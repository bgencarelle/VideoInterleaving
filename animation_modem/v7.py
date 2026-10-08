"""V7 prototype: encoder and decoder following docs/transport_v7_spec.md.

The transport remains display/application neutral. Standalone tools and the
application modem adapter both call this module; it does not import settings,
renderers, or audio devices.

Scope: a faithful, unoptimised simulation of the draft wire -- clock/identity
track (§5), continuous OFDM with scattered/continual pilots (§6), M/S
precoding with mono head blocks, SoftCast gains, rank windows and Hadamard
time groups (§7-§8), and the receiver chain of §9 (edge-counted clock,
time-map resampling, clock cancellation, pilot channel + per-symbol fade
model, per-cell 2x2 MMSE, group LMMSE, confidence gate, tail store).
"""
import math
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
from numba import njit
from PIL import Image
from scipy.linalg import hadamard
from scipy.optimize import curve_fit
from scipy.signal import filtfilt, firwin, savgol_filter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from animation_modem import transport3 as PULSE                             # noqa: E402
from animation_modem.v7_core import (SourceCoder, _sample_at, speed_length,
                                  speed_resample)                  # noqa: E402
from animation_modem.v7_core import bound_emission, shape_emission         # noqa: E402
from animation_modem.v7_input_kernels import (                               # noqa: E402
    _leg_correlation_sums, _mono_gain)
from animation_modem.v7_metadata_kernels import (                            # noqa: E402
    decode_metadata_spectrum as _decode_metadata_spectrum)

# ------------------------------------------------------------------ §6.1
RATE, N, CP, SYM, F = 48000, 128, 16, 144, 24
V7_GRIDS = ((96, 80), (48, 40), (48, 40))
V7_SHAPES = ((48, 40), (24, 20), (24, 20))
FRAME = F*SYM                                   # 3456 samples, 13.889 fps
META_SYMBOL = SYM
PULSE_FRAME = PULSE.SYNC_LEN + FRAME + META_SYMBOL + 32  # 3920, 12.245 fps
PULSE_FPS = RATE/PULSE_FRAME
PULSE_GUARD_BASE = 32
EOF_MARKER_RUNS = (4, 8, 6, 6)
EOF_MARKER_LENGTH = sum(EOF_MARKER_RUNS)
EOF_MARKER_PREFIX = 32-EOF_MARKER_LENGTH
EOF_MARKER_OFFSET = PULSE_FRAME-32+EOF_MARKER_PREFIX
EOF_MARKER_LEVEL = PULSE.PREAMBLE_AMPLITUDE
EOF_MARKER_LEVELS = (1, -1, 1, -1)
EOF_MARKER_EDGES = np.cumsum(EOF_MARKER_RUNS[:-1], dtype=float)-.5
EOF_MARKER_CENTERS = (np.cumsum((0,)+EOF_MARKER_RUNS[:-1], dtype=float)+
                      np.asarray(EOF_MARKER_RUNS, dtype=float)/2)
# Fractions of the level the packet's header arrived at (_header_level).
EOF_MARKER_MIN_LEVEL = .6       # every run of the mark must reach this
EOF_HEADER_PERCENTILE = 90      # of the header word's magnitude: its plateau
EOF_MARKER_RUNS_ARRAY = np.asarray(EOF_MARKER_RUNS, dtype=np.float64)
EOF_SEARCH_FRACTION = .012
# measure_pulses() fits one scale over the preamble edge word, centered near
# this reference-sample coordinate within each packet.
PULSE_PREAMBLE_CENTER = (
    16.0 + float(np.mean(PULSE.NOMINAL_EDGES.astype(float)+.5)))
PULSE_WARP_STATIC_BIAS_DELTA = 100e-6
PULSE_WARP_STATIC_BIAS_GATE = 1000e-6
PULSE_WARP_MIN_SLOPE_DELTA = 500e-6
# Pulse scale bounds are relative to the 48 kHz reference geometry. A capture
# at another sample rate observes raw sample scales multiplied by rate/RATE.
PULSE_MIN_SCALE = .25
PULSE_MAX_SCALE = 100.0   # 0.01x playback; inverse scale sets LiveInput's bound
MIN_PLAYBACK_SPEED = .25
MAX_PLAYBACK_SPEED = 4.0
PILOT_TONE_BINS = (1, 3)
PILOT_TONE_REL_DB = -20.0
# Fixed phase origins are 0 rad (bin 1) and 1 rad (bin 3); bin 2 stays clear.
PILOT_TONE_PHASES = {1: 0.0, 3: 1.0}
PILOT_TONE_DURATION = PULSE_FRAME
PILOT_TONE_NARROW_SNR_DB = 16.0
PILOT_TONE_NARROW_WINDOW = 9
PILOT_TONE_JOINT_MIN_COHERENCE = .35
PILOT_TONE_JOINT_MAX_DISAGREEMENT = 1.5
PILOT_TONE_JOINT_RELATIVE_TOLERANCE = .005
PILOT_TONE_JOINT_BASELINE_TIMING_TRIGGER = .02
PILOT_TONE_MAX_PILOT_RESIDUAL = .01
PILOT_TONE_MAX_CHANNEL_TIMING_RESIDUAL = .5
ENCODING_FILTERS = ('nearest', 'box', 'lanczos', 'bicubic')
ENCODING_FILTER_CODES = {name: code for code, name in
                         enumerate(ENCODING_FILTERS)}


def pulse_sample_scale_bounds(sample_rate=RATE):
    """Bounds for pulse scales measured in the capture's sample coordinates.

    The scale limits describe playback relative to the 48 kHz wire geometry.
    A different capture rate changes the number of input samples per packet,
    not the admissible playback-speed range. The current range is 0.01x--4x.
    """
    sample_rate = float(sample_rate)
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError('sample_rate must be finite and positive')
    factor = sample_rate/RATE
    return PULSE_MIN_SCALE*factor, PULSE_MAX_SCALE*factor


PREPARED_SIZE = (80, 96)        # (width, height) every source is resized to


def prepare_image(image, encode_filter='lanczos'):
    """Prepare an RGB source without depending on the application imaging API."""
    resampling = getattr(Image.Resampling, encode_filter.upper())
    image = image.convert('RGB')
    if (resampling == Image.Resampling.BOX and
            image.width % PREPARED_SIZE[0] == 0 and
            image.height % PREPARED_SIZE[1] == 0):
        # Pillow's integer-factor box reducer avoids the general resize filter
        # setup for common camera/capture sizes. Its rounding can differ from
        # resize(BOX) by one RGB code value, so keep this to BOX and exact grids.
        return image.reduce((image.width//PREPARED_SIZE[0],
                             image.height//PREPARED_SIZE[1]))
    return image.resize(PREPARED_SIZE, resampling)


def image_values(image, shapes=V7_SHAPES, encode_filter='lanczos'):
    """Convert a prepared/source image to the fixed V7 YCbCr vector."""
    resampling = getattr(Image.Resampling, encode_filter.upper())
    rows, cols = shapes[0]
    sampled = image.convert('RGB').resize((cols, rows), resampling)
    planes = sampled.convert('YCbCr').split()
    return np.concatenate([
        np.asarray(plane.resize((shape[1], shape[0]), Image.Resampling.BOX)).ravel()
        for plane, shape in zip(planes, shapes)]).astype(float)/127.5 - 1


def speed_pulse_stream(audio, speed=1.0, rate=RATE):
    """Speed each pulse packet independently, preserving preamble boundaries."""
    audio = np.asarray(audio, np.float32)
    if len(audio) % PULSE_FRAME:
        raise ValueError('pulse stream length must contain complete V7 packets')
    return np.concatenate([
        speed_resample(audio[start:start+PULSE_FRAME], rate, speed)
        for start in range(0, len(audio), PULSE_FRAME)
    ])


@lru_cache(maxsize=4)
def _pilot_tone_wave(length):
    """Unit-amplitude pilot tone sum, N-periodic, long enough for any origin."""
    period = np.arange(N)
    one = sum(np.cos(2*np.pi*bin_index*period/N + PILOT_TONE_PHASES[bin_index])
              for bin_index in PILOT_TONE_BINS)
    wave = np.tile(one, int(length)//N + 2)
    wave.setflags(write=False)
    return wave


def _add_pilot_tones(packet, counter, gate_preamble=False):
    """Add packet-long, body-RMS-normalized M references to one pulse packet.

    Each tone's RMS level is -20 dB relative to the OFDM body RMS. The known
    phase origin advances by one PULSE_FRAME per counter, including across
    separately encoded packets.
    """
    packet = np.asarray(packet, np.float64).copy()
    body = packet[PULSE.SYNC_LEN:PULSE.SYNC_LEN+FRAME]
    body_rms = float(np.sqrt(np.mean(body*body)))
    amplitude = np.sqrt(2)*body_rms*10**(PILOT_TONE_REL_DB/20)
    # Every tone bin divides N, so the sum repeats every N samples: read one
    # precomputed period from the packet's phase origin instead of evaluating
    # the cosines at ever larger absolute sample numbers.
    origin = ((int(counter)-1)*PILOT_TONE_DURATION) % N
    tone = amplitude*_pilot_tone_wave(len(packet))[origin:origin+len(packet)]
    if gate_preamble:
        # Test-only fallback for an edge-biased pulse detector. The phase keeps
        # running while the preamble amplitude is faded out and back in.
        envelope = np.ones(len(packet), dtype=float)
        envelope[:16] = np.cos(np.linspace(0, np.pi/2, 16))**2
        envelope[16:272] = 0
        envelope[272:300] = np.sin(np.linspace(0, np.pi/2, 28))**2
        tone *= envelope
    packet += tone[:, None]
    return packet.astype(np.float32)


# The high bit of each pair is orientation; square ignores it.
V7_ASPECT_RATIOS = (1., 4/3, 3/2, 16/9, 1., 3/4, 2/3, 9/16)
V7_ASPECT_NAMES = ('1:1', '4:3', '3:2', '16:9',
                   '1:1', '3:4', '2:3', '9:16')
BINS = np.arange(4, 35)                         # 1.5-12.75 kHz
# The metadata symbol has 31 usable bins.  Spread eleven pilots across the
# band and use the remaining twenty cells for a three-byte payload plus its
# CRC-16 (40 QPSK bits).  Keeping this inside the existing symbol preserves
# the 3,920-sample pulse packet and its 32-sample terminal guard.
META_PILOTS = np.asarray(BINS[::3])
META_DATA_BINS = np.asarray([b for b in BINS if b not in set(META_PILOTS)])
CONTINUAL = (4, 34)
SCAT = {b: ((b-5)//4) % 3 for b in range(5, 34, 4)}
PILOT_BINS = sorted(set(CONTINUAL) | set(SCAT))
WIN = CP-4                                      # FFT window offset in a symbol
PILOT_AMP = 1.5*np.sqrt(2)                      # complex, vs data E|x|^2 ~ 2
H8 = hadamard(8)/np.sqrt(8)
NOISE_FLOOR = 1e-6
DEBUG = {}


def _hadamard8(values):
    """Apply the fixed 8-point Hadamard transform without a BLAS dispatch.

    This tiny transform does not need vendor matrix-multiply dispatch, and the
    fixed contraction keeps behavior consistent across BLAS backends. Reject
    invalid coefficients rather than encoding a damaged packet if a source or
    model calculation ever goes non-finite.
    """
    if not np.isfinite(values).all():
        raise FloatingPointError('V7 Hadamard input contains non-finite values')
    with np.errstate(over='raise', invalid='raise'):
        return np.einsum('fg,gk->fk', values, H8.T, optimize=False)


# The FFT window opens CP-WIN samples early: undo that known linear phase so
# channel interpolation between pilots sees a smooth response.
EARLY = np.exp(2j*np.pi*np.arange(65)*(CP-WIN)/N)
_EARLY32 = EARLY.astype(np.complex64)
_EARLY32.setflags(write=False)


def _solve_2x2_vec(a, b):
    """Solve batched complex 2x2 systems with no tiny-LAPACK dispatch."""
    det = a[..., 0, 0]*a[..., 1, 1] - a[..., 0, 1]*a[..., 1, 0]
    x0 = (a[..., 1, 1]*b[..., 0] - a[..., 0, 1]*b[..., 1])/det
    x1 = (-a[..., 1, 0]*b[..., 0] + a[..., 0, 0]*b[..., 1])/det
    return np.stack((x0, x1), axis=-1)


def _solve_2x2_mat(a, b):
    """Solve A X=B for batched 2x2 matrices."""
    det = a[..., 0, 0]*a[..., 1, 1] - a[..., 0, 1]*a[..., 1, 0]
    x0 = (a[..., 1, 1, None]*b[..., 0, :] -
          a[..., 0, 1, None]*b[..., 1, :])/det[..., None]
    x1 = (-a[..., 1, 0, None]*b[..., 0, :] +
          a[..., 0, 0, None]*b[..., 1, :])/det[..., None]
    return np.stack((x0, x1), axis=-2)

# ------------------------------------------------------------------ §5
BIT = 48
SYNC = [int(c) for c in '0011111111111101']
CLOCK_REL_DB = -3.0


def crc16(data):
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def _bits(value, width):
    return [(value >> (width-1-i)) & 1 for i in range(width)]


def clock_word(counter, profile=0, aspect=6, folders=0, source_index=None):
    # ``source_index`` is not part of the pulse header; the live pulse wire
    # carries it in metadata_symbols().  The word's historical identity
    # fields remain; it now serves only decode_frame's cancellation template.
    payload = (_bits(counter % (1 << 20), 20) + _bits(profile, 4) + _bits(aspect, 3) +
               _bits(folders, 8) + [0]*4)                        # 39 bits
    padded = [0] + payload
    data = bytes(int(''.join(map(str, padded[i:i+8])), 2) for i in range(0, 40, 8))
    word = SYNC + payload + _bits(crc16(data), 16)
    word.append(sum(word) % 2)                                   # polarity bit 71
    assert len(word) == 72
    return word


# ------------------------------------------------------------------ metadata
# Five bytes, 40 QPSK bits in the metadata symbol:
#
#   byte 0   aspect (3) | screen (1) | model (1) | tail slice (3)
#            model 0 nearest, 1 box (ENCODING_FILTER_CODES); screen 0: the
#            aspect is the picture's own, 1: the aspect is the sender's
#            chosen frame (an aspect layout), the picture fitted inside it
#            with bars.  The field was the 2-bit encode filter; lanczos (2)
#            and bicubic (3) are no longer sent.
#   byte 1-2 direction (1) | source index + 1 (15) -- index zero is reserved,
#            so an erased field is invalid; direction 0 counts up, 1 down
#   byte 3-4 CRC-16 of bytes 0-2, XOR a mask chosen by the tail slice
#
# The tail slice (counter mod 7) names which 96 tail coefficients this packet
# carries, and also rotates what the CRC field carries: slices 0-4 a plain
# CRC, slice 5 the CRC XOR the loop phase p, slice 6 the CRC XOR the loop
# field (loop length N, top bit set for a one-way loop).  A receiver recovers
# p and N as (computed CRC) XOR (received field) and, once each has repeated,
# checks every packet at full CRC strength again.  See loop_index().
TAIL_SLICE_P, TAIL_SLICE_N = 5, 6
INDEX_DOWN = 0x8000             # index field flag: the loop is counting down
MAX_SOURCE_INDEX = 0x7ffe       # 32766: the index field keeps 15 bits
LOOP_ONE_WAY = 0x8000           # loop field flag: 0,1..N-1,0,1.. (no ping-pong)
LOOP_NO_CLOCK = 0xFFFF          # p value: the index does not follow the clock
# Sender-side aspect codes may carry this flag above the 3-bit V7 code: the
# metadata's screen bit (see above). Receivers use ``code & 7`` for ratios.
ASPECT_SCREEN = 8
WIRE_ENCODING_TYPES = (0, 1)    # nearest, box: the models the wire can name


def _tail_slice_mask(tail_slice, loop):
    if loop is None:
        return 0
    if tail_slice == TAIL_SLICE_P:
        return loop.phase_field
    if tail_slice == TAIL_SLICE_N:
        return loop.frames_field
    return 0


def metadata_word(aspect_code, encoding_type=0, tail_slice=0, source_index=0,
                  loop=None, direction=1):
    """Pack picture identity, tail slice, direction and one-based index.

    The public/source-facing index remains zero-based, matching the bake and
    display APIs.  ``direction`` is +1 while the loop counts up and -1 while
    it counts down (a ping-pong index alone cannot say which).  ``loop`` (a
    LoopInfo, or None for no loop information) supplies the p/N masks for tail
    slices 5 and 6.
    """
    if not 0 <= int(source_index) <= MAX_SOURCE_INDEX:
        raise ValueError(f'source_index must be 0..{MAX_SOURCE_INDEX}')
    if not 0 <= int(tail_slice) < TAIL_PHASES:
        raise ValueError('tail_slice must be 0..6')
    if int(encoding_type) not in WIRE_ENCODING_TYPES:
        raise ValueError('the wire names only the nearest and box models; '
                         'lanczos and bicubic encodes are not sent')
    if not 0 <= int(aspect_code) <= (7 | ASPECT_SCREEN):
        raise ValueError('aspect_code must be a V7 aspect code (0..7), '
                         'optionally with ASPECT_SCREEN')
    screen = 1 if int(aspect_code) & ASPECT_SCREEN else 0
    wire_index = int(source_index) + 1
    if int(direction) < 0:
        wire_index |= INDEX_DOWN
    payload = bytes([((int(aspect_code) & 7) << 5) | (screen << 4) |
                     ((int(encoding_type) & 1) << 3) |
                     (int(tail_slice) & 7),
                     (wire_index >> 8) & 0xff,
                     wire_index & 0xff])
    field = crc16(payload) ^ _tail_slice_mask(int(tail_slice), loop)
    return payload + field.to_bytes(2, 'big')


def metadata_symbols(aspect_code, encoding_type=0, tail_slice=0, source_index=0,
                     loop=None, direction=1):
    bits = np.unpackbits(np.frombuffer(
        metadata_word(aspect_code, encoding_type, tail_slice, source_index, loop,
                      direction),
        np.uint8))
    return ((bits[0::2].astype(float)*2-1) +
            1j*(bits[1::2].astype(float)*2-1))/np.sqrt(2)


@dataclass(frozen=True)
class Metadata:
    aspect_code: int
    encoding_type: int
    tail_slice: int
    source_index: int
    mask: int                   # computed CRC XOR received field
    direction: int = 1          # +1 counting up, -1 counting down
    screen_aspect: bool = False  # aspect is the sender's frame, not the picture's


def parse_metadata_word(raw):
    """Unpack the five-byte metadata word.

    Returns a Metadata whose ``mask`` is the computed CRC XOR the received
    field, or None for a structurally impossible word (wrong length, index
    zero, tail slice 7).  The mask must be 0 on slices 0-4; on slices 5/6 it
    is p or N, which LoopLock checks.
    """
    raw = bytes(raw)
    if len(raw) != 5:
        return None
    field = int.from_bytes(raw[1:3], 'big')
    wire_index = field & ~INDEX_DOWN
    tail_slice = raw[0] & 7
    if wire_index == 0 or tail_slice >= TAIL_PHASES:
        return None
    mask = crc16(raw[:3]) ^ int.from_bytes(raw[3:5], 'big')
    return Metadata((raw[0] >> 5) & 7, (raw[0] >> 3) & 1, tail_slice,
                    wire_index - 1, mask,
                    -1 if field & INDEX_DOWN else 1,
                    bool((raw[0] >> 4) & 1))


# ------------------------------------------------------------------ loop clock
# The installation's image index is a pure function of wall time (see
# index_calculator.calculate_free_clock_index): ticks since an epoch at
# LOOP_IPS, folded ping-pong over N images.  Only the epoch modulo the loop
# matters, so it collapses to one small constant, the loop phase p:
#
#     ticks = unix_ns * 30 // 1e9            (any correct clock, no epoch)
#     index = fold((ticks - p) mod period)   period = 2N ping-pong, N one-way
#
# For an epoch on a whole second this is exactly the application's formula.
# The wire carries N and p (tail slices 6 and 5), so a receiver that shares
# nothing with the sender but a correct clock knows where the loop is and how
# late the picture on screen is.
LOOP_IPS = 30


@dataclass(frozen=True)
class LoopInfo:
    frames: int                 # N, 1..32767 (0: no loop information)
    phase: int = LOOP_NO_CLOCK  # p, 0..period-1, or LOOP_NO_CLOCK
    pingpong: bool = True

    @classmethod
    def from_epoch(cls, frames, epoch_ns, pingpong=True):
        frames = int(frames)
        return cls(frames, loop_ticks(epoch_ns) % loop_period(frames, pingpong),
                   bool(pingpong))

    @classmethod
    def from_fields(cls, frames_field, phase_field):
        frames = int(frames_field) & ~LOOP_ONE_WAY
        return cls(frames, int(phase_field), not frames_field & LOOP_ONE_WAY)

    @property
    def frames_field(self):
        if not 0 <= self.frames < LOOP_ONE_WAY:
            raise ValueError('loop length must fit 15 bits')
        return self.frames | (0 if self.pingpong else LOOP_ONE_WAY)

    @property
    def phase_field(self):
        return int(self.phase) & 0xFFFF

    @property
    def clocked(self):
        return (self.frames > 0 and self.phase != LOOP_NO_CLOCK and
                self.phase < loop_period(self.frames, self.pingpong))


def loop_ticks(time_ns):
    return int(time_ns)*LOOP_IPS//1_000_000_000


def loop_period(frames, pingpong=True):
    return 2*frames if pingpong and frames > 1 else max(int(frames), 1)


def loop_index(ticks, loop):
    """The index the loop shows at ``ticks`` (index_calculator's fold)."""
    period = loop_period(loop.frames, loop.pingpong)
    raw = (int(ticks) - loop.phase) % period
    if loop.pingpong and loop.frames > 1 and raw >= loop.frames:
        return period - 1 - raw
    return raw


def loop_direction(ticks, loop):
    """+1 while the loop counts up at ``ticks``, -1 while it counts down."""
    if not (loop.pingpong and loop.frames > 1):
        return 1
    raw = (int(ticks) - loop.phase) % loop_period(loop.frames, loop.pingpong)
    return 1 if raw < loop.frames else -1


def loop_lag_ticks(index, ticks, loop, expected=0, direction=None):
    """How many loop ticks the picture showing ``index`` is behind the live
    loop at ``ticks`` (negative: ahead).

    A ping-pong index occurs twice per period, and near a turn the two
    occurrences are close together, so a single reading can be ambiguous.
    ``direction`` (from the wire) settles it outright.  Otherwise the
    occurrence whose lag is nearest ``expected`` is taken: a receiver
    passes its recent lag, and since lag changes slowly this follows the
    true branch through the turns (and recovers from a wrong first guess as
    soon as the loop moves away from a turn)."""
    period = loop_period(loop.frames, loop.pingpong)
    live = (int(ticks) - loop.phase) % period
    if loop.pingpong and loop.frames > 1 and direction is not None:
        # The wire says which way the loop was going, so the occurrence is
        # known: no guessing at the turns.
        raws = [int(index) if direction > 0 else period - 1 - int(index)]
    else:
        raws = [int(index)]
        if loop.pingpong and loop.frames > 1:
            raws.append(period - 1 - int(index))
    lags = [(live - raw - expected + period//2) % period - period//2 + expected
            for raw in raws]
    return min(lags, key=lambda lag: abs(lag - expected))


class LoopLock:
    """Learns p and N from tail slices 5 and 6 of the rotating CRC field.

    Before a value is known its slice cannot be checked, so a value is only
    adopted after it arrived twice in a row (damage gives random values).
    Once known, those slices are checked like any other; a different value
    repeated twice replaces it (a new sender run).
    """
    CONFIRM = 2

    def __init__(self):
        self.values = {TAIL_SLICE_P: None, TAIL_SLICE_N: None}
        self._candidate = {TAIL_SLICE_P: (None, 0), TAIL_SLICE_N: (None, 0)}

    def check(self, meta):
        """True: CRC verified.  False: damaged.  None: cannot tell yet."""
        if meta.tail_slice not in self.values:
            return meta.mask == 0
        known = self.values[meta.tail_slice]
        value, count = self._candidate[meta.tail_slice]
        count = count + 1 if meta.mask == value else 1
        self._candidate[meta.tail_slice] = (meta.mask, count)
        if known is not None and meta.mask == known:
            return True
        if count >= self.CONFIRM:
            self.values[meta.tail_slice] = meta.mask
            return True
        return False if known is not None else None

    @property
    def loop(self):
        p, n = self.values[TAIL_SLICE_P], self.values[TAIL_SLICE_N]
        if n is None:
            return None
        return LoopInfo.from_fields(n, LOOP_NO_CLOCK if p is None else p)


class TailStore:
    """Receiver memory of recently received coefficients.

    Each tail slice (96 of the 656 tail coefficients) arrives once per seven
    packets.  A value is reused only until its slice comes round again
    (max_age packets), so detail from a much older picture never lingers;
    older values fall back to the model mean.  Head and body arrive in every
    packet and are simply overwritten.
    """

    def __init__(self, max_age=None, enabled=True):
        self.max_age = TAIL_PHASES - 1 if max_age is None else int(max_age)
        self.enabled = enabled
        self._encoding = None
        self._values = self._age = None

    def reset(self):
        """Forget coefficients whose temporal order crossed a discontinuity."""
        self._encoding = None
        self._values = self._age = None

    def prior(self, model):
        if (not self.enabled or self._values is None or
                self._encoding != model.encoding_type):
            return model.mu.copy()
        return np.where(self._age <= self.max_age, self._values, model.mu)

    def update(self, model, coeffs, tail_slice):
        if len(coeffs) != len(model.mu):
            # A wire profile decoded this packet with its own coefficient
            # layout (and keeps its own tail memory); they are not this
            # model's coefficients.
            return
        if self._values is None or self._encoding != model.encoding_type:
            self._encoding = model.encoding_type
            self._values = model.mu.astype(float).copy()
            self._age = np.full(len(model.mu), np.iinfo(np.int64).max//2)
        idx = model.rank_tables[int(tail_slice) % TAIL_PHASES]
        got = idx[idx >= 0]
        self._age += 1
        self._values[got] = np.asarray(coeffs, float)[got]
        self._age[got] = 0


PROVISIONAL_INDEX_WINDOW = 64    # index steps a packet may move (about 26 at 1x)


class PulseState:
    """Everything a live receiver carries from one decode call to the next."""

    def __init__(self, tail_memory=True):
        self.tail = TailStore(enabled=tail_memory)
        self.lock = LoopLock()
        self.last_verified = None
        self.playback_direction = None
        self._require_independent_tail_metadata = False

    def set_playback_direction(self, direction):
        """Reset order-dependent picture state before decoding a turn-around."""
        direction = int(direction)
        if direction not in (-1, 1):
            raise ValueError('playback direction must be +1 or -1')
        changed = (self.playback_direction is not None and
                   self.playback_direction != direction)
        if changed:
            self.tail.reset()
            self.last_verified = None
        self.playback_direction = direction
        return changed

    def accept(self, meta):
        """True if the metadata can be used: CRC-verified, or -- before p/N
        are learned -- consistent with the last verified packet (same encode
        filter and aspect, index within a few steps).  A random word passes
        that about 1 time in 16,000, close to the CRC's own 1 in 65,536."""
        if meta is None:
            return False, False
        verified = self.lock.check(meta)
        if verified:
            self.last_verified = meta
            return True, False
        last = self.last_verified
        provisional = (verified is None and last is not None and
                       meta.encoding_type == last.encoding_type and
                       meta.aspect_code == last.aspect_code and
                       meta.screen_aspect == last.screen_aspect and
                       abs(meta.source_index - last.source_index) <=
                       PROVISIONAL_INDEX_WINDOW)
        return provisional, provisional

    def accept_candidates(self, candidates):
        """Choose the intact metadata track when a transient splits the legs."""
        candidates = list(dict.fromkeys(
            (meta for meta in candidates if meta is not None)))
        if not candidates:
            return None, False, False
        last = self.last_verified

        def score(meta):
            known = self.lock.values.get(meta.tail_slice)
            if known is None and meta.tail_slice in (0, 1, 2, 3, 4):
                known = 0
            crc_match = known is not None and meta.mask == known
            if last is None:
                continuity = 0
            elif (meta.encoding_type == last.encoding_type and
                  meta.aspect_code == last.aspect_code and
                  meta.screen_aspect == last.screen_aspect):
                direction = meta.direction or last.direction or 1
                expected = last.source_index + direction
                delta = abs(meta.source_index-expected)
                continuity = (-delta if delta <= PROVISIONAL_INDEX_WINDOW
                              else -PROVISIONAL_INDEX_WINDOW-1)
            else:
                continuity = -PROVISIONAL_INDEX_WINDOW-2
            return (crc_match, continuity)

        # Stable ordering keeps the historical combined-track decision when
        # candidates tie; an independently clean leg wins on CRC/order when
        # the other has been transiently erased.
        meta = max(candidates, key=score)
        valid, provisional = self.accept(meta)
        return meta, valid, provisional


def aspect_wire_code(size):
    """Return the compact V7 family/orientation code for a source size."""
    width, height = size
    ratio = width/height
    families = (1., 4/3, 3/2, 16/9)
    family = min(range(4), key=lambda i: min(
        abs(np.log(ratio/families[i])),
        abs(np.log(ratio/(1/families[i])))))
    return family | (4 if ratio < 1 and family else 0)


_CLOCK_LP = firwin(255, 1250, fs=RATE, window=('kaiser', 8))


def clock_wave(words):
    """Biphase-mark square wave for consecutive words, zero-phase low-passed."""
    level, out = 1.0, []
    for word in words:
        for bit in word:
            level = -level
            out += [level]*(BIT//2)
            if bit:
                level = -level
            out += [level]*(BIT//2)
    return filtfilt(_CLOCK_LP, [1.0], np.asarray(out))


@lru_cache(maxsize=256)
def clock_cancel_template(counter, dtype=float):
    """Cache the deterministic three-word cancellation template."""
    clk = clock_wave((clock_word(counter-1), clock_word(counter),
                      clock_word(counter+1)))
    return np.asarray(clk[FRAME-64:2*FRAME+64], dtype=dtype)


# ------------------------------------------------------------------ §6.4, §8
def data_blocks():
    blocks = [(b, p) for b in range(5, 34) for p in range(3) if SCAT.get(b) != p]
    return sorted(blocks)                               # health order (§8.1)


BLOCKS = data_blocks()
MONO = [blk for blk in BLOCKS if blk[0] <= 9]
STEREO = [blk for blk in BLOCKS if blk[0] > 9]
# Stereo slot order (§8.3): every M slot, by carrier, then every S slot, by
# carrier.  Mono playback loses every S slot, so it now loses a suffix of the
# rank order -- the least important ranks -- and keeps a whole lower-detail
# picture instead of one with ranks missing all through it.  Measured at 1x
# (4 images, SSIMULACRA2 against the clean decode): mono -56 with S ordered
# as if on carrier b+6, -33 here, which equals the ideal for this capacity
# (M exact, S at the prior mean).  Stereo cases are unchanged except soft
# saturation, which follows each picture's emitted level (see §8.3).
STEREO_SLOTS = sorted(
    [(b, c, q) for b in STEREO for c in 'MS' for q in 'IQ'],
    key=lambda slot: (slot[1] == 'S', slot[0][0], slot[0][1], slot[2]))
GROUPS = [(b, 'M', q) for b in MONO for q in 'IQ'] + STEREO_SLOTS
BLOCK_BINS = np.asarray([blk[0] for blk in BLOCKS], int)
BLOCK_SYMBOLS = np.asarray([phi + 3*np.arange(8)
                            for _, phi in BLOCKS], int)
_block_number = {blk: i for i, blk in enumerate(BLOCKS)}
GROUP_BLOCK = np.asarray([_block_number[(b, phi)]
                          for ((b, phi), _, _) in GROUPS], int)
GROUP_STREAM_INDEX = np.asarray([0 if stream == 'M' else 1
                                 for (_, stream, _) in GROUPS], int)
GROUP_Q_INDEX = np.asarray([0 if q == 'I' else 1
                            for (_, _, q) in GROUPS], int)
BLOCK_GROUP_INDEX = np.full((len(BLOCKS), 2, 2), -1, int)
for _gi, _bi in enumerate(GROUP_BLOCK):
    BLOCK_GROUP_INDEX[_bi, GROUP_STREAM_INDEX[_gi], GROUP_Q_INDEX[_gi]] = _gi
N_HEAD_G, N_TAIL_G = 2*len(MONO), 12                    # 26 head, 12 tail groups
HEAD, BODY_END, TAIL_PER = 208, 2224, 96
TAIL_PHASES = 7
HEAD_MIN_CONFIDENCE = .70
HEAD_MIN_COVERAGE = .50
LIVE_VALID_HEAD_CONFIDENCE = .85
LIVE_VALID_HEAD_COVERAGE = .75
# Pilot-noise limits are relative: the pilot residual's power over the power
# a unit cell is received at (_pilot_reference_power), so they do not move
# with the input level. The model's data cells have power about 2, so 2.0 is
# noise as strong as the picture signal the model expects.
LIVE_MAX_PILOT_NOISE = 2.0
# A symbol whose residual is past this (and well past its channel's median
# or the paired channel) is a local erasure, not packet-wide noise.
ERASURE_PILOT_NOISE = .08
DISPLAY_MIN_HEAD_CONFIDENCE = .60
DISPLAY_MIN_HEAD_COVERAGE = .50
DISPLAY_MAX_PILOT_NOISE = 2.0

# Static placement tables: the wire generations do this kind of work once at setup, not on
# every picture.  The flattened arrays are used by the vectorized scatter in
# encode_frame_coeffs().
GROUP_BINS = np.asarray([blk[0] for (blk, _, _) in GROUPS], int)
GROUP_STREAMS = np.asarray([0 if stream == 'M' else 1
                            for (_, stream, _) in GROUPS], int)
GROUP_QMULT = np.asarray([1 if q == 'I' else 1j for (_, _, q) in GROUPS])
GROUP_SYMBOLS = np.asarray([
    phi + 3*np.arange(8) for ((b, phi), _, _) in GROUPS], int)
# Flat (symbol, bin, stream) cell of every (group, member): I groups write the
# real part of a cell and Q groups the imaginary part, and no two groups of
# the same kind share a cell, so the scatter is a plain assignment.
GROUP_CELLS = ((GROUP_SYMBOLS*65 + GROUP_BINS[:, None])*2 +
               GROUP_STREAMS[:, None])
GROUP_IS_I = GROUP_QMULT == 1
CELLS_I = GROUP_CELLS[GROUP_IS_I].ravel()
CELLS_Q = GROUP_CELLS[~GROUP_IS_I].ravel()
assert len(np.unique(CELLS_I)) == CELLS_I.size
assert len(np.unique(CELLS_Q)) == CELLS_Q.size
SCATTERED_PILOTS = np.zeros((F, 65, 2), complex)
for _s in range(F):
    for _b in CONTINUAL:
        SCATTERED_PILOTS[_s, _b] = PILOT_AMP*np.array([1, 1j*(-1)**_s])
    for _b, _phi in SCAT.items():
        if _s % 3 == _phi:
            _v = (_s-_phi)//3
            SCATTERED_PILOTS[_s, _b] = PILOT_AMP*np.array([1, 1j*(-1)**_v])


def windows(n_groups):
    """(group index within tier, member t) -> position in tier sequence."""
    pos = np.empty((n_groups, 8), int)
    start = 0
    while start < n_groups:
        k = min(8, n_groups-start)
        for j in range(k):
            pos[start+j] = start*8 + j + k*np.arange(8)
        start += k
    return pos


def frame_ranks(order, counter):
    """Coefficient index for every (group, member) of this frame (§7.4, §8.4)."""
    tail = order[BODY_END:]
    p = counter % TAIL_PHASES
    tail_now = tail[TAIL_PER*p:TAIL_PER*(p+1)]
    tail_now = np.pad(tail_now, (0, TAIL_PER-len(tail_now)), constant_values=-1)
    tiers = [(order[:HEAD], slice(0, N_HEAD_G)),
             (order[HEAD:BODY_END], slice(N_HEAD_G, len(GROUPS)-N_TAIL_G)),
             (tail_now, slice(len(GROUPS)-N_TAIL_G, len(GROUPS)))]
    out = np.empty((len(GROUPS), 8), int)
    for seq, sl in tiers:
        n = sl.stop-sl.start
        out[sl] = seq[windows(n)]
    return out


# ------------------------------------------------------------------ §7
@dataclass
class Model:
    coder: SourceCoder
    mu: np.ndarray
    lam: np.ndarray
    order: np.ndarray
    gain: np.ndarray
    phase: np.ndarray
    scale: float
    plane: np.ndarray
    head: np.ndarray
    rank_tables: tuple
    encoding_type: int = 0
    # Per tail phase: the per-cell (M, S) prior power of every data block.
    # A pure function of gain, lam and rank_tables, precomputed so decode does
    # not rebuild it frame by frame (see block_priors()).
    block_prior_tables: tuple = ()
    # Receiver-only float32 views.  Keep the canonical float64 tables as the
    # default wire path; the opt-in ARM path reuses these instead of casting
    # them for every decoded frame.
    mu32: np.ndarray = field(default=None, repr=False)
    lam32: np.ndarray = field(default=None, repr=False)
    gain32: np.ndarray = field(default=None, repr=False)
    phase32: np.ndarray = field(default=None, repr=False)
    block_prior_tables32: tuple = field(default=(), repr=False)


# The canonical V7 profile is frozen data, not a runtime computation.  Its
# tables were derived once from REFERENCE_FIXTURE by derive_tables() and
# phase_table() (tools/v7_freeze_tables.py) and are loaded from MODEL_TABLES,
# whose SHA-256 is part of the wire definition.  Sender and receiver therefore
# agree without the fixture image, Pillow, SciPy's curve fit or NumPy's RNG
# stream -- any of which could drift between library versions and silently
# desynchronise the two ends (a different phase table decodes as garbage).
REFERENCE_FIXTURE = ROOT/'modem_tests/fixtures/v7_reference_face.png'
MODEL_TABLES = Path(__file__).with_name('v7_model_tables.npz')
MODEL_TABLES_SHA256 = '2392943a287fdf1cd2ac773c2fc065179006bfab4646726e72e023f3921ffb1c'
PHASE_SEED = 70001          # provenance of the frozen phase table
CROP_SEED = 1               # provenance of the frozen variance tables
LEVEL_SEED = 6              # provenance of the frozen unit levels


def phase_table():
    """Regenerate the per-cell phase table (provenance / drift check only)."""
    return np.exp(2j*np.pi*np.random.default_rng(PHASE_SEED).random((F, 65)))


def _planes():
    plane = np.empty(sum(r*c for r, c in V7_SHAPES), int)
    off = 0
    for p, (r, c) in enumerate(V7_SHAPES):
        plane[off:off+r*c] = p
        off += r*c
    return plane


def _assemble(tables, phase, target_rms, encode_filter):
    """Model from (mu, lam, order, gain, unit_rms) tables and a phase table."""
    coder = SourceCoder(V7_SHAPES, grids=V7_GRIDS)
    order = np.asarray(tables['order'])
    head = np.zeros(len(tables['lam']), bool); head[order[:HEAD]] = True
    rank_tables = tuple(frame_ranks(order, p) for p in range(TAIL_PHASES))
    gain, lam = np.asarray(tables['gain']), np.asarray(tables['lam'])
    priors = tuple(block_priors(gain, lam, idx) for idx in rank_tables)
    mu32 = np.asarray(tables['mu'], dtype=np.float32)
    lam32 = np.asarray(lam, dtype=np.float32)
    gain32 = np.asarray(gain, dtype=np.float32)
    phase32 = np.asarray(phase, dtype=np.complex64)
    priors32 = tuple(np.asarray(table, dtype=np.float32) for table in priors)
    for array in (mu32, lam32, gain32, phase32, *priors32):
        array.setflags(write=False)
    return Model(coder, np.asarray(tables['mu']), lam, order, gain, phase,
                 target_rms/float(tables['unit_rms']), _planes(), head,
                 rank_tables, ENCODING_FILTER_CODES[encode_filter],
                 priors, mu32, lam32, gain32, phase32, priors32)


def block_priors(gain, lam, idx):
    """Per-cell prior power of each data block's M/S components for one tail
    phase's rank table: the mean transmitted power g^2*lam of each group."""
    ranks = np.maximum(idx, 0)
    tx_var = np.where(idx >= 0, gain[ranks]**2*lam[ranks], 0).mean(axis=1)
    valid_group = BLOCK_GROUP_INDEX >= 0
    table = np.where(valid_group, tx_var[np.maximum(BLOCK_GROUP_INDEX, 0)], 0.0)
    table.setflags(write=False)
    return table


def derive_tables(source, encode_filter='lanczos', phase=None):
    """Derive a profile's statistics from an image (seeded crops + power-law fit).

    Used to create the frozen canonical tables and for explicitly custom
    profiles. The result depends on Pillow, SciPy and NumPy's RNG stream.
    """
    coder = SourceCoder(V7_SHAPES, grids=V7_GRIDS)
    rng = np.random.default_rng(CROP_SEED)
    im = source.convert('RGB'); W, Hh = im.size
    C = []
    for _ in range(120):
        s = rng.uniform(.45, 1.0); w = int(min(W, Hh*.75)*s); h = int(w*4/3)
        x = rng.integers(0, W-w+1); y = rng.integers(0, Hh-h+1)
        crop = im.crop((x, y, x+w, y+h))
        if rng.random() < .5:
            crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
        C.append(coder.forward(
            image_values(prepare_image(crop, encode_filter=encode_filter),
                         coder.grids, encode_filter=encode_filter))/coder.gains)
    C = np.asarray(C)
    mu = np.zeros(C.shape[1]); lam = np.empty(C.shape[1])
    off = 0
    for p, ((r, c), (gr, gc)) in enumerate(zip(V7_SHAPES, V7_GRIDS)):
        sl = slice(off, off+r*c)
        mu[off] = C[:, off].mean()                       # DC library mean
        L = np.mean(C[:, sl]**2, axis=0)
        rad = np.hypot(np.arange(r)[:, None]/gr, np.arange(c)[None, :]/gc).ravel()
        f = lambda R, a, k, q: a - q*np.log1p(k*R)
        m = rad > 0
        (a, k, q), _ = curve_fit(f, rad[m], np.log(np.maximum(L[m], 1e-12)),
                                 p0=(0, 20, 2),
                                 bounds=([-50, 0, 0], [50, 1e4, 10]), maxfev=20000)
        fitted = np.exp(f(rad, a, k, q)); fitted[0] = C[:, off].var() + 1e-6
        lam[sl] = fitted; off += r*c
    order = np.argsort(-lam, kind='stable')
    g = lam**-.25
    g /= np.sqrt(np.mean((g*g*lam)[order[:BODY_END]]))   # unit mean slot power
    tables = {'mu': mu, 'lam': lam, 'order': order, 'gain': g, 'unit_rms': 1.0}
    # Fixed level (§6.6): one-off calibration on seeded synthetic coefficients
    # drawn from the variance table -- a property of the profile, never of the
    # frame being sent.  Stored as the RMS of a unit-scale probe frame.
    probe_model = _assemble(tables, phase_table() if phase is None else phase,
                            1.0, encode_filter)
    synth = np.random.default_rng(LEVEL_SEED).standard_normal(len(lam))*np.sqrt(lam) + mu
    probe = encode_frame_coeffs(probe_model, synth, 1)
    tables['unit_rms'] = float(np.sqrt(np.mean(probe**2)))
    return tables


@lru_cache(maxsize=1)
def _frozen_tables():
    import hashlib
    blob = MODEL_TABLES.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != MODEL_TABLES_SHA256:
        raise ValueError(f'{MODEL_TABLES.name} SHA-256 {digest} does not match the '
                         f'V7 wire definition {MODEL_TABLES_SHA256}')
    with np.load(MODEL_TABLES, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def load_model(target_rms, encode_filter='lanczos'):
    """The canonical V7 model from the frozen, hash-checked tables."""
    if encode_filter not in ENCODING_FILTER_CODES:
        raise ValueError(f'unknown encode filter {encode_filter!r}')
    t = _frozen_tables()
    tables = {k: t[f'{encode_filter}/{k}'] for k in ('mu', 'lam', 'order', 'gain', 'unit_rms')}
    return _assemble(tables, t['phase'], target_rms, encode_filter)


def _is_reference(fixture):
    return fixture is None or Path(fixture).resolve() == REFERENCE_FIXTURE.resolve()


def build_model(fixture, target_rms, encode_filter='lanczos'):
    """Canonical frozen model for the reference fixture (or None); otherwise a
    custom profile derived from ``fixture``, which the receiver must match."""
    if _is_reference(fixture):
        return load_model(target_rms, encode_filter)
    with Image.open(fixture) as source:
        return build_model_from_image(source, target_rms, encode_filter)


def build_model_from_image(source, target_rms, encode_filter='lanczos'):
    """A custom (non-canonical) V7 model derived from an in-memory image."""
    phase = phase_table()
    return _assemble(derive_tables(source, encode_filter, phase), phase,
                     target_rms, encode_filter)


# ------------------------------------------------------------------ encoder
def encode_frame_coeffs(model, coeffs, counter, return_X=False,
                        pilot_values=None, right_coeffs=None):
    """One frame's OFDM body. With ``right_coeffs`` the model must be an
    M-only (mono) wire: ``coeffs`` then goes out on the left channel alone
    and ``right_coeffs`` on the right, each with the pilots, as two
    independent mono wires built in one pass."""
    idx = model.rank_tables[counter % TAIL_PHASES]
    ranks = np.maximum(idx, 0)

    def data_cells(values):
        cells = np.zeros((F, 65, 2), complex)                # symbol, bin, M/S
        vals = model.gain[ranks]*(values - model.mu)[ranks]
        vals[idx < 0] = 0
        tx = _hadamard8(vals)
        flat = cells.reshape(-1)
        flat.real[CELLS_I] = tx[GROUP_IS_I].ravel()
        flat.imag[CELLS_Q] = tx[~GROUP_IS_I].ravel()
        return cells

    X = data_cells(coeffs)
    pilots = SCATTERED_PILOTS if pilot_values is None else pilot_values
    if right_coeffs is None:
        X += pilots
    else:
        other = data_cells(right_coeffs)
        if np.any(X[..., 1]) or np.any(other[..., 1]) or np.any(pilots[..., 1]):
            raise ValueError('two-channel encoding needs an M-only wire')
        # Left = (M+S)/sqrt(2) and right = (M-S)/sqrt(2) must equal each
        # wire's own M/sqrt(2): M is their mean, S half their difference.
        left, right = X[..., 0]+pilots[..., 0], other[..., 0]+pilots[..., 0]
        X[..., 0] = (left+right)/2
        X[..., 1] = (left-right)/2
    if return_X:
        return X
    XL = (X[..., 0]+X[..., 1])/np.sqrt(2)*model.phase
    XR = (X[..., 0]-X[..., 1])/np.sqrt(2)*model.phase
    waves = np.fft.irfft(np.stack((XL, XR), axis=-1), n=N, axis=1)
    waves *= model.scale
    out = np.concatenate((waves[:, -CP:, :], waves), axis=1).reshape(-1, 2)
    return out


# Pulse packets are band-limited here (bound_emission in encode_pulse_frame).
EMISSION_EDGE_HZ = 14000

# Peak reduction (fixes list item 28). The body's rare spikes set the level
# of the whole packet, so they are clipped at PEAK_CLIP_DB over the body's
# average and the clipping's spill outside the used carriers is filtered
# off, PEAK_CLIP_ROUNDS times. What lands on the carriers is a small fixed
# noise; the packet can then be sent louder. None turns it off: off by
# default, since live it cost more picture than it gained except under heavy
# hiss, and dropped packets on one test picture (item 28).
PEAK_CLIP_DB = None
PEAK_CLIP_ROUNDS = 3


def _reduce_peaks(body, clip_db=None, rounds=None):
    """Clip and filter an OFDM body, symbol by symbol, keeping each
    symbol's used bins and cyclic prefix."""
    clip_db = PEAK_CLIP_DB if clip_db is None else clip_db
    rounds = PEAK_CLIP_ROUNDS if rounds is None else rounds
    symbols = np.asarray(body, np.float64).reshape(F, SYM, -1)
    core = symbols[:, CP:, :]
    spectrum = np.fft.rfft(core, axis=1)
    used = np.abs(spectrum) > 1e-9*max(float(np.max(np.abs(spectrum))), 1e-30)
    level = np.sqrt(np.mean(core*core, axis=(0, 1)))
    if not np.all(level > 0):
        return body
    limit = level*10**(clip_db/20)
    wave = core
    for _ in range(int(rounds)):
        wave = np.clip(wave, -limit, limit)
        cells = np.fft.rfft(wave, axis=1)
        cells[~used] = 0
        wave = np.fft.irfft(cells, n=N, axis=1)
    out = np.concatenate((wave[:, -CP:, :], wave), axis=1)
    return out.reshape(np.shape(body)).astype(np.float32)


def max_wire_speed(rate):
    """Playback speed below which a `rate` Hz output retains the full band.

    Speeding the wire up multiplies every frequency, so the emission edge
    (14 kHz at 1x) stays below the output's Nyquist frequency up to this
    speed. Faster playback is permitted, but the speed conversion filters or
    aliases high carriers and can reduce decode quality.
    """
    return float(rate)/(2*EMISSION_EDGE_HZ)


def encode_pulse_frame(model, values, counter, aspect_code=0, source_index=None,
                       loop=None, direction=1, pilot_tones=False,
                       pilot_tone_gate_preamble=False,
                       pulse_profile_code=1, extra_tone_mixer=None):
    """One edge-counted pulse-framed V7 body for low-latency live transport.

    ``counter`` is the packet count: counter mod 7 picks the tail slice, which
    the metadata names so the receiver places the tail correctly.  ``loop``
    (LoopInfo) is carried in the rotating CRC field for receivers that want
    the loop length and the live lag; None sends no loop information.
    """
    coeffs = model.coder.forward(values)/model.coder.gains
    return encode_pulse_frame_coeffs(
        model, coeffs, counter, aspect_code=aspect_code,
        source_index=source_index, loop=loop, direction=direction,
        pilot_tones=pilot_tones,
        pilot_tone_gate_preamble=pilot_tone_gate_preamble,
        pulse_profile_code=pulse_profile_code,
        extra_tone_mixer=extra_tone_mixer)


# Emitted levels. The header's shaped peak is HEADER_PEAK_DB below full scale,
# and the EOF pulses peak at the same level. EOF-marked
# packets are measured after tone mixing: the body stays at least
# 1.5 dB below the lower final header/EOF peak, and metadata stays 0.5 dB below
# the body.
# Timing tones are scaled with body RMS. Header and EOF peak 1.75 dB under
# clipping before the timing tones, 1 to 1.5 dB with them (fixes list
# item 28).
HEADER_PEAK_DB = 1.75
HEADER_PEAK = 10**(-HEADER_PEAK_DB/20)
BODY_BELOW_EOF_DB = 1.5
METADATA_BELOW_BODY_DB = .5

# Per-packet auto-level, on for every wire: each packet's body is raised to
# the ceiling a loud one is held at, by at most BODY_AUTO_LEVEL_MAX_DB
# (fixes list item 28). ``body_auto_level(False)`` turns it off in a block.
BODY_AUTO_LEVEL_MAX_DB = 12.0
_AUTO_LEVEL = threading.local()


@contextmanager
def body_auto_level(enabled=True):
    """Packets encoded in this block (this thread) use the whole headroom
    under the header.  The pilots rise with the body, so a receiver reads
    the packet exactly as before, at a better signal-to-noise ratio."""
    previous = getattr(_AUTO_LEVEL, 'on', True)
    _AUTO_LEVEL.on = bool(enabled)
    try:
        yield
    finally:
        _AUTO_LEVEL.on = previous


@lru_cache(maxsize=16)
def _shaped_preamble(profile_code):
    """The band-limited header of one profile, alone in an empty packet."""
    out = np.zeros((PULSE_FRAME, 2), np.float32)
    preamble = PULSE.profile_preamble(profile_code)
    out[16:16+len(preamble), :] = preamble[:, None]
    shaped = shape_emission(out, EMISSION_EDGE_HZ, RATE)
    gain = HEADER_PEAK/float(np.max(np.abs(shaped)))
    shaped = (shaped*np.float32(gain)).astype(np.float32)
    shaped.setflags(write=False)
    return shaped, float(PULSE.PREAMBLE_AMPLITUDE*gain)


@lru_cache(maxsize=1)
def _shaped_eof_marker():
    """The band-limited EOF mark, alone at the end of an empty packet, its
    shaped peak at the header's (HEADER_PEAK). Shaped like the header, so it
    keeps its level through any later rate conversion instead of
    overshooting (a square mark rang 1.5 dB over at 96 kHz, and the sender's
    resampler then turned the whole packet down to hold it)."""
    out = np.zeros((PULSE_FRAME, 2), np.float32)
    marker = np.concatenate([
        np.full(run, level, np.float32)
        for run, level in zip(EOF_MARKER_RUNS, EOF_MARKER_LEVELS)])
    out[EOF_MARKER_OFFSET:, :] = marker[:, None]
    shaped = shape_emission(out, EMISSION_EDGE_HZ, RATE)
    shaped = (shaped*np.float32(HEADER_PEAK/float(np.max(np.abs(shaped)))))
    shaped = shaped.astype(np.float32)
    shaped.setflags(write=False)
    return shaped


def _shaped_eof_edges():
    """Where the shaped mark's three edges cross zero, in samples from the
    mark's start: the shaping moves them a little off the square mark's run
    boundaries, and the mark is measured against these."""
    from scipy.signal import resample_poly
    fine = 32     # the shaped mark is continuous; find its true crossings
    wave = resample_poly(np.asarray(_shaped_eof_marker()[:, 0], np.float64),
                         fine, 1)
    edges = []
    for nominal in EOF_MARKER_EDGES:
        base = (EOF_MARKER_OFFSET+int(np.floor(nominal)))*fine
        for i in sorted(range(base-3*fine, base+4*fine),
                        key=lambda i: abs(i-base)):
            if np.signbit(wave[i]) != np.signbit(wave[i+1]):
                crossing = i+wave[i]/(wave[i]-wave[i+1])
                edges.append(crossing/fine-EOF_MARKER_OFFSET)
                break
        else:
            raise RuntimeError('shaped EOF mark lost an edge')
    return np.asarray(edges, np.float64)


SHAPED_EOF_EDGES = _shaped_eof_edges()


def emitted_pulse_level(profile_code=1):
    """The level the header's and the end marker's pulses peak at."""
    return HEADER_PEAK


def encode_pulse_frame_coeffs(model, coeffs, counter, aspect_code=0,
                              source_index=None, loop=None, direction=1,
                              pilot_tones=False,
                              pilot_tone_gate_preamble=False,
                              pilot_values=None,
                              pulse_profile_code=1, right_coeffs=None,
                              extra_tone_mixer=None):
    """Pulse-frame transformed source coefficients without another DCT pass.
    ``right_coeffs``: see encode_frame_coeffs (two mono wires, one packet).
    ``extra_tone_mixer``, when supplied, adds deterministic body-RMS-scaled
    tones after EOF insertion; EOF-mode level fitting includes their
    contribution."""
    body = encode_frame_coeffs(model, coeffs, counter,
                               pilot_values=pilot_values,
                               right_coeffs=right_coeffs)
    if PEAK_CLIP_DB is not None:
        body = _reduce_peaks(body)
    out = np.zeros((PULSE_FRAME, 2), np.float32)
    out[PULSE.SYNC_LEN:PULSE.SYNC_LEN+FRAME] = body
    meta = np.zeros((N//2+1, 2), complex)
    if source_index is None:
        source_index = counter - 1
    vals = metadata_symbols(aspect_code, model.encoding_type,
                            counter % TAIL_PHASES, source_index, loop, direction)
    meta[META_PILOTS, 0] = 1
    meta[META_DATA_BINS[:len(vals)], 0] = vals
    mx = (meta[:, 0])/np.sqrt(2)*model.phase[-1]
    meta_wave = np.fft.irfft(mx, n=N)
    meta_pcm = np.concatenate([meta_wave[-CP:], meta_wave])*model.scale
    meta_start = PULSE.SYNC_LEN+FRAME
    # Shape the ordinary pulse/body packet first. The metadata symbol has its
    # own cyclic prefix and is inserted afterward so the long packet shaper
    # cannot smear the preceding image symbol across its pilots/data. The
    # header level is fixed per profile. The packet gets a final-mix limiter
    # below, after marker and timing-tone contributions are present.
    shaped = shape_emission(out, EMISSION_EDGE_HZ, RATE)
    # Start the body at the header's peak; the final-mix limiter below then
    # brings it to exactly BODY_BELOW_EOF_DB under the lower framing peak.
    peak = float(np.max(np.abs(shaped)))
    if peak > HEADER_PEAK:
        shaped *= np.float32(HEADER_PEAK/peak)
    elif getattr(_AUTO_LEVEL, 'on', True) and peak > 0:
        # Per-packet auto-level (see body_auto_level).
        shaped *= np.float32(min(HEADER_PEAK/peak,
                                 10**(BODY_AUTO_LEVEL_MAX_DB/20)))
    header, _ = _shaped_preamble(int(pulse_profile_code))
    eof_mark = _shaped_eof_marker()
    body_region = slice(PULSE.SYNC_LEN, PULSE.SYNC_LEN+FRAME)
    header_region = slice(0, PULSE.SYNC_LEN)
    metadata_region = slice(meta_start, meta_start+META_SYMBOL)
    eof_region = slice(EOF_MARKER_OFFSET, PULSE_FRAME)
    body_ratio = 10**(-BODY_BELOW_EOF_DB/20)
    metadata_ratio = 10**(-METADATA_BELOW_BODY_DB/20)

    def assemble_unmixed(body_gain, metadata_gain):
        packet = shaped*np.float32(body_gain)
        packet += header
        packet[metadata_region] += meta_pcm[:, None]*np.float32(metadata_gain)
        packet += eof_mark
        return packet

    # Both built-in and Fold-500 pilot overlays are body-RMS-scaled, so their
    # complete waveform can be measured once at unit body gain and then scaled
    # with the body during the limiter search. This keeps phase and tone mixing
    # in the measured packet without rebuilding the tone template each trial.
    unit_packet = assemble_unmixed(1.0, 1.0)
    unit_mixed = unit_packet
    if pilot_tones:
        unit_mixed = _add_pilot_tones(
            unit_mixed, counter, gate_preamble=pilot_tone_gate_preamble)
        if extra_tone_mixer is not None:
            unit_mixed = np.asarray(extra_tone_mixer(unit_mixed),
                                    dtype=np.float32)
    elif extra_tone_mixer is not None:
        unit_mixed = np.asarray(extra_tone_mixer(unit_mixed), dtype=np.float32)
    tone_overlay = unit_mixed-unit_packet

    def assemble(body_gain, metadata_gain):
        packet = assemble_unmixed(body_gain, metadata_gain)
        packet += tone_overlay*np.float32(body_gain)
        return packet

    def region_peak(packet, region):
        return float(np.max(np.abs(packet[region])))

    def body_is_below_framing(packet):
        framing_peak = min(region_peak(packet, header_region),
                           region_peak(packet, eof_region))
        return region_peak(packet, body_region) <= framing_peak*body_ratio

    # Find the highest image-body gain that leaves a 1.5 dB final-sample
    # margin under both framing regions. Tone amplitude follows body RMS, so
    # every trial reassembles the actual post-tone packet.
    body_gain = 1.0
    packet = assemble(body_gain, 1.0)
    if not body_is_below_framing(packet):
        low, high = 0.0, body_gain
        for _ in range(16):
            trial_gain = (low+high)*.5
            trial = assemble(trial_gain, 1.0)
            if body_is_below_framing(trial):
                low = trial_gain
            else:
                high = trial_gain
        body_gain = low
        packet = assemble(body_gain, 1.0)

    # Metadata is a separate, quieter payload symbol. Limit it only when its
    # final peak (including the timing tone) would reach the image-body peak.
    metadata_limit = region_peak(packet, body_region)*metadata_ratio
    if region_peak(packet, metadata_region) > metadata_limit:
        low, high = 0.0, 1.0
        for _ in range(16):
            trial_gain = (low+high)*.5
            trial = assemble(body_gain, trial_gain)
            if region_peak(trial, metadata_region) <= metadata_limit:
                low = trial_gain
            else:
                high = trial_gain
        packet = assemble(body_gain, low)
    return packet


def encode_pulse_stream(model, values, start_counter=1, aspect_codes=None,
                        source_indices=None, loop=None, directions=None,
                        pilot_tones=False, pilot_tone_gate_preamble=False,
                        pulse_profile_code=1):
    values = list(values) if np.asarray(values).ndim != 1 else [values]
    codes = aspect_codes or [0]*len(values)
    indexes = (list(source_indices) if source_indices is not None else
               [start_counter+i-1 for i in range(len(values))])
    ways = list(directions) if directions is not None else [1]*len(values)
    if len(codes) != len(values) or len(indexes) != len(values) or len(ways) != len(values):
        raise ValueError('aspect_codes, source_indices and directions must match values')
    return np.concatenate([
        encode_pulse_frame(model, value, start_counter+i, code, source_index,
                            loop, way, pilot_tones,
                            pilot_tone_gate_preamble,
                            pulse_profile_code)
        for i, (value, code, source_index, way) in
        enumerate(zip(values, codes, indexes, ways))])


# ------------------------------------------------------------------ receiver: OFDM
@dataclass
class Result:
    counter: int
    status: str
    coeffs: np.ndarray = field(repr=False)
    diag: dict = field(default_factory=dict)


def _pilot_patterns():
    pats = {}
    for s in range(F):
        for b in CONTINUAL:
            pats[(s, b)] = PILOT_AMP*np.array([1, 1j*(-1)**s])
        for b, phi in SCAT.items():
            if s % 3 == phi:
                pats[(s, b)] = PILOT_AMP*np.array([1, 1j*(-1)**((s-phi)//3)])
    return pats


PATS = _pilot_patterns()
PILOT_OBS = tuple(sorted(PATS))
PILOT_SV = np.asarray([s for s, _ in PILOT_OBS])
PILOT_BV = np.asarray([b for _, b in PILOT_OBS])
PILOT_PV = np.asarray([PATS[o] for o in PILOT_OBS])
PILOT_PV32 = PILOT_PV.astype(np.complex64)
PILOT_BY_SYMBOL = tuple(
    tuple(np.asarray([b for ss, b in PILOT_OBS if ss == s], int))
    for s in range(F))
PILOT_VALUES_BY_SYMBOL = tuple(
    np.asarray([PATS[(s, int(b))] for b in PILOT_BY_SYMBOL[s]])
    for s in range(F))
PILOT_MASKS = tuple(PILOT_BV == b for b in PILOT_BINS)
PILOT_MASK_BY_BIN = {int(b): mask for b, mask in zip(PILOT_BINS, PILOT_MASKS)}
PILOT_BIN_INDEX = np.searchsorted(PILOT_BINS, PILOT_BV)


KNOTS = np.array([0, 4, 8, 12, 16, 20, F-1], float)
_BASIS = np.stack([np.interp(np.arange(F), KNOTS, np.eye(len(KNOTS))[k])
                   for k in range(len(KNOTS))], axis=1)          # (F, knots) hat basis
_BASIS32 = _BASIS.astype(np.float32)
_BASIS_LS = np.linalg.pinv(_BASIS)
_BASIS_LS32 = _BASIS_LS.astype(np.float32)

# The joint channel fit solves one independent two-coefficient least-squares
# problem per pilot bin.  Pack those observations once so the per-frame path
# can solve all pilot bins together instead of entering Python for every bin.
_PILOT_BIN_WIDTH = max(sum(b == pilot_bin for _, b in PILOT_OBS)
                       for pilot_bin in PILOT_BINS)
_PILOT_BIN_FREQUENCIES = np.asarray(PILOT_BINS)
_PILOT_BIN_SYMBOLS = np.zeros((len(PILOT_BINS), _PILOT_BIN_WIDTH), int)
_PILOT_BIN_VALID = np.zeros_like(_PILOT_BIN_SYMBOLS, dtype=bool)
_PILOT_BIN_VALUES = np.zeros(
    (len(PILOT_BINS), _PILOT_BIN_WIDTH, 2), dtype=complex)
_PILOT_BIN_LS = np.zeros(
    (len(PILOT_BINS), 2, _PILOT_BIN_WIDTH), dtype=complex)
for _bi, _pilot_bin in enumerate(PILOT_BINS):
    _pilot_symbols = [s for s, b in PILOT_OBS if b == _pilot_bin]
    _pilot_count = len(_pilot_symbols)
    _PILOT_BIN_SYMBOLS[_bi, :_pilot_count] = _pilot_symbols
    _PILOT_BIN_VALID[_bi, :_pilot_count] = True
    _pilot_values = np.asarray(
        [PATS[(s, _pilot_bin)] for s in _pilot_symbols])
    _PILOT_BIN_VALUES[_bi, :_pilot_count] = _pilot_values
    _gram = _pilot_values.conj().T @ _pilot_values
    _PILOT_BIN_LS[_bi, :, :_pilot_count] = np.linalg.solve(
        _gram, _pilot_values.conj().T)
_PILOT_BIN_OMEGA = (2j*np.pi/N)*_PILOT_BIN_FREQUENCIES[:, None]
_PILOT_BIN_J = (
    (2*np.pi/N*_PILOT_BIN_FREQUENCIES[:, None, None]) *
    _BASIS[_PILOT_BIN_SYMBOLS])
_PILOT_BIN_VALUES32 = _PILOT_BIN_VALUES.astype(np.complex64)
_PILOT_BIN_LS32 = _PILOT_BIN_LS.astype(np.complex64)
_PILOT_BIN_OMEGA32 = _PILOT_BIN_OMEGA.astype(np.complex64)
_PILOT_BIN_J32 = _PILOT_BIN_J.astype(np.float32)
for _table in (_BASIS_LS, _BASIS_LS32, _PILOT_BIN_FREQUENCIES,
               _PILOT_BIN_SYMBOLS, _PILOT_BIN_VALID,
               _PILOT_BIN_VALUES,
               _PILOT_BIN_LS, _PILOT_BIN_OMEGA, _PILOT_BIN_J,
               _PILOT_BIN_VALUES32, _PILOT_BIN_LS32,
               _PILOT_BIN_OMEGA32, _PILOT_BIN_J32):
    _table.setflags(write=False)


def _median_fast(values, axis=None):
    """Finite-array median without NumPy's general masked/NaN dispatch."""
    array = np.asarray(values)
    if axis is None:
        array = array.ravel()
        axis = 0
    length = array.shape[axis]
    middle = length//2
    kth = (middle,) if length % 2 else (middle-1, middle)
    partitioned = np.partition(array, kth, axis=axis)
    upper = np.take(partitioned, middle, axis=axis)
    if length % 2:
        return upper
    lower = np.take(partitioned, middle-1, axis=axis)
    return (lower+upper)*.5


def _unwrap_phase_fast(phase, axis=0):
    """Unwrap short phase tracks using their adjacent principal differences."""
    values = np.asarray(phase)
    values = np.moveaxis(values, axis, 0)
    differences = np.diff(values, axis=0)
    differences = (differences+np.pi) % (2*np.pi)-np.pi
    unwrapped = np.empty(values.shape, dtype=float)
    unwrapped[0] = values[0]
    unwrapped[1:] = values[0]+np.cumsum(differences, axis=0)
    return np.moveaxis(unwrapped, 0, axis)


# savgol_filter(x, PILOT_TONE_NARROW_WINDOW, 2, mode='interp') is linear in x;
# as a fixed (F, F) matrix it can run inside the compiled tone tracker.
_TONE_NARROW_SMOOTHER = np.ascontiguousarray(savgol_filter(
    np.eye(F), PILOT_TONE_NARROW_WINDOW, 2, mode='interp', axis=0))
_TONE_BINS_ARRAY = np.asarray(PILOT_TONE_BINS, dtype=np.int64)
_TONE_EMPTY_BINS = np.asarray((0, 2), dtype=np.int64)
for _table in (_TONE_NARROW_SMOOTHER, _TONE_BINS_ARRAY, _TONE_EMPTY_BINS):
    _table.setflags(write=False)


@njit(cache=True, fastmath=False)
def _pilot_tone_track_kernel(Z, scale, phase, early, tone_bins, empty_bins,
                             sym, n, narrow_snr_db, smoother, basis,
                             basis_ls, tone, stats, per_symbol, knot_track,
                             coherence):
    """Numeric core of the single-tone timing estimator.

    stats receives: [stage, index, narrow, empty_level, knot_rms,
    symbol_rms, amp0, amp1, snr0, snr1]; stage 0 = no tone above the empty
    bins, 1 = incoherent, 2 = usable track.
    """
    nsym = Z.shape[0]
    ntones = tone_bins.shape[0]
    for s in range(nsym):
        for t in range(ntones):
            b = tone_bins[t]
            rotation = scale*phase[s, b]*np.conj(early[b])
            tone[s, t] = (Z[s, b, 0]*rotation + Z[s, b, 1]*rotation)*.5
    empty_sum = 0.0
    for s in range(nsym):
        for e in range(empty_bins.shape[0]):
            b = empty_bins[e]
            rotation = scale*phase[s, b]*np.conj(early[b])
            empty_sum += abs((Z[s, b, 0]*rotation + Z[s, b, 1]*rotation)*.5)
    empty_level = max(empty_sum/(nsym*empty_bins.shape[0]), 1e-12)
    stats[3] = empty_level
    active = np.zeros(ntones, np.bool_)
    any_active = False
    min_snr = np.inf
    for t in range(ntones):
        total = 0.0
        for s in range(nsym):
            total += abs(tone[s, t])
        amplitude = total/nsym
        snr = 20*math.log10(max(amplitude, 1e-12)/empty_level)
        stats[6+t] = amplitude
        stats[8+t] = snr
        if amplitude >= 1e-5 and snr >= 6.0:
            active[t] = True
            any_active = True
            min_snr = min(min_snr, snr)
    if not any_active:
        stats[0] = 0
        return
    index = -1
    for t in range(ntones):
        if active[t] and tone_bins[t] == 3:
            index = t
    if index < 0:
        best = -1.0
        for t in range(ntones):
            if active[t] and stats[6+t] > best:
                best = stats[6+t]
                index = t
    stats[1] = index
    bin_index = tone_bins[index]
    advance = np.exp(-2j*np.pi*bin_index*sym/n)
    delta = np.empty(nsym)
    delta[0] = 0.0
    for s in range(1, nsym):
        step = tone[s, index]*np.conj(tone[s-1, index])*advance
        # An exactly-zero phasor (digitally muted symbol) has no phase.
        if step.real != 0.0 or step.imag != 0.0:
            delta[s] = delta[s-1]+math.atan2(step.imag, step.real)
        else:
            delta[s] = delta[s-1]
    for s in range(nsym):
        delta[s] = delta[s]*n/(2*np.pi*bin_index)
    mean = 0.0
    for s in range(nsym):
        mean += delta[s]
    mean /= nsym
    for s in range(nsym):
        delta[s] -= mean
    narrow = min_snr < narrow_snr_db
    stats[2] = 1.0 if narrow else 0.0
    for s in range(nsym):
        if narrow:
            acc = 0.0
            for k in range(nsym):
                acc += smoother[s, k]*delta[k]
            per_symbol[s] = acc
        else:
            left = delta[max(s-1, 0)]
            right = delta[min(s+1, nsym-1)]
            per_symbol[s] = .25*left + .5*delta[s] + .25*right
    nknots = basis.shape[1]
    theta = np.zeros(nknots)
    for k in range(nknots):
        acc = 0.0
        for s in range(nsym):
            acc += basis_ls[k, s]*per_symbol[s]
        theta[k] = acc
    knot_sq = 0.0
    symbol_sq = 0.0
    for s in range(nsym):
        acc = 0.0
        for k in range(nknots):
            acc += basis[s, k]*theta[k]
        knot_track[s] = acc
        knot_sq += (per_symbol[s]-acc)**2
        symbol_sq += (delta[s]-per_symbol[s])**2
    stats[4] = math.sqrt(knot_sq/nsym)
    stats[5] = math.sqrt(symbol_sq/nsym)
    step_phase = 2*np.pi/n
    worst = np.inf
    for t in range(ntones):
        total = 0j
        magnitude = 0.0
        for s in range(nsym):
            corrected = (tone[s, t] *
                         np.exp(-1j*(step_phase*sym*s*tone_bins[t])) *
                         np.exp(-1j*step_phase*per_symbol[s]*tone_bins[t]))
            total += corrected
            magnitude += abs(corrected)
        coherence[t] = abs(total)/max(magnitude, 1e-12)
        if active[t]:
            worst = min(worst, coherence[t])
    stats[0] = 1 if (worst < .2 or stats[4] > 1.0) else 2


def pilot_tone_timing(Z, model, counter, include_metadata=False,
                      estimator='single'):
    """Estimate the smooth within-frame timing residual from bins 1 and 3.

    Adjacent bin-3 phasors cancel the unknown absolute phase and yield timing
    increments; the lower-slope bin 1 remains an independent lock check. The
    constant delay is unobservable from the tones and is centered out. A weak
    or incoherent reference falls back to the established data-pilot fit.
    The float64 path runs the compiled core; _pilot_tone_timing_numpy is the
    reference and the float32 fallback.
    """
    if estimator not in ('single', 'joint'):
        raise ValueError(f'unknown pilot-tone estimator {estimator!r}')
    if estimator == 'joint':
        return _pilot_tone_timing_joint(
            Z, model, counter, include_metadata=include_metadata)
    if (include_metadata or not _is_float64_frame(Z) or
            tuple(PILOT_TONE_BINS) != (1, 3)):
        return _pilot_tone_timing_numpy(Z, model, counter,
                                        include_metadata=include_metadata)
    tone = np.empty((F, 2), np.complex128)
    stats = np.zeros(10)
    per_symbol = np.empty(F)
    knot_track = np.empty(F)
    coherence = np.zeros(2)
    _pilot_tone_track_kernel(
        Z, float(model.scale), model.phase, EARLY, _TONE_BINS_ARRAY,
        _TONE_EMPTY_BINS, SYM, N, PILOT_TONE_NARROW_SNR_DB,
        _TONE_NARROW_SMOOTHER, _BASIS, _BASIS_LS, tone, stats, per_symbol,
        knot_track, coherence)
    bins = _TONE_BINS_ARRAY
    tone_snr_db = stats[8:10]
    empty_level = float(stats[3])
    active = np.flatnonzero((stats[6:8] >= 1e-5) & (tone_snr_db >= 6.0))
    if stats[0] == 0:
        return None, {
            'detected': False, 'reason': 'tone_level',
            'tone_snr_db': tone_snr_db.tolist(),
            'empty_bin_level': empty_level,
        }
    narrow_track = bool(stats[2])
    knot_residual_rms = float(stats[4])
    symbol_residual_rms = float(stats[5])
    if stats[0] == 1:
        return None, {
            'detected': False, 'reason': 'tone_coherence',
            'coherence': coherence.tolist(),
            'tone_bins_used': bins[active].tolist(),
            'tone_snr_db': tone_snr_db.tolist(),
            'empty_bin_level': empty_level,
            'narrow_track': narrow_track,
            'timing_residual_rms': knot_residual_rms,
            'timing_sample_residual_rms': symbol_residual_rms,
        }
    return {'per_symbol': per_symbol, 'knot_fit': knot_track,
            'active': active}, {
        'detected': True,
        'coherence': coherence.tolist(),
        'tone_bins_used': bins[active].tolist(),
        'tone_snr_db': tone_snr_db.tolist(),
        'empty_bin_level': empty_level,
        'narrow_track': narrow_track,
        'timing_rms': float(np.sqrt(np.sum(per_symbol*per_symbol)/F)),
        'timing_peak': float(np.max(np.abs(per_symbol))),
        'timing_residual_rms': knot_residual_rms,
        'timing_sample_residual_rms': symbol_residual_rms,
    }


def _pilot_tone_timing_numpy(Z, model, counter, include_metadata=False):
    """Reference/fallback single-tone estimator (also the metadata variant)."""
    bins = np.asarray(PILOT_TONE_BINS, dtype=int)
    phase0 = np.asarray([PILOT_TONE_PHASES[int(k)] for k in bins])
    phase_step = 2*np.pi/N
    physical = (np.asarray(Z)[:, bins, :]*model.scale *
                model.phase[:, bins, None] *
                np.conj(EARLY[bins])[None, :, None])
    # Encoder tones are identical in L/R, hence M-only; averaging rejects
    # channel-specific noise while keeping the common reference.
    tone = np.sum(physical, axis=2)*.5
    amplitudes = np.sum(np.abs(tone), axis=0)/F
    # Bin 2 is deliberately left empty as a mains-hum guard; bin 0 is also
    # unoccupied. Compare each tone against those local empty-bin levels so a
    # coincidental carrier/noise phasor cannot be mistaken for a reference.
    empty_bins = np.asarray((0, 2), dtype=int)
    empty = (np.asarray(Z)[:, empty_bins, :]*model.scale *
             model.phase[:, empty_bins, None] *
             np.conj(EARLY[empty_bins])[None, :, None])
    empty_tone = np.sum(empty, axis=2)*.5
    empty_level = max(float(np.sum(np.abs(empty_tone))/(F*len(empty_bins))),
                      1e-12)
    tone_snr_db = 20*np.log10(np.maximum(amplitudes, 1e-12)/empty_level)
    active = np.flatnonzero(
        (amplitudes >= 1e-5) &
        (tone_snr_db >= 6.0))
    if not len(active):
        return None, {
            'detected': False, 'reason': 'tone_level',
            'tone_snr_db': tone_snr_db.tolist(),
            'empty_bin_level': empty_level,
        }

    # Bin 3 supplies fine timing over its ±N/6-sample unambiguous interval;
    # bin 1 remains an independent lock/coherence check. If bin 3 is removed
    # by the medium, fall back to the lower-slope bin 1 track.
    index = (1 if 3 in bins[active] else
             int(active[np.argmax(amplitudes[active])]))
    bin_index = int(bins[index])
    # Adjacent symbol phasors cancel unknown start time and fixed tone phase.
    # The remaining phase is only the within-packet timing increment, so a
    # single short cumulative sum replaces full-packet phase unwrapping.
    advance = np.exp(-2j*np.pi*bin_index*SYM/N)
    phase_steps = _phase_or_zero(
        tone[1:, index]*np.conj(tone[:-1, index])*advance)
    delta = np.r_[0.0, np.cumsum(phase_steps)]*N/(2*np.pi*bin_index)
    delta -= np.sum(delta)/len(delta)
    narrow_track = bool(np.min(tone_snr_db[active]) <
                        PILOT_TONE_NARROW_SNR_DB)
    if narrow_track:
        # Hum harmonics produce low-frequency beat phasors in the low bins.
        # Reduce the tracking bandwidth when the tone is still phase-coherent
        # but has weak SNR; high-SNR/flutter cases keep the wider 3-tap track.
        per_symbol = savgol_filter(
            delta, PILOT_TONE_NARROW_WINDOW, 2, mode='interp')
    else:
        padded = np.pad(delta, (1, 1), mode='edge')
        per_symbol = (.25*padded[:-2] + .5*padded[1:-1] +
                      .25*padded[2:])
    theta = _BASIS_LS@per_symbol
    knot_track = _BASIS@theta
    knot_residual = per_symbol-knot_track
    symbol_residual = delta-per_symbol
    symbol_phase = (phase_step*SYM*np.arange(F)[:, None]*bins[None, :])
    nominal_removed = tone*np.exp(-1j*symbol_phase)
    corrected = nominal_removed*np.exp(
        -1j*phase_step*per_symbol[:, None]*bins[None, :])
    coherence = (np.abs(np.sum(corrected, axis=0)) /
                 np.maximum(np.sum(np.abs(corrected), axis=0), 1e-12))
    knot_residual_rms = float(np.sqrt(np.sum(knot_residual*knot_residual)/F))
    symbol_residual_rms = float(np.sqrt(np.sum(symbol_residual*symbol_residual)/F))
    if np.min(coherence[active]) < .2 or knot_residual_rms > 1.0:
        return None, {
            'detected': False, 'reason': 'tone_coherence',
            'coherence': coherence.tolist(),
            'tone_bins_used': bins[active].tolist(),
            'tone_snr_db': tone_snr_db.tolist(),
            'empty_bin_level': empty_level,
            'narrow_track': narrow_track,
            'timing_residual_rms': knot_residual_rms,
            'timing_sample_residual_rms': symbol_residual_rms,
        }
    track = {'per_symbol': per_symbol, 'knot_fit': knot_track,
             'active': active}
    if include_metadata:
        nominal = ((int(counter)-1)*PULSE_FRAME + PULSE.SYNC_LEN +
                   np.arange(F)*SYM + WIN)
        expected = (phase_step*nominal[:, None]*bins[None, :] +
                    phase0[None, :])
        despread = tone*np.exp(-1j*expected)
        individual_phase = _unwrap_phase_fast(np.angle(despread), axis=0)
        tone_omegas = 2*np.pi*bins/N
        phase_intercepts = _median_fast(
            individual_phase-tone_omegas[None, :]*per_symbol[:, None], axis=0)
        body_centers = (PULSE.SYNC_LEN + np.arange(F)*SYM + WIN +
                        (N-1)/2)
        meta_center = PULSE.SYNC_LEN+FRAME+WIN+(N-1)/2
        fit_x = body_centers[-5:]
        fit_y = per_symbol[-5:]
        fit_x_center = float(np.mean(fit_x))
        fit_y_center = float(np.mean(fit_y))
        fit_slope = float(np.dot(fit_x-fit_x_center, fit_y-fit_y_center) /
                          np.dot(fit_x-fit_x_center, fit_x-fit_x_center))
        track.update({
            'phase_intercept': float(phase_intercepts[1]-phase_intercepts[0]),
            'phase_intercepts': phase_intercepts,
            'meta_delta': fit_y_center+fit_slope*(meta_center-fit_x_center),
        })
    return track, {
        'detected': True,
        'coherence': coherence.tolist(),
        'tone_bins_used': bins[active].tolist(),
        'tone_snr_db': tone_snr_db.tolist(),
        'empty_bin_level': empty_level,
        'narrow_track': narrow_track,
        'timing_rms': float(np.sqrt(np.sum(per_symbol*per_symbol)/F)),
        'timing_peak': float(np.max(np.abs(per_symbol))),
        'timing_residual_rms': knot_residual_rms,
        'timing_sample_residual_rms': symbol_residual_rms,
    }


def _pilot_tone_timing_joint(Z, model, counter, include_metadata=False):
    """Estimate timing increments jointly from both known tone frequencies.

    Differential phase removes each tone's unknown static channel phase. Bin 1
    supplies the wider-range branch decision, while bin 3 supplies finer
    timing resolution. A contaminated empty-bin guard is treated as a
    diagnostic outlier; phase coherence and agreement between the two timing
    estimates decide whether the joint track is usable.
    """
    bins = np.asarray(PILOT_TONE_BINS, dtype=int)
    phase0 = np.asarray([PILOT_TONE_PHASES[int(k)] for k in bins])
    phase_step = 2*np.pi/N
    spectrum = np.asarray(Z)
    physical = (spectrum[:, bins, :]*model.scale *
                model.phase[:, bins, None] *
                np.conj(EARLY[bins])[None, :, None])
    tone = np.sum(physical, axis=2)*.5
    amplitudes = np.mean(np.abs(tone), axis=0)

    # Bin 0 can contain DC; bin 2 is 750 Hz at the 48 kHz reference rate.
    # In particular, the 15th harmonic of 50 Hz can raise bin 2; 750 Hz is
    # not a harmonic of 60 Hz. Use the cleaner of these guards for a
    # conservative signal-presence ratio and retain both levels in diagnostics.
    empty_bins = np.asarray((0, 2), dtype=int)
    empty = (spectrum[:, empty_bins, :]*model.scale *
             model.phase[:, empty_bins, None] *
             np.conj(EARLY[empty_bins])[None, :, None])
    empty_tone = np.sum(empty, axis=2)*.5
    empty_levels = np.maximum(np.mean(np.abs(empty_tone), axis=0), 1e-12)
    empty_level = float(np.min(empty_levels))
    tone_snr_db = 20*np.log10(np.maximum(amplitudes, 1e-12)/empty_level)

    # Remove the known 144-sample symbol-to-symbol tone rotation. A true tone
    # remains coherent even with a varying timing offset; mains/DC leakage does
    # not generally follow both exact V7 phase advances.
    symbols = np.arange(F)
    nominal_phase = phase_step*SYM*symbols[:, None]*bins[None, :]
    nominal_removed = tone*np.exp(-1j*nominal_phase)
    nominal_coherence = (
        np.abs(np.sum(nominal_removed, axis=0)) /
        np.maximum(np.sum(np.abs(tone), axis=0), 1e-12))
    active = np.flatnonzero(
        (amplitudes >= 1e-5) & (tone_snr_db >= 6.0) &
        (nominal_coherence >= PILOT_TONE_JOINT_MIN_COHERENCE))
    base_metrics = {
        'tone_snr_db': tone_snr_db.tolist(),
        'nominal_coherence': nominal_coherence.tolist(),
        'empty_bin_level': empty_level,
        'empty_bin_levels': empty_levels.tolist(),
    }
    if not len(active):
        return None, {
            'detected': False, 'reason': 'tone_level_or_coherence',
            **base_metrics,
        }

    # The residual phase advance from one symbol to the next is
    # 2*pi*k*delta/N. Bin 3's estimate aliases every N/3 samples; the bin-1
    # estimate selects the matching branch before the precision-weighted fit.
    expected_advance = phase_step*SYM*bins
    phase_steps = _phase_or_zero(
        tone[1:, :]*np.conj(tone[:-1, :]) *
        np.exp(-1j*expected_advance)[None, :])
    step_samples = phase_steps*N/(2*np.pi*bins[None, :])
    dual_tone_disagreement = None
    if len(active) == 2 and np.array_equal(bins[active], (1, 3)):
        index1, index3 = int(active[0]), int(active[1])
        delta1 = step_samples[:, index1]
        delta3 = step_samples[:, index3]
        delta3 += np.rint((delta1-delta3)/(N/3))*(N/3)

        # Phase variance scales approximately as 1/SNR and timing variance as
        # 1/k^2, so weight the sample estimates by SNR*k^2 and phase coherence.
        snr_linear = np.clip(10**(tone_snr_db/10), .1, 1e3)
        weights = (snr_linear*bins.astype(float)**2 *
                   np.maximum(nominal_coherence, .05)**2)
        w1, w3 = weights[index1], weights[index3]
        per_step = (w1*delta1+w3*delta3)/(w1+w3)
        dual_tone_disagreement = float(np.sqrt(
            np.mean((delta1-delta3)**2)))
        if dual_tone_disagreement > PILOT_TONE_JOINT_MAX_DISAGREEMENT:
            return None, {
                'detected': False,
                'reason': 'dual_tone_timing_disagreement',
                'tone_bins_used': bins[active].tolist(),
                'dual_tone_timing_disagreement_samples': (
                    dual_tone_disagreement),
                **base_metrics,
            }
    else:
        # If one reference is rejected by the medium, retain a single-tone
        # fallback. Prefer bin 3 for precision; bin 1 remains the broad-range
        # fallback when bin 3 is lost.
        bin3 = np.flatnonzero(bins[active] == 3)
        index = int(active[bin3[0]]) if len(bin3) else int(active[0])
        per_step = step_samples[:, index]

    delta = np.r_[0.0, np.cumsum(per_step)]
    delta -= np.mean(delta)
    narrow_track = bool(np.min(tone_snr_db[active]) <
                        PILOT_TONE_NARROW_SNR_DB)
    if narrow_track:
        per_symbol = savgol_filter(
            delta, PILOT_TONE_NARROW_WINDOW, 2, mode='interp')
    else:
        padded = np.pad(delta, (1, 1), mode='edge')
        per_symbol = (.25*padded[:-2] + .5*padded[1:-1] +
                      .25*padded[2:])
    theta = _BASIS_LS@per_symbol
    knot_track = _BASIS@theta
    knot_residual = per_symbol-knot_track
    symbol_residual = delta-per_symbol
    corrected = nominal_removed*np.exp(
        -1j*phase_step*per_symbol[:, None]*bins[None, :])
    coherence = (np.abs(np.sum(corrected, axis=0)) /
                 np.maximum(np.sum(np.abs(corrected), axis=0), 1e-12))
    knot_residual_rms = float(np.sqrt(np.mean(knot_residual*knot_residual)))
    symbol_residual_rms = float(np.sqrt(np.mean(symbol_residual*symbol_residual)))
    metrics = {
        'coherence': coherence.tolist(),
        'tone_bins_used': bins[active].tolist(),
        'narrow_track': narrow_track,
        'timing_rms': float(np.sqrt(np.mean(per_symbol*per_symbol))),
        'timing_peak': float(np.max(np.abs(per_symbol))),
        'timing_residual_rms': knot_residual_rms,
        'timing_sample_residual_rms': symbol_residual_rms,
        'dual_tone_timing_disagreement_samples': dual_tone_disagreement,
        **base_metrics,
    }
    if np.min(coherence[active]) < .2 or knot_residual_rms > 1.0:
        return None, {
            'detected': False, 'reason': 'tone_coherence', **metrics,
        }

    track = {'per_symbol': per_symbol, 'knot_fit': knot_track,
             'active': active}
    if include_metadata:
        nominal = ((int(counter)-1)*PULSE_FRAME + PULSE.SYNC_LEN +
                   np.arange(F)*SYM + WIN)
        expected = phase_step*nominal[:, None]*bins[None, :] + phase0[None, :]
        despread = tone*np.exp(-1j*expected)
        individual_phase = _unwrap_phase_fast(np.angle(despread), axis=0)
        tone_omegas = 2*np.pi*bins/N
        phase_intercepts = _median_fast(
            individual_phase-tone_omegas[None, :]*per_symbol[:, None], axis=0)
        body_centers = (PULSE.SYNC_LEN + np.arange(F)*SYM + WIN +
                        (N-1)/2)
        meta_center = PULSE.SYNC_LEN+FRAME+WIN+(N-1)/2
        fit_x = body_centers[-5:]
        fit_y = per_symbol[-5:]
        fit_x_center = float(np.mean(fit_x))
        fit_y_center = float(np.mean(fit_y))
        fit_slope = float(np.dot(fit_x-fit_x_center, fit_y-fit_y_center) /
                          np.dot(fit_x-fit_x_center, fit_x-fit_x_center))
        track.update({
            'phase_intercept': float(phase_intercepts[1]-phase_intercepts[0]),
            'phase_intercepts': phase_intercepts,
            'meta_delta': fit_y_center+fit_slope*(meta_center-fit_x_center),
        })
    return track, {'detected': True, **metrics}


def pilot_tone_speed(samples, sample_rate, frame_start, frame_scale,
                     segments=6):
    """Measure raw-recording tone speed relative to the pulse header.

    The pulse count supplies the expected frequencies and removes each packet's
    large nominal phase slope. A short set of windowed in-phase projections
    then measures the residual phase slope for bins 1 and 3. This operates on
    the capture-rate samples, before pulse resampling can normalize the speed.
    """
    sample_rate = float(sample_rate)
    frame_scale = float(frame_scale)
    if (not np.isfinite(sample_rate) or sample_rate <= 0 or
            not np.isfinite(frame_scale) or frame_scale <= 0 or segments < 3):
        raise ValueError('invalid pilot speed measurement parameters')
    audio = np.asarray(samples, dtype=float)
    if audio.ndim == 1:
        mono = audio
    elif audio.ndim == 2 and audio.shape[1]:
        mono = audio.mean(axis=1)
    else:
        raise ValueError('pilot speed samples must be mono or multichannel')
    start = int(round(frame_start + PULSE.SYNC_LEN*frame_scale))
    stop = int(round(frame_start + (PULSE.SYNC_LEN+FRAME)*frame_scale))
    if start < 0 or stop > len(mono) or stop-start < segments*16:
        return {'detected': False, 'reason': 'body_out_of_window'}
    body = mono[start:stop]
    edges = np.linspace(0, len(body), segments+1, dtype=int)
    pulse_speed = sample_rate/(RATE*frame_scale)
    rms = float(np.sqrt(np.mean(body*body)))
    by_bin = []
    for bin_index in PILOT_TONE_BINS:
        expected_hz = bin_index*RATE/N*pulse_speed
        empty_hz = 2*RATE/N*pulse_speed
        phases, amplitudes, empty_amplitudes, centers = [], [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            if hi-lo < 16:
                continue
            window = np.hanning(hi-lo)
            chunk = body[lo:hi]
            chunk = chunk-float(np.mean(chunk))
            indexes = start+np.arange(lo, hi, dtype=float)
            phasor = np.sum(
                chunk*window*np.exp(-2j*np.pi*expected_hz*indexes/sample_rate))
            empty_phasor = np.sum(
                chunk*window*np.exp(-2j*np.pi*empty_hz*indexes/sample_rate))
            phases.append(float(np.angle(phasor)))
            amplitudes.append(float(2*np.abs(phasor)/max(np.sum(window), 1e-12)))
            empty_amplitudes.append(float(
                2*np.abs(empty_phasor)/max(np.sum(window), 1e-12)))
            centers.append((lo+hi-1)/(2*sample_rate))
        if len(phases) < 3:
            continue
        phase = np.unwrap(phases)
        slope = float(np.polyfit(centers, phase, 1)[0])
        speed = pulse_speed + slope/(2*np.pi*bin_index*RATE/N)
        amplitude = float(np.median(amplitudes))
        empty_level = float(np.median(empty_amplitudes))
        ratio = amplitude/max(np.sqrt(2)*rms, 1e-12)
        snr_db = float(20*np.log10(amplitude/max(empty_level, 1e-12)))
        by_bin.append({'bin': int(bin_index), 'speed': float(speed),
                       'amplitude_ratio': ratio,
                       'empty_bin_amplitude': empty_level,
                       'tone_snr_db': snr_db})
    valid = [entry for entry in by_bin
             if entry['amplitude_ratio'] >= .03 and
             entry['tone_snr_db'] >= 6.0]
    if not valid:
        return {'detected': False, 'reason': 'tone_level',
                'pulse_speed': pulse_speed, 'per_bin': by_bin,
                'tone_snr_db': [entry['tone_snr_db'] for entry in by_bin]}
    # Bin 3 has the larger timing slope and is the refinement reference. Bin 1
    # remains a coarse lock/consistency check; if bin 3 is absent, retain the
    # single-tone fallback.
    refine = next((entry for entry in valid if entry['bin'] == 3), None)
    tone_speed = float((refine or valid[0])['speed'])
    return {
        'detected': True,
        'pulse_speed': float(pulse_speed),
        'tone_speed': tone_speed,
        'difference_pct': float(100*(tone_speed/pulse_speed-1)),
        'tone_snr_db': [entry['tone_snr_db'] for entry in by_bin],
        'per_bin': by_bin,
    }


def _pilot_metadata_offset(body, samples, meta_start, scale, model, counter,
                           force_float32=False, estimator='single',
                           metadata_indexes=None):
    """Estimate metadata-symbol timing from the continuing dual-tone phase."""
    windows = np.asarray(body).reshape(F, SYM, 2)[:, WIN:WIN+N, :]
    if force_float32:
        Z = (np.fft.rfft(windows, axis=1)/np.float32(model.scale) *
             np.conj(model.phase32)[:, :, None] *
             EARLY.astype(np.complex64)[None, :, None]).astype(np.complex64)
    else:
        Z = (np.fft.rfft(windows, axis=1)/model.scale *
             np.conj(model.phase)[:, :, None]*EARLY[None, :, None])
    track, metrics = pilot_tone_timing(
        Z, model, counter, include_metadata=True, estimator=estimator)
    if track is None:
        return None, {'used': False, 'reason': 'body_tones_unavailable',
                      'tone_metrics': metrics}

    meta_indexes = (meta_start+np.arange(META_SYMBOL)*scale
                    if metadata_indexes is None else
                    np.asarray(metadata_indexes, dtype=float))
    if meta_indexes[-1] >= len(samples)-1:
        return None, {'used': False, 'reason': 'metadata_out_of_window',
                      'tone_metrics': metrics}
    meta = _sample_at(samples, meta_indexes, taps=4)
    window = meta[WIN:WIN+N]
    if force_float32:
        z = (np.fft.rfft(window.astype(np.float32), axis=0) /
             np.float32(model.scale) *
             np.conj(model.phase32[-1])[:, None] *
             EARLY.astype(np.complex64)[:, None]).astype(np.complex64)
    else:
        z = (np.fft.rfft(window, axis=0)/model.scale *
             np.conj(model.phase[-1])[:, None]*EARLY[:, None])
    bins = np.asarray(PILOT_TONE_BINS, dtype=int)
    phase0 = np.asarray([PILOT_TONE_PHASES[int(k)] for k in bins])
    tone = (z[bins]*model.scale*model.phase[-1, bins, None]*
            np.conj(EARLY[bins])[:, None]).mean(axis=1)
    reference = float(np.median(np.abs(z[BINS]*model.scale)))
    active = track['active']
    if np.any(np.abs(tone[active]) < max(reference*.08, 1e-5)):
        return None, {'used': False, 'reason': 'metadata_tone_level',
                      'tone_metrics': metrics}

    nominal = ((int(counter)-1)*PULSE_FRAME + PULSE.SYNC_LEN + FRAME + WIN)
    expected = 2*np.pi*nominal*bins/N+phase0
    despread = tone*np.exp(-1j*expected)
    if len(active) == 2:
        measured_phase = float(np.angle(despread[1]*np.conj(despread[0])))
        omega = 2*np.pi*(bins[1]-bins[0])/N
        predicted_phase = (track['phase_intercept']+
                           omega*track['meta_delta'])
    else:
        index = int(active[np.argmax(np.abs(tone[active]))])
        measured_phase = float(np.angle(despread[index]))
        omega = 2*np.pi*bins[index]/N
        predicted_phase = (track['phase_intercepts'][index] +
                           omega*track['meta_delta'])
    measured_phase += 2*np.pi*round(
        (predicted_phase-measured_phase)/(2*np.pi))
    offset = (measured_phase-predicted_phase)/omega
    if not np.isfinite(offset) or abs(offset) > 16:
        return None, {'used': False, 'reason': 'metadata_tone_offset_range',
                      'offset_samples': float(offset),
                      'tone_metrics': metrics}
    return float(offset), {
        'used': True, 'offset_samples': float(offset),
        'tone_metrics': metrics,
    }


def channel_joint(Z, iters=2, force_float32=False, pilot_timing='baseline',
                  model=None, counter=None, return_timing_diag=False):
    """Fit channel and timing, optionally using the experimental low-bin tones.

    ``tone-seeded`` initializes the established Gauss-Newton pilot fit from
    the original bin-3 tone track. ``tone-joint`` seeds it from a joint bin-1/
    bin-3 estimate. ``tone-replaced`` holds timing to the selected single-tone
    track while ordinary data pilots continue to estimate the channel.
    """
    modes = ('baseline', 'tone-seeded', 'tone-joint', 'tone-replaced')
    if pilot_timing not in modes:
        raise ValueError(f'unknown pilot timing mode {pilot_timing!r}')
    if pilot_timing == 'tone-joint':
        return _channel_joint_tone_joint(
            Z, iters, force_float32, model, counter, return_timing_diag)
    tone_delta = None
    timing_diag = {'mode_requested': pilot_timing,
                   'mode_applied': 'baseline'}
    if pilot_timing != 'baseline':
        if model is None or counter is None:
            raise ValueError('tone timing requires model and counter')
        tone_delta, metrics = pilot_tone_timing(Z, model, counter)
        timing_diag.update(metrics)
        weak_tone = min(metrics.get('tone_snr_db', [np.inf])) < \
            PILOT_TONE_NARROW_SNR_DB
        max_residual = (.8 if weak_tone else .5)
        residual_key = ('timing_sample_residual_rms'
                        if pilot_timing == 'tone-replaced'
                        else 'timing_residual_rms')
        if pilot_timing == 'tone-seeded' and not weak_tone:
            max_residual = .35
        if (tone_delta is not None and
                timing_diag.get(residual_key, np.inf) > max_residual):
            tone_delta = None
            timing_diag.update({
                'reason': 'timing_quality_gate',
                'timing_accepted': False,
            })
        if tone_delta is not None:
            timing_diag['timing_accepted'] = True
            candidate = _channel_joint_batched(
                Z, iters, force_float32,
                tone_delta=tone_delta['knot_fit']
                if pilot_timing == 'tone-seeded'
                else tone_delta['per_symbol'],
                tone_replaced=(pilot_timing == 'tone-replaced'))
            candidate_score, candidate_timing_residual = _channel_residuals(
                Z, candidate)
            timing_diag.update({
                'tone_pilot_residual': candidate_score,
                'tone_timing_residual_samples': candidate_timing_residual,
                'comparison_fit_performed': False,
            })
            # A clean pilot fit and small phase-derived residual can qualify
            # without a second full channel fit. Ambiguous candidates trigger
            # the baseline fit below, preserving the old relative quality gate
            # while keeping the common accepted path within the CPU budget.
            if (pilot_timing != 'tone-replaced' and
                    candidate_score <= PILOT_TONE_MAX_PILOT_RESIDUAL and
                    candidate_timing_residual <=
                    PILOT_TONE_MAX_CHANNEL_TIMING_RESIDUAL):
                result = candidate
                timing_diag['mode_applied'] = pilot_timing
                timing_diag['timing_quality_gate'] = 'absolute'
            else:
                baseline = _channel_joint_batched(Z, iters, force_float32)
                base_score, baseline_timing_residual = _channel_residuals(
                    Z, baseline)
                timing_diag.update({
                    'baseline_pilot_residual': base_score,
                    'baseline_timing_residual_samples': baseline_timing_residual,
                    'comparison_fit_performed': True,
                })
                if (candidate_score <= base_score*1.05 and
                        candidate_timing_residual <=
                        baseline_timing_residual*1.05+1e-8):
                    result = candidate
                    timing_diag['mode_applied'] = pilot_timing
                    timing_diag['timing_quality_gate'] = 'relative'
                else:
                    result = baseline
                    timing_diag['mode_applied'] = 'baseline'
                    timing_diag['timing_accepted'] = False
                    timing_diag['reason'] = (
                        'pilot_residual_gate'
                        if candidate_score > base_score*1.05
                        else 'timing_residual_gate')
        else:
            result = _channel_joint_batched(Z, iters, force_float32)
            (timing_diag['baseline_pilot_residual'],
             timing_diag['baseline_timing_residual_samples']) = (
                 _channel_residuals(Z, result))
    else:
        result = _channel_joint_batched(Z, iters, force_float32)
    if return_timing_diag:
        if pilot_timing == 'baseline':
            (timing_diag['baseline_pilot_residual'],
             timing_diag['baseline_timing_residual_samples']) = (
                 _channel_residuals(Z, result))
        use_tone = (pilot_timing != 'baseline' and
                    timing_diag.get('mode_applied') == pilot_timing)
        timing_diag['pilot_residual'] = (
            timing_diag.get('tone_pilot_residual') if use_tone else
            timing_diag.get('baseline_pilot_residual'))
        timing_diag['timing_residual_samples'] = (
            timing_diag.get('tone_timing_residual_samples') if use_tone else
            timing_diag.get('baseline_timing_residual_samples'))
        return result, timing_diag
    return result


def _channel_joint_tone_joint(Z, iters, force_float32, model, counter,
                              return_timing_diag):
    """Use the joint tone track only when the ordinary pilot fit needs help."""
    if model is None or counter is None:
        raise ValueError('tone timing requires model and counter')
    result = _channel_joint_batched(Z, iters, force_float32)
    base_score, base_timing = _channel_residuals(Z, result)
    timing_diag = {
        'mode_requested': 'tone-joint',
        'mode_applied': 'baseline',
        'baseline_pilot_residual': base_score,
        'baseline_timing_residual_samples': base_timing,
        'comparison_fit_performed': True,
    }

    # Avoid paying for tone extraction when the data pilots already show a
    # stable timing fit, or when their broad residual says the problem is not a
    # cleanly separable timing error. Metadata CRC retry remains independent.
    if base_score > PILOT_TONE_MAX_PILOT_RESIDUAL:
        timing_diag.update({
            'detected': False, 'tone_evaluation_skipped': True,
            'reason': 'baseline_pilot_residual_high',
        })
    elif base_timing <= PILOT_TONE_JOINT_BASELINE_TIMING_TRIGGER:
        timing_diag.update({
            'detected': False, 'tone_evaluation_skipped': True,
            'reason': 'baseline_timing_fit_sufficient',
        })
    else:
        tone_delta, metrics = pilot_tone_timing(
            Z, model, counter, estimator='joint')
        timing_diag.update(metrics)
        if tone_delta is not None:
            weak_tone = min(metrics.get('tone_snr_db', [np.inf])) < \
                PILOT_TONE_NARROW_SNR_DB
            max_residual = .8 if weak_tone else .5
            if metrics.get('timing_residual_rms', np.inf) > max_residual:
                tone_delta = None
                timing_diag.update({
                    'reason': 'timing_quality_gate',
                    'timing_accepted': False,
                })
        if tone_delta is not None:
            candidate = _channel_joint_batched(
                Z, 1, force_float32, tone_delta=tone_delta['knot_fit'])
            candidate_score, candidate_timing = _channel_residuals(
                Z, candidate)
            timing_diag.update({
                'tone_pilot_residual': candidate_score,
                'tone_timing_residual_samples': candidate_timing,
                'timing_accepted': True,
            })
            tolerance = PILOT_TONE_JOINT_RELATIVE_TOLERANCE
            if (candidate_score <= base_score*(1+tolerance) and
                    candidate_timing <= base_timing*(1+tolerance)+1e-8):
                result = candidate
                timing_diag.update({
                    'mode_applied': 'tone-joint',
                    'timing_quality_gate': 'baseline_relative',
                    'timing_fit_iterations': 1,
                })
            else:
                timing_diag.update({
                    'timing_accepted': False,
                    'reason': ('pilot_residual_gate'
                               if candidate_score > base_score*(1+tolerance)
                               else 'timing_residual_gate'),
                })

    use_tone = timing_diag['mode_applied'] == 'tone-joint'
    timing_diag['pilot_residual'] = (
        timing_diag.get('tone_pilot_residual') if use_tone else base_score)
    timing_diag['timing_residual_samples'] = (
        timing_diag.get('tone_timing_residual_samples')
        if use_tone else base_timing)
    if return_timing_diag:
        return result, timing_diag
    return result


def _phase_or_zero(values):
    """np.angle, but an exactly-zero phasor has phase 0.

    A digitally muted symbol (a dropout to exact zero) yields 0+0j products
    whose np.angle is 0 or +-pi depending on the signs of the zeros, i.e. on
    arithmetic order. Treating them as carrying no phase keeps the timing
    track and the Gauss-Newton step independent of that accident.
    """
    values = np.asarray(values)
    return np.where(values == 0, 0.0, np.angle(values))


_PILOT_BINS_ARRAY = np.asarray(PILOT_BINS)
_PILOT_BINS_ARRAY.setflags(write=False)


def _is_float64_frame(Z, H=None):
    """The compiled kernels cover the default float64 path only."""
    return (np.asarray(Z).dtype == np.complex128 and
            (H is None or np.asarray(H).dtype == np.complex128))


@njit(cache=True, fastmath=False)
def _channel_residuals_kernel(Z, H, symbols, bins, values, n):
    """Pilot misfit and phase-derived timing residual in one pass."""
    numerator = 0.0
    denominator = 0.0
    weight_sum = 0.0
    weighted = 0.0
    for index in range(symbols.shape[0]):
        symbol = symbols[index]
        pilot_bin = bins[index]
        for channel in range(2):
            predicted = (H[symbol, pilot_bin, channel, 0]*values[index, 0] +
                         H[symbol, pilot_bin, channel, 1]*values[index, 1])
            observed = Z[symbol, pilot_bin, channel]
            error = abs(observed-predicted)
            numerator += error*error
            magnitude = abs(observed)
            denominator += magnitude*magnitude
            weight = abs(predicted)*magnitude
            product = observed*np.conj(predicted)
            timing = (math.atan2(product.imag, product.real)*n /
                      (2*np.pi*pilot_bin))
            weight_sum += weight
            weighted += weight*timing*timing
    return (numerator/max(denominator, 1e-12),
            math.sqrt(weighted/max(weight_sum, 1e-12)))


def _channel_residuals(Z, H):
    """Compute pilot misfit and timing residual together in one pass."""
    if _is_float64_frame(Z, H):
        return _channel_residuals_kernel(
            Z, H, PILOT_SV, PILOT_BV, PILOT_PV, N)
    return _channel_residuals_numpy(Z, H)


def _channel_residuals_numpy(Z, H):
    """Float32/reference pair sharing pilot prediction and indexing work."""
    predicted = np.einsum(
        'nci,ni->nc', H[PILOT_SV, PILOT_BV], PILOT_PV)
    observed = Z[PILOT_SV, PILOT_BV]
    numerator = float(np.sum(np.abs(observed-predicted)**2))
    denominator = float(np.sum(np.abs(observed)**2))
    pilot_residual = numerator/max(denominator, 1e-12)
    weights = np.abs(predicted)*np.abs(observed)
    timing_error = (np.angle(observed*np.conj(predicted))*N /
                    (2*np.pi*PILOT_BV[:, None]))
    timing_residual = float(np.sqrt(np.sum(weights*timing_error**2) /
                                    max(float(np.sum(weights)), 1e-12)))
    return pilot_residual, timing_residual


def _channel_pilot_residual(Z, H):
    return float(_channel_residuals(Z, H)[0])


def _channel_timing_residual(Z, H):
    """Weighted data-pilot phase residual expressed in reference samples."""
    return float(_channel_residuals(Z, H)[1])


def _channel_pilot_residual_numpy(Z, H):
    predicted = np.einsum(
        'nci,ni->nc', H[PILOT_SV, PILOT_BV], PILOT_PV)
    observed = Z[PILOT_SV, PILOT_BV]
    numerator = float(np.sum(np.abs(observed-predicted)**2))
    denominator = float(np.sum(np.abs(observed)**2))
    return numerator/max(denominator, 1e-12)


def _channel_timing_residual_numpy(Z, H):
    """Reference/fallback for the compiled residual kernel."""
    predicted = np.einsum(
        'nci,ni->nc', H[PILOT_SV, PILOT_BV], PILOT_PV)
    observed = Z[PILOT_SV, PILOT_BV]
    weights = np.abs(predicted)*np.abs(observed)
    timing_error = (np.angle(observed*np.conj(predicted))*N /
                    (2*np.pi*PILOT_BV[:, None]))
    return float(np.sqrt(np.sum(weights*timing_error**2) /
                         max(float(np.sum(weights)), 1e-12)))


@njit(cache=True, fastmath=False)
def _solve_spd(matrix, rhs):
    """Cholesky solve of a small symmetric positive-definite system (the
    Gauss-Newton normal equations: J^T W J + eps*I)."""
    size = rhs.shape[0]
    lower = np.zeros((size, size))
    for i in range(size):
        for j in range(i+1):
            acc = matrix[i, j]
            for k in range(j):
                acc -= lower[i, k]*lower[j, k]
            lower[i, j] = math.sqrt(acc) if i == j else acc/lower[j, j]
    forward = np.empty(size)
    for i in range(size):
        acc = rhs[i]
        for k in range(i):
            acc -= lower[i, k]*forward[k]
        forward[i] = acc/lower[i, i]
    out = np.empty(size)
    for i in range(size-1, -1, -1):
        acc = forward[i]
        for k in range(i+1, size):
            acc -= lower[k, i]*out[k]
        out[i] = acc/lower[i, i]
    return out


@njit(cache=True, fastmath=False)
def _phasor_powers(delta, n, powers):
    """powers[s, m] = exp(2j*pi*m*delta[s]/n) by repeated multiplication.

    One exp per symbol instead of one per (bin, symbol); the product chain
    over at most ~35 bins stays within a few 1e-15 of the direct exp.
    """
    for s in range(delta.shape[0]):
        step = np.exp((2j*np.pi/n)*delta[s])
        value = 1.0+0j
        for m in range(powers.shape[1]):
            powers[s, m] = value
            value *= step


@njit(cache=True, fastmath=False)
def _channel_joint_kernel(Z, iters, theta0, tone_replaced, tone_delta,
                          bin_symbols, bin_frequencies, bin_valid, values,
                          operator, omega, jacobian, basis, bins, pilot_bins,
                          n, eps, threshold, step_tolerance, H):
    """Joint pilot channel + timing fit (§9.3) for both receive tracks.

    Same weighted Gauss-Newton as _channel_joint_batched_numpy: each pilot
    bin's two-coefficient response is a fixed linear operator of the
    de-rotated observations, and the timing knots take Gauss-Newton steps on
    pilot phase errors weighted by the lesser of predicted and observed
    magnitude.
    """
    nbins, width = bin_symbols.shape
    nknots = basis.shape[1]
    nsym = basis.shape[0]
    y = np.empty((nbins, width), np.complex128)
    rot = np.empty((nbins, width), np.complex128)
    h = np.empty((nbins, 2), np.complex128)
    delta = np.empty(nsym)
    phase_table = np.empty((nsym, bins.shape[0]), np.complex128)
    top = max(bins.max(), bin_frequencies.max())
    powers = np.empty((nsym, top+1), np.complex128)
    target = bins.astype(np.float64)
    known = pilot_bins.astype(np.float64)
    for channel in range(2):
        for b in range(nbins):
            for k in range(width):
                y[b, k] = Z[bin_symbols[b, k], bin_frequencies[b], channel]
        theta = theta0.copy()
        rounds = 1 if tone_replaced else iters
        for _ in range(rounds):
            if tone_replaced:
                for s in range(nsym):
                    delta[s] = tone_delta[s]
            else:
                for s in range(nsym):
                    acc = 0.0
                    for knot in range(nknots):
                        acc += basis[s, knot]*theta[knot]
                    delta[s] = acc
            # exp(2j*pi*bin*delta/n) as integer powers of one phasor per symbol
            _phasor_powers(delta, n, powers)
            for b in range(nbins):
                for k in range(width):
                    rot[b, k] = powers[bin_symbols[b, k], bin_frequencies[b]]
                for column in range(2):
                    acc = 0j
                    for k in range(width):
                        acc += operator[b, column, k]*np.conj(rot[b, k])*y[b, k]
                    h[b, column] = acc
            if tone_replaced:
                break
            normal = np.zeros((nknots, nknots))
            rhs = np.zeros(nknots)
            for b in range(nbins):
                for k in range(width):
                    predicted = rot[b, k]*(values[b, k, 0]*h[b, 0] +
                                           values[b, k, 1]*h[b, 1])
                    # A dropped observation has an arbitrary phase even though
                    # the pilot model still predicts a normal magnitude. Do
                    # not let that symbol pull the packet-wide timing fit.
                    weight = min(abs(predicted), abs(y[b, k]))
                    if not bin_valid[b, k] or weight <= threshold:
                        continue
                    ratio = y[b, k]/predicted
                    phase = (math.atan2(ratio.imag, ratio.real)
                             if ratio.real != 0.0 or ratio.imag != 0.0
                             else 0.0)
                    for i in range(nknots):
                        jw = jacobian[b, k, i]*weight
                        rhs[i] += jw*phase*weight
                        for j in range(nknots):
                            normal[i, j] += jw*jacobian[b, k, j]*weight
            for i in range(nknots):
                normal[i, i] += eps
            step = _solve_spd(normal, rhs)
            largest = 0.0
            for i in range(nknots):
                theta[i] += step[i]
                largest = max(largest, abs(step[i]))
            if largest < step_tolerance:
                break
        if tone_replaced:
            for s in range(nsym):
                delta[s] = tone_delta[s]
        else:
            for s in range(nsym):
                acc = 0.0
                for knot in range(nknots):
                    acc += basis[s, knot]*theta[knot]
                delta[s] = acc
        # One timing rotation per (symbol, bin), shared by both columns.
        _phasor_powers(delta, n, powers)
        for s in range(nsym):
            for j in range(bins.shape[0]):
                phase_table[s, j] = powers[s, bins[j]]
        for column in range(2):
            real = np.interp(target, known, h[:, column].real.copy())
            imag = np.interp(target, known, h[:, column].imag.copy())
            for s in range(nsym):
                for j in range(bins.shape[0]):
                    H[s, bins[j], channel, column] = (
                        (real[j]+1j*imag[j])*phase_table[s, j])


def _channel_joint_batched(Z, iters, force_float32, tone_delta=None,
                           tone_replaced=False):
    if force_float32 or not _is_float64_frame(Z):
        return _channel_joint_batched_numpy(
            Z, iters, force_float32, tone_delta=tone_delta,
            tone_replaced=tone_replaced)
    if tone_delta is None or tone_replaced:
        theta0 = np.zeros(len(KNOTS))
    else:
        theta0 = _BASIS_LS@np.asarray(tone_delta, dtype=np.float64)
    delta = (np.asarray(tone_delta, dtype=np.float64) if tone_replaced
             else np.zeros(F))
    # Bins outside BINS are never read; zeros keep them deterministic.
    H = np.zeros((F, 65, 2, 2), dtype=np.complex128)
    _channel_joint_kernel(
        Z, int(iters), theta0, bool(tone_replaced), delta,
        _PILOT_BIN_SYMBOLS, _PILOT_BIN_FREQUENCIES, _PILOT_BIN_VALID,
        _PILOT_BIN_VALUES, _PILOT_BIN_LS, _PILOT_BIN_OMEGA, _PILOT_BIN_J,
        _BASIS, BINS, _PILOT_BINS_ARRAY, N, 1e-10, 1e-9, 1e-4, H)
    return H


def _channel_joint_batched_numpy(Z, iters, force_float32, tone_delta=None,
                                 tone_replaced=False):
    if force_float32:
        real_dtype, complex_dtype = np.float32, np.complex64
        basis = _BASIS32
        values, operator = _PILOT_BIN_VALUES32, _PILOT_BIN_LS32
        omega, jacobian = _PILOT_BIN_OMEGA32, _PILOT_BIN_J32
        eps = np.float32(1e-10)
        threshold = np.float32(1e-9)
        step_tolerance = np.float32(1e-4)
    else:
        real_dtype, complex_dtype = np.float64, np.complex128
        basis = _BASIS
        values, operator = _PILOT_BIN_VALUES, _PILOT_BIN_LS
        omega, jacobian = _PILOT_BIN_OMEGA, _PILOT_BIN_J
        eps, threshold, step_tolerance = 1e-10, 1e-9, 1e-4

    H = np.empty((F, 65, 2, 2), dtype=complex_dtype)
    for c in range(2):
        y = np.asarray(Z[_PILOT_BIN_SYMBOLS,
                         _PILOT_BIN_FREQUENCIES[:, None], c],
                       dtype=complex_dtype)
        if tone_delta is None or tone_replaced:
            theta = np.zeros(len(KNOTS), dtype=real_dtype)
        else:
            basis_operator = (_BASIS_LS32 if force_float32 else _BASIS_LS)
            theta = basis_operator@np.asarray(tone_delta, dtype=real_dtype)
        if tone_replaced:
            delta = np.asarray(tone_delta, dtype=real_dtype)
            rot = np.exp(omega*delta[_PILOT_BIN_SYMBOLS]).astype(
                complex_dtype, copy=False)
            h = np.einsum('bkn,bn->bk', operator, np.conj(rot)*y)
        else:
            for _ in range(iters):
                delta = basis @ theta
                rot = np.exp(omega*delta[_PILOT_BIN_SYMBOLS]).astype(
                    complex_dtype, copy=False)
                # The pilot phase rotation has unit magnitude, so it changes
                # only the right-hand side; each bin's Gram matrix is static.
                h = np.einsum('bkn,bn->bk', operator, np.conj(rot)*y)
                pred = rot*np.einsum('bni,bi->bn', values, h)
                ok = _PILOT_BIN_VALID & (np.abs(pred) > threshold)
                rho = np.divide(y, pred, out=np.zeros_like(y), where=ok)
                ph = _phase_or_zero(rho).astype(real_dtype, copy=False)
                # Clamp timing-fit weight by the measured pilot magnitude so
                # transient erasures do not act like zero-phase observations.
                w = np.where(ok, np.minimum(np.abs(pred), np.abs(y)), 0).astype(
                    real_dtype, copy=False)

                # This is the same weighted Gauss-Newton fit as the
                # observation-ordered loop: both sides are weighted by w².
                JW = jacobian*w[..., None]
                normal = np.einsum('bni,bnj->ij', JW, JW)
                normal += eps*np.eye(JW.shape[-1], dtype=real_dtype)
                step = np.linalg.solve(
                    normal, np.einsum('bni,bn->i', JW, ph*w))
                theta += step
                if np.max(np.abs(step)) < step_tolerance:
                    break

        delta = (np.asarray(tone_delta, dtype=real_dtype) if tone_replaced
                 else basis @ theta)
        phase = np.exp((2j*np.pi/N)*BINS[None, :]*delta[:, None]).astype(
            complex_dtype, copy=False)
        for k in range(2):
            v = h[:, k]
            hk = (np.interp(BINS, PILOT_BINS, v.real) +
                  1j*np.interp(BINS, PILOT_BINS, v.imag)).astype(
                      complex_dtype, copy=False)
            H[:, BINS, c, k] = hk[None, :]*phase
    return H


def _channel_joint_float32(Z, iters=2):
    """Float32 channel estimate used by the opt-in ARM receiver path."""
    return _channel_joint_batched(Z, iters, True)


# fade_and_noise() works on every symbol and channel at once. Pilot counts
# differ per symbol (4 or 5), so the per-symbol pilot lists are padded to a
# fixed width with a validity mask.
_PMAX = max(len(p) for p in PILOT_BY_SYMBOL)
PILOT_PAD_BINS = np.zeros((F, _PMAX), int)
PILOT_PAD_VALUES = np.zeros((F, _PMAX, 2), complex)
PILOT_PAD_VALID = np.zeros((F, _PMAX), bool)
for _s, (_bins, _vals) in enumerate(zip(PILOT_BY_SYMBOL, PILOT_VALUES_BY_SYMBOL)):
    PILOT_PAD_BINS[_s, :len(_bins)] = _bins
    PILOT_PAD_VALUES[_s, :len(_bins)] = _vals
    PILOT_PAD_VALID[_s, :len(_bins)] = True
PILOT_PAD_KHZ = PILOT_PAD_BINS*RATE/N/1000
PILOT_PAD_VALUES32 = PILOT_PAD_VALUES.astype(np.complex64)
PILOT_PAD_KHZ32 = PILOT_PAD_KHZ.astype(np.float32)
PILOT_DOF = np.maximum(PILOT_PAD_VALID.sum(axis=1)-3, 1)
_BINS_KHZ = BINS*RATE/N/1000
# The compiled fade refit steps the correction geometrically across the
# occupied bins, which needs them evenly spaced.
assert np.allclose(np.diff(_BINS_KHZ), _BINS_KHZ[1]-_BINS_KHZ[0])
_SYMBOL_INDEX = np.arange(F)[:, None]


def _tone_reference_gain(Z, u, v):
    """Estimate a packet-relative broadband gain track from the M-only tones.

    Tone amplitude is normalized to the current packet's body RMS, which the
    receiver does not know. Centering each tone across symbols removes that
    unknown constant and its static channel response, leaving the per-symbol
    M-path gain change. The data pilots still estimate the static response,
    S path, and frequency-dependent loss.
    """
    bins = np.asarray(PILOT_TONE_BINS, dtype=int)
    tone_frequency = bins*RATE/N/1000
    magnitude = np.abs(np.asarray(Z)[:, bins, :])
    empty = np.abs(np.asarray(Z)[:, (0, 2), :])
    empty_level = np.median(empty, axis=(0, 1))
    snr_db = 20*np.log10(
        np.maximum(magnitude, 1e-12)/np.maximum(empty_level[None, None, :],
                                                1e-12))
    valid = (snr_db >= 6.0) & (magnitude >= 1e-5)
    correction = np.zeros((F, 2), dtype=float)
    channel_diag = []

    for channel in range(2):
        observations = np.log(np.maximum(magnitude[:, :, channel], 1e-12))
        predicted = (np.asarray(u)[:, channel, None] -
                     np.asarray(v)[:, channel, None]*tone_frequency[None, :])
        residual = np.zeros((F, len(bins)), dtype=float)
        for tone in range(len(bins)):
            active = valid[:, tone, channel]
            if not np.any(active):
                continue
            residual[:, tone] = (
                observations[:, tone]-np.median(observations[active, tone]) -
                predicted[:, tone]+np.median(predicted[active, tone]))

        active = valid[:, :, channel]
        active_symbols = np.any(active, axis=1)
        count = int(np.sum(active_symbols))
        tone_counts = [int(np.sum(active[:, tone]))
                       for tone in range(len(bins))]
        disagreement = None
        both = active[:, 0] & active[:, 1]
        if np.any(both):
            disagreement = float(np.sqrt(np.mean(
                np.square(residual[both, 0]-residual[both, 1]))))

        diag = {
            'used': False,
            'valid_symbols': count,
            'valid_symbols_by_tone': tone_counts,
            'mean_snr_db': [
                (float(np.mean(snr_db[active[:, tone], tone, channel]))
                 if tone_counts[tone] else None)
                for tone in range(len(bins))],
            'two_tone_disagreement_rms': disagreement,
        }
        if count < F//2:
            diag['reason'] = 'insufficient_tone_reference'
            channel_diag.append(diag)
            continue
        if disagreement is not None and disagreement > .20:
            diag['reason'] = 'tone_gain_tracks_disagree'
            channel_diag.append(diag)
            continue

        weights = np.where(active, 1/(1+10**(-snr_db[:, :, channel]/10)), 0)
        weight_sum = weights.sum(axis=1)
        track = np.divide(
            np.sum(weights*residual, axis=1), weight_sum,
            out=np.zeros(F, dtype=float), where=weight_sum > 0)
        valid_at = np.flatnonzero(active_symbols)
        if len(valid_at) < F:
            track = np.interp(np.arange(F), valid_at, track[valid_at])
        track -= float(np.median(track[active_symbols]))
        if F >= 5:
            track = savgol_filter(track, 5, 2, mode='interp')
            track -= float(np.median(track[active_symbols]))
        rms = float(np.sqrt(np.mean(np.square(track[active_symbols]))))
        if rms < .01:
            diag['reason'] = 'tone_gain_track_negligible'
            diag['rms_log_gain'] = rms
            channel_diag.append(diag)
            continue

        # The reference only constrains the common M-path gain. Keep a poor
        # low-band estimate from extrapolating into an excessive wideband EQ.
        track = np.clip(track, -.22, .22)
        correction[:, channel] = track
        diag.update({
            'used': True,
            'rms_log_gain': rms,
            'max_gain_correction_db': float(
                np.max(np.abs(track))*20/np.log(10)),
        })
        channel_diag.append(diag)

    return correction, {
        'mode': 'm-reference',
        'used': any(item['used'] for item in channel_diag),
        'channels': channel_diag,
    }


def _apply_tone_reference_gain(Z, H, gain_track, diagnostic):
    """Apply tone gain only where the known OFDM pilots agree with it."""
    if not diagnostic.get('used'):
        return H
    valid = PILOT_PAD_VALID
    obs = Z[_SYMBOL_INDEX, PILOT_PAD_BINS]
    accepted_any = False
    for ch, channel_diag in enumerate(diagnostic['channels']):
        if not channel_diag.get('used'):
            channel_diag['accepted_symbols'] = 0
            continue
        pilot_h = H[_SYMBOL_INDEX, PILOT_PAD_BINS, ch]
        predicted = np.einsum('spi,spi->sp', pilot_h, PILOT_PAD_VALUES)
        m_component = pilot_h[:, :, 0]*PILOT_PAD_VALUES[:, :, 0]
        factor = np.exp(gain_track[:, ch])
        candidate = predicted+(factor[:, None]-1)*m_component
        baseline_loss = np.where(valid, np.abs(obs[:, :, ch]-predicted)**2,
                                 0).sum(axis=1)
        candidate_loss = np.where(valid,
                                  np.abs(obs[:, :, ch]-candidate)**2,
                                  0).sum(axis=1)
        # A tone reference is additional training, not permission to worsen
        # the existing known OFDM training fit. This per-symbol check also
        # rejects low-band gain tracks that do not predict the occupied band.
        accepted = candidate_loss <= baseline_loss*1.02+1e-10
        accepted_factor = np.where(accepted, factor, 1.0)
        H[:, BINS, ch, 0] *= accepted_factor[:, None]
        channel_diag.update({
            'accepted_symbols': int(np.sum(accepted)),
            'rejected_symbols': int(F-np.sum(accepted)),
        })
        accepted_any |= bool(np.any(accepted & (np.abs(gain_track[:, ch]) > .01)))
    diagnostic['used'] = accepted_any
    return H


@njit(cache=True, fastmath=False)
def _fade_and_noise_kernel(Z, H, pad_bins, pad_values, pad_valid, pad_khz,
                           dof, bins, frequency, pilot_amp, noise):
    """Compiled §9.4 fade refit and §9.5 noise, same rules as the NumPy path.

    Updates H in place (occupied bins) and fills noise (F, 2).
    """
    nsym, width = pad_bins.shape
    apply = np.zeros((nsym, 2), np.bool_)
    predicted = np.empty(width, np.complex128)
    magnitude = np.empty(width)
    for s in range(nsym):
        for channel in range(2):
            largest = 0.0
            for p in range(width):
                if pad_valid[s, p]:
                    predicted[p] = (
                        H[s, pad_bins[s, p], channel, 0]*pad_values[s, p, 0] +
                        H[s, pad_bins[s, p], channel, 1]*pad_values[s, p, 1])
                    magnitude[p] = abs(predicted[p])
                else:
                    predicted[p] = 0j
                    magnitude[p] = 0.0
                largest = max(largest, magnitude[p])
            count = 0
            fmax = -np.inf
            fmin = np.inf
            s00 = 0.0
            s01 = 0.0
            s11 = 0.0
            r0 = 0.0
            r1 = 0.0
            weight_sum = 0.0
            weighted_log = 0.0
            phasor = 0j
            for p in range(width):
                if not pad_valid[s, p] or magnitude[p] <= .3*largest:
                    continue
                count += 1
                ratio = Z[s, pad_bins[s, p], channel]/predicted[p]
                weight = magnitude[p]
                log_ratio = math.log(abs(ratio)+1e-12)
                khz = pad_khz[s, p]
                fmax = max(fmax, khz)
                fmin = min(fmin, khz)
                w2 = weight*weight
                s00 += w2
                s01 -= w2*khz
                s11 += w2*khz*khz
                r0 += w2*log_ratio
                r1 -= w2*khz*log_ratio
                weight_sum += weight
                weighted_log += weight*log_ratio
                phasor += ratio*weight
            s00 += 1e-10
            s11 += 1e-10
            if count >= 3 and fmax-fmin > 3:
                det = s00*s11-s01*s01
                u = (s11*r0-s01*r1)/det
                v = max((s00*r1-s01*r0)/det, 0.0)
            else:
                u = weighted_log/weight_sum if weight_sum > 0 else 0.0
                v = 0.0
            if count >= 2:
                apply[s, channel] = True
                angle = math.atan2(phasor.imag, phasor.real)
                # exp(u - v*f + j*angle) over evenly spaced bins: one exp for
                # the first bin, then a real geometric step per bin.
                correction = np.exp(u-v*frequency[0]+1j*angle)
                step = math.exp(-v*(frequency[1]-frequency[0]))
                for j in range(bins.shape[0]):
                    H[s, bins[j], channel, 0] *= correction
                    H[s, bins[j], channel, 1] *= correction
                    correction *= step
    raw = np.zeros((nsym, 2))
    for s in range(nsym):
        for channel in range(2):
            if not apply[s, channel]:
                continue
            residual = 0.0
            for p in range(width):
                if pad_valid[s, p]:
                    fitted = (
                        H[s, pad_bins[s, p], channel, 0]*pad_values[s, p, 0] +
                        H[s, pad_bins[s, p], channel, 1]*pad_values[s, p, 1])
                    error = abs(Z[s, pad_bins[s, p], channel]-fitted)
                    residual += error*error
            raw[s, channel] = residual/dof[s]
    smooth = np.empty((nsym, 2))
    for channel in range(2):
        if nsym == 1:
            smooth[0, channel] = raw[0, channel]
        else:
            smooth[0, channel] = (raw[0, channel]+raw[1, channel])/3
            smooth[nsym-1, channel] = (raw[nsym-2, channel] +
                                       raw[nsym-1, channel])/3
            for s in range(1, nsym-1):
                smooth[s, channel] = (raw[s-1, channel]+raw[s, channel] +
                                      raw[s+1, channel])/3
        median = np.median(smooth[:, channel].copy())
        power = 0.0
        for s in range(nsym):
            for j in range(bins.shape[0]):
                for column in range(2):
                    value = abs(H[s, bins[j], channel, column])
                    power += value*value
        floor = 1e-5*pilot_amp*pilot_amp*power/(nsym*bins.shape[0]*2)
        for s in range(nsym):
            noise[s, channel] = max(max(smooth[s, channel], median), floor)


def fade_and_noise(Z, H, force_float32=False, tone_reference=False,
                   return_tone_diag=False):
    """Per-symbol magnitude/phase refit (§9.4) and pilot-residual noise (§9.5).

    The float64 path runs the compiled kernel; the float32 path and the opt-in
    tone reference keep the NumPy implementation, which is also the reference
    the kernel is tested against (modem_tests/test_v7_decode_speed).
    """
    if force_float32:
        return _fade_and_noise_float32(
            Z, H, tone_reference=tone_reference,
            return_tone_diag=return_tone_diag)
    if tone_reference or not _is_float64_frame(Z, H):
        return _fade_and_noise_numpy(Z, H, tone_reference=tone_reference,
                                     return_tone_diag=return_tone_diag)
    noise = np.empty((F, 2))
    _fade_and_noise_kernel(Z, H, PILOT_PAD_BINS, PILOT_PAD_VALUES,
                           PILOT_PAD_VALID, PILOT_PAD_KHZ, PILOT_DOF, BINS,
                           _BINS_KHZ, PILOT_AMP, noise)
    if return_tone_diag:
        return H, noise, {'mode': 'off', 'used': False}
    return H, noise


def _fade_and_noise_numpy(Z, H, tone_reference=False, return_tone_diag=False):
    """Reference/fallback float64 fade/noise refit (batched NumPy)."""
    freq = BINS*RATE/N/1000
    valid = PILOT_PAD_VALID[:, :, None]                                # (F, P, 1)
    Hp = H[_SYMBOL_INDEX, PILOT_PAD_BINS]                              # (F, P, 2ch, 2)
    pred = np.einsum('spci,spi->spc', Hp, PILOT_PAD_VALUES)            # (F, P, 2ch)
    obs = Z[_SYMBOL_INDEX, PILOT_PAD_BINS]                             # (F, P, 2ch)
    mag = np.where(valid, np.abs(pred), 0.0)
    ok = valid & (mag > 0.3*mag.max(axis=1, keepdims=True))
    count = ok.sum(axis=1)                                             # (F, 2ch)
    rho = np.divide(obs, pred, out=np.ones_like(obs), where=ok)
    w = np.where(ok, mag, 0.0)
    logr = np.where(ok, np.log(np.abs(rho)+1e-12), 0.0)
    fb = np.broadcast_to(PILOT_PAD_KHZ[:, :, None], ok.shape)
    fmax = np.where(ok, fb, -np.inf).max(axis=1)
    fmin = np.where(ok, fb, np.inf).min(axis=1)
    use_ls = (count >= 3) & (fmax - fmin > 3)
    # Weighted least squares of log|rho| on [1, -f] (rows scaled by w), as a
    # closed-form 2x2 solve with the loop's 1e-10 ridge.
    w2 = w*w
    s00 = w2.sum(axis=1) + 1e-10
    s01 = -(w2*fb).sum(axis=1)
    s11 = (w2*fb*fb).sum(axis=1) + 1e-10
    r0 = (w2*logr).sum(axis=1)
    r1 = -(w2*fb*logr).sum(axis=1)
    det = s00*s11 - s01*s01
    safe = np.where(use_ls, det, 1.0)
    u_ls = (s11*r0 - s01*r1)/safe
    v_ls = np.maximum((s00*r1 - s01*r0)/safe, 0.0)
    wsum = w.sum(axis=1)
    u_avg = np.divide((w*logr).sum(axis=1), wsum, out=np.zeros_like(wsum), where=wsum > 0)
    u = np.where(use_ls, u_ls, u_avg)
    v = np.where(use_ls, v_ls, 0.0)
    a = np.angle((rho*w).sum(axis=1))
    apply = count >= 2                                                 # else: no refit
    corr = np.exp(u[:, None, :] - v[:, None, :]*freq[None, :, None] + 1j*a[:, None, :])
    corr = np.where(apply[:, None, :], corr, 1.0)                      # (F, bins, 2ch)
    H[:, BINS] *= corr[..., None]
    tone_diag = {'mode': 'off', 'used': False}
    if tone_reference:
        tone_gain, tone_diag = _tone_reference_gain(Z, u, v)
        _apply_tone_reference_gain(Z, H, tone_gain, tone_diag)
    Hp2 = H[_SYMBOL_INDEX, PILOT_PAD_BINS]
    pred2 = np.einsum('spci,spi->spc', Hp2, PILOT_PAD_VALUES)
    resid = np.where(valid, np.abs(obs-pred2)**2, 0.0).sum(axis=1)
    noise = np.where(apply, resid/PILOT_DOF[:, None], 0.0)
    # Smooth over +-1 symbol, without launching one tiny convolution per
    # channel/symbol.  The explicit edge handling matches np.convolve(...,
    # mode='same') with zero outside the frame closely enough for this noise
    # floor, while avoiding a surprising amount of dispatch overhead on M4.
    sm = np.empty_like(noise)
    if F == 1:
        sm[:] = noise
    else:
        sm[0] = (noise[0]+noise[1])/3
        sm[-1] = (noise[-2]+noise[-1])/3
        sm[1:-1] = (noise[:-2]+noise[1:-1]+noise[2:])/3
    for ch in range(2):
        noise[:, ch] = np.maximum(
            np.maximum(sm[:, ch], np.median(sm[:, ch])),
            1e-5*PILOT_AMP**2*np.mean(np.abs(H[:, BINS, ch])**2))
    if return_tone_diag:
        return H, noise, tone_diag
    return H, noise


def _stereo_erasure_mask(noise):
    """Flag localized symbol/channel pilot outliers in the stereo pair.

    A short dropout is a time-local erasure, not a packet-wide noise penalty.
    Residual spikes relative to that leg's packet median identify isolated
    bursts when the paired leg is not also an outlier. This leaves common-mode
    channel-fit excursions to the normal quality gate without requiring the
    paired leg to be perfectly clean. The paired-leg ratio also catches a
    dropout that persists through most of a packet, where a within-packet
    median is no longer a useful reference.
    """
    values = np.asarray(noise)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError('stereo pilot noise must have shape (symbols, 2)')
    limit = float(ERASURE_PILOT_NOISE)
    ratio = 8.0
    local_limit = np.maximum(limit, ratio*np.median(values, axis=0))
    local_outlier = values > local_limit[None, :]
    erase = np.zeros(values.shape, dtype=bool)
    for channel, paired in ((0, 1), (1, 0)):
        erase[:, channel] = (
            (local_outlier[:, channel] & ~local_outlier[:, paired]) |
            ((values[:, channel] > limit) &
             (values[:, channel] >
              ratio*np.maximum(values[:, paired], NOISE_FLOOR)) &
             (values[:, paired] <= limit)))
    return erase


def _pilot_reference_power(H):
    """Per input channel: the mean received power of a unit cell on the
    pilot bins (1.0 for a unity-gain link at the sender's level)."""
    power = np.abs(H[:, _PILOT_BINS_ARRAY])**2
    reference = power.sum(axis=3).mean(axis=(0, 1))
    return np.maximum(reference, NOISE_FLOOR).astype(H.real.dtype)


def _effective_pilot_noise(noise, erased):
    """Per-leg mean residual after excluding localized erasures.

    A leg erased for every symbol is unavailable, not infinitely noisy: the
    other leg may still carry a complete M/S observation through the joint
    equalizer. Report that case at the live rejection boundary so quality
    gates judge the surviving observation rather than the absent one.
    """
    noise = np.asarray(noise)
    erased = np.asarray(erased, dtype=bool)
    effective = np.empty(2, dtype=float)
    for channel in range(2):
        good = ~erased[:, channel]
        effective[channel] = (float(np.mean(noise[good, channel]))
                              if np.any(good) else
                              float(LIVE_MAX_PILOT_NOISE))
    return effective


def _fade_and_noise_float32(Z, H, tone_reference=False,
                            return_tone_diag=False):
    """Float32 version of fade/noise refit for the opt-in receiver path."""
    freq = (BINS*RATE/N/1000).astype(np.float32)
    valid = PILOT_PAD_VALID[:, :, None]
    Hp = H[_SYMBOL_INDEX, PILOT_PAD_BINS]
    pred = np.einsum('spci,spi->spc', Hp, PILOT_PAD_VALUES32)
    obs = Z[_SYMBOL_INDEX, PILOT_PAD_BINS]
    mag = np.where(valid, np.abs(pred), np.float32(0))
    ok = valid & (mag > np.float32(.3)*mag.max(axis=1, keepdims=True))
    count = ok.sum(axis=1)
    rho = np.divide(obs, pred, out=np.ones_like(obs), where=ok)
    w = np.where(ok, mag, np.float32(0))
    logr = np.where(ok, np.log(np.abs(rho)+np.float32(1e-12)), np.float32(0))
    fb = np.broadcast_to(PILOT_PAD_KHZ32[:, :, None], ok.shape)
    fmax = np.where(ok, fb, -np.inf).max(axis=1)
    fmin = np.where(ok, fb, np.inf).min(axis=1)
    use_ls = (count >= 3) & (fmax-fmin > np.float32(3))
    w2 = w*w
    s00 = w2.sum(axis=1) + np.float32(1e-10)
    s01 = -(w2*fb).sum(axis=1)
    s11 = (w2*fb*fb).sum(axis=1) + np.float32(1e-10)
    r0 = (w2*logr).sum(axis=1)
    r1 = -(w2*fb*logr).sum(axis=1)
    det = s00*s11-s01*s01
    safe = np.where(use_ls, det, np.float32(1))
    u_ls = (s11*r0-s01*r1)/safe
    v_ls = np.maximum((s00*r1-s01*r0)/safe, np.float32(0))
    wsum = w.sum(axis=1)
    u_avg = np.divide((w*logr).sum(axis=1), wsum,
                      out=np.zeros_like(wsum), where=wsum > 0)
    u = np.where(use_ls, u_ls, u_avg)
    v = np.where(use_ls, v_ls, np.float32(0))
    a = np.angle((rho*w).sum(axis=1)).astype(np.float32)
    apply = count >= 2
    corr = np.exp(u[:, None, :] - v[:, None, :]*freq[None, :, None] +
                  np.complex64(1j)*a[:, None, :]).astype(np.complex64)
    corr = np.where(apply[:, None, :], corr, np.complex64(1))
    H[:, BINS] *= corr[..., None]
    tone_diag = {'mode': 'off', 'used': False}
    if tone_reference:
        tone_gain, tone_diag = _tone_reference_gain(Z, u, v)
        _apply_tone_reference_gain(Z, H, tone_gain, tone_diag)
    Hp2 = H[_SYMBOL_INDEX, PILOT_PAD_BINS]
    pred2 = np.einsum('spci,spi->spc', Hp2, PILOT_PAD_VALUES32)
    resid = np.where(valid, np.abs(obs-pred2)**2, np.float32(0)).sum(axis=1)
    noise = np.where(apply, resid/PILOT_DOF.astype(np.float32)[:, None],
                     np.float32(0)).astype(np.float32)
    sm = np.empty_like(noise)
    if F == 1:
        sm[:] = noise
    else:
        sm[0] = (noise[0]+noise[1])/np.float32(3)
        sm[-1] = (noise[-2]+noise[-1])/np.float32(3)
        sm[1:-1] = (noise[:-2]+noise[1:-1]+noise[2:])/np.float32(3)
    for ch in range(2):
        noise[:, ch] = np.maximum(
            np.maximum(sm[:, ch], np.median(sm[:, ch])),
            np.float32(1e-5)*np.float32(PILOT_AMP)**2*np.mean(
                np.abs(H[:, BINS, ch])**2).astype(np.float32))
    if return_tone_diag:
        return H, noise, tone_diag
    return H, noise


@njit(cache=True, fastmath=False)
def _equalizer_cell_estimates(Z, H, noise, block_symbols, block_bins,
                              block_prior, noise_floor, est, var):
    """Per-cell stereo MMSE estimates and variances."""
    nblocks, nsym = block_symbols.shape
    for b in range(nblocks):
        k_bin = block_bins[b]
        for t in range(nsym):
            symbol = block_symbols[b, t]
            h00 = H[symbol, k_bin, 0, 0]
            h01 = H[symbol, k_bin, 0, 1]
            h10 = H[symbol, k_bin, 1, 0]
            h11 = H[symbol, k_bin, 1, 1]
            z0 = Z[symbol, k_bin, 0]
            z1 = Z[symbol, k_bin, 1]
            n0 = max(noise[symbol, 0], noise_floor)
            n1 = max(noise[symbol, 1], noise_floor)
            for q in range(2):
                p0 = block_prior[b, 0, q]
                p1 = block_prior[b, 1, q]
                q0 = p0+1e-12
                q1 = p1+1e-12
                s00 = q0*(h00*np.conj(h00))+q1*(h01*np.conj(h01))+n0
                s11 = q0*(h10*np.conj(h10))+q1*(h11*np.conj(h11))+n1
                s01 = q0*h00*np.conj(h10)+q1*h01*np.conj(h11)
                s10 = q0*h10*np.conj(h00)+q1*h11*np.conj(h01)
                # S is Hermitian: s00, s11 and s01*s10 = |s01|^2 are real,
                # so one real reciprocal replaces four complex divisions.
                inverse = 1.0/(s00.real*s11.real-(s01*s10).real)
                i00 = s11*inverse
                i01 = -s01*inverse
                i10 = -s10*inverse
                i11 = s00*inverse
                for k in range(2):
                    pk = q0 if k == 0 else q1
                    prior_k = p0 if k == 0 else p1
                    hk0 = h00 if k == 0 else h01
                    hk1 = h10 if k == 0 else h11
                    w0 = pk*(np.conj(hk0)*i00+np.conj(hk1)*i10)
                    w1 = pk*(np.conj(hk0)*i01+np.conj(hk1)*i11)
                    estimate = w0*z0+w1*z1
                    beta = (w0*hk0+w1*hk1).real
                    variance = max(beta*pk-beta*beta*prior_k, 1e-12)
                    if beta > 1e-6:
                        est[b, k, q, t] = estimate/beta
                        var[b, k, q, t] = variance/(beta*beta)


@njit(cache=True, fastmath=False)
def _equalizer_group_solve(est, var, group_block, group_stream, group_q,
                           ranks, lam, system, weighted, xhat, conf, got):
    """Per-group Cholesky/LMMSE solve and confidence assembly."""
    ngroups = ranks.shape[0]
    S = np.empty((8, 8))
    L = np.zeros((8, 8))
    X = np.empty((8, 9))
    lam_g = np.empty(8)
    y = np.empty(8)
    sig = np.empty(8)
    for group in range(ngroups):
        block = group_block[group]
        stream = group_stream[group]
        quadrature = group_q[group]
        for j in range(8):
            value = est[block, stream, quadrature, j]
            y[j] = value.real if quadrature == 0 else value.imag
            variance = var[block, stream, quadrature, j]
            sig[j] = variance/2 if np.isfinite(variance) else 1e9
            rank = ranks[group, j]
            lam_g[j] = lam[rank] if rank >= 0 else 1e-12

        # A Lam A^T depends only on the rank table (_group_systems); a frame
        # adds only its per-member noise on the diagonal.
        for i in range(8):
            for k in range(8):
                S[i, k] = system[group, i, k]
            S[i, i] += sig[i]

        # Solve the group system by Cholesky with all 8 estimates as RHS.
        for i in range(8):
            for j in range(i+1):
                acc = S[i, j]
                for k in range(j):
                    acc -= L[i, k]*L[j, k]
                if i == j:
                    L[i, i] = np.sqrt(acc)
                else:
                    L[i, j] = acc/L[j, j]
        for i in range(8):
            for column in range(9):
                acc = (weighted[group, i, column]
                       if column < 8 else y[i])
                for k in range(i):
                    acc -= L[i, k]*X[k, column]
                X[i, column] = acc/L[i, i]
        for j in range(8):
            rank = ranks[group, j]
            if rank < 0:
                continue
            estimate = 0.0
            weight = 0.0
            for i in range(8):
                estimate += X[i, j]*X[i, 8]
                weight += X[i, j]*X[i, j]
            confidence = 1.0-(lam_g[j]-weight)/lam_g[j]
            xhat[rank] = estimate
            conf[rank] = min(max(confidence, 0.0), 1.0)
            got[rank] = True


@njit(cache=True, fastmath=False)
def _equalize_numba_kernel(Z, H, noise, block_symbols, block_bins,
                           block_prior, group_block, group_stream, group_q,
                           ranks, lam, system, weighted, noise_floor, xhat,
                           conf, got):
    """Fused production dispatcher for cell estimates and group solves."""
    nblocks, nsym = block_symbols.shape
    est = np.zeros((nblocks, 2, 2, nsym), np.complex128)
    var = np.full((nblocks, 2, 2, nsym), np.inf)
    _equalizer_cell_estimates(
        Z, H, noise, block_symbols, block_bins, block_prior, noise_floor,
        est, var)
    _equalizer_group_solve(
        est, var, group_block, group_stream, group_q, ranks, lam,
        system, weighted, xhat, conf, got)


def _equalize_numba(model, Z, H, noise, counter):
    """Run the fused compiled float64 equalizer; float32 keeps the NumPy path."""
    phase = counter % TAIL_PHASES
    block_prior = model.block_prior_tables[phase]
    ranks = model.rank_tables[phase]
    xhat = np.zeros_like(model.mu)
    conf = np.zeros_like(model.mu)
    got = np.zeros(model.mu.shape, np.bool_)
    _equalize_numba_kernel(
        np.ascontiguousarray(Z, dtype=np.complex128),
        np.ascontiguousarray(H, dtype=np.complex128),
        np.ascontiguousarray(noise, dtype=np.float64),
        BLOCK_SYMBOLS, BLOCK_BINS, block_prior,
        GROUP_BLOCK, GROUP_STREAM_INDEX, GROUP_Q_INDEX,
        ranks, model.lam, *_group_systems(model, phase),
        NOISE_FLOOR, xhat, conf, got)
    return xhat, conf, got


def _group_systems(model, phase):
    """Per tail phase: A Lam A^T and A Lam for every Hadamard group.

    A = H8 diag(gain) over the group's ranks (zero where a slot is empty);
    both are fixed by the model and rank table, so they are built once per
    model and phase and cached on the model.
    """
    cache = model.__dict__.setdefault('_group_system_cache', {})
    if phase not in cache:
        ranks = model.rank_tables[phase]
        live = ranks >= 0
        lam = np.where(live, model.lam[np.maximum(ranks, 0)], 1e-12)
        gain = np.where(live, model.gain[np.maximum(ranks, 0)], 0.)
        A = H8[None, :, :]*gain[:, None, :]
        weighted = np.ascontiguousarray(A*lam[:, None, :])
        system = np.ascontiguousarray(weighted @ A.transpose(0, 2, 1))
        for table in (weighted, system):
            table.setflags(write=False)
        cache[phase] = (system, weighted)
    return cache[phase]


def warmup_equalizer(model):
    """Compile every Numba decoder kernel before a real-time receiver opens
    audio: metadata, channel fit, residuals, fade/noise and the equalizer."""
    shape = (F, 65)
    # LiveInput supplies float32 audio, while the default decoder uses the
    # float64 model phase. Warm exactly that production metadata signature.
    _decode_metadata_spectrum(
        np.zeros((65, 2), np.complex64), model.phase[-1], EARLY,
        model.scale, META_PILOTS, META_DATA_BINS)
    Z = np.zeros(shape+(2,), np.complex128)
    Z[:, BINS] = 1
    H = _channel_joint_batched(Z, 2, False)
    _channel_joint_batched(Z, 2, False, tone_delta=np.zeros(F))
    _channel_joint_batched(Z, 2, False, tone_delta=np.zeros(F),
                           tone_replaced=True)
    _channel_residuals(Z, H)
    Z[:, _TONE_BINS_ARRAY] = 1
    pilot_tone_timing(Z, model, 1)
    H, noise = fade_and_noise(Z, H)
    _equalize_numba(model, Z, H, noise, 1)
    # Acquisition kernels (pulse edges, EOF counter, sample reads) compile per
    # dtype; decode a short float32 stream -- the live receiver's type -- at
    # 1x and slowed (the slow path refines pulse edges with 16-tap reads).
    values = np.zeros(model.coder.source_count)
    wire = encode_pulse_stream(model, [values]*3, pilot_tones=True,
                               ).astype(np.float32)
    # Reverse packets stay as negative-stride views at unity gain. Compile
    # that sample-reader layout here so first live reverse playback cannot
    # stall the capture callback for JIT work.
    _sample_at(wire[::-1], np.asarray([PULSE.SYNC_LEN+16.5]), taps=4)
    for stream in (wire, speed_pulse_stream(wire, .8).astype(np.float32)):
        decode_pulse_stream(model, stream, 
                            pilot_timing='tone-seeded')
        decode_pulse_stream(model, stream, latest_only=True,
                            pilot_timing='tone-seeded')


warmup_decoder = warmup_equalizer


def _equalize_numpy(model, Z, H, noise, counter, force_float32=False):
    """Reference/fallback equalizer, retained for the float32 decode path."""
    idx = model.rank_tables[counter % TAIL_PHASES]
    if force_float32:
        block_prior = (model.block_prior_tables32[counter % TAIL_PHASES]
                       if model.block_prior_tables32 else
                       block_priors(model.gain32, model.lam32, idx).astype(
                           np.float32))
    else:
        block_prior = (model.block_prior_tables[counter % TAIL_PHASES]
                       if model.block_prior_tables else
                       block_priors(model.gain, model.lam, idx))
    Hc = H[BLOCK_SYMBOLS, BLOCK_BINS[:, None]]
    Zc = Z[BLOCK_SYMBOLS, BLOCK_BINS[:, None]]
    real_dtype = np.float32 if force_float32 else float
    complex_dtype = np.complex64 if force_float32 else complex
    estimates = np.zeros((len(BLOCKS), 2, 2, 8), dtype=complex_dtype)
    variances = np.full((len(BLOCKS), 2, 2, 8), np.inf, dtype=real_dtype)
    for q in range(2):
        prior = block_prior[:, :, q]
        safe = prior+(np.float32(1e-12) if force_float32 else 1e-12)
        S = ((Hc*safe[:, None, None, :]) @
             Hc.conj().transpose(0, 1, 3, 2))
        S[..., 0, 0] += np.maximum(noise[BLOCK_SYMBOLS, 0], NOISE_FLOOR)
        S[..., 1, 1] += np.maximum(noise[BLOCK_SYMBOLS, 1], NOISE_FLOOR)
        HP = Hc*safe[:, None, None, :]
        W = _solve_2x2_mat(S, HP).conj().transpose(0, 1, 3, 2)
        xt = np.einsum('btij,btj->bti', W, Zc)
        B = W@Hc
        cov = W@S@W.conj().transpose(0, 1, 3, 2)
        beta = np.diagonal(B, axis1=-2, axis2=-1).real
        var = np.maximum(np.diagonal(cov, axis1=-2, axis2=-1).real-
                         beta**2*prior[:, None, :],
                         np.float32(1e-12) if force_float32 else 1e-12)
        beta_k = beta.transpose(0, 2, 1)
        estimates[:, :, q] = np.divide(
            xt.transpose(0, 2, 1), beta_k,
            out=np.zeros_like(xt.transpose(0, 2, 1)), where=beta_k > 1e-6)
        variances[:, :, q] = np.divide(
            var.transpose(0, 2, 1), beta_k**2,
            out=np.full_like(var.transpose(0, 2, 1), np.inf),
            where=beta_k > 1e-6)
    live_all = idx >= 0
    group_count = len(GROUPS)
    y_all = np.empty((group_count, 8), dtype=real_dtype)
    sig_all = np.empty((group_count, 8), dtype=real_dtype)
    values_all = estimates[GROUP_BLOCK, GROUP_STREAM_INDEX, GROUP_Q_INDEX]
    vars_all = variances[GROUP_BLOCK, GROUP_STREAM_INDEX, GROUP_Q_INDEX]
    y_all[:] = values_all.real
    y_all[GROUP_Q_INDEX == 1] = values_all[GROUP_Q_INDEX == 1].imag
    sig_all[:] = np.where(np.isfinite(vars_all), vars_all/2, 1e9)
    lam_model = model.lam32 if force_float32 else model.lam
    gain_model = model.gain32 if force_float32 else model.gain
    lam_all = np.where(live_all, lam_model[np.maximum(idx, 0)], 1e-12)
    gain_all = np.where(live_all, gain_model[np.maximum(idx, 0)], 0.)
    h8 = H8.astype(np.float32 if force_float32 else float)
    A_all = h8[None, :, :]*gain_all[:, None, :]
    S_all = ((A_all*lam_all[:, None, :]) @
             A_all.transpose(0, 2, 1))
    S_all[:, np.arange(8), np.arange(8)] += sig_all
    M_all = lam_all[:, :, None]*A_all.transpose(0, 2, 1)
    K_all = np.linalg.solve(S_all.transpose(0, 2, 1),
                            M_all.transpose(0, 2, 1)).transpose(0, 2, 1)
    x_all = np.einsum('gij,gj->gi', K_all, y_all)
    post_all = lam_all-np.einsum(
        'gij,gji->gi', K_all, A_all*lam_all[:, None, :])
    conf_all = np.clip(1-post_all/lam_all, 0, 1)
    mu = model.mu32 if force_float32 else model.mu
    xhat = np.zeros_like(mu, dtype=real_dtype)
    conf = np.zeros_like(mu, dtype=real_dtype)
    got = np.zeros_like(model.mu, bool)
    live_ranks = idx[live_all]
    xhat[live_ranks] = x_all[live_all]
    conf[live_ranks] = conf_all[live_all]
    got[live_ranks] = True
    return xhat, conf, got


def _receive_rotation(model):
    """conj(cell phase) * EARLY / scale, the fixed per-cell FFT correction,
    built once per model (was three full-frame multiplies per packet)."""
    cache = model.__dict__.setdefault('_receive_cache', {})
    if 'rotation' not in cache:
        rotation = (np.conj(model.phase)*EARLY[None, :] /
                    model.scale)[:, :, None]
        rotation.setflags(write=False)
        cache['rotation'] = rotation
    return cache['rotation']


def _gate_floor(model, force_float32=False):
    """Per-coefficient confidence floor of the §9.6 gate, built once."""
    cache = model.__dict__.setdefault('_receive_cache', {})
    key = 'floor32' if force_float32 else 'floor'
    if key not in cache:
        floor = np.where(model.head, np.where(model.plane == 0, .05, .15),
                         np.where(model.plane == 0, .45, .60))
        if force_float32:
            floor = floor.astype(np.float32)
        floor.setflags(write=False)
        cache[key] = floor
    return cache[key]


def decode_frame(model, x, tmap, counter, prev_tail, cancel=True,
                 diagnostics=None, direct_body=None, force_float32=False,
                 pilot_timing='baseline', pilot_counter=None,
                 tone_equalization='off', profile_hint=None,
                 erased_symbols=None):
    # ``profile_hint`` (the packet's verified metadata, e.g. its aspect code)
    # is for profile hooks that choose a model before decoding; the base
    # decoder does not use it. ``erased_symbols`` marks OFDM symbols known to
    # be missing (a splice fell in them): both legs are erased there.
    if tone_equalization not in ('off', 'm-reference'):
        raise ValueError(f'unknown tone equalization mode {tone_equalization!r}')
    started = perf_counter()
    stage_started = started
    if direct_body is None:
        c0, nom, d = tmap
        base = (counter-c0)*FRAME
        n = base + np.arange(-64, FRAME+64, dtype=float)
        pos = n + np.interp(n, nom, d)
        if pos[0] < 0 or pos[-1] >= len(x)-1:
            return None
        seg = _sample_at(x, pos, taps=4).astype(
            np.float32 if force_float32 else float)
        # Clock cancellation (§9.2): regenerate from this frame's word, LS gain.
        if cancel:
            clk = clock_cancel_template(
                counter, np.float32 if force_float32 else float)
            for ch in range(2):
                a = np.dot(seg[:, ch], clk)/np.dot(clk, clk)
                seg[:, ch] -= a*clk
        seg = seg[64:64+FRAME]
    else:
        seg = np.asarray(direct_body,
                         dtype=np.float32 if force_float32 else float)
    windows = seg.reshape(F, SYM, 2)[:, WIN:WIN+N, :]
    if force_float32:
        Z = (np.fft.rfft(windows, axis=1)/np.float32(model.scale) *
             np.conj(model.phase32)[:, :, None] *
             EARLY.astype(np.complex64)[None, :, None]).astype(np.complex64)
    else:
        Z = np.fft.rfft(windows, axis=1)*_receive_rotation(model)
    erased = None
    if erased_symbols is not None:
        erased = np.asarray(erased_symbols, dtype=bool)
        if erased.shape != (F,):
            raise ValueError('erased_symbols must mark each OFDM symbol')
        if not erased.any():
            erased = None
        else:
            # A missing symbol carries no observation: zero weight in the
            # channel and timing fits.
            Z = Z.copy()
            Z[erased] = 0
    if diagnostics is not None:
        diagnostics.setdefault('stage_ms', {}).setdefault('sample_fft', []).append(
            (perf_counter()-stage_started)*1000)
    stage_started = perf_counter()
    timing_metrics = {}
    if pilot_timing == 'baseline':
        H, timing_diag = channel_joint(
            Z, force_float32=force_float32, return_timing_diag=True)
        timing_metrics['pilot_residual'] = timing_diag['pilot_residual']
        timing_metrics['timing_residual_samples'] = (
            timing_diag['timing_residual_samples'])
    else:
        H, timing_diag = channel_joint(
            Z, force_float32=force_float32, pilot_timing=pilot_timing,
            model=model, counter=(counter if pilot_counter is None
                                  else pilot_counter),
            return_timing_diag=True)
        timing_metrics['pilot_timing'] = timing_diag
        timing_metrics['pilot_residual'] = timing_diag['pilot_residual']
        timing_metrics['timing_residual_samples'] = (
            timing_diag['timing_residual_samples'])
    fade_input = Z
    if erased is not None:
        # The per-symbol fade refit and pilot residual see the fitted pilots
        # on missing symbols, so they neither correct nor smear noise there.
        fade_input = Z.copy()
        for symbol in np.flatnonzero(erased):
            bins = PILOT_PAD_BINS[symbol][PILOT_PAD_VALID[symbol]]
            values = PILOT_PAD_VALUES[symbol][PILOT_PAD_VALID[symbol]]
            fade_input[symbol, bins, :] = np.einsum(
                'pci,pi->pc', H[symbol, bins], values)
    H, noise, tone_eq_diag = fade_and_noise(
        fade_input, H, force_float32=force_float32,
        tone_reference=(tone_equalization == 'm-reference'),
        return_tone_diag=True)
    # The quality gates judge the pilot residual against the pilots' own
    # received power, so they do not depend on the input level. (The
    # equaliser below keeps the absolute residual, which it weighs against H
    # in the same units.)
    pilot_noise = noise/_pilot_reference_power(H)[None, :]
    stereo_erasures = _stereo_erasure_mask(pilot_noise)
    if erased is not None:
        stereo_erasures[erased, :] = True
    effective_noise = _effective_pilot_noise(pilot_noise, stereo_erasures)
    # The affected observation is absent only for these OFDM symbols. Keep the
    # other channel and the rest of this leg's packet in the joint solve.
    noise[stereo_erasures] = np.maximum(
        1.0, np.max(noise)*1e6)
    if tone_equalization != 'off':
        timing_metrics['tone_equalization'] = tone_eq_diag
    if diagnostics is not None:
        diagnostics.setdefault('stage_ms', {}).setdefault('channel', []).append(
            (perf_counter()-stage_started)*1000)
    stage_started = perf_counter()
    DEBUG['Z'], DEBUG['H'], DEBUG['noise'] = Z, H, noise
    if force_float32:
        xhat, conf, got = _equalize_numpy(
            model, Z, H, noise, counter, force_float32=True)
        real_dtype, mu = np.float32, model.mu32
    else:
        xhat, conf, got = _equalize_numba(model, Z, H, noise, counter)
        real_dtype, mu = float, model.mu
    if diagnostics is not None:
        diagnostics.setdefault('stage_ms', {}).setdefault('equalize', []).append(
            (perf_counter()-stage_started)*1000)
    floor = _gate_floor(model, force_float32)
    gate = np.clip((conf-floor)/(.85-floor), 0, 1)
    current = mu + xhat*gate
    coeffs = np.asarray(prev_tail, dtype=real_dtype).copy()
    head_confidence = float(np.mean(conf[model.head]))
    head_coverage = float(np.mean(conf[model.head] >= .15))
    # Erased symbols take their share of the head with them; the gates judge
    # the symbols that are present.
    present = 1.0 if erased is None else 1.0-float(np.mean(erased))
    if (head_confidence < HEAD_MIN_CONFIDENCE*present or
            head_coverage < HEAD_MIN_COVERAGE*present):
        displayable = (head_confidence >= DISPLAY_MIN_HEAD_CONFIDENCE*present and
                       head_coverage >= DISPLAY_MIN_HEAD_COVERAGE*present)
        display_coeffs = np.asarray(prev_tail, dtype=real_dtype).copy()
        if displayable:
            display_coeffs[got] = current[got]
        result_diag = {
            'noise': effective_noise.tolist(), 'got': int(got.sum()),
            'head_confidence': head_confidence,
            'head_coverage': head_coverage, 'held': not displayable,
            'displayable': displayable,
            'stereo_erased_symbols': stereo_erasures.sum(axis=0).tolist(),
            'display_coeffs': display_coeffs}
        result_diag.update(timing_metrics)
        return Result(counter, 'lost', display_coeffs if displayable else
                      np.asarray(prev_tail, dtype=real_dtype).copy(), result_diag)
    coeffs[got] = current[got]
    result_diag = {'noise': effective_noise.tolist(), 'got': int(got.sum()),
                   'head_confidence': head_confidence,
                   'head_coverage': head_coverage, '_H': H,
                   'present_symbols': present,
                   'displayable': True,
                   'stereo_erased_symbols': stereo_erasures.sum(axis=0).tolist()}
    result_diag.update(timing_metrics)
    return Result(counter, 'verified', coeffs, result_diag)


def decode_metadata(model, samples, start, scale, channel,
                    force_float32=False, sample_indexes=None,
                    return_candidates=False):
    indexes = (start + np.arange(META_SYMBOL)*scale
               if sample_indexes is None else
               np.asarray(sample_indexes, dtype=float))
    if indexes.shape != (META_SYMBOL,):
        raise ValueError('metadata sample indexes must match META_SYMBOL')
    if indexes[-1] >= len(samples)-1:
        return None
    meta = _sample_at(samples, indexes, taps=4)
    window = meta[WIN:WIN+N]
    spectrum = np.fft.rfft(
        window.astype(np.float32) if force_float32 else window, axis=0)
    tracks = [spectrum]
    if spectrum.shape[1] == 2:
        # Metadata is duplicated on both legs. A transient on one leg can
        # corrupt the sum even when the other independent copy is intact.
        tracks.extend((spectrum[:, 0:1], spectrum[:, 1:2]))
    candidates = []
    for track in tracks:
        if force_float32:
            z = (track / np.float32(model.scale) *
                 np.conj(model.phase32[-1])[:, None] *
                 _EARLY32[:, None]).astype(np.complex64)
            observed = z.sum(axis=1)
            if z.shape[1] == 1:
                observed *= np.float32(2.0)
            pilot_z = observed[META_PILOTS]
            if np.sum(np.abs(pilot_z) > 1e-6) < 2:
                continue
            response = (np.interp(META_DATA_BINS, META_PILOTS, pilot_z.real) +
                        1j*np.interp(META_DATA_BINS, META_PILOTS, pilot_z.imag))
            response = response.astype(np.complex64)
            data = np.divide(observed[META_DATA_BINS], response,
                             out=np.zeros(len(META_DATA_BINS), np.complex64),
                             where=np.abs(response) > 1e-9)
            symbols = data[:20]
            bits = np.empty(40, np.uint8)
            bits[0::2] = (symbols.real >= 0).astype(np.uint8)
            bits[1::2] = (symbols.imag >= 0).astype(np.uint8)
            meta = parse_metadata_word(np.packbits(bits).tobytes())
        else:
            valid, packed = _decode_metadata_spectrum(
                track, model.phase[-1], EARLY, model.scale,
                META_PILOTS, META_DATA_BINS)
            meta = parse_metadata_word(packed.tobytes()) if valid else None
        if meta is not None:
            candidates.append(meta)
    if return_candidates:
        return candidates
    if not candidates:
        return None
    # Prefer a plain CRC match when available; PulseState has the loop lock
    # needed to rank the rotated CRC masks on slices 5 and 6.
    return next((meta for meta in candidates if meta.mask == 0), candidates[0])


def _diagnostic_summary(diag, elapsed_ms):
    if diag is None:
        return None
    diag['decode_calls'] = diag.get('decode_calls', 0) + 1
    diag['last_elapsed_ms'] = round(elapsed_ms, 3)
    stages = diag.get('stage_ms', {})
    diag['stage_summary_ms'] = {
        name: {'count': len(values),
               'mean': round(float(np.mean(values)), 3),
               'p95': round(float(np.percentile(values, 95)), 3)}
        for name, values in stages.items() if values
    }
    diag['stage_ms'] = {}
    return diag


POLARITY_THRESHOLD = .3


def leg_polarity(samples, previous=1, threshold=POLARITY_THRESHOLD):
    """Right-leg polarity (+1 or -1) of a stereo capture, with hysteresis.

    The V7 wire is M-dominated (preamble, clock, metadata and head blocks are
    M only), so a correctly wired pair correlates strongly positive: +0.73 to
    +0.85 wideband over one pulse frame across the whole torture matrix.  An
    inverted leg gives the mirror image.  Between -threshold and +threshold
    (silence, one dead leg, non-V7 audio) the previous decision is kept.
    Mono or single-channel input is always +1.
    """
    x = np.asarray(samples)
    if x.ndim != 2 or x.shape[1] != 2 or len(x) < 2:
        return 1
    if x.dtype not in (np.float32, np.float64):
        x = x.astype(np.float64)
    # DC/hum offset must not vote: correlate the mean-removed legs.
    left_power, right_power, cross = _leg_correlation_sums(x)
    energy = left_power*right_power
    if energy <= 1e-24:
        return previous
    correlation = cross/np.sqrt(energy)
    if correlation <= -threshold:
        return -1
    if correlation >= threshold:
        return 1
    return previous


def warmup_leg_polarity():
    """Compile the production stereo-input kernels before capture starts.

    These kernels use Numba's disk cache, so compatible subsequent processes
    can load the compiled signatures rather than compile them again.
    """
    leg_polarity(np.zeros((PULSE_FRAME, 2), dtype=np.float32))
    _mono_gain(np.zeros((PULSE_FRAME, 2), dtype=np.float32), np.float32(1.0))
    _mono_gain(np.zeros((PULSE_FRAME, 1), dtype=np.float32), np.float32(1.0))
    _mono_gain(np.zeros(PULSE_FRAME, dtype=np.float32), np.float32(1.0))


def decode_pulse_stream(model, x, diagnostics=None, latest_only=False,
                        input_gain=1.0, models=None, model_factory=None,
                        force_float32=False, state=None, pulse_starts=None,
                        sample_rate=RATE, pilot_timing='baseline',
                        pilot_speed_diagnostics=False,
                        pulse_timing='baseline',
                        tone_equalization='off'):
    """Decode V7 bodies located by the existing pulse-counted acquisition.

    This is the low-latency live path: each accepted pulse word supplies a
    frame scale, the body is resampled directly to the reference grid, and the
    next pulse search starts after that frame.  There is no continuous clock
    track or buffered clock-template refinement.

    ``input_gain`` is a scalar or one gain per channel; a negative right gain
    applies a polarity correction from leg_polarity().

    ``state`` (PulseState) carries the tail store and the learned loop
    constants between calls; a live receiver passes the same one every time.
    Without it each call starts fresh (fine for a whole recording).

    ``pulse_starts`` optionally supplies ``(start, scale, confidence)`` anchors
    from an upstream incremental scan. The anchors are rechecked locally after
    input gain is applied; if they do not validate, normal acquisition runs.
    ``sample_rate`` is the rate of ``x``; pulse-scale limits are normalized
    against it while sample positions remain in the input's native coordinates.
    ``pilot_timing`` selects the optional low-bin timing reference:
    ``baseline`` (default), ``tone-seeded``, ``tone-joint``, or
    ``tone-replaced``. A packet runs from its header to its EOF marker.
    ``pulse_timing='pulse-warp'`` optionally uses the neighboring pulse fits
    as local-slope anchors for a monotone, within-packet Hermite sample map.
    ``tone_equalization='m-reference'`` uses the known low-bin tone magnitudes
    as a packet-relative M-path gain reference in addition to the ordinary
    data-pilot channel fit. Static response, S-path response, and spectral
    slope remain data-pilot estimates.
    """
    if pilot_timing not in (
            'baseline', 'tone-seeded', 'tone-joint', 'tone-replaced'):
        raise ValueError(f'unknown pilot timing mode {pilot_timing!r}')
    if pulse_timing not in ('baseline', 'pulse-warp'):
        raise ValueError(f'unknown pulse timing mode {pulse_timing!r}')
    if tone_equalization not in ('off', 'm-reference'):
        raise ValueError(f'unknown tone equalization mode {tone_equalization!r}')
    # Live capture is float32 and _sample_at returns float32.  Promoting the
    # complete rolling history to float64 here only doubles allocation and
    # memory traffic; the FFT/equalizer still performs its own complex work at
    # the precision NumPy requires.
    samples = np.asarray(x, np.float32)
    if samples.ndim == 1:
        samples = samples[:, None]              # a bare mono array is one channel
    gain = np.float32(input_gain)
    if gain.ndim and samples.shape[1] != len(gain):
        gain = gain[:samples.shape[1]]          # per-channel gain on mono input
    # The sample readers honor ndarray strides, so leave unity-gain input
    # untouched; this also avoids copying a normalized reverse-playback view.
    if not (gain.size and np.all(gain == 1)):
        samples = samples*gain
    if state is None:
        state = PulseState()
    results, info = _decode_pulse_samples(model, samples, diagnostics,
                                          latest_only, models, model_factory,
                                          force_float32, state, pulse_starts,
                                          sample_rate, pilot_timing,
                                          pilot_speed_diagnostics,
                                          pulse_timing,
                                          tone_equalization)
    if results or samples.shape[1] != 2:
        return results, info
    # One leg polarity-inverted (miswired deck or cable, reversed head lead):
    # the preamble, clock and metadata are M-only, so the L+R acquisition sum
    # cancels and nothing is found.  The 2x2 pilot equalizer itself would cope,
    # so retry once with the right leg inverted.  Normal streams never reach
    # this; silence costs one more (cheap, empty) edge scan.
    flipped, flipped_info = _decode_pulse_samples(
        model, samples*np.float32([1, -1]), diagnostics, latest_only, models,
        model_factory, force_float32, state, pulse_starts, sample_rate,
        pilot_timing, pilot_speed_diagnostics, pulse_timing,
        tone_equalization)
    if not flipped:
        return results, info
    for result in flipped:
        result.diag['polarity_inverted'] = True
    flipped_info['polarity_inverted'] = True
    return flipped, flipped_info


def _mono(samples):
    """Mean of the capture's channels, bit-identical to samples.mean(axis=1)
    for one or two channels (the average of two floats is (a+b)*0.5 in their
    own precision) but without NumPy's slow short-axis reduction, which cost
    ~0.15 ms on every live window."""
    samples = np.asarray(samples)
    if samples.ndim == 1:
        return samples
    if samples.ndim == 2 and samples.shape[1] == 2 and samples.dtype in (
            np.float32, np.float64):
        return (samples[:, 0]+samples[:, 1])*samples.dtype.type(.5)
    if samples.ndim == 2 and samples.shape[1] == 1:
        return samples[:, 0].copy()
    return samples.mean(axis=1)


def pulse_frame_starts(samples, sample_rate=RATE):
    """Every accepted pulse header in a stereo (or (n, 1)) capture.

    Returns ``(frame_start, scale, confidence)`` per header, in order, using
    the same edge-counted acquisition as the decoder: confidence below .45 is
    skipped, and after a hit the scan resumes just before the next expected
    header.  A header needs SYNC_LEN + META_SYMBOL + 32 samples after its
    scan point to be found, so callers scanning a live stream should overlap
    successive scans by that much.
    """
    min_scale, max_scale = pulse_sample_scale_bounds(sample_rate)
    starts = []
    scan = 0
    mono = _mono(samples)      # once, not once per header found
    while scan + PULSE.SYNC_LEN + META_SYMBOL + 32 < len(samples):
        hit = PULSE.measure_pulses(mono[scan:],
                                   min_scale=min_scale,
                                   max_scale=max_scale)
        if hit is None:
            break
        pos, sc, conf = hit
        fs = scan + pos - 16*sc
        if conf < .45:
            scan = int(fs + PULSE_FRAME*sc)
            continue
        starts.append((fs, sc, conf))
        scan = int(fs + (PULSE_FRAME-32)*sc)
    return starts


def pulse_frame_hits(samples, sample_rate=RATE, direction='auto'):
    """Find pulse packet starts in either playback direction.

    Returns ``(packet_start, scale, confidence, direction)`` hits. The pulse
    matcher fits the start of its 256-sample template. Forward packets place
    that template at sample 16; a reversed packet places it at
    ``PULSE_FRAME - 16 - len(PREAMBLE)``. Positions remain fractional capture
    samples, and timing remains exclusively edge-counted.

    Unlike ``pulse_frame_starts``, this helper does not require forward
    metadata samples after a hit. The live input layer decides when enough of
    the selected packet is present to decode it.
    """
    return [hit[:4] for hit in pulse_frame_profile_hits(
        samples, sample_rate=sample_rate, direction=direction)]


def pulse_frame_profile_hits(samples, sample_rate=RATE, direction='auto'):
    """Find packet starts and Schmitt-coded profile IDs in either direction.

    Returns ``(packet_start, scale, confidence, direction, profile_code)``.
    A missing profile code is an ambiguous/unsupported edge word, while the
    timing hit remains useful to the ordinary packet decoder.
    """
    if direction not in ('auto', 'forward', 'reverse'):
        raise ValueError(f'unknown pulse direction {direction!r}')
    min_scale, max_scale = pulse_sample_scale_bounds(sample_rate)
    mono = _mono(samples)
    hits = []
    scan = 0
    minimum = len(PULSE.PREAMBLE)+16
    reverse_offset = PULSE_FRAME-16-len(PULSE.PREAMBLE)
    while scan+minimum <= len(mono):
        hit = PULSE.measure_pulses_profile_both(
            mono[scan:], min_scale=min_scale, max_scale=max_scale,
            direction=direction)
        if hit is None:
            break
        template_start, scale, confidence, way, profile_code = hit
        template_start += scan
        packet_start = (template_start-16*scale if way > 0 else
                        template_start-reverse_offset*scale)
        if confidence >= .45:
            hits.append((float(packet_start), float(scale),
                         float(confidence), int(way), profile_code))
        # Keep enough room before the following packet's template. Both wire
        # orientations have one preamble per packet, though their offsets
        # within that packet differ.
        scan = max(scan+1, int(packet_start+(PULSE_FRAME-32)*scale))
    return hits


REVERSE_PACKET_MARGIN = 64


def reverse_packet_region(samples, packet_start, scale,
                          margin=REVERSE_PACKET_MARGIN):
    """Return a bounded candidate region reversed into forward time.

    The predicted packet bounds must be present in the capture. Margins may be
    clipped at a recording edge; the decoder still has to reacquire the
    preamble and validate the EOF marker before publishing a result.
    Returns ``(region, region_start)`` or ``None`` for an incomplete packet.
    """
    audio = np.asarray(samples)
    packet_start, scale = float(packet_start), float(scale)
    if (audio.ndim not in (1, 2) or not np.isfinite(packet_start+scale) or
            scale <= 0 or margin < 0):
        return None
    packet_end = packet_start+PULSE_FRAME*scale
    if packet_start < -1.0 or packet_end > len(audio)+1.0:
        return None
    lo = max(0, int(np.floor(packet_start-margin*scale)))
    hi = min(len(audio), int(np.ceil(packet_end+margin*scale)))
    if hi <= lo:
        return None
    # Keep the bounded reverse view; the decoder's sample readers honor its
    # stride, and non-unity input gain materializes it only when needed.
    return audio[lo:hi][::-1], lo


def decode_reverse_packet(model, samples, packet_start, scale, **kwargs):
    """Reverse and decode one EOF-marked packet candidate.

    The existing forward decoder remains the only OFDM/metadata path. Reverse
    candidates always require a validated EOF marker and independently valid
    metadata; provisional metadata must not authorize a picture on cold start.
    """
    region = reverse_packet_region(samples, packet_start, scale)
    if region is None:
        return [], {'frames': 0, 'pulse_frames': 0, 'recovered': False,
                    'reverse_rejected': 'incomplete_candidate'}
    candidate, region_start = region
    # Reversal maps the packet's exclusive capture end to its normalized
    # forward-time origin. Seed the normal decoder with that predicted anchor;
    # its usual short-window pulse remeasurement must confirm it before the
    # EOF or image path can proceed, avoiding a second full-window scan.
    normalized_start = (len(candidate)+region_start-
                        (float(packet_start)+PULSE_FRAME*float(scale)))
    options = kwargs
    state = options.get('state')
    if state is None:
        state = PulseState()
        options['state'] = state
    state.set_playback_direction(-1)
    options.pop('latest_only', None)
    options.pop('pulse_starts', None)
    previous_tail_gate = state._require_independent_tail_metadata
    state._require_independent_tail_metadata = True
    try:
        results, info = decode_pulse_stream(
            model, candidate, latest_only=True,
            pulse_starts=((normalized_start, float(scale), 1.0),), **options)
    finally:
        state._require_independent_tail_metadata = previous_tail_gate
    marker_valid = bool(info.get('eof_markers_validated'))
    for result in results:
        result.diag['playback_direction'] = -1
        result.diag['reverse_candidate_start'] = float(packet_start)
        result.diag['reverse_candidate_scale'] = float(scale)
        result.diag['reverse_region_start'] = int(region_start)
        # Loop-field slices may only be provisionally accepted from prior
        # metadata. They are not safe for independent reverse startup.
        result_marker_valid = result.diag.get('eof_marker') is not None
        if (not marker_valid or not result_marker_valid or
                not result.diag.get('metadata_valid') or
                result.diag.get('metadata_provisional')):
            result.status = 'lost'
            result.diag['displayable'] = False
            result.diag['reverse_rejected'] = (
                'eof_marker_not_validated'
                if not marker_valid or not result_marker_valid else
                'metadata_not_independently_valid')
    info['playback_direction'] = -1
    info['reverse_candidate_start'] = float(packet_start)
    return results, info


def _remeasure_pulse_starts(samples, anchors, sample_rate=RATE, mono=None):
    """Recheck upstream header anchors in short windows on decoder samples."""
    min_scale, max_scale = pulse_sample_scale_bounds(sample_rate)
    if mono is None:
        mono = _mono(samples)
    measured = []
    for expected_start, expected_scale, _ in sorted(
            anchors, key=lambda anchor: float(anchor[0])):
        expected_start, expected_scale = (float(expected_start),
                                          float(expected_scale))
        if not np.isfinite(expected_start+expected_scale) or expected_scale <= 0:
            continue
        scan = max(0, int(np.floor(expected_start-8*expected_scale)))
        end = min(len(mono), int(np.ceil(
            expected_start+(PULSE.SYNC_LEN+META_SYMBOL+32)*expected_scale)))
        if end <= scan:
            continue
        hit = PULSE.measure_pulses(
            mono[scan:end], min_scale=min_scale, max_scale=max_scale)
        if hit is None:
            continue
        position, scale, confidence = hit
        frame_start = scan+position-16*scale
        if (abs(frame_start-expected_start) >
                max(16*expected_scale, .03*PULSE_FRAME*expected_scale) or
                abs(scale/expected_scale-1) > .03 or confidence < .45):
            continue
        if measured and frame_start-measured[-1][0] < .5*PULSE_FRAME*scale:
            continue
        measured.append((frame_start, scale, confidence))
    measured.sort(key=lambda hit: hit[0])
    return measured


EOF_EDGE_HYSTERESIS = .35    # Schmitt band, as a fraction of header level


def _header_level(mono, frame_start, scale, fallback=None):
    """The level a packet's header word arrived at: the plateau of its
    runs. When the header is not in the samples given, ``fallback`` (or the
    level it is sent at)."""
    lo = int(np.ceil(frame_start+16*scale))
    hi = int(np.floor(frame_start+(16+len(PULSE.PREAMBLE))*scale))
    if lo < 0 or hi > len(mono) or hi-lo < 32:
        return HEADER_PEAK if fallback is None else float(fallback)
    return float(np.percentile(np.abs(mono[lo:hi]), EOF_HEADER_PERCENTILE))



@njit(cache=True)
def _crossing_fraction(mono, left):
    """Where a band-limited signal crosses zero between samples left and
    left+1, from the cubic through the four samples around it (a straight
    line between the two is biased on a shaped edge)."""
    y1 = mono[left]
    y2 = mono[left+1]
    t = y1/(y1-y2)
    if left < 1 or left+2 >= mono.shape[0]:
        return t
    y0 = mono[left-1]
    y3 = mono[left+2]
    a = -.5*y0+1.5*y1-1.5*y2+.5*y3
    b = y0-2.5*y1+2*y2-.5*y3
    c = -.5*y0+.5*y2
    for _ in range(4):
        value = ((a*t+b)*t+c)*t+y1
        slope = (3*a*t+2*b)*t+c
        if slope == 0:
            break
        t -= value/slope
    if not 0.0 <= t <= 1.0:
        return y1/(y1-y2)
    return t


@njit(cache=True, fastmath=False)
def _eof_marker_kernel(mono, lo, hi, expected_start, radius, start_scale,
                       hysteresis, min_level, edges_nominal, runs, out):
    """Schmitt trigger + run counter for the EOF mark (+, -, +, -).

    Walks [lo, hi) once. Each Schmitt state change is an edge, timed at the
    last zero crossing before it (as the preamble edges are). Three
    consecutive edges -, +, - (or the inverted pattern) whose spacing counts
    8 and 6 samples at one scale, whose runs each peak past min_level, and
    whose fitted start lies inside the search window make a marker. The
    best is the one closest to the packet-clock prediction.

    out receives [found, start, scale, confidence, residual, gap_error,
    min_level, polarity, prediction_error].
    """
    out[0] = 0.0
    times = np.empty(hi-lo)
    signs = np.empty(hi-lo, np.int64)
    peaks = np.empty(hi-lo)          # peak of the run that ends at this edge
    edges = 0
    state = 0
    run_peak = 0.0
    last_crossing = -1
    previous_negative = math.copysign(1.0, mono[lo]) < 0
    for i in range(lo, hi):
        value = mono[i]
        negative = math.copysign(1.0, value) < 0
        if i > lo and negative != previous_negative:
            last_crossing = i-1
        previous_negative = negative
        now = 1 if value > hysteresis else (-1 if value < -hysteresis else 0)
        if now != 0 and state != 0 and now != state and last_crossing >= lo:
            left = last_crossing
            times[edges] = left + _crossing_fraction(mono, left)
            signs[edges] = now
            peaks[edges] = run_peak
            edges += 1
            run_peak = 0.0
        if now != 0:
            state = now
        if state != 0:
            run_peak = max(run_peak, state*value)
    # Peak of the last run after each edge's successor is the next edge's peak;
    # the final run (after the third marker edge) only has to cross the band,
    # which its Schmitt state change already proves.
    nominal_mean = (edges_nominal[0]+edges_nominal[1]+edges_nominal[2])/3
    spread = 0.0
    for k in range(3):
        spread += (edges_nominal[k]-nominal_mean)**2
    best_error = np.inf
    best_quality = np.inf
    for k in range(edges-2):
        first = signs[k]
        if signs[k+1] != -first or signs[k+2] != first:
            continue
        polarity = -first            # (+,-,+,-) marker: first edge enters -
        level = min(peaks[k], peaks[k+1], peaks[k+2])
        if level < min_level:
            continue
        g1 = times[k+1]-times[k]
        g2 = times[k+2]-times[k+1]
        n1 = edges_nominal[1]-edges_nominal[0]
        n2 = edges_nominal[2]-edges_nominal[1]
        marker_scale = .5*(g1/n1+g2/n2)
        if not (.7*start_scale <= marker_scale <= 1.3*start_scale):
            continue
        gap_error = max(abs(g1-n1*marker_scale), abs(g2-n2*marker_scale))
        if gap_error > max(1.5, .40*min(n1, n2)*marker_scale):
            continue
        t_mean = (times[k]+times[k+1]+times[k+2])/3
        cov = 0.0
        for m in range(3):
            cov += (edges_nominal[m]-nominal_mean)*(times[k+m]-t_mean)
        fitted = cov/spread
        if fitted <= 0 or abs(fitted/marker_scale-1) > .08:
            continue
        offset = t_mean-fitted*nominal_mean
        sq = 0.0
        for m in range(3):
            sq += (times[k+m]-offset-fitted*edges_nominal[m])**2
        residual = math.sqrt(sq/3)
        confidence = min(max(1-residual/max(1.5, .45*fitted), 0.0), 1.0)
        if confidence < .45:
            continue
        error = offset-expected_start
        if abs(error) > radius:
            continue
        quality = residual+gap_error*.25
        if (abs(error) < best_error or
                (abs(error) == best_error and quality < best_quality)):
            best_error, best_quality = abs(error), quality
            out[0] = 1.0
            out[1] = offset
            out[2] = fitted
            out[3] = confidence
            out[4] = residual
            out[5] = gap_error
            out[6] = level
            out[7] = polarity
            out[8] = error


def _measure_eof_marker(samples, frame_start, start_scale, radius=None):
    """Find a packet's EOF mark with a Schmitt trigger and a run counter.

    The mark is a known four-run pattern at preamble level, so like the
    header it is counted, not fitted: see _eof_marker_kernel. The search is
    gated around the packet-clock prediction. Returns the marker's measured
    start, end and scale and the packet scale they imply, or None.
    """
    mono = np.asarray(samples)
    if mono.ndim == 2:
        mono = mono.mean(axis=1)
    if mono.ndim != 1 or start_scale <= 0 or mono.dtype not in (
            np.float32, np.float64):
        return None
    start_scale = float(start_scale)
    expected_start = float(frame_start)+EOF_MARKER_OFFSET*start_scale
    if radius is None:
        radius = max(12*start_scale,
                     EOF_SEARCH_FRACTION*PULSE_FRAME*start_scale)
    lo = max(0, int(np.floor(expected_start-radius)))
    hi = min(len(mono), int(np.ceil(expected_start+
                                    EOF_MARKER_LENGTH*start_scale+radius)))
    if hi-lo < EOF_MARKER_LENGTH*start_scale:
        return None
    # The mark is sent at the header's level, so it is judged against the
    # level this packet's header arrived at, not against fixed numbers.
    # Without the header, the loudest sample near the mark stands in: the
    # mark is the loudest thing at the end of a packet.
    level = _header_level(mono, float(frame_start), start_scale,
                          fallback=float(np.max(np.abs(mono[lo:hi]))))
    out = np.zeros(9)
    _eof_marker_kernel(mono, lo, hi, expected_start, radius, start_scale,
                       EOF_EDGE_HYSTERESIS*level, EOF_MARKER_MIN_LEVEL*level,
                       SHAPED_EOF_EDGES, EOF_MARKER_RUNS_ARRAY, out)
    if not out[0]:
        return None
    marker_start, fitted_scale = float(out[1]), float(out[2])
    marker_end = marker_start+EOF_MARKER_LENGTH*fitted_scale
    return {
        'start': marker_start,
        'end': marker_end,
        'scale': fitted_scale,
        'packet_scale': (marker_end-float(frame_start))/PULSE_FRAME,
        'confidence': float(out[3]),
        'edge_residual': float(out[4]),
        'gap_error': float(out[5]),
        'min_level': float(out[6]),
        'polarity': int(out[7]),
        'prediction_error': float(out[8]),
    }


SPLICE_MIN_JUMP = 24            # samples at scale 1: below this, affine map
# A pitch shift that keeps the tempo stretches a packet's header-scale length
# by the inverse factor: half (pitch down 0.5) to double (pitch up 2). Splices may
# shorten a packet to SPLICE_MIN_LENGTH or lengthen it by SPLICE_MAX_FRACTION.
SPLICE_MAX_FRACTION = 1.05      # largest lengthening, of PULSE_FRAME
SPLICE_MIN_LENGTH = .40         # shortest spliced packet, of PULSE_FRAME
SPLICE_MIN_GAIN = 1.5           # summed CP correlation the cut must add
# Below this a symbol is erased. Clean symbols measure down to 0.90 (their
# cyclic prefix can hold little energy), so the limit sits clear of that.
SPLICE_INTACT_CP = .85
SPLICE_SUSPECT_CP = .95         # body fit this poor: question the EOF


_CP_PICK = np.concatenate((
    (np.arange(F)[:, None]*SYM+np.arange(CP)[None, :]).ravel(),
    (np.arange(F)[:, None]*SYM+N+np.arange(CP)[None, :]).ravel()))


def _cp_scores_at(samples, indexes):
    """Cyclic-prefix correlation of each body symbol under a sample map.

    1 means the symbol's prefix matches its tail (an intact symbol at the
    right offset). Only the prefixes and their tails are read.
    """
    picked = np.asarray(_sample_at(samples, np.asarray(indexes)[_CP_PICK],
                                   taps=4), np.float64)
    if picked.ndim == 1:
        picked = picked[:, None]
    prefix = picked[:F*CP].reshape(F, -1)
    tail = picked[F*CP:].reshape(F, -1)
    energy = np.sqrt(np.sum(prefix*prefix, axis=1)*np.sum(tail*tail, axis=1))
    return np.sum(prefix*tail, axis=1)/np.maximum(energy, 1e-12)


TONE_END_MAX_STEP = .03         # tone speed this far off the header: no backup
TONE_END_MIN_STEADINESS = .75   # 1: the tone's phase is one straight line
TONE_END_MARK_RADIUS = 4.0      # samples either side of the tone-given end
TONE_END_CONFIDENCE = .5        # stands in for a counted mark's confidence
_TONE_END_BIN = max(PILOT_TONE_BINS)
# One symbol-long read per body symbol, started half a cyclic prefix in: the
# data carriers are whole cycles in it, so they cancel and only the tone is
# left in its bin.
_TONE_END_READ = (PULSE.SYNC_LEN+CP//2+np.arange(F)[:, None]*SYM +
                  np.arange(N)[None, :]).astype(np.float64)
_TONE_END_PROBE = np.exp(-2j*np.pi*np.arange(N)*_TONE_END_BIN/N)
_TONE_END_ADVANCE = 2*np.pi*_TONE_END_BIN*SYM/N     # radians per symbol


def _packet_tone_scale(mono, frame_start, scale):
    """Samples per packet sample, measured from a packet's own timing tone.

    The tone runs unbroken through the body, so the phase it gains from one
    symbol to the next is the playback speed. Each symbol is read on the
    header's clock, the tone's phase taken, and the clock corrected until
    the phase advances as sent. Returns (scale, steadiness), or None when
    the body is not all in or the tone is not there.
    """
    scale = float(scale)
    lo = max(0, int(frame_start)-8)
    hi = min(len(mono), int(np.ceil(
        frame_start+PULSE_FRAME*scale*(1+2*TONE_END_MAX_STEP)))+8)
    segment = np.ascontiguousarray(mono[lo:hi], np.float64)[:, None]
    # The header's clock drifts off the packet's as the body goes on, and a
    # read that has slid into the next symbol is no use. So the first few
    # symbols correct the clock, then more of them, then all.
    for count in (4, 8, 16, F, F):
        indexes = (frame_start-lo)+_TONE_END_READ[:count].ravel()*scale
        if indexes[-1] >= len(segment)-4:
            return None
        read = _sample_at(segment, indexes, taps=4)[:, 0].reshape(count, N)
        tone = read.astype(np.float64) @ _TONE_END_PROBE
        level = np.abs(tone)
        if not np.all(np.isfinite(level)) or float(np.sum(level)) <= 0:
            return None
        symbols = np.arange(count)
        # A profile may flip the tone's sign symbol by symbol (coded
        # pilots); squaring removes the sign and doubles the phase.
        phase = np.unwrap(np.angle(
            tone*tone*np.exp(-2j*_TONE_END_ADVANCE*symbols)))
        slope = .5*float(np.polyfit(symbols, phase, 1, w=level)[0])
        scale /= 1+slope/_TONE_END_ADVANCE
    # Is it the tone? Its phase, once the clock is right, lies on one line
    # through every symbol; noise or picture data does not.
    fit = np.polyval(np.polyfit(symbols, phase, 1, w=level), symbols)
    weight = level*level
    steadiness = float(np.abs(np.sum(weight*np.exp(1j*(phase-fit)))) /
                       np.sum(weight))
    if steadiness < TONE_END_MIN_STEADINESS:
        return None
    return scale, steadiness


def _tone_eof_marker(mono, frame_start, scale):
    """A packet's end from its own timing tone, when its EOF mark is not
    where the header puts it.

    The tone's pitch is the packet's playback speed, the speed is its
    length, and the header position plus that length is its end. The mark is
    looked for once more, closely, at that end; if it is still absent the
    tone length itself ends the packet (witness 'tones'). Nothing past the
    packet's own last sample decides anything. Returns a
    _measure_eof_marker-style dict, or None when the tone is not there or
    the packet is not all in yet.
    """
    scale = float(scale)
    measured = _packet_tone_scale(mono, frame_start, scale)
    if measured is None:
        return None
    tone_scale, steadiness = measured
    if not abs(tone_scale/scale-1) <= TONE_END_MAX_STEP:
        return None
    marker = _measure_eof_marker(mono, frame_start, tone_scale,
                                 radius=TONE_END_MARK_RADIUS*tone_scale)
    if marker is not None:
        marker = dict(marker)
        marker['tone_search'] = True
        return marker
    end = float(frame_start)+PULSE_FRAME*tone_scale
    if end > len(mono)+1.0:
        return None
    return {'start': float(frame_start)+EOF_MARKER_OFFSET*tone_scale,
            'end': end, 'scale': tone_scale, 'packet_scale': tone_scale,
            'confidence': TONE_END_CONFIDENCE, 'witness': 'tones',
            'tone_steadiness': steadiness}


def _packet_end(mono, frame_start, scale, min_scale, max_scale):
    """Where a packet ends: its EOF mark, found from the header; else from
    its tones; else, for a packet a time-stretcher cut, from the splice
    search; else the length its tones give.
    """
    marker = _measure_eof_marker(mono, frame_start, scale)
    if marker is not None:
        return marker
    by_tones = _tone_eof_marker(mono, frame_start, scale)
    if by_tones is not None and by_tones.get('witness') != 'tones':
        return by_tones
    spliced = _spliced_eof_marker(mono, frame_start, scale, min_scale,
                                  max_scale)
    return spliced if spliced is not None else by_tones


def _spliced_eof_marker(mono, frame_start, scale, min_scale, max_scale,
                        exclude_end=None):
    """EOF mark of a packet whose length differs from its header scale.

    The next header is found in a window wide enough for one splice; the EOF
    mark is then searched where that header puts it (header-to-EOF spacing is
    fixed). The next header only says where to look: the packet's own mark is
    the witness, and without it there is none. Returns a
    _measure_eof_marker-style dict rebased on frame_start, or None.
    ``exclude_end`` rejects a witness at an endpoint already tried.
    """
    scale = float(scale)
    nominal = frame_start+PULSE_FRAME*scale
    lo = max(0, int(frame_start+SPLICE_MIN_LENGTH*PULSE_FRAME*scale))
    hi = min(len(mono), int(nominal+SPLICE_MAX_FRACTION*PULSE_FRAME*scale +
                            (PULSE.SYNC_LEN+32)*scale))
    if hi-lo < PULSE.SYNC_LEN*scale:
        return None
    hit = PULSE.measure_pulses(mono[lo:hi], min_scale=min_scale,
                               max_scale=max_scale)
    if hit is None or hit[2] < .45 or abs(hit[1]/scale-1) > .03:
        return None
    next_start = lo+hit[0]-16*hit[1]
    if exclude_end is not None and abs(next_start-exclude_end) < \
            SPLICE_MIN_JUMP*scale:
        return None
    marker = _measure_eof_marker(mono, next_start-PULSE_FRAME*scale, scale)
    if marker is None:
        return None
    marker = dict(marker)
    marker['packet_scale'] = (marker['end']-float(frame_start))/PULSE_FRAME
    marker['splice_search'] = True
    return marker


SPLICE_OFFSET_STEP = 2          # reference samples between tried offsets
SPLICE_STEP_PENALTY = 2.5       # CP correlation one more splice must earn
SPLICE_UNREACHED = .25          # endpoint offset the body must reach
SPLICE_PEAK_CP = .75            # a CP correlation peak worth identifying
SPLICE_IDENTITY_WEIGHT = 1.0    # pilot-fit residual (0..1) subtracted
SPLICE_IDENTITY_MAX = .5        # above this a symbol is not who it seems
_IDENTITY_TIMING = np.arange(-2.0, 2.01, .5)    # samples tried per peak


def _pilot_observations(model, symbols, windows):
    """Rotated pilot cells of each window read as the given symbol.

    Returns padded arrays: known M/S pilot values (r, p, 2) and observed
    cells (r, p, ch), zero where a symbol has fewer pilots, the pilot bins
    (r, p) and the valid mask (r, p).
    """
    symbols = np.asarray(symbols, int)
    windows = np.asarray(windows, np.float64)
    if windows.shape[-1] == 1:
        windows = np.repeat(windows, 2, axis=-1)
    bins = PILOT_PAD_BINS[symbols]
    valid = PILOT_PAD_VALID[symbols]
    spectra = np.fft.rfft(windows, axis=1)
    rows = np.arange(len(symbols))[:, None]
    rotation = _receive_rotation(model)
    observed = spectra[rows, bins, :]*rotation.reshape(F, rotation.shape[1], -1)[
        symbols[:, None], bins]
    observed[~valid] = 0
    values = np.where(valid[..., None], PILOT_PAD_VALUES[symbols], 0)
    return values, observed, bins, valid


def _ramped(values, bins):
    """Pilot designs for every tried timing ramp: (d, r, p, 2)."""
    ramp = np.exp(2j*np.pi*bins[None]*_IDENTITY_TIMING[:, None, None]/N)
    return values[None]*ramp[..., None]


def _free_fit_error(design, observed):
    """Least-squares residual energy of observed (r, p, ch) on each design
    (d, r, p, 2), per (d, r)."""
    gram = np.einsum('drpi,drpj->drij', design.conj(), design)
    gram = gram+1e-9*np.eye(2)
    projected = np.einsum('drpi,rpc->dric', design.conj(), observed)
    solved = np.linalg.solve(gram, projected)
    captured = np.einsum('dric,dric->dr', projected.conj(), solved).real
    return np.sum(np.abs(observed)**2, axis=(1, 2))[None]-captured


def _pilot_channel(observations):
    """One flat 2x2 (channels x M/S) channel from symbols known to be right.

    Each anchor's small timing ramp is taken from its own best fit first.
    """
    values, observed, bins, _valid = observations
    designs = _ramped(values, bins)
    best = np.argmin(_free_fit_error(designs, observed), axis=0)
    design = designs[best, np.arange(len(best))].reshape(-1, 2)
    return np.linalg.lstsq(design, observed.reshape(-1, observed.shape[-1]),
                           rcond=None)[0]                         # (2, ch)


def _pilot_identity_residual(model, symbols, windows, channel=None):
    """How badly each window's pilots fit the given symbols (0 = perfectly).

    After the named symbol's cell rotation its pilots must match the packet's
    channel (``channel`` from _pilot_channel; per window gain, phase and a
    small timing ramp free). Another symbol's content, scrambled by a
    different rotation, does not. Without a channel each window may choose
    its own flat 2x2 channel: a weaker test (4-5 pilots, 4 unknowns).
    """
    values, observed, bins, valid = _pilot_observations(
        model, symbols, windows)
    energy = np.sum(np.abs(observed)**2, axis=(1, 2))
    usable = (energy > 0) & (np.count_nonzero(valid, axis=1) >= 3)
    designs = _ramped(values, bins)
    safe = np.maximum(energy, 1e-30)
    if channel is None:
        error = _free_fit_error(designs, observed)/safe[None]
    else:
        predicted = designs@channel                            # (d, r, p, ch)
        power = np.sum(np.abs(predicted)**2, axis=(2, 3))
        match = np.abs(np.sum(predicted.conj()*observed[None],
                              axis=(2, 3)))**2
        error = np.where(power > 0, 1.0-match/np.maximum(power, 1e-30)/safe,
                         np.inf)
    return np.where(usable, np.clip(np.min(error, axis=0), 0.0, 1.0), 1.0)


def _staircase(scores, candidates, jump):
    """Best monotone offset path from 0 to ``jump`` (see _symbol_offsets)."""
    step = SPLICE_OFFSET_STEP
    scores = scores.copy()
    # A whole-symbol shift reads a neighbouring symbol, whose cyclic prefix
    # correlates just as well: offsets a whole number of symbols from either
    # anchor are aliases, not a plausible second splice, unless they are the
    # anchors themselves.
    for anchor in (0.0, jump):
        distance = (candidates-anchor)/SYM
        alias = (np.abs(distance-np.rint(distance)) < 4.0/SYM) & \
            (np.rint(distance) != 0)
        scores[:, alias & (np.abs(candidates) > step) &
               (np.abs(candidates-jump) > step)] -= 1.0
    # Monotone staircase from the header offset (0) to the EOF offset (jump):
    # with the candidate axis flipped for a negative jump, an offset index may
    # only stay or grow. Every step costs SPLICE_STEP_PENALTY, including
    # leaving 0 at the start and not ending on the jump.
    flip = jump < 0
    work = scores[:, ::-1] if flip else scores
    count = work.shape[1]
    origin = int(np.argmin(np.abs(candidates)))
    end = int(np.argmin(np.abs(candidates-jump)))
    if flip:
        origin, end = count-1-origin, count-1-end
    penalty = SPLICE_STEP_PENALTY
    best = np.empty_like(work)
    choice = np.zeros(work.shape, int)
    best[0] = work[0]-penalty
    best[0, origin] += penalty
    positions_axis = np.arange(count)
    for symbol in range(1, F):
        previous = best[symbol-1]
        # leader[k]: the best earlier offset index (a strictly earlier one).
        running = np.maximum.accumulate(previous)
        upto = np.maximum.accumulate(
            np.where(previous >= running, positions_axis, 0))
        leader = np.concatenate(([0], upto[:-1]))
        moved = previous[leader]-penalty
        moved[0] = -np.inf
        stay = previous >= moved
        best[symbol] = work[symbol]+np.where(stay, previous, moved)
        choice[symbol] = np.where(stay, positions_axis, leader)
    final = best[-1]-penalty
    final[end] += penalty
    path = np.empty(F, int)
    path[-1] = int(np.argmax(final))
    for symbol in range(F-1, 0, -1):
        path[symbol-1] = choice[symbol][path[symbol]]
    if flip:
        path = count-1-path
    return path


def _symbol_offsets(samples, frame_start, scale, jump, model=None):
    """Per-symbol timing offsets across splices, from the cyclic prefixes.

    The body is read at the header (pitch) scale with every candidate offset
    between 0 (header-anchored) and ``jump`` (EOF-anchored), in reference
    samples. Each symbol's cyclic-prefix correlation scores the offsets; the
    offsets may only move from 0 toward ``jump`` (each splice cuts or repeats
    once), and the best such staircase is found by dynamic programming. Any
    number of splices per packet are allowed. With ``model``, each
    correlation peak is also checked against the symbol's pilots. Returns
    (offsets, CP correlations, pilot residuals or None).
    """
    step = SPLICE_OFFSET_STEP
    low, high = min(0.0, jump), max(0.0, jump)
    candidates = np.arange(np.floor(low/step)*step, high+step, step)
    first = PULSE.SYNC_LEN+int(np.floor(candidates[0]))
    span = FRAME+int(np.ceil(candidates[-1]-candidates[0]))+SYM
    positions = float(frame_start)+(first+np.arange(span))*float(scale)
    inside = (positions >= 0) & (positions < len(samples)-1)
    if not inside.any():
        return None
    # Offsets reaching before the capture (a first packet) or past its end
    # read silence there; they only ever score badly.
    reference = np.asarray(_sample_at(
        samples, np.clip(positions, 0, len(samples)-2), taps=4), np.float64)
    if reference.ndim == 1:
        reference = reference[:, None]
    reference[~inside] = 0
    shifts = np.rint(candidates-candidates[0]).astype(int)
    # Windowed sums over CP samples from running sums: the prefix/tail
    # products and energies at every start at once.
    def windowed(values):
        running = np.concatenate(([0.0], np.cumsum(values)))
        return running[CP:]-running[:-CP]
    energy = windowed(np.sum(reference*reference, axis=1))
    product = windowed(np.sum(reference[:-N]*reference[N:], axis=1))
    heads = np.arange(F)[:, None]*SYM+shifts[None, :]
    scores = product[heads]/np.sqrt(
        np.maximum(energy[heads]*energy[heads+N], 1e-24))
    raw = scores
    path = _staircase(raw, candidates, jump)
    identity = None
    if model is not None:
        # The cyclic prefix says where a symbol starts, not which symbol it
        # is; with many splices (large pitch shifts) another symbol's intact
        # copy is often nearby. The pilots say which. Symbols on the first
        # path that sit on either anchor give the packet's channel; then every
        # correlation peak is checked against it, offsets near a peak inherit
        # its residual, the rest count as unidentified, and the path is found
        # again.
        from scipy.ndimage import maximum_filter1d
        reach = max(1, int(round(4/step)))

        def reads(symbols, indexes):
            starts = symbols*SYM+shifts[indexes]+WIN
            return np.stack([reference[start:start+N] for start in starts])
        offsets = candidates[path]
        anchored = np.flatnonzero(
            ((np.abs(offsets) <= step) | (np.abs(offsets-jump) <= step)) &
            (raw[np.arange(F), path] >= .95))
        channel = None
        if len(anchored) >= 3:
            channel = _pilot_channel(_pilot_observations(
                model, anchored, reads(anchored, path[anchored])))
        on_path = _pilot_identity_residual(
            model, np.arange(F), reads(np.arange(F), path), channel)
        picked = np.arange(F), path
        if np.all((on_path <= SPLICE_IDENTITY_MAX) |
                  (raw[picked] < SPLICE_INTACT_CP)):
            # Every intact chosen symbol is who it seems: the common case.
            return candidates[path], raw[picked], on_path
        local = maximum_filter1d(raw, 2*reach+1, axis=1, mode='nearest')
        peaks = np.argwhere((raw >= SPLICE_PEAK_CP) & (raw >= local))
        identity = np.full(raw.shape, 1.0)
        if len(peaks):
            residual = _pilot_identity_residual(
                model, peaks[:, 0], reads(peaks[:, 0], peaks[:, 1]), channel)
            for (symbol, index), value in zip(peaks, residual):
                low, high = max(0, index-reach), index+reach+1
                identity[symbol, low:high] = np.minimum(
                    identity[symbol, low:high], value)
        path = _staircase(raw-SPLICE_IDENTITY_WEIGHT*identity, candidates,
                          jump)
    picked = np.arange(F), path
    return (candidates[path], raw[picked],
            None if identity is None else identity[picked])


def _splice_map(samples, frame_start, marker_end, scale, affine_indexes,
                force=False, model=None):
    """Splice-aware body/metadata sample map, or None to keep the affine map.

    Symbols are read at the header (pitch) scale with their own offsets
    (_symbol_offsets): header-anchored before a splice, EOF-anchored after
    it. Symbols a splice falls in are returned as erasures. Used when the
    packet is longer or shorter than its header scale predicts and the map
    beats the packet-wide affine map by SPLICE_MIN_GAIN.
    """
    scale = float(scale)
    jump = (float(marker_end)-(float(frame_start)+PULSE_FRAME*scale))/scale
    if abs(jump) < SPLICE_MIN_JUMP and not force:
        return None
    if not -(1-SPLICE_MIN_LENGTH)*PULSE_FRAME <= jump <= \
            SPLICE_MAX_FRACTION*PULSE_FRAME:
        return None
    found = _symbol_offsets(samples, frame_start, scale, jump, model)
    if found is None:
        return None
    offsets, chosen, identity = found
    # The body never reaches the endpoint's offset: the endpoint belongs to a
    # later packet (this one's own end and the next header are lost).
    if abs(float(offsets[-1])-jump) > SPLICE_UNREACHED*PULSE_FRAME:
        return None
    affine = _cp_scores_at(samples, affine_indexes)
    if float(np.sum(chosen)) < float(np.sum(affine))+SPLICE_MIN_GAIN:
        return None
    reference = PULSE.SYNC_LEN+np.arange(FRAME)
    indexes = float(frame_start)+(
        reference+np.repeat(offsets, SYM))*scale
    metadata_indexes = float(marker_end)+(
        PULSE.SYNC_LEN+FRAME+np.arange(META_SYMBOL)-PULSE_FRAME)*scale
    erased = chosen < SPLICE_INTACT_CP
    if identity is not None:
        erased |= identity > SPLICE_IDENTITY_MAX
    cuts = np.flatnonzero(np.abs(np.diff(offsets)) >= SPLICE_MIN_JUMP)
    # A splice inside the metadata symbol leaves one of its two anchorings
    # intact; the decoder tries the header-anchored one if the EOF one fails.
    metadata_before = float(frame_start)+(
        PULSE.SYNC_LEN+FRAME+np.arange(META_SYMBOL)+offsets[-1])*scale
    return indexes, metadata_indexes, {
        'metadata_before': metadata_before,
        'jump': jump*scale, 'cuts': [int(cut)+1 for cut in cuts],
        'cp_score': float(np.sum(chosen)),
        'affine_cp_score': float(np.sum(affine)),
        'damaged_symbols': int(np.count_nonzero(erased)),
        'erased': erased,
    }


def _measure_pulse_after_eof_marker(samples, cursor, scale,
                                    min_scale, max_scale):
    """Acquire the next packet locally, without rescanning the remaining tape."""
    margin = 32*float(scale)
    search_start = max(0, int(cursor-margin))
    search_end = min(len(samples), int(cursor+
                                        (PULSE.SYNC_LEN+32)*scale))
    hit = PULSE.measure_pulses(samples[search_start:search_end],
                               min_scale=min_scale, max_scale=max_scale)
    if hit is None:
        return None
    return (hit[0]+search_start-int(cursor), hit[1], hit[2])


def _pulse_warp_trusted_scale(local, confidence, average):
    weight = float(np.clip((confidence-.45)/.55, 0.0, 1.0))
    return average + weight*(float(local)-average)


def _pulse_warp_anchor_conflict(average, scale_start, scale_end,
                                confidence_start, confidence_end):
    left = _pulse_warp_trusted_scale(
        scale_start, confidence_start, average)
    right = _pulse_warp_trusted_scale(
        scale_end, confidence_end, average)
    nearly_equal = abs(left-right) <= PULSE_WARP_STATIC_BIAS_DELTA*average
    common_bias = abs((left+right)*.5-average) >= \
        PULSE_WARP_STATIC_BIAS_GATE*average
    return bool(nearly_equal and common_bias)


def _pulse_warp_is_near_linear(average, scale_start, scale_end,
                               confidence_start, confidence_end):
    left = _pulse_warp_trusted_scale(
        scale_start, confidence_start, average)
    right = _pulse_warp_trusted_scale(
        scale_end, confidence_end, average)
    return max(abs(left/average-1), abs(right/average-1)) <= \
        PULSE_WARP_MIN_SLOPE_DELTA


def _pulse_warp_map(frame_start, next_start, scale_start, scale_end,
                    confidence_start, confidence_end, positions,
                    endpoint_position=None, endpoint_coordinate=None):
    """Map reference packet coordinates through a pulse-anchored Hermite warp.

    Each pulse fit estimates local scale around the center of its edge word;
    the measured packet interval fixes the integrated scale between them.
    Confidence blends each local slope toward that interval average before the
    monotone-curve check. The returned pair is (capture positions, derivative
    values at the curve's extrema).
    """
    span = float(PULSE_FRAME)
    average = (float(endpoint_position if endpoint_position is not None
                      else next_start)-float(frame_start))/span
    if (not np.isfinite(average) or average <= 0 or
            not np.isfinite(scale_start+scale_end) or
            scale_start <= 0 or scale_end <= 0):
        return None

    left_scale = _pulse_warp_trusted_scale(
        scale_start, confidence_start, average)
    right_scale = _pulse_warp_trusted_scale(
        scale_end, confidence_end, average)
    if _pulse_warp_anchor_conflict(
            average, scale_start, scale_end,
            confidence_start, confidence_end):
        return None
    center = PULSE_PREAMBLE_CENTER
    t0 = center
    t1 = (float(endpoint_coordinate) if endpoint_coordinate is not None
          else span+center)
    reference_span = t1-t0
    if reference_span <= 0:
        return None
    x0 = float(frame_start)+center*left_scale
    x1 = (float(endpoint_position) if endpoint_position is not None else
          float(next_start)+center*right_scale)
    cubic_a = 2*x0-2*x1+reference_span*(left_scale+right_scale)
    cubic_b = (-3*x0+3*x1-reference_span*(2*left_scale+right_scale))
    cubic_c = reference_span*left_scale

    def evaluate(at):
        z = (at-t0)/reference_span
        return ((cubic_a*z+cubic_b)*z+cubic_c)*z+x0

    requested = np.asarray(positions, dtype=float)
    if not np.all(np.isfinite(requested)):
        return None
    if (np.min(requested) < t0 or np.max(requested) > t1):
        return None
    # The derivative is quadratic in normalized interval position. Checking
    # both endpoints and its interior extremum is cheaper and more reliable
    # than allocating a derivative value for every audio sample to be mapped.
    derivative_a = (6*(x0-x1)/reference_span+
                    3*(left_scale+right_scale))
    derivative_b = (6*(x1-x0)/reference_span-
                    4*left_scale-2*right_scale)
    extrema = [0.0, 1.0]
    if abs(derivative_a) > 1e-12:
        vertex = -derivative_b/(2*derivative_a)
        if 0 < vertex < 1:
            extrema.append(float(vertex))
    local_scales = (derivative_a*np.square(extrema) +
                    derivative_b*np.asarray(extrema) + left_scale)
    if (not np.all(np.isfinite(local_scales)) or
            np.min(local_scales) <= .5*average or
            np.max(local_scales) >= 1.5*average):
        return None
    return evaluate(requested), local_scales


def _decode_pulse_samples(model, samples, diagnostics, latest_only, models,
                           model_factory, force_float32, state,
                           pulse_starts=None, sample_rate=RATE,
                           pilot_timing='baseline',
                           pilot_speed_diagnostics=False,
                           pulse_timing='baseline',
                           tone_equalization='off'):
    sample_rate = float(sample_rate)
    cursor = 0
    counter = 1
    results = []
    measured = None
    pending_aspect = 0
    aspect_screen = False           # the last metadata's screen bit
    eof_markers_validated = 0
    selected_marker = None
    last_metadata = None
    tone_witnesses = 0
    previous_candidate = None
    min_scale, max_scale = pulse_sample_scale_bounds(sample_rate)
    # EOF-mode acquisition revisits one packet at a time. Cache this mono
    # view so marker checks and local pulse reacquisition do not repeatedly
    # average the entire capture for every frame.
    mono_samples = _mono(samples)
    if latest_only:
        # The live rolling buffer can contain the previous frame plus the new
        # one. Reuse the input layer's incremental header hits when available;
        # otherwise scan the window as before. In either case only the newest
        # complete frame needs the expensive image decode.
        cached = (_remeasure_pulse_starts(samples, pulse_starts, sample_rate,
                                          mono=mono_samples)
                  if pulse_starts is not None else [])
        starts = cached if cached else pulse_frame_starts(samples, sample_rate)
        candidates = [(fs, sc, conf, 0) for fs, sc, conf in starts]
        if not candidates:
            return [], {'frames': 0, 'pulse_frames': 0, 'recovered': False}
        # LiveInput wakes when a packet's end marker is in, so the
        # newest candidate is normally the complete one. Choose the
        # newest candidate whose marker validates: if a later header has
        # already arrived, its packet is not complete yet and the one
        # before it is.
        selected = None
        frame_min, frame_max = pulse_sample_scale_bounds(sample_rate)
        for position, candidate in reversed(list(enumerate(candidates))):
            candidate_start, candidate_scale, _, _ = candidate
            marker = _packet_end(
                mono_samples, candidate_start, candidate_scale,
                frame_min, frame_max)
            if (marker is not None and
                    frame_min*.98 <= marker['packet_scale'] <= frame_max*1.02):
                selected = candidate
                if position > 0:
                    previous_candidate = candidates[position-1]
                # Keep the validated marker: the walk below commits this
                # same packet and must not count its EOF a second time.
                selected_marker = marker
                break
        if selected is None:
            return [], {'frames': 0, 'pulse_frames': 0, 'recovered': False}
        fs, sc, conf, pending_aspect = selected
        # The walk below restarts from a whole sample (an array index), so
        # the measured start's fraction rides in the position: the packet is
        # timed from where its header was measured, not from the sample
        # before it.
        cursor = int(fs)
        measured = (fs-cursor+16*sc, sc, conf)
    while cursor + PULSE.SYNC_LEN + META_SYMBOL + 32 < len(samples):
        if measured is None:
            pulse_samples = (mono_samples[cursor:] if mono_samples is not None
                             else _mono(samples[cursor:]))
            measured = PULSE.measure_pulses(pulse_samples,
                                          min_scale=min_scale,
                                          max_scale=max_scale)
        if measured is None:
            break
        position, scale, confidence = measured
        position += cursor
        # measure_pulses() returns the first preamble edge (the encoded frame
        # starts 16 samples earlier), matching Receiver.pending's at-16*scale
        # correction.
        frame_start = position - 16*scale
        next_start = None
        aspect_code = pending_aspect
        following_valid = False
        frame_scale = scale
        frame_length = PULSE_FRAME
        following_scale = None
        following_confidence = None
        boundary_diag = None
        if selected_marker is not None:
            marker = dict(selected_marker)
            marker['packet_scale'] = ((marker['end']-frame_start) /
                                      PULSE_FRAME)
            selected_marker = None
        else:
            marker = _packet_end(mono_samples, frame_start, scale,
                                 min_scale, max_scale)
        if marker is not None:
            frame_min, frame_max = pulse_sample_scale_bounds(sample_rate)
            candidate_scale = marker['packet_scale']
            if (frame_min*.98 <= candidate_scale <= frame_max*1.02 and
                    marker['end']-frame_start >
                    .9*SPLICE_MIN_LENGTH*PULSE_FRAME*scale):
                following_valid = True
                next_start = marker['end']
                frame_scale = candidate_scale
                following_scale = marker['scale']
                following_confidence = marker['confidence']
                if marker.get('witness') == 'tones':
                    tone_witnesses += 1
                else:
                    eof_markers_validated += 1
                boundary_diag = marker
        if not following_valid:
            # A frame is committed only after its selected endpoint witness.
            # A whole recording continues at the next header (the packet is
            # held as lost); a live window, or the end of a capture, stops.
            resync = None
            if not latest_only:
                resync_from = int(frame_start+.5*PULSE_FRAME*scale)
                hit = (PULSE.measure_pulses(mono_samples[resync_from:],
                                            min_scale=min_scale,
                                            max_scale=max_scale)
                       if resync_from < len(mono_samples) else None)
                if hit is not None:
                    resync = (resync_from, hit)
            if resync is None:
                break
            results.append(Result(counter, 'lost', state.tail.prior(model),
                                  {'held': True, 'displayable': False,
                                   'reason': 'no_packet_endpoint',
                                   'pulse_confidence': float(confidence)}))
            cursor, measured = resync
            counter += 1
            continue
        start = frame_start + PULSE.SYNC_LEN*scale
        # Consecutive pulse positions provide the packet-average scale. The
        # optional warp bends that straight-line sample map using the local
        # scales fitted at both neighboring pulse words.
        pulse_map = None
        pulse_timing_diag = None
        metadata_indexes = None
        splice_diag = None
        if pulse_timing == 'pulse-warp':
            scale_conflict = _pulse_warp_anchor_conflict(
                frame_scale, scale, following_scale, confidence,
                following_confidence)
            near_linear = _pulse_warp_is_near_linear(
                frame_scale, scale, following_scale, confidence,
                following_confidence)
            if not scale_conflict and not near_linear:
                body_reference = PULSE.SYNC_LEN + np.arange(FRAME)
                metadata_reference = (PULSE.SYNC_LEN+FRAME+
                                     np.arange(META_SYMBOL))
                reference_positions = np.concatenate((body_reference,
                                                      metadata_reference))
                pulse_map = _pulse_warp_map(
                    frame_start, next_start, scale, following_scale,
                    confidence, following_confidence, reference_positions,
                    endpoint_position=next_start,
                    endpoint_coordinate=PULSE_FRAME)
            pulse_timing_diag = {
                'mode_requested': 'pulse-warp',
                'mode_applied': 'baseline',
                'average_scale': float(frame_scale),
                'start_pulse_scale': float(scale),
                'end_pulse_scale': float(following_scale),
            }
            if pulse_map is None:
                pulse_timing_diag['reason'] = (
                    'local_scale_interval_mismatch' if scale_conflict else
                    'pulse_scales_near_average' if near_linear else
                    'invalid_or_non_monotone_map')
            else:
                mapped_indexes, local_scales = pulse_map
                indexes = mapped_indexes[:FRAME]
                metadata_indexes = mapped_indexes[FRAME:]
                pulse_timing_diag.update({
                    'mode_applied': 'pulse-warp',
                    'local_scale_min': float(np.min(local_scales)),
                    'local_scale_max': float(np.max(local_scales)),
                })
        if pulse_map is None:
            # The header origin and measured EOF endpoint define one
            # packet-wide affine time map. Use its scale for both the
            # body origin and every metadata sample; mixing the local
            # preamble scale into either offset breaks that shared clock.
            indexes = (frame_start +
                       (PULSE.SYNC_LEN+np.arange(FRAME))*frame_scale)
            metadata_indexes = (
                frame_start+(PULSE.SYNC_LEN+FRAME+
                             np.arange(META_SYMBOL))*frame_scale)
            spliced = (_splice_map(samples, frame_start, next_start,
                                   scale, indexes, model=model)
                       if boundary_diag is not None else None)
            current_score = (
                spliced[2]['cp_score'] if spliced is not None else
                float(np.sum(_cp_scores_at(samples, indexes))))
            if (boundary_diag is not None and
                    not boundary_diag.get('splice_search') and
                    current_score < F*SPLICE_SUSPECT_CP):
                # The body does not fit the EOF found where the header
                # predicted it: a splice can move the real mark out of
                # reach while something else matches there. Ask the next
                # header, and keep whichever endpoint fits the body.
                alternate = _spliced_eof_marker(
                    mono_samples, frame_start, scale, min_scale,
                    max_scale, exclude_end=next_start)
                alternate_map = (
                    _splice_map(samples, frame_start, alternate['end'],
                                scale, indexes, force=True, model=model)
                    if alternate is not None else None)
                if (alternate_map is not None and
                        alternate_map[2]['cp_score'] >
                        current_score+SPLICE_MIN_GAIN):
                    spliced = alternate_map
                    boundary_diag = alternate
                    next_start = alternate['end']
                    frame_scale = alternate['packet_scale']
                    following_scale = alternate['scale']
                    following_confidence = alternate['confidence']
            if spliced is not None:
                indexes, metadata_indexes, splice_diag = spliced
        if indexes[-1] >= len(samples)-1:
            break
        if confidence < .45:
            lost_diag = {'pulse_confidence': float(confidence), 'held': True}
            if boundary_diag is not None:
                lost_diag['eof_marker'] = boundary_diag
            results.append(Result(counter, 'lost', state.tail.prior(model),
                                  lost_diag))
            pending_aspect = aspect_code
            frame_length = PULSE_FRAME
            cursor = int(next_start if following_valid
                         else frame_start + frame_length*scale)
            if following_valid:
                measured = _measure_pulse_after_eof_marker(
                    mono_samples, cursor, following_scale,
                    min_scale, max_scale)
            else:
                measured = None
            counter += 1
            continue
        body = _sample_at(samples, indexes, taps=4).astype(np.float32)
        if body.shape[1] == 1:
            # The demodulator is M/S two-channel internally.  A mono capture
            # is the shared M observation, so duplicate it without inventing S.
            body = np.repeat(body, 2, axis=1)
        # Metadata is deliberately decoded before the image body.  Its pilots
        # are self-referencing, so the bootstrap model's absolute scale cancels
        # out; the protected encoding ID can therefore select the source model.
        metadata_scale = frame_scale
        meta_start = (frame_start + (PULSE.SYNC_LEN+FRAME)*metadata_scale)
        metadata_candidates = decode_metadata(
            model, samples, meta_start, metadata_scale, None, force_float32,
            sample_indexes=metadata_indexes, return_candidates=True)
        # Slices 0-4 carry a plain CRC; 5 and 6 carry it XOR p / N, which
        # the lock learns and then checks; until then they are accepted only
        # provisionally (see PulseState.accept).
        meta, metadata_valid, provisional = state.accept_candidates(
            metadata_candidates or [])
        if not metadata_valid and splice_diag is not None:
            spliced_candidates = decode_metadata(
                model, samples, meta_start, metadata_scale, None,
                force_float32, sample_indexes=splice_diag['metadata_before'],
                return_candidates=True)
            spliced_meta = state.accept_candidates(spliced_candidates or [])
            if spliced_meta[1]:
                meta, metadata_valid, provisional = spliced_meta
                splice_diag['metadata_anchor'] = 'header'
        if (not metadata_valid and
                pulse_map is None and boundary_diag is not None and
                not boundary_diag.get('splice_search')):
            # The body fitted the EOF found at the predicted place but the
            # metadata did not: a splice near the end can hide the real mark
            # while body content matches there. The next header settles it;
            # the metadata CRC decides between the two endpoints.
            alternate = _spliced_eof_marker(
                mono_samples, frame_start, scale, min_scale, max_scale,
                exclude_end=next_start)
            if alternate is not None:
                header_indexes = frame_start+(
                    PULSE.SYNC_LEN+np.arange(FRAME))*scale
                alternate_map = _splice_map(
                    samples, frame_start, alternate['end'], scale,
                    header_indexes, force=True, model=model)
                if alternate_map is None:
                    alternate_map = (header_indexes, alternate['end']+(
                        PULSE.SYNC_LEN+FRAME+np.arange(META_SYMBOL) -
                        PULSE_FRAME)*scale, None)
                alternate_candidates = decode_metadata(
                    model, samples, meta_start, metadata_scale, None,
                    force_float32, sample_indexes=alternate_map[1],
                    return_candidates=True)
                alternate_meta = state.accept_candidates(
                    alternate_candidates or [])
                if alternate_meta[1]:
                    meta, metadata_valid, provisional = alternate_meta
                    indexes, metadata_indexes, splice_diag = alternate_map
                    boundary_diag = alternate
                    next_start = alternate['end']
                    frame_scale = alternate['packet_scale']
                    following_scale = alternate['scale']
                    following_confidence = alternate['confidence']
                    if indexes[-1] >= len(samples)-1:
                        break
                    body = _sample_at(samples, indexes, taps=4).astype(
                        np.float32)
                    if body.shape[1] == 1:
                        body = np.repeat(body, 2, axis=1)
        metadata_predicted = False
        if (not metadata_valid and latest_only and splice_diag is not None and
                previous_candidate is not None and
                .65*PULSE_FRAME*scale <= frame_start-previous_candidate[0] <=
                1.35*PULSE_FRAME*scale):
            # The live window holds the packet before this one; its metadata
            # (one more symbol) stands in as the neighbour.
            previous_start, previous_scale = previous_candidate[:2]
            previous_meta = state.accept_candidates(decode_metadata(
                model, samples, previous_start+(PULSE.SYNC_LEN+FRAME) *
                previous_scale, previous_scale, None, force_float32,
                return_candidates=True) or [])
            if previous_meta[1] and not previous_meta[2]:
                last_metadata = (previous_meta[0], counter-1)
        if (not metadata_valid and last_metadata is not None and
                splice_diag is not None and
                1 <= counter-last_metadata[1] <= 3):
            # A splice can take the metadata symbol while the body is intact.
            # The neighbour fixes it (the previous packet in a recording, or
            # in the live window): the same encode and aspect, the slice and
            # index advanced by the packets elapsed.
            # Only the rotating tail depends on the slice, and the frame must
            # still pass every quality gate below.
            previous, previous_counter = last_metadata
            elapsed = counter-previous_counter
            meta = replace(previous,
                           tail_slice=(previous.tail_slice+elapsed) %
                           TAIL_PHASES,
                           source_index=(previous.source_index +
                                         elapsed*(previous.direction or 1)) %
                           (MAX_SOURCE_INDEX+1))
            metadata_valid, provisional = True, True
            metadata_predicted = True
        metadata_retry = None
        if not metadata_valid and pilot_timing != 'baseline':
            # A failed CRC may be a timing miss rather than lost metadata bits.
            # The reference tones continue through the metadata symbol, so use
            # their body-to-metadata phase advance for one corrected retry.
            retry_scale = frame_scale
            retry_start = frame_start + (PULSE.SYNC_LEN+FRAME)*retry_scale
            retry_metadata_indexes = metadata_indexes
            shift, metadata_retry = _pilot_metadata_offset(
                body, samples, retry_start, retry_scale, model, counter,
                force_float32,
                estimator=('joint' if pilot_timing == 'tone-joint'
                           else 'single'),
                metadata_indexes=retry_metadata_indexes)
            if shift is not None:
                if pulse_map is None:
                    corrected_start = retry_start-shift*retry_scale
                    retry_candidates = decode_metadata(
                        model, samples, corrected_start, retry_scale, None,
                        force_float32, return_candidates=True)
                else:
                    corrected_map = _pulse_warp_map(
                        frame_start, next_start, scale, following_scale,
                        confidence, following_confidence,
                        PULSE.SYNC_LEN+FRAME+
                        np.arange(META_SYMBOL)-shift,
                        endpoint_position=next_start,
                        endpoint_coordinate=PULSE_FRAME)
                    corrected_indexes = (None if corrected_map is None else
                                         corrected_map[0])
                    retry_candidates = (decode_metadata(
                        model, samples, retry_start, retry_scale, None,
                        force_float32, sample_indexes=corrected_indexes,
                        return_candidates=True)
                        if corrected_indexes is not None else None)
                retry_meta, retry_valid, retry_provisional = \
                    state.accept_candidates(retry_candidates or [])
                metadata_retry.update({
                    'retried': True, 'valid': bool(retry_valid),
                    'provisional': bool(retry_provisional),
                })
                if retry_valid:
                    meta = retry_meta
                    metadata_valid = retry_valid
                    provisional = retry_provisional
            else:
                metadata_retry['retried'] = False
        encoding_type = model.encoding_type
        source_index = None
        tail_slice = None
        direction = None
        if metadata_valid:
            if not metadata_predicted:
                last_metadata = (meta, counter)
            aspect_code = meta.aspect_code
            aspect_screen = bool(meta.screen_aspect)
            encoding_type = meta.encoding_type
            source_index = meta.source_index
            tail_slice = meta.tail_slice
            direction = meta.direction
        selected_model = model
        if metadata_valid:
            selected_model = (models or {}).get(encoding_type)
            if selected_model is None and model_factory is not None:
                selected_model = model_factory(encoding_type)
            if selected_model is None:
                selected_model = model
        nominal = np.array([0., FRAME])
        offset = np.array([64., 64.])
        # The rank table follows the tail slice the packet names; without
        # verified metadata the slice is unknown and the frame is not trusted
        # (status lost), so the local counter is only a placeholder.
        ranks_counter = counter if tail_slice is None else tail_slice
        try:
            result = decode_frame(selected_model, None,
                                  (ranks_counter, nominal, offset),
                                  ranks_counter, state.tail.prior(selected_model),
                                  cancel=False, direct_body=body,
                                  diagnostics=diagnostics,
                                  force_float32=force_float32,
                                  pilot_timing=pilot_timing,
                                  pilot_counter=counter,
                                  tone_equalization=tone_equalization,
                                  profile_hint={
                                      'aspect_code': (aspect_code if
                                                      metadata_valid else None),
                                      'encoding_type': (encoding_type if
                                                        metadata_valid else None)},
                                  **({'erased_symbols': splice_diag['erased']}
                                     if splice_diag is not None else {}))
        except (FloatingPointError, np.linalg.LinAlgError, ValueError,
                IndexError):
            result = None
        if result is not None:
            result.counter = counter
            if result.status != 'lost':
                result.status = 'received' if confidence >= .45 else 'degraded'
            result.diag.pop('_H', None)
            if (result.status != 'lost' and
                    (not metadata_valid or
                     result.diag.get('head_confidence', 0) <
                     LIVE_VALID_HEAD_CONFIDENCE*result.diag.get(
                         'present_symbols', 1.0) or
                     result.diag.get('head_coverage', 0) <
                     LIVE_VALID_HEAD_COVERAGE*result.diag.get(
                         'present_symbols', 1.0) or
                     max(result.diag.get('noise', [np.inf])) >
                     LIVE_MAX_PILOT_NOISE)):
                result.status = 'lost'
            result.diag['displayable'] = bool(
                result.diag.get('displayable', False) and
                max(result.diag.get('noise', [np.inf])) <=
                DISPLAY_MAX_PILOT_NOISE)
            result.diag['pulse_confidence'] = float(confidence)
            result.diag['aspect_code'] = aspect_code
            result.diag['aspect_screen'] = aspect_screen
            result.diag['metadata_valid'] = metadata_valid
            if source_index is not None:
                result.diag['source_index'] = int(source_index)
            result.diag['encoding_type'] = int(encoding_type)
            result.diag['encoding_name'] = ENCODING_FILTERS[int(encoding_type)] \
                if 0 <= int(encoding_type) < len(ENCODING_FILTERS) else 'unknown'
            result.diag['tail_slice'] = tail_slice
            result.diag['direction'] = direction
            result.diag['metadata_provisional'] = provisional
            if metadata_retry is not None:
                result.diag['metadata_pilot_retry'] = metadata_retry
            if metadata_predicted:
                result.diag['metadata_predicted'] = True
            if pulse_timing_diag is not None:
                result.diag['pulse_timing'] = pulse_timing_diag
            if boundary_diag is not None:
                result.diag['eof_marker'] = boundary_diag
            if splice_diag is not None:
                result.diag['splice'] = {
                    key: value for key, value in splice_diag.items()
                    if key not in ('erased', 'metadata_before')}
            result.diag['loop'] = state.lock.loop
            result.diag['pulse_scale'] = float(scale)
            result.diag['frame_scale'] = float(frame_scale)
            result.diag['frame_start'] = float(frame_start)
            result.diag['playback_speed'] = float(
                sample_rate/(RATE*max(frame_scale, 1e-9)))
            result.diag['timing_delta_ppm'] = float(
                (frame_scale/scale-1)*1e6)
            if pilot_speed_diagnostics:
                result.diag['pilot_tone_speed'] = pilot_tone_speed(
                    samples, sample_rate, frame_start, frame_scale)
            results.append(result)
            if (result.status != 'lost' and tail_slice is not None and
                    not (state._require_independent_tail_metadata and
                         provisional)):
                state.tail.update(selected_model, result.coeffs, tail_slice)
            if latest_only:
                break
        pending_aspect = aspect_code
        frame_length = PULSE_FRAME
        cursor = int(next_start if following_valid
                     else frame_start + frame_length*scale)
        if following_valid:
            measured = _measure_pulse_after_eof_marker(
                mono_samples, cursor, following_scale, min_scale, max_scale)
        else:
            measured = None
        counter += 1
    info = {'frames': len(results), 'pulse_frames': len(results),
            'recovered': bool(results),
            'eof_markers_validated': eof_markers_validated,
            'tone_witnesses': tone_witnesses,
            }
    if diagnostics is not None:
        info['diagnostics'] = _diagnostic_summary(diagnostics, 0.0)
    return results, info


def values_from(model, coeffs):
    return model.coder.inverse(coeffs*model.coder.gains)
