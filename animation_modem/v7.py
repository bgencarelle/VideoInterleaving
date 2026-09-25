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
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
from numba import njit
from PIL import Image
from scipy.linalg import hadamard
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt, firwin, savgol_filter, sosfiltfilt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from animation_modem import transport3 as PULSE                             # noqa: E402
from animation_modem.v7_core import (SourceCoder, _sample_at, speed_length,
                                  speed_resample)                  # noqa: E402
from animation_modem.v7_core import bound_emission                         # noqa: E402

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
EOF_MARKER_MIN_LEVEL = .08
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
PULSE_MAX_SCALE = 4.0
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
    not the admissible playback-speed range.
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
    return image.convert('RGB').resize(PREPARED_SIZE, resampling)


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
REFINE = True
DEBUG = {}
# The FFT window opens CP-WIN samples early: undo that known linear phase so
# channel interpolation between pilots sees a smooth response.
EARLY = np.exp(2j*np.pi*np.arange(65)*(CP-WIN)/N)


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
    # carries it in metadata_symbols().  The longer clock-only bench wire
    # retains its historical identity fields for now.
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
#   byte 0   aspect (3) | encode filter (2) | tail slice (3)
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
    wire_index = int(source_index) + 1
    if int(direction) < 0:
        wire_index |= INDEX_DOWN
    payload = bytes([((int(aspect_code) & 7) << 5) |
                     ((int(encoding_type) & 3) << 3) |
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
    return Metadata((raw[0] >> 5) & 7, (raw[0] >> 3) & 3, tail_slice,
                    wire_index - 1, mask,
                    -1 if field & INDEX_DOWN else 1)


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

    def prior(self, model):
        if (not self.enabled or self._values is None or
                self._encoding != model.encoding_type):
            return model.mu.copy()
        return np.where(self._age <= self.max_age, self._values, model.mu)

    def update(self, model, coeffs, tail_slice):
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
                       abs(meta.source_index - last.source_index) <=
                       PROVISIONAL_INDEX_WINDOW)
        return provisional, provisional


def aspect_wire_code(size):
    """Return the compact V7 family/orientation code for a source size."""
    width, height = size
    ratio = width/height
    families = (1., 4/3, 3/2, 16/9)
    family = min(range(4), key=lambda i: min(
        abs(np.log(ratio/families[i])),
        abs(np.log(ratio/(1/families[i])))))
    return family | (4 if ratio < 1 and family else 0)


def parse_word(bits):
    if bits[:16] != SYNC or sum(bits) % 2:
        return None
    payload = bits[16:55]
    data = bytes(int(''.join(map(str, ([0]+payload)[i:i+8])), 2) for i in range(0, 40, 8))
    if int(''.join(map(str, bits[55:71])), 2) != crc16(data):
        return None
    val = lambda a, b: int(''.join(map(str, payload[a:b])), 2)
    return {'counter': val(0, 20), 'profile': val(20, 24), 'aspect': val(24, 27),
            'folders': val(27, 35)}


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
LIVE_MAX_PILOT_NOISE = .08
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
def encode_frame(model, values, counter):
    return encode_frame_coeffs(model, model.coder.forward(values)/model.coder.gains, counter)


def encode_frame_coeffs(model, coeffs, counter, return_X=False):
    c = coeffs - model.mu
    idx = model.rank_tables[counter % TAIL_PHASES]
    X = np.zeros((F, 65, 2), complex)                        # symbol, bin, M/S
    ranks = np.maximum(idx, 0)
    vals = model.gain[ranks]*c[ranks]
    vals[idx < 0] = 0
    tx = vals @ H8.T
    cells = X.reshape(-1)
    cells.real[CELLS_I] = tx[GROUP_IS_I].ravel()
    cells.imag[CELLS_Q] = tx[~GROUP_IS_I].ravel()
    X += SCATTERED_PILOTS
    if return_X:
        return X
    XL = (X[..., 0]+X[..., 1])/np.sqrt(2)*model.phase
    XR = (X[..., 0]-X[..., 1])/np.sqrt(2)*model.phase
    waves = np.fft.irfft(np.stack((XL, XR), axis=-1), n=N, axis=1)
    waves *= model.scale
    out = np.concatenate((waves[:, -CP:, :], waves), axis=1).reshape(-1, 2)
    return out


_EMIT = firwin(63, 13500, fs=RATE, window=('kaiser', 6))
# Pulse packets are band-limited here (bound_emission in encode_pulse_frame).
EMISSION_EDGE_HZ = 14000


def max_wire_speed(rate):
    """Playback speed below which a `rate` Hz output retains the full band.

    Speeding the wire up multiplies every frequency, so the emission edge
    (14 kHz at 1x) stays below the output's Nyquist frequency up to this
    speed. Faster playback is permitted, but the speed conversion filters or
    aliases high carriers and can reduce decode quality.
    """
    return float(rate)/(2*EMISSION_EDGE_HZ)


def encode_stream(model, values, frames, lead=0.25, tail=0.25,
                  start_counter=1):
    """Encode a finite stream, optionally continuing the clock counter.

    Bench callers retain the original default. Live callers use
    ``start_counter`` so successive batches form one logical frame sequence;
    the batch boundary still remains a prototype limitation because the
    current shaping path is offline rather than stateful.
    """
    if np.asarray(values).ndim == 1:
        frames_values = [values]*frames
    else:
        frames_values = list(values)
        if len(frames_values) != frames:
            raise ValueError('values must contain exactly one vector per frame')
    body = np.concatenate([encode_frame(model, frame, start_counter+i)
                           for i, frame in enumerate(frames_values)])
    body = filtfilt(_EMIT, [1.0], body, axis=0)               # 14 kHz bound, no renorm
    words = [clock_word(start_counter+i) for i in range(frames)]
    clk = clock_wave(words)
    ofdm_rms = np.sqrt(np.mean(body**2))
    clk *= ofdm_rms*10**(CLOCK_REL_DB/20)/np.sqrt(np.mean(clk**2))
    sig = body + clk[:, None]
    sig = np.clip(sig, -0.89, 0.89)                           # -1 dBFS safety limiter
    pad = lambda s: np.zeros((int(s*RATE), 2))
    return np.concatenate([pad(lead), sig, pad(tail)]).astype(np.float32)


def encode_pulse_frame(model, values, counter, aspect_code=0, source_index=None,
                       loop=None, direction=1, pilot_tones=False,
                       pilot_tone_gate_preamble=False, eof_marker=False):
    """One edge-counted pulse-framed V7 body for low-latency live transport.

    ``counter`` is the packet count: counter mod 7 picks the tail slice, which
    the metadata names so the receiver places the tail correctly.  ``loop``
    (LoopInfo) is carried in the rotating CRC field for receivers that want
    the loop length and the live lag; None sends no loop information.
    """
    body = encode_frame(model, values, counter)
    out = np.zeros((PULSE_FRAME, 2), np.float32)
    out[PULSE.SYNC_LEN:PULSE.SYNC_LEN+FRAME] = body
    out[16:16+len(PULSE.PREAMBLE), :] = PULSE.PREAMBLE[:, None]
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
    # Shape the ordinary pulse/body packet first.  The metadata symbol has its
    # own cyclic prefix and is inserted afterward so the long packet shaper
    # cannot smear the preceding image symbol across its pilots/data.
    shaped = bound_emission(out, EMISSION_EDGE_HZ, RATE)
    shaped[meta_start:meta_start+META_SYMBOL, :] += meta_pcm[:, None]
    if pilot_tones:
        # Add after the per-packet shaper so phase is exact across packets.
        # Time compression remains downstream and shifts the tone frequencies
        # together with the pulse wire.
        shaped = _add_pilot_tones(shaped, counter,
                                  gate_preamble=pilot_tone_gate_preamble)
    if eof_marker:
        marker = np.concatenate([
            np.full(run, level, np.float32)
            for run, level in zip(EOF_MARKER_RUNS, EOF_MARKER_LEVELS)
        ])*EOF_MARKER_LEVEL
        shaped[EOF_MARKER_OFFSET:PULSE_FRAME, :] += marker[:, None]
    return shaped


def encode_pulse_stream(model, values, start_counter=1, aspect_codes=None,
                        source_indices=None, loop=None, directions=None,
                        pilot_tones=False, pilot_tone_gate_preamble=False,
                        eof_marker=False):
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
                           pilot_tone_gate_preamble, eof_marker)
        for i, (value, code, source_index, way) in
        enumerate(zip(values, codes, indexes, ways))])


# ------------------------------------------------------------------ receiver: clock
_CLK_BP = butter(4, [300, 1300], btype='bandpass', fs=RATE, output='sos')


def clock_signal(x):
    return sosfiltfilt(_CLK_BP, np.asarray(x, float))


def clock_edges(y):
    """Zero crossings confirmed by a Schmitt trigger (floor stops idle noise)."""
    env = np.sqrt(np.convolve(y*y, np.ones(480)/480, mode='same'))*np.sqrt(2)
    th = np.maximum(0.4*env, 0.1*np.percentile(env, 90))
    edges, state = [], 0
    for i in range(1, len(y)):
        if state <= 0 and y[i] > th[i] or state >= 0 and y[i] < -th[i]:
            sign = 1 if y[i] > 0 else -1
            j = i
            while j > 0 and np.sign(y[j-1]) == sign:
                j -= 1
            if j > 0:
                edges.append(j-1 + y[j-1]/(y[j-1]-y[j]))
            state = sign
    return np.asarray(edges)


def biphase_bits(y, edges):
    """Clock-recovery decoder: lock to bit boundaries, read bits by polarity.

    A boundary always has a transition; a `1` adds one mid-bit. The loop
    predicts the next boundary, snaps to the nearest edge within +-U/4 (else
    coasts), and reads the bit as sign(y at U/4) != sign(y at 3U/4). A run of
    boundaries without edges means we locked to mid-bits: shift half a bit.
    """
    out = []
    if len(edges) < 16:
        return out
    gaps = np.diff(edges)
    unit = 2*np.percentile(gaps, 25)
    t, k, misses = edges[0], 0, []
    while t + unit < len(y)-2 and k < len(edges):
        lo, hi = t - unit/3, t + unit/3
        while k < len(edges) and edges[k] < lo:
            k += 1
        hit = k < len(edges) and edges[k] <= hi
        if hit:
            err = edges[k]-t
            t += 0.25*err; unit += 0.02*err
        misses.append(not hit)
        if len(misses) > 8 and sum(misses[-8:]) >= 3:
            t += unit/2; misses = []; out.append((None, t)); continue
        if t+unit >= len(y):
            break
        # A `1` has an edge in the middle half of the bit; a `0` has none.
        # Tolerates +-U/4 of pattern-dependent edge shift (tape phase error).
        j = np.searchsorted(edges, t+unit/4)
        bit = int(j < len(edges) and edges[j] < t+3*unit/4)
        out.append((bit if hit else None, t))
        t += unit
    return out


def find_words(bitstream, max_sync_errors=3):
    """Frame words from a bit stream, tolerant of bit errors (§5.3 steps 5, 7).

    Frame starts are sync matches with <= max_sync_errors differences that are
    confirmed by another near-sync 72 bits before or after. CRC-valid words
    carry a verified counter; the rest inherit counters from the nearest
    verified word by position, or get relative counters if none verified.
    """
    bits = [b for b, _ in bitstream]
    pos = [p for _, p in bitstream]
    n = len(bits)

    def dist(i):
        if i < 0 or i+16 > n:
            return 99
        return sum(1 if b is None else int(b != s_) for b, s_ in zip(bits[i:i+16], SYNC))

    d = [dist(i) for i in range(n)]
    starts = []
    for i in range(n-72):
        if d[i] > max_sync_errors:
            continue
        if min(d[i-72] if i >= 72 else 99, d[i+72] if i+72 < n else 99) > max_sync_errors+1 and d[i] > 0:
            continue
        if starts and i - starts[-1] < 60:                 # keep the better of overlapping hits
            if d[i] < d[starts[-1]]:
                starts[-1] = i
            continue
        starts.append(i)
    frames = []
    for i in starts:
        if i+72 >= n:
            continue
        chunk = bits[i:i+72]
        w = parse_word([0 if b is None else b for b in chunk]) if None not in chunk else None
        frames.append({'start': i, 'bounds': pos[i:i+73], 'counter': w['counter'] if w else None,
                       'verified': bool(w)})
    verified = [f for f in frames if f['verified']]
    period = FRAME
    if len(verified) >= 2:
        span = verified[-1]['bounds'][0]-verified[0]['bounds'][0]
        count = verified[-1]['counter']-verified[0]['counter']
        if count > 0:
            period = span/count
    for f in frames:
        if f['verified']:
            continue
        if verified:
            ref = min(verified, key=lambda v: abs(v['bounds'][0]-f['bounds'][0]))
            f['counter'] = ref['counter'] + int(round((f['bounds'][0]-ref['bounds'][0])/period))
        elif frames:
            f['counter'] = 1 + int(round((f['bounds'][0]-frames[0]['bounds'][0])/period))
    # One frame per counter: prefer verified, then the lower sync distance.
    best = {}
    for f in frames:
        c = f['counter']
        if c is None:
            continue
        if c not in best or (f['verified'] and not best[c]['verified']):
            best[c] = f
    out = [best[c] for c in sorted(best)]
    # Timing trust (flywheel through slips): a frame's bit boundaries enter the
    # time map only if its start agrees with its neighbours' prediction, and
    # only up to the first internal slip.
    starts = {f['counter']: f['bounds'][0] for f in out}
    anchors = [f for f in out if f['verified']] or out
    for f in out:
        c = f['counter']
        others = [a for a in anchors if a['counter'] != c]
        if not others:
            f['timing'] = f['verified']; continue
        near = sorted(others, key=lambda a: abs(a['counter']-c))[:2]
        preds = [a['bounds'][0] + (c-a['counter'])*period for a in near]
        f['timing'] = f['verified'] or min(abs(f['bounds'][0]-q) for q in preds) <= 8
        b = np.asarray(f['bounds']); d = np.diff(b); unit = period/72
        bad = np.flatnonzero((d < .75*unit) | (d > 1.25*unit))
        f['good'] = int(bad[0]) + 1 if len(bad) else len(b)
    return out


def time_map(words):
    """Nominal-sample -> received-position map from verified bit boundaries."""
    if not words:
        return None
    c0 = words[0]['counter']
    nom, rec = [], []
    for w in words:
        if not w.get('timing', True):
            continue
        base = (w['counter']-c0)*FRAME
        for k, p in enumerate(w['bounds'][:w.get('good', len(w['bounds']))]):
            nom.append(base+k*BIT); rec.append(p)
    nom, idx = np.unique(np.asarray(nom, float), return_index=True)
    rec = np.asarray(rec)[idx]
    d = rec-nom
    # Smooth per contiguous run (PLL stand-in, §5.3 step 4).
    runs = np.split(np.arange(len(nom)), np.flatnonzero(np.diff(nom) > BIT*1.5)+1)
    for r in runs:
        if len(r) >= 31:
            d[r] = savgol_filter(d[r], 31, 2)
    return c0, nom, d


def refine_time_map(y, tm, words, win=960, hop=240, iters=4):
    """Least-squares alignment of the regenerated clock to the received band.

    Pulse counting gives lock and identity; this gives precision. The template
    is the clock the decoded words imply, passed through the same band-pass,
    so pattern-dependent edge shifts are in both and cancel.
    """
    c0, nom, d = tm
    counters = sorted({w['counter'] for w in words})
    first, last = counters[0], counters[-1]
    have = {w['counter'] for w in words if w.get('verified', True) and w.get('timing', True)}
    wave = clock_wave([clock_word(c) if c in have else [0]*72 for c in range(first-1, last+2)])
    tpl = clock_signal(wave)                       # nominal time, starts at frame first-1
    origin = (first-1-c0)*FRAME
    dtpl = np.gradient(tpl)
    pts_n, pts_r = [], []
    for start in range(FRAME, len(tpl)-FRAME-win, hop):
        n0 = origin + start
        if n0 < nom[0] or n0+win > nom[-1]:
            continue
        k = np.arange(win)
        seg_t, seg_d = tpl[start:start+win], dtpl[start:start+win]
        if np.dot(seg_t, seg_t) < 1e-9:
            continue
        r0 = n0 + k + np.interp(n0 + k, nom, d)        # coarse received positions
        delta = 0.0
        for _ in range(iters):
            r = np.interp(r0 + delta, np.arange(len(y)), y)
            a = np.dot(r, seg_t)/np.dot(seg_t, seg_t)
            # r(t+delta) ~ a*tpl(t)  =>  dr/ddelta ~ a*dtpl
            resid = r - a*seg_t
            step = np.dot(resid, a*seg_d)/max(np.dot(a*seg_d, a*seg_d), 1e-12)
            delta -= step
            if abs(step) < 1e-3:
                break
        r = np.interp(r0 + delta, np.arange(len(y)), y)
        quality = np.dot(r, seg_t)/np.sqrt(max(np.dot(r, r)*np.dot(seg_t, seg_t), 1e-24))
        if abs(delta) > 3 or quality < 0.6:
            continue                                   # keep the coarse map here
        pts_n.append(n0 + win/2); pts_r.append(r0[win//2] + delta)
    if len(pts_n) < 8:
        return tm
    pts_n, pts_r = np.asarray(pts_n), np.asarray(pts_r)
    # Coarse points far (> 2 windows) from any refined point fill the gaps.
    far = np.min(np.abs(nom[:, None]-pts_n[None, :]), axis=1) > 2*win
    pts_n = np.concatenate([pts_n, nom[far]]); pts_r = np.concatenate([pts_r, nom[far]+d[far]])
    order = np.argsort(pts_n); pts_n, pts_r = pts_n[order], pts_r[order]
    dd = pts_r - pts_n
    runs = np.split(np.arange(len(pts_n)), np.flatnonzero(np.diff(pts_n) > hop*1.5)+1)
    for r in runs:
        if len(r) >= 9 and np.all(np.diff(pts_n[r]) <= hop*1.01):
            dd[r] = savgol_filter(dd[r], 9, 2)
    return c0, pts_n, dd


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
            candidate_score = _channel_pilot_residual(Z, candidate)
            candidate_timing_residual = _channel_timing_residual(Z, candidate)
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
                base_score = _channel_pilot_residual(Z, baseline)
                baseline_timing_residual = _channel_timing_residual(Z, baseline)
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
            timing_diag['baseline_pilot_residual'] = _channel_pilot_residual(
                Z, result)
            timing_diag['baseline_timing_residual_samples'] = (
                _channel_timing_residual(Z, result))
    else:
        result = _channel_joint_batched(Z, iters, force_float32)
    if return_timing_diag:
        if pilot_timing == 'baseline':
            timing_diag['baseline_pilot_residual'] = _channel_pilot_residual(
                Z, result)
            timing_diag['baseline_timing_residual_samples'] = (
                _channel_timing_residual(Z, result))
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
    base_score = _channel_pilot_residual(Z, result)
    base_timing = _channel_timing_residual(Z, result)
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
            candidate_score = _channel_pilot_residual(Z, candidate)
            candidate_timing = _channel_timing_residual(Z, candidate)
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
    return _channel_residuals_kernel(Z, H, PILOT_SV, PILOT_BV, PILOT_PV, N)


def _channel_pilot_residual(Z, H):
    if _is_float64_frame(Z, H):
        return float(_channel_residuals(Z, H)[0])
    return _channel_pilot_residual_numpy(Z, H)


def _channel_timing_residual(Z, H):
    """Weighted data-pilot phase residual expressed in reference samples."""
    if _is_float64_frame(Z, H):
        return float(_channel_residuals(Z, H)[1])
    return _channel_timing_residual_numpy(Z, H)


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
    the |pred|-weighted pilot phase errors.
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
                    weight = abs(predicted)
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
                w = np.where(ok, np.abs(pred), 0).astype(real_dtype, copy=False)

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
def _equalize_numba_kernel(Z, H, noise, block_symbols, block_bins,
                           block_prior, group_block, group_stream, group_q,
                           ranks, lam, system, weighted, noise_floor, xhat,
                           conf, got):
    """Per-cell MMSE and group LMMSE without NumPy's small-matrix overhead."""
    nblocks, nsym = block_symbols.shape
    est = np.zeros((nblocks, 2, 2, nsym), np.complex128)
    var = np.full((nblocks, 2, 2, nsym), np.inf)
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

    ngroups = ranks.shape[0]
    S = np.empty((8, 8))
    L = np.zeros((8, 8))
    R = np.empty((8, 9))
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
            for j in range(8):
                R[i, j] = weighted[group, i, j]
            R[i, 8] = y[i]
        for i in range(8):
            for column in range(9):
                acc = R[i, column]
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


def _equalize_numba(model, Z, H, noise, counter):
    """Run the compiled float64 equalizer; float32 retains the NumPy path."""
    phase = counter % TAIL_PHASES
    xhat = np.zeros_like(model.mu)
    conf = np.zeros_like(model.mu)
    got = np.zeros(model.mu.shape, np.bool_)
    _equalize_numba_kernel(
        np.ascontiguousarray(Z, dtype=np.complex128),
        np.ascontiguousarray(H, dtype=np.complex128),
        np.ascontiguousarray(noise, dtype=np.float64),
        BLOCK_SYMBOLS, BLOCK_BINS, model.block_prior_tables[phase],
        GROUP_BLOCK, GROUP_STREAM_INDEX, GROUP_Q_INDEX,
        model.rank_tables[phase], model.lam, *_group_systems(model, phase),
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
    audio: channel fit, residuals, fade/noise and the equalizer."""
    shape = (F, 65)
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
                               eof_marker=True).astype(np.float32)
    for stream in (wire, speed_pulse_stream(wire, .8).astype(np.float32)):
        decode_pulse_stream(model, stream, frame_boundary='eof',
                            pilot_timing='tone-seeded')
        decode_pulse_stream(model, stream, latest_only=True,
                            frame_boundary='eof', pilot_timing='tone-seeded')


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
                 tone_equalization='off'):
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
    H, noise, tone_eq_diag = fade_and_noise(
        Z, H, force_float32=force_float32,
        tone_reference=(tone_equalization == 'm-reference'),
        return_tone_diag=True)
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
    if head_confidence < HEAD_MIN_CONFIDENCE or head_coverage < HEAD_MIN_COVERAGE:
        displayable = (head_confidence >= DISPLAY_MIN_HEAD_CONFIDENCE and
                       head_coverage >= DISPLAY_MIN_HEAD_COVERAGE)
        display_coeffs = np.asarray(prev_tail, dtype=real_dtype).copy()
        if displayable:
            display_coeffs[got] = current[got]
        result_diag = {
            'noise': noise.mean(0).tolist(), 'got': int(got.sum()),
            'head_confidence': head_confidence,
            'head_coverage': head_coverage, 'held': not displayable,
            'displayable': displayable,
            'display_coeffs': display_coeffs}
        result_diag.update(timing_metrics)
        return Result(counter, 'lost', display_coeffs if displayable else
                      np.asarray(prev_tail, dtype=real_dtype).copy(), result_diag)
    coeffs[got] = current[got]
    result_diag = {'noise': noise.mean(0).tolist(), 'got': int(got.sum()),
                   'head_confidence': head_confidence,
                   'head_coverage': head_coverage, '_H': H,
                   'displayable': True}
    result_diag.update(timing_metrics)
    return Result(counter, 'verified', coeffs, result_diag)


def decode_metadata(model, samples, start, scale, channel,
                    force_float32=False, sample_indexes=None):
    indexes = (start + np.arange(META_SYMBOL)*scale
               if sample_indexes is None else
               np.asarray(sample_indexes, dtype=float))
    if indexes.shape != (META_SYMBOL,):
        raise ValueError('metadata sample indexes must match META_SYMBOL')
    if indexes[-1] >= len(samples)-1:
        return None
    meta = _sample_at(samples, indexes, taps=4)
    window = meta[WIN:WIN+N]
    if force_float32:
        z = (np.fft.rfft(window.astype(np.float32), axis=0) /
             np.float32(model.scale) *
             np.conj(model.phase32[-1])[:, None] *
             EARLY.astype(np.complex64)[:, None]).astype(np.complex64)
    else:
        z = (np.fft.rfft(window, axis=0)/model.scale *
             np.conj(model.phase[-1])[:, None] * EARLY[:, None])
    # The metadata symbol carries known M=1 pilots.  Estimate its own
    # per-symbol complex response from those pilots; this avoids assuming the
    # body-channel phase is unchanged across the symbol boundary.
    observed = z.sum(axis=1)
    pilot_z = observed[META_PILOTS]
    if np.sum(np.abs(pilot_z) > 1e-6) < 2:
        return None
    response = (np.interp(META_DATA_BINS, META_PILOTS, pilot_z.real) +
                1j*np.interp(META_DATA_BINS, META_PILOTS, pilot_z.imag))
    if force_float32:
        response = response.astype(np.complex64)
    data = np.divide(observed[META_DATA_BINS], response,
                     out=np.zeros(len(META_DATA_BINS),
                                  np.complex64 if force_float32 else complex),
                     where=np.abs(response) > 1e-9)
    symbols = data[:20]
    bits = np.empty(40, np.uint8)
    bits[0::2] = (symbols.real >= 0).astype(np.uint8)
    bits[1::2] = (symbols.imag >= 0).astype(np.uint8)
    return parse_metadata_word(np.packbits(bits).tobytes())


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


def decode_stream(model, x, verbose=False, diagnostics=None,
                  force_float32=False):
    started = perf_counter()
    x = np.asarray(x, np.float32 if force_float32 else float)
    def read(sig):
        y = clock_signal(sig)
        return y, find_words(biphase_bits(y, clock_edges(y)))
    y, words = read(x.mean(axis=1))
    # Per-channel fallback (§5.3 step 1) when M yields few words.
    for ch in range(min(2, x.shape[1])):
        if len(words) >= 2:
            break
        y, words = read(x[:, ch])
    if not words:
        info = {'words': 0}
        if diagnostics is not None:
            info['diagnostics'] = _diagnostic_summary(
                diagnostics, (perf_counter()-started)*1000)
        return [], info
    # Keep words consistent with a monotone counter/time relation.
    words.sort(key=lambda w: w['bounds'][0])
    tm = time_map(words)
    if REFINE:
        tm = refine_time_map(y, tm, words)
    verified = {w['counter'] for w in words if w['verified']}
    allc = {w['counter'] for w in words}
    lo, hi = min(allc), max(allc)
    results, tail = [], (model.mu32.copy() if force_float32
                         else model.mu.copy())
    skipped = []
    for counter in range(lo, hi+1):
        try:
            r = decode_frame(model, x, tm, counter, tail,
                             cancel=bool(verified), diagnostics=diagnostics,
                             force_float32=force_float32)
        except (FloatingPointError, np.linalg.LinAlgError, ValueError,
                IndexError) as exc:
            # A damaged frame is an ordinary transport event.  Do not abort
            # the whole buffered run: later clock words may provide a clean
            # re-lock point and a fresh frame.
            skipped.append({'counter': counter, 'error': type(exc).__name__})
            if diagnostics is not None:
                diagnostics['decode_errors'] = diagnostics.get('decode_errors', 0)+1
            continue
        if r is None:
            skipped.append({'counter': counter, 'error': 'out_of_window'})
            continue
        if r.status == 'lost':
            results.append(r)
            continue
        r.status = 'verified' if counter in verified else 'picture_only'
        tail = r.coeffs.copy()
        results.append(r)
    info = {'words': len(words), 'crc_ok': len(verified),
            'frames': len(results), 'skipped_frames': skipped,
            'recovered': bool(results and skipped)}
    if diagnostics is not None:
        diagnostics['frames'] = diagnostics.get('frames', 0) + len(results)
        counts = diagnostics.setdefault('status_counts', {})
        for result in results:
            counts[result.status] = counts.get(result.status, 0) + 1
        diagnostics['input_samples'] = int(len(x))
        info['diagnostics'] = _diagnostic_summary(
            diagnostics, (perf_counter()-started)*1000)
    return results, info


POLARITY_THRESHOLD = .3


@njit(cache=True, fastmath=False)
def _leg_correlation_sums(x):
    """Mean-removed left/right powers and cross product of a stereo block."""
    count = x.shape[0]
    left_mean = 0.0
    right_mean = 0.0
    for i in range(count):
        left_mean += x[i, 0]
        right_mean += x[i, 1]
    left_mean /= count
    right_mean /= count
    left_power = 0.0
    right_power = 0.0
    cross = 0.0
    for i in range(count):
        left = x[i, 0]-left_mean
        right = x[i, 1]-right_mean
        left_power += left*left
        right_power += right*right
        cross += left*right
    return left_power, right_power, cross


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


def decode_pulse_stream(model, x, diagnostics=None, latest_only=False,
                        input_gain=1.0, models=None, model_factory=None,
                        force_float32=False, state=None, pulse_starts=None,
                        sample_rate=RATE, pilot_timing='baseline',
                        pilot_speed_diagnostics=False,
                        pulse_timing='baseline', frame_boundary='baseline',
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
    ``tone-replaced``. ``frame_boundary='eof'`` selects EOF packet boundaries;
    the default waits for the next header for legacy streams.
    ``pulse_timing='pulse-warp'`` optionally uses the neighboring pulse fits
    as local-slope anchors for a monotone, within-packet Hermite sample map.
    ``frame_boundary='eof'`` requires and uses the packet's final 32-sample
    marker instead of waiting for the next header.
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
    if frame_boundary not in ('baseline', 'eof'):
        raise ValueError(f'unknown frame boundary mode {frame_boundary!r}')
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
    samples = samples*gain
    if state is None:
        state = PulseState()
    results, info = _decode_pulse_samples(model, samples, diagnostics,
                                          latest_only, models, model_factory,
                                          force_float32, state, pulse_starts,
                                          sample_rate, pilot_timing,
                                          pilot_speed_diagnostics,
                                          pulse_timing, frame_boundary,
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
        pilot_timing, pilot_speed_diagnostics, pulse_timing, frame_boundary,
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


EOF_EDGE_HYSTERESIS = .04    # Schmitt band around zero for marker edges


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
            times[edges] = left + mono[left]/(mono[left]-mono[left+1])
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


def _measure_eof_marker(samples, frame_start, start_scale):
    """Find a packet's EOF mark with a Schmitt trigger and a run counter.

    The mark is a known four-run pattern at preamble level, so like the
    header it is counted, not fitted: see _eof_marker_kernel. The search is
    gated around the packet-clock prediction. Returns the marker's measured
    start, end and scale and the packet scale they imply, or None.
    _measure_eof_marker_fit is the earlier fitted implementation, kept as the
    reference in tests.
    """
    mono = np.asarray(samples)
    if mono.ndim == 2:
        mono = mono.mean(axis=1)
    if mono.ndim != 1 or start_scale <= 0 or mono.dtype not in (
            np.float32, np.float64):
        return None
    start_scale = float(start_scale)
    expected_start = float(frame_start)+EOF_MARKER_OFFSET*start_scale
    radius = max(12*start_scale, EOF_SEARCH_FRACTION*PULSE_FRAME*start_scale)
    lo = max(0, int(np.floor(expected_start-radius)))
    hi = min(len(mono), int(np.ceil(expected_start+
                                    EOF_MARKER_LENGTH*start_scale+radius)))
    if hi-lo < EOF_MARKER_LENGTH*start_scale:
        return None
    out = np.zeros(9)
    _eof_marker_kernel(mono, lo, hi, expected_start, radius, start_scale,
                       EOF_EDGE_HYSTERESIS, EOF_MARKER_MIN_LEVEL,
                       EOF_MARKER_EDGES, EOF_MARKER_RUNS_ARRAY, out)
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


def _measure_eof_marker_fit(samples, frame_start, start_scale):
    """Validate the three known transitions in a packet's 32-sample EOF mark.

    The marker is a distinct four-run (+, -, +, -) sequence at the same level
    as the acquisition preamble. Search is gated around the packet-clock
    prediction; acceptance then comes from the transition spacing, polarity,
    plateau levels, and a linear fit to the three zero crossings.
    """
    mono = np.asarray(samples, dtype=float)
    if mono.ndim == 2:
        mono = mono.mean(axis=1)
    if mono.ndim != 1 or start_scale <= 0:
        return None

    expected_start = float(frame_start)+EOF_MARKER_OFFSET*float(start_scale)
    radius = max(12*float(start_scale),
                EOF_SEARCH_FRACTION*PULSE_FRAME*float(start_scale))
    lo = max(0, int(np.floor(expected_start-radius)))
    hi = min(len(mono), int(np.ceil(expected_start+
                                    EOF_MARKER_LENGTH*start_scale+radius)))
    if hi-lo < EOF_MARKER_LENGTH*start_scale:
        return None
    window = mono[lo:hi]

    def interpolate(positions):
        positions = np.asarray(positions, dtype=float)
        clipped = np.clip(positions, 0, len(mono)-1)
        left = np.clip(np.floor(clipped).astype(np.intp), 0, len(mono)-2)
        fraction = clipped-left
        return mono[left]+fraction*(mono[left+1]-mono[left])

    crossings = np.flatnonzero(np.diff(np.signbit(window)))
    if len(crossings) < 3:
        return None
    left = crossings
    crossing_values = window[left]
    edge_positions = (lo+left+crossing_values /
                      (crossing_values-window[left+1]))

    edge_coordinates = EOF_MARKER_EDGES
    center_coordinates = EOF_MARKER_CENTERS
    level_signs = np.asarray(EOF_MARKER_LEVELS, dtype=float)
    fit_matrix = np.column_stack((np.ones(3), edge_coordinates))
    nominal_gaps = np.diff(edge_coordinates)
    candidates = []
    for index in range(len(edge_positions)-2):
        observed = edge_positions[index:index+3]
        gaps = np.diff(observed)
        marker_scale = float(np.mean(gaps/nominal_gaps))
        if (not np.isfinite(marker_scale) or marker_scale <= 0 or
                not .7*start_scale <= marker_scale <= 1.3*start_scale):
            continue
        gap_error = float(np.max(np.abs(gaps-nominal_gaps*marker_scale)))
        if gap_error > max(1.5, .40*np.min(nominal_gaps)*marker_scale):
            continue
        offset, fitted_scale = np.linalg.lstsq(
            fit_matrix, observed, rcond=None)[0]
        fitted_scale = float(fitted_scale)
        if (not np.isfinite(offset+fitted_scale) or fitted_scale <= 0 or
                abs(fitted_scale/marker_scale-1) > .08):
            continue
        fit = offset+edge_coordinates*fitted_scale
        residual = float(np.sqrt(np.mean(np.square(observed-fit))))
        residual_limit = max(1.5, .45*fitted_scale)
        confidence = float(np.clip(1-residual/residual_limit, 0, 1))
        marker_start = float(offset)
        if abs(marker_start-expected_start) > radius:
            continue
        centers = marker_start+center_coordinates[:3]*fitted_scale
        levels = interpolate(centers)
        edge_step = max(.75, .75*fitted_scale)
        before_edges = interpolate(observed-edge_step)
        after_edges = interpolate(observed+edge_step)
        marker_polarity = None
        signed_levels = None
        for polarity in (1, -1):
            trial_centers = polarity*level_signs[:3]*levels
            trial_before = polarity*level_signs[:3]*before_edges
            trial_after = polarity*level_signs[1:]*after_edges
            if (float(np.min(trial_centers)) >= EOF_MARKER_MIN_LEVEL and
                    float(np.min(trial_before)) >= .04 and
                    float(np.min(trial_after[:2])) >= .04 and
                    float(trial_after[2]) >= .01):
                marker_polarity = polarity
                signed_levels = np.concatenate((trial_centers,
                                                trial_before, trial_after))
                break
        if marker_polarity is None:
            continue
        if confidence < .45:
            continue
        marker_end = marker_start+EOF_MARKER_LENGTH*fitted_scale
        packet_scale = (marker_end-float(frame_start))/PULSE_FRAME
        candidates.append({
            'start': marker_start,
            'end': marker_end,
            'scale': fitted_scale,
            'packet_scale': packet_scale,
            'confidence': confidence,
            'edge_residual': residual,
            'gap_error': gap_error,
            'min_level': float(np.min(signed_levels)),
            'polarity': marker_polarity,
            'prediction_error': float(marker_start-expected_start),
        })
    if not candidates:
        return None
    return min(candidates, key=lambda item: (
        abs(item['prediction_error']),
        item['edge_residual']+item['gap_error']*.25))


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
                           pulse_timing='baseline', frame_boundary='baseline',
                           tone_equalization='off'):
    sample_rate = float(sample_rate)
    cursor = 0
    counter = 1
    results = []
    measured = None
    preloaded_following = None
    pending_aspect = 0
    eof_markers_validated = 0
    selected_marker = None
    min_scale, max_scale = pulse_sample_scale_bounds(sample_rate)
    # EOF-mode acquisition revisits one packet at a time. Cache this mono
    # view so marker checks and local pulse reacquisition do not repeatedly
    # average the entire capture for every frame.
    mono_samples = _mono(samples) if frame_boundary == 'eof' else None
    if latest_only:
        # The live rolling buffer can contain the previous frame plus the new
        # one. Reuse the input layer's incremental header hits when available;
        # otherwise scan the window as before. In either case only the newest
        # complete frame needs the expensive image decode.
        cached = (_remeasure_pulse_starts(samples, pulse_starts, sample_rate,
                                          mono=mono_samples)
                  if pulse_starts is not None else [])
        required_starts = 1 if frame_boundary == 'eof' else 2
        cache_valid = len(cached) >= required_starts
        if cache_valid and frame_boundary == 'baseline':
            prior, current = cached[-2], cached[-1]
            interval = (current[0]-prior[0])/prior[1]
            cache_valid = (
                abs(interval-PULSE_FRAME) <= max(12, .03*PULSE_FRAME) and
                abs(current[1]/prior[1]-1) <= .03)
        starts = (cached if cache_valid else
                  pulse_frame_starts(samples, sample_rate))
        candidates = [(fs, sc, conf, 0) for fs, sc, conf in starts]
        if len(candidates) < required_starts:
            return [], {'frames': 0, 'pulse_frames': 0, 'recovered': False}
        if frame_boundary == 'eof':
            # LiveInput wakes on a newly arrived header. At that point the
            # newest packet may not have reached its EOF yet, while the prior
            # packet's marker is complete. Choose the newest candidate whose
            # marker validates; a finite one-packet capture still works.
            selected = None
            frame_min, frame_max = pulse_sample_scale_bounds(sample_rate)
            for candidate in reversed(candidates):
                candidate_start, candidate_scale, _, _ = candidate
                marker = _measure_eof_marker(
                    mono_samples, candidate_start, candidate_scale)
                if (marker is not None and
                        frame_min*.98 <= marker['packet_scale'] <= frame_max*1.02):
                    selected = candidate
                    # Keep the validated marker: the walk below commits this
                    # same packet and must not count its EOF a second time.
                    selected_marker = marker
                    break
            if selected is None:
                return [], {'frames': 0, 'pulse_frames': 0, 'recovered': False}
            fs, sc, conf, pending_aspect = selected
        else:
            # The second pulse is the first edge of the next header. It is
            # enough to validate duration; the next body need not exist.
            fs, sc, conf, pending_aspect = candidates[-2]
            preloaded_following = candidates[-1][:3]
        cursor = int(fs)
        measured = (16*sc, sc, conf)
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
        following = None
        next_start = None
        aspect_code = pending_aspect
        following_valid = False
        frame_scale = scale
        frame_length = PULSE_FRAME
        following_scale = None
        following_confidence = None
        boundary_diag = None
        if frame_boundary == 'eof':
            if selected_marker is not None:
                marker = dict(selected_marker)
                marker['packet_scale'] = ((marker['end']-frame_start) /
                                          PULSE_FRAME)
                selected_marker = None
            else:
                marker = _measure_eof_marker(
                    mono_samples, frame_start, scale)
            if marker is not None:
                frame_min, frame_max = pulse_sample_scale_bounds(sample_rate)
                candidate_scale = marker['packet_scale']
                if (frame_min*.98 <= candidate_scale <= frame_max*1.02):
                    following_valid = True
                    next_start = marker['end']
                    frame_scale = candidate_scale
                    following_scale = marker['scale']
                    following_confidence = marker['confidence']
                    eof_markers_validated += 1
                    boundary_diag = marker
        else:
            search = int(frame_start+PULSE_FRAME*scale)
            if preloaded_following is not None:
                following_start, following_scale, following_confidence = \
                    preloaded_following
                candidate = (following_start-search+16*following_scale,
                             following_scale, following_confidence)
                preloaded_following = None
            else:
                candidate = PULSE.measure_pulses(
                    _mono(samples[search:]), min_scale=min_scale,
                    max_scale=max_scale)
            if candidate is not None:
                candidate_start = search + candidate[0] - 16*candidate[1]
                interval = (candidate_start-frame_start)/scale
                current_match = (abs(interval-PULSE_FRAME) <=
                                 max(12, .03*PULSE_FRAME))
                if (candidate[2] >= .45 and
                        abs(candidate[1]/scale-1) <= .03 and current_match):
                    following = candidate
                    next_start = candidate_start
                    following_scale = candidate[1]
                    following_confidence = candidate[2]
            if following is not None:
                following_valid = True
                frame_scale = (next_start-frame_start)/frame_length
        if not following_valid:
            # A frame is committed only after its selected endpoint witness.
            break
        start = frame_start + PULSE.SYNC_LEN*scale
        # Consecutive pulse positions provide the packet-average scale. The
        # optional warp bends that straight-line sample map using the local
        # scales fitted at both neighboring pulse words.
        pulse_map = None
        pulse_timing_diag = None
        metadata_indexes = None
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
                    endpoint_position=(next_start if frame_boundary == 'eof'
                                       else None),
                    endpoint_coordinate=(PULSE_FRAME
                                         if frame_boundary == 'eof' else None))
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
            if frame_boundary == 'eof':
                # The header origin and measured EOF endpoint define one
                # packet-wide affine time map. Use its scale for both the
                # body origin and every metadata sample; mixing the local
                # preamble scale into either offset breaks that shared clock.
                indexes = (frame_start +
                           (PULSE.SYNC_LEN+np.arange(FRAME))*frame_scale)
                metadata_indexes = (
                    frame_start+(PULSE.SYNC_LEN+FRAME+
                                 np.arange(META_SYMBOL))*frame_scale)
            else:
                indexes = start + np.arange(FRAME)*frame_scale
                metadata_indexes = None
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
            if following_valid and frame_boundary == 'baseline':
                measured = (16*following_scale, following_scale,
                            following_confidence)
            elif following_valid and frame_boundary == 'eof':
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
        metadata_scale = (frame_scale if frame_boundary == 'eof' else scale)
        meta_start = (frame_start + (PULSE.SYNC_LEN+FRAME)*metadata_scale)
        meta = decode_metadata(model, samples, meta_start, metadata_scale, None,
                               force_float32,
                               sample_indexes=metadata_indexes)
        # Slices 0-4 carry a plain CRC; 5 and 6 carry it XOR p / N, which
        # the lock learns and then checks; until then they are accepted only
        # provisionally (see PulseState.accept).
        metadata_valid, provisional = state.accept(meta)
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
                    retry_meta = decode_metadata(
                        model, samples, corrected_start, retry_scale, None,
                        force_float32)
                else:
                    corrected_map = _pulse_warp_map(
                        frame_start, next_start, scale, following_scale,
                        confidence, following_confidence,
                        PULSE.SYNC_LEN+FRAME+
                        np.arange(META_SYMBOL)-shift,
                        endpoint_position=(next_start
                                           if frame_boundary == 'eof' else None),
                        endpoint_coordinate=(PULSE_FRAME
                                             if frame_boundary == 'eof'
                                             else None))
                    corrected_indexes = (None if corrected_map is None else
                                         corrected_map[0])
                    retry_meta = (decode_metadata(
                        model, samples, retry_start, retry_scale, None,
                        force_float32, sample_indexes=corrected_indexes)
                        if corrected_indexes is not None else None)
                retry_valid, retry_provisional = state.accept(retry_meta)
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
            aspect_code = meta.aspect_code
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
                                  tone_equalization=tone_equalization)
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
                     LIVE_VALID_HEAD_CONFIDENCE or
                     result.diag.get('head_coverage', 0) <
                     LIVE_VALID_HEAD_COVERAGE or
                     max(result.diag.get('noise', [np.inf])) >
                     LIVE_MAX_PILOT_NOISE)):
                result.status = 'lost'
            result.diag['displayable'] = bool(
                result.diag.get('displayable', False) and
                max(result.diag.get('noise', [np.inf])) <=
                DISPLAY_MAX_PILOT_NOISE)
            result.diag['pulse_confidence'] = float(confidence)
            result.diag['aspect_code'] = aspect_code
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
            if pulse_timing_diag is not None:
                result.diag['pulse_timing'] = pulse_timing_diag
            if boundary_diag is not None:
                result.diag['eof_marker'] = boundary_diag
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
            if result.status != 'lost' and tail_slice is not None:
                state.tail.update(selected_model, result.coeffs, tail_slice)
            if latest_only:
                break
        pending_aspect = aspect_code
        frame_length = PULSE_FRAME
        cursor = int(next_start if following_valid
                     else frame_start + frame_length*scale)
        if following_valid and frame_boundary == 'baseline':
            measured = (16*following_scale, following_scale,
                        following_confidence)
        elif following_valid and frame_boundary == 'eof':
            measured = _measure_pulse_after_eof_marker(
                mono_samples, cursor, following_scale, min_scale, max_scale)
        else:
            measured = None
        counter += 1
    info = {'frames': len(results), 'pulse_frames': len(results),
            'recovered': bool(results), 'frame_boundary': frame_boundary,
            'eof_markers_validated': eof_markers_validated}
    if diagnostics is not None:
        info['diagnostics'] = _diagnostic_summary(diagnostics, 0.0)
    return results, info


def values_from(model, coeffs):
    return model.coder.inverse(coeffs*model.coder.gains)
