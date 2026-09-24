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
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
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
PULSE_MIN_SCALE = .25
PULSE_MAX_SCALE = 2.0
ENCODING_FILTERS = ('nearest', 'box', 'lanczos', 'bicubic')
ENCODING_FILTER_CODES = {name: code for code, name in
                         enumerate(ENCODING_FILTERS)}


def prepare_image(image, encode_filter='lanczos'):
    """Prepare an RGB source without depending on the application imaging API."""
    resampling = getattr(Image.Resampling, encode_filter.upper())
    return image.convert('RGB').resize((80, 96), resampling)


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
# Stereo slot order (§8.3).  An S slot on carrier b is ordered as if it sat on
# carrier b+S_ORDER_OFFSET, so mono playback (which loses every S slot) drops
# less important ranks while a low-pass still removes high carriers last.
# Offset 6 (2.25 kHz) keeps most of the mono gain at almost no low-pass cost.
S_ORDER_OFFSET = 6
STEREO_SLOTS = sorted(
    [(b, c, q) for b in STEREO for c in 'MS' for q in 'IQ'],
    key=lambda slot: (slot[0][0] + (S_ORDER_OFFSET if slot[1] == 'S' else 0),
                      slot[1], slot[0][1], slot[2]))
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
MODEL_TABLES_SHA256 = '0bef7da37a17f9e6f9ab6d0e0015b2e53776912da7dd2bad8fe0e2f5ce196331'
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
    np.add.at(X,
              (GROUP_SYMBOLS.ravel(),
               np.repeat(GROUP_BINS, 8),
               np.repeat(GROUP_STREAMS, 8)),
              (tx*GROUP_QMULT[:, None]).ravel())
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
    """Fastest playback speed a `rate` Hz output can carry.

    Speeding the wire up multiplies every frequency, so the emission edge
    (14 kHz at 1x) must stay below the output's Nyquist frequency: 1.71x at
    48 kHz, 3.43x at 96 kHz.  Faster than this the speed conversion filters
    the top carriers away (measured: 2x from a 48 kHz output passes 4 of 15
    frames; 3.5x from 96 kHz still passes all).
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
                       loop=None, direction=1):
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
    return shaped


def encode_pulse_stream(model, values, start_counter=1, aspect_codes=None,
                        source_indices=None, loop=None, directions=None):
    values = list(values) if np.asarray(values).ndim != 1 else [values]
    codes = aspect_codes or [0]*len(values)
    indexes = (list(source_indices) if source_indices is not None else
               [start_counter+i-1 for i in range(len(values))])
    ways = list(directions) if directions is not None else [1]*len(values)
    if len(codes) != len(values) or len(indexes) != len(values) or len(ways) != len(values):
        raise ValueError('aspect_codes, source_indices and directions must match values')
    return np.concatenate([
        encode_pulse_frame(model, value, start_counter+i, code, source_index,
                           loop, way)
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
for _table in (_PILOT_BIN_FREQUENCIES, _PILOT_BIN_SYMBOLS, _PILOT_BIN_VALID,
               _PILOT_BIN_VALUES,
               _PILOT_BIN_LS, _PILOT_BIN_OMEGA, _PILOT_BIN_J,
               _PILOT_BIN_VALUES32, _PILOT_BIN_LS32,
               _PILOT_BIN_OMEGA32, _PILOT_BIN_J32):
    _table.setflags(write=False)


def channel_joint(Z, iters=2, force_float32=False):
    """Per rx channel: static 1x2 response per pilot bin x smooth timing track.

    y(s,b) = exp(j*2*pi*b*delta(s)/N) * (hM(b)*pM + hS(b)*pS), delta piecewise
    linear over the frame. Alternating LS: responses given delta, then a
    Gauss-Newton phase step for delta. Pilot-bin observations and their static
    least-squares operators are batched, avoiding one Python solve per bin.
    Timing error left by the clock map is common to all carriers of a symbol,
    so the pilots pin it down (§9.3).
    """
    if force_float32:
        return _channel_joint_float32(Z, iters)
    return _channel_joint_batched(Z, iters, False)


def _channel_joint_batched(Z, iters, force_float32):
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
        theta = np.zeros(len(KNOTS), dtype=real_dtype)
        for _ in range(iters):
            delta = basis @ theta
            rot = np.exp(omega*delta[_PILOT_BIN_SYMBOLS]).astype(
                complex_dtype, copy=False)
            # The pilot phase rotation has unit magnitude, so it changes only
            # the right-hand side; each bin's Gram matrix is static and its
            # inverse projection can be precomputed.
            h = np.einsum('bkn,bn->bk', operator, np.conj(rot)*y)
            pred = rot*np.einsum('bni,bi->bn', values, h)
            ok = _PILOT_BIN_VALID & (np.abs(pred) > threshold)
            rho = np.divide(y, pred, out=np.zeros_like(y), where=ok)
            ph = np.angle(rho).astype(real_dtype, copy=False)
            w = np.where(ok, np.abs(pred), 0).astype(real_dtype, copy=False)

            # This is the same weighted Gauss-Newton fit as the observation-
            # ordered loop: both sides of the normal equation carry w**2.
            JW = jacobian*w[..., None]
            normal = np.einsum('bni,bnj->ij', JW, JW)
            normal += eps*np.eye(JW.shape[-1], dtype=real_dtype)
            step = np.linalg.solve(
                normal, np.einsum('bni,bn->i', JW, ph*w))
            theta += step
            if np.max(np.abs(step)) < step_tolerance:
                break

        delta = basis @ theta
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
_SYMBOL_INDEX = np.arange(F)[:, None]


def fade_and_noise(Z, H, force_float32=False):
    """Per-symbol magnitude/phase refit (§9.4) and pilot-residual noise (§9.5).

    Batched over all symbols and both channels; same rules as the original
    per-symbol loop (equivalence is covered by modem_tests/test_v7_decode_speed).
    """
    if force_float32:
        return _fade_and_noise_float32(Z, H)
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
    return H, noise


def _fade_and_noise_float32(Z, H):
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
    return H, noise


def decode_frame(model, x, tmap, counter, prev_tail, cancel=True,
                 diagnostics=None, direct_body=None, force_float32=False):
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
        Z = (np.fft.rfft(windows, axis=1)/model.scale *
             np.conj(model.phase)[:, :, None] * EARLY[None, :, None])
    if diagnostics is not None:
        diagnostics.setdefault('stage_ms', {}).setdefault('sample_fft', []).append(
            (perf_counter()-stage_started)*1000)
    stage_started = perf_counter()
    H = channel_joint(Z, force_float32=force_float32)
    H, noise = fade_and_noise(Z, H, force_float32=force_float32)
    if diagnostics is not None:
        diagnostics.setdefault('stage_ms', {}).setdefault('channel', []).append(
            (perf_counter()-stage_started)*1000)
    stage_started = perf_counter()
    DEBUG['Z'], DEBUG['H'], DEBUG['noise'] = Z, H, noise
    # Per-cell 2x2 MMSE (§9.6 step 1) with priors from group powers.  Keep all
    # blocks and both quadratures in one batch: the small solve is cheap, but
    # entering Python once per block/channel was not.
    idx = model.rank_tables[counter % TAIL_PHASES]
    priors = model.block_prior_tables
    if force_float32:
        block_prior = (model.block_prior_tables32[counter % TAIL_PHASES]
                       if model.block_prior_tables32 else
                       block_priors(model.gain32, model.lam32, idx).astype(
                           np.float32))
    else:
        block_prior = (priors[counter % TAIL_PHASES] if priors
                       else block_priors(model.gain, model.lam, idx))
    Hc = H[BLOCK_SYMBOLS, BLOCK_BINS[:, None]]
    Zc = Z[BLOCK_SYMBOLS, BLOCK_BINS[:, None]]
    estimates = np.zeros((len(BLOCKS), 2, 2, 8),
                         np.complex64 if force_float32 else complex)
    variances = np.full((len(BLOCKS), 2, 2, 8), np.inf,
                        dtype=np.float32 if force_float32 else float)
    for q in range(2):
        prior = block_prior[:, :, q]
        safe = prior + (np.float32(1e-12) if force_float32 else 1e-12)
        S = (Hc*safe[:, None, None, :]) @ Hc.conj().transpose(0, 1, 3, 2)
        S[..., 0, 0] += np.maximum(noise[BLOCK_SYMBOLS, 0],
                                   NOISE_FLOOR)
        S[..., 1, 1] += np.maximum(noise[BLOCK_SYMBOLS, 1],
                                   NOISE_FLOOR)
        # W = P H^H S^-1 (prior on the left, inverse on the right).  S is
        # Hermitian, so W^H = S^-1 (H P): one batched left solve, then the
        # conjugate transpose.  Solving S^-1 (P H^H) instead is only correct
        # when a cell's M and S priors are equal.
        HP = Hc*safe[:, None, None, :]
        W = _solve_2x2_mat(S, HP).conj().transpose(0, 1, 3, 2)
        xt = np.einsum('btij,btj->bti', W, Zc)
        B = W @ Hc
        cov = W @ S @ W.conj().transpose(0, 1, 3, 2)
        beta = np.diagonal(B, axis1=-2, axis2=-1).real
        var = np.maximum(np.diagonal(cov, axis1=-2, axis2=-1).real -
                         beta**2*prior[:, None, :], 1e-12)
        beta_k = beta.transpose(0, 2, 1)
        estimates[:, :, q] = np.divide(
            xt.transpose(0, 2, 1), beta_k,
            out=np.zeros_like(xt.transpose(0, 2, 1)),
            where=beta_k > 1e-6)
        variances[:, :, q] = np.divide(
            var.transpose(0, 2, 1), beta_k**2,
            out=np.full_like(var.transpose(0, 2, 1), np.inf),
            where=beta_k > 1e-6)
    # Group LMMSE (§9.6 step 2) + gate (step 3).  Assemble all groups and use
    # one batched solve instead of 290 Python-level 8x8 SVD/solve calls.
    group_count = len(GROUPS)
    ranks_all = idx
    live_all = ranks_all >= 0
    real_dtype = np.float32 if force_float32 else float
    y_all = np.empty((group_count, 8), dtype=real_dtype)
    sig_all = np.empty((group_count, 8), dtype=real_dtype)
    values_all = estimates[GROUP_BLOCK, GROUP_STREAM_INDEX, GROUP_Q_INDEX]
    vars_all = variances[GROUP_BLOCK, GROUP_STREAM_INDEX, GROUP_Q_INDEX]
    y_all[:] = values_all.real
    y_all[GROUP_Q_INDEX == 1] = values_all[GROUP_Q_INDEX == 1].imag
    sig_all[:] = np.where(np.isfinite(vars_all), vars_all/2, 1e9)
    lam_model = model.lam32 if force_float32 else model.lam
    gain_model = model.gain32 if force_float32 else model.gain
    lam_all = np.where(live_all, lam_model[np.maximum(ranks_all, 0)],
                       np.array(1e-12, dtype=real_dtype))
    gain_all = np.where(live_all, gain_model[np.maximum(ranks_all, 0)],
                        np.array(0, dtype=real_dtype))
    A_all = H8.astype(np.float32 if force_float32 else float)[None, :, :] * \
        gain_all[:, None, :]
    S_all = (A_all*lam_all[:, None, :]) @ A_all.transpose(0, 2, 1)
    S_all[:, np.arange(8), np.arange(8)] += sig_all
    M_all = lam_all[:, :, None]*A_all.transpose(0, 2, 1)
    K_all = np.linalg.solve(S_all.transpose(0, 2, 1),
                            M_all.transpose(0, 2, 1)).transpose(0, 2, 1)
    x_all = np.einsum('gij,gj->gi', K_all, y_all)
    post_all = lam_all - np.einsum(
        'gij,gji->gi', K_all,
        A_all*lam_all[:, None, :])
    conf_all = np.clip(1-post_all/lam_all, 0, 1)
    mu = model.mu32 if force_float32 else model.mu
    xhat = np.zeros_like(mu, dtype=real_dtype)
    conf = np.zeros_like(mu, dtype=real_dtype)
    got = np.zeros_like(model.mu, bool)
    # Every live coefficient occurs in exactly one group.  Flattening the
    # boolean selection turns the final group-order -> coefficient-order copy
    # into three indexed scatters, avoiding one Python loop per frame.
    live_ranks = ranks_all[live_all]
    xhat[live_ranks] = x_all[live_all]
    conf[live_ranks] = conf_all[live_all]
    got[live_ranks] = True
    if diagnostics is not None:
        diagnostics.setdefault('stage_ms', {}).setdefault('equalize', []).append(
            (perf_counter()-stage_started)*1000)
    floor = np.where(model.head, np.where(model.plane == 0, .05, .15),
                     np.where(model.plane == 0, .45, .60))
    if force_float32:
        floor = floor.astype(np.float32)
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
        return Result(counter, 'lost', display_coeffs if displayable else
                      np.asarray(prev_tail, dtype=real_dtype).copy(), {
            'noise': noise.mean(0).tolist(), 'got': int(got.sum()),
            'head_confidence': head_confidence,
            'head_coverage': head_coverage, 'held': not displayable,
            'displayable': displayable,
            'display_coeffs': display_coeffs})
    coeffs[got] = current[got]
    return Result(counter, 'verified', coeffs,
                  {'noise': noise.mean(0).tolist(), 'got': int(got.sum()),
                   'head_confidence': head_confidence,
                   'head_coverage': head_coverage, '_H': H,
                   'displayable': True})


def decode_metadata(model, samples, start, scale, channel,
                    force_float32=False):
    indexes = start + np.arange(META_SYMBOL)*scale
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


def leg_polarity(samples, previous=1, threshold=POLARITY_THRESHOLD):
    """Right-leg polarity (+1 or -1) of a stereo capture, with hysteresis.

    The V7 wire is M-dominated (preamble, clock, metadata and head blocks are
    M only), so a correctly wired pair correlates strongly positive: +0.73 to
    +0.85 wideband over one pulse frame across the whole torture matrix.  An
    inverted leg gives the mirror image.  Between -threshold and +threshold
    (silence, one dead leg, non-V7 audio) the previous decision is kept.
    Mono or single-channel input is always +1.
    """
    x = np.asarray(samples, np.float64)
    if x.ndim != 2 or x.shape[1] != 2 or len(x) < 2:
        return 1
    left = x[:, 0] - x[:, 0].mean()                  # DC/hum offset must not vote
    right = x[:, 1] - x[:, 1].mean()
    energy = float(np.dot(left, left)*np.dot(right, right))
    if energy <= 1e-24:
        return previous
    correlation = float(np.dot(left, right))/np.sqrt(energy)
    if correlation <= -threshold:
        return -1
    if correlation >= threshold:
        return 1
    return previous


def decode_pulse_stream(model, x, diagnostics=None, latest_only=False,
                        input_gain=1.0, models=None, model_factory=None,
                        force_float32=False, state=None, pulse_starts=None):
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
    """
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
                                          force_float32, state, pulse_starts)
    if results or samples.shape[1] != 2:
        return results, info
    # One leg polarity-inverted (miswired deck or cable, reversed head lead):
    # the preamble, clock and metadata are M-only, so the L+R acquisition sum
    # cancels and nothing is found.  The 2x2 pilot equalizer itself would cope,
    # so retry once with the right leg inverted.  Normal streams never reach
    # this; silence costs one more (cheap, empty) edge scan.
    flipped, flipped_info = _decode_pulse_samples(
        model, samples*np.float32([1, -1]), diagnostics, latest_only, models,
        model_factory, force_float32, state, pulse_starts)
    if not flipped:
        return results, info
    for result in flipped:
        result.diag['polarity_inverted'] = True
    flipped_info['polarity_inverted'] = True
    return flipped, flipped_info


def pulse_frame_starts(samples):
    """Every accepted pulse header in a stereo (or (n, 1)) capture.

    Returns ``(frame_start, scale, confidence)`` per header, in order, using
    the same edge-counted acquisition as the decoder: confidence below .45 is
    skipped, and after a hit the scan resumes just before the next expected
    header.  A header needs SYNC_LEN + META_SYMBOL + 32 samples after its
    scan point to be found, so callers scanning a live stream should overlap
    successive scans by that much.
    """
    starts = []
    scan = 0
    while scan + PULSE.SYNC_LEN + META_SYMBOL + 32 < len(samples):
        hit = PULSE.measure_pulses(samples[scan:].mean(axis=1),
                                   min_scale=PULSE_MIN_SCALE,
                                   max_scale=PULSE_MAX_SCALE)
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


def _remeasure_pulse_starts(samples, anchors):
    """Recheck upstream header anchors in short windows on decoder samples."""
    mono = samples.mean(axis=1)
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
            mono[scan:end], min_scale=PULSE_MIN_SCALE,
            max_scale=PULSE_MAX_SCALE)
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


def _decode_pulse_samples(model, samples, diagnostics, latest_only, models,
                           model_factory, force_float32, state,
                           pulse_starts=None):
    cursor = 0
    counter = 1
    results = []
    measured = None
    preloaded_following = None
    pending_aspect = 0
    if latest_only:
        # The live rolling buffer can contain the previous frame plus the new
        # one. Reuse the input layer's incremental header hits when available;
        # otherwise scan the window as before. In either case only the newest
        # complete frame needs the expensive image decode.
        cached = (_remeasure_pulse_starts(samples, pulse_starts)
                  if pulse_starts is not None else [])
        cache_valid = len(cached) >= 2
        if cache_valid:
            prior, current = cached[-2], cached[-1]
            interval = (current[0]-prior[0])/prior[1]
            cache_valid = (
                abs(interval-PULSE_FRAME) <= max(12, .03*PULSE_FRAME) and
                abs(current[1]/prior[1]-1) <= .03)
        starts = cached if cache_valid else pulse_frame_starts(samples)
        candidates = [(fs, sc, conf, 0) for fs, sc, conf in starts]
        if len(candidates) < 2:
            return [], {'frames': 0, 'pulse_frames': 0, 'recovered': False}
        # The second pulse is the first edge of the next header.  It is enough
        # to validate the current frame duration; the next body need not exist.
        fs, sc, conf, pending_aspect = candidates[-2]
        preloaded_following = candidates[-1][:3]
        cursor = int(fs)
        measured = (16*sc, sc, conf)
    while cursor + PULSE.SYNC_LEN + META_SYMBOL + 32 < len(samples):
        if measured is None:
            measured = PULSE.measure_pulses(samples[cursor:].mean(axis=1),
                                         min_scale=PULSE_MIN_SCALE,
                                         max_scale=PULSE_MAX_SCALE)
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
        search = int(frame_start+PULSE_FRAME*scale)
        if preloaded_following is not None:
            following_start, following_scale, following_confidence = \
                preloaded_following
            candidate = (following_start-search+16*following_scale,
                         following_scale, following_confidence)
            preloaded_following = None
        else:
            candidate = PULSE.measure_pulses(
                samples[search:].mean(axis=1), min_scale=PULSE_MIN_SCALE,
                max_scale=PULSE_MAX_SCALE)
        if candidate is not None:
            candidate_start = search + candidate[0] - 16*candidate[1]
            interval = (candidate_start-frame_start)/scale
            current_match = abs(interval-PULSE_FRAME) <= max(12, .03*PULSE_FRAME)
            if (candidate[2] >= .45 and abs(candidate[1]/scale-1) <= .03 and
                    current_match):
                following = candidate
                next_start = candidate_start
        aspect_code = pending_aspect
        following_valid = False
        frame_scale = scale
        frame_length = PULSE_FRAME
        if following is not None:
            following_valid = True
            frame_scale = (next_start-frame_start)/frame_length
        if not following_valid:
            # The next header is the commit boundary.  Do not decode on a
            # coincidental edge inside the current body/metadata.
            break
        start = frame_start + PULSE.SYNC_LEN*scale
        # The first pulse measures the local playback scale at frame start;
        # consecutive pulse positions measure the actual frame duration. Use
        # the latter for the body walk so smooth wow/flutter is corrected
        # across the payload instead of only at its first sample.
        indexes = start + np.arange(FRAME)*frame_scale
        if indexes[-1] >= len(samples)-1:
            break
        if confidence < .45:
            results.append(Result(counter, 'lost', state.tail.prior(model), {
                'pulse_confidence': float(confidence), 'held': True}))
            pending_aspect = aspect_code
            frame_length = PULSE_FRAME
            cursor = int(next_start if following_valid
                         else frame_start + frame_length*scale)
            measured = ((16*following[1], following[1], following[2])
                        if following_valid else None)
            counter += 1
            continue
        # Metadata is deliberately decoded before the image body.  Its pilots
        # are self-referencing, so the bootstrap model's absolute scale cancels
        # out; the protected encoding ID can therefore select the source model.
        meta_start = frame_start + (PULSE.SYNC_LEN+FRAME)*scale
        meta = decode_metadata(model, samples, meta_start, scale, None,
                               force_float32)
        # Slices 0-4 carry a plain CRC; 5 and 6 carry it XOR p / N, which
        # the lock learns and then checks; until then they are accepted only
        # provisionally (see PulseState.accept).
        metadata_valid, provisional = state.accept(meta)
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
        body = _sample_at(samples, indexes, taps=4).astype(np.float32)
        if body.shape[1] == 1:
            # The demodulator is M/S two-channel internally.  A mono capture
            # is the shared M observation, so duplicate it without inventing S.
            body = np.repeat(body, 2, axis=1)
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
                                  force_float32=force_float32)
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
            result.diag['loop'] = state.lock.loop
            result.diag['pulse_scale'] = float(scale)
            result.diag['frame_scale'] = float(frame_scale)
            result.diag['playback_speed'] = float(1/max(scale, 1e-9))
            result.diag['timing_delta_ppm'] = float(
                (frame_scale/scale-1)*1e6)
            results.append(result)
            if result.status != 'lost' and tail_slice is not None:
                state.tail.update(selected_model, result.coeffs, tail_slice)
            if latest_only:
                break
        pending_aspect = aspect_code
        frame_length = PULSE_FRAME
        cursor = int(next_start if following_valid
                     else frame_start + frame_length*scale)
        measured = ((16*following[1], following[1], following[2])
                    if following_valid else None)
        counter += 1
    info = {'frames': len(results), 'pulse_frames': len(results),
            'recovered': bool(results)}
    if diagnostics is not None:
        info['diagnostics'] = _diagnostic_summary(diagnostics, 0.0)
    return results, info


def values_from(model, coeffs):
    return model.coder.inverse(coeffs*model.coder.gains)
