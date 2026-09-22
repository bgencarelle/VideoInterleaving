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


METADATA_OPTION_MONO_SUM = 2
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


def metadata_word(aspect_code, encoding_type=0, revision=0, source_index=0):
    """Pack picture identity and one-based source index into metadata.

    The public/source-facing value remains zero-based, matching the bake and
    display APIs. Wire zero is reserved so an erased field cannot masquerade
    as the first source frame.
    """
    if not 0 <= int(source_index) < 0xffff:
        raise ValueError('source_index must fit a zero-based 16-bit field')
    wire_index = int(source_index) + 1
    payload = bytes([((int(aspect_code) & 7) << 5) |
                     ((int(encoding_type) & 3) << 3) |
                     ((int(revision) & 3) << 1) | 1,
                     (wire_index >> 8) & 0xff,
                     wire_index & 0xff])
    return payload + crc16(payload).to_bytes(2, 'big')


def metadata_symbols(aspect_code, encoding_type=0, revision=0, source_index=0):
    bits = np.unpackbits(np.frombuffer(
        metadata_word(aspect_code, encoding_type, revision, source_index), np.uint8))
    return ((bits[0::2].astype(float)*2-1) +
            1j*(bits[1::2].astype(float)*2-1))/np.sqrt(2)


def parse_metadata_word(raw):
    """Validate and unpack the five-byte live metadata word."""
    raw = bytes(raw)
    if len(raw) != 5 or crc16(raw[:3]) != int.from_bytes(raw[3:5], 'big'):
        return None
    wire_index = int.from_bytes(raw[1:3], 'big')
    if wire_index == 0:
        return None
    return ((raw[0] >> 5) & 7, (raw[0] >> 3) & 3, (raw[0] >> 1) & 3,
            wire_index - 1)


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
def clock_cancel_template(counter):
    """Cache the deterministic three-word cancellation template."""
    clk = clock_wave((clock_word(counter-1), clock_word(counter),
                      clock_word(counter+1)))
    return np.asarray(clk[FRAME-64:2*FRAME+64], float)


# ------------------------------------------------------------------ §6.4, §8
def data_blocks():
    blocks = [(b, p) for b in range(5, 34) for p in range(3) if SCAT.get(b) != p]
    return sorted(blocks)                               # health order (§8.1)


BLOCKS = data_blocks()
MONO = [blk for blk in BLOCKS if blk[0] <= 9]
STEREO = [blk for blk in BLOCKS if blk[0] > 9]
GROUPS = ([(b, 'M', q) for b in MONO for q in 'IQ'] +
          [(b, c, q) for b in STEREO for c in 'MS' for q in 'IQ'])
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
    mono_sum: bool = False


# The canonical V7 profile is frozen data, not a runtime computation.  Its
# tables were derived once from REFERENCE_FIXTURE by derive_tables() and
# phase_table() (tools/v7_freeze_tables.py) and are loaded from MODEL_TABLES,
# whose SHA-256 is part of the wire definition.  Sender and receiver therefore
# agree without the fixture image, Pillow, SciPy's curve fit or NumPy's RNG
# stream -- any of which could drift between library versions and silently
# desynchronise the two ends (a different phase table decodes as garbage).
REFERENCE_FIXTURE = ROOT/'modem_tests/fixtures/v7_reference_face.png'
MODEL_TABLES = Path(__file__).with_name('v7_model_tables.npz')
MODEL_TABLES_SHA256 = '74b28a2ed165b5ba8e9e6d46c8f2a99127e53d03efa5a4d19fe12fa7115b1e01'
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


def _assemble(tables, phase, target_rms, encode_filter, mono_sum):
    """Model from (mu, lam, order, gain, unit_rms) tables and a phase table."""
    coder = SourceCoder(V7_SHAPES, grids=V7_GRIDS)
    order = np.asarray(tables['order'])
    head = np.zeros(len(tables['lam']), bool); head[order[:HEAD]] = True
    rank_tables = tuple(frame_ranks(order, p) for p in range(TAIL_PHASES))
    return Model(coder, np.asarray(tables['mu']), np.asarray(tables['lam']), order,
                 np.asarray(tables['gain']), phase,
                 target_rms/float(tables['unit_rms']), _planes(), head,
                 rank_tables, ENCODING_FILTER_CODES[encode_filter], mono_sum)


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
                            1.0, encode_filter, False)
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


def load_model(target_rms, encode_filter='lanczos', mono_sum=False):
    """The canonical V7 model from the frozen, hash-checked tables."""
    if encode_filter not in ENCODING_FILTER_CODES:
        raise ValueError(f'unknown encode filter {encode_filter!r}')
    t = _frozen_tables()
    tables = {k: t[f'{encode_filter}/{k}'] for k in ('mu', 'lam', 'order', 'gain', 'unit_rms')}
    return _assemble(tables, t['phase'], target_rms, encode_filter, mono_sum)


def _is_reference(fixture):
    return fixture is None or Path(fixture).resolve() == REFERENCE_FIXTURE.resolve()


def build_model(fixture, target_rms, encode_filter='lanczos', mono_sum=False):
    """Canonical frozen model for the reference fixture (or None); otherwise a
    custom profile derived from ``fixture``, which the receiver must match."""
    if _is_reference(fixture):
        return load_model(target_rms, encode_filter, mono_sum)
    with Image.open(fixture) as source:
        return build_model_from_image(source, target_rms, encode_filter,
                                       mono_sum)


def build_model_from_image(source, target_rms, encode_filter='lanczos',
                           mono_sum=False):
    """A custom (non-canonical) V7 model derived from an in-memory image."""
    phase = phase_table()
    return _assemble(derive_tables(source, encode_filter, phase), phase,
                     target_rms, encode_filter, mono_sum)


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


def encode_pulse_frame(model, values, counter, aspect_code=0, source_index=None):
    """One edge-counted pulse-framed V7 body for low-latency live transport."""
    body = encode_frame(model, values, counter)
    out = np.zeros((PULSE_FRAME, 2), np.float32)
    out[PULSE.SYNC_LEN:PULSE.SYNC_LEN+FRAME] = body
    out[16:16+len(PULSE.PREAMBLE), :] = PULSE.PREAMBLE[:, None]
    meta = np.zeros((N//2+1, 2), complex)
    options = METADATA_OPTION_MONO_SUM if model.mono_sum else 0
    if source_index is None:
        source_index = counter - 1
    vals = metadata_symbols(aspect_code, model.encoding_type, options, source_index)
    meta[META_PILOTS, 0] = 1
    meta[META_DATA_BINS[:len(vals)], 0] = vals
    mx = (meta[:, 0])/np.sqrt(2)*model.phase[-1]
    meta_wave = np.fft.irfft(mx, n=N)
    meta_pcm = np.concatenate([meta_wave[-CP:], meta_wave])*model.scale
    meta_start = PULSE.SYNC_LEN+FRAME
    # Shape the ordinary pulse/body packet first.  The metadata symbol has its
    # own cyclic prefix and is inserted afterward so the long packet shaper
    # cannot smear the preceding image symbol across its pilots/data.
    shaped = bound_emission(out, 14000, RATE)
    shaped[meta_start:meta_start+META_SYMBOL, :] += meta_pcm[:, None]
    return shaped


def encode_pulse_stream(model, values, start_counter=1, aspect_codes=None,
                        source_indices=None):
    values = list(values) if np.asarray(values).ndim != 1 else [values]
    codes = aspect_codes or [0]*len(values)
    indexes = (list(source_indices) if source_indices is not None else
               [start_counter+i-1 for i in range(len(values))])
    if len(codes) != len(values) or len(indexes) != len(values):
        raise ValueError('aspect_codes and source_indices must match values')
    return np.concatenate([
        encode_pulse_frame(model, value, start_counter+i, code, source_index)
        for i, (value, code, source_index) in
        enumerate(zip(values, codes, indexes))])


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


def channel_joint(Z, iters=2):
    """Per rx channel: static 1x2 response per pilot bin x smooth timing track.

    y(s,b) = exp(j*2*pi*b*delta(s)/N) * (hM(b)*pM + hS(b)*pS), delta piecewise
    linear over the frame. Alternating LS: responses given delta, then a
    Gauss-Newton phase step for delta. Timing error left by the clock map is
    common to all carriers of a symbol, so the pilots pin it down (§9.3).
    """
    sv, bv, pv = PILOT_SV, PILOT_BV, PILOT_PV
    H = np.empty((F, 65, 2, 2), complex)
    for c in range(2):
        y = np.array([Z[s, b, c] for s, b in PILOT_OBS])
        theta = np.zeros(len(KNOTS))
        for _ in range(iters):
            delta = _BASIS @ theta
            rot = np.exp(2j*np.pi*bv*delta[sv]/N)
            h = np.empty((len(PILOT_BINS), 2), complex)
            for b in PILOT_BINS:
                m = PILOT_MASK_BY_BIN[b]
                A = pv[m]*rot[m, None]
                gram = A.conj().T @ A
                rhs = A.conj().T @ y[m]
                h[PILOT_BIN_INDEX[m][0]] = _solve_2x2_vec(gram, rhs)
            pred = rot*np.einsum('ij,ij->i', h[PILOT_BIN_INDEX], pv)
            ok = np.abs(pred) > 1e-9
            ph = np.angle(y[ok]/pred[ok]); w = np.abs(pred[ok])
            J = (2*np.pi*bv[ok]/N)[:, None]*_BASIS[sv[ok]]
            JW = J*w[:, None]
            normal = JW.T @ JW + 1e-10*np.eye(JW.shape[1])
            step = np.linalg.solve(normal, JW.T @ (ph*w))
            theta += step
            if np.max(np.abs(step)) < 1e-4:
                break
        delta = _BASIS @ theta
        hb = h                                                   # (pilots, 2)
        for k in range(2):
            v = hb[:, k]
            hk = np.interp(BINS, PILOT_BINS, v.real) + 1j*np.interp(BINS, PILOT_BINS, v.imag)
            H[:, BINS, c, k] = hk[None, :]*np.exp(2j*np.pi*BINS[None, :]*delta[:, None]/N)
    return H


def fade_and_noise(Z, H):
    """Per-symbol magnitude/phase refit (§9.4) and pilot-residual noise (§9.5)."""
    freq = BINS*RATE/N/1000
    noise = np.zeros((F, 2))
    for s in range(F):
        pil = PILOT_BY_SYMBOL[s]
        pvals = PILOT_VALUES_BY_SYMBOL[s]
        for ch in range(2):
            pred = np.einsum('bi,bi->b', H[s, pil, ch], pvals)
            obs = Z[s, pil, ch]
            mag = np.abs(pred)
            ok = mag > 0.3*mag.max()
            if ok.sum() < 2:
                continue
            rho = obs[ok]/pred[ok]; fb = np.asarray(pil)[ok]*RATE/N/1000; w = mag[ok]
            if ok.sum() >= 3 and np.ptp(fb) > 3:
                A = np.stack([np.ones(ok.sum()), -fb], 1)*w[:, None]
                normal = A.T @ A + 1e-10*np.eye(2)
                u, v = np.linalg.solve(
                    normal, A.T @ (np.log(np.abs(rho)+1e-12)*w))
                v = max(v, 0.0)
            else:
                u, v = float(np.average(np.log(np.abs(rho)+1e-12), weights=w)), 0.0
            a = float(np.angle(np.sum(rho*w))); bslope = 0.0
            corr = np.exp(u - v*freq + 1j*(a + bslope*freq))
            H[s, BINS, ch, :] *= corr[:, None]
            pred2 = np.einsum('bi,bi->b', H[s, pil, ch], pvals)
            dof = max(len(pil)-3, 1)
            noise[s, ch] = np.sum(np.abs(obs-pred2)**2)/dof
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


def decode_frame(model, x, tmap, counter, prev_tail, cancel=True,
                 diagnostics=None, direct_body=None):
    started = perf_counter()
    stage_started = started
    if direct_body is None:
        c0, nom, d = tmap
        base = (counter-c0)*FRAME
        n = base + np.arange(-64, FRAME+64, dtype=float)
        pos = n + np.interp(n, nom, d)
        if pos[0] < 0 or pos[-1] >= len(x)-1:
            return None
        seg = _sample_at(x, pos, taps=16).astype(float)
        # Clock cancellation (§9.2): regenerate from this frame's word, LS gain.
        if cancel:
            clk = clock_cancel_template(counter)
            for ch in range(2):
                a = np.dot(seg[:, ch], clk)/np.dot(clk, clk)
                seg[:, ch] -= a*clk
        seg = seg[64:64+FRAME]
    else:
        seg = np.asarray(direct_body, dtype=float)
    windows = seg.reshape(F, SYM, 2)[:, WIN:WIN+N, :]
    Z = (np.fft.rfft(windows, axis=1)/model.scale *
         np.conj(model.phase)[:, :, None] * EARLY[None, :, None])
    if diagnostics is not None:
        diagnostics.setdefault('stage_ms', {}).setdefault('sample_fft', []).append(
            (perf_counter()-stage_started)*1000)
    stage_started = perf_counter()
    H = channel_joint(Z)
    H, noise = fade_and_noise(Z, H)
    if diagnostics is not None:
        diagnostics.setdefault('stage_ms', {}).setdefault('channel', []).append(
            (perf_counter()-stage_started)*1000)
    stage_started = perf_counter()
    DEBUG['Z'], DEBUG['H'], DEBUG['noise'] = Z, H, noise
    # Per-cell 2x2 MMSE (§9.6 step 1) with priors from group powers.  Keep all
    # blocks and both quadratures in one batch: the small solve is cheap, but
    # entering Python once per block/channel was not.
    idx = model.rank_tables[counter % TAIL_PHASES]
    tx_var = np.array([np.mean(np.where(r >= 0, model.gain[np.maximum(r, 0)]**2 *
                                        model.lam[np.maximum(r, 0)], 0)) for r in idx])
    valid_group = BLOCK_GROUP_INDEX >= 0
    block_priors = np.where(valid_group, tx_var[np.maximum(BLOCK_GROUP_INDEX, 0)], 0.0)
    Hc = H[BLOCK_SYMBOLS, BLOCK_BINS[:, None]]
    Zc = Z[BLOCK_SYMBOLS, BLOCK_BINS[:, None]]
    estimates = np.zeros((len(BLOCKS), 2, 2, 8), complex)
    variances = np.full((len(BLOCKS), 2, 2, 8), np.inf)
    for q in range(2):
        prior = block_priors[:, :, q]
        safe = prior + 1e-12
        S = (Hc*safe[:, None, None, :]) @ Hc.conj().transpose(0, 1, 3, 2)
        S[..., 0, 0] += np.maximum(noise[BLOCK_SYMBOLS, 0],
                                   NOISE_FLOOR)
        S[..., 1, 1] += np.maximum(noise[BLOCK_SYMBOLS, 1],
                                   NOISE_FLOOR)
        M = safe[:, None, :, None] * Hc.conj().transpose(0, 1, 3, 2)
        W = _solve_2x2_mat(S, M)
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
    y_all = np.empty((group_count, 8)); sig_all = np.empty((group_count, 8))
    values_all = estimates[GROUP_BLOCK, GROUP_STREAM_INDEX, GROUP_Q_INDEX]
    vars_all = variances[GROUP_BLOCK, GROUP_STREAM_INDEX, GROUP_Q_INDEX]
    y_all[:] = values_all.real
    y_all[GROUP_Q_INDEX == 1] = values_all[GROUP_Q_INDEX == 1].imag
    sig_all[:] = np.where(np.isfinite(vars_all), vars_all/2, 1e9)
    lam_all = np.where(live_all, model.lam[np.maximum(ranks_all, 0)], 1e-12)
    gain_all = np.where(live_all, model.gain[np.maximum(ranks_all, 0)], 0)
    A_all = H8[None, :, :]*gain_all[:, None, :]
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
    xhat = np.zeros_like(model.mu); conf = np.zeros_like(model.mu)
    got = np.zeros_like(model.mu, bool)
    for gi in range(group_count):
        r = ranks_all[gi, live_all[gi]]
        xhat[r] = x_all[gi, live_all[gi]]
        conf[r] = conf_all[gi, live_all[gi]]
        got[r] = True
    if diagnostics is not None:
        diagnostics.setdefault('stage_ms', {}).setdefault('equalize', []).append(
            (perf_counter()-stage_started)*1000)
    floor = np.where(model.head, np.where(model.plane == 0, .05, .15),
                     np.where(model.plane == 0, .45, .60))
    gate = np.clip((conf-floor)/(.85-floor), 0, 1)
    current = model.mu + xhat*gate
    coeffs = prev_tail.copy()
    head_confidence = float(np.mean(conf[model.head]))
    head_coverage = float(np.mean(conf[model.head] >= .15))
    if head_confidence < HEAD_MIN_CONFIDENCE or head_coverage < HEAD_MIN_COVERAGE:
        displayable = (head_confidence >= DISPLAY_MIN_HEAD_CONFIDENCE and
                       head_coverage >= DISPLAY_MIN_HEAD_COVERAGE)
        display_coeffs = prev_tail.copy()
        if displayable:
            display_coeffs[got] = current[got]
        return Result(counter, 'lost', display_coeffs if displayable else prev_tail.copy(), {
            'noise': noise.mean(0).tolist(), 'got': int(got.sum()),
            'head_confidence': head_confidence,
            'head_coverage': head_coverage, 'held': not displayable,
            'displayable': displayable,
            'display_coeffs': display_coeffs})
    coeffs[got] = (model.mu + xhat*gate)[got]
    return Result(counter, 'verified', coeffs,
                  {'noise': noise.mean(0).tolist(), 'got': int(got.sum()),
                   'head_confidence': head_confidence,
                   'head_coverage': head_coverage, '_H': H,
                   'displayable': True})


def decode_metadata(model, samples, start, scale, channel):
    indexes = start + np.arange(META_SYMBOL)*scale
    if indexes[-1] >= len(samples)-1:
        return None
    meta = _sample_at(samples, indexes, taps=16)
    window = meta[WIN:WIN+N]
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
    data = np.divide(observed[META_DATA_BINS], response,
                     out=np.zeros(len(META_DATA_BINS), complex),
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


def decode_stream(model, x, verbose=False, diagnostics=None):
    started = perf_counter()
    x = np.asarray(x, float)
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
    results, tail = [], model.mu.copy()
    skipped = []
    for counter in range(lo, hi+1):
        try:
            r = decode_frame(model, x, tm, counter, tail,
                             cancel=bool(verified), diagnostics=diagnostics)
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


def decode_pulse_stream(model, x, diagnostics=None, latest_only=False,
                        input_gain=1.0, models=None, model_factory=None):
    """Decode V7 bodies located by the existing pulse-counted acquisition.

    This is the low-latency live path: each accepted pulse word supplies a
    frame scale, the body is resampled directly to the reference grid, and the
    next pulse search starts after that frame.  There is no continuous clock
    track or buffered clock-template refinement.
    """
    # Live capture is float32 and _sample_at returns float32.  Promoting the
    # complete rolling history to float64 here only doubles allocation and
    # memory traffic; the FFT/equalizer still performs its own complex work at
    # the precision NumPy requires.
    samples = np.asarray(x, np.float32)*np.float32(input_gain)
    cursor = 0
    counter = 1
    results = []
    tail = model.mu.copy()
    measured = None
    pending_aspect = 0
    if latest_only:
        # The live rolling buffer can contain the previous frame plus the new
        # one.  Find all pulse starts, but run the expensive image decode only
        # on the newest complete frame.
        candidates = []
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
            candidates.append((fs, sc, conf, 0))
            scan = int(fs + (PULSE_FRAME-32)*sc)
        if len(candidates) < 2:
            return [], {'frames': 0, 'pulse_frames': 0, 'recovered': False}
        # The second pulse is the first edge of the next header.  It is enough
        # to validate the current frame duration; the next body need not exist.
        fs, sc, conf, pending_aspect = candidates[-2]
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
        candidate = PULSE.measure_pulses(samples[search:].mean(axis=1),
                                      min_scale=PULSE_MIN_SCALE,
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
            results.append(Result(counter, 'lost', tail.copy(), {
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
        decoded_metadata = decode_metadata(model, samples, meta_start, scale, None)
        metadata_valid = decoded_metadata is not None
        encoding_type = model.encoding_type
        revision = 0
        source_index = None
        if metadata_valid:
            aspect_code, encoding_type, revision, source_index = decoded_metadata
            if (revision not in (0, METADATA_OPTION_MONO_SUM) or
                    encoding_type >= len(ENCODING_FILTERS)):
                metadata_valid = False
                encoding_type = model.encoding_type
                revision = 0
        selected_model = model
        if metadata_valid:
            selected_model = (models or {}).get(encoding_type)
            if selected_model is None and model_factory is not None:
                selected_model = model_factory(encoding_type)
            if selected_model is None:
                selected_model = model
        body = _sample_at(samples, indexes, taps=16).astype(np.float32)
        if revision == METADATA_OPTION_MONO_SUM:
            body = np.repeat(body.mean(axis=1, keepdims=True), 2, axis=1)
        elif body.shape[1] == 1:
            # The demodulator is M/S two-channel internally.  A mono capture
            # is the shared M observation, so duplicate it without inventing S.
            body = np.repeat(body, 2, axis=1)
        nominal = np.array([0., FRAME])
        offset = np.array([64., 64.])
        try:
            result = decode_frame(selected_model, None, (counter, nominal, offset),
                                  counter, tail, cancel=False, direct_body=body,
                                  diagnostics=diagnostics)
        except (FloatingPointError, np.linalg.LinAlgError, ValueError,
                IndexError):
            result = None
        if result is not None:
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
            result.diag['revision'] = int(revision)
            result.diag['mono_sum'] = revision == METADATA_OPTION_MONO_SUM
            result.diag['pulse_scale'] = float(scale)
            result.diag['frame_scale'] = float(frame_scale)
            result.diag['playback_speed'] = float(1/max(scale, 1e-9))
            result.diag['timing_delta_ppm'] = float(
                (frame_scale/scale-1)*1e6)
            results.append(result)
            if result.status != 'lost':
                tail = result.coeffs.copy()
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
