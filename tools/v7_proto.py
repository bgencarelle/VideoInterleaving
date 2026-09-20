"""V7 prototype: encoder and decoder following docs/transport_v7_spec.md.

EXPERIMENTAL, bench-only. Nothing in animation_modem imports this, and no live
sender/receiver uses it. It exists to test the V7 draft end to end; see the
"Prototype findings" section of the spec for what it changed in the draft.

Scope: a faithful, unoptimised simulation of the draft wire -- clock/identity
track (§5), continuous OFDM with scattered/continual pilots (§6), M/S
precoding with mono head blocks, SoftCast gains, rank windows and Hadamard
time groups (§7-§8), and the receiver chain of §9 (edge-counted clock,
time-map resampling, clock cancellation, pilot channel + per-symbol fade
model, per-cell 2x2 MMSE, group LMMSE, confidence gate, tail store).
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.linalg import hadamard
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt, firwin, savgol_filter, sosfiltfilt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from animation_modem import v6                                         # noqa: E402
from animation_modem.core import SourceCoder, _sample_at               # noqa: E402
from animation_modem.imaging import image_values, prepare_image        # noqa: E402

# ------------------------------------------------------------------ §6.1
RATE, N, CP, SYM, F = 48000, 128, 16, 144, 24
FRAME = F*SYM                                   # 3456 samples, 13.889 fps
BINS = np.arange(4, 35)                         # 1.5-12.75 kHz
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


def clock_word(counter, profile=0, aspect=6, folders=0):
    payload = (_bits(counter % (1 << 20), 20) + _bits(profile, 4) + _bits(aspect, 3) +
               _bits(folders, 8) + [0]*4)                        # 39 bits
    padded = [0] + payload
    data = bytes(int(''.join(map(str, padded[i:i+8])), 2) for i in range(0, 40, 8))
    word = SYNC + payload + _bits(crc16(data), 16)
    word.append(sum(word) % 2)                                   # polarity bit 71
    assert len(word) == 72
    return word


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


# ------------------------------------------------------------------ §6.4, §8
def data_blocks():
    blocks = [(b, p) for b in range(5, 34) for p in range(3) if SCAT.get(b) != p]
    return sorted(blocks)                               # health order (§8.1)


BLOCKS = data_blocks()
MONO = [blk for blk in BLOCKS if blk[0] <= 9]
STEREO = [blk for blk in BLOCKS if blk[0] > 9]
GROUPS = ([(b, 'M', q) for b in MONO for q in 'IQ'] +
          [(b, c, q) for b in STEREO for c in 'MS' for q in 'IQ'])
N_HEAD_G, N_TAIL_G = 2*len(MONO), 12                    # 26 head, 12 tail groups
HEAD, BODY_END, TAIL_PER = 208, 2224, 96
TAIL_PHASES = 7


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


def build_model(fixture, target_rms):
    coder = SourceCoder(v6.V6_SHAPES, grids=v6.V6_GRIDS)
    rng = np.random.default_rng(1)
    im = Image.open(fixture).convert('RGB'); W, Hh = im.size
    C = []
    for _ in range(120):
        s = rng.uniform(.45, 1.0); w = int(min(W, Hh*.75)*s); h = int(w*4/3)
        x = rng.integers(0, W-w+1); y = rng.integers(0, Hh-h+1)
        crop = im.crop((x, y, x+w, y+h))
        if rng.random() < .5:
            crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
        C.append(coder.forward(image_values(prepare_image(crop), coder.grids))/coder.gains)
    C = np.asarray(C)
    mu = np.zeros(C.shape[1]); lam = np.empty(C.shape[1])
    off, plane = 0, np.empty(C.shape[1], int)
    for p, ((r, c), (gr, gc)) in enumerate(zip(v6.V6_SHAPES, v6.V6_GRIDS)):
        sl = slice(off, off+r*c); plane[sl] = p
        mu[off] = C[:, off].mean()                       # DC library mean
        L = np.mean(C[:, sl]**2, axis=0)
        rad = np.hypot(np.arange(r)[:, None]/gr, np.arange(c)[None, :]/gc).ravel()
        f = lambda R, a, k, q: a - q*np.log1p(k*R)
        m = rad > 0
        (a, k, q), _ = curve_fit(f, rad[m], np.log(L[m]), p0=(0, 20, 2),
                                 bounds=([-50, 0, 0], [50, 1e4, 10]), maxfev=20000)
        model = np.exp(f(rad, a, k, q)); model[0] = C[:, off].var() + 1e-6
        lam[sl] = model; off += r*c
    order = np.argsort(-lam, kind='stable')
    g = lam**-.25
    g /= np.sqrt(np.mean((g*g*lam)[order[:BODY_END]]))   # unit mean slot power
    phase = np.exp(2j*np.pi*np.random.default_rng(70001).random((F, 65)))
    head = np.zeros(C.shape[1], bool); head[order[:HEAD]] = True
    model = Model(coder, mu, lam, order, g, phase, 1.0, plane, head)
    # Fixed level (§6.6): one-off calibration on seeded synthetic coefficients
    # drawn from the variance table -- a property of the profile, never of the
    # frame being sent.
    synth = np.random.default_rng(6).standard_normal(len(lam))*np.sqrt(lam) + mu
    probe = encode_frame_coeffs(model, synth, 1)
    model.scale = target_rms/np.sqrt(np.mean(probe**2))
    return model


# ------------------------------------------------------------------ encoder
def encode_frame(model, values, counter):
    return encode_frame_coeffs(model, model.coder.forward(values)/model.coder.gains, counter)


def encode_frame_coeffs(model, coeffs, counter, return_X=False):
    c = coeffs - model.mu
    idx = frame_ranks(model.order, counter)
    X = np.zeros((F, 65, 2), complex)                        # symbol, bin, M/S
    for gi, (blk, stream, q) in enumerate(GROUPS):
        b, phi = blk
        ranks = idx[gi]
        vals = np.where(ranks >= 0, model.gain[ranks]*c[np.maximum(ranks, 0)], 0.0)
        tx = H8 @ vals
        syms = phi + 3*np.arange(8)
        X[syms, b, 0 if stream == 'M' else 1] += tx*(1 if q == 'I' else 1j)
    for s in range(F):                                       # pilots (§6.3)
        for b in CONTINUAL:
            X[s, b] = PILOT_AMP*np.array([1, 1j*(-1)**s])
        for b, phi in SCAT.items():
            if s % 3 == phi:
                v = (s-phi)//3
                X[s, b] = PILOT_AMP*np.array([1, 1j*(-1)**v])
    if return_X:
        return X
    XL = (X[..., 0]+X[..., 1])/np.sqrt(2)*model.phase
    XR = (X[..., 0]-X[..., 1])/np.sqrt(2)*model.phase
    out = np.empty((FRAME, 2))
    for s in range(F):
        for ch, Xc in enumerate((XL, XR)):
            t = np.fft.irfft(Xc[s], n=N)*model.scale
            out[s*SYM:(s+1)*SYM, ch] = np.concatenate([t[-CP:], t])
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


KNOTS = np.array([0, 4, 8, 12, 16, 20, F-1], float)
_BASIS = np.stack([np.interp(np.arange(F), KNOTS, np.eye(len(KNOTS))[k])
                   for k in range(len(KNOTS))], axis=1)          # (F, knots) hat basis


def channel_joint(Z, iters=5):
    """Per rx channel: static 1x2 response per pilot bin x smooth timing track.

    y(s,b) = exp(j*2*pi*b*delta(s)/N) * (hM(b)*pM + hS(b)*pS), delta piecewise
    linear over the frame. Alternating LS: responses given delta, then a
    Gauss-Newton phase step for delta. Timing error left by the clock map is
    common to all carriers of a symbol, so the pilots pin it down (§9.3).
    """
    obs = sorted(PATS)                                    # (s, b)
    sv = np.array([o[0] for o in obs]); bv = np.array([o[1] for o in obs])
    pv = np.array([PATS[o] for o in obs])                 # (n, 2)
    H = np.empty((F, 65, 2, 2), complex)
    for c in range(2):
        y = np.array([Z[s, b, c] for s, b in obs])
        theta = np.zeros(len(KNOTS))
        for _ in range(iters):
            delta = _BASIS @ theta
            rot = np.exp(2j*np.pi*bv*delta[sv]/N)
            h = {}
            for b in PILOT_BINS:
                m = bv == b
                A = pv[m]*rot[m, None]
                h[b], *_ = np.linalg.lstsq(A, y[m], rcond=None)
            pred = np.array([rot[i]*(h[bv[i]] @ pv[i]) for i in range(len(y))])
            ok = np.abs(pred) > 1e-9
            ph = np.angle(y[ok]/pred[ok]); w = np.abs(pred[ok])
            J = (2*np.pi*bv[ok]/N)[:, None]*_BASIS[sv[ok]]
            step, *_ = np.linalg.lstsq(J*w[:, None], ph*w, rcond=None)
            theta += step
            if np.max(np.abs(step)) < 1e-4:
                break
        delta = _BASIS @ theta
        hb = np.array([h[b] for b in PILOT_BINS])              # (pilots, 2)
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
        pil = [b for (ss, b) in PATS if ss == s]
        for ch in range(2):
            pred = np.array([H[s, b, ch] @ PATS[(s, b)] for b in pil])
            obs = np.array([Z[s, b, ch] for b in pil])
            mag = np.abs(pred)
            ok = mag > 0.3*mag.max()
            if ok.sum() < 2:
                continue
            rho = obs[ok]/pred[ok]; fb = np.asarray(pil)[ok]*RATE/N/1000; w = mag[ok]
            if ok.sum() >= 3 and np.ptp(fb) > 3:
                A = np.stack([np.ones(ok.sum()), -fb], 1)*w[:, None]
                (u, v), *_ = np.linalg.lstsq(A, np.log(np.abs(rho)+1e-12)*w, rcond=None)
                v = max(v, 0.0)
            else:
                u, v = float(np.average(np.log(np.abs(rho)+1e-12), weights=w)), 0.0
            a = float(np.angle(np.sum(rho*w))); bslope = 0.0
            corr = np.exp(u - v*freq + 1j*(a + bslope*freq))
            H[s, BINS, ch, :] *= corr[:, None]
            pred2 = np.array([H[s, b, ch] @ PATS[(s, b)] for b in pil])
            dof = max(len(pil)-3, 1)
            noise[s, ch] = np.sum(np.abs(obs-pred2)**2)/dof
    # Smooth over +-1 symbol, floor relative to pilot power.
    k = np.array([1, 1, 1])/3
    for ch in range(2):
        # Additive noise does not vanish when the signal fades (a dropout
        # leaves tiny residuals): floor each symbol at the frame median.
        sm = np.convolve(noise[:, ch], k, mode='same')
        noise[:, ch] = np.maximum(np.maximum(sm, np.median(sm)),
                                  1e-5*PILOT_AMP**2*np.mean(np.abs(H[:, BINS, ch])**2))
    return H, noise


def decode_frame(model, x, tmap, counter, prev_tail, cancel=True):
    c0, nom, d = tmap
    base = (counter-c0)*FRAME
    n = base + np.arange(-64, FRAME+64, dtype=float)
    pos = n + np.interp(n, nom, d)
    if pos[0] < 0 or pos[-1] >= len(x)-1:
        return None
    seg = _sample_at(x, pos, taps=16).astype(float)            # nominal-time frame
    # Clock cancellation (§9.2): regenerate from this frame's word, LS gain.
    clk = clock_wave([clock_word(counter-1), clock_word(counter), clock_word(counter+1)])
    clk = clk[FRAME-64:2*FRAME+64]
    for ch in range(2 if cancel else 0):
        a = np.dot(seg[:, ch], clk)/np.dot(clk, clk)
        seg[:, ch] -= a*clk
    seg = seg[64:64+FRAME]
    Z = np.empty((F, 65, 2), complex)
    for s in range(F):
        w = seg[s*SYM+WIN:s*SYM+WIN+N]
        Z[s] = (np.fft.rfft(w, axis=0)/model.scale*np.conj(model.phase[s])[:, None] *
                EARLY[:, None])
    H = channel_joint(Z)
    H, noise = fade_and_noise(Z, H)
    DEBUG['Z'], DEBUG['H'], DEBUG['noise'] = Z, H, noise
    # Per-cell 2x2 MMSE (§9.6 step 1) with priors from group powers.
    idx = frame_ranks(model.order, counter)
    tx_var = np.array([np.mean(np.where(r >= 0, model.gain[np.maximum(r, 0)]**2 *
                                        model.lam[np.maximum(r, 0)], 0)) for r in idx])
    pw = {}
    for gi, (blk, stream, q) in enumerate(GROUPS):
        pw.setdefault(blk, np.zeros(2))[0 if stream == 'M' else 1] += tx_var[gi]
    est = {}
    for blk, P in pw.items():
        b, phi = blk
        for t in range(8):
            s = phi + 3*t
            Hc = H[s, b]; Pd = np.diag(P+1e-12); Nn = np.diag(np.maximum(noise[s], NOISE_FLOOR))
            S = Hc @ Pd @ Hc.conj().T + Nn
            W = Pd @ Hc.conj().T @ np.linalg.pinv(S)
            xt = W @ Z[s, b]; B = W @ Hc
            cov = W @ S @ W.conj().T
            for k in range(2):
                if P[k] <= 0:
                    continue
                beta = B[k, k].real
                if beta < 1e-6:
                    est[(blk, k, t)] = (0j, np.inf); continue
                var = max((cov[k, k].real - beta**2*P[k]), 1e-12)/beta**2
                est[(blk, k, t)] = (xt[k]/beta, var)
    # Group LMMSE (§9.6 step 2) + gate (step 3).
    xhat = np.zeros_like(model.mu); conf = np.zeros_like(model.mu); got = np.zeros_like(model.mu, bool)
    for gi, (blk, stream, q) in enumerate(GROUPS):
        k = 0 if stream == 'M' else 1
        ranks = idx[gi]; live = ranks >= 0
        y = np.empty(8); sig = np.empty(8)
        for t in range(8):
            val, var = est.get((blk, k, t), (0j, np.inf))
            y[t] = val.real if q == 'I' else val.imag
            sig[t] = var/2 if np.isfinite(var) else 1e9
        lam = np.where(live, model.lam[np.maximum(ranks, 0)], 1e-12)
        g = np.where(live, model.gain[np.maximum(ranks, 0)], 0)
        A = H8*g[None, :]
        S = A @ np.diag(lam) @ A.T + np.diag(sig)
        K = np.diag(lam) @ A.T @ np.linalg.pinv(S)
        xg = K @ y
        post = lam - np.einsum('ij,ji->i', K, A @ np.diag(lam))
        cg = np.clip(1 - post/lam, 0, 1)
        r = ranks[live]
        xhat[r] = xg[live]; conf[r] = cg[live]; got[r] = True
    floor = np.where(model.head, np.where(model.plane == 0, .05, .15),
                     np.where(model.plane == 0, .45, .60))
    gate = np.clip((conf-floor)/(.85-floor), 0, 1)
    coeffs = prev_tail.copy()
    coeffs[got] = (model.mu + xhat*gate)[got]
    return Result(counter, 'verified', coeffs,
                  {'noise': noise.mean(0).tolist(), 'got': int(got.sum())})


def decode_stream(model, x, verbose=False):
    x = np.asarray(x, float)
    def read(sig):
        y = clock_signal(sig)
        return y, find_words(biphase_bits(y, clock_edges(y)))
    y, words = read(x.mean(axis=1))
    # Per-channel fallback (§5.3 step 1) when M yields few words.
    for ch in range(2):
        if len(words) >= 2:
            break
        y, words = read(x[:, ch])
    if not words:
        return [], {'words': 0}
    # Keep words consistent with a monotone counter/time relation.
    words.sort(key=lambda w: w['bounds'][0])
    tm = time_map(words)
    if REFINE:
        tm = refine_time_map(y, tm, words)
    verified = {w['counter'] for w in words if w['verified']}
    allc = {w['counter'] for w in words}
    lo, hi = min(allc), max(allc)
    results, tail = [], model.mu.copy()
    for counter in range(lo, hi+1):
        r = decode_frame(model, x, tm, counter, tail, cancel=bool(verified))
        if r is None:
            continue
        r.status = 'verified' if counter in verified else 'picture_only'
        tail = r.coeffs.copy()
        results.append(r)
    return results, {'words': len(words), 'crc_ok': len(verified)}


def values_from(model, coeffs):
    return model.coder.inverse(coeffs*model.coder.gains)
