"""V3 transport: countable preamble, edge-capture acquisition, corrected level.

This is the only transport. It owns the preamble and the way the receiver finds
the packet; `core.decode_packet` demodulates once the timing is known.
"""
import numpy as np
from functools import lru_cache
from scipy.signal import correlate
from .audio_common import (pcm, pair, device, wav_blocks, wav_rate, wire_notice,
                           InputLevel, sounddevice)
from .core import (REFERENCE_RATE, N, CP, SYMBOL, SYNC_LEN, GUARD, HEADER_GAIN,
                         IMAGE_GAIN, HEADER_SLOTS, HEADER_FORMAT, HEADER_BYTES,
                         Layout, SourceCoder, Decoded, coefficient_slots,
                         coder_slots,
                         phases, pack_header, pack_folders, decode_packet,
                         default_allocation, resample_packet, _sample_at,
                         _body_walk, _decode_tables, FOLDER_LIMIT,
                         band_limited, emit_length, emit_ratio,
                         PROFILE_CODES, profile_code, profile_name,
                         _recovery_quality)

# --------------------------------------------------------------------------
# Preamble
# --------------------------------------------------------------------------

HALF = 8                   # samples per half-bit: '1' -> 3 kHz, '0' -> 1.5 kHz
PREAMBLE_BITS = (0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0)
PREAMBLE_AMPLITUDE = .55

SHORT, LONG = HALF, 2*HALF
RELOCK_AFTER = 4           # unverified packets before a detected preset is abandoned
EDGE_HYSTERESIS = .12      # Schmitt threshold, relative to preamble amplitude


def _biphase(bits=PREAMBLE_BITS, half=HALF, amplitude=PREAMBLE_AMPLITUDE):
    """Biphase-mark: a transition every bit, plus a mid-bit one for a '1'."""
    level, out = 1., []
    for bit in bits:
        level = -level
        out += [level]*half
        if bit:
            level = -level
        out += [level]*half
    wave = np.asarray(out, float)
    return wave*(amplitude/np.max(np.abs(wave)))


# The one wire. There is exactly one layout now; the old preset table is gone.
#
# Chosen by the EQ+noise sweep: 54 top bin (375-20250 Hz), carriers spread so
# the image planes ride the middle of the band and EQ tints nothing, a LOW-band
# header on the lowest data carriers so identity survives a top-band cut and
# fast playback, and dense_header so the header symbols' idle carriers still
# carry image -- which pays for dropping to 14 image symbols while holding
# the full 2880-coefficient color-dct picture. 3280 values at 15.00 fps.
#
# The header is low AND narrow (header_width=20, four header symbols) on
# purpose, independent of spread_carriers. Measured: widening to 27 drops
# treble+noise identity from ~30/32 to ~19/32 -- the top header carriers sit
# in the treble cut, buried, and three symbols give the tolerance search less
# to work with. A full-band header (the old header_width=50) is faster but
# loses identity under a 15 kHz lowpass or 2x playback; a split header was
# measured strictly worse still. Change the wire HERE, nowhere else.
WIRE = Layout(top_bin=54, image_symbols=14, name='wire',
              progressive=True, orthogonal_training=True,
              spread_carriers=True, dense_header=True, header_width=20)

# v5: HD layout - improved quality at same wire size, with CDF 9/7 + LDPC
# Top bin 54 -> 375-20250 Hz carriers, spread for EQ immunity
# image_symbols=16 gives ~18 fps @ 48 kHz, ~3800 capacity (vs 2880 v3/v4)
# header_width=20, dense_header reclaims header symbol carriers
WIRE_HD = Layout(top_bin=54, image_symbols=16, name='wire-hd',
                 progressive=True, orthogonal_training=True,
                 spread_carriers=True, dense_header=True, header_width=20)

# v5 preamble: shorter biphase-mark for 1600-sample frame
# 12 bits = 1.5kHz/3kHz edges, ~192 samples @ 8 samples/half-bit
HALF_V5 = 8
PREAMBLE_BITS_V5 = (0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1)
PREAMBLE_AMPLITUDE_V5 = .55

def _biphase_v5(bits=PREAMBLE_BITS_V5, half=HALF_V5, amplitude=PREAMBLE_AMPLITUDE_V5):
    level, out = 1., []
    for bit in bits:
        level = -level
        out += [level]*half
        if bit:
            level = -level
        out += [level]*half
    wave = np.asarray(out, float)
    return wave*(amplitude/np.max(np.abs(wave)))

PREAMBLE_V5 = _biphase_v5()
PREAMBLE_ENERGY_V5 = float(np.einsum('i,i->', PREAMBLE_V5, PREAMBLE_V5, optimize=False))
NOMINAL_EDGES_V5 = np.flatnonzero(np.diff(np.signbit(PREAMBLE_V5)))
NOMINAL_SPAN_V5 = float(NOMINAL_EDGES_V5[-1] - NOMINAL_EDGES_V5[0])
MAGIC_V5 = b'V5'

PREAMBLE = _biphase()
PREAMBLE_ENERGY = float(np.einsum('i,i->', PREAMBLE, PREAMBLE, optimize=False))
NOMINAL_EDGES = np.flatnonzero(np.diff(np.signbit(PREAMBLE)))
NOMINAL_SPAN = float(NOMINAL_EDGES[-1] - NOMINAL_EDGES[0])
MAGIC = b'V3'


def edge_intervals(samples, hysteresis=EDGE_HYSTERESIS*PREAMBLE_AMPLITUDE):
    state = np.where(samples > hysteresis, 1,
                     np.where(samples < -hysteresis, -1, 0))
    live = np.flatnonzero(state)
    if len(live) < 2:
        return np.empty(0, int)
    changes = np.flatnonzero(np.diff(state[live]))
    return live[changes + 1]


NOMINAL_GAPS = np.diff(NOMINAL_EDGES).astype(float)
MIN_RUN = len(NOMINAL_GAPS) - 4


def _runs(gaps, tolerance=.28):
    out = []
    start = 0
    n = len(gaps)
    while start < n:
        unit = float(np.min(gaps[start:start+MIN_RUN])) if start < n else 0.
        if unit <= 0:
            start += 1
            continue
        end = start
        while end < n:
            g = gaps[end]
            near_short = abs(g-unit) <= tolerance*unit
            near_long = abs(g-2*unit) <= tolerance*2*unit
            if not (near_short or near_long):
                break
            end += 1
        if end-start >= MIN_RUN:
            out.append((start, end, unit))
            start = end
        else:
            start += 1
    return out


def measure_speed(samples, min_scale=.5, max_scale=4.0):
    at = edge_intervals(samples)
    if len(at) < MIN_RUN+1:
        return None
    gaps = np.diff(at).astype(float)
    for start, end, unit in _runs(gaps):
        run = gaps[start:end]
        short = run[run < 1.5*unit]
        long = run[run >= 1.5*unit]
        if len(short) < 4 or len(long) < 2:
            continue
        unit = float(np.mean(short))
        if not 1.7 <= float(np.mean(long))/max(unit, 1e-9) <= 2.3:
            continue
        span = float(at[end] - at[start])
        scale = span/NOMINAL_SPAN
        if not .98*min_scale <= scale <= 1.02*max_scale:
            continue
        if abs(unit/SHORT - scale) > .15*scale:
            continue
        position = float(at[start]) - NOMINAL_EDGES[0]*scale
        agreement = 1.0 - min(1.0, float(np.std(short))/max(unit, 1e-9))
        return position, scale, agreement
    return None


def measure_pulses(samples, min_scale=.5, max_scale=4.0):
    samples = np.asarray(samples)
    edges = edge_intervals(samples)
    count = len(NOMINAL_EDGES)
    if len(edges) < count:
        return None
    crossings = np.flatnonzero(np.diff(np.signbit(samples)))
    left = crossings[np.searchsorted(crossings, edges-1, side='right')-1]
    values = samples[left]
    positions = left + values/(values-samples[left+1])
    words = np.lib.stride_tricks.sliding_window_view(positions, count)
    scales = (words[:, -1]-words[:, 0])/NOMINAL_SPAN
    valid = (scales >= .98*min_scale) & (scales <= 1.02*max_scale)
    expected = NOMINAL_GAPS[None, :]*scales[:, None]
    valid &= np.all(np.abs(np.diff(words, axis=1) - expected) <=
                    np.maximum(1.2, 0.45 * expected), axis=1)
    nominal = NOMINAL_EDGES.astype(float)+.5
    centered = nominal-nominal.mean()
    for word in words[valid]:
        if (word[-1]-word[0])/NOMINAL_SPAN < .999:
            refined = word.copy()
            mono = samples[:, None]
            for _ in range(3):
                probes = np.concatenate([refined, refined-.05, refined+.05])
                amplitudes = _sample_at(mono, probes, taps=16)[:, 0]
                center, minus, plus = np.split(amplitudes, 3)
                derivative = (plus-minus)/.1
                delta = np.divide(center, derivative, out=np.zeros_like(center),
                                  where=np.abs(derivative) > 1e-6)
                refined = np.clip(refined-np.clip(delta, -.25, .25), word-.5, word+.5)
            word = refined
        scale = float(np.dot(word-word.mean(), centered)/np.dot(centered, centered))
        position = float(word.mean()-scale*nominal.mean())
        residual = float(np.sqrt(np.mean((word-position-scale*nominal)**2)))
        confidence = max(0., 1 - residual / (SHORT * scale))
        if confidence >= 0.45:
            return position, scale, confidence
    return None


def preamble_correlation(x, limit=None, template=None, energy=None):
    if template is None:
        template, energy = PREAMBLE, PREAMBLE_ENERGY
    x = np.asarray(x, float)
    n = len(x) - len(template) + 1
    if limit is not None:
        n = max(0, min(n, limit))
    if n <= 0:
        return np.empty(0)
    peak = float(np.max(np.abs(x)))
    if peak == 0:
        return np.zeros(n)
    x = x/peak
    cumulative = np.empty(len(x)+1)
    cumulative[0] = 0.
    np.cumsum(x*x, out=cumulative[1:])
    window = cumulative[len(template):] - cumulative[:-len(template)]
    floor = 64*np.finfo(float).eps*float(np.max(window))
    corr = correlate(x[:n+len(template)-1], template, mode='valid', method='auto')
    score = np.zeros(n)
    ok = window[:n] > floor
    score[ok] = np.abs(corr[ok])/np.sqrt(window[:n][ok]*energy)
    return np.minimum(score, 1.0)


@lru_cache(maxsize=64)
def _scaled_preamble(scale):
    if scale == 1.0:
        return PREAMBLE
    length = int(round(len(PREAMBLE)*scale))
    return np.interp(np.arange(length)/scale, np.arange(len(PREAMBLE)), PREAMBLE)


def _fit_preamble(samples, at, scale, reach=.02, iterations=5, fast=True):
    radius = 12
    left = max(0, int(at)-radius)
    right = min(len(samples), int(at + len(PREAMBLE)*scale) + radius)
    local = samples[left:right]
    if len(local) < len(PREAMBLE)//2:
        return float(at), float(scale), 0.
    best = (0., float(at-left), float(scale))
    for s in (np.linspace(scale*(1-reach), scale*(1+reach), 7) if reach else ()):
        template = _scaled_preamble(float(s))
        energy = float(np.einsum('i,i->', template, template, optimize=False))
        for c in range(local.shape[1]):
            scores = preamble_correlation(local[:, c], 2*radius+1, template, energy)
            if len(scores):
                p = int(np.argmax(scores))
                if scores[p] > best[0]:
                    best = (float(scores[p]), float(p), float(s))
    _, position, s = best
    reference = PREAMBLE
    energy = PREAMBLE_ENERGY
    for _ in range(iterations):
        z = resample_packet(local, s-1, len(PREAMBLE), offset=position, fast=fast)
        scores = (abs(np.einsum('i,ic->c', reference, z, optimize=False)) /
                  np.sqrt(energy*np.maximum((z*z).sum(0), 1e-20)))
        c = int(np.argmax(scores))
        z = z[:, c].astype(float)
        gain = float(np.einsum('i,i->', reference, z, optimize=False))/energy
        derivative = (resample_packet(local, s-1, len(PREAMBLE), offset=position+.05, fast=fast)[:, c]
                      - resample_packet(local, s-1, len(PREAMBLE), offset=position-.05, fast=fast)[:, c])/.1
        j = np.stack([derivative, derivative*np.arange(len(PREAMBLE))/1000], axis=1)
        j -= reference[:, None]*np.einsum('i,ij->j', reference, j, optimize=False)[None, :]/energy
        gram = np.einsum('ni,nj->ij', j, j, optimize=False) + np.eye(2)*1e-12
        step = np.linalg.solve(gram, np.einsum('ni,n->i', j, z-gain*reference, optimize=False))
        step = np.clip(step, [-1., -s*2], [1., s*2])
        position = max(0, position-step[0])
        s -= step[1]/1000
        if np.max(abs(step)) < 1e-4:
            break
    z = resample_packet(local, s-1, len(PREAMBLE), offset=position, fast=fast)
    scores = (abs(np.einsum('i,ic->c', reference, z, optimize=False)) /
              np.sqrt(energy*np.maximum((z*z).sum(0), 1e-20)))
    return float(left+position), float(s), float(scores.max())


# --------------------------------------------------------------------------
# v5 Preamble detection (uses PREAMBLE_V5)
# --------------------------------------------------------------------------

NOMINAL_GAPS_V5 = np.diff(NOMINAL_EDGES_V5).astype(float)
MIN_RUN_V5 = len(NOMINAL_GAPS_V5) - 4

def _runs_v5(gaps, tolerance=.28):
    out = []
    start = 0
    n = len(gaps)
    while start < n:
        unit = float(np.min(gaps[start:start+MIN_RUN_V5])) if start < n else 0.
        if unit <= 0:
            start += 1
            continue
        end = start
        while end < n:
            g = gaps[end]
            near_short = abs(g-unit) <= tolerance*unit
            near_long = abs(g-2*unit) <= tolerance*2*unit
            if not (near_short or near_long):
                break
            end += 1
        if end-start >= MIN_RUN_V5:
            out.append((start, end, unit))
            start = end
        else:
            start += 1
    return out

def measure_speed_v5(samples, min_scale=.5, max_scale=4.0):
    at = edge_intervals(samples)
    if len(at) < MIN_RUN_V5+1:
        return None
    gaps = np.diff(at).astype(float)
    for start, end, unit in _runs_v5(gaps):
        run = gaps[start:end]
        short = run[run < 1.5*unit]
        long = run[run >= 1.5*unit]
        if len(short) < 4 or len(long) < 2:
            continue
        unit = float(np.mean(short))
        if not 1.7 <= float(np.mean(long))/max(unit, 1e-9) <= 2.3:
            continue
        span = float(at[end] - at[start])
        scale = span/NOMINAL_SPAN_V5
        if not .98*min_scale <= scale <= 1.02*max_scale:
            continue
        if abs(unit/SHORT - scale) > .15*scale:
            continue
        position = float(at[start]) - NOMINAL_EDGES_V5[0]*scale
        agreement = 1.0 - min(1.0, float(np.std(short))/max(unit, 1e-9))
        return position, scale, agreement
    return None

def preamble_correlation_v5(x, limit=None, template=None, energy=None):
    if template is None:
        template, energy = PREAMBLE_V5, PREAMBLE_ENERGY_V5
    x = np.asarray(x, float)
    n = len(x) - len(template) + 1
    if limit is not None:
        n = max(0, min(n, limit))
    if n <= 0:
        return np.empty(0)
    peak = float(np.max(np.abs(x)))
    if peak == 0:
        return np.zeros(n)
    x = x/peak
    cumulative = np.empty(len(x)+1)
    cumulative[0] = 0.
    np.cumsum(x*x, out=cumulative[1:])
    window = cumulative[len(template):] - cumulative[:-len(template)]
    floor = 64*np.finfo(float).eps*float(np.max(window))
    corr = correlate(x[:n+len(template)-1], template, mode='valid', method='auto')
    score = np.zeros(n)
    ok = window[:n] > floor
    score[ok] = np.abs(corr[ok])/np.sqrt(window[:n][ok]*energy)
    return np.minimum(score, 1.0)

@lru_cache(maxsize=64)
def _scaled_preamble_v5(scale):
    if scale == 1.0:
        return PREAMBLE_V5
    length = int(round(len(PREAMBLE_V5)*scale))
    return np.interp(np.arange(length)/scale, np.arange(len(PREAMBLE_V5)), PREAMBLE_V5)

def _fit_preamble_v5(samples, at, scale, reach=.02, iterations=5, fast=True):
    radius = 12
    left = max(0, int(at)-radius)
    right = min(len(samples), int(at + len(PREAMBLE_V5)*scale) + radius)
    local = samples[left:right]
    if len(local) < len(PREAMBLE_V5)//2:
        return float(at), float(scale), 0.
    best = (0., float(at-left), float(scale))
    for s in (np.linspace(scale*(1-reach), scale*(1+reach), 7) if reach else ()):
        template = _scaled_preamble_v5(float(s))
        energy = float(np.einsum('i,i->', template, template, optimize=False))
        scores = preamble_correlation_v5(local, 2*radius+1, template, energy)
        if len(scores):
            p = int(np.argmax(scores))
            if scores[p] > best[0]:
                best = (float(scores[p]), float(p), float(s))
    _, position, s = best
    reference = PREAMBLE_V5
    energy = PREAMBLE_ENERGY_V5
    for _ in range(iterations):
        z = resample_packet(local, s-1, len(PREAMBLE_V5), offset=position, fast=fast)
        scores = (abs(np.einsum('i,i->', reference, z, optimize=False)) /
                  np.sqrt(energy*np.maximum((z*z).sum(0), 1e-20)))
        if len(scores):
            p = int(np.argmax(scores))
            if scores[p] > best[0]:
                best = (float(scores[p]), float(position + p), float(s))
    return best[1], best[2], best[0]

def measure_pulses_v5(samples, min_scale=.5, max_scale=4.0):
    """v5 preamble detection using shorter PREAMBLE_V5."""
    samples = np.asarray(samples)
    edges = edge_intervals(samples)
    count = len(NOMINAL_EDGES_V5)
    if len(edges) < count:
        return None
    crossings = np.flatnonzero(np.diff(np.signbit(samples)))
    left = crossings[np.searchsorted(crossings, edges-1, side='right')-1]
    values = samples[left]
    positions = left + values/(values-samples[left+1])
    words = np.lib.stride_tricks.sliding_window_view(positions, count)
    scales = (words[:, -1]-words[:, 0])/NOMINAL_SPAN_V5
    valid = (scales >= .98*min_scale) & (scales <= 1.02*max_scale)
    expected = NOMINAL_GAPS_V5[None, :]*scales[:, None]
    valid &= np.all(np.abs(np.diff(words, axis=1) - expected) <=
                    np.maximum(1.2, 0.45 * expected), axis=1)
    nominal = NOMINAL_EDGES_V5.astype(float)+.5
    centered = nominal-nominal.mean()
    for word in words[valid]:
        if (word[-1]-word[0])/NOMINAL_SPAN_V5 < .999:
            refined = word.copy()
            for _ in range(3):
                probes = np.concatenate([refined, refined-.05, refined+.05])
                amplitudes = _sample_at(samples[None, :], probes, taps=16)[:, 0]
                center, minus, plus = np.split(amplitudes, 3)
                derivative = (plus-minus)/.1
                delta = np.divide(center, derivative, out=np.zeros_like(center),
                                  where=np.abs(derivative) > 1e-6)
                refined = np.clip(refined-np.clip(delta, -.25, .25), word-.5, word+.5)
            word = refined
        scale = float(np.dot(word-word.mean(), centered)/np.dot(centered, centered))
        position = float(word.mean()-scale*nominal.mean())
        residual = float(np.sqrt(np.mean((word-position-scale*nominal)**2)))
        confidence = max(0., 1 - residual / (SHORT * scale))
        if confidence >= 0.45:
            return position, scale, confidence
    return None


# --------------------------------------------------------------------------
# Encode
# --------------------------------------------------------------------------

def encode(values, layout, coder, absolute, index, count, stamp_ms=0, flags=0,
           headroom=.95, profile=0, aspect_code=0):
    from .aspect import pack_absolute
    expected = getattr(coder, 'source_count', coder.count)
    if len(values) != expected or not np.isfinite(values).all():
        raise ValueError('Expected one finite value per source coefficient')
    if coder.count > layout.capacity:
        raise ValueError(f'{coder.count} coefficients exceed capacity {layout.capacity}')
    carriers = layout.carriers
    data = np.searchsorted(carriers, layout.data_bins)
    pilots = np.searchsorted(carriers, layout.pilots)
    header = np.searchsorted(carriers, layout.header_bins)
    grid = np.zeros((layout.symbols, len(carriers), 2), complex)
    if layout.orthogonal_training:
        tone = phases(layout)
        lock0 = tone[0, :, 0]*np.conj(tone[0, :, 1])
        lock1 = tone[1, :, 0]*np.conj(tone[1, :, 1])
        grid[0, :, 0], grid[0, :, 1] = 1, lock0
        grid[1, :, 0], grid[1, :, 1] = 1, -lock1
    else:
        grid[0, :, 0] = 1
        grid[1, :, 1] = 1

    raw = pack_header(flags, layout.top_bin, pack_absolute(absolute, aspect_code), index, count, stamp_ms,
                      magic=layout.wire_magic,
                      profile=profile)
    bits = np.unpackbits(np.frombuffer(raw, np.uint8)).reshape(HEADER_SLOTS, 2)
    qpsk = ((bits[:, 0]*2.-1) + 1j*(bits[:, 1]*2.-1))/np.sqrt(2)
    if layout.header_split:
        room = layout.header_symbols*layout.header_lanes
        spread = np.resize(qpsk, room).reshape(layout.header_symbols,
                                               len(header), 2)
        for s in range(layout.header_symbols):
            grid[2+s, header, 0] = spread[s, :, 0]*HEADER_GAIN
            grid[2+s, header, 1] = spread[s, :, 1]*HEADER_GAIN
    else:
        room = layout.header_symbols*len(header)
        spread = np.resize(qpsk, room).reshape(layout.header_symbols, len(header))
        for s in range(layout.header_symbols):
            grid[2+s, header, 0] = spread[s]*HEADER_GAIN
            grid[2+s, header, 1] = spread[s]*HEADER_GAIN

    sent = np.zeros(layout.capacity)
    sent[coder_slots(layout, coder)] = coder.forward(values)
    head = layout.header_capacity
    if head:
        spare = np.searchsorted(carriers, layout.spare_bins)
        early = sent[:head].reshape(layout.header_symbols, len(spare), 2, 2)
        grid[2:2+layout.header_symbols, spare, :] = \
            (early[..., 0] + 1j*early[..., 1])*IMAGE_GAIN
    block = sent[head:].reshape(layout.image_symbols, len(data), 2, 2)
    grid[2+layout.header_symbols:, data, :] = (block[..., 0] + 1j*block[..., 1])*IMAGE_GAIN
    grid[2:, pilots, :] = 1

    spectrum = np.zeros((layout.symbols, N//2+1, 2), complex)
    spectrum[:, carriers, :] = grid*phases(layout)
    wave = np.fft.irfft(spectrum, n=N, axis=1)
    wave = np.concatenate([wave[:, -CP:], wave], axis=1).reshape(-1, 2)

    out = np.zeros((layout.frame, 2), np.float32)
    out[SYNC_LEN:layout.packet] = wave
    # Peak normalize only the OFDM part, not the preamble
    peak = float(np.max(np.abs(out[SYNC_LEN:layout.packet])))
    if peak > 0:
        out[SYNC_LEN:layout.packet] *= headroom/peak
    # Add preamble at full amplitude AFTER normalization
    out[16:16+len(PREAMBLE), 0] = PREAMBLE
    out[16:16+len(PREAMBLE), 1] = PREAMBLE
    return out


# --------------------------------------------------------------------------
# v5 Encode (stereo, compatible with v3/v4 wire format)
# --------------------------------------------------------------------------

def encode_v5(values, layout, coder, absolute, index, count, stamp_ms=0,
              headroom=.95, profile=2, aspect_code=0):
    """Encode for v5: stereo output, V3 header format, V3 preamble.
    
    Uses profile=2 (hd-dwt) in standard V3 header. Wire format identical to v3/v4.
    """
    # Use the standard v3 encode - same wire format, just different coder
    return encode(values, layout, coder, absolute, index, count,
                  stamp_ms=stamp_ms, headroom=headroom, profile=profile,
                  aspect_code=aspect_code)


# --------------------------------------------------------------------------
# Receive
# --------------------------------------------------------------------------

# A measured scale this far from unity earns a second decode at unity scale,
# best of the two kept. Dispersion biases the edge estimator (measured
# +0.26% under a highpassed channel); demodulating at a biased scale
# mis-stretches the walk and destroys amplitudes while the bits survive, so
# the biased path decodes "verified" garbage. Unity is right whenever the
# medium runs true; when it genuinely does not, the measured scale wins the
# comparison below. Clean packets (scale ~= 1) never pay the second attempt.
SCALE_SANITY = 1e-3


class Receiver:
    def __init__(self, layout, coder, threshold=.4, rate_window=None,
                 min_speed=.25, max_speed=2.0, recovery=False, fast=True,
                 pulse_only=True, input_rate=None, coders=None, candidates=None,
                 header_tolerance=2, diagnostics=False):
        if not (0 < min_speed <= 1 <= max_speed and min_speed >= .25 and max_speed <= 2):
            raise ValueError('Supported speed range: .25 <= min_speed <= 1 <= max_speed <= 2')
        if input_rate is not None and not (np.isfinite(input_rate) and input_rate > 0):
            raise ValueError('Input rate must be a positive number of hertz, or None')
        if coder.count > layout.capacity:
            raise ValueError('Source coder exceeds layout capacity')
        self.layout, self.coder, self.threshold = layout, coder, threshold
        self.coders = dict(coders) if coders else None
        self._start = (layout, coder, self.coders)
        self.candidates = sorted(candidates or [], key=lambda c: c[0].packet)
        for cand, cand_coder, _ in self.candidates:
            if cand_coder.count > cand.capacity:
                raise ValueError(f'Coder exceeds {cand.name} capacity')
        self.detected = None
        self.recovery = bool(recovery)
        self.fast = bool(fast)
        self.pulse_only = bool(pulse_only)
        self.min_speed, self.max_speed = min_speed, max_speed
        self.input_rate = None if input_rate is None else float(input_rate)
        self.header_tolerance = int(header_tolerance)
        self.diagnostics = bool(diagnostics)
        self.resets = -1  # Construction is not an input discontinuity.
        self.clock = 1. if input_rate is None else float(input_rate)/REFERENCE_RATE
        self.min_scale = self.clock/max_speed
        self.max_scale = self.clock/min_speed
        self.keep = int(np.ceil(len(PREAMBLE)*1.2*self.max_scale)) + 64
        self.reset()

    def reset(self, preserve_timing=False):
        self.resets += 1
        rate_error = self.rate_error if preserve_timing else 0.
        confidence = self.confidence if preserve_timing else 0.
        if not hasattr(self, '_storage'):
            self._storage = np.empty((32768, 2), np.float32)
        self._write_end = 0
        self.buffer = self._storage[:0]
        self.offset = 0
        self.rate_error, self.confidence = rate_error, confidence
        self.pending = None
        self._tried = 0
        self._since_lock = 0
        self.search_after = SYNC_LEN
        self.acquire_ms = 0.
        self.acquisition_path = 'edge'
        self.predicted = None
        self.waiting = False
        self.misses = 0
        self.fallback_every = 1
        self.edge_hits = 0
        self.correlation_hits = 0
        self.locked_packets = 0
        self._aspect_code = 0
        self._pulse_channels = [None, None]
        self._timing_channel = None
        self._diagnostic_counts = {'attempted': 0, 'verified': 0, 'pictures': 0,
                                   'lost': 0, 'single_input': 0, 'no_preamble': 0}
        self._last_decode = None

    def diagnostic_state(self):
        """Bounded snapshot, including failures which never produce a picture.

        The caller decides when/where to report it; no I/O runs in the decoder.
        Counts and sample positions are since the last input discontinuity.
        """
        return {'receiver_resets': self.resets, 'sample_offset': self.offset,
                'buffer_samples': len(self.buffer), 'wire': self.layout.name,
                'detected': self.detected, 'pending': self.pending is not None,
                'predicted_sample': self.predicted, 'confidence': self.confidence,
                'timing_scale': 1+self.rate_error,
                'timing_channel': self._timing_channel,
                'pulse_channels': list(self._pulse_channels),
                'since_verified_header': self._since_lock,
                'decode_counts': dict(self._diagnostic_counts),
                'last_decode': self._last_decode}

    def _acquire(self):
        if self.pulse_only:
            return self._acquire_pulses()
        window = self.buffer[:max(1024, 2*self.keep)]
        self.waiting = False
        if not np.any(window):
            return None
        if self.predicted is not None and self.confidence:
            at = self.predicted - self.offset
            need = at + len(PREAMBLE)*(1+self.rate_error) + 24
            if 0 <= at and need <= len(self.buffer):
                fit = _fit_preamble(self.buffer, at, 1+self.rate_error, reach=0,
                                    iterations=2, fast=self.fast)
                if fit[2] >= max(self.threshold, .5):
                    self.acquisition_path = 'coast'
                    self.locked_packets += 1
                    self.misses = 0
                    self.fallback_every = 1
                    return fit
            elif 0 <= at:
                self.waiting = True
                return None
            self.predicted = None
        for channel in range(window.shape[1]):
            measured = measure_speed(window[:, channel], self.min_scale, self.max_scale)
            if measured is None:
                continue
            position, scale, _ = measured
            fit = _fit_preamble(window, position, scale, reach=0,
                                iterations=3, fast=self.fast)
            if fit[2] >= self.threshold:
                self.acquisition_path = 'edge'
                self.edge_hits += 1
                self.misses = 0
                self.fallback_every = 1
                return fit
        self.misses += 1
        if self.misses % self.fallback_every:
            return None
        found = self._correlate(window)
        if found is None:
            self.fallback_every = min(self.fallback_every*2, 64)
        return found

    def _acquire_pulses(self):
        self.waiting = False
        if self.predicted is not None and self.confidence:
            at = self.predicted-self.offset
            left = max(0, int(at)-24)
            right = int(at+len(PREAMBLE)*(1+self.rate_error))+24
            if at >= 0 and right > len(self.buffer):
                self.waiting = True
                return None
            if at >= 0:
                measured = self._closest_pulses(self.buffer[left:right])
                if measured is not None:
                    position, scale, confidence = measured
                    self.acquisition_path = 'coast'
                    self.locked_packets += 1
                    return position+left, scale, confidence
            self.predicted = None
        window = self.buffer[:max(1024, 2*self.keep)]
        measured = self._closest_pulses(window)
        if measured is not None:
            self.acquisition_path = 'edge'
            self.edge_hits += 1
        return measured

    def _closest_pulses(self, window):
        """Measure the preamble on both legs; trust the one nearer the expected speed.

        The preamble rides both legs identically, so they agree unless one is
        damaged. Taking the first leg that measured anything let a pitch-
        shifted left leg -- whose edges read ~6% fast -- time the whole packet,
        and the clean right leg was never consulted. Expected speed is the
        tracked rate once locked, nominal before. Legs that agree resolve to
        leg 0, as before.
        """
        expected = self.clock*(1+self.rate_error) if self.confidence else self.clock
        best = None
        if self.diagnostics:
            self._pulse_channels = [None, None]
            self._timing_channel = None
        for channel in range(2):
            measured = measure_pulses(window[:, channel], self.min_scale, self.max_scale)
            if measured is None:
                continue
            if self.diagnostics:
                self._pulse_channels[channel] = {
                    'at_in_window': float(measured[0]), 'scale': float(measured[1]),
                    'confidence': float(measured[2])}
            miss = abs(measured[1]/expected - 1)
            if best is None or miss < best[0] - 1e-4:
                best = (miss, measured)
                if self.diagnostics:
                    self._timing_channel = channel
        return None if best is None else best[1]

    def _correlate(self, window):
        scales = np.geomspace(self.min_scale, self.max_scale,
                              max(2, int(np.ceil(np.log(self.max_scale/self.min_scale)/.09))+1))
        best = None
        for scale in scales:
            template = _scaled_preamble(float(scale))
            energy = float(np.einsum('i,i->', template, template, optimize=False))
            for channel in range(window.shape[1]):
                scores = preamble_correlation(window[:, channel], None, template, energy)
                if not len(scores):
                    continue
                at = int(np.argmax(scores))
                if best is None or scores[at] > best[0]:
                    best = (float(scores[at]), at, float(scale))
        if best is None or best[0] < max(self.threshold, .5):
            return None
        fit = _fit_preamble(window, best[1], best[2], reach=.05, iterations=6,
                            fast=self.fast)
        if fit[2] >= self.threshold:
            self.acquisition_path = 'correlation'
            self.correlation_hits += 1
            self.misses = 0
            self.fallback_every = 1
            return fit
        return None

    def feed(self, block):
        block = np.asarray(block, np.float32)
        if block.ndim != 2 or block.shape[1] != 2 or not np.isfinite(block).all():
            raise ValueError('Receiver requires finite stereo samples')

        out = []
        step = min(1024, self.keep) if self.pulse_only else 256
        for start in range(0, len(block), step):
            self._append(block[start:start + step])
            out.extend(self._drain())
        return out

    def _append(self, block):
        count = len(self.buffer)
        need = count + len(block)
        if self._write_end + len(block) > len(self._storage):
            if need > len(self._storage):
                storage = np.empty((max(need, 2*len(self._storage)), 2), np.float32)
                storage[:count] = self.buffer
                self._storage = storage
            else:
                self._storage[:count] = self.buffer.copy()
            self._write_end = count
        begin = self._write_end - count
        self._storage[self._write_end:self._write_end+len(block)] = block
        self._write_end += len(block)
        self.buffer = self._storage[begin:self._write_end]

    def flush(self):
        return self._drain(final=True)

    def _demodulate(self, begin, scale, layout, coder, coders):
        """One decode attempt at a given layout. Returns (result, taps)."""
        result, taps = self._demodulate_once(begin, scale, layout, coder,
                                             coders)
        if abs(scale-1.0) > SCALE_SANITY:
            alt, alt_taps = self._demodulate_once(begin, 1.0, layout, coder,
                                                  coders, fast=False)
            if _recovery_quality(alt) > _recovery_quality(result):
                return alt, alt_taps
        return result, taps

    def _demodulate_once(self, begin, scale, layout, coder, coders,
                         fast=True):
        """A single demodulate + decode. The unity-scale retry above always
        goes through the resampling path: _identify sized the buffer for the
        measured scale, so the fast slice is not guaranteed room."""
        taps = 0
        if (fast and scale == 1 and begin == int(begin) and begin >= 0
                and int(begin)+layout.packet <= len(self.buffer)):
            packet = self.buffer[int(begin):int(begin)+layout.packet]
            body = packet[SYNC_LEN:].reshape(layout.symbols, SYMBOL, 2)[:, CP-4:CP-4+N]
        else:
            highest = layout.top_bin/N/scale
            taps = 32 if highest > .44 else 8
            body = _sample_at(self.buffer, begin+_body_walk(layout)*scale, taps=taps)
            body = body.reshape(layout.symbols, N, 2)

        return decode_packet(None, layout, coder, body=body, coders=coders,
                              header_tolerance=self.header_tolerance,
                              diagnostics=self.diagnostics), taps

    def _identify(self, begin, scale, final=False):
        if not self.candidates or self.detected is not None:
            result, taps = self._demodulate(begin, scale, self.layout,
                                            self.coder, self.coders)
            return result, taps, False
        room = len(self.buffer)-begin
        for index in range(self._tried, len(self.candidates)):
            layout, coder, coders = self.candidates[index]
            if (layout.packet-1)*scale+1 > room:
                break
            self._tried = index+1
            result, taps = self._demodulate(begin, scale, layout, coder, coders)
            if result.identity == 'verified_header':
                self.layout, self.coder, self.coders = layout, coder, coders
                self.detected = layout.name
                return result, taps, False
        if self._tried < len(self.candidates) and not final:
            return None, 0, True
        result, taps = self._demodulate(begin, scale, self.layout,
                                        self.coder, self.coders)
        return result, taps, False

    def _drain(self, final=False):
        import time
        out = []
        while True:
            if self.pending is None:
                if len(self.buffer) < self.search_after and not final:
                    break
                if len(self.buffer) < len(PREAMBLE)//2:
                    break
                started = time.perf_counter()
                self.waiting = False
                acquired = self._acquire()
                self.acquire_ms += (time.perf_counter()-started)*1000
                if acquired is None:
                    if self.waiting:
                        break
                    if self.diagnostics:
                        self._diagnostic_counts['no_preamble'] += 1
                    self.search_after = len(self.buffer)+256
                    if len(self.buffer) > self.keep:
                        self._drop(len(self.buffer)-self.keep)
                    break
                at, scale, score = acquired
                self.search_after = SYNC_LEN
                self.pending = (at-16*scale, scale, score)
                self._tried = 0
            begin, scale, score = self.pending
            searching = self.candidates and self.detected is None
            want = (self.candidates[0][0].packet if searching
                    else self.layout.packet)
            end = begin + (want-1)*scale + 1
            if len(self.buffer) < end - (2 if final else 0):
                break
            started = time.perf_counter()
            result, taps, wait = self._identify(begin, scale, final)
            if wait:
                break
            if self.detected is not None:
                if result.identity == 'verified_header':
                    self._since_lock = 0
                else:
                    self._since_lock += 1
                    if self._since_lock >= RELOCK_AFTER:
                        self.layout, self.coder, self.coders = self._start
                        self.detected = None
                        self._tried = 0
                        self._since_lock = 0
            result.rate_error = scale-1
            # A damaged header cannot change display geometry. Keep the last
            # verified preset until a new header arrives (native after reset).
            from .aspect import ASPECT_RATIOS
            if result.identity == 'verified_header':
                self._aspect_code = result.extra['aspect_code']
            else:
                result.extra.update(aspect_code=self._aspect_code,
                                    aspect=ASPECT_RATIOS[self._aspect_code],
                                    aspect_inferred=True)
            result.rate_confidence = score
            result.extra.update(sync_score=score, at=self.offset+begin,
                                input_path='v3', acquisition_path=self.acquisition_path,
                                timing_method='pulse' if self.pulse_only else 'waveform',
                                timing_scale=scale, complete=True,
                                top_bin_cycles_per_sample=self.layout.top_bin/N/scale,
                                playback_speed=self.clock/scale,
                                preset=self.layout.name, resample_taps=taps,
                                decode_ms=(time.perf_counter()-started)*1000,
                                acquire_ms=self.acquire_ms)
            if self.input_rate is not None:
                result.extra.update(
                    input_rate=self.input_rate,
                    input_nyquist_hz=self.input_rate/2,
                    input_top_hz=result.extra['top_bin_cycles_per_sample']*self.input_rate,
                    frame_seconds=self.layout.frame*scale/self.input_rate)
            result.extra['receive_cpu_ms'] = result.extra['decode_ms']+self.acquire_ms
            if self.diagnostics:
                counts = self._diagnostic_counts
                counts['attempted'] += 1
                counts['verified'] += int(result.identity == 'verified_header')
                counts['pictures'] += int(result.values is not None)
                counts['lost'] += int(result.values is None)
                counts['single_input'] += int('single_input' in result.extra)
                self._last_decode = {
                    'at': float(self.offset+begin), 'frame': result.absolute,
                    'status': result.status, 'identity': result.identity,
                    'single_input': result.extra.get('single_input'),
                    'pilot_error': result.pilot_error, 'coverage': result.coverage,
                    'attempts': result.extra.get('decode_attempts', [])}
                result.extra.update(timing_channel=self._timing_channel,
                                    pulse_channels=list(self._pulse_channels))
            self.pending = None
            if result.values is not None:
                self.rate_error, self.confidence = scale-1, score
                self.predicted = (self.offset + begin
                                  + (self.layout.frame + 16)*scale)
                out.append(result)
                self.acquire_ms = 0.
                self._drop(max(1, int(np.floor(begin+self.layout.packet*scale))))
            else:
                self.confidence = 0.
                self.rate_error = 0.
                self.predicted = None
                self._drop(max(1, int(begin+16*scale+1)))
        return out

    def _drop(self, n):
        n = max(1, min(int(n), len(self.buffer)))
        self.buffer = self.buffer[n:]
        self.offset += n
        self.search_after = max(SYNC_LEN, self.search_after-n)
