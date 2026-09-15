"""V3 transport: countable preamble, edge-capture acquisition, corrected level.

The wire body is byte-identical to v2 -- same OFDM grid, same carriers, same
16-byte CRC header, same source coding. Only the preamble and the way the
receiver finds it change, so `core.decode_packet` demodulates a v3 packet
unmodified once the timing is known.

Four changes, in order of how much they matter:

0. Because timing comes from counting edges, the receiver never needs to know
   its own sample rate, and so never asks a device to run at one. A capture at
   44.1, 96 or 192 kHz differs from a 48 kHz one by a constant factor on every
   interval, which is indistinguishable from a deck running slow or fast -- and
   the decoder already corrects for that. `input_rate`, when a caller passes the
   rate an open device reported, is used only to separate the two for reporting
   and to keep the speed window centred on the capture clock. It is not an
   input to demodulation, and there is no default: unknown stays unknown.

1. The preamble is an LTC-style biphase-mark word instead of a random-phase
   noise burst. Edge intervals take exactly two values, so playback speed comes
   from `measured interval / nominal interval` -- the same thing an LTC reader
   does with a timer capture on each edge. No correlation bank, no scale
   search. v2 spends ~5.5 ms of CPU per 1152-sample buffer correlating against
   48 scaled templates; edge capture answers the same question in ~26 us and is
   ten times more precise.

2. The preamble is on BOTH channels. v2 left channel 1 silent for the whole
   256-sample burst, which is a third of all the dead samples in the frame and
   throws away half the available preamble energy.

3. `encode` normalises up as well as down. v2 only ever attenuated (`if peak >
   .95`), so a typical packet left ~2.8 dB of headroom unused on top of a
   payload already sitting ~21 dB below full scale.

The default receiver reads pulse transitions only. The legacy waveform path
remains available with pulse_only=False for comparison. In that legacy path,
correlation has not gone away. Edge timing is a per-sample decision and has no
processing gain, so it degrades below roughly 10 dB SNR while a 256-sample
correlation is still locking at -5 dB. Acquisition therefore gates on edges and
falls back to correlation, which is only reached on genuinely marginal signal
rather than on every unlocked buffer.
"""
import numpy as np
from functools import lru_cache
from scipy.signal import correlate

from . import core as v2
from .core import (REFERENCE_RATE, N, CP, SYMBOL, SYNC_LEN, GUARD, HEADER_GAIN,
                         IMAGE_GAIN, HEADER_SLOTS, HEADER_FORMAT, HEADER_BYTES,
                         Layout, PRESETS, SourceCoder, Decoded, coefficient_slots,
                         phases, pack_header, pack_folders, decode_packet,
                         default_allocation, resample_packet, _sample_at,
                         _body_walk, _decode_tables, FOLDER_LIMIT,
                         band_limited, emit_length, emit_ratio,
                         PROFILE_ALIASES, PROFILE_CODES, profile_code, profile_name)

# --------------------------------------------------------------------------
# Preamble
# --------------------------------------------------------------------------

HALF = 8                   # samples per half-bit: '1' -> 3 kHz, '0' -> 1.5 kHz
PREAMBLE_BITS = (0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0)
# Chosen by exhaustive search over balanced 16-bit words for the lowest
# autocorrelation sidelobe: 0.312 against 0.531 for an arbitrary word. v2's
# noise burst reaches 0.080, so the correlation fallback is more sidelobe-prone
# here; the edge gate and the header CRC are what disambiguate.
PREAMBLE_AMPLITUDE = .55
# Below v2's .65. The preamble held the packet peak in 73 of 100 measured
# frames, which meant any level set to keep it from clipping buried the picture
# 7.7 dB further down than it needed to be.

SHORT, LONG = HALF, 2*HALF
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


# v3-only layouts. These live here rather than in core.PRESETS because
# orthogonal training needs v3's encoder: a v2 transmitter cannot produce them,
# and transport2's own tests round-trip every entry in PRESETS.
V3_PRESETS = {
    # Recommended default: same band and cadence as 'wide', but both channels
    # drive both training symbols. Same peak, twice the training energy, and
    # measured +1.2 dB clean / +2.1 dB at tape noise / +4.1 dB at heavy noise,
    # with the 0.25-1.6x playback range unchanged.
    'wide-v3': Layout(top_bin=54, image_symbols=15, name='wide-v3',
                      progressive=True, orthogonal_training=True),
    # Trades the top of the speed range for frame rate: the header goes out as
    # two symbols instead of four, 14.35 -> 15.71 fps, but 1.6x playback stops
    # decoding. For a stable transport, not a drifting deck.
    'wide-v3-fast': Layout(top_bin=54, image_symbols=15, name='wide-v3-fast',
                           progressive=True, orthogonal_training=True,
                           header_split=True),
    # Cassette band with the same training improvement.
    'tape-v3': Layout(top_bin=27, image_symbols=35, name='tape-v3',
                      progressive=True, orthogonal_training=True),
    # Lean chroma (profile 'color-lean', 2160 coefficients) fits 11 symbols.
    'lean-v3': Layout(top_bin=54, image_symbols=11, name='lean-v3',
                      progressive=True, orthogonal_training=True,
                      spread_carriers=True),
    # Sub-15 kHz. top_bin 39 = 14625 Hz, which clears 15 kHz with margin and
    # fits inside anything that passes FM-radio bandwidth. 35 data carriers, so
    # the full 2880-value colour picture needs 21 image symbols.
    'mid-v3': Layout(top_bin=39, image_symbols=21, name='mid-v3',
                     progressive=True, orthogonal_training=True,
                     spread_carriers=True),
    # Same band, header halved, for frame rate on a stable transport.
    'mid-v3-fast': Layout(top_bin=39, image_symbols=21, name='mid-v3-fast',
                          progressive=True, orthogonal_training=True,
                          spread_carriers=True, header_split=True),
    # lean-v3's geometry with the header left at the BOTTOM of the band.
    #
    # For tape. spread_carriers moves the header to the middle -- 6750-14250 Hz
    # on this layout -- which is chosen to survive a channel that rolls off at
    # its own bottom edge. A cassette deck has the opposite problem: it rolls
    # off at the top, and the middle is exactly where it stops. Measured with
    # -45 dBFS hiss, lean-v3 loses every header below a 9 kHz low-pass and all
    # of them by 8 kHz, while still decoding all four pictures -- so the symptom
    # is a black screen, not a broken one, because only a verified header is
    # held for display. Here the header sits at 375-8625 Hz and survives 8 kHz
    # plus flutter, 4/4, at the full 17.34 fps.
    #
    # What it gives up is narrow: spread only buys a 300 Hz high-pass, and both
    # placements lose every header by 375 Hz.
    'lean-v3-tape': Layout(top_bin=54, image_symbols=11, name='lean-v3-tape',
                           progressive=True, orthogonal_training=True),
    # lean-v3 with the header symbols' idle carriers carrying image. Identical
    # frame, identical band, identical frame rate: 2768 samples and 17.34 fps
    # at 48 kHz. The header only ever used 20 of the 50 data carriers, so the
    # other 30 were silence in all four header symbols -- 480 slots, 2200 ->
    # 2680, +21.8%, enough for the full 2880-value 'color' picture to shrink
    # far less. Costs nothing in level: the preamble holds the packet peak.
    'lean-v3-dense': Layout(top_bin=54, image_symbols=11, name='lean-v3-dense',
                            progressive=True, orthogonal_training=True,
                            spread_carriers=True, dense_header=True),
    # Sub-14 kHz, for a channel with a ceiling rather than a roll-off.
    #
    # mid-v3 stops at 14625 Hz, which clears 15 kHz but not 14. top_bin 34 is
    # 12750 Hz, and the 1250 Hz of headroom under a 14 kHz cut is the point:
    # paired with core.bound_emission the whole emission lands with 99.99% of
    # its energy at 13973 Hz, at a reconstruction error of 0.0062. Crowding the
    # cut instead -- top_bin 37, 13875 Hz, 125 Hz of guard -- costs 0.036 for
    # the same ceiling, because the filter's transition band eats the top
    # carriers. The guard is cheaper than the carriers it buys back.
    #
    # dense_header pays for the bins top_bin gives up: 2280 -> 2680, which is
    # enough to carry the full 2880-value 'color' picture at 38x46 rather than
    # dropping to a lean profile. 11.41 fps, the same as mid-v3.
    'mid-14k': Layout(top_bin=34, image_symbols=21, name='mid-14k',
                      progressive=True, orthogonal_training=True,
                      spread_carriers=True, dense_header=True),
    # Same band, lean-v3's frame rate. 11 image symbols instead of 21, so the
    # picture drops to 30x36 -- but 17.34 fps under 14 kHz, which the 21-symbol
    # version cannot do. Measured, that budget costs about 1 dB on graphic
    # content and GAINS about 1 dB on smooth content and hard edges, because
    # fewer coefficients means more power in each and less aliasing.
    'lean-14k': Layout(top_bin=34, image_symbols=11, name='lean-14k',
                       progressive=True, orthogonal_training=True,
                       spread_carriers=True, dense_header=True),
}
ALL_PRESETS = {**PRESETS, **V3_PRESETS}

PREAMBLE = _biphase()
PREAMBLE_ENERGY = float(np.einsum('i,i->', PREAMBLE, PREAMBLE, optimize=False))
NOMINAL_EDGES = np.flatnonzero(np.diff(np.signbit(PREAMBLE)))
NOMINAL_SPAN = float(NOMINAL_EDGES[-1] - NOMINAL_EDGES[0])
MAGIC = b'V3'


def edge_intervals(samples, hysteresis=EDGE_HYSTERESIS*PREAMBLE_AMPLITUDE):
    """Schmitt-triggered transition positions. The ICR capture, vectorised.

    Returns positions only; intervals are `np.diff` of them. A hysteresis band
    rather than a bare sign test is what keeps hiss from manufacturing edges
    near the zero crossings.
    """
    state = np.where(samples > hysteresis, 1,
                     np.where(samples < -hysteresis, -1, 0))
    live = np.flatnonzero(state)
    if len(live) < 2:
        return np.empty(0, int)
    changes = np.flatnonzero(np.diff(state[live]))
    return live[changes + 1]


NOMINAL_GAPS = np.diff(NOMINAL_EDGES).astype(float)
MIN_RUN = len(NOMINAL_GAPS) - 4        # tolerate a few corrupted edges


def _runs(gaps, tolerance=.28):
    """Maximal spans of intervals that all sit near one unit or twice it.

    The preamble is followed immediately by OFDM, whose zero crossings are
    dense and irregular. Taking statistics over a whole buffer lets those
    crossings drag the estimate -- measured directly, two body edges moved the
    apparent span from 224 to 266 samples. Looking for a *run* of consistent
    intervals instead ignores everything that is not shaped like the word.
    """
    out = []
    start = 0
    n = len(gaps)
    while start < n:
        # The unit is the SHORT interval, but a word can open with a run of
        # long ones -- this one starts with five. Estimating from the first few
        # gaps therefore locks onto 2*unit and rejects the real word at the
        # first short interval. Take the minimum over a full word's worth.
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
    """Timing scale and preamble position from edge intervals alone.

    Intervals cluster at SHORT and LONG = 2*SHORT. Clustering on the observed
    ratio rather than thresholding against fixed constants is what lifts the
    capture range past the +/-15% an LTC reader's fixed windows allow.

    Returns (position, scale, confidence) or None. `scale` is received duration
    over nominal, in samples of whatever clock captured them. Bounds are given
    in scale rather than playback speed because this function never learns the
    sample rate: a 96 kHz capture of a normal-speed signal and a half-speed
    48 kHz one are the same measurement, and only a caller that knows the rate
    can tell them apart.
    """
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
        # A real biphase word has long == 2*short. Anything else is not our
        # preamble, and rejecting it costs one comparison instead of a fit.
        if not 1.7 <= float(np.mean(long))/max(unit, 1e-9) <= 2.3:
            continue
        # Total span is a far better estimator than any single interval:
        # per-edge sample quantisation averages out over the whole word.
        span = float(at[end] - at[start])
        scale = span/NOMINAL_SPAN
        if not .98*min_scale <= scale <= 1.02*max_scale:
            continue
        if abs(unit/SHORT - scale) > .15*scale:
            continue                       # unit and span must agree
        position = float(at[start]) - NOMINAL_EDGES[0]*scale
        agreement = 1.0 - min(1.0, float(np.std(short))/max(unit, 1e-9))
        return position, scale, agreement
    return None


def measure_pulses(samples, min_scale=.5, max_scale=4.0):
    """Read the biphase word and fit its transition times, without a speed sweep.

    Schmitt edges reject chatter. Interpolated zero crossings remove integer
    sample quantisation before fitting elapsed time against the known pulse
    count. The full short/long word must match; payload crossings are ignored.

    This is the whole reason the decoder needs no sample rate: an LTC reader
    counts its own clock between transitions and never asks what that clock is.
    Bounds are in scale, for the reason given on `measure_speed`.
    """
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
    # Allow sample interpolation error near the endpoints of the scale range.
    valid = (scales >= .98*min_scale) & (scales <= 1.02*max_scale)
    expected = NOMINAL_GAPS[None, :]*scales[:, None]
    valid &= np.all(np.abs(np.diff(words, axis=1)-expected) <=
                    np.maximum(.8, .28*expected), axis=1)
    nominal = NOMINAL_EDGES.astype(float)+.5
    centered = nominal-nominal.mean()
    for word in words[valid]:
        # Linear zero-crossing interpolation is biased when a speed change
        # puts a pulse edge between samples. Refine those SAME observed edges
        # on a bandlimited interpolant; this does not search speeds/templates.
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
        confidence = max(0., 1-residual/(SHORT*scale))
        if confidence >= .7:
            return position, scale, confidence
    return None


def preamble_correlation(x, limit=None, template=None, energy=None):
    """Normalised correlation against the preamble. The low-SNR fallback."""
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
    """Refine position and dilation against the known preamble.

    Same two-parameter projected fit as core._fit_sync. Edge capture
    already lands inside 0.5%, so `reach` is small and the loop is short --
    this is a refinement, not a search.
    """
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
# Encode
# --------------------------------------------------------------------------

def encode(values, layout, coder, absolute, index, count, stamp_ms=0, flags=0,
           headroom=.95, profile=0):
    """One v3 packet: v2's body, a countable stereo preamble, correct level.

    `profile` is the picture geometry's wire code, from `profile_code`. It goes
    in the two spare bits of the header's top_bin byte, so a receiver holding
    the matching coders reconstructs whatever is sent without being told
    separately. Declaring it wrong is worse than not declaring it: the receiver
    will believe the header over its own assumption.
    """
    # A coder that transmits fewer coefficients than it consumes (a truncating
    # coder) has a different input length from its slot count, so validate
    # against the input size it declares.
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
        # v2 sends [1,0] then [0,1]: half of every training symbol is silence.
        # Rows [1,1] and [1,-1] fill both channels for the same 288 samples and
        # the same per-channel peak, doubling the energy behind the estimate.
        #
        # Channel 1 is pre-rotated onto channel 0's phase so that the scramble
        # applied below leaves the training matrix orthogonal; without this the
        # two independent phases can very nearly cancel and the 2x2 solve
        # becomes singular on some carriers.
        tone = phases(layout)
        lock0 = tone[0, :, 0]*np.conj(tone[0, :, 1])
        lock1 = tone[1, :, 0]*np.conj(tone[1, :, 1])
        grid[0, :, 0], grid[0, :, 1] = 1, lock0
        grid[1, :, 0], grid[1, :, 1] = 1, -lock1
    else:
        grid[0, :, 0] = 1
        grid[1, :, 1] = 1

    raw = pack_header(flags, layout.top_bin, absolute, index, count, stamp_ms,
                      magic=layout.wire_magic,
                      profile=profile)
    bits = np.unpackbits(np.frombuffer(raw, np.uint8)).reshape(HEADER_SLOTS, 2)
    qpsk = ((bits[:, 0]*2.-1) + 1j*(bits[:, 1]*2.-1))/np.sqrt(2)
    if layout.header_split:
        # Different halves on each channel rather than the same bits twice.
        # v2 spent 4 symbols -- 48.6% of the packet's fixed overhead -- on a
        # header it then sent identically on both channels. Splitting halves
        # the symbol count; the cost is the ~3 dB diversity combine, which the
        # CRC still catches when it fails.
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
    sent[coefficient_slots(layout, tuple(coder.shapes))] = coder.forward(values)
    head = layout.header_capacity
    if head:
        # The header occupies header_width carriers; the rest of every header
        # symbol was silence. Same symbols, different carriers, so the header
        # is untouched -- and the preamble still holds the packet peak, so this
        # costs nothing in level. Written first in the flat slot array.
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
    # Both channels. v2 wrote the preamble to channel 0 only.
    out[16:16+len(PREAMBLE), 0] = PREAMBLE
    out[16:16+len(PREAMBLE), 1] = PREAMBLE
    out[SYNC_LEN:layout.packet] = wave
    # Normalise in BOTH directions. The training symbols carry the same scale,
    # so the receiver's channel estimate undoes it either way; refusing to
    # scale up simply discarded headroom.
    peak = float(np.max(np.abs(out)))
    if peak > 0:
        out *= headroom/peak
    return out


# --------------------------------------------------------------------------
# Receive
# --------------------------------------------------------------------------

class Receiver:
    """Pulse-timed acquisition; legacy correlation is explicitly opt-in.

    Drop-in for core.Receiver: same constructor, same `feed`/`flush`
    contract, same Decoded results. `acquisition_path` in `extra` reports which
    route found each packet, so a deployment can see how often it is paying for
    the fallback.
    """

    def __init__(self, layout, coder, threshold=.4, rate_window=None,
                 min_speed=.25, max_speed=2.0, recovery=False, fast=True,
                 pulse_only=True, input_rate=None, coders=None, candidates=None):
        """`input_rate` is reported by the open device; it is never requested.

        Decoding itself does not use it. Timing comes from preamble edges, in
        samples, so the receiver works at any capture rate the hardware happens
        to be set to. What the rate buys is interpretation: without it a scale
        of 2.0 could be a 96 kHz capture or a half-speed deck, and the two are
        indistinguishable from the samples. Given the rate, the acquisition
        window is placed around the capture clock -- so min_speed/max_speed keep
        meaning transport speed at 44.1, 96 or 192 kHz instead of quietly
        turning into a rate limit -- and the reported speed, frame duration and
        carrier frequencies come out in real units. Left None, the receiver
        assumes nothing and reports the sample-domain numbers only.
        """
        if not (0 < min_speed <= 1 <= max_speed and min_speed >= .25 and max_speed <= 2):
            raise ValueError('Supported speed range: .25 <= min_speed <= 1 <= max_speed <= 2')
        if input_rate is not None and not (np.isfinite(input_rate) and input_rate > 0):
            raise ValueError('Input rate must be a positive number of hertz, or None')
        if coder.count > layout.capacity:
            raise ValueError('Source coder exceeds layout capacity')
        self.layout, self.coder, self.threshold = layout, coder, threshold
        # Profile code -> coder. With it, the picture geometry follows what the
        # transmitter declares in each header rather than a launch argument.
        self.coders = dict(coders) if coders else None
        # (layout, coder, coders) triples to identify the sender's preset by
        # trying them. Preset cannot be signalled the way profile is: the
        # header carries top_bin, but you need the layout to know where the
        # header IS, so reading it first is circular. The preamble is the one
        # part that does not depend on the layout, so acquisition still works
        # -- and from there the CRC decides, at 1.4 ms an attempt, once.
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
        # Capture clock over reference clock. One full sample period of the
        # wire geometry occupies this many samples of the device's own clock,
        # so it is exactly the factor that separates rate from speed.
        self.clock = 1. if input_rate is None else float(input_rate)/REFERENCE_RATE
        self.min_scale = self.clock/max_speed
        self.max_scale = self.clock/min_speed
        self.keep = int(np.ceil(len(PREAMBLE)*1.2*self.max_scale)) + 64
        self.reset()

    def reset(self, preserve_timing=False):
        rate_error = self.rate_error if preserve_timing else 0.
        confidence = self.confidence if preserve_timing else 0.
        if not hasattr(self, '_storage'):
            self._storage = np.empty((32768, 2), np.float32)
        self._write_end = 0
        self.buffer = self._storage[:0]
        self.offset = 0
        # Timing scale minus one, NOT a sample rate. `input_rate` is the
        # device's clock; this is how far the wire ran from nominal on it.
        self.rate_error, self.confidence = rate_error, confidence
        self.pending = None
        # Candidates already ruled out at the current pending position. Without
        # it, every feed() that arrives while waiting for a longer candidate
        # re-runs the short ones that already failed -- measured at 21 decode
        # attempts for six frames where 13 is the whole search.
        self._tried = 0
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

    # -- acquisition ------------------------------------------------------

    def _acquire(self):
        if self.pulse_only:
            return self._acquire_pulses()
        window = self.buffer[:max(1024, 2*self.keep)]
        self.waiting = False
        if not np.any(window):
            return None
        # 1. Coast. The transmitter emits contiguous frames, so once locked the
        #    next preamble is one frame away and only needs verifying.
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
                # Not arrived yet. This is NOT a failed search: returning a
                # plain None here would let the caller inflate search_after and
                # drop the buffer the prediction points into, which silently
                # disables coasting altogether.
                self.waiting = True
                return None
            self.predicted = None
        # 2. Edge capture. This is the common path and the cheap one.
        for channel in range(window.shape[1]):
            measured = measure_speed(window[:, channel], self.min_scale, self.max_scale)
            if measured is None:
                continue
            position, scale, _ = measured
            # Edge capture lands within ~0.3%, well inside the Newton loop's
            # basin, so no scale sweep is needed -- that sweep is 7 scales x 2
            # channels of correlation and was the bulk of acquisition cost.
            fit = _fit_preamble(window, position, scale, reach=0,
                                iterations=3, fast=self.fast)
            if fit[2] >= self.threshold:
                self.acquisition_path = 'edge'
                self.edge_hits += 1
                self.misses = 0
                self.fallback_every = 1
                return fit
        # 3. Correlation fallback, for signal too weak to count edges on.
        #    This is the only expensive route left, and by construction it only
        #    matters when the input is marginal -- which is exactly when it is
        #    least urgent. Running it on every failed buffer is what kept v2
        #    pinned at ~275% of a core on silence, so it is decimated instead.
        #    Any edge or coast hit resets the counter, so a real signal never
        #    waits more than one buffer for it.
        self.misses += 1
        if self.misses % self.fallback_every:
            return None
        found = self._correlate(window)
        if found is None:
            # Widen the gap between fallback sweeps, to a ceiling, so a dead
            # input costs almost nothing while still being watched.
            self.fallback_every = min(self.fallback_every*2, 64)
        return found

    def _acquire_pulses(self):
        """Measure every preamble; prediction only limits which samples to read."""
        self.waiting = False
        if self.predicted is not None and self.confidence:
            at = self.predicted-self.offset
            left = max(0, int(at)-24)
            right = int(at+len(PREAMBLE)*(1+self.rate_error))+24
            if at >= 0 and right > len(self.buffer):
                self.waiting = True
                return None
            if at >= 0:
                for channel in range(2):
                    measured = measure_pulses(self.buffer[left:right, channel],
                                              self.min_scale, self.max_scale)
                    if measured is not None:
                        position, scale, confidence = measured
                        self.acquisition_path = 'coast'
                        self.locked_packets += 1
                        return position+left, scale, confidence
            self.predicted = None
        window = self.buffer[:max(1024, 2*self.keep)]
        for channel in range(2):
            measured = measure_pulses(window[:, channel], self.min_scale, self.max_scale)
            if measured is not None:
                self.acquisition_path = 'edge'
                self.edge_hits += 1
                return measured
        return None

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

    # -- buffering --------------------------------------------------------

    def feed(self, block):
        block = np.asarray(block, np.float32)
        if block.ndim != 2 or block.shape[1] != 2 or not np.isfinite(block).all():
            raise ValueError('Receiver requires finite stereo samples')
        out = []
        # Consume queued audio in larger pieces. A partial packet remains
        # pending until its complete body is available; no decode per sample.
        step = min(1024, self.keep) if self.pulse_only else 256
        for start in range(0, len(block), step):
            self._append(block[start:start+step])
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
        taps = 0
        if scale == 1 and begin == int(begin) and begin >= 0:
            packet = self.buffer[int(begin):int(begin)+layout.packet]
            body = packet[SYNC_LEN:].reshape(layout.symbols, SYMBOL, 2)[:, CP-4:CP-4+N]
        else:
            # Faster playback moves surviving carriers toward input Nyquist,
            # where the short interpolator has phase-dependent attenuation.
            # Use a longer kernel only when it is needed.
            highest = layout.top_bin/N/scale
            taps = 32 if highest > .44 else 8
            body = _sample_at(self.buffer, begin+_body_walk(layout)*scale, taps=taps)
            body = body.reshape(layout.symbols, N, 2)
        return decode_packet(None, layout, coder, body=body, coders=coders), taps

    def _identify(self, begin, scale, final=False):
        """Decode, working out which preset is on the wire if asked to.

        Tried shortest packet first, and only the ones the buffer can already
        hold. The header CRC decides: a wrong layout reshapes the body wrongly,
        so the 32-bit check fails on top of a magic and a top_bin that also
        have to agree. Once one verifies it is adopted and every later packet
        takes the single-layout path, so the search is paid once per lock, not
        per frame.

        Returns (result, taps, wait). `wait` asks the caller for more samples:
        nothing has verified yet but a longer candidate has not been reachable.
        Waiting for the longest candidate up front instead would be simpler and
        wrong -- tape-v3's packet is 6192 samples, so any recording shorter than
        that would decode nothing at all, however short its own packets are.
        """
        if not self.candidates or self.detected is not None:
            result, taps = self._demodulate(begin, scale, self.layout,
                                            self.coder, self.coders)
            return result, taps, False
        room = len(self.buffer)-begin
        for index in range(self._tried, len(self.candidates)):
            layout, coder, coders = self.candidates[index]
            if (layout.packet-1)*scale+1 > room:
                break                     # sorted by length; the rest are longer
            self._tried = index+1         # ruled out; do not retry on more audio
            result, taps = self._demodulate(begin, scale, layout, coder, coders)
            if result.identity == 'verified_header':
                self.layout, self.coder, self.coders = layout, coder, coders
                self.detected = layout.name
                return result, taps, False
        if self._tried < len(self.candidates) and not final:
            return None, 0, True
        # Every candidate that could be tried has failed. Fall back to the
        # layout this receiver was built with, so an unidentifiable packet
        # fails exactly the way it does without a candidate list at all.
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
                        break             # prediction still good; just early
                    self.search_after = len(self.buffer)+256
                    if len(self.buffer) > self.keep:
                        self._drop(len(self.buffer)-self.keep)
                    break
                at, scale, score = acquired
                self.search_after = SYNC_LEN
                self.pending = (at-16*scale, scale, score)
                self._tried = 0
            begin, scale, score = self.pending
            # While the preset is unknown, the SHORTEST candidate is the one
            # that decides when decoding can start; _identify asks for more
            # samples if it needs them for a longer one.
            searching = self.candidates and self.detected is None
            want = (self.candidates[0][0].packet if searching
                    else self.layout.packet)
            end = begin + (want-1)*scale + 1
            if len(self.buffer) < end - (2 if final else 0):
                break
            started = time.perf_counter()
            result, taps, wait = self._identify(begin, scale, final)
            if wait:
                break                     # a longer candidate needs more audio
            result.rate_error = scale-1
            result.rate_confidence = score
            result.extra.update(sync_score=score, at=self.offset+begin,
                                input_path='v3', acquisition_path=self.acquisition_path,
                                timing_method='pulse' if self.pulse_only else 'waveform',
                                # Sample-domain truth, always available: how
                                # much longer the packet is than the geometry
                                # says, and where the top carrier sits in
                                # cycles per captured sample.
                                timing_scale=scale, complete=True,
                                top_bin_cycles_per_sample=self.layout.top_bin/N/scale,
                                # Real time, against the reference geometry and
                                # the capture clock. Identical to 1/scale when
                                # the device happens to run at the reference
                                # rate, which is why this was safe to hardcode
                                # for as long as the rate was being demanded.
                                playback_speed=self.clock/scale,
                                preset=self.layout.name, resample_taps=taps,
                                decode_ms=(time.perf_counter()-started)*1000,
                                acquire_ms=self.acquire_ms)
            if self.input_rate is not None:
                # Hertz and seconds exist only once a device has told us what
                # it is running at. Nothing here asked it to run at anything.
                result.extra.update(
                    input_rate=self.input_rate,
                    input_nyquist_hz=self.input_rate/2,
                    input_top_hz=result.extra['top_bin_cycles_per_sample']*self.input_rate,
                    frame_seconds=self.layout.frame*scale/self.input_rate)
            result.extra['receive_cpu_ms'] = result.extra['decode_ms']+self.acquire_ms
            self.pending = None
            if result.values is not None:
                self.rate_error, self.confidence = scale-1, score
                # Contiguous frames: predict rather than search next time.
                # Point at the next PREAMBLE, not the next frame start. The
                # preamble sits 16 samples into the frame, and _fit_preamble
                # only searches +/-12, so predicting the frame start puts the
                # target outside the window and silently disables coasting.
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
