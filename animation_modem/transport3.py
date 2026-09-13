"""V3 transport: countable preamble, edge-capture acquisition, corrected level.

The wire body is byte-identical to v2 -- same OFDM grid, same carriers, same
16-byte CRC header, same source coding. Only the preamble and the way the
receiver finds it change, so `core.decode_packet` demodulates a v3 packet
unmodified once the timing is known.

Three changes, in order of how much they matter:

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

Correlation has not gone away. Edge timing is a per-sample decision and has no
processing gain, so it degrades below roughly 10 dB SNR while a 256-sample
correlation is still locking at -5 dB. Acquisition therefore gates on edges and
falls back to correlation, which is only reached on genuinely marginal signal
rather than on every unlocked buffer.
"""
import numpy as np
from functools import lru_cache
from scipy.signal import correlate

from . import core as v2
from .core import (RATE, N, CP, SYMBOL, SYNC_LEN, GUARD, HEADER_GAIN,
                         IMAGE_GAIN, HEADER_SLOTS, HEADER_FORMAT, HEADER_BYTES,
                         Layout, PRESETS, SourceCoder, Decoded, coefficient_slots,
                         phases, pack_header, pack_folders, decode_packet,
                         default_allocation, resample_packet, _sample_at,
                         _body_walk, _decode_tables, FOLDER_LIMIT)

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


def measure_speed(samples, min_speed=.25, max_speed=2.0):
    """Playback speed and preamble position from edge intervals alone.

    Intervals cluster at SHORT and LONG = 2*SHORT. Clustering on the observed
    ratio rather than thresholding against fixed constants is what lifts the
    capture range past the +/-15% an LTC reader's fixed windows allow.

    Returns (position, scale, confidence) or None. `scale` is received duration
    over nominal, matching transport2's convention.
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
        if not min_speed*.98 <= 1/scale <= max_speed*1.02:
            continue
        if abs(unit/SHORT - scale) > .15*scale:
            continue                       # unit and span must agree
        position = float(at[start]) - NOMINAL_EDGES[0]*scale
        agreement = 1.0 - min(1.0, float(np.std(short))/max(unit, 1e-9))
        return position, scale, agreement
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
           headroom=.95):
    """One v3 packet: v2's body, a countable stereo preamble, correct level."""
    if len(values) != coder.count or not np.isfinite(values).all():
        raise ValueError('Expected one finite value per source coefficient')
    if len(values) > layout.capacity:
        raise ValueError(f'{len(values)} values exceed capacity {layout.capacity}')
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
                      magic=MAGIC if layout.progressive else b'V2')
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

    room = layout.image_symbols*len(data)*4
    sent = np.zeros(room)
    sent[coefficient_slots(layout, tuple(coder.shapes))] = coder.forward(values)
    block = sent.reshape(layout.image_symbols, len(data), 2, 2)
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
    """Edge-gated acquisition with a correlation fallback.

    Drop-in for core.Receiver: same constructor, same `feed`/`flush`
    contract, same Decoded results. `acquisition_path` in `extra` reports which
    route found each packet, so a deployment can see how often it is paying for
    the fallback.
    """

    def __init__(self, layout, coder, threshold=.4, rate_window=None,
                 min_speed=.25, max_speed=2.0, recovery=False, fast=True):
        if not (0 < min_speed <= 1 <= max_speed and min_speed >= .25 and max_speed <= 2):
            raise ValueError('Supported search range: .25 <= min_speed <= 1 <= max_speed <= 2')
        if coder.count > layout.capacity:
            raise ValueError('Source coder exceeds layout capacity')
        self.layout, self.coder, self.threshold = layout, coder, threshold
        self.recovery = bool(recovery)
        self.fast = bool(fast)
        self.min_speed, self.max_speed = min_speed, max_speed
        self.keep = int(np.ceil(len(PREAMBLE)*1.2/min_speed)) + 64
        self.reset()

    def reset(self, preserve_timing=False):
        rate = self.rate if preserve_timing else 0.
        confidence = self.confidence if preserve_timing else 0.
        if not hasattr(self, '_storage'):
            self._storage = np.empty((32768, 2), np.float32)
        self._write_end = 0
        self.buffer = self._storage[:0]
        self.offset = 0
        self.rate, self.confidence = rate, confidence
        self.pending = None
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
        window = self.buffer[:max(1024, 2*self.keep)]
        self.waiting = False
        if not np.any(window):
            return None
        # 1. Coast. The transmitter emits contiguous frames, so once locked the
        #    next preamble is one frame away and only needs verifying.
        if self.predicted is not None and self.confidence:
            at = self.predicted - self.offset
            need = at + len(PREAMBLE)*(1+self.rate) + 24
            if 0 <= at and need <= len(self.buffer):
                fit = _fit_preamble(self.buffer, at, 1+self.rate, reach=0,
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
            measured = measure_speed(window[:, channel], self.min_speed, self.max_speed)
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

    def _correlate(self, window):
        scales = np.geomspace(1/self.max_speed, 1/self.min_speed,
                              max(2, int(np.ceil(np.log(self.max_speed/self.min_speed)/.09))+1))
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
        for start in range(0, len(block), 256):
            self._append(block[start:start+256])
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
            begin, scale, score = self.pending
            end = begin + (self.layout.packet-1)*scale + 1
            if len(self.buffer) < end - (2 if final else 0):
                break
            started = time.perf_counter()
            if scale == 1 and begin == int(begin) and begin >= 0:
                packet = self.buffer[int(begin):int(begin)+self.layout.packet]
                body = packet[SYNC_LEN:].reshape(self.layout.symbols, SYMBOL, 2)[:, CP-4:CP-4+N]
            else:
                body = _sample_at(self.buffer, begin+_body_walk(self.layout)*scale)
                body = body.reshape(self.layout.symbols, N, 2)
            result = decode_packet(None, self.layout, self.coder, body=body)
            result.rate_error = scale-1
            result.rate_confidence = score
            result.extra.update(sync_score=score, at=self.offset+begin,
                                input_path='v3', acquisition_path=self.acquisition_path,
                                playback_speed=1/scale, complete=True,
                                decode_ms=(time.perf_counter()-started)*1000,
                                acquire_ms=self.acquire_ms)
            result.extra['receive_cpu_ms'] = result.extra['decode_ms']+self.acquire_ms
            self.pending = None
            if result.values is not None:
                self.rate, self.confidence = scale-1, score
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
                self.rate = 0.
                self.predicted = None
                self._drop(max(1, int(begin+16*scale+1)))
        return out

    def _drop(self, n):
        n = max(1, min(int(n), len(self.buffer)))
        self.buffer = self.buffer[n:]
        self.offset += n
        self.search_after = max(SYNC_LEN, self.search_after-n)
