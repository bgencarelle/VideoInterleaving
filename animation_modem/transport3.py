"""V3 transport: countable preamble, edge-capture acquisition, corrected level.

The wire body is byte-identical to v2 -- same OFDM grid, same carriers, same
16-byte CRC header, same source coding. Only the preamble and the way the
receiver finds it change, so `core.decode_packet` demodulates a v3 packet
unmodified once the timing is known.
"""
import numpy as np
from functools import lru_cache
from scipy.signal import correlate
from .audio_common import (pcm, pair, device, wav_blocks, wav_rate, wire_notice,
                           InputLevel, sounddevice)
from . import core as v2
from .core import (REFERENCE_RATE, N, CP, SYMBOL, SYNC_LEN, GUARD, HEADER_GAIN,
                         IMAGE_GAIN, HEADER_SLOTS, HEADER_FORMAT, HEADER_BYTES,
                         Layout, PRESETS, SourceCoder, Decoded, coefficient_slots,
                         phases, pack_header, pack_folders, decode_packet,
                         default_allocation, resample_packet, _sample_at,
                         _body_walk, _decode_tables, FOLDER_LIMIT,
                         band_limited, emit_length, emit_ratio,
                         PROFILE_CODES, profile_code, profile_name)

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


V3_PRESETS = {
    'wide-v3': Layout(top_bin=54, image_symbols=15, name='wide-v3',
                      progressive=True, orthogonal_training=True),
    'wide-v3-fast': Layout(top_bin=54, image_symbols=15, name='wide-v3-fast',
                           progressive=True, orthogonal_training=True,
                           header_split=True),
    'tape-v3': Layout(top_bin=27, image_symbols=35, name='tape-v3',
                      progressive=True, orthogonal_training=True),
    'lean-v3': Layout(top_bin=54, image_symbols=11, name='lean-v3',
                      progressive=True, orthogonal_training=True,
                      spread_carriers=True),
    'mid-v3': Layout(top_bin=39, image_symbols=21, name='mid-v3',
                     progressive=True, orthogonal_training=True,
                     spread_carriers=True),
    'mid-v3-fast': Layout(top_bin=39, image_symbols=21, name='mid-v3-fast',
                          progressive=True, orthogonal_training=True,
                          spread_carriers=True, header_split=True),
    'lean-v3-tape': Layout(top_bin=54, image_symbols=11, name='lean-v3-tape',
                           progressive=True, orthogonal_training=True),
    'lean-v3-dense': Layout(top_bin=54, image_symbols=11, name='lean-v3-dense',
                            progressive=True, orthogonal_training=True,
                            spread_carriers=True, dense_header=True),
    'mid-14k': Layout(top_bin=34, image_symbols=21, name='mid-14k',
                      progressive=True, orthogonal_training=True,
                      spread_carriers=True, dense_header=True),
    'lean-14k': Layout(top_bin=34, image_symbols=11, name='lean-14k',
                       progressive=True, orthogonal_training=True,
                       spread_carriers=True, dense_header=True),
    'hires-v3': Layout(top_bin=54, image_symbols=12, name='hires-v3',
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
# Encode
# --------------------------------------------------------------------------

def encode(values, layout, coder, absolute, index, count, stamp_ms=0, flags=0,
           headroom=.95, profile=0):
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

    raw = pack_header(flags, layout.top_bin, absolute, index, count, stamp_ms,
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
    sent[coefficient_slots(layout, tuple(coder.shapes))] = coder.forward(values)
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
    out[16:16+len(PREAMBLE), 0] = PREAMBLE
    out[16:16+len(PREAMBLE), 1] = PREAMBLE
    out[SYNC_LEN:layout.packet] = wave
    peak = float(np.max(np.abs(out)))
    if peak > 0:
        out *= headroom/peak
    return out


# --------------------------------------------------------------------------
# Receive
# --------------------------------------------------------------------------

class Receiver:
    def __init__(self, layout, coder, threshold=.4, rate_window=None,
                 min_speed=.25, max_speed=2.0, recovery=False, fast=True,
                 pulse_only=True, input_rate=None, coders=None, candidates=None):
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

    def feed(self, block):
        block = np.asarray(block, np.float32)
        if block.ndim != 2 or block.shape[1] != 2 or not np.isfinite(block).all():
            raise ValueError('Receiver requires finite stereo samples')
        out = []
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
            highest = layout.top_bin/N/scale
            taps = 32 if highest > .44 else 8
            body = _sample_at(self.buffer, begin+_body_walk(layout)*scale, taps=taps)
            body = body.reshape(layout.symbols, N, 2)

        return decode_packet(None, layout, coder, body=body, coders=coders), taps

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