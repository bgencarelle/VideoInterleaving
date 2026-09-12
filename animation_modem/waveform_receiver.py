"""Native-sample, frame-local receiver for the existing transport2 waveform.

The ADC stream is treated as the band-limited waveform.  The device's nominal
sample-rate label is never used as the modem clock: each packet derives its
sample scale from the known L-only preamble and R-only training endpoint.
"""
import time
import numpy as np
from scipy.signal import lfilter

from . import transport2 as v
from .progressive import _Packet
from .reference_receiver import _reference_table


class Receiver:
    native_waveform = True

    def __init__(self, layout, coder):
        if coder.count > layout.capacity:
            raise ValueError('Source coder exceeds layout capacity')
        self.layout, self.coder = layout, coder
        self.keep = max(4 * layout.frame, 4096)
        self.reset()

    def reset(self, preserve_timing=False):
        self.buffer = np.empty((0, 2), np.float32)
        self.offset = 0
        self.scan = 0
        self.edge = None
        self.peak = 0.
        self.energy_state = np.zeros((7, 2))
        self.energy_history = np.empty((0, 2), np.float64)
        self.pending = None
        self.packet = None
        self.acquire_ms = 0.
        self.decode_ms = 0.

    def _append(self, block):
        self.buffer = np.concatenate((self.buffer, np.asarray(block, np.float32)))

    def _drop(self, count):
        count = min(max(0, int(count)), len(self.buffer))
        if count:
            self.buffer = self.buffer[count:].copy()
            self.energy_history = self.energy_history[count:]
            self.offset += count
        self.scan = max(0, self.scan-count)
        if self.edge is not None:
            self.edge -= count
            if self.edge < 0:
                self.edge = None

    def _measure(self, left, right):
        onset = self.buffer[left:min(right, left+128), 0]**2
        # An ADC reconstruction filter can ring before the real onset. Use the
        # local preamble peak, not the first (possibly tiny) detected sample.
        strong = np.flatnonzero(onset > max(float(np.max(onset)), 1e-20)*.1)
        if not len(strong):
            return None
        left += int(strong[0])
        scale = (right-left)/(v.SYNC_LEN+v.SYMBOL-16)
        if not .5 <= scale <= 8:
            return None
        q, design, projection = _reference_table()
        observed = v._sample_at(self.buffer, left+q*scale)[:, 0]
        level = max(float(np.sqrt(np.mean(observed**2))), 1e-12)
        observed /= level
        coefficients = np.einsum('ij,j->i', projection, observed, optimize=False)
        gain, delay, dilation = coefficients
        if abs(gain) < 1e-5:
            return None
        delay, dilation = delay/gain, dilation/gain
        if abs(delay) > 2*scale or abs(dilation) > .03:
            return None
        predicted = np.einsum('ni,i->n', design, coefficients, optimize=False)
        residual = float(np.sum((observed-predicted)**2) /
                         max(np.sum(observed**2), 1e-20))
        if residual > .45:
            # A high-order audio-rate conversion can change the short sync
            # shape while preserving its timing. Keep the geometric clock when
            # the normalized known waveform still correlates; reject silence
            # and unrelated stereo activity.
            reference = v._sample_at(v.SYNC[:, None], q)
            corr = float(np.dot(observed, reference.ravel()) /
                         max(np.linalg.norm(observed)*np.linalg.norm(reference), 1e-20))
            if corr < .35:
                return None
            residual = min(residual, 1-corr)
        corrected = scale/(1+dilation)
        at = left-(delay-q.mean()*dilation)*corrected
        return (at, corrected, max(0., 1-residual)) if .49 <= corrected <= 8.04 else None

    def _acquire(self):
        start = self.scan
        if start >= len(self.buffer):
            return None
        power = self.buffer[start:].astype(float)**2
        energy, self.energy_state = lfilter(np.full(8, 1/8), [1.], power,
                                            axis=0, zi=self.energy_state)
        self.energy_history = np.concatenate((self.energy_history, energy))
        self.scan = len(self.buffer)
        left, right = self.energy_history.T
        floor = max(1e-12, float(np.percentile(np.minimum(left, right), 20))*16)
        if self.edge is None:
            hits = np.flatnonzero((left[start:] > floor) &
                                  (right[start:] < np.maximum(floor, left[start:]*.02)))
            if not len(hits):
                return None
            at = start+int(hits[0])
            lo = max(0, at-7)
            onset = np.flatnonzero(self.buffer[lo:at+1, 0]**2 > floor)
            self.edge = lo+int(onset[0]) if len(onset) else at
            self.peak = float(left[at])
        search = max(self.edge+200, start)
        ages = np.arange(len(left))-self.edge
        limit = 4*self.layout.frame
        hits = np.flatnonzero((ages[search:] >= 200) & (ages[search:] <= limit) &
                              (right[search:] > np.maximum(floor, self.peak*1e-4)) &
                              (left[search:] < right[search:]*.15))
        if not len(hits):
            if len(ages) and ages[-1] > limit:
                self.edge = None
            return None
        hit = search+int(hits[0])
        at = hit
        lo = max(self.edge, at-7)
        onset = np.flatnonzero(self.buffer[lo:at+1, 1]**2 > max(floor, self.peak*1e-4))
        right_edge = lo+int(onset[0]) if len(onset) else at
        result = self._measure(self.edge, right_edge)
        self.edge = None
        return result

    def _drain(self, final=False):
        out = []
        if self.pending is None:
            started = time.perf_counter()
            acquired = self._acquire()
            self.acquire_ms += (time.perf_counter()-started)*1000
            if acquired is None:
                if len(self.buffer) > self.keep:
                    self._drop(len(self.buffer)-self.keep)
                return out
            at, scale, score = acquired
            self.pending = (at-16*scale, scale, score)
            self.packet = _Packet(self.layout, self.coder)
        begin, scale, score = self.pending
        first = self.packet.symbol
        ends = begin+(v.SYNC_LEN+np.arange(first, self.layout.symbols)*v.SYMBOL+
                      v.CP-4+v.N-1)*scale+1
        available = np.flatnonzero(ends <= len(self.buffer) if final else ends+8 <= len(self.buffer))
        last = first+(int(available[-1])+1 if len(available) else 0)
        if last == first:
            return out
        started = time.perf_counter()
        walk = v._body_walk(self.layout).reshape(self.layout.symbols, v.N)[first:last].ravel()
        positions = begin+walk*scale
        if positions[-1] >= len(self.buffer):
            return out
        body = v._sample_at(self.buffer, positions).reshape(last-first, v.N, 2)
        self.packet.push(body)
        self.decode_ms += (time.perf_counter()-started)*1000
        if last < self.layout.symbols:
            return out
        started = time.perf_counter()
        result = self.packet.snapshot(True)
        self.decode_ms += (time.perf_counter()-started)*1000
        elapsed = self.acquire_ms
        self.acquire_ms = 0.
        if result is not None:
            result.rate_error = scale-1
            result.rate_confidence = score
            result.extra.update(input_path='native_waveform', packet_id=self.offset+begin,
                at=self.offset+begin, playback_speed=1/scale,
                packet_duration_samples=float(self.layout.packet*scale),
                packet_age_samples=float(len(self.buffer)-begin),
                decode_ms=self.decode_ms, acquire_ms=elapsed,
                receive_cpu_ms=elapsed+self.decode_ms,
                sample_scale=scale)
            out.append(result)
        self.pending = self.packet = None
        self.decode_ms = 0.
        self._drop(max(1, int(np.floor(begin+self.layout.packet*scale))))
        # Acquisition skipped the payload. Rebuild energy for the remaining
        # samples exactly once; old history no longer indexes this buffer.
        self.scan = 0
        self.energy_history = np.empty((0, 2), np.float64)
        self.energy_state.fill(0)
        return out

    def feed(self, block):
        block = np.asarray(block, np.float32)
        if block.ndim != 2 or block.shape[1] != 2 or not np.isfinite(block).all():
            raise ValueError('Receiver requires finite stereo samples')
        self._append(block)
        out = []
        while True:
            got = self._drain()
            if not got:
                break
            out.extend(got)
        return out

    def flush(self):
        return self._drain(final=True)
