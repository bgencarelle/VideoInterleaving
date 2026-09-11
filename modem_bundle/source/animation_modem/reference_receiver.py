"""One-pass reference-event receiver for the unchanged transport2 waveform.

No correlation scan, scale bank, iterative timing fit or payload replay. This
experimental front end needs distinguishable L-only / R-only training events;
severe crosstalk, mono collapse and independent pitch processing are not solved.
"""
from functools import lru_cache
import numpy as np
from scipy.signal import lfilter
from . import transport2 as v
from .progressive import Receiver as ProgressiveReceiver


@lru_cache(maxsize=1)
def _reference_table():
    # Single local linearization: gain, fractional delay, dilation. Centering
    # keeps the tiny normal equation well conditioned. Never iterate/replay.
    q = np.arange(8, len(v.SYNC)-8, dtype=float)
    derivative = np.fft.irfft(
        np.fft.rfft(v.SYNC)*(2j*np.pi*np.fft.rfftfreq(len(v.SYNC))),
        n=len(v.SYNC))[8:-8]
    design = np.stack((v.SYNC[8:-8], derivative,
                       derivative*(q-q.mean())), axis=1)
    gram = np.einsum('ni,nj->ij', design, design, optimize=False)
    projection = np.linalg.solve(gram, design.T)
    return q, design, projection


class Receiver(ProgressiveReceiver):
    """Reference events drive one affine sample clock per packet.

    The inherited symbol consumer owns fresh coefficients, calibration and
    paced snapshots. Only storage helpers are used from transport2's receiver;
    its acquisition bank is neither constructed nor called.
    """
    def __init__(self, layout, coder, preview_hz=30):
        if not np.isfinite(preview_hz) or preview_hz <= 0:
            raise ValueError('Positive preview_hz required')
        if coder.count > layout.capacity:
            raise ValueError('Source coder exceeds layout capacity')
        self.layout, self.coder = layout, coder
        self.preview_samples = v.RATE/preview_hz
        self.keep = 2304
        self.recovery, self.fast = False, True
        self.reference_fits = 0
        self.reference_rejections = 0
        self.reset()

    def reset(self, preserve_timing=False):
        super().reset(preserve_timing)
        self._scan = 0
        self._edge = None
        self._peak = 0.
        self._energy_state = np.zeros((7, 2))
        self._noise = 1e-10

    def _drop(self, n):
        super()._drop(n)
        self._scan = max(0, self._scan-n)
        if self._edge is not None:
            self._edge -= n
            if self._edge < 0:
                self._edge = None

    def _measure(self, left, right):
        # Preamble starts at 16. Second training starts at SYNC_LEN+SYMBOL.
        onset = self.buffer[left:min(right, left+128), 0]**2
        strong = np.flatnonzero(onset > self._peak*.1)
        if not len(strong):
            return None
        left += int(strong[0])
        scale = (right-left)/(v.SYNC_LEN+v.SYMBOL-16)
        if not .5 <= scale <= 4:
            return None
        q, design, projection = _reference_table()
        observed = v._sample_at(self.buffer, left+q*scale)[:, 0]
        coefficients = np.einsum('ij,j->i', projection, observed, optimize=False)
        gain, delay, dilation = coefficients
        self.reference_fits += 1
        if abs(gain) < 1e-5:
            return None
        delay, dilation = delay/gain, dilation/gain
        if abs(delay) > 2 or abs(dilation) > .02:
            return None
        predicted = np.einsum('ni,i->n', design, coefficients, optimize=False)
        energy = float(np.sum(observed**2))
        residual = float(np.sum((observed-predicted)**2))/max(energy, 1e-20)
        if residual > .35:
            return None
        corrected_scale = scale/(1+dilation)
        at = left-(delay-q.mean()*dilation)*corrected_scale
        if not .49 <= corrected_scale <= 4.04:
            return None
        return at, corrected_scale, 1-residual

    def _acquire(self):
        # Each incoming energy sample is inspected once. Buffer overlap does
        # not repeat event detection; no work is deferred in a candidate queue.
        start = self._scan
        if start >= len(self.buffer):
            return None
        power = self.buffer[start:].astype(float)**2
        energy, self._energy_state = lfilter(
            np.full(8, 1/8), [1.], power, axis=0, zi=self._energy_state)
        self._scan = len(self.buffer)
        for j, (left, right) in enumerate(energy):
            at = start+j
            floor = max(1e-9, self._noise*16)
            if self._edge is None:
                if left > floor and right < max(floor, left*.02):
                    # Backtrack only the fixed envelope support, not a search
                    # window. Locate the first energetic sample in this edge.
                    lo = max(0, at-7)
                    hits = np.flatnonzero(self.buffer[lo:at+1, 0]**2 > floor)
                    self._edge = lo+int(hits[0]) if len(hits) else at
                    self._peak = left
                else:
                    self._noise += .002*(min(left, right)-self._noise)
                continue
            self._peak = max(self._peak, left)
            age = at-self._edge
            if age > 1720:
                self._edge = None
                continue
            # R-only second training is the independent duration endpoint.
            threshold = max(floor, self._peak*1e-4)
            if age >= 200 and right > threshold and left < right*.15:
                lo = max(self._edge, at-7)
                hits = np.flatnonzero(self.buffer[lo:at+1, 1]**2 > threshold)
                edge = lo+int(hits[0]) if len(hits) else at
                result = self._measure(self._edge, edge)
                self._edge = None
                if result is not None:
                    return result
                self.reference_rejections += 1
        return None

    def _drain(self, final=False):
        results = super()._drain(final)
        if self.pending is not None:
            # Payload is consumed by the symbol path, never by the event path.
            self._scan = len(self.buffer)
            self._edge = None
            self._energy_state.fill(0)
        for result in results:
            result.extra.update(input_path='reference_events',
                                reference_fits=self.reference_fits,
                                reference_rejections=self.reference_rejections)
        return results
