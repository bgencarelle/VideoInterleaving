"""Experimental independent delivery-clock and two-pilot measurement.

Reference-only waveform: L carries amplitude timing marks, R two continuous
pilots. This occupies a stereo pair and is NOT the production image format.
The tracker uses fixed-size windows, cached oscillator tables and two complex
projections per hop after acquisition. No image decode or candidate replay.
"""
from collections import deque
from functools import lru_cache
import numpy as np
from scipy.signal import butter, sosfilt, find_peaks

PILOTS = (1300., 3300.)
MARK_HZ = 15.
WINDOW = 1024
HOP = 256


def reference(seconds=4., rate=48000, speed=1., cents=0., wow_percent=0.,
              wow_hz=4., offset_hz=0.):
    """Synthesize independent pitch and timing controls, with continuous phase.

    Controls model ideal frequency/time changes, not a phase-vocoder effect.
    `speed` and wow affect the marker clock AND frequencies; cents affects only
    carrier frequencies. offset_hz adds the same Hz shift to both pilots.
    """
    params = (seconds, rate, speed, cents, wow_percent, wow_hz, offset_hz)
    if not np.isfinite(params).all() or seconds <= 0 or rate < 16000:
        raise ValueError('Use finite values, positive duration and rate >= 16000')
    if (not .25 <= speed <= 2 or abs(cents) > 100 or not 0 <= wow_percent <= 5
            or wow_hz <= 0 or abs(offset_hz) > 100):
        raise ValueError('Use speed .25..2, cents +/-100, wow 0..5%, positive wow Hz, offset +/-100 Hz')
    time = np.arange(round(seconds*rate))/rate
    source_time = speed*(time + (wow_percent/100)/(2*np.pi*wow_hz)*
                         (1-np.cos(2*np.pi*wow_hz*time)))
    pitched_time = source_time*2**(cents/1200)
    amplitude = np.where((source_time*MARK_HZ) % 1 < .25, .70, .14)
    left = amplitude*np.sin(2*np.pi*2000*pitched_time)
    right = sum(.23*np.sin(2*np.pi*(f*pitched_time+offset_hz*time)) for f in PILOTS)
    return np.stack((left, right), axis=1).astype(np.float32)


@lru_cache(maxsize=32)
def _oscillators(bins):
    n = np.arange(WINDOW)
    table = np.exp(-2j*np.pi*n/WINDOW)
    return (np.hanning(WINDOW)[None, :]*table[(np.asarray(bins)[:, None]*n) % WINDOW],
            table[(np.asarray(bins)*HOP) % WINDOW])


class PilotClock:
    """Incremental timing/pilot observer. All sample positions are input-clock based."""
    def __init__(self, rate=48000):
        if not np.isfinite(rate) or rate < 16000:
            raise ValueError('Input rate must be >= 16000 Hz')
        self.rate = float(rate)
        self.filter = butter(2, 200., fs=rate, output='sos')
        self.window = np.hanning(WINDOW)
        self.reset()

    def reset(self):
        self.pending = np.empty((0, 2), np.float32)
        self.samples = 0
        self.audio = np.zeros(WINDOW, np.float32)
        self.filter_state = np.zeros((len(self.filter), 2))
        self.peak = 0.
        self.above = False
        self.last_edge = None
        self.bins = None
        self.previous = None
        self.frequencies = deque(maxlen=1024)
        self.intervals = deque(maxlen=64)
        self.acquisitions = 0

    def _acquire(self):
        spectrum = np.abs(np.fft.rfft(self.audio*self.window))
        peaks, _ = find_peaks(spectrum)
        peaks = [int(k) for k in peaks if 200 < k*self.rate/WINDOW < 7500
                 and spectrum[k] > .1*max(spectrum.max(), 1e-12)]
        peaks = sorted(peaks, key=lambda k: spectrum[k], reverse=True)[:10]
        best = None
        for first in peaks:
            for second in peaks:
                if second <= first:
                    continue
                f1, f2 = np.array([first, second])*self.rate/WINDOW
                scale = (f2-f1)/(PILOTS[1]-PILOTS[0])
                offset = f1-scale*PILOTS[0]
                if .22 <= scale <= 2.2 and abs(offset) < 120:
                    score = min(spectrum[first], spectrum[second])/(1+abs(offset)/100)
                    if best is None or score > best[0]:
                        best = (score, (first, second))
        if best is not None:
            self.bins = best[1]
            self.previous = None
            self.acquisitions += 1

    def _pilots(self):
        if self.samples < WINDOW:
            return
        if self.bins is None:
            self._acquire()
        if self.bins is None:
            return
        weights, advance = _oscillators(self.bins)
        z = np.einsum('kn,n->k', weights, self.audio, optimize=False)
        amplitudes = 2*np.abs(z)/self.window.sum()
        if np.min(amplitudes) < .015:
            self.bins = None
            self.previous = None
            self.frequencies.clear()
            self.intervals.clear()
            self.last_edge = None
            return
        anchors = np.asarray(self.bins)*self.rate/WINDOW
        if self.previous is not None:
            phase = np.angle(z*self.previous.conj()*advance)
            measured = anchors+phase*self.rate/(2*np.pi*HOP)
            if np.any(abs(measured-anchors) > .75*self.rate/WINDOW):
                self.bins = None
                self.previous = None
                return
            center = self.samples-(WINDOW+HOP)/2
            self.frequencies.append((center, float(measured[0]), float(measured[1])))
        self.previous = z

    def _marks(self, block):
        power, self.filter_state = sosfilt(self.filter, block[:, 0].astype(float)**2,
                                           zi=self.filter_state)
        envelope = np.sqrt(np.maximum(power, 0))
        self.peak = max(float(envelope.max()), self.peak*np.exp(-HOP/self.rate))
        if self.peak < .01:
            self.last_edge = None
            return
        # Hysteresis suppresses ripple; a refractory period rejects extra edges.
        for i, value in enumerate(envelope):
            if self.above:
                if value < .4*self.peak:
                    self.above = False
            elif value > .65*self.peak:
                self.above = True
                edge = self.samples-HOP+i
                if self.last_edge is not None:
                    distance = edge-self.last_edge
                    if distance < self.rate/(MARK_HZ*2.2):
                        continue
                    speed = self.rate/(MARK_HZ*distance)
                    if .22 <= speed <= 2.2:
                        self.intervals.append((self.last_edge, edge, speed))
                self.last_edge = edge

    def _reports(self):
        out = []
        while self.intervals and self.frequencies:
            start, end, speed = self.intervals[0]
            if self.frequencies[-1][0] < end:
                break
            self.intervals.popleft()
            history = np.asarray(self.frequencies)
            if history[0, 0] > start:
                continue
            selected = history[(history[:, 0] > start) & (history[:, 0] < end), 0]
            positions = np.concatenate(([start], selected, [end]))
            # Compare frequency and delivery over the SAME time interval.
            # Otherwise ordinary flutter can look like independent pitch drift.
            frequency = []
            for column in (1, 2):
                f = np.interp(positions, history[:, 0], history[:, column])
                frequency.append(float(np.sum((f[1:]+f[:-1])*.5*np.diff(positions))/(end-start)))
            scale = (frequency[1]-frequency[0])/(PILOTS[1]-PILOTS[0])
            if scale <= 0:
                continue
            offset = frequency[0]-scale*PILOTS[0]
            out.append(dict(delivery_speed=speed, frequency_scale=scale,
                            extra_pitch_cents=1200*np.log2(scale/speed),
                            frequency_offset_hz=offset, pilot_hz=frequency,
                            at_sample=end, acquisitions=self.acquisitions))
        return out

    def feed(self, block):
        block = np.asarray(block, np.float32)
        if block.ndim != 2 or block.shape[1] != 2 or not np.isfinite(block).all():
            raise ValueError('Expected finite stereo samples')
        # Keep working storage independent of the caller's block size.
        out = []
        for start in range(0, len(block), HOP):
            self.pending = np.concatenate((self.pending, block[start:start+HOP]))
            if len(self.pending) < HOP:
                continue
            chunk = self.pending[:HOP]
            self.pending = self.pending[HOP:]
            self.audio[:-HOP] = self.audio[HOP:]
            self.audio[-HOP:] = chunk[:, 1]
            self.samples += HOP
            self._pilots()
            self._marks(chunk)
            out.extend(self._reports())
        return out
