"""Bounded handoff from the audio reader to the packet decoder."""
from collections import deque
import threading

import numpy as np


class AudioBuffer:
    def __init__(self, capacity, batch):
        if not 0 < batch <= capacity:
            raise ValueError('Audio buffer must hold at least one decode batch')
        self.capacity, self.batch = int(capacity), int(batch)
        self._condition = threading.Condition()
        self._chunks = deque()
        self._samples = 0
        self._gap = False
        self.closed = False
        self.dropped_samples = 0
        self.input_overflows = 0

    def put(self, audio, overflowed=False):
        audio = np.asarray(audio, np.float32).copy()
        with self._condition:
            if self.closed:
                return
            if overflowed:
                self.input_overflows += 1
            if overflowed:
                self.dropped_samples += self._samples
                self._chunks.clear()
                self._samples = 0
                self._gap = True
            if len(audio) > self.capacity:
                self.dropped_samples += len(audio)-self.capacity
                audio = audio[-self.capacity:]
                self._gap = True
            self._chunks.append(audio)
            self._samples += len(audio)
            # Keep the newest contiguous tail instead of flushing the queue
            # and waiting for it to fill again after every scheduling stall.
            excess = max(0, self._samples-self.capacity)
            self.dropped_samples += excess
            if excess:
                self._gap = True
            while excess:
                oldest = self._chunks.popleft()
                take = min(excess, len(oldest))
                if take < len(oldest):
                    self._chunks.appendleft(oldest[take:])
                self._samples -= take
                excess -= take
            self._condition.notify()

    def take(self, timeout=.1):
        """Return an available batch and its discontinuity flag, or None."""
        with self._condition:
            ready = self._condition.wait_for(
                lambda: self.closed or self._samples >= self.batch, timeout)
            if not ready or not self._samples:
                return None
            audio = np.concatenate(self._chunks)
            gap = self._gap
            self._chunks.clear()
            self._samples = 0
            self._gap = False
            return audio, gap

    def close(self):
        with self._condition:
            self.closed = True
            self._condition.notify_all()

    def configure(self, capacity, batch):
        """Follow the detected packet duration, including playback speed."""
        with self._condition:
            self.capacity = max(1, int(capacity))
            self.batch = max(1, min(int(batch), self.capacity))
            excess = max(0, self._samples-self.capacity)
            self.dropped_samples += excess
            if excess:
                self._gap = True
            while excess:
                oldest = self._chunks.popleft()
                take = min(excess, len(oldest))
                if take < len(oldest):
                    self._chunks.appendleft(oldest[take:])
                self._samples -= take
                excess -= take
            self._condition.notify_all()
