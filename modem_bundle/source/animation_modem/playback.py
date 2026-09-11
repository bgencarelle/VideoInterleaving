"""Packet-safe audio output using scope's ready / single-pending-frame handoff.

No image selection, producer thread, or frame FIFO lives here. XY beam guards
and hold-last-sample behavior from scope_out must not modify modem waveforms.
"""
import argparse
import json
import sys
import time
from dataclasses import dataclass
import threading
import numpy as np
from .audio_common import sounddevice, route
from .transport2 import PRESETS, RATE

_DEFAULT = PRESETS['wide']   # layouts pass their own geometry


def latency(value):
    if value in ('low', 'high'):
        return value
    try:
        number = float(value)
        if not np.isfinite(number) or number <= 0:
            raise ValueError()
        return number
    except ValueError:
        raise argparse.ArgumentTypeError('Latency must be low, high, or positive seconds')


@dataclass(frozen=True)
class Slot:
    start_time: float  # PortAudio stream clock, seconds
    target_time_ns: int  # Chrony-disciplined wall clock, millisecond resolution


class PacketOutput:
    def __init__(self, device=None, channels=(0, 1), requested_latency='low',
                 frame=_DEFAULT.frame, packet=_DEFAULT.packet):
        # v2 layouts have their own geometry -- wide is 3344 samples, not v1's
        # 3200 -- so the packet size cannot be a module constant any more.
        self.frame = int(frame)
        self.packet = int(packet)
        if not 0 < self.packet <= self.frame:
            raise ValueError('Packet must be positive and fit inside the frame')
        self.channels = channels
        self.pending = None
        self.pending_start = None
        self.next_start = None
        self.scheduled = False
        self.deadline_misses = 0
        self.current = None
        self.position = 0
        self.underflows = 0
        self.starvations = 0
        self.completed = 0
        self.finishing = False
        self.error = None
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.sd = sounddevice()
        self.stream = self.sd.OutputStream(
            samplerate=RATE, channels=max(channels)+1, dtype='float32',
            device=device, blocksize=256, latency=requested_latency,
            callback=self._callback, finished_callback=self.done.set)

    def __enter__(self):
        # Timed output starts at reservation; legacy output starts on first submit.
        print(json.dumps({'output_latency_ms': self.stream.latency*1000,
                          'pending_frames': 1, 'block_samples': 256}), file=sys.stderr)
        self.started = False
        return self

    def check(self):
        if self.error is not None:
            raise RuntimeError('Modem audio callback failed') from self.error
        if self.started and (self.done.is_set() or not self.stream.active):
            raise RuntimeError('Modem audio stopped unexpectedly')

    def ready(self):
        self.check()
        with self.lock:
            return self.pending is None

    def reserve(self, prepare_ms=10, receive_margin_ms=15):
        """Reserve a send boundary; map its completion to the shared wall clock.

        stream.time and outputBufferDacTime share PortAudio's clock. Bridge it
        to wall time on every packet so Chrony slew and device drift do not
        accumulate into an independent animation clock.
        """
        if not self.ready():return None
        self.scheduled = True
        if not self.started:
            self.stream.start()
            self.started = True
        before = time.time_ns()
        now = self.stream.time
        after = time.time_ns()
        wall_ns = (before + after)//2
        earliest = now + self.stream.latency + 256/RATE + prepare_ms/1000
        with self.lock:
            start = max(self.next_start or earliest, earliest)
        target = wall_ns + round((start-now + self.packet/RATE + receive_margin_ms/1000)*1e9)
        # Select the image at the exact timestamp the header can represent.
        return Slot(start, (target//1_000_000)*1_000_000)

    def submit(self, audio, slot=None):
        if np.shape(audio) != (self.frame, 2) or not np.isfinite(audio).all():
            raise ValueError('Expected one finite stereo modem frame')
        prepared = route(audio, self.channels)
        self.check()
        if slot is not None and self.stream.time + self.stream.latency >= slot.start_time:
            self.deadline_misses += 1
            return False  # Re-evaluate the shared clock; do not send a stale image.
        with self.lock:
            if self.pending is not None or self.finishing:
                raise RuntimeError('Wait for ready() before submitting another packet')
            self.pending = prepared
            self.pending_start = slot.start_time if slot else None
            if slot:self.next_start = slot.start_time + self.frame/RATE
        if not self.started:
            self.stream.start()
            self.started = True
        return True

    def _callback(self, outdata, frames, timing, status):
        outdata.fill(0)
        if status.output_underflow:
            self.underflows += 1
        eof = False
        try:
            with self.lock:
                written = 0
                while written < frames:
                    if self.current is None:
                        if self.pending is not None and self.pending_start is not None:
                            here = timing.outputBufferDacTime + written/RATE
                            gap = round((self.pending_start-here)*RATE)
                            if gap > 0:
                                written += min(gap, frames-written)
                                continue  # Intentional silence before the reserved send time.
                            if gap < -48:  # 1 ms tolerance matches header timestamp precision.
                                self.pending = None
                                self.pending_start = None
                                self.deadline_misses += 1
                                continue  # Device missed the deadline; never present stale metadata.
                        self.current, self.pending = self.pending, None
                        self.pending_start = None
                        if self.current is None:
                            eof = self.finishing
                            if not eof and not self.scheduled:self.starvations += 1
                            break
                    take = min(frames-written, len(self.current)-self.position)
                    outdata[written:written+take] = self.current[self.position:self.position+take]
                    written += take
                    self.position += take
                    if self.position == len(self.current):
                        self.current = None
                        self.position = 0
                        self.completed += 1
        except Exception as exc:
            self.error = exc
            raise self.sd.CallbackAbort
        if eof:
            # PortAudio drains the final output buffer before finished_callback.
            raise self.sd.CallbackStop

    def finish(self):
        if not self.started:return
        with self.lock:self.finishing = True
        while not self.done.wait(.05):
            if not self.stream.active:
                raise RuntimeError('Modem audio stopped before draining')
        if self.error is not None:
            raise RuntimeError('Modem audio callback failed') from self.error

    def __exit__(self, *args):
        try:
            self.stream.stop()
        finally:
            self.stream.close()
            print(json.dumps({'output_underflows': self.underflows,
                              'producer_starvations': self.starvations,
                              'packets_completed': self.completed,
                              'send_deadline_misses': self.deadline_misses}), file=sys.stderr)
