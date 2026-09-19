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
from .transport3 import WIRE
from .core import band_limited, emit_length

_DEFAULT = WIRE   # layouts pass their own geometry


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
                 frame=_DEFAULT.frame, packet=_DEFAULT.packet, blocksize=256):
        # The wire carries its own geometry -- 3200 samples -- so the packet
        # size is taken from it instead of a module constant.
        self.frame = int(frame)
        self.packet = int(packet)
        if not 0 < self.packet <= self.frame:
            raise ValueError('Packet must be positive and fit inside the frame')
        self.channels = channels
        self.pending = None
        self.pending_start = None
        self.pending_delay = None
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
        # No forced sample rate - device runs at its native rate.
        # blocksize: choose so frame divides evenly at device rate.
        # For WIRE_HD at 96kHz: frame=3488*2=6976 samples, blocksize=64 -> 109 blocks
        self.stream = self.sd.OutputStream(
            channels=max(channels)+1, dtype='float32',
            device=device, blocksize=blocksize, latency=requested_latency,
            callback=self._callback, finished_callback=self.done.set)
        # Whatever the device reported once it was open. Everything below that
        # converts samples to seconds uses this, never a constant.
        self.rate = float(self.stream.samplerate)
        # Callers hand over reference-geometry packets and this adapts them to
        # the device, because this is the only object that knows what the
        # device turned out to be. Above the reference rate that means
        # resampling so the carriers stay at 375-20250 Hz instead of riding the
        # clock up out of the DAC's passband; at or below it, nothing happens.
        self.emit_frame = emit_length(self.frame, self.rate)
        self.emit_packet = emit_length(self.packet, self.rate)
        self.fps = self.rate/self.emit_frame

    def __enter__(self):
        # Timed output starts at reservation; legacy output starts on first submit.
        print(json.dumps({'output_latency_ms': self.stream.latency*1000,
                          'device_rate_hz': self.rate,
                          'frame_rate_fps': round(self.fps, 3),
                          'emit_frame_samples': self.emit_frame,
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
        earliest = now + self.stream.latency + 256/self.rate + prepare_ms/1000
        with self.lock:
            start = max(self.next_start or earliest, earliest)
        target = wall_ns + round((start-now + self.emit_packet/self.rate + receive_margin_ms/1000)*1e9)
        # Select the image at the exact timestamp the header can represent.
        return Slot(start, (target//1_000_000)*1_000_000)

    def submit(self, audio, slot=None):
        """Take a reference-geometry frame; emit one this device can carry."""
        if np.shape(audio) != (self.frame, 2) or not np.isfinite(audio).all():
            raise ValueError('Expected one finite stereo modem frame')
        prepared = route(band_limited(audio, self.rate), self.channels)
        self.check()
        if slot is not None and self.stream.time + self.stream.latency >= slot.start_time:
            self.deadline_misses += 1
            return False  # Re-evaluate the shared clock; do not send a stale image.
        with self.lock:
            if self.pending is not None or self.finishing:
                raise RuntimeError('Wait for ready() before submitting another packet')
            self.pending = prepared
            self.pending_start = slot.start_time if slot else None
            self.pending_delay = None
            if slot:self.next_start = slot.start_time + self.emit_frame/self.rate
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
                if status.output_underflow:
                    # A real gap invalidates the sample-counted reservation.
                    self.pending_delay = None
                written = 0
                while written < frames:
                    if self.current is None:
                        if self.pending is not None and self.pending_start is not None:
                            if self.pending_delay is None:
                                here = timing.outputBufferDacTime + written/self.rate
                                self.pending_delay = round((self.pending_start-here)*self.rate)
                            gap = self.pending_delay
                            if gap > 0:
                                take = min(gap, frames-written)
                                written += take
                                self.pending_delay -= take
                                # Once placed on the output sample timeline, count
                                # samples to the boundary. Re-reading DAC timestamps
                                # each block turns host timestamp jitter into drops.
                                continue  # Intentional silence before the reserved send time.
                            if gap < -.001*self.rate:  # 1 ms, matching header timestamp precision.
                                self.pending = None
                                self.pending_start = None
                                self.pending_delay = None
                                self.deadline_misses += 1
                                continue  # Device missed the deadline; never present stale metadata.
                        self.current, self.pending = self.pending, None
                        self.pending_start = None
                        self.pending_delay = None
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
