"""Runtime routing and packet-aligned audio passthrough for the V7 receiver."""
from collections import deque
import threading
import time

import numpy as np

from tools.v7_device_recovery import device_identity, resolve_device_index

DEFAULT_AUDIO_VOLUME = 1.0  # VLC's default 100% volume (unity gain).


def _checked_audio_volume(value):
    volume = float(value)
    if not np.isfinite(volume) or not 0.0 <= volume <= 1.0:
        raise ValueError('Passthrough volume must be between 0 and 1.')
    return volume


class ReceiverRuntimeOptions:
    """Thread-safe live choices shared by the receiver GUI and decoder."""

    def __init__(self, audio_output_device=None, audio_muted=False,
                 audio_output_identity=None, audio_input_identity=None,
                 freewheel_seconds=2.0, show_sync_warning=True,
                 audio_volume=DEFAULT_AUDIO_VOLUME,
                 audio_diagnostics=False):
        self._values = {
            'audio_output_device': audio_output_device,
            'audio_output_identity': audio_output_identity,
            'audio_input_identity': audio_input_identity,
            'audio_muted': bool(audio_muted),
            'audio_volume': _checked_audio_volume(audio_volume),
            'freewheel_seconds': max(0.0, float(freewheel_seconds)),
            'show_sync_warning': bool(show_sync_warning),
            'audio_diagnostics': bool(audio_diagnostics),
        }
        self._lock = threading.Lock()

    def update(self, **values):
        with self._lock:
            for key in self._values:
                if key not in values:
                    continue
                value = values[key]
                if key == 'freewheel_seconds':
                    value = max(0.0, float(value))
                elif key == 'audio_volume':
                    value = _checked_audio_volume(value)
                elif key in ('audio_muted', 'show_sync_warning',
                             'audio_diagnostics'):
                    value = bool(value)
                self._values[key] = value

    def snapshot(self):
        with self._lock:
            return dict(self._values)


class _ChannelEvidence:
    """Confirm a coded profile after distinct valid packet statuses."""

    CONFIRM_PACKETS = 2

    def __init__(self, loss_seconds):
        self.mode = None
        self.candidate = None
        self.streak = 0
        self.last_valid = None
        self.loss_seconds = float(loss_seconds)

    def observe(self, mode, now, already_confirmed=False):
        if mode is None:
            return False
        mode = int(mode)
        now = float(now)
        if self.last_valid is not None and now <= self.last_valid:
            return False
        if (self.last_valid is not None and
                now-self.last_valid > self.loss_seconds):
            self.candidate = None
            self.streak = 0
        if mode == self.candidate:
            self.streak += 1
        else:
            self.candidate = mode
            self.streak = 1
        if already_confirmed or self.streak >= self.CONFIRM_PACKETS:
            changed = mode != self.mode
            self.mode = mode
            self.last_valid = now
            return changed
        return False

    def active(self, now, loss_seconds):
        return (self.mode is not None and self.last_valid is not None and
                float(now)-self.last_valid <= loss_seconds)


class ReceiverChannelRouter:
    """Metadata-driven video/audio assignment with loss-tolerant side changes."""

    LOSS_SECONDS = 0.75

    def __init__(self, mono_mode, stereo_modes=(), loss_seconds=LOSS_SECONDS,
                 initial_video_side=None):
        if initial_video_side not in (None, 'left', 'right', 'both'):
            raise ValueError('initial video side must be left, right, both, or None')
        if isinstance(mono_mode, (tuple, list, set, frozenset)):
            mono_modes = frozenset(map(int, mono_mode))
            if not mono_modes:
                raise ValueError('at least one mono status mode is required')
            self.mono_mode = min(mono_modes)
            self.mono_modes = mono_modes
        else:
            self.mono_mode = int(mono_mode)
            self.mono_modes = frozenset((self.mono_mode,))
        self.stereo_modes = frozenset(map(int, stereo_modes))
        self.loss_seconds = max(0.0, float(loss_seconds))
        self.channels = (_ChannelEvidence(self.loss_seconds),
                         _ChannelEvidence(self.loss_seconds))
        self.state = 'searching'
        self.video_side = initial_video_side
        self.audio_side = (self._opposite(initial_video_side)
                           if initial_video_side in ('left', 'right') else None)
        self.last_packet = None
        if initial_video_side is not None:
            self.state = ('dual-mono' if initial_video_side == 'both' else
                          f'mono-{initial_video_side}')

    @staticmethod
    def _opposite(side):
        return {'left': 'right', 'right': 'left'}.get(side)

    def observe(self, left_mode=None, right_mode=None, now=None,
                left_seen_at=None, right_seen_at=None,
                left_confirmed=False, right_confirmed=False):
        now = time.monotonic() if now is None else float(now)
        self.channels[0].observe(
            left_mode, now if left_seen_at is None else left_seen_at,
            already_confirmed=left_confirmed)
        self.channels[1].observe(
            right_mode, now if right_seen_at is None else right_seen_at,
            already_confirmed=right_confirmed)
        active = [item.active(now, self.loss_seconds)
                  for item in self.channels]
        modes = [item.mode if is_active else None
                 for item, is_active in zip(self.channels, active)]
        for item in self.channels:
            if item.last_valid is not None:
                self.last_packet = max(self.last_packet or item.last_valid,
                                       item.last_valid)

        if all(active):
            if modes[0] == modes[1] and modes[0] in self.mono_modes:
                self.state, self.video_side, self.audio_side = (
                    'dual-mono', 'both', None)
            elif modes[0] == modes[1] and modes[0] in self.stereo_modes:
                self.state, self.video_side, self.audio_side = (
                    'stereo', 'both', None)
            else:
                self.state, self.audio_side = 'unresolved', None
            return self.snapshot(now)

        active_index = next((index for index, present in enumerate(active)
                             if present), None)
        if active_index is None:
            return self.snapshot(now)

        mode = modes[active_index]
        if mode in self.stereo_modes:
            # A stereo profile owns both input legs, including during one-sided
            # packet loss; never leak the damaged modem leg to audio output.
            self.state, self.video_side, self.audio_side = (
                'stereo', 'both', None)
        elif mode in self.mono_modes:
            side = 'left' if active_index == 0 else 'right'
            current = ({'left': 0, 'right': 1}.get(self.video_side)
                       if self.video_side in ('left', 'right') else None)
            if (self.state.startswith('mono-') and current is not None and
                    current != active_index):
                if active[current]:
                    return self.snapshot(now)
            if self.state in ('stereo', 'dual-mono'):
                # Once both legs identified as modem signal, one-sided loss is
                # not enough evidence to reinterpret a leg as audio.
                return self.snapshot(now)
            self.state, self.video_side, self.audio_side = (
                f'mono-{side}', side, self._opposite(side))
        else:
            # Unknown profile codes are never treated as audio eligibility.
            self.state, self.audio_side = 'unresolved', None
        return self.snapshot(now)

    def sync_state(self, now=None, freewheel_seconds=2.0):
        now = time.monotonic() if now is None else float(now)
        if self.last_packet is None:
            return 'acquiring'
        age = max(0.0, now-self.last_packet)
        if age <= self.loss_seconds:
            return 'playing'
        if age <= max(self.loss_seconds, float(freewheel_seconds)):
            return 'freewheeling'
        return 'sync-lost'

    def observe_profile_decision(self, decision, now=None):
        """Route only confirmed profiles, with independent evidence per leg.

        A profile candidate is not permission to silence an established audio
        route. Even for the active profile, one status hit on the soundtrack
        must not identify that leg as a second modem channel.
        """
        if not decision['confirmed']:
            return self.snapshot(now)
        left, right = decision['channel_modes']
        for evidence, mode in zip(self.channels, (left, right)):
            if mode is None:
                # Absence on this packet breaks a candidate streak, without
                # revoking the established route during a short dropout.
                evidence.candidate = None
                evidence.streak = 0
        return self.observe(left, right, now=now)

    def snapshot(self, now=None):
        now = time.monotonic() if now is None else float(now)
        return {
            'state': self.state,
            'video_side': self.video_side,
            'audio_side': self.audio_side,
            'channel_modes': tuple(item.mode for item in self.channels),
            'channel_active': tuple(item.active(now, self.loss_seconds)
                                    for item in self.channels),
            'last_packet': self.last_packet,
        }


def passthrough_audio_side(route, input_channels):
    """Return the confirmed side-channel audio leg, independent of sync age."""
    if int(input_channels) < 2 or not isinstance(route, dict):
        return None
    side = route.get('audio_side')
    return side if side in ('left', 'right') else None


class _RateMeter:
    """Wall-clock sample rate over a sliding window (diagnostics only).

    Callbacks often arrive in clusters: with a 4,096-frame host buffer,
    PortAudio delivers four 1,024-frame blocks back to back. Counting from
    the first block of a cluster then includes its cluster-mates without
    their period and reads high by (k-1) blocks per window: +3,200 ppm for
    96 kHz capture in clusters of four. Arrivals within COALESCE_SECONDS of
    the first block of a group are therefore counted as one arrival, stamped
    at its last block.
    """

    WINDOW_SECONDS = 10.0
    COALESCE_SECONDS = 0.002

    def __init__(self):
        self._points = deque()  # (last time, cumulative total, group start)
        self._total = 0

    def add(self, count, now):
        now = float(now)
        self._total += int(count)
        if self._points and now-self._points[-1][2] < self.COALESCE_SECONDS:
            self._points[-1] = (now, self._total, self._points[-1][2])
        else:
            self._points.append((now, self._total, now))
        while (len(self._points) > 2 and
               now-self._points[0][0] > self.WINDOW_SECONDS):
            self._points.popleft()

    def rate(self):
        if len(self._points) < 2:
            return None
        (t0, n0, _), (t1, n1, _) = self._points[0], self._points[-1]
        if t1-t0 < 1.0:
            return None
        # Samples up to t0 arrived at or before t0: count only later ones.
        return (n1-n0)/(t1-t0)


class AudioPassthrough:
    """Continuously play the selected captured audio leg."""

    # Keep a short capture-clock buffer: enough to absorb callback jitter while
    # the bounded FIFO servo prevents unbounded monitor delay.
    BUFFER_SECONDS = 0.5
    # Prime by about one V7 packet before the continuous output stream starts.
    TARGET_SECONDS = 0.08
    # Independent BlackHole/Bluetooth clocks can exceed the generic source
    # reader's 0.1% cap; permit up to 0.5% correction for receiver monitoring.
    CLOCK_CORRECTION_LIMIT = 0.005
    FRAME_GAP_CROSSFADE_SECONDS = 0.01
    RECOVERY_FADE_SECONDS = 0.005
    # Every route change, reprime or buffer reset ramps from the last emitted
    # sample instead of stepping to silence or to mid-waveform audio.
    DECLICK_SECONDS = 0.005
    # Fill-level low-pass for the clock servo (see ClockMatchedReader).
    FILL_SMOOTHING_SECONDS = 0.25
    # PI servo gains (per second, per second squared): critically damped at
    # about 0.1 rad/s. The integral holds whatever steady clock offset there
    # is, so the proportional term stays gentle and fill noise moves the
    # ratio by hundreds of ppm instead of rail to rail. (An apparent +3,100
    # ppm capture clock on the reference Mac was a rate-meter artifact of
    # clustered callbacks; see _RateMeter.rate.)
    SERVO_KP = 0.2
    SERVO_KI = 0.01
    # If the FIFO still grows past this (output callbacks starved, or a rate
    # error beyond the servo limit), skip back to the target with a ramp
    # rather than letting latency grow until the FIFO sheds audio unfaded.
    TRIM_SECONDS = 0.25

    # Wake period of the stall probe (see _stall_probe).
    STALL_PROBE_SECONDS = 0.005
    # Frames per output callback. The callback is Python, so every period it
    # must win the GIL before the device deadline. On the reference Mac the
    # host's own 512-frame (11.6 ms) period lost ~2% of output blocks to
    # 12-26 ms process stalls; 1,024 frames lost one block in ~25 s. 2,048
    # frames (46 ms at 44.1 kHz) clears the worst stall seen with margin, for
    # about 35 ms more monitor delay. 0 lets the host choose.
    OUTPUT_BLOCKSIZE = 2048
    # Capture callbacks are delayed by the same stalls (up to ~26 ms seen);
    # the servo target keeps this much audio beyond one output callback.
    CAPTURE_STALL_SECONDS = 0.03

    # Output mode. 'blocking' (default): PortAudio's own C thread plays from
    # its ring buffer (CoreAudio) or the device buffer (ALSA, JACK, WASAPI),
    # and a Python writer thread only refills it. The real-time path never
    # needs the GIL, so a process stall costs buffer headroom, not output.
    # 'callback': the previous Python output callback (OUTPUT_BLOCKSIZE).
    OUTPUT_MODE = 'blocking'
    # Frames per blocking write (and PortAudio framesPerBuffer).
    BLOCKING_WRITE_FRAMES = 512
    # Requested output buffer ahead of the device in blocking mode. It must
    # outlast the longest stall of the writer thread (~26 ms seen on the
    # reference Mac) plus one write. CoreAudio sizes its blocking ring as the
    # next power of two >= 2 x suggested latency x rate, so the request is
    # halved there: 45 ms gives a 2,048-frame (46 ms) ring at 44.1 kHz.
    BLOCKING_BUFFER_SECONDS = 0.045
    # FIFO target in blocking mode. The output buffer now absorbs output-side
    # timing, so the FIFO only covers capture jitter; the target floor still
    # adds one write, one capture block and CAPTURE_STALL_SECONDS (~52 ms at
    # 96 kHz in, 44.1 kHz out). Keeps total monitor delay near the old path.
    BLOCKING_TARGET_SECONDS = 0.05

    def __init__(self, input_rate, sounddevice_module=None, status_callback=None,
                 volume=DEFAULT_AUDIO_VOLUME, target_seconds=None,
                 output_latency=None, output_blocksize=None,
                 output_mode=None):
        if sounddevice_module is None:
            import sounddevice as sounddevice_module
        from tools.v7_source_audio import SampleBuffer

        self.sd = sounddevice_module
        self.input_rate = int(input_rate)
        self.output_mode = (self.OUTPUT_MODE if output_mode is None
                            else str(output_mode))
        if self.output_mode not in ('blocking', 'callback'):
            raise ValueError("audio output mode must be 'blocking' or 'callback'")
        default_target = (self.BLOCKING_TARGET_SECONDS
                          if self.output_mode == 'blocking'
                          else self.TARGET_SECONDS)
        self.target_seconds = (default_target if target_seconds is None
                               else float(target_seconds))
        if not np.isfinite(self.target_seconds) or self.target_seconds <= 0:
            raise ValueError('audio target buffer must be positive')
        self.output_rate = None
        self.device_default_rate = None
        self.channels = 0
        self.device = None
        self.buffer = SampleBuffer(round(self.input_rate*self.BUFFER_SECONDS))
        self.history = deque()
        self._history_end = None
        self._queued_sample_end = None
        self._queued_last_sample = None
        self.reader = None
        self._fade_in_after_underflow = False
        self._audio_primed = False
        self._last_output = 0.0
        self._declick_pending = False
        self._largest_push = 0
        self._input_meter = _RateMeter()
        self._output_meter = _RateMeter()
        self._last_callback_time = None
        self._max_callback_gap = 0.0
        self._late_callbacks = 0
        self._late_ms_total = 0.0
        # Experimental output buffering: a larger host buffer gives the
        # Python callback more slack before a stall costs audible output.
        self.output_latency = output_latency
        default_blocksize = (self.BLOCKING_WRITE_FRAMES
                             if self.output_mode == 'blocking'
                             else self.OUTPUT_BLOCKSIZE)
        self.output_blocksize = int(default_blocksize if output_blocksize
                                    is None else output_blocksize)
        if self.output_mode == 'blocking' and self.output_blocksize == 0:
            raise ValueError('blocking audio output needs a positive blocksize')
        if self.output_blocksize < 0:
            raise ValueError('output blocksize must be 0 or positive')
        self._max_stall = 0.0
        self._stalls = 0
        self._stall_ms_total = 0.0
        self._probe_stop = None
        self._writer_stop = None
        self._writer_thread = None
        self._latency_trims = 0
        self._trimmed_samples = 0
        self.stream = None
        self.route = None
        self.muted = False
        self.volume = _checked_audio_volume(volume)
        self.error = None
        self._status_callback = status_callback
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._output_underflow_events = 0
        self._output_underflow_callbacks = 0
        self._output_underflow_active = False
        self._input_status_counts = {}
        self._output_status_counts = {}

    def _report(self, error):
        self.error = None if error is None else str(error)
        if self._status_callback is not None:
            try:
                self._status_callback(self.error)
            except Exception:
                pass

    @staticmethod
    def _count_status(counts, status):
        if not status:
            return
        name = str(status)
        counts[name] = counts.get(name, 0)+1

    def note_input_status(self, status):
        """Record PortAudio capture xruns for the receiver diagnostics."""
        with self._stats_lock:
            self._count_status(self._input_status_counts, status)

    def stats_snapshot(self):
        """Return audio FIFO, clock correction and PortAudio xrun counters."""
        with self._stats_lock:
            stats = {
                # Process-wide stalls seen by a thread that wakes every 5 ms.
                # If these line up with late callbacks, something holds the
                # GIL (e.g. a compiled kernel without nogil) and the output
                # callback cannot run.
                'max_stall_ms': round(self._max_stall*1000.0, 2),
                'stalls_over_10ms': self._stalls,
                'stall_ms_total': round(self._stall_ms_total, 1),
                'late_ms_total': round(self._late_ms_total, 1),
                'underflow_events': self._output_underflow_events,
                'underflow_callbacks': self._output_underflow_callbacks,
                'input_status': dict(self._input_status_counts),
                'output_status': dict(self._output_status_counts),
                # Measured against the wall clock. A measured output rate
                # below nominal while input matches nominal means output
                # callbacks are being missed (e.g. the GIL is held), which
                # no clock servo can correct.
                'measured_input_rate': self._input_meter.rate(),
                'measured_output_rate': self._output_meter.rate(),
                'max_callback_gap_ms': round(self._max_callback_gap*1000.0, 2),
                'late_callbacks': self._late_callbacks,
                'latency_trims': self._latency_trims,
                'trimmed_ms': round(
                    1000.0*self._trimmed_samples/self.input_rate, 1),
            }
        with self._lock:
            reader = self.reader
            output_rate = self.output_rate
        stats.update({
            'buffered_ms': self.buffered_ms,
            'dropped_samples': int(self.buffer.dropped),
            'current_clock_correction_ppm': (0.0 if reader is None else
                                             reader.current_correction*1e6),
            'max_clock_correction_ppm': (0.0 if reader is None else
                                         reader.max_correction*1e6),
            'input_rate': self.input_rate,
            'output_rate': output_rate,
            'output_latency_ms': self._stream_latency_ms(),
            'output_blocksize': self.output_blocksize,
            'output_mode': self.output_mode,
        })
        return stats

    def reset_report_peaks(self):
        """Start new per-report maxima (call after printing a report)."""
        with self._stats_lock:
            self._max_callback_gap = 0.0
            self._max_stall = 0.0

    def _stream_latency_ms(self):
        with self._lock:
            stream = self.stream
        try:
            return (None if stream is None else
                    round(1000.0*float(stream.latency), 2))
        except Exception:
            return None

    def _stall_probe(self, stop):
        period = self.STALL_PROBE_SECONDS
        while not stop.is_set():
            started = time.monotonic()
            stop.wait(period)
            late = time.monotonic()-started-period
            if late <= 0:
                continue
            with self._stats_lock:
                self._max_stall = max(self._max_stall, late)
                if late > 0.010:
                    self._stalls += 1
                    self._stall_ms_total += late*1000.0

    def _make_clock_reader(self, input_rate, output_rate):
        from tools.v7_source_audio import ClockMatchedReader

        input_rate = float(input_rate)
        output_rate = float(output_rate)
        target = max(1, int(round(input_rate*self.target_seconds)))
        return ClockMatchedReader(
            self.buffer, target, nominal_ratio=input_rate/output_rate,
            correction_limit=self.CLOCK_CORRECTION_LIMIT,
            smoothing_samples=input_rate*self.FILL_SMOOTHING_SECONDS,
            servo_rate=input_rate, servo_kp=self.SERVO_KP,
            servo_ki=self.SERVO_KI)

    @staticmethod
    def _identity(sd, index):
        return device_identity(sd, index)

    @classmethod
    def _resolve_device(cls, sd, device, identity):
        return resolve_device_index(sd, device, identity, 'output')

    def open(self, device, expected_identity=None):
        self.close()
        self.buffer.clear()
        if device is None and expected_identity is None:
            return
        stream = None
        try:
            device = self._resolve_device(
                self.sd, device, expected_identity)
            info = self.sd.query_devices(device, 'output')
            device_default_rate = float(
                info.get('default_samplerate') or self.input_rate)
            channels = min(2, int(info.get('max_output_channels') or 0))
            if channels < 1:
                raise ValueError('selected audio output has no output channels')
            rate = float(info.get('default_samplerate') or self.input_rate)
            self.sd.check_output_settings(
                device=device, channels=channels, dtype='float32',
                samplerate=rate)
            reader = self._make_clock_reader(self.input_rate, rate)
            if self.output_mode == 'blocking':
                latency = (self.output_latency
                           if self.output_latency is not None
                           else self._blocking_latency(device))
                stream = self.sd.OutputStream(
                    samplerate=rate, channels=channels, dtype='float32',
                    device=device, blocksize=self.output_blocksize,
                    latency=latency)
            else:
                extra = ({} if self.output_latency is None else
                         {'latency': self.output_latency})
                stream = self.sd.OutputStream(
                    samplerate=rate, channels=channels, dtype='float32',
                    device=device, blocksize=self.output_blocksize,
                    callback=self._callback, **extra)
            stream.start()
            rate = float(getattr(stream, 'samplerate', rate))
            if rate <= 0 or not np.isfinite(rate):
                raise ValueError('audio output reported an invalid sample rate')
            reader.nominal_ratio = self.input_rate/rate
        except Exception as exc:
            if stream is not None:
                try:
                    stream.stop()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass
            self._report(f'Audio output unavailable: {exc}')
            return
        with self._lock:
            self.device = device
            self.output_rate = rate
            self.device_default_rate = device_default_rate
            self.channels = channels
            self.reader = reader
            self._fade_in_after_underflow = False
            self._audio_primed = False
            self.stream = stream
        if self.output_mode == 'blocking':
            self._writer_stop = threading.Event()
            self._writer_thread = threading.Thread(
                target=self._writer_loop,
                args=(stream, self._writer_stop, self.output_blocksize,
                      channels),
                name='v7-audio-writer', daemon=True)
            self._writer_thread.start()
        self._probe_stop = threading.Event()
        threading.Thread(target=self._stall_probe, args=(self._probe_stop,),
                         name='v7-audio-stall-probe', daemon=True).start()
        self._report(None)

    def _blocking_latency(self, device):
        """PortAudio suggested latency giving ~BLOCKING_BUFFER_SECONDS."""
        seconds = float(self.BLOCKING_BUFFER_SECONDS)
        try:
            hostapi = self.sd.query_devices(device)['hostapi']
            name = str(self.sd.query_hostapis(hostapi)['name'])
        except Exception:
            name = ''
        # CoreAudio's blocking ring is sized from twice the latency.
        return seconds/2 if 'core audio' in name.lower() else seconds

    def _writer_loop(self, stream, stop, frames, channels):
        """Refill PortAudio's output buffer; the only Python on this path.

        PortAudio's own thread plays from the buffer, so a stall here only
        spends buffered audio. The writer never blocks inside PortAudio: it
        writes only when a whole block fits (write_available) and otherwise
        sleeps in Python until about that much has played. Stopping a stream
        while another thread waits in a blocking Pa_WriteStream can deadlock
        (seen with the JACK host API), and close() must be able to end this
        loop promptly before it aborts the stream.
        """
        block = np.zeros((frames, channels), dtype=np.float32)
        rate = float(getattr(stream, 'samplerate', 0) or
                     self.output_rate or self.input_rate)
        while not stop.is_set():
            try:
                available = int(stream.write_available)
            except Exception as exc:
                if not stop.is_set():
                    self._report(f'Audio output failed: {exc}')
                return
            if available < frames:
                stop.wait(max(0.001, (frames-available)/rate))
                continue
            self._callback(block, frames, None, None)
            try:
                underflowed = stream.write(block)
            except Exception as exc:
                if not stop.is_set():
                    self._report(f'Audio output failed: {exc}')
                return
            if underflowed:
                with self._stats_lock:
                    self._count_status(
                        self._output_status_counts, 'output underflow')

    def set_input_rate(self, rate):
        """Use the capture stream's negotiated clock for passthrough resampling."""
        rate = float(rate)
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError('audio input sample rate must be positive')
        with self._lock:
            self.input_rate = rate
            output_rate = self.output_rate
            if output_rate:
                self.reader = self._make_clock_reader(rate, output_rate)
            self.history.clear()
            self._history_end = None
            self._queued_sample_end = None
            self._queued_last_sample = None
            self.buffer.clear()
            self._fade_in_after_underflow = False
            self._audio_primed = False
            self._declick_pending = True

    def record_capture(self, sample_start, block):
        """Retain raw input with the decoder's absolute sample coordinates."""
        values = np.asarray(block, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] < 2 or not len(values):
            return
        sample_start = int(sample_start)
        values = values[:, :2].copy()
        sample_end = sample_start+len(values)
        with self._lock:
            if (self._history_end is not None and
                    sample_start != self._history_end):
                self.history.clear()
            self.history.append((sample_start, values))
            self._history_end = sample_end
            oldest = sample_end-int(round(self.input_rate*self.BUFFER_SECONDS))
            while self.history and self.history[0][0]+len(self.history[0][1]) <= oldest:
                self.history.popleft()
            if self.history and self.history[0][0] < oldest:
                start, first = self.history[0]
                trim = oldest-start
                self.history[0] = (oldest, first[trim:].copy())

    def queue_frame(self, sample_start, sample_count, side):
        """Queue audio samples from the exact input interval of a decoded frame."""
        if side not in ('left', 'right'):
            return False
        sample_start = int(sample_start)
        sample_count = int(sample_count)
        if sample_count <= 0:
            return False
        sample_end = sample_start+sample_count
        channel = 0 if side == 'left' else 1
        pieces = []
        cursor = sample_start
        with self._lock:
            if self.route != side or self.muted:
                return False
            for start, values in self.history:
                end = start+len(values)
                if end <= cursor:
                    continue
                if start > cursor:
                    return False
                take_end = min(end, sample_end)
                pieces.append(values[cursor-start:take_end-start, channel])
                cursor = take_end
                if cursor >= sample_end:
                    break
            if cursor < sample_end:
                return False
            audio = np.concatenate(pieces)
            return self._queue_samples_locked(sample_start, audio)

    def queue_capture(self, sample_start, block):
        """Queue a captured audio block without waiting for video decoding.

        The PortAudio output stream consumes this FIFO continuously. A missed
        picture therefore does not pause the audio leg or restart playback.
        """
        values = np.asarray(block, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] < 2 or not len(values):
            return False
        with self._lock:
            if self.route not in ('left', 'right') or self.muted:
                return False
            channel = 0 if self.route == 'left' else 1
            audio = values[:, channel].copy()
            # Capture blocks set the arrival granularity the servo target must
            # cover (decoded-frame queuing is paced by the decoder instead).
            self._largest_push = max(self._largest_push, len(audio))
            return self._queue_samples_locked(sample_start, audio)

    def _queue_samples_locked(self, sample_start, audio):
        """Append samples and gently bridge timestamp gaps or overlaps."""
        sample_start = int(sample_start)
        sample_end = sample_start+len(audio)
        previous_end = self._queued_sample_end
        if previous_end is not None:
            gap = sample_start-previous_end
            if gap < 0:
                overlap = min(-gap, len(audio))
                if overlap == len(audio):
                    return True
                audio = audio[overlap:].copy()
            elif gap > max(1, int(round(self.input_rate*.001))):
                fade_samples = min(
                    len(audio), max(1, int(round(
                        self.input_rate*self.FRAME_GAP_CROSSFADE_SECONDS))))
                if fade_samples == 1:
                    audio[0] = self._queued_last_sample
                else:
                    phase = np.linspace(
                        0.0, 1.0, fade_samples, dtype=np.float32)
                    gain = phase*phase*(3.0-2.0*phase)
                    audio[:fade_samples] = (
                        self._queued_last_sample*(1.0-gain) +
                        audio[:fade_samples]*gain)
        if not len(audio):
            return True
        self._queued_sample_end = sample_end
        self._queued_last_sample = float(audio[-1])
        with self._stats_lock:
            self._input_meter.add(len(audio), time.monotonic())
        self.buffer.push(audio)
        return True

    def clear_audio(self):
        """Discard captured history and queued output after a capture gap."""
        with self._lock:
            self.history.clear()
            self._history_end = None
            self._queued_sample_end = None
            self._queued_last_sample = None
            self.buffer.clear()
            self._fade_in_after_underflow = False
            self._audio_primed = False
            self._declick_pending = True
            if self.output_rate:
                self.reader = self._make_clock_reader(
                    self.input_rate, self.output_rate)

    @property
    def buffered_ms(self):
        return 1000.0*self.buffer.available/self.input_rate

    def _declick(self, samples, rate):
        """Ramp from the last emitted sample into `samples` (in place)."""
        fade = min(len(samples), int(round(rate*self.DECLICK_SECONDS)))
        if fade <= 0:
            return
        phase = (np.arange(1, fade+1, dtype=np.float32)/fade)
        gain = phase*phase*(3.0-2.0*phase)
        samples[:fade] = self._last_output*(1.0-gain) + samples[:fade]*gain

    def _note_callback_timing(self, frames, output_rate):
        now = time.monotonic()
        with self._stats_lock:
            previous = self._last_callback_time
            self._last_callback_time = now
            self._output_meter.add(frames, now)
            if previous is not None:
                gap = now-previous
                self._max_callback_gap = max(self._max_callback_gap, gap)
                period = frames/output_rate
                if gap > 1.5*period+0.005:
                    self._late_callbacks += 1
                    self._late_ms_total += (gap-period)*1000.0

    def _callback(self, outdata, frames, _timing, status):
        self._note_callback_timing(
            frames, self.output_rate or self.input_rate)
        if status:
            self._report(f'Audio output warning: {status}')
            with self._stats_lock:
                self._count_status(self._output_status_counts, status)
        with self._lock:
            reader = self.reader
            audible = self.route is not None and not self.muted
            volume = self.volume
            primed = self._audio_primed
            declick = self._declick_pending
            self._declick_pending = False
            output_rate = self.output_rate or self.input_rate
        outdata.fill(0)
        samples = np.zeros(frames, dtype=np.float32)
        try:
            if audible and reader is not None:
                # Keep the servo target above one output callback, one
                # capture block and the longest capture delay seen in the
                # field (a GIL stall holds the capture callback too), or
                # large output callbacks underflow when capture runs late.
                needed = int(np.ceil(frames*reader.nominal_ratio))+2
                floor = (needed+self._largest_push+int(round(
                    self.input_rate*self.CAPTURE_STALL_SECONDS)))
                if reader.target_samples < floor:
                    reader.target_samples = floor
                ready = primed
                if not primed:
                    required = max(reader.target_samples, needed)
                    if self.buffer.available >= required:
                        with self._lock:
                            ready = (self.reader is reader and
                                     self.route is not None and not self.muted)
                            if ready:
                                self._audio_primed = True
                        # Starting from silence: ramp in from the last sample.
                        declick = declick or ready
                trim_at = max(int(round(self.input_rate*self.TRIM_SECONDS)),
                              2*reader.target_samples)
                if ready and self.buffer.available > trim_at:
                    drop = self.buffer.available-reader.target_samples
                    self.buffer.read(drop)
                    reader.reset()
                    declick = True
                    with self._stats_lock:
                        self._latency_trims += 1
                        self._trimmed_samples += drop
                if ready:
                    samples = self._read_locked_state(reader, frames)
            if declick:
                self._declick(samples, output_rate)
            elif self._last_output != 0.0 and not np.any(samples):
                # Audio stopped without a reset (route cleared, unprimed):
                # fade the last sample to silence instead of stepping.
                self._declick(samples, output_rate)
            if len(samples):
                self._last_output = float(samples[-1])
            outdata[:] = np.clip(samples*volume, -1.0, 1.0)[:, None]
        except Exception as exc:
            self._report(f'Audio passthrough failed: {exc}')

    def _read_locked_state(self, reader, frames):
        samples = reader.read(frames)
        if reader.underflow:
            with self._stats_lock:
                self._output_underflow_callbacks += 1
                if not self._output_underflow_active:
                    self._output_underflow_events += 1
                self._output_underflow_active = True
            valid_samples = min(len(samples), reader.valid_output_samples)
            if valid_samples:
                fade_samples = min(
                    valid_samples, max(1, int(round(
                        (self.output_rate or self.input_rate)*
                        self.RECOVERY_FADE_SECONDS))))
                fade_start = valid_samples-fade_samples
                if fade_samples == 1:
                    samples[fade_start] = 0.0
                else:
                    phase = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
                    gain = 1.0-phase*phase*(3.0-2.0*phase)
                    samples[fade_start:valid_samples] *= gain
            samples[valid_samples:] = 0.0
            with self._lock:
                if self.reader is reader:
                    self._fade_in_after_underflow = True
                    self._audio_primed = False
        else:
            with self._stats_lock:
                self._output_underflow_active = False
            with self._lock:
                fade_in = (self.reader is reader and
                           self._fade_in_after_underflow)
                if fade_in:
                    self._fade_in_after_underflow = False
                output_rate = self.output_rate or self.input_rate
            if fade_in and len(samples):
                fade_samples = min(
                    len(samples), max(1, int(round(
                        output_rate*self.RECOVERY_FADE_SECONDS))))
                if fade_samples == 1:
                    samples[0] = 0.0
                else:
                    phase = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
                    gain = phase*phase*(3.0-2.0*phase)
                    samples[:fade_samples] *= gain
        return samples

    def set_route(self, side, muted=None):
        if side not in (None, 'left', 'right'):
            raise ValueError('audio route must be left, right, or None')
        with self._lock:
            changed = side != self.route
            self.route = side
            if muted is not None:
                self.muted = bool(muted)
            if changed:
                self.buffer.clear()
                self._queued_sample_end = None
                self._queued_last_sample = None
                self._fade_in_after_underflow = False
                self._audio_primed = False
                self._declick_pending = True
                if self.output_rate:
                    self.reader = self._make_clock_reader(
                        self.input_rate, self.output_rate)

    def set_muted(self, muted):
        with self._lock:
            muted = bool(muted)
            if muted != self.muted:
                self.muted = muted
                # Mute stays immediate (a user action); unmuting ramps in.
                self._last_output = 0.0
                self._declick_pending = not muted
                self.buffer.clear()
                self._queued_sample_end = None
                self._queued_last_sample = None
                self._audio_primed = False
                if self.output_rate:
                    self.reader = self._make_clock_reader(
                        self.input_rate, self.output_rate)

    def set_volume(self, volume):
        volume = _checked_audio_volume(volume)
        with self._lock:
            self.volume = volume

    @property
    def is_open(self):
        with self._lock:
            stream = self.stream
        if stream is None:
            return False
        try:
            return bool(stream.active)
        except Exception:
            return True

    def check_device(self, expected_identity=None):
        """Detect a stopped stream or a PortAudio index reassignment."""
        with self._lock:
            stream, current_device = self.stream, self.device
        if stream is None:
            return False
        try:
            if not stream.active:
                self._report('Audio output stopped; reselect the output device.')
                return False
        except AttributeError:
            pass
        except Exception as exc:
            self._report(f'Audio output health check failed: {exc}')
            return False
        if expected_identity is not None:
            try:
                resolved = self._resolve_device(
                    self.sd, current_device, expected_identity)
            except Exception as exc:
                self._report(f'Audio output unavailable; reselect it: {exc}')
                return False
            if resolved != current_device:
                self._report('Audio output device index changed; reopening it.')
                return False
        return True

    def native_device_sample_rate(self, expected_identity=None):
        """Read the selected output device's currently reported native rate."""
        with self._lock:
            current_device = self.device
        if current_device is None:
            return None
        resolved = self._resolve_device(
            self.sd, current_device, expected_identity)
        info = self.sd.query_devices(resolved, 'output')
        rate = float(info.get('default_samplerate') or 0.0)
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError('audio output reported an invalid native sample rate')
        return rate

    def close(self):
        if self._probe_stop is not None:
            self._probe_stop.set()
            self._probe_stop = None
        writer, writer_stop = self._writer_thread, self._writer_stop
        self._writer_thread = self._writer_stop = None
        if writer_stop is not None:
            writer_stop.set()
        with self._lock:
            stream, self.stream = self.stream, None
            self.reader = None
            self._queued_sample_end = None
            self._queued_last_sample = None
            self._fade_in_after_underflow = False
            self._audio_primed = False
            self.output_rate = None
            self.device_default_rate = None
            self.channels = 0
            self.device = None
        if stream is None:
            return
        if writer is None:
            try:
                stream.stop()
            finally:
                stream.close()
            return
        # Blocking mode: end the writer before touching the stream. It never
        # waits inside PortAudio, so it exits within about one block; only
        # then abort (no drain) and close, so no write races the stop.
        writer.join(timeout=2.0)
        if writer.is_alive():
            self._report('Audio writer did not stop; output stream left open.')
            return
        try:
            stream.abort()
        finally:
            stream.close()
