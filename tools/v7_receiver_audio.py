"""Runtime routing and packet-aligned audio passthrough for the V7 receiver."""
from collections import deque
import threading
import time

import numpy as np

DEFAULT_AUDIO_VOLUME = 1.0  # VLC's default 100% volume (unity gain).


def _checked_audio_volume(value):
    volume = float(value)
    if not np.isfinite(volume) or not 0.0 <= volume <= 1.0:
        raise ValueError('Passthrough volume must be between 0 and 1.')
    return volume


def device_identity(sounddevice_module, index):
    """Return a stable name/backend identity for a PortAudio device index."""
    device = sounddevice_module.query_devices(index)
    hostapis = sounddevice_module.query_hostapis()
    hostapi_index = device.get('hostapi')
    hostapi = (hostapis[int(hostapi_index)].get('name')
               if hostapi_index is not None and
               0 <= int(hostapi_index) < len(hostapis) else '')
    return {'name': str(device.get('name', '')),
            'hostapi': str(hostapi)}


def resolve_device_index(sounddevice_module, preferred, identity, kind):
    """Resolve a persisted device identity without falling back to defaults."""
    if identity is None:
        return preferred
    devices = sounddevice_module.query_devices()
    channel_key = ('max_input_channels' if kind == 'input' else
                   'max_output_channels')
    if preferred is not None:
        try:
            candidate = sounddevice_module.query_devices(preferred)
            current = device_identity(sounddevice_module, preferred)
            if (int(candidate.get(channel_key) or 0) > 0 and
                    current.get('name') == identity.get('name') and
                    current.get('hostapi') == identity.get('hostapi')):
                return preferred
        except Exception:
            pass
    for index, candidate in enumerate(devices):
        if int(candidate.get(channel_key) or 0) < 1:
            continue
        try:
            current = device_identity(sounddevice_module, index)
            if (current.get('name') == identity.get('name') and
                    current.get('hostapi') == identity.get('hostapi')):
                return index
        except Exception:
            continue
    expected = identity.get('name') or f'the selected {kind} device'
    raise ValueError(f'{expected} is unavailable; reselect the {kind} device')


class ReceiverRuntimeOptions:
    """Thread-safe live choices shared by the receiver GUI and decoder."""

    def __init__(self, audio_output_device=None, audio_muted=False,
                 audio_output_identity=None, audio_input_identity=None,
                 freewheel_seconds=2.0, show_sync_warning=True,
                 audio_volume=DEFAULT_AUDIO_VOLUME):
        self._values = {
            'audio_output_device': audio_output_device,
            'audio_output_identity': audio_output_identity,
            'audio_input_identity': audio_input_identity,
            'audio_muted': bool(audio_muted),
            'audio_volume': _checked_audio_volume(audio_volume),
            'freewheel_seconds': max(0.0, float(freewheel_seconds)),
            'show_sync_warning': bool(show_sync_warning),
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
                elif key in ('audio_muted', 'show_sync_warning'):
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
    # Route changes, reprimes and clears ramp from the last emitted sample.
    DECLICK_SECONDS = 0.005
    # Smooth callback-sized FIFO swings before they reach the clock servo.
    FILL_SMOOTHING_SECONDS = 0.25

    def __init__(self, input_rate, sounddevice_module=None, status_callback=None,
                 volume=DEFAULT_AUDIO_VOLUME, target_seconds=None):
        if sounddevice_module is None:
            import sounddevice as sounddevice_module
        from tools.v7_source_audio import SampleBuffer

        self.sd = sounddevice_module
        self.input_rate = int(input_rate)
        self.target_seconds = (self.TARGET_SECONDS if target_seconds is None
                               else float(target_seconds))
        if not np.isfinite(self.target_seconds) or self.target_seconds <= 0:
            raise ValueError('audio target buffer must be positive')
        self.output_rate = None
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
                'underflow_events': self._output_underflow_events,
                'underflow_callbacks': self._output_underflow_callbacks,
                'input_status': dict(self._input_status_counts),
                'output_status': dict(self._output_status_counts),
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
        })
        return stats

    def _make_clock_reader(self, input_rate, output_rate):
        from tools.v7_source_audio import ClockMatchedReader

        input_rate = float(input_rate)
        output_rate = float(output_rate)
        target = max(1, int(round(input_rate*self.target_seconds)))
        return ClockMatchedReader(
            self.buffer, target, nominal_ratio=input_rate/output_rate,
            correction_limit=self.CLOCK_CORRECTION_LIMIT,
            smoothing_samples=input_rate*self.FILL_SMOOTHING_SECONDS)

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
            channels = min(2, int(info.get('max_output_channels') or 0))
            if channels < 1:
                raise ValueError('selected audio output has no output channels')
            rate = float(info.get('default_samplerate') or self.input_rate)
            self.sd.check_output_settings(
                device=device, channels=channels, dtype='float32',
                samplerate=rate)
            reader = self._make_clock_reader(self.input_rate, rate)
            stream = self.sd.OutputStream(
                samplerate=rate, channels=channels, dtype='float32',
                device=device, blocksize=0,
                callback=self._callback)
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
            self.channels = channels
            self.reader = reader
            self._fade_in_after_underflow = False
            self._audio_primed = False
            self._last_output = 0.0
            self._declick_pending = True
            self.stream = stream
        self._report(None)

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
            self._largest_push = 0

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

    def _declick(self, samples, rate, last_output):
        """Ramp from the last emitted sample into `samples` (in place)."""
        fade = min(len(samples), int(round(rate*self.DECLICK_SECONDS)))
        if fade <= 0:
            return
        phase = np.arange(1, fade+1, dtype=np.float32)/fade
        gain = phase*phase*(3.0-2.0*phase)
        samples[:fade] = last_output*(1.0-gain) + samples[:fade]*gain

    def _callback(self, outdata, frames, _timing, status):
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
            last_output = self._last_output
            largest_push = self._largest_push
        outdata.fill(0)
        samples = np.zeros(frames, dtype=np.float32)
        try:
            if audible and reader is not None:
                # Retain enough audio for one output callback and one complete
                # capture callback, even when the host requests large blocks.
                needed = int(np.ceil(frames*reader.nominal_ratio))+2
                floor = needed+largest_push
                if reader.target_samples < floor:
                    reader.target_samples = floor
                ready = primed
                if not primed:
                    # target_samples may have been raised above, so include it
                    # in the actual prime condition as well as the base target.
                    required = max(reader.target_samples, needed)
                    if self.buffer.available >= required:
                        with self._lock:
                            ready = (self.reader is reader and
                                     self.route is not None and not self.muted)
                            if ready:
                                self._audio_primed = True
                        declick = declick or ready
                if ready:
                    samples = self._read_locked_state(reader, frames)

            # Apply transitions in output-amplitude space so the fade begins
            # at the actual last emitted value, including volume and clipping.
            emitted = np.clip(samples*volume, -1.0, 1.0)
            if declick or (last_output != 0.0 and not np.any(emitted)):
                self._declick(emitted, output_rate, last_output)
            if len(emitted):
                with self._lock:
                    self._last_output = float(emitted[-1])
            outdata[:] = emitted[:, None]
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
                    phase = np.linspace(
                        0.0, 1.0, fade_samples, dtype=np.float32)
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
                    phase = np.linspace(
                        0.0, 1.0, fade_samples, dtype=np.float32)
                    gain = phase*phase*(3.0-2.0*phase)
                    samples[:fade_samples] *= gain
        return samples

    def set_route(self, side, muted=None):
        if side not in (None, 'left', 'right'):
            raise ValueError('audio route must be left, right, or None')
        with self._lock:
            route_changed = side != self.route
            muted_changed = (muted is not None and
                             bool(muted) != self.muted)
            self.route = side
            if muted is not None:
                self.muted = bool(muted)
            if route_changed or muted_changed:
                self.buffer.clear()
                self._queued_sample_end = None
                self._queued_last_sample = None
                self._fade_in_after_underflow = False
                self._audio_primed = False
                if self.muted:
                    # A user mute is immediate; subsequent unmute starts from
                    # silence and ramps back in after the FIFO is primed.
                    self._last_output = 0.0
                    self._declick_pending = False
                else:
                    self._declick_pending = True
                if self.output_rate:
                    self.reader = self._make_clock_reader(
                        self.input_rate, self.output_rate)

    def set_muted(self, muted):
        with self._lock:
            muted = bool(muted)
            if muted != self.muted:
                self.muted = muted
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

    def close(self):
        with self._lock:
            stream, self.stream = self.stream, None
            self.reader = None
            self._queued_sample_end = None
            self._queued_last_sample = None
            self._fade_in_after_underflow = False
            self._audio_primed = False
            self.output_rate = None
            self.channels = 0
            self.device = None
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()
