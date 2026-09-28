"""Runtime routing and optional audio passthrough for the V7 receiver."""
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
        self.mono_mode = int(mono_mode)
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
            if modes[0] == modes[1] == self.mono_mode:
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
        elif mode == self.mono_mode:
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
        if age < max(self.loss_seconds, float(freewheel_seconds)):
            return 'freewheeling'
        return 'sync-lost'

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
    """Bounded, independent-clock mono input to selected stereo output."""

    BUFFER_SECONDS = 2.0
    TARGET_SECONDS = 0.12

    def __init__(self, input_rate, sounddevice_module=None, status_callback=None,
                 volume=DEFAULT_AUDIO_VOLUME):
        if sounddevice_module is None:
            import sounddevice as sounddevice_module
        from tools.v7_source_audio import ClockMatchedReader, SampleBuffer

        self.sd = sounddevice_module
        self.input_rate = int(input_rate)
        self.output_rate = None
        self.channels = 0
        self.device = None
        self.buffer = SampleBuffer(round(self.input_rate*self.BUFFER_SECONDS))
        self.reader = None
        self.stream = None
        self.route = None
        self.muted = False
        self.volume = _checked_audio_volume(volume)
        self.error = None
        self._status_callback = status_callback
        self._lock = threading.Lock()

    def _report(self, error):
        self.error = None if error is None else str(error)
        if self._status_callback is not None:
            try:
                self._status_callback(self.error)
            except Exception:
                pass

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
            rate = int(round(float(info.get('default_samplerate') or
                                   self.input_rate)))
            self.sd.check_output_settings(
                device=device, channels=channels, dtype='float32',
                samplerate=rate)
            from tools.v7_source_audio import ClockMatchedReader
            target = round(self.input_rate*self.TARGET_SECONDS)
            ratio = self.input_rate/rate
            reader = ClockMatchedReader(
                self.buffer, target, nominal_ratio=ratio)
            stream = self.sd.OutputStream(
                samplerate=rate, channels=channels, dtype='float32',
                device=device, blocksize=0,
                callback=self._callback)
            stream.start()
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
            self.stream = stream
        self._report(None)

    def _callback(self, outdata, frames, _timing, status):
        if status:
            self._report(f'Audio output warning: {status}')
        with self._lock:
            reader = self.reader
            audible = self.route is not None and not self.muted
            volume = self.volume
        outdata.fill(0)
        if audible and reader is not None:
            try:
                samples = reader.read(frames)
                outdata[:] = np.clip(samples*volume, -1.0, 1.0)[:, None]
            except Exception as exc:
                self._report(f'Audio passthrough failed: {exc}')

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
            with self._lock:
                if self.reader is not None:
                    self.reader.reset()

    def set_muted(self, muted):
        with self._lock:
            self.muted = bool(muted)

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

    def push(self, block):
        with self._lock:
            side = self.route
        if side is None:
            return
        index = 0 if side == 'left' else 1
        values = np.asarray(block)
        if values.ndim == 2 and values.shape[1] > index:
            self.buffer.push(values[:, index])

    def close(self):
        with self._lock:
            stream, self.stream = self.stream, None
            self.reader = None
            self.output_rate = None
            self.channels = 0
            self.device = None
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()
