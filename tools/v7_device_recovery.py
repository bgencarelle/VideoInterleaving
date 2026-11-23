"""Stable device identification and sample-rate debounce for live V7 I/O."""
import math


DEVICE_RATE_STABLE_FRAMES = 5
DEVICE_RATE_FRAME_INTERVAL = 1/30


def _query_device_info(sounddevice_module, index, kind=None):
    if kind is None:
        return sounddevice_module.query_devices(index)
    try:
        return sounddevice_module.query_devices(index, kind)
    except TypeError:
        return sounddevice_module.query_devices(index)


def device_identity(sounddevice_module, index, kind=None):
    """Return a stable name/backend identity for a PortAudio device index."""
    device = _query_device_info(sounddevice_module, index, kind)
    try:
        hostapis = sounddevice_module.query_hostapis()
    except Exception:
        hostapis = ()
    hostapi_index = device.get('hostapi')
    try:
        hostapi_index = int(hostapi_index)
        hostapi = (hostapis[hostapi_index].get('name', '')
                   if 0 <= hostapi_index < len(hostapis) else '')
    except (TypeError, ValueError):
        hostapi = ''
    return {'name': str(device.get('name', '')),
            'hostapi': str(hostapi)}


def resolve_device_index(sounddevice_module, preferred, identity, kind):
    """Resolve an explicit device identity without falling back to defaults."""
    if identity is None:
        if preferred is None:
            raise ValueError(f'no selected {kind} device')
        return preferred
    devices = sounddevice_module.query_devices()
    channel_key = ('max_input_channels' if kind == 'input' else
                   'max_output_channels')
    if preferred is not None:
        try:
            candidate = sounddevice_module.query_devices(preferred)
            current = device_identity(sounddevice_module, preferred, kind)
            if (int(candidate.get(channel_key) or 0) > 0 and
                    current.get('name') == identity.get('name') and
                    (not identity.get('hostapi') or
                     current.get('hostapi') == identity.get('hostapi'))):
                return preferred
        except Exception:
            pass
    matches = []
    for index, candidate in enumerate(devices):
        if int(candidate.get(channel_key) or 0) < 1:
            continue
        try:
            current = device_identity(sounddevice_module, index, kind)
        except Exception:
            continue
        if (current.get('name') == identity.get('name') and
                (not identity.get('hostapi') or
                 current.get('hostapi') == identity.get('hostapi'))):
            matches.append(index)
    if len(matches) == 1:
        return matches[0]
    expected = identity.get('name') or f'the selected {kind} device'
    detail = ' is ambiguous' if matches else ' is unavailable'
    raise ValueError(f'{expected}{detail}; reselect the {kind} device')


def query_device_snapshot(sounddevice_module, preferred, identity, kind):
    """Return the resolved index, device details, identity, and native rate."""
    index = resolve_device_index(
        sounddevice_module, preferred, identity, kind)
    info = sounddevice_module.query_devices(index, kind)
    current_identity = device_identity(sounddevice_module, index, kind)
    rate = float(info.get('default_samplerate') or 0.0)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError(f'{current_identity.get("name") or kind} has no valid '
                         'reported sample rate')
    return index, info, current_identity, rate


class DeviceRateDebouncer:
    """Accept a device/rate pair only after consecutive stable observations."""

    def __init__(self, stable_frames=DEVICE_RATE_STABLE_FRAMES):
        self.stable_frames = max(1, int(stable_frames))
        self._candidate = None
        self._count = 0

    def reset(self):
        self._candidate = None
        self._count = 0

    def observe(self, identity, rate):
        try:
            rate = float(rate)
        except (TypeError, ValueError):
            self.reset()
            return False
        if not math.isfinite(rate) or rate <= 0 or not isinstance(identity, dict):
            self.reset()
            return False
        key = (str(identity.get('name', '')),
               str(identity.get('hostapi', '')), round(rate, 3))
        if key == self._candidate:
            self._count += 1
        else:
            self._candidate = key
            self._count = 1
        return self._count >= self.stable_frames

    @property
    def stable_count(self):
        return self._count

    @property
    def rate(self):
        return None if self._candidate is None else self._candidate[2]
