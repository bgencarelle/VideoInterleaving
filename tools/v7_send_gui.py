#!/usr/bin/env python3
"""Small, event-driven GUI controller for the standalone V7 sender.

The sender runs unchanged in its own process. The GUI has no capture preview,
per-frame polling, or verbose sender logging; it wakes on user input and the
sender's occasional startup/shutdown messages.
"""
import math
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RATE_CANDIDATES = (32000, 44100, 48000, 88200, 96000, 176400, 192000)
PROFILE_CHOICES = (
    ('Fold 500 · recommended', 'fold-500'),
    ('Fold 1000', 'fold-1000'),
    ('Baseline', 'baseline'),
    ('Experimental mono', 'mono'),
)
SOURCE_CHOICES = (
    ('Camera', 'camera'),
    ('Screen', 'screen'),
    ('Video file / stream', 'video'),
    ('Test pattern', 'test'),
    ('Mouse-follow', 'mouse-follow'),
)
FILTER_CHOICES = (
    ('Profile default', 'auto'),
    ('Box', 'box'),
    ('Nearest', 'nearest'),
    ('Lanczos', 'lanczos'),
    ('Bicubic', 'bicubic'),
)
CAPTURE_FILTER_CHOICES = (
    ('Automatic', 'auto'),
    ('Neighbor', 'neighbor'),
    ('Area', 'area'),
    ('Bilinear', 'bilinear'),
    ('Bicubic', 'bicubic'),
    ('Lanczos', 'lanczos'),
)

FIELD_HELP = {
    'device': 'Choose the explicit audio output device that feeds the receiver or recording path.',
    'source': 'Choose what the sender captures. Capture starts only after Start.',
    'rate': 'Audio output sample rate. Native uses the device clock; this is separate from Capture FPS.',
    'profile': 'Choose the wire profile. The receiver must use the matching fold profile.',
    'speed': 'Playback speed from 0.25× to 4×. Faster playback raises the transmitted carrier frequencies.',
    'encode_filter': 'Resize filter. Folded profiles require Box; Profile default selects the profile recommendation.',
    'brightness': 'Optional source brightness multiplier. Blank keeps the profile default.',
    'gamma': 'Source gamma; 1.0 is neutral.',
    'capture_fps': 'Optional capture pacing rate. Blank uses the source-specific default.',
    'video_source': 'Video file path or FFmpeg-supported URL. You can type a path or drop a file on the window.',
    'video_live': 'Treat an HTTP(S) video URL as a live stream rather than a looping clip.',
    'camera': 'Camera index passed to the capture backend (default 0).',
    'ffmpeg_input': 'Optional FFmpeg input specification for a particular camera or display backend.',
    'screen_backend': 'mss is the simple native screen capture path; FFmpeg can be useful when capture rate matters.',
    'display': 'Optional display index for FFmpeg screen capture.',
    'region': 'Optional screen crop as left,top,width,height.',
    'capture_width': 'Capture width for screen/video and initial mouse-follow crop; camera is sampled at 80×96.',
    'capture_filter': 'Optional FFmpeg capture scaler. Automatic follows the sender defaults.',
    'mono_sum': 'Send mono-summed content on one output channel instead of stereo.',
    'pilot_tones': 'Pilot references are required by folded-coded profiles.',
    'eof_marker': 'Packet end marker. The experimental mono profile requires it.',
}
FIELD_LABELS = {
    'device': 'Audio output device',
    'source': 'Capture source',
    'rate': 'Audio output sample rate',
    'capture_fps': 'Capture FPS',
    'video_live': 'Treat URL as live',
    'ffmpeg_input': 'FFmpeg input',
    'mono_sum': 'Mono output',
    'pilot_tones': 'Pilot tones',
    'eof_marker': 'EOF marker',
}


class OutputDevice:
    def __init__(self, index, name, channels, default_rate):
        self.index = int(index)
        self.name = str(name)
        self.channels = int(channels)
        self.default_rate = int(round(float(default_rate or 0)))

    @property
    def label(self):
        rate = f'{self.default_rate/1000:g} kHz' if self.default_rate else 'rate unknown'
        return (f'{self.index}: {self.name} · {self.channels} out · '
                f'{rate}')


def output_devices(sd_module=None):
    """Return output-capable devices without selecting the system default."""
    if sd_module is None:
        import sounddevice as sd_module
    devices = sd_module.query_devices()
    result = []
    for index, device in enumerate(devices):
        channels = int(device.get('max_output_channels') or 0)
        if channels:
            result.append(OutputDevice(
                index, device.get('name', f'Audio device {index}'), channels,
                device.get('default_samplerate')))
    return tuple(result)


def _rate_label(rate):
    if rate is None:
        return 'Native (device clock)'
    return f'{float(rate)/1000:g} kHz'


def sample_rate_options(device, channels, sd_module=None):
    """List common rates the selected device reports as valid, without opening it."""
    if device is None:
        return (('Native (device clock)', None),)
    if sd_module is None:
        import sounddevice as sd_module

    candidates = []
    if device.default_rate > 0:
        candidates.append(device.default_rate)
    candidates.extend(RATE_CANDIDATES)
    rates = []
    for rate in dict.fromkeys(candidates):
        try:
            sd_module.check_output_settings(
                device=device.index, channels=channels, dtype='float32',
                samplerate=rate)
        except Exception:
            continue
        rates.append(rate)
    return (('Native (device clock)', None),) + tuple(
        (_rate_label(rate), rate) for rate in sorted(rates)) + (
            ('Custom…', 'custom'),)


def _float_setting(value, label, optional=False):
    if optional and not str(value).strip():
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{label} must be a number.') from exc
    if not math.isfinite(number):
        raise ValueError(f'{label} must be finite.')
    return number


def validate_settings(settings, devices, sd_module=None):
    """Validate sender GUI state before launching the sender process."""
    by_index = {device.index: device for device in devices}
    device = by_index.get(settings.get('device'))
    if device is None:
        raise ValueError('Select an audio output device before starting.')
    source = settings.get('source')
    if source not in dict(SOURCE_CHOICES).values():
        raise ValueError('Choose a capture source before starting.')
    channels = 1 if settings.get('mono_sum') else 2
    if device.channels < channels:
        raise ValueError(
            f'{device.name} supports {device.channels} output channel(s); '
            f'this configuration needs {channels}. Enable mono output or '
            'choose another device.')

    rate = settings.get('rate')
    if rate is not None:
        try:
            rate = int(rate)
        except (TypeError, ValueError) as exc:
            raise ValueError('Audio sample rate must be a positive integer.') from exc
        if rate <= 0:
            raise ValueError('Audio sample rate must be a positive integer.')
    if sd_module is None:
        import sounddevice as sd_module
    try:
        sd_module.check_output_settings(
            device=device.index, channels=channels, dtype='float32',
            samplerate=rate or device.default_rate or None)
    except Exception as exc:
        sample_text = f'{rate} Hz' if rate else 'the device native rate'
        raise ValueError(
            f'{device.name} cannot open {channels} channel(s) at {sample_text}: {exc}') from exc

    profile = settings.get('profile')
    if profile not in dict(PROFILE_CHOICES).values():
        raise ValueError('Choose a supported wire profile.')
    encode_filter = settings.get('encode_filter', 'auto')
    if encode_filter not in dict(FILTER_CHOICES).values():
        raise ValueError('Choose a supported encode filter.')
    if encode_filter == 'auto':
        encode_filter = 'box' if profile in ('fold-500', 'fold-1000') else 'nearest'
    if profile in ('fold-500', 'fold-1000') and encode_filter != 'box':
        raise ValueError('Folded profiles require the Box encode filter.')
    if profile in ('fold-500', 'fold-1000') and not settings.get('pilot_tones', True):
        raise ValueError('Folded profiles require pilot tones.')
    screen_backend = settings.get('screen_backend', 'mss')
    if source == 'screen' and screen_backend not in ('mss', 'ffmpeg'):
        raise ValueError('Choose a supported screen capture backend.')
    if settings.get('capture_filter', 'auto') not in dict(CAPTURE_FILTER_CHOICES).values():
        raise ValueError('Choose a supported capture filter.')
    if profile == 'mono':
        if not settings.get('pilot_tones', True):
            raise ValueError('Experimental mono requires pilot tones.')
        if not settings.get('eof_marker', True):
            raise ValueError('Experimental mono requires the EOF marker.')

    speed = _float_setting(settings.get('speed', '1'), 'Speed')
    if not .25 <= speed <= 4.0:
        raise ValueError('Speed must be between 0.25 and 4.0.')
    brightness = _float_setting(
        settings.get('brightness', ''), 'Brightness', optional=True)
    if brightness is not None and brightness <= 0:
        raise ValueError('Brightness must be positive.')
    gamma = _float_setting(settings.get('gamma', '1'), 'Gamma')
    if gamma <= 0:
        raise ValueError('Gamma must be positive.')
    capture_fps = _float_setting(
        settings.get('capture_fps', ''), 'Capture FPS', optional=True)
    if capture_fps is not None and capture_fps <= 0:
        raise ValueError('Capture FPS must be positive.')

    video_source = str(settings.get('video_source', '')).strip()
    if source == 'video' and not video_source:
        raise ValueError('Set a video file path or stream URL.')
    camera = (_integer_setting(settings.get('camera', '0'), 'Camera index', 0)
              if source == 'camera' else 0)
    capture_width = (
        _integer_setting(settings.get('capture_width', '160'), 'Capture width', 1)
        if source in ('screen', 'video', 'mouse-follow') else 160)
    display = (_integer_setting(settings.get('display', ''), 'Display index', 0,
                                optional=True)
               if source == 'screen' else None)
    region = (str(settings.get('region', '')).strip()
              if source == 'screen' else '')
    if region:
        try:
            parts = tuple(int(part.strip()) for part in region.split(','))
        except ValueError as exc:
            raise ValueError('Screen region must be left,top,width,height.') from exc
        if len(parts) != 4 or parts[2] < 1 or parts[3] < 1:
            raise ValueError('Screen region must be left,top,width,height with positive size.')

    return {
        'device': device,
        'source': source,
        'channels': channels,
        'rate': rate,
        'profile': profile,
        'encode_filter': encode_filter,
        'speed': speed,
        'brightness': brightness,
        'gamma': gamma,
        'capture_fps': capture_fps,
        'video_source': video_source,
        'camera': camera,
        'ffmpeg_input': (str(settings.get('ffmpeg_input', '')).strip()
                         if (source == 'camera' or
                             source == 'screen' and screen_backend == 'ffmpeg')
                         else ''),
        'screen_backend': screen_backend,
        'capture_width': capture_width,
        'display': display,
        'region': region,
    }


def _integer_setting(value, label, minimum, optional=False):
    if optional and not str(value).strip():
        return None
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{label} must be an integer.') from exc
    if result < minimum:
        raise ValueError(f'{label} must be at least {minimum}.')
    return result


def build_command(settings, devices, sd_module=None, python=None):
    """Build an argv list for the existing V7 CLI; never invokes a shell."""
    checked = validate_settings(settings, devices, sd_module)
    command = [
        python or sys.executable,
        str(ROOT/'tools'/'v7_live.py'),
        'send',
        '--device', str(checked['device'].index),
        '--source', checked['source'],
    ]

    if checked['profile'] == 'fold-500':
        command.extend(('--experimental-fold', '500'))
    elif checked['profile'] == 'fold-1000':
        command.extend(('--experimental-fold', '1000'))
    elif checked['profile'] == 'baseline':
        command.extend(('--experimental-fold', '0'))
    else:
        command.append('--experimental-mono')

    if checked['rate'] is not None:
        command.extend(('--rate', str(checked['rate'])))
    if checked['speed'] != 1.0:
        command.extend(('--speed', str(checked['speed'])))
    if checked['encode_filter'] != ('box' if checked['profile'] in (
            'fold-500', 'fold-1000') else 'nearest'):
        command.extend(('--encode-filter', checked['encode_filter']))
    if checked['brightness'] is not None:
        command.extend(('--brightness', str(checked['brightness'])))
    if checked['gamma'] != 1.0:
        command.extend(('--gamma', str(checked['gamma'])))
    if checked['capture_fps'] is not None:
        command.extend(('--capture-fps', str(checked['capture_fps'])))

    if checked['source'] == 'video':
        command.extend(('--video-source', checked['video_source']))
        if settings.get('video_live'):
            command.append('--video-live')
    elif checked['source'] == 'camera':
        command.extend(('--camera', str(checked['camera'])))
        if checked['ffmpeg_input']:
            command.extend(('--ffmpeg-input', checked['ffmpeg_input']))
    elif checked['source'] == 'screen':
        command.extend(('--screen-backend', checked['screen_backend']))
        if checked['screen_backend'] == 'ffmpeg' and checked['ffmpeg_input']:
            command.extend(('--ffmpeg-input', checked['ffmpeg_input']))
        if checked['display'] is not None:
            command.extend(('--display', str(checked['display'])))
        if checked['region']:
            command.extend(('--region', checked['region']))

    if checked['source'] in ('screen', 'video', 'mouse-follow'):
        command.extend(('--capture-width', str(checked['capture_width'])))
    capture_filter = settings.get('capture_filter', 'auto')
    if capture_filter != 'auto':
        command.extend(('--capture-filter', capture_filter))
    if settings.get('mono_sum'):
        command.append('--mono-sum')
    if not settings.get('pilot_tones', True):
        command.append('--no-pilot-tones')
    if not settings.get('eof_marker', True):
        command.append('--no-eof-marker')
    return command


def _font(size):
    from PIL import ImageFont
    for name in ('DejaVuSans.ttf', 'Arial.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _fit(text, font, width):
    text = str(text)
    while text and font.getlength(text) > width:
        text = text[:-2]+'…'
    return text


def _wrapped(text, font, width):
    words = str(text).split()
    lines, line = [], ''
    for word in words:
        trial = (line+' '+word).strip()
        if line and font.getlength(trial) > width:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    return lines


class SenderGui:
    WINDOW_SIZE = (960, 720)
    TOOLBAR = 54
    ROW_HEIGHT = 39
    BASIC_FIELDS = (
        'device', 'source', 'rate', 'profile', 'speed', 'encode_filter',
        'video_source', 'video_live', 'camera',
    )
    ADVANCED_FIELDS = (
        'capture_fps', 'brightness', 'gamma', 'screen_backend', 'display',
        'region', 'ffmpeg_input', 'capture_width', 'capture_filter', 'mono_sum',
        'pilot_tones', 'eof_marker',
    )

    def __init__(self, devices=(), device_error=''):
        self.devices = tuple(devices)
        self.device_error = device_error
        self.settings = {
            'device': None,
            'source': None,
            'rate': None,
            'profile': 'fold-500',
            'speed': '1',
            'encode_filter': 'auto',
            'brightness': '',
            'gamma': '1',
            'capture_fps': '',
            'video_source': '',
            'video_live': False,
            'camera': '0',
            'ffmpeg_input': '',
            'screen_backend': 'mss',
            'display': '',
            'region': '',
            'capture_width': '160',
            'capture_filter': 'auto',
            'mono_sum': False,
            'pilot_tones': True,
            'eof_marker': True,
        }
        self.page = 'setup'
        self.advanced = False
        self.selected = 'device'
        self.scroll = 0
        self.dropdown = None
        self.dropdown_scroll = 0
        self.editing = False
        self.edit_buffer = ''
        self.notice = 'Choose an output device, capture source, and profile.'
        self.lines = []
        self.events = queue.Queue()
        self.process = None
        self.reader = None
        self.stop_requested = False
        self.close_when_stopped = False
        self.dirty = True
        self.hits = {}
        self.width, self.height = self.WINDOW_SIZE
        self._glfw = None
        self._window = None
        self._sd = None
        self.rate_cache = {}

    def _device(self):
        return next((device for device in self.devices
                     if device.index == self.settings['device']), None)

    def _choices(self, dest):
        if dest == 'device':
            return tuple((device.label, device.index) for device in self.devices)
        if dest == 'source':
            return SOURCE_CHOICES
        if dest == 'rate':
            channels = 1 if self.settings['mono_sum'] else 2
            device = self._device()
            key = (None if device is None else device.index, channels)
            if key in self.rate_cache:
                return self.rate_cache[key]
            try:
                options = sample_rate_options(device, channels, self._sounddevice())
            except Exception as exc:
                self.notice = f'Cannot query supported sample rates: {exc}'
                options = (('Native (device clock)', None),)
            self.rate_cache[key] = options
            return options
        if dest == 'profile':
            return PROFILE_CHOICES
        if dest == 'encode_filter':
            if self.settings['profile'] in ('fold-500', 'fold-1000'):
                return FILTER_CHOICES[:2]
            return FILTER_CHOICES
        if dest == 'screen_backend':
            return (('mss · lightweight', 'mss'), ('FFmpeg', 'ffmpeg'))
        if dest == 'capture_filter':
            return CAPTURE_FILTER_CHOICES
        return ()

    def _sounddevice(self):
        if self._sd is None:
            import sounddevice as sd
            self._sd = sd
        return self._sd

    def _visible_fields(self):
        fields = list(self.BASIC_FIELDS)
        if self.advanced:
            fields.extend(self.ADVANCED_FIELDS)
        source = self.settings['source']
        fields = [dest for dest in fields if not (
            dest == 'video_source' and source != 'video' or
            dest == 'video_live' and source != 'video' or
            dest == 'camera' and source != 'camera' or
            dest == 'ffmpeg_input' and source not in ('camera', 'screen') or
            dest == 'ffmpeg_input' and source == 'screen' and
            self.settings['screen_backend'] != 'ffmpeg' or
            dest in ('screen_backend', 'display', 'region') and source != 'screen' or
            dest == 'capture_width' and source not in ('screen', 'video', 'mouse-follow'))]
        return fields

    def _value_label(self, dest):
        value = self.settings[dest]
        if dest == 'device':
            device = self._device()
            return device.label if device else 'Select output device…'
        if dest in ('source', 'profile', 'encode_filter', 'screen_backend',
                    'capture_filter'):
            choices = (SOURCE_CHOICES if dest == 'source' else
                       PROFILE_CHOICES if dest == 'profile' else
                       FILTER_CHOICES if dest == 'encode_filter' else
                       CAPTURE_FILTER_CHOICES if dest == 'capture_filter' else
                       (('mss · lightweight', 'mss'), ('FFmpeg', 'ffmpeg')))
            label = next((label for label, candidate in choices
                          if candidate == value), None)
            if label is None:
                return 'Profile default' if value == 'auto' else 'Choose…'
            return label
        if dest == 'rate':
            if value is None:
                device = self._device()
                rate = f' · {device.default_rate/1000:g} kHz' if device and device.default_rate else ''
                return 'Native'+rate
            if value == 'custom':
                return 'Enter custom rate in Hz…'
            try:
                return _rate_label(int(value))
            except (TypeError, ValueError):
                return str(value)
        if isinstance(value, bool):
            return 'On' if value else 'Off'
        return str(value) if str(value).strip() else '(not set)'

    def _profile_changed(self, value):
        if self.settings['encode_filter'] == 'auto':
            return
        if value in ('fold-500', 'fold-1000') and self.settings['encode_filter'] != 'box':
            self.settings['encode_filter'] = 'auto'

    def _assign(self, dest, value):
        self.settings[dest] = value
        if dest == 'profile':
            self._profile_changed(value)
        elif dest == 'device':
            self.settings['rate'] = None
        elif dest == 'mono_sum':
            options = self._choices('rate')
            if self.settings['rate'] not in {item[1] for item in options}:
                self.settings['rate'] = None
        elif dest == 'source':
            self.dropdown = None
        self.notice = f'{dest.replace("_", " ").capitalize()} updated.'
        self.dirty = True

    def _select_option(self, dest, value):
        if dest == 'rate' and value == 'custom':
            self.settings['rate'] = None
            self.selected = 'rate'
            self.editing = True
            self.edit_buffer = ''
            self.notice = 'Enter the requested audio sample rate in Hz.'
            self.dropdown = None
            self.dirty = True
            return
        self._assign(dest, value)

    def _build_command(self):
        return build_command(self.settings, self.devices, self._sounddevice())

    def _start(self):
        if self.editing:
            self._finish_edit()
        if self.process is not None:
            return
        try:
            command = self._build_command()
            kwargs = {
                'cwd': str(ROOT),
                'stdin': subprocess.DEVNULL,
                'stdout': subprocess.PIPE,
                'stderr': subprocess.STDOUT,
                'text': True,
                'encoding': 'utf-8',
                'errors': 'replace',
                'bufsize': 1,
            }
            if os.name == 'nt':
                kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs['start_new_session'] = True
            process = subprocess.Popen(command, **kwargs)
        except (ImportError, ValueError, OSError, RuntimeError) as exc:
            self.notice = str(exc)
            self.dirty = True
            return

        self.process = process
        self.stop_requested = False
        self.page = 'live'
        self.notice = 'Starting sender…'
        self.lines.clear()
        self.reader = threading.Thread(
            target=self._read_sender, args=(process,),
            name='v7-send-gui-output', daemon=True)
        self.reader.start()
        self.dirty = True

    def _read_sender(self, process):
        try:
            for line in process.stdout:
                line = line.strip()
                if line:
                    self.events.put(('line', line))
                    self._wake()
        except Exception as exc:
            self.events.put(('line', f'Sender output read failed: {exc}'))
            self._wake()
        return_code = process.wait()
        self.events.put(('exit', return_code))
        self._wake()

    def _wake(self):
        glfw = self._glfw
        if glfw is not None:
            try:
                glfw.post_empty_event()
            except Exception:
                pass

    @staticmethod
    def _stop_after_timeout(process, events, wake):
        try:
            process.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            pass
        events.put(('line', 'Sender did not stop promptly; requesting termination.'))
        wake()
        try:
            process.terminate()
        except OSError:
            return
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except OSError:
                pass

    def _stop(self):
        process = self.process
        if process is None or process.poll() is not None or self.stop_requested:
            return
        self.stop_requested = True
        self.notice = 'Stopping sender and closing its audio stream…'
        self.dirty = True
        try:
            if os.name == 'nt':
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process.send_signal(signal.SIGINT)
        except (OSError, ValueError):
            try:
                process.terminate()
            except OSError:
                pass
        threading.Thread(
            target=self._stop_after_timeout,
            args=(process, self.events, self._wake),
            name='v7-send-gui-stop-timeout', daemon=True).start()

    def _drain_events(self):
        changed = False
        while True:
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            changed = True
            if kind == 'line':
                self.lines.append(value)
                self.lines = self.lines[-12:]
                self.notice = value
            elif kind == 'exit':
                return_code = int(value)
                self.process = None
                self.stop_requested = False
                if return_code == 0:
                    self.notice = 'Sender stopped.'
                else:
                    self.notice = f'Sender exited with status {return_code}.'
                if self.close_when_stopped and self._window is not None:
                    self._glfw.set_window_should_close(self._window, True)
        if changed:
            self.dirty = True

    def _finish_edit(self, commit=True):
        dest = self.selected
        if commit:
            self.settings[dest] = self.edit_buffer
            self.notice = f'{dest.replace("_", " ").capitalize()} updated.'
        self.editing = False
        self.dirty = True

    def _render_setup(self, image, draw, font, small):
        width, height = image.size
        draw.text((24, 76), 'Configure a sender', font=font,
                  fill=(229, 237, 243))
        draw.text((24, 101), 'No capture or audio stream opens until Start.',
                  font=small, fill=(133, 159, 177))
        fields = self._visible_fields()
        row_top, row_bottom = 137, height-131
        visible_count = max(1, (row_bottom-row_top)//self.ROW_HEIGHT)
        self.scroll = max(0, min(self.scroll, max(0, len(fields)-visible_count)))
        if self.selected not in fields and fields:
            self.selected = fields[0]
        shown_fields = fields[self.scroll:self.scroll+visible_count]
        for visible_index, dest in enumerate(shown_fields):
            y = row_top+visible_index*self.ROW_HEIGHT
            selected = dest == self.selected
            fill = (35, 60, 77) if selected else (17, 29, 39)
            draw.rounded_rectangle((18, y, width-18, y+self.ROW_HEIGHT-3),
                                   radius=4, fill=fill,
                                   outline=(74, 111, 134) if selected else (32, 48, 60),
                                   width=1)
            label = FIELD_LABELS.get(
                dest, dest.replace('_', ' ').capitalize())
            value_left = max(300, int(width*.37))
            draw.text((30, y+10), label, font=small, fill=(205, 218, 228))
            value = self.edit_buffer if self.editing and dest == self.selected else self._value_label(dest)
            value = _fit(value, small, width-value_left-45)
            draw.text((value_left, y+10), value,
                      font=small, fill=(237, 242, 246))
            draw.text((width-40, y+9), '▾' if dest in (
                'device', 'source', 'rate', 'profile', 'encode_filter',
                'screen_backend', 'capture_filter') else '',
                font=small, fill=(134, 169, 188))
            self.hits[f'field:{dest}'] = (18, y, width-18, y+self.ROW_HEIGHT-3)

        toggle_y = height-123
        draw.rounded_rectangle((22, toggle_y, 154, toggle_y+28), radius=4,
                               fill=(24, 41, 54), outline=(63, 91, 108))
        draw.text((34, toggle_y+6),
                  'Advanced  On' if self.advanced else 'Advanced  Off',
                  font=small, fill=(205, 218, 228))
        self.hits['advanced'] = (22, toggle_y, 154, toggle_y+28)

        help_text = FIELD_HELP.get(self.selected, '')
        if self.selected == 'rate':
            help_text = FIELD_HELP['rate']
        help_top = height-87
        lines = _wrapped(help_text, small, width-48)
        for index, line in enumerate(lines[:2]):
            draw.text((24, help_top+index*17), line, font=small,
                      fill=(125, 150, 168))

        if self.dropdown is not None:
            self._render_dropdown(draw, small, width, height)

    def _render_dropdown(self, draw, small, width, height):
        dest = self.dropdown
        options = self._choices(dest)
        max_items = min(8, len(options))
        self.dropdown_scroll = max(
            0, min(self.dropdown_scroll, max(0, len(options)-max_items)))
        anchor = self.hits.get(f'field:{dest}')
        if anchor is None:
            return
        left = max(280, int(width*.37))
        right = width-25
        top = anchor[1]+self.ROW_HEIGHT-2
        if top+max_items*28 > height-100:
            top = max(60, anchor[1]-max_items*28)
        draw.rounded_rectangle((left, top, right, top+max_items*28+4),
                               radius=4, fill=(9, 18, 25),
                               outline=(98, 145, 169), width=1)
        for index in range(max_items):
            option_index = self.dropdown_scroll+index
            label, value = options[option_index]
            y = top+2+index*28
            if value == self.settings[dest]:
                draw.rectangle((left+2, y, right-2, y+26), fill=(42, 78, 99))
            draw.text((left+9, y+6), _fit(label, small, right-left-18),
                      font=small, fill=(235, 241, 246))
            self.hits[f'option:{option_index}'] = (left, y, right, y+26)

    def _render_live(self, image, draw, font, small):
        width, height = image.size
        draw.text((24, 76), 'Sender status', font=font,
                  fill=(229, 237, 243))
        state = ('STOPPING' if self.stop_requested else
                 'SENDING' if self.process is not None else 'STOPPED')
        draw.rounded_rectangle((24, 119, width-24, 188), radius=6,
                               fill=(17, 29, 39), outline=(48, 73, 90))
        draw.text((42, 135), state, font=font,
                  fill=(238, 182, 125) if state == 'STOPPING' else
                  (145, 218, 170) if state == 'SENDING' else (188, 202, 213))
        draw.text((42, 165), _fit(self.notice, small, width-84),
                  font=small, fill=(165, 187, 202))

        device = self._device()
        profile_label = next((label for label, value in PROFILE_CHOICES
                              if value == self.settings['profile']), 'Unknown')
        source_label = next((label for label, value in SOURCE_CHOICES
                             if value == self.settings['source']), 'Not selected')
        rate = self.settings['rate']
        if rate is None and device is not None:
            rate_text = (f'Native · {device.default_rate/1000:g} kHz'
                         if device.default_rate else 'Native')
        elif rate is None:
            rate_text = 'Native'
        else:
            rate_text = _rate_label(rate)
        details = (
            ('Source', source_label),
            ('Output device', device.name if device else 'Not selected'),
            ('Sample rate', rate_text),
            ('Wire profile', profile_label),
            ('Speed', f"{self.settings['speed']}×"),
        )
        y = 220
        for label, value in details:
            draw.text((32, y), label, font=small, fill=(132, 158, 176))
            draw.text((235, y), _fit(value, small, width-265),
                      font=small, fill=(218, 229, 237))
            y += 32

        log_top = y+12
        draw.text((24, log_top), 'Sender messages', font=small,
                  fill=(132, 158, 176))
        first_line = log_top+24
        line_count = max(0, min(10, (height-34-first_line)//22))
        for index, line in enumerate(self.lines[-line_count:] if line_count else ()):
            shown = _fit(line, small, width-48)
            draw.text((28, first_line+index*22), shown,
                      font=small, fill=(183, 201, 214))

    def _canvas(self, size):
        from PIL import Image, ImageDraw
        width, height = size
        image = Image.new('RGBA', (width, height), (8, 14, 20, 255))
        draw = ImageDraw.Draw(image)
        font, small = _font(18), _font(13)
        self.hits = {}
        draw.rectangle((0, 0, width, self.TOOLBAR), fill=(10, 18, 25, 255))
        draw.rectangle((0, self.TOOLBAR-1, width, self.TOOLBAR),
                       fill=(47, 68, 83, 255))
        controls = (
            ('setup', 'Setup', 14, 100),
            ('live', 'Live', 108, 180),
            ('start_stop', 'Stop' if self.process is not None else 'Start',
             width-202, width-108),
            ('close', 'Close', width-98, width-12),
        )
        for key, label, left, right in controls:
            rect = (left, 9, right, 45)
            self.hits[key] = rect
            active = ((key == 'setup' and self.page == 'setup') or
                      (key == 'live' and self.page == 'live'))
            color = ((39, 67, 86) if active else
                     (82, 55, 40) if key == 'start_stop' and self.process else
                     (43, 94, 123) if key == 'start_stop' else (22, 35, 46))
            draw.rounded_rectangle(rect, radius=5, fill=color,
                                   outline=(67, 100, 122), width=1)
            draw.text((left+10, 19), label, font=small,
                      fill=(246, 240, 235) if key == 'start_stop' and self.process
                      else (236, 242, 247))

        if self.page == 'setup':
            self._render_setup(image, draw, font, small)
        else:
            self._render_live(image, draw, font, small)

        footer_top = height-34
        draw.rectangle((0, footer_top, width, height), fill=(10, 18, 25))
        status = self.device_error if self.device_error else self.notice
        draw.text((14, footer_top+10), _fit(status, small, width-28),
                  font=small, fill=(255, 182, 132) if self.device_error
                  else (147, 206, 169) if self.process else (167, 187, 202))
        return image

    def _on_mouse(self, glfw, window, button, action, _mods):
        if button != glfw.MOUSE_BUTTON_LEFT or action != glfw.PRESS:
            return
        x, y = glfw.get_cursor_pos(window)
        ordered = list(self.hits)
        if self.dropdown is not None:
            ordered.sort(key=lambda key: 0 if key.startswith('option:') else 1)
        hit = next((key for key in ordered
                    if self.hits[key][0] <= x < self.hits[key][2] and
                    self.hits[key][1] <= y < self.hits[key][3]), None)
        if self.editing:
            self._finish_edit()
        if hit == 'setup':
            self.page, self.dropdown = 'setup', None
        elif hit == 'live':
            self.page, self.dropdown = 'live', None
        elif hit == 'close':
            if self.process is not None:
                self.close_when_stopped = True
                self._stop()
            else:
                glfw.set_window_should_close(window, True)
        elif hit == 'start_stop':
            self._stop() if self.process is not None else self._start()
        elif hit == 'advanced':
            if self.process is not None:
                self.notice = 'Settings are locked while the sender is running.'
            else:
                self.advanced = not self.advanced
                self.scroll = 0
                self.dropdown = None
        elif hit and hit.startswith('option:') and self.dropdown is not None:
            index = int(hit.split(':', 1)[1])
            choices = self._choices(self.dropdown)
            if 0 <= index < len(choices):
                self._select_option(self.dropdown, choices[index][1])
            self.dropdown = None
        elif hit and hit.startswith('field:') and self.page == 'setup':
            dest = hit.split(':', 1)[1]
            self.selected = dest
            if self.process is not None:
                self.notice = 'Settings are locked while the sender is running.'
            elif dest in ('video_live', 'mono_sum', 'pilot_tones', 'eof_marker'):
                self._assign(dest, not self.settings[dest])
            elif dest in ('device', 'source', 'rate', 'profile',
                          'encode_filter', 'screen_backend', 'capture_filter'):
                self.dropdown = dest
                self.dropdown_scroll = 0
            elif dest in self._visible_fields():
                self.editing = True
                current = self.settings[dest]
                self.edit_buffer = '' if current is None else str(current)
        else:
            self.dropdown = None
        self.dirty = True

    def _on_key(self, glfw, window, key, _scancode, action, mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if self.editing:
            if key in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER):
                self._finish_edit()
            elif key == glfw.KEY_ESCAPE:
                self._finish_edit(commit=False)
            elif key == glfw.KEY_BACKSPACE:
                self.edit_buffer = self.edit_buffer[:-1]
            elif key == glfw.KEY_A and mods & glfw.MOD_CONTROL:
                self.edit_buffer = ''
            elif key == glfw.KEY_V and mods & glfw.MOD_CONTROL:
                try:
                    self.edit_buffer += glfw.get_clipboard_string(window) or ''
                except Exception:
                    pass
            self.dirty = True
            return
        if key == glfw.KEY_ESCAPE:
            if self.dropdown is not None:
                self.dropdown = None
            elif self.process is not None:
                self._stop()
            else:
                glfw.set_window_should_close(window, True)
        elif key in (glfw.KEY_C, glfw.KEY_TAB):
            self.page = 'setup' if key == glfw.KEY_C else 'live'
            self.dropdown = None
        elif key == glfw.KEY_I:
            self.page = 'live'
            self.dropdown = None
        elif key == glfw.KEY_SPACE:
            self._stop() if self.process is not None else self._start()
        elif self.dropdown is not None:
            options = self._choices(self.dropdown)
            if key in (glfw.KEY_UP, glfw.KEY_DOWN):
                self.dropdown_scroll = max(
                    0, min(max(0, len(options)-1), self.dropdown_scroll+
                           (1 if key == glfw.KEY_DOWN else -1)))
            elif key in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER) and options:
                self._select_option(self.dropdown,
                                    options[self.dropdown_scroll][1])
                self.dropdown = None
            elif key == glfw.KEY_ESCAPE:
                self.dropdown = None
        elif self.page == 'setup':
            fields = self._visible_fields()
            if self.process is not None and key not in (glfw.KEY_UP, glfw.KEY_DOWN):
                self.notice = 'Settings are locked while the sender is running.'
                self.dirty = True
                return
            if key in (glfw.KEY_UP, glfw.KEY_DOWN) and fields:
                position = fields.index(self.selected) if self.selected in fields else 0
                position = max(0, min(len(fields)-1, position+
                                       (1 if key == glfw.KEY_DOWN else -1)))
                self.selected = fields[position]
                row_top, row_bottom = 137, self.height-131
                visible_count = max(1, (row_bottom-row_top)//self.ROW_HEIGHT)
                if position < self.scroll:
                    self.scroll = position
                elif position >= self.scroll+visible_count:
                    self.scroll = position-visible_count+1
            elif key in (glfw.KEY_LEFT, glfw.KEY_RIGHT) and self.selected in fields:
                dest = self.selected
                if dest in ('video_live', 'mono_sum', 'pilot_tones', 'eof_marker'):
                    self._assign(dest, not self.settings[dest])
                elif dest in ('device', 'source', 'rate', 'profile',
                              'encode_filter', 'screen_backend', 'capture_filter'):
                    self.dropdown = dest
                else:
                    self.editing = True
                    self.edit_buffer = str(self.settings.get(dest) or '')
            elif key in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER) and self.selected in fields:
                dest = self.selected
                if dest in ('device', 'source', 'rate', 'profile',
                            'encode_filter', 'screen_backend', 'capture_filter'):
                    self.dropdown = dest
                elif dest in ('video_live', 'mono_sum', 'pilot_tones', 'eof_marker'):
                    self._assign(dest, not self.settings[dest])
                else:
                    self.editing = True
                    self.edit_buffer = str(self.settings.get(dest) or '')
        self.dirty = True

    def _on_char(self, _window, codepoint):
        if self.editing and codepoint >= 32:
            self.edit_buffer += chr(codepoint)
            self.dirty = True

    def _on_scroll(self, _window, _xoffset, yoffset):
        if self.dropdown is not None:
            options = self._choices(self.dropdown)
            self.dropdown_scroll = max(
                0, min(max(0, len(options)-1), self.dropdown_scroll-
                       int(yoffset)))
        elif self.page == 'setup':
            visible_count = max(
                1, (self.height-131-137)//self.ROW_HEIGHT)
            self.scroll = max(
                0, min(max(0, len(self._visible_fields())-visible_count),
                       self.scroll-int(yoffset)))
        self.dirty = True

    def _on_drop(self, _window, paths):
        if paths and self.settings.get('source') == 'video':
            self.settings['video_source'] = paths[0]
            self.selected = 'video_source'
            self.notice = f'Video path selected: {Path(paths[0]).name}'
            self.dirty = True

    def run(self):
        import glfw
        import moderngl

        if not glfw.init():
            raise RuntimeError('GLFW initialization failed')
        window = context = program = vertex_array = texture = None
        try:
            glfw.default_window_hints()
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
            glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
            window = glfw.create_window(*self.WINDOW_SIZE,
                                        'V7 Sender · Setup', None, None)
            if not window:
                raise RuntimeError('GLFW could not create the V7 sender window')
            glfw.make_context_current(window)
            self._glfw, self._window = glfw, window
            glfw.set_window_size_limits(window, 720, 480,
                                        glfw.DONT_CARE, glfw.DONT_CARE)
            glfw.swap_interval(1)
            context = moderngl.create_context(require=330)
            program = context.program(
                vertex_shader='''#version 330
                out vec2 uv;
                void main() {
                    vec2 p[3] = vec2[3](vec2(-1.0, -1.0),
                                         vec2(3.0, -1.0),
                                         vec2(-1.0, 3.0));
                    vec2 pos = p[gl_VertexID];
                    uv = (pos + 1.0) * 0.5;
                    gl_Position = vec4(pos, 0.0, 1.0);
                }''',
                fragment_shader='''#version 330
                uniform sampler2D image;
                in vec2 uv;
                out vec4 color;
                void main() { color = texture(image, vec2(uv.x, 1.0-uv.y)); }
                ''')
            program['image'].value = 0
            vertex_array = context.vertex_array(program, [])
            glfw.set_mouse_button_callback(
                window, lambda w, b, a, m: self._on_mouse(glfw, w, b, a, m))
            glfw.set_key_callback(
                window, lambda w, k, s, a, m: self._on_key(glfw, w, k, s, a, m))
            glfw.set_char_callback(window, self._on_char)
            glfw.set_drop_callback(window, self._on_drop)
            glfw.set_scroll_callback(window, self._on_scroll)
            glfw.set_window_size_callback(
                window, lambda *_args: setattr(self, 'dirty', True))
            glfw.set_framebuffer_size_callback(
                window, lambda *_args: setattr(self, 'dirty', True))

            # The initial canvas is dirty before the first blocking wait.
            glfw.post_empty_event()
            while not glfw.window_should_close(window):
                glfw.wait_events()
                self._drain_events()
                if not self.dirty:
                    continue
                size = glfw.get_window_size(window)
                size = (max(320, int(size[0])), max(240, int(size[1])))
                self.width, self.height = size
                framebuffer_size = glfw.get_framebuffer_size(window)
                if not all(framebuffer_size):
                    continue
                canvas = self._canvas(size)
                rgba = canvas.tobytes()
                if texture is None or texture.size != size:
                    if texture is not None:
                        texture.release()
                    texture = context.texture(size, 4, rgba, dtype='f1')
                    texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
                    texture.repeat_x = False
                    texture.repeat_y = False
                else:
                    texture.write(rgba)
                context.viewport = (0, 0, *framebuffer_size)
                context.clear(.03, .05, .07, 1.0)
                texture.use(location=0)
                vertex_array.render(mode=moderngl.TRIANGLES, vertices=3)
                glfw.swap_buffers(window)
                self.dirty = False
                title = 'V7 Sender · '+('Sending' if self.process else 'Setup')
                glfw.set_window_title(window, title)
        finally:
            if self.process is not None:
                self.close_when_stopped = True
                self._stop()
                try:
                    self.process.wait(timeout=7)
                except subprocess.TimeoutExpired:
                    try:
                        self.process.terminate()
                    except OSError:
                        pass
                    try:
                        self.process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        try:
                            self.process.kill()
                        except OSError:
                            pass
                        self.process.wait()
            if self.reader is not None:
                self.reader.join(timeout=1)
            self._glfw = None
            self._window = None
            if texture is not None:
                texture.release()
            if vertex_array is not None:
                vertex_array.release()
            if program is not None:
                program.release()
            if context is not None:
                context.release()
            if window is not None:
                glfw.destroy_window(window)
            glfw.terminate()


def main():
    try:
        devices = output_devices()
        device_error = '' if devices else 'No audio output devices are available.'
    except Exception as exc:
        devices, device_error = (), f'Audio device enumeration failed: {exc}'
    SenderGui(devices, device_error).run()


if __name__ == '__main__':
    main()
