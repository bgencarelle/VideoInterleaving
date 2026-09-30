#!/usr/bin/env python3
"""Event-driven GUI controller for the standalone V7 sender.

The sender runs in its own process. Optional source-video playback uses a
desktop media player; encoded-image preview arrives on a separate bounded
loopback channel.
"""
import json
import math
import os
from pathlib import Path
import queue
import re
import signal
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.v7_preview_protocol import parse_preview_datagram


PROFILE_CHOICES = (
    ('Fold 500 stereo · recommended', 'fold-500'),
    ('Mono video · Fold 500', 'mono-fold-500'),
    ('Mono video · colour Fold 500 · experimental', 'mono-colour-500'),
    ('Fold 1000 · advanced', 'fold-1000'),
)
MONO_PROFILES = ('mono-fold-500', 'mono-colour-500')
PRIMARY_PROFILE_CHOICES = PROFILE_CHOICES[:3]
SOURCE_AUDIO_CHOICES = (
    ('Video soundtrack (if present)', 'source'),
    ('Input device', 'device'),
    ('Off · silence', 'off'),
)
SOURCE_AUDIO_SIDE_CHOICES = (
    ('Stereo downmix', 'mix'),
    ('Left input', 'left'),
    ('Right input', 'right'),
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
DOWNSCALER_CHOICES = (
    ('Off', 'off'),
    ('Linear box', 'linear-box'),
    ('Gamma detail', 'gamma-detail'),
    ('Linear detail', 'linear-detail'),
)
MONO_VIDEO_SIDE_CHOICES = (
    ('Left output · right stays clear', 'left'),
    ('Right output · left stays clear', 'right'),
)

FIELD_HELP = {
    'device': 'Choose the explicit audio output device that feeds the receiver or recording path.',
    'source': 'Choose what the sender captures. Capture starts only after Start.',
    'profile': ('Fold 500 stereo is recommended. Mono Fold 500 is available, '
                'and colour-weighted mono remains experimental.'),
    'speed': 'Playback speed from 0.25× to 4×. Faster playback raises the transmitted carrier frequencies.',
    'encode_filter': 'Resize filter. Folded profiles require Box; Profile default selects the profile recommendation.',
    'brightness': 'Live source brightness multiplier. 1.0 is neutral.',
    'gamma': 'Live source gamma; 1.0 is neutral.',
    'capture_fps': 'Choose a frame rate reported by the capture source, or leave it at Source default.',
    'video_source': 'Choose a video with Browse, type a path or URL, or drop a file on the window.',
    'video_preview': ('Open the same file or URL in muted ffplay. If ffplay is '
                      'unavailable, no player is opened.'),
    'image_preview': ('Show the captured source or resized encoder input. '
                      'Use the receiver to inspect encoded/decoded output.'),
    'video_live': 'Treat an HTTP(S) video URL as a live stream rather than a looping clip.',
    'camera': 'Choose a camera discovered from the host capture devices.',
    'screen_target': 'Choose the monitor or screen capture device. Discovery runs only when you open this picker.',
    'ffmpeg_input': ('Optional custom FFmpeg input for a camera or display. '
                     'Use it instead of choosing that device from the picker.'),
    'screen_backend': 'mss is the simple native screen capture path; FFmpeg can be useful when capture rate matters.',
    'region': 'Optional screen crop as left,top,width,height.',
    'capture_width': ('Intermediate FFmpeg width for video/FFmpeg screen and '
                      'initial mouse-follow crop; the sender then prepares the '
                      'fixed 80×96 wire image. Camera default capture is '
                      'sampled directly at 80×96.'),
    'capture_filter': 'Optional FFmpeg capture scaler. Automatic follows the sender defaults.',
    'mono_video_side': ('For the mono video profile, carry the modem on one '
                        'leg and leave the other free for separate audio.'),
    'source_audio': ('In mono-video mode, route the source soundtrack or an '
                     'input device to the free output leg, or choose Off. '
                     'This is separate from the muted source-player preview.'),
    'source_audio_device': 'Choose an explicit microphone, line, or loopback input device.',
    'source_audio_input_side': 'Select one input leg or downmix stereo input to mono.',
    'source_audio_gain': 'Gain applied only to source audio on the free output leg.',
    'source_audio_delay_ms': ('Additional sync delay beyond one emitted video '
                              'packet; zero is the low-latency starting point.'),
    'perceptual_resize': ('Experimental pre-encode downscaler. Requires '
                          'Fold 500, Fold 1000, or Mono video with the Box '
                          'encode filter.'),
    'perceptual_detail_strength': 'Strength for the selected pre-encode downscaler, from 0 to 1.',
}
FIELD_LABELS = {
    'device': 'Audio output device',
    'source': 'Capture source',
    'capture_fps': 'Capture FPS',
    'brightness': 'Brightness · live',
    'gamma': 'Gamma · live',
    'screen_target': 'Screen / display',
    'video_preview': 'Open source in player',
    'image_preview': 'Show image preview',
    'video_live': 'Treat URL as live',
    'camera': 'Camera',
    'ffmpeg_input': 'FFmpeg input',
    'mono_video_side': 'Mono video output side',
    'source_audio': 'Audio source',
    'source_audio_device': 'Audio input device',
    'source_audio_input_side': 'Audio input channels',
    'source_audio_gain': 'Source-audio gain',
    'source_audio_delay_ms': 'Additional audio delay (ms)',
    'perceptual_resize': 'Pre-encode downscaler',
    'perceptual_detail_strength': 'Downscaler strength',
}
VIDEO_FILE_GLOB = '*.mp4 *.m4v *.mov *.mkv *.webm *.avi *.mpeg *.mpg *.wmv *.ts'
DEVICE_REFRESH_SECONDS = 3.0
GUI_EVENT_WAIT_SECONDS = 0.5
SENDER_PREFERENCES_VERSION = 1


def sender_preferences_path():
    """Return the per-user V7 sender GUI preferences path."""
    root = Path(os.environ.get('XDG_CONFIG_HOME') or Path.home()/'.config')
    return root/'modemTest'/'v7_send_gui.json'


def _load_sender_preferences(path):
    try:
        values = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError):
        return {}
    if (not isinstance(values, dict) or
            values.get('version') != SENDER_PREFERENCES_VERSION):
        return {}
    settings = values.get('settings')

    def clean_identity(value):
        if (not isinstance(value, dict) or
                not isinstance(value.get('name'), str)):
            return None
        hostapi = value.get('hostapi', '')
        if not isinstance(hostapi, str):
            hostapi = ''
        return {'name': value['name'], 'hostapi': hostapi}

    return {'settings': settings if isinstance(settings, dict) else {},
            'output_device': clean_identity(values.get('output_device')),
            'source_audio_device': clean_identity(
                values.get('source_audio_device'))}


def _save_sender_preferences(values, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(
        values, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    try:
        temporary.chmod(0o600)
    except OSError:
        pass
    temporary.replace(path)


def _device_identity(device):
    if device is None:
        return None
    return {'name': device.name, 'hostapi': device.hostapi}


def _match_device(devices, identity):
    if not isinstance(identity, dict) or not isinstance(identity.get('name'), str):
        return None
    matches = [device for device in devices
               if device.name == identity['name'] and
               (not identity.get('hostapi') or
                device.hostapi == identity.get('hostapi'))]
    return matches[0] if len(matches) == 1 else None


SAVED_SETTING_FIELDS = (
    'source', 'profile', 'mono_video_side', 'source_audio',
    'source_audio_input_side', 'source_audio_gain',
    'source_audio_delay_ms', 'speed', 'encode_filter', 'brightness', 'gamma',
    'capture_fps', 'video_source', 'video_preview', 'video_live', 'camera',
    'image_preview', 'preview_stage',
    'ffmpeg_input', 'screen_backend', 'screen_target', 'region',
    'capture_width', 'capture_filter', 'perceptual_resize',
    'perceptual_detail_strength',
)


def _serialize_sender_settings(settings):
    saved = {}
    for key in SAVED_SETTING_FIELDS:
        value = settings.get(key)
        if isinstance(value, ScreenTarget):
            value = {'label': value.label, 'region': value.region,
                     'display': value.display}
        if value is None or isinstance(value, (str, int, float, bool, dict)):
            saved[key] = value
    return saved


def _restore_sender_settings(target, saved):
    if not isinstance(saved, dict):
        return
    for key in SAVED_SETTING_FIELDS:
        if key not in saved:
            continue
        value = saved[key]
        if key == 'screen_target':
            if isinstance(value, dict):
                try:
                    target[key] = ScreenTarget(
                        str(value.get('label', 'Saved display')),
                        str(value.get('region', '')),
                        (None if value.get('display') is None else
                         int(value['display'])))
                except (TypeError, ValueError):
                    continue
            elif value is None:
                target[key] = None
            continue
        if key in ('video_preview', 'video_live', 'image_preview'):
            if isinstance(value, bool):
                target[key] = value
        elif value is None or isinstance(value, (str, int, float, bool)):
            target[key] = value
    if 'image_preview' not in saved and isinstance(
            saved.get('encoded_preview'), bool):
        target['image_preview'] = saved['encoded_preview']
    if target.get('preview_stage') not in ('source', 'resized'):
        target['preview_stage'] = 'resized'


class OutputDevice:
    def __init__(self, index, name, channels, default_rate, hostapi=''):
        self.index = int(index)
        self.name = str(name)
        self.channels = int(channels)
        self.default_rate = int(round(float(default_rate or 0)))
        self.hostapi = str(hostapi or '')

    @property
    def label(self):
        rate = f'{self.default_rate/1000:g} kHz' if self.default_rate else 'rate unknown'
        return (f'{self.index}: {self.name} · {self.channels} out · '
                f'{rate}')


class InputDevice:
    def __init__(self, index, name, channels, default_rate, hostapi=''):
        self.index = int(index)
        self.name = str(name)
        self.channels = int(channels)
        self.default_rate = int(round(float(default_rate or 0)))
        self.hostapi = str(hostapi or '')

    @property
    def label(self):
        rate = f'{self.default_rate/1000:g} kHz' if self.default_rate else 'rate unknown'
        return (f'{self.index}: {self.name} · {self.channels} in · '
                f'{rate}')


@dataclass(frozen=True)
class ScreenTarget:
    label: str
    region: str = ''
    display: int | None = None


def linux_camera_sources(video_paths, names=None):
    """Build V4L2 choices from video nodes without opening a capture stream."""
    names = names or {}
    choices = []
    paths = [Path(path) for path in video_paths
             if re.fullmatch(r'video\d+', Path(path).name)]
    for path in sorted(paths, key=lambda item: int(item.name[5:])):
        name = names.get(str(path)) or path.name
        choices.append((f'{name} · {path}', f'v4l2:{path}'))
    return tuple(choices)


def _ffmpeg_devices(command, ffmpeg_path=None, run=None):
    ffmpeg_path = ffmpeg_path or shutil.which('ffmpeg')
    if not ffmpeg_path:
        raise RuntimeError('FFmpeg is required to enumerate capture devices.')
    run = subprocess.run if run is None else run
    result = run(command(ffmpeg_path), capture_output=True, text=True,
                 encoding='utf-8', errors='replace', timeout=10, check=False)
    return '\n'.join((getattr(result, 'stdout', '') or '',
                      getattr(result, 'stderr', '') or ''))


def _is_avfoundation_screen_source(name):
    return bool(re.fullmatch(r'capture screen\s+\d+', str(name).strip(),
                             re.IGNORECASE))


def parse_ffmpeg_camera_sources(output, platform):
    """Parse the camera sections of FFmpeg's AVFoundation / DirectShow listing."""
    choices = []
    if platform == 'darwin':
        in_video = False
        for line in output.splitlines():
            lowered = line.lower()
            if 'avfoundation video devices:' in lowered:
                in_video = True
                continue
            if 'avfoundation audio devices:' in lowered:
                in_video = False
            if not in_video:
                continue
            match = re.search(r'\[(\d+)\]\s+(.+?)\s*$', line)
            if match and not _is_avfoundation_screen_source(match.group(2)):
                index, name = match.groups()
                # Keep FFmpeg's original index in the label for reference, but
                # open the picked camera by its enumerated name. Synthetic
                # screen entries must not affect those reference indices.
                choices.append((f'{name} · AVFoundation {index}',
                                f'avfoundation:{name}'))
    elif platform.startswith('win'):
        in_video = False
        for line in output.splitlines():
            lowered = line.lower()
            if 'directshow video devices' in lowered:
                in_video = True
                continue
            if 'directshow audio devices' in lowered:
                in_video = False
            if not in_video:
                continue
            if 'alternative name' in lowered:
                continue
            match = re.search(r'"([^"]+)"', line)
            if match:
                name = match.group(1)
                choices.append((name, f'dshow:video={name}'))
    return tuple(choices)


def parse_avfoundation_screen_sources(output):
    choices = []
    in_video = False
    for line in output.splitlines():
        lowered = line.lower()
        if 'avfoundation video devices:' in lowered:
            in_video = True
            continue
        if 'avfoundation audio devices:' in lowered:
            in_video = False
        if not in_video:
            continue
        match = re.search(r'\[(\d+)\]\s+(.+?)\s*$', line)
        if match and _is_avfoundation_screen_source(match.group(2)):
            index, name = match.groups()
            choices.append(ScreenTarget(name, display=int(index)))
    return tuple(choices)


def screen_targets_from_monitors(monitors):
    targets = []
    for index, monitor in enumerate(monitors):
        left = int(monitor['left'])
        top = int(monitor['top'])
        width = int(monitor['width'])
        height = int(monitor['height'])
        label = ('All displays' if index == 0 else f'Display {index}')
        label += f' · {width}×{height} · ({left},{top})'
        region = f'{left},{top},{width},{height}'
        targets.append(ScreenTarget(label, region=region))
    return tuple(targets)


def enumerate_camera_sources(platform=None, ffmpeg_path=None, run=None,
                             dev_root='/dev', sys_root='/sys/class/video4linux'):
    platform = sys.platform if platform is None else platform
    if platform.startswith('linux'):
        nodes = list(Path(dev_root).glob('video*'))
        names = {}
        for node in nodes:
            try:
                names[str(node)] = (Path(sys_root)/node.name/'name').read_text(
                    encoding='utf-8').strip()
            except OSError:
                pass
        choices = linux_camera_sources(nodes, names)
    else:
        if platform == 'darwin':
            def command(executable):
                return [executable, '-hide_banner', '-f', 'avfoundation',
                        '-list_devices', 'true', '-i', '']
        elif platform.startswith('win'):
            def command(executable):
                return [executable, '-hide_banner', '-list_devices', 'true',
                        '-f', 'dshow', '-i', 'dummy']
        else:
            raise RuntimeError(f'Camera discovery is not implemented for {platform}.')
        output = _ffmpeg_devices(command, ffmpeg_path, run)
        choices = parse_ffmpeg_camera_sources(output, platform)
    if not choices:
        raise RuntimeError('No camera devices were found.')
    return choices


def parse_capture_fps(output):
    """Parse frame rates reported by the host's capture driver/tools."""
    text = str(output or '')
    found = set()

    def add(value, denominator=None):
        try:
            rate = float(value)
            if denominator is not None:
                divisor = float(denominator)
                if divisor <= 0:
                    return
                rate /= divisor
        except (TypeError, ValueError, ZeroDivisionError):
            return
        if math.isfinite(rate) and 0 < rate <= 1000:
            found.add(round(rate, 3))

    for match in re.finditer(
            r'\bfps\s*[:=]\s*(\d+(?:\.\d+)?)(?:/(\d+(?:\.\d+)?))?',
            text, re.IGNORECASE):
        add(match.group(1), match.group(2))
    for match in re.finditer(
            r'(\d+(?:\.\d+)?)\s*fps\b', text, re.IGNORECASE):
        add(match.group(1))
    for match in re.finditer(
            r'\b(?:avg_frame_rate|r_frame_rate)\s*[=:]\s*'
            r'(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)', text,
            re.IGNORECASE):
        add(match.group(1), match.group(2))
    return tuple(sorted(found))


def _fps_choices(rates):
    choices = [('Source default', '')]
    choices.extend((f'{rate:g} fps', str(rate)) for rate in sorted(set(rates)))
    return tuple(choices)


def enumerate_capture_fps(source, camera=None, video_source='',
                          platform=None, ffmpeg_path=None, which=None,
                          run=None):
    """Return source-specific capture rates, preferring host-reported modes."""
    platform = sys.platform if platform is None else platform
    which = shutil.which if which is None else which
    run = subprocess.run if run is None else run
    output = ''

    def collect(command):
        nonlocal output
        result = run(command, capture_output=True, text=True,
                      encoding='utf-8', errors='replace', timeout=10,
                      check=False)
        output = '\n'.join((getattr(result, 'stdout', '') or '',
                            getattr(result, 'stderr', '') or ''))

    if source == 'camera':
        selected = str(camera if camera is not None else '')
        if platform.startswith('linux'):
            node = selected.split(':', 1)[1] if selected.startswith('v4l2:') else selected
            ctl = which('v4l2-ctl')
            if ctl and node:
                try:
                    collect([ctl, '--list-formats-ext', '--device', node])
                except (OSError, subprocess.SubprocessError):
                    output = ''
            if not parse_capture_fps(output):
                executable = ffmpeg_path or which('ffmpeg')
                if executable and node:
                    try:
                        collect([executable, '-hide_banner', '-f', 'v4l2',
                                 '-list_formats', 'all', '-i', node])
                    except (OSError, subprocess.SubprocessError):
                        output = ''
        elif selected:
            executable = ffmpeg_path or which('ffmpeg')
            if executable:
                if selected.startswith('dshow:'):
                    fmt, device = 'dshow', selected.split(':', 1)[1]
                elif selected.startswith('avfoundation:'):
                    fmt, device = 'avfoundation', selected.split(':', 1)[1]
                else:
                    fmt, device = ('avfoundation' if platform == 'darwin'
                                   else 'dshow'), selected
                try:
                    collect([executable, '-hide_banner', '-f', fmt,
                             '-list_options', 'true', '-i', device])
                except (OSError, subprocess.SubprocessError):
                    output = ''
    elif source == 'video' and video_source:
        executable = which('ffprobe')
        if executable:
            try:
                collect([executable, '-v', 'error', '-select_streams', 'v:0',
                         '-show_entries', 'stream=avg_frame_rate,r_frame_rate',
                         '-of', 'default=noprint_wrappers=1', video_source])
            except (OSError, subprocess.SubprocessError):
                output = ''
    elif source == 'screen':
        if platform.startswith('linux'):
            executable = which('xrandr')
            command = [executable, '--query'] if executable else None
        elif platform == 'darwin':
            executable = which('system_profiler') or '/usr/sbin/system_profiler'
            command = [executable, 'SPDisplaysDataType']
        elif platform.startswith('win'):
            executable = which('powershell.exe') or which('powershell')
            script = (
                'Get-CimInstance -Namespace root/wmi '
                '-ClassName WmiMonitorListedSupportedSourceModes | '
                'ForEach-Object { $_.SupportedDisplayModes } | '
                'ForEach-Object { $_.RefreshRate }')
            command = ([executable, '-NoProfile', '-Command', script]
                       if executable else None)
        else:
            command = None
        if command:
            try:
                collect(command)
            except (OSError, subprocess.SubprocessError):
                output = ''
        # xrandr lists display modes' rates without an "fps" suffix.
        rates = set(parse_capture_fps(output))
        if platform.startswith('linux'):
            for line in output.splitlines():
                if re.match(r'^\s+\d{3,5}x\d{3,5}\s+', line):
                    rates.update(float(value) for value in re.findall(
                        r'(?<![\w.])(\d{2,3}(?:\.\d+)?)[*+]?\b', line))
        elif platform == 'darwin':
            rates.update(float(value) for value in re.findall(
                r'\b(\d{2,3}(?:\.\d+)?)\s*(?:Hz|Hertz)\b', output,
                re.IGNORECASE))
        elif platform.startswith('win'):
            rates.update(float(value) for value in re.findall(
                r'(?m)^\s*(\d{2,3}(?:\.\d+)?)\s*$', output))
        if rates:
            return _fps_choices(rates)

    rates = parse_capture_fps(output)
    if rates:
        return _fps_choices(rates)
    return _fps_choices(())


def enumerate_screen_targets(backend='mss', platform=None, mss_module=None,
                             ffmpeg_path=None, run=None):
    platform = sys.platform if platform is None else platform
    if backend == 'ffmpeg' and platform == 'darwin':
        def command(executable):
            return [executable, '-hide_banner', '-f', 'avfoundation',
                    '-list_devices', 'true', '-i', '']
        output = _ffmpeg_devices(command, ffmpeg_path, run)
        targets = parse_avfoundation_screen_sources(output)
    else:
        if mss_module is None:
            try:
                import mss as mss_module
            except ImportError as exc:
                raise RuntimeError('Screen discovery requires the mss package.') from exc
        factory = getattr(mss_module, 'MSS', None)
        if factory is None:
            factory = mss_module.mss
        capture = factory()
        try:
            targets = screen_targets_from_monitors(capture.monitors)
        finally:
            close = getattr(capture, 'close', None)
            if close is not None:
                close()
    if not targets:
        raise RuntimeError('No screen/display capture targets were found.')
    return targets


def output_devices(sd_module=None):
    """Return output-capable devices without selecting the system default."""
    if sd_module is None:
        import sounddevice as sd_module
    devices = sd_module.query_devices()
    try:
        hostapis = sd_module.query_hostapis()
    except Exception:
        hostapis = ()
    result = []
    for index, device in enumerate(devices):
        channels = int(device.get('max_output_channels') or 0)
        if channels:
            hostapi_index = device.get('hostapi')
            hostapi = (hostapis[int(hostapi_index)].get('name', '')
                       if hostapi_index is not None and
                       0 <= int(hostapi_index) < len(hostapis) else '')
            result.append(OutputDevice(
                index, device.get('name', f'Audio device {index}'), channels,
                device.get('default_samplerate'), hostapi))
    return tuple(result)


def input_devices(sd_module=None):
    """Return input-capable devices without selecting the system default."""
    if sd_module is None:
        import sounddevice as sd_module
    devices = sd_module.query_devices()
    try:
        hostapis = sd_module.query_hostapis()
    except Exception:
        hostapis = ()
    result = []
    for index, device in enumerate(devices):
        channels = int(device.get('max_input_channels') or 0)
        if channels:
            hostapi_index = device.get('hostapi')
            hostapi = (hostapis[int(hostapi_index)].get('name', '')
                       if hostapi_index is not None and
                       0 <= int(hostapi_index) < len(hostapis) else '')
            result.append(InputDevice(
                index, device.get('name', f'Audio device {index}'), channels,
                device.get('default_samplerate'), hostapi))
    return tuple(result)


def device_rate_text(device):
    """The output device's current rate as the OS reports it."""
    if device is None or not device.default_rate:
        return 'OS setting'
    return f'{device.default_rate/1000:g} kHz · OS setting'


def _clipboard_text(glfw, window):
    """Read the clipboard, retrying with the active window if needed."""
    try:
        version = tuple(glfw.get_version())
    except Exception:
        version = (0, 0, 0)
    preferred = None if version >= (3, 4, 0) else window
    candidates = [preferred]
    fallback = window if preferred is None else None
    if fallback not in candidates:
        candidates.append(fallback)
    last_error = None
    read_succeeded = False
    for candidate in candidates:
        try:
            text = glfw.get_clipboard_string(candidate)
        except Exception as exc:
            last_error = exc
            continue
        read_succeeded = True
        if text:
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='replace')
            return text
    if last_error is not None and not read_succeeded:
        raise last_error
    return ''


def pick_video_file(current_path='', platform=None, which=None, run=None):
    """Open the host's native file picker without adding a GUI dependency."""
    platform = sys.platform if platform is None else platform
    which = shutil.which if which is None else which
    run = subprocess.run if run is None else run

    current = Path(str(current_path)).expanduser() if current_path else None
    if current is not None and current.is_dir():
        initial_dir = str(current)
    elif current is not None and current.parent.is_dir():
        initial_dir = str(current.parent)
    else:
        initial_dir = str(ROOT)

    if platform == 'darwin':
        executable = which('osascript')
        if executable is None:
            raise RuntimeError('macOS file picker (osascript) is unavailable.')
        command = [executable, '-e',
                   'POSIX path of (choose file with prompt "Select a video file")']
    elif platform.startswith('win'):
        executable = which('powershell.exe') or which('powershell')
        if executable is None:
            raise RuntimeError('Windows PowerShell file picker is unavailable.')
        script = (
            'Add-Type -AssemblyName System.Windows.Forms; '
            '$d = New-Object System.Windows.Forms.OpenFileDialog; '
            "$d.Title = 'Select a video file'; "
            f"$d.InitialDirectory = '{initial_dir.replace(chr(39), chr(39)*2)}'; "
            f"$d.Filter = 'Video files|{VIDEO_FILE_GLOB.replace(' ', ';')}|All files|*.*'; "
            'if ($d.ShowDialog() -eq '
            '[System.Windows.Forms.DialogResult]::OK) '
            '{ [Console]::Write($d.FileName) }'
        )
        command = [executable, '-NoProfile', '-STA', '-Command', script]
    else:
        executable = which('zenity')
        if executable is not None:
            start_at = initial_dir.rstrip(os.sep)+os.sep
            command = [
                executable, '--file-selection',
                '--title=Select a video file', f'--filename={start_at}',
                f'--file-filter=Video files | {VIDEO_FILE_GLOB}',
                '--file-filter=All files | *',
            ]
        else:
            executable = which('kdialog')
            if executable is None:
                raise RuntimeError(
                    'No native file picker found. Install zenity or kdialog, '
                    'or type the path / drop a video file on the window.')
            command = [
                executable, '--getopenfilename', initial_dir,
                f'Video files ({VIDEO_FILE_GLOB})', '--title',
                'Select a video file',
            ]

    result = run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        error = str(getattr(result, 'stderr', '') or '').strip()
        if error and 'user canceled' not in error.lower():
            raise RuntimeError(error)
        return None
    return str(getattr(result, 'stdout', '') or '').strip() or None


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


def validate_settings(settings, devices, sd_module=None, audio_devices=()):
    """Validate sender GUI state before launching the sender process."""
    by_index = {device.index: device for device in devices}
    device = by_index.get(settings.get('device'))
    if device is None:
        raise ValueError('Select an audio output device before starting.')
    source = settings.get('source')
    if source not in dict(SOURCE_CHOICES).values():
        raise ValueError('Choose a capture source before starting.')
    channels = 2
    if device.channels < channels:
        raise ValueError(
            f'{device.name} supports {device.channels} output channel(s); '
            f'this configuration needs {channels}. Choose another device.')

    # The sender plays at the rate the OS has the output device set to; the
    # GUI does not offer other rates (PortAudio cannot list a device's
    # supported rates portably, and some hosts resample any request).
    if sd_module is None:
        import sounddevice as sd_module
    try:
        sd_module.check_output_settings(
            device=device.index, channels=channels, dtype='float32',
            samplerate=device.default_rate or None)
    except Exception as exc:
        sample_text = 'its current OS rate'
        raise ValueError(
            f'{device.name} cannot open {channels} channel(s) at {sample_text}: {exc}') from exc

    profile = settings.get('profile')
    if profile not in dict(PROFILE_CHOICES).values():
        raise ValueError('Choose a supported wire profile.')
    mono_profile = profile in MONO_PROFILES
    mono_video_side = (settings.get('mono_video_side', 'right')
                       if mono_profile else 'right')
    if (mono_profile and
            mono_video_side not in dict(MONO_VIDEO_SIDE_CHOICES).values()):
        raise ValueError('Choose the left or right mono-video output side.')
    source_audio = (settings.get('source_audio', 'source')
                    if mono_profile else 'off')
    if (mono_profile and
            source_audio not in dict(SOURCE_AUDIO_CHOICES).values()):
        raise ValueError('Choose video soundtrack, an input device, or Off.')
    source_audio_input_side = 'mix'
    source_audio_gain = 1.0
    source_audio_delay_ms = 0.0
    if mono_profile and source_audio != 'off':
        source_audio_gain = _float_setting(
            settings.get('source_audio_gain', '1'), 'Source-audio gain')
        if not 0.0 <= source_audio_gain <= 4.0:
            raise ValueError('Source-audio gain must be between 0 and 4.')
        source_audio_delay_ms = _float_setting(
            settings.get('source_audio_delay_ms', '0'), 'Source-audio delay')
        if source_audio_delay_ms < 0:
            raise ValueError('Source-audio delay cannot be negative.')
    audio_device = None
    if mono_profile and source_audio == 'device':
        source_audio_input_side = settings.get(
            'source_audio_input_side', 'mix')
        if source_audio_input_side not in dict(SOURCE_AUDIO_SIDE_CHOICES).values():
            raise ValueError('Choose a supported source-audio input channel.')
        audio_device = next((item for item in audio_devices
                             if item.index == settings.get('source_audio_device')),
                            None)
        if audio_device is None:
            raise ValueError('Select an input device for source audio.')
        if source_audio_input_side == 'right' and audio_device.channels < 2:
            raise ValueError('Right input selection requires a stereo input device.')
        input_channels = min(audio_device.channels, 2)
        input_rate = device.default_rate or audio_device.default_rate or None
        try:
            sd_module.check_input_settings(
                device=audio_device.index, channels=input_channels,
                dtype='float32', samplerate=input_rate)
        except Exception as exc:
            sample_text = (f'{input_rate} Hz' if input_rate else
                           'the input device native rate')
            raise ValueError(
                f'{audio_device.name} cannot open {input_channels} input '
                f'channel(s) at {sample_text}: {exc}') from exc
    folded_profiles = ('fold-500', 'fold-1000', 'mono-fold-500',
                       'mono-colour-500')
    encode_filter = settings.get('encode_filter', 'auto')
    if encode_filter not in dict(FILTER_CHOICES).values():
        raise ValueError('Choose a supported encode filter.')
    if encode_filter == 'auto':
        encode_filter = 'box' if profile in folded_profiles else 'nearest'
    if profile in folded_profiles and encode_filter != 'box':
        raise ValueError('Folded profiles require the Box encode filter.')
    screen_backend = settings.get('screen_backend', 'mss')
    if source == 'screen' and screen_backend not in ('mss', 'ffmpeg'):
        raise ValueError('Choose a supported screen capture backend.')
    if settings.get('capture_filter', 'auto') not in dict(CAPTURE_FILTER_CHOICES).values():
        raise ValueError('Choose a supported capture filter.')
    perceptual_resize = settings.get('perceptual_resize', 'off')
    if perceptual_resize not in dict(DOWNSCALER_CHOICES).values():
        raise ValueError('Choose a supported pre-encode downscaler.')
    if perceptual_resize != 'off':
        if profile not in ('fold-500', 'fold-1000', 'mono-fold-500',
                           'mono-colour-500'):
            raise ValueError('The pre-encode downscaler requires a folded profile.')
        if encode_filter != 'box':
            raise ValueError('The pre-encode downscaler requires the Box encode filter.')
        perceptual_strength = _float_setting(
            settings.get('perceptual_detail_strength', '0.25'),
            'Downscaler strength')
        if not 0.0 <= perceptual_strength <= 1.0:
            raise ValueError('Downscaler strength must be between 0 and 1.')
    else:
        perceptual_strength = 0.25

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
    raw_ffmpeg_input = str(settings.get('ffmpeg_input', '')).strip()
    ffmpeg_input = (raw_ffmpeg_input
                    if (source == 'camera' or
                        source == 'screen' and screen_backend == 'ffmpeg')
                    else '')
    camera = None
    camera_spec = ''
    if source == 'camera':
        selected_camera = settings.get('camera')
        if ffmpeg_input:
            if selected_camera is not None:
                raise ValueError(
                    'Choose a camera from the picker or enter a custom '
                    'FFmpeg input, not both.')
            camera_spec = ffmpeg_input
        elif selected_camera is None:
            raise ValueError('Select a camera before starting.')
        else:
            try:
                camera = int(selected_camera)
            except (TypeError, ValueError):
                camera_spec = str(selected_camera).strip()
                if not camera_spec:
                    raise ValueError('Select a camera before starting.')
            if camera is not None and camera < 0:
                raise ValueError('Camera index must be non-negative.')
    elif (source == 'screen' and screen_backend == 'ffmpeg' and
          ffmpeg_input and
          isinstance(settings.get('screen_target'), ScreenTarget) and
          settings['screen_target'].display is not None):
        raise ValueError(
            'Choose a display from the picker or enter a custom FFmpeg '
            'input, not both.')
    capture_width = (
        _integer_setting(settings.get('capture_width', '160'), 'Capture width', 1)
        if source in ('screen', 'video', 'mouse-follow') else 160)
    screen_target = settings.get('screen_target') if source == 'screen' else None
    if screen_target is not None and not isinstance(screen_target, ScreenTarget):
        raise ValueError('Choose a screen/display from the picker.')
    display = screen_target.display if screen_target is not None else None
    region_override = (str(settings.get('region', '')).strip()
                       if source == 'screen' else '')
    region = (region_override or
              (screen_target.region if screen_target is not None else ''))
    if (source == 'screen' and screen_target is None and not region and
            not ffmpeg_input):
        raise ValueError('Choose a screen/display before starting.')
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
        'profile': profile,
        'mono_video_side': mono_video_side,
        'source_audio': source_audio,
        'source_audio_device': audio_device,
        'source_audio_input_side': source_audio_input_side,
        'source_audio_gain': source_audio_gain,
        'source_audio_delay_ms': source_audio_delay_ms,
        'encode_filter': encode_filter,
        'speed': speed,
        'brightness': brightness,
        'gamma': gamma,
        'capture_fps': capture_fps,
        'video_source': video_source,
        'camera': camera,
        'camera_spec': camera_spec,
        'ffmpeg_input': ffmpeg_input,
        'screen_backend': screen_backend,
        'capture_width': capture_width,
        'display': display,
        'region': region,
        'perceptual_resize': perceptual_resize,
        'perceptual_detail_strength': perceptual_strength,
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


def build_command(settings, devices, sd_module=None, python=None,
                  audio_devices=(), image_preview_port=None):
    """Build an argv list for the existing V7 CLI; never invokes a shell."""
    checked = validate_settings(settings, devices, sd_module, audio_devices)
    command = [
        python or sys.executable,
        str(ROOT/'tools'/'v7_live.py'),
        'send',
        '--device', str(checked['device'].index),
        '--source', checked['source'],
    ]

    command.extend(('--profile', checked['profile']))
    if checked['profile'] in MONO_PROFILES:
        command.extend(('--mono-video-side',
                        checked['mono_video_side'], '--source-audio',
                        checked['source_audio']))
        if checked['source_audio'] == 'device':
            command.extend(('--source-audio-device',
                            str(checked['source_audio_device'].index),
                            '--source-audio-input-side',
                            checked['source_audio_input_side']))
        if checked['source_audio_gain'] != 1.0:
            command.extend(('--source-audio-gain',
                            str(checked['source_audio_gain'])))
        if checked['source_audio_delay_ms'] != 0.0:
            command.extend(('--source-audio-delay-ms',
                            str(checked['source_audio_delay_ms'])))

    if checked['speed'] != 1.0:
        command.extend(('--speed', str(checked['speed'])))
    if checked['encode_filter'] != ('box' if checked['profile'] in (
            'fold-500', 'fold-1000', 'mono-fold-500',
            'mono-colour-500') else 'nearest'):
        command.extend(('--encode-filter', checked['encode_filter']))
    if checked['brightness'] is not None:
        command.extend(('--brightness', str(checked['brightness'])))
    if checked['gamma'] != 1.0:
        command.extend(('--gamma', str(checked['gamma'])))
    if checked['perceptual_resize'] != 'off':
        command.extend(('--perceptual-resize', checked['perceptual_resize']))
        if checked['perceptual_detail_strength'] != 0.25:
            command.extend(('--perceptual-detail-strength',
                            str(checked['perceptual_detail_strength'])))
    if checked['capture_fps'] is not None:
        command.extend(('--capture-fps', str(checked['capture_fps'])))

    if checked['source'] == 'video':
        command.extend(('--video-source', checked['video_source']))
        if settings.get('video_live'):
            command.append('--video-live')
        if settings.get('video_preview'):
            command.append('--preview')
    elif checked['source'] == 'camera':
        if checked['camera_spec']:
            command.extend(('--ffmpeg-input', checked['camera_spec']))
        else:
            command.extend(('--camera', str(checked['camera'])))
    elif checked['source'] == 'screen':
        command.extend(('--screen-backend', checked['screen_backend']))
        if checked['screen_backend'] == 'ffmpeg' and checked['ffmpeg_input']:
            command.extend(('--ffmpeg-input', checked['ffmpeg_input']))
        if checked['display'] is not None:
            command.extend(('--display', str(checked['display'])))
        if checked['region']:
            command.append(f"--region={checked['region']}")

    if checked['source'] in ('screen', 'video', 'mouse-follow'):
        command.extend(('--capture-width', str(checked['capture_width'])))
    capture_filter = settings.get('capture_filter', 'auto')
    if capture_filter != 'auto':
        command.extend(('--capture-filter', capture_filter))
    command.append('--gui-control')
    if image_preview_port is not None:
        command.extend(('--image-preview-port', str(int(image_preview_port)),))
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
    ROW_HEIGHT = 36
    BASIC_FIELDS = (
        'device', 'source', 'video_source', 'video_preview', 'image_preview',
        'video_live', 'camera',
        'screen_target', 'source_audio', 'source_audio_device',
        'source_audio_input_side', 'source_audio_gain',
        'source_audio_delay_ms', 'capture_fps', 'profile',
        'mono_video_side', 'brightness', 'gamma', 'speed',
    )
    DROPDOWN_FIELDS = (
        'device', 'source', 'capture_fps', 'profile',
        'mono_video_side', 'source_audio', 'source_audio_device',
        'source_audio_input_side', 'screen_backend', 'capture_filter',
        'camera', 'screen_target', 'perceptual_resize',
    )
    ADVANCED_FIELDS = (
        'perceptual_resize', 'perceptual_detail_strength',
        'screen_backend', 'region', 'ffmpeg_input', 'capture_width',
        'capture_filter',
    )

    def __init__(self, devices=(), device_error='', audio_devices=(),
                 audio_device_error='', preference_path=None,
                 restore_preferences=False):
        self.devices = tuple(devices)
        self.device_error = device_error
        self.audio_devices = tuple(audio_devices)
        self.audio_device_error = audio_device_error
        self.preference_path = (Path(preference_path)
                                if preference_path is not None else None)
        self.output_device_identity = None
        self.source_audio_device_identity = None
        self.settings = {
            'device': None,
            'source': None,
            'profile': 'fold-500',
            'mono_video_side': 'right',
            'source_audio': 'source',
            'source_audio_device': None,
            'source_audio_input_side': 'mix',
            'source_audio_gain': '1',
            'source_audio_delay_ms': '0',
            'speed': '1',
            'encode_filter': 'auto',
            'brightness': '',
            'gamma': '1',
            'capture_fps': '',
            'video_source': '',
            'video_preview': False,
            'image_preview': False,
            'preview_stage': 'resized',
            'video_live': False,
            'camera': None,
            'ffmpeg_input': '',
            'screen_backend': 'mss',
            'screen_target': None,
            'region': '',
            'capture_width': '160',
            'capture_filter': 'auto',
            'perceptual_resize': 'off',
            'perceptual_detail_strength': '0.25',
        }
        self.notice = 'Choose an output device, capture source, and profile.'
        if restore_preferences and self.preference_path is not None:
            self._restore_preferences()
        self.page = 'setup'
        self.advanced = False
        self.selected = 'device'
        self.scroll = 0
        self.dropdown = None
        self.dropdown_scroll = 0
        self.editing = False
        self.edit_buffer = ''
        self.lines = []
        self.events = queue.Queue()
        self.process = None
        self.reader = None
        self.preview_reader = None
        self._preview_socket = None
        self.preview_image = None
        self.preview_counter = None
        self.preview_aspect = None
        self.preview_stage = None
        self.preview_handoff_ns = None
        self.preview_stage_frames = {}
        self.stop_requested = False
        self.close_when_stopped = False
        self.sender_device_lost = False
        self.dirty = True
        self.hits = {}
        self.width, self.height = self.WINDOW_SIZE
        self._glfw = None
        self._window = None
        self._sd = None
        self.capture_choice_cache = {}
        self.device_watch_stop = threading.Event()
        self.device_watch_thread = None
        self.change_source_after_stop = False

    def _restore_preferences(self):
        preferences = _load_sender_preferences(self.preference_path)
        _restore_sender_settings(self.settings, preferences.get('settings'))
        self.output_device_identity = preferences.get('output_device')
        self.source_audio_device_identity = preferences.get(
            'source_audio_device')
        output = _match_device(self.devices, self.output_device_identity)
        if output is not None:
            self.settings['device'] = output.index
        elif self.output_device_identity is not None:
            self.settings['device'] = None
            name = self.output_device_identity.get('name', 'Saved output')
            self.notice = f'{name} is unavailable; choose an output device.'
        source_audio = _match_device(
            self.audio_devices, self.source_audio_device_identity)
        if source_audio is not None:
            self.settings['source_audio_device'] = source_audio.index
        elif self.source_audio_device_identity is not None:
            self.settings['source_audio_device'] = None
            name = self.source_audio_device_identity.get(
                'name', 'Saved audio input')
            self.notice = f'{name} is unavailable; choose an input device.'

    def _persist_preferences(self):
        if self.preference_path is None:
            return
        output = self._device()
        if output is not None:
            self.output_device_identity = _device_identity(output)
        source_audio = next((device for device in self.audio_devices
                             if device.index ==
                             self.settings.get('source_audio_device')), None)
        if source_audio is not None:
            self.source_audio_device_identity = _device_identity(source_audio)
        try:
            _save_sender_preferences({
                'version': SENDER_PREFERENCES_VERSION,
                'settings': _serialize_sender_settings(self.settings),
                'output_device': self.output_device_identity,
                'source_audio_device': self.source_audio_device_identity,
            }, self.preference_path)
        except OSError as exc:
            self.notice = f'Could not save sender preferences: {exc}'

    def _device(self):
        return next((device for device in self.devices
                     if device.index == self.settings['device']), None)

    def _choices(self, dest):
        if dest == 'device':
            return tuple((device.label, device.index) for device in self.devices)
        if dest == 'source_audio_device':
            return tuple((device.label, device.index)
                         for device in self.audio_devices)
        if dest == 'source':
            return SOURCE_CHOICES
        if dest == 'capture_fps':
            return self.capture_choice_cache.get(
                dest, _fps_choices(()))
        if dest == 'profile':
            return PROFILE_CHOICES if self.advanced else PRIMARY_PROFILE_CHOICES
        if dest == 'mono_video_side':
            return MONO_VIDEO_SIDE_CHOICES
        if dest == 'source_audio':
            return SOURCE_AUDIO_CHOICES
        if dest == 'source_audio_input_side':
            return SOURCE_AUDIO_SIDE_CHOICES
        if dest == 'screen_backend':
            return (('mss · lightweight', 'mss'), ('FFmpeg', 'ffmpeg'))
        if dest == 'capture_filter':
            return CAPTURE_FILTER_CHOICES
        if dest == 'perceptual_resize':
            return DOWNSCALER_CHOICES
        if dest == 'camera':
            return self.capture_choice_cache.get(dest, ())
        if dest == 'screen_target':
            return tuple((target.label, target) for target in
                         self.capture_choice_cache.get(dest, ()))
        return ()

    def _open_dropdown(self, dest):
        if self.process is not None:
            self.notice = 'Settings are locked while the sender is running.'
            return
        if dest in ('device', 'source_audio_device'):
            choices_available = self._refresh_audio_device_choices(dest)
            if not choices_available:
                self.dropdown = None
                self.notice = (self.device_error if dest == 'device' else
                               self.audio_device_error or
                               'No audio input devices are available.')
                self.dirty = True
                return
        if dest == 'camera':
            try:
                self.capture_choice_cache[dest] = enumerate_camera_sources()
            except Exception as exc:
                self.capture_choice_cache[dest] = ()
                self.notice = f'Camera discovery failed: {exc}'
                self.dropdown = None
                self.dirty = True
                return
        elif dest == 'screen_target':
            try:
                self.capture_choice_cache[dest] = enumerate_screen_targets(
                    self.settings['screen_backend'])
            except Exception as exc:
                self.capture_choice_cache[dest] = ()
                self.notice = f'Screen discovery failed: {exc}'
                self.dropdown = None
                self.dirty = True
                return
        elif dest == 'capture_fps':
            try:
                self.capture_choice_cache[dest] = enumerate_capture_fps(
                    self.settings['source'], self.settings.get('camera'),
                    self.settings.get('video_source', ''))
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                self.capture_choice_cache[dest] = _fps_choices(())
                self.notice = f'Capture-rate discovery failed: {exc}'
                self.dirty = True
                return
        self.dropdown = dest
        options = self._choices(dest)
        if not options:
            self.dropdown = None
            self.notice = {
                'camera': 'No camera sources are available.',
                'screen_target': 'No displays are available.',
            }.get(dest, 'No choices are available for this option.')
            self.dirty = True
            return
        current = self.settings.get(dest)
        self.dropdown_scroll = next(
            (index for index, (_label, value) in enumerate(options)
             if value == current), 0)
        self.dirty = True

    def _refresh_audio_device_choices(self, dest):
        if dest == 'device':
            try:
                devices = output_devices(self._sounddevice())
                error = '' if devices else 'No audio output devices are available.'
            except Exception as exc:
                devices = ()
                error = f'Audio device discovery failed: {exc}'
            self._set_output_devices(devices, error)
            return bool(self.devices)

        try:
            devices = input_devices(self._sounddevice())
            error = '' if devices else 'No audio input devices are available.'
        except Exception as exc:
            devices = ()
            error = f'Audio input enumeration failed: {exc}'
        self._set_input_devices(devices, error)
        return bool(self.audio_devices)

    def _set_output_devices(self, devices, error=''):
        previous = self._device()
        identity = self.output_device_identity or _device_identity(previous)
        old_devices, old_error = self.devices, self.device_error
        changed = (self._audio_device_signature(old_devices) !=
                   self._audio_device_signature(devices))
        self.devices = tuple(devices)
        self.device_error = error
        current = (_match_device(self.devices, identity)
                   if identity is not None else None)
        if self.settings['device'] is not None:
            if current is None:
                self.settings['device'] = None
                self.notice = ('Selected output device is unavailable; '
                               'choose an available device.')
            else:
                self.settings['device'] = current.index
                self.output_device_identity = _device_identity(current)
        elif current is not None and identity is not None:
            self.settings['device'] = current.index
            self.output_device_identity = _device_identity(current)
        if changed:
            self._device_choices_changed('device')
            if 'unavailable' not in self.notice:
                self.notice = 'Audio devices changed; open a picker to review.'
        if old_error != self.device_error:
            self.dirty = True

    def _set_input_devices(self, devices, error=''):
        previous = next((device for device in self.audio_devices
                         if device.index == self.settings['source_audio_device']),
                        None)
        identity = (self.source_audio_device_identity or
                    _device_identity(previous))
        old_devices, old_error = self.audio_devices, self.audio_device_error
        changed = (self._audio_device_signature(old_devices) !=
                   self._audio_device_signature(devices))
        self.audio_devices = tuple(devices)
        self.audio_device_error = error
        current = (_match_device(self.audio_devices, identity)
                   if identity is not None else None)
        if self.settings['source_audio_device'] is not None:
            if current is None:
                self.settings['source_audio_device'] = None
                self.notice = ('Selected audio input is unavailable; '
                               'choose an available device.')
            else:
                self.settings['source_audio_device'] = current.index
                self.source_audio_device_identity = _device_identity(current)
        elif current is not None and identity is not None:
            self.settings['source_audio_device'] = current.index
            self.source_audio_device_identity = _device_identity(current)
        if changed:
            self._device_choices_changed('source_audio_device')
            if 'unavailable' not in self.notice:
                self.notice = 'Audio devices changed; open a picker to review.'
        if old_error != self.audio_device_error:
            self.dirty = True

    def _device_choices_changed(self, dest):
        if self.dropdown == dest:
            options = self._choices(dest)
            selected = self.settings.get(dest)
            self.dropdown_scroll = next(
                (index for index, (_label, value) in enumerate(options)
                 if value == selected), 0)
        self.dirty = True

    @staticmethod
    def _audio_device_signature(devices):
        return tuple((device.index, device.name, device.hostapi, device.channels,
                      device.default_rate) for device in devices)

    def _watch_devices(self, stop):
        known = None
        while not stop.is_set():
            try:
                outputs = output_devices(self._sounddevice())
                output_error = ('' if outputs else
                                'No audio output devices are available.')
            except Exception as exc:
                outputs = self.devices
                output_error = f'Audio device discovery failed: {exc}'
            try:
                inputs = input_devices(self._sounddevice())
                input_error = ('' if inputs else
                               'No audio input devices are available.')
            except Exception as exc:
                inputs = self.audio_devices
                input_error = f'Audio input enumeration failed: {exc}'

            cameras = None
            camera_error = None
            if self.settings.get('source') == 'camera' and self.process is None:
                try:
                    cameras = enumerate_camera_sources()
                except RuntimeError as exc:
                    if 'No camera devices were found' in str(exc):
                        cameras = ()
                        camera_error = 'No camera sources are available.'
                    else:
                        cameras = self.capture_choice_cache.get('camera')
                        camera_error = f'Camera discovery failed: {exc}'
                except Exception as exc:
                    cameras = self.capture_choice_cache.get('camera')
                    camera_error = f'Camera discovery failed: {exc}'

            signature = (
                self._audio_device_signature(outputs), output_error,
                self._audio_device_signature(inputs), input_error,
                cameras if cameras is not None else 'not-scanned',
                camera_error)
            if signature != known:
                known = signature
                self.events.put(('devices', (outputs, output_error, inputs,
                                             input_error, cameras,
                                             camera_error)))
                self._wake()
            if stop.wait(DEVICE_REFRESH_SECONDS):
                return

    def _apply_device_snapshot(self, snapshot):
        outputs, output_error, inputs, input_error, cameras, camera_error = snapshot
        previous_camera_error = self.notice.startswith((
            'Camera discovery failed:', 'No camera sources are available.'))
        self._set_output_devices(outputs, output_error)
        self._set_input_devices(inputs, input_error)
        if cameras is not None:
            old_cameras = self.capture_choice_cache.get('camera')
            self.capture_choice_cache['camera'] = tuple(cameras)
            selected = self.settings.get('camera')
            if selected is not None and selected not in {
                    value for _label, value in cameras}:
                self.notice = ('Selected camera is unavailable; reopen the '
                               'camera picker and select it again.')
            if old_cameras != tuple(cameras):
                self._device_choices_changed('camera')
                if selected is None or selected in {
                        value for _label, value in cameras}:
                    self.notice = 'Camera list changed; open the picker to review.'
            elif previous_camera_error and camera_error is None:
                self.notice = 'Camera discovery recovered; open the picker to review.'
        if camera_error:
            self.notice = camera_error
            self.dirty = True

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
            dest == 'video_preview' and source != 'video' or
            dest == 'video_live' and source != 'video' or
            dest == 'camera' and source != 'camera' or
            dest == 'screen_target' and source != 'screen' or
            dest == 'ffmpeg_input' and source not in ('camera', 'screen') or
            dest == 'ffmpeg_input' and source == 'screen' and
            self.settings['screen_backend'] != 'ffmpeg' or
            dest in ('screen_backend', 'region') and source != 'screen' or
            dest == 'perceptual_detail_strength' and
            self.settings['perceptual_resize'] == 'off' or
            dest == 'mono_video_side' and
            self.settings['profile'] not in MONO_PROFILES or
            dest == 'source_audio' and
            self.settings['profile'] not in MONO_PROFILES or
            dest == 'source_audio_device' and not (
                self.settings['profile'] in MONO_PROFILES and
                self.settings['source_audio'] == 'device') or
            dest == 'source_audio_input_side' and not (
                self.settings['profile'] in MONO_PROFILES and
                self.settings['source_audio'] == 'device') or
            dest in ('source_audio_gain', 'source_audio_delay_ms') and not (
                self.settings['profile'] in MONO_PROFILES and
                self.settings['source_audio'] != 'off') or
            dest == 'capture_width' and source not in ('screen', 'video', 'mouse-follow'))]
        return fields

    def _value_label(self, dest):
        value = self.settings[dest]
        if dest == 'brightness' and not str(value).strip():
            return '1.0 · profile default'
        if dest == 'device':
            device = self._device()
            return device.label if device else 'Select output device…'
        if dest in ('camera', 'screen_target', 'source_audio_device'):
            if value is None:
                if dest == 'camera':
                    return 'Choose a camera…'
                if dest == 'source_audio_device':
                    return 'Choose an input device…'
                return 'Choose a display…'
            if dest == 'screen_target' and isinstance(value, ScreenTarget):
                return value.label
            choices = self._choices(dest)
            return next((label for label, candidate in choices
                         if candidate == value), str(value))
        if dest in ('source', 'profile', 'screen_backend',
                    'capture_filter', 'perceptual_resize', 'mono_video_side',
                    'source_audio', 'source_audio_input_side', 'capture_fps'):
            choices = self._choices(dest)
            label = next((label for label, candidate in choices
                          if candidate == value), None)
            if label is None:
                return 'Profile default' if value == 'auto' else 'Choose…'
            return label
        if isinstance(value, bool):
            return 'On' if value else 'Off'
        return str(value) if str(value).strip() else '(not set)'

    def _profile_changed(self, value):
        if self.settings['encode_filter'] == 'auto':
            return
        if value in ('fold-500', 'fold-1000', 'mono-fold-500',
                     'mono-colour-500') and self.settings['encode_filter'] != 'box':
            self.settings['encode_filter'] = 'auto'

    def _assign(self, dest, value):
        self.settings[dest] = value
        if dest == 'preview_stage':
            frame = self.preview_stage_frames.get(value)
            self.preview_stage = value
            if frame is None:
                self.preview_image = None
                self.preview_counter = None
                self.preview_aspect = None
                self.preview_handoff_ns = None
            else:
                (self.preview_image, self.preview_counter,
                 self.preview_aspect, self.preview_handoff_ns) = frame
        elif dest == 'profile':
            self._profile_changed(value)
            if value not in ('fold-500', 'fold-1000', 'mono-fold-500',
                             'mono-colour-500'):
                self.settings['perceptual_resize'] = 'off'
        elif dest == 'device':
            self.output_device_identity = _device_identity(self._device())
        elif dest == 'source_audio_device':
            device = next((item for item in self.audio_devices
                           if item.index == value), None)
            self.source_audio_device_identity = _device_identity(device)
        elif dest == 'screen_backend':
            self.settings['screen_target'] = None
            self.capture_choice_cache.pop('capture_fps', None)
        elif dest == 'source':
            self.dropdown = None
            self.capture_choice_cache.pop('capture_fps', None)
        elif dest == 'camera':
            self.capture_choice_cache.pop('capture_fps', None)
        self.notice = f'{dest.replace("_", " ").capitalize()} updated.'
        self.dirty = True
        self._persist_preferences()

    def _select_option(self, dest, value):
        self._assign(dest, value)

    def _build_command(self, image_preview_port=None):
        return build_command(self.settings, self.devices, self._sounddevice(),
                             audio_devices=self.audio_devices,
                             image_preview_port=image_preview_port)

    def _start(self):
        if self.editing:
            self._finish_edit()
        if self.process is not None:
            return
        if (self.settings.get('source') == 'camera' and
                not str(self.settings.get('ffmpeg_input', '')).strip()):
            try:
                choices = enumerate_camera_sources()
            except Exception as exc:
                self.notice = f'Camera discovery failed: {exc}'
                self.dirty = True
                return
            self.capture_choice_cache['camera'] = choices
            selected_camera = self.settings.get('camera')
            if selected_camera not in {value for _label, value in choices}:
                self.notice = (
                    'Camera is unavailable or changed; reopen the camera '
                    'picker and select it again.')
                self.dirty = True
                return
        try:
            preview_socket = None
            preview_warning = None
            if self.settings.get('image_preview'):
                try:
                    preview_socket = socket.socket(socket.AF_INET,
                                                   socket.SOCK_DGRAM)
                    preview_socket.setsockopt(socket.SOL_SOCKET,
                                              socket.SO_RCVBUF, 256*1024)
                    preview_socket.bind(('127.0.0.1', 0))
                    preview_socket.settimeout(.2)
                except OSError as exc:
                    if preview_socket is not None:
                        preview_socket.close()
                    preview_socket = None
                    preview_warning = (
                        f'Image preview unavailable; sending without it: {exc}')
            if preview_socket is not None:
                command = self._build_command(
                    image_preview_port=preview_socket.getsockname()[1])
            else:
                command = self._build_command()
            kwargs = {
                'cwd': str(ROOT),
                'stdin': subprocess.PIPE,
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
            if 'preview_socket' in locals() and preview_socket is not None:
                preview_socket.close()
            self.notice = str(exc)
            self.dirty = True
            return

        self.process = process
        self._preview_socket = preview_socket
        self.preview_image = None
        self.preview_counter = None
        self.preview_aspect = None
        self.preview_stage = None
        self.preview_handoff_ns = None
        self.preview_stage_frames = {}
        self.stop_requested = False
        self.page = 'live'
        self.notice = preview_warning or 'Starting sender…'
        self.lines.clear()
        self.reader = threading.Thread(
            target=self._read_sender, args=(process,),
            name='v7-send-gui-output', daemon=True)
        self.reader.start()
        if preview_socket is not None:
            self.preview_reader = threading.Thread(
                target=self._read_image_preview,
                args=(process, preview_socket),
                name='v7-send-gui-image-preview', daemon=True)
            self.preview_reader.start()
        self.dirty = True

    def _read_image_preview(self, process, preview_socket):
        while (self.process is process and
               self._preview_socket is preview_socket):
            try:
                packet, _address = preview_socket.recvfrom(65507)
            except socket.timeout:
                continue
            except OSError:
                return
            newest_by_stage = {}

            def remember(candidate):
                parsed_candidate = parse_preview_datagram(candidate)
                if parsed_candidate is None:
                    return
                stage_candidate = parsed_candidate[3]
                previous = newest_by_stage.get(stage_candidate)
                if previous is None or parsed_candidate[2] >= previous[0]:
                    newest_by_stage[stage_candidate] = (
                        parsed_candidate[2], candidate)

            remember(packet)
            try:
                preview_socket.setblocking(False)
            except OSError:
                return
            while True:
                try:
                    candidate, _address = preview_socket.recvfrom(65507)
                except BlockingIOError:
                    break
                except OSError:
                    return
                remember(candidate)
            try:
                preview_socket.settimeout(.2)
            except OSError:
                return
            for _handoff_ns, newest in newest_by_stage.values():
                parsed = parse_preview_datagram(newest)
                if parsed is None:
                    continue
                counter, aspect, handoff_ns, stage, jpeg = parsed
                try:
                    from PIL import Image
                    import io
                    with Image.open(io.BytesIO(jpeg)) as image:
                        image.load()
                        decoded = image.convert('RGB')
                except (OSError, ValueError):
                    continue
                if (self.process is not process or
                        self._preview_socket is not preview_socket):
                    continue
                previous = self.preview_stage_frames.get(stage)
                if previous is not None and handoff_ns < previous[3]:
                    continue
                image_state = (decoded, int(counter), int(aspect),
                               int(handoff_ns))
                self.preview_stage_frames[stage] = image_state
                if stage == self.settings.get('preview_stage', 'resized'):
                    (self.preview_image, self.preview_counter,
                     self.preview_aspect, self.preview_handoff_ns) = image_state
                    self.preview_stage = stage
                self.dirty = True
                self._wake()

    def _close_preview_socket(self):
        preview_socket, self._preview_socket = self._preview_socket, None
        if preview_socket is not None:
            try:
                preview_socket.close()
            except OSError:
                pass

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
        control = getattr(process, 'stdin', None)
        if control is not None:
            try:
                control.close()
            except OSError:
                pass
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
                try:
                    status = json.loads(value).get('status')
                except (TypeError, ValueError, AttributeError):
                    status = None
                if status == 'sender_device_lost':
                    self.sender_device_lost = True
                elif status == 'sender_device_reconnected':
                    self.sender_device_lost = False
            elif kind == 'exit':
                return_code = int(value)
                self.process = None
                self._close_preview_socket()
                self.stop_requested = False
                self.sender_device_lost = False
                if return_code == 0:
                    self.notice = 'Sender stopped.'
                else:
                    self.notice = f'Sender exited with status {return_code}.'
                if self.change_source_after_stop:
                    self.change_source_after_stop = False
                    if not self.close_when_stopped:
                        self._open_source_picker()
                if self.close_when_stopped and self._window is not None:
                    self._glfw.set_window_should_close(self._window, True)
            elif kind == 'devices':
                self._apply_device_snapshot(value)
        if changed:
            self.dirty = True

    def _finish_edit(self, commit=True):
        dest = self.selected
        if commit:
            previous = self.settings.get(dest)
            self.settings[dest] = self.edit_buffer
            if dest == 'video_source':
                self.capture_choice_cache.pop('capture_fps', None)
            try:
                if self.process is not None and dest in ('brightness', 'gamma'):
                    self._send_live_tone_update()
                self.notice = f'{dest.replace("_", " ").capitalize()} updated.'
                self._persist_preferences()
            except (OSError, ValueError, RuntimeError) as exc:
                self.settings[dest] = previous
                self.notice = str(exc)
        self.editing = False
        self.dirty = True

    def _open_source_picker(self):
        self.page = 'setup'
        self.dropdown = None
        self.selected = 'source'
        self.scroll = 0
        self.notice = 'Choose a new capture source.'
        self._open_dropdown('source')
        self.dirty = True

    def _send_live_tone_update(self):
        brightness = _float_setting(
            self.settings.get('brightness', ''), 'Brightness', optional=True)
        brightness = 1.0 if brightness is None else brightness
        gamma = _float_setting(self.settings.get('gamma', '1'), 'Gamma')
        if brightness <= 0 or gamma <= 0:
            raise ValueError('Brightness and gamma must be positive.')
        control = getattr(self.process, 'stdin', None)
        if control is None:
            raise RuntimeError('Live sender controls are unavailable.')
        control.write(json.dumps({'brightness': brightness, 'gamma': gamma})+'\n')
        control.flush()

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
            if dest == 'video_source':
                browse_left = width-104
                value = _fit(value, small, browse_left-value_left-14)
                browse_rect = (browse_left, y+4, width-26, y+self.ROW_HEIGHT-7)
                draw.rounded_rectangle(browse_rect, radius=4, fill=(30, 58, 76),
                                       outline=(75, 111, 132), width=1)
                draw.text((browse_left+10, y+10), 'Browse', font=small,
                          fill=(229, 239, 246))
                self.hits['browse:video_source'] = browse_rect
                field_right = browse_left-8
            else:
                value = _fit(value, small, width-value_left-45)
                field_right = width-18
            draw.text((value_left, y+10), value,
                      font=small, fill=(237, 242, 246))
            draw.text((width-40, y+9), '▾' if dest in self.DROPDOWN_FIELDS else '',
                font=small, fill=(134, 169, 188))
            self.hits[f'field:{dest}'] = (
                18, y, field_right, y+self.ROW_HEIGHT-3)

        toggle_y = height-123
        draw.rounded_rectangle((22, toggle_y, 154, toggle_y+28), radius=4,
                               fill=(24, 41, 54), outline=(63, 91, 108))
        draw.text((34, toggle_y+6),
                  'Advanced  On' if self.advanced else 'Advanced  Off',
                  font=small, fill=(205, 218, 228))
        self.hits['advanced'] = (22, toggle_y, 154, toggle_y+28)

        help_text = FIELD_HELP.get(self.selected, '')
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
            0, min(self.dropdown_scroll, max(0, len(options)-1)))
        first_option = max(
            0, min(self.dropdown_scroll-max_items+1,
                   max(0, len(options)-max_items)))
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
            option_index = first_option+index
            label, value = options[option_index]
            y = top+2+index*28
            if value == self.settings[dest]:
                draw.rectangle((left+2, y, right-2, y+26), fill=(42, 78, 99))
            if option_index == self.dropdown_scroll:
                draw.rectangle((left+2, y, right-2, y+26),
                               outline=(117, 174, 199), width=1)
            draw.text((left+9, y+6), _fit(label, small, right-left-18),
                      font=small, fill=(235, 241, 246))
            self.hits[f'option:{option_index}'] = (left, y, right, y+26)

    def _render_live(self, image, draw, font, small):
        width, height = image.size
        preview_enabled = bool(self.settings.get('image_preview'))
        detail_right = int(width*.47) if preview_enabled else width-24
        draw.text((24, 76), 'Sender status', font=font,
                  fill=(229, 237, 243))
        state = ('STOPPING' if self.stop_requested else
                 'DEVICE LOST' if self.sender_device_lost else
                 'SENDING' if self.process is not None else 'STOPPED')
        draw.rounded_rectangle((24, 119, width-24, 188), radius=6,
                               fill=(17, 29, 39), outline=(48, 73, 90))
        draw.text((42, 135), state, font=font,
                  fill=(238, 140, 110) if state == 'DEVICE LOST' else
                  (238, 182, 125) if state == 'STOPPING' else
                  (145, 218, 170) if state == 'SENDING' else (188, 202, 213))
        draw.text((42, 165), _fit(self.notice, small, width-84),
                  font=small, fill=(165, 187, 202))

        device = self._device()
        profile_label = next((label for label, value in PROFILE_CHOICES
                              if value == self.settings['profile']), 'Unknown')
        source_label = next((label for label, value in SOURCE_CHOICES
                             if value == self.settings['source']), 'Not selected')
        rate_text = device_rate_text(device)
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
            value_x = 148 if preview_enabled else 235
            draw.text((value_x, y), _fit(value, small,
                                         max(70, detail_right-value_x-8)),
                      font=small, fill=(218, 229, 237))
            y += 32

        log_top = y+12
        draw.text((24, log_top), 'Sender messages', font=small,
                  fill=(132, 158, 176))
        first_line = log_top+24
        line_count = max(0, min(10, (height-34-first_line)//22))
        for index, line in enumerate(self.lines[-line_count:] if line_count else ()):
            shown = _fit(line, small,
                         max(80, detail_right-52) if preview_enabled
                         else width-48)
            draw.text((28, first_line+index*22), shown,
                      font=small, fill=(183, 201, 214))

        if preview_enabled:
            from PIL import ImageOps
            panel_left = int(width*.51)
            panel = (panel_left, 248, width-24, height-48)
            draw.text((panel_left, 220), 'Image preview', font=small,
                      fill=(132, 158, 176))
            for index, (stage, label) in enumerate((
                    ('source', 'Source'), ('resized', 'Resized'))):
                left = panel_left+118+index*88
                rect = (left, 215, left+80, 241)
                active = self.settings.get('preview_stage') == stage
                draw.rounded_rectangle(
                    rect, radius=4,
                    fill=(42, 78, 99) if active else (17, 29, 39),
                    outline=(94, 143, 168) if active else (48, 73, 90))
                draw.text((left+9, 221), label, font=small,
                          fill=(235, 242, 247))
                self.hits[f'preview_stage:{stage}'] = rect
            draw.rounded_rectangle(panel, radius=6, fill=(12, 21, 29),
                                   outline=(48, 73, 90))
            if self.preview_image is None:
                message = ('Starting preview…' if self.process else
                           'Start sending to see source and resized images.')
                draw.text((panel_left+16, panel[1]+16), message, font=small,
                          fill=(165, 187, 202))
            else:
                inner = (max(1, panel[2]-panel[0]-20),
                         max(1, panel[3]-panel[1]-58))
                thumbnail = ImageOps.contain(self.preview_image, inner)
                x = panel_left+(panel[2]-panel_left-thumbnail.width)//2
                y_image = panel[1]+8+(inner[1]-thumbnail.height)//2
                image.paste(thumbnail.convert('RGBA'), (x, y_image))
                age_ms = max(0.0, (time.monotonic_ns()-
                                   int(self.preview_handoff_ns or 0))/1e6)
                stage = (self.preview_stage or 'image').capitalize()
                caption = (f'{stage} · packet {self.preview_counter} · aspect '
                           f'{self.preview_aspect} · {age_ms:.0f} ms after '
                           'output handoff')
                draw.text((panel_left+10, panel[3]-27),
                          _fit(caption, small, panel[2]-panel_left-20),
                          font=small,
                          fill=(145, 218, 170) if age_ms < 500 else
                          (238, 182, 125))

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
        controls = [
            ('setup', 'Setup', 14, 100),
            ('live', 'Live', 108, 180),
        ]
        if self.page == 'live':
            controls.append(('change_source', 'Change source', 190, 310))
        controls.extend((
            ('start_stop', 'Stop' if self.process is not None else 'Start',
             width-202, width-108),
            ('close', 'Close', width-98, width-12),
        ))
        for key, label, left, right in controls:
            rect = (left, 9, right, 45)
            self.hits[key] = rect
            active = ((key == 'setup' and self.page == 'setup') or
                      (key == 'live' and self.page == 'live'))
            color = ((39, 67, 86) if active else
                     (100, 51, 41) if key == 'start_stop' and
                     self.sender_device_lost else
                     (82, 55, 40) if key == 'start_stop' and self.process else
                     (99, 65, 34) if key == 'change_source' else
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
        right_button = getattr(glfw, 'MOUSE_BUTTON_RIGHT', None)
        if (action != glfw.PRESS or
                button not in (glfw.MOUSE_BUTTON_LEFT, right_button)):
            return
        x, y = glfw.get_cursor_pos(window)
        ordered = list(self.hits)
        if self.dropdown is not None:
            ordered.sort(key=lambda key: 0 if key.startswith('option:') else 1)
        hit = next((key for key in ordered
                    if self.hits[key][0] <= x < self.hits[key][2] and
                    self.hits[key][1] <= y < self.hits[key][3]), None)
        if button == right_button:
            if (hit is None or not hit.startswith('field:') or
                    self.page != 'setup'):
                return
            dest = hit.split(':', 1)[1]
            if (dest not in self._visible_fields() or
                    dest in self.DROPDOWN_FIELDS or
                    dest in ('video_live', 'video_preview',
                             'image_preview')):
                return
            if (self.process is not None and
                    dest not in ('brightness', 'gamma')):
                self.notice = 'Settings are locked while the sender is running.'
                self.dirty = True
                return
            if self.editing and self.selected != dest:
                self._finish_edit()
            if not self.editing:
                self.selected = dest
                self.editing = True
                current = self.settings[dest]
                self.edit_buffer = '' if current is None else str(current)
            self._paste_clipboard(glfw, window)
            return
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
        elif hit == 'change_source':
            if self.process is not None:
                self.change_source_after_stop = True
                self.notice = 'Stopping sender before changing source…'
                self._stop()
            else:
                self._open_source_picker()
        elif hit in ('preview_stage:source', 'preview_stage:resized'):
            self._assign('preview_stage', hit.rsplit(':', 1)[1])
        elif hit == 'advanced':
            if self.process is not None:
                self.notice = 'Settings are locked while the sender is running.'
            else:
                self.advanced = not self.advanced
                self.scroll = 0
                self.dropdown = None
        elif hit == 'browse:video_source':
            if self.process is not None:
                self.notice = 'Settings are locked while the sender is running.'
            else:
                try:
                    path = pick_video_file(self.settings['video_source'])
                except (OSError, RuntimeError) as exc:
                    self.notice = str(exc)
                else:
                    if path:
                        self.settings['video_source'] = path
                        self.capture_choice_cache.pop('capture_fps', None)
                        self.selected = 'video_source'
                        self.notice = f'Selected video: {Path(path).name}'
                        self._persist_preferences()
                    else:
                        self.notice = 'Video selection cancelled.'
        elif hit and hit.startswith('option:') and self.dropdown is not None:
            index = int(hit.split(':', 1)[1])
            choices = self._choices(self.dropdown)
            if 0 <= index < len(choices):
                self._select_option(self.dropdown, choices[index][1])
            self.dropdown = None
        elif hit and hit.startswith('field:') and self.page == 'setup':
            dest = hit.split(':', 1)[1]
            self.selected = dest
            if (self.process is not None and
                    dest not in ('brightness', 'gamma')):
                self.notice = 'Settings are locked while the sender is running.'
            elif dest in ('video_live', 'video_preview', 'image_preview'):
                self._assign(dest, not self.settings[dest])
            elif dest in self.DROPDOWN_FIELDS:
                self._open_dropdown(dest)
            elif dest in self._visible_fields():
                self.editing = True
                current = self.settings[dest]
                self.edit_buffer = '' if current is None else str(current)
        else:
            self.dropdown = None
        self.dirty = True

    def _paste_clipboard(self, glfw, window):
        try:
            self.edit_buffer += _clipboard_text(glfw, window)
        except Exception as exc:
            self.notice = f'Clipboard paste failed: {exc}'
        self.dirty = True

    def _on_key(self, glfw, window, key, _scancode, action, mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if self.editing:
            paste_modifiers = (glfw.MOD_CONTROL |
                               getattr(glfw, 'MOD_SUPER', 0))
            if key in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER):
                self._finish_edit()
            elif key == glfw.KEY_ESCAPE:
                self._finish_edit(commit=False)
            elif key == glfw.KEY_BACKSPACE:
                self.edit_buffer = self.edit_buffer[:-1]
            elif key == glfw.KEY_A and mods & paste_modifiers:
                self.edit_buffer = ''
            elif key == glfw.KEY_V and mods & paste_modifiers:
                self._paste_clipboard(glfw, window)
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
            live_tone_selected = self.selected in ('brightness', 'gamma')
            if (self.process is not None and not live_tone_selected and
                    key not in (glfw.KEY_UP, glfw.KEY_DOWN)):
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
                if dest in ('video_live', 'video_preview', 'image_preview'):
                    self._assign(dest, not self.settings[dest])
                elif dest in self.DROPDOWN_FIELDS:
                    self._open_dropdown(dest)
                else:
                    self.editing = True
                    self.edit_buffer = str(self.settings.get(dest) or '')
            elif key in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER) and self.selected in fields:
                dest = self.selected
                if dest in self.DROPDOWN_FIELDS:
                    self._open_dropdown(dest)
                elif dest in ('video_live', 'video_preview', 'image_preview'):
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
        if not yoffset:
            return
        delta = -1 if yoffset > 0 else 1
        if self.dropdown is not None:
            options = self._choices(self.dropdown)
            self.dropdown_scroll = max(
                0, min(max(0, len(options)-1), self.dropdown_scroll+delta))
        elif self.page == 'setup':
            visible_count = max(
                1, (self.height-131-137)//self.ROW_HEIGHT)
            self.scroll = max(
                0, min(max(0, len(self._visible_fields())-visible_count),
                       self.scroll+delta))
        self.dirty = True

    def _on_drop(self, _window, paths):
        if paths and self.settings.get('source') == 'video':
            self.settings['video_source'] = paths[0]
            self.capture_choice_cache.pop('capture_fps', None)
            self.selected = 'video_source'
            self.notice = f'Video path selected: {Path(paths[0]).name}'
            self.dirty = True
            self._persist_preferences()

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
            self.device_watch_stop.clear()
            self.device_watch_thread = threading.Thread(
                target=self._watch_devices, args=(self.device_watch_stop,),
                name='v7-send-gui-device-watch', daemon=True)
            self.device_watch_thread.start()
            while not glfw.window_should_close(window):
                # A bounded wait keeps Python signal handling responsive while
                # remaining event-driven between UI/device notifications.
                glfw.wait_events_timeout(GUI_EVENT_WAIT_SECONDS)
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
        except KeyboardInterrupt:
            # Ctrl-C should follow the same graceful child shutdown as closing
            # the window, without leaving a traceback that looks like a crash.
            pass
        finally:
            self.device_watch_stop.set()
            self._persist_preferences()
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
            self._close_preview_socket()
            if self.preview_reader is not None:
                self.preview_reader.join(timeout=1)
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
    try:
        audio_devices = input_devices()
        audio_device_error = ('' if audio_devices else
                              'No audio input devices are available.')
    except Exception as exc:
        audio_devices = ()
        audio_device_error = f'Audio input enumeration failed: {exc}'
    SenderGui(devices, device_error, audio_devices, audio_device_error,
              preference_path=sender_preferences_path(),
              restore_preferences=True).run()


if __name__ == '__main__':
    main()
