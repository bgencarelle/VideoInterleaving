#!/usr/bin/env python3
"""Event-driven GUI controller for the standalone V7 sender.

The sender runs in its own process. There is one preview setting: off, the
picture in this window (it arrives on a separate bounded loopback channel), or
the same picture in a pop-out window (tools/v7_preview_popout.py, a small
process this GUI forwards the pictures to). For a video file the Live page
also has transport controls, sent to the sender over its control pipe.
"""
import copy
import errno
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
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.v7_preview_protocol import parse_preview_datagram
from animation_modem import v7_kernels


PROFILE_CHOICES = (
    ('Aspect Fold 500 · stereo · recommended', 'aspect-fold-500'),
    ('Fold 500 · stereo', 'fold-500'),
    ('Mono video · aspect colour Fold 500', 'aspect-mono-500'),
    ('Mono video · colour Fold 500', 'mono-colour-500'),
    ('Stereo slices · each channel a whole picture · new', 'stereo-slices'),
)
DEFAULT_PROFILE = 'aspect-fold-500'
MONO_PROFILES = ('mono-colour-500', 'aspect-mono-500')
FOLDED_PROFILES = ('fold-500', 'mono-colour-500', 'aspect-fold-500',
                   'aspect-mono-500', 'stereo-slices')
ASPECT_PROFILES = ('aspect-fold-500', 'aspect-mono-500', 'stereo-slices')
# Only the stereo aspect profile has a V7 tail (mono packets carry none).
ASPECT_TAIL_PROFILES = ('aspect-fold-500',)
ASPECT_LAYOUT_CHOICES = (
    ('Auto · source aspect', 'auto'),
    ('1:1', '1:1'), ('4:3', '4:3'), ('3:2', '3:2'), ('16:9', '16:9'),
    ('3:4', '3:4'), ('2:3', '2:3'), ('9:16', '9:16'),
)
ASPECT_TAIL_CHOICES = (
    ('Fixed · 96 colour every packet, no rotation · recommended', 'fixed'),
    ('Chroma · rotating colour detail (V7) · best for held stills', 'chroma'),
    ('Split · 48 luma + 48 rotating chroma', 'split'),
    ('Luma · 96 luma every packet', 'luma'),
)
PRIMARY_PROFILE_CHOICES = PROFILE_CHOICES
SOURCE_AUDIO_CHOICES = (
    ('Video soundtrack · default', 'source'),
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
    ('Box · Pillow', 'box'),
    ('Nearest · Pillow', 'nearest'),
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
PIXEL_DETAIL_CHOICES = (
    ('Average · exact for block art', 'average'),
    ('Soft · in the transform', 'soft'),
    ('Cut · in the transform, most detail', 'cut'),
    ('Crisp · in the transform, undoes pixel repetition', 'crisp'),
)
PIXEL_GRID_CHOICES = (
    ('Robust · exact on any link (52×36 for 3:2)', 'robust'),
    ('Large · exact on clean links only (60×40 for 3:2)', 'large'),
)
DCT_SHARPEN_CHOICES = (
    ('Off', 'off'),
    ('Taper · sent band only', 'taper'),
    ('Unsharp mask', 'usm'),
)
# Direct DCT encode runs the 500-slot folded profiles with the Box filter.
DCT_PROFILES = FOLDED_PROFILES
BOOL_FIELDS = ('video_live', 'dct_encode',
               'luma_adjust', 'luma_adjust_linear', 'pixel_encode',
               'clip_aware')
# Settings that take effect while the sender runs (the sender reads them
# from its control pipe); everything else needs a stop and a new Start.
LIVE_FIELDS = ('brightness', 'gamma')
# One preview at a time: none, the pane on the Live page, or the same
# pictures in a separate window.
PREVIEW_CHOICES = (
    ('Off', 'off'),
    ('In the window', 'window'),
    ('Pop-out window', 'popout'),
)
PREVIEW_STAGE_LABELS = (('source', 'Source'), ('resized', 'Encoder input'))
MONO_VIDEO_SIDE_CHOICES = (
    ('Left output · right stays clear', 'left'),
    ('Right output · left stays clear', 'right'),
)

FIELD_HELP = {
    'device': 'Choose the explicit audio output device that feeds the receiver or recording path.',
    'source': 'Choose what the sender captures. Capture starts only after Start.',
    'profile': ('Aspect Fold 500 stereo is recommended. Mono video profiles '
                'leave one output free for audio. Stereo slices (new) makes '
                'each channel a whole picture by itself: either channel '
                'alone, or a mono sum, still shows the picture, and both '
                'together show it in full. The receiver follows the profile '
                'from each packet.'),
    'speed': 'Playback speed from 0.25× to 4×. Faster playback raises the transmitted carrier frequencies.',
    'encode_filter': ('Pillow encoder resize to the fixed 80×96 image when '
                      'Direct DCT encode is off. Every profile uses Box.'),
    'brightness': 'Live source brightness multiplier. 1.0 is neutral.',
    'gamma': 'Live source gamma; 1.0 is neutral.',
    'capture_fps': 'Choose a frame rate reported by the capture source, or leave it at Source default.',
    'video_source': 'Choose a video with Browse, type a path or URL, or drop a file on the window.',
    'preview': ('In the window: the Live page shows the captured source or '
                'the encoder input. Pop-out window: the same picture in a '
                'separate window that can be moved and resized; it follows '
                'pause and seek. Use the receiver to inspect the decoded '
                'output.'),
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
    'capture_filter': ('Optional FFmpeg capture scaler before encoder preparation. '
                       'This is separate from the Pillow encoder resize.'),
    'mono_video_side': ('For the mono video profile, carry the modem on one '
                        'leg and leave the other free for separate audio.'),
    'source_audio': ('In mono-video mode, route the source soundtrack by '
                     'default, or select an input device or Off.'),
    'source_audio_device': 'Choose an explicit microphone, line, or loopback input device.',
    'source_audio_input_side': 'Select one input leg or downmix stereo input to mono.',
    'source_audio_gain': 'Gain applied only to source audio on the free output leg.',
    'source_audio_delay_ms': ('Additional sync delay beyond one emitted video '
                              'packet; zero is the low-latency starting point.'),
    'perceptual_resize': ('Experimental pre-encode downscaler for the resize '
                          'path (Direct DCT encode off).'),
    'perceptual_detail_strength': 'Strength for the selected pre-encode downscaler, from 0 to 1.',
    'dct_encode': ('Encode straight from the full-size source frame to DCT '
                   'coefficients instead of resizing to 80×96 first. '
                   'Recommended (default). Not with the pre-encode downscaler.'),
    'luma_adjust': ('Re-fit brightness so each pixel keeps the source\'s '
                    'luminance with the colour the receiver will show. Keeps '
                    'saturated edges from darkening or ringing. Sender only. '
                    'Recommended (default).'),
    'pixel_encode': ('Send the picture as hard pixels on the wire\'s own '
                     'pixel grid, exactly. Pixel art at a whole multiple of '
                     'the grid passes unchanged. On Aspect Fold 500 the grid '
                     'follows the picture shape (see Pixel grid); on Fold '
                     '500 it is 40×48. Use the receiver\'s Pixel display. '
                     'Turns off the DCT enhancements and luma adjustment.'),
    'pixel_grid': ('Pixel encode on Aspect Fold 500. Robust fits the '
                   'ordinary slots and stays exact on tape and MP3: 52×36 '
                   'for 3:2, 58×32 for 16:9, 50×38 for 4:3, 42×42 square. '
                   'Large is 60×40, 64×36, 56×42, 48×48: its finest detail '
                   'rides as fold guests and is exact only on a clean link. '
                   'The receiver follows either.'),
    'pixel_detail': ('How Pixel encode brings the frame down to the pixel '
                     'grid. Average: area average of the pixels, exact for '
                     'block art. Soft / Cut / Crisp downscale inside the '
                     'transform: more detail per pixel, with a faint mesh '
                     'next to hard edges that grows from Soft to Crisp.'),
    'luma_adjust_linear': ('Luma adjustment aims at the light of every source '
                           'pixel instead of the light of the averaged '
                           'picture. Fine patterns keep their true brightness; '
                           'thin dark outlines get lighter. About 2 ms more '
                           'per 1080p frame. Off by default.'),
    'clip_aware': ('Re-fit the sent brightness detail so edge ringing falls '
                   'into the receiver\'s black/white clip. Sender only; '
                   'about 2 ms per frame.'),
    'dct_kernel': ('The filter that brings the picture down to what the wire '
                   'holds. Files in dct_kernels/; add or remove one and press '
                   'R. Left/Right switches kernel, also while sending. The '
                   'wire is unchanged.'),
    'dct_sharpen': ('Taper boosts the upper-middle of the sent band and '
                    'leaves the cutoff alone; unsharp mask boosts everything.'),
    'dct_sharpen_strength': 'Sharpen strength, from 0 to 1. Try 0.25 or 0.5.',
    'dct_clarity': 'Large-radius local contrast, from 0 to 1. Try 0.15 or 0.3.',
    'dct_chroma_gain': ('Saturation boost around neutral, from 1.0 to 1.3. '
                        '1.0 recommended: 1.05 and 1.1 measured no better.'),
    'aspect_layout': ('Which coefficients the aspect profiles send: matched '
                      'to this picture shape. Not signalled; set the receiver '
                      'to the same layout.'),
    'aspect_tail': ('What the 96 tail slots carry. Not signalled; set the '
                    'receiver to the same tail.'),
}
FIELD_LABELS = {
    'device': 'Audio output device',
    'source': 'Capture source',
    'capture_fps': 'Capture FPS',
    'brightness': 'Brightness',
    'gamma': 'Gamma',
    'screen_target': 'Screen / display',
    'preview': 'Preview',
    'video_live': 'Treat URL as live',
    'camera': 'Camera',
    'ffmpeg_input': 'FFmpeg input',
    'mono_video_side': 'Mono video output',
    'encode_filter': 'Encoder resize · Pillow',
    'capture_filter': 'Capture scaler · FFmpeg',
    'source_audio': 'Audio source',
    'source_audio_device': 'Audio input device',
    'source_audio_input_side': 'Input channels',
    'source_audio_gain': 'Audio gain',
    'source_audio_delay_ms': 'Audio delay (ms)',
    'perceptual_resize': 'Downscaler',
    'perceptual_detail_strength': 'Downscale amount',
    'dct_encode': 'Direct DCT encode',
    'luma_adjust': 'Luma adjustment',
    'luma_adjust_linear': 'Luma · linear light',
    'pixel_encode': 'Pixel encode',
    'pixel_detail': 'Pixel downscale',
    'pixel_grid': 'Pixel grid',
    'clip_aware': 'Clip-aware encode',
    'dct_kernel': 'DCT kernel · live',
    'dct_sharpen': 'DCT sharpen',
    'dct_sharpen_strength': 'Sharpen strength',
    'dct_clarity': 'DCT clarity',
    'dct_chroma_gain': 'Chroma gain',
    'aspect_layout': 'Aspect layout',
    'aspect_tail': 'Aspect tail',
}
# DCT downscale kernels: one file each in dct_kernels/ (see the README there),
# read when the GUI starts and again on R. The kernel's own parameters are
# rows named kp:<parameter>; they, the kernel choice and the DCT strengths
# are live: they reach a running sender through its control pipe.
KERNEL_PARAM_PREFIX = 'kp:'
KERNEL_DIRS = []                  # extra folders: --dct-kernel-dir
KERNEL_LIVE_FIELDS = ('dct_kernel', 'dct_sharpen', 'dct_sharpen_strength',
                      'dct_clarity', 'dct_chroma_gain')
# (low, high, step) of the plain numbers that step with Left/Right.
NUMERIC_STEPS = {'dct_sharpen_strength': (0.0, 1.0, 0.05),
                 'dct_clarity': (0.0, 1.0, 0.05),
                 'dct_chroma_gain': (1.0, 1.3, 0.01)}
_KERNELS = None


def kernel_registry(refresh=False):
    """The GUI's kernel registry (scanned on first use, or when asked)."""
    global _KERNELS
    if _KERNELS is None:
        _KERNELS = v7_kernels.KernelRegistry(v7_kernels.kernel_dirs(KERNEL_DIRS))
        refresh = True
    if refresh:
        _KERNELS.scan()
    return _KERNELS


def _is_live_field(dest):
    """Settings that may change while the sender runs."""
    return (dest in LIVE_FIELDS or dest in KERNEL_LIVE_FIELDS or
            dest.startswith(KERNEL_PARAM_PREFIX))


def _kernel_values(settings, kernel):
    """The saved values for ``kernel``, limited to parameters it still has."""
    saved = (settings.get('dct_kernel_params') or {}).get(kernel.name, {})
    return kernel.resolve({key: value for key, value in saved.items()
                           if key in kernel.params})


VIDEO_FILE_GLOB = '*.mp4 *.m4v *.mov *.mkv *.webm *.avi *.mpeg *.mpg *.wmv *.ts'
DEVICE_REFRESH_SECONDS = 3.0
GUI_EVENT_WAIT_SECONDS = 0.5
SENDER_PREFERENCES_VERSION = 5
DEFAULT_ASPECT_TAIL = 'fixed'
# Version 5 made the fixed tail the default (the rotating tails show stale
# colour on moving pictures): an earlier saved tail starts at it once.
V5_RESET_SETTINGS = ('aspect_tail',)
# Settings whose defaults changed in version 4 (four folded profiles, Aspect
# Fold 500 and Direct DCT encode by default; every profile encodes with Box):
# earlier saved values are dropped.
V4_RESET_SETTINGS = ('profile', 'dct_encode', 'encode_filter')


def sender_preferences_path():
    """Return the per-user V7 sender GUI preferences path."""
    root = Path(os.environ.get('XDG_CONFIG_HOME') or Path.home()/'.config')
    return root/'modemTest'/'v7_send_gui.json'


def _load_sender_preferences(path):
    try:
        values = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError):
        return {}
    version = values.get('version') if isinstance(values, dict) else None
    if version not in (1, 2, 3, 4, SENDER_PREFERENCES_VERSION):
        return {}
    settings = values.get('settings')
    if isinstance(settings, dict):
        settings = dict(settings)
        if version == 2 and settings.get('source_audio') == 'off':
            # V2 changed the default to Off; restore the requested soundtrack
            # default unless the user had selected another audio source.
            settings['source_audio'] = 'source'
        if version < 4:
            for key in V4_RESET_SETTINGS:
                settings.pop(key, None)
        if version < 5:
            for key in V5_RESET_SETTINGS:
                settings.pop(key, None)
        if settings.get('encode_filter') not in (
                None, *dict(FILTER_CHOICES).values()):
            # Lanczos and bicubic are no longer wire models.
            settings.pop('encode_filter')

    def clean_identity(value):
        if (not isinstance(value, dict) or
                not isinstance(value.get('name'), str)):
            return None
        hostapi = value.get('hostapi', '')
        if not isinstance(hostapi, str):
            hostapi = ''
        return {'name': value['name'], 'hostapi': hostapi}

    return {'settings': settings if isinstance(settings, dict) else {},
            'resume': _clean_resume(values.get('resume')),
            'output_device': clean_identity(values.get('output_device')),
            'source_audio_device': clean_identity(
                values.get('source_audio_device'))}


RESUME_ENTRY_LIMIT = 40


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _clean_resume(value):
    """Saved playback positions: {file path: position, duration, size, mtime}."""
    cleaned = {}
    if not isinstance(value, dict):
        return cleaned
    for key, entry in value.items():
        if not isinstance(key, str) or not isinstance(entry, dict):
            continue
        position = _number(entry.get('position'))
        if position is None or position <= 0:
            continue
        cleaned[key] = {'position': position,
                        'duration': _number(entry.get('duration')),
                        'size': entry.get('size'),
                        'mtime': entry.get('mtime')}
    return cleaned


def _resume_key(video_source):
    return os.path.abspath(os.path.expanduser(str(video_source).strip()))


def _file_signature(video_source):
    try:
        status = os.stat(_resume_key(video_source))
    except OSError:
        return None
    return {'size': int(status.st_size), 'mtime': int(status.st_mtime)}


def is_file_source(settings):
    """A local video file: the source the transport controls apply to."""
    video_source = str(settings.get('video_source') or '').strip()
    if settings.get('source') != 'video' or not video_source:
        return False
    from tools.v7_capture import _is_stream_url
    return not _is_stream_url(video_source)


def resume_position(resume, video_source):
    """Where Start resumes this file, or 0 for the beginning.

    A missing or changed file (size or modification time) and a position at
    or past the known end start from the beginning.
    """
    entry = resume.get(_resume_key(video_source)) if isinstance(
        resume, dict) else None
    signature = _file_signature(video_source)
    if not isinstance(entry, dict) or signature is None:
        return 0.0
    if (entry.get('size') != signature['size'] or
            entry.get('mtime') != signature['mtime']):
        return 0.0
    position = _number(entry.get('position'))
    duration = _number(entry.get('duration'))
    if position is None or position <= 0:
        return 0.0
    if duration is not None and position >= duration:
        return 0.0
    return position


def _clock_text(seconds):
    seconds = max(0, int(seconds or 0))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    if hours:
        return f'{hours}:{minutes:02d}:{seconds:02d}'
    return f'{minutes}:{seconds:02d}'


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
    'capture_fps', 'video_source', 'preview', 'video_live', 'camera',
    'preview_stage',
    'ffmpeg_input', 'screen_backend', 'screen_target', 'region',
    'capture_width', 'capture_filter', 'perceptual_resize',
    'perceptual_detail_strength', 'dct_encode', 'dct_sharpen',
    'dct_sharpen_strength', 'dct_clarity', 'dct_chroma_gain',
    'aspect_layout', 'aspect_tail', 'clip_aware', 'luma_adjust',
    'luma_adjust_linear', 'pixel_encode', 'pixel_detail', 'pixel_grid',
    'dct_kernel', 'dct_kernel_params',
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
        if key == 'profile' and value not in dict(PROFILE_CHOICES).values():
            continue                    # a removed profile: keep the default
        if key == 'dct_kernel_params':
            if isinstance(value, dict):
                target[key] = {
                    str(name): {str(k): float(v) for k, v in values.items()
                                if isinstance(v, (int, float)) and
                                not isinstance(v, bool) and math.isfinite(v)}
                    for name, values in value.items()
                    if isinstance(values, dict)}
        elif key in BOOL_FIELDS:
            if isinstance(value, bool):
                target[key] = value
        elif value is None or isinstance(value, (str, int, float, bool)):
            target[key] = value
    if saved.get('preview') == 'external':
        # The external player became the pop-out window.
        target['preview'] = 'popout'
    elif saved.get('preview') not in dict(PREVIEW_CHOICES).values():
        # Earlier versions had two switches (and, before that,
        # 'encoded_preview' for the pane). The pane wins over the separate
        # player, which is now the pop-out window.
        image = saved.get('image_preview')
        if not isinstance(image, bool):
            image = saved.get('encoded_preview') is True
        target['preview'] = ('window' if image else
                             'popout' if saved.get('video_preview') is True
                             else 'off')
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
    encode_filter = settings.get('encode_filter', 'auto')
    if encode_filter not in dict(FILTER_CHOICES).values():
        raise ValueError('Choose a supported encode filter.')
    if encode_filter == 'auto':
        encode_filter = 'box' if profile in FOLDED_PROFILES else 'nearest'
    if profile in FOLDED_PROFILES and encode_filter != 'box':
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
        if profile not in FOLDED_PROFILES:
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
    dct_encode = bool(settings.get('dct_encode', False))
    dct_sharpen = settings.get('dct_sharpen', 'off')
    pixel_detail = settings.get('pixel_detail', 'average')
    if pixel_detail not in dict(PIXEL_DETAIL_CHOICES).values():
        raise ValueError('Choose a supported pixel downscale.')
    pixel_grid = settings.get('pixel_grid', 'robust')
    if pixel_grid not in dict(PIXEL_GRID_CHOICES).values():
        raise ValueError('Choose a supported pixel grid.')
    dct_strength, dct_clarity, dct_chroma_gain = .25, 0.0, 1.0
    if dct_encode:
        if profile not in DCT_PROFILES:
            raise ValueError('Direct DCT encode requires a folded profile.')
        if encode_filter != 'box':
            raise ValueError('Direct DCT encode requires the Box encode filter.')
        if perceptual_resize != 'off':
            raise ValueError('Direct DCT encode cannot be combined with the '
                             'pre-encode downscaler.')
        if dct_sharpen not in dict(DCT_SHARPEN_CHOICES).values():
            raise ValueError('Choose a supported DCT sharpen mode.')
        if dct_sharpen != 'off':
            dct_strength = _float_setting(
                settings.get('dct_sharpen_strength', '0.25'),
                'DCT sharpen strength')
            if not 0.0 <= dct_strength <= 1.0:
                raise ValueError('DCT sharpen strength must be between 0 and 1.')
        dct_clarity = _float_setting(settings.get('dct_clarity', '0'),
                                     'DCT clarity')
        if not 0.0 <= dct_clarity <= 1.0:
            raise ValueError('DCT clarity must be between 0 and 1.')
        dct_chroma_gain = _float_setting(settings.get('dct_chroma_gain', '1'),
                                         'DCT chroma gain')
        if not 1.0 <= dct_chroma_gain <= 1.3:
            raise ValueError('DCT chroma gain must be between 1.0 and 1.3.')
    else:
        dct_sharpen = 'off'
    dct_kernel, kernel_values = 'reference', {}
    if dct_encode and not settings.get('pixel_encode'):
        dct_kernel = settings.get('dct_kernel', 'reference') or 'reference'
        if dct_kernel != 'reference':
            try:
                kernel_values = _kernel_values(
                    settings, kernel_registry().get(dct_kernel))
            except v7_kernels.KernelError as exc:
                raise ValueError(f'DCT kernel: {exc}.') from exc
    aspect_layout, aspect_tail = 'auto', DEFAULT_ASPECT_TAIL
    if profile in ASPECT_PROFILES:
        aspect_layout = settings.get('aspect_layout', 'auto')
        if profile in ASPECT_TAIL_PROFILES:
            aspect_tail = settings.get('aspect_tail', DEFAULT_ASPECT_TAIL)
        if aspect_layout not in dict(ASPECT_LAYOUT_CHOICES).values():
            raise ValueError('Choose a supported aspect layout.')
        if aspect_tail not in dict(ASPECT_TAIL_CHOICES).values():
            raise ValueError('Choose a supported aspect tail.')
        if (profile == 'aspect-fold-500' and dct_encode and
                settings.get('pixel_encode') and
                aspect_tail not in ('fixed', 'chroma')):
            raise ValueError('Pixel encode needs the Fixed or Chroma tail.')

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
        'dct_encode': dct_encode,
        'clip_aware': bool(settings.get('clip_aware', False)),
        'luma_adjust': bool(settings.get('luma_adjust', True)),
        'luma_adjust_linear': bool(settings.get('luma_adjust_linear', False)),
        'pixel_encode': bool(settings.get('pixel_encode', False)),
        'pixel_detail': pixel_detail,
        'pixel_grid': pixel_grid,
        'aspect_layout': aspect_layout,
        'aspect_tail': aspect_tail,
        'dct_sharpen': dct_sharpen,
        'dct_sharpen_strength': dct_strength,
        'dct_clarity': dct_clarity,
        'dct_chroma_gain': dct_chroma_gain,
        'dct_kernel': dct_kernel,
        'dct_kernel_values': kernel_values,
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
                  audio_devices=(), image_preview_port=None, video_start=None):
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
    if checked['profile'] in ASPECT_PROFILES:
        if checked['aspect_layout'] != 'auto':
            command.extend(('--aspect-layout', checked['aspect_layout']))
        if checked['aspect_tail'] != DEFAULT_ASPECT_TAIL:
            command.extend(('--aspect-tail', checked['aspect_tail']))
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
    if checked['encode_filter'] != (
            'box' if checked['profile'] in FOLDED_PROFILES else 'nearest'):
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
    if checked['dct_encode']:
        command.append('--dct-encode')
        if checked['dct_sharpen'] != 'off':
            command.extend(('--dct-sharpen', checked['dct_sharpen']))
            if checked['dct_sharpen_strength'] != .25:
                command.extend(('--dct-sharpen-strength',
                                str(checked['dct_sharpen_strength'])))
        if checked['dct_clarity'] != 0.0:
            command.extend(('--dct-clarity', str(checked['dct_clarity'])))
        if checked['dct_chroma_gain'] != 1.0:
            command.extend(('--dct-chroma-gain',
                            str(checked['dct_chroma_gain'])))
        if checked['dct_kernel'] != 'reference':
            kernel = kernel_registry().get(checked['dct_kernel'])
            command.extend(('--dct-kernel', checked['dct_kernel']))
            for key, value in sorted(checked['dct_kernel_values'].items()):
                if value != kernel.params[key].default:
                    command.extend(('--dct-kernel-param', f'{key}={value:g}'))
        for folder in KERNEL_DIRS:
            command.extend(('--dct-kernel-dir', str(folder)))
        if checked['pixel_encode']:
            command.append('--pixel-encode')
            if checked['pixel_detail'] != 'average':
                command.extend(('--pixel-detail', checked['pixel_detail']))
            if (checked['profile'] == 'aspect-fold-500' and
                    checked['pixel_grid'] != 'robust'):
                command.extend(('--pixel-grid', checked['pixel_grid']))
        if checked['luma_adjust']:
            command.append('--luma-adjust')
            if checked['luma_adjust_linear']:
                command.append('--luma-adjust-linear')
    if checked['clip_aware']:
        command.append('--clip-aware-encode')
    if checked['capture_fps'] is not None:
        command.extend(('--capture-fps', str(checked['capture_fps'])))

    if checked['source'] == 'video':
        command.extend(('--video-source', checked['video_source']))
        # 'Treat URL as live' is a saved toggle about stream URLs.  Left on,
        # it must not turn a chosen movie file into live capture, which the
        # sender refuses ('live capture requires a stream URL').
        from tools.v7_capture import _is_stream_url
        if settings.get('video_live'):
            if _is_stream_url(checked['video_source']):
                command.append('--video-live')
        if (video_start and video_start > 0 and
                not _is_stream_url(checked['video_source'])):
            command.extend(('--video-start', f'{float(video_start):.3f}'))
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
    # The pane and the pop-out show the same pictures from one channel.
    if (image_preview_port is not None and
            settings.get('preview', 'window') in ('window', 'popout')):
        command.extend(('--image-preview-port', str(int(image_preview_port)),))
    return command


@lru_cache(maxsize=16)
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
    while text and text != '…' and font.getlength(text) > width:
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
    ROW_HEIGHT = 34
    HEADER_HEIGHT = 26
    GUTTER = 18
    # Settings sit on a grid of equal cells; a line holds as many as fit.
    CELL_MIN_WIDTH = 300
    CELL_GAP = 8
    SETUP_TOP = TOOLBAR+6
    TRANSPORT_HEIGHT = 36
    LOG_LINE = 18
    LIVE_FIELDS = LIVE_FIELDS
    # Not signalled on the wire: the receiver must be set the same.
    MATCH_FIELDS = ('aspect_layout', 'aspect_tail')
    # The setup page: named sections, most used first. A setting appears in
    # exactly one section; _shown_fields hides what the current choices make
    # irrelevant. The encoder resize filter is not offered: every profile is
    # folded and folded profiles only encode with Box ('auto'), so the other
    # choice could only fail validation. --encode-filter remains on the CLI.
    SECTIONS = (
        ('Live controls', LIVE_FIELDS),
        ('Output', ('device', 'speed')),
        ('Source', ('source', 'capture_fps', 'preview', 'video_source',
                    'video_live', 'camera', 'screen_target')),
        ('Wire profile', ('profile', 'aspect_layout', 'aspect_tail',
                          'mono_video_side')),
        ('Source audio', ('source_audio', 'source_audio_device',
                          'source_audio_input_side', 'source_audio_gain',
                          'source_audio_delay_ms')),
        ('Picture encode', ('dct_encode', 'pixel_encode', 'pixel_detail',
                            'pixel_grid', 'luma_adjust',
                            'luma_adjust_linear', 'clip_aware',
                            'dct_kernel', 'dct_sharpen', 'dct_sharpen_strength',
                            'dct_clarity', 'dct_chroma_gain',
                            'perceptual_resize',
                            'perceptual_detail_strength')),
        ('Capture', ('screen_backend', 'region', 'ffmpeg_input',
                     'capture_width', 'capture_filter')),
    )
    # Cell widths in grid columns: long values take the whole line, a few
    # take two cells, everything else (numbers, short pickers, on/off) one.
    FULL_WIDTH_FIELDS = (
        'camera', 'screen_target', 'profile',
        'aspect_tail', 'source_audio_device', 'ffmpeg_input', 'region',
        'pixel_detail', 'pixel_grid',
    )
    DOUBLE_WIDTH_FIELDS = ('device', 'video_source', 'aspect_layout',
                           'source_audio', 'mono_video_side', 'dct_kernel')
    DROPDOWN_FIELDS = (
        'device', 'source', 'capture_fps', 'profile', 'encode_filter',
        'mono_video_side', 'source_audio', 'source_audio_device',
        'source_audio_input_side', 'screen_backend', 'capture_filter',
        'camera', 'screen_target', 'perceptual_resize', 'dct_sharpen',
        'aspect_layout', 'aspect_tail', 'pixel_detail', 'pixel_grid',
        'preview', 'dct_kernel',
    )

    # A new notice replaces the help line in the footer until the selection
    # moves; moving the selection brings the help back.
    @property
    def notice(self):
        return self._notice

    @notice.setter
    def notice(self, value):
        self._notice = value
        self._notice_fresh = True

    @property
    def selected(self):
        return self._selected

    @selected.setter
    def selected(self, value):
        if value != getattr(self, '_selected', None):
            self._notice_fresh = False
        self._selected = value

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
            'profile': DEFAULT_PROFILE,
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
            'preview': 'off',
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
            'dct_encode': True,
            'clip_aware': False,
            'luma_adjust': True,
            'luma_adjust_linear': False,
            'pixel_encode': False,
            'pixel_detail': 'average',
            'pixel_grid': 'robust',
            'dct_sharpen': 'off',
            'dct_sharpen_strength': '0.25',
            'dct_clarity': '0',
            'dct_chroma_gain': '1',
            'dct_kernel': 'reference',
            'dct_kernel_params': {},
            'aspect_layout': 'auto',
            'aspect_tail': DEFAULT_ASPECT_TAIL,
        }
        self.notice = 'Choose an output device, capture source, and profile.'
        # Playback of a video file: the sender's last report while it runs,
        # and the saved positions Start resumes from.
        self.playback = None
        self.resume_positions = {}
        self._resume_saved_at = 0.0
        if restore_preferences and self.preference_path is not None:
            self._restore_preferences()
        self.page = 'setup'
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
        self.preview_datagrams = {}
        self.popout = None
        self._popout_address = None
        self._seek_drag = None
        self._seek_track = None
        self.preview_error = None
        self.preview_reader_error_reported = False
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
        self._notice_fresh = True

    def _restore_preferences(self):
        preferences = _load_sender_preferences(self.preference_path)
        _restore_sender_settings(self.settings, preferences.get('settings'))
        self.resume_positions = dict(preferences.get('resume') or {})
        self.output_device_identity = preferences.get('output_device')
        self.source_audio_device_identity = preferences.get(
            'source_audio_device')
        if self.settings.get('dct_kernel', 'reference') not in \
                kernel_registry().names():
            self.notice = (f"DCT kernel {self.settings['dct_kernel']!r} is "
                           'no longer in the kernel folder; using Reference.')
            self.settings['dct_kernel'] = 'reference'
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
                'resume': self.resume_positions,
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
        if dest == 'preview':
            return PREVIEW_CHOICES
        if dest == 'capture_fps':
            return self.capture_choice_cache.get(
                dest, _fps_choices(()))
        if dest == 'profile':
            return PROFILE_CHOICES
        if dest == 'encode_filter':
            if self.settings.get('profile') in FOLDED_PROFILES:
                return FILTER_CHOICES[:2]
            return FILTER_CHOICES
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
        if dest == 'aspect_layout':
            return ASPECT_LAYOUT_CHOICES
        if dest == 'aspect_tail':
            return ASPECT_TAIL_CHOICES
        if dest == 'dct_kernel':
            return tuple((label, name) for name, label, _help, _params in
                         kernel_registry().describe())
        if dest == 'dct_sharpen':
            return DCT_SHARPEN_CHOICES
        if dest == 'pixel_detail':
            return PIXEL_DETAIL_CHOICES
        if dest == 'pixel_grid':
            return PIXEL_GRID_CHOICES
        if dest == 'camera':
            return self.capture_choice_cache.get(dest, ())
        if dest == 'screen_target':
            return tuple((target.label, target) for target in
                         self.capture_choice_cache.get(dest, ()))
        return ()

    def _open_dropdown(self, dest):
        if self.process is not None and not _is_live_field(dest):
            self.notice = 'Settings are locked while the sender is running.'
            return
        if dest == 'dct_kernel' and self.process is None:
            registry = kernel_registry(refresh=True)
            if registry.errors:
                self.notice = ''.join(
                    f'Kernel {Path(path).name} skipped: {message}  '
                    for path, message in registry.errors)
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
        """Shown settings in page order (also the keyboard order)."""
        return [dest for _title, shown in self._setup_sections()
                for dest in shown]

    def _setup_sections(self):
        """(title, shown settings) for every section that has any."""
        sections = []
        for title, group in self.SECTIONS:
            shown = self._shown_fields(group)
            if shown:
                sections.append((title, shown))
        return sections

    def _columns(self, width):
        """Grid columns that fit the window width."""
        return max(1, (width-2*self.GUTTER+self.CELL_GAP)//
                   (self.CELL_MIN_WIDTH+self.CELL_GAP))

    def _span(self, dest, columns):
        if dest in self.FULL_WIDTH_FIELDS:
            span = columns
        elif dest in self.DOUBLE_WIDTH_FIELDS:
            span = 2
        else:
            span = 1
        return max(1, min(span, columns))

    def _pack(self, dests, columns):
        """Lines of (setting, first column, columns spanned), left to right."""
        lines, line, used = [], [], 0
        for dest in dests:
            span = self._span(dest, columns)
            if used+span > columns:
                lines.append(tuple(line))
                line, used = [], 0
            line.append((dest, used, span))
            used += span
        if line:
            lines.append(tuple(line))
        return lines

    def _setup_items(self, width):
        """Page lines: ('header', title), ('line', cells), ('note', text)."""
        columns = self._columns(width)
        items = []
        for title, shown in self._setup_sections():
            items.append(('header', title))
            items.extend(('line', line) for line in self._pack(shown, columns))
            if title == 'Live controls' and self.process is not None:
                items.append(('note', 'Stop to change the settings below'))
        return items

    def _item_height(self, kind):
        return self.ROW_HEIGHT if kind == 'line' else self.HEADER_HEIGHT

    def _locked(self, dest):
        """True for a setting that cannot change while the sender runs."""
        return self.process is not None and not _is_live_field(dest)

    def _setup_room(self, size=None):
        """Pixels between the toolbar and the footer."""
        width, height = size or (self.width, self.height)
        return max(0, height-self.SETUP_TOP-self._footer_height(width)-4)

    def _items_fitting(self, items, first, room):
        count = used = 0
        for kind, _payload in items[first:]:
            used += self._item_height(kind)
            if used > room:
                break
            count += 1
        return count

    def _max_scroll(self, items, room):
        used = 0
        for index in range(len(items)-1, -1, -1):
            used += self._item_height(items[index][0])
            if used > room:
                return min(index+1, len(items)-1)
        return 0

    def _scroll_to(self, dest, size=None):
        """Scroll the setup page so the line holding dest is in view."""
        size = size or (self.width, self.height)
        items = self._setup_items(size[0])
        room = self._setup_room(size)
        position = next((index for index, (kind, payload) in enumerate(items)
                         if kind == 'line' and
                         any(cell[0] == dest for cell in payload)), None)
        if position is None:
            return
        if position < self.scroll:
            self.scroll = position
            # Keep a section header in view above its first line.
            if position and items[position-1][0] == 'header':
                self.scroll = position-1
        else:
            while (self.scroll < position and position >= self.scroll+
                   self._items_fitting(items, self.scroll, room)):
                self.scroll += 1
        self.scroll = max(0, min(self.scroll, self._max_scroll(items, room)))

    def _footer_content(self):
        """(text, colour, wrap) for the one line at the bottom of the window.

        The selected setting's help, unless something needs attention: a
        fresh notice, a lost device, or a device-discovery error.
        """
        if self.device_error:
            return self.device_error, (255, 182, 132), False
        alert = self.sender_device_lost
        help_text = self._help_text(self.selected)
        if (self.page == 'setup' and help_text and not alert and
                not self._notice_fresh):
            return help_text, (150, 172, 188), True
        color = ((255, 182, 132) if alert else
                 (147, 206, 169) if self.process is not None else
                 (167, 187, 202))
        return self.notice, color, False

    def _footer_lines(self, width):
        text, color, wrap = self._footer_content()
        small = _font(13)
        if wrap:
            lines = _wrapped(text, small, width-28)[:2] or ['']
        else:
            lines = [_fit(text, small, width-28)]
        return lines, color

    def _footer_height(self, width):
        return 13+17*len(self._footer_lines(width)[0])

    def _shown_fields(self, fields):
        source = self.settings['source']
        fields = [dest for dest in fields if not (
            dest == 'video_source' and source != 'video' or
            dest == 'video_live' and source != 'video' or
            dest == 'camera' and source != 'camera' or
            dest == 'screen_target' and source != 'screen' or
            dest == 'ffmpeg_input' and source not in ('camera', 'screen') or
            dest == 'ffmpeg_input' and source == 'screen' and
            self.settings['screen_backend'] != 'ffmpeg' or
            dest in ('screen_backend', 'region') and source != 'screen' or
            dest == 'perceptual_detail_strength' and
            self.settings['perceptual_resize'] == 'off' or
            dest == 'perceptual_resize' and self.settings['dct_encode'] or
            dest == 'dct_encode' and
            self.settings['profile'] not in DCT_PROFILES or
            dest in ('dct_sharpen', 'dct_clarity', 'dct_chroma_gain',
                     'luma_adjust', 'pixel_encode', 'dct_kernel') and
            not self.settings['dct_encode'] or
            dest == 'pixel_detail' and not (
                self.settings['dct_encode'] and
                self.settings.get('pixel_encode')) or
            dest == 'pixel_grid' and not (
                self.settings['dct_encode'] and
                self.settings.get('pixel_encode') and
                self.settings['profile'] == 'aspect-fold-500') or
            # Pixel encode sends the pixels as they are: no enhancement.
            dest in ('dct_sharpen', 'dct_sharpen_strength', 'dct_clarity',
                     'dct_chroma_gain', 'luma_adjust', 'dct_kernel',
                     'luma_adjust_linear') and
            self.settings.get('pixel_encode') or
            # Direct DCT encode takes the frame at its own size: the capture
            # scaler and width do nothing (mouse-follow still uses the width).
            dest == 'capture_filter' and self.settings['dct_encode'] or
            dest == 'capture_width' and self.settings['dct_encode'] and
            source != 'mouse-follow' or
            dest == 'luma_adjust_linear' and not (
                self.settings['dct_encode'] and
                self.settings.get('luma_adjust', True)) or
            dest == 'aspect_layout' and
            self.settings['profile'] not in ASPECT_PROFILES or
            dest == 'aspect_tail' and
            self.settings['profile'] not in ASPECT_TAIL_PROFILES or
            dest == 'dct_sharpen_strength' and not (
                self.settings['dct_encode'] and
                self.settings['dct_sharpen'] != 'off') or
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
        # The chosen kernel's own parameters follow its row.
        return [shown for dest in fields for shown in (
            [dest, *self._kernel_param_fields()] if dest == 'dct_kernel'
            else [dest])]

    def _page_fields(self):
        """Settings shown on the page being displayed."""
        return (self._live_fields() if self.page == 'live'
                else self._visible_fields())

    def _live_fields(self):
        """The settings the sending page offers: tone, then the DCT kernel,
        its parameters and the DCT strengths when the encode uses them."""
        return list(self.LIVE_FIELDS) + self._shown_fields(KERNEL_LIVE_FIELDS)

    def _button_label(self, dest):
        if dest.startswith(KERNEL_PARAM_PREFIX):
            return '    ' + dest[len(KERNEL_PARAM_PREFIX):].replace('_', ' ') + ' · live'
        return FIELD_LABELS.get(dest, dest.replace('_', ' ').capitalize())

    # ---- DCT kernel rows --------------------------------------------------
    def _selected_kernel(self):
        """The chosen kernel, or None (reference, or one no longer on disk)."""
        name = self.settings.get('dct_kernel', 'reference')
        if name in (None, '', 'reference'):
            return None
        try:
            return kernel_registry().get(name)
        except v7_kernels.KernelError:
            return None

    def _kernel_param_fields(self):
        kernel = self._selected_kernel()
        if kernel is None:
            return []
        return [KERNEL_PARAM_PREFIX+name for name in kernel.params]

    def _kernel_param(self, dest):
        """(kernel, parameter name, Param) of a kp: row, or None."""
        kernel = self._selected_kernel()
        name = dest[len(KERNEL_PARAM_PREFIX):]
        if kernel is None or name not in kernel.params:
            return None
        return kernel, name, kernel.params[name]

    def _field_value(self, dest):
        """A setting's value; a kernel parameter's lives in a nested dict."""
        if dest.startswith(KERNEL_PARAM_PREFIX):
            found = self._kernel_param(dest)
            if found is None:
                return ''
            kernel, name, param = found
            saved = (self.settings.get('dct_kernel_params') or {}).get(
                kernel.name, {})
            return saved.get(name, param.default)
        return self.settings.get(dest)

    def _edit_text(self, dest):
        if dest.startswith(KERNEL_PARAM_PREFIX):
            return f'{float(self._field_value(dest) or 0):g}'
        return str(self.settings.get(dest) or '')

    def _set_kernel_param(self, dest, value):
        """Store a kernel parameter (clamped); returns the stored number."""
        kernel, name, param = self._kernel_param(dest)
        try:
            value = param.clamp(_float_setting(value, name))
        except ValueError as exc:
            raise ValueError(f'{name} must be a number.') from exc
        params = self.settings.setdefault('dct_kernel_params', {})
        params.setdefault(kernel.name, {})[name] = value
        return value

    def _help_text(self, dest):
        if dest.startswith(KERNEL_PARAM_PREFIX):
            found = self._kernel_param(dest)
            if found is None:
                return ''
            _kernel, name, param = found
            return (f'{param.help or name} · {param.low:g} to {param.high:g}, '
                    f'default {param.default:g}. Left/Right steps by '
                    f'{param.step:g} (Shift: five times) and is heard on the '
                    f'next frame; Enter types a value.')
        if dest == 'dct_kernel':
            kernel = self._selected_kernel()
            return (FIELD_HELP['dct_kernel'] if kernel is None else
                    f'{kernel.help or kernel.label} (file: {kernel.path.name})')
        return FIELD_HELP.get(dest, '')

    def _numeric_step(self, dest):
        """(low, high, step) for a number Left/Right can step, else None."""
        if dest.startswith(KERNEL_PARAM_PREFIX):
            found = self._kernel_param(dest)
            return None if found is None else (
                found[2].low, found[2].high, found[2].step)
        return NUMERIC_STEPS.get(dest)

    def _step_field(self, dest, direction, fast=False):
        low, high, step = self._numeric_step(dest)
        step *= 5 if fast else 1
        current = _float_setting(self._field_value(dest), dest)
        value = round(min(max(current+direction*step, low), high), 6)
        if dest.startswith(KERNEL_PARAM_PREFIX):
            value = self._set_kernel_param(dest, value)
        else:
            self.settings[dest] = f'{value:g}'
        self.notice = f'{self._button_label(dest).strip()}: {value:g}'
        self._persist_preferences()
        if self.process is not None:
            self._send_live_kernel_update()
        self.dirty = True

    def _cycle_kernel(self, direction):
        names = [name for _label, name in self._choices('dct_kernel')]
        current = self.settings.get('dct_kernel', 'reference')
        index = names.index(current) if current in names else 0
        self._assign('dct_kernel', names[(index+direction) % len(names)])

    def _reload_kernels(self):
        """Rescan the kernel folders (here, and in a running sender)."""
        registry = kernel_registry()
        added, removed = registry.scan()
        if self.settings.get('dct_kernel', 'reference') not in registry.names():
            self.settings['dct_kernel'] = 'reference'
        text = ', '.join([f'+{name}' for name in added] +
                         [f'-{name}' for name in removed]) or 'no change'
        skipped = ''.join(f'; skipped {Path(path).name}: {message}'
                          for path, message in registry.errors)
        self.notice = f'DCT kernels rescanned: {text}{skipped}'
        if self.process is not None:
            try:
                self._write_control({'kernels': 'reload'})
                self._send_live_kernel_update()
            except (OSError, ValueError, RuntimeError) as exc:
                self.notice = str(exc)
        self.dirty = True

    def _write_control(self, message):
        control = getattr(self.process, 'stdin', None)
        if control is None:
            raise RuntimeError('Live sender controls are unavailable.')
        control.write(json.dumps(message)+'\n')
        control.flush()

    def _send_live_kernel_update(self):
        """Send the whole kernel state (kernel, its values, DCT strengths)."""
        strength = _float_setting(self.settings.get('dct_sharpen_strength', '0.25'),
                                  'DCT sharpen strength')
        clarity = _float_setting(self.settings.get('dct_clarity', '0'), 'DCT clarity')
        chroma = _float_setting(self.settings.get('dct_chroma_gain', '1'),
                                'DCT chroma gain')
        if not (0 <= strength <= 1 and 0 <= clarity <= 1 and 1 <= chroma <= 1.3):
            raise ValueError('DCT sharpen and clarity are 0 to 1; chroma gain '
                             'is 1.0 to 1.3.')
        message = {'dct': {'sharpen': self.settings.get('dct_sharpen', 'off'),
                           'sharpen_strength': strength, 'clarity': clarity,
                           'chroma_gain': chroma}}
        kernel = self._selected_kernel()
        message['kernel'] = 'reference' if kernel is None else kernel.name
        if kernel is not None:
            message['kernel_params'] = _kernel_values(self.settings, kernel)
        self._write_control(message)

    def _value_label(self, dest):
        if dest.startswith(KERNEL_PARAM_PREFIX):
            found = self._kernel_param(dest)
            if found is None:
                return '—'
            value = self._field_value(dest)
            param = found[2]
            return f'{float(value):g}   ({param.low:g} to {param.high:g})'
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
        if dest in ('source', 'profile', 'screen_backend', 'preview',
                    'encode_filter', 'capture_filter', 'perceptual_resize',
                    'dct_sharpen', 'aspect_layout', 'aspect_tail',
                    'pixel_detail', 'pixel_grid', 'mono_video_side',
                    'source_audio', 'source_audio_input_side', 'capture_fps',
                    'dct_kernel'):
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
        if (value in FOLDED_PROFILES and
                self.settings['encode_filter'] != 'box'):
            self.settings['encode_filter'] = 'auto'

    def _assign(self, dest, value):
        self.settings[dest] = value
        if dest == 'preview_stage':
            self._forward_popout()
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
            if value not in FOLDED_PROFILES:
                self.settings['perceptual_resize'] = 'off'
            if value not in DCT_PROFILES:
                self.settings['dct_encode'] = False
        elif dest == 'dct_encode' and value:
            # The direct encode replaces the resize the downscaler shapes.
            self.settings['perceptual_resize'] = 'off'
        elif dest == 'perceptual_resize' and value != 'off':
            self.settings['dct_encode'] = False
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
        message = f'{dest.replace("_", " ").capitalize()} updated.'
        if self.process is not None and dest in ('dct_kernel', 'dct_sharpen'):
            try:
                self._send_live_kernel_update()
            except (OSError, ValueError, RuntimeError) as exc:
                message = str(exc)
        self.notice = message
        self.dirty = True
        self._persist_preferences()

    def _select_option(self, dest, value):
        self._assign(dest, value)

    def _build_command(self, image_preview_port=None, video_start=None):
        extra = {'video_start': video_start} if video_start else {}
        return build_command(self.settings, self.devices, self._sounddevice(),
                             audio_devices=self.audio_devices,
                             image_preview_port=image_preview_port, **extra)

    # ---- video-file transport ------------------------------------------
    def _file_source(self):
        return is_file_source(self.settings)

    def _resume_position(self):
        if not self._file_source():
            return 0.0
        return resume_position(self.resume_positions,
                               self.settings['video_source'])

    def _remember_position(self, position, duration=None, persist=False):
        """Keep where this file is, so Start on it resumes there."""
        if not self._file_source():
            return
        key = _resume_key(self.settings['video_source'])
        signature = _file_signature(self.settings['video_source'])
        position = _number(position)
        if signature is None or position is None or position <= 0:
            changed = self.resume_positions.pop(key, None) is not None
        else:
            previous = self.resume_positions.pop(key, None) or {}
            if duration is None:
                duration = previous.get('duration')
            self.resume_positions[key] = {
                'position': round(position, 3), 'duration': _number(duration),
                **signature}
            while len(self.resume_positions) > RESUME_ENTRY_LIMIT:
                self.resume_positions.pop(next(iter(self.resume_positions)))
            changed = True
        if persist and changed:
            self._resume_saved_at = time.monotonic()
            self._persist_preferences()

    def _transport_state(self, dragging=True):
        """(position, duration, paused) shown by the transport controls.

        While the seek bar is dragged the position is the one under the
        pointer, whatever the sender reports meanwhile.
        """
        if self.process is not None and self.playback is not None:
            state = (self.playback['position'], self.playback['duration'],
                     self.playback['paused'])
        else:
            entry = self.resume_positions.get(
                _resume_key(self.settings.get('video_source', ''))) or {}
            state = (self._resume_position(), _number(entry.get('duration')),
                     False)
        if dragging and self._seek_drag is not None and state[1]:
            state = (self._seek_drag*state[1],)+state[1:]
        return state

    # ---- seek bar: press, drag, release ---------------------------------
    def _seek_fraction(self, x):
        left, _top, right, _bottom = self._seek_track
        return max(0.0, min(1.0, (x-left)/max(1, right-left)))

    def _seek_press(self, x):
        """The bar follows the pointer; one seek is sent on release."""
        self._seek_track = self.hits['transport:seek']
        fraction = self._seek_fraction(x)
        if not self._transport_state()[1]:
            self._transport('seek', fraction)   # says the length is unknown
            return
        self._seek_drag = fraction
        self.dirty = True

    def _on_cursor(self, _window, x, _y):
        if self._seek_drag is not None:
            self._seek_drag = self._seek_fraction(x)
            self.dirty = True

    def _seek_release(self, x):
        if self._seek_drag is None:
            return
        fraction = self._seek_fraction(x)
        self._seek_drag = None
        self._transport('seek', fraction)

    # ---- pop-out preview window -----------------------------------------
    def _open_popout(self):
        """Start the pop-out window process; sending never depends on it.

        Returns a message when it could not be started.
        """
        try:
            process = subprocess.Popen(
                [sys.executable, str(ROOT/'tools'/'v7_preview_popout.py')],
                cwd=str(ROOT), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, encoding='utf-8',
                errors='replace', bufsize=1)
        except (OSError, ValueError) as exc:
            return ('Pop-out window unavailable; sending without a preview: '
                    f'{exc}')
        self.popout = process
        self._popout_address = None
        threading.Thread(target=self._read_popout, args=(process,),
                         name='v7-send-gui-popout', daemon=True).start()
        return None

    def _read_popout(self, process):
        """Learn the window's port; report when the window is gone."""
        message = None
        try:
            for line in process.stdout:
                try:
                    record = json.loads(line)
                    status = record.get('status')
                except (TypeError, ValueError, AttributeError):
                    continue
                if status == 'popout_ready' and self.popout is process:
                    self._popout_address = ('127.0.0.1', int(record['port']))
                    self._forward_popout()
                elif status == 'popout_error':
                    message = str(record.get('message') or
                                  'Pop-out window unavailable.')
        except Exception as exc:
            message = f'Pop-out window failed: {exc}'
        self.events.put(('popout_exit', (process, message)))
        self._wake()

    def _forward_popout(self, stage=None):
        """Send the newest picture of the selected stage to the pop-out."""
        selected = self.settings.get('preview_stage', 'resized')
        address, channel = self._popout_address, self._preview_socket
        newest = self.preview_datagrams.get(selected)
        if (stage not in (None, selected) or address is None or
                channel is None or newest is None):
            return
        try:
            channel.sendto(newest[1], address)
        except OSError:
            pass                          # the next picture follows shortly

    @staticmethod
    def _end_popout(process):
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                process.terminate()
            except OSError:
                pass
        except OSError:
            pass

    def _close_popout(self):
        """Close the window: it ends when its control pipe closes."""
        process, self.popout = self.popout, None
        self._popout_address = None
        if process is None:
            return
        try:
            process.stdin.close()
        except (OSError, ValueError, AttributeError):
            pass
        threading.Thread(target=self._end_popout, args=(process,),
                         name='v7-send-gui-popout-close', daemon=True).start()

    def _send_transport(self, command, position=None):
        control = getattr(self.process, 'stdin', None)
        if control is None:
            raise RuntimeError('Live sender controls are unavailable.')
        message = {'transport': command}
        if position is not None:
            message['position'] = round(float(position), 3)
        control.write(json.dumps(message)+'\n')
        control.flush()

    def _transport(self, action, fraction=None):
        """A click on play/pause, restart or the seek bar."""
        if not self._file_source():
            return
        position, duration, paused = self._transport_state(dragging=False)
        running = self.process is not None and not self.stop_requested
        try:
            if action == 'play_pause':
                if not running:
                    self.notice = 'Press Start to send this file.'
                    return
                self._send_transport('play' if paused else 'pause')
                if self.playback is not None:
                    self.playback['paused'] = not paused
                self.notice = ('Playing.' if paused else
                               'Paused: still sending the held picture.')
            elif action == 'restart':
                if running:
                    self._send_transport('restart')
                    if self.playback is not None:
                        self.playback['position'] = 0.0
                self._remember_position(0.0, persist=True)
                self.notice = ('Restarted from the beginning.' if running else
                               'Start will begin at the beginning.')
            elif action == 'seek':
                if not duration:
                    self.notice = 'The length of this file is not known yet.'
                    return
                target = max(0.0, min(1.0, float(fraction)))*duration
                target = min(target, max(0.0, duration-.25))
                if running:
                    self._send_transport('seek', target)
                    if self.playback is not None:
                        self.playback['position'] = target
                self._remember_position(target, duration, persist=True)
                self.notice = f'Position {_clock_text(target)}.'
        except (OSError, ValueError, RuntimeError) as exc:
            self.notice = str(exc)
        finally:
            self.dirty = True

    def _playback_report(self, record):
        """A {"status": "playback"} line from the sender."""
        position = _number(record.get('position'))
        if position is None:
            return
        paused = bool(record.get('paused'))
        was_paused = bool(self.playback and self.playback['paused'])
        self.playback = {'position': position,
                         'duration': _number(record.get('duration')),
                         'paused': paused}
        self._remember_position(
            position, self.playback['duration'],
            persist=(paused != was_paused or
                     time.monotonic()-self._resume_saved_at > 5.0))

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
            if self.settings.get('preview') in ('window', 'popout'):
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
            # A file resumes where it was; nothing is passed from the start.
            resume = self._resume_position()
            extra = {'video_start': resume} if resume > 0 else {}
            if preview_socket is not None:
                command = self._build_command(
                    image_preview_port=preview_socket.getsockname()[1],
                    **extra)
            else:
                command = self._build_command(**extra)
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
        self.playback = None
        self._seek_drag = None
        self._preview_socket = preview_socket
        self.preview_datagrams = {}
        if (preview_socket is not None and
                self.settings.get('preview') == 'popout'):
            preview_warning = self._open_popout()
        self.preview_image = None
        self.preview_counter = None
        self.preview_aspect = None
        self.preview_stage = None
        self.preview_handoff_ns = None
        self.preview_stage_frames = {}
        self.preview_error = preview_warning
        self.preview_reader_error_reported = False
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
        def report_error(context, exc):
            if (self.preview_reader_error_reported or
                    self.process is not process or
                    self._preview_socket is not preview_socket):
                return
            self.preview_reader_error_reported = True
            detail = f'{context}: {type(exc).__name__}: {exc}'
            code = getattr(exc, 'errno', None)
            winerror = getattr(exc, 'winerror', None)
            if code is not None:
                code_name = errno.errorcode.get(code, '')
                detail += (f' (errno {code}'
                           f'{": "+code_name if code_name else ""})')
            if winerror is not None:
                detail += f' (Windows error {winerror})'
            self.events.put(('line', json.dumps({
                'status': 'image_preview_error',
                'message': detail,
            })))
            self._wake()

        while (self.process is process and
               self._preview_socket is preview_socket):
            try:
                packet, _address = preview_socket.recvfrom(65507)
            except socket.timeout:
                continue
            except OSError as exc:
                report_error('receiving preview datagram', exc)
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
            except OSError as exc:
                report_error('draining preview datagrams', exc)
                return
            while True:
                try:
                    candidate, _address = preview_socket.recvfrom(65507)
                except BlockingIOError:
                    break
                except OSError as exc:
                    report_error('draining preview datagrams', exc)
                    return
                remember(candidate)
            try:
                preview_socket.settimeout(.2)
            except OSError as exc:
                report_error('restoring preview socket timeout', exc)
                return
            for _handoff_ns, newest in newest_by_stage.values():
                parsed = parse_preview_datagram(newest)
                if parsed is None:
                    continue
                counter, aspect, handoff_ns, stage, jpeg = parsed
                if (self.process is not process or
                        self._preview_socket is not preview_socket):
                    continue
                earlier = self.preview_datagrams.get(stage)
                if earlier is None or handoff_ns >= earlier[0]:
                    self.preview_datagrams[stage] = (handoff_ns, newest)
                    # The pop-out gets the datagram the pane would decode.
                    self._forward_popout(stage)
                if self.settings.get('preview') == 'popout':
                    continue             # nothing is drawn in this window
                try:
                    from PIL import Image
                    import io
                    with Image.open(io.BytesIO(jpeg)) as image:
                        image.load()
                        decoded = image.convert('RGB')
                except (OSError, ValueError) as exc:
                    report_error(f'decoding {stage} preview image', exc)
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
                self.preview_error = None
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
                try:
                    status_record = json.loads(value)
                    status = status_record.get('status')
                except (TypeError, ValueError, AttributeError):
                    status_record = {}
                    status = None
                if status == 'playback':
                    # Several a second: the transport bar, not the log.
                    self._playback_report(status_record)
                    continue
                if status == 'kernel':
                    message = str(status_record.get('message') or '')
                    if message:
                        self.lines.append(message)
                        self.lines = self.lines[-12:]
                        self.notice = message
                    continue
                self.lines.append(value)
                self.lines = self.lines[-12:]
                self.notice = value
                if status == 'sender_device_lost':
                    self.sender_device_lost = True
                elif status == 'sender_device_reconnected':
                    self.sender_device_lost = False
                elif status in ('image_preview_error',
                                'image_preview_unavailable'):
                    self.preview_error = status_record.get(
                        'message', 'Image preview failed.')
            elif kind == 'exit':
                return_code = int(value)
                self.process = None
                if self.playback is not None:
                    self._remember_position(self.playback['position'],
                                            self.playback['duration'])
                    self.playback = None
                    self._persist_preferences()
                self._close_preview_socket()
                self._close_popout()
                self._seek_drag = None
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
            elif kind == 'popout_exit':
                process, message = value
                if process is self.popout:      # not closed by this GUI
                    self.popout = None
                    self._popout_address = None
                    self.preview_error = self.notice = message or (
                        'Pop-out window closed; still sending.'
                        if self.process is not None else
                        'Pop-out window closed.')
            elif kind == 'devices':
                self._apply_device_snapshot(value)
        if changed:
            self.dirty = True

    def _finish_edit(self, commit=True):
        dest = self.selected
        if commit and dest.startswith(KERNEL_PARAM_PREFIX):
            previous = copy.deepcopy(self.settings.get('dct_kernel_params'))
            try:
                value = self._set_kernel_param(dest, self.edit_buffer)
                if self.process is not None:
                    self._send_live_kernel_update()
                self.notice = f'{self._button_label(dest).strip()}: {value:g}'
                self._persist_preferences()
            except (OSError, ValueError, RuntimeError, TypeError) as exc:
                self.settings['dct_kernel_params'] = previous
                self.notice = str(exc)
            self.editing = False
            self.dirty = True
            return
        if commit:
            previous = self.settings.get(dest)
            self.settings[dest] = self.edit_buffer
            if dest == 'video_source':
                self.capture_choice_cache.pop('capture_fps', None)
            try:
                if self.process is not None and dest in LIVE_FIELDS:
                    self._send_live_tone_update()
                elif self.process is not None and dest in NUMERIC_STEPS:
                    self._send_live_kernel_update()
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

    def _cell_rect(self, width, column, span, y):
        columns = self._columns(width)
        cell = (width-2*self.GUTTER-(columns-1)*self.CELL_GAP)//columns
        left = self.GUTTER+column*(cell+self.CELL_GAP)
        return (left, y, left+span*cell+(span-1)*self.CELL_GAP,
                y+self.ROW_HEIGHT-3)

    def _render_header(self, draw, small, title, y, width):
        draw.text((24, y+6), title, font=small, fill=(229, 237, 243))
        label_right = 24+int(small.getlength(title))+12
        draw.line((label_right, y+14, width-self.GUTTER, y+14),
                  fill=(47, 68, 83), width=1)

    def _render_setup(self, image, draw, font, small):
        width, height = image.size
        fields = self._visible_fields()
        items = self._setup_items(width)
        room = self._setup_room(image.size)
        self.scroll = max(0, min(self.scroll, self._max_scroll(items, room)))
        if self.selected not in fields and fields:
            self.selected = fields[0]
        count = self._items_fitting(items, self.scroll, room)
        y = self.SETUP_TOP
        for kind, payload in items[self.scroll:self.scroll+count]:
            if kind == 'header':
                self._render_header(draw, small, payload, y, width)
            elif kind == 'note':
                draw.text((24, y+5), payload, font=small,
                          fill=(238, 182, 125))
            else:
                for dest, column, span in payload:
                    self._render_cell(
                        draw, small, dest,
                        self._cell_rect(width, column, span, y))
            y += self._item_height(kind)
        if count < len(items):
            # Position marker: the page scrolls by line like the dropdowns.
            track_top, track_bottom = self.SETUP_TOP, self.SETUP_TOP+room
            span = track_bottom-track_top
            thumb = max(18, span*count//len(items))
            offset = (span-thumb)*self.scroll//max(1, len(items)-count)
            draw.rectangle((width-11, track_top, width-8, track_bottom),
                           fill=(17, 29, 39))
            draw.rectangle((width-11, track_top+offset, width-8,
                            track_top+offset+thumb), fill=(74, 111, 134))
        if self.dropdown is not None:
            self._render_dropdown(draw, small, width, height)

    def _render_cell(self, draw, small, dest, rect):
        locked = self._locked(dest)
        if dest in BOOL_FIELDS:
            self._render_button(draw, small, dest, rect, locked)
        else:
            self._render_row(draw, small, dest, rect, locked)

    def _render_button(self, draw, small, dest, rect, locked=False):
        """One on/off setting: filled when on, outlined and dim when off."""
        left, top, right, _bottom = rect
        value = bool(self.settings[dest])
        selected = dest == self.selected
        draw.rounded_rectangle(
            rect, radius=4,
            fill=(28, 52, 66) if locked and value else
            (12, 19, 26) if locked else
            (43, 94, 123) if value else (12, 21, 29),
            outline=(160, 205, 226) if selected and not locked else
            (98, 145, 169) if value and not locked else (47, 68, 83),
            width=2 if selected and not locked else 1)
        state = 'On' if value else 'Off'
        state_left = right-38
        on_ink = (246, 250, 252) if not locked else (140, 158, 170)
        off_ink = (133, 159, 177) if not locked else (96, 114, 126)
        draw.text((left+12, top+8),
                  _fit(self._button_label(dest), small, state_left-left-22),
                  font=small, fill=on_ink if value else off_ink)
        draw.text((state_left, top+8), state, font=small,
                  fill=on_ink if value else off_ink)
        self.hits[f'field:{dest}'] = rect

    def _render_row(self, draw, small, dest, rect, locked=False):
        """One setting: label, value, and the picker arrow or Browse."""
        left, top, right, bottom = rect
        selected = dest == self.selected and not locked
        draw.rounded_rectangle(
            rect, radius=4,
            fill=(35, 60, 77) if selected else
            (13, 21, 28) if locked else (17, 29, 39),
            outline=(74, 111, 134) if selected else
            (28, 42, 53) if locked else (32, 48, 60), width=1)
        label_width = (190 if right-left >= 600 else
                       min(int((right-left)*.55),
                           int(small.getlength(self._button_label(dest)))+22))
        draw.text((left+10, top+8),
                  _fit(self._button_label(dest), small, label_width-16),
                  font=small,
                  fill=(100, 118, 131) if locked else (205, 218, 228))
        edge = right-8
        field_right = right
        if dest in self.DROPDOWN_FIELDS:
            draw.text((right-24, top+7), '▾', font=small,
                      fill=(80, 100, 114) if locked else (134, 169, 188))
            edge = right-28
        if dest == 'video_source':
            browse = (right-78, top+4, right-6, bottom-4)
            draw.rounded_rectangle(
                browse, radius=4, fill=(20, 36, 46) if locked else (30, 58, 76),
                outline=(48, 68, 82) if locked else (75, 111, 132), width=1)
            draw.text((browse[0]+10, top+8), 'Browse', font=small,
                      fill=(110, 128, 142) if locked else (229, 239, 246))
            self.hits['browse:video_source'] = browse
            edge = browse[0]-8
            field_right = browse[0]-4
        if dest in self.MATCH_FIELDS:
            # Not signalled on the wire: the receiver must be set the same.
            tag_width = int(small.getlength('match'))+14
            tag = (edge-tag_width, top+5, edge, bottom-5)
            ink = (92, 120, 136) if locked else (140, 192, 212)
            draw.rounded_rectangle(tag, radius=4, outline=ink, width=1)
            draw.text((tag[0]+7, tag[1]+3), 'match', font=small, fill=ink)
            edge = tag[0]-8
        value = (self.edit_buffer if self.editing and dest == self.selected
                 else self._value_label(dest))
        value_left = left+label_width
        draw.text((value_left, top+8),
                  _fit(value, small, max(20, edge-value_left)), font=small,
                  fill=(120, 136, 148) if locked else (237, 242, 246))
        self.hits[f'field:{dest}'] = (left, top, field_right, bottom)

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
        menu_width = min(max(anchor[2]-anchor[0], 320), width-2*self.GUTTER)
        left = max(self.GUTTER, min(anchor[0], width-self.GUTTER-menu_width))
        right = left+menu_width
        top = anchor[3]+2
        if top+max_items*28+4 > height-self._footer_height(width):
            top = max(self.SETUP_TOP, anchor[1]-max_items*28-4)
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

    def _status_facts(self):
        device = self._device()
        settings = self.settings
        profile = next((label for label, value in PROFILE_CHOICES
                        if value == settings['profile']), 'Unknown')
        source = next((label for label, value in SOURCE_CHOICES
                       if value == settings['source']), 'Not selected')
        facts = [source, profile.split(' · ')[0],
                 (device.name if device else 'No output') + ' · ' +
                 device_rate_text(device),
                 'Direct DCT' if settings.get('dct_encode') else 'Resize']
        if settings.get('profile') in ASPECT_PROFILES:
            layout = settings.get('aspect_layout', 'auto')
            if settings.get('profile') in ASPECT_TAIL_PROFILES:
                layout += ' / '+settings.get('aspect_tail',
                                             DEFAULT_ASPECT_TAIL)
            facts.append('aspect '+layout)
        facts.append(f"{settings['speed']}×")
        return '  ·  '.join(facts)

    def _render_stage_buttons(self, draw, small, left, top, limit):
        """The Source / Encoder input switch; returns where it ends."""
        for stage, label in PREVIEW_STAGE_LABELS:
            right = left+int(small.getlength(label))+20
            if right > limit:
                break
            rect = (left, top, right, top+24)
            active = self.settings.get('preview_stage') == stage
            draw.rounded_rectangle(
                rect, radius=4, fill=(42, 78, 99) if active else (17, 29, 39),
                outline=(94, 143, 168) if active else (48, 73, 90))
            draw.text((left+10, top+5), label, font=small,
                      fill=(235, 242, 247))
            self.hits[f'preview_stage:{stage}'] = rect
            left = right+6
        return left

    def _render_live(self, image, draw, font, small):
        width, height = image.size
        gutter = self.GUTTER
        preview_enabled = self.settings.get('preview') == 'window'
        popout = self.settings.get('preview') == 'popout'
        transport = self._file_source()
        held = (transport and self.process is not None and
                self.playback is not None and self.playback['paused'])
        state = ('STOPPING' if self.stop_requested else
                 'DEVICE LOST' if self.sender_device_lost else
                 'SENDING · PAUSED' if held else
                 'SENDING' if self.process is not None else 'STOPPED')

        # One status strip: the state, then the facts that used to be a table.
        strip = (gutter, self.SETUP_TOP, width-gutter, self.SETUP_TOP+34)
        draw.rounded_rectangle(strip, radius=6, fill=(17, 29, 39),
                               outline=(48, 73, 90))
        draw.text((gutter+12, strip[1]+7), state, font=font,
                  fill=(238, 140, 110) if state == 'DEVICE LOST' else
                  (238, 182, 125) if state == 'STOPPING' else
                  (145, 218, 170) if state.startswith('SENDING')
                  else (188, 202, 213))
        facts_left = gutter+12+int(font.getlength(state))+18
        facts_right = strip[2]-10
        if popout:
            # The pop-out window shows the stage chosen here.
            labels = [label for _stage, label in PREVIEW_STAGE_LABELS]
            switch = sum(int(small.getlength(label))+26 for label in labels)
            facts_right -= switch+66
            self._render_stage_buttons(draw, small, facts_right+66,
                                       strip[1]+5, strip[2]-6)
            draw.text((facts_right+10, strip[1]+10), 'Pop-out', font=small,
                      fill=(132, 158, 176))
        draw.text((facts_left, strip[1]+10),
                  _fit(self._status_facts(), small,
                       max(40, facts_right-facts_left)),
                  font=small, fill=(218, 229, 237))

        # From the bottom up: messages, transport, live controls.
        bottom = height-self._footer_height(width)-6
        if preview_enabled:
            log_top = bottom-3*self.LOG_LINE-2
            for index, line in enumerate(self.lines[-3:]):
                draw.text((gutter+6, log_top+index*self.LOG_LINE),
                          _fit(line, small, width-2*gutter-12), font=small,
                          fill=(183, 201, 214))
            bottom = log_top-4
        if transport:
            transport_top = bottom-self.TRANSPORT_HEIGHT
            self._render_transport(
                draw, small, (gutter, transport_top, width-gutter, bottom))
            bottom = transport_top-6
        columns = self._columns(width)
        live_lines = self._pack(self._live_fields(), columns)
        live_top = bottom-len(live_lines)*self.ROW_HEIGHT
        for index, line in enumerate(live_lines):
            for dest, column, span in line:
                self._render_cell(
                    draw, small, dest,
                    self._cell_rect(width, column, span,
                                    live_top+index*self.ROW_HEIGHT))
        bottom = live_top-6

        area_top = strip[3]+8
        if not preview_enabled:
            lines_fit = max(0, (bottom-area_top)//self.LOG_LINE)
            for index, line in enumerate(
                    self.lines[-lines_fit:] if lines_fit else ()):
                draw.text((gutter+6, area_top+index*self.LOG_LINE),
                          _fit(line, small, width-2*gutter-12), font=small,
                          fill=(183, 201, 214))
            return

        from PIL import ImageOps
        panel = (gutter, area_top, width-gutter, max(area_top+40, bottom))
        draw.rounded_rectangle(panel, radius=6, fill=(12, 21, 29),
                               outline=(48, 73, 90))
        # The switch and the caption sit along the top edge of the picture.
        buttons_end = self._render_stage_buttons(
            draw, small, panel[0]+8, panel[1]+5, panel[2])
        if self.preview_image is None:
            if self.preview_error:
                message = self.preview_error
            elif self.process:
                message = 'Waiting for the first handed-off frame…'
            else:
                message = 'Start sending to see the frame being encoded.'
            draw.text((panel[0]+16, panel[1]+38),
                      _fit(message, small, panel[2]-panel[0]-32),
                      font=small, fill=(165, 187, 202))
            return
        inner = (max(1, panel[2]-panel[0]-16), max(1, panel[3]-panel[1]-36))
        thumbnail = ImageOps.contain(self.preview_image, inner)
        x = panel[0]+(panel[2]-panel[0]-thumbnail.width)//2
        y_image = panel[1]+32+(inner[1]-thumbnail.height)//2
        image.paste(thumbnail.convert('RGBA'), (x, y_image))
        age_ms = max(0.0, (time.monotonic_ns()-
                           int(self.preview_handoff_ns or 0))/1e6)
        stage = dict(PREVIEW_STAGE_LABELS).get(self.preview_stage, 'Image')
        caption = (f'{stage} · packet {self.preview_counter} · aspect '
                   f'{self.preview_aspect} · {age_ms:.0f} ms after '
                   'output handoff')
        room = panel[2]-8-buttons_end-8
        shown = _fit(caption, small, max(40, room))
        draw.text((panel[2]-8-int(small.getlength(shown)), panel[1]+9),
                  shown, font=small,
                  fill=(145, 218, 170) if age_ms < 500 else (238, 182, 125))

    def _render_transport(self, draw, small, rect):
        """Play/pause, restart and a seek bar with position and duration."""
        left, top, right, bottom = rect
        position, duration, paused = self._transport_state()
        running = self.process is not None and not self.stop_requested
        ink = (235, 242, 247) if running else (110, 132, 148)
        size = bottom-top
        middle = (top+bottom)//2
        play_rect = (left, top, left+size+6, bottom)
        restart_rect = (play_rect[2]+6, top, play_rect[2]+6+size+6, bottom)
        for key, button in (('play_pause', play_rect),
                            ('restart', restart_rect)):
            draw.rounded_rectangle(
                button, radius=4,
                fill=(43, 94, 123) if running and key == 'play_pause'
                else (17, 29, 39), outline=(74, 111, 134), width=1)
            self.hits[f'transport:{key}'] = button
        x = (play_rect[0]+play_rect[2])//2
        if running and not paused:              # pause: two bars
            draw.rectangle((x-7, middle-8, x-3, middle+8), fill=ink)
            draw.rectangle((x+3, middle-8, x+7, middle+8), fill=ink)
        else:                                   # play: a triangle
            draw.polygon(((x-6, middle-9), (x-6, middle+9), (x+9, middle)),
                         fill=ink)
        x = (restart_rect[0]+restart_rect[2])//2
        restart_ink = (235, 242, 247)           # also works when stopped
        draw.rectangle((x-9, middle-8, x-6, middle+8), fill=restart_ink)
        draw.polygon(((x+8, middle-9), (x+8, middle+9), (x-5, middle)),
                     fill=restart_ink)
        time_text = (f'{_clock_text(position)} / {_clock_text(duration)}'
                     if duration else _clock_text(position))
        time_width = int(small.getlength(time_text))
        draw.text((right-time_width, middle-8), time_text, font=small,
                  fill=(218, 229, 237))
        track_left = restart_rect[2]+16
        track_right = right-time_width-14
        if track_right-track_left < 24:
            return
        draw.rounded_rectangle((track_left, middle-3, track_right, middle+3),
                               radius=3, fill=(17, 29, 39),
                               outline=(48, 73, 90))
        if duration:
            fraction = max(0.0, min(1.0, position/duration))
            knob = track_left+int((track_right-track_left)*fraction)
            draw.rounded_rectangle((track_left, middle-3, knob, middle+3),
                                   radius=3, fill=(94, 143, 168))
            draw.ellipse((knob-6, middle-6, knob+6, middle+6),
                         fill=(160, 205, 226))
        # The whole height of the row takes the click, not the thin track.
        self.hits['transport:seek'] = (track_left, top, track_right, bottom)

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
            ('setup', 'Setup', 12, 104),
            ('live', 'Live', 112, 184),
            ('start_stop', 'Stop' if self.process is not None else 'Start',
             192, 284),
        ]
        if self.page == 'live':
            controls.append(('change_source', 'Change source', 292, 412))
        for key, label, left, right in controls:
            rect = (left, 9, right, 45)
            self.hits[key] = rect
            active = ((key == 'setup' and self.page == 'setup') or
                      (key == 'live' and self.page == 'live'))
            color = ((39, 67, 86) if active else
                     (100, 51, 41) if key == 'start_stop' and
                     self.sender_device_lost else
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
            if self.dropdown is not None:
                self._render_dropdown(draw, small, width, height)

        lines, color = self._footer_lines(width)
        footer_top = height-self._footer_height(width)
        draw.rectangle((0, footer_top, width, height), fill=(10, 18, 25))
        for index, line in enumerate(lines):
            draw.text((14, footer_top+8+index*17), line, font=small,
                      fill=color)
        return image

    def _on_mouse(self, glfw, window, button, action, _mods):
        right_button = getattr(glfw, 'MOUSE_BUTTON_RIGHT', None)
        if (self._seek_drag is not None and action != glfw.PRESS and
                button == glfw.MOUSE_BUTTON_LEFT):
            self._seek_release(glfw.get_cursor_pos(window)[0])
            return
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
            if hit is None or not hit.startswith('field:'):
                return
            dest = hit.split(':', 1)[1]
            if (dest not in self._page_fields() or
                    dest in self.DROPDOWN_FIELDS or
                    dest in BOOL_FIELDS):
                return
            if self._locked(dest):
                self.notice = 'Settings are locked while the sender is running.'
                self.dirty = True
                return
            if self.editing and self.selected != dest:
                self._finish_edit()
            if not self.editing:
                self.selected = dest
                self.editing = True
                current = self._field_value(dest)
                self.edit_buffer = '' if current is None else str(current)
            self._paste_clipboard(glfw, window)
            return
        if self.editing:
            self._finish_edit()
        if hit == 'setup':
            self.page, self.dropdown = 'setup', None
        elif hit == 'live':
            self.page, self.dropdown = 'live', None
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
        elif hit in ('transport:play_pause', 'transport:restart'):
            self._transport(hit.split(':', 1)[1])
        elif hit == 'transport:seek':
            self._seek_press(x)
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
        elif hit and hit.startswith('field:'):
            dest = hit.split(':', 1)[1]
            self.selected = dest
            if self._locked(dest):
                self.notice = 'Settings are locked while the sender is running.'
            elif dest in BOOL_FIELDS:
                self._assign(dest, not self.settings[dest])
            elif dest in self.DROPDOWN_FIELDS:
                self._open_dropdown(dest)
            elif dest in self._page_fields():
                self.editing = True
                current = self._field_value(dest)
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
        elif key == getattr(glfw, 'KEY_R', -1):
            self._reload_kernels()
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
        elif self.page in ('setup', 'live'):
            fields = self._page_fields()
            live_tone_selected = _is_live_field(self.selected)
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
                if self.page == 'setup':
                    self._scroll_to(self.selected)
            elif key in (glfw.KEY_LEFT, glfw.KEY_RIGHT) and self.selected in fields:
                dest = self.selected
                direction = 1 if key == glfw.KEY_RIGHT else -1
                if dest in BOOL_FIELDS:
                    self._assign(dest, not self.settings[dest])
                elif dest == 'dct_kernel':
                    self._cycle_kernel(direction)         # also while sending
                elif self._numeric_step(dest) is not None:
                    try:
                        self._step_field(dest, direction, bool(mods & getattr(glfw, 'MOD_SHIFT', 1)))
                    except (OSError, ValueError, RuntimeError) as exc:
                        self.notice = str(exc)
                elif dest in self.DROPDOWN_FIELDS:
                    self._open_dropdown(dest)
                else:
                    self.editing = True
                    self.edit_buffer = self._edit_text(dest)
            elif key in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER) and self.selected in fields:
                dest = self.selected
                if dest in self.DROPDOWN_FIELDS:
                    self._open_dropdown(dest)
                elif dest in BOOL_FIELDS:
                    self._assign(dest, not self.settings[dest])
                else:
                    self.editing = True
                    self.edit_buffer = self._edit_text(dest)
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
            self.scroll = max(0, min(
                self._max_scroll(self._setup_items(self.width),
                                 self._setup_room()),
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
            glfw.set_cursor_pos_callback(window, self._on_cursor)
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
            self._close_popout()
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
    import argparse
    parser = argparse.ArgumentParser(description='V7 sender GUI')
    parser.add_argument(
        '--dct-kernel-dir', action='append', default=[], metavar='DIR',
        help=('another folder of DCT downscale kernels, besides dct_kernels/ '
              'and $V7_KERNEL_DIR (repeatable)'))
    KERNEL_DIRS.extend(parser.parse_args().dct_kernel_dir)
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
