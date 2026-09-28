"""Interactive, unstarted configuration and live monitor for the V7 receiver.

The interface is a thin GLFW/ModernGL shell around the existing receiver CLI
parser and `run_receive` path. It does not open an input stream until Start is
pressed, and it reads decoded pictures from the receiver's latest-frame mailbox.
"""
import argparse
import ast
import contextlib
from dataclasses import dataclass
from functools import lru_cache
import io
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
import traceback
from collections import deque

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.v7_gl_viewer import (DISPLAY_LABELS, DISPLAY_MODES,
                                FILTER_PRECOMPUTE_MODES,
                                FLOAT_FRAGMENT_SHADER, FLOAT_MODE_IDS,
                                FRAGMENT_SHADER, VERTEX_SHADER,
                                _diagnostic_image, _float_texture_filter,
                                build_filter_lut, fit_viewport, float_planes,
                                resample_filter_planes)


ROW_HEIGHT = 36
TOOLBAR_HEIGHT = 54
INFO_REFRESH_SECONDS = 0.2
FOOTER_REFRESH_SECONDS = 0.5
QUIET_WAIT_SECONDS = 0.5
RESOURCE_REFRESH_SECONDS = 1.0
FULLSCREEN_TOOLBAR_HIDE_SECONDS = 2.0
FULLSCREEN_TOOLBAR_EDGE = 14
DISPLAY_MENU_ROW_HEIGHT = 29
DISPLAY_MENU_WIDTH = 250
RECEIVER_PREFERENCES_PATH = (
    Path.home()/'.config'/'modemTest'/'v7_receiver_gui.json')


def _shader_for_context(source, use_gles):
    """Adapt the viewer's GLSL 3.30 shaders to GLSL ES 3.00 when needed."""
    if not use_gles:
        return source
    version = '#version 300 es\nprecision highp float;\nprecision highp int;'
    return source.replace('#version 330', version, 1)


def _create_graphics_context(glfw, moderngl, wayland=None):
    """Try a Wayland-friendly GLES/EGL context, with desktop GL fallback."""
    if wayland is None:
        wayland = bool(os.environ.get('WAYLAND_DISPLAY') or
                       os.environ.get('XDG_SESSION_TYPE', '').lower() ==
                       'wayland')
    apis = (('gles', 'opengl') if wayland else ('opengl', 'gles'))
    failures = []
    for api in apis:
        gles = api == 'gles'
        glfw.default_window_hints()
        if gles:
            glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_ES_API)
            glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 0)
            require = 300
        else:
            glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_API)
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
            require = 330
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
        try:
            window = glfw.create_window(
                960, 720, 'V7 Receiver · Setup', None, None)
        except Exception as exc:
            failures.append(f'{api}: window creation raised {exc}')
            continue
        if not window:
            try:
                code, description = glfw.get_error()
            except Exception:
                code, description = None, None
            detail = (f'GLFW {code}: {description!r}' if code else
                      'GLFW returned no window')
            failures.append(f'{api}: {detail}')
            continue

        context = None
        try:
            glfw.make_context_current(window)
            # GLFW owns the window and its EGL/native context. ModernGL must
            # wrap that current context rather than selecting its standalone
            # EGL backend (which rejects the GLFW GLES context as unknown mode).
            context = moderngl.create_context(require=require)
            context_version = getattr(context, 'version_code', 0)
            if context_version < 300:
                raise RuntimeError(
                    f'ModernGL context version {context_version} is below 300')
            return window, context, gles
        except Exception as exc:
            failures.append(f'{api}: context setup failed: {exc}')
            if context is not None:
                try:
                    context.release()
                except Exception:
                    pass
            glfw.make_context_current(None)
            glfw.destroy_window(window)

    details = '; '.join(failures) or 'no context attempts were made'
    raise RuntimeError(
        f'Could not create a supported V7 receiver GL context: {details}')


BASIC_OPTION_DESTS = frozenset((
    'device', 'audio_output_device', 'audio_muted', 'audio_volume',
    'freewheel_seconds',
    'show_sync_warning', 'fullscreen', 'show_diagnostics', 'image_only',
    'save_dir'))
LIVE_RUNTIME_DESTS = frozenset((
    'audio_output_device', 'audio_muted', 'audio_volume', 'freewheel_seconds',
    'show_sync_warning'))
HIDDEN_DECODE_OPTIONS = frozenset((
    'direction', 'fixture', 'experimental_fold', 'baseline',
    'experimental_mono', 'experimental_mono_fold', 'mono_compatible',
    'mono_video_side', 'profile_ui', 'decode_batch', 'decode_history',
    'refine', 'no_tail_memory', 'force_float32', 'pilot_timing',
    'frame_boundary', 'pilot_speed_diagnostics', 'pulse_timing',
    'tone_equalization'))


@dataclass
class OptionField:
    action: object
    value: object
    label: str
    kind: str
    options: tuple = ()
    locked: bool = False

    @property
    def dest(self):
        return self.action.dest if self.action is not None else self.label


class QueueWriter:
    """Thread-safe text sink for the receiver's ordinary diagnostic output."""
    def __init__(self, output, notifier=None):
        self.output = output
        self.notifier = notifier
        self.pending = ''
        self.lock = threading.Lock()

    def notify(self):
        if self.notifier is not None:
            try:
                self.notifier()
            except Exception:
                pass

    def write(self, text):
        if not text:
            return 0
        wrote_lines = False
        with self.lock:
            self.pending += str(text)
            lines = self.pending.split('\n')
            self.pending = lines.pop()
            for line in lines:
                if line:
                    self.output.put(line)
                    wrote_lines = True
        if wrote_lines:
            self.notify()
        return len(text)

    def flush(self):
        wrote_line = False
        with self.lock:
            if self.pending:
                self.output.put(self.pending)
                self.pending = ''
                wrote_line = True
        if wrote_line:
            self.notify()


def _receive_parser(v7_live):
    root_parser = v7_live.parser()
    subparsers = next(action for action in root_parser._actions
                      if isinstance(action, argparse._SubParsersAction))
    return root_parser, subparsers.choices['receive']


def _input_devices():
    try:
        import sounddevice as sd
        devices = sd.query_devices()
    except Exception as exc:
        return (), f'Audio device enumeration failed: {exc}'
    choices = []
    for index, device in enumerate(devices):
        channels = int(device.get('max_input_channels') or 0)
        if channels:
            rate = float(device.get('default_samplerate') or 0)
            label = (f'{index}: {device["name"]} · {channels} in · '
                     f'{rate/1000:g} kHz')
            choices.append((label, index))
    if not choices:
        return (), 'No audio input devices are available.'
    return tuple(choices), ''


def _output_devices():
    try:
        import sounddevice as sd
        devices = sd.query_devices()
    except Exception as exc:
        return (), f'Audio output enumeration failed: {exc}'
    choices = []
    for index, device in enumerate(devices):
        channels = int(device.get('max_output_channels') or 0)
        if channels:
            rate = float(device.get('default_samplerate') or 0)
            label = (f'{index}: {device["name"]} · {channels} out · '
                     f'{rate/1000:g} kHz')
            choices.append((label, index))
    if not choices:
        return (), 'No audio output devices are available.'
    return tuple(choices), ''


def _device_identity(index, kind):
    if index is None:
        return None
    import sounddevice as sd
    device = sd.query_devices(index)
    hostapis = sd.query_hostapis()
    hostapi_index = device.get('hostapi')
    hostapi = (hostapis[int(hostapi_index)].get('name')
               if hostapi_index is not None and
               0 <= int(hostapi_index) < len(hostapis) else '')
    return {'name': str(device.get('name', '')),
            'hostapi': str(hostapi)}


def _find_device(identity, choices, kind):
    if identity is None:
        return None
    for _label, index in choices:
        try:
            current = _device_identity(index, kind)
        except Exception:
            continue
        if (current['name'] == identity.get('name') and
                current['hostapi'] == identity.get('hostapi')):
            return index
    return None


def _load_preferences(path=RECEIVER_PREFERENCES_PATH):
    try:
        values = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError):
        return {}
    return values if isinstance(values, dict) else {}


def _save_preferences(values, path=RECEIVER_PREFERENCES_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, sort_keys=True)+'\n',
                    encoding='utf-8')


def pick_save_directory(current=''):
    """Open the platform's native directory picker for decoded images."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError('A native folder picker is unavailable.') from exc
    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes('-topmost', True)
        except tk.TclError:
            pass
        options = {'parent': root, 'title': 'Save decoded images to'}
        if current and Path(current).is_dir():
            options['initialdir'] = str(current)
        selected = filedialog.askdirectory(**options)
        return str(selected) if selected else None
    except tk.TclError as exc:
        raise RuntimeError(f'Could not open the folder picker: {exc}') from exc
    finally:
        if root is not None:
            root.destroy()


def _process_memory_mib():
    """Current resident memory when available, otherwise the process peak."""
    try:
        import psutil
        return psutil.Process().memory_info().rss/(1024*1024), 'RSS'
    except Exception:
        try:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Darwin reports bytes; Linux and the BSDs report KiB.
            peak_mib = (peak/(1024*1024) if sys.platform == 'darwin'
                        else peak/1024)
            return peak_mib, 'peak RSS'
        except Exception:
            return None, 'RSS'


def _field_label(action):
    friendly = {
        'device': 'Input audio device',
        'audio_output_device': 'Passthrough output device',
        'audio_muted': 'Mute passthrough audio',
        'audio_volume': 'Passthrough volume (0–1)',
        'freewheel_seconds': 'Freewheel before sync warning (s)',
        'show_sync_warning': 'Show sync-loss warning',
        'direction': 'Playback direction',
        'fixture': 'Model fixture',
        'fullscreen': 'Start fullscreen',
        'show_diagnostics': 'Show diagnostics',
        'image_only': 'Start in image-only view',
        'profile_ui': 'Profile viewer CPU',
        'mono_compatible': 'Mono-compatible chroma',
        'no_log': 'Quiet receiver logs',
        'log': 'Log each decoded frame',
        'save_dir': 'Save decoded images to',
        'diagnostics': 'Detailed decoder diagnostics',
        'decode_batch': 'Decode batch',
        'decode_history': 'Decode history',
        'refine': 'Clock refinement',
        'no_tail_memory': 'Disable tail memory',
        'force_float32': 'Experimental float32 decoder',
        'pilot_timing': 'Pilot timing',
        'frame_boundary': 'Frame boundary',
        'pilot_speed_diagnostics': 'Pilot speed diagnostics',
        'pulse_timing': 'Pulse timing',
        'tone_equalization': 'Tone equalization',
        'baseline': 'Use baseline profile',
        'experimental_fold': 'Fold profile',
        'experimental_mono_fold': 'Experimental mono video fold',
        'mono_video_side': 'Mono video input side',
    }
    return friendly.get(action.dest,
                        action.dest.replace('_', ' ').capitalize())


def _make_fields(receive_parser, device_choices, audio_output_choices=()):
    fields = []
    for action in receive_parser._actions:
        if action.dest in ('help', 'mode') or action.dest in HIDDEN_DECODE_OPTIONS:
            continue
        if action.dest == 'headless':
            # The integrated GUI owns display; this internal CLI switch keeps
            # the receiver from opening a second, nested GL viewer.
            fields.append(OptionField(action, True, 'GUI receiver backend',
                                      'bool', locked=True))
            continue
        if action.dest == 'device':
            first_device = device_choices[0][1] if device_choices else None
            fields.append(OptionField(action, first_device, _field_label(action),
                                      'choice', device_choices))
            continue

        if action.dest == 'audio_output_device':
            fields.append(OptionField(action, None, _field_label(action),
                                      'choice',
                                      (('Off · passthrough disabled', None),) +
                                      tuple(audio_output_choices)))
            continue

        if action.dest == 'save_dir':
            fields.append(OptionField(action, None, _field_label(action),
                                      'folder'))
            continue

        value = action.default
        if value is argparse.SUPPRESS:
            value = None
        if action.dest == 'diagnostics':
            # Keep decoder timing for the optional live diagnostic panel.
            value = True
        if action.dest == 'log':
            value = False
        if action.dest == 'no_log':
            value = True
        if action.dest == 'show_diagnostics':
            # Keep the live picture large; details are a one-click drawer.
            value = False
        if isinstance(action, (argparse._StoreTrueAction,
                                argparse._StoreFalseAction)):
            fields.append(OptionField(action, bool(value),
                                      _field_label(action), 'bool'))
        elif action.choices is not None:
            options = tuple((str(option), option) for option in action.choices)
            if action.dest == 'experimental_fold':
                options = (('Default (500)', None),) + options
            fields.append(OptionField(action, value, _field_label(action),
                                      'choice', options))
        else:
            if isinstance(value, Path):
                value = str(value)
            fields.append(OptionField(action, value, _field_label(action),
                                      'text'))
    fields.append(OptionField(None, 'nearest', 'Display upscaler', 'choice',
                              tuple((DISPLAY_LABELS[name], name)
                                    for name in DISPLAY_MODES)))
    return fields


@lru_cache(maxsize=32)
def _font(size, mono=False):
    names = ('DejaVuSansMono.ttf', 'DejaVuSans.ttf') if mono else (
        'DejaVuSans.ttf', 'DejaVuSansMono.ttf')
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _text_lines(text, width, font):
    words = str(text).split()
    lines, current = [], ''
    for word in words:
        trial = f'{current} {word}'.strip()
        if current and font.getlength(trial) > width:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def _fit_text(text, font, width):
    text = str(text)
    while text and font.getlength(text) > width:
        text = text[:-2]+'…'
    return text


def _logical_rect_to_framebuffer(rect, window_size, framebuffer_size):
    """Scale a top-left-origin logical rectangle to GL scissor coordinates."""
    window_width, window_height = window_size
    framebuffer_width, framebuffer_height = framebuffer_size
    if window_width <= 0 or window_height <= 0:
        return 0, 0, 0, 0
    scale_x = framebuffer_width/window_width
    scale_y = framebuffer_height/window_height
    left, top, right, bottom = rect
    x0 = round(left*scale_x)
    x1 = round(right*scale_x)
    y0 = framebuffer_height-round(bottom*scale_y)
    y1 = framebuffer_height-round(top*scale_y)
    return x0, y0, max(0, x1-x0), max(0, y1-y0)


def _draw_scissored_ui(context, framebuffer_size, scissor, texture,
                       vertex_array, triangle_mode):
    """Draw the UI texture over the current picture inside one rectangle."""
    _draw_scissored_ui_regions(context, framebuffer_size, (scissor,), texture,
                               vertex_array, triangle_mode)


def _draw_scissored_ui_regions(context, framebuffer_size, scissors, texture,
                               vertex_array, triangle_mode):
    """Blit the full UI texture through several framebuffer-space clips."""
    context.viewport = (0, 0, *framebuffer_size)
    try:
        texture.use(location=0)
        for scissor in scissors:
            if scissor[2] <= 0 or scissor[3] <= 0:
                continue
            context.scissor = scissor
            vertex_array.render(mode=triangle_mode, vertices=3)
    finally:
        context.scissor = None


def _scissors_outside_viewport(framebuffer_size, viewport):
    """Return nonempty GL scissor rectangles covering everything but a view."""
    framebuffer_width, framebuffer_height = framebuffer_size
    x, y, width, height = viewport
    left = max(0, min(framebuffer_width, x))
    bottom = max(0, min(framebuffer_height, y))
    right = max(left, min(framebuffer_width, x+width))
    top = max(bottom, min(framebuffer_height, y+height))
    regions = (
        (0, 0, framebuffer_width, bottom),
        (0, top, framebuffer_width, framebuffer_height-top),
        (0, bottom, left, top-bottom),
        (right, bottom, framebuffer_width-right, top-bottom),
    )
    return tuple(region for region in regions
                 if region[2] > 0 and region[3] > 0)


class ReceiverGui:
    def __init__(self, v7_live, root_parser, receive_parser, device_choices,
                 device_error='', audio_output_choices=(),
                 audio_output_error='',
                 preference_path=RECEIVER_PREFERENCES_PATH):
        self.v7_live = v7_live
        self.root_parser = root_parser
        self.receive_parser = receive_parser
        self.fields = _make_fields(receive_parser, device_choices,
                                   audio_output_choices)
        self.device_error = device_error
        self.audio_output_error = audio_output_error
        self.runtime_options = None
        self.preference_path = Path(preference_path)
        self.preferences = _load_preferences(self.preference_path)
        self.input_device_identity = self.preferences.get('input_device')
        self.audio_output_identity = self.preferences.get('output_device')
        self.unavailable_input_identity = None
        self.unavailable_output_identity = None
        self.page = 'config'
        self.fullscreen = False
        self.toolbar_visible = True
        self.last_ui_activity = time.monotonic()
        self.windowed_bounds = {'position': (80, 80), 'size': (960, 720)}
        self.selected = 0
        self.scroll = 0
        self.dropdown = None
        self.dropdown_scroll = 0
        self.display_menu_open = False
        self.display_menu_index = 0
        self.editing = False
        self.edit_buffer = ''
        self.advanced_options = False
        self.notice = 'Review the selected input and settings, then start receiving.'
        self.info_scroll = 0
        self.info_follow = True
        self.dirty = True
        self.picture_dirty = False
        self.info_refresh_pending = False
        self.last_info_refresh = 0.0
        self.last_diagnostics_poll = 0.0
        self.hits = {}
        self.width, self.height = 960, 720
        self.output = queue.Queue()
        self.lines = deque(maxlen=500)
        self.latest_report = None
        self.live_diagnostics = None
        self.live_packet = None
        self.live_decode_info = None
        self.live_meter = None
        self.live_meter_json = None
        self.started = False
        self.ever_started = False
        self.receiver_thread = None
        self.receiver_stop = None
        self.last_generation = None
        self.current_frame = None
        self.latest_values_image = None
        self.display_latency_ms = None
        self.last_display_latency_label = None
        self.display_mode = 'nearest'
        self.image_only = False
        self.image_only_previous_page = 'info'
        self.image_only_previous_fullscreen = False
        self._glfw = None
        self._window = None
        self.last_title = None
        self.last_ui_size = None
        self.profile_wall = None
        self.profile_process = None
        self.profile_thread = None
        self.gui_resource_wall = None
        self.gui_resource_process = None
        self.gui_resource_thread = None
        self.gui_resource_next_sample = 0.0
        self.gui_resource_lines = ('thread -- · proc --', 'RSS --')
        self._restore_preferences(device_choices, audio_output_choices)

    def _field_value_label(self, field):
        if field.dest == 'device':
            if self.unavailable_input_identity is not None:
                name = self.unavailable_input_identity.get('name', 'saved input')
                return f'Unavailable · {name} — reselect'
            return next((label for label, value in field.options
                         if value == field.value), 'Select an input device…')
        if field.dest == 'audio_output_device':
            if self.unavailable_output_identity is not None:
                name = self.unavailable_output_identity.get(
                    'name', 'saved output')
                return f'Unavailable · {name} — reselect or choose Off'
            if field.value is None:
                return 'Select output · passthrough off'
            return next((label for label, value in field.options
                         if value == field.value), 'Reselect output device…')
        if field.kind == 'folder':
            return (str(field.value) if field.value else
                    'Choose a folder…')
        if field.kind == 'bool':
            if field.locked:
                return 'On · required for this integrated viewer'
            return 'On' if field.value else 'Off'
        if field.kind == 'choice':
            return next((label for label, value in field.options
                         if value == field.value),
                        'Default (500)' if field.dest == 'experimental_fold'
                        and field.value is None else 'Select…')
        return str(field.value) if field.value not in (None, '') else '(not set)'

    def _restore_preferences(self, input_choices, output_choices):
        by_dest = {field.dest: field for field in self.fields}
        input_field = by_dest.get('device')
        output_field = by_dest.get('audio_output_device')
        if (input_field is not None and self.input_device_identity is None and
                input_field.value is not None):
            try:
                self.input_device_identity = _device_identity(
                    input_field.value, 'input')
            except Exception:
                pass
        if input_field is not None and self.input_device_identity is not None:
            selected = _find_device(self.input_device_identity,
                                    input_choices, 'input')
            if selected is None:
                input_field.value = None
                self.unavailable_input_identity = self.input_device_identity
            else:
                input_field.value = selected
        if output_field is not None and self.audio_output_identity is not None:
            selected = _find_device(self.audio_output_identity,
                                    output_choices, 'output')
            if selected is None:
                output_field.value = None
                self.unavailable_output_identity = self.audio_output_identity
            else:
                output_field.value = selected
        for dest in ('audio_muted', 'show_sync_warning'):
            field = by_dest.get(dest)
            if field is not None and dest in self.preferences:
                field.value = bool(self.preferences[dest])
        freewheel = by_dest.get('freewheel_seconds')
        if freewheel is not None and 'freewheel_seconds' in self.preferences:
            try:
                value = float(self.preferences['freewheel_seconds'])
                if np.isfinite(value) and value >= 0:
                    freewheel.value = value
            except (TypeError, ValueError):
                pass
        volume = by_dest.get('audio_volume')
        if volume is not None and 'audio_volume' in self.preferences:
            try:
                value = float(self.preferences['audio_volume'])
                if np.isfinite(value) and 0.0 <= value <= 1.0:
                    volume.value = value
            except (TypeError, ValueError):
                pass
        save_dir = by_dest.get('save_dir')
        if save_dir is not None:
            save_dir.value = self.preferences.get('save_dir')
        if self.unavailable_input_identity is not None:
            name = self.unavailable_input_identity.get('name', 'Saved input')
            self.notice = f'{name} is unavailable; reselect the input device.'
        elif self.unavailable_output_identity is not None:
            name = self.unavailable_output_identity.get('name', 'Saved output')
            self.notice = f'{name} is unavailable; reselect it or choose Off.'

    def _persist_preferences(self):
        by_dest = {field.dest: field for field in self.fields}
        input_field = by_dest.get('device')
        output_field = by_dest.get('audio_output_device')
        if (input_field is not None and input_field.value is not None and
                self.unavailable_input_identity is None):
            try:
                self.input_device_identity = _device_identity(
                    input_field.value, 'input')
            except Exception:
                pass
        if (output_field is not None and output_field.value is not None and
                self.unavailable_output_identity is None):
            try:
                self.audio_output_identity = _device_identity(
                    output_field.value, 'output')
            except Exception:
                pass
        elif (output_field is not None and output_field.value is None and
              self.unavailable_output_identity is None):
            self.audio_output_identity = None
        values = {
            'input_device': self.input_device_identity,
            'output_device': self.audio_output_identity,
        }
        for dest in ('audio_muted', 'audio_volume', 'freewheel_seconds',
                     'show_sync_warning', 'save_dir'):
            if dest in by_dest:
                values[dest] = by_dest[dest].value
        self.preferences = values
        try:
            _save_preferences(values, self.preference_path)
        except OSError as exc:
            self.notice = f'Could not save receiver preferences: {exc}'

    def _prepare_input_device(self):
        field = next((field for field in self.fields
                      if field.dest == 'device'), None)
        if field is None:
            return False
        if self.input_device_identity is None and field.value is not None:
            try:
                self.input_device_identity = _device_identity(
                    field.value, 'input')
            except Exception:
                self.notice = 'Could not verify the selected input device.'
                return False
        choices, error = _input_devices()
        if not choices:
            self.device_error = error
            self.notice = error or 'No input audio devices are available.'
            field.options = ()
            field.value = None
            return False
        field.options = choices
        selected = _find_device(self.input_device_identity, choices, 'input')
        if selected is None:
            field.value = None
            self.unavailable_input_identity = self.input_device_identity
            name = (self.input_device_identity or {}).get('name',
                                                           'selected input')
            self.notice = f'{name} is unavailable; reselect the input device.'
            return False
        field.value = selected
        self.unavailable_input_identity = None
        return True

    def _toggle_fullscreen(self, glfw, window):
        primary = glfw.get_primary_monitor()
        if primary is None:
            return
        self.display_menu_open = False
        if self.fullscreen:
            x, y = self.windowed_bounds['position']
            width, height = self.windowed_bounds['size']
            glfw.set_window_monitor(window, None, x, y, width, height,
                                    glfw.DONT_CARE)
            self.fullscreen = False
            self.toolbar_visible = True
        else:
            self.windowed_bounds['position'] = glfw.get_window_pos(window)
            self.windowed_bounds['size'] = glfw.get_window_size(window)
            mode = glfw.get_video_mode(primary)
            glfw.set_window_monitor(window, primary, 0, 0, mode.size.width,
                                    mode.size.height, mode.refresh_rate)
            self.fullscreen = True
            self.toolbar_visible = False
        focus_window = getattr(glfw, 'focus_window', None)
        if focus_window is not None:
            focus_window(window)
        self.last_ui_activity = time.monotonic()
        self.dirty = True

    def _reveal_toolbar(self):
        if self.image_only:
            return
        was_hidden = not self.toolbar_visible
        self.toolbar_visible = True
        self.last_ui_activity = time.monotonic()
        if was_hidden:
            self.dirty = True

    def _toggle_details(self):
        self.display_menu_open = False
        self.page = 'info'
        field = next(field for field in self.fields
                     if field.dest == 'show_diagnostics')
        field.value = not field.value
        self.dirty = True

    def _on_cursor_position(self, _window, _x, y):
        if not self.fullscreen or self.image_only:
            return
        now = time.monotonic()
        if self.toolbar_visible:
            self.last_ui_activity = now
        elif y <= FULLSCREEN_TOOLBAR_EDGE:
            self.toolbar_visible = True
            self.last_ui_activity = now
            self.dirty = True

    def _update_toolbar_visibility(self, now, cursor_y):
        was_visible = self.toolbar_visible
        if self.image_only:
            return
        if not self.fullscreen:
            self.toolbar_visible = True
        elif (self.dropdown is not None or self.display_menu_open or
              self.editing or cursor_y <= FULLSCREEN_TOOLBAR_EDGE):
            self.toolbar_visible = True
            if not was_visible:
                self.last_ui_activity = now
        elif (self.toolbar_visible and
              now-self.last_ui_activity >= FULLSCREEN_TOOLBAR_HIDE_SECONDS):
            self.toolbar_visible = False
        if self.toolbar_visible != was_visible:
            self.dirty = True

    def _set_image_only(self, enabled):
        enabled = bool(enabled)
        if enabled == self.image_only:
            return
        self.display_menu_open = False
        if enabled:
            self.image_only_previous_page = self.page
            self.image_only_previous_fullscreen = self.fullscreen
            self.image_only = True
            if self._glfw is not None and self._window is not None:
                self._glfw.set_input_mode(
                    self._window, self._glfw.CURSOR,
                    self._glfw.CURSOR_HIDDEN)
            if not self.fullscreen and self._glfw is not None:
                self._toggle_fullscreen(self._glfw, self._window)
        else:
            self.image_only = False
            self.page = self.image_only_previous_page
            if self._glfw is not None and self._window is not None:
                self._glfw.set_input_mode(
                    self._window, self._glfw.CURSOR,
                    self._glfw.CURSOR_NORMAL)
            if (self.fullscreen and not self.image_only_previous_fullscreen and
                    self._glfw is not None):
                self._toggle_fullscreen(self._glfw, self._window)
        for field in self.fields:
            if field.dest == 'image_only':
                field.value = enabled
                break
        self.picture_dirty = True
        self.dirty = True

    def _build_arguments(self):
        words = ['receive', '--headless']
        for field in self.fields:
            action = field.action
            if action is None:
                if field.label == 'Display upscaler':
                    self.display_mode = field.value
                continue
            dest = action.dest
            if dest in ('help', 'mode', 'headless', 'fullscreen',
                        'image_only'):
                continue
            if dest == 'device' and field.value is None:
                raise ValueError('Select an input audio device before starting.')
            option = action.option_strings[0] if action.option_strings else None
            if not option:
                continue
            if isinstance(action, argparse._StoreTrueAction):
                if field.value:
                    words.append(option)
            elif isinstance(action, argparse._StoreFalseAction):
                if not field.value:
                    words.append(option)
            elif field.value is not None:
                words.extend((option, str(field.value)))

        parsed = self.root_parser.parse_args(words)
        parsed.headless = True
        parsed.fullscreen = False
        return parsed

    def _notify_ui(self):
        glfw = self._glfw
        if glfw is not None:
            glfw.post_empty_event()

    def _start_receiver(self):
        if self.editing:
            self._finish_edit(self.fields[self.selected])
        if self.started:
            self._stop_receiver()
            return
        if self.receiver_thread is not None and self.receiver_thread.is_alive():
            self.notice = 'Waiting for the previous receiver to stop…'
            self.dirty = True
            return
        if not self._prepare_input_device():
            self.dirty = True
            return
        try:
            output = io.StringIO()
            with contextlib.redirect_stdout(output), \
                    contextlib.redirect_stderr(output):
                args = self._build_arguments()
        except (ValueError, SystemExit) as exc:
            details = output.getvalue().strip().splitlines()
            self.notice = (details[-1] if details else
                           str(exc) or 'Receiver options are invalid.')
            self.dirty = True
            return
        from tools.v7_receiver_audio import ReceiverRuntimeOptions
        self.runtime_options = ReceiverRuntimeOptions(
            audio_output_device=args.audio_output_device,
            audio_output_identity=self.audio_output_identity,
            audio_input_identity=self.input_device_identity,
            audio_muted=args.audio_muted,
            freewheel_seconds=args.freewheel_seconds,
            show_sync_warning=args.show_sync_warning,
            audio_volume=args.audio_volume)
        args.runtime_options = self.runtime_options
        start_image_only = any(
            field.value for field in self.fields
            if field.dest == 'image_only')
        self.started = True
        self.ever_started = True
        self.receiver_stop = threading.Event()
        self.latest_report = None
        self.live_diagnostics = None
        self.live_packet = None
        self.live_decode_info = None
        self.live_meter = None
        self.live_meter_json = None
        args.stop_event = self.receiver_stop
        self.notice = f'Starting receiver on {args.device!r}…'
        fullscreen_field = next(
            (field for field in self.fields
             if field.action is not None and field.action.dest == 'fullscreen'),
            None)
        if (fullscreen_field is not None and fullscreen_field.value and
                not self.fullscreen):
            self._toggle_fullscreen(self._glfw, self._window)

        def receive():
            writer = QueueWriter(self.output, self._notify_ui)
            try:
                with contextlib.redirect_stdout(writer), \
                        contextlib.redirect_stderr(writer):
                    self.v7_live.run_receive(args)
            except BaseException:
                writer.write(traceback.format_exc())
            finally:
                writer.flush()
                self.output.put('[receiver thread returned]')
                writer.notify()

        self.receiver_thread = threading.Thread(
            target=receive, name='v7-receiver', daemon=True)
        self.receiver_thread.start()
        self.page = 'info'
        if start_image_only:
            self._set_image_only(True)
        self.dirty = True

    def _stop_receiver(self):
        if self.receiver_stop is not None and self.started:
            self.receiver_stop.set()
            self.started = False
            self.notice = 'Stopping receiver and closing its audio stream…'
            self.dirty = True

    def _poll_receiver_lifecycle(self):
        if (self.receiver_thread is None or
                self.receiver_thread.is_alive()):
            return
        if self.started:
            if not self.lines or not self.lines[-1].endswith(
                    '[receiver thread returned]'):
                self.lines.append('[receiver thread returned]')
            self.notice = 'Receiver stopped; inspect the information panel.'
            self.started = False
            self.dirty = True
        elif (self.receiver_stop is not None and
              self.receiver_stop.is_set() and
              self.notice.startswith('Stopping receiver')):
            self.notice = 'Receiver stopped.'
            self.dirty = True

    def _select_choice(self, field, value):
        if (self.started and field.action is not None and
                field.dest not in LIVE_RUNTIME_DESTS):
            self.dropdown = None
            self.notice = ('Receiver options are locked while receiving; '
                           'close and relaunch to change them.')
            self.dirty = True
            return
        field.value = value
        if field.dest == 'device':
            self.unavailable_input_identity = None
            try:
                self.input_device_identity = _device_identity(value, 'input')
            except Exception:
                self.input_device_identity = None
        elif field.dest == 'audio_output_device':
            self.unavailable_output_identity = None
            try:
                self.audio_output_identity = _device_identity(value, 'output')
            except Exception:
                self.audio_output_identity = None
        if field.dest == 'experimental_fold':
            self._clear_other_profiles(field.dest)
        self.dropdown = None
        self.dropdown_scroll = 0
        if field.label == 'Display upscaler':
            self.display_mode = value
            self.picture_dirty = True
            self.display_menu_open = False
        self.notice = f'{field.label}: {self._field_value_label(field)}'
        self._update_runtime_option(field)
        if field.dest in ('device', 'audio_output_device', 'audio_muted',
                          'freewheel_seconds', 'show_sync_warning',
                          'save_dir'):
            self._persist_preferences()
        self.dirty = True

    def _clear_other_profiles(self, active):
        for item in self.fields:
            if item.dest == active:
                continue
            if item.dest == 'experimental_fold':
                item.value = None
            elif item.dest in ('baseline', 'experimental_mono',
                               'experimental_mono_fold'):
                item.value = False

    def _adjust_field(self, field, direction):
        if (self.started and field.action is not None and
                field.dest not in LIVE_RUNTIME_DESTS):
            self.notice = ('Receiver options are locked while receiving; '
                           'close and relaunch to change them.')
            self.dirty = True
            return
        if field.locked:
            return
        if field.kind == 'bool':
            field.value = not field.value
            if field.dest in ('baseline', 'experimental_mono',
                              'experimental_mono_fold') and field.value:
                self._clear_other_profiles(field.dest)
        elif field.kind == 'choice' and field.options:
            values = [value for _label, value in field.options]
            try:
                index = values.index(field.value)
            except ValueError:
                index = -1
            field.value = values[(index+direction) % len(values)]
            if field.dest == 'experimental_fold':
                self._clear_other_profiles(field.dest)
            if field.label == 'Display upscaler':
                self.display_mode = field.value
                self.picture_dirty = True
                self.display_menu_open = False
        self.notice = f'{field.label}: {self._field_value_label(field)}'
        self._update_runtime_option(field)
        if field.dest in ('audio_muted', 'audio_volume', 'freewheel_seconds',
                          'show_sync_warning'):
            self._persist_preferences()
        self.dirty = True

    def _refresh_audio_output_choices(self):
        field = next((field for field in self.fields
                      if field.dest == 'audio_output_device'), None)
        if field is None:
            return
        choices, error = _output_devices()
        field.options = (('Off · passthrough disabled', None),) + choices
        self.audio_output_error = error
        if self.audio_output_identity is not None:
            selected = _find_device(self.audio_output_identity,
                                    choices, 'output')
            if selected is None:
                field.value = None
                self.unavailable_output_identity = self.audio_output_identity
            else:
                field.value = selected
                self.unavailable_output_identity = None

    def _update_runtime_option(self, field):
        if (self.runtime_options is None or
                field.dest not in LIVE_RUNTIME_DESTS):
            return
        try:
            if field.dest == 'audio_output_device':
                if field.value is not None:
                    try:
                        self.audio_output_identity = _device_identity(
                            field.value, 'output')
                    except Exception:
                        self.audio_output_identity = None
                elif self.unavailable_output_identity is None:
                    self.audio_output_identity = None
                self.runtime_options.update(
                    audio_output_device=field.value,
                    audio_output_identity=self.audio_output_identity)
            elif field.dest == 'audio_muted':
                self.runtime_options.update(audio_muted=field.value)
            elif field.dest == 'audio_volume':
                value = float(field.value)
                if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError('Passthrough volume must be between 0 and 1.')
                field.value = value
                self.runtime_options.update(audio_volume=value)
            elif field.dest == 'freewheel_seconds':
                value = float(field.value)
                if not np.isfinite(value) or value < 0:
                    raise ValueError('Freewheel duration must be non-negative.')
                field.value = value
                self.runtime_options.update(freewheel_seconds=value)
            elif field.dest == 'show_sync_warning':
                self.runtime_options.update(show_sync_warning=field.value)
        except (TypeError, ValueError) as exc:
            self.notice = str(exc)

    def _process_output(self):
        other_output = False
        report_changed = False
        while True:
            try:
                line = self.output.get_nowait()
            except queue.Empty:
                break
            self.lines.append(str(line))
            try:
                record = ast.literal_eval(str(line))
            except (ValueError, SyntaxError):
                record = None
            if isinstance(record, dict):
                self.latest_report = record
                report_changed = True
            else:
                other_output = True
        if other_output:
            self.dirty = True
            self.last_info_refresh = time.monotonic()
            self.info_refresh_pending = False
        elif report_changed:
            now = time.monotonic()
            if now-self.last_info_refresh >= INFO_REFRESH_SECONDS:
                self.dirty = True
                self.last_info_refresh = now
                self.info_refresh_pending = False
            else:
                self.info_refresh_pending = True

    def _profile_ui_enabled(self):
        return any(field.action is not None and
                   field.action.dest == 'profile_ui' and field.value
                   for field in self.fields)

    def _profile_ui_if_enabled(self):
        enabled = self._profile_ui_enabled()
        now = time.monotonic()
        if not enabled:
            self.profile_wall = now
            self.profile_process = time.process_time()
            self.profile_thread = time.thread_time()
            return
        if self.profile_wall is None:
            self.profile_wall = now
            self.profile_process = time.process_time()
            self.profile_thread = time.thread_time()
            return
        elapsed = now-self.profile_wall
        if elapsed < 5:
            return
        process = time.process_time()
        thread = time.thread_time()
        self.lines.append(
            f'UI CPU over {elapsed:.1f}s: process '
            f'{100*(process-self.profile_process)/elapsed:.1f}% · viewer '
            f'{100*(thread-self.profile_thread)/elapsed:.1f}%')
        self.profile_wall, self.profile_process, self.profile_thread = (
            now, process, thread)
        self.dirty = True

    def _sample_gui_resources(self, now):
        if now < self.gui_resource_next_sample:
            return self.gui_resource_lines
        process = time.process_time()
        thread = time.thread_time()
        if self.gui_resource_wall is not None:
            elapsed = now-self.gui_resource_wall
            if elapsed > 0:
                process_percent = 100*(process-self.gui_resource_process)/elapsed
                thread_percent = 100*(thread-self.gui_resource_thread)/elapsed
                self.gui_resource_lines = (
                    f'thread {thread_percent:3.0f}% · proc {process_percent:3.0f}%',
                    self._memory_line())
        else:
            self.gui_resource_lines = ('thread -- · proc --',
                                       self._memory_line())
        self.gui_resource_wall = now
        self.gui_resource_process = process
        self.gui_resource_thread = thread
        self.gui_resource_next_sample = now+RESOURCE_REFRESH_SECONDS
        return self.gui_resource_lines

    @staticmethod
    def _memory_line():
        memory_mib, label = _process_memory_mib()
        return (f'{label} {memory_mib:.0f} MiB' if memory_mib is not None
                else f'{label} unavailable')

    def _poll_frame(self):
        frame = self.v7_live.FRAME_BUFFER.snapshot()
        if frame is None or frame.generation == self.last_generation:
            return
        self.last_generation = frame.generation
        self.current_frame = frame
        # RGB conversion is only used by nearest-neighbour display. All other
        # upscalers consume the decoded planes directly in their GL shader.
        self.latest_values_image = None
        self.picture_dirty = True

    def _poll_diagnostics(self):
        now = time.monotonic()
        if self.page != 'info' or self.image_only:
            return
        diagnostics_visible = self._diagnostics_visible()
        interval = (INFO_REFRESH_SECONDS if diagnostics_visible else
                    FOOTER_REFRESH_SECONDS)
        if now-self.last_diagnostics_poll < interval:
            return
        self.last_diagnostics_poll = now
        snapshot = dict(self.v7_live.RECEIVER_GUI_STATUS)
        meter = snapshot.get('meter')
        if not diagnostics_visible:
            # The compact footer needs only two scalar fields. Avoid building
            # all diagnostic strings and JSON-copying the full meter while its
            # panel is hidden.
            if meter is None:
                return
            old_meter = self.live_meter or {}
            old_footer = (old_meter.get('decoded'),
                          round(float(old_meter.get('input_fps', 0.0)), 1))
            new_footer = (meter.get('decoded'),
                          round(float(meter.get('input_fps', 0.0)), 1))
            self.live_meter = meter
            if new_footer != old_footer:
                self.dirty = True
            return
        provider = snapshot.get('diagnostics')
        if provider is None:
            return
        try:
            diagnostics = dict(provider())
        except Exception:
            return
        diagnostics['resources'] = self._sample_gui_resources(now)
        display_latency_label = (
            None if self.display_latency_ms is None else
            f'{self.display_latency_ms:.1f}')
        packet = snapshot.get('latest_packet')
        decode_info = snapshot.get('decode_info')
        try:
            meter_json = (json.dumps(
                meter, sort_keys=True,
                default=lambda value: value.tolist()
                if hasattr(value, 'tolist') else str(value))
                if meter is not None else None)
        except (TypeError, ValueError):
            meter_json = repr(meter)
        changed = diagnostics != self.live_diagnostics
        changed |= packet is not self.live_packet
        changed |= decode_info is not self.live_decode_info
        changed |= meter_json != self.live_meter_json
        changed |= display_latency_label != self.last_display_latency_label
        if changed:
            self.last_display_latency_label = display_latency_label
            self.live_diagnostics = diagnostics
            self.live_packet = packet
            self.live_decode_info = decode_info
            self.live_meter_json = meter_json
            try:
                self.live_meter = (json.loads(meter_json)
                                   if meter_json is not None else None)
            except json.JSONDecodeError:
                self.live_meter = {'raw': meter_json}
            self.dirty = True

    def _render_config(self, image, draw, font, small, mono):
        width, height = image.size
        draw.text((24, 70), 'Receiver setup',
                  fill=(240, 245, 249), font=font)
        draw.text((24, 98),
                  'First available input is selected; change it or review advanced controls.',
                  fill=(151, 174, 192), font=small)
        order = self._config_field_indexes()
        top = 140
        bottom = height-126
        visible = max(1, (bottom-top)//ROW_HEIGHT)
        self.scroll = max(0, min(self.scroll, max(0, len(order)-visible)))
        if order and self.selected not in order:
            self.selected = order[0]
        self.hits['advanced_toggle'] = (width-256, 94, width-14, 124)
        advanced_label = ('Hide advanced settings' if self.advanced_options else
                          f'Show advanced settings · {len(self.fields)-len(order)}')
        draw.rounded_rectangle(self.hits['advanced_toggle'], radius=5,
                               fill=(22, 35, 46), outline=(67, 100, 122),
                               width=1)
        draw.text((width-244, 101), advanced_label,
                  fill=(196, 216, 229), font=small)
        for row in range(visible):
            order_index = self.scroll+row
            if order_index >= len(order):
                break
            index = order[order_index]
            field = self.fields[index]
            y = top+row*ROW_HEIGHT
            selected = index == self.selected
            if selected:
                draw.rounded_rectangle((14, y, width-14, y+32), radius=4,
                                       fill=(31, 53, 69),
                                       outline=(85, 131, 159), width=1)
            else:
                draw.rounded_rectangle((14, y, width-14, y+32), radius=4,
                                       fill=(17, 28, 38),
                                       outline=(38, 55, 69), width=1)
            label_color = (220, 231, 239) if not field.locked else (135, 153, 166)
            draw.text((26, y+8), field.label, fill=label_color, font=small)
            value_box = (330, y+4, width-28, y+29)
            if field.kind == 'bool':
                draw.rounded_rectangle(value_box, radius=4,
                                       fill=(38, 63, 78) if field.value else (28, 38, 47),
                                       outline=(65, 91, 108), width=1)
                value = self._field_value_label(field)
            elif field.kind == 'choice':
                draw.rounded_rectangle(value_box, radius=4,
                                       fill=(21, 35, 47),
                                       outline=(65, 91, 108), width=1)
                value = self._field_value_label(field)+'  ▾'
            elif field.kind == 'folder':
                draw.rounded_rectangle(value_box, radius=4,
                                       fill=(21, 35, 47),
                                       outline=(65, 91, 108), width=1)
                value = self._field_value_label(field)+'   Browse…'
            else:
                draw.rounded_rectangle(value_box, radius=4,
                                       fill=(21, 35, 47),
                                       outline=(65, 91, 108), width=1)
                value = self.edit_buffer if self.editing and index == self.selected else self._field_value_label(field)
            available = max(8, value_box[2]-value_box[0]-14)
            while value and small.getlength(value) > available:
                value = value[:-2]+'…'
            draw.text((value_box[0]+8, y+8), value,
                      fill=(236, 242, 247) if not field.locked else (135, 153, 166),
                      font=small)
            self.hits[f'row:{index}'] = (14, y, width-14, y+32)
            self.hits[f'value:{index}'] = value_box

        if self.dropdown is not None:
            field = self.fields[self.dropdown]
            menu_items = list(field.options)
            max_items = min(7, len(menu_items))
            self.dropdown_scroll = max(
                0, min(self.dropdown_scroll, len(menu_items)-max_items))
            try:
                field_position = order.index(self.dropdown)
            except ValueError:
                field_position = self.scroll
            yrow = top+(field_position-self.scroll)*ROW_HEIGHT
            menu_top = yrow+ROW_HEIGHT
            if menu_top+max_items*29 > bottom:
                menu_top = max(top, yrow-max_items*29)
            left, right = 330, width-28
            draw.rounded_rectangle((left, menu_top, right,
                                    menu_top+max_items*29+4), radius=4,
                                   fill=(12, 22, 31),
                                   outline=(93, 132, 155), width=1)
            for menu_index in range(max_items):
                option_index = self.dropdown_scroll+menu_index
                label, value = menu_items[option_index]
                option_y = menu_top+2+menu_index*29
                if value == field.value:
                    draw.rectangle((left+1, option_y, right-1, option_y+28),
                                   fill=(42, 75, 96))
                shown = label
                while shown and small.getlength(shown) > right-left-20:
                    shown = shown[:-2]+'…'
                draw.text((left+9, option_y+6), shown,
                          fill=(235, 241, 246), font=small)
                self.hits[f'option:{option_index}'] = (
                    left, option_y, right, option_y+28)

        if self.device_error:
            draw.text((24, height-88), self.device_error[:120],
                      fill=(255, 182, 132), font=small)
        if self.notice:
            draw.text((24, height-60), self.notice[:150],
                      fill=(165, 190, 207), font=small)
        if 0 <= self.selected < len(self.fields):
            field = self.fields[self.selected]
            help_text = getattr(field.action, 'help', '') if field.action else (
                'Display-only option; does not affect the decoded values.')
            if help_text and help_text != argparse.SUPPRESS:
                lines = _text_lines(help_text, width-48, small)
                if lines:
                    draw.text((24, height-118), lines[0][:150],
                              fill=(123, 148, 168), font=small)

    def _diagnostics_visible(self):
        enabled = any(field.value for field in self.fields
                      if field.dest == 'show_diagnostics')
        return enabled and (not self.fullscreen or self.toolbar_visible)

    def _picture_box(self, size):
        width, height = size
        chrome_visible = not self.fullscreen or self.toolbar_visible
        top = (TOOLBAR_HEIGHT if chrome_visible else 0)+8
        details_height = round(height*.27) if self._diagnostics_visible() else 0
        footer_height = 38 if chrome_visible else 0
        picture_height = max(1, height-top-details_height-footer_height-8)
        return (10, top, max(1, width-20), picture_height)

    def _picture_viewport(self, window_size, framebuffer_size, aspect_ratio):
        left, top, picture_w, picture_h = self._picture_box(window_size)
        x, y, width, height = fit_viewport(
            (picture_w, picture_h), aspect_ratio)
        fb_width, fb_height = framebuffer_size
        scale_x = fb_width/window_size[0] if window_size[0] else 0
        scale_y = fb_height/window_size[1] if window_size[1] else 0
        viewport_width = max(0, round(width*scale_x))
        viewport_height = max(0, round(height*scale_y))
        viewport_x = round((left+x)*scale_x)
        viewport_top = round((top+y)*scale_y)
        viewport_y = fb_height-viewport_top-viewport_height
        return viewport_x, viewport_y, viewport_width, viewport_height

    def _render_info(self, image, draw, font, small, mono):
        width, height = image.size
        left, top, picture_w, picture_h = self._picture_box(image.size)
        draw.rectangle((left, top, left+picture_w, top+picture_h),
                       fill=(14, 19, 24), outline=(49, 69, 83), width=1)
        if self.current_frame is None and not self.started:
            label = 'Choose an input in Setup, then press Start.'
            draw.text((left+18, top+18), label,
                      fill=(205, 219, 229), font=font)
        elif self.current_frame is None:
            draw.text((left+18, top+18), 'Waiting for the first decoded picture…',
                      fill=(205, 219, 229), font=font)
        if (self.live_meter or {}).get('sync_warning'):
            warning = (left+14, top+12, left+194, top+43)
            draw.rounded_rectangle(warning, radius=5, fill=(125, 42, 31),
                                   outline=(238, 130, 93), width=1)
            draw.text((warning[0]+9, warning[1]+7), 'SYNC LOST',
                      fill=(255, 239, 228), font=small)

        if self._diagnostics_visible():
            panel_height = round(height*.27)
            footer_height = 38 if not self.fullscreen or self.toolbar_visible else 0
            panel_top = height-footer_height-panel_height
            diagnostics = self.live_diagnostics or {
                'status': ('ACQUIRING',),
                'sync': ('waiting for pulse header',),
                'decode': ('no decoded frames yet',),
                'decode_cpu': ('last -- ms/frame', 'average -- / core'),
                'input': ('waiting for audio',),
                'signal': ('picture appears here as soon as it decodes',),
                'resources': self.gui_resource_lines,
            }
            panel = Image.fromarray(_diagnostic_image(
                (width, panel_height), diagnostics), mode='RGBA')
            image.alpha_composite(panel, (0, panel_top))

        chrome_visible = not self.fullscreen or self.toolbar_visible
        if chrome_visible:
            footer_top = height-38
            draw.rectangle((0, footer_top, width, height), fill=(10, 18, 25))
            state = ('RECEIVER RUNNING' if self.started else
                     'RECEIVER STOPPED' if self.ever_started else 'NOT STARTED')
            if (self.receiver_thread is not None and
                    not self.receiver_thread.is_alive()):
                state = 'RECEIVER STOPPED'
            count = (self.live_meter or {}).get('decoded', 0)
            input_fps = (self.live_meter or {}).get('input_fps', 0.0)
            meter = self.live_meter or {}
            routing = (f'{meter.get("detected_mode", "acquiring")} · '
                       f'video {meter.get("video_side") or "--"} / '
                       f'audio {meter.get("audio_side") or "--"}')
            output = meter.get('audio_output_device') or 'off'
            volume = meter.get('audio_volume', 1.0)
            audio = (f'{"muted" if meter.get("audio_muted") else "live"} '
                     f'{volume:.0%} → {output}'
                     if meter.get('audio_side') else 'off')
            audio_error = meter.get('audio_device_error')
            if audio_error:
                audio = f'RESELECT · {audio_error}'
            sync = meter.get('sync_state', 'acquiring')
            if self.current_frame is not None:
                status = (self.latest_report or {}).get(
                    'status', 'picture decoded')
                display = ('' if self.display_latency_ms is None else
                           f' · GUI handoff {self.display_latency_ms:.1f} ms')
                detail = (f'{state}{display} · {status} · {count} pictures · '
                          f'input {input_fps:.1f} fps · {routing} · '
                          f'audio {audio} · sync {sync}')
            else:
                detail = (f'{state} · {self.notice} · {routing} · '
                          f'audio {audio} · sync {sync}')
            draw.text((14, footer_top+11), _fit_text(detail, small, width-28),
                      fill=(147, 206, 169) if self.started else
                      (189, 203, 214), font=small)

    def _render_display_menu(self, image, draw, small):
        bounds = self._display_menu_bounds(image.size[0])
        if bounds is None:
            return
        left, menu_top, right, menu_bottom = bounds
        draw.rounded_rectangle((left, menu_top, right, menu_bottom), radius=5,
                               fill=(12, 22, 31, 250),
                               outline=(93, 132, 155), width=1)
        for index, mode in enumerate(DISPLAY_MODES):
            top = menu_top+2+index*DISPLAY_MENU_ROW_HEIGHT
            box = (left+2, top, right-2, top+DISPLAY_MENU_ROW_HEIGHT-1)
            if mode == self.display_mode:
                draw.rectangle(box, fill=(42, 75, 96))
            if index == self.display_menu_index:
                draw.rectangle(box, outline=(105, 165, 195), width=1)
            label = _fit_text(DISPLAY_LABELS[mode], small, right-left-20)
            draw.text((left+10, top+6), label, fill=(235, 241, 246),
                      font=small)
            self.hits[f'display_mode:{mode}'] = (
                left, top, right, top+DISPLAY_MENU_ROW_HEIGHT-1)

    def _display_menu_bounds(self, width):
        if (not self.display_menu_open or self.page != 'info' or
                self.image_only or not self.toolbar_visible):
            return None
        button = self.hits.get('mode_button')
        if button is None:
            return None
        left = button[0]
        right = min(width-14, left+DISPLAY_MENU_WIDTH)
        top = TOOLBAR_HEIGHT+2
        bottom = top+len(DISPLAY_MODES)*DISPLAY_MENU_ROW_HEIGHT+4
        return left, top, right, bottom

    def _canvas(self, size):
        width, height = size
        image = Image.new('RGBA', (width, height), (8, 14, 20, 255))
        draw = ImageDraw.Draw(image)
        font, small, mono = _font(17), _font(13), _font(12, mono=True)
        self.hits = {}
        chrome_visible = not self.fullscreen or self.toolbar_visible
        if chrome_visible:
            draw.rectangle((0, 0, width, TOOLBAR_HEIGHT),
                           fill=(10, 18, 25, 255))
            draw.rectangle((0, TOOLBAR_HEIGHT-1, width, TOOLBAR_HEIGHT),
                           fill=(47, 68, 83, 255))
            diagnostics_visible = self._diagnostics_visible()
            controls = (
                ('config_tab', 'Setup', 12, 104),
                ('info_tab', 'Live', 112, 184),
                ('start_stop', 'Stop' if self.started else 'Start', 192, 284),
                ('mode_button', f'{DISPLAY_LABELS[self.display_mode]}  ▾',
                 width-414, width-300),
                ('details_button', 'Info On' if diagnostics_visible else
                 'Info Off', width-292, width-220),
                ('image_only', 'Image only', width-212, width-112),
                ('fullscreen', 'Fullscreen', width-104, width-12),
            )
            for key, label, x1, x2 in controls:
                if (key in ('mode_button', 'details_button', 'image_only') and
                        self.page != 'info'):
                    continue
                self.hits[key] = (x1, 9, x2, 46)
                active = ((key == 'config_tab' and self.page == 'config') or
                          (key == 'info_tab' and self.page == 'info'))
                fill = ((39, 67, 86) if active else
                        (82, 55, 40) if key == 'start_stop' and self.started else
                        (43, 94, 123) if key == 'start_stop' else
                        (22, 35, 46))
                draw.rounded_rectangle(self.hits[key], radius=5, fill=fill,
                                       outline=(67, 100, 122), width=1)
                text_width = max(1, x2-x1-16)
                shown = _fit_text(label, small, text_width)
                draw.text((x1+8, 18), shown,
                          fill=(246, 240, 235) if key == 'start_stop' and self.started
                          else (236, 242, 247), font=small)
        if self.page == 'config':
            self._render_config(image, draw, font, small, mono)
        else:
            self._render_info(image, draw, font, small, mono)
            self._render_display_menu(image, draw, small)
        return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))

    def _config_field_indexes(self):
        indexes = []
        for index, field in enumerate(self.fields):
            if (self.advanced_options or field.dest in BASIC_OPTION_DESTS or
                    field.label == 'Display upscaler'):
                indexes.append(index)
        return indexes

    def _visible_fields(self, height):
        top, bottom = 140, height-126
        return top, max(1, (bottom-top)//ROW_HEIGHT)

    def _change_page(self):
        if self.editing:
            self._finish_edit(self.fields[self.selected])
        self.page = 'info' if self.page == 'config' else 'config'
        self.dropdown = None
        self.display_menu_open = False
        self.editing = False
        self.dirty = True

    def _on_mouse(self, glfw, window, button, action, _mods):
        if button != glfw.MOUSE_BUTTON_LEFT or action != glfw.PRESS:
            return
        self._reveal_toolbar()
        x, y = glfw.get_cursor_pos(window)
        keys = list(self.hits)
        if self.dropdown is not None:
            keys.sort(key=lambda key: 0 if key.startswith('option:') else 1)
        if self.display_menu_open:
            keys.sort(key=lambda key: 0 if key.startswith('display_mode:') else 1)
        hit = next((key for key in keys
                    if self.hits[key][0] <= x < self.hits[key][2] and
                    self.hits[key][1] <= y < self.hits[key][3]), None)
        is_display_choice = (hit is not None and
                             hit.startswith('display_mode:'))
        if (self.display_menu_open and not is_display_choice and
                hit != 'mode_button'):
            self.display_menu_open = False
        if self.editing:
            self._finish_edit(self.fields[self.selected])
        if hit == 'config_tab':
            self.page, self.dropdown = 'config', None
            self.display_menu_open = False
        elif hit == 'info_tab':
            self.page, self.dropdown = 'info', None
            self.display_menu_open = False
        elif hit == 'fullscreen':
            self._toggle_fullscreen(glfw, window)
        elif hit == 'image_only':
            self._set_image_only(True)
        elif hit == 'details_button':
            field = next(field for field in self.fields
                         if field.dest == 'show_diagnostics')
            field.value = not field.value
            self.dirty = True
        elif hit == 'start_stop':
            if self.started:
                self._stop_receiver()
            else:
                self._start_receiver()
        elif hit == 'mode_button':
            self.dropdown = None
            self.display_menu_open = not self.display_menu_open
            self.display_menu_index = DISPLAY_MODES.index(self.display_mode)
            self.dirty = True
        elif is_display_choice:
            mode = hit.split(':', 1)[1]
            field = next(field for field in self.fields
                         if field.label == 'Display upscaler')
            self._select_choice(field, mode)
            self.display_menu_open = False
        elif hit and hit.startswith('option:') and self.dropdown is not None:
            index = int(hit.split(':', 1)[1])
            field = self.fields[self.dropdown]
            self._select_choice(field, field.options[index][1])
        elif hit and hit.startswith('row:') and self.page == 'config':
            index = int(hit.split(':', 1)[1])
            self.selected = index
            field = self.fields[index]
            if (self.started and field.action is not None and
                    field.dest not in LIVE_RUNTIME_DESTS):
                self.notice = ('Receiver options are locked while receiving; '
                               'close and relaunch to change them.')
            elif not field.locked:
                if field.kind == 'bool':
                    self._adjust_field(field, 1)
                elif field.kind == 'choice':
                    if field.dest == 'audio_output_device':
                        self._refresh_audio_output_choices()
                    self.dropdown = (None if self.dropdown == index else index)
                    self.dropdown_scroll = 0
                    if len(field.options) <= 1 and self.audio_output_error:
                        self.notice = self.audio_output_error
                elif field.kind == 'folder':
                    self._choose_save_directory(field)
                else:
                    self.editing = True
                    self.edit_buffer = '' if field.value is None else str(field.value)
            self.dirty = True
        elif hit == 'advanced_toggle':
            self.advanced_options = not self.advanced_options
            self.dropdown = None
            self.scroll = 0
            order = self._config_field_indexes()
            if order and self.selected not in order:
                self.selected = order[0]
            self.dirty = True
        elif self.dropdown is not None or self.display_menu_open:
            self.dropdown = None
            self.display_menu_open = False
            self.dirty = True
        else:
            self.dirty = True

    def _on_key(self, glfw, window, key, _scancode, action, mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if self.image_only and key in (glfw.KEY_F, glfw.KEY_I):
            was_fullscreen = self.image_only_previous_fullscreen
            self._set_image_only(False)
            if key == glfw.KEY_F:
                # Image-only may have entered fullscreen on its own, or may
                # have been enabled from an already-fullscreen Live view.
                # In both cases F must leave fullscreen exactly once.
                if was_fullscreen and self.fullscreen:
                    self._toggle_fullscreen(glfw, window)
            else:
                self._reveal_toolbar()
                self._toggle_details()
            return
        if (self.fullscreen and key in (glfw.KEY_F, glfw.KEY_I)):
            if self.editing:
                self._finish_edit(self.fields[self.selected])
            self._reveal_toolbar()
            if key == glfw.KEY_F:
                self._toggle_fullscreen(glfw, window)
            else:
                self._toggle_details()
            return
        self._reveal_toolbar()
        if self.editing:
            field = self.fields[self.selected]
            if key in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER):
                self._finish_edit(field)
            elif key == glfw.KEY_ESCAPE:
                self.editing = False
            elif key == glfw.KEY_BACKSPACE:
                self.edit_buffer = self.edit_buffer[:-1]
            elif key == glfw.KEY_A and mods & glfw.MOD_CONTROL:
                self.edit_buffer = ''
            self.dirty = True
            return
        if self.image_only:
            if key in (glfw.KEY_ESCAPE, glfw.KEY_P):
                self._set_image_only(False)
            return
        if self.display_menu_open:
            if key == glfw.KEY_ESCAPE:
                self.display_menu_open = False
                self.dirty = True
                return
            if key in (glfw.KEY_UP, glfw.KEY_DOWN):
                direction = -1 if key == glfw.KEY_UP else 1
                self.display_menu_index = (
                    self.display_menu_index+direction) % len(DISPLAY_MODES)
                self.dirty = True
                return
            if key in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER):
                mode = DISPLAY_MODES[self.display_menu_index]
                field = next(field for field in self.fields
                             if field.label == 'Display upscaler')
                self._select_choice(field, mode)
                self.display_menu_open = False
                return
            self.display_menu_open = False
            self.dirty = True
        if key == glfw.KEY_F:
            self._toggle_fullscreen(glfw, window)
        elif key == glfw.KEY_P:
            self._set_image_only(True)
        elif key == glfw.KEY_I:
            self._toggle_details()
        elif key == glfw.KEY_TAB:
            self._change_page()
        elif key == glfw.KEY_C:
            self.page = 'config'
            self.dropdown = None
            self.display_menu_open = False
            self.dirty = True
        elif key == glfw.KEY_ESCAPE:
            if self.dropdown is not None:
                self.dropdown = None
                self.dirty = True
            elif self.fullscreen:
                self._toggle_fullscreen(glfw, window)
            else:
                glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_ENTER and self.page == 'config':
            if self.dropdown is not None:
                self._select_choice(self.fields[self.dropdown],
                                    self.fields[self.dropdown].options[
                                        self.dropdown_scroll][1])
            else:
                field = self.fields[self.selected]
                if field.kind == 'bool':
                    self._adjust_field(field, 1)
                elif field.kind == 'choice':
                    if field.dest == 'audio_output_device':
                        self._refresh_audio_output_choices()
                    self.dropdown = self.selected
                    self.dropdown_scroll = 0
                    if (field.dest == 'audio_output_device' and
                            len(field.options) <= 1 and
                            self.audio_output_error):
                        self.notice = self.audio_output_error
                elif (field.dest in ('freewheel_seconds', 'audio_volume') and
                      (not self.started or
                       field.dest in LIVE_RUNTIME_DESTS) and
                      not field.locked):
                    self.editing = True
                    self.edit_buffer = str(field.value or '')
                elif field.kind == 'folder' and not field.locked:
                    self._choose_save_directory(field)
                elif not self.started and not field.locked:
                    self.editing = True
                    self.edit_buffer = str(field.value or '')
                self.dirty = True
        elif key in (glfw.KEY_UP, glfw.KEY_DOWN) and self.page == 'config':
            if self.dropdown is not None:
                options = self.fields[self.dropdown].options
                self.dropdown_scroll = max(
                    0, min(len(options)-1, self.dropdown_scroll+
                           (1 if key == glfw.KEY_DOWN else -1)))
            else:
                delta = -1 if key == glfw.KEY_UP else 1
                order = self._config_field_indexes()
                position = order.index(self.selected) if self.selected in order else 0
                position = max(0, min(len(order)-1, position+delta))
                if order:
                    self.selected = order[position]
                top, visible = self._visible_fields(self.height)
                if position < self.scroll:
                    self.scroll = position
                elif position >= self.scroll+visible:
                    self.scroll = position-visible+1
            self.dirty = True
        elif key in (glfw.KEY_UP, glfw.KEY_DOWN) and self.page == 'info':
            if key == glfw.KEY_DOWN:
                self.info_follow = True
                self.info_scroll = 10**9
            else:
                self.info_follow = False
                self.info_scroll = max(0, self.info_scroll-1)
            self.dirty = True
        elif key in (glfw.KEY_LEFT, glfw.KEY_RIGHT) and self.page == 'config':
            if self.dropdown is not None:
                self._adjust_field(self.fields[self.dropdown],
                                   -1 if key == glfw.KEY_LEFT else 1)
                self.dropdown = None
            else:
                self._adjust_field(self.fields[self.selected],
                                   -1 if key == glfw.KEY_LEFT else 1)
            self.dirty = True

    def _on_char(self, _window, codepoint):
        if self.editing and codepoint >= 32:
            self.edit_buffer += chr(codepoint)
            self.dirty = True

    def _finish_edit(self, field):
        value = self.edit_buffer.strip()
        if field.dest in ('freewheel_seconds', 'audio_volume'):
            try:
                number = float(value)
                valid = (np.isfinite(number) and
                         (number >= 0 if field.dest == 'freewheel_seconds'
                          else 0.0 <= number <= 1.0))
                if not valid:
                    raise ValueError
            except ValueError:
                self.editing = False
                self.notice = (
                    'Freewheel duration must be non-negative.'
                    if field.dest == 'freewheel_seconds' else
                    'Passthrough volume must be between 0 and 1.')
                self.dirty = True
                return
            field.value = number
        else:
            field.value = value
        self.editing = False
        self.notice = f'{field.label} updated.'
        self._update_runtime_option(field)
        if field.dest in ('freewheel_seconds', 'audio_volume', 'save_dir'):
            self._persist_preferences()
        self.dirty = True

    def _choose_save_directory(self, field):
        try:
            selected = pick_save_directory(field.value or '')
        except (OSError, RuntimeError) as exc:
            self.notice = str(exc)
        else:
            if selected:
                field.value = selected
                self.notice = f'Save folder: {selected}'
                self._persist_preferences()
        self.dirty = True

    def _on_scroll(self, _window, _xoffset, yoffset):
        self._reveal_toolbar()
        delta = -1 if yoffset > 0 else 1
        if self.display_menu_open:
            self.display_menu_index = (
                self.display_menu_index+delta) % len(DISPLAY_MODES)
        elif self.dropdown is not None:
            field = self.fields[self.dropdown]
            self.dropdown_scroll = max(
                0, min(max(0, len(field.options)-1),
                       self.dropdown_scroll+delta))
        elif self.page == 'config':
            _top, visible = self._visible_fields(self.height)
            self.scroll = max(0, min(max(0, len(self._config_field_indexes())-visible),
                                     self.scroll+delta))
        else:
            if delta > 0:
                self.info_follow = True
                self.info_scroll = 10**9
            else:
                self.info_follow = False
                self.info_scroll = max(0, self.info_scroll-1)
        self.dirty = True

    def run(self):
        import glfw
        import moderngl

        if not glfw.init():
            raise RuntimeError('GLFW initialization failed')
        window = context = program = vertex_array = None
        float_program = float_array = None
        ui_texture = picture_texture = None
        plane_textures = []
        plane_texture_shapes = None
        kernel_textures = {}
        picture_texture_size = None
        picture_texture_mode = None
        frame_buffer = getattr(self.v7_live, 'FRAME_BUFFER', None)
        set_frame_notifier = getattr(frame_buffer, 'set_notifier', None)
        try:
            window, context, use_gles = _create_graphics_context(
                glfw, moderngl)
            self._window = window
            self._glfw = glfw
            if set_frame_notifier is not None:
                set_frame_notifier(glfw.post_empty_event)
            glfw.set_window_size_limits(window, 720, 480,
                                        glfw.DONT_CARE, glfw.DONT_CARE)
            glfw.swap_interval(1)
            vertex_shader = _shader_for_context(VERTEX_SHADER, use_gles)
            fragment_shader = _shader_for_context(FRAGMENT_SHADER, use_gles)
            float_fragment_shader = _shader_for_context(
                FLOAT_FRAGMENT_SHADER, use_gles)
            program = context.program(vertex_shader=vertex_shader,
                                      fragment_shader=fragment_shader)
            program['image'].value = 0
            vertex_array = context.vertex_array(program, [])
            glfw.set_key_callback(
                window, lambda w, k, s, a, m: self._on_key(glfw, w, k, s, a, m))
            glfw.set_char_callback(window, self._on_char)
            glfw.set_cursor_pos_callback(window, self._on_cursor_position)
            glfw.set_mouse_button_callback(
                window, lambda w, b, a, m: self._on_mouse(glfw, w, b, a, m))
            glfw.set_scroll_callback(window, self._on_scroll)
            glfw.set_window_refresh_callback(
                window, lambda _window: setattr(self, 'dirty', True))

            def upload_picture():
                nonlocal picture_texture, picture_texture_size
                nonlocal plane_textures, plane_texture_shapes
                frame = self.current_frame
                if frame is None:
                    return
                if self.display_mode == 'nearest':
                    if self.latest_values_image is None:
                        try:
                            self.latest_values_image = self.v7_live.values_image(
                                frame.values, frame.shapes).convert('RGB')
                        except Exception as exc:
                            self.lines.append(
                                f'Could not render decoded frame: {exc}')
                            self.dirty = True
                            return
                    pixels = np.ascontiguousarray(np.asarray(
                        self.latest_values_image, dtype=np.uint8))
                    picture_size = (pixels.shape[1], pixels.shape[0])
                    pixel_bytes = pixels.tobytes()
                    if (picture_texture is None or
                            picture_texture_size != picture_size):
                        if picture_texture is not None:
                            picture_texture.release()
                        picture_texture = context.texture(
                            picture_size, 3, pixel_bytes, dtype='f1')
                        picture_texture_size = picture_size
                        picture_texture.repeat_x = False
                        picture_texture.repeat_y = False
                    else:
                        picture_texture.write(pixel_bytes)
                else:
                    planes = float_planes(
                        frame.values, frame.shapes)
                    if self.display_mode in FILTER_PRECOMPUTE_MODES:
                        planes = resample_filter_planes(
                            planes, self.display_mode)
                    plane_sizes = tuple((plane.shape[1], plane.shape[0])
                                        for plane in planes)
                    if (plane_texture_shapes != plane_sizes or
                            len(plane_textures) != len(planes)):
                        for plane_texture in plane_textures:
                            plane_texture.release()
                        plane_textures = [
                            context.texture(plane_size, 1, plane.tobytes(),
                                            dtype='f4')
                            for plane_size, plane in zip(plane_sizes, planes)]
                        for plane_texture in plane_textures:
                            plane_texture.repeat_x = False
                            plane_texture.repeat_y = False
                        plane_texture_shapes = plane_sizes
                    else:
                        for plane_texture, plane in zip(plane_textures, planes):
                            plane_texture.write(plane.tobytes())
                if picture_texture is not None:
                    picture_texture.filter = (moderngl.NEAREST,
                                              moderngl.NEAREST)
                filtering = _float_texture_filter(self.display_mode, moderngl)
                for plane_texture in plane_textures:
                    plane_texture.filter = (filtering, filtering)

            def picture_uploaded():
                return (picture_texture is not None
                        if self.display_mode == 'nearest' else
                        len(plane_textures) == 3)

            def ensure_float_renderer():
                nonlocal float_program, float_array
                if float_program is not None:
                    return
                float_program = context.program(
                    vertex_shader=vertex_shader,
                    fragment_shader=float_fragment_shader)
                float_program['plane_y'].value = 0
                float_program['plane_cb'].value = 1
                float_program['plane_cr'].value = 2
                float_program['kernel_lut'].value = 3
                float_array = context.vertex_array(float_program, [])

            def kernel_texture_for(mode):
                texture_for_mode = kernel_textures.get(mode)
                if texture_for_mode is None:
                    weights = build_filter_lut(mode)
                    texture_for_mode = context.texture(
                        (weights.size, 1), 1, weights.tobytes(), dtype='f4')
                    texture_for_mode.filter = (moderngl.LINEAR,
                                               moderngl.LINEAR)
                    texture_for_mode.repeat_x = False
                    texture_for_mode.repeat_y = False
                    kernel_textures[mode] = texture_for_mode
                return texture_for_mode

            def render_picture(viewport):
                context.viewport = viewport
                if self.display_mode == 'nearest':
                    picture_texture.use(location=0)
                    vertex_array.render(mode=moderngl.TRIANGLES, vertices=3)
                    return
                ensure_float_renderer()
                for unit, plane_texture in enumerate(plane_textures):
                    plane_texture.use(location=unit)
                kernel_texture_for(self.display_mode).use(location=3)
                float_program['reconstruction'].value = FLOAT_MODE_IDS[
                    self.display_mode]
                float_program['output_size'].value = (
                    float(viewport[2]), float(viewport[3]))
                float_program['filtered_intermediate'].value = int(
                    self.display_mode in FILTER_PRECOMPUTE_MODES)
                float_array.render(mode=moderngl.TRIANGLES, vertices=3)

            def next_event_timeout(now):
                # GLFW's blocking wait delays Python signal handling. Keep a
                # low-frequency fallback so Ctrl-C is observed while idle.
                timeout = QUIET_WAIT_SECONDS
                receiver_active = (self.receiver_thread is not None and
                                   self.receiver_thread.is_alive())
                receiver_stopped = (self.receiver_thread is not None and
                                    not receiver_active and
                                    (self.started or self.notice.startswith(
                                        'Stopping receiver')))
                if (receiver_active and not self.image_only and
                        self.page == 'info'):
                    timeout = (INFO_REFRESH_SECONDS
                               if self._diagnostics_visible() else
                               FOOTER_REFRESH_SECONDS)

                if receiver_stopped or not self.output.empty():
                    timeout = (0.001 if timeout is None else
                               min(timeout, 0.001))

                if self.info_refresh_pending:
                    remaining = max(
                        0.001, self.last_info_refresh+
                        INFO_REFRESH_SECONDS-now)
                    timeout = (remaining if timeout is None else
                               min(timeout, remaining))

                if (self.fullscreen and self.toolbar_visible and
                        not self.image_only and self.dropdown is None and
                        not self.display_menu_open and not self.editing and
                        glfw.get_cursor_pos(window)[1] >
                        FULLSCREEN_TOOLBAR_EDGE):
                    remaining = max(
                        0.001, self.last_ui_activity+
                        FULLSCREEN_TOOLBAR_HIDE_SECONDS-now)
                    timeout = (remaining if timeout is None else
                               min(timeout, remaining))

                if self._profile_ui_enabled():
                    remaining = (0.001 if self.profile_wall is None else
                                 max(0.001, self.profile_wall+5.0-now))
                    timeout = (remaining if timeout is None else
                               min(timeout, remaining))
                return timeout

            first_iteration = True
            while not glfw.window_should_close(window):
                # Frame publication and window callbacks wake GLFW immediately.
                # Timed waits are reserved for visible status, toolbar, and
                # profiler refreshes; a stopped, idle window blocks in GLFW.
                if first_iteration:
                    first_iteration = False
                else:
                    timeout = next_event_timeout(time.monotonic())
                    glfw.wait_events_timeout(timeout)
                self._process_output()
                self._poll_frame()
                self._poll_diagnostics()
                self._profile_ui_if_enabled()
                self._poll_receiver_lifecycle()
                now = time.monotonic()
                cursor_y = glfw.get_cursor_pos(window)[1]
                self._update_toolbar_visibility(now, cursor_y)
                if (self.info_refresh_pending and
                        now-self.last_info_refresh >= INFO_REFRESH_SECONDS):
                    self.dirty = True
                    self.info_refresh_pending = False
                    self.last_info_refresh = now
                window_size = glfw.get_window_size(window)
                window_size = (max(320, int(window_size[0])),
                               max(240, int(window_size[1])))
                framebuffer_size = glfw.get_framebuffer_size(window)
                if window_size != self.last_ui_size:
                    self.width, self.height = window_size
                    self.last_ui_size = window_size
                    self.dirty = True
                if (ui_texture is not None and
                        ui_texture.size != window_size and not self.image_only):
                    self.dirty = True

                picture_needs_draw = (self.picture_dirty and
                                      (self.page == 'info' or self.image_only))
                if picture_needs_draw:
                    upload_picture()
                    self.picture_dirty = False

                if (picture_texture is not None and
                        picture_texture_mode != self.display_mode):
                    picture_texture.filter = (moderngl.NEAREST,
                                              moderngl.NEAREST)
                    filtering = _float_texture_filter(
                        self.display_mode, moderngl)
                    for plane_texture in plane_textures:
                        plane_texture.filter = (filtering, filtering)
                    picture_texture_mode = self.display_mode

                if self.image_only:
                    if self.dirty or picture_needs_draw:
                        context.viewport = (0, 0, *framebuffer_size)
                        context.clear(.035, .045, .055, 1.0)
                        if picture_uploaded() and self.current_frame is not None:
                            aspect = self.v7_live.P.V7_ASPECT_RATIOS[
                                self.current_frame.aspect & 7]
                            render_picture(fit_viewport(
                                framebuffer_size, aspect))
                        glfw.swap_buffers(window)
                        if picture_needs_draw and self.current_frame is not None:
                            self.display_latency_ms = max(
                                0.0, 1000*(time.monotonic()-
                                          self.current_frame.published_at))
                    self.dirty = False
                    title = 'V7 Receiver · Image only'
                    if title != self.last_title:
                        glfw.set_window_title(window, title)
                        self.last_title = title
                    continue

                ui_needs_draw = self.dirty
                if ui_needs_draw:
                    rgba = self._canvas(window_size)
                    ui_size = (rgba.shape[1], rgba.shape[0])
                    if ui_texture is None or ui_texture.size != ui_size:
                        if ui_texture is not None:
                            ui_texture.release()
                        ui_texture = context.texture(
                            ui_size, 4, rgba.tobytes(), dtype='f1')
                    else:
                        ui_texture.write(rgba.tobytes())
                    ui_texture.filter = (moderngl.LINEAR,
                                         moderngl.LINEAR)
                    ui_texture.repeat_x = False
                    ui_texture.repeat_y = False
                    self.dirty = False

                if ui_needs_draw or picture_needs_draw:
                    picture_viewport = None
                    if (self.page == 'info' and self.current_frame is not None
                            and picture_uploaded()):
                        aspect = self.v7_live.P.V7_ASPECT_RATIOS[
                            self.current_frame.aspect & 7]
                        picture_viewport = self._picture_viewport(
                            window_size, framebuffer_size, aspect)
                        if not picture_viewport[2] or not picture_viewport[3]:
                            picture_viewport = None
                    if picture_viewport is None:
                        context.viewport = (0, 0, *framebuffer_size)
                        ui_texture.use(location=0)
                        vertex_array.render(
                            mode=moderngl.TRIANGLES, vertices=3)
                    else:
                        # Recompose both buffers every swap, but shade the
                        # image rectangle only once instead of under the UI.
                        _draw_scissored_ui_regions(
                            context, framebuffer_size,
                            _scissors_outside_viewport(
                                framebuffer_size, picture_viewport),
                            ui_texture, vertex_array, moderngl.TRIANGLES)
                        render_picture(picture_viewport)
                        menu_bounds = self._display_menu_bounds(
                            window_size[0])
                        if menu_bounds is not None:
                            scissor = _logical_rect_to_framebuffer(
                                menu_bounds, window_size, framebuffer_size)
                            _draw_scissored_ui(
                                context, framebuffer_size, scissor,
                                ui_texture, vertex_array,
                                moderngl.TRIANGLES)
                    glfw.swap_buffers(window)
                    if picture_needs_draw and self.current_frame is not None:
                        self.display_latency_ms = max(
                            0.0, 1000*(time.monotonic()-
                                      self.current_frame.published_at))
                title = ('V7 Receiver · ' +
                         ('Receiving' if self.started else
                          'Stopped' if self.ever_started else 'Not started') +
                         f' · {self.page.capitalize()}')
                if title != self.last_title:
                    glfw.set_window_title(window, title)
                    self.last_title = title
        finally:
            if self.receiver_stop is not None:
                self.receiver_stop.set()
            if self.receiver_thread is not None:
                self.receiver_thread.join(timeout=2)
            if set_frame_notifier is not None:
                set_frame_notifier(None)
            # The receiver may outlive the bounded join; suppress its optional
            # event notification before destroying GLFW's global state.
            self._window = None
            self._glfw = None
            if ui_texture is not None:
                ui_texture.release()
            if picture_texture is not None:
                picture_texture.release()
            for plane_texture in plane_textures:
                plane_texture.release()
            for kernel_texture in kernel_textures.values():
                kernel_texture.release()
            if vertex_array is not None:
                vertex_array.release()
            if float_array is not None:
                float_array.release()
            if float_program is not None:
                float_program.release()
            if program is not None:
                program.release()
            if context is not None:
                context.release()
            if window is not None:
                glfw.destroy_window(window)
            glfw.terminate()
            self._window = None
            self._glfw = None


def main(v7_live_module=None):
    if v7_live_module is None:
        from tools import v7_live as v7_live_module
    v7_live = v7_live_module
    root_parser, receive_parser = _receive_parser(v7_live)
    devices, device_error = _input_devices()
    audio_outputs, audio_output_error = _output_devices()
    ReceiverGui(v7_live, root_parser, receive_parser, devices,
                device_error, audio_outputs, audio_output_error).run()


if __name__ == '__main__':
    main()
