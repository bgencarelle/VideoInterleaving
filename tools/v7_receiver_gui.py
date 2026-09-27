"""Interactive, unstarted configuration and live monitor for the V7 receiver.

The interface is a thin GLFW/ModernGL shell around the existing receiver CLI
parser and `run_receive` path. It does not open an input stream until Start is
pressed, and it reads decoded pictures from the receiver's latest-frame mailbox.
"""
import argparse
import ast
import contextlib
from dataclasses import dataclass
import io
import json
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

from tools.v7_gl_viewer import FRAGMENT_SHADER, VERTEX_SHADER, fit_viewport


ROW_HEIGHT = 36
TOOLBAR_HEIGHT = 54
DISPLAY_MODES = ('nearest', 'bilinear')
INFO_REFRESH_SECONDS = 0.2


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
    def __init__(self, output):
        self.output = output
        self.pending = ''
        self.lock = threading.Lock()

    def write(self, text):
        if not text:
            return 0
        with self.lock:
            self.pending += str(text)
            lines = self.pending.split('\n')
            self.pending = lines.pop()
            for line in lines:
                if line:
                    self.output.put(line)
        return len(text)

    def flush(self):
        with self.lock:
            if self.pending:
                self.output.put(self.pending)
                self.pending = ''


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


def _field_label(action):
    friendly = {
        'device': 'Input audio device',
        'direction': 'Playback direction',
        'fixture': 'Model fixture',
        'fullscreen': 'Start fullscreen',
        'show_diagnostics': 'Show diagnostics',
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
    }
    return friendly.get(action.dest,
                        action.dest.replace('_', ' ').capitalize())


def _make_fields(receive_parser, device_choices):
    fields = []
    for action in receive_parser._actions:
        if action.dest in ('help', 'mode'):
            continue
        if action.dest == 'headless':
            # The integrated GUI owns display; this internal CLI switch keeps
            # the receiver from opening a second, nested GL viewer.
            fields.append(OptionField(action, True, 'GUI receiver backend',
                                      'bool', locked=True))
            continue
        if action.dest == 'device':
            fields.append(OptionField(action, None, _field_label(action),
                                      'choice', device_choices))
            continue

        value = action.default
        if value is argparse.SUPPRESS:
            value = None
        if action.dest in ('diagnostics', 'log'):
            # The information view should be useful on the first run.
            value = True
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
                              tuple((name.capitalize(), name)
                                    for name in DISPLAY_MODES)))
    return fields


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


class ReceiverGui:
    def __init__(self, v7_live, root_parser, receive_parser, device_choices,
                 device_error=''):
        self.v7_live = v7_live
        self.root_parser = root_parser
        self.receive_parser = receive_parser
        self.fields = _make_fields(receive_parser, device_choices)
        self.device_error = device_error
        self.page = 'config'
        self.fullscreen = False
        self.windowed_bounds = {'position': (80, 80), 'size': (960, 720)}
        self.selected = 0
        self.scroll = 0
        self.dropdown = None
        self.dropdown_scroll = 0
        self.editing = False
        self.edit_buffer = ''
        self.notice = 'Choose an input device, review settings, then start receiving.'
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
        self.display_mode = 'nearest'
        self.last_title = None
        self.last_ui_size = None
        self.profile_wall = None
        self.profile_process = None
        self.profile_thread = None

    def _field_value_label(self, field):
        if field.dest == 'device':
            return next((label for label, value in field.options
                         if value == field.value), 'Select an input device…')
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

    def _toggle_fullscreen(self, glfw, window):
        primary = glfw.get_primary_monitor()
        if primary is None:
            return
        if self.fullscreen:
            x, y = self.windowed_bounds['position']
            width, height = self.windowed_bounds['size']
            glfw.set_window_monitor(window, None, x, y, width, height,
                                    glfw.DONT_CARE)
            self.fullscreen = False
        else:
            self.windowed_bounds['position'] = glfw.get_window_pos(window)
            self.windowed_bounds['size'] = glfw.get_window_size(window)
            mode = glfw.get_video_mode(primary)
            glfw.set_window_monitor(window, primary, 0, 0, mode.size.width,
                                    mode.size.height, mode.refresh_rate)
            self.fullscreen = True
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
            if dest in ('help', 'mode', 'headless', 'fullscreen'):
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
            writer = QueueWriter(self.output)
            try:
                with contextlib.redirect_stdout(writer), \
                        contextlib.redirect_stderr(writer):
                    self.v7_live.run_receive(args)
            except BaseException:
                writer.write(traceback.format_exc())
            finally:
                writer.flush()
                self.output.put('[receiver thread returned]')

        self.receiver_thread = threading.Thread(
            target=receive, name='v7-receiver', daemon=True)
        self.receiver_thread.start()
        self.page = 'info'
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
        if self.started and field.action is not None:
            self.dropdown = None
            self.notice = ('Receiver options are locked while receiving; '
                           'close and relaunch to change them.')
            self.dirty = True
            return
        field.value = value
        if field.dest == 'experimental_fold':
            baseline = next((item for item in self.fields
                             if item.dest == 'baseline'), None)
            if baseline is not None:
                baseline.value = False
        self.dropdown = None
        self.dropdown_scroll = 0
        if field.label == 'Display upscaler':
            self.display_mode = value
        self.notice = f'{field.label}: {self._field_value_label(field)}'
        self.dirty = True

    def _adjust_field(self, field, direction):
        if self.started and field.action is not None:
            self.notice = ('Receiver options are locked while receiving; '
                           'close and relaunch to change them.')
            self.dirty = True
            return
        if field.locked:
            return
        if field.kind == 'bool':
            field.value = not field.value
            if field.dest == 'baseline' and field.value:
                fold = next((item for item in self.fields
                             if item.dest == 'experimental_fold'), None)
                if fold is not None:
                    fold.value = None
        elif field.kind == 'choice' and field.options:
            values = [value for _label, value in field.options]
            try:
                index = values.index(field.value)
            except ValueError:
                index = -1
            field.value = values[(index+direction) % len(values)]
            if field.dest == 'experimental_fold':
                baseline = next((item for item in self.fields
                                 if item.dest == 'baseline'), None)
                if baseline is not None:
                    baseline.value = False
            if field.label == 'Display upscaler':
                self.display_mode = field.value
        self.notice = f'{field.label}: {self._field_value_label(field)}'
        self.dirty = True

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

    def _profile_ui_if_enabled(self):
        enabled = any(field.action is not None and
                      field.action.dest == 'profile_ui' and field.value
                      for field in self.fields)
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

    def _poll_frame(self):
        frame = self.v7_live.FRAME_BUFFER.snapshot()
        if frame is None or frame.generation == self.last_generation:
            return
        self.last_generation = frame.generation
        self.current_frame = frame
        try:
            self.latest_values_image = self.v7_live.values_image(
                frame.values, frame.shapes).convert('RGB')
            self.picture_dirty = True
        except Exception as exc:
            self.lines.append(f'Could not render decoded frame: {exc}')
            self.dirty = True

    def _poll_diagnostics(self):
        now = time.monotonic()
        if now-self.last_diagnostics_poll < INFO_REFRESH_SECONDS:
            return
        self.last_diagnostics_poll = now
        snapshot = dict(self.v7_live.RECEIVER_GUI_STATUS)
        provider = snapshot.get('diagnostics')
        if provider is None:
            return
        try:
            diagnostics = provider()
        except Exception:
            return
        packet = snapshot.get('latest_packet')
        decode_info = snapshot.get('decode_info')
        meter = snapshot.get('meter')
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
        if changed:
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
        draw.text((24, 70), 'Receiver configuration',
                  fill=(240, 245, 249), font=font)
        draw.text((24, 98),
                  'Idle · choose an input, review options, then Start. Wheel or ↑/↓ scrolls.',
                  fill=(151, 174, 192), font=small)
        top = 132
        bottom = height-126
        visible = max(1, (bottom-top)//ROW_HEIGHT)
        self.scroll = max(0, min(self.scroll, len(self.fields)-visible))
        draw.text((width-190, 74),
                  f'{self.scroll+1}–{min(len(self.fields), self.scroll+visible)} / {len(self.fields)}',
                  fill=(120, 147, 166), font=small)
        self.hits.update({'start': (width-236, height-62,
                                    width-18, height-16)})
        for row in range(visible):
            index = self.scroll+row
            if index >= len(self.fields):
                break
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
            yrow = top+(self.dropdown-self.scroll)*ROW_HEIGHT
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
        draw.rounded_rectangle(
            self.hits['start'], radius=6,
            fill=(82, 65, 48) if self.started else (43, 94, 123),
            outline=(170, 122, 79) if self.started else (90, 159, 192),
            width=1)
        draw.text((self.hits['start'][0]+18, self.hits['start'][1]+12),
                  'Stop receiver' if self.started else 'Start receiver',
                  fill=(246, 250, 252), font=font)
        if self.started:
            draw.text((self.hits['start'][0]-275, self.hits['start'][1]+13),
                      'Receiver running',
                      fill=(255, 204, 140), font=small)

        if 0 <= self.selected < len(self.fields):
            field = self.fields[self.selected]
            help_text = getattr(field.action, 'help', '') if field.action else (
                'Display-only option; does not affect the decoded values.')
            if help_text and help_text != argparse.SUPPRESS:
                lines = _text_lines(help_text, width-48, small)
                if lines:
                    draw.text((24, height-118), lines[0][:150],
                              fill=(123, 148, 168), font=small)

    def _picture_box(self, size):
        width, height = size
        return (18, 68, max(240, round(width*.57)),
                max(160, round((height-92)*.62)))

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
        self.hits.update({'mode_button': (width-400, 9,
                                          width-250, 46)})
        mode_box = self.hits['mode_button']
        draw.rounded_rectangle(mode_box, radius=5, fill=(22, 35, 46),
                               outline=(67, 100, 122), width=1)
        draw.text((mode_box[0]+10, 18),
                  f'Upscale {self.display_mode.capitalize()}  ▾',
                  fill=(236, 242, 247), font=small)
        left, top, picture_w, picture_h = self._picture_box(image.size)
        picture_area = (picture_w, picture_h)
        draw.rounded_rectangle((left, top, left+picture_w, top+picture_h),
                               radius=6, fill=(4, 8, 12),
                               outline=(49, 69, 83), width=1)
        if self.latest_values_image is None and not self.started:
            draw.text((left+20, top+22), 'Receiver not started',
                      fill=(205, 219, 229), font=font)
            draw.text((left+20, top+55),
                      'Open Configuration, choose an input device, and press Start.',
                      fill=(134, 159, 179), font=small)
        else:
            draw.text((left+20, top+22), 'Waiting for a decoded frame…',
                      fill=(205, 219, 229), font=font)

        right = left+picture_w+16
        panel_w = max(220, width-right-18)
        state = ('RECEIVER RUNNING' if self.started else
                 'RECEIVER STOPPED' if self.ever_started else 'NOT STARTED')
        if self.receiver_thread is not None and not self.receiver_thread.is_alive():
            state = 'RECEIVER STOPPED'
        draw.rounded_rectangle((right, top, width-18, top+72), radius=5,
                               fill=(15, 26, 35), outline=(49, 69, 83), width=1)
        draw.text((right+12, top+10), state,
                  fill=(147, 206, 169) if self.started else (189, 203, 214),
                  font=font)
        if self.current_frame is not None:
            status = (self.latest_report or {}).get('status', 'picture decoded')
            index = (self.latest_report or {}).get('source_index', '--')
            detail = f'{status} · source index {index}'
        else:
            detail = self.notice[:80]
        draw.text((right+12, top+42), _fit_text(detail, small, panel_w-24),
                  fill=(156, 177, 192), font=small)
        device_field = next(
            (field for field in self.fields
             if field.action is not None and field.action.dest == 'device'),
            None)
        device_text = (self._field_value_label(device_field)
                       if device_field is not None else 'not selected')
        draw.text((right+12, top+60),
                  _fit_text(f'Input: {device_text}', small, panel_w-24),
                  fill=(131, 154, 171), font=small)

        info_top = top+84
        info_bottom = height-20
        draw.rounded_rectangle((right, info_top, width-18, info_bottom),
                               radius=5, fill=(13, 23, 32),
                               outline=(49, 69, 83), width=1)
        show_details = next(
            (field.value for field in self.fields
             if field.action is not None and
             field.action.dest == 'show_diagnostics'), True)
        if not show_details:
            draw.text((right+12, info_top+12),
                      'Diagnostics hidden. Enable Show diagnostics in Configuration.',
                      fill=(175, 194, 208), font=small)
        else:
            draw.text((right+12, info_top+9), 'LATEST DECODER INFORMATION',
                      fill=(134, 166, 188), font=small)
            if self.live_diagnostics is not None:
                sections = []
                for key in ('status', 'sync', 'decode', 'input', 'signal'):
                    values = self.live_diagnostics.get(key, ())
                    sections.append(f'{key.upper()}')
                    sections.extend(f'  {line}' for line in values)
                if self.live_meter is not None:
                    sections.extend(('LIVE METER', json.dumps(
                        self.live_meter, indent=2, default=str)))
                packet = self.live_packet or self.latest_report
                if packet is not None:
                    sections.extend(('LATEST PACKET', json.dumps(
                        packet, indent=2, default=str)))
                if self.live_decode_info is not None:
                    sections.extend(('DECODE DETAILS', json.dumps(
                        self.live_decode_info, indent=2, default=str)))
                report_text = '\n'.join(sections)
            elif self.latest_report is not None:
                report_text = json.dumps(self.latest_report, indent=2, default=str)
            elif self.lines:
                report_text = '\n'.join(self.lines)
            elif self.started:
                report_text = 'Waiting for receiver status…'
            else:
                report_text = ('The receiver is idle. No audio stream is open.\n'
                               'Configuration and device selection remain available.')
            if self.lines:
                report_text += '\n\nRECENT OUTPUT\n' + '\n'.join(
                    str(line) for line in list(self.lines)[-3:])
            chars = max(24, int((panel_w-26)/max(1, mono.getlength('M'))))
            lines = []
            for raw in report_text.splitlines():
                while len(raw) > chars:
                    lines.append(raw[:chars])
                    raw = raw[chars:]
                lines.append(raw)
            max_lines = max(1, (info_bottom-info_top-44)//17)
            max_start = max(0, len(lines)-max_lines)
            if self.info_follow:
                self.info_scroll = max_start
            else:
                self.info_scroll = max(0, min(self.info_scroll, max_start))
            start = self.info_scroll
            for index, line in enumerate(lines[start:start+max_lines]):
                draw.text((right+12, info_top+32+index*17), line,
                          fill=(223, 232, 239), font=mono)
        draw.text((left, top+picture_h+10), _fit_text(
                  f'View: {self.display_mode} · I toggles config/info · F fullscreen · wheel scrolls details',
                  small, picture_w),
                  fill=(135, 159, 178), font=small)

    def _canvas(self, size):
        width, height = size
        image = Image.new('RGBA', (width, height), (8, 14, 20, 255))
        draw = ImageDraw.Draw(image)
        font, small, mono = _font(17), _font(13), _font(12, mono=True)
        draw.rectangle((0, 0, width, TOOLBAR_HEIGHT), fill=(10, 18, 25, 255))
        draw.rectangle((0, TOOLBAR_HEIGHT-1, width, TOOLBAR_HEIGHT),
                       fill=(47, 68, 83, 255))
        self.hits = {}
        tabs = (('config_tab', 'Configuration', 14, 160),
                ('info_tab', 'Information', 168, 310))
        for key, label, x1, x2 in tabs:
            active = (key == 'config_tab' and self.page == 'config') or (
                key == 'info_tab' and self.page == 'info')
            box = (x1, 9, x2, 46)
            self.hits[key] = box
            draw.rounded_rectangle(box, radius=5,
                                   fill=(39, 67, 86) if active else (22, 35, 46),
                                   outline=(67, 100, 122), width=1)
            draw.text((x1+12, 18), label, fill=(236, 242, 247), font=small)
        full_box = (width-120, 9, width-14, 46)
        self.hits['fullscreen'] = full_box
        draw.rounded_rectangle(full_box, radius=5, fill=(22, 35, 46),
                               outline=(67, 100, 122), width=1)
        draw.text((width-106, 18), 'Fullscreen',
                  fill=(236, 242, 247), font=small)
        if self.started:
            stop_box = (width-240, 9, width-132, 46)
            self.hits['start_stop'] = stop_box
            draw.rounded_rectangle(stop_box, radius=5, fill=(82, 55, 40),
                                   outline=(170, 122, 79), width=1)
            draw.text((stop_box[0]+14, 18), 'Stop',
                      fill=(246, 240, 235), font=small)
        if self.page == 'config':
            self._render_config(image, draw, font, small, mono)
        else:
            self._render_info(image, draw, font, small, mono)
        return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))

    def _visible_fields(self, height):
        top, bottom = 132, height-126
        return top, max(1, (bottom-top)//ROW_HEIGHT)

    def _change_page(self):
        if self.editing:
            self._finish_edit(self.fields[self.selected])
        self.page = 'info' if self.page == 'config' else 'config'
        self.dropdown = None
        self.editing = False
        self.dirty = True

    def _on_mouse(self, glfw, window, button, action, _mods):
        if button != glfw.MOUSE_BUTTON_LEFT or action != glfw.PRESS:
            return
        x, y = glfw.get_cursor_pos(window)
        keys = list(self.hits)
        if self.dropdown is not None:
            keys.sort(key=lambda key: 0 if key.startswith('option:') else 1)
        hit = next((key for key in keys
                    if self.hits[key][0] <= x < self.hits[key][2] and
                    self.hits[key][1] <= y < self.hits[key][3]), None)
        if self.editing:
            self._finish_edit(self.fields[self.selected])
        if hit == 'config_tab':
            self.page, self.dropdown = 'config', None
        elif hit == 'info_tab':
            self.page, self.dropdown = 'info', None
        elif hit == 'fullscreen':
            self._toggle_fullscreen(glfw, window)
        elif hit == 'start':
            self._start_receiver()
        elif hit == 'start_stop':
            self._stop_receiver()
        elif hit == 'mode_button':
            field = next(field for field in self.fields
                         if field.label == 'Display upscaler')
            self._adjust_field(field, 1)
        elif hit and hit.startswith('option:') and self.dropdown is not None:
            index = int(hit.split(':', 1)[1])
            field = self.fields[self.dropdown]
            self._select_choice(field, field.options[index][1])
        elif hit and hit.startswith('row:') and self.page == 'config':
            index = int(hit.split(':', 1)[1])
            self.selected = index
            field = self.fields[index]
            if self.started and field.action is not None:
                self.notice = ('Receiver options are locked while receiving; '
                               'close and relaunch to change them.')
            elif not field.locked:
                if field.kind == 'bool':
                    self._adjust_field(field, 1)
                elif field.kind == 'choice':
                    self.dropdown = (None if self.dropdown == index else index)
                    self.dropdown_scroll = 0
                else:
                    self.editing = True
                    self.edit_buffer = '' if field.value is None else str(field.value)
            self.dirty = True
        elif self.dropdown is not None:
            self.dropdown = None
            self.dirty = True
        else:
            self.dirty = True

    def _on_key(self, glfw, window, key, _scancode, action, mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
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
        if key == glfw.KEY_F:
            self._toggle_fullscreen(glfw, window)
        elif key in (glfw.KEY_I, glfw.KEY_TAB):
            self._change_page()
        elif key == glfw.KEY_C:
            self.page = 'config'
            self.dropdown = None
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
                    self.dropdown = self.selected
                    self.dropdown_scroll = 0
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
                self.selected = max(0, min(len(self.fields)-1,
                                           self.selected+delta))
                top, visible = self._visible_fields(self.height)
                if self.selected < self.scroll:
                    self.scroll = self.selected
                elif self.selected >= self.scroll+visible:
                    self.scroll = self.selected-visible+1
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
        field.value = self.edit_buffer.strip()
        self.editing = False
        self.notice = f'{field.label} updated.'
        self.dirty = True

    def _on_scroll(self, _window, _xoffset, yoffset):
        delta = -1 if yoffset > 0 else 1
        if self.dropdown is not None:
            field = self.fields[self.dropdown]
            self.dropdown_scroll = max(
                0, min(max(0, len(field.options)-1),
                       self.dropdown_scroll+delta))
        elif self.page == 'config':
            _top, visible = self._visible_fields(self.height)
            self.scroll = max(0, min(max(0, len(self.fields)-visible),
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
        ui_texture = picture_texture = None
        picture_texture_size = None
        picture_texture_mode = None
        try:
            glfw.default_window_hints()
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
            glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
            window = glfw.create_window(960, 720, 'V7 Receiver · Setup', None, None)
            if not window:
                raise RuntimeError('GLFW could not create the V7 receiver window')
            glfw.make_context_current(window)
            self._window = window
            glfw.set_window_size_limits(window, 720, 480,
                                        glfw.DONT_CARE, glfw.DONT_CARE)
            glfw.swap_interval(1)
            context = moderngl.create_context(require=330)
            program = context.program(vertex_shader=VERTEX_SHADER,
                                      fragment_shader=FRAGMENT_SHADER)
            program['image'].value = 0
            vertex_array = context.vertex_array(program, [])
            glfw.set_key_callback(
                window, lambda w, k, s, a, m: self._on_key(glfw, w, k, s, a, m))
            glfw.set_char_callback(window, self._on_char)
            glfw.set_mouse_button_callback(
                window, lambda w, b, a, m: self._on_mouse(glfw, w, b, a, m))
            glfw.set_scroll_callback(window, self._on_scroll)

            while not glfw.window_should_close(window):
                glfw.wait_events_timeout(.04)
                self._process_output()
                self._poll_frame()
                self._poll_diagnostics()
                self._profile_ui_if_enabled()
                self._poll_receiver_lifecycle()
                now = time.monotonic()
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
                        ui_texture.size != window_size):
                    self.dirty = True
                picture_needs_draw = (self.picture_dirty and
                                      self.page == 'info')
                if self.dirty or picture_needs_draw:
                    if self.dirty:
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
                        if (self.page == 'info' and self.current_frame is not None
                                and self.latest_values_image is not None):
                            self.picture_dirty = True
                            picture_needs_draw = True

                    if picture_needs_draw:
                        if self.latest_values_image is not None:
                            pixels = np.ascontiguousarray(np.asarray(
                                self.latest_values_image, dtype=np.uint8))
                            picture_size = (pixels.shape[1], pixels.shape[0])
                            if (picture_texture is None or
                                    picture_texture_size != picture_size):
                                if picture_texture is not None:
                                    picture_texture.release()
                                picture_texture = context.texture(
                                    picture_size, 3, pixels.tobytes(), dtype='f1')
                                picture_texture_size = picture_size
                                picture_texture.repeat_x = False
                                picture_texture.repeat_y = False
                            else:
                                picture_texture.write(pixels.tobytes())
                        self.picture_dirty = False

                    if (picture_texture is not None and
                            picture_texture_mode != self.display_mode):
                        filtering = (moderngl.NEAREST
                                     if self.display_mode == 'nearest'
                                     else moderngl.LINEAR)
                        picture_texture.filter = (filtering, filtering)
                        picture_texture_mode = self.display_mode

                    context.viewport = (0, 0, *framebuffer_size)
                    context.clear(.03, .05, .07, 1.0)
                    ui_texture.use(location=0)
                    vertex_array.render(mode=moderngl.TRIANGLES, vertices=3)
                    if (self.page == 'info' and self.current_frame is not None
                            and picture_texture is not None):
                        aspect = self.v7_live.P.V7_ASPECT_RATIOS[
                            self.current_frame.aspect & 7]
                        picture_viewport = self._picture_viewport(
                            window_size, framebuffer_size, aspect)
                        if picture_viewport[2] and picture_viewport[3]:
                            context.viewport = picture_viewport
                            picture_texture.use(location=0)
                            vertex_array.render(
                                mode=moderngl.TRIANGLES, vertices=3)
                    glfw.swap_buffers(window)
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
            if ui_texture is not None:
                ui_texture.release()
            if picture_texture is not None:
                picture_texture.release()
            if vertex_array is not None:
                vertex_array.release()
            if program is not None:
                program.release()
            if context is not None:
                context.release()
            if window is not None:
                glfw.destroy_window(window)
            glfw.terminate()


def main(v7_live_module=None):
    if v7_live_module is None:
        from tools import v7_live as v7_live_module
    v7_live = v7_live_module
    root_parser, receive_parser = _receive_parser(v7_live)
    devices, device_error = _input_devices()
    ReceiverGui(v7_live, root_parser, receive_parser, devices,
                device_error).run()


if __name__ == '__main__':
    main()
