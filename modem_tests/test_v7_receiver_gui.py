"""Receiver GUI configuration maps every receive CLI setting safely."""
import io
import json
import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from animation_modem import v7
from tools import v7_live
from tools.v7_gl_viewer import (DISPLAY_MODES, FLOAT_FRAGMENT_SHADER,
                                DCT_RECONSTRUCTION_MODES,
                                FRAGMENT_SHADER, VERTEX_SHADER)
from tools.v7_receiver_gui import (ROOT, OptionField, ReceiverGui, _make_fields,
                                   FULLSCREEN_TOOLBAR_EDGE,
                                   FULLSCREEN_TOOLBAR_HIDE_SECONDS,
                                   HIDDEN_DECODE_OPTIONS,
                                   QueueWriter,
                                   _create_graphics_context,
                                   _logical_rect_to_framebuffer,
                                   _shader_for_context,
                                   _scissors_outside_viewport,
                                   _receive_parser)


class MouseStub:
    MOUSE_BUTTON_LEFT = 0
    PRESS = 1

    def __init__(self, position):
        self.position = position

    def get_cursor_pos(self, _window):
        return self.position


class KeyStub:
    PRESS = 1
    REPEAT = 2
    KEY_F = 70
    KEY_I = 73
    KEY_C = 67
    KEY_TAB = 258
    KEY_ESCAPE = 256
    KEY_ENTER = 257
    KEY_DOWN = 264
    KEY_UP = 265
    KEY_KP_ENTER = 335
    KEY_P = 80
    MOD_CONTROL = 2


class FullscreenKeyStub(KeyStub):
    DONT_CARE = 0

    def get_primary_monitor(self):
        return object()

    def set_window_monitor(self, *_args):
        pass


class GraphicsGlfwStub:
    CLIENT_API = 1
    OPENGL_ES_API = 2
    OPENGL_API = 3
    CONTEXT_CREATION_API = 4
    EGL_CONTEXT_API = 5
    CONTEXT_VERSION_MAJOR = 6
    CONTEXT_VERSION_MINOR = 7
    OPENGL_PROFILE = 8
    OPENGL_CORE_PROFILE = 9
    OPENGL_FORWARD_COMPAT = 10
    RESIZABLE = 11
    TRUE = 1

    def __init__(self, fail_desktop=False):
        self.fail_desktop = fail_desktop
        self.hints = {}
        self.apis = []
        self.window_sizes = []
        self.focused_windows = []

    def default_window_hints(self):
        self.hints = {}

    def window_hint(self, hint, value):
        self.hints[hint] = value

    def create_window(self, *_args):
        self.window_sizes.append(_args[:2])
        api = self.hints[self.CLIENT_API]
        self.apis.append(api)
        if self.fail_desktop and api == self.OPENGL_API:
            return None
        return object()

    def get_error(self):
        return 65543, b'EGL: Failed to create context: Arguments are inconsistent'

    def make_context_current(self, _window):
        pass

    def destroy_window(self, _window):
        pass

    def focus_window(self, window):
        self.focused_windows.append(window)


class GraphicsModernGlStub:
    def __init__(self):
        self.options = []

    def create_context(self, **options):
        self.options.append(options)
        return SimpleNamespace(version_code=300, release=Mock())


class ReceiverGuiGraphicsContextTests(unittest.TestCase):
    def test_setup_window_starts_within_a_small_display_mode(self):
        glfw = GraphicsGlfwStub()
        glfw.get_primary_monitor = lambda: object()
        glfw.get_video_mode = lambda _monitor: SimpleNamespace(
            size=SimpleNamespace(width=800, height=600))
        moderngl = GraphicsModernGlStub()

        window, _context, _use_gles = _create_graphics_context(
            glfw, moderngl, wayland=False)

        self.assertEqual(glfw.window_sizes, [(768, 520)])
        self.assertEqual(glfw.focused_windows, [window])

    def test_wayland_prefers_gles_egl(self):
        glfw = GraphicsGlfwStub()
        moderngl = GraphicsModernGlStub()

        _window, _context, use_gles = _create_graphics_context(
            glfw, moderngl, wayland=True)

        self.assertTrue(use_gles)
        self.assertEqual(glfw.apis, [glfw.OPENGL_ES_API])
        self.assertEqual(moderngl.options, [{'require': 300}])

    def test_desktop_context_failure_falls_back_to_gles(self):
        glfw = GraphicsGlfwStub(fail_desktop=True)
        moderngl = GraphicsModernGlStub()

        _window, _context, use_gles = _create_graphics_context(
            glfw, moderngl, wayland=False)

        self.assertTrue(use_gles)
        self.assertEqual(glfw.apis,
                         [glfw.OPENGL_API, glfw.OPENGL_ES_API])
        self.assertEqual(moderngl.options, [{'require': 300}])

    def test_gles_shader_variant_uses_es_300_and_precision(self):
        for shader in (VERTEX_SHADER, FRAGMENT_SHADER,
                       FLOAT_FRAGMENT_SHADER):
            with self.subTest(shader=shader[:32]):
                converted = _shader_for_context(shader, use_gles=True)
                self.assertTrue(converted.startswith(
                    '#version 300 es\nprecision highp float;'))
                self.assertNotIn('#version 330', converted)
        self.assertEqual(_shader_for_context(VERTEX_SHADER, use_gles=False),
                         VERTEX_SHADER)


class ReceiverGuiOptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root_parser, cls.receive_parser = _receive_parser(v7_live)

    def test_every_receive_option_is_represented(self):
        fields = _make_fields(self.receive_parser, ())
        represented = {field.action.dest for field in fields
                       if field.action is not None}
        expected = {action.dest for action in self.receive_parser._actions
                    if action.dest not in ('help', 'mode') and
                    action.dest not in HIDDEN_DECODE_OPTIONS}
        self.assertEqual(represented, expected)

    def test_builds_headless_receiver_args_without_opening_audio(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser,
                          (('test input device', 3),))
        device = next(field for field in gui.fields
                      if field.action is not None and
                      field.action.dest == 'device')
        device.value = 3
        args = gui._build_arguments()
        self.assertEqual(args.mode, 'receive')
        self.assertEqual(args.device, 3)
        self.assertTrue(args.headless)
        self.assertTrue(args.diagnostics)
        self.assertFalse(args.log)
        self.assertTrue(args.no_log)
        self.assertIsNone(args.save_dir)
        self.assertEqual(args.mono_video_side, 'auto')

    def test_start_routes_audio_diagnostics_to_launch_terminal(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser,
                          (('test input device', 3),),
                          audio_output_choices=(('test output device', 4),))
        output = next(field for field in gui.fields
                      if field.action is not None and
                      field.action.dest == 'audio_output_device')
        output.value = 4
        diagnostics = next(field for field in gui.fields
                           if field.dest == 'show_diagnostics')
        diagnostics.value = True
        gui.v7_live = SimpleNamespace(run_receive=Mock())
        gui._prepare_input_device = Mock(return_value=True)
        terminal = io.StringIO()

        with patch('tools.v7_receiver_gui.sys.stdout', terminal):
            gui._start_receiver()
            gui.receiver_thread.join(timeout=2)

        args = gui.v7_live.run_receive.call_args.args[0]
        self.assertTrue(args.runtime_options.snapshot()['audio_diagnostics'])
        self.assertIs(args.audio_diagnostics_stream, terminal)
        gui.receiver_stop.set()

    def test_audio_diagnostics_are_off_until_the_details_button_is_enabled(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser,
                          (('test input device', 3),),
                          audio_output_choices=(('test output device', 4),))
        output = next(field for field in gui.fields
                      if field.action is not None and
                      field.action.dest == 'audio_output_device')
        output.value = 4
        gui.v7_live = SimpleNamespace(run_receive=Mock())
        gui._prepare_input_device = Mock(return_value=True)
        terminal = io.StringIO()

        with patch('tools.v7_receiver_gui.sys.stdout', terminal):
            gui._start_receiver()
            gui.receiver_thread.join(timeout=2)

        args = gui.v7_live.run_receive.call_args.args[0]
        self.assertFalse(args.runtime_options.snapshot()['audio_diagnostics'])
        self.assertIs(args.audio_diagnostics_stream, terminal)
        gui._set_show_diagnostics(True)
        self.assertTrue(args.runtime_options.snapshot()['audio_diagnostics'])
        gui._set_show_diagnostics(False)
        self.assertFalse(args.runtime_options.snapshot()['audio_diagnostics'])
        gui.receiver_stop.set()

    def test_cli_passthrough_volume_defaults_to_vlc_level_and_is_bounded(self):
        args = self.root_parser.parse_args([
            'receive', '--device', '3', '--audio-output-device', '4'])
        self.assertEqual(args.audio_volume, 1.0)
        args = self.root_parser.parse_args([
            'receive', '--device', '3', '--audio-volume', '.35'])
        self.assertEqual(args.audio_volume, .35)
        with self.assertRaises(SystemExit):
            self.root_parser.parse_args([
                'receive', '--device', '3', '--audio-volume', '1.1'])

    def test_receiver_output_wakes_the_event_driven_window(self):
        output = queue.Queue()
        notify = Mock()
        writer = QueueWriter(output, notify)

        writer.write('receiver ready')
        self.assertEqual(notify.call_count, 0)
        writer.write('\nstatus updated\n')

        self.assertEqual(output.get_nowait(), 'receiver ready')
        self.assertEqual(output.get_nowait(), 'status updated')
        self.assertEqual(notify.call_count, 1)

    def test_start_requires_an_explicit_input_device(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
        with self.assertRaisesRegex(ValueError, 'Select an input audio device'):
            gui._build_arguments()

    def test_enter_on_empty_device_list_closes_dropdown_without_crashing(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
        gui.dropdown = next(index for index, field in enumerate(gui.fields)
                            if field.dest == 'device')

        gui._on_key(KeyStub, None, KeyStub.KEY_ENTER, 0, KeyStub.PRESS, 0)

        self.assertIsNone(gui.dropdown)
        self.assertIn('No choices', gui.notice)

    def test_empty_input_picker_does_not_open_a_blank_menu(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
        field_index = next(index for index, field in enumerate(gui.fields)
                           if field.dest == 'device')
        gui._canvas((960, 720))
        left, top, right, bottom = gui.hits[f'row:{field_index}']

        with patch('tools.v7_receiver_gui._input_devices',
                   return_value=((), 'No audio inputs attached.')):
            gui._on_mouse(
                MouseStub(((left+right)/2, (top+bottom)/2)), None, 0, 1, 0)

        self.assertIsNone(gui.dropdown)
        self.assertEqual(gui.notice, 'No audio inputs attached.')

    def test_numpad_enter_selects_the_highlighted_device(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser,
                          (('first input', 3), ('second input', 9)))
        field_index = next(index for index, field in enumerate(gui.fields)
                           if field.dest == 'device')
        gui._canvas((960, 720))
        left, top, right, bottom = gui.hits[f'row:{field_index}']

        choices = (('first input', 3), ('second input', 9))
        identities = {
            3: {'name': 'first input', 'hostapi': 'test'},
            9: {'name': 'second input', 'hostapi': 'test'},
        }
        with patch('tools.v7_receiver_gui._input_devices',
                   return_value=(choices, '')), \
                patch('tools.v7_receiver_gui._device_identity',
                      side_effect=lambda index, _kind: identities[index]), \
                patch.object(gui, '_persist_preferences'):
            gui._on_mouse(
                MouseStub(((left+right)/2, (top+bottom)/2)), None, 0, 1, 0)
            gui._on_key(KeyStub, None, KeyStub.KEY_DOWN, 0,
                        KeyStub.PRESS, 0)
            gui._on_key(KeyStub, None, KeyStub.KEY_KP_ENTER, 0,
                        KeyStub.PRESS, 0)

        self.assertEqual(gui.fields[field_index].value, 9)
        self.assertIsNone(gui.dropdown)

    def test_zero_vertical_scroll_does_not_move_receiver_dropdown(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser,
                          (('test input', 3),))
        field_index = next(index for index, field in enumerate(gui.fields)
                           if field.label == 'Display upscaler')
        gui.dropdown = field_index
        gui.dropdown_scroll = 0

        gui._on_scroll(None, 1, 0)
        self.assertEqual(gui.dropdown_scroll, 0)
        gui._on_scroll(None, 0, -0.5)

        self.assertEqual(gui.dropdown_scroll, 1)

    def test_wire_profile_and_decoder_tuners_are_not_gui_options(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser,
                          (('test input device', 3),))
        represented = {field.dest for field in gui.fields
                       if field.action is not None}
        self.assertTrue(HIDDEN_DECODE_OPTIONS.isdisjoint(represented))
        self.assertNotIn('experimental_mono_colour', represented)
        args = gui._build_arguments()
        self.assertFalse(args.experimental_mono_fold)
        self.assertFalse(args.experimental_mono_colour)
        self.assertIsNone(args.experimental_fold)
        self.assertEqual(args.frame_boundary, 'eof')
        self.assertEqual(args.pilot_timing, 'tone-seeded')
        receiver_help = self.receive_parser.format_help()
        for hidden in ('--pilot-timing', '--frame-boundary',
                       '--pulse-timing', '--tone-equalization',
                       '--experimental-mono-fold',
                       '--experimental-mono-colour'):
            self.assertNotIn(hidden, receiver_help)

    def test_gui_uses_packet_profile_dispatch_without_a_startup_latch(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser,
                          (('test input device', 3),))
        args = gui._build_arguments()
        args.no_log = True
        args.runtime_options = SimpleNamespace(
            snapshot=lambda: {'audio_input_identity': None})

        fold, profiles = Mock(), Mock()
        with (patch.object(v7_live, '_experimental_fold', return_value=fold),
              patch.object(v7_live, '_make_auto_profile_decoder',
                           return_value=profiles),
              patch.object(v7_live, '_detect_mono_fold_side') as detect,
              patch.object(v7_live, '_run_receive', return_value='decoded') as run):
            self.assertEqual(v7_live.run_receive(args), 'decoded')

        detect.assert_not_called()
        profiles.install.assert_called_once_with()
        profiles.uninstall.assert_called_once_with()
        self.assertIs(run.call_args.args[1], fold)
        self.assertIs(run.call_args.kwargs['adaptive_profile'], profiles)
        self.assertFalse(args.experimental_mono_fold)
        self.assertFalse(args.experimental_mono_colour)

    def test_first_available_input_is_preselected(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser,
                          (('first input', 4), ('second input', 9)))
        device = next(field for field in gui.fields
                      if field.dest == 'device')
        self.assertEqual(device.value, 4)
        self.assertEqual(gui._build_arguments().device, 4)

    def test_windowed_setup_tab_click_returns_from_live_page(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
        gui.page = 'info'
        gui._canvas((960, 720))
        left, top, right, bottom = gui.hits['config_tab']

        gui._on_mouse(MouseStub(((left+right)/2, (top+bottom)/2)),
                      None, MouseStub.MOUSE_BUTTON_LEFT, MouseStub.PRESS, 0)

        self.assertEqual(gui.page, 'config')
        self.assertTrue(gui.dirty)

    def test_input_picker_refreshes_hotplugged_devices_when_opened(self):
        with patch('tools.v7_receiver_gui._load_preferences', return_value={}):
            gui = ReceiverGui(
                self, self.root_parser, self.receive_parser, (),
                preference_path='unused-preferences.json')
        field_index = next(index for index, field in enumerate(gui.fields)
                           if field.dest == 'device')
        gui._canvas((960, 720))
        left, top, right, bottom = gui.hits[f'row:{field_index}']
        choices = (('newly connected input', 12),)

        with patch('tools.v7_receiver_gui._input_devices',
                   return_value=(choices, '')), \
                patch('tools.v7_receiver_gui._device_identity',
                      return_value={'name': 'newly connected input',
                                    'hostapi': 'test'}), \
                patch.object(gui, '_persist_preferences'):
            gui._on_mouse(
                MouseStub(((left+right)/2, (top+bottom)/2)), None, 0, 1, 0)

            self.assertEqual(gui.fields[field_index].options, choices)
            self.assertEqual(gui.dropdown, field_index)
            gui._on_key(KeyStub, None, KeyStub.KEY_ENTER, 0,
                        KeyStub.PRESS, 0)

        self.assertEqual(gui.fields[field_index].value, 12)

    def test_background_device_refresh_updates_input_and_output_pickers(self):
        new_inputs = (('newly connected input · 2 in · 48 kHz', 12),)
        new_outputs = (('newly connected output · 2 out · 48 kHz', 14),)
        with patch('tools.v7_receiver_gui._load_preferences', return_value={}):
            gui = ReceiverGui(
                self, self.root_parser, self.receive_parser, (),
                audio_output_choices=(),
                preference_path='unused-preferences.json')
        gui.device_updates.put((new_inputs, '', new_outputs, ''))

        gui._process_device_updates()

        input_field = next(field for field in gui.fields
                           if field.dest == 'device')
        output_field = next(field for field in gui.fields
                            if field.dest == 'audio_output_device')
        self.assertEqual(input_field.options, new_inputs)
        self.assertEqual(input_field.value, 12)
        self.assertEqual(output_field.options,
                         (('Off · passthrough disabled', None),) + new_outputs)
        self.assertIsNone(output_field.value)

    def test_selected_cli_options_reach_receiver_arguments(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser,
                          (('test input device', 3),))
        values = {field.dest: field for field in gui.fields
                  if field.action is not None}
        values['device'].value = 3
        values['show_diagnostics'].value = False
        args = gui._build_arguments()
        self.assertEqual(args.direction, 'auto')
        self.assertEqual(args.decode_history, 1)
        self.assertEqual(args.pilot_timing, 'tone-seeded')
        self.assertFalse(args.show_diagnostics)
        self.assertFalse(args.no_tail_memory)

    def test_aspect_fold_options_are_in_wire_and_default_to_auto_chroma(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser,
                          (('test input device', 3),))
        fields = {field.dest: field for field in gui.fields
                  if field.action is not None}
        sections = dict(gui._config_sections())
        live = {gui.fields[index].dest for index in sections['Live controls']}
        wire = {gui.fields[index].dest for index in sections['Wire']}
        self.assertNotIn('aspect_layout', live)
        self.assertEqual(wire, {'aspect_layout', 'aspect_tail'})
        self.assertIn(('Luma · 96 luma every packet', 'luma'),
                      fields['aspect_tail'].options)
        fields['device'].value = 3
        args = gui._build_arguments()
        self.assertEqual((args.aspect_layout, args.aspect_tail), ('auto', 'fixed'))
        fields['aspect_layout'].value = '4:3'
        fields['aspect_tail'].value = 'luma'
        args = gui._build_arguments()
        self.assertEqual((args.aspect_layout, args.aspect_tail), ('4:3', 'luma'))

    def test_display_grain_is_a_basic_display_choice_off_by_default(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser,
                          (('test input device', 3),))
        field = next(item for item in gui.fields
                     if item.label == 'Display grain')
        self.assertEqual(field.value, 'off')
        self.assertEqual(gui.grain_mode, 'off')
        basic = {gui.fields[index].label
                 for index in gui._config_field_indexes()}
        self.assertIn('Display grain', basic)
        self.assertEqual([value for _label, value in field.options],
                         ['off', 'flat'])
        gui._adjust_field(field, 1)
        self.assertEqual(gui.grain_mode, 'flat')
        self.assertTrue(gui.picture_dirty)

    def test_pixel_display_is_a_toggle_that_locks_the_other_display_choices(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser,
                          (('test input device', 3),))
        by_label = {field.label: field for field in gui.fields}
        pixel = by_label['Pixel display']
        smooth = [by_label[label] for label in (
            'Display upscaler', 'DCT reconstruction', 'Display grain',
            'Edge reconstruction', 'Edge strength')]
        self.assertEqual((pixel.kind, pixel.value, gui.pixel_display),
                         ('bool', False, False))
        self.assertIn(gui.fields.index(pixel), gui._config_field_indexes())
        self.assertLess(gui.fields.index(pixel), gui.fields.index(smooth[0]))
        before = [field.value for field in smooth]

        gui.picture_dirty = False
        gui._adjust_field(pixel, 1)
        self.assertTrue(gui.pixel_display and gui.picture_dirty)
        self.assertEqual((gui.dct_reconstruction, gui.display_mode,
                          gui.edge_mode, gui.grain_mode),
                         ('pixel', 'nearest', 'off', 'off'))
        self.assertTrue(all(field.locked for field in smooth))
        self.assertEqual(gui._field_value_label(smooth[0]),
                         'Off while Pixel display is on')
        # Locked: neither stepping nor choosing changes them.
        for field in smooth:
            gui._adjust_field(field, 1)
        gui._select_choice(smooth[1], '8x')
        self.assertEqual([field.value for field in smooth], before)
        self.assertEqual(gui.dct_reconstruction, 'pixel')
        # Starting the receiver keeps the pixel display.
        by_label_dest = {field.dest: field for field in gui.fields}
        by_label_dest['device'].value = 3
        gui._build_arguments()
        self.assertEqual((gui.dct_reconstruction, gui.display_mode),
                         ('pixel', 'nearest'))

        gui._adjust_field(pixel, 1)
        self.assertFalse(gui.pixel_display)
        self.assertFalse(any(field.locked for field in smooth))
        self.assertEqual((gui.display_mode, gui.dct_reconstruction,
                          gui.edge_mode, gui.grain_mode),
                         ('bicubic', '4x', 'on', 'off'))
        gui._select_choice(smooth[1], '8x')
        self.assertEqual(gui.dct_reconstruction, '8x')

    def test_edge_reconstruction_is_a_basic_display_choice_on_by_default(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser,
                          (('test input device', 3),))
        field = next(item for item in gui.fields
                     if item.label == 'Edge reconstruction')
        self.assertEqual(field.value, 'on')
        self.assertEqual(gui.edge_mode, 'on')
        basic = {gui.fields[index].label
                 for index in gui._config_field_indexes()}
        self.assertIn('Edge reconstruction', basic)
        gui._adjust_field(field, 1)
        self.assertEqual(gui.edge_mode, 'high')
        self.assertTrue(gui.picture_dirty)
        gui._adjust_field(field, 1)
        self.assertEqual(gui.edge_mode, 'off')
        strength = next(item for item in gui.fields
                        if item.label == 'Edge strength')
        self.assertEqual((strength.value, gui.edge_strength), (.75, .75))
        self.assertIn('Edge strength', {gui.fields[index].label for index
                                        in gui._config_field_indexes()})
        gui._adjust_field(strength, 1)
        self.assertEqual(gui.edge_strength, .5)

    def test_save_directory_is_used_only_when_explicit(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser,
                          (('test input device', 3),))
        values = {field.dest: field for field in gui.fields
                  if field.action is not None}
        values['device'].value = 3
        values['save_dir'].value = str(ROOT/'tmp'/'receiver-captures')
        args = gui._build_arguments()
        self.assertEqual(args.save_dir, ROOT/'tmp'/'receiver-captures')

    def test_stopped_receiver_clears_stopping_notice_after_thread_exit(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
        thread = threading.Thread(target=lambda: None)
        thread.start()
        thread.join()
        gui.receiver_thread = thread
        gui.receiver_stop = threading.Event()
        gui.receiver_stop.set()
        gui.ever_started = True
        gui.notice = 'Stopping receiver and closing its audio stream…'

        gui._poll_receiver_lifecycle()

        self.assertEqual(gui.notice, 'Receiver stopped.')
        self.assertTrue(gui.dirty)

    def test_display_upscaler_remains_changeable_during_receive(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
        gui.started = True
        gui.page = 'info'
        gui._canvas((960, 720))
        x1, y1, x2, y2 = gui.hits['mode_button']
        gui._on_mouse(MouseStub(((x1+x2)/2, (y1+y2)/2)), None, 0, 1, 0)
        self.assertTrue(gui.display_menu_open)
        gui._canvas((720, 480))
        display_mode_hits = [key for key in gui.hits
                             if key.startswith('display_mode:')]
        self.assertEqual(
            tuple(key.split(':', 1)[1] for key in display_mode_hits),
            DISPLAY_MODES)
        self.assertLessEqual(max(gui.hits[key][3]
                                 for key in display_mode_hits), 480)
        x1, y1, x2, y2 = gui.hits['display_mode:ewa-jinc']
        gui._on_mouse(MouseStub(((x1+x2)/2, (y1+y2)/2)), None, 0, 1, 0)
        self.assertEqual(gui.display_mode, 'ewa-jinc')
        self.assertIn('EWA Jinc', gui.notice)
        self.assertFalse(gui.display_menu_open)
        self.assertTrue(gui.picture_dirty)

        field = next(field for field in gui.fields
                     if field.label == 'Display upscaler')
        self.assertEqual(tuple(value for _label, value in field.options),
                         DISPLAY_MODES)
        gui._canvas((720, 480))
        x1, y1, x2, y2 = gui.hits['mode_button']
        gui._on_mouse(MouseStub(((x1+x2)/2, (y1+y2)/2)), None, 0, 1, 0)
        gui._on_key(KeyStub, None, KeyStub.KEY_DOWN, 0, KeyStub.PRESS, 0)
        self.assertEqual(gui.display_menu_index, 0)
        gui._on_key(KeyStub, None, KeyStub.KEY_ENTER, 0, KeyStub.PRESS, 0)
        self.assertEqual(gui.display_mode, 'nearest')
        self.assertFalse(gui.display_menu_open)

        device = next(field for field in gui.fields
                      if field.dest == 'device')
        before = device.value
        gui._adjust_field(device, 1)
        self.assertEqual(device.value, before)
        self.assertIn('locked while receiving', gui.notice)

    def test_display_menu_bounds_scale_to_a_hidpi_scissor(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
        gui.page = 'info'
        gui.display_menu_open = True
        gui._canvas((960, 720))

        bounds = gui._display_menu_bounds(960)
        self.assertIsNotNone(bounds)
        first_mode = gui.hits['display_mode:'+DISPLAY_MODES[0]]
        self.assertGreaterEqual(first_mode[1], bounds[1])
        self.assertLessEqual(first_mode[3], bounds[3])
        self.assertEqual(
            _logical_rect_to_framebuffer(
                bounds, (960, 720), (1920, 1440)),
            (bounds[0]*2, 1440-bounds[3]*2,
             (bounds[2]-bounds[0])*2, (bounds[3]-bounds[1])*2))

    def test_picture_viewport_complement_has_four_nonoverlapping_regions(self):
        self.assertEqual(
            _scissors_outside_viewport((8, 8), (2, 2, 4, 4)),
            ((0, 0, 8, 2), (0, 6, 8, 2),
             (0, 2, 2, 4), (6, 2, 2, 4)))

    def test_picture_viewport_uses_framebuffer_pixels_on_hidpi(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
        self.assertEqual(
            gui._picture_viewport((960, 720), (1920, 1440), 4/3),
            (144, 92, 1632, 1224))

    def test_setup_lists_every_option_in_sections_but_never_hidden_ones(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
        self.assertFalse(hasattr(gui, 'advanced_options'))
        sections = dict(gui._config_sections())
        live = {gui.fields[index].dest for index in sections['Live controls']}
        self.assertIn('device', {gui.fields[index].dest
                                 for index in sections['Input']})
        self.assertIn('DCT reconstruction', live)
        self.assertNotIn('aspect_tail', live)

        shown = {gui.fields[index].dest
                 for index in gui._config_field_indexes()}
        self.assertTrue(HIDDEN_DECODE_OPTIONS.isdisjoint(shown))
        self.assertNotIn('experimental_fold', shown)
        self.assertNotIn('direction', shown)
        self.assertNotIn('decode_history', shown)
        self.assertIn('DCT reconstruction', shown)
        self.assertIn('aspect_tail', shown)
        self.assertEqual(sorted(gui._config_field_indexes()),
                         list(range(len(gui.fields))))

    def test_dct_reconstruction_is_a_display_only_basic_choice(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser,
                          (('test input device', 3),))
        field = next(field for field in gui.fields
                     if field.label == 'DCT reconstruction')
        self.assertIn(gui.fields.index(field), gui._config_field_indexes())
        # Recommended display: 4x DCT reconstruction, shader bicubic.
        self.assertEqual(gui.display_mode, 'bicubic')
        self.assertEqual(field.value, '4x')
        self.assertEqual(gui.dct_reconstruction, '4x')
        # Pixel is its own toggle, not a reconstruction choice.
        self.assertEqual(tuple(value for _label, value in field.options),
                         tuple(mode for mode in DCT_RECONSTRUCTION_MODES
                               if mode != 'pixel'))

        gui.picture_dirty = False
        gui._select_choice(field, 'off')
        self.assertEqual(gui.dct_reconstruction, 'off')
        self.assertTrue(gui.picture_dirty)
        self.assertEqual(gui.notice, 'DCT reconstruction: Off')

        gui.picture_dirty = False
        gui._select_choice(field, '4x')

        self.assertEqual(gui.dct_reconstruction, '4x')
        self.assertTrue(gui.picture_dirty)
        self.assertEqual(gui.notice, 'DCT reconstruction: 4× · recommended')

        gui._build_arguments()
        self.assertEqual(gui.dct_reconstruction, '4x')

    def test_image_only_view_restores_the_live_page(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
        gui.page = 'info'

        gui._set_image_only(True)
        self.assertTrue(gui.image_only)
        gui._set_image_only(False)

        self.assertFalse(gui.image_only)
        self.assertEqual(gui.page, 'info')

    def test_fullscreen_toolbar_hides_and_reappears_at_top_edge(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
        gui.fullscreen = True
        gui.last_ui_activity = 10.0

        gui._update_toolbar_visibility(
            10.0+FULLSCREEN_TOOLBAR_HIDE_SECONDS, FULLSCREEN_TOOLBAR_EDGE+1)
        self.assertFalse(gui.toolbar_visible)

        gui._update_toolbar_visibility(
            11.0, FULLSCREEN_TOOLBAR_EDGE)
        self.assertTrue(gui.toolbar_visible)

    def test_entering_fullscreen_starts_with_the_hud_hidden(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
        focused = []
        glfw = SimpleNamespace(
            DONT_CARE=0,
            get_primary_monitor=lambda: object(),
            get_window_pos=lambda _window: (0, 0),
            get_window_size=lambda _window: (960, 720),
            get_video_mode=lambda _monitor: SimpleNamespace(
                size=SimpleNamespace(width=1920, height=1080),
                refresh_rate=60),
            set_window_monitor=lambda *_args: None,
            focus_window=lambda window: focused.append(window))
        window = object()

        gui._toggle_fullscreen(glfw, window)

        self.assertTrue(gui.fullscreen)
        self.assertFalse(gui.toolbar_visible)
        self.assertEqual(focused, [window])

    def test_hidden_fullscreen_toolbar_is_removed_from_live_canvas(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
        gui.page = 'info'
        gui.fullscreen = True
        gui.toolbar_visible = False
        next(field for field in gui.fields
             if field.dest == 'show_diagnostics').value = True

        canvas = Image.fromarray(gui._canvas((960, 720))).convert('RGB')

        self.assertNotIn('fullscreen', gui.hits)
        self.assertTrue(gui._diagnostics_visible())
        self.assertEqual(gui._picture_box((960, 720)), (0, 0, 960, 432))
        self.assertEqual(canvas.getpixel((0, 0)), (0, 0, 0))
        self.assertNotEqual(canvas.getpixel((0, 710)), (0, 0, 0))

    def test_fullscreen_without_info_is_black_image_only(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
        gui.page = 'info'
        gui.fullscreen = True
        gui.toolbar_visible = True

        canvas = Image.fromarray(gui._canvas((960, 720))).convert('RGB')

        self.assertTrue(gui._video_fullscreen())
        self.assertIsNone(canvas.getbbox())
        self.assertEqual(gui._picture_box((960, 720)), (0, 0, 960, 720))

    def test_sync_warning_clear_marks_hidden_info_view_dirty(self):
        meter = {'decoded': 4, 'input_fps': 12.0, 'sync_warning': True}
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
        gui.page = 'info'
        gui.last_diagnostics_poll = 0.0
        gui.live_meter = dict(meter)
        gui.live_sync_warning = True
        gui.dirty = False
        gui.v7_live = SimpleNamespace(RECEIVER_GUI_STATUS={'meter': meter})
        meter['sync_warning'] = False

        with patch('tools.v7_receiver_gui.time.monotonic', return_value=1.0):
            gui._poll_diagnostics()

        self.assertTrue(gui.dirty)
        self.assertFalse(gui.live_meter['sync_warning'])
        self.assertFalse(gui.live_sync_warning)

    def test_sync_warning_clears_before_footer_refresh_interval(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
        gui.page = 'info'
        gui.last_diagnostics_poll = 0.8
        gui.live_meter = {'decoded': 4, 'input_fps': 12.0,
                          'sync_warning': True}
        gui.live_sync_warning = True
        gui.dirty = False
        gui.v7_live = SimpleNamespace(RECEIVER_GUI_STATUS={
            'meter': {'decoded': 5, 'input_fps': 12.0,
                      'sync_warning': False}})

        with patch('tools.v7_receiver_gui.time.monotonic', return_value=1.0):
            gui._poll_diagnostics()

        self.assertFalse(gui.live_sync_warning)
        self.assertFalse(gui.live_meter['sync_warning'])
        self.assertTrue(gui.dirty)

    def test_sync_warning_clears_even_if_diagnostics_provider_fails(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
        gui.page = 'info'
        gui.last_diagnostics_poll = 0.0
        gui.live_sync_warning = True
        gui.live_meter = {'sync_warning': True}
        gui.dirty = False
        gui.v7_live = SimpleNamespace(RECEIVER_GUI_STATUS={
            'meter': {'sync_warning': False},
            'diagnostics': Mock(side_effect=RuntimeError('temporary status race')),
        })
        next(field for field in gui.fields
             if field.dest == 'show_diagnostics').value = True

        with patch('tools.v7_receiver_gui.time.monotonic', return_value=1.0):
            gui._poll_diagnostics()

        self.assertTrue(gui.dirty)
        self.assertFalse(gui.live_sync_warning)
        self.assertFalse(gui.live_meter['sync_warning'])

    def test_c_returns_directly_to_setup_from_image_only(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
        gui.image_only = True
        gui.image_only_previous_page = 'info'
        gui.fullscreen = True

        gui._on_key(FullscreenKeyStub(), object(), KeyStub.KEY_C, 0,
                    KeyStub.PRESS, 0)

        self.assertFalse(gui.image_only)
        self.assertEqual(gui.page, 'config')
        self.assertFalse(gui.fullscreen)
        self.assertTrue(gui.toolbar_visible)

    def test_c_returns_directly_to_setup_from_fullscreen_live_view(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
        gui.page = 'info'
        gui.fullscreen = True
        gui.toolbar_visible = False

        gui._on_key(FullscreenKeyStub(), object(), KeyStub.KEY_C, 0,
                    KeyStub.PRESS, 0)
        gui._update_toolbar_visibility(100.0, 200.0)

        self.assertEqual(gui.page, 'config')
        self.assertTrue(gui.toolbar_visible)
        self.assertFalse(gui.fullscreen)

    def test_compact_toolbar_controls_fit_a_lower_resolution_canvas(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
        gui.page = 'info'

        canvas = Image.fromarray(gui._canvas((640, 480)))
        controls = [gui.hits[key] for key in (
            'config_tab', 'info_tab', 'start_stop', 'mode_button',
            'details_button', 'image_only', 'fullscreen')]

        self.assertEqual(canvas.size, (640, 480))
        self.assertTrue(all(left[2] <= right[0]
                            for left, right in zip(controls, controls[1:])))

    def test_compact_setup_uses_space_above_help_and_footer(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())

        gui._canvas((520, 400))

        visible_rows = sum(key.startswith('row:') for key in gui.hits)
        self.assertGreaterEqual(visible_rows, 6)
        footer_top = 400-gui._footer_height(520)
        for key, rect in gui.hits.items():
            if key.startswith('row:'):
                self.assertLessEqual(rect[3], footer_top)

    def test_compact_info_panel_gets_more_height_for_readability(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
        gui.page = 'info'
        gui.fullscreen = True
        next(field for field in gui.fields
             if field.dest == 'show_diagnostics').value = True

        self.assertEqual(gui._picture_box((640, 480)), (0, 0, 640, 216))

    def test_fullscreen_i_reveals_and_toggles_diagnostics(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
        gui.fullscreen = True
        gui.toolbar_visible = False
        gui.page = 'info'

        gui._on_key(KeyStub(), object(), KeyStub.KEY_I, 0, KeyStub.PRESS, 0)

        field = next(field for field in gui.fields
                     if field.dest == 'show_diagnostics')
        self.assertFalse(gui.toolbar_visible)
        self.assertTrue(field.value)
        self.assertTrue(gui._diagnostics_visible())

    def test_image_only_does_not_change_fullscreen(self):
        # Image only hides the interface; fullscreen is a separate switch.
        for fullscreen in (False, True):
            with self.subTest(fullscreen=fullscreen):
                gui = ReceiverGui(self, self.root_parser, self.receive_parser,
                                  ())
                gui.page = 'info'
                gui.fullscreen = fullscreen
                gui._glfw = SimpleNamespace(
                    CURSOR=1, CURSOR_NORMAL=0, CURSOR_HIDDEN=2,
                    set_input_mode=lambda *_args: None)
                gui._window = object()
                toggles = []
                gui._toggle_fullscreen = lambda *_args: toggles.append(True)
                gui._set_image_only(True)
                self.assertTrue(gui.image_only)
                self.assertEqual(gui.fullscreen, fullscreen)
                gui._set_image_only(False)
                self.assertFalse(gui.image_only)
                self.assertEqual(gui.fullscreen, fullscreen)
                self.assertEqual(toggles, [])

    def test_image_only_keeps_the_pointer_unless_fullscreen(self):
        modes = []
        glfw = SimpleNamespace(
            CURSOR=1, CURSOR_NORMAL=0, CURSOR_HIDDEN=2,
            set_input_mode=lambda _window, _what, mode: modes.append(mode))
        for fullscreen, hidden in ((False, False), (True, True)):
            gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
            gui._glfw, gui._window = glfw, object()
            gui.fullscreen = fullscreen
            del modes[:]
            gui._set_image_only(True)
            self.assertEqual(modes, [2 if hidden else 0])

    def test_clicking_the_image_only_window_returns_to_the_live_view(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
        gui.page = 'info'
        gui._set_image_only(True)
        gui.hits = {'config_tab': (0, 0, 400, 400)}   # stale: nothing is drawn
        gui._on_mouse(MouseStub((10, 10)), None, 0, 1, 0)
        self.assertFalse(gui.image_only)
        self.assertEqual(gui.page, 'info')
        self.assertTrue(gui.dirty)

    def test_image_only_f_toggles_fullscreen_and_stays_image_only(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
        gui.image_only = True
        gui.image_only_previous_page = 'info'
        gui.fullscreen = True
        toggles = []
        gui._toggle_fullscreen = lambda *_args: (
            toggles.append(True), setattr(gui, 'fullscreen', False))

        gui._on_key(KeyStub(), object(), KeyStub.KEY_F, 0,
                    KeyStub.PRESS, 0)

        self.assertTrue(gui.image_only)
        self.assertFalse(gui.fullscreen)
        self.assertEqual(toggles, [True])

    def test_image_only_i_returns_to_live_with_diagnostics_visible(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
        gui.image_only = True
        gui.image_only_previous_page = 'info'
        gui.fullscreen = True
        gui.toolbar_visible = False

        gui._on_key(KeyStub(), object(), KeyStub.KEY_I, 0,
                    KeyStub.PRESS, 0)

        diagnostics = next(field for field in gui.fields
                           if field.dest == 'show_diagnostics')
        self.assertFalse(gui.image_only)
        self.assertTrue(gui.fullscreen)
        self.assertEqual(gui.page, 'info')
        self.assertFalse(gui.toolbar_visible)
        self.assertTrue(diagnostics.value)

    def test_fullscreen_f_works_while_text_editing(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
        gui.fullscreen = True
        text_field = next(field for field in gui.fields
                          if field.dest == 'freewheel_seconds')
        gui.selected = gui.fields.index(text_field)
        gui.editing = True
        gui.edit_buffer = '2.5'
        gui._persist_preferences = lambda: None
        toggles = []
        gui._toggle_fullscreen = lambda *_args: toggles.append(True)

        gui._on_key(KeyStub(), object(), KeyStub.KEY_F, 0, KeyStub.PRESS, 0)

        self.assertEqual(toggles, [True])
        self.assertFalse(gui.editing)
        self.assertEqual(text_field.value, 2.5)

    def test_gui_resource_monitor_reports_thread_process_and_memory(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
        gui.gui_resource_wall = 10.0
        gui.gui_resource_process = 5.0
        gui.gui_resource_thread = 2.0
        with (patch('tools.v7_receiver_gui.time.process_time', return_value=5.5),
              patch('tools.v7_receiver_gui.time.thread_time', return_value=2.2),
              patch('tools.v7_receiver_gui._process_memory_mib',
                    return_value=(256.0, 'RSS'))):
            lines = gui._sample_gui_resources(11.0)

        self.assertIn('20%', lines[0])
        self.assertIn('50%', lines[0])
        self.assertEqual(lines[1], 'RSS 256 MiB')

    def test_hidden_diagnostics_only_copy_the_footer_meter(self):
        meter = {'decoded': 12, 'input_fps': 11.25}
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
        gui.page = 'info'
        gui.v7_live = SimpleNamespace(RECEIVER_GUI_STATUS={
            'meter': meter,
            'diagnostics': lambda: self.fail(
                'hidden diagnostics should not be formatted'),
        })
        gui.dirty = False

        gui._poll_diagnostics()

        self.assertEqual(gui.live_meter, meter)
        self.assertIsNot(gui.live_meter, meter)
        self.assertIsNone(gui.live_diagnostics)
        self.assertTrue(gui.dirty)

    def test_visible_diagnostics_still_refresh_the_full_panel(self):
        meter = {'decoded': 12, 'input_fps': 11.25}
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
        gui.page = 'info'
        next(field for field in gui.fields
             if field.dest == 'show_diagnostics').value = True
        gui.v7_live = SimpleNamespace(RECEIVER_GUI_STATUS={
            'meter': meter,
            'diagnostics': lambda: {'decode': ('frame 12',)},
        })
        gui.dirty = False

        with patch.object(gui, '_sample_gui_resources',
                          return_value=('thread 0% · proc 0%', 'RSS 1 MiB')):
            gui._poll_diagnostics()

        self.assertEqual(gui.live_diagnostics['decode'], ('frame 12',))
        self.assertEqual(gui.live_meter['decoded'], 12)
        self.assertTrue(gui.dirty)

    def test_receiver_cli_accepts_image_only_mode(self):
        args = v7_live.parser().parse_args([
            'receive', '--device', 'named-loopback', '--image-only'])
        self.assertTrue(args.image_only)

    def test_new_picture_does_not_dirty_the_full_ui_canvas(self):
        image = Image.new('RGB', (80, 96), (90, 120, 160))
        frame = SimpleNamespace(
            generation=1,
            values=v7.image_values(image, v7.V7_GRIDS),
            shapes=v7.V7_GRIDS,
            aspect=v7.aspect_wire_code(image.size))
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
        gui.v7_live = SimpleNamespace(
            FRAME_BUFFER=SimpleNamespace(snapshot=lambda: frame),
            values_image=v7_live.values_image)
        gui.dirty = False

        gui._poll_frame()

        self.assertTrue(gui.picture_dirty)
        self.assertFalse(gui.dirty)
        self.assertEqual(gui.current_frame, frame)
        self.assertIsNone(gui.latest_values_image)

    def test_clicking_start_commits_active_text_edit_first(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
        text_field = next(field for field in gui.fields
                          if field.dest == 'freewheel_seconds')
        gui.selected = gui.fields.index(text_field)
        gui.editing = True
        gui.edit_buffer = '2.5'
        gui._persist_preferences = lambda: None
        gui.hits = {'start_stop': (0, 0, 100, 40)}
        started_with = []
        gui._start_receiver = lambda: started_with.append(text_field.value)

        gui._on_mouse(MouseStub((20, 20)), None, 0, 1, 0)

        self.assertEqual(text_field.value, 2.5)
        self.assertEqual(started_with, [2.5])
        self.assertFalse(gui.editing)

    def test_invalid_active_edit_blocks_start_but_not_stop(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
        volume = next(field for field in gui.fields
                       if field.dest == 'audio_volume')
        gui.selected = gui.fields.index(volume)
        gui.editing = True
        gui.edit_buffer = '1.5'
        gui.hits = {'start_stop': (0, 0, 100, 40)}
        gui._start_receiver = Mock()

        gui._on_mouse(MouseStub((20, 20)), None, 0, 1, 0)

        gui._start_receiver.assert_not_called()
        self.assertEqual(volume.value, 1.0)
        self.assertIn('between 0 and 1', gui.notice)

        gui.started = True
        gui.editing = True
        gui.edit_buffer = 'invalid'
        gui._stop_receiver = Mock()

        gui._on_mouse(MouseStub((20, 20)), None, 0, 1, 0)

        gui._stop_receiver.assert_called_once_with()

    def test_switching_tabs_commits_active_text_edit(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
        text_field = next(field for field in gui.fields
                          if field.dest == 'freewheel_seconds')
        gui.selected = gui.fields.index(text_field)
        gui.editing = True
        gui.edit_buffer = '3.5'
        gui._persist_preferences = lambda: None
        gui.hits = {'info_tab': (0, 0, 100, 40)}

        gui._on_mouse(MouseStub((20, 20)), None, 0, 1, 0)

        self.assertEqual(gui.page, 'info')
        self.assertEqual(text_field.value, 3.5)
        self.assertFalse(gui.editing)

    def test_save_directory_uses_native_folder_picker(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, (),
                          preference_path=ROOT/'tmp'/'receiver-gui-test.json')
        save_dir = next(field for field in gui.fields
                        if field.dest == 'save_dir')
        self.assertEqual(save_dir.kind, 'folder')
        index = gui.fields.index(save_dir)
        gui.hits = {f'row:{index}': (0, 0, 300, 40)}
        with patch('tools.v7_receiver_gui.pick_save_directory',
                   return_value='captures') as picker:
            gui._on_mouse(MouseStub((20, 20)), None, 0, 1, 0)
        picker.assert_called_once()
        self.assertEqual(save_dir.value, 'captures')

    def test_saved_audio_devices_restore_by_identity_not_old_index(self):
        path = ROOT/'tmp'/'receiver-gui-preferences-test.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            'input_device': {'name': 'Loopback In', 'hostapi': 'test'},
            'output_device': {'name': 'Loopback Out', 'hostapi': 'test'},
            'audio_muted': True,
            'freewheel_seconds': 3.5,
            'show_sync_warning': False,
        }), encoding='utf-8')
        identities = {
            (12, 'input'): {'name': 'Loopback In', 'hostapi': 'test'},
            (17, 'output'): {'name': 'Loopback Out', 'hostapi': 'test'},
        }
        try:
            with patch('tools.v7_receiver_gui._device_identity',
                       side_effect=lambda index, kind:
                       identities[(index, kind)]):
                gui = ReceiverGui(
                    self, self.root_parser, self.receive_parser,
                    (('renumbered input', 12),), audio_output_choices=(
                        ('renumbered output', 17),), preference_path=path)
            values = {field.dest: field.value for field in gui.fields}
            self.assertEqual(values['device'], 12)
            self.assertEqual(values['audio_output_device'], 17)
            self.assertTrue(values['audio_muted'])
            self.assertEqual(values['freewheel_seconds'], 3.5)
            self.assertFalse(values['show_sync_warning'])
        finally:
            path.unlink(missing_ok=True)

    def test_malformed_saved_device_identities_are_ignored(self):
        path = ROOT/'tmp'/'receiver-gui-malformed-preferences-test.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            'input_device': 'not an identity object',
            'output_device': ['also invalid'],
        }), encoding='utf-8')
        try:
            with patch('tools.v7_receiver_gui._device_identity',
                       return_value={'name': 'Available input',
                                     'hostapi': 'test'}):
                gui = ReceiverGui(
                    self, self.root_parser, self.receive_parser,
                    (('available input', 3),), audio_output_choices=(
                        ('available output', 5),), preference_path=path)
            values = {field.dest: field.value for field in gui.fields}
            self.assertEqual(values['device'], 3)
            self.assertIsNone(values['audio_output_device'])
            self.assertIsNone(gui.unavailable_input_identity)
            self.assertIsNone(gui.unavailable_output_identity)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_saved_audio_devices_require_reselection(self):
        path = ROOT/'tmp'/'receiver-gui-missing-preferences-test.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            'input_device': {'name': 'Missing In', 'hostapi': 'test'},
            'output_device': {'name': 'Missing Out', 'hostapi': 'test'},
        }), encoding='utf-8')
        try:
            with patch('tools.v7_receiver_gui._device_identity',
                       return_value={'name': 'Other', 'hostapi': 'test'}):
                gui = ReceiverGui(
                    self, self.root_parser, self.receive_parser,
                    (('other input', 4),), audio_output_choices=(
                        ('other output', 5),), preference_path=path)
            values = {field.dest: field.value for field in gui.fields}
            self.assertIsNone(values['device'])
            self.assertIsNone(values['audio_output_device'])
            self.assertIn('Missing In', gui._field_value_label(
                next(field for field in gui.fields
                     if field.dest == 'device')))
            self.assertIn('Missing Out', gui._field_value_label(
                next(field for field in gui.fields
                     if field.dest == 'audio_output_device')))
        finally:
            path.unlink(missing_ok=True)

    def test_passthrough_device_and_mute_update_live_runtime(self):
        path = ROOT/'tmp'/'receiver-gui-live-preferences-test.json'
        identity = {'name': 'Output', 'hostapi': 'test'}
        with patch('tools.v7_receiver_gui._device_identity',
                   return_value=identity):
            gui = ReceiverGui(
                self, self.root_parser, self.receive_parser, (),
                audio_output_choices=(('Output', 9),), preference_path=path)
            from tools.v7_receiver_audio import ReceiverRuntimeOptions
            gui.runtime_options = ReceiverRuntimeOptions()
            gui.started = True
            output = next(field for field in gui.fields
                          if field.dest == 'audio_output_device')
            gui._select_choice(output, 9)
            mute = next(field for field in gui.fields
                        if field.dest == 'audio_muted')
            gui._adjust_field(mute, 1)
            volume = next(field for field in gui.fields
                          if field.dest == 'audio_volume')
            gui.selected = gui.fields.index(volume)
            gui.editing = True
            gui.edit_buffer = '0.25'
            gui._finish_edit(volume)
        snapshot = gui.runtime_options.snapshot()
        self.assertEqual(snapshot['audio_output_device'], 9)
        self.assertEqual(snapshot['audio_output_identity'], identity)
        self.assertTrue(snapshot['audio_muted'])
        self.assertEqual(snapshot['audio_volume'], .25)
        self.assertEqual(json.loads(path.read_text())['audio_volume'], .25)
        path.unlink(missing_ok=True)


class ReceiverGuiButtonLayoutTests(unittest.TestCase):
    """On/off settings are buttons on a grid; the page is named sections."""

    SIZE = (960, 720)

    @classmethod
    def setUpClass(cls):
        cls.root_parser, cls.receive_parser = _receive_parser(v7_live)

    def _gui(self, preferences=None):
        with patch('tools.v7_receiver_gui._load_preferences',
                   return_value=dict(preferences or {})):
            return ReceiverGui(
                self, self.root_parser, self.receive_parser,
                (('test input device', 3),),
                preference_path='unused-preferences.json')

    def _show(self, gui, index):
        gui._scroll_to(index, self.SIZE)
        gui._canvas(self.SIZE)
        return gui.hits[f'row:{index}']

    def _click(self, gui, rect):
        gui._on_mouse(MouseStub(((rect[0]+rect[2])/2, (rect[1]+rect[3])/2)),
                      None, 0, 1, 0)

    def _values(self, gui):
        return [field.value for field in gui.fields]

    def test_every_boolean_setting_is_a_button_and_a_click_flips_only_it(self):
        gui = self._gui()
        booleans = [index for index, field in enumerate(gui.fields)
                    if field.kind == 'bool']
        self.assertGreaterEqual(len(booleans), 8)
        # Every argparse on/off flag the page offers is among them.
        for index, field in enumerate(gui.fields):
            if field.action is not None and field.action.nargs == 0:
                self.assertIn(index, booleans, field.dest)
        for index in booleans:
            gui = self._gui()
            field = gui.fields[index]
            rect = self._show(gui, index)
            self.assertLess(rect[2]-rect[0], self.SIZE[0]//2, field.label)
            self.assertNotIn(f'value:{index}', gui.hits)
            before = self._values(gui)
            with patch('tools.v7_receiver_gui._save_preferences'):
                self._click(gui, rect)
                after = self._values(gui)
                changed = [position for position, value in enumerate(after)
                           if value != before[position]]
                if field.locked:
                    # The integrated viewer's own switch stays on.
                    self.assertEqual(changed, [], field.label)
                    continue
                self.assertEqual(changed, [index], field.label)
                self.assertIs(field.value, not before[index])
                self.assertIsNone(gui.dropdown)
                self.assertFalse(gui.editing)
                self._click(gui, self._show(gui, index))
            self.assertEqual(self._values(gui), before, field.label)

    def test_cells_share_lines_and_do_not_overlap(self):
        gui = self._gui()
        items = gui._config_items(self.SIZE[0])
        self.assertTrue(any(kind == 'line' and len(payload) > 1
                            for kind, payload in items))
        gui._canvas(self.SIZE)
        rects = [rect for key, rect in gui.hits.items()
                 if key.startswith('row:')]
        self.assertGreater(len(rects), 6)
        for index, first in enumerate(rects):
            self.assertGreaterEqual(first[0], 0)
            self.assertLessEqual(first[2], self.SIZE[0])
            for second in rects[index+1:]:
                self.assertFalse(
                    first[0] < second[2] and second[0] < first[2] and
                    first[1] < second[3] and second[1] < first[3],
                    (first, second))
        self.assertLess(gui._columns(640), gui._columns(960))

    def test_settings_are_named_sections_without_a_toggle(self):
        gui = self._gui()
        gui._canvas(self.SIZE)
        self.assertNotIn('advanced_toggle', gui.hits)
        items = gui._config_items(self.SIZE[0])
        headers = [payload for kind, payload in items if kind == 'header']
        self.assertEqual(headers, ['Live controls', 'Input', 'Wire',
                                   'Startup view', 'Output and logging'])
        shown = {index for kind, payload in items if kind == 'line'
                 for index, _column, _span in payload}
        self.assertEqual(shown, set(range(len(gui.fields))))
        dests = {gui.fields[index].dest for index in shown}
        for dest in ('aspect_layout', 'aspect_tail', 'log', 'no_log',
                     'diagnostics', 'device'):
            self.assertIn(dest, dests)
        # Each is reachable on the page by scrolling alone.
        for index in shown:
            gui.scroll = 0
            self.assertEqual(len(self._show(gui, index)), 4)
        # Non-boolean settings keep their control type.
        layout = next(index for index, field in enumerate(gui.fields)
                      if field.dest == 'aspect_layout')
        self.assertEqual(gui.fields[layout].kind, 'choice')
        self._click(gui, self._show(gui, layout))
        self.assertEqual(gui.dropdown, layout)
        gui._canvas(self.SIZE)
        self.assertIn('option:0', gui.hits)

    def test_keyboard_walks_every_setting_and_flips_a_button(self):
        gui = self._gui()
        order = gui._config_field_indexes()
        gui.selected = order[0]
        for expected in order[1:]:
            gui._on_key(KeyStub, None, KeyStub.KEY_DOWN, 0, KeyStub.PRESS, 0)
            self.assertEqual(gui.selected, expected)
            gui._canvas(self.SIZE)
            self.assertIn(f'row:{expected}', gui.hits)
        log = next(index for index, field in enumerate(gui.fields)
                   if field.dest == 'log')
        gui.selected = log
        gui._on_key(KeyStub, None, KeyStub.KEY_ENTER, 0, KeyStub.PRESS, 0)
        self.assertTrue(gui.fields[log].value)

    def test_setup_page_scrolls_by_line_when_it_does_not_fit(self):
        size = (720, 480)
        gui = self._gui()
        gui.width, gui.height = size
        items = gui._config_items(size[0])
        room = gui._config_room(size)
        top = gui._max_scroll(items, room)
        self.assertGreater(top, 0)
        for _ in range(len(items)):
            gui._on_scroll(None, 0, -1)
        self.assertEqual(gui.scroll, top)
        gui._canvas(size)
        for key, rect in gui.hits.items():
            if key.startswith('row:'):
                self.assertLessEqual(rect[3], size[1]-gui._footer_height(size[0]))

    def test_the_default_window_shows_the_whole_setup_page(self):
        gui = self._gui()
        items = gui._config_items(self.SIZE[0])
        self.assertEqual(gui._max_scroll(items, gui._config_room(self.SIZE)), 0)

    def test_every_setting_is_in_exactly_one_section(self):
        gui = self._gui()
        placed = [index for _title, indexes in gui._config_sections()
                  for index in indexes]
        self.assertEqual(sorted(placed), list(range(len(gui.fields))))

    def test_a_setting_named_in_no_section_goes_to_the_last_one(self):
        gui = self._gui()
        gui.fields.append(OptionField(None, False, 'Something new', 'bool'))
        title, indexes = gui._config_sections()[-1]
        self.assertEqual(title, 'Output and logging')
        self.assertEqual(indexes[-1], len(gui.fields)-1)

    def test_live_section_holds_exactly_what_changes_while_receiving(self):
        gui = self._gui()
        title, indexes = gui._config_sections()[0]
        self.assertEqual(title, 'Live controls')
        self.assertEqual(
            {gui.fields[index].dest for index in indexes},
            {'audio_output_device', 'audio_volume', 'audio_muted',
             'freewheel_seconds', 'show_sync_warning', 'Pixel display',
             'Display upscaler', 'DCT reconstruction', 'Display grain',
             'Edge reconstruction', 'Edge strength'})
        gui.started = True
        live = set(indexes)
        for index, field in enumerate(gui.fields):
            self.assertEqual(gui._runtime_locked(field), index not in live,
                             field.dest)
        notes = [payload for kind, payload in gui._config_items(self.SIZE[0])
                 if kind == 'note']
        self.assertEqual(notes, ['Stop to change the settings below'])
        gui.started = False
        self.assertFalse(any(gui._runtime_locked(field)
                             for field in gui.fields))

    def test_grid_collapses_to_one_column_in_a_narrow_window(self):
        gui = self._gui()
        self.assertEqual(gui._columns(960), 3)
        self.assertEqual(gui._columns(400), 1)
        for kind, payload in gui._config_items(400):
            if kind == 'line':
                self.assertEqual(len(payload), 1)

    def test_locked_setting_reports_while_receiving_and_live_ones_edit(self):
        gui = self._gui()
        gui.started = True
        device = next(index for index, field in enumerate(gui.fields)
                      if field.dest == 'device')
        self._click(gui, self._show(gui, device))
        self.assertIsNone(gui.dropdown)
        self.assertIn('locked', gui.notice)
        mute = next(index for index, field in enumerate(gui.fields)
                    if field.dest == 'audio_muted')
        with patch('tools.v7_receiver_gui._save_preferences'):
            self._click(gui, self._show(gui, mute))
        self.assertTrue(gui.fields[mute].value)

    def test_a_tab_click_repaints_at_once(self):
        # The page used to change only on the next click in the window.
        gui = self._gui()
        gui.page = 'info'
        for key, page in (('config_tab', 'config'), ('info_tab', 'info')):
            gui._canvas(self.SIZE)
            gui.dirty = False
            self._click(gui, gui.hits[key])
            self.assertEqual(gui.page, page)
            self.assertTrue(gui.dirty, key)

    def test_toolbar_has_start_beside_the_tabs(self):
        gui = self._gui()
        gui._canvas(self.SIZE)
        self.assertEqual(gui.hits['config_tab'][0], 12)
        self.assertEqual(gui.hits['start_stop'][0],
                         gui.hits['info_tab'][2]+8)

    def test_live_page_toolbar_mutes_and_steps_the_volume(self):
        gui = self._gui()
        gui.page = 'info'
        gui._canvas(self.SIZE)
        for key in ('mute_button', 'volume_down', 'volume_up'):
            self.assertIn(key, gui.hits)
        by_dest = {field.dest: field for field in gui.fields}
        with patch('tools.v7_receiver_gui._save_preferences'):
            self._click(gui, gui.hits['mute_button'])
            self.assertTrue(by_dest['audio_muted'].value)
            gui._canvas(self.SIZE)
            self._click(gui, gui.hits['mute_button'])
            self.assertFalse(by_dest['audio_muted'].value)
            by_dest['audio_volume'].value = 1.0
            gui._canvas(self.SIZE)
            self._click(gui, gui.hits['volume_down'])
            self.assertEqual(by_dest['audio_volume'].value, 0.9)
            gui._canvas(self.SIZE)
            for _ in range(3):
                self._click(gui, gui.hits['volume_up'])
            self.assertEqual(by_dest['audio_volume'].value, 1.0)
        # They give way before the window gets too narrow for them.
        gui._canvas((800, 600))
        self.assertNotIn('mute_button', gui.hits)
        self.assertIn('mode_button', gui.hits)
        gui.page = 'config'
        gui._canvas(self.SIZE)
        self.assertNotIn('mute_button', gui.hits)

    def test_footer_shows_the_selected_help_until_a_notice_needs_it(self):
        gui = self._gui()
        device = next(index for index, field in enumerate(gui.fields)
                      if field.dest == 'device')
        gui.selected = device
        self.assertEqual(gui._footer_content()[0],
                         gui.fields[device].action.help)
        gui.notice = 'Receiver options are locked while receiving'
        self.assertEqual(gui._footer_content()[0], gui.notice)
        gui.selected = device+1
        self.assertNotEqual(gui._footer_content()[0], gui.notice)
        gui.device_error = 'No audio input devices are available.'
        self.assertEqual(gui._footer_content()[0], gui.device_error)

    def test_settings_that_must_match_the_sender_are_tagged(self):
        from tools.v7_receiver_gui import MATCH_SETTINGS
        self.assertEqual(MATCH_SETTINGS, ('aspect_layout', 'aspect_tail'))
        gui = self._gui()
        for field in gui.fields:
            if field.dest in MATCH_SETTINGS:
                self.assertNotIn('match', field.label.lower())

    def test_old_preferences_with_an_advanced_flag_still_load(self):
        gui = self._gui({'advanced': True, 'advanced_options': True,
                         'audio_muted': True, 'freewheel_seconds': 3.5,
                         'input_device': None, 'output_device': None})
        by_dest = {field.dest: field for field in gui.fields}
        self.assertTrue(by_dest['audio_muted'].value)
        self.assertEqual(by_dest['freewheel_seconds'].value, 3.5)
        self.assertFalse(hasattr(gui, 'advanced_options'))
        self.assertEqual(gui._canvas(self.SIZE).shape[:2],
                         (self.SIZE[1], self.SIZE[0]))
        with patch('tools.v7_receiver_gui._save_preferences') as save:
            gui._persist_preferences()
        self.assertNotIn('advanced', save.call_args.args[0])
        self.assertNotIn('advanced_options', save.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
