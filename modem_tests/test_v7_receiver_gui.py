"""Receiver GUI configuration maps every receive CLI setting safely."""
import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from animation_modem import v7
from tools import v7_live
from tools.v7_gl_viewer import DISPLAY_MODES
from tools.v7_receiver_gui import (ROOT, ReceiverGui, _make_fields,
                                   FULLSCREEN_TOOLBAR_EDGE,
                                   FULLSCREEN_TOOLBAR_HIDE_SECONDS,
                                   QueueWriter,
                                   _logical_rect_to_framebuffer,
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
    KEY_ESCAPE = 256
    KEY_ENTER = 257
    KEY_DOWN = 264
    KEY_UP = 265
    KEY_KP_ENTER = 335
    KEY_P = 80
    MOD_CONTROL = 2


class ReceiverGuiOptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root_parser, cls.receive_parser = _receive_parser(v7_live)

    def test_every_receive_option_is_represented(self):
        fields = _make_fields(self.receive_parser, ())
        represented = {field.action.dest for field in fields
                       if field.action is not None}
        expected = {action.dest for action in self.receive_parser._actions
                    if action.dest not in ('help', 'mode')}
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

    def test_first_available_input_is_preselected(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser,
                          (('first input', 4), ('second input', 9)))
        device = next(field for field in gui.fields
                      if field.dest == 'device')
        self.assertEqual(device.value, 4)
        self.assertEqual(gui._build_arguments().device, 4)

    def test_selected_cli_options_reach_receiver_arguments(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser,
                          (('test input device', 3),))
        values = {field.dest: field for field in gui.fields
                  if field.action is not None}
        values['device'].value = 3
        values['direction'].value = 'reverse'
        values['decode_history'].value = 4
        values['pilot_timing'].value = 'tone-joint'
        values['show_diagnostics'].value = False
        values['no_tail_memory'].value = True
        args = gui._build_arguments()
        self.assertEqual(args.direction, 'reverse')
        self.assertEqual(args.decode_history, 4)
        self.assertEqual(args.pilot_timing, 'tone-joint')
        self.assertFalse(args.show_diagnostics)
        self.assertTrue(args.no_tail_memory)

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

        direction = next(field for field in gui.fields
                         if field.dest == 'direction')
        before = direction.value
        gui._adjust_field(direction, 1)
        self.assertEqual(direction.value, before)
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

    def test_basic_setup_hides_advanced_options_until_requested(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
        basic = {gui.fields[index].dest
                 for index in gui._config_field_indexes()}
        self.assertIn('device', basic)
        self.assertIn('experimental_fold', basic)
        self.assertNotIn('decode_history', basic)

        gui.advanced_options = True
        advanced = {gui.fields[index].dest
                    for index in gui._config_field_indexes()}
        self.assertIn('decode_history', advanced)

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

        gui._canvas((960, 720))

        self.assertNotIn('fullscreen', gui.hits)
        self.assertFalse(gui._diagnostics_visible())
        self.assertEqual(gui._picture_box((960, 720))[1], 8)

    def test_fullscreen_i_reveals_and_toggles_diagnostics(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
        gui.fullscreen = True
        gui.toolbar_visible = False
        gui.page = 'info'

        gui._on_key(KeyStub(), object(), KeyStub.KEY_I, 0, KeyStub.PRESS, 0)

        field = next(field for field in gui.fields
                     if field.dest == 'show_diagnostics')
        self.assertTrue(gui.toolbar_visible)
        self.assertTrue(field.value)
        self.assertTrue(gui._diagnostics_visible())

    def test_image_only_f_exits_fullscreen_once_when_entered_from_windowed(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
        gui.image_only = True
        gui.image_only_previous_page = 'info'
        gui.image_only_previous_fullscreen = False
        gui.fullscreen = True
        gui._glfw = SimpleNamespace(
            CURSOR=1, CURSOR_NORMAL=0,
            set_input_mode=lambda *_args: None)
        gui._window = object()
        toggles = []

        def toggle(*_args):
            toggles.append(True)
            gui.fullscreen = False

        gui._toggle_fullscreen = toggle
        gui._on_key(KeyStub(), gui._window, KeyStub.KEY_F, 0,
                    KeyStub.PRESS, 0)

        self.assertFalse(gui.image_only)
        self.assertFalse(gui.fullscreen)
        self.assertEqual(toggles, [True])

    def test_image_only_f_exits_fullscreen_once_when_already_fullscreen(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
        gui.image_only = True
        gui.image_only_previous_page = 'info'
        gui.image_only_previous_fullscreen = True
        gui.fullscreen = True
        toggles = []
        gui._toggle_fullscreen = lambda *_args: (
            toggles.append(True), setattr(gui, 'fullscreen', False))

        gui._on_key(KeyStub(), object(), KeyStub.KEY_F, 0,
                    KeyStub.PRESS, 0)

        self.assertFalse(gui.image_only)
        self.assertFalse(gui.fullscreen)
        self.assertEqual(toggles, [True])

    def test_image_only_i_returns_to_live_with_diagnostics_visible(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
        gui.image_only = True
        gui.image_only_previous_page = 'info'
        gui.image_only_previous_fullscreen = True
        gui.fullscreen = True
        gui.toolbar_visible = False

        gui._on_key(KeyStub(), object(), KeyStub.KEY_I, 0,
                    KeyStub.PRESS, 0)

        diagnostics = next(field for field in gui.fields
                           if field.dest == 'show_diagnostics')
        self.assertFalse(gui.image_only)
        self.assertTrue(gui.fullscreen)
        self.assertEqual(gui.page, 'info')
        self.assertTrue(gui.toolbar_visible)
        self.assertTrue(diagnostics.value)

    def test_fullscreen_f_works_while_text_editing(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
        gui.fullscreen = True
        save_dir = next(field for field in gui.fields
                        if field.dest == 'save_dir')
        gui.selected = gui.fields.index(save_dir)
        gui.editing = True
        gui.edit_buffer = 'captures'
        toggles = []
        gui._toggle_fullscreen = lambda *_args: toggles.append(True)

        gui._on_key(KeyStub(), object(), KeyStub.KEY_F, 0, KeyStub.PRESS, 0)

        self.assertEqual(toggles, [True])
        self.assertFalse(gui.editing)
        self.assertEqual(save_dir.value, 'captures')

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

        self.assertIs(gui.live_meter, meter)
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
        save_dir = next(field for field in gui.fields
                        if field.dest == 'save_dir')
        gui.selected = gui.fields.index(save_dir)
        gui.editing = True
        gui.edit_buffer = 'captures with spaces'
        gui.hits = {'start_stop': (0, 0, 100, 40)}
        started_with = []
        gui._start_receiver = lambda: started_with.append(save_dir.value)

        gui._on_mouse(MouseStub((20, 20)), None, 0, 1, 0)

        self.assertEqual(save_dir.value, 'captures with spaces')
        self.assertEqual(started_with, ['captures with spaces'])
        self.assertFalse(gui.editing)

    def test_switching_tabs_commits_active_text_edit(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
        save_dir = next(field for field in gui.fields
                        if field.dest == 'save_dir')
        gui.selected = gui.fields.index(save_dir)
        gui.editing = True
        gui.edit_buffer = 'captures with spaces'
        gui.hits = {'info_tab': (0, 0, 100, 40)}

        gui._on_mouse(MouseStub((20, 20)), None, 0, 1, 0)

        self.assertEqual(gui.page, 'info')
        self.assertEqual(save_dir.value, 'captures with spaces')
        self.assertFalse(gui.editing)


if __name__ == '__main__':
    unittest.main()
