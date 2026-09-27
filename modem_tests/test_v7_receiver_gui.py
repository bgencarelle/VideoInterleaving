"""Receiver GUI configuration maps every receive CLI setting safely."""
import threading
import unittest
from types import SimpleNamespace

from PIL import Image

from animation_modem import v7
from tools import v7_live
from tools.v7_receiver_gui import (ROOT, ReceiverGui, _make_fields,
                                   FULLSCREEN_TOOLBAR_EDGE,
                                   FULLSCREEN_TOOLBAR_HIDE_SECONDS,
                                   _receive_parser)


class MouseStub:
    MOUSE_BUTTON_LEFT = 0
    PRESS = 1

    def __init__(self, position):
        self.position = position

    def get_cursor_pos(self, _window):
        return self.position


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

    def test_start_requires_an_explicit_input_device(self):
        gui = ReceiverGui(self, self.root_parser, self.receive_parser, ())
        with self.assertRaisesRegex(ValueError, 'Select an input audio device'):
            gui._build_arguments()

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
        gui.hits = {'mode_button': (0, 0, 40, 20)}
        gui._on_mouse(MouseStub((10, 10)), None, 0, 1, 0)

        self.assertEqual(gui.display_mode, 'bilinear')
        self.assertEqual(gui.notice, 'Display upscaler: Bilinear')

        direction = next(field for field in gui.fields
                         if field.dest == 'direction')
        before = direction.value
        gui._adjust_field(direction, 1)
        self.assertEqual(direction.value, before)
        self.assertIn('locked while receiving', gui.notice)

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
        glfw = SimpleNamespace(
            DONT_CARE=0,
            get_primary_monitor=lambda: object(),
            get_window_pos=lambda _window: (0, 0),
            get_window_size=lambda _window: (960, 720),
            get_video_mode=lambda _monitor: SimpleNamespace(
                size=SimpleNamespace(width=1920, height=1080),
                refresh_rate=60),
            set_window_monitor=lambda *_args: None)

        gui._toggle_fullscreen(glfw, object())

        self.assertTrue(gui.fullscreen)
        self.assertFalse(gui.toolbar_visible)

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
