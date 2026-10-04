"""The sender GUI's DCT kernel chooser and its live parameters."""
import json
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tools import v7_send_gui as gui_module
from tools.v7_send_gui import (OutputDevice, SenderGui, build_command,
                                validate_settings,
                                _restore_sender_settings,
                                _serialize_sender_settings)


class KeyStub:
    PRESS, REPEAT = 1, 2
    KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN = 263, 262, 265, 264
    KEY_ENTER, KEY_KP_ENTER, KEY_ESCAPE, KEY_TAB = 257, 335, 256, 258
    KEY_C, KEY_I, KEY_R, KEY_SPACE = 67, 73, 82, 32
    KEY_A, KEY_V, KEY_BACKSPACE = 65, 86, 259
    KEY_B, KEY_D, KEY_DELETE = 66, 68, 261
    MOD_CONTROL, MOD_SHIFT = 2, 1


def _gui(devices):
    gui = SenderGui(devices)
    gui.settings.update(device=3, source='test', profile='fold-500',
                        dct_encode=True)
    return gui


class KernelGuiTests(unittest.TestCase):
    def setUp(self):
        self.devices = (OutputDevice(3, 'Test output', 2, 48000),)
        self.sd = Mock()
        self.sd.check_output_settings.return_value = None
        gui_module.kernel_registry(refresh=True)

    def command(self, gui):
        return build_command(gui.settings, self.devices, self.sd)

    def test_the_reference_adds_nothing_to_the_command_line(self):
        gui = _gui(self.devices)
        self.assertEqual(gui.settings['dct_kernel'], 'reference')
        self.assertNotIn('--dct-kernel', self.command(gui))
        self.assertFalse([d for d in gui._visible_fields() if d.startswith('kp:')])

    def test_the_chooser_lists_the_kernel_folder(self):
        gui = _gui(self.devices)
        names = [name for _label, name in gui._choices('dct_kernel')]
        self.assertEqual(names[0], 'reference')
        for expected in ('lanczos', 'mitchell', 'antiring'):
            self.assertIn(expected, names)
        self.assertIn('dct_kernel', gui._visible_fields())

    def test_a_chosen_kernel_shows_its_parameters_and_reaches_the_command(self):
        gui = _gui(self.devices)
        gui._assign('dct_kernel', 'lanczos')
        fields = gui._visible_fields()
        start = fields.index('dct_kernel')
        self.assertEqual(fields[start+1:start+5],
                         ['kp:lobes', 'kp:width', 'kp:luma_mix', 'kp:chroma_mix'])
        command = self.command(gui)
        self.assertEqual(command[command.index('--dct-kernel')+1], 'lanczos')
        self.assertNotIn('--dct-kernel-param', command)          # all defaults
        gui._set_kernel_param('kp:width', '0.8')
        command = self.command(gui)
        self.assertIn('width=0.8', command)
        self.assertEqual(command.count('--dct-kernel-param'), 1)

    def test_kernel_specific_host_default_is_shown_and_not_sent_as_an_override(self):
        gui = _gui(self.devices)
        gui._assign('dct_kernel', 'csf_diamond')
        self.assertEqual(gui._field_value('kp:luma_mix'), 0.25)
        self.assertTrue(gui._is_default('kp:luma_mix'))
        self.assertIn('default 0.25', gui._help_text('kp:luma_mix'))
        command = self.command(gui)
        self.assertNotIn('--dct-kernel-param', command)
        gui._set_kernel_param('kp:luma_mix', 1.0)
        command = self.command(gui)
        self.assertIn('luma_mix=1', command)

    def test_aspect_mono_shows_its_profile_kernel_defaults(self):
        gui = _gui(self.devices)
        gui.settings['profile'] = 'aspect-mono-500'
        gui._assign('dct_kernel', 'lanczos')
        self.assertEqual(gui._field_value('kp:width'), 0.5)
        self.assertEqual(gui._field_value('kp:luma_mix'), 0.25)
        command = self.command(gui)
        self.assertNotIn('--dct-kernel-param', command)

    def test_aspect_stereo_shows_its_profile_kernel_defaults(self):
        gui = _gui(self.devices)
        gui.settings['profile'] = 'aspect-fold-500'
        gui._assign('dct_kernel', 'band_taper')
        self.assertEqual(gui._field_value('kp:edge'), 0.9)
        self.assertEqual(gui._field_value('kp:luma_mix'), 0.05)
        self.assertEqual(gui._field_value('kp:use_guests'), 1)
        command = self.command(gui)
        self.assertNotIn('--dct-kernel-param', command)

    def test_pixel_encode_and_direct_dct_off_hide_and_drop_the_kernel(self):
        gui = _gui(self.devices)
        gui._assign('dct_kernel', 'gaussian')
        gui.settings['pixel_encode'] = True
        self.assertNotIn('dct_kernel', gui._visible_fields())
        self.assertNotIn('--dct-kernel', self.command(gui))
        gui.settings['pixel_encode'] = False
        gui.settings['dct_encode'] = False
        self.assertNotIn('dct_kernel', gui._visible_fields())
        self.assertNotIn('--dct-kernel', self.command(gui))

    def test_an_unknown_saved_kernel_refuses_to_start(self):
        gui = _gui(self.devices)
        gui.settings['dct_kernel'] = 'vanished'
        with self.assertRaisesRegex(ValueError, 'DCT kernel'):
            validate_settings(gui.settings, self.devices, self.sd)

    def test_stale_saved_parameters_are_ignored(self):
        gui = _gui(self.devices)
        gui._assign('dct_kernel', 'lanczos')
        gui.settings['dct_kernel_params'] = {'lanczos': {'width': 1.0, 'gone': 3}}
        command = self.command(gui)
        self.assertIn('width=1', command)
        self.assertFalse([c for c in command if c.startswith('gone')])

    def test_typed_values_are_clamped_and_bad_text_is_refused(self):
        gui = _gui(self.devices)
        gui._assign('dct_kernel', 'lanczos')
        gui.selected, gui.editing, gui.edit_buffer = 'kp:width', True, '9'
        gui._finish_edit()
        self.assertEqual(gui._field_value('kp:width'), 1.6)
        gui.selected, gui.editing, gui.edit_buffer = 'kp:width', True, 'wide'
        gui._finish_edit()
        self.assertEqual(gui._field_value('kp:width'), 1.6)
        self.assertIn('number', gui.notice)

    def test_stepping_a_parameter_updates_a_running_sender(self):
        gui = _gui(self.devices)
        gui._assign('dct_kernel', 'lanczos')
        control = Mock()
        gui.process = SimpleNamespace(stdin=control)
        gui.selected = 'kp:width'
        gui._step_field('kp:width', -1)
        payload = json.loads(control.write.call_args.args[0])
        self.assertEqual(payload['kernel'], 'lanczos')
        self.assertAlmostEqual(payload['kernel_params']['width'], 0.85)
        self.assertEqual(payload['dct']['sharpen'], 'off')
        control.flush.assert_called()
        gui._step_field('kp:width', -1, fast=True)
        self.assertAlmostEqual(gui._field_value('kp:width'), 0.6)

    def test_arrow_keys_step_and_cycle_while_sending(self):
        gui = _gui(self.devices)
        control = Mock()
        gui.process = SimpleNamespace(stdin=control)
        gui.selected = 'dct_kernel'
        gui._on_key(KeyStub, None, KeyStub.KEY_RIGHT, 0, KeyStub.PRESS, 0)
        names = [name for _label, name in gui._choices('dct_kernel')]
        self.assertEqual(gui.settings['dct_kernel'], names[1])
        self.assertEqual(json.loads(control.write.call_args.args[0])['kernel'],
                         names[1])
        gui._on_key(KeyStub, None, KeyStub.KEY_LEFT, 0, KeyStub.PRESS, 0)
        self.assertEqual(gui.settings['dct_kernel'], 'reference')
        gui._assign('dct_kernel', 'gaussian')
        gui.selected = 'kp:sigma'
        before = gui._field_value('kp:sigma')
        gui._on_key(KeyStub, None, KeyStub.KEY_RIGHT, 0, KeyStub.PRESS, 0)
        self.assertGreater(gui._field_value('kp:sigma'), before)
        gui._on_key(KeyStub, None, KeyStub.KEY_RIGHT, 0, KeyStub.PRESS,
                    KeyStub.MOD_SHIFT)
        self.assertAlmostEqual(gui._field_value('kp:sigma'),
                               before+0.025*6, places=6)

    def test_other_settings_stay_locked_while_sending(self):
        gui = _gui(self.devices)
        gui.process = SimpleNamespace(stdin=Mock())
        gui.selected = 'profile'
        gui._on_key(KeyStub, None, KeyStub.KEY_ENTER, 0, KeyStub.PRESS, 0)
        self.assertIn('locked', gui.notice)

    def test_dct_strengths_are_live_too(self):
        gui = _gui(self.devices)
        control = Mock()
        gui.process = SimpleNamespace(stdin=control)
        gui.settings['dct_sharpen'] = 'taper'
        gui.selected = 'dct_sharpen_strength'
        gui._on_key(KeyStub, None, KeyStub.KEY_RIGHT, 0, KeyStub.PRESS, 0)
        dct = json.loads(control.write.call_args.args[0])['dct']
        self.assertEqual(dct['sharpen'], 'taper')
        self.assertAlmostEqual(dct['sharpen_strength'], 0.30)
        gui.selected, gui.editing, gui.edit_buffer = 'dct_clarity', True, '2'
        gui._finish_edit()
        self.assertEqual(gui.settings['dct_clarity'], '0')       # refused, kept

    def test_values_survive_a_restart_of_the_gui(self):
        gui = _gui(self.devices)
        gui._assign('dct_kernel', 'mitchell')
        gui._set_kernel_param('kp:B', 0.1)
        saved = json.loads(json.dumps(_serialize_sender_settings(gui.settings)))
        fresh = _gui(self.devices)
        _restore_sender_settings(fresh.settings, saved)
        self.assertEqual(fresh.settings['dct_kernel'], 'mitchell')
        self.assertEqual(fresh._field_value('kp:B'), 0.1)
        _restore_sender_settings(fresh.settings, {'dct_kernel_params': {
            'x': {'y': 'not a number', 'z': 2}, 'bad': 7}})
        self.assertEqual(fresh.settings['dct_kernel_params'], {'x': {'z': 2.0}})

    def test_kernel_dir_files_appear_after_r_and_vanish_when_deleted(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(gui_module, 'KERNEL_DIRS', [folder]), \
                patch.object(gui_module, '_KERNELS', None):
            gui = _gui(self.devices)
            self.assertNotIn('dropped_in',
                             [n for _l, n in gui._choices('dct_kernel')])
            path = Path(folder)/'dropped_in.py'
            path.write_text(textwrap.dedent('''
                import numpy as np
                SUPPORT = 1.0
                def kernel(x):
                    return np.clip(1.0 - np.abs(x), 0.0, None)
            '''))
            gui._on_key(KeyStub, None, KeyStub.KEY_R, 0, KeyStub.PRESS, 0)
            self.assertIn('+dropped_in', gui.notice)
            gui._assign('dct_kernel', 'dropped_in')
            command = self.command(gui)
            self.assertEqual(command[command.index('--dct-kernel')+1], 'dropped_in')
            self.assertEqual(command[command.index('--dct-kernel-dir')+1], folder)
            path.unlink()
            gui._on_key(KeyStub, None, KeyStub.KEY_R, 0, KeyStub.PRESS, 0)
            self.assertEqual(gui.settings['dct_kernel'], 'reference')
            self.assertIn('-dropped_in', gui.notice)

    def test_r_on_a_running_sender_asks_it_to_rescan_too(self):
        gui = _gui(self.devices)
        control = Mock()
        gui.process = SimpleNamespace(stdin=control)
        gui._on_key(KeyStub, None, KeyStub.KEY_R, 0, KeyStub.PRESS, 0)
        sent = [json.loads(call.args[0]) for call in control.write.call_args_list]
        self.assertIn({'kernels': 'reload'}, sent)

    def test_sender_kernel_messages_show_in_the_notice_not_as_json(self):
        gui = _gui(self.devices)
        gui.events.put(('line', json.dumps({
            'status': 'kernel', 'message': 'DCT kernel antiring skipped: boom'})))
        gui._drain_events()
        self.assertEqual(gui.notice, 'DCT kernel antiring skipped: boom')
        self.assertEqual(gui.lines, ['DCT kernel antiring skipped: boom'])

    def test_the_sending_page_offers_the_kernel_and_steps_it_by_key(self):
        gui = _gui(self.devices)
        gui._assign('dct_kernel', 'lanczos')
        control = Mock()
        gui.process = SimpleNamespace(stdin=control)
        gui.page, gui.selected = 'live', 'dct_kernel'
        live = gui._live_fields()
        self.assertEqual(live[:2], ['brightness', 'gamma'])
        for dest in ('dct_kernel', 'kp:width', 'dct_sharpen_strength'
                     if gui.settings['dct_sharpen'] != 'off' else 'dct_clarity'):
            self.assertIn(dest, live)
        gui._on_key(KeyStub, None, KeyStub.KEY_DOWN, 0, KeyStub.PRESS, 0)
        self.assertEqual(gui.selected, live[live.index('dct_kernel')+1])
        gui.selected = 'kp:width'
        gui._on_key(KeyStub, None, KeyStub.KEY_LEFT, 0, KeyStub.PRESS, 0)
        self.assertEqual(json.loads(control.write.call_args.args[0])
                         ['kernel_params']['width'], 0.85)
        gui._on_key(KeyStub, None, KeyStub.KEY_R, 0, KeyStub.PRESS, 0)
        sent = [json.loads(c.args[0]) for c in control.write.call_args_list]
        self.assertIn({'kernels': 'reload'}, sent)

    def test_the_kernel_can_be_picked_with_the_mouse_on_the_live_page(self):
        from PIL import Image, ImageDraw
        gui = _gui(self.devices)
        gui.page = 'live'
        image = Image.new('RGB', (1100, 800))
        gui._render_live(image, ImageDraw.Draw(image), gui_module._font(16),
                         gui_module._font(13))
        left, top, right, bottom = gui.hits['field:dct_kernel']
        glfw = SimpleNamespace(
            PRESS=1, MOUSE_BUTTON_LEFT=0, MOUSE_BUTTON_RIGHT=1,
            get_cursor_pos=lambda _w: ((left+right)/2, (top+bottom)/2))
        gui._on_mouse(glfw, None, 0, 1, 0)
        self.assertEqual(gui.dropdown, 'dct_kernel')
        image = Image.new('RGB', (1100, 800))
        draw = ImageDraw.Draw(image)
        gui._render_dropdown(draw, gui_module._font(13), 1100, 800)
        option = gui.hits['option:2']
        glfw.get_cursor_pos = lambda _w: ((option[0]+option[2])/2,
                                          (option[1]+option[3])/2)
        gui._on_mouse(glfw, None, 0, 1, 0)
        names = [name for _label, name in gui._choices('dct_kernel')]
        self.assertEqual(gui.settings['dct_kernel'], names[2])
        self.assertIsNone(gui.dropdown)

    def test_the_sending_page_hides_the_kernel_when_pixel_encode_is_on(self):
        gui = _gui(self.devices)
        gui.settings['pixel_encode'] = True
        self.assertEqual(gui._live_fields(), ['brightness', 'gamma'])

    def test_the_pre_shrink_is_its_own_live_setting_for_any_kernel(self):
        gui = _gui(self.devices)
        self.assertIn('dct_preshrink', gui._visible_fields())
        self.assertNotIn('--dct-preshrink', self.command(gui))        # 4: shipped
        gui._assign('dct_kernel', 'lanczos')
        control = Mock()
        gui.process = SimpleNamespace(stdin=control)
        gui.page, gui.selected = 'live', 'dct_preshrink'
        self.assertIn('dct_preshrink', gui._live_fields())
        gui._on_key(KeyStub, None, KeyStub.KEY_RIGHT, 0, KeyStub.PRESS, 0)
        sent = json.loads(control.write.call_args.args[0])
        self.assertEqual(sent['dct']['preshrink'], 4.25)
        self.assertEqual(sent['kernel'], 'lanczos')
        gui.process = None
        gui.settings['dct_preshrink'] = '6'
        command = self.command(gui)
        self.assertEqual(command[command.index('--dct-preshrink')+1], '6.0')
        gui.settings['dct_preshrink'] = '12'
        with self.assertRaisesRegex(ValueError, 'pre-shrink'):
            self.command(gui)
        gui.settings['dct_preshrink'] = '6'
        gui.settings['pixel_encode'] = True
        self.assertNotIn('dct_preshrink', gui._visible_fields())

    # ---- defaults, reset, A/B, grouping, folding, speed ---------------------
    def press(self, gui, key, mods=0):
        gui._on_key(KeyStub, None, key, 0, KeyStub.PRESS, mods)

    def test_rows_say_what_their_default_is_and_mark_changes(self):
        gui = _gui(self.devices)
        gui._assign('dct_kernel', 'lanczos')
        self.assertTrue(gui._value_label('kp:width').endswith('default'))
        gui._set_kernel_param('kp:width', 0.8)
        self.assertEqual(gui._value_label('kp:width'), '0.8  ·  default 0.9')
        self.assertFalse(gui._is_default('kp:width'))
        self.assertIn('default Reference', gui._value_label('dct_kernel'))
        gui.settings['dct_preshrink'] = '6'
        self.assertEqual(gui._value_label('dct_preshrink'), '6  ·  default 4')
        self.assertIn('Default: 4', gui._help_text('dct_preshrink'))
        self.assertIn('Delete', gui._help_text('kp:width'))

    def test_delete_resets_the_selected_row_and_tells_a_running_sender(self):
        gui = _gui(self.devices)
        gui._assign('dct_kernel', 'lanczos')
        gui._set_kernel_param('kp:width', 0.8)
        gui.settings['dct_preshrink'] = '6'
        control = Mock()
        gui.process = SimpleNamespace(stdin=control)
        gui.selected = 'kp:width'
        self.press(gui, KeyStub.KEY_DELETE)
        self.assertEqual(gui._field_value('kp:width'), 0.9)
        sent = json.loads(control.write.call_args_list[-1].args[0])
        self.assertEqual(sent['kernel_params']['width'], 0.9)
        self.assertEqual(sent['dct']['preshrink'], 6.0)         # others kept
        gui.selected = 'dct_preshrink'
        self.press(gui, KeyStub.KEY_BACKSPACE)
        self.assertEqual(gui.settings['dct_preshrink'], '4')
        self.press(gui, KeyStub.KEY_BACKSPACE)
        self.assertIn('already', gui.notice)

    def test_the_defaults_button_and_key_reset_every_tweak(self):
        gui = _gui(self.devices)
        gui._assign('dct_kernel', 'mitchell')
        gui._set_kernel_param('kp:B', 0.1)
        gui.settings.update(dct_preshrink='7', dct_clarity='0.4', gamma='1.2',
                            brightness='0.8', dct_sharpen='usm')
        control = Mock()
        gui.process = SimpleNamespace(stdin=control)
        self.press(gui, KeyStub.KEY_D)
        self.assertEqual(gui.settings['dct_kernel'], 'reference')
        self.assertEqual((gui.settings['dct_preshrink'], gui.settings['dct_clarity'],
                          gui.settings['gamma'], gui.settings['brightness'],
                          gui.settings['dct_sharpen']), ('4', '0', '1', '', 'off'))
        self.assertEqual(gui.settings['dct_kernel_params'], {})
        payloads = [json.loads(c.args[0]) for c in control.write.call_args_list]
        self.assertIn({'brightness': 1.0, 'gamma': 1.0}, payloads)
        self.assertEqual(payloads[-1]['kernel'], 'reference')
        # the toolbar button does the same
        gui.settings['dct_clarity'] = '0.3'
        gui.page = 'live'
        gui._canvas((1100, 800))
        left, top, right, bottom = gui.hits['reset_tweaks']
        glfw = SimpleNamespace(
            PRESS=1, MOUSE_BUTTON_LEFT=0, MOUSE_BUTTON_RIGHT=1,
            get_cursor_pos=lambda _w: ((left+right)/2, (top+bottom)/2))
        gui._on_mouse(glfw, None, 0, 1, 0)
        self.assertEqual(gui.settings['dct_clarity'], '0')

    def test_ab_shows_the_defaults_and_brings_your_settings_back(self):
        gui = _gui(self.devices)
        gui._assign('dct_kernel', 'lanczos')
        gui._set_kernel_param('kp:width', 0.8)
        gui.settings['dct_preshrink'] = '6'
        control = Mock()
        gui.process = SimpleNamespace(stdin=control)
        self.press(gui, KeyStub.KEY_B)
        self.assertEqual(gui.settings['dct_kernel'], 'reference')
        self.assertEqual(json.loads(control.write.call_args.args[0])['kernel'],
                         'reference')
        self.assertIn('A/B', gui._status_facts())
        # edits are refused while B is shown, so nothing of A can be lost
        gui.selected = 'dct_preshrink'
        gui._step_field('dct_preshrink', 1)
        gui._assign('dct_clarity', '0.5')
        self.assertEqual(gui.settings['dct_preshrink'], '4')
        self.assertEqual(gui.settings['dct_clarity'], '0')
        self.assertIn('A/B', gui.notice)
        # preferences keep A, not the defaults shown for the comparison
        saved = _serialize_sender_settings({**gui.settings, **gui._ab_stash})
        self.assertEqual(saved['dct_kernel'], 'lanczos')
        self.press(gui, KeyStub.KEY_B)
        self.assertEqual(gui.settings['dct_kernel'], 'lanczos')
        self.assertEqual(gui.settings['dct_preshrink'], '6')
        self.assertEqual(gui._field_value('kp:width'), 0.8)
        payload = json.loads(control.write.call_args.args[0])
        self.assertEqual(payload['kernel_params']['width'], 0.8)
        self.assertIsNone(gui._ab_stash)

    def test_the_sending_screen_groups_what_belongs_together(self):
        gui = _gui(self.devices)
        gui._assign('dct_kernel', 'lanczos')
        titles = [title for title, _fields in gui._live_groups()]
        self.assertEqual(titles, ['Tone', 'Downscale', 'Sharpness'])
        groups = dict(gui._live_groups())
        self.assertEqual(groups['Tone'], ['brightness', 'gamma'])
        self.assertEqual(groups['Downscale'][0], 'dct_kernel')
        self.assertIn('kp:width', groups['Downscale'])
        self.assertEqual(groups['Downscale'][-1], 'dct_preshrink')
        self.assertEqual(groups['Sharpness'][0], 'dct_sharpen')
        gui.settings['pixel_encode'] = True
        self.assertEqual([t for t, _ in gui._live_groups()], ['Tone'])

    def test_sections_fold_and_the_choice_is_remembered(self):
        gui = _gui(self.devices)
        self.assertIn('device', gui._visible_fields())
        gui._canvas((1100, 800))
        self.assertIn('section:Output', gui.hits)
        self.assertNotIn('section:Live controls', gui.hits)
        left, top, right, bottom = gui.hits['section:Output']
        glfw = SimpleNamespace(
            PRESS=1, MOUSE_BUTTON_LEFT=0, MOUSE_BUTTON_RIGHT=1,
            get_cursor_pos=lambda _w: ((left+right)/2, (top+bottom)/2))
        gui._on_mouse(glfw, None, 0, 1, 0)
        self.assertNotIn('device', gui._visible_fields())
        self.assertEqual(gui.settings['collapsed_sections'], ['Output'])
        items = gui._setup_items(1100)
        self.assertIn(('header', 'Output'), items)
        saved = json.loads(json.dumps(_serialize_sender_settings(gui.settings)))
        fresh = _gui(self.devices)
        _restore_sender_settings(fresh.settings, saved)
        self.assertNotIn('device', fresh._visible_fields())
        _restore_sender_settings(fresh.settings, {'collapsed_sections': 'junk'})
        self.assertEqual(fresh.settings['collapsed_sections'], ['Output'])
        gui._toggle_section('Output')
        self.assertIn('device', gui._visible_fields())

    def test_the_toolbar_says_whether_start_will_work(self):
        gui = _gui(self.devices)
        gui._sd = self.sd
        ready, text = gui._readiness()
        self.assertTrue(ready, text)
        gui.settings['device'] = None
        ready, text = gui._readiness()
        self.assertFalse(ready)
        self.assertTrue(text)
        gui.settings['device'] = 3
        self.assertTrue(gui._readiness()[0])

    def test_cached_text_draws_the_same_picture_as_plain_drawing(self):
        import numpy as np
        from PIL import Image, ImageDraw
        font = gui_module._font(13)
        plain = Image.new('RGBA', (240, 60), (8, 14, 20, 255))
        fast = plain.copy()
        ImageDraw.Draw(plain).text((10, 8), 'DCT kernel · live', font=font,
                                   fill=(205, 218, 228))
        drawer = gui_module._CachedDraw(fast)
        drawer.text((10, 8), 'DCT kernel · live', font=font, fill=(205, 218, 228))
        difference = np.abs(np.asarray(plain, int) - np.asarray(fast, int))
        self.assertLess(difference.max(), 12)
        # the second use comes from the cache and draws the same again
        again = Image.new('RGBA', (240, 60), (8, 14, 20, 255))
        gui_module._CachedDraw(again).text(
            (10, 8), 'DCT kernel · live', font=font, fill=(205, 218, 228))
        np.testing.assert_array_equal(np.asarray(fast), np.asarray(again))

    def test_a_repaint_reuses_the_scaled_preview(self):
        from PIL import Image, ImageDraw
        gui = _gui(self.devices)
        gui.settings['preview'] = 'window'
        gui.page = 'live'
        gui.preview_image = Image.new('RGB', (480, 640), (90, 120, 60))
        gui.preview_counter, gui.preview_aspect = 1, 1
        gui.preview_handoff_ns = 0
        image = Image.new('RGBA', (1100, 800))
        gui._render_live(image, gui_module._CachedDraw(image),
                         gui_module._font(16), gui_module._font(13))
        first = gui._thumbnail_cache[2]
        gui._render_live(image, gui_module._CachedDraw(image),
                         gui_module._font(16), gui_module._font(13))
        self.assertIs(gui._thumbnail_cache[2], first)
        gui.preview_image = Image.new('RGB', (480, 640), (10, 10, 10))
        gui._render_live(image, gui_module._CachedDraw(image),
                         gui_module._font(16), gui_module._font(13))
        self.assertIsNot(gui._thumbnail_cache[2], first)

    def test_the_help_line_describes_the_selected_row(self):
        gui = _gui(self.devices)
        gui._assign('dct_kernel', 'lanczos')
        self.assertIn('lanczos.py', gui._help_text('dct_kernel'))
        self.assertIn('kernel width', gui._help_text('kp:width'))
        self.assertIn('Left/Right', gui._help_text('kp:width'))


if __name__ == '__main__':
    unittest.main()
