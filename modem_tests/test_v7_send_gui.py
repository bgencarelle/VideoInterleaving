"""The sender GUI builds valid CLI settings without opening an audio stream."""
import unittest
import signal
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tools import v7_live
from tools.v7_send_gui import (InputDevice, OutputDevice,
                                PRIMARY_PROFILE_CHOICES, ScreenTarget, SenderGui,
                                build_command, enumerate_screen_targets,
                                enumerate_camera_sources, linux_camera_sources,
                                enumerate_capture_fps, parse_capture_fps,
                                input_devices, output_devices,
                                parse_avfoundation_screen_sources,
                                parse_ffmpeg_camera_sources, pick_video_file,
                                device_rate_text, validate_settings,
                                _clipboard_text)


class SenderKeyStub:
    PRESS = 1
    REPEAT = 2
    KEY_ESCAPE = 256
    KEY_ENTER = 257
    KEY_TAB = 258
    KEY_BACKSPACE = 259
    KEY_RIGHT = 262
    KEY_LEFT = 263
    KEY_DOWN = 264
    KEY_UP = 265
    KEY_C = 67
    KEY_I = 73
    KEY_SPACE = 32
    KEY_A = 65
    KEY_V = 86
    KEY_KP_ENTER = 335
    MOD_CONTROL = 2


class SenderGuiTests(unittest.TestCase):
    def setUp(self):
        self.device = OutputDevice(3, 'Test output', 2, 48000)
        self.devices = (self.device,)
        self.settings = {
            'device': 3,
            'source': 'screen',
            'rate': None,
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
            'video_live': False,
            'camera': None,
            'ffmpeg_input': '',
            'screen_backend': 'mss',
            'screen_target': ScreenTarget(
                'Display 1 · 1920×1080 · (0,0)', '0,0,1920,1080'),
            'region': '',
            'capture_width': '160',
            'capture_filter': 'auto',
            'perceptual_resize': 'off',
            'perceptual_detail_strength': '0.25',
        }
        self.sd = Mock()
        self.sd.check_output_settings.return_value = None

    def test_colour_mono_profile_is_available_in_primary_picker(self):
        self.assertEqual(
            tuple(value for _label, value in PRIMARY_PROFILE_CHOICES),
            ('fold-500', 'mono-fold-500', 'mono-colour-500'))

    def test_device_list_contains_only_output_devices_and_does_not_default(self):
        sd = Mock()
        sd.query_devices.return_value = [
            {'name': 'Input only', 'max_input_channels': 2,
             'max_output_channels': 0, 'default_samplerate': 48000},
            {'name': 'Output', 'max_input_channels': 0,
             'max_output_channels': 2, 'default_samplerate': 96000},
        ]

        devices = output_devices(sd)

        self.assertEqual([device.name for device in devices], ['Output'])
        self.assertEqual(devices[0].index, 1)
        gui = SenderGui(devices)
        self.assertIsNone(gui.settings['device'])

    def test_input_device_picker_lists_only_input_capable_devices(self):
        sd = Mock()
        sd.query_devices.return_value = [
            {'name': 'Input', 'max_input_channels': 2,
             'max_output_channels': 0, 'default_samplerate': 48000},
            {'name': 'Output', 'max_input_channels': 0,
             'max_output_channels': 2, 'default_samplerate': 96000},
        ]

        devices = input_devices(sd)

        self.assertEqual([(item.index, item.channels) for item in devices],
                         [(0, 2)])
        self.assertIsInstance(devices[0], InputDevice)

    def test_sender_uses_the_output_rate_the_os_is_set_to(self):
        # A rate left in older saved settings must not reach the CLI.
        self.settings['rate'] = '96000'
        command = build_command(self.settings, self.devices, self.sd)

        self.assertNotIn('--rate', command)
        kwargs = self.sd.check_output_settings.call_args.kwargs
        self.assertEqual(kwargs['device'], 3)
        self.assertEqual(kwargs['channels'], 2)
        self.assertEqual(kwargs['samplerate'], self.device.default_rate)
        self.assertEqual(device_rate_text(self.device),
                         f'{self.device.default_rate/1000:g} kHz · OS setting')
        self.assertEqual(device_rate_text(None), 'OS setting')
        self.assertNotIn('rate', SenderGui(self.devices)._visible_fields())

    def test_folded_default_builds_sender_cli_without_verbose_logging(self):
        command = build_command(self.settings, self.devices, self.sd)
        args = v7_live.parser().parse_args(command[2:])

        self.assertEqual(args.mode, 'send')
        self.assertEqual(args.device, 3)
        self.assertEqual(args.source, 'screen')
        self.assertEqual(args.profile, 'fold-500')
        self.assertIsNone(args.experimental_fold)
        self.assertEqual(args.encode_filter, None)
        self.assertFalse(args.log)
        self.assertNotIn('--rate', command)

    def test_selected_sample_rate_profile_and_video_path_reach_cli(self):
        self.settings.update(source='video', video_source='a clip with spaces.mp4',
                             video_live=True, video_preview=True,
                             profile='fold-1000', speed='1.5')

        command = build_command(self.settings, self.devices, self.sd)
        args = v7_live.parser().parse_args(command[2:])

        self.assertEqual(args.profile, 'fold-1000')
        self.assertIsNone(args.rate)
        self.assertEqual(args.video_source, 'a clip with spaces.mp4')
        self.assertTrue(args.video_live)
        self.assertTrue(args.preview)
        self.assertIn('--preview', command)
        self.assertEqual(args.speed, 1.5)

    def test_preview_is_not_forwarded_for_non_video_sources(self):
        self.settings.update(source='screen', video_preview=True)
        command = build_command(self.settings, self.devices, self.sd)
        self.assertNotIn('--preview', command)

    def test_mono_video_fold_profile_reaches_cli_and_uses_box_model(self):
        self.settings.update(profile='mono-fold-500', mono_video_side='right')
        command = build_command(self.settings, self.devices, self.sd)
        args = v7_live.parser().parse_args(command[2:])

        self.assertEqual(args.profile, 'mono-fold-500')
        self.assertFalse(args.experimental_mono_fold)
        self.assertEqual(args.mono_video_side, 'right')
        self.assertIsNone(args.experimental_fold)
        self.assertIsNone(args.encode_filter)
        self.assertIn('--profile', command)
        self.assertNotIn('--experimental-mono-fold', command)
        self.assertEqual(args.source_audio, 'source')

    def test_mono_colour_profile_reaches_sender_cli_as_a_profile_choice(self):
        self.settings.update(profile='mono-colour-500', mono_video_side='left')

        command = build_command(self.settings, self.devices, self.sd)
        args = v7_live.parser().parse_args(command[2:])

        v7_live._apply_profile_option(args)

        self.assertEqual(args.profile, 'mono-colour-500')
        self.assertEqual(args.mono_video_side, 'left')
        self.assertTrue(args.experimental_mono_fold)
        self.assertTrue(args.experimental_mono_colour)
        self.assertEqual(command[command.index('--profile')+1],
                         'mono-colour-500')
        self.assertNotIn('--experimental-mono-colour', command)

    def test_mono_video_can_route_an_explicit_audio_input_device(self):
        audio_device = InputDevice(8, 'Loopback', 2, 48000)
        self.settings.update(
            profile='mono-fold-500', source='video',
            video_source='clip.mp4', source_audio='device',
            source_audio_device=8, source_audio_input_side='left',
            source_audio_gain='0.7', source_audio_delay_ms='12')

        command = build_command(
            self.settings, self.devices, self.sd,
            audio_devices=(audio_device,))
        args = v7_live.parser().parse_args(command[2:])

        self.assertEqual(args.mono_video_side, 'right')
        self.assertEqual(args.source_audio, 'device')
        self.assertEqual(args.source_audio_device, 8)
        self.assertEqual(args.source_audio_input_side, 'left')
        self.assertEqual(args.source_audio_gain, .7)
        self.assertEqual(args.source_audio_delay_ms, 12)
        self.sd.check_input_settings.assert_called_once()

    def test_mono_video_audio_device_is_available_for_every_capture_source(self):
        gui = SenderGui(self.devices)
        gui.settings.update(profile='mono-fold-500', source_audio='device')
        for _label, source in (
                ('Camera', 'camera'), ('Screen', 'screen'),
                ('Video', 'video'), ('Test', 'test'),
                ('Mouse-follow', 'mouse-follow')):
            with self.subTest(source=source):
                gui.settings['source'] = source
                self.assertIn('source_audio', gui._visible_fields())
                self.assertIn('source_audio_device', gui._visible_fields())

    def test_video_audio_source_and_input_pickers_are_clickable(self):
        audio_device = InputDevice(8, 'Loopback', 2, 48000)
        gui = SenderGui(self.devices, audio_devices=(audio_device,))
        gui.settings.update(source='video', video_source='clip.mp4',
                            profile='mono-fold-500')

        def click(key):
            rect = gui.hits[key]
            position = ((rect[0]+rect[2])/2, (rect[1]+rect[3])/2)
            glfw = SimpleNamespace(
                MOUSE_BUTTON_LEFT=1, PRESS=1,
                get_cursor_pos=lambda _window: position)
            gui._on_mouse(glfw, None, 1, 1, 0)

        gui._canvas((960, 720))
        self.assertIn('field:source_audio', gui.hits)
        click('field:source_audio')
        self.assertEqual(gui.dropdown, 'source_audio')
        gui._canvas((960, 720))
        click('option:1')
        self.assertEqual(gui.settings['source_audio'], 'device')

        gui._canvas((960, 720))
        self.assertIn('field:source_audio_device', gui.hits)
        self.assertIn('field:source_audio_input_side', gui.hits)
        with patch('tools.v7_send_gui.input_devices',
                   return_value=(audio_device,)):
            click('field:source_audio_device')
        self.assertEqual(gui.dropdown, 'source_audio_device')
        gui._canvas((960, 720))
        click('option:0')
        self.assertEqual(gui.settings['source_audio_device'], 8)

        gui._canvas((960, 720))
        click('field:source_audio_input_side')
        self.assertEqual(gui.dropdown, 'source_audio_input_side')
        gui._canvas((960, 720))
        click('option:1')
        self.assertEqual(gui.settings['source_audio_input_side'], 'left')

    def test_output_picker_refreshes_hotplugged_devices_without_defaulting(self):
        gui = SenderGui(())
        new_device = OutputDevice(7, 'New interface', 2, 48000)

        with patch('tools.v7_send_gui.output_devices',
                   return_value=(new_device,)), \
                patch.object(gui, '_sounddevice', return_value=object()):
            gui._open_dropdown('device')

        self.assertEqual(gui.devices, (new_device,))
        self.assertEqual(gui._choices('device'),
                         (('7: New interface · 2 out · 48 kHz', 7),))
        self.assertIsNone(gui.settings['device'])
        self.assertEqual(gui.dropdown, 'device')

    def test_background_device_refresh_adds_devices_without_fallback(self):
        gui = SenderGui(())
        gui.settings.update(source='camera',
                            camera='avfoundation:Disconnected camera')
        output = OutputDevice(7, 'New interface', 2, 48000)
        audio_input = InputDevice(8, 'New loopback', 2, 48000)
        cameras = (('Capture card', 'avfoundation:Capture card'),)

        gui._apply_device_snapshot(((output,), '', (audio_input,), '',
                                    cameras, None))

        self.assertEqual(gui._choices('device'),
                         (('7: New interface · 2 out · 48 kHz', 7),))
        self.assertEqual(gui._choices('source_audio_device'),
                         (('8: New loopback · 2 in · 48 kHz', 8),))
        self.assertIsNone(gui.settings['device'])
        self.assertEqual(gui.settings['camera'],
                         'avfoundation:Disconnected camera')
        self.assertEqual(gui._choices('camera'), cameras)
        self.assertIn('unavailable', gui.notice)

    def test_advanced_downscaler_picker_is_visible_and_selectable(self):
        gui = SenderGui(self.devices)
        gui.settings.update(source='video', profile='fold-500')
        gui.advanced = True
        gui._canvas((960, 720))
        self.assertIn('field:perceptual_resize', gui.hits)

        rect = gui.hits['field:perceptual_resize']
        position = ((rect[0]+rect[2])/2, (rect[1]+rect[3])/2)
        glfw = SimpleNamespace(
            MOUSE_BUTTON_LEFT=1, PRESS=1,
            get_cursor_pos=lambda _window: position)
        gui._on_mouse(glfw, None, 1, 1, 0)
        self.assertEqual(gui.dropdown, 'perceptual_resize')

        gui._canvas((960, 720))
        options = gui._choices('perceptual_resize')
        option_index = next(index for index, item in enumerate(options)
                            if item[1] == 'gamma-detail')
        rect = gui.hits[f'option:{option_index}']
        position = ((rect[0]+rect[2])/2, (rect[1]+rect[3])/2)
        gui._on_mouse(glfw, None, 1, 1, 0)
        self.assertEqual(gui.settings['perceptual_resize'], 'gamma-detail')

    def test_mono_video_side_selector_is_visible_only_for_that_profile(self):
        gui = SenderGui(self.devices)
        self.assertNotIn('mono_video_side', gui._visible_fields())

        gui.settings.update(profile='mono-fold-500', mono_video_side='right')
        self.assertIn('mono_video_side', gui._visible_fields())
        self.assertIn('source_audio', gui._visible_fields())
        self.assertNotIn('source_audio_device', gui._visible_fields())
        self.assertEqual(gui._value_label('mono_video_side'),
                         'Right output · left stays clear')

        gui.settings['profile'] = 'fold-500'
        self.assertNotIn('mono_video_side', gui._visible_fields())
        gui.settings['profile'] = 'mono-fold-500'
        gui.settings['source_audio'] = 'device'
        self.assertIn('source_audio_device', gui._visible_fields())

    def test_mono_video_keeps_wire_defaults_and_allows_pre_resize(self):
        self.settings.update(profile='mono-fold-500', perceptual_resize='linear-box')
        command = build_command(self.settings, self.devices, self.sd)
        self.assertEqual(command[command.index('--perceptual-resize')+1],
                         'linear-box')

        self.assertNotIn('--mono-sum', command)
        self.assertNotIn('--no-pilot-tones', command)
        self.assertNotIn('--no-eof-marker', command)

    def test_ffmpeg_screen_input_is_forwarded_only_for_ffmpeg_capture(self):
        self.settings.update(source='screen', screen_backend='ffmpeg',
                             ffmpeg_input='x11grab::0.0')

        command = build_command(self.settings, self.devices, self.sd)
        args = v7_live.parser().parse_args(command[2:])

        self.assertEqual(args.screen_backend, 'ffmpeg')
        self.assertEqual(args.ffmpeg_input, 'x11grab::0.0')

        self.settings['screen_target'] = ScreenTarget(
            'Capture screen 1', display=1)
        with self.assertRaisesRegex(ValueError, 'display from the picker'):
            build_command(self.settings, self.devices, self.sd)

    def test_camera_picker_values_and_screen_targets_map_to_sender_cli(self):
        self.settings.update(
            source='camera', camera='avfoundation:HDMI Capture Card',
            screen_target=None)
        command = build_command(self.settings, self.devices, self.sd)
        args = v7_live.parser().parse_args(command[2:])
        self.assertEqual(args.ffmpeg_input,
                         'avfoundation:HDMI Capture Card')
        self.assertNotIn('--camera', command)

        self.settings.update(source='camera', camera='v4l2:/dev/video2',
                             screen_target=None)
        command = build_command(self.settings, self.devices, self.sd)
        args = v7_live.parser().parse_args(command[2:])
        self.assertEqual(args.ffmpeg_input, 'v4l2:/dev/video2')
        self.assertNotIn('--camera', command)

        self.settings.update(source='screen', screen_backend='mss',
                             screen_target=ScreenTarget(
                                 'Display 2', '-1920,0,1920,1080'))
        command = build_command(self.settings, self.devices, self.sd)
        args = v7_live.parser().parse_args(command[2:])
        self.assertEqual(args.region, '-1920,0,1920,1080')

        self.settings.update(screen_backend='ffmpeg', ffmpeg_input='',
                             screen_target=ScreenTarget(
                                 'Capture screen 1', display=1))
        command = build_command(self.settings, self.devices, self.sd)
        args = v7_live.parser().parse_args(command[2:])
        self.assertEqual(args.display, 1)

    def test_custom_camera_input_cannot_silently_override_picker_selection(self):
        self.settings.update(
            source='camera', camera='avfoundation:HDMI Capture Card',
            ffmpeg_input='avfoundation:0')

        with self.assertRaisesRegex(ValueError, 'camera from the picker'):
            build_command(self.settings, self.devices, self.sd)

        self.settings['camera'] = None
        command = build_command(self.settings, self.devices, self.sd)
        args = v7_live.parser().parse_args(command[2:])
        self.assertEqual(args.ffmpeg_input, 'avfoundation:0')

    def test_downscaler_is_forwarded_and_constrained(self):
        self.settings.update(source='video', video_source='clip.mp4',
                             perceptual_resize='gamma-detail',
                             perceptual_detail_strength='0.6')
        command = build_command(self.settings, self.devices, self.sd)
        args = v7_live.parser().parse_args(command[2:])
        self.assertEqual(args.perceptual_resize, 'gamma-detail')
        self.assertEqual(args.perceptual_detail_strength, 0.6)

        self.settings.update(profile='mono-fold-500')
        command = build_command(self.settings, self.devices, self.sd)
        self.assertEqual(command[command.index('--perceptual-resize')+1],
                         'gamma-detail')

    def test_capture_device_discovery_parsers(self):
        cameras = linux_camera_sources(
            ['/dev/video10', '/dev/video2', '/dev/notvideo'],
            {'/dev/video2': 'USB Camera'})
        self.assertEqual(cameras, (
            ('USB Camera · /dev/video2', 'v4l2:/dev/video2'),
            ('video10 · /dev/video10', 'v4l2:/dev/video10'),
        ))

        listing = ('[AVFoundation indev @ 0x1] AVFoundation video devices:\n'
                   '[AVFoundation indev @ 0x1] [0] Face Camera\n'
                   '[AVFoundation indev @ 0x1] [1] Capture Screen Pro\n'
                   '[AVFoundation indev @ 0x1] [2] USB Screen Capture\n'
                   '[AVFoundation indev @ 0x1] [3] Capture screen 0\n'
                   '[AVFoundation indev @ 0x1] [4] HDMI Capture Card\n'
                   '[AVFoundation indev @ 0x1] AVFoundation audio devices:\n'
                   '[AVFoundation indev @ 0x1] [0] Microphone')
        self.assertEqual(parse_ffmpeg_camera_sources(listing, 'darwin'), (
            ('Face Camera · AVFoundation 0', 'avfoundation:Face Camera'),
            ('Capture Screen Pro · AVFoundation 1',
             'avfoundation:Capture Screen Pro'),
            ('USB Screen Capture · AVFoundation 2',
             'avfoundation:USB Screen Capture'),
            ('HDMI Capture Card · AVFoundation 4',
             'avfoundation:HDMI Capture Card')))
        self.assertEqual(parse_avfoundation_screen_sources(listing), (
            ScreenTarget('Capture screen 0', display=3),))

        directshow = ('[dshow @ 0x1] DirectShow video devices\n'
                      '[dshow @ 0x1] "USB Camera" (video)\n'
                      '[dshow @ 0x1]   Alternative name "@device_pnp_\\\\..."\n'
                      '[dshow @ 0x1] DirectShow audio devices\n'
                      '[dshow @ 0x1] "Microphone" (audio)')
        self.assertEqual(parse_ffmpeg_camera_sources(directshow, 'win32'), (
            ('USB Camera', 'dshow:video=USB Camera'),))

    def test_linux_camera_discovery_reads_device_names_without_consuming_nodes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dev_root = root/'dev'
            sys_root = root/'sys'
            dev_root.mkdir()
            (dev_root/'video0').touch()
            name_path = sys_root/'video0'/'name'
            name_path.parent.mkdir(parents=True)
            name_path.write_text('USB Capture', encoding='utf-8')

            choices = enumerate_camera_sources(
                platform='linux', dev_root=dev_root, sys_root=sys_root)

        self.assertEqual(choices, (
            ('USB Capture · '+str(dev_root/'video0'),
             'v4l2:'+str(dev_root/'video0')),
        ))

    def test_screen_enumeration_is_closed_after_listing_monitors(self):
        capture = SimpleNamespace(
            monitors=[{'left': 0, 'top': 0, 'width': 3000, 'height': 1080},
                      {'left': 0, 'top': 0, 'width': 1920, 'height': 1080},
                      {'left': 1920, 'top': 0, 'width': 1080, 'height': 1080}],
            close=Mock())
        module = SimpleNamespace(MSS=Mock(return_value=capture))

        targets = enumerate_screen_targets(mss_module=module, platform='linux')

        self.assertEqual(len(targets), 3)
        self.assertEqual(targets[0].region, '0,0,3000,1080')
        self.assertEqual(targets[2].region, '1920,0,1080,1080')
        capture.close.assert_called_once_with()

    def test_screen_target_picker_opens_and_selects_screen_targets(self):
        gui = SenderGui(self.devices)
        gui.settings['source'] = 'screen'
        targets = (
            ScreenTarget('Display 1 · 1920×1080 · (0,0)',
                         region='0,0,1920,1080'),
            ScreenTarget('Display 2 · 1280×720 · (1920,0)',
                         region='1920,0,1280,720'),
        )

        with patch('tools.v7_send_gui.enumerate_screen_targets',
                   return_value=targets):
            gui._open_dropdown('screen_target')
            self.assertEqual(gui.dropdown_scroll, 0)
            self.assertEqual(
                gui._choices('screen_target'),
                tuple((target.label, target) for target in targets))
            gui._on_key(SenderKeyStub, None, SenderKeyStub.KEY_DOWN, 0,
                        SenderKeyStub.PRESS, 0)
            gui._on_key(SenderKeyStub, None, SenderKeyStub.KEY_ENTER, 0,
                        SenderKeyStub.PRESS, 0)

        self.assertEqual(gui.settings['screen_target'], targets[1])
        self.assertEqual(gui._value_label('screen_target'), targets[1].label)

    def test_capture_picker_enumerates_only_when_opened(self):
        gui = SenderGui(self.devices)
        gui.settings.update(source='camera')
        with patch('tools.v7_send_gui.enumerate_camera_sources',
                   return_value=(('USB camera', 'v4l2:/dev/video0'),)) as list_devices:
            self.assertEqual(gui._choices('camera'), ())
            list_devices.assert_not_called()
            gui._open_dropdown('camera')
            self.assertEqual(gui._choices('camera'),
                             (('USB camera', 'v4l2:/dev/video0'),))
            list_devices.assert_called_once_with()

    def test_empty_camera_picker_shows_a_notice_without_opening_blank_menu(self):
        gui = SenderGui(self.devices)
        gui.settings['source'] = 'camera'

        with patch('tools.v7_send_gui.enumerate_camera_sources',
                   return_value=()):
            gui._open_dropdown('camera')

        self.assertIsNone(gui.dropdown)
        self.assertIn('No camera sources', gui.notice)

    def test_keyboard_camera_picker_starts_at_selected_item_and_moves_focus(self):
        gui = SenderGui(self.devices)
        gui.settings.update(source='camera', camera='avfoundation:Camera 1')
        choices = (('Camera 0', 'avfoundation:Camera 0'),
                   ('Camera 1', 'avfoundation:Camera 1'),
                   ('Capture card', 'avfoundation:Capture card'))

        with patch('tools.v7_send_gui.enumerate_camera_sources',
                   return_value=choices):
            gui._open_dropdown('camera')
            self.assertEqual(gui.dropdown_scroll, 1)
            gui._on_key(SenderKeyStub, None, SenderKeyStub.KEY_ENTER, 0,
                        SenderKeyStub.PRESS, 0)
            self.assertEqual(gui.settings['camera'],
                             'avfoundation:Camera 1')

            gui._open_dropdown('camera')
            gui._on_key(SenderKeyStub, None, SenderKeyStub.KEY_DOWN, 0,
                        SenderKeyStub.PRESS, 0)
            gui._on_key(SenderKeyStub, None, SenderKeyStub.KEY_KP_ENTER, 0,
                        SenderKeyStub.PRESS, 0)

        self.assertEqual(gui.settings['camera'],
                         'avfoundation:Capture card')

    def test_camera_start_reenumerates_and_rejects_stale_picker_values(self):
        gui = SenderGui(self.devices)
        gui.settings.update(source='camera', camera='avfoundation:4')

        with patch('tools.v7_send_gui.enumerate_camera_sources',
                   return_value=(('HDMI Capture Card · AVFoundation 4',
                                  'avfoundation:HDMI Capture Card'),)) as scan, \
                patch.object(gui, '_build_command') as build_command:
            gui._start()

        scan.assert_called_once_with()
        build_command.assert_not_called()
        self.assertIsNone(gui.process)
        self.assertIn('reopen the camera picker', gui.notice)

    def test_camera_start_uses_freshly_enumerated_name_selection(self):
        gui = SenderGui(self.devices)
        gui.settings.update(source='camera',
                            camera='avfoundation:HDMI Capture Card')
        child = Mock()
        child.poll.return_value = None

        with patch('tools.v7_send_gui.enumerate_camera_sources',
                   return_value=(('HDMI Capture Card · AVFoundation 4',
                                  'avfoundation:HDMI Capture Card'),)) as scan, \
                patch.object(gui, '_build_command',
                             return_value=['python', 'send']) as build_command, \
                patch('tools.v7_send_gui.subprocess.Popen',
                      return_value=child), \
                patch('tools.v7_send_gui.threading.Thread') as thread:
            gui._start()

        scan.assert_called_once_with()
        build_command.assert_called_once_with()
        self.assertEqual(gui.page, 'live')
        thread.return_value.start.assert_called_once_with()

    def test_fractional_scroll_moves_sender_dropdown_and_zero_does_nothing(self):
        gui = SenderGui(self.devices)
        gui.dropdown = 'source'
        gui.dropdown_scroll = 2

        gui._on_scroll(None, 1, 0)
        self.assertEqual(gui.dropdown_scroll, 2)
        gui._on_scroll(None, 0, 0.5)

        self.assertEqual(gui.dropdown_scroll, 1)

    def test_video_and_camera_pick_controls_are_visible_and_clickable(self):
        gui = SenderGui(self.devices)
        gui.settings.update(source='video')
        gui._canvas((960, 720))
        self.assertIn('browse:video_source', gui.hits)
        rect = gui.hits['browse:video_source']
        position = ((rect[0]+rect[2])/2, (rect[1]+rect[3])/2)
        glfw = SimpleNamespace(
            MOUSE_BUTTON_LEFT=1, PRESS=1,
            get_cursor_pos=lambda _window: position)
        with patch('tools.v7_send_gui.pick_video_file',
                   return_value='/media/clip.mp4'):
            gui._on_mouse(glfw, None, 1, 1, 0)
        self.assertEqual(gui.settings['video_source'], '/media/clip.mp4')

        gui.settings.update(source='camera')
        gui._canvas((960, 720))
        self.assertIn('field:camera', gui.hits)
        rect = gui.hits['field:camera']
        position = ((rect[0]+rect[2])/2, (rect[1]+rect[3])/2)
        with patch('tools.v7_send_gui.enumerate_camera_sources',
                   return_value=(('USB camera', 'v4l2:/dev/video0'),)):
            gui._on_mouse(glfw, None, 1, 1, 0)
        self.assertEqual(gui.dropdown, 'camera')
        self.assertEqual(gui._choices('camera'),
                         (('USB camera', 'v4l2:/dev/video0'),))
        gui._canvas((960, 720))
        rect = gui.hits['option:0']
        position = ((rect[0]+rect[2])/2, (rect[1]+rect[3])/2)
        gui._on_mouse(glfw, None, 1, 1, 0)
        self.assertEqual(gui.settings['camera'], 'v4l2:/dev/video0')

    def test_video_source_accepts_control_v_and_command_v_paste(self):
        gui = SenderGui(self.devices)
        gui.settings['source'] = 'video'
        gui.selected = 'video_source'
        gui.editing = True

        for modifier in (SenderKeyStub.MOD_CONTROL, 8):
            with self.subTest(modifier=modifier):
                gui.edit_buffer = 'old-url'
                clipboard = Mock(
                    return_value='https://example.test/clip.mp4')
                glfw = SimpleNamespace(
                    PRESS=SenderKeyStub.PRESS,
                    REPEAT=SenderKeyStub.REPEAT,
                    KEY_ENTER=SenderKeyStub.KEY_ENTER,
                    KEY_KP_ENTER=SenderKeyStub.KEY_KP_ENTER,
                    KEY_ESCAPE=SenderKeyStub.KEY_ESCAPE,
                    KEY_BACKSPACE=SenderKeyStub.KEY_BACKSPACE,
                    KEY_A=SenderKeyStub.KEY_A,
                    KEY_V=SenderKeyStub.KEY_V,
                    MOD_CONTROL=SenderKeyStub.MOD_CONTROL,
                    MOD_SUPER=8,
                    get_version=lambda: (3, 4, 0),
                    get_clipboard_string=clipboard)

                gui._on_key(glfw, None, glfw.KEY_A, 0, glfw.PRESS, modifier)
                self.assertEqual(gui.edit_buffer, '')
                window = object()
                gui._on_key(glfw, window, glfw.KEY_V, 0, glfw.PRESS, modifier)

                self.assertEqual(gui.edit_buffer,
                                 'https://example.test/clip.mp4')
                clipboard.assert_called_once_with(None)
                gui._on_key(glfw, None, glfw.KEY_ENTER, 0,
                            glfw.PRESS, modifier)
                self.assertEqual(gui.settings['video_source'],
                                 'https://example.test/clip.mp4')
                gui.editing = True

    def test_right_click_pastes_into_video_source_field(self):
        gui = SenderGui(self.devices)
        gui.settings['source'] = 'video'
        gui._canvas((960, 720))
        rect = gui.hits['field:video_source']
        position = ((rect[0]+rect[2])/2, (rect[1]+rect[3])/2)
        clipboard = Mock(return_value=b'https://example.test/clip.mp4')
        glfw = SimpleNamespace(
            MOUSE_BUTTON_LEFT=1, MOUSE_BUTTON_RIGHT=2, PRESS=1,
            get_cursor_pos=lambda _window: position,
            get_version=lambda: (3, 4, 0),
            get_clipboard_string=clipboard)

        gui._on_mouse(glfw, object(), glfw.MOUSE_BUTTON_RIGHT,
                      glfw.PRESS, 0)

        self.assertTrue(gui.editing)
        self.assertEqual(gui.selected, 'video_source')
        self.assertEqual(gui.edit_buffer, 'https://example.test/clip.mp4')
        clipboard.assert_called_once_with(None)

    def test_clipboard_read_keeps_window_argument_for_legacy_glfw(self):
        window = object()
        clipboard = Mock(return_value='clipboard text')
        glfw = SimpleNamespace(
            get_version=lambda: (3, 3, 8),
            get_clipboard_string=clipboard)

        self.assertEqual(_clipboard_text(glfw, window), 'clipboard text')
        clipboard.assert_called_once_with(window)

    def test_clipboard_read_falls_back_to_window_when_null_lookup_is_empty(self):
        window = object()
        clipboard = Mock(side_effect=(None, 'clipboard text'))
        glfw = SimpleNamespace(
            get_version=lambda: (3, 4, 0),
            get_clipboard_string=clipboard)

        self.assertEqual(_clipboard_text(glfw, window), 'clipboard text')
        self.assertEqual(clipboard.call_args_list, [
            unittest.mock.call(None), unittest.mock.call(window)])

    def test_capture_fps_dropdown_uses_rates_reported_by_camera_driver(self):
        output = ('Interval: Discrete 0.033s (30.000 fps)\n'
                  'Interval: Discrete 0.017s (59.940 fps)\n')
        self.assertEqual(parse_capture_fps(output), (30.0, 59.94))
        run = Mock(return_value=SimpleNamespace(stdout=output, stderr=''))
        choices = enumerate_capture_fps(
            'camera', 'v4l2:/dev/video2', platform='linux',
            which=lambda name: '/usr/bin/v4l2-ctl'
            if name == 'v4l2-ctl' else None, run=run)
        self.assertEqual(choices, (
            ('Source default', ''), ('30 fps', '30.0'),
            ('59.94 fps', '59.94')))
        run.assert_called_once_with(
            ['/usr/bin/v4l2-ctl', '--list-formats-ext', '--device',
             '/dev/video2'], capture_output=True, text=True,
            encoding='utf-8', errors='replace', timeout=10, check=False)

    def test_capture_fps_is_a_basic_gui_dropdown(self):
        gui = SenderGui(self.devices)
        gui.settings['source'] = 'camera'
        gui.settings['camera'] = 'v4l2:/dev/video2'
        with patch('tools.v7_send_gui.enumerate_capture_fps', return_value=(
                ('Source default', ''), ('30 fps', '30'))):
            gui._open_dropdown('capture_fps')
        self.assertIn('capture_fps', gui._visible_fields())
        self.assertEqual(gui._choices('capture_fps'),
                         (('Source default', ''), ('30 fps', '30')))

        self.settings['capture_fps'] = '30'
        command = build_command(self.settings, self.devices, self.sd)
        self.assertEqual(command[command.index('--capture-fps')+1], '30.0')

    def test_native_video_picker_returns_selected_path(self):
        run = Mock(return_value=SimpleNamespace(
            returncode=0, stdout='/media/clips/a movie.mp4\n', stderr=''))
        which = lambda name: '/usr/bin/zenity' if name == 'zenity' else None

        selected = pick_video_file(platform='linux', which=which, run=run)

        self.assertEqual(selected, '/media/clips/a movie.mp4')
        args = run.call_args.args[0]
        self.assertEqual(args[0], '/usr/bin/zenity')
        self.assertIn('--file-selection', args)

    def test_native_video_picker_cancel_and_missing_picker_are_handled(self):
        cancel = Mock(return_value=SimpleNamespace(
            returncode=1, stdout='', stderr=''))
        which = lambda name: '/usr/bin/zenity' if name == 'zenity' else None
        self.assertIsNone(pick_video_file(platform='linux', which=which, run=cancel))

        with self.assertRaisesRegex(RuntimeError, 'Install zenity or kdialog'):
            pick_video_file(platform='linux', which=lambda _name: None,
                            run=cancel)

    def test_profile_default_and_explicit_filter_are_validated(self):
        self.settings['profile'] = 'mono-fold-500'
        self.settings['encode_filter'] = 'auto'
        result = validate_settings(self.settings, self.devices, self.sd)
        self.assertEqual(result['encode_filter'], 'box')

        self.settings.update(profile='fold-500', encode_filter='nearest')
        with self.assertRaisesRegex(ValueError, 'require the Box'):
            build_command(self.settings, self.devices, self.sd)

    def test_values_from_hidden_mono_audio_controls_do_not_block_other_profiles(self):
        self.settings.update(
            source='test', profile='fold-500', mono_video_side='invalid',
            source_audio='invalid', source_audio_input_side='invalid',
            source_audio_gain='not a number', source_audio_delay_ms='invalid')

        command = build_command(self.settings, self.devices, self.sd)
        args = v7_live.parser().parse_args(command[2:])

        self.assertEqual(args.profile, 'fold-500')
        self.assertFalse(any(option in command for option in (
            '--source-audio', '--source-audio-gain',
            '--source-audio-delay-ms')))

    def test_disabled_mono_audio_ignores_its_hidden_tuning_fields(self):
        self.settings.update(
            source='test', profile='mono-fold-500', source_audio='off',
            source_audio_input_side='invalid', source_audio_gain='invalid',
            source_audio_delay_ms='invalid')

        command = build_command(self.settings, self.devices, self.sd)
        args = v7_live.parser().parse_args(command[2:])

        self.assertEqual(args.source_audio, 'off')
        self.assertNotIn('--source-audio-gain', command)
        self.assertNotIn('--source-audio-delay-ms', command)

    def test_missing_device_source_video_path_and_unsupported_output_are_errors(self):
        for update, message in (
                ({'device': None}, 'Select an audio output device'),
                ({'source': None}, 'Choose a capture source'),
                ({'source': 'video'}, 'Set a video file path')):
            settings = dict(self.settings)
            settings.update(update)
            with self.subTest(message=message), self.assertRaisesRegex(
                    ValueError, message):
                validate_settings(settings, self.devices, self.sd)

        self.sd.check_output_settings.side_effect = RuntimeError('unsupported')
        with self.assertRaisesRegex(ValueError, 'cannot open'):
            validate_settings(self.settings, self.devices, self.sd)

    def test_invalid_speed_and_screen_region_are_rejected(self):
        self.settings['speed'] = '5'
        with self.assertRaisesRegex(ValueError, 'Speed must be between'):
            validate_settings(self.settings, self.devices, self.sd)

        self.settings.update(speed='1', region='1,2,0,4')
        with self.assertRaisesRegex(ValueError, 'Screen region'):
            validate_settings(self.settings, self.devices, self.sd)

    def test_basic_profile_picker_defaults_to_fold500_stereo(self):
        gui = SenderGui(self.devices)
        self.assertEqual(gui.settings['profile'], 'fold-500')
        self.assertEqual(tuple(value for _label, value in
                               gui._choices('profile')),
                         ('fold-500', 'mono-fold-500', 'mono-colour-500'))
        gui.advanced = True
        self.assertIn('fold-1000', [value for _label, value in
                                    gui._choices('profile')])
        parser = v7_live.parser()
        send = parser._subparsers._group_actions[0].choices['send']
        option_strings = {option for action in send._actions
                          for option in action.option_strings}
        self.assertNotIn('--mono-sum', option_strings)
        self.assertNotIn('--pilot-tones', option_strings)
        self.assertNotIn('--no-pilot-tones', option_strings)
        self.assertNotIn('--eof-marker', option_strings)
        self.assertTrue(send.get_default('pilot_tones'))
        self.assertTrue(send.get_default('eof_marker'))
        self.assertFalse(send.get_default('mono_sum'))
        send_help = send.format_help()
        self.assertIn('--profile', send_help)
        self.assertNotIn('--baseline', send_help)
        self.assertNotIn('--experimental-fold', send_help)
        self.assertNotIn('--no-eof-marker', send_help)

    def test_gui_hides_irrelevant_source_fields_and_renders_canvas(self):
        gui = SenderGui(self.devices)
        gui.settings.update(device=3, source='screen')
        visible = gui._visible_fields()
        self.assertIn('screen_backend', gui.ADVANCED_FIELDS)
        self.assertIn('screen_target', visible)
        self.assertNotIn('video_source', visible)
        self.assertNotIn('video_preview', visible)
        self.assertNotIn('camera', visible)
        self.assertNotIn('screen_backend', visible)

        image = gui._canvas((960, 720))
        self.assertEqual(image.size, (960, 720))
        self.assertIn('start_stop', gui.hits)

        gui.settings.update(source='video', profile='fold-500',
                            video_source='clip.mp4')
        gui.page = 'setup'
        self.assertIn('video_source', gui._visible_fields())
        self.assertIn('video_preview', gui._visible_fields())
        self.assertNotIn('camera', gui._visible_fields())
        self.assertNotIn('screen_target', gui._visible_fields())
        gui._canvas((960, 720))
        self.assertIn('browse:video_source', gui.hits)
        gui.page = 'live'
        self.assertEqual(gui._canvas((960, 720)).size, (960, 720))
        self.assertEqual(gui._canvas((720, 480)).size, (720, 480))

    def test_live_change_source_stops_then_opens_source_picker(self):
        gui = SenderGui(self.devices)
        gui.page = 'live'
        gui.process = object()
        gui.settings.update(source='video', video_source='clip.mp4',
                            video_preview=True, camera='camera-one')
        gui._canvas((960, 720))
        rect = gui.hits['change_source']
        position = ((rect[0]+rect[2])/2, (rect[1]+rect[3])/2)
        glfw = SimpleNamespace(
            MOUSE_BUTTON_LEFT=1, PRESS=1,
            get_cursor_pos=lambda _window: position)

        with patch.object(gui, '_stop') as stop:
            gui._on_mouse(glfw, None, 1, 1, 0)

        stop.assert_called_once()
        self.assertTrue(gui.change_source_after_stop)
        gui.events.put(('exit', 0))
        gui._drain_events()

        self.assertEqual(gui.page, 'setup')
        self.assertEqual(gui.selected, 'source')
        self.assertEqual(gui.dropdown, 'source')
        gui._assign('source', 'screen')
        self.assertEqual(gui.settings['video_source'], 'clip.mp4')
        self.assertTrue(gui.settings['video_preview'])
        self.assertEqual(gui.settings['camera'], 'camera-one')

    def test_sender_preferences_restore_last_settings_and_stable_devices(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'sender.json'
            output = OutputDevice(3, 'Stable output', 2, 48000, 'Test API')
            audio = InputDevice(8, 'Stable input', 2, 44100, 'Test API')
            gui = SenderGui((output,), audio_devices=(audio,),
                            preference_path=path)
            gui.settings.update(
                device=3, source='video', video_source='clip with spaces.mp4',
                video_preview=True, video_live=False, profile='fold-500',
                source_audio='device', source_audio_device=8,
                source_audio_input_side='right')
            gui._persist_preferences()

            restored = SenderGui(
                (OutputDevice(17, 'Stable output', 2, 96000, 'Test API'),),
                audio_devices=(InputDevice(
                    23, 'Stable input', 2, 48000, 'Test API'),),
                preference_path=path, restore_preferences=True)

        self.assertEqual(restored.settings['device'], 17)
        self.assertEqual(restored.settings['source_audio_device'], 23)
        self.assertEqual(restored.settings['source'], 'video')
        self.assertEqual(restored.settings['video_source'],
                         'clip with spaces.mp4')
        self.assertTrue(restored.settings['video_preview'])
        self.assertFalse(restored.settings['video_live'])
        self.assertIsNone(restored.process)

    def test_unavailable_saved_output_is_not_replaced_by_a_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'sender.json'
            path.write_text(json.dumps({
                'version': 1,
                'settings': {'source': 'video',
                             'video_source': 'clip.mp4'},
                'output_device': {'name': 'Missing output',
                                  'hostapi': 'Missing API'},
                'source_audio_device': None,
            }), encoding='utf-8')
            gui = SenderGui(self.devices, preference_path=path,
                            restore_preferences=True)

        self.assertIsNone(gui.settings['device'])
        self.assertIn('Missing output', gui.notice)
        self.assertEqual(gui.settings['source'], 'video')
        self.assertEqual(gui.settings['video_source'], 'clip.mp4')
        self.assertIsNone(gui.process)

    def test_malformed_saved_device_identity_is_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'sender.json'
            path.write_text(json.dumps({
                'version': 1, 'settings': {'source': 'test'},
                'output_device': 'not-a-device-identity',
            }), encoding='utf-8')
            gui = SenderGui(self.devices, preference_path=path,
                            restore_preferences=True)

        self.assertIsNone(gui.settings['device'])
        self.assertEqual(gui.settings['source'], 'test')
        self.assertIsNone(gui.process)

    def test_corrupt_sender_preferences_fall_back_to_setup_defaults(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'sender.json'
            path.write_text('{broken', encoding='utf-8')
            gui = SenderGui(self.devices, preference_path=path,
                            restore_preferences=True)

        self.assertIsNone(gui.settings['source'])
        self.assertFalse(gui.settings['video_preview'])
        self.assertIsNone(gui.process)

    def test_start_runs_cli_as_child_and_stop_requests_graceful_interrupt(self):
        gui = SenderGui(self.devices)
        gui.settings.update(device=3, source='screen')
        child = Mock()
        child.poll.return_value = None

        with (patch.object(gui, '_build_command', return_value=['python', 'send']),
              patch('tools.v7_send_gui.subprocess.Popen', return_value=child) as popen,
              patch('tools.v7_send_gui.threading.Thread') as thread):
            gui._start()
            self.assertIs(gui.process, child)
            self.assertEqual(gui.page, 'live')
            self.assertEqual(popen.call_args.args[0], ['python', 'send'])
            self.assertTrue(popen.call_args.kwargs['start_new_session'])
            thread.return_value.start.assert_called_once()

            gui._stop()

        child.send_signal.assert_called_once_with(signal.SIGINT)
        self.assertTrue(gui.stop_requested)

    def test_sender_exit_updates_live_state_and_allows_another_start(self):
        gui = SenderGui(self.devices)
        gui.process = Mock()
        gui.events.put(('line', 'V7 send stopped after 12 frames'))
        gui.events.put(('exit', 0))

        gui._drain_events()

        self.assertIsNone(gui.process)
        self.assertEqual(gui.notice, 'Sender stopped.')
        self.assertEqual(gui.lines, ['V7 send stopped after 12 frames'])

    def test_brightness_and_gamma_can_be_updated_while_sender_runs(self):
        gui = SenderGui(self.devices)
        control = Mock()
        gui.process = SimpleNamespace(stdin=control)
        gui.selected = 'brightness'
        gui.settings['brightness'] = '1.25'
        gui.settings['gamma'] = '0.8'
        gui.edit_buffer = '1.25'
        gui.editing = True

        gui._finish_edit()

        payload = json.loads(control.write.call_args.args[0])
        self.assertEqual(payload, {'brightness': 1.25, 'gamma': 0.8})
        control.flush.assert_called_once_with()

    def test_live_sender_accepts_valid_tone_updates_and_rejects_bad_values(self):
        controls = v7_live.LiveToneControls(1.0, 1.0)
        self.assertTrue(controls.update('{"brightness":1.3,"gamma":0.9}'))
        self.assertEqual(controls.snapshot(),
                         {'brightness': 1.3, 'gamma': 0.9})
        self.assertFalse(controls.update('{"brightness":0}'))
        self.assertFalse(controls.update('not json'))
        self.assertEqual(controls.snapshot(),
                         {'brightness': 1.3, 'gamma': 0.9})


if __name__ == '__main__':
    unittest.main()
