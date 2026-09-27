"""The sender GUI builds valid CLI settings without opening an audio stream."""
import unittest
import signal
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tools import v7_live
from tools.v7_send_gui import (InputDevice, OutputDevice, ScreenTarget, SenderGui,
                               build_command, enumerate_screen_targets,
                               enumerate_camera_sources, linux_camera_sources,
                               enumerate_capture_fps, parse_capture_fps,
                               input_devices, output_devices,
                               parse_avfoundation_screen_sources,
                               parse_ffmpeg_camera_sources, pick_video_file,
                               sample_rate_options, validate_settings)


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

    def test_sample_rates_are_probed_for_selected_device_and_channel_count(self):
        options = sample_rate_options(self.device, 2, self.sd)

        self.assertEqual(options[0], ('Native (device clock)', None))
        self.assertIn(('48 kHz', 48000), options)
        self.assertEqual(options[-1], ('Custom…', 'custom'))
        self.assertEqual(self.sd.check_output_settings.call_args.kwargs['device'], 3)
        self.assertEqual(self.sd.check_output_settings.call_args.kwargs['channels'], 2)

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
                             video_live=True, rate='96000',
                             profile='fold-1000', speed='1.5')

        command = build_command(self.settings, self.devices, self.sd)
        args = v7_live.parser().parse_args(command[2:])

        self.assertEqual(args.profile, 'fold-1000')
        self.assertEqual(args.rate, 96000)
        self.assertEqual(args.video_source, 'a clip with spaces.mp4')
        self.assertTrue(args.video_live)
        self.assertEqual(args.speed, 1.5)

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
        gui.settings['source_audio'] = 'device'
        for _label, source in (
                ('Camera', 'camera'), ('Screen', 'screen'),
                ('Video', 'video'), ('Test', 'test'),
                ('Mouse-follow', 'mouse-follow')):
            with self.subTest(source=source):
                gui.settings['source'] = source
                self.assertIn('source_audio', gui._visible_fields())
                self.assertIn('source_audio_device', gui._visible_fields())

    def test_mono_video_side_selector_is_visible_only_for_that_profile(self):
        gui = SenderGui(self.devices)
        self.assertIn('mono_video_side', gui._visible_fields())

        gui.settings.update(profile='fold-500', mono_video_side='right')
        self.assertNotIn('mono_video_side', gui._visible_fields())
        gui.settings.update(profile='mono-fold-500')
        self.assertIn('source_audio', gui._visible_fields())
        self.assertNotIn('source_audio_device', gui._visible_fields())
        self.assertEqual(gui._value_label('mono_video_side'),
                         'Right output · left stays clear')

        gui.settings['source_audio'] = 'device'
        self.assertIn('source_audio_device', gui._visible_fields())

    def test_mono_video_always_uses_wire_defaults_and_disables_pre_resize(self):
        self.settings.update(profile='mono-fold-500', perceptual_resize='linear-box')
        with self.assertRaisesRegex(ValueError, 'requires the pre-encode downscaler off'):
            validate_settings(self.settings, self.devices, self.sd)

        self.settings.update(perceptual_resize='off')
        command = build_command(self.settings, self.devices, self.sd)
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

    def test_camera_picker_values_and_screen_targets_map_to_sender_cli(self):
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

    def test_downscaler_is_forwarded_and_constrained(self):
        self.settings.update(source='video', video_source='clip.mp4',
                             perceptual_resize='gamma-detail',
                             perceptual_detail_strength='0.6')
        command = build_command(self.settings, self.devices, self.sd)
        args = v7_live.parser().parse_args(command[2:])
        self.assertEqual(args.perceptual_resize, 'gamma-detail')
        self.assertEqual(args.perceptual_detail_strength, 0.6)

        self.settings.update(profile='mono-fold-500')
        with self.assertRaisesRegex(ValueError, 'requires the pre-encode downscaler off'):
            validate_settings(self.settings, self.devices, self.sd)

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
                   '[AVFoundation indev @ 0x1] [1] Capture screen 0\n'
                   '[AVFoundation indev @ 0x1] AVFoundation audio devices:\n'
                   '[AVFoundation indev @ 0x1] [2] Microphone')
        self.assertEqual(parse_ffmpeg_camera_sources(listing, 'darwin'), (
            ('Face Camera · AVFoundation 0', 'avfoundation:0'),))
        self.assertEqual(parse_avfoundation_screen_sources(listing), (
            ScreenTarget('Capture screen 0', display=1),))

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

    def test_basic_profile_picker_defaults_to_mono_with_fold500_stereo(self):
        gui = SenderGui(self.devices)
        self.assertEqual(gui.settings['profile'], 'mono-fold-500')
        self.assertEqual(tuple(value for _label, value in
                               gui._choices('profile')),
                         ('mono-fold-500', 'fold-500'))
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
        self.assertNotIn('camera', visible)
        self.assertNotIn('screen_backend', visible)

        image = gui._canvas((960, 720))
        self.assertEqual(image.size, (960, 720))
        self.assertIn('start_stop', gui.hits)

        gui.settings.update(source='video', rate='88200', profile='fold-500',
                            video_source='clip.mp4')
        gui.page = 'setup'
        self.assertIn('video_source', gui._visible_fields())
        self.assertNotIn('camera', gui._visible_fields())
        self.assertNotIn('screen_target', gui._visible_fields())
        gui._canvas((960, 720))
        self.assertIn('browse:video_source', gui.hits)
        gui.page = 'live'
        self.assertEqual(gui._canvas((960, 720)).size, (960, 720))
        self.assertEqual(gui._canvas((720, 480)).size, (720, 480))

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
