"""The sender GUI builds valid CLI settings without opening an audio stream."""
import unittest
import signal
from unittest.mock import Mock, patch

from tools import v7_live
from tools.v7_send_gui import (OutputDevice, SenderGui, build_command,
                               output_devices, sample_rate_options,
                               validate_settings)


class SenderGuiTests(unittest.TestCase):
    def setUp(self):
        self.device = OutputDevice(3, 'Test output', 2, 48000)
        self.devices = (self.device,)
        self.settings = {
            'device': 3,
            'source': 'screen',
            'rate': None,
            'profile': 'fold-500',
            'speed': '1',
            'encode_filter': 'auto',
            'brightness': '',
            'gamma': '1',
            'capture_fps': '',
            'video_source': '',
            'video_live': False,
            'camera': '0',
            'screen_backend': 'mss',
            'display': '',
            'region': '',
            'capture_width': '160',
            'capture_filter': 'auto',
            'mono_sum': False,
            'pilot_tones': True,
            'eof_marker': True,
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
        self.assertEqual(args.experimental_fold, 500)
        self.assertEqual(args.encode_filter, None)
        self.assertFalse(args.log)
        self.assertNotIn('--rate', command)

    def test_selected_sample_rate_profile_and_video_path_reach_cli(self):
        self.settings.update(source='video', video_source='a clip with spaces.mp4',
                             video_live=True, rate='96000',
                             profile='fold-1000', speed='1.5')

        command = build_command(self.settings, self.devices, self.sd)
        args = v7_live.parser().parse_args(command[2:])

        self.assertEqual(args.experimental_fold, 1000)
        self.assertEqual(args.rate, 96000)
        self.assertEqual(args.video_source, 'a clip with spaces.mp4')
        self.assertTrue(args.video_live)
        self.assertEqual(args.speed, 1.5)

    def test_ffmpeg_screen_input_is_forwarded_only_for_ffmpeg_capture(self):
        self.settings.update(source='screen', screen_backend='ffmpeg',
                             ffmpeg_input='x11grab::0.0')

        command = build_command(self.settings, self.devices, self.sd)
        args = v7_live.parser().parse_args(command[2:])

        self.assertEqual(args.screen_backend, 'ffmpeg')
        self.assertEqual(args.ffmpeg_input, 'x11grab::0.0')

    def test_profile_default_and_explicit_filter_are_validated(self):
        self.settings['profile'] = 'baseline'
        self.settings['encode_filter'] = 'auto'
        result = validate_settings(self.settings, self.devices, self.sd)
        self.assertEqual(result['encode_filter'], 'nearest')

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

    def test_mono_profile_requires_its_pilot_and_eof_invariants(self):
        self.settings.update(profile='mono', pilot_tones=False)
        with self.assertRaisesRegex(ValueError, 'requires pilot tones'):
            validate_settings(self.settings, self.devices, self.sd)

        self.settings.update(pilot_tones=True, eof_marker=False)
        with self.assertRaisesRegex(ValueError, 'requires the EOF marker'):
            validate_settings(self.settings, self.devices, self.sd)

    def test_gui_hides_irrelevant_source_fields_and_renders_canvas(self):
        gui = SenderGui(self.devices)
        gui.settings.update(device=3, source='screen')
        visible = gui._visible_fields()
        self.assertIn('screen_backend', gui.ADVANCED_FIELDS)
        self.assertNotIn('video_source', visible)
        self.assertNotIn('camera', visible)
        self.assertNotIn('screen_backend', visible)

        image = gui._canvas((960, 720))
        self.assertEqual(image.size, (960, 720))
        self.assertIn('start_stop', gui.hits)

        gui.settings.update(source='video', rate='88200',
                            video_source='clip.mp4')
        gui.page = 'live'
        self.assertIn('video_source', gui._visible_fields())
        self.assertNotIn('camera', gui._visible_fields())
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


if __name__ == '__main__':
    unittest.main()
