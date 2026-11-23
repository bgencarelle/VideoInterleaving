"""Playback/recording checks use files and fake devices, never audio hardware."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np

from animation_modem.audio_common import input_pair
from tools import v7_live
from tools.v7_capture import CaptureEndOfStream, CapturedFrame, FilePlayback
from tools.v7_file import export_wav, transform_frame
from tools.v7_wav_io import WavWriter, wav_blocks, wav_info

ROOT = Path(__file__).resolve().parents[1]


class PlaybackEndTests(unittest.TestCase):
    def test_finite_passes_reopen_then_end_without_holding_or_wrapping(self):
        opened, closed = [], []

        def reader(start):
            opened.append(start)
            frames = iter([int(start*10), int(start*10)+1])

            def grab():
                try:
                    return next(frames)
                except StopIteration:
                    raise CaptureEndOfStream
            grab.close = lambda: closed.append(start)
            return grab

        playback = FilePlayback(reader, duration=1, fps=10, passes=2, start=.2)
        self.assertEqual([playback() for _ in range(4)], [2, 3, 0, 1])
        with self.assertRaises(CaptureEndOfStream):
            playback()
        self.assertTrue(playback.ended)
        self.assertEqual(playback.completed_passes, 2)
        self.assertEqual(opened, [.2, 0])
        self.assertEqual(closed, [.2, 0])
        self.assertEqual(playback.position, 1)
        playback.close()

    def test_transforms_keep_source_geometry_and_do_not_mutate_input(self):
        image = np.arange(2*3*3, dtype=np.uint8).reshape(2, 3, 3)
        wrapped = CapturedFrame(image, (300, 200), True)
        transformed = transform_frame(wrapped, 90, True)
        self.assertEqual(transformed.source_size, (200, 300))
        self.assertTrue(transformed.prepared)
        np.testing.assert_array_equal(transformed.rgb, np.rot90(image, -1)[:, ::-1])
        np.testing.assert_array_equal(image.ravel(), np.arange(18))

    def test_channel_selection_preserves_logical_order_and_rejects_missing_pair(self):
        self.assertEqual(input_pair(None, 1), (0,))
        self.assertEqual(input_pair(None, 8), (0, 1))
        self.assertEqual(input_pair((5, 2), 8), (5, 2))
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            input_pair((2, 5), 4)


class GuiControlTests(unittest.TestCase):
    def test_sender_export_needs_no_device_and_restored_settings_reach_cli(self):
        from tools.v7_send_gui import SenderGui, build_command, _restore_sender_settings
        gui = SenderGui()
        _restore_sender_settings(gui.settings, {
            'output_mode': 'export', 'wav_path': 'tmp/gui-export.wav',
            'source': 'test', 'frames': '7', 'seconds': '2',
            'rate': '96000', 'rotation': 90, 'mirror': True,
            'luma_adjust_linear': True})
        sd = mock.Mock()
        command = build_command(gui.settings, (), sd_module=sd)
        sd.check_output_settings.assert_not_called()
        args = v7_live.parser().parse_args(command[2:])
        self.assertEqual(args.write, Path('tmp/gui-export.wav'))
        self.assertEqual((args.frames, args.seconds, args.rate), (7, 2, 96000))
        self.assertEqual(args.rotation, 90)
        self.assertTrue(args.mirror)
        self.assertTrue(args.luma_adjust_linear)
        self.assertIsNone(args.device)

    def test_receiver_file_and_channel_selection_do_not_require_a_device(self):
        from tools.v7_receiver_gui import ReceiverGui, _receive_parser
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as directory:
            root, receive = _receive_parser(v7_live)
            gui = ReceiverGui(v7_live, root, receive, (),
                              preference_path=Path(directory)/'prefs.json')
            gui._field('read').value = 'tmp/input.wav'
            gui._field('channels').value = '5,3'
            args = gui._build_arguments()
            self.assertEqual(args.read, Path('tmp/input.wav'))
            self.assertEqual(args.channels, (4, 2))
            self.assertIsNone(args.device)


class FileRoundTripTests(unittest.TestCase):
    def test_recording_roundtrip_and_append_preserve_float_samples(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as directory:
            path = Path(directory)/'recording.wav'
            samples = np.array([[1.1, -.2], [.3, -.4]], np.float32)
            with WavWriter(path, 48000) as writer:
                writer.write(samples)
            with WavWriter(path, 48000, append=True) as writer:
                writer.write(samples)
            self.assertEqual(wav_info(path), (48000, 2, 4))
            np.testing.assert_array_equal(np.concatenate(list(wav_blocks(path, 1))),
                                          np.concatenate([samples, samples]))
            with self.assertRaisesRegex(ValueError, 'rate/channels'):
                WavWriter(path, 44100, append=True)

    def test_export_is_deterministic_at_two_rates_and_decodes_without_devices(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as directory:
            for rate in (48000, 96000):
                output = Path(directory)/f'{rate}.wav'
                args = v7_live.parser().parse_args([
                    'send', '--source', 'test', '--write', str(output),
                    '--frames', '4', '--rate', str(rate), '--no-log'])
                with mock.patch('sounddevice.OutputStream', side_effect=AssertionError('device opened')):
                    self.assertEqual(export_wav(args), 4)
                first = np.concatenate(list(wav_blocks(output)))
                count = len(v7_live.P.speed_pulse_stream(
                    np.zeros(v7_live.P.PULSE_FRAME), 1, rate=rate))
                self.assertEqual(wav_info(output), (rate, 2, count*4))
                export_wav(args)
                np.testing.assert_array_equal(first, np.concatenate(list(wav_blocks(output))))
                receive = v7_live.parser().parse_args([
                    'receive', '--read', str(output), '--headless', '--no-log'])
                with mock.patch('sounddevice.InputStream', side_effect=AssertionError('device opened')), \
                        mock.patch('sounddevice.query_devices', side_effect=AssertionError('device enumerated')):
                    result = v7_live.run_receive(receive)
                self.assertGreaterEqual(result['decoded'], 3)

    def test_live_limit_trims_batches_and_recording_matches_routed_output(self):
        class Output:
            samplerate = 48000
            active = True

            def __init__(self, **kwargs):
                self.options = kwargs
                self.blocks = []

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                pass

            def write(self, samples):
                self.blocks.append(samples.copy())

        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as directory:
            path = Path(directory)/'record.wav'
            args = v7_live.parser().parse_args([
                'send', '--source', 'test', '--device', 'fake',
                '--frames', '3', '--batch-frames', '2', '--channels', '3,5',
                '--record', str(path), '--no-log'])
            output = Output()

            def open_output(**kwargs):
                output.options = kwargs
                return output

            with mock.patch('sounddevice.OutputStream', side_effect=open_output):
                v7_live._run_send_session(args)
            wire = np.concatenate(output.blocks)
            self.assertEqual(output.options['channels'], 5)
            self.assertEqual(args._sender_total, 3)
            self.assertEqual(len(wire), 3*v7_live.P.PULSE_FRAME)
            np.testing.assert_array_equal(wire[:, [0, 1, 3]], 0)
            np.testing.assert_array_equal(wire[:, [2, 4]],
                                          np.concatenate(list(wav_blocks(path))))

    @unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg required')
    def test_video_passes_and_embedded_soundtrack_export(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as directory:
            source = Path(directory)/'clip.mkv'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                            'testsrc2=size=160x120:rate=24:duration=0.5',
                            '-f', 'lavfi', '-i', 'sine=frequency=440:duration=0.5',
                            '-c:v', 'ffv1', '-c:a', 'pcm_s16le', str(source)], check=True)
            output = Path(directory)/'mono.wav'
            args = v7_live.parser().parse_args([
                'send', '--source', 'video', '--video-source', str(source),
                '--write', str(output), '--source-passes', '2',
                '--profile', 'aspect-mono-500', '--no-log'])
            count = export_wav(args)
            self.assertGreater(count, 4)
            sound = np.concatenate(list(wav_blocks(output)))[:, 0]
            self.assertGreater(np.max(np.abs(sound)), .01)


if __name__ == '__main__':
    unittest.main()
