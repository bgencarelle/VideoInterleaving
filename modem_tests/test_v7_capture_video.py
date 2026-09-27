"""Video-file/stream selection for the V7 live sender."""
import io
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import sys
import unittest
from unittest import mock

import numpy as np

from animation_modem import v7
from tools.v7_capture import (CapturedFrame, _showinfo_source_size,
                              ffmpeg_source, video_source)
from tools.v7_live import _capture, _resolve_send_source, _values, run_send


class VideoSourceSelectionTests(unittest.TestCase):
    def test_missing_source_prompts_for_kind_then_video_path(self):
        args = Namespace(source=None, video_source=None)
        answers = iter(('video', 'rtsp://camera.example/live'))
        with redirect_stdout(io.StringIO()):
            _resolve_send_source(args, interactive=True,
                                 input_fn=lambda _prompt: next(answers))
        self.assertEqual(args.source, 'video')
        self.assertEqual(args.video_source, 'rtsp://camera.example/live')

    def test_video_argument_infers_source_kind(self):
        args = Namespace(source=None, video_source='clip.mp4')
        _resolve_send_source(args, interactive=False)
        self.assertEqual(args.source, 'video')

    def test_missing_video_argument_requires_interaction(self):
        args = Namespace(source='video', video_source=None)
        with self.assertRaisesRegex(ValueError, '--video-source PATH_OR_URL'):
            _resolve_send_source(args, interactive=False)

    def test_noninteractive_missing_source_fails_clearly(self):
        args = Namespace(source=None, video_source=None)
        with self.assertRaisesRegex(ValueError, 'specify --source'):
            _resolve_send_source(args, interactive=False)


class VideoSourceCommandTests(unittest.TestCase):
    class Process:
        def __init__(self):
            self.stdout = io.BytesIO()

        def poll(self):
            return 0

    def test_local_file_loops_and_is_realtime_paced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'clip.mp4'
            path.write_bytes(b'placeholder')
            process = self.Process()
            with mock.patch('tools.v7_capture.shutil.which', return_value='ffmpeg'), \
                    mock.patch('tools.v7_capture.subprocess.Popen',
                               return_value=process) as popen:
                grab = video_source(path, width=160,
                                    scale_flags='bicubic')
                cmd = popen.call_args.args[0]
                self.assertIn('-stream_loop', cmd)
                self.assertIn('-re', cmd)
                self.assertIn(str(path), cmd)
                self.assertIn('scale=160:-2:flags=bicubic', cmd)
                self.assertTrue(grab.paced)
                grab.close()

    def test_stream_url_is_neither_looped_nor_file_paced(self):
        process = self.Process()
        with mock.patch('tools.v7_capture.shutil.which', return_value='ffmpeg'), \
                mock.patch('tools.v7_capture.subprocess.Popen',
                           return_value=process) as popen:
            grab = video_source('rtsp://camera.example/live')
            cmd = popen.call_args.args[0]
            self.assertNotIn('-stream_loop', cmd)
            self.assertNotIn('-re', cmd)
            self.assertIn('-rw_timeout', cmd)
            self.assertIn('rtsp://camera.example/live', cmd)
            grab.close()

    def test_https_vod_loops_unless_explicitly_marked_live(self):
        for live in (False, True):
            process = self.Process()
            with self.subTest(live=live), \
                    mock.patch('tools.v7_capture.shutil.which', return_value='ffmpeg'), \
                    mock.patch('tools.v7_capture.subprocess.Popen',
                               return_value=process) as popen:
                grab = video_source('https://media.example/master.m3u8',
                                    live=live)
                cmd = popen.call_args.args[0]
                self.assertEqual('-stream_loop' in cmd, not live)
                self.assertEqual('-re' in cmd, not live)
                self.assertIn('-rw_timeout', cmd)
                grab.close()

    def test_direct_v7_grid_scale_keeps_original_aspect_metadata(self):
        ppm = b'P6\n80 96\n255\n'+bytes(80*96*3)
        showinfo = (b'[Parsed_showinfo_0 @ 0x1] n: 0 fmt:yuv420p '
                    b'sar:1/1 s:1920x1080 i:P\n')

        class DirectGridProcess:
            def __init__(self):
                self.stdout = io.BytesIO(ppm)
                self.stderr = io.BytesIO(showinfo)
                self.terminated = False

            def poll(self):
                return 0 if self.terminated else None

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                self.terminated = True
                return 0

        process = DirectGridProcess()
        with mock.patch('tools.v7_capture.shutil.which', return_value='ffmpeg'), \
                mock.patch('tools.v7_capture.subprocess.Popen',
                           return_value=process) as popen:
            grab = ffmpeg_source('v4l2:/dev/video0', 15,
                                 output_size=(80, 96), scale_flags='area')
            frame = grab()
            cmd = popen.call_args.args[0]
            self.assertIn('showinfo=checksum=0,scale=80:96:flags=area', cmd)
            self.assertIn('-loglevel', cmd)
            self.assertEqual(cmd[cmd.index('-loglevel')+1], 'info')
            self.assertIsInstance(frame, CapturedFrame)
            self.assertEqual(frame.rgb.shape, (96, 80, 3))
            self.assertEqual(frame.source_size, (1920, 1080))
            self.assertTrue(frame.prepared)
            grab.close()

    def test_showinfo_parser_rejects_missing_or_invalid_source_geometry(self):
        self.assertEqual(_showinfo_source_size(
            b'Parsed_showinfo n:0 fmt:yuv420p s:640x480 i:P'), (640, 480))
        self.assertIsNone(_showinfo_source_size(b'frame has no dimensions'))
        self.assertIsNone(_showinfo_source_size(b's:0x480'))


class SenderPreparationTests(unittest.TestCase):
    def test_prepared_camera_grid_uses_original_geometry_for_packet_aspect(self):
        frame = CapturedFrame(np.zeros((96, 80, 3), np.uint8),
                              (1920, 1080), prepared=True)
        model = type('Model', (), {
            'coder': type('Coder', (), {'grids': v7.V7_GRIDS})()})()
        with mock.patch('tools.v7_live.prepare_image',
                        side_effect=AssertionError('prepared frame resized twice')), \
                mock.patch('tools.v7_live.image_values',
                           return_value=np.zeros(4)) as image_values:
            _, aspect = _values(
                model, frame, encode_filter='box', brightness=1.0, gamma=1.0)
        self.assertEqual(image_values.call_args.args[0].size, (80, 96))
        self.assertEqual(image_values.call_args.kwargs['encode_filter'], 'box')
        self.assertEqual(aspect, v7.aspect_wire_code((1920, 1080)))

    def test_camera_capture_maps_encode_filter_to_direct_v7_scale(self):
        for encode_filter, scale_flags in (('box', 'area'),
                                           ('nearest', 'neighbor'),
                                           ('lanczos', 'lanczos'),
                                           ('bicubic', 'bicubic')):
            args = Namespace(
                source='camera', camera=0, capture_fps=None,
                capture_width=160, ffmpeg_input=None,
                region=None,
                encode_filter=encode_filter, capture_filter=None,
                perceptual_resize='off')
            with self.subTest(encode_filter=encode_filter), \
                    mock.patch('tools.v7_capture.camera_source') as camera:
                _capture(args)
                self.assertEqual(camera.call_args.kwargs['output_size'],
                                 (80, 96))
                self.assertEqual(camera.call_args.kwargs['scale_flags'],
                                 scale_flags)

    def test_opt_in_perceptual_resize_retains_intermediate_camera_path(self):
        args = Namespace(
            source='camera', camera=0, capture_fps=None,
            capture_width=160, ffmpeg_input=None, region=None,
            encode_filter='box',
            capture_filter=None, perceptual_resize='linear-box')
        with mock.patch('tools.v7_capture.camera_source') as camera:
            _capture(args)
        self.assertNotIn('output_size', camera.call_args.kwargs)

    def test_camera_capture_filter_override_wins_over_encode_profile(self):
        args = Namespace(
            source='camera', camera=0, capture_fps=None,
            capture_width=160, ffmpeg_input=None, region=None,
            encode_filter='box', capture_filter='lanczos',
            perceptual_resize='off')
        with mock.patch('tools.v7_capture.camera_source') as camera:
            _capture(args)
        self.assertEqual(camera.call_args.kwargs['scale_flags'], 'lanczos')


class SenderFailureTests(unittest.TestCase):
    def test_producer_errors_return_to_the_sender_thread(self):
        class OutputStream:
            samplerate = 48000

            def __init__(self, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

        class LatestGrab:
            def __init__(self, *_args):
                pass

            def __call__(self):
                return None

            def close(self):
                pass

        args = Namespace(
            fixture=None, encode_filter='nearest', rate=None, source='video',
            capture_fps=None, screen_backend='mss', speed=1.0,
            batch_frames=1, seconds=0, mono_sum=False, device=0,
            no_log=True, log=False, brightness=1.0, gamma=1.0,
            baseline=True,
        )
        fake_sounddevice = type('SoundDevice', (), {'OutputStream': OutputStream})
        with mock.patch.dict(sys.modules, {'sounddevice': fake_sounddevice}), \
                mock.patch('tools.v7_live._model', return_value=object()), \
                mock.patch('tools.v7_live._capture', return_value=lambda: None), \
                mock.patch('tools.v7_capture.Throttled', LatestGrab), \
                mock.patch('tools.v7_live._values',
                           side_effect=RuntimeError('broken test source')):
            with self.assertRaisesRegex(RuntimeError, 'broken test source'):
                run_send(args)


class SenderSchedulingTests(unittest.TestCase):
    class OutputStream:
        samplerate = 96000

        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def write(self, _audio):
            pass

    class Throttle:
        instances = []

        def __init__(self, grab, hz):
            self.grab = grab
            self.initial_hz = hz
            self.retuned_hz = None
            type(self).instances.append(self)

        def retune(self, hz):
            self.retuned_hz = hz

        def __call__(self):
            return self.grab()

        def close(self):
            pass

    def _run_for_speed(self, speed, source='camera'):
        grab_count = [0]

        def grab():
            grab_count[0] += 1
            return object()

        args = Namespace(
            fixture=None, encode_filter='nearest', rate=None, source=source,
            capture_fps=30, screen_backend='mss', speed=speed,
            batch_frames=1, seconds=.4, mono_sum=False, device=0,
            no_log=True, log=False, brightness=1.0, gamma=1.0, camera=0,
            capture_width=160, capture_filter='neighbor', ffmpeg_input=None,
            region=None, display=None, baseline=True,
        )
        fake_sounddevice = type(
            'SoundDevice', (), {'OutputStream': self.OutputStream})
        audio = np.zeros((v7.PULSE_FRAME, 2), np.float32)
        self.Throttle.instances = []
        with mock.patch.dict(sys.modules, {'sounddevice': fake_sounddevice}), \
                mock.patch('tools.v7_live._model', return_value=object()), \
                mock.patch('tools.v7_live._capture', return_value=grab), \
                mock.patch('tools.v7_capture.Throttled', self.Throttle), \
                mock.patch('tools.v7_live._values',
                           return_value=(np.zeros(1), 0)), \
                mock.patch('tools.v7_live.P.encode_pulse_stream',
                           return_value=audio), \
                mock.patch('tools.v7_live.P.speed_pulse_stream',
                           return_value=audio):
            run_send(args)
        self.assertEqual(len(self.Throttle.instances), 1)
        return grab_count[0], self.Throttle.instances[0].retuned_hz

    def test_capture_producer_tracks_accelerated_wire_rate(self):
        one_x, one_x_retune = self._run_for_speed(1.0)
        two_x, two_x_retune = self._run_for_speed(2.0)
        self.assertAlmostEqual(one_x_retune, v7.PULSE_FPS)
        self.assertAlmostEqual(two_x_retune, 2*v7.PULSE_FPS)
        self.assertGreaterEqual(two_x, 1.6*one_x)

    def test_screen_capture_producer_tracks_accelerated_wire_rate(self):
        one_x, _ = self._run_for_speed(1.0, source='screen')
        two_x, two_x_retune = self._run_for_speed(2.0, source='screen')
        self.assertAlmostEqual(two_x_retune, 2*v7.PULSE_FPS)
        self.assertGreaterEqual(two_x, 1.6*one_x)


if __name__ == '__main__':
    unittest.main()
