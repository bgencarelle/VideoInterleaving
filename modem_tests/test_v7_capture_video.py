"""Video-file/stream selection for the V7 live sender."""
import io
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import sys
import threading
import unittest
from unittest import mock

import numpy as np

from animation_modem import v7
from tools.v7_capture import (CapturedFrame, _showinfo_source_size,
                              CaptureEndOfStream, Throttled, ffmpeg_source,
                              video_source, video_source_loops)
from tools.v7_live import (_capture, _resolve_send_source,
                           SENDER_STARTUP_BUFFER_SECONDS,
                           _sender_queue_batches, _values, run_send)

ROOT = Path(__file__).resolve().parents[1]


class VideoSourceSelectionTests(unittest.TestCase):
    def test_live_sender_bounds_queue_by_startup_audio_duration(self):
        self.assertAlmostEqual(SENDER_STARTUP_BUFFER_SECONDS,
                               v7.PULSE_FRAME/v7.RATE)
        self.assertEqual(_sender_queue_batches(
            SENDER_STARTUP_BUFFER_SECONDS, 48000, 3920), 1)
        self.assertEqual(_sender_queue_batches(
            SENDER_STARTUP_BUFFER_SECONDS/4, 96000, 1960), 1)

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
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
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

    def test_dct_video_capture_keeps_the_decoded_frame_dimensions(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
            path = Path(tmp)/'source.mp4'
            path.write_bytes(b'placeholder')
            process = self.Process()
            with mock.patch('tools.v7_capture.shutil.which', return_value='ffmpeg'), \
                    mock.patch('tools.v7_capture.subprocess.Popen',
                               return_value=process) as popen:
                grab = video_source(path, width=160, preserve_size=True)
                command = popen.call_args.args[0]
                self.assertNotIn('-vf', command)
                self.assertEqual(command[command.index('-i')+1], str(path))
                grab.close()

    def test_untagged_hd_files_are_read_with_the_hd_colour_matrix(self):
        import shutil
        import subprocess
        import numpy as np
        from tools.v7_capture import untagged_hd_matrix, video_scale_filter
        if shutil.which('ffmpeg') is None or shutil.which('ffprobe') is None:
            self.skipTest('ffmpeg/ffprobe not installed')
        colours = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255],
                            [255, 255, 0]], np.uint8)

        def encode(path, size, extra):
            frame = np.repeat(np.repeat(colours[None], size[1], 0),
                              size[0]//4, 1)
            made = subprocess.run(
                ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo',
                 '-pixel_format', 'rgb24', '-video_size',
                 f'{size[0]}x{size[1]}', '-framerate', '30', '-i', 'pipe:0',
                  '-vf', 'scale=out_color_matrix=bt709,setparams=colorspace=unknown',
                  '-pix_fmt', 'yuv420p',
                  '-c:v', 'libx264', '-crf', '10',
                  '-colorspace', 'unknown', *extra, str(path)],
                input=np.tile(frame[None], (3, 1, 1, 1)).tobytes())
            if made.returncode:
                self.skipTest('this ffmpeg cannot encode the test clip')

        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
            hd, tagged, sd = (Path(tmp)/name for name in
                              ('hd.mp4', 'tagged.mp4', 'sd.mp4'))
            encode(hd, (1280, 720), [])
            encode(tagged, (1280, 720), ['-colorspace', 'bt709'])
            encode(sd, (640, 480), [])
            self.assertEqual(untagged_hd_matrix(hd), 'bt709')
            self.assertIsNone(untagged_hd_matrix(tagged))
            self.assertIsNone(untagged_hd_matrix(sd))
            self.assertIsNone(untagged_hd_matrix('rtsp://camera.example/live'))
            self.assertEqual(video_scale_filter(hd, 160, 'bicubic', True),
                             'scale=in_color_matrix=bt709')
            self.assertEqual(video_scale_filter(hd, 160, 'bicubic', False),
                             'scale=160:-2:flags=bicubic:in_color_matrix=bt709')
            self.assertIsNone(video_scale_filter(sd, 160, 'bicubic', True))
            grab = video_source(hd, preserve_size=True, realtime=False,
                                loop=False)
            try:
                frame = np.asarray(grab())
            finally:
                grab.close()
            self.assertEqual(frame.shape, (720, 1280, 3))
            patches = np.array([frame[360, 160+320*i] for i in range(4)], int)
            # Within compression error of the source colours (the SD matrix
            # reads green as about (19, 255, 8)).
            self.assertLess(np.abs(patches-colours).max(), 8)

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

    def test_loop_policy_of_the_sender_source(self):
        self.assertTrue(video_source_loops('clip.mp4'))
        self.assertTrue(video_source_loops(
            'https://media.example/clip.mp4?token=x'))
        self.assertFalse(video_source_loops('rtsp://camera.example/live'))
        self.assertFalse(video_source_loops(
            'https://camera.example/live', live=True))

    def test_avfoundation_camera_opens_the_selected_device_name(self):
        process = self.Process()
        with mock.patch('tools.v7_capture.shutil.which', return_value='ffmpeg'), \
                mock.patch('tools.v7_capture.subprocess.Popen',
                           return_value=process) as popen:
            grab = ffmpeg_source(
                'avfoundation:HDMI Capture Card', 30)
            command = popen.call_args.args[0]
            grab.close()

        self.assertEqual(command[command.index('-i')+1],
                         'HDMI Capture Card')


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
                self.assertIn('-re', cmd)
                self.assertIn('-rw_timeout', cmd)
                grab.close()

    def test_live_finite_mp4_url_is_paced_once(self):
        process = self.Process()
        with mock.patch('tools.v7_capture.shutil.which', return_value='ffmpeg'), \
                mock.patch('tools.v7_capture.subprocess.Popen',
                           return_value=process) as popen:
            grab = video_source(
                'https://media.example/flower.mp4?token=abc', live=True)
            command = popen.call_args.args[0]
            grab.close()

        self.assertNotIn('-stream_loop', command)
        self.assertIn('-re', command)

    def test_ffmpeg_screen_capture_applies_selected_windows_monitor_region(self):
        process = self.Process()
        with mock.patch('tools.v7_capture.shutil.which', return_value='ffmpeg'), \
                mock.patch('tools.v7_capture.subprocess.Popen',
                           return_value=process) as popen:
            grab = ffmpeg_source('gdigrab:desktop', 59.94,
                                 region=(-1920, 0, 1920, 1080))
            cmd = popen.call_args.args[0]
            self.assertEqual(cmd[cmd.index('-framerate')+1], '59.94')
            self.assertEqual(cmd[cmd.index('-offset_x')+1], '-1920')
            self.assertEqual(cmd[cmd.index('-offset_y')+1], '0')
            self.assertEqual(cmd[cmd.index('-video_size')+1], '1920x1080')
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


class ThrottledCleanupTests(unittest.TestCase):
    def test_clean_end_of_stream_is_not_a_capture_failure(self):
        class FiniteCapture:
            paced = True

            def __init__(self):
                self.calls = 0

            def __call__(self):
                self.calls += 1
                if self.calls > 1:
                    raise CaptureEndOfStream
                return np.zeros((2, 2, 3), dtype=np.uint8)

            def close(self):
                pass

        throttled = Throttled(FiniteCapture(), 30)
        throttled._thread.join(timeout=1)

        self.assertFalse(throttled._thread.is_alive())
        self.assertTrue(throttled.ended)
        self.assertIsNone(throttled())
        np.testing.assert_array_equal(throttled.first_frame, 0)

    def test_close_releases_a_blocked_source_without_a_process_handle(self):
        class Capture:
            paced = True

            def __init__(self):
                self.calls = 0
                self.release = threading.Event()
                self.closed = False

            def __call__(self):
                self.calls += 1
                if self.calls > 1:
                    self.release.wait(2)
                return np.zeros((2, 2, 3), dtype=np.uint8)

            def close(self):
                self.closed = True
                self.release.set()

        capture = Capture()
        throttled = Throttled(capture, 30)
        np.testing.assert_array_equal(throttled.first_frame, 0.0)

        throttled.close()

        self.assertTrue(capture.closed)
        self.assertFalse(throttled._thread.is_alive())


class SenderPreparationTests(unittest.TestCase):
    def test_avfoundation_picker_passes_camera_name_to_capture(self):
        args = Namespace(
            source='camera', camera=0, capture_fps=None,
            capture_width=160,
            ffmpeg_input='avfoundation:HDMI Capture Card', region=None,
            encode_filter='box', capture_filter=None,
            perceptual_resize='off')
        with mock.patch('tools.v7_capture.camera_source') as camera:
            _capture(args)

        self.assertEqual(camera.call_args.kwargs['spec'],
                         'avfoundation:HDMI Capture Card')

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
            experimental_fold=0,
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

    def test_finite_live_video_eof_stops_sender_without_producer_failure(self):
        class OutputStream:
            samplerate = 48000

            def __init__(self, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

            def write(self, _samples):
                pass

        class EndedGrab:
            ended = True
            first_frame = None

            def __init__(self, *_args):
                pass

            def __call__(self):
                return None

            def close(self):
                pass

        args = Namespace(
            fixture=None, encode_filter='nearest', rate=None, source='video',
            video_source='https://media.example/flower.mp4', video_live=True,
            capture_fps=None, screen_backend='mss', speed=1.0,
            batch_frames=1, seconds=0, mono_sum=False, device=0,
            no_log=True, log=False, brightness=1.0, gamma=1.0,
            experimental_fold=0)
        audio = np.zeros((v7.PULSE_FRAME, 2), dtype=np.float32)
        fake_sounddevice = type('SoundDevice', (), {
            'OutputStream': OutputStream})

        with mock.patch.dict(sys.modules, {'sounddevice': fake_sounddevice}), \
                mock.patch('tools.v7_live._model', return_value=object()), \
                mock.patch('tools.v7_live._capture', return_value=lambda: None), \
                mock.patch('tools.v7_capture.Throttled', EndedGrab), \
                mock.patch('tools.v7_live._values',
                           return_value=(np.zeros(1), 0)), \
                mock.patch('tools.v7_live.P.encode_pulse_stream',
                           return_value=audio), \
                mock.patch('tools.v7_live.P.speed_pulse_stream',
                           return_value=audio):
            run_send(args)

    def test_sender_reopens_selected_device_after_sample_rate_change(self):
        class SoundDevice:
            devices = [{
                'name': 'Recoverable loopback', 'hostapi': 0,
                'max_output_channels': 2, 'max_input_channels': 0,
                'default_samplerate': 48000,
            }]
            opened_rates = []
            writes = []
            lost_once = False

            @classmethod
            def query_devices(cls, device=None, _kind=None):
                if device is None:
                    return list(cls.devices)
                return cls.devices[int(device)]

            @staticmethod
            def query_hostapis():
                return [{'name': 'Test API'}]

            class OutputStream:
                def __init__(self, **_kwargs):
                    self.samplerate = float(
                        SoundDevice.devices[0]['default_samplerate'])
                    self.active = True
                    SoundDevice.opened_rates.append(self.samplerate)

                def __enter__(self):
                    return self

                def __exit__(self, *_exc):
                    self.active = False

                def write(self, samples):
                    if not SoundDevice.lost_once:
                        SoundDevice.lost_once = True
                        SoundDevice.devices[0]['default_samplerate'] = 96000
                        raise RuntimeError('simulated output unplug')
                    SoundDevice.writes.append(samples.copy())

        args = Namespace(
            fixture=None, encode_filter='nearest', rate=None, source='test',
            video_source=None, video_live=False, preview=False,
            capture_fps=30, screen_backend='mss', speed=1.0,
            batch_frames=1, seconds=.65, mono_sum=False, device=0,
            no_log=True, log=False, brightness=1.0, gamma=1.0,
            camera=0, capture_width=160, capture_filter='neighbor',
            ffmpeg_input=None, region=None, display=None, experimental_fold=0,
        )
        audio = np.zeros((v7.PULSE_FRAME, 2), dtype=np.float32)

        with mock.patch.dict(sys.modules, {'sounddevice': SoundDevice}), \
                mock.patch('tools.v7_live._model', return_value=object()), \
                mock.patch('tools.v7_live._capture',
                           return_value=lambda: object()), \
                mock.patch('tools.v7_live._values',
                           return_value=(np.zeros(1), 0)), \
                mock.patch('tools.v7_live.P.encode_pulse_stream',
                           return_value=audio), \
                mock.patch('tools.v7_live.P.speed_pulse_stream',
                           return_value=audio):
            run_send(args)

        self.assertGreaterEqual(len(SoundDevice.opened_rates), 2)
        self.assertEqual(SoundDevice.opened_rates[:2], [48000.0, 96000.0])
        self.assertTrue(SoundDevice.writes)


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
            region=None, display=None, experimental_fold=0,
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

    def test_playback_waits_for_encoded_startup_cushion(self):
        encoded = [0]
        writes = []

        class RecordingOutputStream:
            samplerate = 96000

            def __init__(self, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

            def write(self, _audio):
                writes.append(encoded[0])

        args = Namespace(
            fixture=None, encode_filter='nearest', rate=None, source='test',
            capture_fps=30, screen_backend='mss', speed=1.0,
            batch_frames=1, seconds=.35, mono_sum=False, device=0,
            no_log=True, log=False, brightness=1.0, gamma=1.0,
            camera=0, capture_width=160, capture_filter='neighbor',
            ffmpeg_input=None, region=None, display=None, experimental_fold=0,
        )
        fake_sounddevice = type(
            'SoundDevice', (), {'OutputStream': RecordingOutputStream})
        audio = np.zeros((v7.PULSE_FRAME, 2), np.float32)

        def encode(*_args, **_kwargs):
            encoded[0] += 1
            return audio

        with mock.patch.dict(sys.modules, {'sounddevice': fake_sounddevice}), \
                mock.patch('tools.v7_live._model', return_value=object()), \
                mock.patch('tools.v7_live._capture',
                           return_value=lambda: object()), \
                mock.patch('tools.v7_capture.Throttled', self.Throttle), \
                mock.patch('tools.v7_live._values',
                           return_value=(np.zeros(1), 0)), \
                mock.patch('tools.v7_live.P.encode_pulse_stream',
                           side_effect=encode), \
                mock.patch('tools.v7_live.P.speed_pulse_stream',
                           return_value=audio), \
                mock.patch('tools.v7_live.SENDER_STARTUP_BUFFER_SECONDS', .08):
            run_send(args)

        self.assertTrue(writes)
        # One call compiles the path before capture; the single startup packet
        # is enough to begin output.
        self.assertGreaterEqual(writes[0], 2)


class _TrickleStream(io.RawIOBase):
    """A raw pipe that returns at most ``chunk`` bytes per read."""

    def __init__(self, data, chunk):
        self.data, self.chunk, self.position = data, chunk, 0

    def readable(self):
        return True

    def readinto(self, buffer):
        piece = self.data[self.position:self.position+min(len(buffer),
                                                            self.chunk)]
        buffer[:len(piece)] = piece
        self.position += len(piece)
        return len(piece)


class PipeFrameReadTests(unittest.TestCase):
    def test_short_pipe_reads_assemble_one_writable_frame(self):
        from tools.v7_capture import _read_ppm
        pixels = np.arange(640*360*3, dtype=np.uint32).astype(np.uint8)
        stream = _TrickleStream(b'P6\n640 360\n255\n'+pixels.tobytes(),
                                65536)
        frame = _read_ppm(stream)
        self.assertEqual(frame.shape, (360, 640, 3))
        np.testing.assert_array_equal(frame.ravel(), pixels)
        self.assertTrue(frame.flags.writeable)

    def test_truncated_frame_is_an_error_not_zeros(self):
        from tools.v7_capture import _read_ppm
        stream = _TrickleStream(b'P6\n4 4\n255\n'+bytes(20), 7)
        with self.assertRaisesRegex(RuntimeError, 'partway'):
            _read_ppm(stream)

    def test_dct_preview_is_a_toned_thumbnail(self):
        from tools.v7_live import _dct_preview_image
        rgb = np.random.default_rng(0).integers(
            0, 256, (1080, 1920, 3), dtype=np.uint8)
        preview = _dct_preview_image(rgb, 1.05, 1.2)
        self.assertLessEqual(preview.width, 1920//3)
        self.assertEqual(preview.mode, 'RGB')
        corner = np.asarray(_dct_preview_image(rgb[:320, :256], 1.05, 1.2))
        expected = np.rint(np.clip(rgb[:320, :256]/255*1.05, 0, 1)**(1/1.2)*255)
        np.testing.assert_array_equal(corner, expected.astype(np.uint8))


if __name__ == '__main__':
    unittest.main()
