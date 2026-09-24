"""Video-file/stream selection for the V7 live sender."""
import io
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import sys
import unittest
from unittest import mock

from tools.v7_capture import video_source
from tools.v7_live import _resolve_send_source, run_send


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


if __name__ == '__main__':
    unittest.main()
