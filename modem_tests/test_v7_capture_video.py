"""Video-file/stream selection for the V7 live sender."""
import io
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools.v7_capture import video_source
from tools.v7_live import _resolve_send_source


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


if __name__ == '__main__':
    unittest.main()
