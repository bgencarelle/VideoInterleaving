"""Capture startup failures are bounded and visible without camera hardware."""
import threading
import unittest
from unittest.mock import Mock, patch

from tools.scope_screen import ffmpeg_source


class CaptureStartupTests(unittest.TestCase):
    def test_early_ffmpeg_exit_reports_failure_and_preserves_stderr(self):
        process = Mock()
        with patch("shutil.which", return_value="ffmpeg"), \
                patch("subprocess.Popen", return_value=process) as spawn, \
                patch("tools.scope_screen._read_exact", return_value=None):
            grab = ffmpeg_source(input_spec="v4l2:/dev/video0")
            with self.assertRaisesRegex(RuntimeError, "before its first frame"):
                grab()
        self.assertIsNone(spawn.call_args.kwargs["stderr"])
        process.stdout.close.assert_called_once()

    def test_first_frame_timeout_kills_capture_and_unblocks_reader(self):
        killed = threading.Event()
        process = Mock()
        process.kill.side_effect = killed.set

        def read(_pipe, _size):
            self.assertTrue(killed.wait(2), "startup watchdog did not unblock capture")
            return None

        with patch("shutil.which", return_value="ffmpeg"), \
                patch("subprocess.Popen", return_value=process), \
                patch("tools.scope_screen._read_exact", side_effect=read):
            grab = ffmpeg_source(input_spec="v4l2:/dev/video0", startup_timeout=.05)
            with self.assertRaisesRegex(RuntimeError, "startup timed out"):
                grab()
        process.kill.assert_called_once()
        process.wait.assert_called_once()
