"""Capture startup failures are bounded and visible without camera hardware."""
import threading
import unittest
from unittest.mock import Mock, patch
import numpy as np

from tools.scope_screen import ffmpeg_source, modem_camera_source, main


class CaptureStartupTests(unittest.TestCase):
    def test_camera_interrupt_closes_shared_backend(self):
        source = Mock(side_effect=KeyboardInterrupt)
        with patch("tools.v7_capture.camera_source", return_value=source):
            grab = modem_camera_source("avfoundation:0", fps=60)
            with self.assertRaises(KeyboardInterrupt):
                grab()
        source.close.assert_called_once()

    def test_capture_is_closed_if_output_setup_fails_before_main_loop(self):
        def source():
            return np.zeros((8, 8), np.float32)
        source.close = Mock()
        with patch("tools.scope_screen.modem_camera_source", return_value=source), \
                patch("tools.scope_screen.Scope", side_effect=RuntimeError("output setup failed")):
            with self.assertRaisesRegex(RuntimeError, "output setup failed"):
                main(["--source", "camera", "--ffmpeg-input", "v4l2:/dev/test", "--device", "null"])
        self.assertTrue(source.close.called)
    def test_camera_reuses_modem_backend_with_default_rate_and_rgb_geometry(self):
        source = Mock(return_value=np.full((12, 20, 3), 255, np.uint8))
        with patch("tools.v7_capture.camera_source", return_value=source) as capture, \
                patch("tools.v7_send_gui.enumerate_capture_fps", return_value=(("Source default", ""),)):
            grab = modem_camera_source("avfoundation:0", width=160)
            lum = grab()
        capture.assert_called_once_with(fps=None, width=160, spec="avfoundation:0", scale_flags="area")
        self.assertEqual(lum.shape, (12, 20))
        np.testing.assert_allclose(lum, 1)
        self.assertEqual(grab.timing_snapshot()["source_sequence"], 1)
        grab.close()

    def test_modem_camera_permission_diagnostic_is_retained(self):
        source = Mock(side_effect=RuntimeError("camera permission denied by AVFoundation"))
        with patch("tools.v7_capture.camera_source", return_value=source):
            grab = modem_camera_source("avfoundation:0", fps=60)
            with self.assertRaisesRegex(RuntimeError, "permission denied"):
                grab()
        source.close.assert_called_once()

    def test_source_default_uses_reported_rate_instead_of_unsupported_30(self):
        source = Mock(return_value=np.zeros((8, 8, 3), np.uint8))
        with patch("tools.v7_capture.camera_source", return_value=source) as capture, \
                patch("tools.v7_send_gui.enumerate_capture_fps", return_value=(("Source default", ""), ("60 fps", "60"))):
            grab = modem_camera_source("avfoundation:0")
            grab()
            grab.close()
        self.assertEqual(capture.call_args.kwargs["fps"], 60)

    def test_source_default_retries_avfoundation_driver_reported_rate(self):
        rejected = Mock(side_effect=RuntimeError("Selected framerate not supported\n1280x720@[59.940060 59.940060]fps"))
        working = Mock(return_value=np.zeros((8, 12, 3), np.uint8))
        with patch("tools.v7_capture.camera_source", side_effect=(rejected, working)) as capture, \
                patch("tools.v7_send_gui.enumerate_capture_fps", return_value=(("Source default", ""),)):
            grab = modem_camera_source("avfoundation:0")
            self.assertEqual(grab().shape, (8, 12))
            self.assertIs(grab.proc, working.proc)
            grab.close()
        self.assertAlmostEqual(capture.call_args.kwargs["fps"], 59.940060)
        rejected.close.assert_called_once()
        working.close.assert_called_once()
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
