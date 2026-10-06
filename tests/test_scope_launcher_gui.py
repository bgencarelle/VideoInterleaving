import copy
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from tools.scope_launcher_gui import (DEFAULT_SETTINGS, build_command,
                                      load_preferences, save_preferences,
                                      ScopeLauncher,
                                      validate_settings,
                                      window_size_for_workarea)


class ScopeLauncherCommandTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = copy.deepcopy(DEFAULT_SETTINGS)
        self.settings.update({
            "device": "null", "channels": "1,2", "x_only": True,
            "trigger": True, "xy_dir": "", "image_dir": str(self.root),
        })

    def test_app_scope_command_carries_image_renderer_and_mono_output(self):
        self.settings.update({"run_mode": "app", "app_source": "images",
                              "render_mode": "raster", "scope_gui": True})
        command = build_command(self.settings, root=self.root, python="python")
        self.assertEqual(command[:3], ["python", str(self.root / "main.py"),
                                       "--mode"])
        self.assertIn("images", command)
        self.assertIn("--scope-x-only", command)
        self.assertIn("--scope-channels", command)
        self.assertIn("--scope-gui", command)
        self.assertIn("--no-scope-mix", command)

    def test_launcher_can_start_the_scope_tuner_fullscreen_image_only(self):
        self.settings.update({"run_mode": "app", "app_source": "images",
                              "render_mode": "raster",
                              "scope_gui_fullscreen": True})
        command = build_command(self.settings, root=self.root, python="python")
        self.assertIn("--scope-gui", command)
        self.assertIn("--scope-gui-image-only", command)
        self.assertIn("--scope-gui-fullscreen", command)

    def test_runtime_images_reject_baked_geometry_modes(self):
        self.settings.update({"run_mode": "app", "app_source": "images",
                              "render_mode": "vector"})
        with self.assertRaisesRegex(ValueError, "Runtime images"):
            validate_settings(self.settings, root=self.root)

    def test_live_video_command_includes_transport_and_saved_position(self):
        video = self.root / "clip.mp4"
        video.write_bytes(b"placeholder")
        self.settings.update({"run_mode": "live", "live_source": "video",
                              "video_file": str(video)})
        command = build_command(self.settings, root=self.root, python="python",
                                video_start=12.5)
        self.assertIn(str(self.root / "tools" / "scope_screen.py"), command)
        self.assertIn("--source", command)
        self.assertIn("--control", command)
        self.assertEqual(command[command.index("--start-at") + 1], "12.5")
        self.assertIn("--scope-x-only", command)

    def test_live_pipeline_can_open_the_native_scope_visualizer(self):
        self.settings.update({"run_mode": "live", "live_source": "test",
                              "scope_gui": True})
        command = build_command(self.settings, root=self.root, python="python")

        self.assertIn("--scope-gui", command)

    def test_live_scope_parser_accepts_visualizer_option(self):
        from tools.scope_screen import build_parser

        args = build_parser().parse_args(["--source", "test", "--scope-gui"])

        self.assertTrue(args.scope_gui)

    def test_camera_source_requires_a_selected_device_input(self):
        self.settings.update({"run_mode": "live", "live_source": "camera"})
        with self.assertRaisesRegex(ValueError, "FFmpeg input"):
            validate_settings(self.settings, root=self.root)
        self.settings["ffmpeg_input"] = "v4l2:/dev/video0"
        validate_settings(self.settings, root=self.root)
        command = build_command(self.settings, root=self.root, python="python")
        self.assertEqual(command[command.index("--ffmpeg-input") + 1],
                         "v4l2:/dev/video0")

    def test_whole_trace_mix_cannot_be_combined_with_explicit_samples(self):
        self.settings.update({"run_mode": "app", "app_source": "bake",
                              "render_mode": "vector", "mix": "120",
                              "samples": "3200"})
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            validate_settings(self.settings, root=self.root)

    def test_stream_command_does_not_restore_a_file_position(self):
        self.settings.update({"run_mode": "live", "live_source": "video",
                              "video_file": "https://example.invalid/live.m3u8"})
        command = build_command(self.settings, root=self.root, python="python",
                                video_start=12.5)
        self.assertNotIn("--start-at", command)
        self.assertIn("--rotation", command)
        self.assertIn("--no-mirror", command)

    def test_live_oversampling_must_be_an_integer(self):
        video = self.root / "clip.mp4"
        video.write_bytes(b"placeholder")
        self.settings.update({"run_mode": "live", "live_source": "video",
                              "video_file": str(video), "live_oversample": "1.5"})
        with self.assertRaisesRegex(ValueError, "Anti-alias oversampling"):
            validate_settings(self.settings, root=self.root)

    def test_output_device_must_support_selected_channel_pair(self):
        self.settings.update({"run_mode": "live", "live_source": "test",
                              "device": "2", "channels": "18,19",
                              "x_only": True})
        outputs = ((0, "Stereo", "API", 48000),)
        with self.assertRaisesRegex(ValueError, "enough output channels"):
            validate_settings(self.settings, outputs, root=self.root)

    def test_preferences_round_trip_settings_and_resume_positions(self):
        path = self.root / "scope.json"
        values = {"run_mode": "live", "live_source": "video"}
        resume = {"clip.mp4": {"position": 4.25, "duration": 20.0,
                                "paused": False}}
        save_preferences(path, values, resume)
        self.assertEqual(load_preferences(path), (values, resume))

    def test_launcher_window_fits_a_small_monitor_workarea(self):
        self.assertEqual(window_size_for_workarea(800, 600), (768, 520))
        self.assertEqual(window_size_for_workarea(1920, 1080), (1040, 760))


class CameraDevicePickerTests(unittest.TestCase):
    @staticmethod
    def launcher():
        gui = ScopeLauncher.__new__(ScopeLauncher)
        gui.settings = {"live_source": "camera", "ffmpeg_input": ""}
        gui._camera_cache = None
        gui.dropdown = None
        gui.notice = ""
        gui.dirty = False
        return gui

    def test_camera_picker_reuses_and_caches_modem_device_discovery(self):
        gui = self.launcher()
        devices = (("USB camera · /dev/video0", "v4l2:/dev/video0"),)

        with patch("tools.scope_launcher_gui._enumerate_camera_sources",
                   return_value=devices) as enumerate_devices:
            gui._open_dropdown("ffmpeg_input")
            self.assertEqual(gui.dropdown, "ffmpeg_input")
            self.assertEqual(gui._choices("ffmpeg_input"), devices)
            enumerate_devices.assert_called_once_with()

        gui.settings["ffmpeg_input"] = "v4l2:/dev/video0"
        self.assertEqual(gui._display_value("ffmpeg_input"), devices[0][0])

    def test_empty_camera_picker_explains_the_ffmpeg_fallback(self):
        gui = self.launcher()

        with patch("tools.scope_launcher_gui._enumerate_camera_sources",
                   return_value=()):
            gui._open_dropdown("ffmpeg_input")

        self.assertIsNone(gui.dropdown)
        self.assertIn("No camera devices found", gui.notice)
        self.assertIn("Screen / FFmpeg input", gui.notice)

    def test_manual_ffmpeg_input_remains_a_text_field(self):
        gui = self.launcher()
        gui.settings["live_source"] = "ffmpeg"

        self.assertFalse(gui._is_dropdown_field("ffmpeg_input"))

    def test_live_pipeline_exposes_the_native_visualizer_controls(self):
        gui = ScopeLauncher.__new__(ScopeLauncher)
        gui.settings = {"run_mode": "live", "live_source": "test"}

        self.assertIn("scope_gui", gui._visible_fields())


class VideoSourceTransportTests(unittest.TestCase):
    class FakeClock:
        def __init__(self):
            self.now = 10.0

        def __call__(self):
            return self.now

        def advance(self, seconds):
            self.now += float(seconds)

    class FakeCapture:
        def __init__(self):
            self.index = 0
            self.reads = 0
            self.released = False

        def isOpened(self):
            return True

        def get(self, prop):
            cv2 = VideoSourceTransportTests.FakeCV2
            if prop == cv2.CAP_PROP_FPS:
                return 10.0
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 40.0
            if prop == cv2.CAP_PROP_POS_FRAMES:
                return float(self.index)
            if prop == cv2.CAP_PROP_POS_MSEC:
                return self.index * 100.0
            return 0.0

        def set(self, prop, value):
            cv2 = VideoSourceTransportTests.FakeCV2
            if prop == cv2.CAP_PROP_POS_FRAMES:
                self.index = int(value)
            elif prop == cv2.CAP_PROP_POS_MSEC:
                self.index = int(float(value) / 100.0)
            return True

        def read(self):
            if self.index >= 40:
                return False, None
            frame = np.full((4, 4, 3), self.index, dtype=np.uint8)
            self.index += 1
            self.reads += 1
            return True, frame

        def release(self):
            self.released = True

    class FakeCV2:
        CAP_PROP_FPS = 5
        CAP_PROP_FRAME_COUNT = 7
        CAP_PROP_POS_FRAMES = 1
        CAP_PROP_POS_MSEC = 0

        def __init__(self):
            self.capture = VideoSourceTransportTests.FakeCapture()

        def VideoCapture(self, _path):
            return self.capture

    def test_pause_seek_restart_and_play_keep_source_thread_safe(self):
        from tools.scope_screen import VideoFileSource

        cv2 = self.FakeCV2()
        source = VideoFileSource("clip.mp4", downto=16, cv2_module=cv2)
        first = source()
        self.assertEqual(source.playback()["duration"], 4.0)
        source.transport("pause")
        reads = cv2.capture.reads
        np.testing.assert_array_equal(source(), first)
        self.assertEqual(cv2.capture.reads, reads)

        source.transport("seek", 1.5)
        sought = source()
        self.assertAlmostEqual(float(sought[0, 0]), 15.0 / 255.0)
        self.assertTrue(source.playback()["paused"])
        source.transport("restart")
        restarted = source()
        self.assertAlmostEqual(float(restarted[0, 0]), 0.0)
        self.assertTrue(source.playback()["paused"])

        source.transport("play")
        self.assertAlmostEqual(float(source()[0, 0]), 1.0 / 255.0)
        source.close()
        self.assertTrue(cv2.capture.released)

    def test_saved_start_position_seeks_before_first_frame(self):
        from tools.scope_screen import VideoFileSource

        cv2 = self.FakeCV2()
        source = VideoFileSource("clip.mp4", downto=16, start_at=1.2,
                                 cv2_module=cv2)
        frame = source()
        self.assertAlmostEqual(float(frame[0, 0]), 12.0 / 255.0)
        self.assertEqual(source.playback()["position"], 1.2)
        source.close()

    def test_media_clock_drops_late_frames_and_preserves_transport_time(self):
        from tools.scope_screen import VideoFileSource

        cv2 = self.FakeCV2()
        clock = self.FakeClock()
        source = VideoFileSource("clip.mp4", downto=16, cv2_module=cv2,
                                 clock=clock)
        try:
            first = source.read_latest_due()
            self.assertAlmostEqual(float(first[0, 0]), 0.0)
            clock.advance(0.25)
            due = source.read_latest_due()
            self.assertAlmostEqual(float(due[0, 0]), 2.0 / 255.0)
            self.assertAlmostEqual(
                source.timing_snapshot()["media_position_s"], 0.2)

            clock.advance(0.25)
            due = source.read_latest_due()
            self.assertAlmostEqual(float(due[0, 0]), 5.0 / 255.0)
            source.transport("pause")
            reads = cv2.capture.reads
            clock.advance(1.0)
            paused = source.read_latest_due()
            np.testing.assert_array_equal(paused, due)
            self.assertEqual(cv2.capture.reads, reads)

            source.transport("seek", 1.2)
            # Seeking does not replace the last complete picture with a
            # synthetic black placeholder while the new frame is pending.
            np.testing.assert_array_equal(source.last, due)
            sought = source.read_latest_due()
            self.assertAlmostEqual(float(sought[0, 0]), 12.0 / 255.0)
            self.assertTrue(source.playback()["paused"])
            source.transport("play")
            clock.advance(0.25)
            resumed = source.read_latest_due()
            self.assertAlmostEqual(float(resumed[0, 0]), 14.0 / 255.0)

            clock.advance(3.0)
            looped = source.read_latest_due()
            self.assertAlmostEqual(float(looped[0, 0]), 4.0 / 255.0)
            self.assertAlmostEqual(
                source.timing_snapshot()["media_position_s"], 0.4)
        finally:
            source.close()

    def test_paused_seek_publishes_first_frame_after_off_boundary_target(self):
        from tools.scope_screen import VideoFileSource

        class ForwardSeekCapture(self.FakeCapture):
            def set(self, prop, value):
                cv2 = VideoSourceTransportTests.FakeCV2
                if prop == cv2.CAP_PROP_POS_MSEC:
                    # Model a decoder that lands on the first frame after a
                    # non-frame-aligned seek position.
                    self.index = int(np.ceil(float(value) / 100.0))
                    return True
                return super().set(prop, value)

        cv2 = self.FakeCV2()
        cv2.capture = ForwardSeekCapture()
        source = VideoFileSource("clip.mp4", downto=16, cv2_module=cv2)
        try:
            before = source()
            source.transport("pause")
            source.transport("seek", 1.55)
            np.testing.assert_array_equal(source.last, before)

            sought = source.read_latest_due()
            self.assertAlmostEqual(float(sought[0, 0]), 16.0 / 255.0)
            self.assertAlmostEqual(
                source.timing_snapshot()["media_position_s"], 1.6)
            self.assertTrue(source.playback()["paused"])
        finally:
            source.close()


class LatestCaptureSnapshotTests(unittest.TestCase):
    def test_raw_pipe_reader_returns_only_complete_frames(self):
        from tools.scope_screen import _read_exact

        class ChunkedPipe:
            def __init__(self, chunks):
                self.chunks = list(chunks)

            def read(self, _count):
                return self.chunks.pop(0) if self.chunks else b""

        self.assertEqual(
            _read_exact(ChunkedPipe([b"a", b"bc", b"def"]), 6),
            b"abcdef")
        self.assertIsNone(_read_exact(ChunkedPipe([b"abc"]), 6))

    def test_reader_publishes_matching_latest_pixels_and_timestamps(self):
        from tools.scope_screen import Throttled

        class TimedReader:
            def __init__(self):
                self.sequence = 0
                self.metadata = {}

            def __call__(self):
                self.sequence += 1
                self.metadata = {
                    "source_kind": "test-capture",
                    "source_sequence": self.sequence,
                    "requested_at_ns": time.monotonic_ns(),
                    "decode_started_at_ns": time.monotonic_ns(),
                    "ready_at_ns": time.monotonic_ns(),
                    "media_position_s": None,
                }
                return np.full((4, 4), self.sequence, dtype=np.float32)

            def timing_snapshot(self):
                return dict(self.metadata)

        reader = TimedReader()
        latest = Throttled(reader, fps=200.0, source_kind="test")
        try:
            deadline = time.monotonic() + 1.0
            while latest.captures < 2 and time.monotonic() < deadline:
                time.sleep(0.002)
            frame, _version, metadata = latest.timed_snapshot()
            self.assertGreaterEqual(latest.captures, 2)
            self.assertTrue(np.all(frame == metadata["source_sequence"]))
            self.assertLessEqual(metadata["requested_at_ns"],
                                 metadata["decode_started_at_ns"])
            self.assertLessEqual(metadata["decode_started_at_ns"],
                                 metadata["ready_at_ns"])
            self.assertEqual(latest.snapshot_metrics()["source_kind"],
                             "test-capture")
        finally:
            latest.close()

    def test_drain_reader_continuously_consumes_paced_capture_frames(self):
        from tools.scope_screen import Throttled

        class PacedReader:
            def __init__(self):
                self.sequence = 0
                self.metadata = {}

            def __call__(self):
                time.sleep(0.002)
                self.sequence += 1
                now = time.monotonic_ns()
                self.metadata = {
                    "source_kind": "ffmpeg-test",
                    "source_sequence": self.sequence,
                    "requested_at_ns": now - 2_000_000,
                    "decode_started_at_ns": now - 1_000_000,
                    "ready_at_ns": now,
                    "media_position_s": None,
                }
                return np.full((4, 4), self.sequence, dtype=np.float32)

            def timing_snapshot(self):
                return dict(self.metadata)

        reader = PacedReader()
        latest = Throttled(reader, fps=1.0, drain=True)
        try:
            deadline = time.monotonic() + 0.2
            while latest.captures < 10 and time.monotonic() < deadline:
                time.sleep(0.002)
            frame, _version, metadata = latest.timed_snapshot()
            self.assertGreaterEqual(latest.captures, 10)
            self.assertTrue(np.all(frame == metadata["source_sequence"]))
            self.assertLessEqual(latest.snapshot_metrics()["polls"],
                                 latest.captures + 1)
        finally:
            latest.close()


if __name__ == "__main__":
    unittest.main()
