import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.scope_launcher_gui import (DEFAULT_SETTINGS, build_command,
                                      load_preferences, save_preferences,
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

    def test_camera_source_requires_a_custom_ffmpeg_input(self):
        self.settings.update({"run_mode": "live", "live_source": "camera"})
        with self.assertRaisesRegex(ValueError, "FFmpeg input"):
            validate_settings(self.settings, root=self.root)
        self.settings["ffmpeg_input"] = "v4l2:/dev/video0"
        validate_settings(self.settings, root=self.root)

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


class VideoSourceTransportTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
