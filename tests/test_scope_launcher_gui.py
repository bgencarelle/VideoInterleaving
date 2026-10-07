import copy
import queue
import threading
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

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

    def test_trajectory_controls_are_forwarded_to_both_pipelines(self):
        self.settings.update({"run_mode": "app", "app_source": "bake",
                              "render_mode": "raster",
                              "geometry_samples": "2400",
                              "traversal_hz": "60"})
        app = build_command(self.settings, root=self.root, python="python")
        self.assertEqual(app[app.index("--scope-geometry-samples") + 1], "2400")
        self.assertEqual(app[app.index("--scope-traversal-hz") + 1], "60")

        self.settings.update({"run_mode": "live", "live_source": "test"})
        live = build_command(self.settings, root=self.root, python="python")
        self.assertEqual(live[live.index("--geometry-samples") + 1], "2400")
        self.assertEqual(live[live.index("--traversal-hz") + 1], "60")

    def test_runtime_images_ignore_saved_baked_folder(self):
        self.settings.update({"run_mode": "app", "app_source": "images",
                              "render_mode": "raster",
                              "xy_dir": str(self.root / "missing_xy")})
        command = build_command(self.settings, root=self.root, python="python")
        self.assertNotIn("--xy-dir", command)
        self.assertEqual(command[command.index("--scope-source") + 1], "images")
        self.assertEqual(command[command.index("--dir") + 1], str(self.root))

    def test_baked_source_keeps_selected_baked_folder(self):
        self.settings.update({"run_mode": "app", "app_source": "bake",
                              "render_mode": "raster", "xy_dir": str(self.root)})
        command = build_command(self.settings, root=self.root, python="python")
        self.assertEqual(command[command.index("--xy-dir") + 1], str(self.root))

    def test_trajectory_controls_reject_unsupported_combinations(self):
        self.settings.update({"run_mode": "app", "app_source": "bake",
                              "render_mode": "stochastic",
                              "geometry_samples": "2400"})
        with self.assertRaisesRegex(ValueError, "baked vector, raster, or stipple"):
            validate_settings(self.settings, root=self.root)
        self.settings.update({"render_mode": "raster", "traversal_hz": "60",
                              "trigger": False})
        with self.assertRaisesRegex(ValueError, "requires the scope trigger"):
            validate_settings(self.settings, root=self.root)

    def test_baked_vector_command_accepts_independent_trajectory_controls(self):
        self.settings.update({"run_mode": "app", "app_source": "bake",
                              "render_mode": "vector", "geometry_samples": "3200",
                              "traversal_hz": "15"})
        validate_settings(self.settings, root=self.root)
        command = build_command(self.settings, root=self.root, python="python")
        self.assertEqual(command[command.index("--scope-mode") + 1], "vector")
        self.assertEqual(command[command.index("--scope-geometry-samples") + 1], "3200")
        self.assertEqual(command[command.index("--scope-traversal-hz") + 1], "15")

    def test_stipple_and_physical_dwell_controls_are_forwarded(self):
        self.settings.update({"run_mode": "app", "app_source": "bake",
                              "render_mode": "stipple", "geometry_samples": "2400",
                              "traversal_hz": "20", "physical_dwell": "0.4"})
        validate_settings(self.settings, root=self.root)
        command = build_command(self.settings, root=self.root, python="python")
        self.assertIn("--scope-mode", command)
        self.assertEqual(command[command.index("--scope-mode") + 1], "stipple")
        self.assertEqual(command[command.index("--scope-geometry-samples") + 1], "2400")
        self.assertEqual(command[command.index("--scope-traversal-hz") + 1], "20")
        self.assertEqual(command[command.index("--scope-physical-dwell") + 1], "0.4")

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

    def test_camera_commands_forward_all_supported_drawing_modes(self):
        self.settings.update(run_mode="live", live_source="camera",
                             ffmpeg_input="v4l2:/dev/video0")
        for mode in ("raster", "stochastic", "stipple"):
            with self.subTest(mode=mode):
                self.settings["live_render_mode"] = mode
                command = build_command(self.settings, root=self.root)
                self.assertEqual(command[command.index("--render-mode") + 1], mode)
        self.settings.update(live_render_mode="stipple", stream=True)
        with self.assertRaisesRegex(ValueError, "whole traces"):
            build_command(self.settings, root=self.root)

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
        gui._camera_pending = False
        gui.events = queue.Queue()
        gui.dropdown = None
        gui.notice = ""
        gui.dirty = False
        return gui

    def test_camera_picker_reuses_and_caches_modem_device_discovery(self):
        gui = self.launcher()
        devices = (("USB camera · /dev/video0", "v4l2:/dev/video0"),)

        with patch("tools.scope_launcher_gui._enumerate_camera_sources",
                   return_value=devices) as enumerate_devices, \
                patch("tools.scope_launcher_gui.threading.Thread") as worker:
            gui._open_dropdown("ffmpeg_input")
            self.assertIsNone(gui.dropdown)
            self.assertTrue(gui._camera_pending)
            enumerate_devices.assert_not_called()
            worker.assert_called_once()
            gui._discover_cameras()
            gui._drain_events()
            gui._open_dropdown("ffmpeg_input")
            self.assertEqual(gui.dropdown, "ffmpeg_input")
            self.assertEqual(gui._choices("ffmpeg_input"), devices)
            enumerate_devices.assert_called_once_with()

        gui.settings["ffmpeg_input"] = "v4l2:/dev/video0"
        self.assertEqual(gui._display_value("ffmpeg_input"), devices[0][0])

    def test_empty_camera_picker_explains_the_ffmpeg_fallback(self):
        gui = self.launcher()

        with patch("tools.scope_launcher_gui._enumerate_camera_sources",
                   return_value=()), \
                patch("tools.scope_launcher_gui.threading.Thread"):
            gui._open_dropdown("ffmpeg_input")
            gui._discover_cameras()
            gui._drain_events()

        self.assertIsNone(gui.dropdown)
        self.assertIn("No camera devices found", gui.notice)
        self.assertIn("Screen / FFmpeg input", gui.notice)

    def test_slow_discovery_does_not_block_or_launch_duplicate_workers(self):
        gui = self.launcher()
        entered, release = threading.Event(), threading.Event()

        def discover():
            entered.set()
            release.wait(2)
            return (("Camera", "v4l2:/dev/video0"),)

        with patch("tools.scope_launcher_gui._enumerate_camera_sources",
                   side_effect=discover) as probe:
            try:
                gui._open_dropdown("ffmpeg_input")
                self.assertTrue(entered.wait(1))
                self.assertTrue(gui._camera_pending)
                gui._camera_choices(refresh=True)
                self.assertEqual(probe.call_count, 1)
                self.assertIn("Looking for cameras", gui.notice)
            finally:
                release.set()
            event = gui.events.get(timeout=2)
            gui.events.put(event)
            gui._drain_events()
        self.assertFalse(gui._camera_pending)
        self.assertEqual(gui._choices("ffmpeg_input")[0][1], "v4l2:/dev/video0")

    def test_manual_ffmpeg_input_remains_a_text_field(self):
        gui = self.launcher()
        gui.settings["live_source"] = "ffmpeg"

        self.assertFalse(gui._is_dropdown_field("ffmpeg_input"))

    def test_live_pipeline_exposes_the_native_visualizer_controls(self):
        gui = ScopeLauncher.__new__(ScopeLauncher)
        gui.settings = {"run_mode": "live", "live_source": "test"}

        self.assertIn("scope_gui", gui._visible_fields())


class RunningSettingsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        with patch.object(ScopeLauncher, "_init_graphics"):
            self.gui = ScopeLauncher(
                preference_path=Path(self.temp.name) / "preferences.json",
                restore_preferences=False)
        self.gui.settings.update({"run_mode": "live", "live_source": "video",
                                  "video_file": str(Path(self.temp.name) / "old.mp4")})
        self.gui.active_settings = dict(self.gui.settings)
        self.gui.process = Mock()
        self.gui.process.poll.return_value = None
        self.gui.process.stdin = None
        self.gui.active_video_transport = True
        self.gui.playback = {"position": 12.5, "duration": 40.0, "paused": False}

    def test_apply_waits_for_old_process_exit_and_preserves_its_resume(self):
        gui = self.gui
        old_key = gui._resume_key()
        gui._assign("video_file", str(Path(self.temp.name) / "new.mp4"))
        self.assertTrue(gui._settings_pending())
        self.assertEqual(gui._resume_key(), old_key)
        self.assertEqual(gui._current_resume(), 0.0)
        with patch.object(gui, "_build_command", return_value=["python"]), \
                patch("tools.scope_launcher_gui.threading.Thread"), \
                patch.object(gui, "_start") as start:
            gui._apply_settings()
            self.assertTrue(gui.restart_requested)
            gui.process.send_signal.assert_called_once()
            start.assert_not_called()
            gui.events.put(("exit", 0))
            gui._drain_events()
            start.assert_called_once()
        self.assertEqual(gui.resume[old_key]["position"], 12.5)
        self.assertIsNone(gui.active_settings)

    def test_invalid_pending_settings_do_not_interrupt_running_scope(self):
        gui = self.gui
        gui._assign("live_fps", "invalid")
        with patch.object(gui, "_build_command", side_effect=ValueError("Invalid FPS")), \
                patch.object(gui, "_stop") as stop:
            gui._apply_settings()
            stop.assert_not_called()
        self.assertFalse(gui.restart_requested)
        self.assertEqual(gui.notice, "Invalid FPS")

    def test_invalid_active_editor_blocks_apply_and_keeps_user_input(self):
        gui = self.gui
        gui._assign("capture_fps", "12")
        gui._edit_start("live_fps")
        gui.edit_buffer = "not-a-number"
        with patch.object(gui, "_build_command") as build, \
                patch.object(gui, "_stop") as stop:
            gui._apply_settings()
        build.assert_not_called()
        stop.assert_not_called()
        self.assertTrue(gui.editing)
        self.assertEqual(gui.edit_buffer, "not-a-number")
        self.assertFalse(gui.restart_requested)
        self.assertIn("Invalid value", gui.notice)
        self.assertTrue(gui._edit_finish(commit=False))
        self.assertFalse(gui.editing)

    def test_invalid_editor_blocks_start_until_corrected(self):
        gui = self.gui
        gui.process = None
        gui._edit_start("live_fps")
        gui.edit_buffer = "bad"
        with patch("tools.scope_launcher_gui.subprocess.Popen") as start:
            gui._start()
        start.assert_not_called()
        self.assertTrue(gui.editing)
        gui.edit_buffer = "25"
        self.assertTrue(gui._edit_finish())
        self.assertEqual(gui.settings["live_fps"], "25")

    def test_task_tabs_keep_advanced_controls_available_without_clutter(self):
        gui = self.gui
        self.assertIn("video_file", gui._visible_fields())
        self.assertNotIn("blocksize", gui._visible_fields())
        gui.settings_tab = "output"
        self.assertIn("device", gui._visible_fields())
        self.assertNotIn("blocksize", gui._visible_fields())
        gui.show_advanced = True
        self.assertIn("blocksize", gui._visible_fields())
        gui.settings_tab = "preview"
        self.assertIn("scope_gui", gui._visible_fields())

    def test_stop_cancels_queued_restart(self):
        gui = self.gui
        gui.restart_requested = True
        gui.stop_requested = True
        gui._stop()
        with patch.object(gui, "_start") as start:
            gui.events.put(("exit", 0))
            gui._drain_events()
            start.assert_not_called()

    def test_current_video_controls_still_target_active_source_after_edit(self):
        gui = self.gui
        gui._assign("run_mode", "app")
        self.assertTrue(gui._has_video_selection())
        self.assertTrue(gui._is_control_process())
        self.assertTrue(gui._settings_pending())

    def test_stopping_after_source_edit_does_not_reuse_old_video_position(self):
        gui = self.gui
        old_key = gui._resume_key()
        gui._assign("video_file", str(Path(self.temp.name) / "new.mp4"))
        gui.events.put(("exit", 0))
        gui._drain_events()
        self.assertEqual(gui.resume[old_key]["position"], 12.5)
        self.assertEqual(gui._current_resume(), 0.0)

    def test_live_source_offers_captured_image_renderers(self):
        gui = self.gui
        self.assertIn("live_render_mode", gui._visible_fields())
        self.assertNotIn("render_mode", gui._visible_fields())
        self.assertEqual([value for _label, value in gui._choices("live_render_mode")],
                         ["raster", "stochastic", "stipple"])
        gui._assign("run_mode", "app")
        gui._assign("app_source", "images")
        self.assertEqual([value for _label, value in gui._choices("render_mode")],
                         ["raster", "stochastic", "stipple"])


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
