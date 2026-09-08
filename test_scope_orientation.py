"""Rotation and mirror reach every local mode from the command line.

Run with: python -m unittest test_scope_orientation

Orientation used to be settable only by editing
constantStorage/display_constants.py, and only rotation was honoured by the
scope path at all. These tests pin the CLI wiring and the output-boundary
mirror against the two ways it silently breaks: a by-value constant import
that predates argument parsing, and a mirror applied after the Y-T marker.
"""
import ast
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import settings
from scope_controls import KeyMap, as_flags
from scope_out import LEVEL, Scope, mirror_frame, rotate_frame


def picture_frame(n=3200):
    """An asymmetric frame: a mirror of it is never equal to itself."""
    # X is the wide fast sweep and Y the slow one, as in a real raster trace.
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    return np.column_stack((0.85 * (2.0 * t - 1.0),
                            0.3 * np.sin(9.0 * t))).astype(np.float32)


class MirrorFrameTests(unittest.TestCase):
    def test_mirror_negates_x_and_leaves_y_alone(self):
        f = picture_frame()
        out = mirror_frame(f, True)
        np.testing.assert_allclose(out[:, 0], -f[:, 0], atol=1e-7)
        np.testing.assert_allclose(out[:, 1], f[:, 1], atol=1e-7)

    def test_mirror_off_is_a_passthrough(self):
        f = picture_frame()
        np.testing.assert_array_equal(mirror_frame(f, False), f)

    def test_mirroring_twice_returns_the_original(self):
        f = picture_frame()
        np.testing.assert_allclose(mirror_frame(mirror_frame(f, True), True),
                                   f, atol=1e-7)

    def test_mirror_does_not_move_the_sweep_off_x(self):
        # The reason mirror may live at the output while rotation may not: a
        # 90-degree turn puts the fast axis on Y, a mirror does not.
        f = picture_frame()
        self.assertGreater(float(np.ptp(mirror_frame(f, True)[:, 0])),
                           float(np.ptp(mirror_frame(f, True)[:, 1])))
        turned = rotate_frame(f, 90)
        self.assertLess(float(np.ptp(turned[:, 0])), float(np.ptp(turned[:, 1])))


class ScopeMirrorOutputTests(unittest.TestCase):
    def test_show_frame_applies_the_configured_mirror(self):
        # trigger off: it would overwrite the first samples of both channels.
        scope = Scope(device="null", samplerate=96000, samples=3200,
                      mirror=True, trigger=False)
        self.addCleanup(scope.stream.close)
        f = picture_frame()
        scope.show_frame(f)
        np.testing.assert_allclose(scope._pending[:, 0], -f[:, 0], atol=1e-6)
        np.testing.assert_allclose(scope._pending[:, 1], f[:, 1], atol=1e-6)

    def test_set_mirror_takes_effect_without_a_restart(self):
        scope = Scope(device="null", samplerate=96000, samples=3200,
                      trigger=False)
        self.addCleanup(scope.stream.close)
        f = picture_frame()
        scope.show_frame(f)
        np.testing.assert_allclose(scope._pending[:, 0], f[:, 0], atol=1e-6)
        scope.set_mirror(True)
        scope.show_frame(f)
        np.testing.assert_allclose(scope._pending[:, 0], -f[:, 0], atol=1e-6)

    def test_mirror_does_not_invert_the_yt_trigger_edge(self):
        # Applied after the marker, the -0.99 -> +0.99 rise becomes a fall and
        # a rising-edge trigger stops locking entirely.
        scope = Scope(device="null", samplerate=96000, samples=3200,
                      mirror=True, yt_trigger_us=500.0)
        self.addCleanup(scope.stream.close)
        scope.show_frame(picture_frame())
        x = scope._pending[:, 0]
        rising = np.flatnonzero((x[:-1] < 0.95) & (x[1:] >= 0.95))
        self.assertEqual(len(rising), 1, "the trigger edge must survive mirroring")

    def test_browser_preview_is_mirrored_to_match_the_hardware(self):
        lum = np.zeros((8, 8), dtype=np.float32)
        lum[:, 0] = 1.0                       # a bar down the left edge
        scope = Scope(device="null", samplerate=96000, samples=3200, mirror=True)
        self.addCleanup(scope.stream.close)
        Scope.publish_luma(lum)
        _seq, data = Scope.read_luma()
        self.assertGreater(int(data[0, -1]), int(data[0, 0]),
                           "published luminance must be flipped with the output")
        scope.set_mirror(False)


class LiveControlTests(unittest.TestCase):
    def test_m_toggles_mirror_and_asks_for_a_transform_rebuild(self):
        state = {"mirror": False}
        keys = KeyMap(state)
        self.assertTrue(keys.feed("m"))
        self.assertTrue(state["mirror"])
        self.assertTrue(keys.transform_dirty)
        keys.feed("m")
        self.assertFalse(state["mirror"])

    def test_printed_flags_round_trip_orientation(self):
        flags = as_flags({"mode": "raster", "rotation": 270, "mirror": True})
        self.assertIn("--rotation 270", flags)
        self.assertIn("--mirror", flags)
        self.assertNotIn("--rotation 0", as_flags({"mode": "raster"}))
        self.assertNotIn("--mirror", as_flags({"mode": "raster"}))


class CommandLineTests(unittest.TestCase):
    """Drives main.configure_runtime() for real, without starting servers."""

    def setUp(self):
        self.saved = vars(settings).copy()
        self.addCleanup(self.restore)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        path = Path(__file__).with_name("main.py")
        tree = ast.parse(path.read_text(), filename=str(path))
        stop = next(i for i, node in enumerate(tree.body)
                    if isinstance(node, ast.Assign)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "configure_runtime")
        tree.body = tree.body[:stop]
        self.ns = {"__file__": str(Path(self.temp.name) / "main.py")}
        exec(compile(tree, str(path), "exec"), self.ns)
        self.ns["LOGS_DIR"] = str(Path(self.temp.name) / "logs")

    def restore(self):
        for key in set(vars(settings)) - set(self.saved):
            delattr(settings, key)
        vars(settings).update(self.saved)

    def configure(self, *flags, mode="scope"):
        argv = ["main.py", "--mode", mode, "--xy-dir", self.temp.name, *flags]
        if mode == "scope":
            argv += ["--device", "null"]
        with patch.object(sys, "argv", argv), \
                contextlib.redirect_stdout(io.StringIO()):
            return self.ns["configure_runtime"]()

    def test_rotation_and_mirror_reach_settings_in_scope_mode(self):
        self.configure("--rotation", "180", "--mirror")
        self.assertEqual(settings.INITIAL_ROTATION, 180)
        self.assertEqual(settings.INITIAL_MIRROR, 1)

    def test_rotation_and_mirror_reach_settings_in_local_mode(self):
        # "all local modes" -- orientation is not a scope-only concern.
        self.configure("--rotation", "90", "--mirror", mode="local")
        self.assertEqual(settings.INITIAL_ROTATION, 90)
        self.assertEqual(settings.INITIAL_MIRROR, 1)

    def test_no_mirror_switches_the_constant_off(self):
        settings.INITIAL_MIRROR = 1
        self.configure("--no-mirror")
        self.assertEqual(settings.INITIAL_MIRROR, 0)

    def test_omitting_the_flags_leaves_the_constants_alone(self):
        settings.INITIAL_ROTATION = 90
        settings.INITIAL_MIRROR = 1
        self.configure()
        self.assertEqual(settings.INITIAL_ROTATION, 90)
        self.assertEqual(settings.INITIAL_MIRROR, 1)

    def test_a_non_quarter_turn_is_rejected_at_the_boundary(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as err:
                self.configure("--rotation", "45")
        self.assertEqual(err.exception.code, 2)

    def test_display_state_reads_the_override_not_the_import_time_constant(self):
        # display_manager binds INITIAL_ROTATION by value at import, so this
        # is the check that a CLI override is not silently dropped there.
        settings.INITIAL_ROTATION = 270
        settings.INITIAL_MIRROR = 1
        source = Path(__file__).with_name("display_manager.py").read_text()
        tree = ast.parse(source)
        cls = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.ClassDef) and n.name == "DisplayState")
        ns = {"settings": settings, "INITIAL_ROTATION": 0, "INITIAL_MIRROR": 0,
              "tuple": tuple}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), "<ds>", "exec"), ns)
        state = ns["DisplayState"]()
        self.assertEqual(state.rotation, 270)
        self.assertEqual(state.mirror, 1)


if __name__ == "__main__":
    unittest.main()
