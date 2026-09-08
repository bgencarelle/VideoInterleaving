"""Trigger and row-timing configuration; run with python -m unittest test_scope_yt_cli.

The trigger used to be a MODE, and these tests pinned the five combinations it
refused to start in. It is a property of the output now -- stamped on the
finished frame in Scope.show_frame, which every renderer already passes through
-- so there is nothing left for it to be incompatible with, and what these
tests pin instead is that the old flags still land somewhere sensible and that
the awkward combinations degrade with a message rather than exiting.
"""
import ast
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import settings
import scope_display
from scope_controls import KeyMap, as_flags
from scope_out import Scope


class YtConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.saved_settings = vars(settings).copy()
        self.addCleanup(self.restore_settings)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        # Execute the real configuration code without starting main's servers.
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
        settings.SCOPE_REALTIME = False

    def restore_settings(self):
        for key in set(vars(settings)) - set(self.saved_settings):
            delattr(settings, key)
        vars(settings).update(self.saved_settings)

    def configure(self, *flags):
        argv = ["main.py", "--mode", "scope", "--device", "null",
                "--xy-dir", self.temp.name, *flags]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            return self.ns["configure_runtime"]()

    def test_every_duration_alias_reaches_the_output(self):
        for flag in ("--scope-trigger-us", "--scope-trigger-duration",
                     "--scope-yt-trigger", "--scope-yt-trigger-us"):
            with self.subTest(flag=flag):
                self.configure(flag, "500")
                self.assertEqual(settings.SCOPE_TRIGGER_US, 500.0)
                scope = Scope(device="null", samplerate=96000,
                              yt_trigger_us=settings.SCOPE_TRIGGER_US)
                try:
                    self.assertEqual(scope.yt_trigger_samples, 48)
                    state = dict(yt=True, mode="raster", raster=True,
                                 yt_trigger_us=scope.yt_trigger_us)
                    self.assertIn("--scope-trigger-us 500", as_flags(state))
                finally:
                    scope.stream.close()

    def test_the_trigger_is_on_by_default_and_can_be_switched_off(self):
        self.configure()
        self.assertTrue(settings.SCOPE_TRIGGER)
        self.assertEqual(settings.SCOPE_TRIGGER_SHAPE, "ramp")
        self.configure("--no-scope-trigger")
        self.assertFalse(settings.SCOPE_TRIGGER)

    def test_shape_selection_reaches_settings(self):
        for shape in ("ramp", "step"):
            self.configure("--scope-trigger-shape", shape)
            self.assertEqual(settings.SCOPE_TRIGGER_SHAPE, shape)

    def test_invalid_duration_rejected(self):
        for duration in ("0", "-1", "nan", "inf"):
            with self.subTest(duration=duration), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as err:
                    self.configure("--scope-trigger-us", duration)
                self.assertEqual(err.exception.code, 2)

    def test_timing_selection_reaches_settings_and_printed_flags(self):
        for timing in ("fixed", "dwell"):
            self.configure("--scope-yt-timing", timing)
            self.assertEqual(settings.SCOPE_YT_TIMING, timing)
        # Only the non-default is worth printing back.
        self.assertIn("--scope-yt-timing fixed",
                      as_flags(dict(yt=True, yt_timing="fixed")))
        self.assertNotIn("--scope-yt-timing",
                         as_flags(dict(yt=True, yt_timing="dwell")))

    def test_deprecated_scope_yt_still_selects_fixed_row_timing(self):
        self.configure("--scope-yt")
        self.assertEqual(settings.SCOPE_YT_TIMING, "fixed")
        # ...and no longer drags a renderer, a sweep or a mode lock with it.
        self.assertTrue(settings.SCOPE_TRIGGER)

    def test_combinations_that_used_to_be_refused_now_start(self):
        # Each of these called parser.error() before. The marker is applied to
        # the finished frame, so none of them can conflict with it.
        for flags in (("--scope-realtime",),
                      ("--scope-mode", "stochastic"),
                      ("--scope-stipple",),
                      ("--scope-mix", "120"),
                      ("--scope-sweep", "palindrome")):
            with self.subTest(flags=flags):
                self.configure(*flags)
                self.assertTrue(settings.SCOPE_TRIGGER)

    def test_fixed_timing_degrades_instead_of_exiting(self):
        # Raster-only, so asking for it in stochastic has to give way -- with
        # a message on stdout, not a SystemExit.
        settings.SCOPE_YT_TIMING = "fixed"
        settings.SCOPE_RENDER_MODE = "stochastic"
        settings.SCOPE_REALTIME = False
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            with self.assertRaises(Exception) as err:
                scope_display.run_scope()
        # It got past configuration and failed later, on missing image data.
        self.assertNotIsInstance(err.exception, SystemExit)
        self.assertIn("raster only", out.getvalue())


if __name__ == "__main__":
    unittest.main()
