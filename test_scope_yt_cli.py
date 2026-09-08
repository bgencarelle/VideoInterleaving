"""Y-T configuration regressions; run with python -m unittest test_scope_yt_cli."""
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

    def test_aliases_reach_the_output_and_printed_flags(self):
        for flag in ("--scope-yt-trigger", "--scope-yt-trigger-us"):
            with self.subTest(flag=flag):
                self.configure("--scope-yt", flag, "500")
                self.assertTrue(settings.SCOPE_YT)
                self.assertEqual(settings.SCOPE_RENDER_MODE, "raster")
                self.assertEqual(settings.SCOPE_SWEEP, "retrace")
                scope = Scope(device="null", samplerate=96000, yt_mode=settings.SCOPE_YT,
                              yt_trigger_us=settings.SCOPE_YT_TRIGGER_US)
                try:
                    self.assertEqual(scope.yt_trigger_samples, 48)
                    state = dict(yt=True, mode="raster", raster=True,
                                 yt_trigger_us=scope.yt_trigger_us, sweep="retrace")
                    KeyMap(state).feed("v")
                    self.assertEqual(state["mode"], "raster")
                    self.assertIn("--scope-yt-trigger 500", as_flags(state))
                finally:
                    scope.stream.close()

    def test_invalid_duration_rejected(self):
        for duration in ("0", "-1", "nan", "inf"):
            with self.subTest(duration=duration), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as err:
                    self.configure("--scope-yt", "--scope-yt-trigger", duration)
                self.assertEqual(err.exception.code, 2)

    def test_timing_selection_reaches_settings_and_printed_flags(self):
        for timing in ("fixed", "dwell"):
            self.configure("--scope-yt", "--scope-yt-timing", timing)
            self.assertEqual(settings.SCOPE_YT_TIMING, timing)
            flags = as_flags(dict(yt=True, yt_timing=settings.SCOPE_YT_TIMING))
            self.assertIn("--scope-yt-timing " + timing, flags)

    def test_realtime_cli_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as err:
                self.configure("--scope-yt", "--scope-realtime")
            self.assertEqual(err.exception.code, 2)

    def test_realtime_settings_rejected_before_playback(self):
        settings.SCOPE_YT = True
        settings.SCOPE_REALTIME = True
        with self.assertRaisesRegex(ValueError, "requires complete traces"):
            scope_display.run_scope()


if __name__ == "__main__":
    unittest.main()
