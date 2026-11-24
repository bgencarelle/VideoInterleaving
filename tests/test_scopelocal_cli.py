"""--mode scopelocal: scope mode with its trace preview fullscreen on screen.

Runs main's real configuration code (as test_scope_yt_cli does) without
starting servers, audio or a window.
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


class ScopeLocalTests(unittest.TestCase):
    def setUp(self):
        self.saved_settings = vars(settings).copy()
        self.addCleanup(self.restore_settings)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        path = Path(__file__).resolve().parents[1] / "main.py"
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
        argv = ["main.py", "--mode", "scopelocal", "--device", "null", *flags]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            return self.ns["configure_runtime"]()

    def test_fullscreen_preview_of_live_images(self):
        self.configure("--dir", self.temp.name)
        self.assertTrue(settings.SCOPE_GUI)
        self.assertTrue(settings.SCOPE_GUI_FULLSCREEN)
        self.assertTrue(settings.SCOPE_GUI_IMAGE_ONLY)
        self.assertEqual(settings.SCOPE_SOURCE, "images")

    def test_a_bake_can_still_be_chosen(self):
        self.configure("--scope-source", "bake", "--xy-dir", self.temp.name)
        self.assertEqual(settings.SCOPE_SOURCE, "bake")
        self.assertTrue(settings.SCOPE_GUI_FULLSCREEN)

    def test_other_views_belong_to_scope_mode(self):
        for flag in ("--no-scope-gui", "--no-scope-gui-fullscreen"):
            with self.subTest(flag=flag), self.assertRaises(SystemExit):
                self.configure(flag)


if __name__ == "__main__":
    unittest.main()
