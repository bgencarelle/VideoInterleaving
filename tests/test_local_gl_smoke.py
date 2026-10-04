"""Local-mode GL smoke test using Xvfb and Mesa software rendering.

Run with: .venv/bin/python -m unittest tests.test_local_gl_smoke -v
"""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SMOKE_RUNNER = ROOT / "tools" / "local_gl_smoke.py"


def _make_images(root):
    for folder, colors in (
        (root / "face" / "0_main", ((210, 40, 30, 255),
                                     (30, 180, 90, 255))),
        (root / "float" / "255_overlay", ((20, 40, 230, 150),
                                           (230, 210, 20, 150))),
    ):
        folder.mkdir(parents=True)
        for index, color in enumerate(colors):
            Image.new("RGB", (160, 120), color[:3]).save(
                folder / f"frame_{index}.jpg", quality=90)


class LocalGLSmokeTests(unittest.TestCase):
    def test_local_mode_renders_with_headless_software_gl(self):
        xvfb_run = shutil.which("xvfb-run")
        if (not sys.platform.startswith("linux") or not xvfb_run or
                not shutil.which("xauth")):
            self.skipTest("requires Linux xvfb-run and Mesa GL")

        # Local mode performs its normal startup port check even though the
        # smoke runner disables listener startup.
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", 8888))
            except OSError:
                self.skipTest("local-mode monitor port 8888 is busy")

        tmp_root = ROOT / "tmp"
        tmp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="local-gl-smoke-",
                                         dir=tmp_root) as temp:
            work = Path(temp)
            source = work / f"source_{work.name}"
            source.mkdir()
            _make_images(source)
            logs = work / "logs"
            result_path = work / "result.json"
            env = dict(os.environ)
            env["LIBGL_ALWAYS_SOFTWARE"] = "1"
            command = [
                xvfb_run, "-a", sys.executable, str(SMOKE_RUNNER),
                "--dir", str(source), "--seconds", "3",
                "--logs-dir", str(logs), "--result", str(result_path),
            ]
            suffix = f"{source.name}_local_None"
            try:
                run = subprocess.run(command, cwd=ROOT, env=env, text=True,
                                     capture_output=True, timeout=30)

                self.assertEqual(run.returncode, 0,
                                 run.stdout + "\n" + run.stderr)
                report = json.loads(result_path.read_text(encoding="utf-8"))
                self.assertTrue(report["gl_ready"], report)
                self.assertIn("GL_RENDERER:", report["renderer"])
                self.assertGreater(report["gl_composite"], 0, report)
                self.assertGreater(report["cpu_composite"], 0, report)
                self.assertEqual(report["bridge_frame_shape"], [120, 80, 3],
                                 report)
                self.assertTrue(report["timer_expired"], report)
            finally:
                for cache_name in (f"folders_processed_{suffix}",
                                   f"generated_lists_{suffix}"):
                    shutil.rmtree(ROOT / "_cache" / cache_name,
                                  ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
