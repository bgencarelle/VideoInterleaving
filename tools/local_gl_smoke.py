"""Bounded local-mode GL smoke runner for Xvfb/Mesa.

Example:
    xvfb-run -a env LIBGL_ALWAYS_SOFTWARE=1 \
        .venv/bin/python tools/local_gl_smoke.py --dir images_sbs

The runner counts actual GL composite calls and rejects the CPU compositor
path. It disables app listeners and the Wayland-only pointer helper so this
diagnostic has no network or desktop side effects.
"""
import argparse
import importlib
import io
import json
from pathlib import Path
import signal
import sys


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default="images_sbs", help="image source tree")
    ap.add_argument("--seconds", type=float, default=5.0,
                    help="bounded local render duration")
    ap.add_argument("--logs-dir", default="tmp/local_gl_smoke/logs")
    ap.add_argument("--result", help="optional JSON result path")
    args = ap.parse_args()

    source_root = Path(args.dir).resolve()
    logs_dir = Path(args.logs_dir).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    source_name = source_root.name.replace(" ", "_")
    log_path = logs_dir / f"runtime_{source_name}_local_None.log"
    result_path = Path(args.result).resolve() if args.result else None

    # main.py parses argv at module execution and initializes its log stream
    # there. Redirect only its log directory in this in-memory copy, leaving
    # the application source untouched.
    sys.argv = [str(MAIN), "--mode", "local", "--dir", str(source_root)]
    source = MAIN.read_text(encoding="utf-8")
    original_log_setting = 'LOGS_DIR = "logs"'
    if source.count(original_log_setting) != 1:
        raise RuntimeError("could not isolate the main.py log directory")
    source = source.replace(original_log_setting,
                            f"LOGS_DIR = {str(logs_dir)!r}", 1)

    # main's Tee still records the run to its per-run logfile, but its console
    # leg should not contend with the render loop during this smoke test.
    sys.stdout = io.StringIO()
    namespace = {"__file__": str(MAIN), "__name__": "_local_gl_smoke_main"}
    exec(compile(source, str(MAIN), "exec"), namespace)

    renderer = importlib.import_module("renderer")
    counts = {"gl_composite": 0, "cpu_composite": 0}
    gl_composite = renderer.overlay_images_single_pass
    cpu_composite = renderer.composite_cpu

    def counted_gl(*call_args, **call_kwargs):
        counts["gl_composite"] += 1
        return gl_composite(*call_args, **call_kwargs)

    def counted_cpu(*call_args, **call_kwargs):
        counts["cpu_composite"] += 1
        return cpu_composite(*call_args, **call_kwargs)

    renderer.overlay_images_single_pass = counted_gl
    renderer.composite_cpu = counted_cpu

    # This is an X11 smoke run, so avoid the optional Wayland pointer utility.
    display_manager = importlib.import_module("display_manager")
    display_manager._move_wlrctl_offscreen_once = lambda _window: None

    # No viewer is needed to verify local GL compositing; keep the test from
    # opening any application listeners.
    lightweight_monitor = importlib.import_module("lightweight_monitor")
    lightweight_monitor.start_monitor = lambda *a, **k: None
    web_service = importlib.import_module("web_service")
    web_service.start_server = lambda *a, **k: None

    timer_expired = False

    def stop_run(_signum, _frame):
        nonlocal timer_expired
        timer_expired = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGALRM, stop_run)
    signal.setitimer(signal.ITIMER_REAL, args.seconds)
    app_status = 0
    try:
        namespace["main"]()
    except SystemExit as exc:
        app_status = exc.code if isinstance(exc.code, int) else 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

    log_text = log_path.read_text(encoding="utf-8", errors="replace") \
        if log_path.exists() else ""
    renderer_line = next((line for line in log_text.splitlines()
                          if "[DISPLAY] GL_RENDERER:" in line), None)
    backend_ready = ("[DISPLAY] Renderer backend: moderngl" in log_text or
                     "[DISPLAY] Renderer backend: legacy (PyOpenGL)" in log_text)
    gl_ready = backend_ready and renderer_line is not None
    report = {
        **counts,
        "gl_ready": gl_ready,
        "renderer": renderer_line,
        "timer_expired": timer_expired,
        "app_status": app_status,
        "log_path": str(log_path),
    }
    if result_path:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(report, indent=2) + "\n",
                               encoding="utf-8")
    print(json.dumps(report, indent=2), file=sys.__stdout__)

    return 0 if (app_status == 0 and gl_ready and timer_expired and
                  counts["gl_composite"] > 0 and
                  counts["cpu_composite"] == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
