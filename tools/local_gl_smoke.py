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
import socket
import sys
import threading
import time


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

    # Receive one real rendered frame through the same bridge used by the
    # sender source, while main.py runs its ordinary local display loop.
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(('127.0.0.1', 0))
    listener.listen(1)
    listener.settimeout(.2)
    bridge = {}

    def receive_bridge_frame():
        try:
            from local_frame_bridge import read_frame
            deadline = time.monotonic()+args.seconds+10
            while True:
                try:
                    connection, _address = listener.accept()
                    break
                except socket.timeout:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('local display did not connect to bridge')
            with connection:
                connection.settimeout(3.0)
                frame = read_frame(connection)
            bridge['shape'] = tuple(frame.shape)
            bridge['has_pixels'] = bool(frame.any())
            bridge['contrast'] = float(frame.std())
        except Exception as exc:
            bridge['error'] = str(exc)

    receiver = threading.Thread(target=receive_bridge_frame, daemon=True)
    receiver.start()

    # main.py parses argv at module execution and initializes its log stream
    # there. Redirect only its log directory in this in-memory copy, leaving
    # the application source untouched.
    sys.argv = [str(MAIN), "--mode", "local", "--dir", str(source_root),
                "--local-frame-port", str(listener.getsockname()[1])]
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
    counts = {"gl_composite": 0, "cpu_composite": 0,
              "frame_readback": 0}
    gl_composite = renderer.overlay_images_single_pass
    cpu_composite = renderer.composite_cpu
    read_frame_rgb = renderer.read_frame_rgb

    def counted_gl(*call_args, **call_kwargs):
        counts["gl_composite"] += 1
        return gl_composite(*call_args, **call_kwargs)

    def counted_cpu(*call_args, **call_kwargs):
        counts["cpu_composite"] += 1
        return cpu_composite(*call_args, **call_kwargs)

    def counted_readback(*call_args, **call_kwargs):
        counts['frame_readback'] += 1
        return read_frame_rgb(*call_args, **call_kwargs)

    renderer.overlay_images_single_pass = counted_gl
    renderer.composite_cpu = counted_cpu
    renderer.read_frame_rgb = counted_readback

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
        receiver.join(timeout=3.0)
        listener.close()

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
        "bridge_frame_shape": bridge.get('shape'),
        "bridge_frame_has_pixels": bridge.get('has_pixels', False),
        "bridge_frame_contrast": bridge.get('contrast', 0.0),
        "bridge_error": bridge.get('error'),
        "app_status": app_status,
        "log_path": str(log_path),
    }
    if result_path:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(report, indent=2) + "\n",
                               encoding="utf-8")
    print(json.dumps(report, indent=2), file=sys.__stdout__)

    return 0 if (app_status == 0 and gl_ready and timer_expired and
                  bridge.get('shape') and bridge.get('contrast', 0.0) > 1.0 and
                  counts["gl_composite"] > 0 and
                  counts["cpu_composite"] == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
