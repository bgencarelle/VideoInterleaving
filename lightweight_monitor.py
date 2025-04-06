import http.server
import socketserver
import threading
import json
import time
import os
import psutil
import socket
import platform
import subprocess
from collections import deque
from settings import WEB_PORT
web_port = WEB_PORT

monitor_data = {
    "index": 0,
    "displayed": 0,
    "delta": 0,
    "fps": 0,
    "fifo_depth": 0,
    "staleness_ms": 0,
    "comp_slope": 0,
    "latency_state": "unknown",
    # System diagnostics:
    "cpu_percent": 0,
    "mem_used": "0 MB",
    "mem_total": "0 MB",
    "uptime": "0:00:00",
    "load_avg": "0.0, 0.0, 0.0",
    "disk_root": "0%",
    "threads": 0,
    "proc_count": 0,
    "python_mem_mb": "0 MB",
    "hostname": socket.gethostname(),
    "platform": platform.platform(),
    # Tracking failures
    "failed_load_count": 0,
    "failed_indices": "",
    "last_error": "",
}

HTML_TEMPLATE = """
<html>
<head><meta http-equiv='refresh' content='2'><style>
body { background: #111; color: #0f0; font-family: monospace; padding: 2em; }
.label { color: #888; margin-right: 1em; }
</style></head>
<body>
<h1>🎧 Live Playback Monitor</h1>
%s
</body></html>
"""

class MonitorHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress HTTP log messages to terminal
        return

    def do_GET(self):
        if self.path == "/":
            html = "".join(
                f"<div><span class='label'>{k}:</span> <span class='value'>{v}</span></div>"
                for k, v in monitor_data.items()
            )
            response = HTML_TEMPLATE % html
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(response.encode("utf-8"))

        elif self.path == "/data":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(monitor_data).encode("utf-8"))
        else:
            self.send_error(404)

def kill_old_server(port=web_port):
    """Attempt to kill any existing server using the webport."""
    import subprocess
    try:
        output = subprocess.check_output(
            ["lsof", "-ti", f":{port}"], stderr=subprocess.DEVNULL
        ).decode()
        pids = output.strip().split("\n")
        for pid in pids:
            if pid:
                os.kill(int(pid), 9)
    except Exception:
        pass

def start_monitor_server():
    handler = MonitorHTTPRequestHandler
    print(f"🛰️  Starting monitor on port {web_port}")  # Confirm actual port value
    with socketserver.TCPServer(("", web_port), handler) as httpd:
        httpd.serve_forever()

def start_monitor():
    kill_old_server()  # Kill any existing monitor instance first
    thread = threading.Thread(target=start_monitor_server, daemon=True)
    thread.start()
    return MonitorUpdater()

class MonitorUpdater:
    def __init__(self):
        self.last_success_time = time.monotonic()
        self.comp_history = []
        self.failed_indices = deque(maxlen=20)
        self.failed_load_count = 0

    def record_load_error(self, index, err):
        """Called when an async image load fails."""
        self.failed_load_count += 1
        self.failed_indices.append(index)
        monitor_data["failed_load_count"] = self.failed_load_count
        monitor_data["failed_indices"] = ",".join(map(str, self.failed_indices))
        monitor_data["last_error"] = str(err)

    def update(self, payload):
        now = time.monotonic()

        idx = payload.get("index", 0)
        displayed = payload.get("displayed", 0)
        delta = idx - displayed
        offset = payload.get("offset", 0)
        fps = payload.get("fps", 0)

        # If we displayed a frame successfully, reset staleness timer
        if payload.get("successful_frame", False):
            self.last_success_time = now

        monitor_data.update({
            "index": idx,
            "displayed": displayed,
            "delta": delta,
            "fps": f"{fps:.1f}" if isinstance(fps, float) else fps,
            "fifo_depth": payload.get("fifo_depth", monitor_data["fifo_depth"]),
            "staleness_ms": f"{(now - self.last_success_time) * 1000:.0f}"
        })

        # Rolling slope for comp history
        self.comp_history.append(offset)
        if len(self.comp_history) > 30:
            self.comp_history.pop(0)

        slope = (self.comp_history[-1] - self.comp_history[0]) if len(self.comp_history) >= 2 else 0
        monitor_data["comp_slope"] = slope

        if abs(delta) <= 1:
            state = "Delta synced"
        elif delta < -1:
            state = "Delta ahead"
        else:
            state = "Delta behind"
        monitor_data["latency_state"] = state

        # System stats
        try:
            vm = psutil.virtual_memory()
            du = psutil.disk_usage("/")
            uptime_seconds = time.time() - psutil.boot_time()
            load = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)
            proc = psutil.Process(os.getpid())

            monitor_data.update({
                "cpu_percent": psutil.cpu_percent(),
                "mem_used": f"{vm.used // (1024**2)} MB",
                "mem_total": f"{vm.total // (1024**2)} MB",
                "uptime": time.strftime("%H:%M:%S", time.gmtime(uptime_seconds)),
                "load_avg": ", ".join(f"{x:.2f}" for x in load),
                "disk_root": f"{du.percent}%",
                "threads": threading.active_count(),
                "proc_count": len(psutil.pids()),
                "python_mem_mb": f"{proc.memory_info().rss // (1024**2)} MB"
            })
        except Exception:
            pass

        return monitor_data
