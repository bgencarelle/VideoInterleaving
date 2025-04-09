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
    "cpu_per_core": [],  # <--- New field for per-core CPU usage
    "mem_used": "0 MB",
    "mem_total": "0 MB",
    "script_uptime": "0d 00:00:00",
    "machine_uptime": "0d 00:00:00",
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
<head>
  <style>
    body { background: #111; color: #0f0; font-family: monospace; padding: 2em; }
    .label { color: #888; margin-right: 1em; }
  </style>
</head>
<body>
  <h1>🎧 Live Playback Monitor</h1>
  <div id="monitor_data">
    <!-- Monitor data will be updated here -->
  </div>
  <script>
    function updateMonitor() {
      fetch('/data')
        .then(response => response.json())
        .then(data => {
          let html = "";
          for (const [key, value] of Object.entries(data)) {
            html += `<div><span class='label'>${key}:</span> <span class='value'>${value}</span></div>`;
          }
          document.getElementById("monitor_data").innerHTML = html;
        })
        .catch(error => console.error('Error fetching monitor data:', error));
    }

    // Update monitor data every 2 seconds; adjust as needed.
    setInterval(updateMonitor, 100);
    </script>
    // Initial data fetch
    updateMonitor();
  </script>
</body>
</html>
"""


class MonitorHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress HTTP log messages to terminal
        return

    def do_GET(self):
        if self.path == "/":
            # Send the full HTML page (which uses AJAX to update dynamic parts)
            response = HTML_TEMPLATE
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(response.encode("utf-8"))
        elif self.path == "/data":
            # Send only the JSON data payload
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(monitor_data).encode("utf-8"))
        else:
            self.send_error(404)


def kill_old_server(port=web_port):
    """Attempt to kill any existing server using the webport."""
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
        self.script_start_time = time.time()  # Track when the script started
        # Initialize a timer for heavy (expensive) updates:
        self.last_heavy_update = 0
        self.heavy_interval = 0.5  # seconds; adjust as needed

    @staticmethod
    def _format_duration(seconds):
        days = int(seconds) // 86400
        hours = (int(seconds) % 86400) // 3600
        minutes = (int(seconds) % 3600) // 60
        secs = int(seconds) % 60
        return f"{days}d {hours:02}:{minutes:02}:{secs:02}"

    def record_load_error(self, index, err):
        """Called when an async image load fails."""
        self.failed_load_count += 1
        self.failed_indices.append(index)
        monitor_data["failed_load_count"] = self.failed_load_count
        monitor_data["failed_indices"] = ",".join(map(str, self.failed_indices))
        monitor_data["last_error"] = str(err)

    def update(self, payload):
        now = time.monotonic()

        # ---- Fast, free updates from pygame:
        idx = payload.get("index", 0)
        displayed = payload.get("displayed", 0)
        delta = idx - displayed
        offset = payload.get("offset", 0)
        fps = payload.get("fps", 0)

        # If we displayed a frame successfully, reset the staleness timer.
        if payload.get("successful_frame", False):
            self.last_success_time = now

        # Accept new folder fields from pygame (which are free)
        main_folder = payload.get("main_folder", 0)
        float_folder = payload.get("float_folder", 0)
        rand_mult = payload.get("rand_mult", 0)

        # Update the quick part of monitor_data:
        monitor_data.update({
            "index": idx,
            "displayed": displayed,
            "delta": delta,
            "fps": f"{fps:.1f}" if isinstance(fps, float) else fps,
            "fifo_depth": payload.get("fifo_depth", monitor_data["fifo_depth"]),
            "staleness_ms": f"{(now - self.last_success_time) * 1000:.0f}",
            "main_folder": main_folder,
            "float_folder": float_folder,
            "rand_mult": rand_mult,
        })

        # Rolling slope update remains fast as it's just arithmetic:
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

        # ---- Heavy (expensive) system telemetry updates:
        # Only update heavy telemetry if sufficient time has passed:
        if now - self.last_heavy_update >= self.heavy_interval:
            try:
                vm = psutil.virtual_memory()
                du = psutil.disk_usage("/")
                uptime_seconds = time.time() - psutil.boot_time()
                load = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)
                proc = psutil.Process(os.getpid())

                # Gather per-core CPU usage
                cpu_percents = psutil.cpu_percent(percpu=True)
                average_cpu = sum(cpu_percents) / len(cpu_percents) if cpu_percents else 0

                monitor_data.update({
                    "cpu_percent": average_cpu,
                    "cpu_per_core": [f"{p:.1f}%" for p in cpu_percents],
                    "mem_used": f"{vm.used // (1024 ** 2)} MB",
                    "mem_total": f"{vm.total // (1024 ** 2)} MB",
                    "machine_uptime": self._format_duration(uptime_seconds),
                    "script_uptime": self._format_duration(time.time() - self.script_start_time),
                    "load_avg": ", ".join(f"{x:.2f}" for x in load),
                    "disk_root": f"{du.percent}%",
                    "threads": threading.active_count(),
                    "proc_count": len(psutil.pids()),
                    "python_mem_mb": f"{proc.memory_info().rss // (1024 ** 2)} MB"
                })
            except Exception:
                pass
            # Reset the heavy update timer
            self.last_heavy_update = now

        return monitor_data
