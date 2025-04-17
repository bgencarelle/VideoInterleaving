import http.server
import socketserver
import threading
import json
import time
import os
import psutil
import socket
import platform
from collections import deque
from math import log2  # for entropy calc
from settings import WEB_PORT

# Subclass TCPServer to allow address reuse selectively
class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

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
    # Folder‑selection probes
    "main_folder": 0,
    "float_folder": 0,
    "rand_mult": 0,
    "main_covered": "0/0",
    "float_covered": "0/0",
    "main_entropy": "0.00",
    "float_entropy": "0.00",
    "main_samples": 0,
    "float_samples": 0,
    # System diagnostics
    "cpu_percent": 0,
    "cpu_per_core": [],
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
  <div id=\"monitor_data\"></div>
  <script>
    function updateMonitor() {
      fetch('/data')
        .then(r => r.json())
        .then(d => {
          let html = '';
          for (const [k, v] of Object.entries(d))
            html += `<div><span class='label'>${k}:</span> <span class='value'>${v}</span></div>`;
          document.getElementById('monitor_data').innerHTML = html;
        });
    }
    setInterval(updateMonitor, 100);
    updateMonitor();
  </script>
</body>
</html>
"""

# Helper – Shannon entropy (0‑1)
def _entropy(counts):
    tot = sum(counts)
    if tot == 0:
        return 0.0
    return -sum((c/tot) * log2(c/tot) for c in counts if c) / log2(len(counts))

class MonitorHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        return

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode())
        elif self.path == "/data":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(monitor_data).encode())
        else:
            self.send_error(404)


def start_monitor_server():
    handler = MonitorHTTPRequestHandler
    print(f"🛰️  Starting monitor on port {web_port}")
    with ReusableTCPServer(("", web_port), handler) as httpd:
        httpd.serve_forever()


def start_monitor():
    thread = threading.Thread(target=start_monitor_server, daemon=True)
    thread.start()
    return MonitorUpdater()

class MonitorUpdater:
    def __init__(self):
        self.last_success_time = time.monotonic()
        self.comp_history = []
        self.failed_indices = deque(maxlen=20)
        self.failed_load_count = 0
        self.script_start_time = time.time()
        self.last_heavy_update = 0
        self.heavy_interval = 0.5
        # Reduced system-call frequency: cache sys stats every 1s
        self.sys_stats_interval = 1.0
        self.last_sys_stats_time = 0
        self.cached_sys_stats = {}
        # CPU smoothing history buffer
        self.cpu_per_core_history = deque(maxlen=5)
        # Folder coverage tracking
        self.main_counts = None
        self.float_counts = None

    @staticmethod
    def _format_duration(seconds):
        days = int(seconds) // 86400
        hours = (int(seconds) % 86400) // 3600
        minutes = (int(seconds) % 3600) // 60
        secs = int(seconds) % 60
        return f"{days}d {hours:02}:{minutes:02}:{secs:02}"

    def record_load_error(self, index, err):
        self.failed_load_count += 1
        self.failed_indices.append(index)
        monitor_data.update({
            "failed_load_count": self.failed_load_count,
            "failed_indices": ",".join(map(str, self.failed_indices)),
            "last_error": str(err),
        })

    def update(self, payload):
        now = time.monotonic()
        idx = payload.get("index", 0)
        displayed = payload.get("displayed", 0)
        delta = idx - displayed
        offset = payload.get("offset", 0)
        fps = payload.get("fps", 0)

        if payload.get("successful_frame", False):
            self.last_success_time = now

        main_folder = payload.get("main_folder", 0)
        float_folder = payload.get("float_folder", 0)
        rand_mult = payload.get("rand_mult", 0)
        # Initialize counts on first payload
        N_main = payload.get("main_folder_count")
        N_float = payload.get("float_folder_count")
        if N_main is not None and self.main_counts is None:
            self.main_counts = [0] * N_main
        if N_float is not None and self.float_counts is None:
            self.float_counts = [0] * N_float
        # Increment counts
        if self.main_counts and 0 <= main_folder < len(self.main_counts):
            self.main_counts[main_folder] += 1
        if self.float_counts and 0 <= float_folder < len(self.float_counts):
            self.float_counts[float_folder] += 1

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

        # Rolling slope calculation
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

        # Heavy telemetry + sys stats caching
        if now - self.last_heavy_update >= self.heavy_interval:
            # Refresh cached sys stats only every sys_stats_interval
            if now - self.last_sys_stats_time >= self.sys_stats_interval:
                try:
                    # System memory and disk
                    vm = psutil.virtual_memory()
                    du = psutil.disk_usage("/")
                    uptime_seconds = time.time() - psutil.boot_time()
                    load = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)
                    proc = psutil.Process(os.getpid())

                    # Cache values
                    self.cached_sys_stats = {
                        "mem_used": f"{vm.used // (1024 ** 2)} MB",
                        "mem_total": f"{vm.total // (1024 ** 2)} MB",
                        "machine_uptime": self._format_duration(uptime_seconds),
                        "load_avg": ", ".join(f"{x:.2f}" for x in load),
                        "disk_root": f"{du.percent}%",
                        "threads": threading.active_count(),
                        "proc_count": len(psutil.pids()),
                        "python_mem_mb": f"{proc.memory_info().rss // (1024 ** 2)} MB",
                    }
                except Exception:
                    pass
                self.last_sys_stats_time = now

            # CPU smoothing remains at heavy_interval
            raw_percents = psutil.cpu_percent(percpu=True)
            self.cpu_per_core_history.append(raw_percents)
            avg_per_core = [sum(core)/len(self.cpu_per_core_history) for core in zip(*self.cpu_per_core_history)]
            average_cpu = sum(avg_per_core) / len(avg_per_core) if avg_per_core else 0

            # Update monitor_data with cached sys stats + cpu
            monitor_data.update({
                "cpu_percent": average_cpu,
                "cpu_per_core": [f"{p:.1f}%" for p in avg_per_core],
                **self.cached_sys_stats
            })

            # Folder coverage & entropy
            if self.main_counts:
                covered = sum(1 for c in self.main_counts if c)
                total = len(self.main_counts)
                monitor_data.update({
                    "main_covered": f"{covered}/{total}",
                    "main_entropy": f"{_entropy(self.main_counts):.2f}",
                    "main_samples": sum(self.main_counts),
                })
            if self.float_counts:
                covered = sum(1 for c in self.float_counts if c)
                total = len(self.float_counts)
                monitor_data.update({
                    "float_covered": f"{covered}/{total}",
                    "float_entropy": f"{_entropy(self.float_counts):.2f}",
                    "float_samples": sum(self.float_counts),
                })

            self.last_heavy_update = now

        return monitor_data
