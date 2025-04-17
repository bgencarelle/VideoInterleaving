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

# Subclass TCPServer to allow immediate port reuse
class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

web_port = WEB_PORT

# Initial monitor data with all expected keys
monitor_data = {
    "index": 0,
    "displayed": 0,
    "delta": 0,
    "fps": 0,
    "fifo_depth": 0,
    "staleness_ms": 0,
    "comp_slope": 0,
    "latency_state": "unknown",
    # Folder selection probes
    "main_folder": 0,
    "float_folder": 0,
    "rand_mult": 0,
    "main_folder_count": 0,
    "float_folder_count": 0,
    "successful_frame": False,
    "fifo_miss_count": 0,
    "last_fifo_miss": -1,
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
    # Error tracking
    "failed_load_count": 0,
    "failed_indices": "",
    "last_error": "",
}

# HTML template for the monitor web UI
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
  <div id="monitor_data"></div>
  <script>
    function updateMonitor() {
      fetch('/data')
        .then(r => r.json())
        .then(d => {
          let html = '';
          for (const [k, v] of Object.entries(d)) {
            html += `<div><span class='label'>${k}:</span> <span class='value'>${v}</span></div>`;
          }
          document.getElementById('monitor_data').innerHTML = html;
        })
        .catch(console.error);
    }
    setInterval(updateMonitor, 100);
    updateMonitor();
  </script>
</body>
</html>
"""

# Compute normalized Shannon entropy (0.0‒1.0)
def _entropy(counts):
    tot = sum(counts)
    if tot == 0:
        return 0.0
    h = -sum((c/tot) * log2(c/tot) for c in counts if c)
    return h / log2(len(counts))

# HTTP handler
class MonitorHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        return  # suppress logs

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode('utf-8'))
        elif self.path == "/data":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(monitor_data).encode('utf-8'))
        else:
            self.send_error(404)

# Start the web server
def _serve():
    print(f"🛰️  Starting monitor on port {web_port}")
    with ReusableTCPServer(("", web_port), MonitorHTTPRequestHandler) as httpd:
        httpd.serve_forever()

# Public API to begin monitoring
def start_monitor():
    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()
    return MonitorUpdater()

# Main updater
class MonitorUpdater:
    def __init__(self):
        self.last_success_time    = time.monotonic()
        self.comp_history         = deque(maxlen=30)
        self.failed_indices       = deque(maxlen=20)
        self.failed_load_count    = 0
        self.script_start_time    = time.monotonic()
        self.last_heavy_update    = 0
        self.heavy_interval       = 0.5
        self.sys_stats_interval   = 1.0
        self.last_sys_stats_time  = 0
        self.cached_sys_stats     = {}
        self.cpu_per_core_history = deque(maxlen=5)
        self.main_counts          = None
        self.float_counts         = None

    @staticmethod
    def _fmt_duration(seconds):
        d, rem = divmod(int(seconds), 86400)
        h, rem  = divmod(rem, 3600)
        m, s    = divmod(rem, 60)
        return f"{d}d {h:02}:{m:02}:{s:02}"

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
        # Unpack quick payload fields
        idx       = payload.get("index", 0)
        displayed = payload.get("displayed", 0)
        # Reset staleness timer whenever displayed catches up to index
        if idx == displayed:
            self.last_success_time = now
        # Unpack quick payload fields
        idx               = payload.get("index", 0)
        displayed = payload.get("displayed", 0)
        # Track current playback offset
        offset = payload.get("offset", 0)
        delta             = idx - displayed
        fps               = payload.get("fps", 0)
        fifo_depth        = payload.get("fifo_depth", monitor_data["fifo_depth"])
        successful_frame  = payload.get("successful_frame", monitor_data.get("successful_frame", False))
        main_folder       = payload.get("main_folder", monitor_data["main_folder"])
        float_folder      = payload.get("float_folder", monitor_data["float_folder"])
        rand_mult         = payload.get("rand_mult", monitor_data["rand_mult"])
        mf_count          = payload.get("main_folder_count", monitor_data.get("main_folder_count", 0))
        ff_count          = payload.get("float_folder_count",monitor_data.get("float_folder_count",0))
        fifo_miss_count   = payload.get("fifo_miss_count",   monitor_data["fifo_miss_count"])
        last_fifo_miss    = payload.get("last_fifo_miss",    monitor_data["last_fifo_miss"])

        # Initialize folder histograms once
        if mf_count and self.main_counts is None:
            self.main_counts = [0] * mf_count
        if ff_count and self.float_counts is None:
            self.float_counts = [0] * ff_count
        # Count picks
        if self.main_counts and 0 <= main_folder < len(self.main_counts):
            self.main_counts[main_folder] += 1
        if self.float_counts and 0 <= float_folder < len(self.float_counts):
            self.float_counts[float_folder] += 1

        # Quick update of monitor_data
        monitor_data.update({
            "index": idx,
            "displayed": displayed,
            "delta": delta,
            "fps": f"{fps:.1f}" if isinstance(fps, float) else fps,
            "fifo_depth": fifo_depth,
            "staleness_ms": f"{(now - self.last_success_time) * 1000:.0f}",
            "main_folder": main_folder,
            "float_folder": float_folder,
            "rand_mult": rand_mult,
            "main_folder_count": mf_count,
            "float_folder_count": ff_count,
            "successful_frame": successful_frame,
            "fifo_miss_count": fifo_miss_count,
            "last_fifo_miss": last_fifo_miss,
        })

        # Rolling slope computation
        # Track offset for slope computation
        self.comp_history.append(offset)
        if len(self.comp_history) >= 2:
            monitor_data["comp_slope"] = self.comp_history[-1] - self.comp_history[0]
        # Latency state
        if abs(delta) <= 1:
            monitor_data["latency_state"] = "Delta synced"
        elif delta < -1:
            monitor_data["latency_state"] = "Delta ahead"
        else:
            monitor_data["latency_state"] = "Delta behind"

        # Heavy updates
        if now - self.last_heavy_update >= self.heavy_interval:
            # Cache expensive sys stats only every sys_stats_interval
            if now - self.last_sys_stats_time >= self.sys_stats_interval:
                try:
                    vm = psutil.virtual_memory()
                    du = psutil.disk_usage("/")
                    uptime = time.time() - psutil.boot_time()
                    load = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)
                    proc = psutil.Process(os.getpid())
                    self.cached_sys_stats = {
                        "mem_used": f"{vm.used // (1024**2)} MB",
                        "mem_total": f"{vm.total // (1024**2)} MB",
                        "machine_uptime": self._fmt_duration(uptime),
                        "load_avg": ", ".join(f"{x:.2f}" for x in load),
                        "disk_root": f"{du.percent}%",
                        "threads": threading.active_count(),
                        "proc_count": len(psutil.pids()),
                        "python_mem_mb": f"{proc.memory_info().rss // (1024**2)} MB",
                    }
                except Exception:
                    pass
                self.last_sys_stats_time = now

            # CPU smoothing each heavy tick
            try:
                self.cpu_per_core_history.append(psutil.cpu_percent(percpu=True))
                avg_core = [sum(c)/len(self.cpu_per_core_history) for c in zip(*self.cpu_per_core_history)]
                avg_cpu  = sum(avg_core)/len(avg_core) if avg_core else 0
            except Exception:
                avg_core = []
                avg_cpu  = 0

            # Merge cached stats + CPU + script uptimes
            monitor_data.update({
                **self.cached_sys_stats,
                "cpu_percent": avg_cpu,
                "cpu_per_core": [f"{p:.1f}%" for p in avg_core],
                "script_uptime": self._fmt_duration(now - self.script_start_time),
            })

            # Folder coverage & entropy
            if self.main_counts:
                covered = sum(1 for c in self.main_counts if c)
                total   = len(self.main_counts)
                monitor_data.update({
                    "main_covered": f"{covered}/{total}",
                    "main_entropy": f"{_entropy(self.main_counts):.2f}",
                    "main_samples": sum(self.main_counts),
                })
            if self.float_counts:
                covered = sum(1 for c in self.float_counts if c)
                total   = len(self.float_counts)
                monitor_data.update({
                    "float_covered": f"{covered}/{total}",
                    "float_entropy": f"{_entropy(self.float_counts):.2f}",
                    "float_samples": sum(self.float_counts),
                })

            self.last_heavy_update = now

        return monitor_data
