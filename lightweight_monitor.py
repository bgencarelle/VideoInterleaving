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
from math import log2  # for entropy calc
from settings import WEB_PORT

# ---------------------------------------------------------------
# Configuration --------------------------------------------------
# ---------------------------------------------------------------
web_port = WEB_PORT

# Initial payload the browser sees; new keys auto‑appear there.
monitor_data = {
    # Fast playback diagnostics
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

# ---------------------------------------------------------------
# HTML template (browser pulls /data 10×/s) ----------------------
# ---------------------------------------------------------------
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

# ---------------------------------------------------------------
# Helper – Shannon entropy normalised to [0,1] ------------------
# ---------------------------------------------------------------

def _entropy(counts):
    tot = sum(counts)
    if tot == 0:
        return 0.0
    return -sum((c / tot) * log2(c / tot) for c in counts if c) / log2(len(counts))

# ---------------------------------------------------------------
# Web server -----------------------------------------------------
# ---------------------------------------------------------------
class MonitorHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        return  # silence spam

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


def _kill_old_server(port=web_port):
    """Kill any previous instance so dev restarts are painless."""
    try:
        pids = subprocess.check_output(["lsof", "-ti", f":{port}"], stderr=subprocess.DEVNULL).decode().split()
        for pid in pids:
            os.kill(int(pid), 9)
    except Exception:
        pass


def _serve():
    with socketserver.TCPServer(("", web_port), MonitorHTTPRequestHandler) as httpd:
        print(f"🛰️  Monitor on http://localhost:{web_port}")
        httpd.serve_forever()


def start_monitor():
    _kill_old_server()
    threading.Thread(target=_serve, daemon=True).start()
    return MonitorUpdater()

# ---------------------------------------------------------------
# Core updater ---------------------------------------------------
# ---------------------------------------------------------------
class MonitorUpdater:
    """Call .update(payload_dict) from your pygame loop."""

    def __init__(self):
        self.last_success_time = time.monotonic()
        self.comp_history = []
        self.failed_indices = deque(maxlen=20)
        self.failed_load_count = 0
        self.script_start_time = time.time()
        # heavy‑telemetry pacing
        self.last_heavy = 0
        self.heavy_interval = 0.5
        # folder hit arrays (lazy‑init when we know N)
        self.main_hits = None
        self.float_hits = None

    # -------------------------- helpers -------------------------
    @staticmethod
    def _fmt_dur(sec):
        d, h = divmod(int(sec), 86400)
        h, m = divmod(h, 3600)
        m, s = divmod(m, 60)
        return f"{d}d {h:02}:{m:02}:{s:02}"

    def record_load_error(self, idx, err):
        self.failed_load_count += 1
        self.failed_indices.append(idx)
        monitor_data.update({
            "failed_load_count": self.failed_load_count,
            "failed_indices": ",".join(map(str, self.failed_indices)),
            "last_error": str(err),
        })

    # -------------------------- update --------------------------
    def update(self, payload):
        now = time.monotonic()

        # ---------- fast fields ---------------------------------
        idx = payload.get("index", 0)
        displayed = payload.get("displayed", 0)
        delta = idx - displayed
        fps = payload.get("fps", 0)
        offset = payload.get("offset", 0)

        if payload.get("successful_frame"):  # resets staleness timer
            self.last_success_time = now

        main_folder = payload.get("main_folder", 0)
        float_folder = payload.get("float_folder", 0)
        rand_mult = payload.get("rand_mult", 0)

        # ---------- folder bookkeeping --------------------------
        N_main = payload.get("main_folder_count")
        N_float = payload.get("float_folder_count")

        if N_main is not None and self.main_hits is None:
            self.main_hits = [0] * N_main
        if N_float is not None and self.float_hits is None:
            self.float_hits = [0] * N_float

        if self.main_hits and 0 <= main_folder < len(self.main_hits):
            self.main_hits[main_folder] += 1
        if self.float_hits and 0 <= float_folder < len(self.float_hits):
            self.float_hits[float_folder] += 1

        # ---------- quick push to dict --------------------------
        monitor_data.update({
            "index": idx,
            "displayed": displayed,
            "delta": delta,
            "fps": f"{fps:.1f}" if isinstance(fps, float) else fps,
            "fifo_depth": payload.get("fifo_depth", monitor_data["fifo_depth"]),
            "staleness_ms": f"{(now - self.last_success_time)*1000:.0f}",
            "main_folder": main_folder,
            "float_folder": float_folder,
            "rand_mult": rand_mult,
        })

        # compensation slope (30‑sample window)
        self.comp_history.append(offset)
        if len(self.comp_history) > 30:
            self.comp_history.pop(0)
        if len(self.comp_history) >= 2:
            monitor_data["comp_slope"] = self.comp_history[-1] - self.comp_history[0]

        monitor_data["latency_state"] = (
            "Delta synced" if abs(delta) <= 1 else "Delta ahead" if delta < -1 else "Delta behind"
        )

        # ---------- heavy telemetry every self.heavy_interval ----
        if now - self.last_heavy >= self.heavy_interval:
            try:
                vm = psutil.virtual_memory()
                du = psutil.disk_usage("/")
                uptime = time.time() - psutil.boot_time()
                load = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)
                proc = psutil.Process(os.getpid())

                cpu_perc = psutil.cpu_percent(percpu=True)
                monitor_data.update({
                    "cpu_percent": sum(cpu_perc)/len(cpu_perc) if cpu_perc else 0,
                    "cpu_per_core": [f"{p:.1f}%" for p in cpu_perc],
                    "mem_used": f"{vm.used//(1024**2)} MB",
                    "mem_total": f"{vm.total//(1024**2)} MB",
                    "machine_uptime": self._fmt_dur(uptime),
                    "script_uptime": self._fmt_dur(time.time() - self.script_start_time),
                    "load_avg": ", ".join(f"{x:.2f}" for x in load),
                    "disk_root": f"{du.percent}%",
                    "threads": threading.active_count(),
                    "proc_count": len(psutil.pids()),
                    "python_mem_mb": f"{proc.memory_info().rss//(1024**2)} MB",
                })

                # folder metrics
                if self.main_hits:
                    covered = sum(1 for c in self.main_hits if c)
                    total = len(self.main_hits)
                    monitor_data.update({
                        "main_covered": f"{covered}/{total}",
                        "main_entropy": f"{_entropy(self.main_hits):.2f}",
                        "main_samples": sum(self.main_hits),
                    })
                if self.float_hits:
                    covered = sum(1 for c in self.float_hits if c)
                    total = len(self.float_hits)
                    monitor_data.update({
                        "float_covered": f"{covered}/{total}",
                        "float_entropy": f"{_entropy(self.float_hits):.2f}",
                        "float_samples": sum(self.float_hits),
                    })
            except Exception:
                pass
            self.last_heavy = now

        return monitor_data
