import time
import os
import psutil
import socket
import platform
import threading
from collections import deque
from math import log2

from settings import WEB_PORT

# --- DATA STORE ---
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
    "last_http_crash": "",
}

HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Interleaving Project Live Playback Monitor</title>
  <style>
    body { background:#111; color:#0f0; font-family:monospace; padding:2em; }
    .label{color:#888;margin-right:1em;}
    a { color:#0f0; }
  </style>
</head>
<body>
  <h1>Interleaving Project Live Playback Monitor</h1>
  <div><a href="log">click for error log</a></div>
  <div id="monitor_data"></div>

  <script>
    function render(d){
      let html = '';
      for (const [k, v] of Object.entries(d)) {
        html += `<div><span class='label'>${k}:</span>${v}</div>`;
      }
      const el = document.getElementById('monitor_data');
      if (el) el.innerHTML = html;
    }

    function poll(){
      // Fetch relative "data" endpoint on port 1978
      fetch('data', { cache: 'no-store' })
        .then(r => r.json())
        .then(render)
        .catch(() => {});
    }

    document.addEventListener('DOMContentLoaded', () => {
      poll();
      setInterval(poll, 100); // 1Hz poll to save CPU
    });
  </script>
</body>
</html>
"""

def _entropy(counts):
    tot = sum(counts)
    if tot == 0:
        return 0.0
    h = -sum((c/tot) * log2(c/tot) for c in counts if c)
    return h / log2(len(counts))

# Call this from image_display.py
def start_monitor():
    # Just returns the updater class.
    # The SERVER is now started by web_service.py
    return MonitorUpdater()

class MonitorUpdater:
    def __init__(self):
        self.last_success_time = time.monotonic()
        self.comp_history = deque(maxlen=30)
        self.failed_indices = deque(maxlen=20)
        self.failed_load_count = 0
        self.script_start_time = time.monotonic()
        self.last_heavy_update = 0
        self.heavy_interval = 5.0
        self.sys_stats_interval = 6.0
        self.last_sys_stats_time = 0
        self.cached_sys_stats = {}
        self.cpu_per_core_history = deque(maxlen=5)
        self.main_counts = None
        self.float_counts = None

    @staticmethod
    def _fmt_duration(secs):
        d, rem = divmod(int(secs), 86400)
        h, rem = divmod(rem, 3600)
        m, s = divmod(rem, 60)
        return f"{d}d {h:02}:{m:02}:{s:02}"

    def record_load_error(self, index, err):
        self.failed_load_count += 1
        self.failed_indices.append(index)
        monitor_data.update({
            'failed_load_count': self.failed_load_count,
            'failed_indices': ','.join(map(str, self.failed_indices)),
            'last_error': str(err)
        })

    def update(self, payload):
        # ... (Identical update logic as previous versions) ...
        # Copied for completeness
        now = time.monotonic()
        if payload.get('successful_frame', False):
            self.last_success_time = now

        idx = payload.get('index', 0)
        displayed = payload.get('displayed', 0)
        delta = idx - displayed
        fps = payload.get('fps', 0)
        fifo_depth = payload.get('fifo_depth', monitor_data['fifo_depth'])
        succ = payload.get('successful_frame', False)
        mf = payload.get('main_folder', monitor_data['main_folder'])
        ff = payload.get('float_folder', monitor_data['float_folder'])
        rm = payload.get('rand_mult', monitor_data['rand_mult'])
        mfc = payload.get('main_folder_count', monitor_data['main_folder_count'])
        ffc = payload.get('float_folder_count', monitor_data['float_folder_count'])
        fmc = payload.get('fifo_miss_count', monitor_data['fifo_miss_count'])
        lfm = payload.get('last_fifo_miss', monitor_data['last_fifo_miss'])

        if mfc and self.main_counts is None:
            self.main_counts = [0] * mfc
        if ffc and self.float_counts is None:
            self.float_counts = [0] * ffc
        if self.main_counts and 0 <= mf < len(self.main_counts):
            self.main_counts[mf] += 1
        if self.float_counts and 0 <= ff < len(self.float_counts):
            self.float_counts[ff] += 1

        monitor_data.update({
            'index': idx,
            'displayed': displayed,
            'delta': delta,
            'fps': f"{fps:.1f}" if isinstance(fps, float) else fps,
            'fifo_depth': fifo_depth,
            'staleness_ms': f"{(now - self.last_success_time) * 1000:.0f}",
            'main_folder': mf,
            'float_folder': ff,
            'rand_mult': rm,
            'main_folder_count': mfc,
            'float_folder_count': ffc,
            'successful_frame': succ,
            'fifo_miss_count': fmc,
            'last_fifo_miss': lfm,
        })

        self.comp_history.append(delta)
        if len(self.comp_history) > 1:
            monitor_data['comp_slope'] = self.comp_history[-1] - self.comp_history[0]

        if abs(delta) <= 1:
            monitor_data['latency_state'] = 'Delta synced'
        elif delta < -1:
            monitor_data['latency_state'] = 'Delta ahead'
        else:
            monitor_data['latency_state'] = 'Delta behind'

        if now - self.last_heavy_update >= self.heavy_interval:
            if now - self.last_sys_stats_time >= self.sys_stats_interval:
                try:
                    vm = psutil.virtual_memory()
                    du = psutil.disk_usage('/')
                    upt_sec = time.time() - psutil.boot_time()
                    load = os.getloadavg() if hasattr(os, 'getloadavg') else (0.0, 0.0, 0.0)
                    proc = psutil.Process(os.getpid())
                    self.cached_sys_stats = {
                        'mem_used': f"{vm.used // (1024**2)} MB",
                        'mem_total': f"{vm.total // (1024**2)} MB",
                        'machine_uptime': self._fmt_duration(upt_sec),
                        'load_avg': ', '.join(f"{x:.2f}" for x in load),
                        'disk_root': f"{du.percent}%",
                        'threads': threading.active_count(),
                        'proc_count': len(psutil.pids()),
                        'python_mem_mb': f"{proc.memory_info().rss // (1024**2)} MB",
                    }
                except Exception:
                    pass
                self.last_sys_stats_time = now

            try:
                self.cpu_per_core_history.append(psutil.cpu_percent(percpu=True))
                cores = [sum(c) / len(self.cpu_per_core_history) for c in zip(*self.cpu_per_core_history)]
                avg_cpu = sum(cores) / len(cores)
            except Exception:
                cores, avg_cpu = [], 0.0

            monitor_data.update({
                **self.cached_sys_stats,
                'cpu_percent': avg_cpu,
                'cpu_per_core': [f"{p:.1f}%" for p in cores],
                'script_uptime': self._fmt_duration(now - self.script_start_time),
            })

            if self.main_counts:
                cov = sum(1 for c in self.main_counts if c)
                tot = len(self.main_counts)
                monitor_data.update({
                    'main_covered': f"{cov}/{tot}",
                    'main_entropy': f"{_entropy(self.main_counts):.2f}",
                    'main_samples': sum(self.main_counts),
                })
            if self.float_counts:
                cov = sum(1 for c in self.float_counts if c)
                tot = len(self.float_counts)
                monitor_data.update({
                    'float_covered': f"{cov}/{tot}",
                    'float_entropy': f"{_entropy(self.float_counts):.2f}",
                    'float_samples': sum(self.float_counts),
                })

            self.last_heavy_update = now

        return monitor_data