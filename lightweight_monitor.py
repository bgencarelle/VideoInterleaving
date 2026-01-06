import time
import os
import psutil
import socket
import platform
import threading
from collections import deque
from math import log2

# Try to import ASCII dimensions (may not be available in all modes)
try:
    from shared_state import get_ascii_dimensions
    _ascii_dims_available = True
except ImportError:
    _ascii_dims_available = False

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
    # ASCII dimensions (if in ASCII mode)
    "ascii_width": 0,
    "ascii_height": 0,
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
    .control-panel {
      margin-top: 2em;
      padding: 1em;
      border: 1px solid #333;
      background: #0a0a0a;
    }
    .control-panel h2 {
      margin-top: 0;
      color: #0f0;
      border-bottom: 1px solid #333;
      padding-bottom: 0.5em;
    }
    .dim-controls {
      display: flex;
      gap: 1em;
      align-items: center;
      margin: 1em 0;
    }
    .dim-controls input {
      background: #111;
      border: 1px solid #333;
      color: #0f0;
      padding: 0.5em;
      font-family: monospace;
      width: 80px;
    }
    .dim-controls button {
      background: #0f0;
      color: #000;
      border: none;
      padding: 0.5em 1em;
      font-family: monospace;
      cursor: pointer;
      font-weight: bold;
    }
    .dim-controls button:hover {
      background: #0a0;
    }
    .presets {
      display: flex;
      gap: 0.5em;
      flex-wrap: wrap;
      margin: 1em 0;
    }
    .preset-btn {
      background: #222;
      color: #0f0;
      border: 1px solid #333;
      padding: 0.5em 1em;
      font-family: monospace;
      cursor: pointer;
    }
    .preset-btn:hover {
      background: #333;
      border-color: #0f0;
    }
    .status-message {
      margin-top: 1em;
      padding: 0.5em;
      font-size: 0.9em;
    }
    .status-message.success {
      color: #0f0;
      background: #003300;
    }
    .status-message.error {
      color: #f00;
      background: #330000;
    }
    .current-dims {
      color: #0f0;
      font-weight: bold;
      margin: 0.5em 0;
    }
  </style>
</head>
<body>
  <h1>Interleaving Project Live Playback Monitor</h1>
  <div><a href="log">click for error log</a></div>
  <div id="monitor_data"></div>

  <div class="control-panel" id="ascii-controls" style="display:none;">
    <h2>ASCII Terminal Dimensions</h2>
    <div class="current-dims" id="current-dims">Loading...</div>
    <div class="dim-controls">
      <label>Width: <input type="number" id="width-input" min="1" max="500" value="90"></label>
      <label>Height: <input type="number" id="height-input" min="1" max="500" value="60"></label>
      <button onclick="applyDimensions()">Apply</button>
    </div>
    <div class="presets">
      <button class="preset-btn" onclick="setPreset(30, 20)">30×20</button>
      <button class="preset-btn" onclick="setPreset(90, 60)">90×60</button>
      <button class="preset-btn" onclick="setPreset(300, 200)">300×200</button>
      <button class="preset-btn" onclick="setPreset(500, 300)">500×300</button>
    </div>
    <div class="status-message" id="status-message" style="display:none;"></div>
  </div>

  <script>
    let dimensionPollInterval = null;

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

    // ASCII Dimension Control Functions
    async function getDimensions() {
      try {
        const r = await fetch('/ascii/size', { cache: 'no-store' });
        if (r.ok) {
          return await r.json();
        }
        return null;
      } catch (e) {
        return null;
      }
    }

    async function setDimensions(width, height) {
      try {
        const r = await fetch('/ascii/size', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({width, height})
        });
        if (r.ok) {
          return await r.json();
        } else {
          const error = await r.json().catch(() => ({error: 'Unknown error'}));
          throw new Error(error.error || 'Failed to set dimensions');
        }
      } catch (e) {
        throw e;
      }
    }

    function updateDimensionDisplay(dims) {
      if (!dims) return;
      const currentEl = document.getElementById('current-dims');
      const widthInput = document.getElementById('width-input');
      const heightInput = document.getElementById('height-input');
      
      if (currentEl) {
        currentEl.textContent = `Current: ${dims.width} cols × ${dims.height} rows`;
      }
      if (widthInput) widthInput.value = dims.width;
      if (heightInput) heightInput.value = dims.height;
    }

    function showStatus(message, isError = false) {
      const statusEl = document.getElementById('status-message');
      if (statusEl) {
        statusEl.textContent = message;
        statusEl.className = 'status-message ' + (isError ? 'error' : 'success');
        statusEl.style.display = 'block';
        setTimeout(() => {
          statusEl.style.display = 'none';
        }, 3000);
      }
    }

    async function applyDimensions() {
      const widthInput = document.getElementById('width-input');
      const heightInput = document.getElementById('height-input');
      
      if (!widthInput || !heightInput) return;
      
      const width = parseInt(widthInput.value);
      const height = parseInt(heightInput.value);
      
      if (isNaN(width) || isNaN(height) || width < 1 || height < 1) {
        showStatus('Invalid dimensions. Must be positive numbers.', true);
        return;
      }
      
      try {
        const result = await setDimensions(width, height);
        if (result) {
          updateDimensionDisplay(result);
          showStatus(`✓ Dimensions updated to ${result.width}×${result.height}`);
        }
      } catch (e) {
        showStatus(`Error: ${e.message}`, true);
      }
    }

    function setPreset(width, height) {
      const widthInput = document.getElementById('width-input');
      const heightInput = document.getElementById('height-input');
      
      if (widthInput) widthInput.value = width;
      if (heightInput) heightInput.value = height;
      applyDimensions();
    }

    async function pollDimensions() {
      const dims = await getDimensions();
      if (dims) {
        // Show controls if dimensions endpoint is available (ASCII mode active)
        const controlsEl = document.getElementById('ascii-controls');
        if (controlsEl) {
          controlsEl.style.display = 'block';
        }
        updateDimensionDisplay(dims);
      } else {
        // Hide controls if not in ASCII mode
        const controlsEl = document.getElementById('ascii-controls');
        if (controlsEl) {
          controlsEl.style.display = 'none';
        }
      }
    }

    document.addEventListener('DOMContentLoaded', () => {
      poll();
      setInterval(poll, 100); // Poll monitor data
      
      // Poll dimensions and show/hide controls
      pollDimensions();
      dimensionPollInterval = setInterval(pollDimensions, 2000); // Poll dimensions every 2 seconds
    });
  </script>
</body>
</html>
"""

def _entropy(counts):
    """
    Calculate normalized Shannon entropy for folder selection distribution.
    
    Normalizes by the maximum achievable entropy for ALL available folders,
    not just the folders that have been used. This measures randomness across
    the full set of available folders and shows progress toward using all folders.
    
    Returns a value between 0.0 (completely ordered or few folders used) and 1.0
    (maximally random across all available folders).
    
    - Early in run: Low entropy (few folders used)
    - Mid run: Increasing entropy (more folders used)
    - Late run: High entropy (all folders used randomly)
    """
    tot = sum(counts)
    if tot == 0:
        return 0.0
    
    n = len(counts)  # Total folders available
    if n <= 1:
        return 0.0
    
    # Calculate raw Shannon entropy (only sum over non-zero bins)
    h = -sum((c / tot) * log2(c / tot) for c in counts if c > 0)
    
    # Normalize by maximum entropy for ALL folders
    # This measures randomness across the full set of available folders
    max_h = log2(n)
    
    return h / max_h if max_h > 0 else 0.0

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

            # Update ASCII dimensions if available
            if _ascii_dims_available:
                try:
                    dims = get_ascii_dimensions()
                    monitor_data.update({
                        'ascii_width': dims.get_width(),
                        'ascii_height': dims.get_height(),
                    })
                except Exception:
                    # ASCII dimensions not available or error
                    monitor_data.update({
                        'ascii_width': 0,
                        'ascii_height': 0,
                    })

            self.last_heavy_update = now

        return monitor_data