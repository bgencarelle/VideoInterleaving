import json
import time
import os
import threading
import psutil
import socket
import platform
from collections import deque
from math import log2
import settings


def _entropy(counts):
    tot = sum(counts)
    if tot == 0: return 0.0
    h = -sum((c / tot) * log2(c / tot) for c in counts if c)
    return h / log2(len(counts))


class DetailedFileMonitor:
    def __init__(self):
        # 1. Identity
        if getattr(settings, 'ASCII_MODE', False):
            self.mode = "ascii"
        elif getattr(settings, 'SERVER_MODE', False):
            self.mode = "web"
        else:
            self.mode = "local"

        self.filename = f"stats_{self.mode}.json"
        self.temp_filename = f"{self.filename}.tmp"

        # 2. State & History (Restored from your original code)
        self.last_success_time = time.monotonic()
        self.script_start_time = time.monotonic()
        self.comp_history = deque(maxlen=30)
        self.failed_indices = deque(maxlen=20)
        self.cpu_per_core_history = deque(maxlen=5)
        self.main_counts = None
        self.float_counts = None

        # 3. Rate Limiting
        self.last_write = 0
        self.write_interval = 0.5  # Write twice a second for snappiness
        self.last_heavy_update = 0
        self.heavy_interval = 2.0  # Sys stats every 2s

        # 4. Base Data Structure
        self.data = {
            "mode": self.mode,
            "status": "booting",
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "cpu_percent": 0,
            "mem_used": "0 MB",
            "disk_root": "0%",
            # Default values to prevent UI undefined errors
            "index": 0, "fps": 0, "fifo_depth": 0, "main_entropy": 0
        }

    def record_load_error(self, index, err):
        self.data['last_error'] = str(err)
        self.data['failed_load_count'] = self.data.get('failed_load_count', 0) + 1

    def update(self, payload):
        now = time.monotonic()

        # --- A. Update Counters (Entropy etc) ---
        mf = payload.get('main_folder')
        mfc = payload.get('main_folder_count')

        # Initialize counts if needed
        if mfc and self.main_counts is None: self.main_counts = [0] * mfc
        if self.main_counts and mf is not None and 0 <= mf < len(self.main_counts):
            self.main_counts[mf] += 1

        # Update basic payload
        self.data.update(payload)
        self.data['uptime'] = self._fmt_duration(now - self.script_start_time)

        # --- B. Heavy System Stats (CPU/RAM) ---
        if now - self.last_heavy_update > self.heavy_interval:
            self._update_sys_stats()
            self.last_heavy_update = now

        # --- C. Write to Disk ---
        if now - self.last_write > self.write_interval:
            self._flush()
            self.last_write = now

    def _update_sys_stats(self):
        try:
            # CPU
            self.cpu_per_core_history.append(psutil.cpu_percent(percpu=True))
            cores = [sum(c) / len(self.cpu_per_core_history) for c in zip(*self.cpu_per_core_history)]
            self.data['cpu_percent'] = round(sum(cores) / len(cores), 1)
            self.data['cpu_per_core'] = [round(x, 1) for x in cores]

            # RAM
            vm = psutil.virtual_memory()
            self.data['mem_used'] = f"{vm.used // (1024 ** 2)} MB"
            self.data['mem_total'] = f"{vm.total // (1024 ** 2)} MB"

            # Entropy Calculation
            if self.main_counts:
                self.data['main_entropy'] = f"{_entropy(self.main_counts):.2f}"

        except Exception:
            pass

    def _flush(self):
        # We add a timestamp so the UI knows if the data is stale
        self.data['timestamp'] = time.time()
        threading.Thread(target=self._write_file, daemon=True).start()

    def _write_file(self):
        try:
            with open(self.temp_filename, 'w') as f:
                json.dump(self.data, f)
            os.replace(self.temp_filename, self.filename)
        except:
            pass

    @staticmethod
    def _fmt_duration(secs):
        m, s = divmod(int(secs), 60)
        h, m = divmod(m, 60)
        return f"{h:02}:{m:02}:{s:02}"


def start_monitor():
    return DetailedFileMonitor()