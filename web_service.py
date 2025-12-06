#!/usr/bin/env python3
import time
import threading
import json
import uuid
import socketserver
import http.server
from urllib.parse import urlparse, parse_qs
import mimetypes
import os
import socket
import settings
from shared_state import exchange
from lightweight_monitor import monitor_data, HTML_TEMPLATE

# --- CONFIGURATION ---
MAX_VIEWERS = getattr(settings, 'MAX_VIEWERS', 20)

# --- STATE ---
_hb_lock = threading.Lock()
_client_heartbeats = {}
_current_viewer_count = 0
_count_lock = threading.Lock()
_viewer_semaphore = threading.Semaphore(MAX_VIEWERS)


def _set_heartbeat(cid):
    with _hb_lock:
        _client_heartbeats[cid] = time.monotonic()


def _clear_heartbeat(cid):
    with _hb_lock:
        if cid in _client_heartbeats:
            del _client_heartbeats[cid]


# --- SERVER CLASSES ---

class ThreadedTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class MonitorHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode('utf-8'))
        elif self.path == "/data":
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(json.dumps(monitor_data).encode('utf-8'))
        elif self.path == "/log":
            try:
                with open("runtime.log", "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'text/plain; charset=utf-8')
                self.end_headers()
                self.wfile.write(content)
            except Exception:
                self.send_error(404)
        else:
            self.send_error(404)


class StreamHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/video_feed":
            cid = query.get('id', [None])[0]
            if not cid: cid = uuid.uuid4().hex
            self._handle_mjpeg_stream(cid)
        elif path == "/stats":
            with _count_lock:
                data = {
                    "current": _current_viewer_count,
                    "max": MAX_VIEWERS,
                    "full": _current_viewer_count >= MAX_VIEWERS
                }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode('utf-8'))
        elif path == "/":
            try:
                with open("templates/index.html", "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(content)
            except:
                self.send_error(404)
        else:
            self.send_error(404)

    def _handle_mjpeg_stream(self, cid):
        global _current_viewer_count
        # ... (Identical to previous logic) ...
        # Copied for brevity, ensure previous logic for semaphore/socket opts matches

        if not _viewer_semaphore.acquire(blocking=False):
            self.send_error(503, "Server Full")
            return

        with _count_lock:
            _current_viewer_count += 1

        _set_heartbeat(cid)

        try:
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()

            header_bytes = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
            newline = b'\r\n'

            while True:
                frame_data = exchange.get_frame()
                if not frame_data: continue
                # In ASCII mode, frame_data is a string, which would crash this handler.
                # Since we selectively disable this handler in main.py, we are safe.
                _set_heartbeat(cid)
                self.wfile.write(header_bytes)
                self.wfile.write(frame_data)
                self.wfile.write(newline)
        except:
            pass
        finally:
            _viewer_semaphore.release()
            with _count_lock:
                _current_viewer_count -= 1
            _clear_heartbeat(cid)


def run_monitor_server():
    port = getattr(settings, 'WEB_PORT', 1978)
    print(f"📊 Monitor UI started at http://127.0.0.1:{port}")
    httpd = ThreadedTCPServer(('0.0.0.0', port), MonitorHandler)
    httpd.serve_forever()


def run_stream_server():
    port = getattr(settings, 'STREAM_PORT', 8080)
    print(f"🛰️  MJPEG Stream started at http://127.0.0.1:{port}")
    httpd = ThreadedTCPServer((getattr(settings, 'STREAM_HOST', '127.0.0.1'), port), StreamHandler)
    httpd.serve_forever()


def start_server(monitor=True, stream=True):
    """
    Selectively start the Monitor and/or the MJPEG Streamer.
    """
    if monitor:
        threading.Thread(target=run_monitor_server, daemon=True, name="MonitorServer").start()

    if stream:
        threading.Thread(target=run_stream_server, daemon=True, name="StreamServer").start()