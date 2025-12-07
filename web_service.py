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

MAX_VIEWERS = getattr(settings, 'MAX_VIEWERS', 20)
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
            self.end_headers()
            self.wfile.write(json.dumps(monitor_data).encode('utf-8'))

        elif self.path == "/log":
            try:
                with open("runtime.log", "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'text/plain')
                self.end_headers()
                self.wfile.write(content)
            except:
                self.send_error(404)

        # --- NEW: Serve Static Files (Added this block) ---
        elif self.path.startswith("/static/"):
            try:
                # Security: Prevent directory traversal
                clean_path = os.path.normpath(self.path.lstrip('/')).replace('\\', '/')
                if not clean_path.startswith("static") or ".." in clean_path:
                    self.send_error(403)
                    return

                # Serve file
                with open(clean_path, "rb") as f:
                    content = f.read()

                mime, _ = mimetypes.guess_type(clean_path)
                self.send_response(200)
                self.send_header('Content-Type', mime or 'application/octet-stream')
                self.end_headers()
                self.wfile.write(content)
            except:
                self.send_error(404)
        # --------------------------------------------------

        elif self.path == "/ascii":
            try:
                with open("templates/ascii_viewer.html", "r", encoding="utf-8") as f:
                    content = f.read()

                # Inject Settings
                cols = getattr(settings, 'ASCII_WIDTH', 120)
                rows = getattr(settings, 'ASCII_HEIGHT', 96)
                content = content.replace("{{ASCII_WIDTH}}", str(cols))
                content = content.replace("{{ASCII_HEIGHT}}", str(rows))

                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(content.encode('utf-8'))
            except Exception as e:
                print(f"Error serving ASCII viewer: {e}")
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
            cid = query.get('id', [None])[0] or uuid.uuid4().hex
            self._handle_mjpeg_stream(cid)
        elif path == "/stats":
            with _count_lock:
                data = {"current": _current_viewer_count, "max": MAX_VIEWERS,
                        "full": _current_viewer_count >= MAX_VIEWERS}
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode('utf-8'))
        elif path.startswith("/static/"):
            # --- FIX: Static File Handler ---
            try:
                clean_path = os.path.normpath(path.lstrip('/')).replace('\\', '/')
                if not clean_path.startswith("static") or ".." in clean_path:
                    self.send_error(403)
                    return
                with open(clean_path, "rb") as f:
                    content = f.read()
                mime, _ = mimetypes.guess_type(clean_path)
                self.send_response(200)
                self.send_header('Content-Type', mime or 'application/octet-stream')
                self.end_headers()
                self.wfile.write(content)
            except:
                self.send_error(404)
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
        if not _viewer_semaphore.acquire(blocking=False):
            self.send_error(503, "Server Full")
            return
        with _count_lock:
            _current_viewer_count += 1
        _set_heartbeat(cid)

        try:
            self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        except:
            pass

        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()

        header = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
        newline = b'\r\n'

        try:
            while True:
                frame_data = exchange.get_frame()
                if not frame_data or not isinstance(frame_data, bytes): continue
                _set_heartbeat(cid)
                self.wfile.write(header)
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
    httpd = ThreadedTCPServer(('0.0.0.0', port), MonitorHandler)
    httpd.serve_forever()


def run_stream_server():
    port = getattr(settings, 'STREAM_PORT', 8080)
    httpd = ThreadedTCPServer((getattr(settings, 'STREAM_HOST', '127.0.0.1'), port), StreamHandler)
    httpd.serve_forever()


def start_server(monitor=True, stream=True):
    if monitor: threading.Thread(target=run_monitor_server, daemon=True, name="MonitorServer").start()
    if stream: threading.Thread(target=run_stream_server, daemon=True, name="StreamServer").start()