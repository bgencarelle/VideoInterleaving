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
import socket  # <--- NEW: Needed for TCP Keepalive options

import settings
from shared_state import exchange
from lightweight_monitor import monitor_data, HTML_TEMPLATE

# --- CONFIGURATION ---
# Default to 20 viewers if not in settings.py
MAX_VIEWERS = getattr(settings, 'MAX_VIEWERS', 20)

# --- SAFETY STATE ---
_hb_lock = threading.Lock()
_client_heartbeats = {}

# The "Bouncer": Limits concurrent connections to MAX_VIEWERS
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
    """
    Handles the Monitoring Interface on WEB_PORT (e.g. 1978)
    """

    def log_message(self, format, *args):
        pass  # Silence console logs

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
                self.send_error(404, "Log not found")
        else:
            self.send_error(404)


class StreamHandler(http.server.BaseHTTPRequestHandler):
    """
    Handles the Video Stream on STREAM_PORT (e.g. 8080)
    """

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        # 1. Video Stream
        if path == "/video_feed":
            cid = query.get('id', [None])[0]
            if not cid:
                cid = uuid.uuid4().hex
            self._handle_mjpeg_stream(cid)

        # 2. Heartbeat
        elif path == "/stream_alive":
            cid = query.get('id', [''])[0]
            with _hb_lock:
                ts = _client_heartbeats.get(cid, 0.0)
                now = time.monotonic()
            resp = json.dumps({"ok": bool(ts), "ts": ts, "now": now})

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(resp.encode('utf-8'))

        # 3. Serve CSS (Static Files)
        elif path.startswith("/static/"):
            try:
                clean_path = path.lstrip('/')
                clean_path = os.path.normpath(clean_path)

                if not clean_path.startswith("static") or ".." in clean_path:
                    self.send_error(403, "Forbidden")
                    return

                with open(clean_path, "rb") as f:
                    content = f.read()

                mime, _ = mimetypes.guess_type(clean_path)
                self.send_response(200)
                self.send_header('Content-Type', mime or 'application/octet-stream')
                self.end_headers()
                self.wfile.write(content)
            except FileNotFoundError:
                # print(f"[HTTP] Static file not found: {clean_path}")
                self.send_error(404, "File not found")
            except Exception as e:
                # print(f"[HTTP] Error serving static: {e}")
                self.send_error(500)

        # 4. Serve Index HTML
        elif path == "/":
            try:
                with open("templates/index.html", "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(content)
            except Exception as e:
                fallback = b"<html><body style='background:black;'><img src='/video_feed'></body></html>"
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                self.wfile.write(fallback)

        else:
            self.send_error(404)

    def _handle_mjpeg_stream(self, cid):
        print(f"[STREAM] Client connecting: {cid}")

        # --- SAFETY 1: THE BOUNCER (Semaphore) ---
        # Check if we have room for another viewer
        if not _viewer_semaphore.acquire(blocking=False):
            print(f"[STREAM] Rejected {cid}: Server Full ({MAX_VIEWERS} limit)")
            self.send_error(503, "Server Full")
            return

        # If we passed the bouncer, we proceed
        _set_heartbeat(cid)

        # --- SAFETY 2: TCP KEEPALIVE (Kick Zombies) ---
        # Forces OS to kill connection if client vanishes silently
        try:
            self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            # Linux/VPS specific options:
            # Idle for 1 sec, then check every 1 sec, fail after 5 tries
            if hasattr(socket, 'TCP_KEEPIDLE'):
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 1)
            if hasattr(socket, 'TCP_KEEPINTVL'):
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 1)
            if hasattr(socket, 'TCP_KEEPCNT'):
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5)
        except Exception:
            # Windows might not support all options, which is fine
            pass

        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Pragma', 'no-cache')
        self.end_headers()

        # Pre-calc headers for performance
        header_bytes = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
        newline = b'\r\n'

        try:
            while True:
                frame_data = exchange.get_frame()
                if not frame_data:
                    continue

                _set_heartbeat(cid)

                self.wfile.write(header_bytes)
                self.wfile.write(frame_data)
                self.wfile.write(newline)

        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            print(f"[STREAM] Error {cid}: {e}")
        finally:
            # --- SAFETY 1: RETURN THE TOKEN ---
            _viewer_semaphore.release()

            _clear_heartbeat(cid)
            print(f"[STREAM] Client disconnected: {cid}")


def run_monitor_server():
    host = '127.0.0.1'
    port = getattr(settings, 'WEB_PORT', 1978)
    print(f"🛰️  Monitor active on: http://{host}:{port}")
    httpd = ThreadedTCPServer((host, port), MonitorHandler)
    httpd.serve_forever()


def run_stream_server():
    host = getattr(settings, 'STREAM_HOST', '0.0.0.0')
    port = getattr(settings, 'STREAM_PORT', 8080)
    print(f"🛰️  Stream active on:  http://{host}:{port}")
    httpd = ThreadedTCPServer((host, port), StreamHandler)
    httpd.serve_forever()


def start_server():
    t_mon = threading.Thread(target=run_monitor_server, daemon=True, name="MonitorServer")
    t_mon.start()

    t_str = threading.Thread(target=run_stream_server, daemon=True, name="StreamServer")
    t_str.start()