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
import sys  # Added for logging output
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


def get_real_ip(handler):
    """Extracts the real IP from Nginx headers or falls back to socket address."""
    # 1. Try X-Real-IP (Standard Nginx)
    real_ip = handler.headers.get('X-Real-IP')

    # 2. Try X-Forwarded-For (Standard Proxy Chain)
    if not real_ip:
        forwarded = handler.headers.get('X-Forwarded-For')
        if forwarded:
            # Get the first IP in the list
            real_ip = forwarded.split(',')[0].strip()

    # 3. Fallback to direct connection
    if not real_ip:
        real_ip = handler.client_address[0]

    return real_ip


def _serve_static_file(handler):
    """Helper to serve static files safely."""
    try:
        # 1. Parse URL to strip query strings (Fixes the ?v=1 bug)
        parsed_path = urlparse(handler.path).path

        # 2. Clean and Normalize
        # Remove leading slash so it becomes a relative path (e.g., "static/style.css")
        rel_path = parsed_path.lstrip('/')
        clean_path = os.path.normpath(rel_path).replace('\\', '/')

        # 3. Security Checks
        # Must start with 'static' and not try to go up directories
        if not clean_path.startswith("static") or ".." in clean_path:
            handler.send_error(403)
            return

        # 4. Existence Check
        if not os.path.isfile(clean_path):
            handler.send_error(404)
            return

        # 5. Serve
        with open(clean_path, "rb") as f:
            content = f.read()

        mime, _ = mimetypes.guess_type(clean_path)
        handler.send_response(200)
        # Force encoding for text files (CSS/JS) to prevent browser warnings
        if mime and (mime.startswith("text/") or mime == "application/javascript"):
            handler.send_header('Content-Type', f'{mime}; charset=utf-8')
        else:
            handler.send_header('Content-Type', mime or 'application/octet-stream')

        handler.send_header('Content-Length', len(content))
        handler.end_headers()
        handler.wfile.write(content)

    except Exception as e:
        print(f"Static serve error: {e}")
        handler.send_error(404)


class ThreadedTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class MonitorHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Nginx-aware logging
        ip = get_real_ip(self)
        sys.stderr.write("%s - - [%s] %s\n" %
                         (ip, self.log_date_time_string(), format % args))

    def do_GET(self):
        # Monitor is often on port 1978/1980
        # Check static first to handle /static/style.css requests
        if self.path.startswith("/static/"):
            _serve_static_file(self)
            return

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
                # Safely read only the tail of the log
                log_size = os.path.getsize("runtime.log")
                read_size = 64 * 1024  # 64KB

                with open("runtime.log", "rb") as f:
                    if log_size > read_size:
                        f.seek(-read_size, os.SEEK_END)
                    content = f.read()

                self.send_response(200)
                self.send_header('Content-Type', 'text/plain; charset=utf-8')
                self.end_headers()
                self.wfile.write(content)
            except Exception:
                self.send_error(404, "Log file not found or empty")

        elif self.path == "/ascii":
            try:
                with open("templates/ascii_viewer.html", "r", encoding="utf-8") as f:
                    content = f.read()

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
    timeout = 5

    def log_message(self, format, *args):
        # Nginx-aware logging
        ip = get_real_ip(self)
        sys.stderr.write("%s - - [%s] %s\n" %
                         (ip, self.log_date_time_string(), format % args))

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
            _serve_static_file(self)

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

        # Send Headers
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.end_headers()

        header = b'--frame\r\n'  # Content-Type added dynamically
        newline = b'\r\n'

        try:
            while True:
                # BLOCKING WAIT
                raw_payload = exchange.get_frame()

                if not raw_payload or not isinstance(raw_payload, bytes):
                    break

                # Format Detection (First Byte)
                fmt_byte = raw_payload[0:1]
                frame_data = raw_payload[1:]

                if fmt_byte == b'w':
                    ctype = b'Content-Type: image/webp\r\n\r\n'
                else:
                    ctype = b'Content-Type: image/jpeg\r\n\r\n'

                _set_heartbeat(cid)

                try:
                    self.wfile.write(header)
                    self.wfile.write(ctype)
                    self.wfile.write(frame_data)
                    self.wfile.write(newline)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break
        except Exception:
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