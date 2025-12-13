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
import sys
import settings
from shared_state import exchange
from lightweight_monitor import monitor_data, HTML_TEMPLATE

MAX_VIEWERS = getattr(settings, 'MAX_VIEWERS', 20)
_hb_lock = threading.Lock()
_client_heartbeats = {}
_current_viewer_count = 0
_count_lock = threading.Lock()
_viewer_semaphore = threading.Semaphore(MAX_VIEWERS)

# Pre-calculated headers for efficiency
HEADER_BOUNDARY = b'--frame\r\n'
HEADER_NEWLINE = b'\r\n'
HEADER_CTYPE_JPEG = b'Content-Type: image/jpeg\r\n\r\n'
HEADER_CTYPE_WEBP = b'Content-Type: image/webp\r\n\r\n'


def _set_heartbeat(cid):
    with _hb_lock:
        _client_heartbeats[cid] = time.monotonic()


def _clear_heartbeat(cid):
    with _hb_lock:
        if cid in _client_heartbeats:
            del _client_heartbeats[cid]


def get_real_ip(handler):
    """Extracts the real IP from Nginx headers or falls back to socket address."""
    real_ip = handler.headers.get('X-Real-IP')
    if not real_ip:
        forwarded = handler.headers.get('X-Forwarded-For')
        if forwarded:
            real_ip = forwarded.split(',')[0].strip()
    if not real_ip:
        real_ip = handler.client_address[0]
    return real_ip


def _serve_static_file(handler):
    """Helper to serve static files safely and handle query strings."""
    try:
        parsed_path = urlparse(handler.path).path
        rel_path = parsed_path.lstrip('/')
        clean_path = os.path.normpath(rel_path).replace('\\', '/')

        if not clean_path.startswith("static") or ".." in clean_path:
            handler.send_error(403)
            return

        if not os.path.isfile(clean_path):
            handler.send_error(404)
            return

        with open(clean_path, "rb") as f:
            content = f.read()

        mime, _ = mimetypes.guess_type(clean_path)
        handler.send_response(200)

        if mime and (mime.startswith("text/") or mime == "application/javascript"):
            handler.send_header('Content-Type', f'{mime}; charset=utf-8')
        else:
            handler.send_header('Content-Type', mime or 'application/octet-stream')

        handler.send_header('Content-Length', len(content))
        handler.end_headers()
        handler.wfile.write(content)

    except Exception:
        handler.send_error(404)


class ThreadedTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class OptimizedHandlerMixin:
    """Mixin to handle logging suppression and DNS optimization."""

    def address_string(self):
        # Disable reverse DNS lookups (major speedup)
        return str(self.client_address[0])

    def log_message(self, format, *args):
        # 1. Quiet Mode: Filter out noise
        # Don't log poll requests (/stats, /data) or static files
        if self.path in ['/stats', '/data'] or self.path.startswith('/static/'):
            return

        # 2. Nginx-aware logging for everything else
        ip = get_real_ip(self)
        sys.stderr.write("%s - - [%s] %s\n" %
                         (ip, self.log_date_time_string(), format % args))


class MonitorHandler(OptimizedHandlerMixin, http.server.BaseHTTPRequestHandler):
    def do_GET(self):
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
                # USE THE PATH FROM SETTINGS
                log_path = getattr(settings, 'LOG_FILE_PATH', 'runtime.log')
                log_size = os.path.getsize(log_path)
                read_size = 64 * 1024

                with open(log_path, "rb") as f:
                    if log_size > read_size:
                        f.seek(-read_size, os.SEEK_END)
                    content = f.read()

                self.send_response(200)
                self.send_header('Content-Type', 'text/plain; charset=utf-8')
                self.end_headers()
                self.wfile.write(content)
            except Exception:
                self.send_error(404, "Log file missing")

        elif self.path == "/ascii":
            try:
                with open("templates/ascii_viewer.html", "r", encoding="utf-8") as f:
                    content = f.read()

                cols = getattr(settings, 'ASCII_WIDTH', 120)
                rows = getattr(settings, 'ASCII_HEIGHT', 96)
                # [NEW] Pass the ratio so frontend can calculate spacing
                ratio = getattr(settings, 'ASCII_FONT_RATIO', 0.5)

                content = content.replace("{{ASCII_WIDTH}}", str(cols))
                content = content.replace("{{ASCII_HEIGHT}}", str(rows))
                content = content.replace("{{ASCII_FONT_RATIO}}", str(ratio))

                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(content.encode('utf-8'))
            except Exception:
                self.send_error(404)
        else:
            self.send_error(404)

class StreamHandler(OptimizedHandlerMixin, http.server.BaseHTTPRequestHandler):
    # Optimization: Increase timeout to prevent disconnects on lag
    timeout = 30

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
            # 1. Keepalive (Detect broken pipes)
            self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            # 2. No Delay (Disable Nagle's Algorithm) - CRITICAL for streaming latency
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except:
            pass

        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.end_headers()

        try:
            while True:
                # 1. BLOCKING WAIT
                raw_payload = exchange.get_frame()

                if not raw_payload or not isinstance(raw_payload, bytes):
                    break

                if len(raw_payload) < 1: continue

                fmt_byte = raw_payload[0:1]
                frame_data = raw_payload[1:]

                # 2. Use Pre-calculated Headers
                if fmt_byte == b'w':
                    ctype = HEADER_CTYPE_WEBP
                else:
                    ctype = HEADER_CTYPE_JPEG

                _set_heartbeat(cid)

                # 3. Send Frame
                try:
                    self.wfile.write(HEADER_BOUNDARY)
                    self.wfile.write(ctype)
                    self.wfile.write(frame_data)
                    self.wfile.write(HEADER_NEWLINE)
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
    httpd = ThreadedTCPServer(('127.0.0.1', port), MonitorHandler)
    httpd.serve_forever()


def run_stream_server():
    port = getattr(settings, 'STREAM_PORT', 8080)
    httpd = ThreadedTCPServer((getattr(settings, 'STREAM_HOST', '127.0.0.1'), port), StreamHandler)
    httpd.serve_forever()


def start_server(monitor=True, stream=True):
    if monitor: threading.Thread(target=run_monitor_server, daemon=True, name="MonitorServer").start()
    if stream: threading.Thread(target=run_stream_server, daemon=True, name="StreamServer").start()