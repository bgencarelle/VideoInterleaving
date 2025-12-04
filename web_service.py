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
import settings
from shared_state import exchange
from lightweight_monitor import monitor_data, HTML_TEMPLATE

# --- HEARTBEAT STATE (For Stream) ---
_client_heartbeats = {}
_hb_lock = threading.Lock()


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
        # Root -> HTML
        if self.path == "/":
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode('utf-8'))

        # Data -> JSON
        elif self.path == "/data":
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(json.dumps(monitor_data).encode('utf-8'))

        # Log -> Text
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
                # Remove leading slash and normalize path (handles windows/linux slashes)
                # url path: /static/stream.css -> file system: static/stream.css
                clean_path = path.lstrip('/')
                clean_path = os.path.normpath(clean_path)

                # Security check: Ensure we are still inside the 'static' folder
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
                print(f"[HTTP] Static file not found: {clean_path}")
                self.send_error(404, "File not found")
            except Exception as e:
                print(f"[HTTP] Error serving static: {e}")
                self.send_error(500)

        # 4. Serve Index HTML
        elif path == "/":
            try:
                # We simply read and serve the file.
                # Since we updated index.html to use standard paths,
                # we don't need any complex string replacement here.
                with open("templates/index.html", "rb") as f:
                    content = f.read()

                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(content)
            except Exception as e:
                print(f"[HTTP] Error serving template: {e}")
                # Fallback if template is missing
                fallback = b"<html><body style='background:black;'><img src='/video_feed'></body></html>"
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                self.wfile.write(fallback)

        else:
            self.send_error(404)

    def _handle_mjpeg_stream(self, cid):
        # ... (Keep your existing _handle_mjpeg_stream logic exactly as is) ...
        # (I am omitting it here to save space, but DO NOT delete it from your file)
        print(f"[STREAM] Client connected: {cid}")
        _set_heartbeat(cid)

        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Pragma', 'no-cache')
        self.end_headers()

        try:
            while True:
                frame_data = exchange.get_frame()
                if not frame_data:
                    continue

                _set_heartbeat(cid)

                header = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                self.wfile.write(header)
                self.wfile.write(frame_data)
                self.wfile.write(b'\r\n')
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            _clear_heartbeat(cid)
            print(f"[STREAM] Client disconnected: {cid}")

def run_monitor_server():
    host = '127.0.0.1'  # Monitor is usually local/tunneled
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
    # We start TWO distinct servers on TWO ports
    t_mon = threading.Thread(target=run_monitor_server, daemon=True, name="MonitorServer")
    t_mon.start()

    t_str = threading.Thread(target=run_stream_server, daemon=True, name="StreamServer")
    t_str.start()

    # We don't join threads because the main thread continues to run image_display logic