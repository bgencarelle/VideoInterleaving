#!/usr/bin/env python3
import time
import threading
import json
import uuid
import socketserver
import http.server
from urllib.parse import urlparse, parse_qs, unquote
import mimetypes
import os
import socket
import sys
import settings
from shared_state import exchange_web
from lightweight_monitor import monitor_data, HTML_TEMPLATE
from server_config import get_config

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
    """
    Extracts the real IP from Nginx headers (X-Real-IP / X-Forwarded-For)
    or falls back to the socket address if accessed directly.
    """
    real_ip = handler.headers.get('X-Real-IP')
    if not real_ip:
        forwarded = handler.headers.get('X-Forwarded-For')
        if forwarded:
            real_ip = forwarded.split(',')[0].strip()
    if not real_ip:
        real_ip = handler.client_address[0]
    return real_ip


def _serve_static_file(handler):
    """
    Serve static files safely (prevents traversal, supports query strings).
    Updated to decode URL entities before path validation.
    """
    try:
        raw_path = urlparse(handler.path).path
        decoded_path = unquote(raw_path)  # Decode %2e%2e -> ..
        rel_path = decoded_path.lstrip('/')

        # Normalize path separators to forward slashes for consistency
        clean_path = os.path.normpath(rel_path).replace('\\', '/')

        # Security: Enforce static/ prefix after normalization
        if not clean_path.startswith("static/"):
            handler.send_error(403)
            return

        # Extra paranoia: reject any remaining traversal tokens
        if "/../" in f"/{clean_path}/" or clean_path.endswith("/.."):
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

        handler.send_header('Content-Length', str(len(content)))
        handler.end_headers()
        handler.wfile.write(content)

    except Exception:
        handler.send_error(404)


def _serve_dynamic_template(handler):
    """
    Tries to map /pagename to templates/pagename.html
    Returns True if successful, False if file not found.
    """
    try:
        # 1. Decode and clean the path
        raw_path = urlparse(handler.path).path
        decoded_path = unquote(raw_path)
        # Remove leading slash and normalize
        clean_path = os.path.normpath(decoded_path.lstrip('/')).replace('\\', '/')

        # 2. Security: Prevent Directory Traversal
        if "/../" in f"/{clean_path}/" or clean_path.startswith(".."):
            return False

        # 3. Construct target file path (e.g., /about -> templates/about.html)
        # Note: We enforce the .html extension so users can't read source code files
        target_file = os.path.join("templates", f"{clean_path}.html")

        # 4. If file exists, serve it
        if os.path.isfile(target_file):
            with open(target_file, "rb") as f:
                content = f.read()

            handler.send_response(200)
            handler.send_header('Content-Type', 'text/html; charset=utf-8')
            handler.send_header('Content-Length', str(len(content)))
            handler.end_headers()
            handler.wfile.write(content)
            return True

    except Exception as e:
        # Fail silently so the main handler can send 404
        pass

    return False


class ThreadedTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class RobustHandlerMixin:
    """
    Mixin to harden the server against bots and reduce log noise.
    Wraps request handling to catch network disconnects silently.
    """

    def address_string(self):
        # Disable reverse DNS lookups for speed
        return str(self.client_address[0])

    def log_request(self, code='-', size='-'):
        """
        Overrides the default request logger.
        Filters out 400/403/404/408 errors to suppress bot scan logs.
        """
        try:
            c = int(code)
        except Exception:
            c = None

        if c in (400, 403, 404, 408):
            return

        self.log_message('"%s" %s %s', self.requestline, str(code), str(size))

    def log_error(self, format, *args):
        # Suppress error logging for common HTTP error codes
        if args and isinstance(args[0], int) and args[0] in (400, 403, 404, 408):
            return

        ip = get_real_ip(self)
        sys.stderr.write("%s - - [%s] %s\n" %
                         (ip, self.log_date_time_string(), format % args))

    def log_message(self, format, *args):
        # Quiet Mode: Filter out internal stats polling and static files
        path = getattr(self, "path", "")
        if path in ['/stats', '/data'] or path.startswith('/static/'):
            return

        ip = get_real_ip(self)
        sys.stderr.write("%s - - [%s] %s\n" %
                         (ip, self.log_date_time_string(), format % args))

    def handle(self):
        """
        Wraps the standard handle() to catch network disconnects silently.
        """
        try:
            super().handle()
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError, socket.timeout):
            # Normal network weather (client disconnected, timeout, etc.)
            pass
        except Exception as e:
            # Log unexpected server crashes
            print(f"⚠️ [Web Error] {e}", file=sys.stderr)


def _scope_opts(path):
    """(size, exposure) from the query string, clamped.

    Size is the resolution knob AND the cpu knob and there is no way to
    separate them: the scanlines are about 92 rows, so below ~400 px they
    merge and the picture turns into a blob. 512 is the smallest that still
    resolves features. Cost here is 2.4 / 4.4 / 8.0 / 13.7 ms at
    256 / 384 / 512 / 700; a Pi will be several times that.
    """
    size, exposure = 512, 1.0
    try:
        q = parse_qs(urlparse(path).query)
        size = max(128, min(768, int(q.get("size", [size])[0])))
        exposure = max(0.1, min(4.0, float(q.get("exposure", [exposure])[0])))
    except Exception:
        pass
    return size, exposure


def _scope_jpeg(size, pts=None, quality=82, exposure=1.0):
    """Render the parked trace to JPEG bytes, or None if there is nothing."""
    try:
        import cv2
        from scope_out import Scope
        from scope_bake import preview_frame
        if pts is None:
            Scope.want_tap(3.0)
            _, pts = Scope.read_tap()
        if pts is None:
            return None
        img = preview_frame(pts, size=size, exposure=exposure)
        ok, buf = cv2.imencode(".jpg", img[:, :, ::-1],
                               [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None
    except Exception:
        return None


class MonitorHandler(RobustHandlerMixin, http.server.BaseHTTPRequestHandler):
    # Short timeout to prevent Slow Loris attacks on control pages
    timeout = 5

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

        elif self.path.startswith("/scope/stream.mjpg"):
            self._scope_mjpeg()

        elif self.path.startswith("/scope/luma.mjpg"):
            # The LUMINANCE the trace was built from, as MJPEG. The browser
            # renders its own trace from this, at its own AudioContext rate,
            # and plays it on its own sound card -- which is the only place it
            # can reach the visitor's scope from.
            self._scope_luma_stream()

        elif self.path.startswith("/scope/frame.jpg"):
            # Single still, for anything that will not hold a connection open.
            _sz, _ex = _scope_opts(self.path)
            blob = _scope_jpeg(_sz, exposure=_ex)
            if blob is None:
                self.send_response(204); self.end_headers(); return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(blob)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(blob)

        elif self.path == "/log":
            try:
                log_path = getattr(settings, 'LOG_FILE_PATH', 'runtime.log')
                if os.path.exists(log_path):
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
                else:
                    self.send_error(404, "Log file not found")
            except Exception as e:
                print(f"[Web] Error serving log file: {e}", file=sys.stderr)
                self.send_error(500)

        elif self.path == "/ascii":
            try:
                with open("templates/ascii_viewer.html", "r", encoding="utf-8") as f:
                    content = f.read()

                cols = getattr(settings, 'ASCII_WIDTH', 120)
                rows = getattr(settings, 'ASCII_HEIGHT', 96)
                ratio = getattr(settings, 'ASCII_FONT_RATIO', 0.5)
                # Get WebSocket port from config (for asciiweb mode)
                try:
                    ws_port = get_config().get_ascii_websocket_port()
                    if ws_port is None:
                        ws_port = getattr(settings, 'WEBSOCKET_PORT', 2424)
                except RuntimeError:
                    # Config not initialized yet, use settings fallback
                    ws_port = getattr(settings, 'WEBSOCKET_PORT', 2424)

                content = content.replace("{{ASCII_WIDTH}}", str(cols))
                content = content.replace("{{ASCII_HEIGHT}}", str(rows))
                content = content.replace("{{ASCII_FONT_RATIO}}", str(ratio))
                content = content.replace("{{WEBSOCKET_PORT}}", str(ws_port))

                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
                self.send_header('Pragma', 'no-cache')
                self.end_headers()
                self.wfile.write(content.encode('utf-8'))
            except Exception:
                self.send_error(404)

        # [NEW] Dynamic Template Fallback
        elif _serve_dynamic_template(self):
            return

        else:
            self.send_error(404)

    def _scope_luma_stream(self):
        """multipart JPEG of the source luminance, not the rendered trace."""
        try:
            import cv2
            from scope_out import Scope
        except Exception:
            self.send_error(503); return
        try:
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.end_headers()
        except (OSError, AttributeError):
            return
        last, idle = -1, 0.0
        # q88: below about q80 the compression pushes cells across `trim`,
        # which flips them from drawn to undrawn and reroutes the serpentine.
        # Visually harmless but it is a cliff, not a slope, so stay above it.
        enc = [cv2.IMWRITE_JPEG_QUALITY, 88]
        try:
            while True:
                Scope.want_tap(3.0)
                seq, lum = Scope.read_luma()
                if lum is None or seq == last:
                    time.sleep(0.01); idle += 0.01
                    if idle > 30.0:
                        return
                    continue
                idle = 0.0; last = seq
                ok, buf = cv2.imencode(".jpg", lum, enc)
                if not ok:
                    continue
                try:
                    self.wfile.write(HEADER_BOUNDARY)
                    self.wfile.write(HEADER_CTYPE_JPEG)
                    self.wfile.write(buf.tobytes())
                    self.wfile.write(HEADER_NEWLINE)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError,
                        ConnectionAbortedError, OSError):
                    return
                time.sleep(1.0 / max(getattr(settings, "IPS", 30), 1))
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError,
                socket.timeout, OSError):
            return
        except Exception as e:
            print(f"[Scope luma] {e}", file=sys.stderr)

    def _scope_mjpeg(self):
        """multipart/x-mixed-replace, same shape as StreamHandler.

        Rendering runs HERE, on the request thread, not in the audio path.  A
        slow client or a heavy size setting therefore costs frames on the
        preview and nothing at all on the trace deadline.
        """
        size, exposure = _scope_opts(self.path)
        try:
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control",
                             "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.end_headers()
        except (OSError, AttributeError):
            return

        last_seq = -1
        idle = 0.0
        try:
            from scope_out import Scope
        except Exception:
            return
        try:
            while True:
                Scope.want_tap(3.0)          # holding the connection IS the ask
                seq, pts = Scope.read_tap()
                if pts is None or seq == last_seq:
                    time.sleep(0.02)
                    idle += 0.02
                    if idle > 30.0:          # nothing is producing; let go
                        return
                    continue
                idle = 0.0
                last_seq = seq
                blob = _scope_jpeg(size, pts, exposure=exposure)
                if blob is None:
                    continue
                try:
                    self.wfile.write(HEADER_BOUNDARY)
                    self.wfile.write(HEADER_CTYPE_JPEG)
                    self.wfile.write(blob)
                    self.wfile.write(HEADER_NEWLINE)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError,
                        ConnectionAbortedError, OSError):
                    return
                # Cap the preview well under the trace rate. The picture only
                # changes at IPS anyway, and this is the CPU knob.
                time.sleep(1.0 / max(getattr(settings, "SCOPE_PREVIEW_FPS", 12), 1))
        except (BrokenPipeError, ConnectionResetError,
                ConnectionAbortedError, socket.timeout, OSError):
            return
        except Exception as e:
            print(f"[Scope preview] {e}", file=sys.stderr)

    def do_POST(self):
        """Control endpoints.

        The first POST handler in this file.  Note the ASCII panel in
        HTML_TEMPLATE already POSTs to /ascii/size, which has never had a
        handler and whose get_ascii_dimensions() helper is not defined
        anywhere -- so that panel has always been dead.  This does not revive
        it; it only adds the scope one.
        """
        if self.path == "/scope/device":
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if length > 4096:                      # nothing legitimate is
                    self.send_error(413)               # anywhere near this
                    return
                body = self.rfile.read(length) if length else b"{}"
                payload = json.loads(body.decode("utf-8") or "{}")
                # "" and null both mean the system default, which is what the
                # page sends for its first option.
                spec = payload.get("device")
                if isinstance(spec, str) and not spec.strip():
                    spec = None

                import scope_display
                scope_display.request_device(spec)
                result = {"ok": True, "requested": spec}
            except Exception as e:
                result = {"ok": False, "error": str(e)}
            blob = json.dumps(result).encode("utf-8")
            self.send_response(200 if result.get("ok") else 400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)
            return

        self.send_error(404)


class StreamHandler(RobustHandlerMixin, http.server.BaseHTTPRequestHandler):
    # Timeout for streaming connections
    timeout = 15

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
            except (IOError, OSError) as e:
                print(f"[Web] Error serving index.html: {e}", file=sys.stderr)
                self.send_error(404)

        # [NEW] Dynamic Template Fallback
        elif _serve_dynamic_template(self):
            return

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
        except (OSError, AttributeError) as e:
            # Socket options may fail on some platforms or if connection is already closed
            # This is non-critical, so we silently continue
            pass

        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.end_headers()

        # [NEW] Stall Timeout Logic
        # If the producer stops sending frames for >10s, we disconnect the client.
        stall_timeout = getattr(settings, "STREAM_STALL_TIMEOUT", 10.0)
        last_frame_time = time.monotonic()

        try:
            while True:
                # Poll every 1.0s to check for new frames OR stall conditions
                raw_payload = exchange_web.get_frame(timeout=1.0)

                if raw_payload is None:
                    # No frame arrived in the last second. Check if we have stalled.
                    elapsed = time.monotonic() - last_frame_time
                    if elapsed > stall_timeout:
                        # Producer is dead/stuck. Break loop to free the slot.
                        print(f"[Stream] Client {cid}: Stall timeout ({elapsed:.1f}s > {stall_timeout}s). Disconnecting.")
                        break
                    # Frames are coming slowly but not stalled - continue waiting
                    # Don't send anything, just wait for next frame
                    continue

                if not isinstance(raw_payload, bytes) or len(raw_payload) < 1:
                    print(f"[Stream] Client {cid}: Invalid frame payload (type={type(raw_payload)}, len={len(raw_payload) if hasattr(raw_payload, '__len__') else 'N/A'})")
                    continue

                # Valid frame received, update watchdog
                last_frame_time = time.monotonic()

                fmt_byte = raw_payload[0:1]
                frame_data = raw_payload[1:]

                # Validate frame data
                if len(frame_data) == 0:
                    print(f"[Stream] Client {cid}: Empty frame data, skipping")
                    continue

                # 2. Determine Header Type
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
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError) as write_err:
                    # Connection closed during write - normal disconnect
                    print(f"[Stream] Client {cid}: Write error: {write_err}")
                    break

        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, socket.timeout, OSError):
            # Normal disconnect or network error
            pass
        except Exception as e:
            # Log unusual errors with traceback for debugging
            import traceback
            print(f"[Stream Error] {e}", file=sys.stderr)
            print(f"[Stream Error] Traceback:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        finally:
            _viewer_semaphore.release()
            with _count_lock:
                _current_viewer_count -= 1
            _clear_heartbeat(cid)


def run_monitor_server():
    port = get_config().get_monitor_port()
    # Default to 127.0.0.1 (Localhost only) for security.
    host = getattr(settings, 'WEB_HOST', '127.0.0.1')
    try:
        httpd = ThreadedTCPServer((host, port), MonitorHandler)
        print(f">> Monitor running on {host}:{port}")
        httpd.serve_forever()
    except OSError as e:
        if e.errno == 98:  # Address already in use
            print(f"❌ ERROR: Port {port} is already in use")
            print(f"   -> Is another instance running?")
            print(f"   -> Try: sudo systemctl stop vi-web")
            print(f"   -> Or check: sudo lsof -i :{port}")
            raise
        raise


def run_stream_server():
    port = get_config().get_stream_port()
    if port is None:
        raise RuntimeError("Stream port not configured for current mode")
    # Default to 0.0.0.0 (Public) for the stream.
    host = getattr(settings, 'STREAM_HOST', '0.0.0.0')
    try:
        httpd = ThreadedTCPServer((host, port), StreamHandler)
        print(f">> Stream running on {host}:{port}")
        httpd.serve_forever()
    except OSError as e:
        if e.errno == 98:  # Address already in use
            print(f"❌ ERROR: Port {port} is already in use")
            print(f"   -> Is another instance running?")
            print(f"   -> Try: sudo systemctl stop vi-web")
            print(f"   -> Or check: sudo lsof -i :{port}")
            raise
        raise


def start_server(monitor=True, stream=True):
    if monitor: threading.Thread(target=run_monitor_server, daemon=True, name="MonitorServer").start()
    if stream: threading.Thread(target=run_stream_server, daemon=True, name="StreamServer").start()