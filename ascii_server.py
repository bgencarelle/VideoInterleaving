import socketserver
import socket
import sys
import threading
import select
import time
from ascii_terminal import WindowSize, pictures, NEGOTIATE, ENTER, LEAVE
import settings
from shared_state import exchange_ascii, ascii_client_count
from server_config import get_config

# --- CONFIGURATION ---
HOST = getattr(settings, 'ASCII_HOST', '127.0.0.1')
MAX_CLIENTS = getattr(settings, 'MAX_VIEWERS', 20)

_sem = threading.Semaphore(MAX_CLIENTS)

# Pre-encoded ANSI bytes to save CPU during the loop
ANSI_CLEAR = b"\033[?25l\033[2J"
ANSI_HOME = b"\033[H"
ANSI_SHOW_CURSOR = b"\033[?25h"


class ThreadedTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class AsciiHandler(socketserver.BaseRequestHandler):
    def handle(self):
        # 1. Capacity Check
        if not _sem.acquire(blocking=False):
            print(f"[ASCII] Rejected {self.client_address}: Server Full")
            try:
                self.request.sendall(b"Server Full. Try again later.\r\n")
            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                # Connection may be closed before we can send rejection message
                print(f"[ASCII] Could not send rejection message: {e}")
            return

        ascii_client_count.increment()
        print(f"[ASCII] Client connected: {self.client_address} (Total: {ascii_client_count.get_count()})")

        try:
            # 2. Socket Optimization
            # Keepalive: Detect dropped WiFi/Internet connections
            self.request.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

            # NoDelay: CRITICAL for ASCII animation. Disables Nagle's algo
            # to prevent buffering small packets (reduces stutter).
            self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            # Linux-specific Keepalive tuning (approx 60s timeout)
            if hasattr(socket, 'TCP_KEEPIDLE'):
                self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 10)
            elif hasattr(socket, 'TCP_KEEPALIVE') and sys.platform == 'darwin':
                # macOS uses TCP_KEEPALIVE for the idle time (seconds)
                self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPALIVE, 10)
            if hasattr(socket, 'TCP_KEEPINTVL'):
                self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
            if hasattr(socket, 'TCP_KEEPCNT'):
                self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5)

            # 3. Setup Terminal
            self.request.settimeout(2.0)
            self.request.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16384)
            self.request.sendall(NEGOTIATE + ENTER)
            window = WindowSize()
            identity = None
            applied_size = window.size
            last_resize = 0.0
            input_started = time.monotonic()
            input_bytes = 0
            reported_size = None
            fallback_logged = False
            # Explicit experimental opt-in; ASCII-web never uses this encoder.
            delta = None
            if getattr(settings, 'ASCII_DELTA_UPDATES', False):
                from ascii_delta import TerminalDelta
                delta = TerminalDelta()

            while True:
                # The producer publishes on the ASCII_FPS ticker. Wait for
                # each new frame instead of adding a second, drifting rate cap.
                readable, _, _ = select.select([self.request], [], [], 0.02)
                if readable:
                    incoming = self.request.recv(4096)
                    if not incoming:
                        break
                    now = time.monotonic()
                    if now - input_started >= 1.0:
                        input_started, input_bytes = now, 0
                    input_bytes += len(incoming)
                    if input_bytes > 8192:
                        break
                    reply = window.feed(incoming)
                    if window.reported != reported_size:
                        reported_size = window.reported
                        print(f'[ASCII] Window {self.client_address}: reported={reported_size}, output={window.size}')
                    if reply:
                        self.request.sendall(reply)
                now = time.monotonic()
                if not fallback_logged and now - input_started >= 1.0:
                    fallback_logged = True
                    if reported_size is None:
                        print(f'[ASCII] Window {self.client_address}: no NAWS size received; fallback={window.size}')
                if window.size != applied_size and now - last_resize >= 0.5:
                    applied_size = window.size
                    last_resize = now
                    self.request.sendall(b'\x1b[2J')
                    if delta is not None:
                        delta.reset()
                identity, payload = pictures.picture(applied_size, identity)
                if payload is None:
                    continue

                # Payload Construction
                try:
                    if delta is not None:
                        payload = delta.encode(payload, applied_size)
                    self.request.sendall(payload)

                except (BrokenPipeError, ConnectionResetError):
                    break
                except Exception as e:
                    print(f"[ASCII] Send Error: {e}")
                    break

        except (ConnectionResetError, BrokenPipeError):
            # Client disconnected gracefully or abruptly
            pass
        except Exception as e:
            print(f"[ASCII] Connection Error: {e}")

        finally:
            # 7. Cleanup
            try:
                self.request.sendall(LEAVE)
            except (BrokenPipeError, ConnectionResetError, OSError):
                # Connection already closed, cursor restore not needed
                pass
            _sem.release()
            ascii_client_count.decrement()
            print(f"[ASCII] Client disconnected: {self.client_address} (Total: {ascii_client_count.get_count()})")


def start_server():
    port = get_config().get_ascii_telnet_port()
    if port is None:
        raise RuntimeError("ASCII telnet port not configured for current mode")
    print(f"📠 ASCII Telnet Server started on {HOST}:{port}")
    server = ThreadedTCPServer((HOST, port), AsciiHandler)
    server.serve_forever()
