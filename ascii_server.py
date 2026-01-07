import socketserver
import socket
import sys
import threading
import time
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
            self.request.sendall(ANSI_CLEAR)

            # Timing setup
            target_fps = getattr(settings, 'ASCII_FPS', 15)
            min_interval = 1.0 / target_fps
            last_send_time = 0

            while True:
                # 4. Blocking Wait (0% CPU usage)
                # Waits for the main display loop to signal a new frame
                frame_data = exchange_ascii.get_frame()

                if not frame_data:
                    break

                # 5. Rate Limiting (Frame Skip)
                # If frames are coming too fast, skip sending to save bandwidth
                now = time.monotonic()
                if now - last_send_time < min_interval:
                    continue

                # 6. Payload Construction
                try:
                    if isinstance(frame_data, bytes):
                        # Optimization: If it's already bytes, don't decode/encode
                        payload = ANSI_HOME + frame_data
                    else:
                        # Standard String Path
                        # Combine Home + Text and encode once
                        payload = (f"\033[H{frame_data}").encode('utf-8', errors='ignore')

                    self.request.sendall(payload)
                    last_send_time = now

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
                self.request.sendall(ANSI_SHOW_CURSOR)
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