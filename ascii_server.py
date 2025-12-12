import socketserver
import socket
import threading
import time
import settings
from shared_state import exchange

# --- CONFIGURATION ---
HOST = getattr(settings, 'ASCII_HOST', '0.0.0.0')
PORT = getattr(settings, 'ASCII_PORT', 2323)
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
            except:
                pass
            return

        print(f"[ASCII] Client connected: {self.client_address}")

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
                frame_data = exchange.get_frame()

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

        except Exception as e:
            print(f"[ASCII] Connection Error: {e}")

        finally:
            # 7. Cleanup
            try:
                self.request.sendall(ANSI_SHOW_CURSOR)
            except:
                pass
            _sem.release()
            print(f"[ASCII] Client disconnected: {self.client_address}")


def start_server():
    print(f"📠 ASCII Telnet Server started on {HOST}:{PORT}")
    server = ThreadedTCPServer((HOST, PORT), AsciiHandler)
    server.serve_forever()