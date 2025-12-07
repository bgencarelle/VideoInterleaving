import socketserver
import socket
import threading
import time
import settings
from shared_state import exchange

# --- CONFIGURATION ---
# Default to 0.0.0.0 to allow external connections (Required for LAN access)
HOST = getattr(settings, 'ASCII_HOST', '0.0.0.0')
PORT = getattr(settings, 'ASCII_PORT', 2323)
MAX_CLIENTS = getattr(settings, 'MAX_VIEWERS', 20)

# Semaphore to prevent CPU overload from too many active telnet sessions
_sem = threading.Semaphore(MAX_CLIENTS)


class ThreadedTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class AsciiHandler(socketserver.BaseRequestHandler):
    def handle(self):
        # 1. Capacity Check (From your code)
        if not _sem.acquire(blocking=False):
            print(f"[ASCII] Rejected connection from {self.client_address}: Server Full")
            try:
                self.request.sendall(b"Server Full. Try again later.\r\n")
            except:
                pass
            return

        print(f"[ASCII] Client connected: {self.client_address} ({_sem._value + 1} slots left)")

        try:
            # 2. TCP Keepalive (Crucial for detecting dropped WiFi connections)
            self.request.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            # Linux-specific tuning (detect dead connection in ~60 seconds)
            if hasattr(socket, 'TCP_KEEPIDLE'):
                self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 10)
            if hasattr(socket, 'TCP_KEEPINTVL'):
                self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
            if hasattr(socket, 'TCP_KEEPCNT'):
                self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5)

            # 3. ANSI Setup: Hide Cursor + Clear Screen
            self.request.sendall(b"\033[?25l\033[2J")

            fps = getattr(settings, 'ASCII_FPS', 15)
            interval = 1.0 / fps
            last_frame_time = 0

            while True:
                # Rate Limiting
                now = time.monotonic()
                if now - last_frame_time < interval:
                    time.sleep(0.01)
                    continue

                # Blocking wait logic handled by main loop timing,
                # here we just grab the latest available frame.
                frame_data = exchange.get_frame()

                if not frame_data:
                    time.sleep(0.05)
                    continue

                # Defensive coding: Convert bytes to string if needed
                text = frame_data
                if isinstance(frame_data, bytes):
                    try:
                        text = frame_data.decode('utf-8', errors='ignore')
                    except:
                        continue

                # 4. Construct Payload: Home Cursor + Text
                payload = f"\033[H{text}".encode('utf-8', errors='ignore')
                self.request.sendall(payload)

                last_frame_time = now

        except (BrokenPipeError, ConnectionResetError):
            # Normal disconnection
            pass
        except Exception as e:
            print(f"[ASCII] Error handling client {self.client_address}: {e}")
        finally:
            # 5. Cleanup
            try:
                # Restore Cursor
                self.request.sendall(b"\033[?25h")
            except:
                pass

            _sem.release()
            print(f"[ASCII] Client disconnected: {self.client_address}")


def start_server():
    print(f"📠 ASCII Telnet Server started on {HOST}:{PORT}")
    server = ThreadedTCPServer((HOST, PORT), AsciiHandler)
    server.serve_forever()