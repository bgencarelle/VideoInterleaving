import socketserver
import threading
import time
import settings
from shared_state import exchange

MAX_CLIENTS = getattr(settings, 'MAX_VIEWERS', 20)
_sem = threading.Semaphore(MAX_CLIENTS)


class AsciiHandler(socketserver.BaseRequestHandler):
    def handle(self):
        # Capacity Check
        if not _sem.acquire(blocking=False):
            self.request.sendall(b"Server Full.\r\n")
            return

        print(f"[ASCII] Client connected: {self.client_address}")

        try:
            # ANSI Setup: Hide Cursor (\033[?25l) + Clear (\033[2J)
            init_code = b"\033[?25l\033[2J"
            self.request.sendall(init_code)

            fps = getattr(settings, 'ASCII_FPS', 10)
            interval = 1.0 / fps

            while True:
                start_t = time.time()

                # Blocking wait for next frame
                frame_data = exchange.get_frame()

                if frame_data:
                    # 1. Move Cursor Home (\033[H)
                    # 2. Send Frame Data (encoded to bytes)
                    payload = b"\033[H" + frame_data.encode('utf-8', errors='ignore')
                    self.request.sendall(payload)

                # Rate Limiting
                dt = time.time() - start_t
                sleep_t = interval - dt
                if sleep_t > 0:
                    time.sleep(sleep_t)

        except (BrokenPipeError, ConnectionResetError):
            pass  # Normal disconnect
        except Exception as e:
            print(f"[ASCII] Error: {e}")
        finally:
            # Reset Cursor (\033[?25h)
            try:
                self.request.sendall(b"\033[?25h")
            except:
                pass
            _sem.release()
            print(f"[ASCII] Client disconnected: {self.client_address}")


class ThreadedTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def start_server():
    if not getattr(settings, 'ASCII_MODE', False):
        return

    host = getattr(settings, 'ASCII_HOST', '0.0.0.0')
    port = getattr(settings, 'ASCII_PORT', 2323)

    server = ThreadedTCPServer((host, port), AsciiHandler)
    print(f"📠 ASCII Telnet Server started on port {port}")
    server.serve_forever()