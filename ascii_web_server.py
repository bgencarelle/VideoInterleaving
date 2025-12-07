import threading
import time
import settings
from shared_state import exchange
from SimpleWebSocketServer import SimpleWebSocketServer, WebSocket

# Configuration
HOST = getattr(settings, 'ASCII_HOST', '0.0.0.0')
PORT = 2324  # Dedicated WebSocket Port
MAX_CLIENTS = getattr(settings, 'MAX_VIEWERS', 20)

_sem = threading.Semaphore(MAX_CLIENTS)
clients = []


class AsciiWebSocket(WebSocket):
    def handleConnected(self):
        if not _sem.acquire(blocking=False):
            self.close()
            return

        print(f"[WS] Client connected: {self.address}")
        clients.append(self)
        # Clear Screen ANSI
        self.sendMessage("\033[?25l\033[2J")

    def handleClose(self):
        if self in clients:
            clients.remove(self)
            _sem.release()
            print(f"[WS] Client disconnected: {self.address}")

    def handleMessage(self):
        pass


def broadcast_loop():
    """Pushes frames to all connected WS clients."""
    fps = getattr(settings, 'ASCII_FPS', 15)
    interval = 1.0 / fps
    last_frame_time = 0

    while True:
        now = time.monotonic()
        if now - last_frame_time < interval:
            time.sleep(0.01)
            continue

        frame_data = exchange.get_frame()
        if not frame_data:
            time.sleep(0.05)
            continue

        # Decode bytes to string for WebSocket
        if isinstance(frame_data, bytes):
            try:
                text = frame_data.decode('utf-8', errors='ignore')
            except:
                continue
        else:
            text = frame_data

        # Payload: Home Cursor + Text
        payload = f"\033[H{text}"

        # Broadcast
        for client in list(clients):
            try:
                client.sendMessage(payload)
            except:
                pass

        last_frame_time = now


def start_server():
    print(f"🕸️  ASCII WebSocket Server started on {HOST}:{PORT}")

    # Background broadcaster
    t = threading.Thread(target=broadcast_loop, daemon=True, name="WS-Broadcaster")
    t.start()

    # Main Server Loop
    server = SimpleWebSocketServer(HOST, PORT, AsciiWebSocket)
    server.serveforever()