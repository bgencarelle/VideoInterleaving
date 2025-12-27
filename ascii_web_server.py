import threading
import socket
import settings
from shared_state import exchange_ascii
from SimpleWebSocketServer import SimpleWebSocketServer, WebSocket
from server_config import get_config

# Configuration
HOST = getattr(settings, 'ASCII_HOST', '127.0.0.1')
MAX_CLIENTS = getattr(settings, 'MAX_VIEWERS', 20)

_sem = threading.Semaphore(MAX_CLIENTS)
clients = []

ANSI_CLEAR = "\033[?25l\033[2J"
ANSI_HOME = "\033[H"


class AsciiWebSocket(WebSocket):
    def handleConnected(self):
        if not _sem.acquire(blocking=False):
            self.close()
            return

        sem_acquired = True
        client_added = False
        try:
            # Keep the OS buffer small too, just in case
            self.client.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
            # Disable Nagle for instant updates
            self.client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            print(f"[WS] Client connected: {self.address}")
            clients.append(self)
            client_added = True
            sem_acquired = False  # Client added, handleClose will release semaphore
            self.sendMessage(ANSI_CLEAR)
        except Exception as e:
            print(f"[WS] Handshake Error: {e}")
            # Cleanup: remove client from list if it was added, and release semaphore
            if client_added:
                if self in clients:
                    clients.remove(self)
                _sem.release()
            elif sem_acquired:
                # Semaphore acquired but client not added - release it
                _sem.release()
            self.close()

    def handleClose(self):
        # Ensure cleanup even if called multiple times
        if self in clients:
            clients.remove(self)
            _sem.release()
            print(f"[WS] Client disconnected: {self.address}")

    def handleMessage(self):
        pass


def broadcast_loop():
    """Pushes frames to all connected WS clients."""
    while True:
        # Blocking Wait
        frame_data = exchange_ascii.get_frame()

        if not frame_data:
            continue

        try:
            if isinstance(frame_data, bytes):
                text = frame_data.decode('utf-8', errors='ignore')
            else:
                text = frame_data
            payload = ANSI_HOME + text
        except Exception:
            continue

        # Broadcast
        dead_clients = []
        for client in list(clients):
            try:
                # --- THE "LEAKY BUCKET" FIX ---
                # Check the internal library buffer.
                # If 'sendq' has data, the client is lagging (tab hidden).
                # Skip this frame for this specific client.
                if hasattr(client, 'sendq') and client.sendq:
                    continue

                # If buffer is empty, send the new frame
                client.sendMessage(payload)

            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                # Connection error - mark for cleanup
                print(f"[WS] Client {getattr(client, 'address', 'unknown')} connection error: {e}")
                dead_clients.append(client)
            except Exception as e:
                # Unexpected error - log and mark for cleanup
                print(f"[WS] Unexpected error sending to client {getattr(client, 'address', 'unknown')}: {e}")
                dead_clients.append(client)
        
        # Clean up dead clients
        for client in dead_clients:
            try:
                if client in clients:
                    clients.remove(client)
                    _sem.release()
                client.close()
            except Exception as e:
                print(f"[WS] Error cleaning up dead client: {e}")


def start_server():
    port = get_config().get_ascii_websocket_port()
    if port is None:
        raise RuntimeError("ASCII WebSocket port not configured for current mode")
    print(f"🕸️  ASCII WebSocket Server started on {HOST}:{port}")
    t = threading.Thread(target=broadcast_loop, daemon=True, name="WS-Broadcaster")
    t.start()

    server = SimpleWebSocketServer(HOST, port, AsciiWebSocket)
    server.serveforever()