import threading
import socket
import gzip
import ssl
import json
from collections import deque
import settings
from shared_state import exchange_ascii

# SimpleWebSocketServer 0.1.2 evaluates this removed constant while defining
# its unused SSL server, even when importing only the plain WebSocket server.
# Supply a modern server-context default for that import only; do not leave
# a deprecated-protocol alias installed process-wide (Python 3.14+).
_missing_legacy_tls_default = not hasattr(ssl, 'PROTOCOL_TLSv1_2')
if _missing_legacy_tls_default:
    ssl.PROTOCOL_TLSv1_2 = ssl.PROTOCOL_TLS_SERVER
try:
    from SimpleWebSocketServer import SimpleWebSocketServer, WebSocket
finally:
    if _missing_legacy_tls_default:
        del ssl.PROTOCOL_TLSv1_2
from server_config import get_config

# Configuration
HOST = getattr(settings, 'ASCII_HOST', '127.0.0.1')
MAX_CLIENTS = getattr(settings, 'MAX_VIEWERS', 20)

_sem = threading.Semaphore(MAX_CLIENTS)
clients = []

ANSI_CLEAR = "\033[?25l\033[2J"
ANSI_HOME = "\033[H"


class AsciiWebSocket(WebSocket):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._delivery_lock = threading.Lock()
        self._controls = deque()
        self._pending_picture = None
        self._flow = False
        self._awaiting_ack = None
        self._picture_id = 0

    def sendMessage(self, message):
        # Control messages may originate outside the socket thread too. Never
        # let a producer mutate the library's sendq (including a partial head).
        with self._delivery_lock:
            self._controls.append(message)

    def queue_picture(self, plain, compressed=None):
        with self._delivery_lock:
            self._pending_picture = (plain, compressed)

    def flush_delivery(self):
        """Socket thread only: preserve controls and finish every started frame."""
        with self._delivery_lock:
            while self._controls:
                super().sendMessage(self._controls.popleft())
            if (self.sendq or self._pending_picture is None or
                    (self._flow and self._awaiting_ack is not None)):
                return
            plain, compressed = self._pending_picture
            self._pending_picture = None
        message = (compressed if self.frame_encoding == 'gzip' and
                   compressed is not None else plain)
        if self._flow:
            self._picture_id += 1
            self._awaiting_ack = self._picture_id
            super().sendMessage(json.dumps({'type': 'frame', 'id': self._picture_id}))
        super().sendMessage(message)

    def handleConnected(self):
        self.frame_encoding = 'text'
        if not _sem.acquire(blocking=False):
            self.close()
            return

        sem_acquired = True
        client_added = False
        sem_released = False  # Track if semaphore was already released
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
            width, height = (getattr(settings, 'ASCII_WEB_SOURCE_SIZE', None) or
                             (getattr(settings, 'ASCII_WIDTH', 60),
                              getattr(settings, 'ASCII_HEIGHT', 40)))
            self.sendMessage(json.dumps({'type': 'resize', 'cols': width, 'rows': height}))
        except Exception as e:
            print(f"[WS] Handshake Error: {e}")
            # Cleanup: remove client from list if it was added, and release semaphore
            if client_added:
                if self in clients:
                    clients.remove(self)
                _sem.release()
                sem_released = True  # Mark semaphore as released
            elif sem_acquired:
                # Semaphore acquired but client not added - release it
                _sem.release()
                sem_released = True  # Mark semaphore as released
            self.close()
            # Store flag so handleClose knows not to release again
            self._sem_released_in_exception = sem_released

    def handleClose(self):
        # Ensure cleanup even if called multiple times
        if self in clients:
            clients.remove(self)
            # Only release semaphore if it wasn't already released in exception handler or broadcast cleanup
            if not getattr(self, '_sem_released_in_exception', False) and not getattr(self, '_sem_released_in_broadcast', False):
                _sem.release()
            print(f"[WS] Client disconnected: {self.address}")

    def handleMessage(self):
        """Handle incoming WebSocket messages (e.g., dimension change requests)."""
        try:
            if not isinstance(self.data, str) or len(self.data) > 4096:
                return
            data = json.loads(self.data)
            if not isinstance(data, dict):
                return
            if data.get('type') == 'encoding' and data.get('encoding') == 'gzip':
                self.frame_encoding = 'gzip'
            if data.get('type') == 'flow':
                # Repeated negotiation must not grant extra frame credit.
                self._flow = True
            if (data.get('type') == 'ack' and self._flow and
                    type(data.get('id')) is int and
                    data['id'] == self._awaiting_ack):
                self._awaiting_ack = None
            if data.get('type') == 'resize':
                # Client requesting dimension change (optional feature)
                # For now, we'll let the server control dimensions via HTTP API
                # This could be extended to allow client-initiated changes
                pass
        except (json.JSONDecodeError, AttributeError, KeyError):
            # Not a JSON message or not a resize command - ignore
            pass


class LatestWebSocketServer(SimpleWebSocketServer):
    def serveonce(self):
        # Admission and library writes share one owner. Producers only replace
        # mailboxes; neither they nor acknowledgements ever clear sendq.
        for client in list(self.connections.values()):
            client.flush_delivery()
        super().serveonce()


def broadcast_loop():
    """Push each frame published by the ASCII_FPS-paced image producer."""
    while True:
        frame_data = exchange_ascii.get_frame()

        if not frame_data:
            continue

        try:
            if isinstance(frame_data, bytes):
                text = frame_data.decode('utf-8', errors='ignore')
            else:
                text = frame_data
            payload = ANSI_HOME + text
            # Encode ONCE per publication, never per viewer or zoom level.
            compressed = (b'g' + gzip.compress(payload.encode('utf-8'), compresslevel=1, mtime=0)
                          if any(getattr(client, 'frame_encoding', 'text') == 'gzip'
                                 for client in list(clients)) else None)
        except Exception:
            continue

        # Broadcast
        dead_clients = []
        for client in list(clients):
            try:
                # One replaceable waiting picture; encoded once for all clients.
                client.queue_picture(payload, compressed)

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
                    # Mark that semaphore was released so handleClose() doesn't release again
                    client._sem_released_in_broadcast = True
                client.close()
            except Exception as e:
                print(f"[WS] Error cleaning up dead client: {e}")


def broadcast_dimension_change(width, height):
    """Broadcast dimension change to all connected WebSocket clients."""
    try:
        import json
        message = json.dumps({
            "type": "resize",
            "cols": width,
            "rows": height
        })
        # Broadcast to all clients
        dead_clients = []
        for client in list(clients):
            try:
                client.sendMessage(message)
            except (BrokenPipeError, ConnectionResetError, OSError):
                dead_clients.append(client)
            except Exception as e:
                print(f"[WS] Error broadcasting dimension change: {e}")
                dead_clients.append(client)
        
        # Clean up dead clients
        for client in dead_clients:
            try:
                if client in clients:
                    clients.remove(client)
                    _sem.release()
                    client._sem_released_in_broadcast = True
                client.close()
            except Exception:
                pass
    except Exception as e:
        print(f"[WS] Error in broadcast_dimension_change: {e}")


def start_server():
    port = get_config().get_ascii_websocket_port()
    if port is None:
        raise RuntimeError("ASCII WebSocket port not configured for current mode")
    print(f"🕸️  ASCII WebSocket Server started on {HOST}:{port}")
    t = threading.Thread(target=broadcast_loop, daemon=True, name="WS-Broadcaster")
    t.start()

    # Bound mailbox admission latency without changing the producer's cadence.
    server = LatestWebSocketServer(HOST, port, AsciiWebSocket,
                                  selectInterval=0.01)
    server.serveforever()
