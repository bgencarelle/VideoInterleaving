import threading
import socket
import gzip
import ssl
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
    def _sendBuffer(self, buff, send_all=False):
        remaining = super()._sendBuffer(buff, send_all)
        self._head_started = remaining is not None
        return remaining

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
            import json
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
            import json
            if not isinstance(self.data, str) or len(self.data) > 4096:
                return
            data = json.loads(self.data)
            if data.get('type') == 'encoding' and data.get('encoding') == 'gzip':
                self.frame_encoding = 'gzip'
            if data.get('type') == 'resize':
                # Client requesting dimension change (optional feature)
                # For now, we'll let the server control dimensions via HTTP API
                # This could be extended to allow client-initiated changes
                pass
        except (json.JSONDecodeError, AttributeError, KeyError):
            # Not a JSON message or not a resize command - ignore
            pass


def _keep_only_latest(client):
    """Drop queued pictures no viewer needs; the newest one replaces them.

    The library's head entry is a picture it has already begun to write when
    _head_started is set: that one must stay whole, so only the entries behind
    it are dropped. Otherwise nothing has begun and the whole queue is stale.
    """
    queue = client.sendq
    if getattr(client, '_head_started', False) is True:
        while len(queue) > 1:
            queue.pop()
    else:
        queue.clear()


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
                # --- NEAR-ZERO BUFFERING ---
                # The library never blocks on a slow client: sendMessage
                # just appends to client.sendq, and the select loop drains
                # it once the socket is writable. A tab the browser has
                # frozen or throttled stops reading, its queue grows without
                # bound, and when it wakes it replays the entire backlog --
                # minutes of stale frames. We do not want history on the
                # screen, we want the current frame: so if a client is still
                # draining more than a couple of frames, throw the backlog
                # away and enqueue only this one. A healthy client's queue
                # empties between broadcasts (the select loop runs every
                # 10ms and frames arrive at ~15fps), so it never trips this;
                # a frozen one caps at three frames.
                _keep_only_latest(client)

                # If buffer is empty, send the new frame
                client.sendMessage(compressed if getattr(client, 'frame_encoding', 'text') == 'gzip'
                                   and compressed is not None else payload)

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

    # sendMessage queues data on a producer thread; it does not wake the
    # library's select(). Its 100-ms default added a polling-period delay to
    # frames already ready for presentation. Bound that handoff to 10 ms.
    server = SimpleWebSocketServer(HOST, port, AsciiWebSocket,
                                  selectInterval=0.01)
    server.serveforever()
