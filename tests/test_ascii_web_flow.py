"""Real WebSocket delivery: preserve setup/partial writes; never wait for ACKs."""
import base64
import gzip
import json
import os
import socket
import struct
import threading
import time
import unittest
from unittest.mock import patch

import ascii_web_server as web


class SocketDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.clients = patch.object(web, 'clients', [])
        self.clients.start()
        self.server = web.LatestWebSocketServer('127.0.0.1', 0, web.AsciiWebSocket,
                                                selectInterval=.005)
        self.stop = threading.Event()
        self.errors = []
        def serve():
            try:
                while not self.stop.is_set():
                    self.server.serveonce()
            except Exception as exc:
                self.errors.append(exc)
        self.thread = threading.Thread(target=serve)
        self.thread.start()
        self.sock = socket.create_connection(self.server.serversocket.getsockname(), timeout=2)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall(('GET / HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\n'
                           'Connection: Upgrade\r\nSec-WebSocket-Version: 13\r\n'
                           f'Sec-WebSocket-Key: {key}\r\n\r\n').encode())
        header = b''
        while not header.endswith(b'\r\n\r\n'):
            header += self.exact(1)
        self.assertIn(b'101', header)
        self.assertEqual(self.receive(), web.ANSI_CLEAR)
        self.assertEqual(json.loads(self.receive())['type'], 'resize')
        self.client = web.clients[0]

    def tearDown(self):
        self.sock.close()
        self.stop.set()
        self.thread.join(timeout=3)
        self.server.close()
        self.clients.stop()
        self.assertFalse(self.thread.is_alive())
        self.assertEqual(self.errors, [])

    def exact(self, count):
        data = b''
        while len(data) < count:
            part = self.sock.recv(count - len(data))
            if not part:
                raise EOFError('WebSocket closed')
            data += part
        return data

    def receive(self):
        header = self.exact(2)
        size = header[1] & 127
        if size == 126:
            size = struct.unpack('!H', self.exact(2))[0]
        elif size == 127:
            size = struct.unpack('!Q', self.exact(8))[0]
        payload = self.exact(size)
        return payload.decode() if header[0] & 15 == 1 else payload

    def send(self, **message):
        payload = json.dumps(message).encode()
        mask = os.urandom(4)
        self.sock.sendall(bytes((0x81, 0x80 | len(payload))) + mask +
                          bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload)))

    def wait_for(self, predicate):
        end = time.monotonic() + 2
        while not predicate() and time.monotonic() < end:
            time.sleep(.005)
        self.assertTrue(predicate())

    def test_continuous_gzip_delivery_needs_no_ack_even_after_old_flow_message(self):
        self.send(type='encoding', encoding='gzip')
        self.send(type='flow')  # a cached viewer must not opt back into gating
        self.wait_for(lambda: self.client.frame_encoding == 'gzip')
        for value in ('first', 'second', 'third'):
            self.client.queue_picture(value, b'g' + gzip.compress(value.encode()))
            self.assertEqual(gzip.decompress(self.receive()[1:]), value.encode())

    def test_15fps_producer_is_not_capped_by_missing_browser_feedback(self):
        count = 12
        def produce():
            start = time.monotonic()
            for i in range(count):
                self.client.queue_picture(str(i))
                time.sleep(max(0, start + (i + 1) / 15 - time.monotonic()))
        producer = threading.Thread(target=produce)
        start = time.monotonic()
        producer.start()
        try:
            self.assertEqual([self.receive() for _ in range(count)],
                             [str(i) for i in range(count)])
            self.assertLess(time.monotonic() - start, 1.5)
        finally:
            producer.join(timeout=2)

    def test_legacy_client_keeps_receiving_without_flow_negotiation(self):
        for value in ('first', 'second'):
            self.client.queue_picture(value)
            self.assertEqual(self.receive(), value)

    def test_partial_large_frame_is_finished_without_corrupting_next_picture(self):
        value = 'x' * (512 * 1024)
        self.client.queue_picture(value)
        self.wait_for(lambda: self.client._pending_picture is None)
        for i in range(50):
            self.client.queue_picture('newest-' + str(i))
        self.assertEqual(self.receive(), value)
        self.assertEqual(self.receive(), 'newest-49')
