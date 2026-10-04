"""Small loopback bridge for publishing the app's rendered local-mode frames.

This module is deliberately independent of both the UI and modem packages so
the local display can publish its output without importing the sender stack.
"""
import socket
import struct
import threading

import numpy as np


_MAGIC = b'VIF1'
_HEADER = struct.Struct('!4sII')
_MAX_FRAME_BYTES = 64 * 1024 * 1024


def read_frame(sock):
    """Read one RGB frame from a connected socket; return None on clean EOF."""
    header = _read_exact(sock, _HEADER.size, allow_eof=True)
    if header is None:
        return None
    magic, width, height = _HEADER.unpack(header)
    size = width * height * 3
    if (magic != _MAGIC or width < 1 or height < 1 or
            size > _MAX_FRAME_BYTES):
        raise ValueError('invalid local-frame header')
    data = _read_exact(sock, size)
    return np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)


def write_frame(sock, frame):
    """Write one contiguous uint8 RGB frame to a connected socket."""
    pixels = np.asarray(frame)
    if (pixels.dtype != np.uint8 or pixels.ndim != 3 or
            pixels.shape[2] != 3):
        raise ValueError('local frames must be HxWx3 uint8 RGB arrays')
    height, width = pixels.shape[:2]
    size = width * height * 3
    if width < 1 or height < 1 or size > _MAX_FRAME_BYTES:
        raise ValueError('local frame dimensions are outside supported limits')
    pixels = np.ascontiguousarray(pixels)
    sock.sendall(_HEADER.pack(_MAGIC, width, height))
    sock.sendall(memoryview(pixels).cast('B'))


def _read_exact(sock, count, allow_eof=False):
    chunks = bytearray(count)
    view = memoryview(chunks)
    have = 0
    while have < count:
        received = sock.recv_into(view[have:])
        if not received:
            if allow_eof and have == 0:
                return None
            raise ConnectionError('local-frame stream ended mid-frame')
        have += received
    return chunks


class LatestFramePublisher:
    """Asynchronously publish frames while keeping only the newest pending one."""

    def __init__(self, port):
        port = int(port)
        if not 1 <= port <= 65535:
            raise ValueError('local-frame port must be between 1 and 65535')
        self.address = ('127.0.0.1', port)
        self._condition = threading.Condition()
        self._pending = None
        self._closed = False
        self._socket = None
        self._thread = threading.Thread(
            target=self._run, name='local-frame-publisher', daemon=True)
        self._thread.start()

    def publish(self, frame):
        pixels = np.asarray(frame)
        if (pixels.dtype != np.uint8 or pixels.ndim != 3 or
                pixels.shape[2] != 3):
            raise ValueError('local frames must be HxWx3 uint8 RGB arrays')
        height, width = pixels.shape[:2]
        if (height < 1 or width < 1 or
                height * width * 3 > _MAX_FRAME_BYTES):
            raise ValueError('local frame dimensions are outside supported limits')
        pixels = np.ascontiguousarray(pixels)
        with self._condition:
            if self._closed:
                return
            self._pending = pixels
            self._condition.notify()

    def _connect(self):
        while not self._closed:
            try:
                connection = socket.create_connection(self.address, timeout=.5)
                connection.settimeout(1.0)
                self._socket = connection
                return connection
            except OSError:
                with self._condition:
                    self._condition.wait(.1)
        return None

    def _run(self):
        connection = None
        while True:
            with self._condition:
                while self._pending is None and not self._closed:
                    self._condition.wait()
                if self._closed:
                    break
                pixels = self._pending
                self._pending = None
            if connection is None:
                connection = self._connect()
                if connection is None:
                    break
            try:
                write_frame(connection, pixels)
            except (OSError, ValueError):
                try:
                    connection.close()
                except OSError:
                    pass
                connection = None
                with self._condition:
                    if not self._closed and self._pending is None:
                        self._pending = pixels
                        self._condition.notify()
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass

    def close(self):
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()
        connection = self._socket
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                connection.close()
            except OSError:
                pass
        self._thread.join(timeout=1.5)
