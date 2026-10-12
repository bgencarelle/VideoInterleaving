"""Small loopback bridge for publishing the app's rendered local-mode frames.

This module is deliberately independent of both the UI and modem packages so
the local display can publish its output without importing the sender stack.
"""
import socket
import struct
import threading

import numpy as np


_MAGIC = b'VIF1'
# VIF2: the same frame plus which image it shows and the app's clock, so a
# modem sender can stamp packets with the real image index and loop.
_MAGIC_INFO = b'VIF2'
_HEADER = struct.Struct('!4sII')
# index, frames in the loop, ping-pong, clocked, launch time ns, shown at ns
_INFO = struct.Struct('!iI??qq')
_MAX_FRAME_BYTES = 64 * 1024 * 1024


def read_frame(sock):
    """Read one RGB frame from a connected socket; return None on clean EOF."""
    result = read_frame_info(sock)
    return None if result is None else result[0]


def read_frame_info(sock):
    """(frame, info) from a connected socket, or None on clean EOF. ``info``
    is None, or a dict: index, frames, pingpong, clocked, launch_ns,
    shown_ns (see write_frame)."""
    header = _read_exact(sock, _HEADER.size, allow_eof=True)
    if header is None:
        return None
    magic, width, height = _HEADER.unpack(header)
    size = width * height * 3
    if (magic not in (_MAGIC, _MAGIC_INFO) or width < 1 or height < 1 or
            size > _MAX_FRAME_BYTES):
        raise ValueError('invalid local-frame header')
    info = None
    if magic == _MAGIC_INFO:
        fields = _INFO.unpack(bytes(_read_exact(sock, _INFO.size)))
        info = dict(zip(('index', 'frames', 'pingpong', 'clocked',
                         'launch_ns', 'shown_ns'), fields))
    data = _read_exact(sock, size)
    return np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3), info


def write_frame(sock, frame, info=None):
    """Write one contiguous uint8 RGB frame to a connected socket, with
    ``info`` (index, frames, pingpong, clocked, launch_ns, shown_ns) when
    given."""
    pixels = np.asarray(frame)
    if (pixels.dtype != np.uint8 or pixels.ndim != 3 or
            pixels.shape[2] != 3):
        raise ValueError('local frames must be HxWx3 uint8 RGB arrays')
    height, width = pixels.shape[:2]
    size = width * height * 3
    if width < 1 or height < 1 or size > _MAX_FRAME_BYTES:
        raise ValueError('local frame dimensions are outside supported limits')
    pixels = np.ascontiguousarray(pixels)
    if info is None:
        sock.sendall(_HEADER.pack(_MAGIC, width, height))
    else:
        sock.sendall(_HEADER.pack(_MAGIC_INFO, width, height) + _INFO.pack(
            int(info['index']), int(info['frames']), bool(info['pingpong']),
            bool(info['clocked']), int(info['launch_ns']),
            int(info['shown_ns'])))
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

    def publish(self, frame, info=None):
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
            self._pending = (pixels, info)
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
                pending = self._pending
                pixels, info = pending
                self._pending = None
            if connection is None:
                connection = self._connect()
                if connection is None:
                    break
            try:
                write_frame(connection, pixels, info)
            except (OSError, ValueError):
                try:
                    connection.close()
                except OSError:
                    pass
                connection = None
                with self._condition:
                    if not self._closed and self._pending is None:
                        self._pending = pending
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
