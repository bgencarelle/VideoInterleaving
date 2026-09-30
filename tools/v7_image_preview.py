"""Non-blocking source/resized-frame preview for the V7 sender GUI."""
from io import BytesIO
import queue
import socket
import threading
import time

import numpy as np
from PIL import Image, ImageOps

from tools.v7_preview_protocol import pack_preview_datagram


_MAX_JPEG_BYTES = 60_000
_THUMBNAIL_SIZE = (256, 320)


def _preview_jpeg(frame):
    if isinstance(frame, Image.Image):
        image = frame.convert('RGB')
    else:
        image = Image.fromarray(np.asarray(frame)).convert('RGB')
    image = ImageOps.contain(
        image, _THUMBNAIL_SIZE, method=Image.Resampling.LANCZOS)
    for quality in (80, 68):
        output = BytesIO()
        image.save(output, format='JPEG', quality=quality, optimize=False)
        jpeg = output.getvalue()
        if len(jpeg) <= _MAX_JPEG_BYTES:
            return jpeg
    image.thumbnail((192, 240), Image.Resampling.BILINEAR)
    output = BytesIO()
    image.save(output, format='JPEG', quality=60, optimize=False)
    return output.getvalue()


class ImagePreviewWorker:
    """Publish both stages of only the newest frame to loopback."""

    def __init__(self, port):
        self.destination = ('127.0.0.1', int(port))
        self.jobs = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.generation = 0
        self.closed = False
        self.thread = threading.Thread(
            target=self._run, name='v7-image-preview', daemon=True)

    def start(self):
        self.thread.start()

    def submit(self, counter, aspect, source, resized, handoff_ns=None):
        """Queue both image stages after handoff without waiting for rendering."""
        if self.closed:
            return
        try:
            item = (int(counter), int(aspect), source, resized,
                    time.monotonic_ns() if handoff_ns is None else int(handoff_ns))
            with self.lock:
                self.generation += 1
                generation = self.generation
            job = (generation, item)
            try:
                self.jobs.put_nowait(job)
            except queue.Full:
                try:
                    self.jobs.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.jobs.put_nowait(job)
                except queue.Full:
                    pass
        except Exception:
            return

    def _run(self):
        publisher = None
        try:
            publisher = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            publisher.setblocking(False)
            while not self.stop.is_set():
                try:
                    generation, item = self.jobs.get(timeout=.1)
                except queue.Empty:
                    continue
                try:
                    counter, aspect, source, resized, handoff_ns = item
                    for stage, frame in (('source', source),
                                         ('resized', resized)):
                        jpeg = _preview_jpeg(frame)
                        with self.lock:
                            if generation != self.generation:
                                break
                        message = pack_preview_datagram(
                            counter, aspect, handoff_ns, stage, jpeg)
                        publisher.sendto(message, self.destination)
                except Exception:
                    continue
        except OSError:
            return
        finally:
            if publisher is not None:
                publisher.close()

    def close(self, timeout=1.0):
        if self.closed:
            return
        self.closed = True
        self.stop.set()
        with self.lock:
            self.generation += 1
        if self.thread.is_alive():
            self.thread.join(timeout=max(0.0, float(timeout)))
