"""Non-blocking source/resized-frame preview for the V7 sender GUI."""
import errno
import json
import queue
import socket
import threading
import time
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps

from tools.v7_preview_protocol import HEADER, pack_preview_datagram


# One preview picture is one loopback UDP datagram. macOS limits a UDP
# datagram to net.inet.udp.maxdgram, 9216 bytes by default, and rejects a
# larger one with EMSGSIZE ("Message too long", errno 40); Linux allows far
# more, which hid this. Stay under the macOS limit, header included.
_MAX_DATAGRAM_BYTES = 8*1024
_MAX_JPEG_BYTES = _MAX_DATAGRAM_BYTES-HEADER.size
_THUMBNAIL_SIZE = (256, 320)


def _preview_jpeg(frame):
    if isinstance(frame, Image.Image):
        image = frame.convert('RGB')
    else:
        image = Image.fromarray(np.asarray(frame)).convert('RGB')
    image = ImageOps.contain(
        image, _THUMBNAIL_SIZE, method=Image.Resampling.LANCZOS)
    while True:
        for quality in (80, 68, 56, 44, 32):
            output = BytesIO()
            image.save(output, format='JPEG', quality=quality, optimize=False)
            jpeg = output.getvalue()
            if len(jpeg) <= _MAX_JPEG_BYTES:
                return jpeg
        if image.width == 1 and image.height == 1:
            # Even this minimal JPEG is far below the production datagram cap.
            return jpeg
        reduced_size = (max(1, int(image.width*.75)),
                        max(1, int(image.height*.75)))
        image = image.resize(reduced_size, Image.Resampling.BILINEAR)


class ImagePreviewWorker:
    """Publish both stages of only the newest frame to loopback."""

    def __init__(self, port):
        self.destination = ('127.0.0.1', int(port))
        self.jobs = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.generation = 0
        self.closed = False
        self.error_reported = False
        self.thread = threading.Thread(
            target=self._run, name='v7-image-preview', daemon=True)

    def start(self):
        self.thread.start()

    def submit(self, counter, aspect, source, resized, handoff_ns=None):
        """Queue both image stages after handoff without waiting for rendering."""
        try:
            item = (int(counter), int(aspect), source, resized,
                    time.monotonic_ns() if handoff_ns is None else int(handoff_ns))
            with self.lock:
                if self.closed:
                    return
                self.generation += 1
                generation = self.generation
                try:
                    self.jobs.put_nowait((generation, item))
                except queue.Full:
                    try:
                        self.jobs.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self.jobs.put_nowait((generation, item))
                    except queue.Full:
                        pass
        except Exception:
            return

    def _report_error(self, exc, context='preview worker'):
        if self.error_reported:
            return
        self.error_reported = True
        code = getattr(exc, 'errno', None)
        code_name = errno.errorcode.get(code, '') if code is not None else ''
        detail = f'{context}: {type(exc).__name__}: {exc}'
        if code is not None:
            detail += f' (errno {code}{": "+code_name if code_name else ""})'
        try:
            print(json.dumps({
                'status': 'image_preview_error',
                'message': detail,
            }), flush=True)
        except Exception:
            pass

    def _run(self):
        publisher = None
        try:
            publisher = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            publisher.setblocking(False)
            while True:
                try:
                    generation, item = self.jobs.get(timeout=.1)
                except queue.Empty:
                    if self.stop.is_set():
                        break
                    continue
                try:
                    counter, aspect, source, resized, handoff_ns = item
                    for stage, frame in (('source', source),
                                         ('resized', resized)):
                        try:
                            jpeg = _preview_jpeg(frame)
                        except Exception as exc:
                            self._report_error(exc, f'rendering {stage} image')
                            continue
                        with self.lock:
                            if generation != self.generation:
                                break
                        message = pack_preview_datagram(
                            counter, aspect, handoff_ns, stage, jpeg)
                        try:
                            publisher.sendto(message, self.destination)
                        except OSError as exc:
                            self._report_error(exc, f'sending {stage} datagram')
                except Exception as exc:
                    self._report_error(exc, 'building preview datagram')
                    continue
        except OSError as exc:
            self._report_error(exc, 'opening preview socket')
            return
        finally:
            if publisher is not None:
                publisher.close()

    def close(self, timeout=1.0):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.stop.set()
        if self.thread.is_alive():
            self.thread.join(timeout=max(0.0, float(timeout)))
