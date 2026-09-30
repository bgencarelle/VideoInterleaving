"""Non-blocking reconstruction previews for the standalone V7 sender."""
from collections import deque
import io
import queue
import socket
import threading
import time

import numpy as np
from PIL import Image

from animation_modem import v7
from animation_modem.imaging import values_image
from tools.v7_preview_protocol import (pack_preview_datagram,
                                       parse_preview_datagram)


_HISTORY_FRAMES = v7.TAIL_PHASES


preview_datagram = parse_preview_datagram


def _image_jpeg(image):
    output = io.BytesIO()
    image.save(output, format='JPEG', quality=82, optimize=False)
    return output.getvalue()


class EncodedFrameReconstructor:
    """Rebuild the clean-link image represented by one sender packet."""

    def __init__(self, model, fold=None, mono_wire=None):
        self.base_model = model
        self.fold = fold
        self.mono_wire = mono_wire
        self.model = (mono_wire.model_for(model) if mono_wire is not None
                      else model)
        self.tail_memory = not bool(
            getattr(mono_wire, 'disable_tail_memory', False))
        self.codec = None
        if mono_wire is not None and getattr(mono_wire, 'use_fold', False):
            self.codec = mono_wire._codec(self.model)
        elif fold is not None:
            self.codec = fold.codec(self.model)
        self.tail = v7.TailStore(enabled=self.tail_memory)

    def _encoded_coefficients(self, values):
        if self.codec is not None:
            return np.asarray(self.codec.encode_coefficients(values), dtype=float)
        return (self.base_model.coder.forward(values) /
                self.base_model.coder.gains)

    def _decode_clean_packet(self, counter, values):
        model = self.model
        encoded = self._encoded_coefficients(values)
        tail_slice = int(counter) % v7.TAIL_PHASES
        ranks = model.rank_tables[tail_slice]
        selected = ranks[ranks >= 0]
        previous = (self.tail.prior(model) if self.tail_memory else
                    model.mu.copy())
        decoded = previous.copy()
        decoded[selected] = encoded[selected]

        if self.codec is not None:
            # In the clean-link reconstruction the equalizer has unit
            # confidence and recovers the transmitted coefficient exactly.
            equalized = np.zeros_like(model.mu)
            equalized[selected] = encoded[selected]-model.mu[selected]
            confidence = np.zeros_like(model.mu)
            confidence[selected] = 1.0
            coefficients = self.codec.decode(
                decoded, equalized, confidence, fallback=True,
                metadata_confirmed=True)
            values = self.codec.grid.inverse(coefficients)
        else:
            values = v7.values_from(model, decoded)

        if self.tail_memory:
            self.tail.update(model, decoded, tail_slice)
        return values

    def reconstruct_values(self, history):
        """Reconstruct newest clean-link values after rebuilding one tail cycle."""
        self.tail.reset()
        decoded_values = None
        for counter, _aspect, values, _handoff_ns in history:
            decoded_values = self._decode_clean_packet(counter, values)
        return decoded_values

    def reconstruct(self, history):
        """Render the newest packet from at most one complete tail cycle."""
        values = self.reconstruct_values(history)
        return (None if values is None else
                values_image(values, self.model.coder.grids))


class EncodedPreviewWorker:
    """Latest-frame-only renderer and loopback UDP publisher.

    Sender-side calls only retain small frame values and replace a bounded
    queue entry. DCT reconstruction, image rendering, JPEG encoding, and socket
    output all happen on this worker thread.
    """

    def __init__(self, model, port, fold=None, mono_wire=None, history=()):
        self.model = model
        self.fold = fold
        self.mono_wire = mono_wire
        self.destination = ('127.0.0.1', int(port))
        self.history = deque(history, maxlen=_HISTORY_FRAMES)
        self.jobs = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.generation = 0
        self.closed = False
        self.thread = threading.Thread(
            target=self._run, name='v7-encoded-preview', daemon=True)

    def start(self):
        self.thread.start()

    def submit(self, counter, aspect, values, handoff_ns=None):
        """Queue a post-handoff frame without waiting for preview work."""
        if self.closed:
            return
        try:
            item = (int(counter), int(aspect), np.asarray(values),
                    (time.monotonic_ns() if handoff_ns is None else
                     int(handoff_ns)))
            self.history.append(item)
            with self.lock:
                self.generation += 1
                generation = self.generation
            job = (generation, tuple(self.history))
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
            # A preview mailbox failure must never unwind into the send loop.
            return

    def _run(self):
        publisher = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        publisher.setblocking(False)
        try:
            reconstructor = EncodedFrameReconstructor(
                self.model, self.fold, self.mono_wire)
            while not self.stop.is_set():
                try:
                    generation, history = self.jobs.get(timeout=.1)
                except queue.Empty:
                    continue
                try:
                    image = reconstructor.reconstruct(history)
                    if image is None:
                        continue
                    counter, aspect, _values, handoff_ns = history[-1]
                    jpeg = _image_jpeg(image)
                    with self.lock:
                        if generation != self.generation:
                            continue
                    message = pack_preview_datagram(
                        counter, aspect, handoff_ns, jpeg)
                    publisher.sendto(message, self.destination)
                except Exception:
                    # Preview is observational: failures are discarded rather
                    # than propagated into the audio producer/output path.
                    continue
        finally:
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


def decode_preview_image(jpeg):
    """Decode a preview JPEG for the sender GUI."""
    with Image.open(io.BytesIO(jpeg)) as image:
        image.load()
        return image.convert('RGB')
