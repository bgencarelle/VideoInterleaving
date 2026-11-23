"""Display-neutral handoff for the standalone V7 receiver.

The decoder publishes only the newest usable frame.  A future main-app sink
can consume this object from its UI/render thread and upload the pixels to the
existing ModernGL renderer without making the transport import display code.
"""
from dataclasses import dataclass
import threading
import time


@dataclass(frozen=True)
class DisplayFrame:
    values: object
    shapes: object
    aspect: int
    generation: int
    published_at: float
    # ((rows, cols) luma, chroma, chroma) when the picture was sent on a
    # pixel grid: what the Pixel display shows instead of half the planes.
    pixel_shapes: object = None
    # What the decoder knows about the luma coefficients, when it says: a
    # luma-grid array, infinite where a coefficient was not sent, otherwise
    # how far the true value may lie from the decoded one (see
    # animation_modem.v7_dct_display.edge_consistent_plane).
    luma_room: object = None


class LatestFrame:
    """Thread-safe newest-frame mailbox; no queue and no frame backlog."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._generation = 0
        self._notifier = None

    def set_notifier(self, notifier):
        """Set an optional thread-safe wake callback for a waiting viewer."""
        with self._lock:
            self._notifier = notifier

    def publish(self, values, shapes, aspect=0, pixel_shapes=None, luma_room=None):
        """Publish a frame reference and discard any older unpublished frame."""
        with self._lock:
            self._generation += 1
            self._frame = DisplayFrame(values, shapes, int(aspect),
                                       self._generation, time.monotonic(),
                                       pixel_shapes, luma_room)
            notifier = self._notifier
        if notifier is not None:
            try:
                notifier()
            except Exception:
                # A closing viewer must not break the decoder's publication path.
                pass

    def snapshot(self):
        """Return the newest frame, or ``None`` before the first decode."""
        with self._lock:
            return self._frame
