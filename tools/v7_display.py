"""Display-neutral handoff for the standalone V7 receiver.

The decoder publishes only the newest usable frame.  A future main-app sink
can consume this object from its UI/render thread and upload the pixels to the
existing ModernGL renderer without making the transport import display code.
"""
from dataclasses import dataclass
import threading


@dataclass(frozen=True)
class DisplayFrame:
    values: object
    shapes: object
    aspect: int
    generation: int


class LatestFrame:
    """Thread-safe newest-frame mailbox; no queue and no frame backlog."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._generation = 0

    def publish(self, values, shapes, aspect=0):
        """Publish a frame reference and discard any older unpublished frame."""
        with self._lock:
            self._generation += 1
            self._frame = DisplayFrame(values, shapes, int(aspect),
                                       self._generation)

    def snapshot(self):
        """Return the newest frame, or ``None`` before the first decode."""
        with self._lock:
            return self._frame
