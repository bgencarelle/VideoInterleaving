"""Choose complete frames or deadline previews without changing decode work."""
import time
from .timing import PresentationBuffer


class DeadlinePresentationBuffer(PresentationBuffer):
    """Coalesce refinements until their presentation deadline.

    Shared-time targets retain their existing authority. Otherwise, allow a
    complete picture through immediately and hold partials until half the last
    completed packet's received duration, anchored to this packet's sample age.
    The GUI polls normally, so a truncated packet can appear even in silence.
    """
    def __init__(self, rate, packet_samples, value_count, capacity=64):
        super().__init__(capacity)
        self.rate = rate
        self.packet_samples = packet_samples
        self.value_count = value_count
        self._last_duration = None
        self._key = None
        self._ended = True
        self._deadline = None
        self._pending = None

    def put(self, value, due_ns, *, now_ns=None):
        if isinstance(value, str) or getattr(value, 'values', None) is None:
            # Preserve errors/status messages and the existing display hold.
            return super().put(value, due_ns)
        now_ns = time.time_ns() if now_ns is None else now_ns
        extra = value.extra
        key = (extra.get('packet_id'), value.absolute, value.stamp_ms)
        target = getattr(value, 'target_time_ns', None)
        duration = extra.get('packet_duration_samples',
                             self.packet_samples*(1+value.rate_error))/self.rate
        complete = extra.get('complete', True)
        full = complete and extra.get('received_values', self.value_count) >= self.value_count
        with self.lock:
            new_packet = self._ended or key != self._key or extra.get('frame_start', False)
            if new_packet:
                age = extra.get('packet_age_samples', 0)/self.rate
                previous = self._last_duration if self._last_duration is not None else duration
                self._deadline = now_ns + round((.5*previous-age)*1e9)
                self._key = key
                self._pending = None
                if target is None:
                    # Latest wins for unscheduled input: an older preview must
                    # never wake up after a newer frame has been displayed.
                    kept = [(due, item) for due, item in self.items
                            if isinstance(item, str) or getattr(item, 'target_time_ns', None) is not None]
                    self.dropped += len(self.items)-len(kept)
                    self.items = kept
            # One queued reconstruction per packet, including shared-time
            # frames. Completion removes an older partial's future deadline.
            if self._pending is not None:
                kept = [(due, item) for due, item in self.items if item is not self._pending]
                self.dropped += len(self.items)-len(kept)
                self.items = kept
            deadline = target if target is not None else now_ns if full else self._deadline
            self.items.append((deadline, value))
            self.items.sort(key=lambda item: item[0])
            if len(self.items) > self.capacity:
                self.items.pop(0)
                self.dropped += 1
            self._pending = value
            self._ended = complete
            if complete:
                self._last_duration = duration
