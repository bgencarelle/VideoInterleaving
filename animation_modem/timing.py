"""Shared-wall-time metadata and bounded presentation scheduling (no new clock)."""
from collections import deque
import statistics
import threading
import time


def expand_timestamp(ms32, now_ns):
    """Nearest epoch for milliseconds modulo 2**32; live age must be <24.85 days."""
    now_ms = now_ns//1_000_000
    delta = ((int(ms32) - now_ms + (1 << 31)) & 0xffffffff) - (1 << 31)
    return (now_ms + delta)*1_000_000


class TimingStats:
    def __init__(self):self.errors=deque(maxlen=60)
    def record(self, error_ms):
        self.errors.append(error_ms)
        ordered=sorted(self.errors)
        return {'median_ms':statistics.median(ordered),
                'p95_ms':ordered[min(len(ordered)-1,int(len(ordered)*.95))]}


class PresentationBuffer:
    """Keep early frames; at each GUI tick choose the newest frame already due."""
    def __init__(self, capacity=64):
        self.items=[]
        self.capacity=capacity
        self.dropped=0
        self.lock=threading.Lock()
    def put(self, value, due_ns):
        with self.lock:
            self.items.append((due_ns,value))
            self.items.sort(key=lambda item:item[0])
            if len(self.items)>self.capacity:
                self.items.pop(0)
                self.dropped+=1
    def pop_due(self, now_ns):
        with self.lock:
            n=0
            while n<len(self.items) and self.items[n][0]<=now_ns:n+=1
            if not n:return None
            value=self.items[n-1][1]
            self.dropped+=n-1
            del self.items[:n]
            return value


class ProgressSummary:
    """One line per pass through the bake, instead of one per packet.

    A live receiver runs for hours at 14 packets a second, so per-packet JSON
    is thousands of lines nobody reads. Silence is worse -- there is no way to
    tell a working link from a dead one -- so the default digests a whole loop
    and prints when the source index comes back to zero. That ties the cadence
    to the content rather than to a clock: one line per lap, whatever the
    frame rate or playback speed.

    A time fallback covers the case the index cannot: a link delivering
    pictures whose headers never verify has no index to loop on, and without
    it that would look exactly like a dead link.
    """

    def __init__(self, fallback_seconds=30.0):
        self.fallback_seconds = fallback_seconds
        self.deadline = None
        self.reset()

    def reset(self):
        self.packets = 0
        self.verified = 0
        self.tiers = {}
        self.speeds = []
        self.coverage = []

    def record(self, result, now=None):
        """Accumulate one result. True when a summary line is due."""
        self.packets += 1
        self.verified += getattr(result, 'identity', None) == 'verified_header'
        tier = getattr(result, 'tier', None)
        if tier:
            self.tiers[tier] = self.tiers.get(tier, 0) + 1
        rate = getattr(result, 'rate_error', None)
        if rate is not None:
            self.speeds.append(1/(1 + rate))
        if getattr(result, 'coverage', None) is not None:
            self.coverage.append(result.coverage)

        now = time.monotonic() if now is None else now
        if self.deadline is None:
            self.deadline = now + self.fallback_seconds
        index = getattr(result, 'source_index', None)
        # A lap closes on index zero, but not on the very first packet -- that
        # would report a one-packet lap before anything has been counted.
        if index == 0 and self.packets > 1:
            self.deadline = now + self.fallback_seconds
            return True
        if index is None and now >= self.deadline:
            self.deadline = now + self.fallback_seconds
            return True
        return False

    def line(self):
        if not self.packets:
            return 'no packets'
        tiers = ' '.join(f'{k} {v}' for k, v in sorted(self.tiers.items()))
        speed = statistics.median(self.speeds) if self.speeds else float('nan')
        cover = statistics.median(self.coverage) if self.coverage else float('nan')
        return (f'{self.packets} packets, {self.verified} verified, {tiers}, '
                f'speed {speed:.4f}x, coverage {cover:.2f}')


class ProgressSummary:
    """One line every `every` seconds instead of one per packet.

    A live receiver runs at 14 fps for hours; per-packet JSON is thousands of
    lines nobody reads and a real cost on the thread that must keep reading the
    input stream. Silence is worse though -- there is no way to tell a working
    link from a dead one -- so the default is a periodic digest.
    """

    def __init__(self, every=5.0):
        self.every = every
        self.reset()

    def reset(self):
        self.packets = 0
        self.verified = 0
        self.tiers = {}
        self.speeds = []
        self.coverage = []
        self.due = None

    def record(self, result):
        self.packets += 1
        self.verified += getattr(result, 'identity', None) == 'verified_header'
        tier = getattr(result, 'tier', None)
        if tier:
            self.tiers[tier] = self.tiers.get(tier, 0) + 1
        rate = getattr(result, 'rate_error', None)
        if rate is not None:
            self.speeds.append(1/(1 + rate))
        if getattr(result, 'coverage', None) is not None:
            self.coverage.append(result.coverage)

    def due_now(self, now):
        if self.due is None:
            self.due = now + self.every
            return False
        if now < self.due:
            return False
        self.due = now + self.every
        return True

    def line(self):
        if not self.packets:
            return 'no packets'
        tiers = ' '.join(f'{k} {v}' for k, v in sorted(self.tiers.items()))
        speed = statistics.median(self.speeds) if self.speeds else float('nan')
        cover = statistics.median(self.coverage) if self.coverage else float('nan')
        return (f'{self.packets} packets, {self.verified} verified, {tiers}, '
                f'speed {speed:.4f}x, coverage {cover:.2f}')
