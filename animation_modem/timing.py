"""Shared-wall-time metadata and bounded presentation scheduling (no new clock)."""
from collections import deque
import statistics
import threading


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
