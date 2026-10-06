"""Bounded, consumption-driven scheduling for complete scope traces."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time


@dataclass(frozen=True)
class FrameRequest:
    scope: object
    render: object
    version: object
    repeat: bool
    period: float


class FieldGroupLatch:
    """Pin a render request until all fields of its picture are accepted.

    ``begin`` may be called repeatedly when a render fails. The field position
    and pinned request advance only through ``accept``, so an empty or failed
    render cannot skip a field or pair it with a different source image.
    """

    def __init__(self, fields=1):
        self.configure(fields)

    def configure(self, fields):
        self.fields = max(1, int(fields))
        self.reset()

    def reset(self):
        self.index = 0
        self.request = None

    def begin(self, request):
        if self.index == 0 or self.request is None:
            self.request = request
        return self.index, self.request

    def accept(self):
        self.index = (self.index + 1) % self.fields
        if self.index == 0:
            self.request = None


class ScopeFrameScheduler:
    """Render at most one trace ahead, independently of the UI/input loop.

    ``publish`` replaces a single desired request. The worker waits until the
    Scope callback has consumed the pending frame (`ready()`), then renders
    under ``state_lock``. The lock is also available to the owner thread for
    short renderer reconfiguration/reset operations. A repeating request is
    used by progressive frame modes; vector requests set ``repeat=False`` and
    render once for each new version.
    """

    POLL_SECONDS = 0.001

    def __init__(self, name="scope-frame-producer", clock=None):
        self._clock = clock or time.monotonic
        self.state_lock = threading.RLock()
        self._condition = threading.Condition()
        self._request = None
        self._stopping = False
        self._busy = False
        self._epoch = 0
        self._thread = threading.Thread(
            target=self._run, name=name, daemon=True)
        self._stats = {
            "requested": 0,
            "attempted": 0,
            "rendered": 0,
            "failed": 0,
            "replaced": 0,
            "deadline_misses": 0,
            "deadline_lateness_ms": deque(maxlen=2048),
            "render_duration_ms": deque(maxlen=2048),
            "last_error": None,
        }
        self._thread.start()

    def publish(self, scope, render, *, version=None, repeat=True,
                period=None):
        """Publish the latest render state, replacing any waiting state."""
        if period is None:
            period = (float(scope.trace_samples) / max(
                float(scope.samplerate), 1.0))
        request = FrameRequest(scope, render, version, bool(repeat),
                               max(0.0, float(period)))
        with self._condition:
            if self._stopping:
                return False
            current = self._request
            if (request.version is not None and current is not None
                    and current.scope is scope
                    and current.version == request.version
                    and current.repeat == request.repeat
                    and current.period == request.period):
                # A caller may publish from a fast UI/source polling loop.
                # Identical state is already represented by the worker; do
                # not replace its callback, wake it, or count a false drop.
                return True
            if self._request is not None:
                self._stats["replaced"] += 1
            self._stats["requested"] += 1
            self._request = request
            self._condition.notify_all()
        return True

    def clear(self, *, wait=False):
        """Withdraw pending work; optionally wait for an in-flight render."""
        with self._condition:
            self._request = None
            self._epoch += 1
            self._condition.notify_all()
            while wait and self._busy:
                self._condition.wait()

    def snapshot(self):
        """Return bounded producer diagnostics without exposing mutable lists."""
        with self._condition:
            lateness = tuple(self._stats["deadline_lateness_ms"])
            duration = tuple(self._stats["render_duration_ms"])
            return {
                "requested": self._stats["requested"],
                "attempted": self._stats["attempted"],
                "rendered": self._stats["rendered"],
                "failed": self._stats["failed"],
                "replaced": self._stats["replaced"],
                "deadline_misses": self._stats["deadline_misses"],
                "deadline_lateness_ms": lateness,
                "render_duration_ms": duration,
                "last_error": self._stats["last_error"],
                "busy": self._busy,
                "pending": self._request is not None,
            }

    def close(self, timeout=2.0):
        """Stop scheduling and join the worker before its Scope is closed."""
        with self._condition:
            self._stopping = True
            self._request = None
            self._condition.notify_all()
        self._thread.join(timeout=max(0.0, float(timeout)))
        return not self._thread.is_alive()

    def _wait(self, timeout):
        with self._condition:
            if not self._stopping:
                self._condition.wait(timeout=max(0.0001, float(timeout)))
            return self._stopping

    def _run(self):
        active_scope = None
        active_epoch = -1
        last_version = object()
        next_deadline = None
        while True:
            with self._condition:
                while self._request is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                request = self._request
                epoch = self._epoch

            if epoch != active_epoch:
                active_epoch = epoch
                active_scope = None
                last_version = object()
                next_deadline = None

            if request.scope is not active_scope:
                active_scope = request.scope
                last_version = object()
                next_deadline = None

            if not request.repeat and request.version == last_version:
                if self._wait(self.POLL_SECONDS):
                    return
                continue

            try:
                ready = bool(request.scope.ready())
            except Exception as exc:
                self._record_failure(exc)
                if self._wait(self.POLL_SECONDS):
                    return
                continue

            now = self._clock()
            if not ready:
                wait = self.POLL_SECONDS
                if next_deadline is not None:
                    wait = min(wait, max(0.0001, next_deadline - now))
                if self._wait(wait):
                    return
                continue

            if next_deadline is not None and now < next_deadline:
                if self._wait(min(self.POLL_SECONDS, next_deadline - now)):
                    return
                continue

            with self._condition:
                if self._stopping:
                    return
                if self._request is not request:
                    continue
                self._busy = True
                self._stats["attempted"] += 1

            started = self._clock()
            lateness_ms = (max(0.0, started - next_deadline) * 1000.0
                           if next_deadline is not None else 0.0)
            try:
                with self.state_lock:
                    request.render()
                finished = self._clock()
                last_version = request.version
                # Anchor the next deadline to this observed ready/consumption
                # boundary. Rendering is deliberately done in the time before
                # that next DAC boundary; anchoring to render completion would
                # add render time to every trace and force a repeated frame.
                next_deadline = started + request.period
                self._record_success(lateness_ms,
                                     (finished - started) * 1000.0)
            except Exception as exc:
                finished = self._clock()
                next_deadline = started + request.period
                self._record_failure(exc)
            finally:
                with self._condition:
                    self._busy = False
                    self._condition.notify_all()

    def _record_success(self, lateness_ms, duration_ms):
        with self._condition:
            self._stats["rendered"] += 1
            if lateness_ms > 0.0:
                self._stats["deadline_misses"] += 1
            self._stats["deadline_lateness_ms"].append(float(lateness_ms))
            self._stats["render_duration_ms"].append(float(duration_ms))

    def _record_failure(self, exc):
        with self._condition:
            self._stats["failed"] += 1
            self._stats["last_error"] = f"{type(exc).__name__}: {exc}"
