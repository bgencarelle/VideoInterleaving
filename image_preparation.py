"""Bounded picture preparation for clock-owned presentation deadlines.

This prepares data; it never advances a playback clock or selects a fallback.
All public operations run on the owning display/source thread.
"""
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import math
import time


def advance_tick(previous, period, now, *, swap_consumed=False):
    """Keep the existing tick grid without sleeping twice for a blocking swap."""
    deadline = previous+period
    remaining = deadline-now
    if remaining < -.001:
        elapsed = int((-remaining)//period)
        # A blocking local swap has already paced this iteration. Poll the
        # current clock next; never wait another refresh or replay old ticks.
        deadline += (elapsed if swap_consumed else elapsed+1)*period
    return deadline, max(0.0, deadline-now)


@dataclass
class _Picture:
    future: object
    deadline: float
    result: object = None
    ready: bool = False


class DeadlinePreparer:
    def __init__(self, make, *, workers=2, capacity=4, period=1/30,
                 clock=time.monotonic, on_error=None, executor=None,
                 with_submission_time=False):
        self.make = make
        self.workers = workers
        self.capacity = max(workers, int(capacity))
        self.period = float(period)
        self.clock = clock
        self.anchor = clock()
        self.on_error = on_error
        self.with_submission_time = with_submission_time
        self._pool = executor or ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix='picture-prepare')
        self._jobs = {}
        self._retired = []
        self._failed_until = {}
        self._durations = deque(maxlen=24)
        self._folders = None
        self._render_seconds = 0.0
        self.stats = dict(requested=0, completed=0, expired=0, cancelled=0,
                          hits=0, misses=0, capacity_skips=0, peak_pairs=0)

    @property
    def lead(self):
        # Quick response to a slow job; a finite history allows recovery.
        duration = 0.0
        if self._durations:
            ordered = sorted(self._durations)
            duration = ordered[min(len(ordered)-1, int(.9*len(ordered)))]
        return max(self.period, .025, duration*1.25 + .005) + self._render_seconds

    def record_render(self, seconds):
        self._render_seconds = max(float(seconds), self._render_seconds*.95)

    def seed(self, index, folders, result):
        """Adopt the synchronously loaded startup picture, including held MIDI."""
        folders = tuple(folders)
        future = Future()
        future.set_result((result, 0.0))
        self._folders = folders
        self._jobs[(int(index), folders)] = _Picture(
            future, self.clock(), result=result, ready=True)

    def _prepare(self, key, submitted):
        result = (self.make(*key, submitted) if self.with_submission_time else
                  self.make(*key))
        return result, self.clock()-submitted

    def _collect(self):
        retired = []
        for job in self._retired:
            if not job.future.done():
                retired.append(job)
                continue
            try:
                _, duration = job.future.result()
            except Exception:
                pass
            else:
                # Expired work still measures the storage/CPU conditions. If
                # only on-time jobs trained lead, a slowdown could never grow it.
                self._durations.append(duration)
                self.stats['completed'] += 1
        self._retired = retired
        for key, job in tuple(self._jobs.items()):
            if job.ready or not job.future.done():
                continue
            try:
                job.result, duration = job.future.result()
            except Exception as exc:
                self._jobs.pop(key)
                self._failed_until[key] = self.clock()+1.0
                while len(self._failed_until) > self.capacity*2:
                    self._failed_until.pop(next(iter(self._failed_until)))
                if self.on_error is not None:
                    self.on_error(key, exc)
            else:
                job.ready = True
                self._durations.append(duration)
                self.stats['completed'] += 1

    def _remove(self, key):
        job = self._jobs.pop(key)
        if not job.future.done():
            if job.future.cancel():
                self.stats['cancelled'] += 1
            else:
                # Running jobs still count against capacity until they finish.
                self._retired.append(job)
        self.stats['expired'] += 1

    def plan(self, folders, current_index, index_at=None, *, now=None):
        """Reserve exact future ticks, or the latest held MIDI identity.

        ``index_at`` takes a monotonic presentation deadline. No future folder
        selector is executed: a real folder change invalidates old preparation.
        """
        now = self.clock() if now is None else now
        folders = tuple(folders)
        self._collect()
        self._failed_until = {key:until for key,until in self._failed_until.items()
                              if until > now}
        if folders != self._folders:
            for key in tuple(self._jobs):
                self._remove(key)
            self._folders = folders
        current = (int(current_index), folders)
        for key, job in tuple(self._jobs.items()):
            if job.deadline + self.period < now and key != current:
                self._remove(key)
        if index_at is None:
            targets = [(current_index, now+self.lead)]
            for key in tuple(self._jobs):
                if key != current:
                    self._remove(key)
        else:
            first = math.ceil((now+self.lead-self.anchor)/self.period)
            # A moving preparation window, not a FIFO of adjacent source frames.
            targets = [(index_at(self.anchor+tick*self.period),
                        self.anchor+tick*self.period)
                       for tick in (first, first+1)]
        for index, deadline in targets:
            key = (int(index), folders)
            if key in self._failed_until:
                continue
            if key in self._jobs:
                self._jobs[key].deadline = max(self._jobs[key].deadline, deadline)
                continue
            outstanding = sum(not j.ready for j in self._jobs.values())+len(self._retired)
            if (len(self._jobs)+len(self._retired) >= self.capacity
                    or outstanding >= self.workers):
                self.stats['capacity_skips'] += 1
                continue
            self._jobs[key] = _Picture(
                self._pool.submit(self._prepare, key, self.clock()), deadline)
            self.stats['requested'] += 1
            self.stats['peak_pairs'] = max(self.stats['peak_pairs'],
                                           len(self._jobs)+len(self._retired))

    def get(self, index, folders):
        self._collect()
        job = self._jobs.get((int(index), tuple(folders)))
        if job is not None and job.ready:
            self.stats['hits'] += 1
            return job.result
        self.stats['misses'] += 1
        return None

    def upcoming(self, now=None):
        self._collect()
        now = self.clock() if now is None else now
        return sorted(((key, job.result, job.deadline)
                       for key, job in self._jobs.items()
                       if job.ready and job.deadline > now), key=lambda item:item[2])

    def snapshot(self):
        return dict(self.stats, pairs=len(self._jobs),
                    ready=sum(j.ready for j in self._jobs.values()),
                    pending=sum(not j.ready for j in self._jobs.values())+len(self._retired),
                    lead_ms=self.lead*1000)

    def close(self):
        self._pool.shutdown(wait=True, cancel_futures=True)
        self._jobs.clear()
        self._retired.clear()
        self._failed_until.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
