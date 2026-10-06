"""Bounded consumption/deadline behavior for the frame producer worker."""
import threading
import time
import time

import numpy as np

from scope_frame_scheduler import ScopeFrameScheduler
from scope_frame_scheduler import FieldGroupLatch
from scope_bake import SweepSource, TraceEmitter
from scope_out import Scope


class FakeScope:
    samplerate = 1000
    trace_samples = 10

    def __init__(self, ready=True):
        self._ready = ready
        self._lock = threading.Lock()

    def ready(self):
        with self._lock:
            return self._ready

    def set_ready(self, ready):
        with self._lock:
            self._ready = ready


def _wait_for(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.001)
    return bool(predicate())


def test_waits_for_consumption_and_uses_only_latest_bounded_request():
    scope = FakeScope(ready=False)
    scheduler = ScopeFrameScheduler()
    emitted = []
    done = threading.Event()
    try:
        for value in (1, 2, 3):
            scheduler.publish(
                scope, lambda value=value: (emitted.append(value), done.set()),
                version=value, repeat=False)
        time.sleep(0.02)
        assert emitted == []

        scope.set_ready(True)
        assert done.wait(1.0)
        assert emitted == [3]
        assert scheduler.snapshot()["replaced"] >= 2
    finally:
        scheduler.close()


def test_field_group_reconfigure_discards_the_old_pinned_request():
    from scope_display import _configure_field_groups

    latch = FieldGroupLatch(3)
    mix_latch = FieldGroupLatch(3)
    old_request = object()
    new_request = object()
    assert latch.begin(old_request) == (0, old_request)
    assert mix_latch.begin(old_request) == (0, old_request)
    latch.accept()
    mix_latch.accept()
    assert latch.begin(new_request) == (1, old_request)
    assert mix_latch.begin(new_request) == (1, old_request)

    # Replacing an output stream starts a new group at field zero, with no
    # request captured from the previous stream.
    _configure_field_groups(latch, mix_latch, 2)
    assert latch.begin(new_request) == (0, new_request)
    assert mix_latch.begin(new_request) == (0, new_request)


def test_vector_request_runs_once_per_version_and_accepts_new_latest_state():
    scope = FakeScope()
    scheduler = ScopeFrameScheduler()
    emitted = []
    first = threading.Event()
    second = threading.Event()

    def publish(version, event):
        scheduler.publish(
            scope, lambda: (emitted.append(version), event.set()),
            version=version, repeat=False)

    try:
        publish("image-1", first)
        assert first.wait(1.0)
        time.sleep(0.035)
        assert emitted == ["image-1"]

        publish("image-2", second)
        assert second.wait(1.0)
        assert emitted == ["image-1", "image-2"]
    finally:
        scheduler.close()


def test_identical_published_version_is_not_replaced_or_repeated():
    scope = FakeScope()
    scheduler = ScopeFrameScheduler()
    emitted = []
    done = threading.Event()
    try:
        scheduler.publish(
            scope, lambda: (emitted.append("first"), done.set()),
            version=(1, "same"), repeat=False)
        assert done.wait(1.0)
        scheduler.publish(
            scope, lambda: emitted.append("duplicate"),
            version=(1, "same"), repeat=False)
        time.sleep(0.02)
        assert emitted == ["first"]
        stats = scheduler.snapshot()
        assert stats["replaced"] == 0
        assert stats["rendered"] == 1
        assert stats["requested"] == 1
        assert stats["attempted"] == 1
    finally:
        scheduler.close()


def test_cleared_scheduler_can_reaccept_same_vector_version():
    scope = FakeScope()
    scheduler = ScopeFrameScheduler()
    first = threading.Event()
    second = threading.Event()
    calls = []
    try:
        scheduler.publish(
            scope, lambda: (calls.append("before-clear"), first.set()),
            version="unchanged-key", repeat=False)
        assert first.wait(1.0)
        scheduler.clear(wait=True)
        scheduler.publish(
            scope, lambda: (calls.append("after-clear"), second.set()),
            version="unchanged-key", repeat=False)
        assert second.wait(1.0)
        assert calls == ["before-clear", "after-clear"]
    finally:
        scheduler.close()


def test_latest_frame_state_waits_for_consumption_before_next_render():
    scope = FakeScope()
    scheduler = ScopeFrameScheduler()
    emitted = []
    first = threading.Event()
    latest = threading.Event()

    def render(value, done):
        emitted.append(value)
        scope.set_ready(False)  # the frame is now pending at the DAC boundary
        done.set()

    try:
        scheduler.publish(scope, lambda: render(1, first),
                          version="frame-1", repeat=False)
        assert first.wait(1.0)

        # Source/UI changes while the first frame is pending; only the latest
        # state should be emitted after output consumes that frame.
        scheduler.publish(scope, lambda: render(2, threading.Event()),
                          version="frame-2", repeat=False)
        scheduler.publish(scope, lambda: render(3, latest),
                          version="frame-3", repeat=False)
        time.sleep(0.02)
        assert emitted == [1]

        scope.set_ready(True)
        assert latest.wait(1.0)
        assert emitted == [1, 3]
        assert scheduler.snapshot()["replaced"] == 2
        assert scheduler.snapshot()["requested"] == 3
        assert scheduler.snapshot()["attempted"] == 2
    finally:
        scheduler.close()


def test_live_tuning_replaces_request_without_changing_source_key():
    scope = FakeScope(ready=False)
    scheduler = ScopeFrameScheduler()
    emitted = []
    tuned = threading.Event()
    source_key = (12, 1, 3)
    try:
        scheduler.publish(
            scope, lambda: emitted.append((source_key, 2.2)),
            version=(1, source_key, ("raster", 2.2)), repeat=True)
        time.sleep(0.01)
        scheduler.publish(
            scope, lambda: (emitted.append((source_key, 1.8)), tuned.set()),
            version=(1, source_key, ("raster", 1.8)), repeat=True)
        scope.set_ready(True)
        assert tuned.wait(1.0)
        assert emitted == [(source_key, 1.8)]
        stats = scheduler.snapshot()
        assert stats["requested"] == 2
        assert stats["replaced"] == 1
    finally:
        scheduler.close()


def test_repeating_requests_follow_monotonic_trace_deadlines():
    scope = FakeScope()
    scheduler = ScopeFrameScheduler()
    emitted_at = []
    enough = threading.Event()

    def render():
        emitted_at.append(time.monotonic())
        if len(emitted_at) >= 4:
            enough.set()

    try:
        scheduler.publish(scope, render, repeat=True, period=0.025)
        assert enough.wait(1.0)
        intervals = [b - a for a, b in zip(emitted_at, emitted_at[1:])]
        assert min(intervals) >= 0.018
        stats = scheduler.snapshot()
        assert stats["rendered"] >= 4
        assert len(stats["render_duration_ms"]) >= 4
    finally:
        scheduler.close()


def test_over_budget_render_records_miss_without_catchup_burst():
    scope = FakeScope()
    scheduler = ScopeFrameScheduler()
    emitted_at = []
    enough = threading.Event()

    def slow_render():
        emitted_at.append(time.monotonic())
        time.sleep(0.035)
        if len(emitted_at) >= 3:
            enough.set()

    try:
        scheduler.publish(scope, slow_render, repeat=True, period=0.01)
        assert enough.wait(1.0)
        intervals = [b - a for a, b in zip(emitted_at, emitted_at[1:])]
        assert min(intervals) >= 0.033
        assert scheduler.snapshot()["deadline_misses"] >= 2
    finally:
        scheduler.close()


def test_delayed_consumption_rebases_deadline_without_catchup_burst():
    scope = FakeScope()
    scheduler = ScopeFrameScheduler()
    emitted_at = []
    first = threading.Event()

    def render(number, completed):
        emitted_at.append(time.monotonic())
        # Model a queued trace that remains at the DAC until the test releases
        # it; this deliberately delays the next observed consumption boundary.
        scope.set_ready(False)
        completed.set()

    try:
        scheduler.publish(scope, lambda: render(1, first), repeat=True,
                          period=0.03)
        assert first.wait(1.0)
        time.sleep(0.07)  # exceed two trace periods while output is not ready
        scope.set_ready(True)
        assert _wait_for(lambda: len(emitted_at) >= 2)
        # The third emission must be scheduled one period after the delayed
        # second boundary, rather than catching up against the original clock.
        scope.set_ready(True)
        assert _wait_for(lambda: len(emitted_at) >= 3)
        assert emitted_at[2] - emitted_at[1] >= 0.022
    finally:
        scheduler.close()


def test_deadlines_follow_virtual_consumption_clock_after_delayed_wakeup():
    class VirtualClock:
        def __init__(self):
            self.value = 0.0
            self.lock = threading.Lock()

        def __call__(self):
            with self.lock:
                return self.value

        def advance(self, amount):
            with self.lock:
                self.value += amount

    clock = VirtualClock()
    scope = FakeScope()
    scheduler = ScopeFrameScheduler(clock=clock)
    rendered = [threading.Event() for _ in range(3)]
    emitted_at = []

    def wake_worker():
        with scheduler._condition:
            scheduler._condition.notify_all()

    def render(number):
        emitted_at.append(clock())
        scope.set_ready(False)
        rendered[number].set()

    try:
        scheduler.publish(
            scope, lambda: render(len(emitted_at)), repeat=True, period=0.1)
        assert rendered[0].wait(1.0)

        # The DAC stays busy while the producer is effectively suspended for
        # ten virtual seconds. The next accepted boundary is the new anchor.
        clock.advance(10.0)
        wake_worker()
        scope.set_ready(True)
        wake_worker()
        assert rendered[1].wait(1.0), scheduler.snapshot()
        assert emitted_at == [0.0, 10.0]

        # A deadline rebased at 10.0 cannot fire early at 10.099, and fires on
        # the next wake at 10.100. This distinguishes rebasing from accumulating
        # lateness against the original 0.1 deadline.
        clock.advance(0.099)
        wake_worker()
        time.sleep(0.02)
        assert len(emitted_at) == 2
        clock.advance(0.001)
        wake_worker()
        scope.set_ready(True)
        wake_worker()
        assert rendered[2].wait(1.0)
        assert emitted_at == [0.0, 10.0, 10.1]
    finally:
        scheduler.close()


def test_alternating_raster_repeat_seam_is_bounded_in_scope_callback():
    samples = 3200
    luminance = np.ones((16, 24), dtype=np.float32)
    for border, extra in (
            (0.0, {}), (0.1, {}), (0.0, {"dc_comp": 30}),
            (0.0, {"oversample": 2})):
        emitter = TraceEmitter(
            96000, samples, grid=(16, 24), sweep="alternate",
            close_frame=True, border=border, **extra)
        frame = emitter.emit(luminance)
        next_frame = emitter.emit(luminance)
        np.testing.assert_allclose(next_frame[0], frame[-1], atol=1e-7)

        scope = Scope(device="null", samplerate=96000, samples=samples,
                      trigger=False, channel_pair=(24, 25), invert_y=False)
        try:
            scope._frame = frame
            scope._pending = None
            scope._pos = 0
            output = np.zeros((len(frame) + 1, scope.output_channels),
                              dtype=np.float32)
            scope._callback(output, len(output), None, None)
            xy = output[:, list(scope.channel_indices)]
            seam = float(np.linalg.norm(xy[len(frame)] - xy[len(frame) - 1]))
            frame_steps = np.linalg.norm(np.diff(frame, axis=0), axis=1)
            assert seam <= 1.25 * float(frame_steps.max()) + 1e-6
        finally:
            scope.stream.close()

        scope = Scope(device="null", samplerate=96000, samples=samples,
                      trigger=False, channel_pair=(24, 25), invert_y=False)
        try:
            scope._frame, scope._pending, scope._pos = frame, next_frame, 0
            output = np.zeros((len(frame) + 1, scope.output_channels),
                              dtype=np.float32)
            scope._callback(output, len(output), None, None)
            xy = output[:, list(scope.channel_indices)]
            handoff = float(np.linalg.norm(xy[len(frame)] - xy[len(frame) - 1]))
            assert handoff <= 1e-7
        finally:
            scope.stream.close()


def test_scope_postfilter_handoff_keeps_compensated_loop_continuous():
    from scope_display import _emit

    class FixtureLibrary:
        def __len__(self):
            return 1

        def thumb(self, _index):
            return np.full((16, 24, 2), 255, dtype=np.uint8)

    samples = 3200
    scope = Scope(device="null", samplerate=96000, samples=samples,
                  trigger=False, lowpass_hz=None, rotation=90, mirror=True,
                  channel_pair=(24, 25), invert_y=False)
    emitter = TraceEmitter(
        96000, samples, grid=(16, 24), sweep="alternate", close_frame=True,
        dc_comp=30, oversample=2)
    try:
        beam_end = _emit(
            scope, FixtureLibrary(), None, 0, "raster", {}, "alternate",
            2.2, 0.02, 1.0, None, 0.02, lowpass=12000,
            emitter=emitter)
        first = scope._pending.copy()
        scope._pending = None
        beam_end = _emit(
            scope, FixtureLibrary(), None, 0, "raster", {}, "alternate",
            2.2, 0.02, 1.0, None, 0.02, lowpass=12000,
            emitter=emitter, beam_start=beam_end)
        second = scope._pending.copy()
        assert np.isfinite(beam_end).all()
        assert np.linalg.norm(second[0] - first[-1]) <= 1e-6
        assert np.linalg.norm(second[-1] - second[0]) < 0.1
        assert np.linalg.norm(second[1] - second[0]) < 0.1

        scope._frame, scope._pending, scope._pos = first, second, 0
        output = np.zeros((samples + 1, scope.output_channels), np.float32)
        scope._callback(output, len(output), None, None)
        xy = output[:, list(scope.channel_indices)]
        assert np.linalg.norm(xy[samples] - xy[samples - 1]) <= 1e-6
    finally:
        scope.stream.close()


def test_unaccepted_raster_candidate_does_not_advance_field_or_sweep():
    lum = np.ones((16, 24), dtype=np.float32)
    emitter = TraceEmitter(
        96000, 3200, grid=(16, 24), fields=2,
        sweep="alternate", close_frame=True)

    first = emitter.emit(lum, field=0, commit=False)
    assert emitter._field == 0
    assert emitter._rev is False
    assert emitter._end is None

    retried = emitter.emit(lum, field=0, commit=False)
    np.testing.assert_array_equal(retried, first)
    assert emitter._field == 0
    assert emitter._rev is False
    assert emitter._end is None

    emitter.accept(first[-1])
    assert emitter._field == 1
    assert emitter._rev is True
    np.testing.assert_array_equal(emitter._end, first[-1])

    second = emitter.emit(
        lum, field=1, commit=False, start=emitter._end)
    assert emitter._field == 1
    assert emitter._rev is True
    assert not np.array_equal(first, second)


def test_standalone_candidate_commits_only_after_scope_accepts_it():
    from tools.scope_screen import _queue_raster_candidate

    class QueueScope:
        def __init__(self):
            self.frames_accepted = 0
            self.accept = False

        def show_frame(self, frame, handoff=None):
            if not self.accept:
                return None
            self.frames_accepted += 1
            return np.asarray(frame[-1], dtype=np.float32).copy()

    lum = np.ones((16, 24), dtype=np.float32)
    emitter = TraceEmitter(
        96000, 3200, grid=(16, 24), fields=2,
        sweep="alternate", close_frame=True)
    scope = QueueScope()
    frame = emitter.emit(lum, commit=False)

    assert not _queue_raster_candidate(scope, emitter, frame)
    assert (emitter._field, emitter._rev, emitter._end) == (0, False, None)

    scope.accept = True
    assert _queue_raster_candidate(scope, emitter, frame)
    assert emitter._field == 1
    assert emitter._rev is True
    np.testing.assert_array_equal(emitter._end, frame[-1])


def test_scope_internal_lowpass_returns_chained_output_endpoint():
    samples = 3200
    luminance = np.ones((16, 24), dtype=np.float32)
    scope = Scope(device="null", samplerate=96000, samples=samples,
                  trigger=False, lowpass_hz=12000, rotation=90, mirror=True,
                  channel_pair=(24, 25), invert_y=False)
    emitter = TraceEmitter(
        96000, samples, grid=(16, 24), sweep="alternate", close_frame=True,
        dc_comp=30, oversample=2)
    try:
        first = emitter.emit(luminance)
        first_end = scope.show_frame(first)
        first_output = scope._pending.copy()
        scope._pending = None

        emitter._end = first_end.copy()
        second = emitter.emit(luminance)
        second_end = scope.show_frame(second, handoff=first_end)
        second_output = scope._pending.copy()
        assert np.isfinite(second_end).all()
        assert np.linalg.norm(second_output[-1] - second_output[0]) < 0.1
        assert np.linalg.norm(second_output[1] - second_output[0]) < 0.1

        scope._frame, scope._pending, scope._pos = first_output, second_output, 0
        output = np.zeros((samples + 1, scope.output_channels), np.float32)
        scope._callback(output, len(output), None, None)
        xy = output[:, list(scope.channel_indices)]
        assert np.linalg.norm(xy[samples] - xy[samples - 1]) <= 1e-6
    finally:
        scope.stream.close()


def test_standalone_interlaced_capture_and_levels_stay_pinned_through_retry():
    from tools.scope_screen import (_begin_raster_field,
                                    _queue_raster_candidate)

    class QueueScope:
        def __init__(self):
            self.frames_accepted = 0
            self.accept = False

        def show_frame(self, frame, handoff=None):
            if not self.accept:
                return None
            self.frames_accepted += 1
            return np.asarray(frame[-1], dtype=np.float32).copy()

    images = [np.full((16, 24), value, dtype=np.float32)
              for value in (0.2, 0.4, 0.8)]
    grabs = []
    level_calls = []
    adaptive_state = {"lo": None, "hi": None}

    def grab():
        image = images[len(grabs)]
        grabs.append(image)
        return image

    def levels_for(image):
        value = float(image[0, 0])
        level_calls.append(value)
        target_hi = value * 2.0
        if adaptive_state["hi"] is None:
            proposed_hi = target_hi
        else:
            proposed_hi = adaptive_state["hi"] + 0.5 * (
                target_hi - adaptive_state["hi"])
        proposed = {"lo": 0.0, "hi": proposed_hi}
        return (proposed["lo"], proposed["hi"]), proposed

    groups = FieldGroupLatch(2)
    emitter = TraceEmitter(
        96000, 3200, grid=(16, 24), fields=2,
        sweep="alternate", close_frame=True)
    scope = QueueScope()

    # A rejected first candidate is not output state, so the next attempt may
    # use the latest source. Once field zero is accepted, both source and levels
    # are held until field one is accepted.
    field, captured = _begin_raster_field(groups, grab, levels_for)
    first = emitter.emit(captured["lum"], levels=captured["levels"],
                         field=field, commit=False)
    assert not _queue_raster_candidate(
        scope, emitter, first, field_group=groups,
        levels_commit=(adaptive_state, captured["level_state"]))
    assert groups.index == emitter._field == 0
    assert adaptive_state == {"lo": None, "hi": None}

    field, captured = _begin_raster_field(groups, grab, levels_for)
    accepted_picture = captured
    assert np.isclose(float(captured["lum"][0, 0]), 0.4)
    assert np.allclose(captured["levels"], (0.0, 0.8))
    first_field = emitter.emit(
        captured["lum"], levels=captured["levels"],
        field=field, commit=False)
    scope.accept = True
    assert _queue_raster_candidate(
        scope, emitter, first_field, field_group=groups,
        levels_commit=(adaptive_state, captured["level_state"]))
    assert groups.index == emitter._field == 1
    assert adaptive_state["lo"] == 0.0
    assert np.isclose(adaptive_state["hi"], 0.8)

    scope.accept = False
    field, retried_picture = _begin_raster_field(groups, grab, levels_for)
    assert field == 1
    assert retried_picture is accepted_picture
    second_field = emitter.emit(
        retried_picture["lum"], levels=retried_picture["levels"],
        field=field, commit=False, start=emitter._end)
    assert not _queue_raster_candidate(
        scope, emitter, second_field, field_group=groups)
    assert groups.index == emitter._field == 1
    assert adaptive_state["lo"] == 0.0
    assert np.isclose(adaptive_state["hi"], 0.8)

    scope.accept = True
    field, retried_picture = _begin_raster_field(groups, grab, levels_for)
    assert field == 1
    assert retried_picture is accepted_picture
    second_field = emitter.emit(
        retried_picture["lum"], levels=retried_picture["levels"],
        field=field, commit=False, start=emitter._end)
    assert _queue_raster_candidate(
        scope, emitter, second_field, field_group=groups)
    assert groups.index == 0
    assert len(grabs) == len(level_calls) == 2

    # The next picture captures the third source only after the previous pair
    # has completed.
    field, next_picture = _begin_raster_field(groups, grab, levels_for)
    assert field == 0
    assert np.isclose(float(next_picture["lum"][0, 0]), 0.8)
    assert np.allclose(next_picture["levels"], (0.0, 1.2))
    assert len(grabs) == len(level_calls) == 3


def test_standalone_raster_commits_when_scope_raises_after_acceptance():
    from tools.scope_screen import _queue_raster_candidate

    class AcceptedThenErrorScope:
        frames_accepted = 0
        last_accepted_endpoint = None

        def show_frame(self, frame, handoff=None):
            self.frames_accepted += 1
            self.last_accepted_endpoint = np.array(
                [0.25, -0.3], dtype=np.float32)
            raise RuntimeError("injected after acceptance")

    groups = FieldGroupLatch(2)
    field, request = groups.begin({"image": "pinned"})
    emitter = TraceEmitter(96000, 3200, fields=2, sweep="alternate")
    levels = {"lo": None, "hi": None}
    candidate_levels = {"lo": 0.1, "hi": 0.9}
    frame = np.zeros((32, 2), dtype=np.float32)
    frame[-1] = (0.7, 0.6)

    accepted = _queue_raster_candidate(
        AcceptedThenErrorScope(), emitter, frame, field_group=groups,
        levels_commit=(levels, candidate_levels))

    assert request == {"image": "pinned"}
    assert accepted
    assert emitter._field == groups.index == 1
    np.testing.assert_allclose(emitter._end, (0.25, -0.3))
    assert levels == candidate_levels


def test_field_group_latch_pins_sources_until_all_fields_are_accepted():
    groups = FieldGroupLatch(2)

    field, source = groups.begin("image-A")
    assert (field, source) == (0, "image-A")
    # An empty/failed first field is retried against the latest request.
    assert groups.begin("image-B") == (0, "image-B")
    groups.accept()

    # Once field 0 was queued, newer source requests cannot split the picture.
    assert groups.begin("image-C") == (1, "image-B")
    assert groups.begin("image-D") == (1, "image-B")
    # Failure on field 1 leaves the group pinned and its field unadvanced.
    assert groups.index == 1
    groups.accept()
    assert groups.index == 0
    assert groups.begin("image-E") == (0, "image-E")


def test_scope_counts_accepted_and_callback_adopted_presentations_separately():
    samples = 64
    scope = Scope(device="null", samplerate=96000, samples=samples,
                  trigger=False, channel_pair=(24, 25), invert_y=False)
    frame = np.column_stack((np.linspace(-0.5, 0.5, samples),
                             np.linspace(0.4, -0.4, samples))).astype(np.float32)
    first_identity = (1, "raster", 7, 2, 4, 0, 2)
    latest_identity = (1, "raster", 7, 2, 4, 1, 2)
    try:
        scope.show_frame(frame, identity=first_identity)
        scope.show_frame(frame, identity=latest_identity)
        assert scope.frames_accepted == 2
        assert scope.frames_adopted == 0
        assert scope.frames_dropped == 1

        out = np.zeros((samples, scope.output_channels), np.float32)
        scope._callback(out, samples, None, None)
        assert scope.frames_adopted == 1
        assert scope.last_adopted_identity == latest_identity
        first_adopted_at_ns = scope.last_adopted_monotonic_ns
        assert isinstance(first_adopted_at_ns, int)
        assert first_adopted_at_ns <= time.monotonic_ns()
        assert scope.distinct_source_adoptions == 1
        assert scope.adoption_snapshot()[-1] == (
            latest_identity, first_adopted_at_ns)

        endpoint = scope.show_frame(
            frame, identity=(1, "raster", 8, 2, 4, 0, 2))
        np.testing.assert_array_equal(scope.last_accepted_endpoint, endpoint)
        out.fill(0)
        scope._callback(out, samples, None, None)
        assert scope.frames_accepted == 3
        assert scope.frames_adopted == 2
        assert scope.last_adopted_identity == (1, "raster", 8, 2, 4, 0, 2)
        assert scope.last_adopted_monotonic_ns >= first_adopted_at_ns
        assert scope.distinct_source_adoptions == 2
    finally:
        scope.stream.close()


def test_callback_adoption_preserves_runtime_source_timing_provenance():
    samples = 64
    scope = Scope(device="null", samplerate=96000, samples=samples,
                  trigger=False, channel_pair=(24, 25), invert_y=False)
    frame = np.column_stack((np.linspace(-0.5, 0.5, samples),
                             np.linspace(0.4, -0.4, samples))).astype(np.float32)
    selected_ns = time.monotonic_ns() - 3_000_000
    requested_ns = selected_ns + 100_000
    decode_started_ns = requested_ns + 200_000
    ready_ns = decode_started_ns + 800_000
    identity = (2, "stipple", 3, 1, 0, 0, 1, "runtime-images",
                selected_ns, requested_ns, decode_started_ns, ready_ns)
    try:
        scope.show_frame(frame, identity=identity)
        out = np.zeros((samples, scope.output_channels), np.float32)
        scope._callback(out, samples, None, None)
        assert scope.last_adopted_identity == identity
        assert scope.last_adopted_monotonic_ns >= ready_ns
        assert scope.distinct_source_adoptions == 1
        assert scope.adoption_snapshot()[-1] == (
            identity, scope.last_adopted_monotonic_ns)
    finally:
        scope.stream.close()


def test_clear_waits_for_inflight_render_and_prevents_next_repeat():
    scope = FakeScope()
    scheduler = ScopeFrameScheduler()
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def render():
        calls.append(1)
        entered.set()
        release.wait(1.0)

    try:
        scheduler.publish(scope, render, repeat=True)
        assert entered.wait(1.0)
        cleared = threading.Event()

        def clear():
            scheduler.clear(wait=True)
            cleared.set()

        thread = threading.Thread(target=clear)
        thread.start()
        time.sleep(0.01)
        assert not cleared.is_set()
        release.set()
        assert cleared.wait(1.0)
        time.sleep(0.03)
        assert calls == [1]
    finally:
        release.set()
        scheduler.close()


def test_sweep_source_live_tuning_waits_for_generator_chunk_boundary():
    entered_luminance = threading.Event()
    release_luminance = threading.Event()
    configured = threading.Event()
    config_lock_attempted = threading.Event()
    output = []

    class ObservedLock:
        def __init__(self, lock):
            self._lock = lock

        def __enter__(self):
            if threading.current_thread().name == "sweep-configurer":
                config_lock_attempted.set()
            self._lock.acquire()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self._lock.release()

    def luminance():
        entered_luminance.set()
        assert release_luminance.wait(1.0)
        return np.full((16, 24), 0.75, dtype=np.float32)

    source = SweepSource(samples_per_pass=3200, lum_fn=luminance,
                         grid_rows=16, grid_cols=24)
    source._config_lock = ObservedLock(source._config_lock)
    producer = threading.Thread(target=lambda: output.append(source(64)))
    updater = None

    def update_settings():
        source.configure(gamma=1.4, trim=0.08)
        configured.set()

    producer.start()
    try:
        assert entered_luminance.wait(1.0)
        updater = threading.Thread(target=update_settings,
                                   name="sweep-configurer")
        updater.start()
        assert config_lock_attempted.wait(1.0)
        assert not configured.wait(0.02)
        release_luminance.set()
        producer.join(1.0)
        updater.join(1.0)
        assert not producer.is_alive()
        assert not updater.is_alive()
        assert configured.is_set()
        assert output[0].shape == (64, 2)
        assert source.gamma == 1.4
        assert source.trim == 0.08
    finally:
        release_luminance.set()
        producer.join(1.0)
        if updater is not None:
            updater.join(1.0)


def test_unchanged_sweep_tuning_preserves_inflight_row_plan():
    source = SweepSource(
        samples_per_pass=3200, lum_fn=lambda: np.ones((16, 24), np.float32),
        grid_rows=16, grid_cols=24)
    source(64)
    plan, budgets = source._plan, source._budgets
    row_i, passes = source._row_i, source.passes
    source.configure(gamma=source.gamma, trim=source.trim,
                     grid_rows=16, grid_cols=24, levels=None)
    assert source._plan is plan
    assert source._budgets is budgets
    assert source._row_i == row_i
    assert source.passes == passes

    source.configure(gamma=1.4)
    assert source.gamma == 1.4
    assert source._plan is None
    assert source._budgets is None
