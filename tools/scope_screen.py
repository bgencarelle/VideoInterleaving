"""
scope_screen.py -- use an oscilloscope as a (very) low resolution display.

Feeds any live source into the same dwell-modulated sweep the VideoInterleaving
scope mode uses, so anything you can put in a numpy array can go on the tube:

    python scope_screen.py --source screen --device BlackHole
    python scope_screen.py --source screen --region 0,0,800,600
    python scope_screen.py --source video --file clip.mp4
    python scope_screen.py --source test

BE REALISTIC ABOUT THE RESOLUTION.  The grid is bounded by samples per trace,
which is sample_rate / fps -- 3200 at 96 kHz and 30 fps.  A full-frame source
has no dark margins for trim to reclaim, so that is about 56x56 cells,
grayscale.  Good for silhouettes, large shapes, a clock, a moving figure.
Not for text and not for a desktop UI.

Lower --fps for a bigger grid at the cost of refresh:

    48 kHz, 30 fps -> 1600 samples -> ~40x40
    96 kHz, 30 fps -> 3200 samples -> ~56x56
    96 kHz, 15 fps -> 6400 samples -> ~80x80   (15 Hz flickers on short phosphor)

Screen capture needs `mss` (pip install mss).  On Wayland mss cannot grab the
screen; use X11, or feed frames another way with --source video.
"""
import argparse
import math
import json
import os
import queue
import sys
import threading
import time

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scope_bake import (SweepSource, plan_grid, TraceEmitter,
                        StochasticEmitter, StippleEmitter)   # noqa: E402
from scope_out import Scope, BufferedSource, choose_device  # noqa: E402
from scope_frame_scheduler import FieldGroupLatch  # noqa: E402
from scope_prepared_cache import PreparedImageCache  # noqa: E402
from scope_numeric import (captured_luma, gray_units, positive_percentile,
                           moving_test_image)  # noqa: E402


def _begin_raster_field(field_group, grab, levels_for):
    """Capture one immutable source/tone-map snapshot per accepted picture."""
    request = None
    if field_group.index == 0:
        timed_snapshot = getattr(grab, "timed_snapshot", None)
        if callable(timed_snapshot):
            source, source_version, source_metadata = timed_snapshot()
        else:
            snapshot = getattr(grab, "prepared_snapshot", None)
            if callable(snapshot):
                source, source_version = snapshot()
            else:
                source, source_version = grab(), None
            source_metadata = None
        lum = np.asarray(source, dtype=np.float32).copy()
        levels, level_state = levels_for(lum)
        if levels is not None:
            levels = tuple(levels)
        request = {"lum": lum, "levels": levels,
                   "level_state": level_state,
                   "source_version": source_version,
                   "source_metadata": source_metadata,
                   "selected_at_ns": time.monotonic_ns()}
    field, active = field_group.begin(request)
    if active is None:
        raise RuntimeError("interlaced raster field has no pinned source")
    return field, active


def _queue_raster_candidate(scope, emitter, frame, handoff=None,
                            field_group=None, levels_commit=None,
                            identity=None):
    """Advance raster sweep/field state only when Scope accepts a trace."""
    accepted_before = getattr(scope, "frames_accepted", None)
    if getattr(emitter, "traversal_hz", None) is not None:
        # Time traversal uses the periodic trigger retrace as its frame
        # boundary. Scope's optional handoff anchoring would perturb its clock.
        handoff = None
    try:
        if identity is not None:
            endpoint = scope.show_frame(frame, handoff=handoff,
                                        identity=identity)
        else:
            endpoint = (scope.show_frame(frame, handoff=handoff)
                        if handoff is not None else scope.show_frame(frame))
    except Exception:
        # Some Scope implementations can report an error after placing the
        # candidate in the output queue. Do not retry an already accepted
        # field, or the interlace group and trajectory will fall behind DAC.
        accepted_after = getattr(scope, "frames_accepted", accepted_before)
        if (accepted_before is None or accepted_after <= accepted_before):
            raise
        endpoint = getattr(scope, "last_accepted_endpoint", None)
        if endpoint is None:
            endpoint = frame[-1]
    accepted = (
        scope.frames_accepted > accepted_before
        if accepted_before is not None else endpoint is not None)
    if accepted:
        accept = getattr(emitter, "accept", None) or emitter.chain_from
        accept(frame[-1] if endpoint is None else endpoint)
        if field_group is not None:
            field_group.accept()
        if levels_commit is not None:
            target, candidate = levels_commit
            if candidate is not None:
                target.update(candidate)
    return accepted


def _push_live_image(scope, emitter, captured, mode):
    """Queue a captured-image trajectory, rolling back rejected walk state."""
    checkpoint = emitter.checkpoint()
    handoff = None if emitter._end is None else emitter._end.copy()
    try:
        frame = emitter.emit(captured["lum"])
        accepted = _queue_raster_candidate(
            scope, emitter, frame, handoff=handoff,
            identity=_source_presentation_identity(captured, 0, 1, mode))
    except Exception:
        emitter.restore(checkpoint)
        raise
    if not accepted:
        emitter.restore(checkpoint)
    return accepted


def live_capture_modes(args):
    """Whole captured images share the runtime-image renderer choices."""
    if (args.stream or args.fields != 1 or args.geometry_samples is not None
            or args.traversal_hz is not None):
        return ("raster",)
    return ("raster", "stochastic", "stipple")


def _source_presentation_identity(captured, field, fields, mode="raster"):
    """Carry the chosen capture/video timestamp through Scope adoption."""
    metadata = captured.get("source_metadata") or {}
    source_version = captured.get("source_version")
    sequence = metadata.get("source_sequence")
    if sequence is None:
        sequence = id(source_version)
    return (
        0, f"screen-{mode}", int(sequence), 0, 0, int(field), int(fields),
        str(metadata.get("source_kind", "capture")),
        captured.get("selected_at_ns"), metadata.get("requested_at_ns"),
        metadata.get("decode_started_at_ns"), metadata.get("ready_at_ns"),
        metadata.get("media_position_s"),
    )


def _adopted_source_metrics(scope, playback_position_s=None):
    """Summarize source timestamps at the output callback's adoption boundary."""
    record = getattr(scope, "last_adopted_record", None)
    if isinstance(record, tuple) and len(record) == 2:
        identity, adopted_ns = record
    else:
        identity = getattr(scope, "last_adopted_identity", None)
        adopted_ns = getattr(scope, "last_adopted_monotonic_ns", None)
    if (not isinstance(identity, tuple) or len(identity) < 12
            or adopted_ns is None):
        return {}
    selected_ns, requested_ns, decode_started_ns, ready_ns = identity[8:12]
    now_ns = time.monotonic_ns()
    metrics = {
        "adopted_source_kind": identity[7],
        "adopted_source_sequence": identity[2],
        "source_selected_at_ns": selected_ns,
        "source_requested_at_ns": requested_ns,
        "source_decode_started_at_ns": decode_started_ns,
        "source_ready_at_ns": ready_ns,
        "source_adopted_at_ns": adopted_ns,
        "adopted_media_position_s": (
            identity[12] if len(identity) > 12 else None),
        "distinct_source_adoptions": int(
            getattr(scope, "distinct_source_adoptions", 0)),
        "source_age_at_adoption_ms": (
            max(0.0, (adopted_ns - requested_ns) / 1e6)
            if requested_ns is not None else None),
        "selection_to_adoption_ms": (
            max(0.0, (adopted_ns - selected_ns) / 1e6)
            if selected_ns is not None else None),
        "source_age_now_ms": (
            max(0.0, (now_ns - requested_ns) / 1e6)
            if requested_ns is not None else None),
        "source_decode_queue_ms": (
            max(0.0, (decode_started_ns - requested_ns) / 1e6)
            if decode_started_ns is not None and requested_ns is not None
            else None),
        "source_decode_ms": (
            max(0.0, (ready_ns - decode_started_ns) / 1e6)
            if ready_ns is not None and decode_started_ns is not None else None),
        "source_ready_to_adoption_ms": (
            max(0.0, (adopted_ns - ready_ns) / 1e6)
            if ready_ns is not None else None),
    }
    metrics.update(_source_update_rate(scope))
    media_position = metrics["adopted_media_position_s"]
    metrics["media_lag_s"] = (
        max(0.0, float(playback_position_s) - float(media_position))
        if playback_position_s is not None and media_position is not None
        else None)
    return metrics


def _source_update_rate(scope):
    records = scope.adoption_snapshot()
    source_records = [
        (tuple(identity[2:5]), adopted_ns)
        for identity, adopted_ns in records
        if isinstance(identity, tuple) and len(identity) >= 7]
    if len(source_records) < 2:
        return {"fresh_source_adoptions_per_second": 0.0,
                "fresh_source_adoptions_in_window": len(source_records)}
    changes = sum(a[0] != b[0]
                  for a, b in zip(source_records, source_records[1:]))
    elapsed = max((source_records[-1][1] - source_records[0][1]) / 1e9,
                  1e-6)
    return {"fresh_source_adoptions_per_second": changes / elapsed,
            "fresh_source_adoptions_in_window": changes}


# ---------------------------------------------------------------- sources

def shrink(raw, downto):
    """Downscale a captured frame to `downto` pixels wide, cheaply.

    cv2.resize with INTER_AREA straight from 3840 to 256 averages 15x15 blocks
    per output pixel and costs ~52 ms on a 4K frame -- more than a core at 30
    grabs/sec.  Striding first is a VIEW, so it costs nothing, and the area
    average then runs on an image 16x smaller.  Measured 12.8x faster, and the
    residual aliasing is invisible once the grid is ~56 cells wide.
    """
    import cv2
    h, w = raw.shape[:2]
    if w <= downto:
        src = raw
    else:
        # stride down to roughly 2x the target, then area-average the rest
        step = max(1, int(w // (downto * 2)))
        src = raw[::step, ::step]
        sh, sw = src.shape[:2]
        if sw > downto:
            src = cv2.resize(src, (downto, max(1, int(sh * downto / sw))),
                             interpolation=cv2.INTER_AREA)
    return captured_luma(src)


def screen_source(region=None, downto=160):
    """Grab the screen (or a region) as luminance.

    np.asarray(shot) copies the whole BGRA buffer -- ~30 MB on a Retina panel,
    every grab.  np.frombuffer wraps the same bytes without copying, which is
    free.  Capture is by far the dominant cost of screen mode (measured 34 ms
    per grab on a Retina Mac against 1 ms for the trace build), so the copy is
    worth removing even though the win is only part of it.
    """
    try:
        from mss import MSS as _MSS          # newer API
    except ImportError:
        from mss import mss as _MSS

    sct = _MSS()
    mon = sct.monitors[1] if region is None else {
        "left": region[0], "top": region[1],
        "width": region[2], "height": region[3]}

    def grab():
        shot = sct.grab(mon)
        raw = np.frombuffer(shot.raw, dtype=np.uint8).reshape(
            shot.height, shot.width, 4)
        return shrink(raw[:, :, :3], downto)
    return grab


def _read_exact(stream, nbytes):
    """Read one whole raw-video frame from a possibly short-reading pipe."""
    frame = bytearray(int(nbytes))
    view = memoryview(frame)
    offset = 0
    while offset < len(frame):
        chunk = stream.read(len(frame) - offset)
        if not chunk:
            return None
        view[offset:offset + len(chunk)] = chunk
        offset += len(chunk)
    return frame


def ffmpeg_source(width=160, fps=12, region=None, input_spec=None,
                  display=None, source_kind="capture", startup_timeout=15.0):
    """
    Capture via ffmpeg instead of in Python.

    mss goes through CoreGraphics on macOS and costs ~34 ms per grab at Retina
    backing resolution -- more than everything else in the pipeline combined.
    ffmpeg uses the platform's fast path (avfoundation / x11grab / gdigrab) AND
    does the downscale and grayscale conversion itself, so Python receives a
    ~160x104 single-channel frame and does no image work at all.

    The output size is pinned explicitly so each frame is a known number of
    bytes; guessing it from ffmpeg's stderr is fragile.
    """
    import shutil
    import subprocess
    import threading

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found. brew install ffmpeg / apt install ffmpeg")

    # source dimensions, so the scaled height is exact (no grab required)
    try:
        import mss as _mss
        with (_mss.MSS() if hasattr(_mss, "MSS") else _mss.mss()) as _s:
            mon = _s.monitors[1]
            sw, sh = mon["width"], mon["height"]
    except Exception:
        sw, sh = 1920, 1080
    if region:
        sw, sh = region[2], region[3]

    w = int(width)
    h = max(2, int(round(w * sh / float(sw))) // 2 * 2)

    if input_spec:
        fmt, src = input_spec.split(":", 1)
    elif sys.platform == "darwin":
        fmt, src = "avfoundation", f"{display if display is not None else 1}:none"
    elif sys.platform.startswith("win"):
        fmt, src = "gdigrab", "desktop"
    else:
        fmt, src = "x11grab", os.environ.get("DISPLAY", ":0.0")

    cmd = ["ffmpeg", "-loglevel", "error", "-f", fmt,
           "-framerate", str(int(max(fps, 1)))]
    if fmt == "x11grab" and region:
        cmd += ["-video_size", f"{region[2]}x{region[3]}",
                "-i", f"{src}+{region[0]},{region[1]}"]
    else:
        cmd += ["-i", src]
    cmd += ["-vf", f"scale={w}:{h}", "-pix_fmt", "gray",
            "-f", "rawvideo", "-an", "-sn", "-"]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=None, bufsize=0)
    # Drivers can wait indefinitely for permission or an unavailable camera.
    # Bound the first read; inherited stderr reaches the launcher's log pipe.
    startup_done = threading.Event()
    startup_expired = threading.Event()

    def startup_watchdog():
        if not startup_done.wait(startup_timeout):
            startup_expired.set()
            try:
                proc.kill()
            except OSError:
                pass

    threading.Thread(target=startup_watchdog, daemon=True,
                     name="scope-capture-startup").start()
    nbytes = w * h
    last = [np.zeros((h, w), np.float32)]
    metadata = [{"source_kind": str(source_kind), "source_sequence": 0,
                 "requested_at_ns": None, "decode_started_at_ns": None,
                 "ready_at_ns": None, "media_position_s": None}]

    def grab():
        started_ns = time.monotonic_ns()
        buf = _read_exact(proc.stdout, nbytes)
        if buf is None:
            if metadata[0]["source_sequence"] == 0:
                startup_done.set()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2.0)
                proc.stdout.close()
                if startup_expired.is_set():
                    raise RuntimeError("Camera/capture startup timed out: no frame within "
                                       f"{startup_timeout:g}s. Check camera permissions, "
                                       "device availability and capture frame rate.")
                raise RuntimeError("FFmpeg capture stopped before its first frame. "
                                   "Check the FFmpeg error above, camera permissions "
                                   "and supported capture frame rate.")
            return last[0]
        startup_done.set()
        last[0] = gray_units(np.frombuffer(buf, np.uint8).reshape(h,w))
        metadata[0] = {
            "source_kind": str(source_kind),
            "source_sequence": int(metadata[0]["source_sequence"]) + 1,
            "requested_at_ns": started_ns,
            "decode_started_at_ns": started_ns,
            "ready_at_ns": time.monotonic_ns(),
            "media_position_s": None,
        }
        return last[0]

    grab.proc = proc
    grab.size = (w, h)
    grab.timing_snapshot = lambda: dict(metadata[0])
    return grab


class Throttled:
    """Publish immutable latest-frame snapshots from a reader worker.

    The sweep asks for a frame every trace, but a ~56 cell display does not
    need the screen sampled 30 times a second, and capture is the expensive
    part. Decoupling means the producer never waits on a grab. ``drain`` is for
    already-paced FFmpeg pipes: reading continuously prevents pipe buffering
    from turning a latest-frame display into delayed FIFO playback.
    """

    def __init__(self, grab, fps=12.0, *, source_kind="capture", drain=False):
        import threading
        self._grab = grab
        if hasattr(grab, "proc"):
            self.proc = grab.proc
        self.source_kind = str(source_kind)
        self._drain = bool(drain)
        self._source_owner = getattr(grab, "__self__", None)
        self._timing_source = getattr(grab, "timing_snapshot", None)
        if not callable(self._timing_source):
            self._timing_source = getattr(
                self._source_owner, "timing_snapshot", None)
        self._latest_lock = threading.Lock()
        started_ns = time.monotonic_ns()
        self._latest = grab()
        ready_ns = time.monotonic_ns()
        self._prepared_source_version = object()
        self._latest_metadata = self._metadata(
            started_ns, ready_ns, source_sequence=1)
        self._source_sequence = int(
            self._latest_metadata.get("source_sequence", 1))
        self._stop = threading.Event()
        self.captures = 0
        self.polls = 0
        self.failures = 0
        self.last_capture_duration_ms = max(0.0, (ready_ns - started_ns) / 1e6)
        self._period = 1.0 / max(fps, 0.5)
        self._t = threading.Thread(target=self._run, daemon=True,
                                   name="scope-capture")
        self._t.start()

    def _metadata(self, started_ns, ready_ns, *, source_sequence):
        metadata = None
        if callable(self._timing_source):
            try:
                metadata = self._timing_source()
            except Exception:
                metadata = None
        metadata = dict(metadata or {})
        metadata.setdefault("source_kind", self.source_kind)
        metadata.setdefault("source_sequence", int(source_sequence))
        metadata.setdefault("requested_at_ns", int(started_ns))
        metadata.setdefault("decode_started_at_ns", int(started_ns))
        metadata.setdefault("ready_at_ns", int(ready_ns))
        metadata.setdefault("media_position_s", None)
        return metadata

    def _run(self):
        while not self._stop.is_set():
            t0 = time.perf_counter()
            started_ns = time.monotonic_ns()
            try:
                latest = self._grab()
                ready_ns = time.monotonic_ns()
                metadata = self._metadata(
                    started_ns, ready_ns,
                    source_sequence=self._source_sequence + 1)
                sequence = int(metadata["source_sequence"])
                with self._latest_lock:
                    self.polls += 1
                    changed = sequence != self._source_sequence
                    if changed:
                        self._latest = latest
                        self._latest_metadata = metadata
                        self._prepared_source_version = object()
                        self._source_sequence = sequence
                        self.captures += 1
                    self.last_capture_duration_ms = max(
                        0.0, (ready_ns - started_ns) / 1e6)
            except Exception:
                self.failures += 1
                changed = False
            if self._drain:
                # A pipe may return its final frame immediately at EOF. Avoid
                # a hot loop while retaining prompt wake-up for the next frame.
                if not changed:
                    self._stop.wait(0.001)
            else:
                self._stop.wait(max(
                    0.0, self._period - (time.perf_counter() - t0)))

    def __call__(self):
        return self.prepared_snapshot()[0]

    def prepared_snapshot(self):
        """Atomically return a frame and its immutable-content version token."""
        with self._latest_lock:
            return self._latest, self._prepared_source_version

    def timed_snapshot(self):
        """Return the frame, content token, and source/decode timestamps."""
        with self._latest_lock:
            return (self._latest, self._prepared_source_version,
                    dict(self._latest_metadata))

    def snapshot_metrics(self):
        with self._latest_lock:
            ready_ns = int(self._latest_metadata.get("ready_at_ns", 0))
            return {
                "captures": self.captures,
                "polls": self.polls,
                "failures": self.failures,
                "pending_latest_age_ms": max(
                    0.0, (time.monotonic_ns() - ready_ns) / 1e6),
                "last_capture_duration_ms": self.last_capture_duration_ms,
                "source_kind": self._latest_metadata.get("source_kind"),
                "source_sequence": self._latest_metadata.get("source_sequence"),
                "media_position_s": self._latest_metadata.get("media_position_s"),
            }

    def close(self):
        self._stop.set()
        self._t.join(timeout=2.0)


class VideoFileSource:
    """Timestamp-paced video decoder with nonblocking latest-frame snapshots.

    ``read_latest_due`` is used by the reader worker. It advances according to
    elapsed media time and drops decoded frames that are already late. ``__call__``
    retains the original one-frame-at-a-time behavior for compatibility callers.
    """

    def __init__(self, path, downto=160, loop=True, start_at=0.0,
                 cv2_module=None, clock=None):
        if cv2_module is None:
            import cv2 as cv2_module
        self.cv2 = cv2_module
        self._clock = clock or time.monotonic
        self.cap = cv2_module.VideoCapture(path)
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open {path}")
        self.downto = max(1, int(downto))
        self.loop = bool(loop)
        self.lock = threading.RLock()
        self.paused = False
        self.last = None
        self.position = 0.0
        fps = float(self.cap.get(cv2_module.CAP_PROP_FPS) or 0.0)
        count = float(self.cap.get(cv2_module.CAP_PROP_FRAME_COUNT) or 0.0)
        self.frame_rate = fps if math.isfinite(fps) and fps > 0 else None
        self.duration = (count / fps if count > 0 and fps > 0 else None)
        self._playback_anchor_wall = self._clock()
        self._playback_anchor_position = 0.0
        self._next_decoded = None
        self._eof = False
        self._seek_pending = False
        self._generation = 0
        self._sequence = 0
        self._frame_metadata = {
            "source_kind": "video", "source_sequence": 0,
            "requested_at_ns": None, "decode_started_at_ns": None,
            "ready_at_ns": None, "media_position_s": None,
            "generation": self._generation,
        }
        if start_at > 0:
            self.transport("seek", start_at)

    def _frame_position_locked(self):
        if self.frame_rate:
            frame_index = float(
                self.cap.get(self.cv2.CAP_PROP_POS_FRAMES) or 0)
            return max(0.0, (frame_index - 1.0) / self.frame_rate)
        position = float(self.cap.get(self.cv2.CAP_PROP_POS_MSEC) or 0)
        return max(0.0, position / 1000.0)

    def _read_next_locked(self):
        started_ns = time.monotonic_ns()
        ok, frame = self.cap.read()
        if not ok:
            return None
        image = np.asarray(shrink(frame, self.downto), dtype=np.float32).copy()
        ready_ns = time.monotonic_ns()
        image.setflags(write=False)
        return {
            "image": image,
            "position": self._frame_position_locked(),
            "requested_at_ns": started_ns,
            "decode_started_at_ns": started_ns,
            "ready_at_ns": ready_ns,
        }

    def _publish_locked(self, candidate):
        self.last = candidate["image"]
        self.position = float(candidate["position"])
        self._seek_pending = False
        self._sequence += 1
        self._frame_metadata = {
            "source_kind": "video",
            "source_sequence": self._sequence,
            "requested_at_ns": int(candidate["requested_at_ns"]),
            "decode_started_at_ns": int(candidate["decode_started_at_ns"]),
            "ready_at_ns": int(candidate["ready_at_ns"]),
            "media_position_s": self.position,
            "generation": self._generation,
        }

    def __call__(self):
        with self.lock:
            if self.paused and self.last is not None and not self._seek_pending:
                return self.last
            candidate = self._read_next_locked()
            if candidate is None and self.loop:
                self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, 0)
                self._eof = False
                candidate = self._read_next_locked()
            if candidate is None:
                if self.last is not None:
                    return self.last
                return np.zeros((64, 64), np.float32)
            self._publish_locked(candidate)
            return self.last

    def read_latest_due(self):
        """Decode through the latest frame due at the current media time."""
        with self.lock:
            now = self._clock()
            if (self.paused and self.last is not None
                    and not self._seek_pending):
                return self.last
            target = (self.position if self.paused else
                      self._playback_anchor_position
                      + max(0.0, now - self._playback_anchor_wall))
            if self.duration is not None and target >= self.duration:
                if not self.loop:
                    target = max(0.0, self.duration -
                                 (1.0 / self.frame_rate
                                  if self.frame_rate else 0.001))
                else:
                    target %= self.duration
                    self.cap.set(self.cv2.CAP_PROP_POS_MSEC, target * 1000.0)
                    self._next_decoded = None
                    self._eof = False
                    self._playback_anchor_position = target
                    self._playback_anchor_wall = now

            while True:
                if self._next_decoded is None:
                    if self._eof:
                        break
                    self._next_decoded = self._read_next_locked()
                    if self._next_decoded is None:
                        self._eof = True
                        break
                # A decoder can land just after a non-frame-aligned seek
                # target. Publish that first seek result even while paused,
                # otherwise the frozen media clock can leave it pending.
                if (self._next_decoded["position"] <= target + 1e-9
                        or self._seek_pending):
                    self._publish_locked(self._next_decoded)
                    self._next_decoded = None
                    continue
                break

            if self.last is not None:
                return self.last
            return np.zeros((64, 64), np.float32)

    def timing_snapshot(self):
        with self.lock:
            return dict(self._frame_metadata)

    def transport(self, command, position=None):
        """Apply play, pause, restart, or seek without stopping scope output."""
        command = str(command or "").strip().lower()
        with self.lock:
            if command == "pause":
                if not self.paused:
                    self.position = (
                        self._playback_anchor_position
                        + max(0.0, self._clock()
                              - self._playback_anchor_wall))
                    if self.duration is not None and self.loop:
                        self.position %= self.duration
                    self._playback_anchor_position = self.position
                    self._playback_anchor_wall = self._clock()
                self.paused = True
            elif command == "play":
                self.paused = False
                self._playback_anchor_position = self.position
                self._playback_anchor_wall = self._clock()
            elif command == "restart":
                was_paused = self.paused
                self._seek_locked(0.0)
                self.paused = was_paused
            elif command == "seek":
                try:
                    target = float(position)
                except (TypeError, ValueError) as exc:
                    raise ValueError("seek position must be a number") from exc
                if not math.isfinite(target) or target < 0:
                    raise ValueError("seek position must be finite and non-negative")
                self._seek_locked(target)
            elif command == "shutdown":
                return "shutdown"
            else:
                raise ValueError(f"unknown video transport command: {command}")
        return None

    def _seek_locked(self, seconds):
        if self.duration is not None:
            seconds = min(seconds, max(0.0, self.duration - 0.001))
        self.cap.set(self.cv2.CAP_PROP_POS_MSEC, seconds * 1000.0)
        self.position = seconds
        # Keep the last complete decoded image visible until the seek target
        # has been decoded. A transport discontinuity must not flash black.
        self._next_decoded = None
        self._eof = False
        self._seek_pending = True
        self._generation += 1
        self._playback_anchor_position = seconds
        self._playback_anchor_wall = self._clock()

    def playback(self):
        with self.lock:
            position = self.position
            if not self.paused:
                position = (self._playback_anchor_position
                            + max(0.0, self._clock()
                                  - self._playback_anchor_wall))
                if self.duration is not None and self.loop:
                    position %= self.duration
            return {
                "status": "playback",
                "position": round(position, 3),
                "duration": (round(self.duration, 3)
                             if self.duration is not None else None),
                "paused": self.paused,
                "frame_position": self._frame_metadata.get("media_position_s"),
                "frame_sequence": self._sequence,
                "generation": self._generation,
            }

    def close(self):
        with self.lock:
            self.cap.release()


def video_source(path, downto=160, loop=True, start_at=0.0,
                 cv2_module=None):
    """Compatibility factory for callers that need a callable frame source."""
    return VideoFileSource(path, downto, loop, start_at, cv2_module)


def _control_reader(source, messages, stop):
    """Read newline-delimited JSON commands without blocking the scope loop."""
    for line in sys.stdin:
        try:
            message = json.loads(line)
            if not isinstance(message, dict):
                continue
            result = source.transport(message.get("transport"),
                                      message.get("position"))
            if result == "shutdown":
                stop.set()
                return
            messages.put({"status": "control", "ok": True})
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            messages.put({"status": "control", "ok": False,
                          "message": str(exc)})


def test_source():
    """Known geometry, so you can tell distortion from content."""
    t = [0]

    def grab():
        t[0] += 1
        return moving_test_image(t[0])
    return grab


# ---------------------------------------------------------------- main

def _profile(args, grab):
    """Report the real cost split on this machine.

    Worth doing rather than guessing: screen capture on macOS goes through
    CoreGraphics and can dwarf everything downstream, but it cannot be
    measured anywhere except on the machine doing it.
    """
    import cv2

    n = int(args.samples or (96000 // max(args.fps, 1)))

    inner = grab._grab if hasattr(grab, "_grab") else grab
    ffmpeg_mode = hasattr(grab, "proc") or hasattr(getattr(grab, "_grab", None), "proc")
    for _ in range(3):
        inner()
    t0 = time.perf_counter()
    for _ in range(20):
        inner()
    cap = (time.perf_counter() - t0) / 20 * 1000

    lum = np.asarray(inner(), dtype=np.float32)
    profile_grid = None
    if args.geometry_samples is not None or args.traversal_hz is not None:
        geometry_budget = args.geometry_samples or 3200
        profile_grid = plan_grid(
            lum, geometry_budget, density=args.density, trim=args.trim,
            rows=args.rows, fields=max(1, args.fields))
    # Time the REAL path, not a hand-rolled approximation of it -- a probe that
    # measures something the program never runs is worse than no probe.
    _probe_em = TraceEmitter(48000, n, gamma=args.gamma, trim=args.trim,
                             density=args.density, rows=args.rows,
                             fields=max(1, args.fields),
                             border=getattr(args, "border", 0.0),
                             oversample=getattr(args, "oversample", 1),
                             geometry_samples=args.geometry_samples,
                             traversal_hz=args.traversal_hz,
                             grid=profile_grid,
                             close_frame=not args.scope_trigger)
    for _ in range(3):
        _probe_em.emit(lum)
    t0 = time.perf_counter()
    for _ in range(30):
        _probe_em.emit(lum)
    build = (time.perf_counter() - t0) / 30 * 1000

    cap_hz = args.capture_fps
    print()
    if ffmpeg_mode:
        print(f"  pipe read + convert : {cap:6.2f} ms x {cap_hz:.0f}/s "
              f"= {cap * cap_hz / 10:5.1f}% of a core   (Python side only;")
        print( "                        ffmpeg does the capture and scaling in "
               "its own process)")
    else:
        print(f"  capture + downscale : {cap:6.2f} ms x {cap_hz:.0f}/s "
              f"= {cap * cap_hz / 10:5.1f}% of a core")
    print(f"  trace build         : {build:6.2f} ms x {args.fps}/s "
          f"= {build * args.fps / 10:5.1f}% of a core")
    bs = args.blocksize or 256
    print(f"  audio callbacks     : {96000 / bs:6.0f}/s at blocksize {bs} "
          "(Python overhead per wake-up)")
    print()
    if cap * cap_hz / 10 > 5 and args.source == "screen":
        print("  capture dominates. How it scales with region size on THIS "
              "machine:")
        for wh in ((1920, 1200), (1280, 800), (800, 600), (640, 400)):
            try:
                g = screen_source((0, 0, wh[0], wh[1]), downto=args.downto)
                for _ in range(2):
                    g()
                t0 = time.perf_counter()
                for _ in range(10):
                    g()
                ms = (time.perf_counter() - t0) / 10 * 1000
                print(f"    --region 0,0,{wh[0]},{wh[1]:<5} {ms:6.2f} ms "
                      f"= {ms * cap_hz / 10:5.1f}% of a core at "
                      f"{cap_hz:.0f} grabs/s")
            except Exception as e:
                print(f"    --region 0,0,{wh[0]},{wh[1]}: {e}")
        print()
        print("  Also worth knowing: this is a macOS problem. Capture there "
              "goes through")
        print("  CoreGraphics and is slow. On Linux/X11 the same grab uses "
              "XShm and is")
        print("  typically 5-10x cheaper, so a Pi 5 may well beat the Mac at "
              "this.")


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[1],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("screen", "ffmpeg", "camera",
                                         "video", "test"),
                    default="test",
                    help="screen = mss; ffmpeg = desktop capture via ffmpeg; "
                         "camera = custom FFmpeg input; video = looping file")
    ap.add_argument("--ffmpeg-input", metavar="FMT:SRC",
                    help="FFmpeg input, e.g. avfoundation:1:none or v4l2:/dev/video0")
    ap.add_argument("--display", type=int,
                    help="avfoundation screen index (see: ffmpeg -f "
                         "avfoundation -list_devices true -i \"\")")
    ap.add_argument("--file", help="video file for --source video")
    ap.add_argument("--region", help="screen region as left,top,width,height")
    ap.add_argument("--fps", type=int, default=30,
                    help="traces per second; lower = bigger grid, more flicker")
    ap.add_argument("--samples", type=int, help="samples per trace (overrides --fps)")
    ap.add_argument("--render-mode", choices=("raster", "stochastic", "stipple"),
                    default="raster", help="drawing style for captured images")
    ap.add_argument("--geometry-samples", type=int, metavar="N",
                    help="raster trajectory detail budget independent of DAC samples")
    ap.add_argument("--traversal-hz", type=float, metavar="HZ",
                    help="timed raster traversal speed in cycles/second; requires trigger and one field")
    ap.add_argument("--physical-dwell", type=float, default=0.0, metavar="F",
                    help="physical sample-time redistribution toward drawing segments (0=off, 1=strong)")
    ap.add_argument("--trim", type=float, default=0.02,
                    help="drop cells dimmer than this. Useful on dark content; "
                         "leave low for a full-frame source")
    ap.add_argument("--gamma", type=float, default=1.8)
    ap.add_argument("--border", type=float, default=0.0, metavar="F",
                    help="Fixed rectangle at the full extent, F of the trace's "
                         "samples (try 0.03). Stops the framing moving with "
                         "the content.")
    ap.add_argument("--oversample", type=int, default=1, metavar="N",
                    help="Anti-alias the path. ~5%% change on real content.")
    ap.add_argument("--fields", type=int, default=1, metavar="N",
                    help="Interlace: N=2 draws every other row per trace and "
                         "alternates, so the beam repaints at N x the picture "
                         "rate on the SAME grid. Refresh without paying "
                         "resolution for it. Use --fps N x capture-fps.")
    ap.add_argument("--density", type=float, default=1.0)
    ap.add_argument("--rows", type=int, help="scanline count (default: auto)")
    ap.add_argument("--adapt", type=float, default=4.0, metavar="SEC",
                    help="tone-mapping time constant. Live content cannot be "
                         "pre-scanned, but adapting fast makes cells flicker, "
                         "so this is deliberately slow. 0 = fixed 0..1")
    ap.add_argument("--capture-fps", type=float, default=12.0,
                    help="how often to grab the source, independent of the "
                         "trace rate. Capture is the expensive part and a ~56 "
                         "cell display does not need 30 grabs a second.")
    ap.add_argument("--downto", type=int, default=160,
                    help="width to shrink the source to before gridding "
                         "(default 160; the grid is ~56 wide, so this is "
                         "already oversampled)")
    ap.add_argument("--stream", action="store_true",
                    help="generate row by row using compiled preparation and "
                         "sampling; the default builds complete traces. Queued "
                         "block depth determines stream latency.")
    ap.add_argument("--buffer-blocks", type=int, default=6,
                    help="blocks queued ahead of the audio callback")
    ap.add_argument("--dc-comp", type=float, metavar="HZ",
                    help="cancel the output's AC coupling (try 20-50). The "
                         "vertical sweep runs at the trace rate, so a headphone "
                         "jack funnels the picture without this.")
    ap.add_argument("--blocksize", type=int, default=1024,
                    help="audio callback size. Bigger = fewer Python wake-ups "
                         "and less CPU, at the cost of latency (1024 at 96 kHz "
                         "is 10.7 ms). 0 lets the driver choose.")
    ap.add_argument("--scope-lowpass", type=float, metavar="HZ",
                    help="low-pass output samples at this corner frequency")
    ap.add_argument("--scope-gui", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="open the native scope preview and tuner")
    ap.add_argument("--scope-channels", default="1,2", metavar="X,Y",
                    help="1-based PortAudio channels carrying X and Y")
    ap.add_argument("--scope-x-only", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="send X only through one output channel for a Y-T scope")
    ap.add_argument("--scope-trigger", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="include the unique X trigger edge (default on)")
    ap.add_argument("--scope-trigger-us", type=float, default=250.0,
                    metavar="US", help="Y-T trigger marker duration")
    ap.add_argument("--scope-trigger-shape", choices=("ramp", "step"),
                    default="ramp", help="shape of the X trigger marker")
    ap.add_argument("--rotation", type=int, choices=(0, 90, 180, 270),
                    default=0, help="rotate the displayed scope output")
    ap.add_argument("--mirror", action=argparse.BooleanOptionalAction,
                    default=False, help="mirror the displayed scope output")
    ap.add_argument("--start-at", type=float, default=0.0, metavar="SECONDS",
                    help=argparse.SUPPRESS)
    ap.add_argument("--control", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--profile", action="store_true",
                    help="measure where the time actually goes on THIS machine "
                         "and exit: capture, downscale, trace build")
    ap.add_argument("--device", help="audio output: index or name fragment")
    ap.add_argument("--ask", action="store_true")
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.render_mode not in live_capture_modes(args):
        ap.error("stochastic/stipple require whole traces, one field, and no "
                 "raster geometry/traversal override")

    if args.source == "video" and not args.file:
        ap.error("--source video needs --file")
    if args.source == "camera" and not args.ffmpeg_input:
        ap.error("--source camera needs --ffmpeg-input")
    if args.source not in ("video",) and (args.start_at or args.control):
        ap.error("--start-at and --control require --source video")
    if not math.isfinite(args.start_at) or args.start_at < 0:
        ap.error("--start-at must be finite and non-negative")
    if args.geometry_samples is not None and args.geometry_samples < 2:
        ap.error("--geometry-samples must be at least 2")
    if not math.isfinite(args.physical_dwell) or not 0.0 <= args.physical_dwell <= 1.0:
        ap.error("--physical-dwell must be between 0 and 1")
    if args.traversal_hz is not None and (
            not math.isfinite(args.traversal_hz) or args.traversal_hz <= 0):
        ap.error("--traversal-hz must be finite and greater than zero")
    if args.traversal_hz is not None and not args.scope_trigger:
        ap.error("--traversal-hz requires the scope trigger; remove --no-scope-trigger")
    if args.traversal_hz is not None and args.fields > 1:
        ap.error("--traversal-hz requires --fields 1")
    if (args.geometry_samples is not None or args.traversal_hz is not None) and args.stream:
        ap.error("trajectory controls require whole-trace rendering; remove --stream")
    try:
        channel_pair = tuple(int(value.strip())
                             for value in args.scope_channels.split(","))
        from scope_out import parse_channel_pair, required_output_channels
        channel_pair = parse_channel_pair(channel_pair)
    except (TypeError, ValueError) as exc:
        ap.error(str(exc))
    if not math.isfinite(args.scope_trigger_us) or args.scope_trigger_us <= 0:
        ap.error("--scope-trigger-us must be finite and greater than zero")
    if args.scope_lowpass is not None and (
            not math.isfinite(args.scope_lowpass) or args.scope_lowpass <= 0):
        ap.error("--scope-lowpass must be finite and greater than zero")
    gui_enabled = bool(args.scope_gui)

    region = [int(v) for v in args.region.split(",")] if args.region else None
    if args.source in ("ffmpeg", "camera"):
        raw_grab = ffmpeg_source(width=args.downto, fps=args.capture_fps,
                                 region=region, input_spec=args.ffmpeg_input,
                                 display=args.display,
                                 source_kind=args.source)
        grab = (raw_grab if args.profile else
                Throttled(raw_grab, fps=args.capture_fps,
                          source_kind=args.source, drain=True))
    elif args.source == "screen":
        raw_grab = screen_source(region, downto=args.downto)
        grab = (raw_grab if args.profile else
                Throttled(raw_grab, fps=args.capture_fps,
                          source_kind="screen"))
    elif args.source == "video":
        video = video_source(args.file, downto=args.downto,
                             start_at=args.start_at)
        grab = (video if args.profile else
                Throttled(video.read_latest_due, fps=args.capture_fps,
                          source_kind="video"))
    else:
        raw_grab = test_source()
        grab = (raw_grab if args.profile else
                Throttled(raw_grab, fps=args.capture_fps,
                          source_kind="test"))

    probe = grab()
    print(f"[SCREEN] source {args.source}, {probe.shape[1]}x{probe.shape[0]} "
          "after downscale")

    if args.profile:
        try:
            _profile(args, grab)
        finally:
            if hasattr(grab, "close"):
                grab.close()
            if args.source == "video":
                video.close()
            if getattr(grab, "proc", None) is not None:
                grab.proc.terminate()
        return

    min_channels = required_output_channels(channel_pair, args.scope_x_only)
    scope = Scope(
        fps=args.fps, samples=args.samples, blocksize=args.blocksize,
        device=choose_device(ask=args.ask, device=args.device,
                             min_channels=min_channels),
        lowpass_hz=args.scope_lowpass, x_only=args.scope_x_only,
        channel_pair=channel_pair, trigger=args.scope_trigger,
        trigger_shape=args.scope_trigger_shape,
        yt_trigger_us=args.scope_trigger_us, rotation=args.rotation,
        mirror=args.mirror, physical_dwell=args.physical_dwell)
    if gui_enabled:
        scope.set_tap_fields(max(1, args.fields))
    n = scope.samples_per_frame
    stop = threading.Event()
    render_lock = threading.RLock()
    control_messages = queue.Queue()
    control_thread = None
    producer_thread = None
    if args.control:
        control_thread = threading.Thread(
            target=_control_reader,
            args=(video, control_messages, stop),
            name="scope-video-control", daemon=True)
        control_thread.start()

    # Tone mapping for unknown live content.  Adapting per frame is what makes
    # cells flicker, so this is a slow exponential average -- seconds, not
    # frames.
    lv = {"lo": None, "hi": None}

    def levels_for(g):
        if args.adapt <= 0:
            return None, None
        current = {"lo": lv["lo"], "hi": lv["hi"]}
        lit = g[g > 0.01]
        if lit.size < 16:
            levels = ((current["lo"], current["hi"])
                      if current["lo"] is not None else None)
            return levels, current
        lo_n, _ = positive_percentile(lit,2.0)
        hi_n, _ = positive_percentile(lit,98.0)
        if current["lo"] is None:
            proposed = {"lo": lo_n, "hi": hi_n}
        else:
            a = min(1.0, (1.0 / max(args.fps, 1)) / args.adapt)
            proposed = {
                "lo": current["lo"] + a * (lo_n - current["lo"]),
                "hi": current["hi"] + a * (hi_n - current["hi"]),
            }
        return (proposed["lo"], proposed["hi"]), proposed

    # --- fix the grid once -------------------------------------------------
    # autofit re-derives rows/cols from the fraction of cells surviving trim.
    # On live screen content that fraction moves whenever the picture does, so
    # leaving autofit on per frame meant the grid resized under the image and
    # every cell boundary re-quantized -- cells popping in and out. Same reason
    # mode scope calls calibrate() at startup and holds the result.
    #
    # levels are handled separately above and are deliberately adaptive, but
    # slowly (--adapt, in seconds). Geometry cannot be adaptive at all.
    _probe = np.asarray(grab(), dtype=np.float32)
    geometry_budget = (args.geometry_samples if args.geometry_samples is not None
                       else (3200 if args.traversal_hz is not None else n))
    _grid_rows, _grid_cols = plan_grid(_probe, geometry_budget, density=args.density,
                                       trim=args.trim, rows=args.rows,
                                       fields=max(1, args.fields))
    print(f"[SCREEN] grid fixed at {_grid_cols}x{_grid_rows} "
          f"({geometry_budget * max(1, args.fields) / max(_grid_rows * _grid_cols, 1):.2f} "
          f"samples/cell)"
          + (f", interlace x{args.fields}" if args.fields > 1 else ""))

    emitter = None
    gen = None
    image_emitters = {}
    selected_mode = [args.render_mode]
    prepared_cache = PreparedImageCache()
    if args.stream:
        gen = SweepSource(lum_fn=grab, samples_per_pass=n, gamma=args.gamma,
                          trim=args.trim, density=args.density, rows=args.rows,
                          auto_levels=args.adapt)
        scope.source = BufferedSource(gen, blocksize=max(args.blocksize, 256),
                                      depth=args.buffer_blocks)
        gen(256)
        rws, cls = gen._dims
    else:
        # Whole-trace build: one vectorised pass instead of ~40 tiny per-row
        # ones.  Measured ~1.6 ms per trace against ~12% of a core streaming.
        # One emitter, shared with mode scope. Tuning lives on the object, so
        # a knob added there arrives here without anyone porting it.
        emitter = TraceEmitter(
            scope.samplerate, n,
            gamma=args.gamma, trim=args.trim, density=args.density,
            rows=args.rows, fields=max(1, args.fields),
            border=getattr(args, "border", 0.0),
            oversample=getattr(args, "oversample", 1),
            sweep="alternate", dc_comp=args.dc_comp,
            grid=(_grid_rows, _grid_cols), close_frame=not scope.trigger,
            geometry_samples=args.geometry_samples,
            traversal_hz=args.traversal_hz)
        field_group = FieldGroupLatch(max(1, args.fields))
        if len(live_capture_modes(args)) > 1 and (gui_enabled or args.render_mode != "raster"):
            # Constructors warm compiled kernels before the stream starts,
            # including modes that may be selected interactively later.
            options = dict(gamma=args.gamma, trim=args.trim,
                           dc_comp=args.dc_comp, border=args.border)
            image_emitters = {
                "stochastic": StochasticEmitter(scope.samplerate, n, **options),
                "stipple": StippleEmitter(scope.samplerate, n, **options),
            }

        def push():
            # Gate on the callback having taken the last frame. Without it a
            # frame can be queued over an unconsumed one and dropped while
            # sweep["end"] advances anyway -- so the next trace starts from a
            # position the beam was never at, which creates an unbudgeted
            # full-screen jump.
            # It also guarantees interlaced fields arrive one per trace, in
            # order, instead of one silently replacing the other.
            if not scope.ready():
                return False
            with render_lock:
                mode = selected_mode[0]
                field, captured = _begin_raster_field(
                    field_group, grab, levels_for)
                if mode != "raster":
                    accepted = _push_live_image(
                        scope, image_emitters[mode], captured, mode)
                    if accepted:
                        field_group.accept()
                    return accepted
                handoff = (None if emitter._end is None
                           else emitter._end.copy())
                source_key = PreparedImageCache.versioned_array_key(
                    captured.get("source_version"))
                levels = (captured["levels"] if captured["levels"] is not None
                          else emitter.levels)
                prepared_grid = None
                # Adaptive percentiles intentionally change tone per captured
                # picture; do not fill the bounded cache with one-off grids.
                if (source_key is not None and args.adapt <= 0
                        and args.geometry_samples is None
                        and args.traversal_hz is None):
                    grid = emitter.grid or (None, None)
                    prepared_grid = prepared_cache.raster_grid(
                        captured["lum"], source_key, n=emitter.n,
                        density=emitter.density, trim=emitter.trim,
                        rows=emitter.rows, cols=None,
                        autofit=emitter.autofit, grid_rows=grid[0],
                        grid_cols=grid[1], row_bias=emitter.row_bias,
                        levels=levels, stretch=True, fields=emitter.fields,
                        precondition=emitter.precondition,
                        yt_fixed=(emitter.yt_timing == "fixed"))
                fr = emitter.emit(
                    captured["lum"], levels=captured["levels"],
                    field=field, commit=False, prepared_grid=prepared_grid)
                if fr is not None:
                    return _queue_raster_candidate(
                        scope, emitter, fr, handoff=handoff,
                        field_group=field_group,
                        levels_commit=((lv, captured["level_state"])
                                       if field == 0 else None),
                        identity=_source_presentation_identity(
                            captured, field, args.fields))
            return False

        trace_period = scope.trace_samples / max(scope.samplerate, 1)
        next_deadline = [0.0]
        rws = cls = 0

        def pump():
            while not stop.is_set():
                if not scope.ready():
                    stop.wait(0.001)
                    continue
                now = time.monotonic()
                delay = next_deadline[0] - now
                if delay > 0:
                    stop.wait(min(delay, 0.001))
                    continue
                started = now
                try:
                    produced = push()
                except Exception:
                    produced = False
                if produced:
                    # Anchor to the observed ready boundary. This leaves the
                    # render time available before the next DAC boundary.
                    next_deadline[0] = started + trace_period
                else:
                    # A transient capture/render failure must not turn the
                    # producer into a tight retry loop while the callback is
                    # ready for another frame.
                    stop.wait(0.001)

        producer_thread = threading.Thread(
            target=pump, daemon=True, name="scope-frames")

    gui = None
    try:
        last_reported_adoption_sequence = 0
        scope.stream.start()
        if producer_thread is not None:
            producer_thread.start()
        if gui_enabled:
            from scope_gui import ScopeGUI

            live_state = {
                "trim": args.trim, "density": args.density,
                "gamma": args.gamma, "rows": args.rows or 0,
                "lowpass": args.scope_lowpass or 0.0,
                "mode": args.render_mode, "raster": args.render_mode == "raster",
                "mode_locked": len(live_capture_modes(args)) == 1,
                "clock_locked": True,
                "disabled_sliders": ("ips", "fps", "fields", "density", "rows"),
                "audio_muted": False, "fps": args.fps, "ips": args.fps,
                "fields": args.fields, "available_modes": live_capture_modes(args),
            }
            gui = ScopeGUI(live_state)
        print("[SCREEN] running -- Ctrl+C to stop", flush=True)
        last_report = 0.0
        while not stop.is_set():
            if gui is not None:
                source = getattr(scope, "source", None)
                buffered_samples = (
                    source.buffered_samples
                    if source is not None and
                    hasattr(source, "buffered_samples") else
                    (scope.trace_samples if not scope.ready() else 0))
                buffer_capacity = (
                    source.capacity
                    if source is not None and hasattr(source, "capacity") else
                    scope.trace_samples)
                stream_latency = getattr(scope.stream, "latency", 0.0)
                if isinstance(stream_latency, (tuple, list)):
                    stream_latency = stream_latency[-1] if stream_latency else 0.0
                try:
                    dac_latency_ms = 1000.0 * float(stream_latency)
                except (TypeError, ValueError):
                    dac_latency_ms = 0.0
                metrics = {
                    "device": str(args.device or "Scope output"),
                    "sample_rate": int(scope.samplerate),
                    "trace_hz": scope.samplerate / max(scope.trace_samples, 1),
                    "picture_hz": (0.0 if live_state["mode"] == "stochastic" else scope.samplerate /
                                   max(scope.trace_samples *
                                       max(1, args.fields), 1)),
                    "samples": int(scope.samples_per_frame),
                    "fields": int(args.fields),
                    "grid": (f"{_grid_cols}x{_grid_rows}" if live_state["raster"] else "—"),
                    "dropouts": int(scope.dac_dropouts),
                    "underruns": int(getattr(source, "underruns", 0)),
                    "buffered_ms": 1000.0 * buffered_samples /
                    max(scope.samplerate, 1),
                    "buffer_capacity_ms": 1000.0 * buffer_capacity /
                    max(scope.samplerate, 1),
                    "buffer_kind": ("source" if source is not None and
                                    hasattr(source, "buffered_samples") else
                                    "trace queue"),
                    "dac_latency_ms": dac_latency_ms,
                    "adoption_dac_schedule_offset_ms": getattr(
                        scope, "last_adopted_dac_schedule_offset_ms", None),
                }
                playback_position = (video.playback()["position"]
                                     if args.source == "video" else None)
                metrics.update(_adopted_source_metrics(
                    scope, playback_position_s=playback_position))
                if hasattr(grab, "snapshot_metrics"):
                    metrics["capture_reader"] = grab.snapshot_metrics()
                try:
                    actions = gui.poll(live_state, metrics)
                except Exception as exc:
                    print(f"[SCREEN] GUI update failed: {exc}", flush=True)
                    gui.close()
                    gui = None
                    actions = ()
                for action in actions:
                    if action[0] == "slider":
                        name, value = action[1], action[2]
                        if name == "gamma":
                            live_state["gamma"] = float(value)
                            with render_lock:
                                if emitter is not None:
                                    emitter.gamma = float(value)
                                for image_emitter in image_emitters.values():
                                    image_emitter.gamma = float(value)
                            if gen is not None:
                                gen.configure(gamma=float(value))
                        elif name == "trim":
                            live_state["trim"] = float(value)
                            with render_lock:
                                if emitter is not None:
                                    emitter.trim = float(value)
                                for image_emitter in image_emitters.values():
                                    image_emitter.trim = float(value)
                            if gen is not None:
                                gen.configure(trim=float(value))
                        elif name == "lowpass":
                            with render_lock:
                                scope.lowpass_hz = float(value) or None
                            live_state["lowpass"] = float(value)
                        elif name == "exposure":
                            gui.set_preview_exposure(value)
                    elif action[0] == "mode" and action[1] in live_capture_modes(args) and emitter is not None:
                        with render_lock:
                            selected_mode[0] = action[1]
                            field_group.reset()
                            active = (emitter if action[1] == "raster" else image_emitters[action[1]])
                            active.reset()
                            endpoint = scope.last_accepted_endpoint
                            if endpoint is not None:
                                accept = getattr(active, "accept", None) or active.chain_from
                                accept(endpoint)
                        live_state.update(mode=action[1], raster=action[1] == "raster")
                        scope.set_tap_fields(1)
                    elif action[0] == "audio":
                        audible = bool(action[1])
                        scope.set_output_audio(muted=not audible)
                        live_state["audio_muted"] = not audible
                    elif action[0] == "fullscreen":
                        gui.set_fullscreen(action[1])
                if gui is not None and gui.close_requested:
                    break
            while True:
                try:
                    message = control_messages.get_nowait()
                except queue.Empty:
                    break
                print(json.dumps(message), flush=True)
            if args.control and time.monotonic() - last_report >= 0.25:
                status = video.playback()
                status.update(_adopted_source_metrics(
                    scope, playback_position_s=status["position"]))
                status["adoption_dac_schedule_offset_ms"] = getattr(
                    scope, "last_adopted_dac_schedule_offset_ms", None)
                # Export cumulative counters so clients can calculate measured
                # trace cadence independently of source-identity transitions.
                status["output_trace_count"] = scope.frames_drawn
                status["callback_adoption_count"] = scope.frames_adopted
                status["source_adoption_event_count"] = (
                    scope.distinct_source_adoptions)
                adoption_records = scope.adoption_snapshot()
                new_adoption_count = (
                    scope._adoption_sequence
                    - last_reported_adoption_sequence)
                new_adoptions = (adoption_records[-new_adoption_count:]
                                 if new_adoption_count > 0 else ())
                status["adoption_events"] = [
                    {
                        "source_identity": list(identity[2:5]),
                        "source_kind": identity[7],
                        "selected_at_ns": identity[8],
                        "requested_at_ns": identity[9],
                        "decode_started_at_ns": identity[10],
                        "ready_at_ns": identity[11],
                        "media_position_s": (
                            identity[12] if len(identity) > 12 else None),
                        "adopted_at_ns": adopted_ns,
                    }
                    for identity, adopted_ns in new_adoptions
                    if isinstance(identity, tuple) and len(identity) >= 12]
                last_reported_adoption_sequence = scope._adoption_sequence
                status["capture_reader"] = grab.snapshot_metrics()
                print(json.dumps(status), flush=True)
                last_report = time.monotonic()
            u = getattr(getattr(scope, "source", None), "underruns", 0)
            if u:
                print(f"[SCREEN] {u} underruns -- raise --buffer-blocks or "
                      "lower --fps", flush=True)
                scope.source.underruns = 0
            time.sleep(0.02 if gui is not None else 0.1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            stop.set()
        except NameError:
            pass
        if producer_thread is not None and producer_thread.is_alive():
            producer_thread.join(timeout=5.0)
        proc = getattr(grab, "proc", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)
        if hasattr(grab, "close"):
            grab.close()
        if args.source == "video":
            video.close()
        if gui is not None:
            gui.close()
        cache_stats = (gen.preparation_stats() if gen is not None
                       else prepared_cache.snapshot())
        if any(cache_stats.get(key, 0)
               for key in ("hits", "misses", "bypasses")):
            print("[SCREEN] prepared cache "
                  + json.dumps(cache_stats), flush=True)
        if getattr(scope, "source", None) is not None:
            scope.source.close()
        scope.stream.stop()
        scope.stream.close()


if __name__ == "__main__":
    main()
