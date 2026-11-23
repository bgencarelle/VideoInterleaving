"""
scope_bake.py -- shared vector/raster/stochastic/stipple/fusion scope toolkit.

Both sides import from here and nothing here imports either side:
    utilities/convert_to_xy.py  (offline)  -> geometry helpers, format constants
    scope_display.py            (runtime)  -> XYLibrary, merge

Library format (one directory per baked image folder, all mmap-able):
    verts.npy         (V, 2) int16  vertices in [-1, 1] * Q, screen y-down
    poly_starts.npy   (P+1,) int32  vertex offsets per polyline
    frame_starts.npy  (F+1,) int32  polyline offsets per frame
    flags.npy         (P,)  uint8   1 = closed silhouette loop (matte edge)
    names.json                      source filenames, bake provenance
"""
import copy
import math
import json
from pathlib import Path
import threading

from numba import njit, prange
import numpy as np
from scope_numeric import (area_grid, importance_grid, systematic_samples,
                           candidate_importance, cumulative_mass, preview_rows,
                           positive_percentile, finite_array, circular_filter,
                           warm_scope_numeric, stochastic_stream,
                           sanitize_probability, stretch_grid, threshold_fraction,
                           precondition_grid, raster_points, composite_thumbnail,
                           trace_weights, mux_positions, retime_weighted,
                           normalize_mass, combine_fields, grid_axes,
                           polygon_crossings, segment_midpoints, inside_parity,
                           outside_runs, trace_border, argmax_first, pixel_positions,
                           pixel_route, cloud_route, dwell_route, stipple_geometry,
                           density_polyline, dilate_density, linear_axis,
                           linear_trace, yt_row, clip_x, live_luma,
                           row_budgets, sweep_row, scale_float32, floor_weights,
                           baked_vertices, nearest_path_vertex, subdivision_counts,
                           subdivided_points, alpha_samples, candidate_cloud,
                           overscan_path, border_extension, array_percentile, lit_bounds,
                           mass_search)

Q = 32767.0


# ---------------------------------------------------------------- geometry

@njit(cache=True, nogil=True, fastmath=False)
def path_length(p):
    total = 0.0
    for i in range(1,len(p)):
        total += np.hypot(p[i,0]-p[i-1,0],p[i,1]-p[i-1,1])
    return total


def subdivide(p, max_seg):
    """Insert vertices so no segment exceeds max_seg.  Bake-time step that
    bounds the error of the runtime midpoint-in-matte occlusion test."""
    n,total,changed = subdivision_counts(np.asarray(p),float(max_seg))
    if not changed:
        return p
    return subdivided_points(np.asarray(p),n,total)


def fit_epsilon(contours, budget, closed, lo=0.25, hi=32.0, iters=22):
    """Binary-search the approxPolyDP tolerance that lands on a vertex budget."""
    import cv2
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        n = sum(len(cv2.approxPolyDP(c, mid, closed)) for c in contours)
        if n > budget:
            lo = mid
        else:
            hi = mid
    return hi


def order_paths(paths, closed, start=None):
    """
    Greedy nearest-neighbour tour; a closed loop may be entered at any vertex,
    so rotate each to start at the point nearest the beam.

    closed : one bool for every path, or a per-path sequence of bools.
    start  : beam position now, so successive tours chain instead of each
             restarting from the origin.
    """
    flags = [closed] * len(paths) if isinstance(closed, bool) else list(closed)
    remaining = list(range(len(paths)))
    pos = np.zeros(2) if start is None else np.asarray(start, np.float64)
    out = []
    while remaining:
        best = None
        for r in remaining:
            c = paths[r]
            distance,j = nearest_path_vertex(np.asarray(c),pos,bool(flags[r]))
            cand = (distance,r,j)
            if best is None or cand[0] < best[0]:
                best = cand
        _, r, j = best
        remaining.remove(r)
        c = paths[r]
        if flags[r]:
            c = np.vstack([c[j:], c[:j + 1]])      # rotate, then re-close
        elif j == -1:
            c = c[::-1]
        out.append(c)
        pos = c[-1]
    return out


# ---------------------------------------------------------------- runtime

class XYLibrary:
    """Mmap access to one baked folder.  frame(i) -> (polylines, closed_flags)."""

    def __init__(self, path, flip_y=True):
        p = Path(path)
        # Baked arrays are immutable for this library lifetime. Reopening a bake
        # creates a new version even when its path/index is unchanged.
        self.prepared_source_version = object()
        self.flip_y = flip_y
        self.verts = np.load(p / "verts.npy", mmap_mode="r")
        self.poly = np.load(p / "poly_starts.npy")
        self.fstart = np.load(p / "frame_starts.npy")
        fpath = p / "flags.npy"
        self.flags = np.load(fpath) if fpath.exists() else None
        npath = p / "names.json"
        self.names = json.loads(npath.read_text()) if npath.exists() else []
        meta_path = p / "format.json"
        try:
            self.format = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        except (OSError, ValueError):
            self.format = {}
        tpath = p / "thumbs.npy"
        self.thumbs = np.load(tpath, mmap_mode="r") if tpath.exists() else None
        if self.thumbs is not None:
            if self.thumbs.ndim != 4 or self.thumbs.shape[-1] < 2:
                raise ValueError(
                    f"invalid thumbs.npy shape {self.thumbs.shape}; expected "
                    "(frames, height, width, channels>=2)")
            if len(self.thumbs) != len(self):
                raise ValueError(
                    f"thumbs.npy has {len(self.thumbs)} frames but geometry "
                    f"has {len(self)}; rebake this folder")
        sx = p / "stipple_xy.npy"
        sl = p / "stipple_lae.npy"
        sm = p / "stipple_mass.npy"
        self.stipple_xy = np.load(sx, mmap_mode="r") if sx.exists() else None
        self.stipple_lae = np.load(sl, mmap_mode="r") if sl.exists() else None
        self.stipple_mass = np.load(sm, mmap_mode="r") if sm.exists() else None
        stores = (self.stipple_xy, self.stipple_lae, self.stipple_mass)
        if any(v is not None for v in stores):
            if not all(v is not None for v in stores):
                raise ValueError(f"incomplete stipple candidate store in {p}")
            if (len(self.stipple_xy) != len(self)
                    or len(self.stipple_lae) != len(self)
                    or len(self.stipple_mass) != len(self)):
                raise ValueError(f"stipple candidate frame count differs in {p}")
            if (self.stipple_xy.shape[-1] != 2
                    or self.stipple_lae.shape[-1] != 3
                    or self.stipple_xy.shape[:2] != self.stipple_lae.shape[:2]):
                raise ValueError(f"invalid stipple candidate geometry in {p}")

    def __len__(self):
        return len(self.fstart) - 1

    def frame(self, i):
        a, b = self.fstart[i], self.fstart[i + 1]
        polys, flags = [], []
        for k in range(a, b):
            v = baked_vertices(self.verts[self.poly[k]:self.poly[k+1]],bool(self.flip_y))
            polys.append(v)
            if self.flags is not None:
                flags.append(int(self.flags[k]))
            else:                                  # legacy bake: infer
                flags.append(int(len(v)>2 and v[0,0]==v[-1,0] and v[0,1]==v[-1,1]))
        return polys, flags


    def thumb(self, i):
        """Baked channels, or None.

        Compact v2 bakes are [raw L, alpha]. Interim bakes may be
        [preconditioned raster L, alpha, raw L]; older two-channel files remain
        readable and are treated as legacy raster data without new correction.
        """
        if self.thumbs is None:
            return None
        return np.asarray(self.thumbs[i % len(self.thumbs)])

    def stipple(self, i):
        """Return compact (xy, luminance/alpha/edge, proposal mass)."""
        if self.stipple_xy is None:
            return None
        j = i % len(self.stipple_xy)
        return (np.asarray(self.stipple_xy[j]),
                np.asarray(self.stipple_lae[j]),
                float(self.stipple_mass[j]))

    @property
    def raw_thumbnail(self):
        return self.format.get("thumbnail_channels") == [
            "raw_luminance", "alpha"]

    @property
    def raster_precondition(self):
        return (float(self.format.get("raster_precondition", 0.0))
                if self.raw_thumbnail else 0.0)


def _walk(P, w, n, oversample=1):
    """
    Sample n points along the polyline P, spending time per weights w.

    oversample > 1 generates n*oversample points, bandlimits them circularly,
    and decimates.  Point-sampling a path is not anti-aliased: geometry finer
    than the sample spacing folds back as noise instead of averaging into the
    signal.  Oversample-and-decimate turns that detail into correct low-
    frequency content, which is what lets a grid finer than one cell per
    sample be usable rather than just noisy.  It cannot exceed Nyquist -- the
    gain is accuracy below it, not more bandwidth.
    """
    oversample = max(1, int(oversample))
    if oversample > 1:
        fine = _walk_raw(P, w, n * oversample)
        k = n * oversample
        out = circular_filter(fine, k, 0.45 * n, order=8, dtype=np.float64)
        return out[::oversample]
    return _walk_raw(P, w, n)


@njit(cache=True, nogil=True, fastmath=False)
def _walk_raw_numba_kernel(P, weights, n):
    """Sample a weighted polyline at uniformly spaced cumulative weights."""
    count = len(weights)
    cumulative = np.empty(count + 1, dtype=np.float64)
    cumulative[0] = 0.0
    for i in range(count):
        cumulative[i + 1] = cumulative[i] + weights[i]

    out = np.empty((n, P.shape[1]), dtype=np.float64)
    if n == 0:
        return out

    step = cumulative[count] / n
    segment = 0
    for sample in range(n):
        t = step * sample
        # `side="right"` in searchsorted means an exact boundary belongs to
        # the following segment. Since t is monotonic, advance once rather
        # than doing a binary search for every output sample.
        while segment < count - 1 and cumulative[segment + 1] <= t:
            segment += 1
        fraction = (t - cumulative[segment]) / weights[segment]
        for axis in range(P.shape[1]):
            out[sample, axis] = (
                P[segment, axis]
                + fraction * (P[segment + 1, axis] - P[segment, axis]))
    return out


def _warm_walk_raw_numba(n, point_dtype=np.float32):
    """Compile the selected raster sampler signature before stream start."""
    warm_scope_numeric()
    points = np.zeros((2, 2), dtype=point_dtype)
    weights = np.ones(1, dtype=np.float64)
    _walk_raw_numba_kernel(points, weights, max(0, int(n)))


@njit(cache=True, nogil=True, fastmath=False)
def _trajectory_sample_kernel(path, phase, step, count):
    """Sample a cyclic, canonical path at time-parameterized positions."""
    out = np.empty((count, 2), dtype=np.float32)
    size = len(path)
    for i in range(count):
        p = (phase + i * step) % 1.0
        x = p * size
        j = int(x)
        f = x - j
        k = (j + 1) % size
        out[i, 0] = path[j, 0] + f * (path[k, 0] - path[j, 0])
        out[i, 1] = path[j, 1] + f * (path[k, 1] - path[j, 1])
    return out


def _walk_raw(P, w, n):
    """Sample n points along polyline P, spending time per arbitrary weights w.
    Same machinery as rasterize, but the weights are brightness rather than
    length -- that substitution is what turns dwell time into intensity."""
    w = floor_weights(np.asarray(w,np.float64))
    P = np.ascontiguousarray(P)
    return _walk_raw_numba_kernel(P, w, int(n))


def _box(img, rows, cols):
    """Exact area-average resample to EXACTLY (rows, cols).

    An integer block-size reshape does not work here: the source is a 96-wide
    thumbnail, so h // rows rounds and the derived grid comes out finer than
    asked for -- more cells than samples, which renders as broken sparse rows
    instead of solid scanlines.  Summed-area table gives the exact grid.
    """
    return area_grid(np.ascontiguousarray(img),int(rows),int(cols),img.dtype==np.float32)


class SweepSource:
    """
    Continuous sample generator: always draws whatever index is CURRENT, from
    wherever the beam happens to be.

    Frame-buffer playback can only change content at a trace boundary, because
    swapping mid-trace teleports the beam.  This generates row by row instead
    and re-reads the state at every row boundary, so an index change takes
    effect within ~1/rows of a trace rather than waiting up to a full one.

    Splicing granularity is a row and not a sample on purpose: sample #k of two
    different indices are at different screen positions (dwell follows
    brightness), but row 30 is at the same y in every frame, so resuming there
    with new content is geometrically continuous.

    Raster switches content at row boundaries.  Vector has no positional
    correspondence between frames, so it switches at pass boundaries -- as does
    the raster/vector mode itself, which always finishes the pass it is in.
    """

    def __init__(self, state_fn=None, samples_per_pass=1600, gamma=2.2,
                 floor=0.012, trim=0.02, density=1.0, rows=None, bbox=None,
                 level=0.9, grid_rows=None, grid_cols=None, levels=None,
                 lum_fn=None, auto_levels=0.0, precondition=0.0,
                 invert=False, rotation=0, alternate=True):
        """
        lum_fn      : optional callable returning an (H, W) float array in
                      0..1 -- any live source (screen grab, camera, video,
                      a buffer you drew).  Supplied instead of state_fn, it
                      decouples this generator from the baked libraries and
                      makes the scope a general low-resolution display.
        auto_levels : seconds of time constant for adapting the tone mapping
                      to unknown content.  Live input cannot be pre-scanned,
                      but adapting per frame is what caused the flecking, so
                      this is a deliberately SLOW exponential average -- a few
                      seconds, not a few frames.  0 disables.
        """
        self.lum_fn = lum_fn
        self._config_lock = threading.RLock()
        self.auto_levels = float(auto_levels)
        self._lv = None                     # running (lo, hi)
        self._lum_shape = None
        # grid_rows/grid_cols/levels come from calibrate(), exactly as in
        # raster_frame.  Without them this path computes its own grid and a
        # per-frame stretch, so realtime looked different from frame mode --
        # fewer rows and drifting levels.
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.levels = levels
        self.state_fn = state_fn
        self.n_pass = max(64, int(samples_per_pass))
        self.gamma, self.floor, self.trim = gamma, floor, trim
        self.density, self.rows_override, self.bbox = density, rows, bbox
        self.level = level
        self.precondition = max(0.0, float(precondition))
        self.invert = bool(invert)
        self.rotation = int(rotation) % 360
        if self.rotation % 90:
            raise ValueError("scope rotation must be a multiple of 90 degrees")
        self.alternate = bool(alternate)
        self._out = np.zeros((0, 2), np.float32)
        self._plan = None
        self._budgets = None
        self._row_i = 0
        self._reverse = False
        self._last = None
        self._grid_key = None
        self._grid = None
        self._composite_cache_key = None
        self._composite_grid_hits = 0
        self._composite_grid_misses = 0
        self._composite_grid_bypasses = 0
        self._live_prepared_key = None
        self._live_grid_hits = 0
        self._live_grid_misses = 0
        self._live_grid_bypasses = 0
        self.passes = 0
        self.row_switches = 0

    def set_rotation(self, degrees, grid=None):
        """Rotate source content while leaving X as the fast sweep axis."""
        angle = int(degrees) % 360
        if angle % 90:
            raise ValueError("scope rotation must be a multiple of 90 degrees")
        with self._config_lock:
            self.rotation = angle
            if grid is not None:
                self.grid_rows, self.grid_cols = map(int, grid)
            self._out = np.zeros((0, 2), np.float32)
            self._plan = None
            self._budgets = None
            self._row_i = 0
            self._last = None
            self._lum_shape = None
            self._grid_key = None
            self._grid = None
            self._composite_cache_key = None
            self._live_prepared_key = None

    def configure(self, **values):
        """Publish live tuning atomically at a generator-chunk boundary."""
        allowed = {
            "gamma", "floor", "trim", "density", "rows_override", "invert",
            "grid_rows", "grid_cols", "levels", "precondition",
        }
        unknown = set(values) - allowed
        if unknown:
            raise TypeError("unsupported SweepSource setting(s): "
                            + ", ".join(sorted(unknown)))
        with self._config_lock:
            def comparable(value):
                # Tuning values are scalars/optional tuples; array controls
                # compare as lists without object-array numerical dispatch.
                if isinstance(value,np.ndarray):value=value.tolist()
                return tuple(value) if isinstance(value,(list,tuple)) else value
            values = {
                name: value for name, value in values.items()
                if comparable(getattr(self,name)) != comparable(value)
            }
            if not values:
                return
            for name, value in values.items():
                setattr(self, name, value)
            if set(values) & {
                    "density", "rows_override", "invert", "grid_rows",
                    "grid_cols", "levels", "precondition"}:
                self._grid_key = None
                self._grid = None
                self._composite_cache_key = None
                self._live_prepared_key = None
                self._lum_shape = None
            if set(values) & {
                    "gamma", "trim", "floor", "density", "rows_override",
                    "grid_rows", "grid_cols", "levels", "precondition"}:
                self._plan = None
                self._budgets = None

    def _live(self):
        """Grid + axes for a live luminance source."""
        snapshot = getattr(self.lum_fn, "prepared_snapshot", None)
        if callable(snapshot):
            source, source_version = snapshot()
        else:
            source, source_version = self.lum_fn(), None
        lum = np.asarray(source, dtype=np.float32)
        lum = live_luma(lum,bool(self.invert))
        if self.rotation:
            lum = np.rot90(lum, k=self.rotation // 90)
            lum = np.ascontiguousarray(lum)
        h, w = lum.shape
        settings = (
            (h, w), self.rotation, self.invert, self.n_pass,
            float(self.density), self.rows_override, self.grid_rows,
            self.grid_cols,
            None if self.levels is None else tuple(map(float, self.levels)),
            float(self.precondition))
        live_key = None
        if source_version is not None and self.auto_levels <= 0:
            try:
                hash(source_version)
                live_key = (source_version, settings)
            except TypeError:
                pass
        if live_key is None:
            self._live_grid_bypasses += 1
        elif live_key == self._live_prepared_key and self._grid is not None:
            self._live_grid_hits += 1
            return self._grid
        else:
            self._live_grid_misses += 1

        if self._grid is None or self._lum_shape != (h, w):
            self._lum_shape = (h, w)
            aspect = h / float(w)
            if self.grid_rows and self.grid_cols:
                rws, cls = int(self.grid_rows), int(self.grid_cols)
            else:
                cells = max(64.0, self.n_pass / max(self.density, 0.25))
                cls = max(8, int(math.sqrt(cells / max(aspect, 1e-6))))
                rws = max(6, int(round(cls * aspect)))
            rws, cls = min(rws, h), min(cls, w)
            sx = 1.0 if w >= h else w / float(h)
            sy = 1.0 if h >= w else h / float(w)
            xs,ys = grid_axes(w,h,rws,cls)
            self._axes = (np.asarray(xs,dtype=np.float32),np.asarray(ys,dtype=np.float32))
            self._dims = (rws, cls)
        rws, cls = self._dims
        g = _box(lum, rws, cls)

        if self.levels is not None:
            lo, hi = self.levels
        elif self.auto_levels > 0:
            lo_n, count = positive_percentile(g,2.0,0.01)
            if count > 16:
                hi_n, _ = positive_percentile(g,98.0,0.01)
                if self._lv is None:
                    self._lv = (lo_n, hi_n)
                else:
                    # slow: a few seconds, so the picture cannot breathe
                    a = min(1.0, (self.n_pass / 48000.0) / self.auto_levels)
                    self._lv = (self._lv[0] + a * (lo_n - self._lv[0]),
                                self._lv[1] + a * (hi_n - self._lv[1]))
            lo, hi = self._lv if self._lv else (0.0, 1.0)
        else:
            lo, hi = 0.0, 1.0
        if hi > lo:
            g = stretch_grid(g,float(lo),float(hi))
        g = _precondition_grid(g, self.precondition)
        xs, ys = self._axes
        self._grid = (g, xs, ys)
        self._live_prepared_key = live_key
        return self._grid

    def preparation_stats(self):
        """One-entry live/composite grid cache counters and retained bytes."""
        retained = (0 if self._grid is None else
                    sum(int(np.asarray(a).nbytes) for a in self._grid))
        return dict(
            hits=self._live_grid_hits + self._composite_grid_hits,
            misses=self._live_grid_misses + self._composite_grid_misses,
            bypasses=self._live_grid_bypasses + self._composite_grid_bypasses,
            live_grid_hits=self._live_grid_hits,
            live_grid_misses=self._live_grid_misses,
            live_grid_bypasses=self._live_grid_bypasses,
            composite_grid_hits=self._composite_grid_hits,
            composite_grid_misses=self._composite_grid_misses,
            composite_grid_bypasses=self._composite_grid_bypasses,
            entries=int(self._live_prepared_key is not None or
                        self._composite_cache_key is not None),
            bytes=retained)

    # -- geometry -------------------------------------------------------
    def _composite(self, st):
        ml, fl = st.get("main"), st.get("float")
        mi, fi = st.get("mi", 0), st.get("fi", 0)
        versions = tuple(
            getattr(lib, "prepared_source_version", None)
            if lib is not None else None for lib in (ml, fl))
        settings = (
            self.invert, self.rotation,
            None if self.bbox is None else tuple(self.bbox),
            self.n_pass, float(self.density), self.rows_override,
            self.grid_rows, self.grid_cols,
            None if self.levels is None else tuple(map(float, self.levels)),
            float(self.precondition))
        key = (id(ml), mi, id(fl), fi, versions, settings)
        versioned = all(lib is None or version is not None
                        for lib, version in zip((ml, fl), versions))
        cache_key = key if versioned else None
        if (cache_key is not None and cache_key == self._composite_cache_key
                and self._grid is not None):
            self._composite_grid_hits += 1
            return self._grid
        if cache_key is None:
            self._composite_grid_bypasses += 1
        else:
            self._composite_grid_misses += 1
        self._grid_key = key
        self._composite_cache_key = cache_key
        tm = ml.thumb(mi) if ml is not None and len(ml) else None
        tf = fl.thumb(fi) if fl is not None and len(fl) else None
        if tm is None and tf is None:
            self._grid = None
            self._composite_cache_key = None
            return None

        # Preserve this legacy path's float-only choice on mismatched geometry.
        if tf is not None and tm is not None and tf.shape[:2] != tm.shape[:2]: tm = None
        empty = np.empty((0,0,2),np.uint8)
        lum = composite_thumbnail(tm if tm is not None else empty,
                                  tf if tf is not None else empty,False,bool(self.invert))

        if self.bbox is not None:
            hh, ww = lum.shape
            x0, y0, x1, y1 = self.bbox
            lum = lum[int(y0 * hh):max(int(y1 * hh), int(y0 * hh) + 1),
                      int(x0 * ww):max(int(x1 * ww), int(x0 * ww) + 1)]

        if self.rotation:
            lum = np.rot90(lum, k=self.rotation // 90)
            lum = np.ascontiguousarray(lum)

        h, w = lum.shape
        aspect = h / float(w)
        if self.grid_rows and self.grid_cols:
            rws, cls = int(self.grid_rows), int(self.grid_cols)
        else:
            cells = max(64.0, self.n_pass / max(self.density, 0.25))
            if self.rows_override:
                rws = int(self.rows_override); cls = max(8, int(cells / rws))
            else:
                cls = max(8, int(math.sqrt(cells / max(aspect, 1e-6))))
                rws = max(6, int(round(cls * aspect)))
        g = _box(lum, min(rws, h), min(cls, w))
        rws, cls = g.shape
        g = _stretch_grid(g, self.levels)
        g = _precondition_grid(g, self.precondition)
        sx = 1.0 if w >= h else w / float(h)
        sy = 1.0 if h >= w else h / float(w)
        xs,ys = grid_axes(w,h,rws,cls)
        self._grid = (g,xs,ys)
        return self._grid

    def _start_pass(self, st):
        grid = self._live() if self.lum_fn is not None else self._composite(st)
        if grid is None:
            return False
        g, xs, ys = grid
        seq = list(range(g.shape[0]))
        if self._reverse:
            seq = seq[::-1]

        # Allocate each row a share of the pass proportional to how much light
        # it carries.  Equal shares give a dim wide row the same beam time as a
        # bright narrow one, which flattens the tone and blooms the edges --
        # that is why realtime used to look unlike frame mode.
        self._budgets = row_budgets(g,bool(self._reverse),float(self.trim),
                                   float(self.floor),float(self.gamma),int(self.n_pass))

        self._plan = seq
        self._row_i = 0
        self.passes += 1
        return True

    def _row_samples(self, st, r, budget):
        """Points+weights for one row of the CURRENT source, chained from the
        beam's present position."""
        grid = self._grid if self.lum_fn is not None else self._composite(st)
        if grid is None:
            return None
        g, xs, ys = grid
        if r >= g.shape[0]:
            return None
        start = np.zeros(2,np.float64) if self._last is None else np.asarray(self._last,dtype=np.float64)
        P,W = sweep_row(g,xs,ys,int(r),float(self.trim),float(self.floor),float(self.gamma),start,self._last is not None)
        if len(P) < 2:
            return None
        return _walk(P, W, max(2, int(budget)))

    # -- audio callback interface --------------------------------------
    def __call__(self, n):
        # The source worker, not the audio callback, owns this lock. GUI/input
        # tuning takes effect atomically between generated blocks rather than
        # mixing old/new parameters partway through a trajectory chunk.
        with self._config_lock:
            return self._generate(n)

    def _generate(self, n):
        while len(self._out) < n:
            st = self.state_fn() if self.state_fn is not None else None
            if self._plan is None or self._row_i >= len(self._plan):
                self._reverse = (not self._reverse) if self.alternate else False
                if not self._start_pass(st):
                    return np.zeros((n, 2), np.float32)
            budget = int(self._budgets[self._row_i]) if self._budgets is not None \
                else max(2, self.n_pass // max(len(self._plan), 1))
            before = self._grid_key
            chunk = self._row_samples(st, self._plan[self._row_i], budget)
            if before is not None and self._grid_key != before:
                self.row_switches += 1
            self._row_i += 1
            if chunk is None:
                continue
            self._last = chunk[-1]
            self._out = np.vstack([self._out, scale_float32(chunk,float(self.level))])
        out, self._out = self._out[:n], self._out[n:]
        return np.ascontiguousarray(out, dtype=np.float32)


def apply_overscan(P, W, travel, overscan, level=0.9, travel_frac=0.12):
    """
    Route travel moves OFF-SCREEN so they are invisible, without a Z channel.

    The trick is deflection, not amplitude: scale the picture down to +-level/
    overscan and turn the scope's V/div up so that range fills the screen.
    Anything beyond it deflects past the phosphor and simply is not drawn.
    Costs nothing in resolution -- the DAC has 16 bits and a tube resolves
    maybe 9.

    Scaling amplitude ALONE achieves nothing: the reconstruction filter is
    linear, so a jump's settling time is set by bandwidth, not by how big it
    is.  What buys you the blanking is the off-screen excursion.

    Each travel segment A->B becomes A -> A' -> B' -> B, where A' and B' are
    pushed out through the NEAREST edge.  Nearest matters: in a serpentine the
    row ends already sit at the content's left/right extremes, so the visible
    stub is short.

    IMPORTANT: the beam only goes where samples put it.  The DAC interpolates
    between consecutive samples, so if the excursion gets no samples the beam
    simply slides from A to B straight across the screen and the waypoints do
    nothing.  Blanking this way therefore COSTS samples -- travel_frac is the
    share of the budget spent getting off-screen and back.  That is the trade:
    a slice of the sample budget in exchange for removing the travel ink.

    Caveat: the output is AC-coupled, so brief excursions shift the mean and
    therefore the image's position slightly.  Travel is a few percent of
    samples, so the shift is small, but it does vary with content.
    """
    if overscan <= 1.0:
        return P, W
    points,weights,count = overscan_path(np.asarray(P),np.asarray(W),
        np.asarray(travel,dtype=np.bool_),float(overscan),float(travel_frac))
    return points,(weights if count else W)


def plan_grid(lum, n, density=1.0, trim=0.02, rows=None, cols=None,
              aspect=None, fields=1, autofit=True, row_bias=1.0):
    """Decide the grid ONCE. Returns (rows, cols).

    Face features are mostly HORIZONTAL edges -- eyelids, brow, lip line,
    nostril, the beard boundary -- and a horizontal edge is resolved by
    VERTICAL sampling, i.e. by rows.  The MTF measurement says vertical is the
    strong axis by roughly 4x at 8 cycles, so rows are also the cheap axis.
    Both point the same way, and on a real face autofit's square-cell split
    lands about 30% short: it picked 63x85 where 29x110 reads visibly sharper
    at the same sample cost.

    row_bias multiplies rows and divides columns by the same factor, so the
    cell COUNT is unchanged and the sample budget is untouched -- only the
    shape of the cells moves.  1.0 is the old behaviour.  ~1.3 is the measured
    sweet spot for faces.  Past ~1.6 the columns get too few and the mouth
    smears horizontally while the silhouette goes blocky, so this is a real
    optimum and not a "more is better" knob.

    Extracted from render_luma so there is exactly one implementation of the
    sizing rule.  scope_screen.py needs it to fix a grid at startup the way
    calibrate() does for mode scope; before this it re-derived the grid on
    every frame, which is the per-frame adaptation that shows as flicker.
    """
    lum = np.asarray(lum, dtype=np.float32)
    if aspect is None:
        aspect = lum.shape[0] / max(lum.shape[1], 1)
    density = max(float(density), 0.25)
    fields = max(1, int(fields))
    cells = max(64.0, n * fields / density)
    if rows and cols:
        rows, cols = int(rows), int(cols)
    elif rows:
        rows = int(rows)
        cols = max(8, int(cells / rows))
    elif cols:
        cols = int(cols)
        rows = max(6, int(cells / cols))
    else:
        cols = max(8, int(math.sqrt(cells / max(aspect, 1e-6))))
        rows = max(6, int(round(cols * aspect)))

    if autofit and trim > 0:
        probe = _box(lum, rows, cols)
        probe = _stretch_grid(probe)
        frac = threshold_fraction(probe,float(trim))
        if 0.05 < frac < 0.95:
            grow = min(1.0 / math.sqrt(frac), 2.5)
            rows = max(6, min(int(round(rows * grow)), lum.shape[0]))
            cols = max(8, min(int(round(cols * grow)), lum.shape[1]))
    rows, cols = _apply_row_bias(rows, cols, row_bias, lum.shape)
    return int(rows), int(cols)


def _apply_row_bias(rows, cols, bias, shape):
    """Trade columns for rows at constant cell count."""
    b = float(bias)
    if b == 1.0 or b <= 0:
        return rows, cols
    k = math.sqrt(b)
    return (max(6, min(int(round(rows * k)), shape[0])),
            max(8, min(int(round(cols / k)), shape[1])))


@njit(cache=True, nogil=True, fastmath=False)
def _preview_splat_kernel(samples, size, max_split, out):
    """Accumulate energy-preserving pixel splats without expanded index arrays."""
    count = len(samples) - 1
    x0 = np.empty(count, dtype=np.float32)
    y0 = np.empty(count, dtype=np.float32)
    x1 = np.empty(count, dtype=np.float32)
    y1 = np.empty(count, dtype=np.float32)
    subdivisions = np.empty(count, dtype=np.int32)
    out.fill(0.0)
    total = 0
    scale = np.float32(0.5 * (size - 1))
    one = np.float32(1.0)
    for i in range(count):
        x0[i] = np.float32(np.float32(samples[i, 0] + one) * scale)
        y0[i] = np.float32(np.float32(one - samples[i, 1]) * scale)
        x1[i] = np.float32(np.float32(samples[i + 1, 0] + one) * scale)
        y1[i] = np.float32(np.float32(one - samples[i + 1, 1]) * scale)
        dx = np.float32(x1[i] - x0[i])
        dy = np.float32(y1[i] - y0[i])
        distance = np.float32(np.sqrt(np.float32(dx * dx + dy * dy)))
        pieces = int(np.ceil(distance))
        if pieces < 1:
            pieces = 1
        if pieces > max_split:
            pieces = max_split
        subdivisions[i] = pieces
        total += pieces

    for i in range(count):
        pieces = subdivisions[i]
        weight = np.float32(1.0 / pieces)
        dx = np.float32(x1[i] - x0[i])
        dy = np.float32(y1[i] - y0[i])
        for j in range(pieces):
            t = (j + 0.5) / pieces
            x = int(x0[i] + dx * t)
            y = int(y0[i] + dy * t)
            if x < 0:
                x = 0
            elif x >= size:
                x = size - 1
            if y < 0:
                y = 0
            elif y >= size:
                y = size - 1
            out[y, x] += float(weight)
    return total


@njit(cache=True, nogil=True, fastmath=False, parallel=True)
def _preview_tonemap_kernel(acc, gain, out):
    """Fuse preview tone curve, phosphor tint, clip and uint8 conversion."""
    h, w = acc.shape
    g = np.float32(gain)
    for y in prange(h):
        for x in range(w):
            value = np.float32(acc[y, x])
            v = np.float32(1.0 - np.exp(np.float32(-value * g)))
            red = np.float32(v * np.float32(0.35) + np.float32(0.03))
            green = np.float32(v + np.float32(0.03))
            blue = np.float32(v * np.float32(0.25) + np.float32(0.03))
            if red < 0.0:
                red = 0.0
            elif red > 1.0:
                red = 1.0
            if green < 0.0:
                green = 0.0
            elif green > 1.0:
                green = 1.0
            if blue < 0.0:
                blue = 0.0
            elif blue > 1.0:
                blue = 1.0
            out[y, x, 0] = np.uint8(red * np.float32(255.0))
            out[y, x, 1] = np.uint8(green * np.float32(255.0))
            out[y, x, 2] = np.uint8(blue * np.float32(255.0))
    return out


class PreviewWorkspace:
    """Reusable bounded scratch/output storage for repeated preview renders."""

    def __init__(self, size):
        self.size = int(size)
        if self.size < 1:
            raise ValueError("preview size must be positive")
        shape = (self.size, self.size)
        self.splats = np.empty(shape, dtype=np.float64)
        self.blur_input = np.empty(shape, dtype=np.float32)
        self.blurred = np.empty(shape, dtype=np.float32)
        self.rgb = np.empty(shape + (3,), dtype=np.uint8)
        self.filter_size = (self.size + 1) // 2
        low_shape = (self.filter_size, self.filter_size)
        self.filter_input = np.empty(low_shape, dtype=np.float32)
        self.filter_blurred = np.empty(low_shape, dtype=np.float32)


def _warm_preview_kernels():
    """Compile preview production signatures before preview rendering starts."""
    warm_scope_numeric()
    workspace = PreviewWorkspace(8)
    _preview_splat_kernel(
        np.zeros((2, 2), dtype=np.float32), 8, 4, workspace.splats)
    _preview_tonemap_kernel(
        workspace.blur_input, 1.0, workspace.rgb)


def preview_frame(samples, size=384, spot=None, exposure=1.0, max_split=192,
                  workspace=None):
    """Simulated scope screen: splat beam positions, blur, tonemap, tint green.

    THE SPLAT IS THE POINT.  Dwell is brightness -- render_luma spends more
    SAMPLES on brighter cells, it does not vary any intensity value.  So the
    image only appears if each sample deposits energy independently and they
    accumulate.  Drawing the path as a connected polyline instead gives every
    segment the same brightness however many samples were on it, which throws
    the picture away and leaves an outline of the rows with the retrace
    diagonals as the brightest thing on screen.

    Subdivision is per segment and measured in PIXELS.  One sample-time is one
    unit of energy no matter how far the beam travelled during it, so a segment
    covering d pixels gets ceil(d) splats of weight 1/ceil(d): total energy
    stays 1, and energy per pixel comes out as 1/d.  That is the physics --
    fast travel is faint, slow travel is bright -- and it is gapless.

    The previous fixed interpolation (a constant count per segment, budget //
    n, capped at 24) could not do this.  Constant subdivision means a long
    segment gets the same few points as a short one, so at size 700 roughly a
    fifth of segments were left with gaps of up to 19 px.  Travel strokes came
    out as dotted lines rather than faint continuous ones, and the effect got
    worse the larger the render, which is the opposite of what raising the
    resolution is for.

    Canonical implementation.  test_scope_pair.render_trace() delegates here
    rather than keeping a second copy -- SweepSource and raster_frame already
    demonstrated what happens when one algorithm has two implementations.
    """
    import cv2
    size = int(size)
    if size < 1:
        raise ValueError("preview size must be positive")
    samples = np.asarray(samples, dtype=np.float32)
    if len(samples) < 2:
        return np.zeros((size, size, 3), np.uint8)

    if spot is None:
        # SCALE THE SPOT WITH THE ROW PITCH, not with nothing.
        #
        # A fixed 1.2 px blur was a quarter of the row pitch at 256 px and a
        # sixteenth of it at 700, so raising the render size opened the gaps
        # between scanlines instead of resolving more detail -- which is why
        # the SMALLEST preview looked best. That is backwards: the small one
        # was not sharper, it was accidentally doing what a real beam spot
        # does, filling the space between rows.
        #
        # A CRT's spot is a fixed fraction of the screen, so it always
        # overlaps its neighbours by the same amount however many lines are
        # drawn. Matching that means deriving the spot from the row pitch.
        _rows = preview_rows(samples)
        # 0.40: at k=0.18 the rows stay separate and you get the outline
        # look; 0.24 and 0.30 still show the sweep as a stack of bars; 0.40 is
        # the first value where a face reads as a face. Erring soft is correct
        # here -- a real tube's spot overlaps generously, and the eye recovers
        # detail from a soft continuous image far better than from a sharp
        # discontinuous one.
        spot = max(0.6, 0.40 * size / _rows)

    samples = np.ascontiguousarray(samples, dtype=np.float32)
    if workspace is None:
        workspace = PreviewWorkspace(size)
    elif workspace.size != size:
        raise ValueError("preview workspace size does not match requested size")
    _splat_count = _preview_splat_kernel(
        samples, size, int(max_split), workspace.splats)
    np.copyto(workspace.blur_input, workspace.splats, casting="unsafe")
    if size >= 1024 and spot >= 4.0:
        # At large image-only sizes the full-resolution Gaussian dominates
        # preview latency. Filter at half linear resolution and reconstruct;
        # measured RGB error on decoded content stays sub-code-value on average.
        filtered = cv2.resize(
            workspace.blur_input,
            (workspace.filter_size, workspace.filter_size),
            dst=workspace.filter_input,
            interpolation=cv2.INTER_AREA)
        cv2.GaussianBlur(filtered, (0, 0), spot * 0.5,
                         dst=workspace.filter_blurred)
        cv2.resize(workspace.filter_blurred, (size, size),
                   dst=workspace.blurred, interpolation=cv2.INTER_LINEAR)
    else:
        cv2.GaussianBlur(workspace.blur_input, (0, 0), spot,
                         dst=workspace.blurred)
    acc = workspace.blurred
    percentile, count = positive_percentile(acc, 75.0)
    gain = exposure * 2.5 / max(percentile, 1e-6) if count else 1.0
    return _preview_tonemap_kernel(acc, gain, workspace.rgb)


def calibrate(main_libs, float_libs, n_samples, density=1.0, trim=0.02,
              rows=None, cols=None, bbox=None, frames=24, fields=1,
              row_bias=1.0, autofit=True, invert=False):
    """
    Compute a FIXED grid size and tone mapping from a sample of the sequence.

    Both the autofit grid and the percentile stretch were being derived per
    frame, which makes the output depend on content statistics that move.  The
    grid changing by even one row re-quantizes every cell boundary, and a
    drifting stretch pushes cells back and forth across the trim threshold --
    they wink in and out as black flecks.  Same failure as auto-exposure
    flicker in video.

    Sampling once and holding the result fixes both.  Returns a dict to pass
    to raster_frame as grid_rows / grid_cols / levels.
    """
    libs = [l for l in (list(main_libs) + list(float_libs)) if l is not None
            and getattr(l, "thumbs", None) is not None]
    if not libs:
        return {}

    ref = libs[0].thumb(0)
    h, w = ref.shape[0], ref.shape[1]
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        h = max(1, int(y1 * h) - int(y0 * h))
        w = max(1, int(x1 * w) - int(x0 * w))
    aspect = h / float(w)

    # n_samples is per TRACE; a picture costs n_samples*fields.  Must match
    # render_luma's sizing exactly or the calibrated grid and the live grid
    # disagree and the picture re-quantizes the moment calibration lands.
    cells = max(64.0, n_samples * max(1, int(fields)) / max(density, 0.25))
    if rows and cols:
        r0, c0 = int(rows), int(cols)
    elif rows:
        r0 = int(rows); c0 = max(8, int(cells / r0))
    elif cols:
        c0 = int(cols); r0 = max(6, int(cells / c0))
    else:
        c0 = max(8, int(math.sqrt(cells / max(aspect, 1e-6))))
        r0 = max(6, int(round(c0 * aspect)))

    # gather luminance statistics and trim coverage over a spread of frames
    los, his, fracs = [], [], []
    for lib in libs[:8]:
        F = len(lib.thumbs)
        step = max(1, F // max(frames // max(len(libs[:8]), 1), 1))
        for i in range(0, F, step):
            t = np.asarray(lib.thumb(i))
            v = composite_thumbnail(t,np.empty((0,0,2),np.uint8),False,bool(invert))
            g = _box(v, min(r0, v.shape[0]), min(c0, v.shape[1]))
            lo,count = positive_percentile(g,2.0,0.01)
            if count < 16:
                continue
            hi,_ = positive_percentile(g,98.0,0.01)
            los.append(lo); his.append(hi)
            if hi > lo:
                fracs.append(threshold_fraction(stretch_grid(g,lo,hi),float(trim)))

    out = {}
    if los:
        # median, not mean: robust to the odd blank or blown-out frame
        out["levels"] = (array_percentile(np.asarray(los),50.0),array_percentile(np.asarray(his),50.0))
    if autofit and fracs and trim > 0:
        frac = array_percentile(np.asarray(fracs),50.0)
        if 0.05 < frac < 0.95:
            grow = min(1.0 / math.sqrt(frac), 2.5)
            r0 = int(round(r0 * grow)); c0 = int(round(c0 * grow))
    # must match render_luma/plan_grid exactly, bias included, or the
    # calibrated grid and the live grid disagree
    r0, c0 = _apply_row_bias(r0, c0, row_bias, ref.shape)
    out["grid_rows"] = max(6, min(r0, ref.shape[0]))
    out["grid_cols"] = max(8, min(c0, ref.shape[1]))
    return out


def content_bbox(libs, samples=24, thresh=0.06, pad=0.01):
    """
    Union bounding box of visible content across libraries, as fractions
    (x0, y0, x1, y1).  Computed ONCE over a sample of frames so the framing is
    identical for every frame and every folder -- a per-frame bbox would make
    the image breathe and jump on folder switches.
    """
    y0 = x0 = 1.0
    y1 = x1 = 0.0
    for lib in libs:
        if lib is None or getattr(lib, "thumbs", None) is None:
            continue
        F = len(lib.thumbs)
        step = max(1, F // max(samples, 1))
        for i in range(0, F, step):
            t = np.asarray(lib.thumb(i))
            v=composite_thumbnail(t,np.empty((0,0,2),np.uint8),False,False)
            left,top,right,bottom=lit_bounds(v,float(thresh))
            if right<=left or bottom<=top:
                continue
            y0=min(y0,top);y1=max(y1,bottom)
            x0=min(x0,left);x1=max(x1,right)
    if y1 <= y0 or x1 <= x0:
        return None
    return (max(0.0, x0 - pad), max(0.0, y0 - pad),
            min(1.0, x1 + pad), min(1.0, y1 + pad))


def apply_trace_border(frame, fraction, aspect=1.0, level=0.9):
    """Replace a trace tail with a full-extent rectangle, then rejoin it.

    Continuous renderers cannot append samples: their output buffer is the
    DAC's fixed time budget. They also must not reset hidden walk/route state
    at each border. Generate the complete image trace first, spend
    ``fraction`` of its existing samples on the rectangle, and finish at the
    trace's original endpoint. The next buffer therefore resumes exactly as
    it would have with the border disabled.

    The one-sample entry and exit connectors cross quickly and stay faint;
    every corner itself is represented exactly. ``aspect`` is height/width
    before square padding, matching all three image-renderer conventions.
    """
    source = np.asarray(frame, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != 2:
        raise ValueError(f"trace must have shape (N, 2), got {source.shape}")
    fraction = min(0.5,max(0.0,float(fraction)))
    count = min(len(source) - 1, int(round(len(source) * fraction)))
    if fraction <= 0.0 or count < 6:
        return np.ascontiguousarray(source, dtype=np.float32)

    return trace_border(np.ascontiguousarray(source),count,float(aspect),float(level))


class TraceEmitter:
    """Turns a luminance array into the next trace. The single implementation.

    Every caller that drives a scope needs the same seven things: fixed grid
    and levels, field rotation, sweep chaining, the border, DC compensation,
    and a consistent set of tuning parameters. Before this class each caller
    assembled that list itself, and they drifted -- scope_screen was still
    re-deriving its grid every frame, and gained `border` and `oversample`
    only when someone remembered to port them across.

    Callers differ ONLY in where the luminance comes from: mode scope
    composites two baked libraries, scope_screen grabs the screen. Both hand
    the array here and get a frame back.

    Sweep state lives in the object because it has to persist between traces
    and because two callers keeping their own copies is how it goes wrong.
    """

    def __init__(self, samplerate, samples, *, gamma=2.2, trim=0.02,
                 density=1.0, rows=None, fields=1, border=0.0, oversample=1,
                  sweep="alternate", dc_comp=None, grid=None, levels=None,
                  autofit=True, row_bias=1.0, precondition=0.0,
                 yt_timing=None, yt_trigger_samples=0,
                 close_frame=False, geometry_samples=None,
                 traversal_hz=None):
        self.samplerate = samplerate
        self.n = int(samples)
        self.gamma, self.trim, self.density = gamma, trim, density
        self.rows, self.fields = rows, max(1, int(fields))
        self.border, self.oversample = border, oversample
        self.sweep_mode, self.dc_comp = sweep, dc_comp
        self.grid, self.levels, self.autofit = grid, levels, autofit
        self.row_bias = row_bias
        self.precondition = max(0.0, float(precondition))
        if yt_timing not in (None, "fixed", "dwell"):
            raise ValueError("Y-T timing must be fixed or dwell")
        self.yt_timing = yt_timing
        self.yt_trigger_samples = int(yt_trigger_samples)
        self.close_frame = bool(close_frame)
        self.samplerate = float(samplerate)
        if not math.isfinite(self.samplerate) or self.samplerate <= 0:
            raise ValueError("samplerate must be finite and positive")
        if self.n < 1:
            raise ValueError("samples must be positive")
        if geometry_samples is not None and int(geometry_samples) < 2:
            raise ValueError("geometry_samples must be at least 2")
        if traversal_hz is not None and (
                not math.isfinite(float(traversal_hz))
                or float(traversal_hz) <= 0):
            raise ValueError("traversal_hz must be finite and positive")
        if traversal_hz is not None and self.fields != 1:
            raise ValueError("time traversal does not support interlace fields > 1")
        if traversal_hz is not None and self.close_frame:
            raise ValueError(
                "time traversal does not support close_frame; use trigger retrace")
        if ((traversal_hz is not None or geometry_samples is not None)
                and self.yt_timing == "fixed"):
            raise ValueError("independent geometry does not support fixed Y-T timing")
        self.geometry_samples = (int(geometry_samples) if geometry_samples is not None
                                 else (3200 if traversal_hz is not None else self.n))
        self.traversal_hz = (None if traversal_hz is None
                             else float(traversal_hz))
        self._phase = 0.0
        self._candidate_phase = None
        self._rev, self._end, self._field = False, None, 0
        _warm_walk_raw_numba(
            self.n,
            point_dtype=(np.float64 if self.yt_timing == "fixed"
                         else np.float32))
        if self.traversal_hz is not None or geometry_samples is not None:
            _trajectory_sample_kernel(
                np.zeros((2, 2), np.float32), 0.0,
                ((self.traversal_hz / self.samplerate)
                 if self.traversal_hz is not None else 1.0 / self.n), self.n)

    def reset(self):
        """Drop chain state. Call after a device swap: the beam is not where
        the chain thinks it is, and continuing would jump the full screen."""
        self._rev, self._end, self._field = False, None, 0
        self._phase = 0.0
        self._candidate_phase = None

    def emit(self, lum, levels=None, *, field=None, commit=True, start=None,
             prepared_grid=None):
        """One trace, or None if there is nothing to draw.

        ``commit=False`` renders a candidate without advancing the sweep or
        interlace state. The caller can pass the accepted output endpoint to
        :meth:`accept` after it has been queued for presentation.
        """
        if lum is None:
            return None
        timed = self.traversal_hz is not None
        independent_geometry = timed or self.geometry_samples != self.n
        alt = self.sweep_mode == "alternate"
        kw = {}
        if self.grid:
            kw["grid_rows"], kw["grid_cols"] = self.grid
        lv = levels if levels is not None else self.levels
        if alt and not timed:
            start = self._end if start is None else start
        else:
            start = None
        frame = render_luma(
            lum, self.geometry_samples if independent_geometry else self.n,
            gamma=self.gamma, trim=self.trim,
            density=self.density, rows=self.rows, autofit=self.autofit,
            oversample=self.oversample, border=self.border,
            row_bias=self.row_bias, precondition=self.precondition,
            fields=self.fields,
            field=(self._field if field is None else int(field)) % self.fields,
            levels=lv, palindrome=(self.sweep_mode == "palindrome"),
            reverse=(alt and self._rev and not timed),
            start=start,
            # With the trigger marker enabled, Scope supplies an off-picture
            # retrace at each trace boundary. Without it, a missed producer
            # deadline makes Scope replay this whole trace; a short dim close
            # path then bounds the loop seam. The next trace still chains from
            # this endpoint.
            close=(True if timed else
                   (self.sweep_mode == "retrace" or
                    (alt and self.close_frame))),
            loop_anchor=(alt and self.close_frame and not timed),
            yt_fixed=(self.yt_timing == "fixed"),
            yt_trigger_samples=self.yt_trigger_samples,
            prepared_grid=(None if independent_geometry else prepared_grid),
            **kw)
        if frame is None:
            return None
        if independent_geometry:
            # Prepared grids are budget-dependent and are deliberately ignored
            # here: the canonical grid must use geometry_samples, not DAC n.
            rate = ((self.traversal_hz / float(self.samplerate)) if timed
                    else 1.0 / self.n)
            phase = self._phase
            frame = _trajectory_sample_kernel(
                np.ascontiguousarray(frame, dtype=np.float32), phase, rate,
                self.n)
            if timed:
                self._candidate_phase = (phase + self.n * rate) % 1.0
        if self.dc_comp:
            from scope_out import precompensate_hpf
            frame = precompensate_hpf(frame, self.dc_comp, self.samplerate)
        if alt and not timed:
            if start is not None:
                from scope_out import anchor_periodic_frame
                frame = anchor_periodic_frame(frame, start)
        if commit:
            self.accept(frame[-1])
        return frame

    def accept(self, endpoint):
        """Commit progression after a rendered trace was accepted for output."""
        self._field += 1
        if self.sweep_mode == "alternate":
            self._rev = not self._rev
            self._end = np.asarray(endpoint, dtype=np.float32)[:2].copy()
        if self._candidate_phase is not None:
            self._phase = self._candidate_phase
            self._candidate_phase = None


def _stochastic_probability(lum, gamma, trim, edge_gain):
    """Return Osci-style per-candidate acceptance probabilities.

    Osci does not draw one random mask and walk that mask for an entire trace.
    It makes a fresh probability decision every time it examines a candidate.
    That distinction is what turns luminance into visit density instead of a
    collection of permanently connected random islands.
    """
    if lum is None:
        return None
    a = np.asarray(lum, dtype=np.float64)
    if a.ndim != 2 or not a.size:
        return None
    prob, mass = importance_grid(np.ascontiguousarray(a), float(gamma),
                                 float(trim), float(edge_gain), True)
    return prob if mass > 1e-12 else None


class StochasticEmitter:
    """Continuous Osci-style walk whose state survives trace buffers.

    Three clocks are deliberately separate:

    * the image/index clock chooses the current probability field;
    * ``walk_hz`` chooses new image targets (48 kHz matches Osci's usual
      one-target-per-sample behaviour at a 48 kHz device);
    * the DAC rate samples that continuing path.  A 96 kHz DAC therefore gets
      two samples for each 48 kHz target interval instead of drawing twice as
      much image merely because the hardware sample rate doubled.

    A trace is only an audio buffer.  It is not a stochastic image frame and
    it has no waypoint budget.
    """

    _DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))

    def __init__(self, samplerate, samples, *, gamma=2.0, trim=0.02,
                 radius=10, stride=0, edge_gain=0.0, reseed_ms=5.0,
                 walk_hz=48000.0, seed=0, dc_comp=None, level=0.9,
                 border=0.0):
        warm_scope_numeric()
        self.samplerate = max(1, int(samplerate))
        self.n = max(2, int(samples))
        self.gamma, self.trim = gamma, trim
        self.radius, self.stride = radius, stride
        self.edge_gain, self.reseed_ms = edge_gain, reseed_ms
        self.walk_hz = max(1.0, float(walk_hz))
        self.dc_comp = dc_comp
        self.level = float(level)
        self.border = min(0.5,max(0.0,float(border)))
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self._end = None
        self._shape = None
        self._pixel = None
        self._visited = None
        self._count = 0
        self._global_searches = 0
        self._cdf_source = None
        self._cdf_value = None
        self._phase = 1.0       # choose a target for the first output sample
        self._idle_phase = 0
        self._handoff_pending = False
        self._lowpass = None
        self._lowpass_cutoff = None

    def checkpoint(self):
        """Capture mutable trajectory/filter state for an output transaction."""
        lowpass = self._lowpass
        return {
            "end": self._end,
            "shape": self._shape,
            "pixel": self._pixel,
            "visited": (None if self._visited is None
                        else self._visited.copy()),
            "count": self._count,
            "global_searches": self._global_searches,
            "phase": self._phase,
            "idle_phase": self._idle_phase,
            "handoff_pending": self._handoff_pending,
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "lowpass": lowpass,
            "lowpass_cutoff": self._lowpass_cutoff,
            "lowpass_state": (
                None if lowpass is None else
                (lowpass.enabled, lowpass.a, lowpass.z.copy())),
        }

    def restore(self, checkpoint):
        """Restore a candidate that was not accepted by the output queue."""
        self._end = checkpoint["end"]
        self._shape = checkpoint["shape"]
        self._pixel = checkpoint["pixel"]
        self._visited = checkpoint["visited"]
        self._count = checkpoint["count"]
        self._global_searches = checkpoint.get("global_searches", 0)
        self._phase = checkpoint["phase"]
        self._idle_phase = checkpoint["idle_phase"]
        self._handoff_pending = checkpoint["handoff_pending"]
        self.rng.bit_generator.state = copy.deepcopy(checkpoint["rng_state"])
        self._lowpass = checkpoint["lowpass"]
        self._lowpass_cutoff = checkpoint["lowpass_cutoff"]
        lowpass_state = checkpoint["lowpass_state"]
        if self._lowpass is not None and lowpass_state is not None:
            self._lowpass.enabled, self._lowpass.a = lowpass_state[:2]
            self._lowpass.z[:] = lowpass_state[2]

    def start_at(self, point):
        """Resume after a mode handoff at the beam's actual position.

        The walk clock and short visited history survive. Resetting them here
        made every stochastic pass in the whole-trace mix behave like a brand-new
        renderer, even though the emitter itself was deliberately persistent.
        Only the geometric cursor is remapped to the beam's current endpoint.
        """
        self._end = np.asarray(point, dtype=np.float32)[:2].copy()
        self._pixel = None      # remap this endpoint when the image shape is known
        if self._count == 0:
            # Do not turn the first handoff into an immediate random reseed.
            self._count = 1
        self._handoff_pending = True
        if self._lowpass is not None:
            self._lowpass.z[:] = self._end

    def chain_from(self, point):
        """Record the endpoint actually sent to the DAC after filtering."""
        self._end = np.asarray(point, dtype=np.float32)[:2].copy()

    def handoff_from(self, point):
        """Use an exact first sample only when entering from another mode."""
        if self._pixel is None:
            self.start_at(point)
            return True
        else:
            self.chain_from(point)
            return False

    def apply_lowpass(self, frame, cutoff_hz):
        """Stateful filter for a walk that does not loop at buffer edges."""
        enabled = bool(cutoff_hz and 0 < cutoff_hz < self.samplerate / 2)
        if self._lowpass is None:
            if not enabled:
                return frame
            from scope_lowpass import CascadedOnePole
            self._lowpass = CascadedOnePole(
                cutoff_hz, self.samplerate, order=4, channels=2)
            self._lowpass.z[:] = frame[0]
            self._lowpass_cutoff = float(cutoff_hz)
        elif cutoff_hz != self._lowpass_cutoff:
            was_enabled = self._lowpass.enabled
            self._lowpass.set_cutoff(cutoff_hz, self.samplerate)
            if enabled and not was_enabled:
                self._lowpass.z[:] = frame[0]
            self._lowpass_cutoff = float(cutoff_hz) if cutoff_hz else None
        return self._lowpass.process(frame)

    def _xy_to_pixel(self, point, shape):
        h, w = shape
        m = float(max(w, h))
        scale = max(abs(self.level),1e-9)
        x = (float(point[0])/scale+1.0)*0.5*m-(m-w)*0.5
        y = (1.0-float(point[1])/scale)*0.5*m-(m-h)*0.5
        return (min(w-1,max(0,round(x))),min(h-1,max(0,round(y))))

    def _pixel_to_xy(self, pixel, shape):
        h, w = shape
        x, y = pixel
        m = float(max(w, h))
        return np.asarray([
            2.0 * (x + (m - w) * 0.5) / m - 1.0,
            1.0 - 2.0 * (y + (m - h) * 0.5) / m,
        ], dtype=np.float32) * self.level

    def _ensure_state(self, shape):
        h, w = shape
        if self._shape != shape:
            self._shape = shape
            self._visited = np.zeros(shape, dtype=bool)
            self._pixel = None
        if self._pixel is None:
            if self._end is not None:
                self._pixel = self._xy_to_pixel(self._end, shape)
            else:
                self._pixel = (int(self.rng.integers(w)),
                               int(self.rng.integers(h)))

    def _find_white(self, prob, cdf=None):
        """Osci's global rejection fallback, with a bounded exact fallback."""
        h, w = prob.shape
        for _ in range(100):
            x = int(self.rng.integers(w)); y = int(self.rng.integers(h))
            p = float(prob[y, x])
            if p > 0.0 and self.rng.random() < p:
                return x, y

        # Rejection can miss a very small bright feature 100 times.  Sampling
        # proportional to p is the same accepted distribution without letting
        # sparse images become a CPU or blank-frame lottery.
        if callable(cdf):
            cdf = cdf()
        weights = prob.ravel()
        if cdf is None:
            cumulative, total = cumulative_mass(weights)
        else:
            cumulative, total = cdf
            total = float(total)
        if total <= 1e-12:
            return self._pixel
        flat = int(mass_search(cumulative,self.rng.random()*total))
        flat = min(flat, weights.size - 1)
        return flat % w, flat // w

    def _stride_for_width(self, width):
        return (max(1, int(round(width / 120.0))) if int(self.stride) <= 0
                else max(1, int(self.stride)))

    def _advance(self, prob, cdf=None):
        h, w = prob.shape
        reseed_targets = max(1, round(
            min(self.walk_hz, float(self.samplerate))
            * max(float(self.reseed_ms), 0.0) / 1000.0))
        if self._count % reseed_targets == 0:
            self._pixel = (int(self.rng.integers(w)),
                           int(self.rng.integers(h)))

        # This is what current Osci-render actually executes: due to operator
        # precedence, `count % 10 * jumpFrequency() == 0` clears every ten
        # target samples, not every ten jump intervals.
        if self._count % 10 == 0:
            self._visited.fill(False)

        x, y = self._pixel
        sx, sy = x, y
        direction = int(self.rng.integers(4))
        radius = max(1, min(int(self.radius), 64))
        # Osci's default stride 4 is measured in full-resolution source pixels
        # (about 480 px wide for the supplied portrait). Applying the same
        # integer to a 96 px bake made every move five times too large and
        # produced the visible orthogonal maze. Zero means scale that physical
        # step to the stored width: 96/120 -> 1, 256/120 -> 2, 480/120 -> 4.
        stride = self._stride_for_width(w)
        found = None
        local = False
        for arm in range(1, 2 * radius + 1):
            for _ in range(2):
                dx, dy = self._DIRS[direction]
                for _ in range(arm):
                    sx += stride * dx; sy += stride * dy
                    if sx < 0 or sx >= w or sy < 0 or sy >= h:
                        break
                    p = float(prob[sy, sx])
                    if (p > 0.0 and not self._visited[sy, sx]
                            and self.rng.random() < p):
                        found = (sx, sy)
                        local = True
                        break
                if found is not None:
                    break
                direction = (direction + 1) % 4
            if found is not None:
                break

        if found is None:
            found = self._find_white(prob, cdf=cdf)
        self._pixel = found
        if local:
            self._visited[found[1], found[0]] = True
        self._count += 1

    def emit(self, lum):
        prob = _stochastic_probability(
            lum, self.gamma, self.trim, self.edge_gain)
        return self.emit_probability(prob)

    def emit_probability(self, probability, *, cdf=None):
        """Walk an already-combined probability field.

        Fusion uses this entry point because each component has already had
        its own tone rule applied and has been mass-normalized. Running the
        result through the stochastic luminance threshold a second time would
        erase the weaker component after blending.
        """
        if probability is None:
            return None
        prob = np.asarray(probability, dtype=np.float64)
        if prob.ndim != 2 or not prob.size:
            return None
        prob, peak = sanitize_probability(np.ascontiguousarray(prob))
        if peak <= 1e-12:
            return None
        self._ensure_state(prob.shape)

        # A DAC faster than the walk clock samples the same continuing target
        # stream more densely.  It does not receive extra image decisions.
        step = min(self.walk_hz, float(self.samplerate)) / self.samplerate
        h, w = prob.shape
        cumulative, total = (cdf if cdf is not None and not callable(cdf)
                             else (np.empty(0,np.float64), 0.0))
        if (callable(cdf) and self._cdf_source is probability
                and not np.asarray(probability).flags.writeable):
            cumulative,total = self._cdf_value
        handoff = self._handoff_pending and self._end is not None
        start = self._end if handoff else np.zeros(2,np.float32)
        out, x, y, self._count, self._phase, fallback, searches = stochastic_stream(
            prob, self._visited, self.rng, *self._pixel, self._count, self._phase,
            self.n, step, max(1, round(min(self.walk_hz,float(self.samplerate))*
                max(float(self.reseed_ms),0.0)/1000.0)),
            max(1,min(int(self.radius),64)), self._stride_for_width(w),
            self.level, np.asarray(start,dtype=np.float32), bool(handoff),
            np.asarray(cumulative,dtype=np.float64), float(total))
        self._pixel = (x,y)
        self._global_searches += searches
        self._handoff_pending = False
        if fallback and callable(cdf):
            prepared = cdf() # lazy admission; reuse only immutable source data
            if not np.asarray(probability).flags.writeable:
                self._cdf_source = probability
                self._cdf_value = prepared

        out = apply_trace_border(
            out, self.border, aspect=h / float(max(w, 1)), level=self.level)
        if self.dc_comp:
            from scope_out import precompensate_hpf
            out = precompensate_hpf(out, self.dc_comp, self.samplerate)
        self._end = out[-1].copy()
        return np.ascontiguousarray(out, dtype=np.float32)


def _stipple_importance(lum, gamma=2.0, trim=0.02, edge_gain=0.0):
    """Continuous image importance without stochastic's fixed 0.2 gate."""
    if lum is None:
        return None
    a = np.asarray(lum, dtype=np.float64)
    if a.ndim != 2 or not a.size:
        return None
    importance, total = importance_grid(np.ascontiguousarray(a), float(gamma),
                                        float(trim), float(edge_gain), False)
    return importance if total > 1e-12 else None


def _stipple_image_samples(importance, points):
    """Systematic image-importance quantiles, independent of endpoint/tour."""
    if importance is None:
        return None
    return systematic_samples(np.ascontiguousarray(importance), int(points))


@njit(cache=True, nogil=True, fastmath=False)
def _greedy_nearest_order_kernel(points, start):
    """Greedy squared-Euclidean tour; strict comparison preserves first ties."""
    count = points.shape[0]
    order = np.empty(count, dtype=np.int64)
    used = np.zeros(count, dtype=np.uint8)
    px, py = start[0], start[1]
    for i in range(count):
        best_index = -1
        best_distance = np.inf
        for j in range(count):
            if used[j] == 0:
                dx = points[j, 0] - px
                dy = points[j, 1] - py
                distance = dx * dx + dy * dy
                # Strict comparison matches np.argmin: the first index wins ties.
                if distance < best_distance:
                    best_distance = distance
                    best_index = j
        order[i] = best_index
        used[best_index] = 1
        px, py = points[best_index, 0], points[best_index, 1]
    return order


def _warm_stipple_tour_kernel():
    """Compile the production contiguous float64 signature during setup."""
    _greedy_nearest_order_kernel(
        np.zeros((1, 2), dtype=np.float64),
        np.zeros(2, dtype=np.float64))


def _greedy_nearest_order(points, start):
    """Return a stable greedy nearest-neighbour permutation without mutation."""
    points = np.ascontiguousarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if not finite_array(points):
        raise ValueError("points must be finite")
    if not len(points):
        return np.empty(0, dtype=np.int64)
    pos = np.ascontiguousarray(np.asarray(start, dtype=np.float64).reshape(-1)[:2])
    if pos.size != 2 or not finite_array(pos):
        raise ValueError("start must contain two finite coordinates")
    # Keep mmap/cache read-only arrays on the already-warmed mutable signature.
    if not points.flags.writeable:
        points = points.copy()
    if not pos.flags.writeable:
        pos = pos.copy()
    return _greedy_nearest_order_kernel(points, pos)


def _stipple_candidate_samples(cloud, points, gamma, trim, edge_gain):
    """Return systematic candidate indices/dwell, excluding endpoint tour state."""
    if cloud is None:
        return None
    xy = np.asarray(cloud["xy"], dtype=np.float64)
    lum = np.asarray(cloud["luminance"], dtype=np.float64)
    edge = np.asarray(cloud["edge"], dtype=np.float64)
    correction = np.asarray(cloud["correction"], dtype=np.float64)
    if not len(xy) or not (len(xy) == len(lum) == len(correction)):
        return None
    importance = candidate_importance(lum, edge, correction, float(gamma),
                                     float(trim), float(edge_gain))
    return systematic_samples(importance, int(points))


class StippleEmitter(StochasticEmitter):
    """Stable luminance-weighted points joined by an unrestricted local tour.

    The point set belongs to the image, not to an output-rate clock. Systematic
    weighted sampling gives every frame the same evenly spaced probability
    quantiles; repeated pixels become dwell, and a Euclidean nearest-neighbour
    tour keeps unavoidable no-Z connectors short. The finished route is then
    resampled to the caller's array length.
    """

    def __init__(self, samplerate, samples, *, points=768, gamma=2.0,
                 trim=0.02, edge_gain=0.0, dc_comp=None, level=0.9,
                 border=0.0, geometry_samples=None, traversal_hz=None):
        super().__init__(
            samplerate, samples, gamma=gamma, trim=trim,
            edge_gain=edge_gain, dc_comp=dc_comp, level=level, border=border)
        self.points = max(8, int(points))
        self.geometry_samples = (None if geometry_samples is None
                                 else max(2, int(geometry_samples)))
        self.traversal_hz = (None if traversal_hz is None
                             else float(traversal_hz))
        if self.traversal_hz is not None and (
                not math.isfinite(self.traversal_hz) or self.traversal_hz <= 0):
            raise ValueError("traversal_hz must be finite and positive")
        self._trajectory_phase = 0.0
        self._candidate_trajectory_phase = None
        _warm_stipple_tour_kernel()
        if geometry_samples is not None or traversal_hz is not None:
            _trajectory_sample_kernel(
                np.zeros((2, 2), dtype=np.float32), 0.0, 0.5, 2)

    def checkpoint(self):
        state = super().checkpoint()
        state["trajectory_phase"] = self._trajectory_phase
        state["candidate_trajectory_phase"] = self._candidate_trajectory_phase
        return state

    def restore(self, checkpoint):
        super().restore(checkpoint)
        self._trajectory_phase = checkpoint.get("trajectory_phase", 0.0)
        self._candidate_trajectory_phase = checkpoint.get(
            "candidate_trajectory_phase")

    def reset(self):
        super().reset()
        self._trajectory_phase = 0.0
        self._candidate_trajectory_phase = None

    def accept(self, endpoint):
        super().chain_from(endpoint)
        if self._candidate_trajectory_phase is not None:
            self._trajectory_phase = self._candidate_trajectory_phase
            self._candidate_trajectory_phase = None

    def _sample_route(self, route, singleton_radius):
        """Resample stipple geometry independently of DAC buffer length."""
        canonical_n = self.geometry_samples or self.n
        route = stipple_geometry(np.ascontiguousarray(route),canonical_n,float(singleton_radius))
        if self.traversal_hz is None:
            return _trajectory_sample_kernel(
                np.ascontiguousarray(route, dtype=np.float32), 0.0,
                1.0 / canonical_n, self.n)
        rate = self.traversal_hz / self.samplerate
        self._candidate_trajectory_phase = (
            self._trajectory_phase + self.n * rate) % 1.0
        return _trajectory_sample_kernel(
            np.ascontiguousarray(route, dtype=np.float32),
            self._trajectory_phase, rate, self.n)

    def _sample_pixels(self, importance, *, prepared_samples=None,
                       tour_cache=None, source_key=None):
        h, w = importance.shape
        samples = (prepared_samples if prepared_samples is not None else
                   _stipple_image_samples(importance, self.points))
        if samples is None:
            return None
        unique, dwell = samples
        pixels = pixel_positions(unique,w)

        if self._end is not None and self.traversal_hz is None:
            start = np.asarray(self._xy_to_pixel(self._end, (h, w)), np.float64)
        else:
            start = pixels[argmax_first(dwell)]
        settings = ("image", self.points, (h, w), float(self.level),
                    float(self.gamma), float(self.trim), float(self.edge_gain))
        order = (tour_cache.stipple_order(
            pixels, start, source_key, settings=settings)
                 if tour_cache is not None else
                 _greedy_nearest_order(pixels, start))
        return dwell_route(pixels,order,dwell)

    def emit(self, lum):
        importance = _stipple_importance(
            lum, self.gamma, self.trim, self.edge_gain)
        return self.emit_importance(importance)

    def emit_importance(self, importance, *, prepared_samples=None,
                        tour_cache=None, source_key=None):
        """Render prepared image importance with the current endpoint/tour."""
        if importance is None:
            return None
        pixels = self._sample_pixels(
            importance, prepared_samples=prepared_samples,
            tour_cache=tour_cache, source_key=source_key)
        if pixels is None or not len(pixels):
            return None
        h, w = importance.shape
        m = float(max(w, h))
        route = pixel_route(pixels,w,h,float(self.level))
        if self._end is not None and self.traversal_hz is None:
            route = np.vstack([self._end, route])
        out = self._sample_route(route, self.level / max(importance.shape))
        out = apply_trace_border(
            out, self.border, aspect=h / float(max(w, 1)), level=self.level)
        if self.dc_comp:
            from scope_out import precompensate_hpf
            out = precompensate_hpf(out, self.dc_comp, self.samplerate)
        self._end = out[-1].copy()
        self._handoff_pending = False
        return out

    def emit_candidates(self, cloud, prepared_samples=None, *,
                        tour_cache=None, source_key=None):
        """Render a compact baked candidate cloud with live tone controls."""
        if cloud is None:
            return None
        xy = np.asarray(cloud["xy"], dtype=np.float64)
        samples = (prepared_samples if prepared_samples is not None else
                   _stipple_candidate_samples(
                       cloud, self.points, self.gamma, self.trim,
                       self.edge_gain))
        if samples is None:
            return None
        unique, dwell = samples
        points = xy[unique]

        aspect = max(float(cloud.get("aspect", 1.0)), 1e-9)
        sx = 1.0 if aspect <= 1.0 else 1.0 / aspect
        sy = aspect if aspect <= 1.0 else 1.0
        display = cloud_route(points,aspect,float(self.level))

        start = (self._end if self._end is not None
                 and self.traversal_hz is None
                  else display[argmax_first(dwell)])
        settings = ("candidates", self.points, float(self.gamma),
                    float(self.trim), float(self.edge_gain), aspect,
                    float(self.level))
        order = (tour_cache.stipple_order(
            display, start, source_key, settings=settings)
                 if tour_cache is not None else
                 _greedy_nearest_order(display, start))
        route = dwell_route(display,order,dwell)
        if self._end is not None and self.traversal_hz is None:
            route = np.vstack([self._end, route])
        out = self._sample_route(route, self.level / 256.0)
        out = apply_trace_border(
            out, self.border, aspect=aspect, level=self.level)
        if self.dc_comp:
            from scope_out import precompensate_hpf
            out = precompensate_hpf(out, self.dc_comp, self.samplerate)
        self._end = out[-1].copy()
        self._handoff_pending = False
        return out


class TriangleMixScheduler:
    """Whole-trace V/R/S/stipple scheduler (historical public class name).

    ``raster_duty`` retains the old mix contract: it is the long-term fraction
    of traces assigned to raster. The remaining traces cycle vector,
    stochastic, and stipple, so all three receive equal shares. At the default
    0.5 duty the exact repeating route is V, R, S, R, T, R (T = stipple).

    Whole traces are deliberate. Blending unrelated XY sample positions would
    draw the interpolation between them; phosphor persistence already performs
    the useful visual sum without inventing those connector shapes.
    """

    def __init__(self, raster_duty=0.5):
        duty = float(raster_duty)
        if not math.isfinite(duty):
            raise ValueError("raster_duty must be finite")
        self.raster_duty = min(1.0, max(0.0, duty))
        self._raster_error = 0.0
        self._outer = 0

    @staticmethod
    def coverage_traces(raster_duty=0.5, raster_fields=1):
        """Traces needed for one complete mixed preview exposure.

        The window must contain every raster field and, when present, at
        least one vector, stochastic, and stipple pass. It is also used as the
        honest combined-picture refresh period reported by the monitor.
        """
        duty = float(raster_duty)
        if not math.isfinite(duty):
            raise ValueError("raster_duty must be finite")
        duty = min(1.0, max(0.0, duty))
        fields = max(1, int(raster_fields))
        spans = [1]
        if duty > 0.0:
            spans.append(int(math.ceil(fields / duty - 1e-12)))
        if duty < 1.0:
            # Vector, stochastic, and stipple split the outer share equally.
            spans.append(int(math.ceil(3.0 / (1.0 - duty) - 1e-12)))
        return max(spans)

    def next_mode(self):
        mode = self.peek_next_mode()
        self.advance()
        return mode

    def peek_next_mode(self):
        """Return the next route without advancing output state."""
        error = self._raster_error + self.raster_duty
        if error >= 1.0 - 1e-12:
            return "raster"
        return ("vector", "stochastic", "stipple")[self._outer % 3]

    def advance(self):
        """Commit one route after its trace was accepted for output."""
        self._raster_error += self.raster_duty
        if self._raster_error >= 1.0 - 1e-12:
            self._raster_error -= 1.0
        else:
            self._outer += 1


def stochastic_luma(lum, n, *, rng=None, gamma=2.0, trim=0.02,
                    radius=10, stride=0, edge_gain=0.0,
                    reseed_samples=240, start=None, level=0.9):
    """Standalone one-buffer form of :class:`StochasticEmitter`.

    The runtime keeps an emitter alive across buffers.  This convenience form
    exists for tests and previews and still chooses a fresh target on every
    returned sample; it never constructs or resamples a smaller waypoint list.
    """
    if int(n) < 2:
        return None
    reference_rate = 48000
    emitter = StochasticEmitter(
        reference_rate, int(n), gamma=gamma, trim=trim, radius=radius,
        stride=stride, edge_gain=edge_gain,
        reseed_ms=1000.0 * max(1, int(reseed_samples)) / reference_rate,
        walk_hz=reference_rate, seed=0, level=level)
    if rng is not None:
        emitter.rng = rng
    if start is not None:
        emitter.start_at(start)
    return emitter.emit(lum)


def composite_luma(main_lib, main_idx, float_lib, float_idx, bbox=None,
                   raw=False, invert=False, thumbnails=None):
    """The interleaved composite as a luminance array, and nothing else.

    Split out of raster_frame so there is ONE place that decides what the
    picture is, separate from the one place that decides how to sweep it.
    Everything downstream -- mode scope, scope_screen, the preview -- takes a
    luminance array, so they can share a single emitter instead of each
    re-assembling the same argument list and drifting apart. ``raw`` selects
    the third channel of new bakes for directionless stochastic motion; legacy
    two-channel bakes fall back to their raster luminance.
    """
    if thumbnails is None:
        tm = (main_lib.thumb(main_idx)
              if main_lib is not None and len(main_lib) else None)
        tf = (float_lib.thumb(float_idx)
              if float_lib is not None and len(float_lib) else None)
    else:
        tm, tf = thumbnails
    if tm is None and tf is None:
        return None

    if tf is not None and tm is not None and tf.shape[:2] != tm.shape[:2]:
        raise ValueError(
            "main/float thumbnail geometry differs "
            f"({tm.shape[1]}x{tm.shape[0]} vs {tf.shape[1]}x{tf.shape[0]}); "
            "rebake all layers with the same --thumb-width")

    empty = np.empty((0,0,2),np.uint8)
    lum = composite_thumbnail(tm if tm is not None else empty,
                              tf if tf is not None else empty,bool(raw),bool(invert))

    if bbox is not None:                       # crop to the subject so the
        hh, ww = lum.shape                     # budget is spent on content
        x0, y0, x1, y1 = bbox
        lum = lum[int(y0 * hh):max(int(y1 * hh), int(y0 * hh) + 1),
                  int(x0 * ww):max(int(x1 * ww), int(x0 * ww) + 1)]
    return lum


def raster_precondition_for(main_lib, float_lib=None):
    """Recommended grid-domain compensation, or zero for legacy bakes.

    A legacy two-channel bake may already be preconditioned and a three-channel
    bake definitely has a separate preconditioned raster channel. Applying the
    new grid correction to either would sharpen twice, so only an explicitly
    tagged raw-thumbnail format enables it.
    """
    libs = [lib for lib in (main_lib, float_lib) if lib is not None]
    if not libs or not all(getattr(lib, "raw_thumbnail", False) for lib in libs):
        return 0.0
    values = [float(getattr(lib, "raster_precondition", 0.0)) for lib in libs]
    return min(values) if values else 0.0


def _alpha_at(alpha, xy):
    """Nearest matte lookup at normalized uint16/float candidate positions."""
    if alpha is None or not len(xy):
        return np.zeros(len(xy), dtype=np.float64)
    return alpha_samples(np.ascontiguousarray(alpha),np.asarray(xy,dtype=np.float64))


def composite_stipple_candidates(main_lib, main_idx, float_lib, float_idx,
                                  invert=False):
    """Combine sparse source-detail candidates for an arbitrary layer pair.

    Candidate coordinates were sampled from a broad proposal at bake time.
    ``correction`` is the proposal-density correction; StippleEmitter applies
    the live gamma and then uses it to recover the intended image distribution.
    The compact raster alpha is sufficient for float-over-main occlusion while
    the uint16 coordinates retain source-resolution feature placement.
    """
    cm = (main_lib.stipple(main_idx)
          if main_lib is not None and hasattr(main_lib, "stipple") else None)
    cf = (float_lib.stipple(float_idx)
          if float_lib is not None and hasattr(float_lib, "stipple") else None)
    if cm is None and cf is None:
        return None

    tm = main_lib.thumb(main_idx) if main_lib is not None else None
    tf = float_lib.thumb(float_idx) if float_lib is not None else None
    reference = tm if tm is not None else tf
    if reference is None:
        return None
    aspect = reference.shape[0] / float(max(reference.shape[1], 1))

    clouds = []

    def add(candidate, occluder=None):
        if candidate is None:
            return
        xy, lae, mass = candidate
        occlusion=_alpha_at(occluder,xy)
        clouds.append(candidate_cloud(np.asarray(xy),np.asarray(lae),float(mass),occlusion,bool(invert)))

    add(cm, tf[..., 1] if tf is not None else None)
    add(cf)
    if not clouds:
        return None
    return {
        "xy": np.concatenate([c[0] for c in clouds]),
        "luminance": np.concatenate([c[1] for c in clouds]),
        "edge": np.concatenate([c[2] for c in clouds]),
        "correction": np.concatenate([c[3] for c in clouds]),
        "aspect": aspect,
    }


FUSION_COMPONENTS = ("vrs", "vr", "sv", "sr")


def normalize_fusion_components(value):
    """Canonicalize a fusion component spelling to one supported preset."""
    key = "".join(ch for ch in str(value).lower() if ch in "vrs")
    canonical = {
        frozenset("vrs"): "vrs",
        frozenset("vr"): "vr",
        frozenset("sv"): "sv",
        frozenset("sr"): "sr",
    }.get(frozenset(key))
    if canonical is None or len(set(key)) != len(key):
        raise ValueError(
            "fusion components must be one of: vrs, vr, sv, sr")
    return canonical


def trace_luminance_weights(lum, trace, gamma=2.0, trim=0.02, level=0.9):
    """Sample image luminance at every XY position in a component trace.

    Fusion uses these values as temporal selection weights. Bilinear sampling
    avoids making the selector flicker merely because a moving point crosses
    a thumbnail cell boundary. Coordinates outside the square-padded image
    are dark, and ``trim``/``gamma`` give bright points progressively more
    beam time without changing their positions.
    """
    if lum is None or trace is None:
        return None
    image = np.asarray(lum, dtype=np.float64)
    points = np.asarray(trace, dtype=np.float64)
    if image.ndim != 2 or not image.size:
        return None
    if points.ndim != 2 or points.shape[1] != 2 or not len(points):
        return None
    return trace_weights(np.ascontiguousarray(image),np.ascontiguousarray(points),
                         float(gamma),float(trim),float(level))


def retime_trace_by_weights(trace, weights, travel_floor=0.03):
    """Re-time a fixed vector path so weighted regions receive more dwell.

    A two-channel scope cannot invert a vector with a Z/intensity value that
    does not exist. The useful no-Z equivalent is velocity modulation: retain
    the baked path and sample count, move slowly through high-weight regions,
    and cross low-weight regions quickly. Large inter-polyline jumps are kept
    at the minimum share so inversion never turns travel into bright strokes.
    """
    points = np.asarray(trace, dtype=np.float32)
    value = np.asarray(weights, dtype=np.float64).reshape(-1)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2
            or len(value) != len(points)):
        raise ValueError("trace and weights must have matching N and trace (N, 2)")
    return retime_weighted(np.ascontiguousarray(points),np.ascontiguousarray(value),float(travel_floor))


class PositionMultiplexer:
    """Select corresponding entries from V/R/S arrays in temporal proportion.

    Fusion is temporal selection, not coordinate arithmetic. With no weights,
    the historical V/R/S round-robin is preserved exactly. With per-position
    luminance weights, a deficit scheduler gives bright candidates more of
    the nearby DAC samples and dark candidates fewer. Credit survives buffer
    boundaries, so weighting changes density rather than introducing random
    frame-to-frame choices.
    """

    def __init__(self):
        self.phase = 0
        self.components = None
        self.credit = None

    def reset(self):
        self.phase = 0
        self.components = None
        self.credit = None

    def checkpoint(self):
        return (self.phase, self.components,
                None if self.credit is None else self.credit.copy())

    def restore(self, checkpoint):
        self.phase, self.components, credit = checkpoint
        self.credit = None if credit is None else credit.copy()

    def emit(self, vector=None, raster=None, stochastic=None,
             components="vrs", weights=None):
        components = normalize_fusion_components(components)
        available = {"v": vector, "r": raster, "s": stochastic}
        names, arrays = [], []
        for name in components:
            trace = available[name]
            if trace is None:
                continue
            trace = np.asarray(trace, dtype=np.float32)
            if trace.ndim != 2 or trace.shape[1] != 2 or not len(trace):
                raise ValueError(
                    f"invalid {name} fusion trace shape {trace.shape}")
            names.append(name)
            arrays.append(trace)
        if not arrays:
            return None
        shape = arrays[0].shape
        if any(trace.shape != shape for trace in arrays[1:]):
            raise ValueError("fusion component trace lengths differ")
        key = tuple(names)
        if key != self.components:
            self.phase = 0
            self.components = key
            self.credit = np.zeros(len(arrays), dtype=np.float64)

        supplied = weights or {}
        weight_arrays = []
        for name in names:
            value = supplied.get(name)
            if value is None:
                weight_arrays.append(np.ones(shape[0], dtype=np.float64))
                continue
            value = np.asarray(value, dtype=np.float64).reshape(-1)
            if len(value) != shape[0]:
                raise ValueError(
                    f"{name} fusion weight length {len(value)} != {shape[0]}")
            weight_arrays.append(value)
        out, self.phase = mux_positions(
            np.stack(arrays),np.column_stack(weight_arrays),self.credit,
            int(self.phase),bool(supplied))
        return out


def fuse_positions(vector=None, raster=None, stochastic=None,
                   components="vrs", weights=None):
    """Stateless convenience wrapper for per-index position multiplexing."""
    return PositionMultiplexer().emit(
        vector, raster, stochastic, components=components, weights=weights)


def vector_density(polylines, shape, thickness=1):
    """Rasterize baked XY lines into the thumbnail's square-padded space.

    This is a dwell-density source, not a displayed raster. Compiled DDA and
    dilation avoid expanding every segment into NumPy repeat/index arrays.
    """
    h, w = map(int, shape[:2])
    out = np.zeros((h, w), dtype=np.float64)
    if h <= 0 or w <= 0:
        return out
    for polyline in polylines or ():
        p = np.asarray(polyline, dtype=np.float64).reshape(-1, 2)
        if len(p) < 2:
            continue
        density_polyline(p,out)
    # A one-pixel contour can be missed too easily by a source-scale walk.
    # A small max-neighbourhood makes it a probability ridge without turning
    # it into the broad tonal mass supplied by raster/raw luminance.
    radius = max(0, int(thickness))
    if radius:
        out = dilate_density(out,radius)
    return out


def _mass_normalize(density):
    if density is None:return None
    return normalize_mass(np.asarray(density,dtype=np.float64))


def fuse_density(vector=None, raster=None, stochastic=None, components="vrs",
                 raster_gamma=2.2, stochastic_gamma=2.0, trim=0.02,
                 edge_gain=0.0):
    """Combine component *dwell distributions*, never instantaneous XY.

    Every available component is normalized to unit probability mass before
    equal mixing. Sparse vector ridges therefore receive the same requested
    dwell share as a full-frame luminance field instead of disappearing under
    its much larger pixel sum. The returned maximum is one; relative mass is
    unchanged and can be consumed directly by ``emit_probability``.
    """
    components = normalize_fusion_components(components)
    fields = []
    if "v" in components and vector is not None:
        fields.append(_mass_normalize(vector))
    if "r" in components and raster is not None:
        fields.append(_mass_normalize(_stipple_importance(raster,raster_gamma,trim,0.0)))
    if "s" in components and stochastic is not None:
        fields.append(_mass_normalize(_stochastic_probability(
            stochastic, stochastic_gamma, trim, edge_gain)))
    fields = [field for field in fields if field is not None]
    if not fields:
        return None
    shape = fields[0].shape
    if any(field.shape != shape for field in fields[1:]):
        raise ValueError("fusion component geometry differs")
    return combine_fields(np.stack(fields))


def fusion_probability(main_lib, main_idx, float_lib, float_idx,
                       components="vrs", bbox=None, min_feature=0.02,
                       raster_gamma=2.2, stochastic_gamma=2.0, trim=0.02,
                       edge_gain=0.0, vector_thickness=1, invert=False):
    """Build one normalized field from the requested baked render sources."""
    components = normalize_fusion_components(components)
    raster = (composite_luma(main_lib, main_idx, float_lib, float_idx,
                             bbox=bbox, raw=False, invert=invert)
              if "r" in components else None)
    stochastic = (composite_luma(main_lib, main_idx, float_lib, float_idx,
                                 bbox=bbox, raw=True, invert=invert)
                  if "s" in components else None)
    reference = raster if raster is not None else stochastic
    vector = None
    if "v" in components:
        if reference is None:
            return None
        vector = vector_density(
            merge(main_lib, main_idx, float_lib, float_idx,
                  min_feature=min_feature),
            reference.shape, thickness=vector_thickness)
    return fuse_density(
        vector=vector, raster=raster, stochastic=stochastic,
        components=components, raster_gamma=raster_gamma,
        stochastic_gamma=stochastic_gamma, trim=trim, edge_gain=edge_gain)


def raster_frame(main_lib, main_idx, float_lib, float_idx, n,
                 gamma=2.2, floor=0.012, level=0.9, rows=None, cols=None,
                 density=1.0, trim=0.02, stretch=True, bbox=None, autofit=True,
                 oversample=1, grid_rows=None, grid_cols=None, levels=None,
                 fields=1, field=0, palindrome=False, reverse=False, start=None,
                 close=None, overscan=1.0, border=0.0, row_bias=1.0,
                 subcell=True, precondition=0.0, invert=False):
    """
    Dwell-modulated serpentine: the 2-channel equivalent of a video-to-scope
    adapter.  The beam sweeps every row and lingers on bright cells, so
    brightness comes from time rather than a Z channel.

    Layers composite in the THUMBNAIL domain using real alpha -- the same
    formula as renderer.py's shader -- so inverse mattes need no special case.

    Trace duration is 1/fps no matter what is in the image; complexity does not
    change it.  What complexity changes is how thinly the fixed sample budget
    is spread:

      density : samples per grid cell.  The grid is sized to n/density cells.
                1.0 (default) maximises resolution.  Above 1 is a TRADE, not an
                improvement: fewer, longer, brighter scanlines with less detail.
                Worth raising only if thin traces read too dim on your tube.
      autofit : size the grid against the cells that will actually be DRAWN
                rather than the whole rectangle.  Trim discards the dark ones --
                typically half the grid on a portrait over black -- so without
                this correction the budget is spread over cells that are never
                swept, and the picture comes out coarser than the samples allow.
      trim    : cells dimmer than this are dropped from the sweep entirely, so
                black margins cost nothing and their samples go to the subject.
                On a portrait over black this is most of the frame.
      reverse/start : the good way to avoid a retrace.  Each trace sweeps the
                rows in ONE direction only, alternating per trace, and starts
                where the previous trace ended -- so consecutive traces chain
                into a continuous path with no flyback, while each trace draws
                a DIFFERENT index at full sample density.  `start` is simply
                the previous frame's last sample.
      close   : append a closing segment back to the first point.  Defaults to
                False whenever the trace is chained (reverse/start given),
                because the NEXT trace continues from here -- appending a
                flyback would put the last samples mid-jump.
      palindrome : fallback.  Sweeps down then back up over the same path in a
                single trace.  Also avoids a retrace, but the return pass
                redraws the same image, so every cell is visited twice at half
                density.  Only needed when a fresh frame is not guaranteed for
                every trace.
      fields  : interlacing.  fields=2 draws every other row per trace,
                alternating, so the full image is covered across two traces at
                full density.  ONLY use when fps is a multiple of ips -- each
                index must get `fields` traces or you see half a picture.
                Buys refresh rate (less flicker), never resolution.
    """
    lum = composite_luma(
        main_lib, main_idx, float_lib, float_idx, bbox=bbox, invert=invert)
    if lum is None:
        return None
    if precondition is None:  # explicit metadata opt-in for old callers only
        precondition = raster_precondition_for(main_lib, float_lib)
    return render_luma(lum, n, gamma=gamma, floor=floor, level=level,
                       rows=rows, cols=cols, density=density, trim=trim,
                       stretch=stretch, bbox=bbox, autofit=autofit, border=border,
                       row_bias=row_bias, subcell=subcell,
                       oversample=oversample, grid_rows=grid_rows,
                       grid_cols=grid_cols, levels=levels, fields=fields,
                       field=field, palindrome=palindrome, reverse=reverse,
                       start=start, close=close, overscan=overscan,
                       precondition=precondition)


def _precondition_grid(grid, amount):
    """Small horizontal-only beam compensation on the final sweep grid."""
    g = np.asarray(grid, dtype=np.float64)
    if not amount or amount <= 0.0 or g.shape[1] <= 2:
        return g
    return precondition_grid(np.ascontiguousarray(g), float(amount))


def _stretch_grid(grid, levels=None, stretch=True):
    if levels is not None:
        low, high = levels
    elif stretch:
        low, count = positive_percentile(grid, 2.0, 0.01)
        if count <= 16: return grid
        high, _ = positive_percentile(grid, 98.0, 0.01)
    else: return grid
    return stretch_grid(grid, float(low), float(high)) if high > low else grid


@njit(cache=True, nogil=True, fastmath=False)
def yt_timeline(n, rows, fields=1, trigger_samples=0, border=0.0):
    """Integer sample boundaries independent of luminance and field parity.

    Layout: reserved trigger, equal row slots, border, return. Uneven field
    lengths get a padding slot, so an odd row count cannot resize the picture.
    """
    n, fields = int(n), max(1, int(fields))
    slots = (int(rows) + fields - 1) // fields
    trigger = max(0, int(trigger_samples))
    border_n = int(np.floor(n * min(0.5,max(0.0,float(border))) + 0.5))
    if border_n < 5:
        border_n = 0
    retrace = max(2, int(np.floor(n * 0.004 + 0.5)))
    picture_end = n - border_n - retrace
    picture_n = picture_end - trigger
    if slots < 1 or picture_n < 2 * slots:
        raise ValueError("fixed row timing needs at least two samples per row; "
                         "reduce --scope-rows/--scope-trigger-us/--scope-border "
                         "or increase --scope-samples")
    edges = trigger + (np.arange(slots + 1, dtype=np.int64) * picture_n) // slots
    return edges, picture_end, n - retrace


def render_yt_grid(g, n, *, aspect=1.0, gamma=2.2, trim=0.02, floor=0.012,
                   level=0.9, fields=1, field=0, border=0.0,
                   trigger_samples=0, oversample=1):
    """Render a tone-mapped grid on a fixed Y-T time axis, still on X.

    Dwell is normalized within each row only. A row can no longer borrow
    time from any other row; empty rows retain their time at a negative
    pedestal outside the normal picture range. The pedestal is visible if
    the scope shows that voltage; there is no blanking/Z channel.
    """
    g = np.asarray(g, dtype=np.float64)
    rows, cols = g.shape
    edges, picture_end, return_start = yt_timeline(n, rows, fields, trigger_samples, border)
    sx = level * min(1.0, 1.0 / float(aspect))
    sy = level * min(1.0, float(aspect))
    xs, ys = linear_axis(-sx,sx,cols),linear_axis(sy,-sy,rows)
    # No positive trigger crossing outside Scope's reserved marker.
    pedestal = -min(0.98, level * 1.04)
    out = np.empty((n, 2), dtype=np.float32)
    out[:] = (pedestal, sy)
    for slot, (begin, end) in enumerate(zip(edges[:-1], edges[1:])):
        r = slot * max(1, int(fields)) + int(field) % max(1, int(fields))
        if r >= rows:
            out[begin:end] = (pedestal, -sy)
            continue
        out[begin:end] = (pedestal, ys[r])
        points,weights=yt_row(g,xs,r,float(trim),float(floor),float(gamma),float(ys[r]))
        if not len(points):
            continue
        if len(points) == 1:
            out[begin:end] = points[0]
            continue
        # Explicit endpoints keep row-to-row transitions at fixed indices.
        body = _walk(points, weights, int(end - begin - 1), oversample=oversample)
        clip_x(body,-sx,sx)
        out[begin:end - 1] = body
        out[begin] = points[0]
        out[end - 1] = points[-1]
    if return_start > picture_end:
        # Fixed corner and direction: changing the subject cannot rotate the
        # border's phase, as nearest-corner entry would do.
        corners = np.array([[-sx, -sy], [sx, -sy], [sx, sy], [-sx, sy], [-sx, -sy]])
        weights = np.asarray([2*sx,2*sy,2*sx,2*sy],np.float64)
        out[picture_end:return_start] = _walk(corners, weights, return_start - picture_end)
    out[return_start:] = linear_trace(out[return_start-1],out[0],n-return_start)
    return np.ascontiguousarray(out)


def prepare_render_grid(lum, n, *, density=1.0, trim=0.02, rows=None,
                        cols=None, autofit=True, grid_rows=None,
                        grid_cols=None, row_bias=1.0, levels=None,
                        stretch=True, fields=1, precondition=0.0,
                        yt_fixed=False):
    """Prepare reusable tone/grid arrays; contains no beam or field state."""
    h, w = lum.shape
    aspect = h / float(w)
    if density < 0.25:
        print(f"[SCOPE] density {density} clamped to 0.25 -- below that the grid "
              "outruns the sample count so far that most cells go unvisited")
        density = 0.25
    fields = max(1, int(fields))
    cells = max(64.0, n * fields / density)
    if rows and cols:
        rows, cols = int(rows), int(cols)
    elif rows:
        rows = int(rows)
        cols = max(8, int(cells / rows))
    elif cols:
        cols = int(cols)
        rows = max(6, int(cells / cols))
    else:
        cols = max(8, int(math.sqrt(cells / max(aspect, 1e-6))))
        rows = max(6, int(round(cols * aspect)))

    if grid_rows and grid_cols:
        rows, cols = int(grid_rows), int(grid_cols)
        autofit = False

    if autofit and trim > 0 and not yt_fixed:
        probe = _box(lum, rows, cols)
        probe = _stretch_grid(probe)
        frac = threshold_fraction(probe, float(trim))
        if 0.05 < frac < 0.95:
            grow = min(1.0 / math.sqrt(frac), 2.5)
            rows = max(6, min(int(round(rows * grow)), lum.shape[0]))
            cols = max(8, min(int(round(cols * grow)), lum.shape[1]))

    if grid_rows is None or grid_cols is None:
        rows, cols = _apply_row_bias(rows, cols, row_bias, lum.shape)
    g = _box(lum, rows, cols)
    rows, cols = g.shape
    g = _stretch_grid(g, levels, stretch and not yt_fixed)
    g = _precondition_grid(g, precondition)

    sx = 1.0 if w >= h else w / float(h)
    sy = 1.0 if h >= w else h / float(w)
    xs,ys = grid_axes(w,h,rows,cols)
    return g, xs, ys


def render_luma(lum, n, gamma=2.2, floor=0.012, level=0.9, rows=None,
                cols=None, density=1.0, trim=0.02, stretch=True, bbox=None,
                autofit=True, oversample=1, grid_rows=None, grid_cols=None,
                border=0.0, row_bias=1.0, subcell=True,
                levels=None, fields=1, field=0, palindrome=False,
                reverse=False, start=None, close=None, overscan=1.0,
                precondition=0.0, yt_fixed=False, yt_trigger_samples=0,
                loop_anchor=False, prepared_grid=None):
    """
    Render a luminance image to XY samples.  This is the whole display engine:
    everything above it just decides what the image is.

    lum: 2D float array in [0,1].  Anything that can produce one of those --
    a video frame, a screen grab, a plot -- can be drawn on a scope with this.

    Beam sweeps rows and lingers where the image is bright, so brightness is
    dwell time; that is how a 2-channel DAC with no intensity input paints
    greyscale.  See raster_frame for the VideoInterleaving compositing that
    feeds it.
    """
    h, w = lum.shape
    aspect = h / float(w)
    fields = max(1, int(fields))
    if prepared_grid is None:
        prepared_grid = prepare_render_grid(
            lum, n, density=density, trim=trim, rows=rows, cols=cols,
            autofit=autofit, grid_rows=grid_rows, grid_cols=grid_cols,
            row_bias=row_bias, levels=levels, stretch=stretch,
            fields=fields, precondition=precondition, yt_fixed=yt_fixed)
    g, xs, ys = prepared_grid
    rows, cols = g.shape

    if yt_fixed:
        return render_yt_grid(g, n, aspect=aspect, gamma=gamma, trim=trim,
                              floor=floor, level=level, fields=fields, field=field,
                              border=border, trigger_samples=yt_trigger_samples,
                              oversample=oversample)

    # Aspect-correct extents.  Mapping both axes to +-1 would stretch a
    # non-square image to a square -- vector mode normalises by max(w, h), and
    # raster must match or the same frame is a different shape in each mode.
    sx = 1.0 if w >= h else w / float(h)
    sy = 1.0 if h >= w else h / float(w)

    anchor = (np.zeros(2,np.float32) if start is None
              else np.asarray(start,dtype=np.float32))
    P, wgt, trav = raster_points(
        np.ascontiguousarray(g), xs, ys, float(trim), float(floor), float(gamma),
        fields, int(field), bool(reverse), bool(subcell), anchor, start is not None)
    if not len(P):
        return None

    if close is None:
        close = not (reverse or start is not None)

    if palindrome and len(P) > 2:
        # walk back down the same path: the loop closes on itself, so there is
        # no retrace segment at all.  Same cells, same weights, reversed order.
        P = np.vstack([P, P[-2::-1]])
        wgt = np.concatenate([wgt, wgt[::-1]])
        trav = np.concatenate([trav, trav[::-1]])
    elif loop_anchor and start is not None:
        # A closed repeatable trace must also hand off continuously to the
        # next trace. Make the prior trace endpoint the actual first vertex
        # (not merely an orientation hint), then close the path back to it
        # after overscan/border geometry has been appended below.
        anchor = np.asarray(start, dtype=np.float32).reshape(1, 2)
        if abs(float(level)) > 1e-12:
            anchor = anchor * (float(overscan) / float(level))
        else:
            anchor.fill(0.0)
        _,total = cumulative_mass(wgt)
        entry_weight = max(total * 0.004, 1e-9)
        P = np.vstack((anchor, P))
        wgt = np.concatenate((np.asarray([entry_weight], dtype=wgt.dtype), wgt))
        trav = np.concatenate((np.ones(1, dtype=bool), trav))
    elif close and not loop_anchor:
        P = np.vstack([P, P[0]])
        _,total = cumulative_mass(wgt)
        wgt = np.concatenate([wgt, [total * 0.004]])   # fast dim retrace
        trav = np.concatenate([trav, [True]])

    if overscan > 1.0:
        P, wgt = apply_overscan(P, wgt, trav, overscan)

    if border > 0.0:
        P, wgt = _append_border(P, wgt, sx, sy, border)

    if loop_anchor:
        # Put the repeat seam after all optional geometry. In particular,
        # _append_border appends its own perimeter, which must not leave an
        # otherwise closed frame open at the callback repeat boundary.
        P = np.vstack((P, P[0]))
        _,total = cumulative_mass(wgt)
        wgt = np.concatenate((wgt, [total * 0.004]))
        trav = np.concatenate((trav, np.ones(1, dtype=bool)))

    out = scale_float32(_walk(P,wgt,n,oversample=oversample),float(level))
    # deliberately not mean-centred: the output AC-couples anyway, and
    # subtracting a content-dependent mean would double the brightness drift
    return np.ascontiguousarray(out, dtype=np.float32)


def _append_border(P, wgt, sx, sy, frac):
    """Append a fixed rectangle at the full extent, at the END of the path.

    Without it the drawn extent is whatever the content happens to occupy, so
    dark margins on one side pull that side in and the picture appears to skew
    and rescale as the subject changes. The scope has no absolute reference in
    XY -- the only reference is what the beam actually touches -- so the fix is
    to touch the corners every trace, regardless of content.

    Deliberately AFTER overscan: the border defines the extent, so scaling it
    would defeat the point.

    MEASURED, so as not to over-claim it:
      - extent is pinned exactly. X range holds at +-0.675 whether the subject
        is centred, shifted, narrow or wide; without it the same four cases
        ranged from +-0.175 to +-0.673.
      - the trace END is NOT fixed. Entry is at whichever corner the content
        finished nearest, which keeps the connector short and dim but means
        the exit corner follows the content. Chain continuity is unaffected --
        it just is not the constant it would be nice to claim.
      - it does NOT fix DC drift. At a 4% share the mean is still content-
        dominated (-0.2425 -> -0.2331 on a left-shifted subject). Use
        --scope-dc-comp for coupling, not this.

    `frac` is the share of the trace's samples spent on it -- the honest cost,
    stated the way it is paid. Weight is spread along the perimeter in
    proportion to length so the box is evenly lit rather than bright at the
    corners.
    """
    frac = min(0.5,max(0.0,float(frac)))
    if frac <= 0.0 or len(P) < 2:
        return P, wgt

    _,content_w = cumulative_mass(wgt)
    if content_w <= 0:
        return P, wgt
    return border_extension(P,wgt,float(sx),float(sy),frac)


def _inside(points, loops):
    """Even-odd test of points against closed loops.  Holes come free."""
    cross = np.zeros(len(points), np.int64)
    for V in loops:
        polygon_crossings(np.asarray(points),np.asarray(V),cross)
    return inside_parity(cross)


def merge(main_lib, main_idx, float_lib, float_idx, min_feature=0.02):
    """
    The composite the selector chose, as one polyline list ready for
    Scope.show(): float drawn in full, main culled where the float's
    silhouette matte covers it -- occlusion, not transparency.  Fragments
    shorter than min_feature are dropped as sub-resolution clutter.
    Either library may be None or empty.

    Flags: 0 = stroke, 1 = silhouette (drawn AND occludes), 2 = matte-only
    (occludes, never drawn).  Flag 2 is what makes inverse mattes -- opaque
    background with a face-shaped hole -- occlude correctly: the canvas
    border participates in the even-odd test without drawing a frame.
    """
    m_polys = []
    if main_lib is not None and len(main_lib):
        mp, mf = main_lib.frame(main_idx % len(main_lib))
        m_polys = [p for p, f in zip(mp, mf) if f != 2]
    f_polys, f_flags = [], []
    if float_lib is not None and len(float_lib):
        f_polys, f_flags = float_lib.frame(float_idx % len(float_lib))

    matte = [p for p, f in zip(f_polys, f_flags) if f >= 1]
    f_drawn = [p for p, f in zip(f_polys, f_flags) if f != 2]
    if not matte:
        return m_polys + f_drawn

    out = []
    for p in m_polys:
        mid = segment_midpoints(p)
        runs = outside_runs(_inside(mid,matte))
        for first,last in runs:
            piece = p[first:last]
            if len(piece) >= 2 and path_length(piece) >= min_feature:
                out.append(piece)
    out.extend(f_drawn)
    return out


if __name__ == "__main__":
    # This module is the shared library (format, geometry, merge, raster).
    # The BAKER is utilities/convert_to_xy.py.  The names are close enough
    # that running this by mistake is easy, and exiting silently is the worst
    # possible response -- so forward instead.
    import os
    import runpy
    import sys

    _baker = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "utilities", "convert_to_xy.py")
    if not os.path.exists(_baker):
        sys.exit("scope_bake.py is the shared library, not the baker.\n"
                 "The baker is utilities/convert_to_xy.py, which is missing.")
    if len(sys.argv) == 1:
        print("scope_bake.py is the shared library used by scope_display.py "
              "and the baker;\nit has nothing to run on its own.\n\n"
              "You probably want the baker:\n"
              "  python utilities/convert_to_xy.py -i images -o images_xy "
              "--thumb-width 256\n\n"
              "Other entry points:\n"
              "  python main.py --mode scope ...   run the mode\n"
              "  python test_scope_pair.py ...     preview one pair\n"
              "  python scope_out.py               bench pattern\n"
              "  python scope_lowpass.py ...       output filter test\n"
              "  python verify_scope_files.py      check the file set")
        sys.exit(1)
    print("[note] scope_bake.py is the shared library; the baker is "
          "utilities/convert_to_xy.py.\n[note] forwarding your arguments to "
          "it.\n")
    sys.argv[0] = _baker
    runpy.run_path(_baker, run_name="__main__")
