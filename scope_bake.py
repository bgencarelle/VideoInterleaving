"""
scope_bake.py -- shared vector-library toolkit for scope mode.

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
import json
from pathlib import Path

import numpy as np

Q = 32767.0


# ---------------------------------------------------------------- geometry

def path_length(p):
    d = np.diff(p, axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def subdivide(p, max_seg):
    """Insert vertices so no segment exceeds max_seg.  Bake-time step that
    bounds the error of the runtime midpoint-in-matte occlusion test."""
    seg = np.diff(p, axis=0)
    L = np.hypot(seg[:, 0], seg[:, 1])
    n = np.maximum(1, np.ceil(L / max_seg).astype(int))
    if (n == 1).all():
        return p
    out = [p[:1]]
    for i, k in enumerate(n):
        if k == 1:
            out.append(p[i + 1:i + 2])
        else:
            t = np.linspace(0.0, 1.0, k + 1)[1:, None]
            out.append(p[i] + t * (p[i + 1] - p[i]))
    return np.vstack(out)


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
            d = np.hypot(c[:, 0] - pos[0], c[:, 1] - pos[1])
            if flags[r]:
                j = int(np.argmin(d))
                cand = (float(d[j]), r, j)
            else:
                cand = ((float(d[0]), r, 0) if d[0] <= d[-1]
                        else (float(d[-1]), r, -1))
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
        self.flip_y = flip_y
        self.verts = np.load(p / "verts.npy", mmap_mode="r")
        self.poly = np.load(p / "poly_starts.npy")
        self.fstart = np.load(p / "frame_starts.npy")
        fpath = p / "flags.npy"
        self.flags = np.load(fpath) if fpath.exists() else None
        npath = p / "names.json"
        self.names = json.loads(npath.read_text()) if npath.exists() else []
        tpath = p / "thumbs.npy"
        self.thumbs = np.load(tpath, mmap_mode="r") if tpath.exists() else None

    def __len__(self):
        return len(self.fstart) - 1

    def frame(self, i):
        a, b = self.fstart[i], self.fstart[i + 1]
        polys, flags = [], []
        for k in range(a, b):
            v = np.asarray(self.verts[self.poly[k]:self.poly[k + 1]],
                           np.float32) / Q
            if self.flip_y:
                v = v * np.array([1.0, -1.0], np.float32)
            polys.append(v)
            if self.flags is not None:
                flags.append(int(self.flags[k]))
            else:                                  # legacy bake: infer
                flags.append(int(len(v) > 2 and np.array_equal(v[0], v[-1])))
        return polys, flags


    def thumb(self, i):
        """(h, w, 2) uint8 [luminance, alpha], or None if not baked."""
        if self.thumbs is None:
            return None
        return np.asarray(self.thumbs[i % len(self.thumbs)])


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
        freqs = np.fft.rfftfreq(k, d=1.0 / k)          # cycles per frame
        # keep everything the decimated rate can represent, roll off above
        mag = 1.0 / np.sqrt(1.0 + (freqs / (0.45 * n)) ** 16)
        out = np.empty_like(fine)
        for ch in range(fine.shape[1]):
            out[:, ch] = np.fft.irfft(np.fft.rfft(fine[:, ch]) * mag, n=k)
        return out[::oversample]
    return _walk_raw(P, w, n)


def _walk_raw(P, w, n):
    """Sample n points along polyline P, spending time per arbitrary weights w.
    Same machinery as rasterize, but the weights are brightness rather than
    length -- that substitution is what turns dwell time into intensity."""
    w = np.maximum(np.asarray(w, np.float64), 1e-12)
    cum = np.concatenate([[0.0], np.cumsum(w)])
    t = np.linspace(0.0, cum[-1], n, endpoint=False)
    i = np.clip(np.searchsorted(cum, t, side="right") - 1, 0, len(w) - 1)
    f = ((t - cum[i]) / w[i])[:, None]
    return P[i] + f * (P[i + 1] - P[i])


def _box(img, rows, cols):
    """Exact area-average resample to EXACTLY (rows, cols).

    An integer block-size reshape does not work here: the source is a 96-wide
    thumbnail, so h // rows rounds and the derived grid comes out finer than
    asked for -- more cells than samples, which renders as broken sparse rows
    instead of solid scanlines.  Summed-area table gives the exact grid.
    """
    h, w = img.shape[:2]
    rows = max(1, min(int(rows), h))
    cols = max(1, min(int(cols), w))
    ys = (np.arange(rows + 1) * h) // rows
    xs = (np.arange(cols + 1) * w) // cols
    ys[-1], xs[-1] = h, w
    c = np.pad(img.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    y0, y1, x0, x1 = ys[:-1], ys[1:], xs[:-1], xs[1:]
    total = (c[np.ix_(y1, x1)] - c[np.ix_(y0, x1)]
             - c[np.ix_(y1, x0)] + c[np.ix_(y0, x0)])
    area = np.outer(np.maximum(y1 - y0, 1), np.maximum(x1 - x0, 1))
    return total / area


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
                 lum_fn=None, auto_levels=0.0):
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
        self._out = np.zeros((0, 2), np.float32)
        self._plan = None
        self._budgets = None
        self._row_i = 0
        self._reverse = False
        self._last = None
        self._grid_key = None
        self._grid = None
        self.passes = 0
        self.row_switches = 0

    def _live(self):
        """Grid + axes for a live luminance source."""
        lum = np.asarray(self.lum_fn(), dtype=np.float32)
        if lum.ndim == 3:
            lum = lum.mean(axis=2)
        if lum.max() > 1.5:
            lum = lum / 255.0
        h, w = lum.shape
        if self._grid is None or self._lum_shape != (h, w):
            self._lum_shape = (h, w)
            aspect = h / float(w)
            if self.grid_rows and self.grid_cols:
                rws, cls = int(self.grid_rows), int(self.grid_cols)
            else:
                cells = max(64.0, self.n_pass / max(self.density, 0.25))
                cls = max(8, int(np.sqrt(cells / max(aspect, 1e-6))))
                rws = max(6, int(round(cls * aspect)))
            rws, cls = min(rws, h), min(cls, w)
            sx = 1.0 if w >= h else w / float(h)
            sy = 1.0 if h >= w else h / float(w)
            self._axes = (np.linspace(-sx, sx, cls, dtype=np.float32),
                          -np.linspace(-sy, sy, rws, dtype=np.float32))
            self._dims = (rws, cls)
        rws, cls = self._dims
        g = _box(lum, rws, cls)

        if self.levels is not None:
            lo, hi = self.levels
        elif self.auto_levels > 0:
            lit = g[g > 0.01]
            if lit.size > 16:
                lo_n, hi_n = np.percentile(lit, 2), np.percentile(lit, 98)
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
            g = np.clip((g - lo) / (hi - lo), 0.0, 1.0)
        xs, ys = self._axes
        self._grid = (g, xs, ys)
        return self._grid

    # -- geometry -------------------------------------------------------
    def _composite(self, st):
        key = (id(st.get("main")), st.get("mi"), id(st.get("float")), st.get("fi"))
        if key == self._grid_key and self._grid is not None:
            return self._grid
        ml, fl = st.get("main"), st.get("float")
        tm = ml.thumb(st["mi"]) if ml is not None and len(ml) else None
        tf = fl.thumb(st["fi"]) if fl is not None and len(fl) else None
        if tm is None and tf is None:
            return None

        def split(t):
            return (t[..., 0].astype(np.float64) / 255.0,
                    t[..., 1].astype(np.float64) / 255.0)

        if tf is not None and tm is not None and tf.shape[:2] == tm.shape[:2]:
            lf, af = split(tf); lm, am = split(tm)
            lum = lf * af + lm * am * (1.0 - af)
        elif tf is not None:
            lf, af = split(tf); lum = lf * af
        else:
            lm, am = split(tm); lum = lm * am

        if self.bbox is not None:
            hh, ww = lum.shape
            x0, y0, x1, y1 = self.bbox
            lum = lum[int(y0 * hh):max(int(y1 * hh), int(y0 * hh) + 1),
                      int(x0 * ww):max(int(x1 * ww), int(x0 * ww) + 1)]

        h, w = lum.shape
        aspect = h / float(w)
        if self.grid_rows and self.grid_cols:
            rws, cls = int(self.grid_rows), int(self.grid_cols)
        else:
            cells = max(64.0, self.n_pass / max(self.density, 0.25))
            if self.rows_override:
                rws = int(self.rows_override); cls = max(8, int(cells / rws))
            else:
                cls = max(8, int(np.sqrt(cells / max(aspect, 1e-6))))
                rws = max(6, int(round(cls * aspect)))
        g = _box(lum, min(rws, h), min(cls, w))
        rws, cls = g.shape
        if self.levels is not None:
            lo, hi = self.levels
            if hi > lo:
                g = np.clip((g - lo) / (hi - lo), 0.0, 1.0)
        else:
            lit = g[g > 0.01]
            if lit.size > 16:
                lo, hi = np.percentile(lit, 2), np.percentile(lit, 98)
                if hi > lo:
                    g = np.clip((g - lo) / (hi - lo), 0.0, 1.0)
        sx = 1.0 if w >= h else w / float(h)
        sy = 1.0 if h >= w else h / float(w)
        self._grid_key = key
        self._grid = (g, np.linspace(-sx, sx, cls), -np.linspace(-sy, sy, rws))
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
        wsum = np.zeros(len(seq), dtype=np.float64)
        for k, r in enumerate(seq):
            row = g[r]
            lit = row[row > self.trim] if self.trim > 0 else row
            if lit.size > 1:
                wsum[k] = float((np.maximum(lit[1:], self.floor) ** self.gamma).sum())
        total = wsum.sum()
        if total <= 0:
            self._budgets = np.full(len(seq), max(2, self.n_pass // max(len(seq), 1)))
        else:
            share = wsum / total * self.n_pass
            self._budgets = np.maximum(2, np.round(share)).astype(int)

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
        row = g[r]
        idx = np.flatnonzero(row > self.trim) if self.trim > 0 else np.arange(len(row))
        if idx.size == 0:
            return None
        a, b = int(idx[0]), int(idx[-1]) + 1
        seg_x, seg_v = xs[a:b], row[a:b]
        if self._last is None:
            flip = (r % 2) == 1
        else:
            flip = abs(seg_x[-1] - self._last[0]) < abs(seg_x[0] - self._last[0])
        if flip:
            seg_x, seg_v = seg_x[::-1], seg_v[::-1]
        P = np.stack([seg_x, np.full(seg_x.shape, ys[r])], axis=1)
        W = np.maximum(seg_v[1:], self.floor) ** self.gamma
        if self._last is not None:
            P = np.vstack([self._last[None, :], P])
            W = np.concatenate([[1e-3], W])
        if len(P) < 2:
            return None
        return _walk(P, W, max(2, int(budget)))

    # -- audio callback interface --------------------------------------
    def __call__(self, n):
        while len(self._out) < n:
            st = self.state_fn() if self.state_fn is not None else None
            if self._plan is None or self._row_i >= len(self._plan):
                self._reverse = not self._reverse
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
            self._out = np.vstack([self._out, (chunk * self.level).astype(np.float32)])
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
    v = 1.0 / overscan                      # visible half-extent after scaling
    park = 1.0                              # excursions go to the full range
    P = np.asarray(P, dtype=np.float64) * v
    travel = np.asarray(travel, dtype=bool)

    def push(pt):
        """Shortest way out of the visible box."""
        dx = v - abs(pt[0])
        dy = v - abs(pt[1])
        out = pt.copy()
        if dx <= dy:
            out[0] = park if pt[0] >= 0 else -park
        else:
            out[1] = park if pt[1] >= 0 else -park
        return out

    n_travel = int(np.count_nonzero(travel[:len(P) - 1]))
    if n_travel == 0:
        return P, W
    # Weight each excursion enough that samples actually land on it, otherwise
    # the beam never leaves the screen and the whole exercise is a no-op.
    content_w = float(W[~travel[:len(W)]].sum()) if len(W) else 1.0
    per_leg = (content_w * travel_frac / max(1.0 - travel_frac, 1e-6)
               / max(n_travel * 3, 1))

    new_P, new_W = [P[0]], []
    for i in range(len(P) - 1):
        if i < len(travel) and travel[i]:
            a, b = push(P[i]), push(P[i + 1])
            new_P.extend([a, b, P[i + 1]])
            new_W.extend([per_leg, per_leg, per_leg])
        else:
            new_P.append(P[i + 1])
            new_W.append(W[i])
    return np.vstack(new_P), np.asarray(new_W, dtype=np.float64)


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
        cols = max(8, int(np.sqrt(cells / max(aspect, 1e-6))))
        rows = max(6, int(round(cols * aspect)))

    if autofit and trim > 0:
        probe = _box(lum, rows, cols)
        lit0 = probe[probe > 0.01]
        if lit0.size > 16:
            lo0, hi0 = np.percentile(lit0, 2), np.percentile(lit0, 98)
            if hi0 > lo0:
                probe = np.clip((probe - lo0) / (hi0 - lo0), 0.0, 1.0)
        frac = float((probe > trim).mean())
        if 0.05 < frac < 0.95:
            grow = min(1.0 / np.sqrt(frac), 2.5)
            rows = max(6, min(int(round(rows * grow)), lum.shape[0]))
            cols = max(8, min(int(round(cols * grow)), lum.shape[1]))
    rows, cols = _apply_row_bias(rows, cols, row_bias, lum.shape)
    return int(rows), int(cols)


def _apply_row_bias(rows, cols, bias, shape):
    """Trade columns for rows at constant cell count."""
    b = float(bias)
    if b == 1.0 or b <= 0:
        return rows, cols
    k = np.sqrt(b)
    return (max(6, min(int(round(rows * k)), shape[0])),
            max(8, min(int(round(cols / k)), shape[1])))


def preview_frame(samples, size=384, spot=1.2, exposure=1.0, max_split=192):
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
    samples = np.asarray(samples, dtype=np.float32)
    if len(samples) < 2:
        return np.zeros((size, size, 3), np.uint8)

    px = (samples[:, 0] + 1.0) * 0.5 * (size - 1)
    py = (1.0 - samples[:, 1]) * 0.5 * (size - 1)     # y up -> row down
    x0, x1 = px[:-1], px[1:]
    y0, y1 = py[:-1], py[1:]

    # max_split bounds the cost: a single sample cannot cost more than this
    # many splats however far the beam jumped.  Only a flyback gets near it.
    d = np.hypot(x1 - x0, y1 - y0)
    m = np.clip(np.ceil(d), 1, max_split).astype(np.int64)

    seg = np.repeat(np.arange(len(m)), m)
    starts = np.concatenate(([0], np.cumsum(m)[:-1]))
    j = np.arange(int(m.sum())) - np.repeat(starts, m)
    t = (j + 0.5) / m[seg]

    xs = np.clip(x0[seg] + (x1[seg] - x0[seg]) * t, 0, size - 1).astype(np.int32)
    ys = np.clip(y0[seg] + (y1[seg] - y0[seg]) * t, 0, size - 1).astype(np.int32)
    wt = (1.0 / m[seg]).astype(np.float32)

    acc = np.bincount(ys * size + xs, weights=wt,
                      minlength=size * size).reshape(size, size)
    acc = cv2.GaussianBlur(acc.astype(np.float32), (0, 0), spot)
    lit = acc[acc > 0]
    gain = exposure * 2.5 / max(float(np.percentile(lit, 75)), 1e-6) if lit.size else 1.0
    v = 1.0 - np.exp(-acc * gain)
    img = np.stack([v * 0.35, v * 1.0, v * 0.25], axis=-1) + 0.03
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def calibrate(main_libs, float_libs, n_samples, density=1.0, trim=0.02,
              rows=None, cols=None, bbox=None, frames=24, fields=1,
              row_bias=1.0):
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
        c0 = max(8, int(np.sqrt(cells / max(aspect, 1e-6))))
        r0 = max(6, int(round(c0 * aspect)))

    # gather luminance statistics and trim coverage over a spread of frames
    los, his, fracs = [], [], []
    for lib in libs[:8]:
        F = len(lib.thumbs)
        step = max(1, F // max(frames // max(len(libs[:8]), 1), 1))
        for i in range(0, F, step):
            t = np.asarray(lib.thumbs[i])
            v = (t[..., 0] / 255.0) * (t[..., 1] / 255.0)
            g = _box(v, min(r0, v.shape[0]), min(c0, v.shape[1]))
            lit = g[g > 0.01]
            if lit.size < 16:
                continue
            lo, hi = np.percentile(lit, 2), np.percentile(lit, 98)
            los.append(lo); his.append(hi)
            if hi > lo:
                fracs.append(float((np.clip((g - lo) / (hi - lo), 0, 1) > trim).mean()))

    out = {}
    if los:
        # median, not mean: robust to the odd blank or blown-out frame
        out["levels"] = (float(np.median(los)), float(np.median(his)))
    if fracs and trim > 0:
        frac = float(np.median(fracs))
        if 0.05 < frac < 0.95:
            grow = min(1.0 / np.sqrt(frac), 2.5)
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
            t = np.asarray(lib.thumbs[i])
            v = (t[..., 0] / 255.0) * (t[..., 1] / 255.0)
            ys, xs = np.nonzero(v > thresh)
            if ys.size == 0:
                continue
            hh, ww = v.shape
            y0 = min(y0, ys.min() / hh); y1 = max(y1, (ys.max() + 1) / hh)
            x0 = min(x0, xs.min() / ww); x1 = max(x1, (xs.max() + 1) / ww)
    if y1 <= y0 or x1 <= x0:
        return None
    return (max(0.0, x0 - pad), max(0.0, y0 - pad),
            min(1.0, x1 + pad), min(1.0, y1 + pad))


# hoisted: these were being allocated once per row per frame and showed in the
# profile.  They are read-only, so one shared copy is safe.
_TRAVEL_W = np.float32([1e-3])
_TRAVEL_T = np.ones(1, dtype=bool)


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
                 autofit=True):
        self.samplerate = samplerate
        self.n = int(samples)
        self.gamma, self.trim, self.density = gamma, trim, density
        self.rows, self.fields = rows, max(1, int(fields))
        self.border, self.oversample = border, oversample
        self.sweep_mode, self.dc_comp = sweep, dc_comp
        self.grid, self.levels, self.autofit = grid, levels, autofit
        self._rev, self._end, self._field = False, None, 0

    def reset(self):
        """Drop chain state. Call after a device swap: the beam is not where
        the chain thinks it is, and continuing would jump the full screen."""
        self._rev, self._end, self._field = False, None, 0

    def emit(self, lum, levels=None):
        """One trace, or None if there is nothing to draw."""
        if lum is None:
            return None
        alt = self.sweep_mode == "alternate"
        kw = {}
        if self.grid:
            kw["grid_rows"], kw["grid_cols"] = self.grid
        lv = levels if levels is not None else self.levels
        frame = render_luma(
            lum, self.n, gamma=self.gamma, trim=self.trim,
            density=self.density, rows=self.rows, autofit=self.autofit,
            oversample=self.oversample, border=self.border,
            fields=self.fields, field=self._field % self.fields,
            levels=lv, palindrome=(self.sweep_mode == "palindrome"),
            reverse=(alt and self._rev),
            start=self._end if alt else None,
            close=(self.sweep_mode == "retrace"), **kw)
        if frame is None:
            return None
        self._field += 1
        if alt:
            self._rev = not self._rev
            # captured BEFORE compensation: the chain continues from where the
            # geometry says the beam is, not from the boosted sample
            self._end = frame[-1]
        if self.dc_comp:
            from scope_out import precompensate_hpf
            frame = precompensate_hpf(frame, self.dc_comp, self.samplerate)
        return frame


def composite_luma(main_lib, main_idx, float_lib, float_idx, bbox=None):
    """The interleaved composite as a luminance array, and nothing else.

    Split out of raster_frame so there is ONE place that decides what the
    picture is, separate from the one place that decides how to sweep it.
    Everything downstream -- mode scope, scope_screen, the preview -- takes a
    luminance array, so they can share a single emitter instead of each
    re-assembling the same argument list and drifting apart.
    """
    tm = main_lib.thumb(main_idx) if main_lib is not None and len(main_lib) else None
    tf = float_lib.thumb(float_idx) if float_lib is not None and len(float_lib) else None
    if tm is None and tf is None:
        return None

    def split(t):
        return (t[..., 0].astype(np.float64) / 255.0,
                t[..., 1].astype(np.float64) / 255.0)

    if tf is not None and tm is not None and tf.shape[:2] == tm.shape[:2]:
        lf, af = split(tf)
        lm, am = split(tm)
        lum = lf * af + lm * am * (1.0 - af)
    elif tf is not None:
        lf, af = split(tf)
        lum = lf * af
    else:
        lm, am = split(tm)
        lum = lm * am

    if bbox is not None:                       # crop to the subject so the
        hh, ww = lum.shape                     # budget is spent on content
        x0, y0, x1, y1 = bbox
        lum = lum[int(y0 * hh):max(int(y1 * hh), int(y0 * hh) + 1),
                  int(x0 * ww):max(int(x1 * ww), int(x0 * ww) + 1)]
    return lum


def raster_frame(main_lib, main_idx, float_lib, float_idx, n,
                 gamma=2.2, floor=0.012, level=0.9, rows=None, cols=None,
                 density=1.0, trim=0.02, stretch=True, bbox=None, autofit=True,
                 oversample=1, grid_rows=None, grid_cols=None, levels=None,
                 fields=1, field=0, palindrome=False, reverse=False, start=None,
                 close=None, overscan=1.0, border=0.0, row_bias=1.0,
                 subcell=True):
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
    lum = composite_luma(main_lib, main_idx, float_lib, float_idx, bbox=bbox)
    if lum is None:
        return None
    return render_luma(lum, n, gamma=gamma, floor=floor, level=level,
                       rows=rows, cols=cols, density=density, trim=trim,
                       stretch=stretch, bbox=bbox, autofit=autofit, border=border,
                       row_bias=row_bias, subcell=subcell,
                       oversample=oversample, grid_rows=grid_rows,
                       grid_cols=grid_cols, levels=levels, fields=fields,
                       field=field, palindrome=palindrome, reverse=reverse,
                       start=start, close=close, overscan=overscan)


def render_luma(lum, n, gamma=2.2, floor=0.012, level=0.9, rows=None,
                cols=None, density=1.0, trim=0.02, stretch=True, bbox=None,
                autofit=True, oversample=1, grid_rows=None, grid_cols=None,
                border=0.0, row_bias=1.0, subcell=True,
                levels=None, fields=1, field=0, palindrome=False,
                reverse=False, start=None, close=None, overscan=1.0):
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
    if density < 0.25:
        # guard against division blow-up, but say so rather than clamp silently
        print(f"[SCOPE] density {density} clamped to 0.25 -- below that the grid "
              "outruns the sample count so far that most cells go unvisited")
        density = 0.25
    fields = max(1, int(fields))
    # Size the grid for the WHOLE picture, not one field.  A field draws
    # rows[field::fields] using n samples, so a complete picture costs
    # n*fields samples spread over `fields` traces -- that total is the
    # budget the grid must match.  Sizing from n alone shrank the grid by
    # sqrt(fields) and handed the freed samples back as pointless extra
    # dwell, which is why interlacing bought refresh at the cost of
    # resolution instead of for free.
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
        cols = max(8, int(np.sqrt(cells / max(aspect, 1e-6))))
        rows = max(6, int(round(cols * aspect)))

    if grid_rows and grid_cols:
        # calibrated once for the whole run: a grid that changes size between
        # frames re-quantizes every cell boundary and shows as popping
        rows, cols = int(grid_rows), int(grid_cols)
        autofit = False

    if autofit and trim > 0:
        # One cheap probe pass: measure what fraction of the grid survives
        # trim, then grow the grid so the SURVIVING cells match the budget.
        probe = _box(lum, rows, cols)
        lit0 = probe[probe > 0.01]
        if lit0.size > 16:
            lo0, hi0 = np.percentile(lit0, 2), np.percentile(lit0, 98)
            if hi0 > lo0:
                probe = np.clip((probe - lo0) / (hi0 - lo0), 0.0, 1.0)
        frac = float((probe > trim).mean())
        if 0.05 < frac < 0.95:
            grow = min(1.0 / np.sqrt(frac), 2.5)      # cap the correction
            rows = max(6, min(int(round(rows * grow)), lum.shape[0]))
            cols = max(8, min(int(round(cols * grow)), lum.shape[1]))

    if grid_rows is None or grid_cols is None:
        rows, cols = _apply_row_bias(rows, cols, row_bias, lum.shape)
    g = _box(lum, rows, cols)
    rows, cols = g.shape
    if levels is not None:
        # fixed tone mapping: a per-frame stretch drifts, pushing cells across
        # the trim threshold so they wink in and out as flecks
        lo, hi = levels
        if hi > lo:
            g = np.clip((g - lo) / (hi - lo), 0.0, 1.0)
    elif stretch:
        lit = g[g > 0.01]
        if lit.size > 16:
            lo, hi = np.percentile(lit, 2), np.percentile(lit, 98)
            if hi > lo:
                g = np.clip((g - lo) / (hi - lo), 0.0, 1.0)

    # Aspect-correct extents.  Mapping both axes to +-1 would stretch a
    # non-square image to a square -- vector mode normalises by max(w, h), and
    # raster must match or the same frame is a different shape in each mode.
    sx = 1.0 if w >= h else w / float(h)
    sy = 1.0 if h >= w else h / float(w)
    xs = np.linspace(-sx, sx, cols)
    ys = -np.linspace(-sy, sy, rows)            # screen y-down -> scope y-up

    # Build the serpentine row by row, keeping only the lit span of each row.
    # Empty rows are skipped outright; the beam jumps to the next row of
    # content, and that jump is weighted low so it stays dim.
    # Build the sweep row by row, appending whole ARRAYS rather than points.
    # The per-point version cost ~29k list appends per frame and dominated the
    # profile; assembling per row is the same geometry in ~1/300th the calls.
    #
    # Segment layout: within a row of K points there are K-1 drawn segments;
    # between rows there is exactly one travel segment, weighted low so the
    # beam crosses fast and dim.
    row_pts, row_w, row_t = [], [], []
    prev = None if start is None else np.asarray(start, dtype=np.float32)
    row_seq = list(range(int(field) % fields, rows, fields))
    if reverse:
        row_seq = row_seq[::-1]

    # one masked reduction for the whole grid beats flatnonzero per row
    if trim > 0:
        lit_mask = g > trim
        any_lit = lit_mask.any(axis=1)
        first_lit = lit_mask.argmax(axis=1)
        last_lit = cols - 1 - lit_mask[:, ::-1].argmax(axis=1)
    else:
        any_lit = None

    for r in row_seq:
        row = g[r]
        if any_lit is not None:
            if not any_lit[r]:
                continue
            a0, b0 = int(first_lit[r]), int(last_lit[r]) + 1
        else:
            a0, b0 = 0, cols
        seg_x = xs[a0:b0]
        seg_v = row[a0:b0]

        if subcell and trim > 0 and len(seg_x) > 1:
            # VERNIER ACUITY.  The eye resolves a misaligned edge roughly ten
            # times finer than it resolves two separate lines -- a few arcsec
            # against about a minute.  So the silhouette's POSITION is read far
            # more precisely than the grid that produced it, and snapping each
            # row's end to a whole cell is visible as stair-stepping even
            # though the cell itself is below the resolution limit.
            #
            # The luminance crossing between the last dark cell and the first
            # lit one gives the edge to a fraction of a cell, and moving the
            # endpoint there costs nothing: same point count, same samples,
            # same brightness.  It is the cheapest perceptual win in the
            # renderer, and it only works because the beam is analogue -- there
            # is no pixel to snap to.
            xstep = xs[1] - xs[0]
            if a0 > 0:
                g0, g1 = row[a0 - 1], row[a0]
                if g1 > g0:
                    seg_x = seg_x.copy()
                    seg_x[0] -= xstep * float(np.clip((g1 - trim) / (g1 - g0), 0.0, 1.0))
            if b0 < cols:
                g0, g1 = row[b0 - 1], row[b0]
                if g0 > g1:
                    seg_x = seg_x if seg_x.base is None else seg_x
                    seg_x = np.array(seg_x, copy=True)
                    seg_x[-1] += xstep * float(np.clip((g0 - trim) / (g0 - g1), 0.0, 1.0))
        if prev is None:
            flip = ((r % 2) == 1) if not reverse else ((r % 2) == 0)
        else:
            flip = abs(seg_x[-1] - prev[0]) < abs(seg_x[0] - prev[0])
        if flip:
            seg_x = seg_x[::-1]
            seg_v = seg_v[::-1]

        k = seg_x.shape[0]
        pts = np.empty((k, 2), dtype=np.float32)
        pts[:, 0] = seg_x
        pts[:, 1] = ys[r]
        w_in = np.maximum(seg_v[1:], floor).astype(np.float32)

        if row_pts:                                   # travel into this row
            row_w.append(_TRAVEL_W)
            row_t.append(_TRAVEL_T)
        row_pts.append(pts)
        row_w.append(w_in)
        row_t.append(np.zeros(k - 1, dtype=bool))
        prev = pts[-1]

    if not row_pts:
        return None
    P = np.concatenate(row_pts, axis=0)
    wgt = np.concatenate(row_w) if row_w else np.ones(max(len(P) - 1, 1), np.float32)
    trav = np.concatenate(row_t) if row_t else np.zeros(max(len(P) - 1, 1), bool)
    if len(wgt) != len(P) - 1:                        # defensive; shapes should agree
        wgt = np.resize(wgt, max(len(P) - 1, 1))
        trav = np.resize(trav, max(len(P) - 1, 1))
    wgt = np.maximum(wgt, 1e-9) ** gamma

    if close is None:
        close = not (reverse or start is not None)

    if palindrome and len(P) > 2:
        # walk back down the same path: the loop closes on itself, so there is
        # no retrace segment at all.  Same cells, same weights, reversed order.
        P = np.vstack([P, P[-2::-1]])
        wgt = np.concatenate([wgt, wgt[::-1]])
        trav = np.concatenate([trav, trav[::-1]])
    elif close:
        P = np.vstack([P, P[0]])
        wgt = np.concatenate([wgt, [wgt.sum() * 0.004]])   # fast dim retrace
        trav = np.concatenate([trav, [True]])

    if overscan > 1.0:
        P, wgt = apply_overscan(P, wgt, trav, overscan)

    if border > 0.0:
        P, wgt = _append_border(P, wgt, sx, sy, border)

    out = _walk(P, wgt, n, oversample=oversample) * level
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
    frac = float(np.clip(frac, 0.0, 0.5))
    if frac <= 0.0 or len(P) < 2:
        return P, wgt

    corners = np.array([[-sx, -sy], [sx, -sy], [sx, sy], [-sx, sy]], np.float32)
    # enter at whichever corner the content ended nearest, so the connector is
    # as short -- and therefore as dim and as cheap -- as it can be
    k = int(np.argmin(np.hypot(*(corners - P[-1]).T)))
    loop = np.vstack([corners[k:], corners[:k], corners[k:k + 1]])

    seg = np.hypot(*np.diff(loop, axis=0).T)          # 4 sides
    content_w = float(wgt.sum())
    if content_w <= 0:
        return P, wgt
    border_w = content_w * frac / (1.0 - frac)

    # connector from the content's last point to the entry corner: travel, so
    # give it almost nothing and let it stay a faint line
    P = np.vstack([P, loop])
    wgt = np.concatenate([
        wgt,
        [content_w * 0.002],                          # the connector
        (seg / seg.sum() * border_w).astype(wgt.dtype),
    ])
    return P, wgt


def _inside(points, loops):
    """Even-odd test of points against closed loops.  Holes come free."""
    cross = np.zeros(len(points), np.int64)
    px, py = points[:, 0], points[:, 1]
    for V in loops:
        if not np.array_equal(V[0], V[-1]):
            V = np.vstack([V, V[:1]])
        x1, y1 = V[:-1, 0][:, None], V[:-1, 1][:, None]
        x2, y2 = V[1:, 0][:, None], V[1:, 1][:, None]
        cond = (y1 > py) != (y2 > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
        cross += (cond & (px < xi)).sum(axis=0)
    return (cross % 2) == 1


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
        mid = 0.5 * (p[:-1] + p[1:])
        keep = ~_inside(mid, matte)
        idx = np.flatnonzero(keep)
        if len(idx) == 0:
            continue
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        for r in runs:
            piece = p[r[0]:r[-1] + 2]
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
              "--thumb-width 128\n\n"
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