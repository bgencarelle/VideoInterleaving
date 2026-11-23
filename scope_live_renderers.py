"""Runtime contour and fusion emitters for unbaked captured luminance."""
import numpy as np
from numba import njit

from scope_bake import (TraceEmitter, StochasticEmitter, PositionMultiplexer, plan_grid,
                        trace_luminance_weights, apply_trace_border)
from scope_out import precompensate_hpf, rasterize
from scope_numeric import live_luma


class LiveCaptureTone:
    """Pin transformed luminance once per version and interlace picture."""
    def __init__(self):
        self.invert = False
        self._key = self._lum = None

    def prepare(self, source, version):
        key = (version, self.invert)
        if version is None or key != self._key:
            lum = np.ascontiguousarray(source, dtype=np.float32).copy()
            self._lum = live_luma(lum, True) if self.invert else lum
            self._key = key
        return self._lum, key if version is not None else None


@njit(cache=True, nogil=True, fastmath=False)
def capture_rgb_luma(rgb):
    out = np.empty(rgb.shape[:2], np.float32)
    for y in range(rgb.shape[0]):
        for x in range(rgb.shape[1]):
            out[y, x] = (rgb[y, x, 0]*.2126 + rgb[y, x, 1]*.7152 + rgb[y, x, 2]*.0722)/255.0
    return out


@njit(cache=True, nogil=True, fastmath=False)
def source_reference(gray, size):
    out = np.zeros((size, size, 3), np.uint8)
    h, w = gray.shape
    scale = size / max(h, w)
    height, width = max(1, int(h*scale)), max(1, int(w*scale))
    top, left = (size-height)//2, (size-width)//2
    for y in range(height):
        for x in range(width):
            value = gray[min(h-1, int(y/scale)), min(w-1, int(x/scale))]
            for channel in range(3):
                out[top+y, left+x, channel] = value
    return out


def warm_live_preview():
    """Compile trace and source-reference layouts before audio output starts."""
    from scope_bake import _warm_preview_kernels
    _warm_preview_kernels()
    gray = np.zeros((2, 2), np.uint8)
    source_reference(gray, 8)
    source_reference(gray[:, ::-1], 8)


@njit(cache=True, nogil=True, fastmath=False)
def _band_mask(lum, threshold, gamma, trim):
    out = np.empty(lum.shape, np.uint8)
    for y in range(lum.shape[0]):
        for x in range(lum.shape[1]):
            value = max(0.0, min(1.0, lum[y, x]))
            out[y, x] = 255 if value >= trim and value ** gamma >= threshold else 0
    return out


@njit(cache=True, nogil=True, fastmath=False)
def _contour_trace(points, offsets, width, height, count, start):
    """Order closed contours from the beam, then sample by arc length."""
    paths = len(offsets) - 1
    route = np.empty((len(points) + paths + 2, 2), np.float32)
    route[0] = start
    used = np.zeros(paths, np.bool_)
    length = 1
    scale = max(width, height) / 2.0
    for _ in range(paths):
        best, vertex, distance = -1, 0, 1e30
        for path in range(paths):
            if used[path]:
                continue
            for j in range(offsets[path], offsets[path + 1]):
                px = .9 * (points[j, 0] - width / 2.0) / scale
                py = .9 * (height / 2.0 - points[j, 1]) / scale
                d = (px - route[length - 1, 0]) ** 2 + (py - route[length - 1, 1]) ** 2
                if d < distance:
                    best, vertex, distance = path, j, d
        if best < 0:
            break
        used[best] = True
        begin, end = offsets[best], offsets[best + 1]
        for k in range(end - begin + 1):
            j = begin + (vertex - begin + k) % (end - begin)
            route[length, 0] = .9 * (points[j, 0] - width / 2.0) / scale
            route[length, 1] = .9 * (height / 2.0 - points[j, 1]) / scale
            length += 1
    if length == 1:
        out = np.empty((count, 2), np.float32)
        for i in range(count):
            out[i] = start
        return out
    cumulative = np.zeros(length, np.float64)
    for i in range(1, length):
        dx, dy = route[i, 0] - route[i-1, 0], route[i, 1] - route[i-1, 1]
        cumulative[i] = cumulative[i-1] + max(1e-7, (dx*dx + dy*dy) ** .5)
    out = np.empty((count, 2), np.float32)
    segment = 1
    for i in range(count):
        target = cumulative[-1] * i / max(1, count - 1)
        while segment < length - 1 and cumulative[segment] < target:
            segment += 1
        f = (target - cumulative[segment-1]) / max(1e-7, cumulative[segment] - cumulative[segment-1])
        for axis in range(2):
            out[i, axis] = route[segment-1, axis] + f * (route[segment, axis] - route[segment-1, axis])
    return out


class LiveVectorEmitter:
    """Three luminance-band contours; native extraction, compiled traversal."""
    def __init__(self, samplerate, samples, gamma=1.8, trim=.02, **_options):
        import cv2
        self._cv2 = cv2
        self.samplerate = samplerate
        self.dc_comp = _options.get("dc_comp")
        self.border = _options.get("border", 0.0)
        self.n = int(samples)
        self._idle = rasterize([], self.n)
        self.gamma, self.trim = gamma, trim
        self._end = None
        self._key = self._points = self._offsets = None
        lum = np.zeros((8, 8), np.float32)
        _band_mask(lum, .5, float(gamma), float(trim))
        _band_mask(lum.astype(np.float64), .5, float(gamma), float(trim))
        _contour_trace(np.zeros((0, 2), np.float32), np.zeros(1, np.int64),
                       8, 8, self.n, np.zeros(2, np.float32))

    def reset(self):
        self._end = None

    def checkpoint(self):
        return self._end

    def restore(self, state):
        self._end = state

    def accept(self, endpoint):
        self._end = np.asarray(endpoint, dtype=np.float32).copy()

    def emit(self, lum):
        cv2 = self._cv2
        key = (id(lum), self.gamma, self.trim)
        if key != self._key:
            contours = []
            for threshold in (.2, .45, .7):
                mask = _band_mask(lum, threshold, float(self.gamma), float(self.trim))
                found, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                for contour in found:
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter >= 8:
                        points = cv2.approxPolyDP(contour, .7, True).reshape(-1, 2)
                        if len(points) >= 3:
                            contours.append((perimeter, points))
            contours.sort(key=lambda item: -item[0])
            arrays = [item[1] for item in contours[:64]]
            offsets = [0]
            for array in arrays:
                offsets.append(offsets[-1] + len(array))
            self._points = (np.ascontiguousarray(np.concatenate(arrays), dtype=np.float32)
                            if arrays else np.empty((0, 2), np.float32))
            self._offsets = np.asarray(offsets, dtype=np.int64)
            # Retain the source itself so Python cannot reuse its id for a new frame.
            self._source, self._key = lum, key
        start = np.zeros(2, np.float32) if self._end is None else self._end
        frame = (_contour_trace(self._points, self._offsets, lum.shape[1], lum.shape[0], self.n, start)
                 if len(self._points) else self._idle.copy())
        if self.border:
            frame = apply_trace_border(frame, self.border, aspect=lum.shape[0]/float(lum.shape[1]))
        if self.dc_comp:
            frame = precompensate_hpf(frame, self.dc_comp, self.samplerate)
        return frame


class LiveFusionEmitter:
    """Aligned V/R/S traces through the existing temporal fusion multiplexer."""
    def __init__(self, samplerate, samples, gamma=1.8, trim=.02, **options):
        self.samplerate = samplerate
        self.dc_comp = options.pop("dc_comp", None)
        self.border = options.pop("border", 0.0)
        self.vector = LiveVectorEmitter(samplerate, samples, gamma, trim)
        self.raster = TraceEmitter(samplerate, samples, gamma=gamma, trim=trim,
                                   fields=1, close_frame=False, **options)
        self.walk = StochasticEmitter(samplerate, samples, gamma=gamma, trim=trim)
        self.mux = PositionMultiplexer()
        self._end = None
        # Warm the mux's exact array signature before output starts.
        blank = np.zeros((samples, 2), np.float32)
        self.mux.emit(blank, blank, blank)
        self.mux.reset()

    @property
    def gamma(self):
        return self.vector.gamma

    @gamma.setter
    def gamma(self, value):
        self.vector.gamma = self.raster.gamma = self.walk.gamma = value

    @property
    def trim(self):
        return self.vector.trim

    @trim.setter
    def trim(self, value):
        self.vector.trim = self.raster.trim = self.walk.trim = value

    def reset(self):
        self._end = None
        for emitter in (self.vector, self.raster, self.walk, self.mux):
            emitter.reset()

    def checkpoint(self):
        raster = self.raster
        return (self._end, self.vector.checkpoint(), self.walk.checkpoint(),
                self.mux.checkpoint(), (raster._end, raster._rev, raster._field,
                                        raster._phase, raster._candidate_phase))

    def restore(self, state):
        self._end, vector, walk, mux, raster = state
        self.vector.restore(vector)
        self.walk.restore(walk)
        self.mux.restore(mux)
        (self.raster._end, self.raster._rev, self.raster._field,
         self.raster._phase, self.raster._candidate_phase) = raster

    def accept(self, endpoint):
        self._end = np.asarray(endpoint, dtype=np.float32).copy()
        self.vector.accept(endpoint)
        self.raster.accept(endpoint)
        self.walk.chain_from(endpoint)

    def emit(self, lum):
        if self.raster.grid is None:
            self.raster.grid = plan_grid(lum, self.raster.n, density=self.raster.density,
                                          trim=self.raster.trim, fields=1)
        vector = self.vector.emit(lum)
        raster = self.raster.emit(lum, commit=False)
        walk = self.walk.emit(lum)
        weights = {name: trace_luminance_weights(lum, frame, gamma=self.gamma, trim=self.trim)
                   for name, frame in (("v", vector), ("r", raster), ("s", walk))}
        frame = self.mux.emit(vector, raster, walk, weights=weights)
        if self.border:
            frame = apply_trace_border(frame, self.border, aspect=lum.shape[0]/float(lum.shape[1]))
        if self.dc_comp:
            frame = precompensate_hpf(frame, self.dc_comp, self.samplerate)
        return frame
