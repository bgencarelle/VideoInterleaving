"""Lazy runtime thumbnails for the unbaked scope image source.

The baked scope path stays entirely image-loader-free. This adapter is only
constructed for ``--scope-source images``; it decodes the current image pair
on a small worker pool and exposes the same ``thumb()`` interface as an
``XYLibrary``. Geometry remains an offline-bake feature.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

import numpy as np


class _ThumbShape:
    """Array-like shape/length facade used by scope calibration and checks."""

    def __init__(self, library):
        self.library = library
        self.shape = (library.source.frames, library.source.height,
                      library.source.width, 2)
        self.ndim = 4

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        return self.library.thumb(index)


class RuntimeThumbnailLibrary:
    """One folder of source images presented as lazily decoded thumbnails."""

    def __init__(self, source, layer, folder):
        self.source = source
        self.layer = layer
        self.folder = int(folder)
        self.thumbs = _ThumbShape(self)
        self.stipple_xy = None
        self.stipple_lae = None
        self.stipple_mass = None
        self.verts = np.empty((0, 2), dtype=np.int16)
        self.flags = None
        self.format = {"thumbnail_channels": ["raw_luminance", "alpha"]}

    def __len__(self):
        return self.source.frames

    def thumb(self, index):
        return self.source.thumb(self.layer, self.folder, index)

    @property
    def raw_thumbnail(self):
        return True

    @property
    def raster_precondition(self):
        return 0.0

    def stipple(self, _index):
        return None


class RuntimeScopeImageSource:
    """Decode source images on demand and prefetch the current pair ahead.

    ``main_paths`` and ``float_paths`` use make_file_lists' existing ordering,
    so the image-clock and folder-selector choices agree with the other modes.
    The fixed thumbnail canvas prevents a mixed source aspect ratio from
    changing the renderer's sample grid; each image is contained without
    stretching, with transparent padding.
    """

    def __init__(self, main_paths, float_paths, *, width=128, workers=2,
                 cache_size=16):
        self.main_paths = self._normalize_paths(main_paths, "main")
        self.float_paths = self._normalize_paths(float_paths, "float")
        if not self.main_paths or not self.float_paths:
            raise ValueError("live scope needs both face and float image folders")
        if len(self.main_paths) != len(self.float_paths):
            raise ValueError("live scope face/float frame counts differ")
        if not self.main_paths[0] or not self.float_paths[0]:
            raise ValueError("live scope has an empty face or float folder list")

        self.frames = len(self.main_paths)
        self.width = max(16, int(width))
        self.height = self._canvas_height(self.main_paths[0][0], self.width)
        self.main_libs = [RuntimeThumbnailLibrary(self, "main", i)
                          for i in range(len(self.main_paths[0]))]
        self.float_libs = [RuntimeThumbnailLibrary(self, "float", i)
                           for i in range(len(self.float_paths[0]))]
        self.cache_size = max(2, int(cache_size))
        self._cache = OrderedDict()
        self._pending = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, int(workers)), thread_name_prefix="scope-images")

    @staticmethod
    def _normalize_paths(paths, layer):
        root = Path(__file__).resolve().parent
        normalized = []
        for frame in paths:
            row = []
            for path in frame:
                if path in (None, ""):
                    raise ValueError(f"live scope {layer} list contains an empty image path")
                path = Path(path)
                row.append(str(path if path.is_absolute() else (root / path).resolve()))
            normalized.append(row)
        if normalized and any(len(row) != len(normalized[0]) for row in normalized):
            raise ValueError(f"live scope {layer} image list is not rectangular")
        return normalized

    @staticmethod
    def _open_image(path):
        from PIL import Image

        suffix = Path(path).suffix.lower()
        if suffix in (".npy", ".spy", ".npz", ".spz"):
            if suffix in (".npy", ".spy"):
                array = np.load(path, mmap_mode="r")
            else:
                with np.load(path) as archive:
                    key = "image" if "image" in archive else "arr_0"
                    if key not in archive:
                        raise ValueError(f"no image array in {path}")
                    array = archive[key].copy()
            if array.ndim != 3 or array.shape[2] not in (3, 4):
                raise ValueError(f"unsupported live scope image array: {path} "
                                 f"({array.shape})")
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            return Image.fromarray(np.asarray(array).copy())
        return Image.open(path)

    @classmethod
    def _canvas_height(cls, first_path, width):
        with cls._open_image(first_path) as image:
            source_width, source_height = image.size
            if image.mode not in ("RGBA", "LA") and "transparency" not in image.info:
                if Path(first_path).suffix.lower() in (".jpg", ".jpeg"):
                    source_width = max(1, source_width // 2)
        return max(1, int(round(source_height * width / max(source_width, 1))))

    def _path(self, layer, folder, index):
        paths = self.main_paths if layer == "main" else self.float_paths
        return paths[index % self.frames][folder]

    def _decode(self, path):
        from PIL import Image

        size = (self.width, self.height)
        with self._open_image(path) as image:
            has_alpha = image.mode in ("RGBA", "LA") or "transparency" in image.info
            if has_alpha:
                rgba = image.convert("RGBA")
                rgb = rgba.convert("RGB")
                alpha = rgba.getchannel("A")
            elif Path(path).suffix.lower() in (".jpg", ".jpeg"):
                rgb = image.convert("RGB")
                half = rgb.width // 2
                if half < 1:
                    raise ValueError(f"invalid side-by-side JPEG: {path}")
                alpha = rgb.crop((half, 0, half * 2, rgb.height)).convert("L")
                rgb = rgb.crop((0, 0, half, rgb.height))
            else:
                rgb = image.convert("RGB")
                alpha = Image.new("L", rgb.size, 255)

            source_width, source_height = rgb.size
            scale = min(size[0] / max(source_width, 1),
                        size[1] / max(source_height, 1))
            resized = (max(1, int(round(source_width * scale))),
                       max(1, int(round(source_height * scale))))
            resampling = Image.Resampling.LANCZOS
            rgb = rgb.resize(resized, resampling)
            alpha = alpha.resize(resized, resampling)

        x = (self.width - resized[0]) // 2
        y = (self.height - resized[1]) // 2
        luminance = np.zeros((self.height, self.width), dtype=np.uint8)
        coverage = np.zeros_like(luminance)
        lum_pixels = np.asarray(rgb.convert("L"), dtype=np.uint8)
        alpha_pixels = np.asarray(alpha, dtype=np.uint8)
        luminance[y:y + resized[1], x:x + resized[0]] = lum_pixels
        coverage[y:y + resized[1], x:x + resized[0]] = alpha_pixels
        thumb = np.stack((luminance, coverage), axis=-1)
        thumb.setflags(write=False)
        return thumb

    def _load(self, key):
        layer, folder, index = key
        return self._decode(self._path(layer, folder, index))

    def _request(self, key):
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return None
            future = self._pending.get(key)
            if future is not None:
                return future
            future = self._pool.submit(self._load, key)
            self._pending[key] = future

        def complete(done):
            try:
                image = done.result()
            except Exception:
                image = None
            with self._lock:
                if self._pending.get(key) is done:
                    self._pending.pop(key, None)
                if image is not None:
                    self._cache[key] = image
                    self._cache.move_to_end(key)
                    while len(self._cache) > self.cache_size:
                        self._cache.popitem(last=False)

        future.add_done_callback(complete)
        return future

    def thumb(self, layer, folder, index):
        key = (layer, int(folder), int(index) % self.frames)
        future = self._request(key)
        if future is not None:
            return future.result()
        with self._lock:
            image = self._cache.get(key)
            if image is not None:
                self._cache.move_to_end(key)
                return image
            # A completed failed prefetch was removed; retry synchronously so
            # the caller receives the actual decode exception.
            future = self._pool.submit(self._load, key)
            self._pending[key] = future
        return future.result()

    def prefetch(self, index, main_folder, float_folder):
        """Queue the current and next image pair without unbounded work."""
        index = int(index) % self.frames
        following = (index + 1) % self.frames
        for frame in (index, following):
            self._request(("main", int(main_folder), frame))
            self._request(("float", int(float_folder), frame))

    def close(self):
        self._pool.shutdown(wait=True, cancel_futures=True)

    def __del__(self):
        pool = getattr(self, "_pool", None)
        if pool is not None:
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
