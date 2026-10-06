"""Lazy runtime thumbnails for the unbaked scope image source.

The baked scope path stays entirely image-loader-free. This adapter is only
constructed for ``--scope-source images``; it decodes the current image pair
on a small worker pool and exposes the same ``thumb()`` interface as an
``XYLibrary``. Geometry remains an offline-bake feature.
"""
from collections import OrderedDict
from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import threading
import time

import numpy as np
from scope_numeric import image_bytes


@dataclass(frozen=True)
class DecodedImagePair:
    """An immutable, atomically decoded main/float image pair."""

    main: np.ndarray
    floating: np.ndarray
    requested_at_ns: int
    decode_started_at_ns: int
    ready_at_ns: int


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
    """Decode complete source pairs off-thread with bounded prefetch.

    ``main_paths`` and ``float_paths`` use make_file_lists' existing ordering,
    so the image-clock and folder-selector choices agree with the other modes.
    The fixed thumbnail canvas prevents a mixed source aspect ratio from
    changing the renderer's sample grid; each image is contained without
    stretching, with transparent padding.
    """

    def __init__(self, main_paths, float_paths, *, width=128, workers=2,
                 cache_size=16, max_pending_pairs=None):
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
        self.cache_pair_limit = max(1, self.cache_size // 2)
        self.max_pending_pairs = max(
            max(1, int(workers)),
            int(max_pending_pairs if max_pending_pairs is not None
                else max(1, int(workers)) * 2))
        self._cache = OrderedDict()
        self._pending_pairs = OrderedDict()
        self._failed_pairs = OrderedDict()
        self._lock = threading.RLock()
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, int(workers)), thread_name_prefix="scope-images")
        self._stats = {
            "pair_requests": 0,
            "pair_ready": 0,
            "pair_misses": 0,
            "decode_failures": 0,
            "prefetch_evictions": 0,
            "queue_full": 0,
        }
        self._retry_failed_after_ns = 1_000_000_000

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
                array = image_bytes(np.ascontiguousarray(array))
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

    def _pair_key(self, index, main_folder, float_folder):
        return (int(index) % self.frames,
                int(main_folder) % len(self.main_paths[0]),
                int(float_folder) % len(self.float_paths[0]))

    def _load_pair(self, key, requested_at_ns):
        index, main_folder, float_folder = key
        started_ns = time.monotonic_ns()
        main = self._decode(self._path("main", main_folder, index))
        floating = self._decode(self._path("float", float_folder, index))
        return DecodedImagePair(
            main, floating, int(requested_at_ns), started_ns,
            time.monotonic_ns())

    def _request_pair(self, key, *, priority):
        """Submit one pair while keeping running plus queued work bounded."""
        with self._lock:
            if key in self._cache or key in self._pending_pairs:
                return True
            failed = self._failed_pairs.get(key)
            if failed is not None:
                if time.monotonic_ns() - failed[1] < self._retry_failed_after_ns:
                    return False
                self._failed_pairs.pop(key, None)

            if len(self._pending_pairs) >= self.max_pending_pairs:
                evicted = False
                if priority:
                    # Superseded queued lookahead is disposable; keep already
                    # running decodes bounded and let them finish harmlessly.
                    for old_key, old_future in tuple(self._pending_pairs.items()):
                        if old_future.running():
                            continue
                        if old_future.cancel():
                            self._pending_pairs.pop(old_key, None)
                            self._stats["prefetch_evictions"] += 1
                            evicted = True
                            break
                if not evicted:
                    self._stats["queue_full"] += 1
                    return False

            self._stats["pair_requests"] += 1
            requested_at_ns = time.monotonic_ns()
            future = self._pool.submit(self._load_pair, key, requested_at_ns)
            self._pending_pairs[key] = future

        def complete(done):
            try:
                pair = done.result()
            except CancelledError:
                return
            except Exception as exc:
                with self._lock:
                    if self._pending_pairs.get(key) is done:
                        self._pending_pairs.pop(key, None)
                    self._failed_pairs[key] = (
                        f"{type(exc).__name__}: {exc}", time.monotonic_ns())
                    self._failed_pairs.move_to_end(key)
                    while len(self._failed_pairs) > self.cache_pair_limit:
                        self._failed_pairs.popitem(last=False)
                    self._stats["decode_failures"] += 1
                return

            with self._lock:
                if self._pending_pairs.get(key) is done:
                    self._pending_pairs.pop(key, None)
                self._failed_pairs.pop(key, None)
                self._cache[key] = pair
                self._cache.move_to_end(key)
                while len(self._cache) > self.cache_pair_limit:
                    self._cache.popitem(last=False)

        future.add_done_callback(complete)
        return True

    def thumb(self, layer, folder, index):
        """Synchronous compatibility access for startup/calibration callers.

        Playback uses :meth:`ready_pair` and never calls this method on the
        render worker. A synchronous decode here preserves the array-like
        ``XYLibrary.thumb`` contract for calibration and external callers.
        """
        folder = int(folder) % (len(self.main_paths[0]) if layer == "main"
                                else len(self.float_paths[0]))
        index = int(index) % self.frames
        with self._lock:
            for key in reversed(self._cache):
                pair_index, main_folder, float_folder = key
                if pair_index != index:
                    continue
                if layer == "main" and main_folder == folder:
                    self._cache.move_to_end(key)
                    return self._cache[key].main
                if layer == "float" and float_folder == folder:
                    self._cache.move_to_end(key)
                    return self._cache[key].floating
        return self._decode(self._path(layer, folder, index))

    def ready_pair(self, index, main_folder, float_folder):
        """Return both cached thumbnails, or request them and return ``None``.

        A partially decoded pair is never exposed. The returned record also
        carries monotonic decode timestamps for presentation-age accounting.
        """
        key = self._pair_key(index, main_folder, float_folder)
        with self._lock:
            pair = self._cache.get(key)
            if pair is not None:
                self._cache.move_to_end(key)
                self._stats["pair_ready"] += 1
                return pair
            self._stats["pair_misses"] += 1
        self._request_pair(key, priority=True)
        return None

    @staticmethod
    def _lookahead(index, direction, frames, count, *, pingpong=True):
        """Return distinct upcoming indices in the requested playback direction."""
        if frames <= 1 or count <= 0:
            return ()
        current = int(index) % frames
        step = -1 if int(direction) < 0 else 1
        result = []
        for _ in range(int(count)):
            following = current + step
            if pingpong:
                if following < 0:
                    step = 1
                    following = 1
                elif following >= frames:
                    step = -1
                    following = frames - 2
            else:
                following %= frames
            if following not in result and following != index:
                result.append(following)
            current = following
        return tuple(result)

    def prefetch(self, index, main_folder, float_folder, *, direction=1,
                 lookahead=2, pingpong=True):
        """Prioritize the current pair, then bounded directional lookahead."""
        index = int(index) % self.frames
        main_folder = int(main_folder) % len(self.main_paths[0])
        float_folder = int(float_folder) % len(self.float_paths[0])
        self._request_pair((index, main_folder, float_folder), priority=True)
        for frame in self._lookahead(index, direction, self.frames, lookahead,
                                     pingpong=pingpong):
            self._request_pair((frame, main_folder, float_folder), priority=False)

    def snapshot(self):
        """Return bounded decode/cache activity and queue state."""
        with self._lock:
            return {
                **self._stats,
                "cache_pairs": len(self._cache),
                "cache_pair_limit": self.cache_pair_limit,
                "pending_pairs": len(self._pending_pairs),
                "pending_pair_limit": self.max_pending_pairs,
                "failed_pairs": len(self._failed_pairs),
            }

    def close(self):
        self._pool.shutdown(wait=True, cancel_futures=True)

    def __del__(self):
        pool = getattr(self, "_pool", None)
        if pool is not None:
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
