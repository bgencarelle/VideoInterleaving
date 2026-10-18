import threading
from collections import deque, OrderedDict
import ctypes
import numpy as np
import os
from settings import MAIN_FOLDER_PATH, FLOAT_FOLDER_PATH, TOLERANCE

from turbojpeg import TJPF_RGB
from turbojpeg_loader import get_turbojpeg

jpeg = get_turbojpeg()

from libwebp_loader import init_libwebp

_libwebp = init_libwebp(verbose=False)

class ImageLoader:
    def __init__(self, main_folder_path=MAIN_FOLDER_PATH, float_folder_path=FLOAT_FOLDER_PATH, png_paths_len=0):
        self.main_folder_path = main_folder_path
        self.float_folder_path = float_folder_path
        self.png_paths_len = png_paths_len
        # Repeated assets (e.g. transparent layers) often have different file
        # names at every source index. Two content entries avoid decoding the
        # same bytes again without caching an image sequence.
        self._jpeg_cache = OrderedDict()
        self._jpeg_lock = threading.Lock()

    def set_paths(self, main_folder_path, float_folder_path):
        self.main_folder_path = main_folder_path
        self.float_folder_path = float_folder_path

    def set_png_paths_len(self, value):
        self.png_paths_len = value

    def _read_webp(self, image_path):
        if _libwebp is None: raise RuntimeError("libwebp not loaded.")
        with open(image_path, "rb") as f:
            data = f.read()
        w, h = ctypes.c_int(), ctypes.c_int()
        if not _libwebp.WebPGetInfo(data, len(data), ctypes.byref(w), ctypes.byref(h)):
            raise ValueError(f"Invalid WebP: {image_path}")
        img = np.empty((h.value, w.value, 4), dtype=np.uint8)
        if not _libwebp.WebPDecodeRGBAInto(data, len(data), img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                                           h.value * w.value * 4, w.value * 4):
            raise RuntimeError(f"Decode failed: {image_path}")
        return img, False

    def read_image(self, image_path, jpeg_scaling_factor=None,
                   jpeg_min_size=None):
        ext = image_path.split('.')[-1].lower()

        # --- [INSERTED] Hybrid Asset Support ---
        if ext == "spy":
            # Raw Memory Map (Fastest)
            return np.load(image_path, mmap_mode='r'), False

        if ext == "spz":
            # Compressed Archive
            with np.load(image_path) as data:
                # Key is 'image' from our baker, or fallback 'arr_0'
                key = 'image' if 'image' in data else 'arr_0'
                return data[key], False

        if ext == "npy":
            # ASCII Stack
            stack = np.load(image_path, mmap_mode='r')
            chars = stack[0].view('S1')
            colors = stack[1]
            return {'chars': chars, 'colors': colors}, False
        # ---------------------------------------

        # --- NPZ PATH (Smart Handling: Headless vs Legacy) ---
        if ext == "npz":
            # We must load the archive to check keys
            with np.load(image_path) as data:
                # 1. HEADLESS IMAGE? (Look for 'image' key)
                if 'image' in data:
                    return data['image'], False

                # 2. LEGACY ASCII? (Look for 'chars' and 'colors')
                elif 'chars' in data and 'colors' in data:
                    return {'chars': data['chars'], 'colors': data['colors']}, False

                # 3. GENERIC FALLBACK
                else:
                    # Just grab the first array found (usually 'arr_0')
                    key = data.files[0]
                    return data[key], False

        # --- STANDARD IMAGES ---
        if ext == "webp": return self._read_webp(image_path)
        if ext in ("jpg", "jpeg"):
            # TurboJPEG decode: TJPF_RGB is fastest format, decode happens in worker thread
            # This is optimal - no unnecessary copies, uses native library
            with open(image_path, "rb") as f: data = f.read()
            try:
                decode_scale = jpeg_scaling_factor
                if decode_scale is not None and jpeg_min_size is not None:
                    width, height, _, _ = jpeg.decode_header(data)
                    if width < jpeg_min_size[0] or height < jpeg_min_size[1]:
                        decode_scale = None
                key = (data, decode_scale)
                with self._jpeg_lock:
                    cached = self._jpeg_cache.get(key)
                    if cached is not None:
                        self._jpeg_cache.move_to_end(key)
                        return cached, True
                image = jpeg.decode(
                    data, pixel_format=TJPF_RGB,
                    scaling_factor=decode_scale)
                image.setflags(write=False)
                with self._jpeg_lock:
                    self._jpeg_cache[key] = image
                    while len(self._jpeg_cache) > 2:
                        self._jpeg_cache.popitem(last=False)
                return image, True
            except Exception as e:
                print(f"[JPEG ERROR] Failed to decode: {image_path}")
                print(f"  Error: {e}")
                raise  # Re-raise to maintain existing error handling

        raise ValueError(f"Unsupported: {image_path}")

    def load_images(self, index, main_folder, float_folder,
                    jpeg_scaling_factor=None, jpeg_min_size=None):
        mpath = self.main_folder_path[index][main_folder]
        fpath = self.float_folder_path[index][float_folder]
        main_img, main_sbs = self.read_image(
            mpath, jpeg_scaling_factor=jpeg_scaling_factor,
            jpeg_min_size=jpeg_min_size)
        float_img, float_sbs = self.read_image(
            fpath, jpeg_scaling_factor=jpeg_scaling_factor,
            jpeg_min_size=jpeg_min_size)
        return main_img, float_img, main_sbs, float_sbs


class FIFOImageBuffer:
    def __init__(self, max_size=5):
        self.queue = deque()
        self.max_size = max_size
        self.lock = threading.Lock()
        self.dropped_count = 0  # Track dropped frames due to backpressure
        self.total_updates = 0  # Track total update attempts

    def is_full(self):
        """Check if buffer is at capacity (thread-safe)."""
        with self.lock:
            return len(self.queue) >= self.max_size

    def current_depth(self):
        """Get current buffer depth (thread-safe)."""
        with self.lock:
            return len(self.queue)

    def update(self, index, data_tuple, folders=None):
        """
        Update buffer with new frame data.
        Implements backpressure: drops oldest frame if buffer is full.
        Returns True if frame was added, False if dropped due to backpressure.
        """
        with self.lock:
            self.total_updates += 1
            dropped = False
            if len(self.queue) >= self.max_size:
                self.queue.popleft()
                self.dropped_count += 1
                dropped = True
            self.queue.append((index, data_tuple, folders))
            return not dropped

    def get(self, current_index, folders=None, total_images=None, *, exact=False):
        with self.lock:
            def distance(index):
                delta = abs(index - current_index)
                if total_images:
                    delta %= total_images
                    delta = min(delta, total_images - delta)
                return delta
            # Completion order is not source order. An old first entry must
            # never discard an exact entry that finished behind it.
            candidates = [item for item in self.queue
                          if item[2] == folders and
                          distance(item[0]) <= (0 if exact else TOLERANCE)]
            if not candidates: return None
            best_idx, best_data, _ = min(candidates, key=lambda item: distance(item[0]))
            if distance(best_idx) <= TOLERANCE:
                return best_idx, best_data[0], best_data[1], best_data[2], best_data[3]
            return None

    def retain(self, keys):
        """Release obsolete decoded layers; keys are (index, folder-pair)."""
        with self.lock:
            self.queue = deque(item for item in self.queue
                               if (item[0], item[2]) in keys)

    def get_stats(self):
        """Get buffer statistics (thread-safe)."""
        with self.lock:
            return {
                'depth': len(self.queue),
                'max_size': self.max_size,
                'dropped_count': self.dropped_count,
                'total_updates': self.total_updates,
                'drop_rate': self.dropped_count / max(1, self.total_updates)
            }
