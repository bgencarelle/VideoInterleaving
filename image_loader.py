import threading
from collections import deque
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

    def read_image(self, image_path):
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
            return jpeg.decode(data, pixel_format=TJPF_RGB), True

        raise ValueError(f"Unsupported: {image_path}")

    def load_images(self, index, main_folder, float_folder):
        mpath = self.main_folder_path[index][main_folder]
        fpath = self.float_folder_path[index][float_folder]
        main_img, main_sbs = self.read_image(mpath)
        float_img, float_sbs = self.read_image(fpath)
        return main_img, float_img, main_sbs, float_sbs


class FIFOImageBuffer:
    def __init__(self, max_size=5):
        self.queue = deque()
        self.max_size = max_size
        self.lock = threading.Lock()

    def update(self, index, data_tuple):
        with self.lock:
            if len(self.queue) >= self.max_size: self.queue.popleft()
            self.queue.append((index, data_tuple))

    def get(self, current_index):
        with self.lock:
            while self.queue and abs(self.queue[0][0] - current_index) > TOLERANCE:
                self.queue.popleft()
            if not self.queue: return None
            best_idx, best_data = min(self.queue, key=lambda item: abs(item[0] - current_index))
            if abs(best_idx - current_index) <= TOLERANCE:
                return best_idx, best_data[0], best_data[1], best_data[2], best_data[3]
            return None
