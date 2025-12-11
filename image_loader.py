import threading
from collections import deque
import ctypes
import numpy as np
from settings import MAIN_FOLDER_PATH, FLOAT_FOLDER_PATH, TOLERANCE

from turbojpeg import TurboJPEG, TJPF_RGB
from ctypes.util import find_library

jpeg = TurboJPEG()

# ---------------------------------------------------------------------
# libwebp dynamic loader (fast path for .webp decoding)
# ---------------------------------------------------------------------

_libwebp = None

# 1) Try system/loader default using ctypes.util.find_library
libname = find_library("webp")
_candidate_libs = []

if libname:
    # On macOS this often returns something like "libwebp.dylib" or a full path.
    _candidate_libs.append(libname)

# 2) Fallback: common bare names
_candidate_libs.extend([
    "libwebp.so",
    "libwebp.so.7",
    "libwebp.so.6",
    "libwebp.dylib",
    "libwebp-7.dll",
])

# 3) Fallback: common Homebrew locations (Apple Silicon / Intel)
_candidate_libs.extend([
    "/opt/homebrew/opt/webp/lib/libwebp.dylib",   # Apple Silicon default
    "/usr/local/opt/webp/lib/libwebp.dylib",      # Intel default
])

for lib in _candidate_libs:
    try:
        _libwebp = ctypes.CDLL(lib)
        # Uncomment this if you want to see which one hits:
        # print(f"[IMAGE_LOADER] Using libwebp from: {lib}")
        break
    except OSError:
        continue

if _libwebp:
    _libwebp.WebPGetInfo.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    _libwebp.WebPGetInfo.restype = ctypes.c_int

    _libwebp.WebPDecodeRGBAInto.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    _libwebp.WebPDecodeRGBAInto.restype = ctypes.POINTER(ctypes.c_uint8)



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

        # --- OPTIMIZED NPY PATH (Uncompressed) ---
        if ext == "npy":
            # Load raw memory block (2, H, W)
            stack = np.load(image_path)

            # Layer 0 is Chars (uint8), View as S1 (Bytes/Char)
            # This is a metadata change only, zero CPU cost
            chars = stack[0].view('S1')
            colors = stack[1]

            return {'chars': chars, 'colors': colors}, False

        # --- LEGACY NPZ PATH (Compressed) ---
        if ext == "npz":
            data = np.load(image_path)
            return {'chars': data['chars'], 'colors': data['colors']}, False

        # --- STANDARD IMAGES ---
        if ext == "webp": return self._read_webp(image_path)
        if ext in ("jpg", "jpeg"):
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