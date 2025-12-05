import threading
from collections import deque
import ctypes
import numpy as np
import cv2
import os
from settings import MAIN_FOLDER_PATH, FLOAT_FOLDER_PATH, TOLERANCE

# --- TurboJPEG Init ---
from turbojpeg import TurboJPEG, TJPF_RGBA, TJPF_RGB

jpeg = TurboJPEG()

# --- Load libwebp ---
_libwebp = None
# Common Linux/Mac/Windows paths
for lib in ("libwebp.so", "libwebp.so.7", "libwebp.so.6", "libwebp.dylib", "libwebp-7.dll"):
    try:
        _libwebp = ctypes.CDLL(lib)
        break
    except OSError:
        continue

if _libwebp is None:
    raise RuntimeError("Could not load libwebp. Ensure libwebp is installed.")

# --- Define ctypes signatures ---
_libwebp.WebPGetInfo.argtypes = [
    ctypes.c_char_p, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
]
_libwebp.WebPGetInfo.restype = ctypes.c_int

_libwebp.WebPDecodeRGBAInto.argtypes = [
    ctypes.c_char_p, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.c_int
]
_libwebp.WebPDecodeRGBAInto.restype = ctypes.POINTER(ctypes.c_uint8)


class ImageLoader:
    def __init__(self,
                 main_folder_path=MAIN_FOLDER_PATH,
                 float_folder_path=FLOAT_FOLDER_PATH,
                 png_paths_len=0):
        self.main_folder_path = main_folder_path
        self.float_folder_path = float_folder_path
        self.png_paths_len = png_paths_len

    def set_paths(self, main_folder_path, float_folder_path):
        self.main_folder_path = main_folder_path
        self.float_folder_path = float_folder_path

    def set_png_paths_len(self, value):
        self.png_paths_len = value

    def _read_webp(self, image_path):
        """
        Zero-Copy WebP Decoder.
        Returns: (numpy_array, is_sbs=False)
        """
        with open(image_path, "rb") as f:
            data = f.read()

        w = ctypes.c_int()
        h = ctypes.c_int()
        if not _libwebp.WebPGetInfo(data, len(data), ctypes.byref(w), ctypes.byref(h)):
            raise ValueError(f"Invalid WebP header: {image_path}")

        width, height = w.value, h.value
        # WebP is always Standard RGBA in this pipeline
        img = np.empty((height, width, 4), dtype=np.uint8)

        out_ptr = img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        stride = width * 4
        buf_size = height * stride

        res = _libwebp.WebPDecodeRGBAInto(data, len(data), out_ptr, buf_size, stride)
        if not res:
            raise RuntimeError(f"Failed to decode WebP data: {image_path}")

        return img, False

    def read_image(self, image_path):
        """
        Loads an image and detects format strategy.
        Returns: (image_data, is_sbs_bool)
        """
        ext = image_path.split('.')[-1].lower()

        # --- WEBP STRATEGY (Standard RGBA) ---
        if ext == "webp":
            return self._read_webp(image_path)

        # --- JPEG STRATEGY (Side-by-Side) ---
        if ext in ("jpg", "jpeg"):
            try:
                with open(image_path, "rb") as f:
                    data = f.read()
                # Decode to RGB.
                # For SBS, visual data is RGB, Alpha mask is encoded in the image structure.
                # We do NOT want TJPF_RGBA here.
                img = jpeg.decode(data, pixel_format=TJPF_RGB)
                return img, True  # is_sbs = True
            except Exception:
                pass

        # --- FALLBACK (OpenCV) ---
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

        # IMPORTANT: Fallback must also return the boolean flag
        return img, False

    def load_images(self, index, main_folder, float_folder):
        """
        Returns 4 items per frame:
        (Main_Img, Float_Img, Main_Is_SBS, Float_Is_SBS)
        """
        mpath = self.main_folder_path[index][main_folder]
        fpath = self.float_folder_path[index][float_folder]

        # Use read_image to get tuples (img, bool)
        main_img, main_sbs = self.read_image(mpath)
        float_img, float_sbs = self.read_image(fpath)

        return main_img, float_img, main_sbs, float_sbs


class FIFOImageBuffer:
    def __init__(self, max_size=5):
        self.queue = deque()
        self.max_size = max_size
        self.lock = threading.Lock()

    def update(self, index, data_tuple):
        """
        data_tuple is now (main_img, float_img, m_sbs, f_sbs)
        """
        with self.lock:
            if len(self.queue) >= self.max_size:
                self.queue.popleft()
            self.queue.append((index, data_tuple))

    def get(self, current_index):
        with self.lock:
            while self.queue and abs(self.queue[0][0] - current_index) > TOLERANCE:
                self.queue.popleft()
            if not self.queue:
                return None
            best_idx, best_data = min(
                self.queue,
                key=lambda item: abs(item[0] - current_index)
            )
            if abs(best_idx - current_index) <= TOLERANCE:
                # Unpack the tuple stored in update
                # Returns: index, m_img, f_img, m_sbs, f_sbs
                return best_idx, best_data[0], best_data[1], best_data[2], best_data[3]
            return None