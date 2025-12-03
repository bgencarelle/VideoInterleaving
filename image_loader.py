import threading
from collections import deque
import ctypes
import numpy as np
import cv2
import os
from settings import MAIN_FOLDER_PATH, FLOAT_FOLDER_PATH, TOLERANCE

# --- TurboJPEG Init ---
# You already have this installed for the display loop.
# We use it here to accelerate JPG loading.
from turbojpeg import TurboJPEG, TJPF_RGBA

jpeg = TurboJPEG()

# --- Load libwebp ---
_libwebp = None
# Added common Linux paths for VPS (libwebp.so.7, libwebp.so.6)
for lib in ("libwebp.so", "libwebp.so.7", "libwebp.so.6", "libwebp.dylib", "libwebp-7.dll"):
    try:
        _libwebp = ctypes.CDLL(lib)
        break
    except OSError:
        continue
if _libwebp is None:
    raise RuntimeError("Could not load libwebp. Ensure libwebp is installed (apt-get install libwebp-dev or similar).")

# --- Define ctypes signatures ---

# WebPGetInfo: Reads width/height without decoding
_libwebp.WebPGetInfo.argtypes = [
    ctypes.c_char_p,  # data
    ctypes.c_size_t,  # data_size
    ctypes.POINTER(ctypes.c_int),  # width ptr
    ctypes.POINTER(ctypes.c_int)  # height ptr
]
_libwebp.WebPGetInfo.restype = ctypes.c_int

# WebPDecodeRGBAInto: Decodes DIRECTLY into an existing buffer (Zero-Copy)
# Signature: uint8_t* WebPDecodeRGBAInto(const uint8_t* data, size_t data_size,
#                                        uint8_t* output_buffer, size_t output_buffer_size, int output_stride);
_libwebp.WebPDecodeRGBAInto.argtypes = [
    ctypes.c_char_p,  # data
    ctypes.c_size_t,  # data_size
    ctypes.POINTER(ctypes.c_uint8),  # output_buffer (pointer to numpy array data)
    ctypes.c_size_t,  # output_buffer_size
    ctypes.c_int  # output_stride (width * 4)
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
        Allocates a NumPy array and tells libwebp to write directly into it.
        """
        # 1. Read raw bytes (Python manages this memory)
        with open(image_path, "rb") as f:
            data = f.read()

        # 2. Get dimensions
        w = ctypes.c_int()
        h = ctypes.c_int()
        # Note: ctypes handles 'data' (bytes) as c_char_p directly. No create_string_buffer needed.
        if not _libwebp.WebPGetInfo(data, len(data), ctypes.byref(w), ctypes.byref(h)):
            raise ValueError(f"Invalid WebP header: {image_path}")

        width, height = w.value, h.value

        # 3. Allocate NumPy array (H, W, 4) for RGBA
        # This is fast and managed by Python GC.
        img = np.empty((height, width, 4), dtype=np.uint8)

        # 4. Get a C-pointer to the NumPy array's data
        # .ctypes.data_as(...) returns a pointer to the memory block
        out_ptr = img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        # Calculate size and stride (4 bytes per pixel for RGBA)
        stride = width * 4
        buf_size = height * stride

        # 5. Decode directly into the array
        res = _libwebp.WebPDecodeRGBAInto(
            data, len(data),
            out_ptr, buf_size, stride
        )

        if not res:
            raise RuntimeError(f"Failed to decode WebP data: {image_path}")

        # 'img' is now populated with pixel data.
        return img

    def read_image(self, image_path):
        """
        Loads an image.
        - .webp: Uses Zero-Copy libwebp (Fastest)
        - .jpg/.jpeg: Uses TurboJPEG (Fast)
        - others: Uses OpenCV (Fallback)
        """
        ext = image_path.split('.')[-1].lower()

        if ext == "webp":
            return self._read_webp(image_path)

        if ext in ("jpg", "jpeg"):
            try:
                with open(image_path, "rb") as f:
                    data = f.read()
                # Decode directly to RGBA to match WebP output
                return jpeg.decode(data, pixel_format=TJPF_RGBA)
            except Exception:
                # If TurboJPEG fails (rare), fall through to OpenCV
                pass

        # Fallback: OpenCV
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Ensure RGBA consistency
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        elif img.shape[2] == 4:
            # OpenCV loads BGRA, we need RGBA
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

        return img

    def load_images(self, index, main_folder, float_folder):
        mpath = self.main_folder_path[index][main_folder]
        fpath = self.float_folder_path[index][float_folder]

        # These run in the ThreadPoolExecutor from image_display.py
        # ctypes releases the GIL, so this is truly parallel on Xeons.
        main_img = self.read_image(mpath)
        float_img = self.read_image(fpath)
        return main_img, float_img


class FIFOImageBuffer:
    def __init__(self, max_size=5):
        self.queue = deque()
        self.max_size = max_size
        self.lock = threading.Lock()

    def update(self, index, images):
        with self.lock:
            if len(self.queue) >= self.max_size:
                self.queue.popleft()
            self.queue.append((index, images))

    def get(self, current_index):
        with self.lock:
            while self.queue and abs(self.queue[0][0] - current_index) > TOLERANCE:
                self.queue.popleft()
            if not self.queue:
                return None
            best_idx, best_imgs = min(
                self.queue,
                key=lambda item: abs(item[0] - current_index)
            )
            if abs(best_idx - current_index) <= TOLERANCE:
                return best_idx, best_imgs[0], best_imgs[1]
            return None