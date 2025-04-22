import threading
from collections import deque
import ctypes
import numpy as np
import cv2
from settings import MAIN_FOLDER_PATH, FLOAT_FOLDER_PATH, TOLERANCE

# --- Load libwebp ---
_libwebp = None
for lib in ("libweb", "webp", "libwebp.so", "libwebp.dylib", "libwebp-7.dll", "libwebp.dll", "libwebp-8.dll"):
    try:
        _libwebp = ctypes.CDLL(lib)
        break
    except OSError:
        continue
if _libwebp is None:
    raise RuntimeError("Could not load libwebp (libwebp.so / .dylib / .dll)")

# Prototype the functions we need
_libwebp.WebPGetInfo.argtypes = [ctypes.c_char_p, ctypes.c_size_t,
                                 ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
_libwebp.WebPGetInfo.restype  = ctypes.c_int

_libwebp.WebPDecodeRGBA.argtypes = [ctypes.c_char_p, ctypes.c_size_t,
                                    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
_libwebp.WebPDecodeRGBA.restype  = ctypes.POINTER(ctypes.c_uint8)

_libwebp.WebPFree.argtypes = [ctypes.c_void_p]

class ImageLoader:
    def __init__(self,
                 main_folder_path=MAIN_FOLDER_PATH,
                 float_folder_path=FLOAT_FOLDER_PATH,
                 png_paths_len=0):
        self.main_folder_path  = main_folder_path
        self.float_folder_path = float_folder_path
        self.png_paths_len     = png_paths_len

    def set_paths(self, main_folder_path, float_folder_path):
        self.main_folder_path  = main_folder_path
        self.float_folder_path = float_folder_path

    def set_png_paths_len(self, value):
        self.png_paths_len = value

    def _read_webp(self, image_path):
        # Read raw bytes
        with open(image_path, "rb") as f:
            data = f.read()
        size = len(data)
        buf  = ctypes.create_string_buffer(data)

        # Get width & height
        w = ctypes.c_int()
        h = ctypes.c_int()
        ok = _libwebp.WebPGetInfo(buf, size, ctypes.byref(w), ctypes.byref(h))
        if not ok:
            raise ValueError(f"Invalid or corrupted WebP file: {image_path}")

        # Decode to RGBA (returns a pointer we must free)
        ptr = _libwebp.WebPDecodeRGBA(
            buf, size,
            ctypes.byref(w), ctypes.byref(h)
        )
        if not ptr:
            raise MemoryError("WebPDecodeRGBA returned NULL")

        # Wrap it in a numpy array without copy, then copy into Python memory
        array = np.ctypeslib.as_array(
            ptr, shape=(h.value, w.value, 4)
        )
        img = array.copy()

        # Free the C buffer
        _libwebp.WebPFree(ptr)
        return img

    def read_image(self, image_path):
        """
        Loads an image from the given path.
        - .webp is decoded via libwebp (fast, BGRA→RGBA swizzle on GPU).
        - other formats use OpenCV + minimal swizzle.
        """
        ext = image_path.lower().rsplit('.', 1)[-1]
        if ext == "webp":
            return self._read_webp(image_path)

        # Fallback: let OpenCV handle .png/.jpg/.jpeg
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Convert BGR→RGBA if needed
        c = img.shape[2] if img.ndim == 3 else 1
        if c == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        elif c == 4:
            # BGRA→RGBA channel swap
            img = img[:, :, [2, 1, 0, 3]]
        return img

    def load_images(self, index, main_folder, float_folder):
        """
        Loads a pair of images for display:
           - main_image  = (index, main_folder)
           - float_image = (index, float_folder)
        """
        mpath = self.main_folder_path[index][main_folder]
        fpath = self.float_folder_path[index][float_folder]
        main_img  = self.read_image(mpath)
        float_img = self.read_image(fpath)
        return main_img, float_img

class FIFOImageBuffer:
    """
    Unchanged from your baseline:
    A thread‑safe FIFO that prunes out‑of‑tolerance frames
    and picks the closest match without removing it.
    """
    def __init__(self, max_size=5):
        self.queue    = deque()
        self.max_size = max_size
        self.lock     = threading.Lock()

    def update(self, index, images):
        with self.lock:
            if len(self.queue) >= self.max_size:
                self.queue.popleft()
            self.queue.append((index, images))

    def get(self, current_index):
        with self.lock:
            # prune out-of-window
            while self.queue and abs(self.queue[0][0] - current_index) > TOLERANCE:
                self.queue.popleft()
            if not self.queue:
                return None
            # pick closest
            best_idx, best_imgs = min(
                self.queue,
                key=lambda item: abs(item[0] - current_index)
            )
            if abs(best_idx - current_index) <= TOLERANCE:
                main_img, float_img = best_imgs
                return best_idx, main_img, float_img
            return None
