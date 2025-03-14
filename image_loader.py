import cv2
from collections import deque

# Module-level variables for folder paths and maximum index.
_main_folder_path = "images/foreground"
_float_folder_path = "images/floatground"
_png_paths_len = 0


def set_folder_paths(main_folder_path, float_folder_path):
    """
    Sets the folder paths to be used for image loading.
    """
    global _main_folder_path, _float_folder_path
    _main_folder_path = main_folder_path
    _float_folder_path = float_folder_path


def set_png_paths_len(value):
    """
    Sets the maximum index (png_paths_len) for clamping.
    """
    global _png_paths_len
    _png_paths_len = value


def read_image(image_path):
    """
    Loads an image from the given path.
    Supported formats: .webp and .png.
    """
    if image_path.lower().endswith(('.webp', '.png')):
        image_np = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image_np is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGBA)
        return image_np
    else:
        raise ValueError("Unsupported image format.")


def load_images(index, main_folder, float_folder):
    """
    Loads a pair of images from the stored folder paths based on the provided indices.
    Expects that set_folder_paths has been called.
    """
    main_image = read_image(_main_folder_path[index][main_folder])
    float_image = read_image(_float_folder_path[index][float_folder])
    return main_image, float_image


class ImageLoaderBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def add_image_future(self, index, future):
        # Clamp index based on _png_paths_len.
        clamped_index = max(0, min(index, _png_paths_len - 1))
        self.buffer.append((clamped_index, future))

    def get_future_for_index(self, index):
        for item in list(self.buffer):
            buf_index, future = item
            if buf_index == index:
                self.buffer.remove(item)
                return future
        return None
