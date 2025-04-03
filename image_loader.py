import cv2
from collections import deque
from settings import MAIN_FOLDER_PATH, FLOAT_FOLDER_PATH
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

    def read_image(self, image_path):
        """
        Loads an image from the given path.
        Supported formats: .webp and .png.
        """
        if image_path.lower().endswith(('.webp', '.png', '.jpg', '.jxl')):
            image_np = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image_np is None:
                raise ValueError(f"Failed to load image: {image_path}")
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGBA)
            return image_np
        else:
            raise ValueError("Unsupported image format.")

    def load_images(self, index, main_folder, float_folder):
        """
        Loads a pair of images from the stored folder paths.
        """
        main_image = self.read_image(self.main_folder_path[index][main_folder])
        float_image = self.read_image(self.float_folder_path[index][float_folder])
        return main_image, float_image

class ImageLoaderBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def add_image_future(self, index, future, png_paths_len):
        # Clamp index based on png_paths_len.
        clamped_index = max(0, min(index, png_paths_len - 1))
        self.buffer.append((clamped_index, future))

    def get_future_for_index(self, index):
        for item in list(self.buffer):
            buf_index, future = item
            if buf_index == index:
                self.buffer.remove(item)
                return future
        return None
