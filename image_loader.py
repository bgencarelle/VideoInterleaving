import cv2
import threading
from collections import deque
from settings import MAIN_FOLDER_PATH, FLOAT_FOLDER_PATH, TOLERANCE

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
        if image_path.lower().endswith(('.webp', '.png')):
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


class FIFOImageBuffer:
    """
    A simple FIFO buffer that stores (index, (main_image, float_image)) pairs.
    The 'index' acts like a simplified presentation timestamp (PTS).
    """

    def __init__(self, max_size=5):
        """
        :param max_size: Maximum number of frames to hold in the queue.
        """
        self.queue = deque()
        self.max_size = max_size
        self.lock = threading.Lock()

    def update(self, index, images):
        """
        Called by the producer after loading a new pair of images.
        Enqueues (index, images), discarding the oldest entry if full.
        """
        with self.lock:
            if len(self.queue) >= self.max_size:
                self.queue.popleft()  # Discard the oldest if full
            self.queue.append((index, images))

    def get(self, current_index):
        """
        Called by the consumer (display loop) to retrieve the best
        (index, images) pair for the current_index.

        We do the following:
          1) Discard frames whose index is outside the +/- TOLERANCE window
             relative to the current_index.
          2) Return the first valid frame if present.
          3) Otherwise return None.
        """
        with self.lock:
            # 1) Discard any frames that are too far from current_index (in either direction).
            while self.queue and abs(self.queue[0][0] - current_index) > TOLERANCE:
                self.queue.popleft()

            if not self.queue:
                return None

            # 2) Now check the first frame in the queue
            stored_index, images = self.queue[0]
            if abs(stored_index - current_index) <= TOLERANCE:
                # We do NOT pop it yet so that repeated calls for the same index can still retrieve it.
                return images

            return None
