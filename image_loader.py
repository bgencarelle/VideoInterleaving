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
    A FIFO buffer that stores (index, (main_image, float_image)) pairs.
    'index' acts like a simplified presentation timestamp (PTS).
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
                self.queue.popleft()  # Discard the oldest if at capacity
            self.queue.append((index, images))

    def get(self, current_index):
        """
        Called by the consumer (display loop) to retrieve the frame whose index
        is closest to current_index (while still within +/- TOLERANCE).

        Returns:
           (stored_index, main_image, float_image) if a suitable frame is found,
           else None.

        Steps:
          1) Discard frames whose index is outside the +/- TOLERANCE window.
          2) Among remaining frames, find the one with the smallest abs difference.
          3) Return that frame if within TOLERANCE, otherwise None.
        """
        with self.lock:
            # 1) Discard any frames that are too far from current_index (in either direction).
            while self.queue and abs(self.queue[0][0] - current_index) > TOLERANCE:
                self.queue.popleft()

            # If the queue is empty after discarding, nothing to display.
            if not self.queue:
                return None

            # 2) Find the closest frame in the queue.
            best_diff = float('inf')
            best_i = -1
            best_stored_index = None
            best_images = None

            # We won't pop frames here; we only pick the best to display.
            for i, (stored_index, images) in enumerate(self.queue):
                diff = abs(stored_index - current_index)
                if diff < best_diff:
                    best_diff = diff
                    best_i = i
                    best_stored_index = stored_index
                    best_images = images

            # 3) Check if the best match is within TOLERANCE.
            if best_diff <= TOLERANCE and best_images is not None:
                main_image, float_image = best_images
                return best_stored_index, main_image, float_image
            else:
                return None
