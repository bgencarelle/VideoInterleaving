import cv2
import threading
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

class TripleImageBuffer:
    def __init__(self):
        # Initialize three slots as (index, image_pair) tuples.
        # Initially, all indices are set to None.
        self.buffers = [(None, None) for _ in range(3)]
        self.front = 0      # Index of the buffer currently being displayed.
        self.pending = None # Index of the newly loaded buffer.
        self.lock = threading.Lock()

    def update(self, index, images):
        """
        Called by the producer after loading a new pair of images.
        Writes (index, images) into an idle buffer slot and marks it as pending.
        """
        with self.lock:
            idle = None
            for i in range(3):
                if i != self.front and i != self.pending:
                    idle = i
                    break
            if idle is None:
                # Fallback: if no idle slot is found, override the front buffer.
                idle = self.front
            self.buffers[idle] = (index, images)
            self.pending = idle

    def get(self, current_index):
        """
        Called by the consumer (display loop) to retrieve the latest image pair.
        If a new pending buffer is available and its stored index is within tolerance
        of the current index, it swaps it in as the front buffer.
        Returns the image pair from the front buffer if its stored index is within tolerance;
        otherwise, returns None.
        """
        tolerance = 1  # Accept buffered images within ±1 of current_index.
        with self.lock:
            # Check if the pending buffer is available and valid.
            if self.pending is not None:
                stored_index, _ = self.buffers[self.pending]
                if stored_index is not None and abs(stored_index - current_index) <= tolerance:
                    self.front = self.pending
                    self.pending = None
            stored_index, images = self.buffers[self.front]
            if stored_index is None:
                return None
            if abs(stored_index - current_index) <= tolerance:
                return images
            else:
                return None
