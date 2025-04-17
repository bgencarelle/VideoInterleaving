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
        Supported formats: .webp, .png, .jpg, .jpeg
        """
        if image_path.lower().endswith(('.webp', '.png', '.jpg', '.jpeg')):
            image_np = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image_np is None:
                raise ValueError(f"Failed to load image: {image_path}")
            # Ensure RGBA ordering
            if image_np.shape[2] == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGBA)
            elif image_np.shape[2] == 4:
                # BGRA -> RGBA by channel swap
                image_np = image_np[:, :, [2, 1, 0, 3]]
            return image_np
        else:
            raise ValueError("Unsupported image format.")

    def load_images(self, index, main_folder, float_folder):
        """
        Loads a pair of images from the stored folder paths.
        Returns a (main_image, float_image) tuple.
        """
        main_image = self.read_image(self.main_folder_path[index][main_folder])
        float_image = self.read_image(self.float_folder_path[index][float_folder])
        return main_image, float_image

class FIFOImageBuffer:
    """
    A FIFO buffer that stores (index, (main_image, float_image)) pairs.
    'index' acts like a simplified presentation timestamp (PTS).

    This implementation batches pruning and lookup to minimize lock contention.
    """
    def __init__(self, max_size=5):
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
                self.queue.popleft()
            self.queue.append((index, images))

    def get(self, target_index):
        """
        Retrieve the frame whose stored index is closest to target_index,
        while still within +/- TOLERANCE.

        Phases:
          1) Under lock: prune any frames outside the TOLERANCE window.
          2) Snapshot the remaining queue and find the best match outside the lock.
          3) Under lock: remove all frames up to and including the matched one.

        Returns:
            (stored_index, main_image, float_image) if a match is found,
            else None.
        """
        # Phase 1: prune old frames under lock and snapshot
        with self.lock:
            while self.queue and abs(self.queue[0][0] - target_index) > TOLERANCE:
                self.queue.popleft()
            snapshot = list(self.queue)

        # Phase 2: find the closest match without holding the lock
        best_diff = float('inf')
        best_item = None
        for stored_index, images in snapshot:
            diff = abs(stored_index - target_index)
            if diff < best_diff:
                best_diff = diff
                best_item = (stored_index, images)

        if best_item is None or best_diff > TOLERANCE:
            return None

        matched_index, (main_image, float_image) = best_item

        # Phase 3: remove matched frame and any older ones under lock
        with self.lock:
            while self.queue and self.queue[0][0] != matched_index:
                self.queue.popleft()
            if self.queue and self.queue[0][0] == matched_index:
                self.queue.popleft()

        return matched_index, main_image, float_image
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
        Supported formats: .webp, .png, .jpg, .jpeg
        """
        if image_path.lower().endswith(('.webp', '.png', '.jpg', '.jpeg')):
            image_np = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image_np is None:
                raise ValueError(f"Failed to load image: {image_path}")
            # Ensure RGBA ordering
            if image_np.shape[2] == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGBA)
            elif image_np.shape[2] == 4:
                # BGRA -> RGBA by channel swap
                image_np = image_np[:, :, [2, 1, 0, 3]]
            return image_np
        else:
            raise ValueError("Unsupported image format.")

    def load_images(self, index, main_folder, float_folder):
        """
        Loads a pair of images from the stored folder paths.
        Returns a (main_image, float_image) tuple.
        """
        main_image = self.read_image(self.main_folder_path[index][main_folder])
        float_image = self.read_image(self.float_folder_path[index][float_folder])
        return main_image, float_image

class FIFOImageBuffer:
    """
    A FIFO buffer that stores (index, (main_image, float_image)) pairs.
    'index' acts like a simplified presentation timestamp (PTS).

    This implementation batches pruning and lookup to minimize lock contention.
    """
    def __init__(self, max_size=5):
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
                self.queue.popleft()
            self.queue.append((index, images))

    def get(self, target_index):
        """
        Retrieve the frame whose stored index is closest to target_index,
        while still within +/- TOLERANCE.

        Phases:
          1) Under lock: prune any frames outside the TOLERANCE window.
          2) Snapshot the remaining queue and find the best match outside the lock.
          3) Under lock: remove all frames up to and including the matched one.

        Returns:
            (stored_index, main_image, float_image) if a match is found,
            else None.
        """
        # Phase 1: prune old frames under lock and snapshot
        with self.lock:
            while self.queue and abs(self.queue[0][0] - target_index) > TOLERANCE:
                self.queue.popleft()
            snapshot = list(self.queue)

        # Phase 2: find the closest match without holding the lock
        best_diff = float('inf')
        best_item = None
        for stored_index, images in snapshot:
            diff = abs(stored_index - target_index)
            if diff < best_diff:
                best_diff = diff
                best_item = (stored_index, images)

        if best_item is None or best_diff > TOLERANCE:
            return None

        matched_index, (main_image, float_image) = best_item

        # Phase 3: remove matched frame and any older ones under lock
        with self.lock:
            while self.queue and self.queue[0][0] != matched_index:
                self.queue.popleft()
            if self.queue and self.queue[0][0] == matched_index:
                self.queue.popleft()

        return matched_index, main_image, float_image
