# buffer_controller.py
from collections import deque

class ImageLoaderBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def add_image_future(self, index, future):
        # Uses the global png_paths_len (assumed to be defined in the main script)
        global png_paths_len
        clamped_index = max(0, min(index, png_paths_len - 1))
        self.buffer.append((clamped_index, future))

    def get_future_for_index(self, index):
        for item in list(self.buffer):
            buf_index, future = item
            if buf_index == index:
                self.buffer.remove(item)
                return future
        return None
