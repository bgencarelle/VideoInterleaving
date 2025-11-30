import threading

class FrameExchange:
    """Thread-safe storage to pass frames from the Render Loop to the Web Server."""
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None

    def set_frame(self, frame_bytes):
        with self.lock:
            self.frame = frame_bytes

    def get_frame(self):
        with self.lock:
            return self.frame

exchange = FrameExchange()