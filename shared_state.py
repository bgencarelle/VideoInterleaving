import threading

class FrameExchange:
    """Thread-safe storage to pass frames from the Render Loop to the Web Server."""
    def __init__(self):
        self.condition = threading.Condition()
        self.frame = None

    def set_frame(self, frame_bytes):
        with self.condition:
            self.frame = frame_bytes
            self.condition.notify_all()  # Wake up all Flask threads  # Wake up all Flask threads

    def get_frame(self):
        with self.condition:
            self.condition.wait()
            return self.frame

exchange = FrameExchange()