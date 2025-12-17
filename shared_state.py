import threading

class FrameExchange:
    """Thread-safe storage to pass frames from the Render Loop to the Web Server."""
    def __init__(self):
        self.condition = threading.Condition()
        self.frame = None

    def set_frame(self, frame_bytes):
        with self.condition:
            self.frame = frame_bytes
            self.condition.notify_all()  # Wake up all waiting threads

    def get_frame(self, timeout=None):
        """
        Wait for a new frame.
        If timeout is set (seconds) and expires, returns None.
        """
        with self.condition:
            # wait() returns True if notified, False if timed out
            notified = self.condition.wait(timeout=timeout)
            if not notified:
                return None
            return self.frame

# Separate exchanges for web and ASCII to prevent frame format conflicts
exchange = FrameExchange()  # Legacy: single exchange (backward compatibility, not used by servers)
exchange_web = FrameExchange()  # Web stream (JPEG/WebP frames) - used by web_service.py
exchange_ascii = FrameExchange()  # ASCII servers (text frames) - used by ascii_server.py and ascii_web_server.py