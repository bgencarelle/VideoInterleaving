# main.py
import sys
import threading

# --- RESTORED: Tee Class for Monitor Visibility ---
class Tee:
    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file
        self.lock = threading.Lock()

    def write(self, data):
        with self.lock:
            # Write to systemd/console
            self.stream.write(data)
            self.stream.flush()
            # Write to file for Monitor
            self.log_file.write(data)
            self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

# Open log file for the web monitor to read
log_file = open("runtime.log", "w", buffering=1)
sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)
# --------------------------------------------------

import make_file_lists
import image_display
import web_service
from settings import CLOCK_MODE


def start_stream_server():
    """
    Run the MJPEG web server (Flask/Raw) that serves / and /video_feed.
    """
    web_service.start_server()


def main(clock=CLOCK_MODE):
    # 1. Build/refresh the file lists and CSVs
    make_file_lists.process_files()

    # 2. Start the MJPEG server on STREAM_PORT in the background
    server_thread = threading.Thread(
        target=start_stream_server,
        daemon=True,
        name="MJPEG-WebServer",
    )
    server_thread.start()

    print("🛰️  MJPEG server thread started (see web_service for port).")

    # 3. Run the display loop (this owns the GL context)
    image_display.run_display(clock)


if __name__ == "__main__":
    main()