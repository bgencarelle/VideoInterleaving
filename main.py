# main.py
import sys
import threading
import make_file_lists
import image_display
import web_service
from settings import CLOCK_MODE

# --- REMOVED: Tee Class and sys.stdout redirection ---
# Systemd now handles the logging automatically without locking your CPU.

def start_stream_server():
    """
    Run the MJPEG web server (Flask/Raw) that serves / and /video_feed.

    This is started in a background thread so the main thread can own
    the OpenGL context for image_display.run_display().
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