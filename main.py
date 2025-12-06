import sys
import threading
import make_file_lists
import image_display
import settings
from settings import CLOCK_MODE

# --- Output Modules ---
import web_service
import ascii_server


class Tee:
    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file
        self.lock = threading.Lock()

    def write(self, data):
        with self.lock:
            self.stream.write(data)
            self.stream.flush()
            self.log_file.write(data)
            self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()


try:
    log_file = open("runtime.log", "w", buffering=1)
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
except Exception as e:
    print(f"Failed to setup logging: {e}")


def main(clock=CLOCK_MODE):
    # 1. Build File Lists
    make_file_lists.process_files()

    # Determine if Monitor is wanted (Global setting)
    want_monitor = getattr(settings, 'HTTP_MONITOR', True)

    # 2. Determine Mode (Mutually Exclusive)
    if getattr(settings, 'ASCII_MODE', False):
        # --- PATH A: ASCII MODE ---
        print("MODE: ASCII Telnet Server")

        # 1. Start Telnet
        server_thread = threading.Thread(
            target=ascii_server.start_server,
            daemon=True,
            name="ASCII-TelnetServer"
        )
        server_thread.start()

        # 2. Start Monitor ONLY (No MJPEG)
        if want_monitor:
            web_service.start_server(monitor=True, stream=False)

    elif getattr(settings, 'SERVER_MODE', False):
        # --- PATH B: WEB MODE ---
        print("MODE: MJPEG Web Server")

        # Start Monitor AND MJPEG
        web_service.start_server(monitor=want_monitor, stream=True)

    else:
        # --- PATH C: LOCAL MODE ---
        print("MODE: Local Standalone")
        # Optional: You might want the monitor locally too?
        if want_monitor:
            web_service.start_server(monitor=True, stream=False)

    # 3. Start Display Engine
    image_display.run_display(clock)


if __name__ == "__main__":
    main()