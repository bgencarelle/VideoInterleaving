import sys
import os
import argparse
import threading

# 1. Import Settings FIRST so we can patch them
import settings


# -----------------------------------------------------------------------------
# CONFIGURATION OVERRIDE LOGIC
# -----------------------------------------------------------------------------
def configure_runtime():
    parser = argparse.ArgumentParser(description="Video Interleaving Server")

    # Mode Override
    parser.add_argument(
        "--mode",
        choices=["web", "ascii", "local"],
        help="Force a specific output mode (overrides settings.py)"
    )

    # Directory Override
    parser.add_argument(
        "--dir",
        help="Path to image source folder (overrides settings.py)"
    )

    args = parser.parse_args()

    # 1. Apply Mode Override
    if args.mode == "ascii":
        print(">> CLI OVERRIDE: Mode set to ASCII")
        settings.ASCII_MODE = True
        settings.SERVER_MODE = False
    elif args.mode == "web":
        print(">> CLI OVERRIDE: Mode set to WEB (MJPEG)")
        settings.ASCII_MODE = False
        settings.SERVER_MODE = True
    elif args.mode == "local":
        print(">> CLI OVERRIDE: Mode set to LOCAL")
        settings.ASCII_MODE = False
        settings.SERVER_MODE = False

    # 2. Apply Directory Override
    if args.dir:
        abs_path = os.path.abspath(args.dir)
        if not os.path.isdir(abs_path):
            print(f"❌ ERROR: Directory not found: {abs_path}")
            sys.exit(1)

        print(f">> CLI OVERRIDE: Images set to {abs_path}")
        settings.IMAGES_DIR = abs_path
        settings.MAIN_FOLDER_PATH = os.path.join(abs_path, "face")
        settings.FLOAT_FOLDER_PATH = os.path.join(abs_path, "float")

    # 3. AUTOMATIC PORT OFFSETTING
    # Base Port comes from settings.py (Default 1978)
    base_port = getattr(settings, 'WEB_PORT', 1978)

    if settings.ASCII_MODE:
        # ASCII: Base + 2 (e.g. 1980)
        settings.WEB_PORT = base_port + 2
        print(f">> PORT CONFIG: Monitor shifted to {settings.WEB_PORT} (ASCII Mode)")

    elif settings.SERVER_MODE:
        # WEB: Keep Base (e.g. 1978) - Compatibility Mode
        settings.WEB_PORT = base_port
        print(f">> PORT CONFIG: Monitor on default {settings.WEB_PORT} (Web Mode)")

    else:
        # LOCAL: Base + 1 (e.g. 1979) - Avoids clash if Web is also running
        settings.WEB_PORT = base_port + 1
        print(f">> PORT CONFIG: Monitor shifted to {settings.WEB_PORT} (Local Mode)")


# Run configuration immediately
configure_runtime()

# -----------------------------------------------------------------------------
# STANDARD IMPORTS
# -----------------------------------------------------------------------------
import make_file_lists
import image_display
import web_service
import ascii_server
from settings import CLOCK_MODE


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

    want_monitor = getattr(settings, 'HTTP_MONITOR', True)

    # 2. Determine Mode
    if getattr(settings, 'ASCII_MODE', False):
        # --- PATH A: ASCII MODE ---
        print("MODE: ASCII Telnet Server")
        server_thread = threading.Thread(
            target=ascii_server.start_server,
            daemon=True,
            name="ASCII-TelnetServer"
        )
        server_thread.start()

        if want_monitor:
            web_service.start_server(monitor=True, stream=False)

    elif getattr(settings, 'SERVER_MODE', False):
        # --- PATH B: WEB MODE ---
        print("MODE: MJPEG Web Server")
        web_service.start_server(monitor=want_monitor, stream=True)

    else:
        # --- PATH C: LOCAL MODE ---
        print("MODE: Local Standalone")
        if want_monitor:
            web_service.start_server(monitor=True, stream=False)

    # 3. Start Display Engine
    image_display.run_display(clock)


if __name__ == "__main__":
    main()