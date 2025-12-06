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

    # Apply Mode Override
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

    # Apply Directory Override
    if args.dir:
        abs_path = os.path.abspath(args.dir)
        if not os.path.isdir(abs_path):
            print(f"❌ ERROR: Directory not found: {abs_path}")
            sys.exit(1)

        print(f">> CLI OVERRIDE: Images set to {abs_path}")
        settings.IMAGES_DIR = abs_path

        # We must re-derive the subpaths because settings.py logic has already run
        # This assumes your folder structure is always /face and /float inside the root
        settings.MAIN_FOLDER_PATH = os.path.join(abs_path, "face")
        settings.FLOAT_FOLDER_PATH = os.path.join(abs_path, "float")


# Run configuration immediately
configure_runtime()

# -----------------------------------------------------------------------------
# STANDARD IMPORTS
# (Now safe to import because settings are patched)
# -----------------------------------------------------------------------------
import make_file_lists
import image_display
# We import these conditionally in main() or let the modules handle flags,
# but importing them here is safe now.
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
    # 1. Build File Lists (Uses the potentially overridden folder)
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
        # Optional: Monitor locally
        if want_monitor:
            web_service.start_server(monitor=True, stream=False)

    # 3. Start Display Engine
    image_display.run_display(clock)


if __name__ == "__main__":
    main()