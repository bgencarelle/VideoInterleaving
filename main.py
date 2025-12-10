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

    # --- 1. Apply Mode & Namespace Folders ---
    if args.mode == "ascii":
        print(">> CLI OVERRIDE: Mode set to ASCII")
        settings.ASCII_MODE = True
        settings.SERVER_MODE = False
        settings.PROCESSED_DIR = "folders_processed_ascii"
        settings.GENERATED_LISTS_DIR = "generated_img_lists_ascii"

        settings.WEB_PORT = 1980
        settings.TELNET_PORT = 2323
        settings.WEBSOCKET_PORT = 2324
        print(f">> PORTS: Monitor={settings.WEB_PORT}, Telnet={settings.TELNET_PORT}")

    elif args.mode == "web":
        print(">> CLI OVERRIDE: Mode set to WEB (MJPEG)")
        settings.ASCII_MODE = False
        settings.SERVER_MODE = True
        settings.PROCESSED_DIR = "folders_processed_web"
        settings.GENERATED_LISTS_DIR = "generated_img_lists_web"

        settings.WEB_PORT = 1978
        settings.STREAM_PORT = 8080
        print(f">> PORTS: Monitor={settings.WEB_PORT}, Stream={settings.STREAM_PORT}")

    elif args.mode == "local":
        print(">> CLI OVERRIDE: Mode set to LOCAL")
        settings.ASCII_MODE = False
        settings.SERVER_MODE = False
        settings.PROCESSED_DIR = "folders_processed_local"
        settings.GENERATED_LISTS_DIR = "generated_img_lists_local"

        settings.WEB_PORT = 8888
        print(f">> PORTS: Monitor={settings.WEB_PORT} (Local)")

    # --- 2. Apply Directory Override ---
    if args.dir:
        abs_path = os.path.abspath(args.dir)
        if not os.path.isdir(abs_path):
            print(f"❌ ERROR: Directory not found: {abs_path}")
            sys.exit(1)

        print(f">> CLI OVERRIDE: Images set to {abs_path}")
        settings.IMAGES_DIR = abs_path
        settings.MAIN_FOLDER_PATH = os.path.join(abs_path, "face")
        settings.FLOAT_FOLDER_PATH = os.path.join(abs_path, "float")


# Run configuration immediately
configure_runtime()

# -----------------------------------------------------------------------------
# STANDARD IMPORTS
# -----------------------------------------------------------------------------
import make_file_lists
import image_display
import web_service
import ascii_server
import ascii_web_server
from settings import CLOCK_MODE


class Tee:
    """
    Optimized Logger: Redirects stdout/stderr to both console and file.
    Removed aggressive flushing to prevent I/O bottlenecks.
    """

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file
        self.lock = threading.Lock()

    def write(self, data):
        with self.lock:
            # Console Write
            self.stream.write(data)
            # File Write (Buffered by Python, safe)
            self.log_file.write(data)

            # Only flush console on newlines to keep it snappy for the user
            # File flushing is handled by buffering=1 (line buffering) in open()
            if '\n' in data:
                self.stream.flush()

    def flush(self):
        with self.lock:
            self.stream.flush()
            self.log_file.flush()


# Setup Logging with Line Buffering
try:
    log_file = open("runtime.log", "w", buffering=1, encoding='utf-8')
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
        print("MODE: ASCII Server (Telnet + WebSocket)")
        threading.Thread(target=ascii_server.start_server, daemon=True, name="ASCII-Telnet").start()
        threading.Thread(target=ascii_web_server.start_server, daemon=True, name="ASCII-WS").start()
 #       web_service.start_server(monitor=want_monitor, stream=True)

        if want_monitor:
            print(f"Starting Monitor on port {settings.WEB_PORT}...")
            web_service.start_server(monitor=True, stream=False)

    elif getattr(settings, 'SERVER_MODE', False):
        print("MODE: MJPEG Web Server")
        web_service.start_server(monitor=want_monitor, stream=True)

    else:
        print("MODE: Local Standalone")
        if want_monitor:
            web_service.start_server(monitor=True, stream=False)

    # 3. Start Display Engine
    # Wrapped in try/except for clean exit on Ctrl+C
    try:
        image_display.run_display(clock)
    except KeyboardInterrupt:
        print("\n[MAIN] Shutdown requested via Ctrl+C")
    except Exception as e:
        print(f"\n[MAIN] Crash: {e}")
    finally:
        print("[MAIN] Exiting...")
        # Tee might have locked stdout, so we just let the OS close files
        # but explicit flush helps ensure log integrity
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    main()