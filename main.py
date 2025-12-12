import sys
import os
import argparse
import threading
import socket

# 1. Import Settings FIRST so we can patch them
import settings

# --- CONSTANTS ---
RESERVED_PORTS = {2323, 2324}
SYSTEM_PORTS_LIMIT = 1024
LOGS_DIR = "logs"
CACHE_DIR = "_cache"  # <--- New Clean Home for generated files


# -----------------------------------------------------------------------------
# HELPER: Port Availability & Safety Checks
# -----------------------------------------------------------------------------
def is_port_free(port):
    """Returns True if the port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('0.0.0.0', port))
            return True
        except socket.error:
            return False


def require_ports(ports):
    """Checks a list of ports. Exits if any are taken."""
    blocked = [p for p in ports if not is_port_free(p)]
    if blocked:
        print(f"❌ ERROR: The following ports are ALREADY IN USE: {blocked}")
        print("   -> Is another instance running?")
        print("   -> Try a different --port")
        sys.exit(1)


def validate_ascii_port(port):
    """Enforces safety rules for the 'ascii' (Telnet) mode."""
    if port < SYSTEM_PORTS_LIMIT:
        print(f"❌ ERROR: Port {port} is a system port (<1024).")
        sys.exit(1)

    if port in RESERVED_PORTS or (port + 1) in RESERVED_PORTS:
        print(f"❌ ERROR: Ports 2323/2324 are RESERVED for 'asciiweb' mode.")
        print(f"   -> Please use the default (2300) or specify a different range.")
        sys.exit(1)


# -----------------------------------------------------------------------------
# CONFIGURATION OVERRIDE LOGIC
# -----------------------------------------------------------------------------
def configure_runtime():
    parser = argparse.ArgumentParser(description="Video Interleaving Server")

    parser.add_argument(
        "--mode",
        choices=["web", "ascii", "asciiweb", "local"],
        default="local",
        help="Operating Mode (default: local)"
    )

    parser.add_argument(
        "--port",
        type=int,
        help="Primary Port override"
    )

    parser.add_argument(
        "--dir",
        help="Path to image source folder (overrides settings.py)"
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force a rebuild of image lists (Default: Reuse existing lists if found)"
    )

    args = parser.parse_args()

    # 1. Apply Directory Override
    if args.dir:
        abs_path = os.path.abspath(args.dir)
        if not os.path.isdir(abs_path):
            print(f"❌ ERROR: Directory not found: {abs_path}")
            sys.exit(1)
        settings.IMAGES_DIR = abs_path
        settings.MAIN_FOLDER_PATH = os.path.join(abs_path, "face")
        settings.FLOAT_FOLDER_PATH = os.path.join(abs_path, "float")

    # 2. Determine Primary Port
    if args.mode == "web":
        primary_port = 8080
    elif args.mode == "local":
        primary_port = 8888
    elif args.mode == "ascii":
        primary_port = args.port or 2300
    elif args.mode == "asciiweb":
        primary_port = args.port or 2323

    # 3. Dynamic Naming & Cache Setup
    source_name = os.path.basename(os.path.normpath(settings.IMAGES_DIR)).replace(" ", "_")
    suffix = f"{source_name}_{args.mode}_{primary_port}"

    # [CHANGE] Prepend CACHE_DIR to keep root clean
    settings.PROCESSED_DIR = os.path.join(CACHE_DIR, f"folders_processed_{suffix}")
    settings.GENERATED_LISTS_DIR = os.path.join(CACHE_DIR, f"generated_lists_{suffix}")

    # 4. Log Path Setup
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)

    log_path = os.path.join(LOGS_DIR, f"runtime_{suffix}.log")
    settings.LOG_FILE_PATH = log_path

    # --- MODE SWITCHING ---

    if args.mode == "web":
        if args.port:
            print("⚠️  WARNING: --port argument ignored in WEB mode. Using fixed port 8080.")
        print(f">> MODE: WEB (MJPEG) [{source_name}]")
        settings.ASCII_MODE = False
        settings.SERVER_MODE = True
        settings.WEB_PORT = 1978
        settings.STREAM_PORT = 8080
        require_ports([settings.WEB_PORT, settings.STREAM_PORT])

    elif args.mode == "local":
        print(f">> MODE: LOCAL (Window) [{source_name}]")
        settings.ASCII_MODE = False
        settings.SERVER_MODE = False
        settings.WEB_PORT = 8888
        require_ports([settings.WEB_PORT])

    elif args.mode == "ascii":
        validate_ascii_port(primary_port)
        print(f">> MODE: ASCII (Telnet) [{source_name}] @ {primary_port}")
        settings.ASCII_MODE = True
        settings.SERVER_MODE = False
        settings.ASCII_PORT = primary_port
        settings.WEB_PORT = primary_port + 1
        print(f">> PORTS: Telnet={settings.ASCII_PORT}, Monitor={settings.WEB_PORT}")
        require_ports([settings.ASCII_PORT, settings.WEB_PORT])

    elif args.mode == "asciiweb":
        validate_ascii_port(primary_port)
        print(f">> MODE: ASCII-WEB (WebSocket) [{source_name}]")
        settings.ASCII_MODE = True
        settings.SERVER_MODE = False
        settings.WEB_PORT = 1980
        settings.WEBSOCKET_PORT = args.port or 2324

        # Specific Logic: If default 2323, WebSocket is 2324.
        # If user overrode port (e.g. 2400), WebSocket is 2401.
        settings.WEBSOCKET_PORT = primary_port + 1

        print(f">> PORTS: Viewer={settings.WEB_PORT}, WebSocket={settings.WEBSOCKET_PORT}")
        require_ports([settings.WEB_PORT, settings.WEBSOCKET_PORT])

    return args, log_path


# Run configuration immediately
cli_args, log_filename = configure_runtime()

# -----------------------------------------------------------------------------
# STANDARD IMPORTS
# -----------------------------------------------------------------------------
import make_file_lists
import image_display
import web_service
import ascii_server
import ascii_stats_server
import ascii_web_server
from settings import CLOCK_MODE


class Tee:
    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file
        self.lock = threading.Lock()

    def write(self, data):
        with self.lock:
            self.stream.write(data)
            self.log_file.write(data)
            if '\n' in data: self.stream.flush()

    def flush(self):
        with self.lock:
            self.stream.flush()
            self.log_file.flush()


try:
    log_file = open(log_filename, "w", buffering=1, encoding='utf-8')
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
except Exception as e:
    print(f"⚠️  Logging setup failed: {e}")


def main(clock=CLOCK_MODE):
    # 1. Process Files (Reuse Logic)
    lists_exist = False
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gen_dir_full = os.path.join(script_dir, settings.GENERATED_LISTS_DIR)

    # Note: make_file_lists uses os.makedirs(), so it handles creating _cache automatically.

    if os.path.exists(gen_dir_full) and os.listdir(gen_dir_full):
        lists_exist = True

    if cli_args.rebuild or not lists_exist:
        print(">> Building file lists...")
        make_file_lists.process_files()
    else:
        print(f">> Skipping build. Reusing existing lists in: {settings.GENERATED_LISTS_DIR}")

    # 2. Launch Servers
    mode = cli_args.mode

    if mode == "ascii":
        threading.Thread(target=ascii_server.start_server, daemon=True, name="ASCII-Telnet").start()
        threading.Thread(target=ascii_stats_server.start_server, daemon=True, name="ASCII-Stats").start()

    elif mode == "asciiweb":
        threading.Thread(target=ascii_web_server.start_server, daemon=True, name="ASCII-WS").start()
        web_service.start_server(monitor=True, stream=False)

    elif mode == "web":
        web_service.start_server(monitor=True, stream=True)

    elif mode == "local":
        web_service.start_server(monitor=True, stream=False)

    # 3. Start Display Engine
    try:
        image_display.run_display(clock)
    except KeyboardInterrupt:
        print("\n[MAIN] Shutdown requested via Ctrl+C")
    except Exception as e:
        print(f"\n[MAIN] Crash: {e}")
    finally:
        print("[MAIN] Exiting...")
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    main()