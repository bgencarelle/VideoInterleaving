import traceback
import sys
import os
import argparse
import threading
import socket
import atexit
import shutil

# 1. Import Settings FIRST so we can patch them
import settings
from server_config import ServerConfig, get_config, MODE_WEB, MODE_LOCAL, MODE_ASCII, MODE_ASCIIWEB, MODE_SCOPE

# --- CONSTANTS ---
# [CHANGE] Updated reserved ports to the new 24xx range
RESERVED_PORTS = {2423, 2424}
SYSTEM_PORTS_LIMIT = 1024
LOGS_DIR = "logs"
CACHE_DIR = "_cache"


# -----------------------------------------------------------------------------
# HELPER: Port Availability & Safety Checks
# -----------------------------------------------------------------------------
def is_port_free(port):
    """Returns True if the port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # [FIX] Allow reusing the address if it's in TIME_WAIT from a recent shutdown
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(('127.0.0.1', port))
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
        # [CHANGE] Updated error message to reflect new ports
        print(f"❌ ERROR: Ports 2423/2424 are RESERVED for 'asciiweb' mode.")
        print(f"   -> Please use the default (2323) or specify a different range.")
        sys.exit(1)


# -----------------------------------------------------------------------------
# CONFIGURATION OVERRIDE LOGIC
# -----------------------------------------------------------------------------
def configure_runtime():
    parser = argparse.ArgumentParser(description="Video Interleaving Server")

    parser.add_argument(
        "--mode",
        choices=["web", "ascii", "asciiweb", "local", "scope"],
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

    # --- Options for --mode scope (XY output on the sound card) ---
    parser.add_argument("--xy-dir", help="Baked XY libraries (default: settings.XY_DIR)")
    parser.add_argument("--scope-raster", action="store_true",
                        help="Scope: scanline/dwell mode instead of vector line art")
    parser.add_argument("--scope-realtime", action="store_true",
                        help="Scope: stream continuously so index changes land "
                             "within a row instead of at a trace boundary (raster only)")
    parser.add_argument("--scope-fps", type=int, help="Scope trace rate (default: IPS)")
    parser.add_argument("--scope-samples", type=int, help="Scope samples per trace")
    parser.add_argument("--scope-fields", type=int, metavar="N",
                        help="Scope raster: interlace. N=2 draws every other "
                             "row per trace and alternates, so the beam "
                             "repaints at N x the picture rate at the SAME "
                             "grid. Requires --scope-fps = N * IPS. "
                             "Default: settings.SCOPE_FIELDS (1 = progressive)")
    parser.add_argument("--scope-trim", type=float,
                        help="Scope raster: drop cells dimmer than this (0.08-0.16 "
                             "reduces stray lines on dark backgrounds). "
                             "Default: settings.SCOPE_TRIM")
    parser.add_argument("--scope-gamma", type=float,
                        help="Default: settings.SCOPE_GAMMA")
    parser.add_argument("--scope-density", type=float,
                        help="Default: settings.SCOPE_DENSITY")
    parser.add_argument("--scope-rows", type=int)
    parser.add_argument("--scope-border", type=float, metavar="F",
                        help="Draw a fixed one-cell rectangle at the full "
                             "extent every trace, spending F of the trace's "
                             "samples on it (try 0.03). Without it the drawn "
                             "extent is whatever the content occupies, so dark "
                             "margins pull that side in and the picture skews "
                             "and rescales as the subject changes. 0 = off.")
    parser.add_argument("--scope-dc-comp", type=float, metavar="HZ",
                        help="Pre-compensate the output's AC coupling at this "
                             "corner frequency, so a DC-ish trace holds its "
                             "shape instead of sagging. Start at 30. Costs "
                             "amplitude: ~26%% at a 20 Hz corner. Both handoff "
                             "documents already list this flag; until now it "
                             "existed only in scope_screen.py as --dc-comp.")
    parser.add_argument("--scope-lowpass", type=float, metavar="HZ",
                        help="Low-pass the XY output at this corner, to emulate "
                             "a softer DAC or a physical RC filter. Try 800-8000; "
                             "see scope_lowpass.py to audition it first.")
    parser.add_argument("--scope-oversample", type=int, metavar="N",
                        help="Anti-alias the beam path: generate N x samples, "
                             "bandlimit, decimate. Fixes aliasing when the grid "
                             "is finer than the sample rate; on smooth content "
                             "the difference is small. 4 is plenty.")
    parser.add_argument("--scope-list-from-images", action="store_true",
                        help="Build the folder manifest by scanning the images "
                             "instead of reading it from the bake. Only needed "
                             "for a bake made before manifests were supported.")
    parser.add_argument("--scope-no-autofit", action="store_true",
                        help="Scope raster: size the grid to the whole frame "
                             "instead of to the cells that survive trim "
                             "(autofit is on by default and is usually a 2x "
                             "resolution win on a dark background)")
    parser.add_argument("--scope-mix", nargs="?", type=float, const=120.0,
                        metavar="HZ",
                        help="Scope: alternate raster and vector every trace at "
                             "this rate (default 120). Above flicker fusion the "
                             "phosphor sums them: raster gives tone, vector gives "
                             "outline.")
    parser.add_argument("--scope-mix-duty", type=float, default=None,
                        help="Scope: fraction of mixed passes spent on raster "
                             "(0.6-0.8 if the vector outline overpowers the tone)")
    parser.add_argument("--scope-sweep", choices=("alternate", "palindrome", "retrace"),
                        default=None,
                        help="Scope raster: alternate (default) chains one-way "
                             "sweeps with no flyback; palindrome is safe when "
                             "traces repeat; retrace shows the CRT flyback")
    parser.add_argument("--scope-min-feature", type=float, default=None,
                        help="Scope vector: shortest stroke kept by the occlusion cull")
    parser.add_argument("--device", "--scope-device", dest="scope_device",
                        help="Audio output: index or name fragment, e.g. "
                             "--device Scarlett. Prefer the name over an index: "
                             "PortAudio indices reshuffle when hardware is "
                             "plugged or unplugged.")
    parser.add_argument("--ask", "--scope-ask", dest="scope_ask",
                        action="store_true",
                        help="Choose the audio output interactively. Only "
                             "prompts when more than one output exists, and "
                             "only when there is a terminal -- safe to leave in "
                             "a kiosk launch script.")

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force a rebuild of image lists (Default: Reuse existing lists if found)"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Override hosts to '0.0.0.0' for network testing (default: '127.0.0.1')"
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
        # settings computed XY_DIR at import time from the DEFAULT images dir,
        # so it is stale once --dir moves us. Re-derive unless --xy-dir won.
        if not args.xy_dir:
            settings.XY_DIR = settings._find_xy_dir(abs_path)

    if args.xy_dir:
        abs_xy = os.path.abspath(args.xy_dir)
        if not os.path.isdir(abs_xy):
            print(f"❌ ERROR: XY directory not found: {abs_xy}")
            print("   -> Bake it: python utilities/convert_to_xy.py -i <images> -o " + abs_xy)
            sys.exit(1)
        settings.XY_DIR = abs_xy

    # Scope options only mean anything in scope mode; say so rather than
    # silently ignoring them.
    if args.mode != "scope":
        _short = {"scope_ask": "--ask", "scope_device": "--device"}
        used = [_short.get(a, f"--{a.replace('_', '-')}") for a in vars(args)
                if a.startswith("scope_") and getattr(args, a) not in (None, False)]
        if args.xy_dir:
            used.append("--xy-dir")
        if used:
            print(f"⚠️  {', '.join(used)} ignored: these apply to --mode scope. "
                  f"Each mode in VideoInterleaving runs standalone.")

    # 2. Determine Primary Port (for ASCII modes)
    if args.mode == "web":
        primary_port = None  # Not used in web mode
    elif args.mode == "local":
        primary_port = None  # Not used in local mode
    elif args.mode == "scope":
        primary_port = None  # Not used in scope mode
    elif args.mode == "ascii":
        primary_port = args.port or 2323
    elif args.mode == "asciiweb":
        # Default updated to 2423
        primary_port = args.port or 2423

    # 2.5. Clean up existing cache directories for this instance
    # Determine instance identifier pattern
    if args.mode == "scope":
        # Scope only: key the pattern on source AND mode, so two scope
        # instances running different image trees never delete each other's
        # lists.  source_name is computed below, so derive it locally here
        # rather than moving upstream code around.
        _src = os.path.basename(os.path.normpath(settings.IMAGES_DIR)).replace(" ", "_")
        instance_pattern = f"_{_src}_{args.mode}_"
    elif args.mode in ("web", "local"):
        instance_pattern = f"_{args.mode}_"  # Match any port
    else:
        # ASCII modes: match specific port
        instance_pattern = f"_{args.mode}_{primary_port}"

    # Clean up existing cache directories for this instance
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_base = os.path.join(script_dir, CACHE_DIR)

    if os.path.exists(cache_base):
        for item in os.listdir(cache_base):
            if (item.startswith("folders_processed_") or item.startswith(
                    "generated_lists_")) and instance_pattern in item:
                full_path = os.path.join(cache_base, item)
                if os.path.isdir(full_path):
                    shutil.rmtree(full_path)
                    print(f">> Deleted old cache: {item}")

    # 3. Dynamic Naming & Cache Setup
    source_name = os.path.basename(os.path.normpath(settings.IMAGES_DIR)).replace(" ", "_")
    suffix = f"{source_name}_{args.mode}_{primary_port}"

    settings.PROCESSED_DIR = os.path.join(CACHE_DIR, f"folders_processed_{suffix}")
    settings.GENERATED_LISTS_DIR = os.path.join(CACHE_DIR, f"generated_lists_{suffix}")

    # 4. Log Path Setup
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)

    log_path = os.path.join(LOGS_DIR, f"runtime_{suffix}.log")
    settings.LOG_FILE_PATH = log_path

    # --- MODE SWITCHING ---
    # Initialize ServerConfig with the selected mode
    config = get_config()

    if args.mode == "web":
        if args.port:
            print("⚠️  WARNING: --port argument ignored in WEB mode. Using fixed ports.")
        print(f">> MODE: WEB (MJPEG) [{source_name}]")
        settings.ASCII_MODE = False
        settings.SERVER_MODE = True
        config.set_mode(MODE_WEB)
        ports = config.get_ports()
        print(f">> PORTS: Monitor={ports.monitor}, Stream={ports.stream}")
        require_ports(ports.get_all_ports())
        # Update settings for backward compatibility
        settings.WEB_PORT = ports.monitor
        settings.STREAM_PORT = ports.stream

    elif args.mode == "local":
        print(f">> MODE: LOCAL (Window) [{source_name}]")
        settings.ASCII_MODE = False
        settings.SERVER_MODE = False
        config.set_mode(MODE_LOCAL)
        ports = config.get_ports()
        print(f">> PORTS: Monitor={ports.monitor}")
        require_ports(ports.get_all_ports())
        # Update settings for backward compatibility
        settings.WEB_PORT = ports.monitor
        # Local mode always uses --test flag for network monitoring
        args.test = True
        print(">> Local mode: Enabling --test flag for network monitoring")

    elif args.mode == "scope":
        if args.port:
            print("⚠️  WARNING: --port ignored in SCOPE mode.")
        print(f">> MODE: SCOPE (XY audio) [{source_name}]")
        settings.ASCII_MODE = False
        settings.SERVER_MODE = False
        settings.SCOPE_MODE = True
        config.set_mode(MODE_SCOPE)
        ports = config.get_ports()
        # Deliberately NOT require_ports() here.  Scope binds nothing -- the
        # launch block below is `pass  # No servers` -- but require_ports()
        # sys.exit(1)s on a busy port.  So scope refused to start whenever web
        # or ascii already held the monitor port, over a port it never opens,
        # and under `Restart=always` a port still in TIME_WAIT after a crash
        # would fail the unit fast enough to trip StartLimitBurst and stop it
        # permanently.  WEB_PORT is still published for anything that reads it.
        settings.WEB_PORT = ports.monitor

        # Publish CLI overrides into settings; scope_display reads settings,
        # exactly as the other modes read ASCII_MODE / SERVER_MODE.
        if args.scope_raster:
            settings.SCOPE_RASTER = True
        if args.scope_realtime:
            settings.SCOPE_REALTIME = True
        if args.scope_fps:
            settings.SCOPE_FPS = args.scope_fps
        if getattr(args, "scope_fields", None):
            settings.SCOPE_FIELDS = args.scope_fields
        if getattr(args, "scope_dc_comp", None):
            settings.SCOPE_DC_COMP = args.scope_dc_comp
        if getattr(args, "scope_border", None) is not None:
            settings.SCOPE_BORDER = args.scope_border
        if args.scope_samples:
            settings.SCOPE_SAMPLES = args.scope_samples
        if args.scope_trim is not None:
            settings.SCOPE_TRIM = args.scope_trim
        if args.scope_gamma is not None:
            settings.SCOPE_GAMMA = args.scope_gamma
        if args.scope_density is not None:
            settings.SCOPE_DENSITY = args.scope_density
        if args.scope_rows:
            settings.SCOPE_ROWS = args.scope_rows
        if args.scope_no_autofit:
            settings.SCOPE_AUTOFIT = False
        if args.scope_list_from_images:
            settings.SCOPE_LIST_FROM_IMAGES = True
        if args.scope_lowpass:
            settings.SCOPE_LOWPASS = args.scope_lowpass
        if args.scope_oversample:
            settings.SCOPE_OVERSAMPLE = args.scope_oversample
        if args.scope_min_feature is not None:
            settings.SCOPE_MIN_FEATURE = args.scope_min_feature
        if args.scope_sweep:
            settings.SCOPE_SWEEP = args.scope_sweep
        if args.scope_mix:
            settings.SCOPE_MIX = args.scope_mix
        if args.scope_mix_duty is not None:
            settings.SCOPE_MIX_DUTY = args.scope_mix_duty
        # Resolve the audio device NOW, before file lists are built and before
        # stdout is wrapped. Prompting from deep inside run_scope meant the
        # question appeared after a long silence, so it read as a hang.
        if args.scope_ask or args.scope_device:
            from scope_out import choose_device as _choose, scrub as _scrub
            # argv is decoded with surrogateescape, so a stray byte in shell
            # history arrives as a lone surrogate and breaks any later encode
            args.scope_device = _scrub(args.scope_device)
            settings.SCOPE_DEVICE = _choose(ask=args.scope_ask,
                                            device=args.scope_device)
            import sounddevice as _sd
            try:
                _name = _scrub(_sd.query_devices(settings.SCOPE_DEVICE)["name"]) \
                    if settings.SCOPE_DEVICE is not None else "system default"
            except Exception:
                _name = str(settings.SCOPE_DEVICE)
            print(f">> AUDIO OUT: {_name}")
            # Explicit flag: scope_display must NOT re-resolve, or the choice
            # silently reverts to the system default.
            settings.SCOPE_DEVICE_RESOLVED = True
        settings.SCOPE_DEVICE_SPEC = None
        settings.SCOPE_ASK = False
        print(f">> XY LIBRARIES: {getattr(settings, 'XY_DIR', 'images_xy')}")

    elif args.mode == "ascii":
        validate_ascii_port(primary_port)
        print(f">> MODE: ASCII (Telnet) [{source_name}] @ {primary_port}")
        settings.ASCII_MODE = True
        settings.SERVER_MODE = False
        config.set_mode(MODE_ASCII, primary_port=primary_port)
        ports = config.get_ports()
        print(f">> PORTS: Telnet={ports.ascii_telnet}, Monitor={ports.monitor}")
        require_ports(ports.get_all_ports())
        # Update settings for backward compatibility
        settings.ASCII_PORT = ports.ascii_telnet
        settings.WEB_PORT = ports.monitor

    elif args.mode == "asciiweb":
        # [NOTE] Validation skipped here so asciiweb can use its own reserved ports
        print(f">> MODE: ASCII-WEB (WebSocket) [{source_name}]")
        settings.ASCII_MODE = True
        settings.SERVER_MODE = False
        config.set_mode(MODE_ASCIIWEB, primary_port=primary_port)
        ports = config.get_ports()
        print(f">> PORTS: Viewer={ports.monitor}, WebSocket={ports.ascii_websocket}")
        require_ports(ports.get_all_ports())
        # Update settings for backward compatibility
        settings.WEB_PORT = ports.monitor
        settings.WEBSOCKET_PORT = ports.ascii_websocket

    # Apply --test flag: Override hosts to '0.0.0.0' for network testing
    if args.test:
        settings.WEB_HOST = '0.0.0.0'
        settings.ASCII_HOST = '0.0.0.0'
        # STREAM_HOST already defaults to '0.0.0.0', no change needed
        print("⚠️  TEST MODE: Servers will bind to '0.0.0.0' (accessible from network)")

    return args, log_path


# Run configuration immediately
cli_args, log_filename = configure_runtime()

# -----------------------------------------------------------------------------
# STANDARD IMPORTS
# -----------------------------------------------------------------------------
import make_file_lists

# Imported here, as upstream, so startup behaviour for the existing modes is
# unchanged.  The failure is only DEFERRED (not skipped) so that scope mode --
# which needs neither TurboJPEG nor GL -- can still run on a box without them.
try:
    import image_display

    _image_display_error = None
except Exception as _e:  # pragma: no cover - environment dependent
    image_display = None
    _image_display_error = _e
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
            try:
                self.stream.write(data)
                self.stream.flush()  # Always flush stream to ensure output
            except (BrokenPipeError, OSError):
                # Stream might be closed (e.g., redirected and closed), continue anyway
                pass
            try:
                self.log_file.write(data)
                # Flush log file frequently to ensure it's written, especially when running remotely
                if '\n' in data or len(data) > 0:
                    self.log_file.flush()
            except Exception:
                # If log file write fails, at least we tried
                pass

    def flush(self):
        with self.lock:
            try:
                self.stream.flush()
            except (BrokenPipeError, OSError):
                pass
            try:
                self.log_file.flush()
            except Exception:
                pass


# Store original streams and log file for cleanup
_original_stdout = sys.stdout
_original_stderr = sys.stderr
_log_file = None

# Check if running under systemd (stdout/stderr already redirected)
# In this case, we should be more careful about additional redirection
_is_systemd = os.environ.get('INVOCATION_ID') is not None

try:
    # Ensure logs directory exists
    log_dir = os.path.dirname(log_filename)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    _log_file = open(log_filename, "w", buffering=1, encoding='utf-8')

    # Always use Tee to capture all output to both terminal and log file
    # This works whether stdout/stderr are redirected or not
    # If stdout/stderr are already redirected (e.g., by wrapper script),
    # Tee will write to both the redirected stream and the log file
    sys.stdout = Tee(sys.stdout, _log_file)
    sys.stderr = Tee(sys.stderr, _log_file)

    # Write initial log message (use original stderr if systemd to avoid issues)
    if _is_systemd:
        _original_stderr.write(f"[MAIN] Logging to {log_filename}\n")
        _log_file.write(f"[MAIN] Logging to {log_filename}\n")
        _log_file.flush()
    else:
        print(f"[MAIN] Logging to {log_filename}")
except Exception as e:
    # Use original stderr to avoid recursion if stdout/stderr are broken
    try:
        _original_stderr.write(f"⚠️  Logging setup failed: {e}\n")
        _original_stderr.flush()
    except Exception:
        pass  # If even stderr is broken, we can't do anything


def main(clock=CLOCK_MODE):
    # Register cleanup handler for display resolution restoration
    try:
        from display_manager import _restore_display_resolution
        atexit.register(_restore_display_resolution)
    except ImportError:
        pass  # display_manager may not be imported yet

    # 1. Process Files (Reuse Logic)
    lists_exist = False
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gen_dir_full = os.path.join(script_dir, settings.GENERATED_LISTS_DIR)

    print(f"[MAIN] Mode={cli_args.mode} | Images={settings.IMAGES_DIR} | Cache={settings.GENERATED_LISTS_DIR}")

    if os.path.exists(gen_dir_full) and os.listdir(gen_dir_full):
        lists_exist = True

    # Scope mode reads its folder manifest from the bake and never opens an
    # image, so the image scan is pure startup cost for it.  Skipped only when
    # the bake can actually supply the manifest; every other mode is untouched.
    _scope_has_manifest = False
    if cli_args.mode == "scope" and not getattr(settings, "SCOPE_LIST_FROM_IMAGES", False):
        _xy = getattr(settings, "XY_DIR", None)
        if _xy and os.path.isdir(_xy):
            for _root, _dirs, _files in os.walk(_xy):
                if "frame_starts.npy" in _files:
                    _scope_has_manifest = True
                    break

    if _scope_has_manifest:
        print(">> Skipping image scan: scope mode takes its manifest from the "
              "bake (--scope-list-from-images to override)")
    elif cli_args.rebuild or not lists_exist:
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

    elif mode == "scope":
        # Scope is reached the same way every other mode is: over the network.
        # It has no window and, under systemd, no tty either -- so without this
        # there is no way to see whether it is running, what device it grabbed,
        # or whether it is dropping traces.  Monitor only; there is no video
        # frame to stream.
        #
        # NOT gated on require_ports(): run_monitor_server() runs in a daemon
        # thread and its OSError on a busy port dies in that thread.  The audio
        # keeps going.  A monitor port held by another instance must never take
        # the installation off the air.
        web_service.start_server(monitor=True, stream=False)

        # 3. Start Display Engine
    try:
        if mode == "scope":
            # Standalone, like every other mode: audio only. No GL context, no
            # TurboJPEG, no ImageLoader, no FIFO.
            import scope_display
            scope_display.run_scope(clock)
        else:
            if image_display is None:
                raise _image_display_error
            image_display.run_display(clock)
    except KeyboardInterrupt:
        print("\n[MAIN] Shutdown requested via Ctrl+C")
    except Exception as e:
        print(f"\n[MAIN] CRASH DETAILS: {e}")
        traceback.print_exc()
    finally:
        print("[MAIN] Exiting...")
        # Restore display resolution if it was changed
        try:
            from display_manager import _restore_display_resolution
            _restore_display_resolution()
        except Exception as e:
            # Don't fail if restoration fails
            pass
        # Ensure all output is flushed before closing
        sys.stdout.flush()
        sys.stderr.flush()
        # Restore original streams
        sys.stdout = _original_stdout
        sys.stderr = _original_stderr
        # Close log file
        if _log_file is not None:
            try:
                _log_file.flush()
                _log_file.close()
            except Exception as e:
                _original_stderr.write(f"⚠️  Error closing log file: {e}\n")


if __name__ == "__main__":
    main()