import traceback
import sys
import os
import argparse
import math
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


def _set_process_title(mode):
    """Give the running application a short, mode-specific process title."""
    try:
        import setproctitle
    except ImportError:
        # Keep an existing environment usable until its requirements are refreshed.
        return
    setproctitle.setproctitle(f"vi.{mode}")


def _parse_scope_channels(value):
    """Parse the scope's user-facing, 1-based X,Y output-channel pair."""
    try:
        pair = (tuple(int(part.strip()) for part in value.split(","))
                if isinstance(value, str)
                else tuple(int(channel) for channel in value))
    except (TypeError, ValueError) as exc:
        raise ValueError("scope channels must be two comma-separated integers") from exc
    if len(pair) != 2 or min(pair) < 1 or pair[0] == pair[1]:
        raise ValueError("scope channels must be two distinct positive channel numbers")
    return pair


# -----------------------------------------------------------------------------
# CONFIGURATION OVERRIDE LOGIC
# -----------------------------------------------------------------------------
def configure_runtime():
    parser = argparse.ArgumentParser(description="Video Interleaving Server")

    parser.add_argument(
        "--mode",
        choices=["web", "ascii", "asciiweb", "local", "scope", "modem"],
        default="local",
        help="Operating Mode (default: local)"
    )

    parser.add_argument(
        "--port",
        type=int,
        help="Primary Port override"
    )

    # Orientation is not scope-specific: the local window and the scope output
    # are two renderings of the same picture, and having one of them settable
    # only by editing constantStorage/display_constants.py meant the constant
    # got edited instead -- a source change to turn a display sideways.
    parser.add_argument(
        "--rotation",
        type=int,
        choices=[0, 90, 180, 270],
        help="Quarter-turn rotation for local and scope output "
             "(default: INITIAL_ROTATION). Press r live to change it"
    )

    parser.add_argument(
        "--mirror",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Left-right flip for local and scope output "
             "(default: INITIAL_MIRROR). Press m live to toggle it"
    )

    # The ASCII grading stage, in the order it is applied. All three only mean
    # anything in --mode ascii and asciiweb; passing them elsewhere says so.
    parser.add_argument(
        "--ascii-contrast",
        type=float,
        metavar="F",
        help="ASCII grading contrast, scaled about mid-grey "
             "(default: ASCII_CONTRAST, 1.0 = neutral, >1 more punch, "
             "<1 flatter). Applies to --mode ascii and asciiweb"
    )

    parser.add_argument(
        "--ascii-brightness",
        type=float,
        metavar="F",
        help="ASCII grading brightness, a straight multiply on value "
             "(default: ASCII_BRIGHTNESS, 1.0 = neutral, 0 = black). "
             "Applies after contrast"
    )

    parser.add_argument(
        "--ascii-gamma",
        type=float,
        metavar="F",
        help="ASCII gamma on the grey the character is chosen from "
             "(default: ASCII_GAMMA, 1.0 = linear, <1 lifts shadows). "
             "Applied last, after contrast and brightness"
    )

    parser.add_argument(
        "--dir",
        help="Path to image source folder (overrides settings.py)"
    )

    # Private handoff used only when the sender GUI owns a local-mode process.
    parser.add_argument("--local-frame-port", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--local-frame-fps", type=float, help=argparse.SUPPRESS)

    # --- Options for --mode scope (scope signal output on the sound card) ---
    parser.add_argument("--xy-dir", help="Baked XY libraries (default: settings.XY_DIR)")
    parser.add_argument("--scope-source", choices=("bake", "images"),
                        default=None,
                        help="Scope source: baked XY libraries or images decoded "
                             "at runtime (default: bake)")
    parser.add_argument("--scope-channels", default=None, metavar="X,Y",
                        help="1-based PortAudio output channels for XY, e.g. "
                             "18,19 (default: 1,2)")
    parser.add_argument("--scope-live-size", type=int, default=None,
                        metavar="PX",
                        help="Live-image thumbnail width (default: 128)")
    parser.add_argument("--scope-gui", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="Open the optional native live scope preview and tuner")
    parser.add_argument("--scope-gui-image-only", "--image-only",
                        dest="scope_gui_image_only",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Start the scope GUI with only the preview; click to restore controls")
    parser.add_argument("--scope-gui-fullscreen", "--fullscreen",
                        dest="scope_gui_fullscreen",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Start the scope GUI fullscreen in image-only view")
    scope_render = parser.add_mutually_exclusive_group()
    scope_render.add_argument("--scope-mode",
                              choices=("vector", "raster", "stochastic",
                                       "stipple", "fusion"),
                              help="Scope renderer (default: settings.SCOPE_RENDER_MODE)")
    scope_render.add_argument("--scope-raster", action="store_true",
                              help="Alias for --scope-mode raster")
    scope_render.add_argument("--scope-stochastic", action="store_true",
                              help="Alias for --scope-mode stochastic: Osci-style "
                                   "luminance-weighted XY walk with no Z channel")
    scope_render.add_argument("--scope-stipple", action="store_true",
                              help="Alias for --scope-mode stipple: stable "
                                   "weighted points with a Euclidean route")
    parser.add_argument("--scope-invert", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="Invert covered image luminance in raster, "
                             "stochastic, stipple, mix, and fusion. Vector "
                             "keeps its geometry but shifts beam dwell toward "
                             "originally dark regions. Transparent padding "
                             "stays dark. Press i to toggle live")
    parser.add_argument("--scope-x-only",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Scope: send only the X signal through a one-channel "
                             "output for a single-input Y-T scope")
    parser.add_argument("--scope-realtime", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="Scope: stream continuously so index changes land "
                             "within a row instead of at a trace boundary "
                             "(raster only)")
    parser.add_argument("--scope-trigger",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Scope: one unique rising edge on X per trace, so "
                             "a single-channel scope in Y-T can lock onto the "
                             "picture. On by default and invisible on an XY "
                             "display; --no-scope-trigger removes it")
    parser.add_argument("--scope-trigger-us", "--scope-trigger-duration",
                        "--scope-yt-trigger", "--scope-yt-trigger-us",
                        dest="scope_trigger_us", type=float, metavar="US",
                        help="Scope trigger marker duration in microseconds "
                             "(default: 250)")
    parser.add_argument("--scope-trigger-shape", choices=("ramp", "step"),
                        help="ramp (default): the marker sweeps and parks "
                             "outside the picture box, so an XY display never "
                             "shows it. step: the original two-dwell marker, "
                             "for a scope whose trigger will not take a ramp")
    parser.add_argument("--scope-yt-timing", choices=("fixed", "dwell"),
                        help="Raster row timing. dwell (default): brightness "
                             "shares out the whole trace. fixed: equal row "
                             "slots, so unrelated brightness cannot move or "
                             "resize a row -- at the cost of tonal balance "
                             "and a rail on empty rows")
    parser.add_argument("--scope-yt", action="store_true",
                        help=argparse.SUPPRESS)  # deprecated: = fixed timing
    parser.add_argument("--scope-fps", type=int, help="Scope trace rate (default: IPS)")
    parser.add_argument("--scope-samples", type=int, help="Scope samples per trace")
    parser.add_argument("--scope-geometry-samples", type=int, metavar="N",
                        help="Baked vector/raster/stipple trajectory detail budget independent of DAC samples")
    parser.add_argument("--scope-traversal-hz", type=float, metavar="HZ",
                        help="Timed baked vector/raster/stipple traversal speed in cycles/second; requires trigger and one field")
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
                        help="Luminance exponent for the active raster or "
                             "stochastic renderer. Stochastic default 2 keeps "
                             "portrait midtones; raster default is "
                             "settings.SCOPE_GAMMA. In mix this sets both; "
                             "fusion also initializes both; "
                             "--scope-stochastic-gamma can then override the "
                             "stochastic side")
    parser.add_argument("--scope-density", type=float,
                        help="Default: settings.SCOPE_DENSITY")
    parser.add_argument("--scope-precondition", type=float, metavar="F",
                        help="Raster horizontal compensation on the final "
                             "sweep grid. Default 0 preserves natural facial "
                             "tone and full spatial resolution. Positive "
                             "values are optional display-specific sharpening")
    parser.add_argument("--scope-walk-radius", type=int, metavar="PX",
                        help="Scope stochastic: nearest-neighbour search radius")
    parser.add_argument("--scope-walk-stride", type=int, metavar="PX",
                        help="Scope stochastic: source-pixel step. Default 0 "
                             "auto-scales to the baked width (1 at the compact "
                             "default width 128)")
    parser.add_argument("--scope-walk-reseed-ms", type=float, metavar="MS",
                        help="Scope stochastic: time between random reseeds "
                             "(default 5 ms, matching Osci-render)")
    parser.add_argument("--scope-stochastic-gamma", "--scope-walk-gamma",
                        dest="scope_walk_gamma", type=float, metavar="F",
                        help="Scope stochastic luminance exponent. Useful in "
                             "mix when --scope-gamma is being used for raster; "
                             "also shapes stochastic position weights in "
                             "fusion; "
                             "the old --scope-walk-gamma spelling remains an "
                             "accepted compatibility alias")
    parser.add_argument("--scope-fusion", choices=("vrs", "vr", "sv", "sr"),
                        help="Scope fusion components: vector+raster+stochastic, "
                             "vector+raster, stochastic+vector, or "
                             "stochastic+raster (default settings.SCOPE_FUSION)")
    parser.add_argument("--scope-walk-edge", type=float, metavar="F",
                        help="Scope stochastic: optional edge probability. "
                             "Default 0 matches Osci-render")
    parser.add_argument("--scope-walk-hz", type=float, metavar="HZ",
                        help="Scope stochastic target clock (default 48000). "
                             "Independent of the image rate and of faster DACs")
    parser.add_argument("--scope-stipple-points", type=int, metavar="N",
                        help="Scope stipple: stable luminance-weighted image "
                             "positions before proximity ordering (default 768)")
    parser.add_argument("--scope-physical-dwell", type=float, metavar="F",
                        help="Redistribute fixed physical XY sample budget toward drawing segments (0=off, 1=strong; does not change preview exposure)")
    parser.add_argument("--scope-rows", type=int)
    parser.add_argument("--scope-row-bias", type=float, metavar="F",
                        help="Trade columns for rows at constant cell count. "
                             "Face features are mostly horizontal edges "
                             "(eyelids, lip line, brow) and horizontal edges "
                             "are resolved by rows, which the MTF says is also "
                             "the cheap axis. On a real face 1.3 reads visibly "
                             "sharper than the default 1.0. Past ~1.6 the "
                             "mouth smears and the silhouette blocks up.")
    parser.add_argument("--scope-border", type=float, metavar="F",
                        help="Draw a fixed rectangle at the full extent in "
                             "raster, stochastic, stipple, and fusion, "
                             "spending F of the trace's samples on it (try "
                             "0.03). Without it the drawn "
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
    parser.add_argument("--scope-list-from-images",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Build the folder manifest by scanning the images "
                             "instead of reading it from the bake. Only needed "
                             "for a bake made before manifests were supported.")
    parser.add_argument("--scope-autofit", dest="scope_autofit",
                        action="store_true", default=None,
                        help="Scope raster: size the grid against cells that "
                             "survive trim (default on)")
    parser.add_argument("--scope-no-autofit", "--no-scope-autofit",
                        dest="scope_autofit", action="store_false", default=None,
                        help="Scope raster: size the grid to the whole frame "
                             "instead of to the cells that survive trim "
                             "(autofit is on by default and is usually a 2x "
                             "resolution win on a dark background)")
    parser.add_argument("--no-scope-fps", dest="scope_clear_fps",
                        action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-scope-samples", dest="scope_clear_samples",
                        action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-scope-rows", dest="scope_clear_rows",
                        action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-scope-dc-comp", dest="scope_clear_dc_comp",
                        action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-scope-lowpass", dest="scope_clear_lowpass",
                        action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--scope-mix", nargs="?", type=float, const=120.0,
                        metavar="HZ",
                        help="Scope: whole-trace mix at this rate (default "
                             "120): vector -> raster -> stochastic -> raster "
                             "-> stipple -> raster. Above flicker fusion the "
                             "phosphor sums all four renderers.")
    parser.add_argument("--no-scope-mix", action="store_true",
                        help="Disable a mix rate configured in settings.py")
    parser.add_argument("--scope-mix-duty", type=float, default=None,
                        help="Scope: fraction of mixed passes spent on raster "
                             "(the remainder is split equally between vector, "
                             "stochastic, and stipple)")
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
        "--test",
        action="store_true",
        help="Override hosts to '0.0.0.0' for network testing (default: '127.0.0.1')"
    )

    # Modem is independent of the XY waveform and the video/GL display stack.
    parser.add_argument("--modem-source", choices=("bake", "images"),
                        default="bake",
                        help="V7 source: prebaked slabs or normal images "
                             "encoded at runtime (default: bake)")
    parser.add_argument("--modem-dir", help="RGBA modem bake from utilities/convert_to_modem_dct.py")
    parser.add_argument("--modem-channels", default="1,2", help="1-based stereo output pair")
    parser.add_argument("--modem-latency", default="low", help="low, high, or seconds")
    parser.add_argument("--modem-time-offset-ms", type=float, default=0.0,
                        help="Additional receiver allowance; scheduled transmission is already compensated")
    parser.add_argument("--modem-receive-margin-ms", type=float, default=15.0,
                        help="Time after packet completion reserved for input/decode/GUI (default 15)")
    parser.add_argument("--modem-prepare-ms", type=float, default=25.0,
                        help="Minimum encoding lead before a send deadline (default 25)")
    parser.add_argument("--modem-speed", "--speed", dest="modem_speed",
                        type=float, default=1.0,
                        help="V7 playback speed 0.25..4x (default 1; 1x is tape-safe). "
                             "Above the output Nyquist limit, high-frequency detail is lost")
    parser.add_argument("--modem-pilot-tones",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="V7 bin-1/bin-3 timing references (required for Fold 500)")
    parser.add_argument("--modem-eof-marker",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="V7 packet EOF marker (required for Fold 500)")
    parser.add_argument("--modem-encode-filter",
                        choices=("nearest", "box"),
                        default=None,
                        help="V7 source resize filter (default: box; stereo Fold 500 "
                             "requires box)")
    modem_length = parser.add_mutually_exclusive_group()
    modem_length.add_argument(
        "--modem-frames", type=int, default=0,
        help="Exact packet count; 0 = unlimited live / one source pass for WAV")
    modem_length.add_argument(
        "--modem-cycles", type=int, default=None,
        help="Full application-clock cycles for WAV export; ping-pong cycles "
             "include the return trip")
    parser.add_argument("--modem-wav", help="Export a deterministic pair to PCM16 WAV instead of live playback")
    parser.add_argument("--modem-pair", help="Fixed zero-based face,float pair for inspection, e.g. 1,0")
    parser.add_argument("--modem-index-offset-ms", type=float, default=0.0,
        help="Shift only which image the clock returns, not when it is shown; "
             "use to align modem output with local/scope/ascii. One image is 1000/IPS ms.")
    parser.add_argument("--modem-log-frames", action="store_true")
    parser.add_argument("--modem-clock", type=int, choices=[0,1,2,3,255], default=settings.CLOCK_MODE)
    parser.add_argument("--modem-frame-duration", type=float, default=1.0,
                        help="Existing MIDI clock's frame scaling factor")
    args = parser.parse_args()
    if args.local_frame_port is not None:
        if args.mode != "local":
            parser.error("--local-frame-port can only be used with --mode local")
        if not 1 <= args.local_frame_port <= 65535:
            parser.error("--local-frame-port must be between 1 and 65535")
    if args.local_frame_fps is not None:
        if args.local_frame_port is None:
            parser.error("--local-frame-fps requires --local-frame-port")
        if (not math.isfinite(args.local_frame_fps) or
                not 0 < args.local_frame_fps <= 1000):
            parser.error("--local-frame-fps must be greater than 0 and at most 1000")
    settings.LOCAL_FRAME_PORT = args.local_frame_port
    settings.LOCAL_FRAME_FPS = (args.local_frame_fps or
                                getattr(settings, "FPS", 30))
    _set_process_title(args.mode)

    if args.mode == "modem":
        if args.modem_encode_filter is None:
            args.modem_encode_filter = "box"
        if args.modem_encode_filter != "box":
            parser.error("stereo Fold 500 requires --modem-encode-filter box")
        if not args.modem_pilot_tones:
            parser.error("stereo Fold 500 requires coded pilot tones")
        if not args.modem_eof_marker:
            parser.error("stereo Fold 500 requires EOF markers")
        if args.modem_frames < 0:
            parser.error("Modem frame count must be nonnegative")
        if args.modem_cycles is not None and args.modem_cycles <= 0:
            parser.error("Modem cycle count must be positive")
        if args.modem_cycles is not None and not args.modem_wav:
            parser.error("--modem-cycles requires --modem-wav")
        if not math.isfinite(args.modem_time_offset_ms):
            parser.error("--modem-time-offset-ms must be finite")
        if (not math.isfinite(args.modem_receive_margin_ms) or args.modem_receive_margin_ms < 0
                or not math.isfinite(args.modem_prepare_ms) or args.modem_prepare_ms <= 0
                or not 0 <= args.modem_receive_margin_ms + args.modem_time_offset_ms <= 2000):
            parser.error("Modem prepare time must be positive and total receiver allowance must be 0..2000 ms")
        if not math.isfinite(args.modem_index_offset_ms) or abs(args.modem_index_offset_ms) > 10000:
            parser.error("--modem-index-offset-ms must be finite and within +/-10000")
        if not math.isfinite(args.modem_frame_duration) or args.modem_frame_duration <= 0:
            parser.error("--modem-frame-duration must be finite and positive")
        if (not math.isfinite(args.modem_speed) or
                not 0.25 <= args.modem_speed <= 4.0):
            parser.error("--modem-speed must be finite and between 0.25 and 4")
        if args.modem_clock == settings.CLIENT_MODE:
            parser.error("--modem-clock 3 (CLIENT_MODE) is not supported by the V7 modem")
        if args.modem_wav and args.modem_clock != settings.FREE_CLOCK:
            parser.error("--modem-wav requires the free clock (--modem-clock 255); "
                         "a live MIDI clock cannot drive a deterministic WAV")
        if args.scope_ask:
            parser.error("V7 modem mode takes --device explicitly; use "
                         "--scope-device with the selected audio device")
        # --dir names the directory to use, in this mode as in every other.
        # It is not a source tree to append a suffix to; only the fallback,
        # when no directory is given at all, derives one from IMAGES_DIR.
        if args.dir:
            settings.IMAGES_DIR = os.path.abspath(args.dir)
        if args.modem_source == "images":
            if args.modem_dir:
                parser.error("--modem-dir cannot be combined with --modem-source images")
            settings.MAIN_FOLDER_PATH = os.path.join(settings.IMAGES_DIR, "face")
            settings.FLOAT_FOLDER_PATH = os.path.join(settings.IMAGES_DIR, "float")
            source_name = os.path.basename(os.path.normpath(settings.IMAGES_DIR)).replace(" ", "_")
            suffix = f"{source_name}_modem"
            settings.PROCESSED_DIR = os.path.join(CACHE_DIR, f"folders_processed_{suffix}")
            settings.GENERATED_LISTS_DIR = os.path.join(CACHE_DIR, f"generated_lists_{suffix}")
            os.makedirs(LOGS_DIR, exist_ok=True)
            settings.CLOCK_MODE = args.modem_clock
            return args, os.path.join(LOGS_DIR, "runtime_modem.log")
        if not args.modem_dir:
            args.modem_dir = (os.path.abspath(args.dir) if args.dir
                              else settings.IMAGES_DIR + "_modem")
        args.modem_dir = os.path.abspath(args.modem_dir)
        if not os.path.isdir(args.modem_dir):
            parser.error(f"Directory not found: {args.modem_dir}")
        if not os.path.isfile(os.path.join(args.modem_dir, "modem.json")):
            parser.error(f"No modem.json in {args.modem_dir}; "
                         "run utilities/convert_to_modem_dct.py first")
        settings.CLOCK_MODE = args.modem_clock
        os.makedirs(LOGS_DIR, exist_ok=True)
        # Baked-only mode: no image scan, cache removal, ports, or GL setup.
        return args, os.path.join(LOGS_DIR, "runtime_modem.log")

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

    # Orientation, before anything reads it.  display_manager binds these at
    # import time and scope_display reads them when the engine starts, so both
    # have to be settled here rather than inside a per-mode branch.
    if args.rotation is not None:
        settings.INITIAL_ROTATION = int(args.rotation) % 360
    if args.mirror is not None:
        settings.INITIAL_MIRROR = 1 if args.mirror else 0

    # ASCII grading, before the converter builds anything from it.  Each is
    # rejected here rather than left to misbehave later: the failure modes are
    # specific, so the messages are too.
    #
    #   contrast   negative scales the picture through mid-grey and comes back
    #              INVERTED -- a tone inversion, not a contrast, and far more
    #              likely a stray minus sign than a request.  0 is the honest
    #              endpoint: everything flattens to one tone.
    #   brightness a straight multiply, so 0 is black and negative is
    #              meaningless -- it clips to black anyway, silently.
    #   gamma      an exponent on i/255.  0 makes every non-zero input 1.0
    #              (a flat white field) and negative divides by zero at i=0,
    #              which overflows to 255 rather than raising.
    for flag, value, floor, closed in (
            ("--ascii-contrast", args.ascii_contrast, 0.0, True),
            ("--ascii-brightness", args.ascii_brightness, 0.0, True),
            ("--ascii-gamma", args.ascii_gamma, 0.0, False)):
        if value is None:
            continue
        if not math.isfinite(value):
            parser.error(f"{flag} must be a finite number")
        if value < floor or (not closed and value == floor):
            parser.error(f"{flag} must be greater than {floor:g}"
                         if not closed else
                         f"{flag} cannot be negative "
                         f"({floor:g} is the low end, 1.0 is neutral)")
    if args.ascii_contrast is not None:
        settings.ASCII_CONTRAST = args.ascii_contrast
    if args.ascii_brightness is not None:
        settings.ASCII_BRIGHTNESS = args.ascii_brightness
    if args.ascii_gamma is not None:
        settings.ASCII_GAMMA = args.ascii_gamma

    # Options that only mean anything in one mode; say so rather than silently
    # ignoring them. The whole point of a knob is that turning it does
    # something, and a flag accepted in the wrong mode does nothing quietly.
    if args.mode not in ("ascii", "asciiweb"):
        _ascii_used = [f for f, v in (("--ascii-contrast", args.ascii_contrast),
                                      ("--ascii-brightness", args.ascii_brightness),
                                      ("--ascii-gamma", args.ascii_gamma))
                       if v is not None]
        if _ascii_used:
            print(f"⚠️  {', '.join(_ascii_used)} ignored: these apply to "
                  f"--mode ascii and asciiweb. Each mode in VideoInterleaving "
                  f"runs standalone.")

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
        # Scope either reads the bake manifest or validates its image-list
        # cache itself. Deleting those directories here forced a full cache
        # restore on every launch and made separate scope instances interfere.
        instance_pattern = None
    elif args.mode in ("web", "local"):
        instance_pattern = f"_{args.mode}_"  # Match any port
    else:
        # ASCII modes: match specific port
        instance_pattern = f"_{args.mode}_{primary_port}"

    # Clean up existing cache directories for this instance
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_base = os.path.join(script_dir, CACHE_DIR)

    if instance_pattern is not None and os.path.exists(cache_base):
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
        if args.local_frame_port is None:
            print(f">> PORTS: Monitor={ports.monitor}")
            require_ports(ports.get_all_ports())
        else:
            print(">> LOCAL VIDEO SOURCE: HTTP monitor disabled")
        # Update settings for backward compatibility
        settings.WEB_PORT = ports.monitor
        # Local mode always uses --test flag for network monitoring
        args.test = True
        print(">> Local mode: Enabling --test flag for network monitoring")

    elif args.mode == "scope":
        scope_source = (args.scope_source or
                        getattr(settings, "SCOPE_SOURCE", "bake"))
        if scope_source not in ("bake", "images"):
            parser.error("--scope-source must be bake or images")
        settings.SCOPE_SOURCE = scope_source
        if args.scope_gui is not None:
            settings.SCOPE_GUI = args.scope_gui
        start_image_only = (getattr(settings, "SCOPE_GUI_IMAGE_ONLY", False)
                            if args.scope_gui_image_only is None else
                            args.scope_gui_image_only)
        start_fullscreen = (getattr(settings, "SCOPE_GUI_FULLSCREEN", False)
                            if args.scope_gui_fullscreen is None else
                            args.scope_gui_fullscreen)
        start_image_only = bool(start_image_only or start_fullscreen)
        if ((start_image_only or start_fullscreen) and args.scope_gui is False):
            parser.error("scope GUI startup views cannot be combined with --no-scope-gui")
        settings.SCOPE_GUI_IMAGE_ONLY = start_image_only
        settings.SCOPE_GUI_FULLSCREEN = bool(start_fullscreen)
        if start_image_only or start_fullscreen:
            settings.SCOPE_GUI = True
        if args.scope_live_size is not None:
            if args.scope_live_size < 16:
                parser.error("--scope-live-size must be at least 16 pixels")
            settings.SCOPE_LIVE_SIZE = args.scope_live_size
        try:
            settings.SCOPE_CHANNELS = _parse_scope_channels(
                args.scope_channels if args.scope_channels is not None
                else getattr(settings, "SCOPE_CHANNELS", (1, 2)))
        except ValueError as exc:
            parser.error(str(exc))

        render_was_selected = bool(
            args.scope_mode or args.scope_raster or args.scope_stochastic
            or args.scope_stipple)
        if args.scope_mode:
            requested_render = args.scope_mode
        elif args.scope_raster:
            requested_render = "raster"
        elif args.scope_stochastic:
            requested_render = "stochastic"
        elif args.scope_stipple:
            requested_render = "stipple"
        else:
            requested_render = getattr(settings, "SCOPE_RENDER_MODE", "vector")
        if scope_source == "images":
            if args.scope_list_from_images:
                parser.error("--scope-list-from-images is redundant with "
                             "--scope-source images")
            if not render_was_selected:
                requested_render = "raster"
            if requested_render not in ("raster", "stochastic", "stipple"):
                parser.error("--scope-source images supports raster, stochastic, "
                             "or stipple rendering; vector/fusion require a bake")
            settings.SCOPE_RENDER_MODE = requested_render
            if args.scope_mix is not None:
                parser.error("--scope-mix requires baked vector geometry and "
                             "cannot be used with --scope-source images")

        geometry_samples = args.scope_geometry_samples
        traversal_hz = args.scope_traversal_hz
        if geometry_samples is not None or traversal_hz is not None:
            if scope_source != "bake" or requested_render not in ("vector", "raster", "stipple"):
                parser.error("scope geometry/traversal controls require baked vector, raster, or stipple rendering")
            if args.scope_mix is not None:
                parser.error("scope geometry/traversal controls cannot be combined with --scope-mix")
            if args.scope_realtime:
                parser.error("scope geometry/traversal controls cannot be combined with --scope-realtime")
            if geometry_samples is not None and geometry_samples < 2:
                parser.error("--scope-geometry-samples must be at least 2")
            if traversal_hz is not None and (not math.isfinite(traversal_hz) or traversal_hz <= 0):
                parser.error("--scope-traversal-hz must be finite and greater than zero")
            if traversal_hz is not None and args.scope_trigger is False:
                parser.error("--scope-traversal-hz requires the scope trigger; remove --no-scope-trigger")
            if traversal_hz is not None and args.scope_fields not in (None, 1):
                parser.error("--scope-traversal-hz requires one field")
            if args.scope_yt_timing == "fixed" or args.scope_yt:
                parser.error("independent geometry controls cannot be combined with fixed Y-T timing")

        if (args.scope_mix_duty is not None
                and (not math.isfinite(args.scope_mix_duty)
                     or not 0.0 <= args.scope_mix_duty <= 1.0)):
            parser.error("--scope-mix-duty must be between 0 and 1")
        if args.scope_mix is not None:
            if not math.isfinite(args.scope_mix) or args.scope_mix <= 0:
                parser.error("--scope-mix must be a finite rate greater than zero")
            if args.scope_samples is not None:
                parser.error("--scope-samples cannot be combined with "
                             "--scope-mix; the mix rate is the trace clock")
            if args.scope_mode == "fusion":
                parser.error("--scope-mode fusion and --scope-mix are "
                             "alternative combiners; choose one")
        if args.scope_mix is not None and args.no_scope_mix:
            parser.error("--scope-mix and --no-scope-mix cannot be combined")
        if args.port:
            print("⚠️  WARNING: --port ignored in SCOPE mode.")
        if args.scope_x_only is not None:
            settings.SCOPE_X_ONLY = args.scope_x_only
        _scope_signal = "X-only mono" if settings.SCOPE_X_ONLY else "XY stereo"
        print(f">> MODE: SCOPE ({_scope_signal}) [{source_name}]")
        settings.ASCII_MODE = False
        settings.SERVER_MODE = False
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
        if args.scope_mode:
            settings.SCOPE_RENDER_MODE = args.scope_mode
        elif args.scope_raster:
            settings.SCOPE_RENDER_MODE = "raster"
        elif args.scope_stochastic:
            settings.SCOPE_RENDER_MODE = "stochastic"
        elif args.scope_stipple:
            settings.SCOPE_RENDER_MODE = "stipple"
        # Compatibility for callers outside main.py that still inspect this.
        settings.SCOPE_RASTER = settings.SCOPE_RENDER_MODE == "raster"
        if args.scope_invert is not None:
            settings.SCOPE_INVERT = args.scope_invert
        # The marker used to be a MODE, and a mode has to reject every
        # combination it cannot represent -- five parser.error()s that made
        # Y-T an either/or against realtime, mix, and every renderer but
        # raster. It is a property of the output now: it is stamped on the
        # finished frame in Scope.show_frame, which every renderer already
        # passes through, so there is nothing left to be incompatible with.
        if args.scope_trigger is not None:
            settings.SCOPE_TRIGGER = args.scope_trigger
        if args.scope_trigger_shape is not None:
            settings.SCOPE_TRIGGER_SHAPE = args.scope_trigger_shape
        if args.scope_trigger_us is not None:
            if (not math.isfinite(args.scope_trigger_us)
                    or args.scope_trigger_us <= 0):
                parser.error("--scope-trigger-us must be finite and greater "
                             "than zero")
            settings.SCOPE_TRIGGER_US = args.scope_trigger_us
            settings.SCOPE_YT_TRIGGER_US = None   # the CLI wins over the alias
        if args.scope_yt_timing is not None:
            settings.SCOPE_YT_TIMING = args.scope_yt_timing
        elif args.scope_yt:
            # The flag's remaining meaning. Everything else it used to imply
            # -- raster, retrace, the marker -- is either the default now or
            # arranged by the timing itself.
            settings.SCOPE_YT_TIMING = "fixed"
            print("[SCOPE] --scope-yt is now --scope-yt-timing fixed; the "
                  "trigger marker is on by default in every mode.")
        if args.scope_realtime is not None:
            settings.SCOPE_REALTIME = args.scope_realtime
        if args.scope_fps is not None:
            settings.SCOPE_FPS = args.scope_fps
        if args.scope_clear_fps:
            settings.SCOPE_FPS = None
        if getattr(args, "scope_fields", None) is not None:
            settings.SCOPE_FIELDS = args.scope_fields
            settings.SCOPE_FIELDS_EXPLICIT = True
        if args.scope_dc_comp is not None:
            settings.SCOPE_DC_COMP = args.scope_dc_comp
        if args.scope_clear_dc_comp:
            settings.SCOPE_DC_COMP = None
        if getattr(args, "scope_border", None) is not None:
            settings.SCOPE_BORDER = args.scope_border
        if getattr(args, "scope_row_bias", None):
            settings.SCOPE_ROW_BIAS = args.scope_row_bias
        if args.scope_samples is not None:
            settings.SCOPE_SAMPLES = args.scope_samples
        if geometry_samples is not None:
            settings.SCOPE_GEOMETRY_SAMPLES = geometry_samples
        if traversal_hz is not None:
            settings.SCOPE_TRAVERSAL_HZ = traversal_hz
        if args.scope_clear_samples:
            settings.SCOPE_SAMPLES = None
        if args.scope_trim is not None:
            settings.SCOPE_TRIM = args.scope_trim
        if args.scope_gamma is not None:
            # One public gamma follows the selected renderer. Mix contains
            # both luminance renderers, so it deliberately sets both; the
            # stochastic-specific flag below can then override just its side.
            if (args.scope_mix is not None
                    or settings.SCOPE_RENDER_MODE == "fusion"):
                settings.SCOPE_GAMMA = args.scope_gamma
                settings.SCOPE_WALK_GAMMA = args.scope_gamma
            elif settings.SCOPE_RENDER_MODE in ("stochastic", "stipple"):
                settings.SCOPE_WALK_GAMMA = args.scope_gamma
            else:
                settings.SCOPE_GAMMA = args.scope_gamma
        if args.scope_density is not None:
            settings.SCOPE_DENSITY = args.scope_density
        if args.scope_precondition is not None:
            if args.scope_precondition < 0:
                parser.error("--scope-precondition must be non-negative")
            settings.SCOPE_PRECONDITION = args.scope_precondition
        if args.scope_walk_radius is not None:
            settings.SCOPE_WALK_RADIUS = args.scope_walk_radius
        if args.scope_walk_stride is not None:
            settings.SCOPE_WALK_STRIDE = args.scope_walk_stride
        if args.scope_walk_reseed_ms is not None:
            settings.SCOPE_WALK_RESEED_MS = args.scope_walk_reseed_ms
        if args.scope_walk_gamma is not None:
            settings.SCOPE_WALK_GAMMA = args.scope_walk_gamma
        if args.scope_fusion is not None:
            settings.SCOPE_FUSION = args.scope_fusion
        if args.scope_walk_edge is not None:
            settings.SCOPE_WALK_EDGE = args.scope_walk_edge
        if args.scope_walk_hz is not None:
            settings.SCOPE_WALK_HZ = args.scope_walk_hz
        if args.scope_stipple_points is not None:
            if args.scope_stipple_points < 8:
                parser.error("--scope-stipple-points must be at least 8")
            settings.SCOPE_STIPPLE_POINTS = args.scope_stipple_points
        if args.scope_physical_dwell is not None:
            if not math.isfinite(args.scope_physical_dwell) or not 0.0 <= args.scope_physical_dwell <= 1.0:
                parser.error("--scope-physical-dwell must be between 0 and 1")
            settings.SCOPE_PHYSICAL_DWELL = args.scope_physical_dwell
        if args.scope_rows is not None:
            settings.SCOPE_ROWS = args.scope_rows
        if args.scope_clear_rows:
            settings.SCOPE_ROWS = None
        if args.scope_autofit is not None:
            settings.SCOPE_AUTOFIT = args.scope_autofit
        if args.scope_list_from_images is not None:
            settings.SCOPE_LIST_FROM_IMAGES = args.scope_list_from_images
        if args.scope_lowpass is not None:
            settings.SCOPE_LOWPASS = args.scope_lowpass
        if args.scope_clear_lowpass:
            settings.SCOPE_LOWPASS = None
        if args.scope_oversample:
            settings.SCOPE_OVERSAMPLE = args.scope_oversample
        if args.scope_min_feature is not None:
            settings.SCOPE_MIN_FEATURE = args.scope_min_feature
        if args.scope_sweep:
            settings.SCOPE_SWEEP = args.scope_sweep
        if args.scope_mix is not None:
            settings.SCOPE_MIX = args.scope_mix
        elif args.no_scope_mix:
            settings.SCOPE_MIX = None
        if args.scope_mix_duty is not None:
            settings.SCOPE_MIX_DUTY = args.scope_mix_duty
        # Fixed row timing lives inside render_luma, so it is genuinely raster
        # only -- but that is a reason to say so and fall back, not to refuse
        # to start. scope_display does the falling back, with the warning.
        # Resolve the audio device NOW, before file lists are built and before
        # stdout is wrapped. Prompting from deep inside run_scope meant the
        # question appeared after a long silence, so it read as a hang.
        _configured_device = (args.scope_device if args.scope_device is not None
                              else getattr(settings, "SCOPE_DEVICE", None))
        _configured_ask = bool(
            args.scope_ask or getattr(settings, "SCOPE_ASK", False))
        if _configured_ask or _configured_device is not None:
            from scope_out import choose_device as _choose, scrub as _scrub
            # argv is decoded with surrogateescape, so a stray byte in shell
            # history arrives as a lone surrogate and breaks any later encode
            _configured_device = _scrub(_configured_device)
            _minimum_channels = (
                settings.SCOPE_CHANNELS[0] if settings.SCOPE_X_ONLY else
                max(settings.SCOPE_CHANNELS))
            settings.SCOPE_DEVICE = _choose(
                ask=_configured_ask, device=_configured_device,
                min_channels=_minimum_channels)
            if settings.SCOPE_DEVICE == "null":
                _name = "none (browser renders)"
            else:
              try:
                # NOT a top-level import. sounddevice runs Pa_Initialize() on
                # import, which throws PortAudioError -- not OSError -- on a
                # host with no sound server. Catch broadly.
                import sounddevice as _sd
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
        _scope_axes = (settings.SCOPE_CHANNELS[:1] if settings.SCOPE_X_ONLY
                       else settings.SCOPE_CHANNELS)
        print(">> AUDIO CHANNELS: " + ",".join(map(str, _scope_axes)) +
              " (PortAudio, 1-based)")
        if scope_source == "images":
            print(f">> SCOPE SOURCE: live images ({settings.IMAGES_DIR})")
        else:
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
from settings import CLOCK_MODE

# Everything a MODE needs is imported by that mode, at the point of use.
#
# These used to be module-level, which meant every mode paid for every other
# mode's dependencies: scope mode -- which needs neither TurboJPEG nor GL nor a
# websocket library -- could not start on a box that was missing any of them.
# image_display was already wrapped in try/except for exactly this reason, but
# that only deferred the FAILURE; the import still ran and its stack still had
# to be installed. ascii_web_server had no such wrapper at all, so a missing
# SimpleWebSocketServer stopped scope, local and web modes dead.
#
# The mapping, so the branches below are not the only record of it:
#
#   make_file_lists     every mode except scope reading a baked manifest
#   image_display       local, web, ascii, asciiweb   (turbojpeg, glfw, moderngl)
#   web_service         local, web, asciiweb, scope   (stdlib only)
#   ascii_server        ascii                         (stdlib only)
#   ascii_stats_server  ascii                         (stdlib only)
#   ascii_web_server    asciiweb                      (SimpleWebSocketServer)
#   scope_display       scope                         (numpy, sounddevice; optional GL GUI)
#
# Python caches modules, so importing inside a branch costs nothing on repeat.


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
    if cli_args.mode == "modem":
        try:
            from modem_v7_display import run_modem
            run_modem(cli_args)
        except KeyboardInterrupt:
            print("\n[MODEM] Shutdown requested")
        except Exception as exc:
            print(f"[MODEM] {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = _original_stdout
            sys.stderr = _original_stderr
            if _log_file is not None:
                _log_file.close()
        return
    # display_manager imports ModernGL and renderer; scope is audio-only and
    # must not import the graphics stack on its way into the scope engine.
    if cli_args.mode != "scope":
        try:
            from display_manager import _restore_display_resolution
            atexit.register(_restore_display_resolution)
        except ImportError:
            pass  # display_manager may not be imported yet

    # 1. Process Files (fresh on each startup)

    print(f"[MAIN] Mode={cli_args.mode} | Images={settings.IMAGES_DIR} | Cache={settings.GENERATED_LISTS_DIR}")

    if cli_args.mode == "scope":
        # The scope engine probes the bake once and initializes image lists
        # only for a live source or a bake without a usable manifest.
        print(f">> Scope source: {getattr(settings, 'SCOPE_SOURCE', 'bake')}")
    else:
        print(">> Building file lists...")
        import make_file_lists          # PIL + numpy; not needed off this path
        make_file_lists.process_files()

    # 2. Launch Servers
    mode = cli_args.mode

    if mode == "ascii":
        import ascii_server
        import ascii_stats_server
        threading.Thread(target=ascii_server.start_server, daemon=True, name="ASCII-Telnet").start()
        threading.Thread(target=ascii_stats_server.start_server, daemon=True, name="ASCII-Stats").start()

    elif mode == "asciiweb":
        import ascii_web_server          # SimpleWebSocketServer lives behind this
        import web_service
        threading.Thread(target=ascii_web_server.start_server, daemon=True, name="ASCII-WS").start()
        web_service.start_server(monitor=True, stream=False)

    elif mode == "web":
        import web_service
        web_service.start_server(monitor=True, stream=True)

    elif mode == "local":
        if cli_args.local_frame_port is None:
            import web_service
            web_service.start_server(monitor=True, stream=False)

    elif mode == "scope":
        # Keep the web monitor available whether the optional native scope
        # window is enabled or the process is running headless.
        #
        # NOT gated on require_ports(): run_monitor_server() runs in a daemon
        # thread and its OSError on a busy port dies in that thread.  The audio
        # keeps going.  A monitor port held by another instance must never take
        # the installation off the air.
        import web_service
        web_service.start_server(monitor=True, stream=False)

        # 3. Start Display Engine
    failed = False
    try:
        if mode == "scope":
            # Graphics stay optional and are imported by scope_display only
            # when --scope-gui is enabled. Scope never loads the video path.
            import scope_display
            scope_display.run_scope(clock)
        else:
            # TurboJPEG, glfw and moderngl are reached through this, and only
            # this. Say which mode wanted them, because "No module named
            # 'turbojpeg'" from a bare traceback does not.
            try:
                import image_display
            except Exception as e:
                raise RuntimeError(
                    f"{mode} mode needs the video display stack "
                    f"(TurboJPEG / GL) and it could not be loaded: {e}. "
                    f"Scope mode does not need it and still runs."
                ) from e
            image_display.run_display(clock)
    except KeyboardInterrupt:
        print("\n[MAIN] Shutdown requested via Ctrl+C")
    except Exception as e:
        failed = True
        print(f"\n[MAIN] CRASH DETAILS: {e}")
        traceback.print_exc()
    finally:
        print("[MAIN] Exiting...")
        # Restore display resolution only for modes that can use the window
        # manager. Scope must stay independent of graphics libraries.
        if mode != "scope":
            try:
                from display_manager import _restore_display_resolution
                _restore_display_resolution()
            except Exception:
                pass  # Don't fail if restoration fails
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
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
