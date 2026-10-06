"""
scope_display.py -- the scope-mode engine.

A standalone mode, like ascii / asciiweb / local / web: selected once and it
owns the run. Renders the interleaved composition as XY vectors, dwell raster,
an Osci-style stochastic luminance walk, a stable weighted stipple route, or a
    per-index multiplexer across complete XY position arrays on the audio output,
    driven by the same image-index path and folder selector as every other mode.
    Normal startup currently uses the free-running clock; it does not initialize
    the retained legacy MIDI clock path.

Baked playback needs none of the image machinery: no ImageLoader, no FIFO, no
TurboJPEG, and no GL context. Optional ``--scope-source images`` decodes small
runtime thumbnails for raster/stochastic/stipple playback. Vector geometry
still comes from libraries baked offline by utilities/convert_to_xy.py.

    python main.py --mode scope --dir images --xy-dir images_xy --scope-raster

or directly, using the settings.py defaults:

    python scope_display.py [--dir images] [--xy-dir images_xy]

All tuning is read from settings.SCOPE_*; main.py publishes CLI overrides
there, exactly as the other modes read ASCII_MODE / SERVER_MODE.
"""
import argparse
import inspect
import math
import os
import threading
import time
from pathlib import Path
from types import MappingProxyType

import numpy as np

import settings
import scope_out as _scope_out
from scope_frame_scheduler import FieldGroupLatch, ScopeFrameScheduler
from scope_prepared_cache import PreparedImageCache

_REQUIRED_SCOPE_OUT_API = 7
_scope_out_api = getattr(_scope_out, "SCOPE_OUT_API_VERSION", 0)
_scope_signature = inspect.signature(_scope_out.Scope.__init__)
if (_scope_out_api != _REQUIRED_SCOPE_OUT_API or
        "rotation" not in _scope_signature.parameters or
        "mirror" not in _scope_signature.parameters or
        "trigger_shape" not in _scope_signature.parameters or
        "x_only" not in _scope_signature.parameters or
        "channel_pair" not in _scope_signature.parameters):
    raise RuntimeError(
        "scope_display.py and scope_out.py are from different revisions. "
        f"Loaded scope_out from {_scope_out.__file__!r}; "
        f"API={_scope_out_api}, constructor={_scope_signature}. "
        "Replace scope_out.py with the rotation/mirror/trigger/channel-pair file "
        "from the same runtime bundle as scope_display.py."
    )

from time import monotonic as _time_mono
from scope_out import (Scope, choose_device, BufferedSource, rasterize,
                       precompensate_hpf, rotate_frame, parse_channel_pair,
                       required_output_channels)
from scope_bake import (XYLibrary, merge, SweepSource, calibrate,
                        composite_luma, composite_stipple_candidates,
                        TraceEmitter, StochasticEmitter,
                        StippleEmitter,
                        TriangleMixScheduler, PositionMultiplexer,
                        normalize_fusion_components, apply_trace_border,
                        trace_luminance_weights, retime_trace_by_weights)
try:
    from scope_lowpass import lowpass_circular
except Exception:            # optional tool; absence must not break the mode
    lowpass_circular = None


def _quarter_turn(value):
    """Return a validated clockwise-control quarter turn in degrees."""
    angle = int(value) % 360
    if angle % 90:
        raise ValueError("scope rotation must be a multiple of 90 degrees")
    return angle


def _rotate_luma(lum, rotation):
    """Rotate image content before trace generation.

    Raster timing depends on X remaining the fast sweep axis. Rotating the
    completed XY trace would swap that carrier onto Y at 90/270 degrees;
    rotating the luminance first lets the renderer build horizontal rows for
    the new orientation instead.
    """
    if lum is None:
        return None
    k = _quarter_turn(rotation) // 90
    src = np.asarray(lum)
    return np.ascontiguousarray(np.rot90(src, k=k)) if k else src


def _rotate_stipple_cloud(cloud, rotation):
    """Rotate normalized baked stipple candidates in image space."""
    if cloud is None:
        return None
    k = _quarter_turn(rotation) // 90
    if not k:
        return cloud
    out = dict(cloud)
    xy = np.asarray(cloud["xy"], dtype=np.float64)
    u, v = xy[:, 0], xy[:, 1]
    if k == 1:
        out["xy"] = np.column_stack((v, 1.0 - u))
    elif k == 2:
        out["xy"] = np.column_stack((1.0 - u, 1.0 - v))
    else:
        out["xy"] = np.column_stack((1.0 - v, u))
    if k % 2:
        out["aspect"] = 1.0 / max(float(cloud.get("aspect", 1.0)), 1e-9)
    return out


def _rotation_grid(cal, rotation):
    """Return calibrated (rows, columns) for the oriented image."""
    if not cal:
        return None
    rows, cols = int(cal["grid_rows"]), int(cal["grid_cols"])
    return (cols, rows) if (_quarter_turn(rotation) // 90) % 2 else (rows, cols)


# ---------------------------------------------------------------- bootstrap

def _bootstrap():
    """Only used when running this file directly: does what main.py's
    configure_runtime() would otherwise do."""
    ap = argparse.ArgumentParser(description="Scope mode (XY audio output)")
    ap.add_argument("--dir", help="image source folder. Optional: scope mode "
                    "reads its manifest from the bake and never opens an image.")
    ap.add_argument("--xy-dir", help="baked XY libraries")
    ap.add_argument("--scope-source", choices=("bake", "images"),
                    default=None,
                    help="baked XY or runtime-decoded images (default: bake)")
    ap.add_argument("--scope-channels", metavar="X,Y",
                    help="1-based PortAudio output channels, e.g. 18,19")
    ap.add_argument("--scope-live-size", type=int, metavar="PX")
    ap.add_argument("--scope-gui", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="open the optional native scope preview and tuner")
    ap.add_argument("--scope-gui-image-only", "--image-only",
                    dest="scope_gui_image_only",
                    action=argparse.BooleanOptionalAction, default=None,
                    help="start the scope GUI with only the preview; click to restore controls")
    ap.add_argument("--scope-gui-fullscreen", "--fullscreen",
                    dest="scope_gui_fullscreen",
                    action=argparse.BooleanOptionalAction, default=None,
                    help="start the scope GUI fullscreen in image-only view")
    ap.add_argument("--scope-fps", type=int)
    ap.add_argument("--scope-samples", type=int)
    ap.add_argument("--device", "--scope-device", dest="scope_device",
                    help="audio output device name fragment or output index")
    ap.add_argument("--ask", "--scope-ask", dest="scope_ask",
                    action="store_true", help="choose an output interactively")
    render = ap.add_mutually_exclusive_group()
    render.add_argument("--scope-mode",
                        choices=("vector", "raster", "stochastic", "stipple",
                                 "fusion"))
    render.add_argument("--scope-raster", "--raster", dest="scope_raster",
                        action="store_true")
    render.add_argument("--scope-stochastic", "--stochastic",
                        dest="scope_stochastic", action="store_true")
    render.add_argument("--scope-stipple", "--stipple",
                        dest="scope_stipple", action="store_true")
    ap.add_argument("--scope-invert", action=argparse.BooleanOptionalAction,
                    default=None)
    ap.add_argument("--scope-x-only", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="send only X through a one-channel output")
    ap.add_argument("--scope-trigger", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="one rising X edge per trace (on by default)")
    ap.add_argument("--scope-trigger-us", "--scope-trigger-duration",
                    "--scope-yt-trigger", "--scope-yt-trigger-us",
                    dest="scope_trigger_us", type=float, metavar="US")
    ap.add_argument("--scope-trigger-shape", choices=("ramp", "step"))
    ap.add_argument("--scope-yt-timing", choices=("fixed", "dwell"))
    ap.add_argument("--scope-yt", action="store_true",
                    help=argparse.SUPPRESS)  # deprecated: = fixed timing
    ap.add_argument("--rotation", type=int, choices=[0, 90, 180, 270],
                    help="quarter-turn rotation of the output")
    ap.add_argument("--mirror", action=argparse.BooleanOptionalAction,
                    default=None, help="left-right flip of the output")
    ap.add_argument("--scope-walk-radius", type=int)
    ap.add_argument("--scope-walk-stride", type=int)
    ap.add_argument("--scope-walk-reseed-ms", type=float)
    ap.add_argument("--scope-gamma", type=float,
                    help="active raster/stochastic luminance exponent")
    ap.add_argument("--scope-stochastic-gamma", "--scope-walk-gamma",
                    dest="scope_walk_gamma", type=float)
    ap.add_argument("--scope-fusion", choices=("vrs", "vr", "sv", "sr"))
    ap.add_argument("--scope-walk-edge", type=float)
    ap.add_argument("--scope-walk-hz", type=float)
    ap.add_argument("--scope-stipple-points", type=int)
    ap.add_argument("--scope-precondition", type=float)
    args = ap.parse_args()

    if args.dir:
        p = os.path.abspath(args.dir)
        if not os.path.isdir(p):
            raise SystemExit(f"Directory not found: {p}")
        settings.IMAGES_DIR = p
        settings.MAIN_FOLDER_PATH = os.path.join(p, "face")
        settings.FLOAT_FOLDER_PATH = os.path.join(p, "float")
        if not args.xy_dir and hasattr(settings, "_find_xy_dir"):
            settings.XY_DIR = settings._find_xy_dir(p)
    if args.xy_dir:
        xp = os.path.abspath(args.xy_dir)
        if not os.path.isdir(xp):
            raise SystemExit(
                f"XY directory not found: {xp}\n"
                "   Bake it: python utilities/convert_to_xy.py "
                f"-i {settings.IMAGES_DIR} -o {xp}")
        settings.XY_DIR = xp
    if args.scope_mode:
        settings.SCOPE_RENDER_MODE = args.scope_mode
    elif args.scope_raster:
        settings.SCOPE_RENDER_MODE = "raster"
    elif args.scope_stochastic:
        settings.SCOPE_RENDER_MODE = "stochastic"
    elif args.scope_stipple:
        settings.SCOPE_RENDER_MODE = "stipple"
    if args.scope_source is not None:
        settings.SCOPE_SOURCE = args.scope_source
    if args.scope_gui is not None:
        settings.SCOPE_GUI = args.scope_gui
    start_image_only = (getattr(settings, "SCOPE_GUI_IMAGE_ONLY", False)
                        if args.scope_gui_image_only is None else
                        args.scope_gui_image_only)
    start_fullscreen = (getattr(settings, "SCOPE_GUI_FULLSCREEN", False)
                        if args.scope_gui_fullscreen is None else
                        args.scope_gui_fullscreen)
    start_image_only = bool(start_image_only or start_fullscreen)
    if (start_image_only or start_fullscreen) and args.scope_gui is False:
        ap.error("scope GUI startup views cannot be combined with --no-scope-gui")
    settings.SCOPE_GUI_IMAGE_ONLY = start_image_only
    settings.SCOPE_GUI_FULLSCREEN = bool(start_fullscreen)
    if start_image_only or start_fullscreen:
        settings.SCOPE_GUI = True
    settings.SCOPE_SOURCE = getattr(settings, "SCOPE_SOURCE", "bake")
    if args.scope_live_size is not None:
        if args.scope_live_size < 16:
            ap.error("--scope-live-size must be at least 16 pixels")
        settings.SCOPE_LIVE_SIZE = args.scope_live_size
    if args.scope_fps is not None:
        settings.SCOPE_FPS = args.scope_fps
    if args.scope_samples is not None:
        settings.SCOPE_SAMPLES = args.scope_samples
    try:
        settings.SCOPE_CHANNELS = _scope_out.parse_channel_pair(
            args.scope_channels if args.scope_channels is not None
            else getattr(settings, "SCOPE_CHANNELS", (1, 2)))
    except ValueError as exc:
        ap.error(str(exc))
    render_was_selected = bool(
        args.scope_mode or args.scope_raster or args.scope_stochastic
        or args.scope_stipple)
    if settings.SCOPE_SOURCE == "images":
        if not render_was_selected:
            settings.SCOPE_RENDER_MODE = "raster"
        if settings.SCOPE_RENDER_MODE not in ("raster", "stochastic", "stipple"):
            ap.error("--scope-source images supports raster, stochastic, or "
                     "stipple rendering; vector/fusion require a bake")
        settings.SCOPE_RASTER = settings.SCOPE_RENDER_MODE == "raster"
    if args.scope_device is not None:
        settings.SCOPE_DEVICE = args.scope_device
    settings.SCOPE_DEVICE_SPEC = args.scope_device
    settings.SCOPE_ASK = args.scope_ask
    if (args.scope_mode or args.scope_raster or args.scope_stochastic
            or args.scope_stipple):
        settings.SCOPE_RASTER = settings.SCOPE_RENDER_MODE == "raster"
    if args.scope_invert is not None:
        settings.SCOPE_INVERT = args.scope_invert
    if args.scope_x_only is not None:
        settings.SCOPE_X_ONLY = args.scope_x_only
    if args.rotation is not None:
        settings.INITIAL_ROTATION = int(args.rotation) % 360
    if args.mirror is not None:
        settings.INITIAL_MIRROR = 1 if args.mirror else 0
    if args.scope_trigger is not None:
        settings.SCOPE_TRIGGER = args.scope_trigger
    if args.scope_trigger_shape is not None:
        settings.SCOPE_TRIGGER_SHAPE = args.scope_trigger_shape
    if args.scope_trigger_us is not None:
        if (not math.isfinite(args.scope_trigger_us)
                or args.scope_trigger_us <= 0):
            ap.error("--scope-trigger-us must be finite and greater than zero")
        settings.SCOPE_TRIGGER_US = args.scope_trigger_us
        settings.SCOPE_YT_TRIGGER_US = None   # the CLI wins over the alias
    if args.scope_yt_timing is not None:
        settings.SCOPE_YT_TIMING = args.scope_yt_timing
    elif args.scope_yt:
        settings.SCOPE_YT_TIMING = "fixed"
    if args.scope_gamma is not None:
        if settings.SCOPE_RENDER_MODE == "fusion":
            settings.SCOPE_GAMMA = args.scope_gamma
            settings.SCOPE_WALK_GAMMA = args.scope_gamma
        elif settings.SCOPE_RENDER_MODE in ("stochastic", "stipple"):
            settings.SCOPE_WALK_GAMMA = args.scope_gamma
        else:
            settings.SCOPE_GAMMA = args.scope_gamma
    for arg, setting_name in (
            (args.scope_walk_radius, "SCOPE_WALK_RADIUS"),
            (args.scope_walk_stride, "SCOPE_WALK_STRIDE"),
            (args.scope_walk_reseed_ms, "SCOPE_WALK_RESEED_MS"),
            (args.scope_walk_gamma, "SCOPE_WALK_GAMMA"),
            (args.scope_walk_edge, "SCOPE_WALK_EDGE"),
            (args.scope_walk_hz, "SCOPE_WALK_HZ"),
            (args.scope_stipple_points, "SCOPE_STIPPLE_POINTS"),
            (args.scope_precondition, "SCOPE_PRECONDITION")):
        if arg is not None:
            setattr(settings, setting_name, arg)
    if args.scope_fusion is not None:
        settings.SCOPE_FUSION = args.scope_fusion

    source = os.path.basename(os.path.normpath(settings.IMAGES_DIR)).replace(" ", "_")
    suffix = f"{source}_scope_None"
    settings.PROCESSED_DIR = os.path.join("_cache", f"folders_processed_{suffix}")
    settings.GENERATED_LISTS_DIR = os.path.join("_cache", f"generated_lists_{suffix}")

    # make_file_lists captures its cache dirs at import time, so import it only
    # after settings are final -- main.py achieves the same by configuring
    # before its own imports.
    import make_file_lists

    if (settings.SCOPE_SOURCE == "images"
            or getattr(settings, "SCOPE_LIST_FROM_IMAGES", False)):
        print(">> Live image lists will be validated by the scope engine")
    else:
        print(">> Scope will use the baked manifest when available")


# ---------------------------------------------------------------- resolution

def _xy_root():
    explicit = getattr(settings, "XY_DIR", None)
    if explicit:
        return explicit
    base = os.path.basename(os.path.normpath(settings.IMAGES_DIR))
    for cand in (f"{settings.IMAGES_DIR}_xy", f"{base}_xy", "images_xy"):
        if os.path.isdir(cand):
            return cand
    return "images_xy"


def _folder_to_library_dir(image_path, xy_root):
    """The baker mirrors the source tree, so a folder's path relative to
    IMAGES_DIR is its library's path relative to XY_DIR.  Falls back to slicing
    from the face/float component when prefixes don't line up."""
    folder = os.path.dirname(image_path)
    rel = os.path.relpath(folder, settings.IMAGES_DIR)
    if rel.startswith(".."):
        parts = Path(folder).parts
        for key in ("face", "float"):
            if key in parts:
                rel = os.path.join(*parts[parts.index(key):])
                break
    return os.path.join(xy_root, rel)


def _manifest_from_xy(xy_root):
    """
    Build the folder manifest from the BAKED tree instead of the images.

    Scope mode never opens an image -- it needs only the folder order, the
    folder counts and the frame count.  The bake mirrors the source tree, so
    it already carries all three, and reading them here avoids rescanning tens
    of thousands of files at every launch.  It also means a scope-only machine
    does not need the source images on disk at all.

    make_file_lists' OWN ordering functions are reused rather than
    reimplemented: folder_selector indexes into this order, so a divergence
    would silently pair the wrong folders.

    Returns (frames, main_dirs, float_dirs) or None if the tree cannot supply it.
    """
    from make_file_lists import natural_sort_key, check_folder_prefix

    root = Path(xy_root)
    if not root.is_dir():
        return None
    libs = sorted({p.parent for p in root.rglob("frame_starts.npy")},
                  key=lambda p: natural_sort_key(str(p)))
    if not libs:
        return None

    mains, floats = [], []
    for d in libs:
        parts = d.relative_to(root).parts
        kind = "float" if "float" in parts else "main"
        if not check_folder_prefix(str(d), kind):
            continue                      # same rule the list builder applies
        (floats if kind == "float" else mains).append(d)
    if not mains or not floats:
        return None

    def folder_key(path):
        # create_folder_csv_files() sorts by numeric prefix, then basename.
        # The relative path is only a deterministic tie-breaker for nested
        # folders with the same leaf name; it does not alter normal ordering.
        name = path.name
        prefix = name.partition("_")[0]
        number = int(prefix) if prefix.isdigit() else float("inf")
        return (number, natural_sort_key(name),
                natural_sort_key(str(path.relative_to(root))))

    mains.sort(key=folder_key)
    floats.sort(key=folder_key)

    # Match find_default_csvs(): normal modes group folders by image count,
    # keep only counts that exist in BOTH layers, then choose the largest
    # common count. Counting every bake and replacing mismatches with None
    # shifts folder_selector's numeric indices and pairs the wrong face/float.
    by_count = {"main": {}, "float": {}}
    for kind, paths in (("main", mains), ("float", floats)):
        for path in paths:
            try:
                count = len(np.load(path / "frame_starts.npy", mmap_mode="r")) - 1
            except Exception as e:
                print(f"[SCOPE] warning: cannot read bake manifest {path} ({e})")
                continue
            if count > 0:
                by_count[kind].setdefault(count, []).append(path)

    common = set(by_count["main"]) & set(by_count["float"])
    if not common:
        print("[SCOPE] warning: no face/float bake pair has the same frame "
              "count; cannot reproduce normal folder selection")
        return None
    frames = max(common)
    if len(common) > 1:
        print(f"[SCOPE] multiple common bake lengths {sorted(common)}; "
              f"using {frames}, matching normal modes")
    selected_mains = by_count["main"][frames]
    selected_floats = by_count["float"][frames]
    skipped = len(mains) + len(floats) - len(selected_mains) - len(selected_floats)
    if skipped:
        print(f"[SCOPE] manifest: excluded {skipped} baked folder(s) whose "
              f"frame count is not the selected face/float length {frames}")
    return frames, selected_mains, selected_floats


def _open_dirs(dirs, layer_name, expected_frames):
    libs = []
    for i, d in enumerate(dirs):
        try:
            lib = XYLibrary(d)
            if len(lib) != expected_frames:
                print(f"[SCOPE] warning: {layer_name} folder {i}: {len(lib)} "
                      f"baked frames vs {expected_frames} expected; ignoring "
                      f"this unregistered library ({d})")
                libs.append(None)
                continue
            libs.append(lib)
        except Exception as e:
            print(f"[SCOPE] warning: {layer_name} folder {i}: {d} ({e})")
            libs.append(None)
    return libs


def _open_layer(paths_by_index, xy_root, layer_name, expected_frames):
    libs = []
    for f in range(len(paths_by_index[0])):
        d = _folder_to_library_dir(paths_by_index[0][f], xy_root)
        try:
            lib = XYLibrary(d)
            if len(lib) != expected_frames:
                print(f"[SCOPE] warning: {layer_name} folder {f}: {len(lib)} "
                      f"baked frames vs {expected_frames} listed; ignoring "
                      f"this unregistered library -- rebake ({d})")
                libs.append(None)
                continue
            libs.append(lib)
        except Exception as e:
            print(f"[SCOPE] warning: {layer_name} folder {f}: "
                  f"no library at {d} ({e})")
            libs.append(None)
    return libs


# ---------------------------------------------------------------- engine

# The web handler runs on a server thread and must not touch PortAudio: the
# stream has to be torn down and rebuilt on the thread that owns the render
# loop, or the callback can fire against a half-closed stream.  So the handler
# only parks a request here and the loop picks it up.
_device_request = {"spec": None, "pending": False, "message": ""}
_timing_request = {"fps": None, "fields": None, "ips": None,
                   "pending": False, "message": ""}
_device_lock = threading.Lock()


def request_device(spec):
    """Ask the running scope to move to another output. Thread-safe.

    Returns immediately; the swap happens on the render loop's next tick.
    `spec` is a name fragment, an output-list index, or None for the system
    default -- exactly what --device accepts.
    """
    with _device_lock:
        _device_request["spec"] = spec
        _device_request["pending"] = True
        _device_request["message"] = "requested"
    return True


def device_status():
    """Current request state, for the web page to echo back."""
    with _device_lock:
        return dict(_device_request)


def request_timing(fps=None, fields=None, ips=None):
    """Queue a timing update for the scope producer thread to apply safely.

    Trace rate and interlace change the DAC frame size, so their stream is
    rebuilt by the thread that owns it.  ``ips`` changes the scope's free-running
    image clock; external MIDI clocks are left to their source.
    """
    values = {"fps": fps, "fields": fields, "ips": ips}
    if all(value is None for value in values.values()):
        raise ValueError("timing update needs fps, fields, or ips")
    if fps is not None and (not math.isfinite(float(fps)) or float(fps) <= 0):
        raise ValueError("trace rate must be finite and positive")
    if fields is not None and int(fields) not in (1, 2, 3, 4):
        raise ValueError("raster fields must be between 1 and 4")
    if ips is not None and (not math.isfinite(float(ips)) or float(ips) <= 0):
        raise ValueError("picture rate must be finite and positive")
    with _device_lock:
        _timing_request.update(values)
        _timing_request["pending"] = True
        _timing_request["message"] = "timing update queued"
    return True


def timing_status():
    """Return the most recent timing request and its apply status."""
    with _device_lock:
        return dict(_timing_request)



def _dev_name_of(scope):
    """Human-readable name of whatever output a Scope ended up on."""
    if getattr(scope, "null", False):
        return "none (browser renders)"
    try:
        import sounddevice as _sd
        from scope_out import scrub as _scrub
        return _scrub(_sd.query_devices(scope.stream.device)["name"])
    except Exception:
        return "?"


def _swap_device(old_scope, spec, source, fps, samples, main_libs, float_libs,
                 density, trim, rows, fields, row_bias, autofit, invert=False,
                 x_only=False, channel_pair=(1, 2), resolved_device=False,
                 geometry_budget=None):
    """Move the running scope to another output device.

    Returns (new_scope, new_cal, fresh_sweep_state).

    Two things make this more than close-and-reopen:

    1.  A different device can have a different DEFAULT SAMPLE RATE, and
        samples_per_trace is samplerate/fps.  A different sample budget is a
        different grid -- so the tone mapping and grid MUST be recalibrated,
        not carried over.  This is the same reason the handoff says the grid
        has to stay runtime-derived: it depends on the sample budget, which is
        a fact about the device.
    2.  The chained alternating sweep carries the previous trace's last sample
        forward.  Across a stream teardown the beam is not where that says it
        is, so the chain is reset rather than continued -- otherwise the first
        trace on the new device starts with a full-screen jump.

    The old stream is closed BEFORE the new one opens: some exclusive-mode
    routes (raw ALSA hw:, WASAPI exclusive) will refuse a second handle, and
    holding both would fail on exactly the devices worth using.
    """
    from scope_out import Scope, resolve_device
    dev = (spec if resolved_device else resolve_device(
        spec, min_channels=required_output_channels(channel_pair, x_only)))
    try:
        old_scope.stream.stop()
        old_scope.stream.close()
    except Exception:
        pass
    # Matches the original construction at run_scope() exactly.  Note NO
    # lowpass_hz: the SCOPE_LOWPASS setting is applied per frame through
    # lowpass_circular() in _emit, not by the Scope.  Passing it here as well
    # would filter twice after a device change and only after a device change,
    # which is the kind of difference that gets blamed on the new device.
    new_scope = Scope(
        fps=fps, samples=samples, device=dev, source=source,
        invert_y=False, rotation=getattr(old_scope, "rotation", 0),
        mirror=getattr(old_scope, "mirror", False),
        trigger=getattr(old_scope, "trigger", True),
        trigger_shape=getattr(old_scope, "trigger_shape", "ramp"),
        x_only=x_only,
        channel_pair=channel_pair,
        yt_trigger_us=getattr(old_scope, "yt_trigger_us", 250.0),
        yt_trigger_level=getattr(old_scope, "yt_trigger_level", 0.99))
    set_output_audio = getattr(new_scope, "set_output_audio", None)
    if callable(set_output_audio):
        set_output_audio(muted=getattr(old_scope, "output_muted", False))
    new_cal = {}
    try:
        new_cal = calibrate(main_libs, float_libs,
                            (geometry_budget or new_scope.samples_per_frame),
                            density=density, trim=trim, rows=rows,
                            fields=fields, row_bias=row_bias, autofit=autofit,
                            invert=invert)
    except Exception as e:
        print(f"[SCOPE] recalibration after device change skipped ({e})")
    new_scope.stream.start()
    print(f"[SCOPE] output now: {_dev_name_of(new_scope)} "
          f"({new_scope.samplerate} Hz, {new_scope.samples_per_frame} "
          f"samples/trace)", flush=True)
    return new_scope, new_cal, {"rev": False, "end": None}


def run_scope(clock_source=None):
    """Caller (main.py, or _bootstrap) must have prepared the generated lists."""
    if clock_source is None:
        clock_source = settings.CLOCK_MODE
    start_image_only = bool(getattr(settings, "SCOPE_GUI_IMAGE_ONLY", False))
    start_fullscreen = bool(getattr(settings, "SCOPE_GUI_FULLSCREEN", False))
    gui_enabled = bool(getattr(settings, "SCOPE_GUI", False)
                        or start_image_only or start_fullscreen)

    from settings import IPS, PINGPONG

    # --- configuration: all of it from settings ---
    fps = getattr(settings, "SCOPE_FPS", None) or IPS
    samples = getattr(settings, "SCOPE_SAMPLES", None)
    geometry_samples = getattr(settings, "SCOPE_GEOMETRY_SAMPLES", None)
    traversal_hz = getattr(settings, "SCOPE_TRAVERSAL_HZ", None)
    render_mode = getattr(settings, "SCOPE_RENDER_MODE", None)
    x_only = bool(getattr(settings, "SCOPE_X_ONLY", False))
    trigger_on = bool(getattr(settings, "SCOPE_TRIGGER", True))
    trigger_shape = getattr(settings, "SCOPE_TRIGGER_SHAPE", "ramp")
    if trigger_shape not in ("ramp", "step"):
        raise ValueError("SCOPE_TRIGGER_SHAPE must be ramp or step")
    yt_timing = getattr(settings, "SCOPE_YT_TIMING", "dwell")
    if getattr(settings, "SCOPE_YT", False):
        yt_timing = "fixed"          # deprecated settings.py spelling
    if yt_timing not in ("fixed", "dwell"):
        raise ValueError("SCOPE_YT_TIMING must be fixed or dwell")
    # SCOPE_YT_TRIGGER_US was the name this shipped under; an existing
    # settings.py that sets it should not quietly stop being honoured.
    trigger_us = float(getattr(settings, "SCOPE_TRIGGER_US", 250.0)
                       if getattr(settings, "SCOPE_YT_TRIGGER_US", None) is None
                       else settings.SCOPE_YT_TRIGGER_US)
    if not math.isfinite(trigger_us) or trigger_us <= 0:
        raise ValueError("SCOPE_TRIGGER_US must be finite and greater than zero")
    if getattr(settings, "SCOPE_RASTER", False) and render_mode == "vector":
        render_mode = "raster"       # compatibility with older settings.py
    if render_mode not in ("vector", "raster", "stochastic", "stipple",
                           "fusion"):
        render_mode = "raster" if getattr(settings, "SCOPE_RASTER", False) else "vector"
    channel_pair = parse_channel_pair(
        getattr(settings, "SCOPE_CHANNELS", (1, 2)))
    scope_source = getattr(settings, "SCOPE_SOURCE", "bake")
    if scope_source not in ("bake", "images"):
        raise ValueError("SCOPE_SOURCE must be 'bake' or 'images'")
    live_size = max(16, int(getattr(settings, "SCOPE_LIVE_SIZE", 128)))
    if scope_source == "images" and render_mode == "vector":
        render_mode = "raster"
    if scope_source == "images" and render_mode not in (
            "raster", "stochastic", "stipple"):
        raise ValueError("live scope images support raster, stochastic, "
                         "or stipple; vector/fusion need an XY bake")
    if geometry_samples is not None or traversal_hz is not None:
        if scope_source != "bake" or render_mode != "raster":
            raise ValueError("scope geometry/traversal controls require baked raster rendering")
        if traversal_hz is not None and not trigger_on:
            raise ValueError("SCOPE_TRAVERSAL_HZ requires the scope trigger")
    use_raster = render_mode == "raster"
    use_stochastic = render_mode == "stochastic"
    use_stipple = render_mode == "stipple"
    use_fusion = render_mode == "fusion"
    invert = bool(getattr(settings, "SCOPE_INVERT", False))
    rotation = int(getattr(settings, "INITIAL_ROTATION", 0) or 0) % 360
    if rotation % 90:
        print(f"[SCOPE] INITIAL_ROTATION {rotation} is not a quarter turn; using 0")
        rotation = 0
    mirror = bool(getattr(settings, "INITIAL_MIRROR", 0))
    try:
        fusion_components = normalize_fusion_components(
            getattr(settings, "SCOPE_FUSION", "vrs"))
    except ValueError as e:
        print(f"[SCOPE] {e}; using vrs")
        fusion_components = "vrs"
    realtime = getattr(settings, "SCOPE_REALTIME", False)
    # Degrade, do not refuse. Fixed row timing is implemented inside
    # render_luma, so it really is raster-only -- but the answer to asking for
    # it in stochastic is to say so and draw something, not to exit.
    if yt_timing == "fixed" and render_mode != "raster":
        print(f"[SCOPE] fixed row timing is raster only; {render_mode} keeps "
              f"dwell timing. The trigger marker is unaffected.")
        yt_timing = "dwell"
    if yt_timing == "fixed" and realtime:
        # The marker stays periodic either way -- it is stamped on a sample
        # counter, not on the row stream -- but fixed slots assume a whole
        # trace, which realtime does not deliver.
        print("[SCOPE] realtime streams partial traces; using dwell timing. "
              "The trigger stays periodic, so a Y-T scope still locks, but "
              "the picture may drift against it.")
        yt_timing = "dwell"
    import make_file_lists
    import index_calculator as _index_calculator
    update_index = _index_calculator.update_index
    from folder_selector import update_folder_selection, folder_dictionary

    min_feature = getattr(settings, "SCOPE_MIN_FEATURE", 0.02)
    trim = getattr(settings, "SCOPE_TRIM", 0.02)
    gamma = getattr(settings, "SCOPE_GAMMA", 2.2)
    density = getattr(settings, "SCOPE_DENSITY", 1.0)
    walk_radius = max(1, int(getattr(settings, "SCOPE_WALK_RADIUS", 10)))
    walk_stride = max(0, int(getattr(settings, "SCOPE_WALK_STRIDE", 0)))
    walk_reseed_ms = max(0.1, float(getattr(settings, "SCOPE_WALK_RESEED_MS", 5.0)))
    walk_gamma = max(0.01, float(getattr(settings, "SCOPE_WALK_GAMMA", 2.0)))
    walk_edge = max(0.0, float(getattr(settings, "SCOPE_WALK_EDGE", 0.0)))
    walk_hz = max(1.0, float(getattr(settings, "SCOPE_WALK_HZ", 48000.0)))
    stipple_points = max(8, int(getattr(
        settings, "SCOPE_STIPPLE_POINTS", 768)))
    rows = getattr(settings, "SCOPE_ROWS", None)
    autofit = getattr(settings, "SCOPE_AUTOFIT", True)
    lowpass = getattr(settings, "SCOPE_LOWPASS", None)
    oversample = int(getattr(settings, "SCOPE_OVERSAMPLE", 1) or 1)
    sweep_mode = getattr(settings, "SCOPE_SWEEP", "alternate")
    # The sweep substitution is deliberately NOT made here: mix and realtime
    # below can still demote fixed timing to dwell, and announcing that the
    # sweep is being ignored before deciding whether fixed timing survives
    # printed a claim that then stopped being true. See after those checks.
    fields = max(1, int(getattr(settings, "SCOPE_FIELDS", 1) or 1))
    if geometry_samples is not None and int(geometry_samples) < 2:
        raise ValueError("SCOPE_GEOMETRY_SAMPLES must be at least 2")
    if traversal_hz is not None:
        traversal_hz = float(traversal_hz)
        if not math.isfinite(traversal_hz) or traversal_hz <= 0:
            raise ValueError("SCOPE_TRAVERSAL_HZ must be finite and greater than zero")
        if fields > 1:
            raise ValueError("SCOPE_TRAVERSAL_HZ requires one field")
    if ((geometry_samples is not None or traversal_hz is not None)
            and yt_timing == "fixed"):
        raise ValueError("independent geometry controls cannot be combined with fixed Y-T timing")
    if (geometry_samples is not None or traversal_hz is not None) and realtime:
        raise ValueError("scope geometry/traversal controls cannot be used in realtime mode")
    fields_explicit = bool(getattr(settings, "SCOPE_FIELDS_EXPLICIT", False))
    dc_comp = getattr(settings, "SCOPE_DC_COMP", None)
    border = float(getattr(settings, "SCOPE_BORDER", 0.0) or 0.0)
    row_bias = float(getattr(settings, "SCOPE_ROW_BIAS", 1.0) or 1.0)
    mix_hz = getattr(settings, "SCOPE_MIX", None)
    if (geometry_samples is not None or traversal_hz is not None) and mix_hz:
        raise ValueError("scope geometry/traversal controls cannot be combined with mix")
    _raw_mix_duty = float(getattr(settings, "SCOPE_MIX_DUTY", 0.5))
    if not math.isfinite(_raw_mix_duty):
        print("[SCOPE] non-finite SCOPE_MIX_DUTY; using 0.5")
        _raw_mix_duty = 0.5
    mix_duty = min(1.0, max(0.0, _raw_mix_duty))
    if mix_duty != _raw_mix_duty:
        print(f"[SCOPE] SCOPE_MIX_DUTY {_raw_mix_duty:g} clamped to "
              f"{mix_duty:g}")
    device_spec = getattr(settings, "SCOPE_DEVICE_SPEC", None)
    ask = getattr(settings, "SCOPE_ASK", False)
    if scope_source == "images":
        if mix_hz:
            raise ValueError("live scope images cannot use --scope-mix; "
                             "it includes baked vector geometry")

    if realtime:
        _d = getattr(settings, "SCOPE_BUFFER_BLOCKS", 6)
        print(f"[SCOPE] realtime: generated on a worker thread, {_d} x 256 "
              "sample ring")
    if realtime and not use_raster:
        print("[SCOPE] realtime applies to raster only; ignoring.")
        realtime = False
    if mix_hz and realtime:
        print("[SCOPE] mix needs whole passes; ignoring realtime.")
        realtime = False
    if yt_timing == "fixed" and mix_hz:
        # Mix alternates whole traces between renderers, and only the raster
        # ones can honour fixed slots. The marker is periodic across all of
        # them regardless, so mix keeps working -- it is only the row timing
        # that has to give way.
        print("[SCOPE] mix alternates renderers; using dwell timing so every "
              "trace is built the same way.")
        yt_timing = "dwell"
    # Now that fixed timing has survived every demotion above, it can claim
    # the sweep. render_yt_grid builds its own closed timeline and ignores the
    # chaining a sweep mode sets up, so the two would otherwise disagree
    # silently. Not an error any more, just a stated substitution.
    if yt_timing == "fixed" and sweep_mode != "retrace":
        print(f"[SCOPE] fixed row timing supplies its own retrace; "
              f"ignoring --scope-sweep {sweep_mode}.")
        sweep_mode = "retrace"
    if mix_hz is not None:
        mix_hz = float(mix_hz)
        if not math.isfinite(mix_hz) or mix_hz <= 0.0:
            raise ValueError("SCOPE_MIX must be a finite rate greater than zero")
        if samples is not None:
            print("[SCOPE] mix owns the trace clock; ignoring SCOPE_SAMPLES. "
                  "Use SCOPE_MIX/--scope-mix to trade pass rate for samples.")
            samples = None
        fps = mix_hz                   # the switch rate IS the trace rate

    # --- interlace ---------------------------------------------------------
    # The visible flicker is the TRACE rate, not the content rate.  At 30 fps
    # the beam repaints 30 times a second, well under fusion, so the sweep is
    # legible as a sweep.  Raising fps progressively shrinks the grid, because
    # a trace is rate/fps samples and the grid is sized from that.
    #
    # Interlace breaks the coupling: each trace draws every Nth row, so a
    # picture still costs rate/IPS samples in total and the grid is unchanged,
    # but the beam covers the full screen height N times as often.  Refresh
    # goes up, resolution does not go down.  Same reason broadcast television
    # did it.
    if mix_hz:
        # Mix hands only `mix_duty` of the traces to raster, so a raster
        # picture is assembled from mix_hz*duty/IPS traces -- and that is
        # exactly a field count.  Sizing the grid per TRACE here throws away
        # the same resolution interlace was written to recover: at mix 120 /
        # duty 0.5 the raster grid came out a quarter of the 30 fps grid, when
        # only half of that loss is the real cost of sharing the beam.
        _raster_traces = mix_hz * mix_duty / max(IPS, 1)
        _mix_traces_per_index = mix_hz / max(IPS, 1)
        _aligned = (abs(_mix_traces_per_index - round(_mix_traces_per_index))
                    < 1e-6
                    and abs(_raster_traces - round(_raster_traces)) < 1e-6)
        if mix_duty <= 0.0:
            if fields > 1:
                print("[SCOPE] mix duty is zero, so there are no raster fields; "
                      "ignoring --scope-fields.")
            fields = 1
        elif (fields <= 1 and not fields_explicit
              and _raster_traces >= 1.9 and _aligned):
            fields = int(round(_raster_traces))
            print(f"[SCOPE] mix: raster gets {_raster_traces:.1f} traces per "
                  f"index, so interlacing it x{fields} -- the grid is sized "
                  f"for {fields} traces, not one. Pass --scope-fields 1 to "
                  "size it per trace as before.")
        elif (fields <= 1 and not fields_explicit
              and _raster_traces >= 1.9 and not _aligned):
            print(f"[SCOPE] mix: raster averages {_raster_traces:.2f} traces "
                  "per index, not a whole field count; keeping it progressive "
                  "so fields never straddle two source images.")
        elif fields > 1 and not _aligned:
            print(f"[SCOPE] warning: mix rate/duty gives {_raster_traces:.2f} "
                  f"raster traces per index, so --scope-fields {fields} cannot "
                  "stay registered to source images. Use a rate/duty whose "
                  "raster traces per index is an integer.")

    if fields > 1:
        if not use_raster and not mix_hz:
            print("[SCOPE] interlace is a raster technique (vector traces have "
                  "no row structure, and neither does stochastic); ignoring "
                  "--scope-fields.")
            fields = 1
        elif realtime:
            print("[SCOPE] interlace needs whole traces and realtime streams "
                  "rows; ignoring --scope-fields.")
            fields = 1
        elif mix_hz:
            # mix already fixed fps to the switch rate; do not touch it
            pass
        elif samples:
            print(f"[SCOPE] interlace x{fields}: --scope-samples set "
                  "explicitly, so the trace rate is whatever that implies. "
                  f"For a stable picture it must come to {fields} x {IPS} Hz.")
        elif getattr(settings, "SCOPE_FPS", None) and fps != fields * IPS:
            print(f"[SCOPE] --scope-fps {fps} with --scope-fields {fields} "
                  f"is not {fields} x {IPS} ips; fields will not line up with "
                  f"indices. Using {fields * IPS}.")
            fps = fields * IPS
        else:
            fps = fields * IPS

    tap_traces = (TriangleMixScheduler.coverage_traces(mix_duty, fields)
                  if mix_hz else fields)

    # --- libraries ---
    live_images = None
    if scope_source == "images":
        make_file_lists.process_files(reuse_existing=True)
        _, main_paths, float_paths = make_file_lists.initialize_image_lists(
            clock_source)
        from scope_image_source import RuntimeScopeImageSource
        live_images = RuntimeScopeImageSource(
            main_paths, float_paths, width=live_size)
        png_paths_len = live_images.frames
        main_libs, float_libs = live_images.main_libs, live_images.float_libs
        main_folder_count, float_folder_count = len(main_libs), len(float_libs)
        print(f"[SCOPE] live image source: {live_images.width}x"
              f"{live_images.height} thumbnails, lazy decode")
    else:
        # Prefer the baked tree: it carries the same folder names in the same
        # order, so it supplies the manifest without scanning source images.
        # Fall back to image lists when a bake predates manifests or the tree
        # cannot be read.
        xy_root = _xy_root()
        print(f"[SCOPE] XY libraries: {xy_root}")
        manifest = None
        if not getattr(settings, "SCOPE_LIST_FROM_IMAGES", False):
            manifest = _manifest_from_xy(xy_root)

        if manifest is not None:
            png_paths_len, main_dirs, float_dirs = manifest
            main_folder_count = len(main_dirs)
            float_folder_count = len(float_dirs)
            print("[SCOPE] manifest from the bake (images not read)")
            main_libs = _open_dirs(main_dirs, "main", png_paths_len)
            float_libs = _open_dirs(float_dirs, "float", png_paths_len)
        else:
            make_file_lists.process_files(reuse_existing=True)
            _, main_paths, float_paths = make_file_lists.initialize_image_lists(
                clock_source)
            png_paths_len = len(main_paths)
            main_folder_count = len(main_paths[0])
            float_folder_count = len(float_paths[0])
            main_libs = _open_layer(main_paths, xy_root, "main", png_paths_len)
            float_libs = _open_layer(float_paths, xy_root, "float", png_paths_len)
    if not any(l is not None for l in main_libs + float_libs):
        if scope_source == "images":
            raise RuntimeError("No usable face/float folders in the live image lists")
        raise RuntimeError(
            f"No XY libraries found under '{xy_root}'. Run:\n"
            f"  python utilities/convert_to_xy.py -i {settings.IMAGES_DIR} "
            f"-o {xy_root}")
    if ((mix_hz and mix_duty < 1.0)
            or (use_fusion and "v" in fusion_components)):
        def _has_drawable_geometry(lib):
            if lib is None or len(lib.verts) == 0:
                return False
            return lib.flags is None or bool(np.any(lib.flags != 2))

        if not any(_has_drawable_geometry(lib)
                   for lib in main_libs + float_libs):
            raise RuntimeError(
                "Vector, vector-fusion, and mix modes need contour geometry, "
                "but this is a "
                "thumbs-only bake. Rebake normally without --thumbs-only.")
    missing_thumbs = [lib for lib in main_libs + float_libs
                      if lib is not None and lib.thumbs is None]
    if (use_raster or use_stochastic or use_stipple or use_fusion
            or mix_hz) and missing_thumbs:
        raise RuntimeError(
            "Raster, stochastic, stipple, and fusion modes need thumbnails and this "
            "bake has "
            f"{len(missing_thumbs)} library/libraries without them. Rebake "
            "normally; thumbnails are part of every standard bake.")
    if use_raster or use_stochastic or use_stipple or use_fusion or mix_hz:
        thumb_shapes = {
            tuple(lib.thumbs.shape[1:3])
            for lib in main_libs + float_libs
            if lib is not None and lib.thumbs is not None
        }
        if len(thumb_shapes) > 1:
            shown = ", ".join(f"{w}x{h}" for h, w in sorted(thumb_shapes))
            raise RuntimeError(
                "Baked thumbnail geometry is inconsistent across layers "
                f"({shown}). Rebake the complete tree with one --thumb-width; "
                "mixing partial old/new bakes breaks layer registration.")
    if (use_stochastic or use_stipple or use_fusion
            or (mix_hz and mix_duty < 1.0)):
        raw_active = (use_stochastic or use_stipple
                      or (use_fusion and "s" in fusion_components)
                      or (mix_hz and mix_duty < 1.0))
        legacy_luma = ([lib for lib in main_libs + float_libs
                        if lib is not None and lib.thumbs is not None
                        and lib.thumbs.ndim >= 4
                        and lib.thumbs.shape[-1] < 3
                        and not getattr(lib, "raw_thumbnail", False)]
                       if raw_active else [])
        if legacy_luma:
            print("[SCOPE] raw-luminance modes: this older bake has only the raster-"
                  "preconditioned luminance channel. It remains compatible, "
                  "but rebake once for the raw luminance channel.")
        first_thumb = next(
            (lib.thumbs for lib in main_libs + float_libs
             if lib is not None and lib.thumbs is not None), None)
        walk_width = int(first_thumb.shape[2])
        resolved_stride = StochasticEmitter(
            48000, 2, stride=walk_stride)._stride_for_width(walk_width)
        stride_text = (f"auto -> {resolved_stride}" if walk_stride <= 0
                       else str(resolved_stride))
        stochastic_active = (use_stochastic
                             or (use_fusion and "s" in fusion_components)
                             or (mix_hz and mix_duty < 1.0))
        if stochastic_active:
            print(f"[SCOPE] stochastic settings: gamma={walk_gamma:g} "
                  f"stride={stride_text} at {walk_width}px radius={walk_radius} "
                  f"reseed={walk_reseed_ms:g}ms walk={walk_hz:g}Hz", flush=True)
        if use_stipple or (mix_hz and mix_duty < 1.0):
            candidate_counts = {
                int(lib.stipple_xy.shape[1])
                for lib in main_libs + float_libs
                if lib is not None and lib.stipple_xy is not None
            }
            candidate_text = (f" from {min(candidate_counts)} baked source-detail "
                              "candidates" if candidate_counts else
                              " (legacy full-thumbnail fallback)")
            print(f"[SCOPE] stipple settings: {stipple_points} stable weighted "
                  f"positions{candidate_text}, gamma={walk_gamma:g}, "
                  f"edge={walk_edge:g}")

    requested_precondition = getattr(settings, "SCOPE_PRECONDITION", None)
    if requested_precondition is not None:
        raster_precondition = max(0.0, float(requested_precondition))
    else:
        # Old compact bakes may recommend 0.45 in format.json. Do not silently
        # re-enable it: it preserves cell count but makes faces look hollow.
        raster_precondition = 0.0
    if (use_raster or mix_hz or (use_fusion and "r" in fusion_components)):
        domain = "sweep-grid" if raster_precondition else "off, natural tone"
        print(f"[SCOPE] raster precondition: {raster_precondition:g} ({domain})")

    # Calibrate the grid and tone mapping ONCE.  Deriving either per frame
    # makes them follow content statistics: the grid changing by a row
    # re-quantizes every cell boundary, and a drifting stretch pushes cells
    # across the trim threshold so they wink in and out as black flecks.
    cal = {}

    # --- audio ---
    live = {"main": None, "mi": 0, "float": None, "fi": 0}
    if getattr(settings, "SCOPE_DEVICE_RESOLVED", False):
        dev = getattr(settings, "SCOPE_DEVICE", None)   # main.py already chose
    else:
        # SCOPE_DEVICE_SPEC is set to None unconditionally by main.py, so this
        # branch could only ever reach the system default -- there was no way
        # to pin an output from settings.py at all.  That is fine interactively
        # and fatal headless, where there is no CLI to carry --device and the
        # system default is whatever HDMI enumerated first.  Fall back to
        # SCOPE_DEVICE so settings.py can name one.
        if device_spec is None:
            device_spec = getattr(settings, "SCOPE_DEVICE", None)
        dev = choose_device(ask=ask, device=device_spec,
                            min_channels=required_output_channels(
                                channel_pair, x_only))
    source = None
    if realtime:
        probe = Scope(fps=fps, samples=samples, device=dev, invert_y=False,
                      x_only=x_only, channel_pair=channel_pair)
        n_pass = probe.samples_per_frame
        probe.stream.close()
        # Calibrate HERE, before the generator is built: it needs the same
        # grid and levels frame mode uses, or the two modes render the same
        # content differently.
        try:
            cal = calibrate(main_libs, float_libs, n_pass,
                            density=density, trim=trim, rows=rows,
                            fields=fields, row_bias=row_bias, autofit=autofit,
                            invert=invert)
        except Exception as e:
            print(f"[SCOPE] calibration skipped ({e})")
            cal = {}
        if cal:
            _spc = n_pass * fields / max(cal["grid_rows"] * cal["grid_cols"], 1)
            print(f"[SCOPE] grid {cal['grid_cols']}x{cal['grid_rows']} "
                  f"({_spc:.2f} samples/cell), calibrated once")
        _gen_grid = _rotation_grid(cal, rotation)
        gen = SweepSource(
            lambda: live, n_pass, gamma=gamma, trim=trim,
            density=density, rows=rows,
            precondition=raster_precondition, invert=invert,
            rotation=rotation,
            alternate=(yt_timing != "fixed"),
            grid_rows=(_gen_grid[0] if _gen_grid else None),
            grid_cols=(_gen_grid[1] if _gen_grid else None),
            levels=(cal.get("levels") if cal else None))
        # Generate on a worker thread rather than in the audio callback: only
        # the AVERAGE has to keep up, and the ring absorbs the spikes.  Depth
        # is latency, and low latency is the point of realtime mode, so keep
        # it small.
        source = BufferedSource(gen, blocksize=256,
                                depth=getattr(settings, "SCOPE_BUFFER_BLOCKS", 6))
    # invert_y=False: everything out of scope_bake is ALREADY in scope
    # space (y up).  XYLibrary.frame() applies flip_y, and render_luma
    # builds its rows with ys = -linspace(...).  Scope.show()'s invert_y
    # is for callers handing it raw screen-space polylines; applying it
    # here flips a second time and stands the vector picture on its head.
    # Rotation belongs in renderer/image space. Scope's final-output rotation
    # swaps the fast raster carrier from X to Y at quarter turns, which breaks
    # X-triggered and X-only displays. Every renderer below receives the
    # orientation while the physical output axes remain fixed.
    # Mirror is the exception to the paragraph above: it only negates X, so
    # the fast carrier stays on X and it is safe -- and cheaper -- to apply it
    # at the output rather than re-orienting every renderer's source.
    scope = Scope(fps=fps, samples=samples, device=dev, source=source,
                  invert_y=False, rotation=0, mirror=mirror,
                  trigger=trigger_on, trigger_shape=trigger_shape,
                  x_only=x_only,
                  channel_pair=channel_pair,
                  yt_trigger_us=trigger_us)
    if gui_enabled:
        # Keep the native visualizer quiet on launch. The preview tap remains
        # upstream of this output-stage mute, so the GUI image is unaffected.
        scope.set_output_audio(muted=True)
    # An explicit samples/trace budget takes precedence over the requested FPS.
    # Keep the runtime's time controls and diagnostics anchored to the rate the
    # DAC will actually emit, not the superseded convenience argument.
    fps = max(1, int(round(scope.samplerate / max(scope.samples_per_frame, 1))))

    if trigger_on:
        marker_us = scope.yt_trigger_us
        trigger_hz = scope.samplerate / max(scope.trace_samples, 1)
        print(f"[SCOPE] X trigger: {trigger_shape} marker, {marker_us:g} us, "
              f"{trigger_hz:g} Hz. Rising edge near +0.95.")
        if trigger_shape == "ramp":
            print("[SCOPE] The marker sweeps and parks outside +-0.9, so an "
                  "XY display set to fill the screen never shows it. Use "
                  "--scope-trigger-shape step if your trigger will not hold.")
        else:
            print("[SCOPE] step marker dwells at both rails: two bright dots "
                  "on an XY display. Y-T only.")
        if x_only:
            print("[SCOPE] X-only mono: connect the output to the Y-T input; "
                  f"set the timebase to one trace ({1e6 / max(trigger_hz, 1e-9):g} us).")
        else:
            print("[SCOPE] One channel, Y-T: take X, set the timebase to one "
                  f"complete trace ({1e6 / max(trigger_hz, 1e-9):g} us).")
    elif x_only:
        print("[SCOPE] X-only mono output: connect the output to the Y-T input; "
              "the trigger marker is disabled.")
    if yt_timing == "fixed":
        print("[SCOPE] fixed row slots: image width stays registered; "
              "brightness controls dwell within each row")

    # The baked thumbnail is a hard ceiling on scanlines; clamping silently
    # would look like the row setting being ignored.
    if use_raster or mix_hz:
        ref = next((l for l in main_libs if l is not None
                    and l.thumbs is not None), None)
        if ref is not None:
            cap_rows, cap_cols = ref.thumbs.shape[1], ref.thumbs.shape[2]
            if rows and rows > cap_rows:
                print(f"[SCOPE] rows={rows} exceeds the baked thumbnail "
                      f"({cap_cols}x{cap_rows}); clamping to {cap_rows}. "
                      f"Rebake with --thumb-width {int(rows * cap_cols / cap_rows)} "
                      "for more scanlines.")
            else:
                cells = scope.samples_per_frame * fields / max(density, 0.25)
                want = int(round((cells * cap_rows / cap_cols) ** 0.5))
                if want > cap_rows:
                    print(f"[SCOPE] the sample budget could resolve ~{want} "
                          f"scanlines but the bake caps it at {cap_rows}; "
                          "rebake with a larger --thumb-width to use it.")

    if getattr(scope, "null", False):
        _dev_name = "none (browser renders)"
    else:
        try:
            import sounddevice as _sd
            from scope_out import scrub as _scrub
            _dev_name = _scrub(_sd.query_devices(scope.stream.device)["name"])
        except Exception:
            _dev_name = "?"
    print(f"[SCOPE] output: {_dev_name}")
    if not getattr(scope, "null", False):
        _active_channels = channel_pair[:1] if x_only else channel_pair
        print("[SCOPE] PortAudio channels: " +
              ",".join(map(str, _active_channels)) + " (1-based)")

    if (use_raster or mix_hz) and not cal:
        try:
            cal = calibrate(main_libs, float_libs,
                            (geometry_samples or (3200 if traversal_hz is not None
                                                  else scope.samples_per_frame)),
                            density=density, trim=trim, rows=rows,
                            fields=fields, row_bias=row_bias, autofit=autofit,
                            invert=invert)
            if cal:
                spc = scope.samples_per_frame * fields / max(
                    cal["grid_rows"] * cal["grid_cols"], 1)
                print(f"[SCOPE] grid {cal['grid_cols']}x{cal['grid_rows']} "
                      f"({spc:.2f} samples/cell), calibrated once")
                if spc < 0.3:
                    print(f"[SCOPE] note: below ~0.3 samples/cell the lit "
                          "pattern shifts frame to frame and reads as moving "
                          "black flecks. Raise --scope-density toward 1.0.")
        except Exception as e:
            print(f"[SCOPE] calibration skipped ({e}); per-frame adaptation")

    mode_name = "MIX" if mix_hz else render_mode.upper()
    # trace_samples, not samples_per_frame: the marker has its own samples,
    # so the picture budget and the emitted trace length are no longer equal.
    _trace_hz = scope.samplerate / scope.trace_samples
    print(f"[SCOPE] {'X-ONLY ' if x_only else ''}"
          f"{mode_name}{' REALTIME' if realtime else ''}"
          f"{f' INTERLACE x{fields}' if fields > 1 else ''} | "
          f"{scope.samples_per_frame} samples/trace @ {scope.samplerate} Hz "
          f"({_trace_hz:.0f} passes/sec) | "
          f"latency ~{scope.stream.latency * 1000:.0f} ms")
    if dc_comp:
        print(f"[SCOPE] DC compensation at {dc_comp:.0f} Hz -- flat regions "
              "hold instead of sagging, at the cost of amplitude. Raise the "
              "scope's gain to compensate.")
    if invert:
        print("[SCOPE] inverse luminance ON (covered image content only; "
              "transparent padding remains dark)")
    if fields > 1 and not mix_hz:
        print(f"[SCOPE] refresh {_trace_hz:.0f} Hz, picture "
              f"{_trace_hz / fields:.0f} Hz "
              f"({scope.samples_per_frame * fields} samples per picture -- "
              "the grid is sized for that, not for one field)")
    print(f"[SCOPE] {main_folder_count} main / {float_folder_count} float "
          f"folders, {png_paths_len} frames, content {IPS} ips")
    if mix_hz:
        _outer_hz = (1.0 - mix_duty) * mix_hz / 3.0
        print(f"[SCOPE] four-way mix duty {mix_duty:.2f} "
              f"({_outer_hz:.0f} vector + {mix_duty * mix_hz:.0f} raster + "
              f"{_outer_hz:.0f} stochastic + {_outer_hz:.0f} stipple "
              "passes/sec)")
        print(f"[SCOPE] mix raster: {scope.samples_per_frame * fields} samples "
              f"per picture across {fields} trace(s). Baseline for comparison "
              f"is {int(scope.samplerate / max(IPS, 1))} at --scope-fps {IPS} "
              "with no mix; the shortfall is the beam time shared with vector, "
              "stochastic, and stipple.")
        print(f"[SCOPE] one complete mixed exposure spans {tap_traces} traces "
              f"({_trace_hz / tap_traces:.1f} combined pictures/sec)")
    elif use_fusion:
        _route = " -> ".join(fusion_components.upper())
        print(f"[SCOPE] fusion components {fusion_components.upper()}: "
              f"luminance-weighted position mux {_route} -> ...")
    if lowpass:
        print(f"[SCOPE] low-pass {lowpass:.0f} Hz in the output path "
              "(emulating a softer DAC / RC filter)")
    if fps < IPS and not mix_hz:
        print(f"[SCOPE] note: {fps} traces/sec < {IPS} ips, so some indices are "
              "skipped -- still on time, never late")
    if (sweep_mode == "alternate" and fps > IPS and not mix_hz
            and render_mode == "vector"):
        # Raster no longer has this problem: it renders a fresh chained frame
        # for every trace, so a one-way sweep always continues rather than
        # looping back over an unbudgeted jump.  Vector still emits per index.
        print(f"[SCOPE] note: {fps} traces/sec > {IPS} ips means vector traces "
              "repeat, and a repeated one-way sweep shows a flyback. "
              "Consider SCOPE_SWEEP='palindrome'.")

    # Measure the real per-frame cost once, so a slow machine says so up front
    # instead of quietly dropping traces.  The budget is one index period.
    if use_raster or use_stochastic or use_stipple or use_fusion or mix_hz:
        try:
            import time as _t
            ml0 = next((l for l in main_libs if l is not None), None)
            fl0 = next((l for l in float_libs if l is not None), None)
            if ml0 is not None:
                _frame_count = min(
                    len(lib) for lib in (ml0, fl0) if lib is not None)
                _probes = []

                def _measure(label, emit_fn):
                    for _k in range(3):
                        emit_fn(_k % _frame_count)
                    _t0 = _t.perf_counter()
                    for _k in range(10):
                        emit_fn(_k % _frame_count)
                    _probes.append(
                        (label, (_t.perf_counter() - _t0) / 10 * 1000.0))

                # Probe every path that can really be scheduled. The old mix
                # probe timed raster only and could miss an over-budget
                # stochastic pass entirely.
                if use_raster or (mix_hz and mix_duty > 0.0):
                    _raster_probe = TraceEmitter(
                        scope.samplerate, scope.samples_per_frame,
                        gamma=gamma, trim=trim, density=density, rows=rows,
                        fields=fields, border=border, oversample=oversample,
                        sweep=sweep_mode, autofit=autofit, row_bias=row_bias,
                        precondition=raster_precondition,
                        yt_timing=yt_timing,
                        yt_trigger_samples=0,   # Scope prepends the marker its own window
                        grid=((cal["grid_rows"], cal["grid_cols"])
                              if cal else None),
                        levels=(cal.get("levels") if cal else None),
                        geometry_samples=geometry_samples,
                        traversal_hz=traversal_hz)
                    _measure("raster", lambda k: _raster_probe.emit(
                        composite_luma(ml0, k, fl0, k, invert=invert)))

                if use_stochastic or (mix_hz and mix_duty < 1.0):
                    _stochastic_probe = StochasticEmitter(
                        scope.samplerate, scope.samples_per_frame,
                        gamma=walk_gamma, trim=trim, radius=walk_radius,
                        stride=walk_stride, edge_gain=walk_edge,
                        reseed_ms=walk_reseed_ms, walk_hz=walk_hz,
                        dc_comp=dc_comp, border=border)
                    _measure("stochastic", lambda k: _stochastic_probe.emit(
                        composite_luma(
                            ml0, k, fl0, k, raw=True, invert=invert)))

                if use_stipple or (mix_hz and mix_duty < 1.0):
                    _stipple_probe = StippleEmitter(
                        scope.samplerate, scope.samples_per_frame,
                        points=stipple_points, gamma=walk_gamma, trim=trim,
                        edge_gain=walk_edge, dc_comp=dc_comp, border=border)
                    def _probe_stipple(k):
                        cloud = composite_stipple_candidates(
                            ml0, k, fl0, k, invert=invert)
                        return (_stipple_probe.emit_candidates(cloud)
                                if cloud is not None else _stipple_probe.emit(
                                    composite_luma(
                                        ml0, k, fl0, k, raw=True,
                                        invert=invert)))
                    _measure("stipple", _probe_stipple)

                if use_fusion:
                    _fusion_raster = TraceEmitter(
                        scope.samplerate, scope.samples_per_frame,
                        gamma=gamma, trim=trim, density=density, rows=rows,
                        fields=fields, border=0.0, oversample=oversample,
                        sweep=sweep_mode, autofit=autofit, row_bias=row_bias,
                        precondition=raster_precondition,
                        grid=((cal["grid_rows"], cal["grid_cols"])
                              if cal else None),
                        levels=(cal.get("levels") if cal else None),
                        geometry_samples=geometry_samples,
                        traversal_hz=traversal_hz)
                    _fusion_stochastic = StochasticEmitter(
                        scope.samplerate, scope.samples_per_frame,
                        gamma=walk_gamma, trim=trim,
                        radius=walk_radius, stride=walk_stride,
                        edge_gain=walk_edge,
                        reseed_ms=walk_reseed_ms, walk_hz=walk_hz,
                        dc_comp=None, border=0.0)
                    _fusion_mux = PositionMultiplexer()

                    def _fusion_probe(k):
                        lum = composite_luma(
                            ml0, k, fl0, k, raw=True, invert=invert)
                        vector = (rasterize(
                            merge(ml0, k, fl0, k,
                                  min_feature=min_feature),
                            scope.samples_per_frame)
                                  if "v" in fusion_components else None)
                        raster = (_fusion_raster.emit(composite_luma(
                            ml0, k, fl0, k, invert=invert))
                                  if "r" in fusion_components else None)
                        stochastic = (_fusion_stochastic.emit(lum)
                                      if "s" in fusion_components else None)
                        weights = {
                            "v": trace_luminance_weights(
                                lum, vector, gamma=gamma, trim=trim),
                            "r": trace_luminance_weights(
                                lum, raster, gamma=gamma, trim=trim),
                            "s": trace_luminance_weights(
                                lum, stochastic, gamma=walk_gamma, trim=trim),
                        }
                        return _fusion_mux.emit(
                            vector=vector, raster=raster,
                            stochastic=stochastic,
                            components=fusion_components, weights=weights)

                    _measure("fusion", _fusion_probe)

                if mix_hz and mix_duty < 1.0:
                    from scope_out import rasterize as _rasterize
                    def _probe_vector(k):
                        vector = _rasterize(
                            merge(ml0, k, fl0, k,
                                  min_feature=min_feature),
                            scope.samples_per_frame)
                        if invert:
                            lum = composite_luma(
                                ml0, k, fl0, k, raw=True, invert=True)
                            weights = trace_luminance_weights(
                                lum, vector, gamma=gamma, trim=trim)
                            vector = retime_trace_by_weights(vector, weights)
                        return vector
                    _measure("vector", _probe_vector)

                _budget = 1000.0 / max(scope.samplerate / scope.samples_per_frame, 1)
                _costs = ", ".join(f"{name} {ms:.1f} ms"
                                   for name, ms in _probes)
                _worst = max((ms for _name, ms in _probes), default=0.0)
                print(f"[SCOPE] trace costs: {_costs}; budget {_budget:.1f} ms "
                      f"(worst {_worst / _budget:.0%})")
                if _worst > 0.7 * _budget:
                    if use_stipple or (mix_hz and mix_duty < 1.0):
                        print("[SCOPE] stipple is CPU-tight: lower "
                              "--scope-stipple-points or --scope-fps. The "
                              "current index clock stays authoritative, so "
                              "an over-budget route repeats the previous "
                              "trace rather than drifting late")
                    elif use_fusion:
                        print("[SCOPE] fusion is CPU-tight: try --scope-fusion "
                              "sr, or lower --scope-walk-hz before changing "
                              "the DAC rate")
                    else:
                        print("[SCOPE] tight: raise --scope-density, lower "
                              "--scope-fields, or drop --scope-oversample")
        except Exception as e:
            print(f"[SCOPE] startup performance probe skipped ({e})")

    # --- live controls ---
    # Everything below is adjustable while watching the scope; restarting to
    # try a different trim is useless when the thing you are judging is a beam.
    live_state = dict(trim=trim, density=density,
                      gamma=(walk_gamma
                             if (use_stochastic or use_stipple
                                 or (use_fusion and "s" in fusion_components))
                             else gamma),
                      raster_gamma=gamma, stochastic_gamma=walk_gamma, rows=rows,
                      lowpass=lowpass, mode=render_mode, raster=use_raster,
                      sweep=sweep_mode, autofit=autofit,
                      invert=invert, rotation=rotation, mirror=mirror,
                      x_only=x_only,
                      yt=trigger_on,
                      yt_trigger_us=scope.yt_trigger_us,
                      trigger_shape=trigger_shape,
                      yt_timing=yt_timing,
                      precondition=raster_precondition,
                      mode_locked=bool(realtime or mix_hz
                                       or geometry_samples is not None
                                       or traversal_hz is not None),
                      clock_locked=bool(_index_calculator.midi_mode),
                      audio_muted=gui_enabled,
                      mix_hz=mix_hz, mix_duty=mix_duty,
                      fps=fps, ips=IPS, fields=fields,
                      available_modes=(
                          ("raster", "stochastic", "stipple")
                          if scope_source == "images"
                          else ("vector", "raster", "stochastic", "stipple",
                                "fusion")),
                      stipple_points=stipple_points,
                      fusion_components=fusion_components)
    # --- monitoring ---
    # Same two-part contract every other mode uses: main.py starts the server,
    # the engine feeds lightweight_monitor. This remains useful in headless
    # runs and alongside the optional native tuner.
    # The web page is the only display many people will have, so its preview
    # must show a whole picture, not one interlaced field.
    try:
        scope.set_tap_fields(tap_traces)
    except Exception:
        pass

    monitor = None
    try:
        from lightweight_monitor import start_monitor, monitor_data
        monitor = start_monitor()
        monitor_data["scope_device"] = _dev_name
        monitor_data["scope_x_only"] = x_only
        # Static: enumerated once. The page needs it to build the selector, and
        # putting it in /data avoids a second endpoint.
        try:
            from scope_out import list_output_devices, default_output_index
            _dflt = default_output_index()
            monitor_data["scope_devices"] = [
                {"index": i, "name": nm, "api": api, "rate": rate,
                 "default": (i == _dflt)}
                for (i, nm, api, rate) in list_output_devices(
                    min_channels=required_output_channels(
                        channel_pair, x_only))]
        except Exception:
            monitor_data["scope_devices"] = []
        monitor_data["scope_mode"] = mode_name + (" REALTIME" if realtime else "")
        monitor_data["scope_samplerate"] = int(scope.samplerate)
        monitor_data["scope_samples_per_trace"] = int(scope.samples_per_frame)
        monitor_data["scope_fields"] = int(fields)
        monitor_data["scope_raster_gamma"] = round(gamma, 2)
        monitor_data["scope_stochastic_gamma"] = round(walk_gamma, 2)
        monitor_data["scope_fusion"] = fusion_components if use_fusion else None
        monitor_data["scope_invert"] = invert
        monitor_data["scope_rotation"] = rotation
        monitor_data["scope_mirror"] = mirror
        monitor_data["scope_trigger"] = trigger_on
        monitor_data["scope_trigger_shape"] = trigger_shape
        monitor_data["scope_yt_trigger_us"] = scope.yt_trigger_us
        monitor_data["scope_yt_timing"] = yt_timing
        monitor_data["scope_border"] = border
        monitor_data["scope_yt_grid"] = _rotation_grid(cal, rotation) if cal else None
        monitor_data["scope_yt_levels"] = cal.get("levels") if cal else None
        monitor_data["scope_gamma"] = round(
            walk_gamma
            if ((use_stochastic or use_stipple
                 or (use_fusion and "s" in fusion_components))
                and not mix_hz) else gamma, 2)
        monitor_data["scope_refresh_hz"] = round(
            scope.samplerate / max(scope.trace_samples, 1), 1)
        monitor_data["scope_picture_hz"] = round(
            scope.samplerate / max(scope.trace_samples * tap_traces, 1), 1)
        # Seed both counters so the dashboard shows a 0 from the first poll
        # rather than an empty field that looks like the metric is missing.
        monitor_data["scope_beam_parked"] = int(scope.beams_unparked)
        monitor_data["scope_dac_dropouts"] = int(scope.dac_dropouts)
        if cal:
            monitor_data["scope_grid"] = f"{cal['grid_cols']}x{cal['grid_rows']}"
            monitor_data["scope_samples_per_cell"] = round(
                scope.samples_per_frame * fields
                / max(cal["grid_rows"] * cal["grid_cols"], 1), 2)
    except Exception as e:
        print(f"[SCOPE] monitor unavailable ({e})")

    keys = term = None
    try:
        from scope_controls import KeyMap, Terminal, as_flags
        term = Terminal()
        if term.enabled or gui_enabled:
            keys = KeyMap(live_state)
        if term.enabled:
            print("[SCOPE] live controls active -- h for keys, p to print "
                  "flags, q to quit")
    except Exception:
        pass

    # --- loop ---
    # Poll well inside a trace period: the raster path hands over one frame per
    # trace and has to notice the handover promptly, or the beam re-runs the
    # frame it already drew.
    tick = 1.0 / max(2 * IPS, 4 * fps)
    prev_index = -1
    prev_key = None
    prefetched_key = None
    prefetched_direction = 1
    source_selected_key = None
    source_selected_at_ns = None
    source_direction = 1
    mix_scheduler = TriangleMixScheduler(mix_duty)
    mix_last_mode = None
    # One emitter, shared with scope_screen.py. Tuning and sweep state live on
    # the object so neither caller keeps its own copy -- that divergence is
    # what test_scope_parity.py exists to catch.
    emitter = TraceEmitter(
        scope.samplerate, scope.samples_per_frame,
        gamma=gamma, trim=trim, density=density, rows=rows, fields=fields,
        border=border, oversample=oversample, sweep=sweep_mode,
        dc_comp=dc_comp, autofit=autofit, row_bias=row_bias,
        precondition=raster_precondition,
        yt_timing=yt_timing,
        close_frame=not scope.trigger,
        yt_trigger_samples=0,   # Scope prepends the marker its own window
        grid=_rotation_grid(cal, rotation),
        levels=(cal.get("levels") if cal else None),
        geometry_samples=geometry_samples, traversal_hz=traversal_hz)
    _configure_raster_emitter_fields(emitter, render_mode, fields)
    stochastic_emitter = StochasticEmitter(
        scope.samplerate, scope.samples_per_frame,
        gamma=walk_gamma, trim=trim, radius=walk_radius, stride=walk_stride,
        edge_gain=walk_edge, reseed_ms=walk_reseed_ms, walk_hz=walk_hz,
        dc_comp=dc_comp, border=border)
    stipple_emitter = StippleEmitter(
        scope.samplerate, scope.samples_per_frame,
        points=stipple_points, gamma=walk_gamma, trim=trim,
        edge_gain=walk_edge, dc_comp=dc_comp, border=border)
    fusion_multiplexer = PositionMultiplexer()

    prepared_cache = PreparedImageCache()

    field_group = FieldGroupLatch(fields)
    mix_field_group = FieldGroupLatch(fields)
    sweep = {"rev": False, "end": None}
    beam_end = None                 # actual last sample handed to the DAC
    render_generation = 0
    requested_presentation = None
    last_report = time.time()
    last_monitor = 0.0
    last_monitor_adopted = 0
    last_source_metrics_at = time.monotonic()
    last_source_adoptions = int(scope.distinct_source_adoptions)

    def clear_frame_requests(wait=True):
        """Cancel queued work and force the current key to be republished."""
        nonlocal prev_key
        if frame_producer is not None:
            frame_producer.clear(wait=wait)
        prev_key = None

    def apply_timing_request(request):
        """Reopen the stream and rebuild timing-dependent render state."""
        nonlocal scope, cal, sweep, fps, samples, fields, tap_traces, IPS
        nonlocal tick, emitter, stochastic_emitter, stipple_emitter
        nonlocal fusion_multiplexer, beam_end
        nonlocal mix_last_mode, prev_key, dev, _dev_name, render_generation

        # A queued job retains its Scope and renderer objects. Drain it before
        # replacing either, so the producer can never publish to a closed
        # stream or carry state across a new sample budget.
        clear_frame_requests(wait=True)

        next_ips = int(round(request["ips"] if request["ips"] is not None
                             else IPS))
        next_fields = int(request["fields"] if request["fields"] is not None
                           else fields)
        if request["fps"] is not None:
            next_fps = int(round(request["fps"]))
        elif request["fields"] is not None and next_fields > 1:
            next_fps = next_ips * next_fields
        else:
            next_fps = fps
        if next_fields > 1 and next_fps != next_ips * next_fields:
            # A raster field must land on a stable source-image boundary.
            next_fields = 1
        if next_fps < 1 or next_ips < 1 or not 1 <= next_fields <= 4:
            raise ValueError("invalid picture rate, trace rate, or field count")
        if traversal_hz is not None and next_fields != 1:
            raise ValueError("timed raster traversal requires one field")

        # The timing request is a new sample budget, so an old explicit
        # --scope-samples value must no longer override the requested rate.
        next_samples = None
        new_scope, new_cal, new_sweep = _swap_device(
            scope, dev, source, next_fps, next_samples,
            main_libs, float_libs, density, trim, rows, next_fields,
            row_bias, autofit, invert, x_only=x_only,
            channel_pair=channel_pair, resolved_device=True,
            geometry_budget=(geometry_samples or (3200 if traversal_hz is not None else None)))
        scope, cal, sweep = new_scope, new_cal, new_sweep
        fps, samples, fields, IPS = next_fps, next_samples, next_fields, next_ips
        settings.IPS = IPS
        settings.SCOPE_FPS = fps
        settings.SCOPE_SAMPLES = None
        settings.SCOPE_FIELDS = fields
        settings.SCOPE_FIELDS_EXPLICIT = True
        try:
            _index_calculator.IPS = IPS
        except Exception:
            pass
        tap_traces = fields if use_raster else 1
        scope.set_tap_fields(tap_traces)
        tick = 1.0 / max(2 * IPS, 4 * fps)

        emitter = TraceEmitter(
            scope.samplerate, scope.samples_per_frame,
            gamma=gamma, trim=trim, density=density, rows=rows,
            fields=fields, border=border, oversample=oversample,
            sweep=sweep_mode, dc_comp=dc_comp, autofit=autofit,
            row_bias=row_bias, precondition=raster_precondition,
            yt_timing=yt_timing, yt_trigger_samples=0,
            close_frame=not scope.trigger,
            grid=_rotation_grid(cal, rotation),
            levels=(cal.get("levels") if cal else None),
            geometry_samples=geometry_samples, traversal_hz=traversal_hz)
        _configure_raster_emitter_fields(emitter, render_mode, fields)
        stochastic_emitter = StochasticEmitter(
            scope.samplerate, scope.samples_per_frame,
            gamma=walk_gamma, trim=trim, radius=walk_radius,
            stride=walk_stride, edge_gain=walk_edge,
            reseed_ms=walk_reseed_ms, walk_hz=walk_hz,
            dc_comp=dc_comp, border=border)
        stipple_emitter = StippleEmitter(
            scope.samplerate, scope.samples_per_frame,
            points=stipple_points, gamma=walk_gamma, trim=trim,
            edge_gain=walk_edge, dc_comp=dc_comp, border=border)
        fusion_multiplexer.reset()
        beam_end = None
        _configure_field_groups(field_group, mix_field_group, fields)
        mix_last_mode = None
        prev_key = None
        render_generation += 1
        dev = "null" if getattr(scope, "null", False) else scope.stream.device
        _dev_name = _dev_name_of(scope)
        live_state.update(fps=fps, ips=IPS, fields=fields)
        with _device_lock:
            _timing_request["message"] = (
                f"applied: {IPS} IPS · {fps} traces/s · {fields} fields")
        try:
            from lightweight_monitor import monitor_data as _md_timing
            _md_timing["scope_device"] = _dev_name
            _md_timing["scope_samplerate"] = int(scope.samplerate)
            _md_timing["scope_samples_per_trace"] = int(scope.samples_per_frame)
            _md_timing["scope_fields"] = int(fields)
            _md_timing["scope_refresh_hz"] = round(
                scope.samplerate / max(scope.trace_samples, 1), 1)
            _md_timing["scope_picture_hz"] = round(
                scope.samplerate / max(scope.trace_samples * tap_traces, 1), 1)
            if cal:
                _md_timing["scope_grid"] = (f"{cal['grid_cols']}x"
                                             f"{cal['grid_rows']}")
                _md_timing["scope_samples_per_cell"] = round(
                    scope.samples_per_frame * fields
                    / max(cal["grid_rows"] * cal["grid_cols"], 1), 2)
        except Exception:
            pass

    def presentation_identity(request, mode, field, decoded_pair=None):
        index, main_folder, float_folder = request["key"]
        source_kind = "runtime-images" if live_images is not None else "bake"
        return (request["generation"], mode, int(index), int(main_folder),
                int(float_folder), int(field), int(request["fields"]),
                source_kind, request.get("source_selected_at_ns"),
                (decoded_pair.requested_at_ns
                 if decoded_pair is not None else None),
                (decoded_pair.decode_started_at_ns
                 if decoded_pair is not None else None),
                (decoded_pair.ready_at_ns
                 if decoded_pair is not None else None))

    def commit_accepted_trace(request, endpoint, mode, *, group=None,
                              mix_mode=None):
        """Commit producer progression only after Scope queued a trace."""
        nonlocal beam_end, mix_last_mode
        if endpoint is None:
            endpoint = request["scope"].last_accepted_endpoint
        if endpoint is not None:
            beam_end = np.asarray(endpoint, dtype=np.float32)[:2].copy()
        has_raster = (mode == "raster" or
                      (mode == "fusion"
                       and "r" in request["fusion_components"]))
        if has_raster and beam_end is not None:
            request["emitter"].accept(beam_end)
            request["sweep"]["rev"] = request["emitter"]._rev
            request["sweep"]["end"] = request["emitter"]._end
        if group is not None:
            group.accept()
        if mix_mode is not None:
            mix_scheduler.advance()
            if mix_mode == "raster":
                mix_field_group.accept()
            mix_last_mode = mix_mode

    def checkpoint_render_state(request, mode):
        """Checkpoint mutable non-raster emitters for failed queue recovery."""
        checkpoints = []
        if mode == "stochastic":
            emitter = request["stochastic_emitter"]
            checkpoints.append((emitter, emitter.checkpoint()))
        elif mode == "stipple":
            emitter = request["stipple_emitter"]
            checkpoints.append((emitter, emitter.checkpoint()))
        elif mode == "fusion":
            if "s" in request["fusion_components"]:
                emitter = request["stochastic_emitter"]
                checkpoints.append((emitter, emitter.checkpoint()))
            mux = request["fusion_multiplexer"]
            checkpoints.append((mux, mux.checkpoint()))
        return checkpoints

    def restore_render_state(checkpoints):
        for renderer, checkpoint in reversed(checkpoints):
            renderer.restore(checkpoint)

    def render_frame_request(request):
        """Render latest state; progression commits only on queueing."""

        if request["mix"]:
            mix_mode = mix_scheduler.peek_next_mode()
            active = request
            field = 0
            if mix_mode == "raster":
                field, active = mix_field_group.begin(request)
            active_scope = active["scope"]
            decoded_pair = None
            if live_images is not None:
                decoded_pair = live_images.ready_pair(
                    active["index"], active["key"][1], active["key"][2])
                if decoded_pair is None:
                    return
            state_checkpoint = checkpoint_render_state(active, mix_mode)

            def render_active():
                return _emit(
                    active["scope"], active["ml"], active["fl"],
                    active["index"], mix_mode, active["sweep"],
                    active["sweep_mode"], active["gamma"], active["trim"],
                    active["density"], active["rows"],
                    active["min_feature"], active["autofit"],
                    active["lowpass"], active["oversample"], active["cal"],
                    dc_comp=active["dc_comp"], border=active["border"],
                    invert=active["invert"], emitter=active["emitter"],
                    stochastic_emitter=active["stochastic_emitter"],
                    stipple_emitter=active["stipple_emitter"],
                    beam_start=beam_end,
                    mode_handoff=(mix_last_mode is not None
                                  and mix_mode != mix_last_mode),
                    field=field, fields=active["fields"],
                    rotation=active["rotation"],
                    presentation=presentation_identity(
                        active, mix_mode, field, decoded_pair),
                    # Whole-application P0 measurements showed no gain from
                    # caching stochastic preparation in its walk-dominated
                    # renderer. Keep the shared cache for raster/stipple and
                    # for the other reusable stages in a fused frame.
                    prepared_cache=(None if mix_mode == "stochastic"
                                    else prepared_cache),
                    source_thumbnails=(
                        (decoded_pair.main, decoded_pair.floating)
                        if decoded_pair is not None else None),
                    defer_state=(mix_mode == "raster"))

            def commit_active(endpoint):
                commit_accepted_trace(
                    active, endpoint, mix_mode, mix_mode=mix_mode)

            _execute_frame_transaction(
                active_scope, render_active, state_checkpoint,
                restore_render_state, commit_active)
            return

        mode = request["mode"]
        active = request
        field = 0
        if mode == "raster":
            field, active = field_group.begin(request)
        active_scope = active["scope"]
        decoded_pair = None
        if live_images is not None:
            decoded_pair = live_images.ready_pair(
                active["index"], active["key"][1], active["key"][2])
            if decoded_pair is None:
                return
        state_checkpoint = checkpoint_render_state(active, mode)

        def render_active():
            return _emit(
                active["scope"], active["ml"], active["fl"],
                active["index"], mode, active["sweep"],
                active["sweep_mode"], active["gamma"], active["trim"],
                active["density"], active["rows"], active["min_feature"],
                active["autofit"], active["lowpass"],
                active["oversample"], active["cal"],
                dc_comp=active["dc_comp"], border=active["border"],
                invert=active["invert"], emitter=active["emitter"],
                stochastic_emitter=active["stochastic_emitter"],
                stipple_emitter=active["stipple_emitter"],
                fusion_multiplexer=active["fusion_multiplexer"],
                beam_start=beam_end,
                fusion_components=active["fusion_components"],
                stochastic_gamma=active["stochastic_gamma"],
                stochastic_edge=active["stochastic_edge"], field=field,
                fields=active["fields"], rotation=active["rotation"],
                presentation=presentation_identity(
                    active, mode, field, decoded_pair),
                # The random walk dominates stochastic render cost; measured
                # cache hits did not improve end-to-end timing and added a
                # small cost. Keep its preparation uncached in application mode.
                prepared_cache=(None if mode == "stochastic"
                                else prepared_cache),
                source_thumbnails=(
                    (decoded_pair.main, decoded_pair.floating)
                    if decoded_pair is not None else None),
                defer_state=(mode in ("raster", "fusion")))

        def commit_active(endpoint):
            commit_accepted_trace(
                active, endpoint, mode,
                group=field_group if mode == "raster" else None)

        _execute_frame_transaction(
            active_scope, render_active, state_checkpoint,
            restore_render_state, commit_active)

    def publish_frame_request(ml, fl, index, key, selected_at_ns):
        """Publish an immutable view of the latest desired frame state."""
        request = MappingProxyType({
            "scope": scope, "ml": ml, "fl": fl, "index": index,
            "key": key, "generation": render_generation,
            "source_selected_at_ns": selected_at_ns,
            "mode": render_mode, "mix": bool(mix_hz), "sweep": sweep,
            "sweep_mode": sweep_mode, "gamma": gamma, "trim": trim,
            "density": density, "rows": rows,
            "min_feature": min_feature, "autofit": autofit,
            "lowpass": lowpass, "oversample": oversample, "cal": cal,
            "dc_comp": dc_comp, "border": border, "invert": invert,
            "emitter": emitter, "stochastic_emitter": stochastic_emitter,
            "stipple_emitter": stipple_emitter,
            "fusion_multiplexer": fusion_multiplexer,
            "fusion_components": fusion_components,
            "stochastic_gamma": walk_gamma, "stochastic_edge": walk_edge,
            "fields": fields, "rotation": rotation,
        })
        repeat = bool(mix_hz or render_mode != "vector")
        version = (
            render_generation, key, render_mode, bool(mix_hz), sweep_mode,
            gamma, trim, density, rows, min_feature, bool(autofit),
            lowpass, oversample, dc_comp, border, bool(invert), fields,
            fusion_components, walk_gamma, walk_radius, walk_stride,
            walk_edge, walk_reseed_ms, walk_hz, stipple_points, rotation,
            mix_duty)
        request = MappingProxyType({**request, "version": version})
        frame_producer.publish(
            scope, lambda request=request: render_frame_request(request),
            version=request["version"], repeat=repeat)

    # NOT `with scope:` -- the device can be changed at runtime, which means
    # rebinding `scope`.  A with-block would call __exit__ on the object it
    # entered, i.e. the already-closed old stream, and raise on the way out.
    if lowpass and lowpass_circular is not None:
        # Stochastic/fusion modes use a stateful causal filter. Compile its
        # fixed array signature before the callback starts, not on the first
        # filtered picture after a live mode switch.
        from scope_lowpass import warm_cascaded_one_pole
        warm_cascaded_one_pole(order=4, channels=2)
    # GUI and producer startup are inside the cleanup scope so a partial start
    # cannot leave either a PortAudio stream or a worker behind.
    frame_producer = None
    gui = None
    try:
        scope.stream.start()
        frame_producer = ScopeFrameScheduler()
        if gui_enabled:
            from scope_gui import ScopeGUI
            gui = ScopeGUI(live_state, start_image_only=start_image_only,
                           start_fullscreen=start_fullscreen)
        while True:
            if gui is not None:
                try:
                    _scope_gui_metrics = {
                        "device": _dev_name,
                        "sample_rate": int(scope.samplerate),
                        "trace_hz": scope.samplerate / max(scope.trace_samples, 1),
                        "picture_hz": scope.samplerate / max(
                            scope.trace_samples * max(tap_traces, 1), 1),
                        "samples": int(scope.samples_per_frame),
                        "fields": int(fields),
                        "grid": (f"{cal['grid_cols']}x{cal['grid_rows']}"
                                 if cal else "--"),
                        "dropouts": int(scope.dac_dropouts),
                        "underruns": int(getattr(source, "underruns", 0)
                                         if source is not None else 0),
                        "buffered_samples": (
                            source.buffered_samples
                            if source is not None and
                            hasattr(source, "buffered_samples") else
                            (scope.trace_samples
                             if not scope.ready() else 0)),
                        "buffer_capacity_samples": (
                            source.capacity
                            if source is not None and hasattr(source, "capacity")
                            else scope.trace_samples),
                        "buffer_kind": ("source" if source is not None
                                        and hasattr(source, "buffered_samples")
                                        else "trace queue"),
                    }
                    _scope_gui_metrics["buffered_ms"] = (
                        1000.0 * _scope_gui_metrics["buffered_samples"]
                        / max(scope.samplerate, 1))
                    _scope_gui_metrics["buffer_capacity_ms"] = (
                        1000.0 * _scope_gui_metrics["buffer_capacity_samples"]
                        / max(scope.samplerate, 1))
                    _stream_latency = getattr(scope.stream, "latency", 0.0)
                    if isinstance(_stream_latency, (tuple, list)):
                        _stream_latency = _stream_latency[-1] if _stream_latency else 0.0
                    try:
                        _scope_gui_metrics["dac_latency_ms"] = (
                            1000.0 * float(_stream_latency))
                    except (TypeError, ValueError):
                        _scope_gui_metrics["dac_latency_ms"] = 0.0
                    _gui_actions = gui.poll(live_state, _scope_gui_metrics)
                except Exception as e:
                    print(f"[SCOPE] GUI update failed: {e}", flush=True)
                    _gui_actions = []
                if gui.close_requested:
                    break
                for _action in _gui_actions:
                    if _action[0] == "slider":
                        _name, _value = _action[1], _action[2]
                        if _name == "ips":
                            _ips = int(round(_value))
                            _fields = fields if use_raster else 1
                            _fps = (_ips * _fields if _fields > 1 else fps)
                            request_timing(fps=_fps, fields=_fields, ips=_ips)
                            gui.message = "Applying image clock / trace timing…"
                        elif _name == "fps":
                            request_timing(fps=int(round(_value)),
                                           fields=None)
                            gui.message = "Reopening output for the new trace rate…"
                        elif _name == "fields":
                            _fields = int(round(_value))
                            request_timing(fps=int(round(IPS * _fields)),
                                           fields=_fields)
                            gui.message = "Reopening output for the new field rate…"
                        else:
                            keys.set_value(_name, _value)
                    elif _action[0] == "mode":
                        keys.set_mode(_action[1])
                    elif _action[0] == "audio":
                        audible = bool(_action[1])
                        scope.set_output_audio(muted=not audible)
                        live_state["audio_muted"] = not audible
                        gui.message = (
                            "XY DAC output enabled"
                            if audible else
                            "XY DAC muted · physical scope holds center dot")
                    elif _action[0] == "fullscreen":
                        gui.set_fullscreen(_action[1])
                    elif _action[0] == "image_only":
                        gui.set_image_only(_action[1])
                    elif _action[0] == "key":
                        keys.feed(_action[1])

            # --- pending device change, parked by request_device() ----------
            with _device_lock:
                _want = (_device_request["spec"]
                         if _device_request["pending"] else False)
                if _device_request["pending"]:
                    _device_request["pending"] = False
            if _want is not False and realtime:
                # BufferedSource wraps a generator built around the CURRENT
                # samples_per_frame.  A new device can have a different default
                # sample rate, which changes that -- and the generator would go
                # on emitting the old length.  Refusing is honest; swapping
                # would look like it worked and drift.
                _msg = ("device change is not available in realtime mode "
                        "(restart with --device instead)")
                print(f"[SCOPE] {_msg}", flush=True)
                with _device_lock:
                    _device_request["message"] = f"failed: {_msg}"
                _want = False
            if _want is not False:
                try:
                    clear_frame_requests(wait=True)
                    scope, cal, sweep = _swap_device(
                        scope, _want, source, fps, samples,
                        main_libs, float_libs, density, trim, rows, fields,
                        row_bias, autofit, invert, x_only=x_only,
                        channel_pair=channel_pair,
                        geometry_budget=(geometry_samples or (3200 if traversal_hz is not None else None)))
                    dev = ("null" if getattr(scope, "null", False)
                           else scope.stream.device)
                    # The emitter owns the chain and the geometry, so it has to
                    # be rebuilt, not just reset: a new device can mean a new
                    # sample rate, which changes samples_per_frame and with it
                    # the grid. Reusing the old one would keep drawing to the
                    # previous device's budget.
                    emitter = TraceEmitter(
                        scope.samplerate, scope.samples_per_frame,
                        gamma=gamma, trim=trim, density=density, rows=rows,
                        fields=fields, border=border, oversample=oversample,
                        sweep=sweep_mode, dc_comp=dc_comp, autofit=autofit,
                        row_bias=row_bias, precondition=raster_precondition,
                        yt_timing=yt_timing,
                        close_frame=not scope.trigger,
                        yt_trigger_samples=0,   # Scope prepends the marker its own window
                        grid=_rotation_grid(cal, rotation),
                        levels=(cal.get("levels") if cal else None),
                        geometry_samples=geometry_samples,
                        traversal_hz=traversal_hz)
                    _configure_raster_emitter_fields(emitter, render_mode, fields)
                    stochastic_emitter = StochasticEmitter(
                        scope.samplerate, scope.samples_per_frame,
                        gamma=walk_gamma, trim=trim, radius=walk_radius,
                        stride=walk_stride, edge_gain=walk_edge,
                        reseed_ms=walk_reseed_ms, walk_hz=walk_hz,
                        dc_comp=dc_comp, border=border)
                    stipple_emitter = StippleEmitter(
                        scope.samplerate, scope.samples_per_frame,
                        points=stipple_points, gamma=walk_gamma, trim=trim,
                        edge_gain=walk_edge, dc_comp=dc_comp, border=border)
                    fusion_multiplexer.reset()
                    beam_end = None
                    _configure_field_groups(field_group, mix_field_group, fields)
                    mix_last_mode = None
                    render_generation += 1
                    with _device_lock:
                        _device_request["message"] = f"now on {_dev_name_of(scope)}"
                    _dev_name = _dev_name_of(scope)
                    try:
                        from lightweight_monitor import monitor_data as _md2
                        _md2["scope_device"] = _dev_name
                        _md2["scope_samplerate"] = int(scope.samplerate)
                        _md2["scope_samples_per_trace"] = int(scope.samples_per_frame)
                        # A new device can mean a new sample rate, so these
                        # three move too.  Leaving them stale made the page
                        # report the OLD refresh rate against the NEW device,
                        # which is worse than reporting nothing.
                        _md2["scope_refresh_hz"] = round(
                            scope.samplerate / max(scope.trace_samples, 1), 1)
                        _md2["scope_picture_hz"] = round(
                            scope.samplerate
                            / max(scope.trace_samples * tap_traces, 1), 1)
                        if cal:
                            _md2["scope_grid"] = (f"{cal['grid_cols']}x"
                                                  f"{cal['grid_rows']}")
                            _md2["scope_samples_per_cell"] = round(
                                scope.samples_per_frame * fields
                                / max(cal["grid_rows"] * cal["grid_cols"], 1), 2)
                    except Exception:
                        pass
                    prev_key = None          # force a redraw on the new stream
                except Exception as e:
                    print(f"[SCOPE] device change failed: {e}", flush=True)
                    with _device_lock:
                        _device_request["message"] = f"failed: {e}"
            with _device_lock:
                _timing_want = (dict(_timing_request)
                                if _timing_request["pending"] else None)
                if _timing_request["pending"]:
                    _timing_request["pending"] = False
            if _timing_want is not None:
                if realtime or mix_hz:
                    _reason = ("timing changes are unavailable in realtime/mix "
                               "mode; restart in frame mode")
                    with _device_lock:
                        _timing_request["message"] = f"failed: {_reason}"
                    if gui is not None:
                        gui.message = _reason
                elif (_timing_want.get("ips") is not None
                      and _index_calculator.midi_mode):
                    _reason = "picture rate follows the external MIDI clock"
                    with _device_lock:
                        _timing_request["message"] = f"failed: {_reason}"
                    if gui is not None:
                        gui.message = _reason
                else:
                    try:
                        apply_timing_request(_timing_want)
                        if gui is not None:
                            gui.message = timing_status()["message"]
                    except Exception as e:
                        _reason = f"timing change failed: {e}"
                        print(f"[SCOPE] {_reason}", flush=True)
                        with _device_lock:
                            _timing_request["message"] = _reason
                        if gui is not None:
                            gui.message = _reason
            mode_changed = False
            if keys is not None:
                for ch in term.read():
                    if keys.feed(ch) and keys.message:
                        print(keys.message, flush=True)
                        keys.message = ""
                if keys.quit:
                    break
                if keys.transform_dirty:
                    keys.transform_dirty = False
                    clear_frame_requests(wait=True)
                    rotation = _quarter_turn(live_state.get("rotation", 0))
                    # Rebuild in image space. The Scope output transform stays
                    # at zero so X remains the fast raster/trigger axis.
                    with frame_producer.state_lock:
                        scope.set_rotation(0)
                        # Mirror needs no rebuild -- it is a sign flip at the
                        # output -- but it rides the same dirty flag so one
                        # keypress cannot leave the two transforms disagreeing.
                        mirror = bool(live_state.get("mirror", False))
                        scope.set_mirror(mirror)
                        emitter.reset()
                        emitter.grid = _rotation_grid(cal, rotation)
                        stochastic_emitter.reset()
                        stipple_emitter.reset()
                        fusion_multiplexer.reset()
                        beam_end = None
                        sweep = {"rev": False, "end": None}
                        field_group.reset()
                        mix_field_group.reset()
                        mix_last_mode = None
                    if realtime:
                        gen.set_rotation(rotation,
                                         grid=_rotation_grid(cal, rotation))
                    try:
                        from lightweight_monitor import monitor_data as _md_rotate
                        _md_rotate["scope_rotation"] = rotation
                        _md_rotate["scope_mirror"] = mirror
                    except Exception:
                        pass
                    prev_key = None          # queue the current image rotated
                    render_generation += 1
                if keys.dirty or keys.changed:
                    _recalibrate = bool(keys.dirty)
                    keys.dirty = False
                    keys.changed = False
                    clear_frame_requests(wait=True)
                    # A tuning/configuration generation change must not finish
                    # half of the old interlace picture with the new grid or
                    # tone settings. Restart the group at field zero; source
                    # index changes alone continue to use the pinned group.
                    with frame_producer.state_lock:
                        field_group.reset()
                        mix_field_group.reset()
                        emitter._field = 0
                    trim = live_state["trim"]; density = live_state["density"]
                    rows = live_state["rows"]; autofit = live_state["autofit"]
                    next_mode = live_state["mode"]
                    mode_changed = next_mode != render_mode
                    next_fusion = live_state.get("fusion_components", "vrs")
                    fusion_changed = next_fusion != fusion_components
                    next_invert = bool(live_state.get("invert", False))
                    invert_changed = next_invert != invert
                    if mode_changed or fusion_changed or invert_changed:
                        # A chain endpoint belongs to the beam, not to a render
                        # mode. Do not resume either procedural emitter from the
                        # stale position it had before cycling away from it.
                        with frame_producer.state_lock:
                            emitter.reset()
                            stochastic_emitter.reset()
                            stipple_emitter.reset()
                            fusion_multiplexer.reset()
                            field_group.reset()
                            mix_field_group.reset()
                    invert = next_invert
                    render_mode = next_mode
                    use_raster = render_mode == "raster"
                    use_stochastic = render_mode == "stochastic"
                    use_stipple = render_mode == "stipple"
                    use_fusion = render_mode == "fusion"
                    fusion_components = next_fusion
                    if mode_changed:
                        tap_traces = fields if use_raster else 1
                        try:
                            scope.set_tap_fields(tap_traces)
                        except Exception:
                            pass
                    if _recalibrate and (use_raster or mix_hz):
                        try:
                            cal = calibrate(main_libs, float_libs,
                                            (geometry_samples or
                                             (3200 if traversal_hz is not None
                                              else scope.samples_per_frame)),
                                            density=density, trim=trim,
                                            rows=rows, fields=fields,
                                            row_bias=row_bias,
                                            autofit=autofit, invert=invert)
                            spc = scope.samples_per_frame * fields / max(
                                cal["grid_rows"] * cal["grid_cols"], 1)
                            print(f"  grid {cal['grid_cols']}x{cal['grid_rows']} "
                                  f"({spc:.2f} samples/cell)"
                                  + ("  <-- below 0.3, expect flecking"
                                     if spc < 0.3 else ""), flush=True)
                        except Exception as e:
                            print(f"  recalibration failed: {e}", flush=True)
                    prev_key = None          # force a redraw with the new values
                    render_generation += 1
                active_gamma = live_state["gamma"]
                if (render_mode in ("stochastic", "stipple")
                        or (render_mode == "fusion" and "s" in fusion_components)):
                    walk_gamma = active_gamma
                    live_state["stochastic_gamma"] = walk_gamma
                else:
                    gamma = active_gamma
                    live_state["raster_gamma"] = gamma
                lowpass = live_state["lowpass"]
                sweep_mode = live_state["sweep"]
                # Push every live value onto the emitter. It holds the tuning
                # now, so a key that only updated a local would silently stop
                # working -- which is the same class of bug as the two paths
                # drifting, just inside one file.
                with frame_producer.state_lock:
                    if mode_changed:
                        _configure_raster_emitter_fields(
                            emitter, render_mode, fields)
                    emitter.gamma = gamma
                    emitter.trim = trim
                    emitter.density = density
                    emitter.rows = rows
                    emitter.autofit = autofit
                    emitter.sweep_mode = sweep_mode
                    if cal:
                        emitter.grid = _rotation_grid(cal, rotation)
                        emitter.levels = cal.get("levels")
                    if realtime:
                        _gen_settings = {
                            "gamma": gamma, "trim": trim,
                            "density": density, "rows_override": rows,
                            "invert": invert,
                        }
                        if cal:
                            (_gen_settings["grid_rows"],
                             _gen_settings["grid_cols"]) = _rotation_grid(
                                 cal, rotation)
                            _gen_settings["levels"] = cal.get("levels")
                        gen.configure(**_gen_settings)
                    stochastic_emitter.gamma = walk_gamma
                    stochastic_emitter.trim = trim
                    stipple_emitter.gamma = walk_gamma
                    stipple_emitter.trim = trim

            index, index_direction = update_index(png_paths_len, PINGPONG)
            if index != prev_index:
                # sole caller of the stateful selector in this mode
                update_folder_selection(index, float_folder_count,
                                        main_folder_count)
                if index_direction in (-1, 1):
                    source_direction = int(index_direction)
                elif not PINGPONG:
                    source_direction = 1
                elif prev_index >= 0:
                    source_direction = 1 if index > prev_index else -1
                prev_index = index
            mf, ff = folder_dictionary["Main_and_Float_Folders"]
            key = (index, mf, ff)
            if key != source_selected_key:
                source_selected_key = key
                source_selected_at_ns = time.monotonic_ns()
            requested_presentation = (
                render_generation, "MIX" if mix_hz else render_mode,
                int(index), int(mf), int(ff), int(fields))

            if (live_images is not None
                    and (key != prefetched_key
                         or source_direction != prefetched_direction)):
                live_images.prefetch(index, mf, ff,
                                     direction=source_direction,
                                     pingpong=PINGPONG)
                prefetched_key = key
                prefetched_direction = source_direction

            now = time.time()
            ml = main_libs[mf % main_folder_count]
            fl = float_libs[ff % float_folder_count]

            if realtime:
                if key != prev_key:
                    # publish state; the audio thread picks it up at the next
                    # row boundary, so a new index lands within ~1 ms
                    live.update(main=ml, mi=index, float=fl, fi=index)
                    prev_key = key
            elif mix_hz:
                # The worker owns trace production and waits for callback
                # consumption. Main/UI work can continue without replacing an
                # in-flight frame; only the desired image/config snapshot is
                # latest-wins.
                publish_frame_request(ml, fl, index, key,
                                      source_selected_at_ns)
                prev_key = key
            elif use_raster:
                publish_frame_request(ml, fl, index, key,
                                      source_selected_at_ns)
                prev_key = key
            elif use_stochastic:
                publish_frame_request(ml, fl, index, key,
                                      source_selected_at_ns)
                prev_key = key
            elif use_stipple:
                publish_frame_request(ml, fl, index, key,
                                      source_selected_at_ns)
                prev_key = key
            elif use_fusion:
                publish_frame_request(ml, fl, index, key,
                                      source_selected_at_ns)
                prev_key = key
            else:
                # Match the one-pending-frame admission used by the raster and
                # other frame modes. The index clock may advance while the DAC
                # is drawing, but replacing an unconsumed vector trace discards
                # a complete picture and advances beam state past output that
                # was never presented. When ready again, publish the latest
                # selected index; intermediate wall-clock indices are skipped.
                if key != prev_key:
                    publish_frame_request(ml, fl, index, key,
                                          source_selected_at_ns)
                    prev_key = key

            if monitor is not None and now - last_monitor >= 1.0:
                last_monitor = now
                try:
                    from lightweight_monitor import monitor_data as _md
                    _adopted_identity = scope.last_adopted_identity
                    _displayed_index = (
                        int(_adopted_identity[2])
                        if isinstance(_adopted_identity, tuple)
                        and len(_adopted_identity) >= 3 else 0)
                    _accepted_in_interval = (
                        scope.frames_adopted > last_monitor_adopted)
                    last_monitor_adopted = scope.frames_adopted
                    # trim/density/gamma are live-tunable, so re-publish them
                    _md["scope_trim"] = round(trim, 3)
                    _md["scope_density"] = round(density, 3)
                    _md["scope_raster_gamma"] = round(gamma, 2)
                    _md["scope_stochastic_gamma"] = round(walk_gamma, 2)
                    _md["scope_fusion"] = (fusion_components
                                             if use_fusion else None)
                    _md["scope_invert"] = invert
                    _md["scope_rotation"] = rotation
                    _md["scope_mirror"] = mirror
                    _md["scope_trigger"] = trigger_on
                    _md["scope_trigger_shape"] = trigger_shape
                    _md["scope_yt_trigger_us"] = scope.yt_trigger_us
                    _md["scope_yt_timing"] = yt_timing
                    _md["scope_yt_grid"] = _rotation_grid(cal, rotation) if cal else None
                    _md["scope_yt_levels"] = cal.get("levels") if cal else None
                    _md["scope_gamma"] = round(
                        walk_gamma
                        if ((use_stochastic or use_stipple
                             or (use_fusion and "s" in fusion_components))
                            and not mix_hz) else gamma,
                        2)
                    _md["scope_sweep"] = sweep_mode
                    _md["scope_mode"] = ("MIX" if mix_hz else render_mode.upper())
                    _active_fields = fields if (use_raster or mix_hz) else 1
                    _md["scope_fields"] = int(_active_fields)
                    _md["scope_picture_hz"] = round(
                        scope.samplerate
                        / max(scope.samples_per_frame
                              * (tap_traces if mix_hz else _active_fields), 1), 1)
                    _md["scope_traces_drawn"] = int(scope.frames_drawn)
                    _md["scope_indices_skipped"] = int(scope.frames_dropped)
                    if live_images is not None:
                        _md["scope_runtime_image_source"] = live_images.snapshot()
                    _source_metrics_at = time.monotonic()
                    _source_adoptions = int(scope.distinct_source_adoptions)
                    _source_interval = max(
                        _source_metrics_at - last_source_metrics_at, 1e-6)
                    if _source_adoptions >= last_source_adoptions:
                        _source_delta = _source_adoptions - last_source_adoptions
                    else:  # output device/timing replacement created a new Scope
                        _source_delta = _source_adoptions
                    _md["scope_fresh_source_adoptions_per_second"] = (
                        _source_delta / _source_interval)
                    last_source_metrics_at = _source_metrics_at
                    last_source_adoptions = _source_adoptions
                    _adopted_at_ns = scope.last_adopted_monotonic_ns
                    _adopted_identity = scope.last_adopted_identity
                    if (isinstance(_adopted_identity, tuple)
                            and len(_adopted_identity) >= 12
                            and _adopted_identity[7] == "runtime-images"
                            and _adopted_at_ns is not None):
                        _selected_ns, _requested_ns, _decode_started_ns, _ready_ns = (
                            _adopted_identity[8:12])
                        _md["scope_source_age_ms"] = (
                            max(0.0, (_adopted_at_ns - _selected_ns) / 1e6)
                            if _selected_ns is not None else None)
                        _md["scope_source_queue_wait_ms"] = (
                            max(0.0, (_decode_started_ns - _requested_ns) / 1e6)
                            if _decode_started_ns is not None
                            and _requested_ns is not None else None)
                        _md["scope_source_decode_ms"] = (
                            max(0.0, (_ready_ns - _decode_started_ns) / 1e6)
                            if _ready_ns is not None
                            and _decode_started_ns is not None else None)
                        _md["scope_source_ready_to_adoption_ms"] = (
                            max(0.0, (_adopted_at_ns - _ready_ns) / 1e6)
                            if _ready_ns is not None else None)
                    else:
                        _md["scope_source_age_ms"] = None
                        _md["scope_source_queue_wait_ms"] = None
                        _md["scope_source_decode_ms"] = None
                        _md["scope_source_ready_to_adoption_ms"] = None
                    _runtime_adoptions = [
                        (identity, adopted_ns)
                        for identity, adopted_ns in scope.adoption_snapshot()
                        if (isinstance(identity, tuple) and len(identity) >= 12
                            and identity[7] == "runtime-images")]
                    _source_timing_values = {
                        "source_age": [], "queue_wait": [], "decode": [],
                        "ready_to_adoption": [],
                    }
                    for _identity, _adopted_ns in _runtime_adoptions:
                        (_selected_ns, _requested_ns, _decode_started_ns,
                         _ready_ns) = _identity[8:12]
                        if _selected_ns is not None:
                            _source_timing_values["source_age"].append(
                                max(0.0, (_adopted_ns - _selected_ns) / 1e6))
                        if (_requested_ns is not None
                                and _decode_started_ns is not None):
                            _source_timing_values["queue_wait"].append(
                                max(0.0, (_decode_started_ns
                                          - _requested_ns) / 1e6))
                        if (_decode_started_ns is not None and _ready_ns is not None):
                            _source_timing_values["decode"].append(
                                max(0.0, (_ready_ns
                                          - _decode_started_ns) / 1e6))
                        if _ready_ns is not None:
                            _source_timing_values["ready_to_adoption"].append(
                                max(0.0, (_adopted_ns - _ready_ns) / 1e6))
                    _md["scope_source_adoption_samples"] = len(_runtime_adoptions)
                    _md["scope_source_age_ms_window"] = {}
                    for _name, _values in _source_timing_values.items():
                        if _values:
                            _md["scope_source_age_ms_window"][_name] = {
                                "p50": float(np.percentile(_values, 50)),
                                "p95": float(np.percentile(_values, 95)),
                                "p99": float(np.percentile(_values, 99)),
                                "max": float(max(_values)),
                            }
                        else:
                            _md["scope_source_age_ms_window"][_name] = None
                    _md["scope_underruns"] = int(getattr(source, "underruns", 0)
                                                  if source is not None else 0)
                    # Both of these are centre-dot causes.  A rising
                    # scope_beam_parked means frames are arriving stationary
                    # and the guard is ringing them; a rising
                    # scope_dac_dropouts means PortAudio filled a block with
                    # silence and the dot was already on screen.  Which one
                    # moves tells you where to look.
                    _md["scope_beam_parked"] = int(scope.beams_unparked)
                    _md["scope_dac_dropouts"] = int(scope.dac_dropouts)
                    _producer_stats = frame_producer.snapshot()
                    _md["scope_prepared_cache"] = prepared_cache.snapshot()
                    _md["scope_producer_frames"] = int(
                        _producer_stats["rendered"])
                    _md["scope_producer_deadline_misses"] = int(
                        _producer_stats["deadline_misses"])
                    _md["scope_producer_failures"] = int(
                        _producer_stats["failed"])
                    _md["scope_producer_last_error"] = (
                        _producer_stats["last_error"])
                    _deadline_lateness = _producer_stats[
                        "deadline_lateness_ms"]
                    _md["scope_producer_p95_lateness_ms"] = (
                        float(np.percentile(_deadline_lateness, 95))
                        if _deadline_lateness else 0.0)
                    _md["scope_presentations_requested"] = int(
                        _producer_stats["attempted"])
                    _md["scope_render_state_requests"] = int(
                        _producer_stats["requested"])
                    _md["scope_presentations_accepted"] = int(
                        scope.frames_accepted)
                    _md["scope_presentations_adopted"] = int(
                        scope.frames_adopted)
                    _md["scope_presentations_dropped_before_adoption"] = int(
                        scope.frames_dropped)
                    _md["scope_requested_presentation"] = requested_presentation
                    _md["scope_accepted_presentation"] = (
                        scope.last_accepted_identity)
                    _md["scope_adopted_presentation"] = _adopted_identity
                    monitor.update({
                        "index": index,
                        "displayed": _displayed_index,
                        "fps": round(scope.samplerate
                                     / max(scope.samples_per_frame, 1), 1),
                        "fifo_depth": 0,
                        "successful_frame": _accepted_in_interval,
                        "main_folder": mf,
                        "float_folder": ff,
                        "main_folder_count": main_folder_count,
                        "float_folder_count": float_folder_count,
                    })
                except Exception:
                    pass

            if now - last_report >= 60.0:
                last_report = now
                drawn, dropped = scope.frames_drawn, scope.frames_dropped
                if drawn and dropped:
                    print(f"[SCOPE] {drawn} traces, {dropped} indices skipped "
                          f"({dropped / max(drawn + dropped, 1):.0%})")
                if source is not None and getattr(source, "underruns", 0):
                    print(f"[SCOPE] {source.underruns} audio underruns -- the "
                          "generator is not keeping up; raise "
                          "SCOPE_BUFFER_BLOCKS or use frame mode")
            time.sleep(tick)
    finally:
        if frame_producer is not None:
            frame_producer.clear(wait=True)
            frame_producer.close(timeout=10.0)
        if gui is not None:
            gui.close()
        if term is not None:
            term.restore()
        if source is not None and hasattr(source, "close"):
            source.close()
        if live_images is not None:
            live_images.close()
        try:
            scope.stream.stop()
            scope.stream.close()
        except Exception:
            pass


def _queue_presented_frame(scope, frame, *, handoff=None, presentation=None):
    kwargs = {}
    if handoff is not None:
        kwargs["handoff"] = handoff
    if presentation is not None:
        kwargs["identity"] = presentation
    return scope.show_frame(frame, **kwargs)


def _configure_raster_emitter_fields(emitter, render_mode,
                                     configured_raster_fields):
    """Use interlace only in raster mode; other modes are progressive."""
    if render_mode == "raster":
        emitter.fields = max(1, int(configured_raster_fields))
    else:
        emitter.fields = 1
    emitter._field = 0


def _configure_field_groups(field_group, mix_field_group, fields):
    """Reset both accepted-output field latches for a new output lifecycle."""
    field_group.configure(fields)
    mix_field_group.configure(fields)


def _execute_frame_transaction(scope, render, checkpoint, restore, commit):
    """Restore renderer state unless the candidate was accepted by Scope."""
    accepted_before = scope.frames_accepted
    try:
        endpoint = render()
    except Exception:
        if scope.frames_accepted > accepted_before:
            commit(scope.last_accepted_endpoint)
            return True
        restore(checkpoint)
        raise
    if scope.frames_accepted > accepted_before:
        commit(endpoint)
        return True
    restore(checkpoint)
    return False


def _emit(scope, ml, fl, index, render_mode, sweep, sweep_mode,
          gamma, trim, density, rows, min_feature, autofit=True, lowpass=None,
          oversample=1, cal=None, field=0, fields=1, dc_comp=None,
          border=0.0, emitter=None, stochastic_emitter=None, beam_start=None,
          mode_handoff=False, fusion_components="vrs",
          stochastic_gamma=2.0, stochastic_edge=0.0, stipple_emitter=None,
          fusion_multiplexer=None, invert=False, rotation=0,
          presentation=None, defer_state=False, prepared_cache=None,
          source_thumbnails=None):
    n = scope.samples_per_frame

    def prepare_luma(main, main_index, floating, float_index, *, raw=False,
                     invert=False):
        if prepared_cache is not None:
            return prepared_cache.luma(
                main, main_index, floating, float_index, raw=raw,
                invert=invert, rotation=rotation,
                thumbnails=source_thumbnails)
        return _rotate_luma(composite_luma(
            main, main_index, floating, float_index, raw=raw,
            invert=invert, thumbnails=source_thumbnails), rotation)

    def emit_stochastic(target, luminance):
        if prepared_cache is None:
            return target.emit(luminance)
        source_key = PreparedImageCache.source_key(
            ml, index, fl, index, raw=True, invert=invert,
            rotation=rotation)
        probability = prepared_cache.stochastic_probability(
            luminance, source_key, gamma=target.gamma, trim=target.trim,
            edge_gain=target.edge_gain)
        cdf_provider = lambda: prepared_cache.stochastic_cdf(
            probability, source_key, gamma=target.gamma, trim=target.trim,
            edge_gain=target.edge_gain)
        return target.emit_probability(probability, cdf=cdf_provider)

    def prepare_raster_grid(target, luminance):
        if prepared_cache is None:
            return None
        if (getattr(target, "traversal_hz", None) is not None
                or getattr(target, "geometry_samples", target.n) != target.n):
            return None
        source_key = PreparedImageCache.source_key(
            ml, index, fl, index, raw=False, invert=invert,
            rotation=rotation)
        grid = target.grid or (None, None)
        return prepared_cache.raster_grid(
            luminance, source_key, n=target.n, density=target.density,
            trim=target.trim, rows=target.rows, cols=None,
            autofit=target.autofit, grid_rows=grid[0], grid_cols=grid[1],
            row_bias=target.row_bias, levels=target.levels, stretch=True,
            fields=target.fields, precondition=target.precondition,
            yt_fixed=(target.yt_timing == "fixed"))

    if render_mode == "fusion":
        # Build corresponding V/R/S position arrays, then select their entries
        # by sampled source light. No coordinates are averaged and no
        # component is converted into a probability field.
        components = normalize_fusion_components(fusion_components)
        old_raster_dc = emitter.dc_comp
        old_stochastic_dc = stochastic_emitter.dc_comp
        old_raster_border = emitter.border
        old_stochastic_border = stochastic_emitter.border
        emitter.dc_comp = None
        stochastic_emitter.dc_comp = None
        # Component-local rectangles would be interleaved into several broken
        # boxes. Fusion gets one rectangle after its position multiplexer.
        emitter.border = 0.0
        stochastic_emitter.border = 0.0
        try:
            fusion_luma = prepare_luma(
                ml, index, fl, index, raw=True, invert=invert)
            if "v" in components:
                vector_frame = (
                    prepared_cache.vector_frame(
                        ml, index, fl, index, samples=n,
                        min_feature=min_feature, rotation=rotation)
                    if prepared_cache is not None else
                    rotate_frame(rasterize(
                        merge(ml, index, fl, index,
                              min_feature=min_feature), n), rotation))
            else:
                vector_frame = None
            raster_frame = None
            if "r" in components:
                if (beam_start is not None and not defer_state
                        and emitter.sweep_mode == "alternate"):
                    emitter._end = np.asarray(
                        beam_start, dtype=np.float32).copy()
                raster_luma = prepare_luma(
                    ml, index, fl, index, raw=False, invert=invert)
                raster_grid = prepare_raster_grid(emitter, raster_luma)
                if defer_state:
                    raster_frame = emitter.emit(
                        raster_luma, field=field, commit=False,
                        start=beam_start, prepared_grid=raster_grid)
                else:
                    raster_frame = emitter.emit(
                        raster_luma, prepared_grid=raster_grid)
            stochastic_frame = None
            if "s" in components:
                if beam_start is not None:
                    # start_at preserves the walk clock and only aligns the
                    # first physical sample with the fused beam endpoint.
                    stochastic_emitter.start_at(beam_start)
                # Reuse the identical raw composite, rotation, and inversion
                # already prepared for fusion_luma above.
                stochastic_frame = emit_stochastic(
                    stochastic_emitter, fusion_luma)
            mux = fusion_multiplexer or PositionMultiplexer()
            fusion_weights = {
                "v": trace_luminance_weights(
                    fusion_luma, vector_frame, gamma=gamma, trim=trim,
                    level=stochastic_emitter.level),
                "r": trace_luminance_weights(
                    fusion_luma, raster_frame, gamma=gamma, trim=trim,
                    level=stochastic_emitter.level),
                "s": trace_luminance_weights(
                    fusion_luma, stochastic_frame, gamma=stochastic_gamma,
                    trim=trim, level=stochastic_emitter.level),
            }
            frame = mux.emit(vector=vector_frame, raster=raster_frame,
                             stochastic=stochastic_frame,
                             components=components, weights=fusion_weights)
        finally:
            emitter.dc_comp = old_raster_dc
            stochastic_emitter.dc_comp = old_stochastic_dc
            emitter.border = old_raster_border
            stochastic_emitter.border = old_stochastic_border
        if frame is None:
            return None
        reference = fusion_luma
        if reference is not None:
            h, w = reference.shape
            frame = apply_trace_border(
                frame, border, aspect=h / float(max(w, 1)),
                level=stochastic_emitter.level)
        if dc_comp:
            frame = precompensate_hpf(frame, dc_comp, scope.samplerate)
        if "s" in components:
            frame = stochastic_emitter.apply_lowpass(frame, lowpass)
        elif lowpass and lowpass_circular is not None:
            frame = lowpass_circular(frame, lowpass, scope.samplerate)
        queued_end = _queue_presented_frame(
            scope, frame, handoff=beam_start, presentation=presentation)
        end = np.asarray(
            frame[-1] if queued_end is None else queued_end,
            dtype=np.float32).copy()
        if not defer_state:
            emitter._end = end.copy()
        stochastic_emitter.chain_from(end)
        if not defer_state:
            sweep["rev"], sweep["end"] = emitter._rev, emitter._end
        return end

    if render_mode in ("raster", "stochastic", "stipple"):
        # Composite here, sweep in the emitter. Everything about HOW the trace
        # is drawn -- grid, fields, chaining, border, dc-comp -- lives on the
        # emitter, which scope_screen.py shares. Nothing about it is restated
        # in this file, so there is no second parameter list to drift.
        _lum = prepare_luma(
            ml, index, fl, index,
            raw=(render_mode in ("stochastic", "stipple")), invert=invert)
        if Scope._tap_until > _time_mono():
            # only while a browser is watching -- same gate as the trace tap
            Scope.publish_luma(_lum)
        exact_handoff = False
        if beam_start is not None:
            if (render_mode == "raster" and not defer_state
                    and emitter.sweep_mode == "alternate"):
                emitter._end = np.asarray(beam_start, dtype=np.float32).copy()
            elif render_mode in ("stochastic", "stipple"):
                if mode_handoff:
                    (stipple_emitter if render_mode == "stipple"
                     else stochastic_emitter).start_at(beam_start)
                    exact_handoff = True
                else:
                    exact_handoff = (
                        stipple_emitter if render_mode == "stipple"
                        else stochastic_emitter).handoff_from(beam_start)
        if render_mode == "raster":
            raster_grid = prepare_raster_grid(emitter, _lum)
            if defer_state:
                frame = emitter.emit(
                    _lum, field=field, commit=False, start=beam_start,
                    prepared_grid=raster_grid)
            else:
                frame = emitter.emit(_lum, prepared_grid=raster_grid)
        elif render_mode == "stipple":
            if prepared_cache is None:
                cloud = _rotate_stipple_cloud(composite_stipple_candidates(
                    ml, index, fl, index, invert=invert), rotation)
                samples = None
                candidate_key = None
            else:
                cloud = prepared_cache.stipple_candidates(
                    ml, index, fl, index, invert=invert, rotation=rotation)
                candidate_key = PreparedImageCache.source_key(
                    ml, index, fl, index, raw=True, invert=invert,
                    rotation=rotation)
                samples = (prepared_cache.stipple_candidate_samples(
                    cloud, candidate_key, points=stipple_emitter.points,
                    gamma=stipple_emitter.gamma, trim=stipple_emitter.trim,
                    edge_gain=stipple_emitter.edge_gain)
                    if cloud is not None else None)
            if cloud is not None:
                frame = stipple_emitter.emit_candidates(
                    cloud, prepared_samples=samples,
                    tour_cache=None,
                    source_key=candidate_key)
            elif prepared_cache is not None:
                source_key = PreparedImageCache.source_key(
                    ml, index, fl, index, raw=True, invert=invert,
                    rotation=rotation)
                importance = prepared_cache.stipple_importance(
                    _lum, source_key, gamma=stipple_emitter.gamma,
                    trim=stipple_emitter.trim,
                    edge_gain=stipple_emitter.edge_gain)
                samples = prepared_cache.stipple_image_samples(
                    importance, source_key, points=stipple_emitter.points,
                    gamma=stipple_emitter.gamma,
                    trim=stipple_emitter.trim,
                    edge_gain=stipple_emitter.edge_gain)
                frame = stipple_emitter.emit_importance(
                    importance, prepared_samples=samples,
                    tour_cache=None, source_key=source_key)
            else:
                frame = stipple_emitter.emit(_lum)
        else:
            frame = emit_stochastic(stochastic_emitter, _lum)
        if frame is not None:
            if render_mode in ("stochastic", "stipple"):
                frame = (stipple_emitter if render_mode == "stipple"
                         else stochastic_emitter).apply_lowpass(frame, lowpass)
            elif lowpass and lowpass_circular is not None:
                frame = lowpass_circular(frame, lowpass, scope.samplerate)
            # The overshoot clip that used to sit here, gated on raster +
            # fixed timing, now lives in Scope.show_frame as clip_for_trigger.
            # It has to cover every renderer once the marker is always on, and
            # show_frame is the one place they all arrive post-filter.
            handoff = (beam_start if beam_start is not None
                       and getattr(emitter, "traversal_hz", None) is None
                       and (render_mode == "raster" or exact_handoff) else None)
            queued_end = _queue_presented_frame(
                scope, frame, handoff=handoff, presentation=presentation)
            end = np.asarray(
                frame[-1] if queued_end is None else queued_end,
                dtype=np.float32).copy()
            # Compensation and output-side filtering can move both endpoints.
            # Chain the next trace from the final queued waveform, not from the
            # pre-output renderer buffer.
            if render_mode == "raster" and emitter.sweep_mode == "alternate":
                if not defer_state:
                    emitter._end = end.copy()
                    sweep["rev"], sweep["end"] = emitter._rev, emitter._end
            elif render_mode in ("stochastic", "stipple"):
                (stipple_emitter if render_mode == "stipple"
                 else stochastic_emitter).chain_from(end)
            return end
    else:
        # empty -> safe idle circle, never a parked dot
        frame = (prepared_cache.vector_frame(
            ml, index, fl, index, samples=n, min_feature=min_feature,
            rotation=rotation) if prepared_cache is not None else
            rotate_frame(rasterize(
                merge(ml, index, fl, index, min_feature=min_feature), n),
                rotation))
        if invert:
            inverse_luma = prepare_luma(
                ml, index, fl, index, raw=True, invert=True)
            weights = trace_luminance_weights(
                inverse_luma, frame, gamma=gamma, trim=trim)
            if weights is not None:
                frame = retime_trace_by_weights(frame, weights)
        if lowpass and lowpass_circular is not None:
            frame = lowpass_circular(frame, lowpass, scope.samplerate)
        _queue_presented_frame(scope, frame, presentation=presentation)
        return frame[-1].copy()
    return None


if __name__ == "__main__":
    _bootstrap()
    try:
        run_scope()
    except KeyboardInterrupt:
        print("\n[SCOPE] Shutdown.")
