"""
scope_display.py -- the scope-mode engine.

A standalone mode, like ascii / asciiweb / local / web: selected once and it
owns the run.  Renders the interleaved composition as XY vectors, dwell raster,
or an Osci-style stochastic luminance walk on the audio output, driven by the
same clock (update_index, MIDI included) and the same folder selector as every
other mode.

Needs none of the image machinery: no ImageLoader, no FIFO, no TurboJPEG, no
GL context.  Geometry comes from libraries baked offline by
utilities/convert_to_xy.py into settings.XY_DIR.

    python main.py --mode scope --dir images --xy-dir images_xy --scope-raster

or directly, using the settings.py defaults:

    python scope_display.py [--dir images] [--xy-dir images_xy] [--rebuild]

All tuning is read from settings.SCOPE_*; main.py publishes CLI overrides
there, exactly as the other modes read ASCII_MODE / SERVER_MODE.
"""
import argparse
import os
import shutil
import threading
import time
from pathlib import Path

import numpy as np

import settings

from time import monotonic as _time_mono
from scope_out import Scope, choose_device, BufferedSource
from scope_bake import (XYLibrary, merge, SweepSource, calibrate,
                        composite_luma, TraceEmitter, StochasticEmitter)
try:
    from scope_lowpass import lowpass_circular
except Exception:            # optional tool; absence must not break the mode
    lowpass_circular = None


# ---------------------------------------------------------------- bootstrap

def _bootstrap():
    """Only used when running this file directly: does what main.py's
    configure_runtime() would otherwise do."""
    ap = argparse.ArgumentParser(description="Scope mode (XY audio output)")
    ap.add_argument("--dir", help="image source folder. Optional: scope mode "
                    "reads its manifest from the bake and never opens an image.")
    ap.add_argument("--xy-dir", help="baked XY libraries")
    ap.add_argument("--rebuild", action="store_true",
                    help="force rebuild of image lists")
    render = ap.add_mutually_exclusive_group()
    render.add_argument("--scope-mode", choices=("vector", "raster", "stochastic"))
    render.add_argument("--scope-raster", "--raster", dest="scope_raster",
                        action="store_true")
    render.add_argument("--scope-stochastic", "--stochastic",
                        dest="scope_stochastic", action="store_true")
    ap.add_argument("--scope-walk-radius", type=int)
    ap.add_argument("--scope-walk-stride", type=int)
    ap.add_argument("--scope-walk-reseed-ms", type=float)
    ap.add_argument("--scope-gamma", type=float,
                    help="active raster/stochastic luminance exponent")
    ap.add_argument("--scope-walk-gamma", type=float, help=argparse.SUPPRESS)
    ap.add_argument("--scope-walk-edge", type=float)
    ap.add_argument("--scope-walk-hz", type=float)
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
    if args.scope_mode or args.scope_raster or args.scope_stochastic:
        settings.SCOPE_RASTER = settings.SCOPE_RENDER_MODE == "raster"
    if args.scope_gamma is not None:
        settings.SCOPE_GAMMA = args.scope_gamma
        if settings.SCOPE_RENDER_MODE == "stochastic":
            settings.SCOPE_WALK_GAMMA = args.scope_gamma
    for arg, setting_name in (
            (args.scope_walk_radius, "SCOPE_WALK_RADIUS"),
            (args.scope_walk_stride, "SCOPE_WALK_STRIDE"),
            (args.scope_walk_reseed_ms, "SCOPE_WALK_RESEED_MS"),
            (args.scope_walk_gamma, "SCOPE_WALK_GAMMA"),
            (args.scope_walk_edge, "SCOPE_WALK_EDGE"),
            (args.scope_walk_hz, "SCOPE_WALK_HZ")):
        if arg is not None:
            setattr(settings, setting_name, arg)

    source = os.path.basename(os.path.normpath(settings.IMAGES_DIR)).replace(" ", "_")
    suffix = f"{source}_scope_None"
    settings.PROCESSED_DIR = os.path.join("_cache", f"folders_processed_{suffix}")
    settings.GENERATED_LISTS_DIR = os.path.join("_cache", f"generated_lists_{suffix}")

    # make_file_lists captures its cache dirs at import time, so import it only
    # after settings are final -- main.py achieves the same by configuring
    # before its own imports.
    import make_file_lists

    script_dir = os.path.dirname(os.path.abspath(__file__))
    gen_full = os.path.join(script_dir, settings.GENERATED_LISTS_DIR)
    if args.rebuild:
        for d in (settings.PROCESSED_DIR, settings.GENERATED_LISTS_DIR):
            shutil.rmtree(os.path.join(script_dir, d), ignore_errors=True)
    if not (os.path.isdir(gen_full) and os.listdir(gen_full)):
        print(">> Building file lists...")
        make_file_lists.process_files()
    else:
        print(f">> Reusing lists in {settings.GENERATED_LISTS_DIR}")


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

    mains.sort(key=lambda p: natural_sort_key(p.name))
    floats.sort(key=lambda p: natural_sort_key(p.name))
    try:
        frames = len(np.load(mains[0] / "frame_starts.npy")) - 1
    except Exception:
        return None
    return frames, mains, floats


def _open_dirs(dirs, layer_name, expected_frames):
    libs = []
    for i, d in enumerate(dirs):
        try:
            lib = XYLibrary(d)
            if len(lib) != expected_frames:
                print(f"[SCOPE] warning: {layer_name} folder {i}: {len(lib)} "
                      f"baked frames vs {expected_frames} expected ({d})")
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
                      f"baked frames vs {expected_frames} listed -- rebake? ({d})")
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
                 density, trim, rows, fields):
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
    dev = resolve_device(spec)
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
    new_scope = Scope(fps=fps, samples=samples, device=dev, source=source,
                      invert_y=False)
    new_cal = {}
    try:
        new_cal = calibrate(main_libs, float_libs, new_scope.samples_per_frame,
                            density=density, trim=trim, rows=rows,
                            fields=fields, row_bias=row_bias)
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

    import make_file_lists
    from index_calculator import update_index
    from folder_selector import update_folder_selection, folder_dictionary
    from settings import IPS, PINGPONG

    # --- configuration: all of it from settings ---
    fps = getattr(settings, "SCOPE_FPS", None) or IPS
    samples = getattr(settings, "SCOPE_SAMPLES", None)
    render_mode = getattr(settings, "SCOPE_RENDER_MODE", None)
    if getattr(settings, "SCOPE_RASTER", False) and render_mode == "vector":
        render_mode = "raster"       # compatibility with older settings.py
    if render_mode not in ("vector", "raster", "stochastic"):
        render_mode = "raster" if getattr(settings, "SCOPE_RASTER", False) else "vector"
    use_raster = render_mode == "raster"
    use_stochastic = render_mode == "stochastic"
    realtime = getattr(settings, "SCOPE_REALTIME", False)
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
    rows = getattr(settings, "SCOPE_ROWS", None)
    autofit = getattr(settings, "SCOPE_AUTOFIT", True)
    lowpass = getattr(settings, "SCOPE_LOWPASS", None)
    oversample = int(getattr(settings, "SCOPE_OVERSAMPLE", 1) or 1)
    sweep_mode = getattr(settings, "SCOPE_SWEEP", "alternate")
    fields = max(1, int(getattr(settings, "SCOPE_FIELDS", 1) or 1))
    dc_comp = getattr(settings, "SCOPE_DC_COMP", None)
    border = float(getattr(settings, "SCOPE_BORDER", 0.0) or 0.0)
    row_bias = float(getattr(settings, "SCOPE_ROW_BIAS", 1.0) or 1.0)
    mix_hz = getattr(settings, "SCOPE_MIX", None)
    mix_duty = min(1.0, max(0.0, getattr(settings, "SCOPE_MIX_DUTY", 0.5)))
    device_spec = getattr(settings, "SCOPE_DEVICE_SPEC", None)
    ask = getattr(settings, "SCOPE_ASK", False)

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
    if mix_hz:
        fps = int(round(mix_hz))        # the switch rate IS the trace rate

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
        if fields <= 1 and _raster_traces >= 1.9:
            fields = int(round(_raster_traces))
            print(f"[SCOPE] mix: raster gets {_raster_traces:.1f} traces per "
                  f"index, so interlacing it x{fields} -- the grid is sized "
                  f"for {fields} traces, not one. Pass --scope-fields 1 to "
                  "size it per trace as before.")

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

    # --- libraries ---
    # Prefer the baked tree: it carries the same folder names in the same
    # order, so it can supply the manifest without rescanning the images.
    # Fall back to the image lists when the bake predates this or the tree
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
        print(f"[SCOPE] manifest from the bake (images not read)")
        main_libs = _open_dirs(main_dirs, "main", png_paths_len)
        float_libs = _open_dirs(float_dirs, "float", png_paths_len)
    else:
        _, main_paths, float_paths = make_file_lists.initialize_image_lists(clock_source)
        png_paths_len = len(main_paths)
        main_folder_count = len(main_paths[0])
        float_folder_count = len(float_paths[0])
        main_libs = _open_layer(main_paths, xy_root, "main", png_paths_len)
        float_libs = _open_layer(float_paths, xy_root, "float", png_paths_len)
    if not any(l is not None for l in main_libs + float_libs):
        raise RuntimeError(
            f"No XY libraries found under '{xy_root}'. Run:\n"
            f"  python utilities/convert_to_xy.py -i {settings.IMAGES_DIR} "
            f"-o {xy_root}")
    missing_thumbs = [lib for lib in main_libs + float_libs
                      if lib is not None and lib.thumbs is None]
    if (use_raster or use_stochastic or mix_hz) and missing_thumbs:
        raise RuntimeError(
            "Raster and stochastic modes need thumbnails and this bake has "
            f"{len(missing_thumbs)} library/libraries without them. Rebake "
            "normally; thumbnails are part of every standard bake.")
    if use_stochastic:
        legacy_luma = [lib for lib in main_libs + float_libs
                       if lib is not None and lib.thumbs is not None
                       and lib.thumbs.ndim >= 4 and lib.thumbs.shape[-1] < 3]
        if legacy_luma:
            print("[SCOPE] stochastic: this older bake has only the raster-"
                  "preconditioned luminance channel. It remains compatible, "
                  "but rebake once for the raw stochastic luminance channel.")
        first_thumb = next(
            (lib.thumbs for lib in main_libs + float_libs
             if lib is not None and lib.thumbs is not None), None)
        walk_width = int(first_thumb.shape[2])
        resolved_stride = StochasticEmitter(
            48000, 2, stride=walk_stride)._stride_for_width(walk_width)
        stride_text = (f"auto -> {resolved_stride}" if walk_stride <= 0
                       else str(resolved_stride))
        print(f"[SCOPE] stochastic settings: gamma={walk_gamma:g} "
              f"stride={stride_text} at {walk_width}px radius={walk_radius} "
              f"reseed={walk_reseed_ms:g}ms walk={walk_hz:g}Hz", flush=True)

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
        dev = choose_device(ask=ask, device=device_spec)
    source = None
    if realtime:
        probe = Scope(fps=fps, samples=samples, device=dev, invert_y=False)
        n_pass = probe.samples_per_frame
        probe.stream.close()
        # Calibrate HERE, before the generator is built: it needs the same
        # grid and levels frame mode uses, or the two modes render the same
        # content differently.
        try:
            cal = calibrate(main_libs, float_libs, n_pass,
                            density=density, trim=trim, rows=rows,
                            fields=fields, row_bias=row_bias)
        except Exception as e:
            print(f"[SCOPE] calibration skipped ({e})")
            cal = {}
        if cal:
            _spc = n_pass * fields / max(cal["grid_rows"] * cal["grid_cols"], 1)
            print(f"[SCOPE] grid {cal['grid_cols']}x{cal['grid_rows']} "
                  f"({_spc:.2f} samples/cell), calibrated once")
        gen = SweepSource(lambda: live, n_pass, gamma=gamma, trim=trim,
                          density=density, rows=rows, **(cal or {}))
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
    scope = Scope(fps=fps, samples=samples, device=dev, source=source,
                  invert_y=False)

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

    if (use_raster or mix_hz) and not cal:
        try:
            cal = calibrate(main_libs, float_libs, scope.samples_per_frame,
                            density=density, trim=trim, rows=rows,
                            fields=fields, row_bias=row_bias)
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
    _trace_hz = scope.samplerate / scope.samples_per_frame
    print(f"[SCOPE] {mode_name}{' REALTIME' if realtime else ''}"
          f"{f' INTERLACE x{fields}' if fields > 1 else ''} | "
          f"{scope.samples_per_frame} samples/trace @ {scope.samplerate} Hz "
          f"({_trace_hz:.0f} passes/sec) | "
          f"latency ~{scope.stream.latency * 1000:.0f} ms")
    if dc_comp:
        print(f"[SCOPE] DC compensation at {dc_comp:.0f} Hz -- flat regions "
              "hold instead of sagging, at the cost of amplitude. Raise the "
              "scope's gain to compensate.")
    if fields > 1:
        print(f"[SCOPE] refresh {_trace_hz:.0f} Hz, picture "
              f"{_trace_hz / fields:.0f} Hz "
              f"({scope.samples_per_frame * fields} samples per picture -- "
              "the grid is sized for that, not for one field)")
    print(f"[SCOPE] {main_folder_count} main / {float_folder_count} float "
          f"folders, {png_paths_len} frames, content {IPS} ips")
    if mix_hz:
        print(f"[SCOPE] mix duty {mix_duty:.2f} "
              f"({mix_duty * mix_hz:.0f} raster + {(1 - mix_duty) * mix_hz:.0f} "
              f"vector passes/sec)")
        print(f"[SCOPE] mix raster: {scope.samples_per_frame * fields} samples "
              f"per picture across {fields} trace(s). Baseline for comparison "
              f"is {int(scope.samplerate / max(IPS, 1))} at --scope-fps {IPS} "
              "with no mix; the shortfall is the beam time spent on vector.")
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
    if use_raster or use_stochastic or mix_hz:
        try:
            import time as _t
            ml0 = next((l for l in main_libs if l is not None), None)
            fl0 = next((l for l in float_libs if l is not None), None)
            if ml0 is not None:
                # Measure the REAL path. A probe that times something the
                # program never runs is worse than no probe.
                if use_stochastic and not mix_hz:
                    _pe = StochasticEmitter(
                        scope.samplerate, scope.samples_per_frame,
                        gamma=walk_gamma, trim=trim, radius=walk_radius,
                        stride=walk_stride, edge_gain=walk_edge,
                        reseed_ms=walk_reseed_ms, walk_hz=walk_hz,
                        dc_comp=dc_comp)
                else:
                    _pe = TraceEmitter(
                        scope.samplerate, scope.samples_per_frame,
                        gamma=gamma, trim=trim, density=density, rows=rows,
                        fields=fields, border=border, oversample=oversample,
                        sweep=sweep_mode, autofit=autofit, row_bias=row_bias,
                        grid=((cal["grid_rows"], cal["grid_cols"]) if cal else None),
                        levels=(cal.get("levels") if cal else None))
                for _ in range(3):
                    _pe.emit(composite_luma(
                        ml0, 0, fl0, 0, raw=(use_stochastic and not mix_hz)))
                _t0 = _t.perf_counter()
                for _k in range(10):
                    _pe.emit(composite_luma(
                        ml0, _k, fl0, _k,
                        raw=(use_stochastic and not mix_hz)))
                _ms = (_t.perf_counter() - _t0) / 10 * 1000.0
                _budget = 1000.0 / max(scope.samplerate / scope.samples_per_frame, 1)
                print(f"[SCOPE] {_ms:.1f} ms per trace, budget {_budget:.1f} ms "
                      f"({_ms / _budget:.0%} of one trace period)")
                if _ms > 0.7 * _budget:
                    print("[SCOPE] tight: raise --scope-density, lower "
                          "--scope-fields, or drop --scope-oversample")
        except Exception:
            pass

    # --- live controls ---
    # Everything below is adjustable while watching the scope; restarting to
    # try a different trim is useless when the thing you are judging is a beam.
    live_state = dict(trim=trim, density=density,
                      gamma=(walk_gamma if use_stochastic else gamma),
                      raster_gamma=gamma, stochastic_gamma=walk_gamma, rows=rows,
                      lowpass=lowpass, mode=render_mode, raster=use_raster,
                      sweep=sweep_mode, autofit=autofit,
                      mode_locked=bool(realtime or mix_hz))
    # --- monitoring ---
    # Same two-part contract every other mode uses: main.py starts the server,
    # the engine feeds lightweight_monitor.  Without this scope is invisible to
    # /data, to the dashboard and to multimonitor.py -- which matters far more
    # here than elsewhere, because scope has no window to look at and no tty
    # under systemd.
    # The web page is the only display many people will have, so its preview
    # must show a whole picture, not one interlaced field.
    try:
        scope.set_tap_fields(fields)
    except Exception:
        pass

    monitor = None
    try:
        from lightweight_monitor import start_monitor, monitor_data
        monitor = start_monitor()
        monitor_data["scope_device"] = _dev_name
        # Static: enumerated once. The page needs it to build the selector, and
        # putting it in /data avoids a second endpoint.
        try:
            from scope_out import list_output_devices, default_output_index
            _dflt = default_output_index()
            monitor_data["scope_devices"] = [
                {"index": i, "name": nm, "api": api, "rate": rate,
                 "default": (i == _dflt)}
                for (i, nm, api, rate) in list_output_devices()]
        except Exception:
            monitor_data["scope_devices"] = []
        monitor_data["scope_mode"] = mode_name + (" REALTIME" if realtime else "")
        monitor_data["scope_samplerate"] = int(scope.samplerate)
        monitor_data["scope_samples_per_trace"] = int(scope.samples_per_frame)
        monitor_data["scope_fields"] = int(fields)
        monitor_data["scope_refresh_hz"] = round(
            scope.samplerate / max(scope.samples_per_frame, 1), 1)
        monitor_data["scope_picture_hz"] = round(
            scope.samplerate / max(scope.samples_per_frame * fields, 1), 1)
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
        if term.enabled:
            keys = KeyMap(live_state)
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
    duty_acc = 0.0
    # One emitter, shared with scope_screen.py. Tuning and sweep state live on
    # the object so neither caller keeps its own copy -- that divergence is
    # what test_scope_parity.py exists to catch.
    emitter = TraceEmitter(
        scope.samplerate, scope.samples_per_frame,
        gamma=gamma, trim=trim, density=density, rows=rows, fields=fields,
        border=border, oversample=oversample, sweep=sweep_mode,
        dc_comp=dc_comp, autofit=autofit, row_bias=row_bias,
        grid=((cal["grid_rows"], cal["grid_cols"]) if cal else None),
        levels=(cal.get("levels") if cal else None))
    stochastic_emitter = StochasticEmitter(
        scope.samplerate, scope.samples_per_frame,
        gamma=walk_gamma, trim=trim, radius=walk_radius, stride=walk_stride,
        edge_gain=walk_edge, reseed_ms=walk_reseed_ms, walk_hz=walk_hz,
        dc_comp=dc_comp)

    field_i = 0                     # free-running; survives a slipped deadline
    mix_field_i = 0                 # advances only on the RASTER traces of mix
    sweep = {"rev": False, "end": None}
    beam_end = None                 # actual last sample handed to the DAC
    last_report = time.time()
    last_monitor = 0.0

    # NOT `with scope:` -- the device can be changed at runtime, which means
    # rebinding `scope`.  A with-block would call __exit__ on the object it
    # entered, i.e. the already-closed old stream, and raise on the way out.
    scope.stream.start()
    try:
        while True:
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
                    scope, cal, sweep = _swap_device(
                        scope, _want, source, fps, samples,
                        main_libs, float_libs, density, trim, rows, fields)
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
                        row_bias=row_bias,
                        grid=((cal["grid_rows"], cal["grid_cols"]) if cal else None),
                        levels=(cal.get("levels") if cal else None))
                    stochastic_emitter = StochasticEmitter(
                        scope.samplerate, scope.samples_per_frame,
                        gamma=walk_gamma, trim=trim, radius=walk_radius,
                        stride=walk_stride, edge_gain=walk_edge,
                        reseed_ms=walk_reseed_ms, walk_hz=walk_hz,
                        dc_comp=dc_comp)
                    beam_end = None
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
                            scope.samplerate / max(scope.samples_per_frame, 1), 1)
                        _md2["scope_picture_hz"] = round(
                            scope.samplerate
                            / max(scope.samples_per_frame * fields, 1), 1)
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
            if keys is not None:
                for ch in term.read():
                    if keys.feed(ch) and keys.message:
                        print(keys.message, flush=True)
                        keys.message = ""
                if keys.quit:
                    break
                if keys.dirty:
                    keys.dirty = False
                    trim = live_state["trim"]; density = live_state["density"]
                    rows = live_state["rows"]; autofit = live_state["autofit"]
                    next_mode = live_state["mode"]
                    mode_changed = next_mode != render_mode
                    if mode_changed:
                        # A chain endpoint belongs to the beam, not to a render
                        # mode. Do not resume either procedural emitter from the
                        # stale position it had before cycling away from it.
                        emitter.reset()
                        stochastic_emitter.reset()
                    render_mode = next_mode
                    use_raster = render_mode == "raster"
                    use_stochastic = render_mode == "stochastic"
                    if mode_changed:
                        field_i = 0
                        try:
                            scope.set_tap_fields(fields if use_raster else 1)
                        except Exception:
                            pass
                    if use_raster or mix_hz:
                        try:
                            cal = calibrate(main_libs, float_libs,
                                            scope.samples_per_frame,
                                            density=density, trim=trim,
                                            rows=rows, fields=fields,
                                            row_bias=row_bias)
                            spc = scope.samples_per_frame * fields / max(
                                cal["grid_rows"] * cal["grid_cols"], 1)
                            print(f"  grid {cal['grid_cols']}x{cal['grid_rows']} "
                                  f"({spc:.2f} samples/cell)"
                                  + ("  <-- below 0.3, expect flecking"
                                     if spc < 0.3 else ""), flush=True)
                        except Exception as e:
                            print(f"  recalibration failed: {e}", flush=True)
                    prev_key = None          # force a redraw with the new values
                active_gamma = live_state["gamma"]
                if render_mode == "stochastic":
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
                emitter.gamma = gamma
                emitter.trim = trim
                emitter.density = density
                emitter.rows = rows
                emitter.autofit = autofit
                emitter.sweep_mode = sweep_mode
                if cal:
                    emitter.grid = (cal["grid_rows"], cal["grid_cols"])
                    emitter.levels = cal.get("levels")
                stochastic_emitter.gamma = walk_gamma
                stochastic_emitter.trim = trim

            index, _ = update_index(png_paths_len, PINGPONG)
            if index != prev_index:
                # sole caller of the stateful selector in this mode
                update_folder_selection(index, float_folder_count,
                                        main_folder_count)
                prev_index = index
            mf, ff = folder_dictionary["Main_and_Float_Folders"]
            key = (index, mf, ff)

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
                # Same ready() gate as plain raster, for the same reason: the
                # old wall-clock pacer drifted against the actual trace rate,
                # so frames were occasionally queued two-deep and the second
                # replaced the first.  Losing a frame that way costs a whole
                # interlace field, and it also desynchronised duty_acc from
                # the traces the beam really drew.
                if scope.ready():
                    duty_acc += mix_duty
                    frame_raster = duty_acc >= 1.0
                    if frame_raster:
                        duty_acc -= 1.0
                    _end = _emit(
                        scope, ml, fl, index,
                        "raster" if frame_raster else "vector", sweep, sweep_mode,
                        gamma, trim, density, rows, min_feature, autofit,
                        lowpass, oversample, cal, dc_comp=dc_comp,
                        border=border, emitter=emitter,
                        stochastic_emitter=stochastic_emitter,
                        beam_start=beam_end,
                        field=mix_field_i % fields, fields=fields)
                    if _end is not None:
                        beam_end = _end
                    if frame_raster:
                        mix_field_i += 1
                    prev_key = key
            elif use_raster:
                # One frame per TRACE, gated on the callback having taken the
                # last one.  Three things fall out of this that the
                # emit-on-index-change version got wrong:
                #   - interlaced fields are handed over in order, so both
                #     halves of the picture actually reach the beam;
                #   - the chained alternating sweep gets the fresh frame it
                #     assumes, instead of the callback looping a frame whose
                #     close segment was deliberately omitted -- that loop is a
                #     full-screen jump with no samples budgeted for it, i.e. a
                #     bright flyback line on every repeat;
                #   - a frame is never queued on top of an unconsumed one, so
                #     sweep["end"] can no longer advance to the end of a frame
                #     the beam never drew.
                if scope.ready():
                    _end = _emit(
                        scope, ml, fl, index, "raster", sweep, sweep_mode,
                        gamma, trim, density, rows, min_feature, autofit,
                        lowpass, oversample, cal, dc_comp=dc_comp,
                        border=border, emitter=emitter,
                        stochastic_emitter=stochastic_emitter,
                        beam_start=beam_end,
                        field=field_i % fields, fields=fields)
                    if _end is not None:
                        beam_end = _end
                    field_i += 1
                    prev_key = key
            elif use_stochastic:
                # Like raster, stochastic is a continuing beam walk, so hand it
                # a fresh endpoint-chained trace whenever the callback is ready.
                if scope.ready():
                    _end = _emit(
                        scope, ml, fl, index, "stochastic", sweep, sweep_mode,
                        gamma, trim, density, rows, min_feature, autofit,
                        lowpass, oversample, cal, dc_comp=dc_comp,
                        border=border, emitter=emitter,
                        stochastic_emitter=stochastic_emitter,
                        beam_start=beam_end)
                    if _end is not None:
                        beam_end = _end
                    prev_key = key
            else:
                if key != prev_key:
                    _end = _emit(
                        scope, ml, fl, index, "vector", sweep, sweep_mode,
                        gamma, trim, density, rows, min_feature, autofit,
                        lowpass, oversample, cal, dc_comp=dc_comp,
                        border=border, emitter=emitter,
                        stochastic_emitter=stochastic_emitter,
                        beam_start=beam_end)
                    if _end is not None:
                        beam_end = _end
                    prev_key = key

            if monitor is not None and now - last_monitor >= 1.0:
                last_monitor = now
                try:
                    from lightweight_monitor import monitor_data as _md
                    # trim/density/gamma are live-tunable, so re-publish them
                    _md["scope_trim"] = round(trim, 3)
                    _md["scope_density"] = round(density, 3)
                    _md["scope_gamma"] = round(gamma, 2)
                    _md["scope_sweep"] = sweep_mode
                    _md["scope_mode"] = ("MIX" if mix_hz else render_mode.upper())
                    _active_fields = fields if (use_raster or mix_hz) else 1
                    _md["scope_fields"] = int(_active_fields)
                    _md["scope_picture_hz"] = round(
                        scope.samplerate
                        / max(scope.samples_per_frame * _active_fields, 1), 1)
                    _md["scope_traces_drawn"] = int(scope.frames_drawn)
                    _md["scope_indices_skipped"] = int(scope.frames_dropped)
                    _md["scope_underruns"] = int(getattr(source, "underruns", 0)
                                                 if source is not None else 0)
                    # displayed == index: scope has no FIFO, so the trace being
                    # drawn IS the current index.  Reporting them equal keeps
                    # the shared dashboard's delta meaningful rather than blank.
                    monitor.update({
                        "index": index,
                        "displayed": index,
                        "fps": round(scope.samplerate
                                     / max(scope.samples_per_frame, 1), 1),
                        "fifo_depth": 0,
                        "successful_frame": True,
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
        if term is not None:
            term.restore()
        if source is not None and hasattr(source, "close"):
            source.close()
        try:
            scope.stream.stop()
            scope.stream.close()
        except Exception:
            pass


def _emit(scope, ml, fl, index, render_mode, sweep, sweep_mode,
          gamma, trim, density, rows, min_feature, autofit=True, lowpass=None,
          oversample=1, cal=None, field=0, fields=1, dc_comp=None,
          border=0.0, emitter=None, stochastic_emitter=None, beam_start=None):
    n = scope.samples_per_frame
    if render_mode in ("raster", "stochastic"):
        # Composite here, sweep in the emitter. Everything about HOW the trace
        # is drawn -- grid, fields, chaining, border, dc-comp -- lives on the
        # emitter, which scope_screen.py shares. Nothing about it is restated
        # in this file, so there is no second parameter list to drift.
        _lum = composite_luma(
            ml, index, fl, index, raw=(render_mode == "stochastic"))
        if Scope._tap_until > _time_mono():
            # only while a browser is watching -- same gate as the trace tap
            Scope.publish_luma(_lum)
        exact_handoff = False
        if beam_start is not None:
            if render_mode == "raster" and emitter.sweep_mode == "alternate":
                emitter._end = np.asarray(beam_start, dtype=np.float32).copy()
            elif render_mode == "stochastic":
                exact_handoff = stochastic_emitter.handoff_from(beam_start)
        frame = (emitter.emit(_lum) if render_mode == "raster"
                 else stochastic_emitter.emit(_lum))
        if frame is not None:
            # mirrored back for the live-controls print and anything else
            # reading sweep state; the emitter owns the real copy
            if render_mode == "raster":
                sweep["rev"], sweep["end"] = emitter._rev, emitter._end
            if render_mode == "stochastic":
                frame = stochastic_emitter.apply_lowpass(frame, lowpass)
            elif lowpass and lowpass_circular is not None:
                frame = lowpass_circular(frame, lowpass, scope.samplerate)
            if beam_start is not None and (render_mode == "raster"
                                           or exact_handoff):
                # Circular filters and DC compensation can move sample zero.
                # Restore the exact handoff point so the frame boundary itself
                # never introduces an unbudgeted visible connector.
                frame[0] = beam_start
            # Compensation/filtering can move the final sample. Chain from the
            # sample the DAC actually receives, not the unfiltered geometry.
            if render_mode == "raster" and emitter.sweep_mode == "alternate":
                emitter._end = frame[-1].copy()
            elif render_mode == "stochastic":
                stochastic_emitter.chain_from(frame[-1])
            scope.show_frame(frame)
            return frame[-1].copy()
    else:
        # empty -> safe idle circle, never a parked dot
        polys = merge(ml, index, fl, index, min_feature=min_feature)
        from scope_out import rasterize
        frame = rasterize(polys, n)
        if lowpass and lowpass_circular is not None:
            frame = lowpass_circular(frame, lowpass, scope.samplerate)
        scope.show_frame(frame)
        return frame[-1].copy()
    return None


if __name__ == "__main__":
    _bootstrap()
    try:
        run_scope()
    except KeyboardInterrupt:
        print("\n[SCOPE] Shutdown.")
