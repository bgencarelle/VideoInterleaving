"""
scope_display.py -- the scope-mode engine.

A standalone mode, like ascii / asciiweb / local / web: selected once and it
owns the run.  Renders the interleaved composition as XY vectors or dwell
raster on the audio output, driven by the same clock (update_index, MIDI
included) and the same folder selector as every other mode.

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
import time
from pathlib import Path

import settings

from scope_out import Scope, choose_device
from scope_bake import XYLibrary, merge, raster_frame, SweepSource, calibrate
try:
    from scope_lowpass import lowpass_circular
except Exception:            # optional tool; absence must not break the mode
    lowpass_circular = None


# ---------------------------------------------------------------- bootstrap

def _bootstrap():
    """Only used when running this file directly: does what main.py's
    configure_runtime() would otherwise do."""
    ap = argparse.ArgumentParser(description="Scope mode (XY audio output)")
    ap.add_argument("--dir", help="image source folder (overrides settings.py)")
    ap.add_argument("--xy-dir", help="baked XY libraries")
    ap.add_argument("--rebuild", action="store_true",
                    help="force rebuild of image lists")
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
    use_raster = getattr(settings, "SCOPE_RASTER", False)
    realtime = getattr(settings, "SCOPE_REALTIME", False)
    min_feature = getattr(settings, "SCOPE_MIN_FEATURE", 0.02)
    trim = getattr(settings, "SCOPE_TRIM", 0.02)
    gamma = getattr(settings, "SCOPE_GAMMA", 2.2)
    density = getattr(settings, "SCOPE_DENSITY", 1.0)
    rows = getattr(settings, "SCOPE_ROWS", None)
    autofit = getattr(settings, "SCOPE_AUTOFIT", True)
    lowpass = getattr(settings, "SCOPE_LOWPASS", None)
    oversample = int(getattr(settings, "SCOPE_OVERSAMPLE", 1) or 1)
    sweep_mode = getattr(settings, "SCOPE_SWEEP", "alternate")
    mix_hz = getattr(settings, "SCOPE_MIX", None)
    mix_duty = min(1.0, max(0.0, getattr(settings, "SCOPE_MIX_DUTY", 0.5)))
    device_spec = getattr(settings, "SCOPE_DEVICE_SPEC", None)
    ask = getattr(settings, "SCOPE_ASK", False)

    if realtime and not use_raster:
        print("[SCOPE] realtime applies to raster only (vector frames have no "
              "positional correspondence); ignoring.")
        realtime = False
    if mix_hz and realtime:
        print("[SCOPE] mix needs whole passes; ignoring realtime.")
        realtime = False
    if mix_hz:
        fps = int(round(mix_hz))        # the switch rate IS the trace rate

    # --- libraries: same list init as every mode, so folder numbering matches
    #     what folder_selector indexes into, by construction ---
    _, main_paths, float_paths = make_file_lists.initialize_image_lists(clock_source)
    png_paths_len = len(main_paths)
    main_folder_count = len(main_paths[0])
    float_folder_count = len(float_paths[0])

    xy_root = _xy_root()
    print(f"[SCOPE] XY libraries: {xy_root}")
    main_libs = _open_layer(main_paths, xy_root, "main", png_paths_len)
    float_libs = _open_layer(float_paths, xy_root, "float", png_paths_len)
    if not any(l is not None for l in main_libs + float_libs):
        raise RuntimeError(
            f"No XY libraries found under '{xy_root}'. Run:\n"
            f"  python utilities/convert_to_xy.py -i {settings.IMAGES_DIR} "
            f"-o {xy_root}")
    if (use_raster or mix_hz) and main_libs[0] is not None \
            and main_libs[0].thumbs is None:
        raise RuntimeError(
            "Raster needs thumbnails and this bake has none. Rebake without "
            "--no-thumbs, or use --thumbs-only for raster alone.")

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
        dev = choose_device(ask=ask, device=device_spec)
    source = None
    if realtime:
        probe = Scope(fps=fps, samples=samples, device=dev)
        n_pass = probe.samples_per_frame
        probe.stream.close()
        source = SweepSource(lambda: live, n_pass, gamma=gamma, trim=trim,
                             density=density, rows=rows)
    scope = Scope(fps=fps, samples=samples, device=dev, source=source)

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
                cells = scope.samples_per_frame / max(density, 0.25)
                want = int(round((cells * cap_rows / cap_cols) ** 0.5))
                if want > cap_rows:
                    print(f"[SCOPE] the sample budget could resolve ~{want} "
                          f"scanlines but the bake caps it at {cap_rows}; "
                          "rebake with a larger --thumb-width to use it.")

    try:
        import sounddevice as _sd
        from scope_out import scrub as _scrub
        _dev_name = _scrub(_sd.query_devices(scope.stream.device)["name"])
    except Exception:
        _dev_name = "?"
    print(f"[SCOPE] output: {_dev_name}")

    if use_raster or mix_hz:
        try:
            cal = calibrate(main_libs, float_libs, scope.samples_per_frame,
                            density=density, trim=trim, rows=rows)
            if cal:
                spc = scope.samples_per_frame / max(
                    cal["grid_rows"] * cal["grid_cols"], 1)
                print(f"[SCOPE] grid {cal['grid_cols']}x{cal['grid_rows']} "
                      f"({spc:.2f} samples/cell), calibrated once")
                if spc < 0.3:
                    print(f"[SCOPE] note: below ~0.3 samples/cell the lit "
                          "pattern shifts frame to frame and reads as moving "
                          "black flecks. Raise --scope-density toward 1.0.")
        except Exception as e:
            print(f"[SCOPE] calibration skipped ({e}); per-frame adaptation")

    mode_name = "MIX" if mix_hz else ("RASTER" if use_raster else "VECTOR")
    print(f"[SCOPE] {mode_name}{' REALTIME' if realtime else ''} | "
          f"{scope.samples_per_frame} samples/trace @ {scope.samplerate} Hz "
          f"({scope.samplerate / scope.samples_per_frame:.0f} passes/sec) | "
          f"latency ~{scope.stream.latency * 1000:.0f} ms")
    print(f"[SCOPE] {main_folder_count} main / {float_folder_count} float "
          f"folders, {png_paths_len} frames, content {IPS} ips")
    if mix_hz:
        print(f"[SCOPE] mix duty {mix_duty:.2f} "
              f"({mix_duty * mix_hz:.0f} raster + {(1 - mix_duty) * mix_hz:.0f} "
              f"vector passes/sec)")
    if lowpass:
        print(f"[SCOPE] low-pass {lowpass:.0f} Hz in the output path "
              "(emulating a softer DAC / RC filter)")
    if fps < IPS and not mix_hz:
        print(f"[SCOPE] note: {fps} traces/sec < {IPS} ips, so some indices are "
              "skipped -- still on time, never late")
    if sweep_mode == "alternate" and fps > IPS and not mix_hz:
        print(f"[SCOPE] note: {fps} traces/sec > {IPS} ips means traces repeat, "
              "and a repeated one-way sweep shows a flyback. "
              "Consider SCOPE_SWEEP='palindrome'.")

    # --- loop ---
    tick = 1.0 / max(2 * IPS, 2 * fps)
    prev_index = -1
    prev_key = None
    last_push = 0.0
    duty_acc = 0.0
    sweep = {"rev": False, "end": None}
    last_report = time.time()

    with scope:
        while True:
            index, _ = update_index(png_paths_len, PINGPONG)
            if index != prev_index:
                # sole caller of the stateful selector in this mode
                update_folder_selection(index, float_folder_count,
                                        main_folder_count)
                prev_index = index
            mf, ff = folder_dictionary["Main_and_Float_Folders"]
            key = (index, mf, ff)

            now = time.time()
            due = (mix_hz and now - last_push >= 1.0 / mix_hz)
            if key != prev_key or due:
                ml = main_libs[mf % main_folder_count]
                fl = float_libs[ff % float_folder_count]

                if realtime:
                    # publish state; the audio thread picks it up at the next
                    # row boundary, so a new index lands within ~1 ms
                    live.update(main=ml, mi=index, float=fl, fi=index)
                elif mix_hz:
                    duty_acc += mix_duty
                    frame_raster = duty_acc >= 1.0
                    if frame_raster:
                        duty_acc -= 1.0
                    last_push = now
                    _emit(scope, ml, fl, index, frame_raster, sweep, sweep_mode,
                          gamma, trim, density, rows, min_feature, autofit,
                          lowpass, oversample, cal)
                else:
                    _emit(scope, ml, fl, index, use_raster, sweep, sweep_mode,
                          gamma, trim, density, rows, min_feature, autofit,
                          lowpass, oversample, cal)
                prev_key = key

            if now - last_report >= 60.0:
                last_report = now
                drawn, dropped = scope.frames_drawn, scope.frames_dropped
                if drawn and dropped:
                    print(f"[SCOPE] {drawn} traces, {dropped} indices skipped "
                          f"({dropped / max(drawn + dropped, 1):.0%})")
            time.sleep(tick)


def _emit(scope, ml, fl, index, as_raster, sweep, sweep_mode,
          gamma, trim, density, rows, min_feature, autofit=True, lowpass=None,
          oversample=1, cal=None):
    n = scope.samples_per_frame
    if as_raster:
        frame = raster_frame(
            ml, index, fl, index, n,
            gamma=gamma, trim=trim, density=density, rows=rows,
            autofit=autofit, oversample=oversample, **(cal or {}),
            palindrome=(sweep_mode == "palindrome"),
            reverse=(sweep_mode == "alternate" and sweep["rev"]),
            start=sweep["end"] if sweep_mode == "alternate" else None,
            close=(sweep_mode == "retrace"))
        if frame is not None:
            if sweep_mode == "alternate":
                sweep["rev"] = not sweep["rev"]
                sweep["end"] = frame[-1]
            if lowpass and lowpass_circular is not None:
                frame = lowpass_circular(frame, lowpass, scope.samplerate)
            scope.show_frame(frame)
    else:
        # empty -> safe idle circle, never a parked dot
        polys = merge(ml, index, fl, index, min_feature=min_feature)
        if lowpass and lowpass_circular is not None:
            from scope_out import rasterize
            scope.show_frame(lowpass_circular(
                rasterize(polys, scope.samples_per_frame), lowpass,
                scope.samplerate))
        else:
            scope.show(polys)


if __name__ == "__main__":
    _bootstrap()
    try:
        run_scope()
    except KeyboardInterrupt:
        print("\n[SCOPE] Shutdown.")