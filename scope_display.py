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

import numpy as np

import settings

from scope_out import Scope, choose_device, BufferedSource
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
    ap.add_argument("--dir", help="image source folder. Optional: scope mode "
                    "reads its manifest from the bake and never opens an image.")
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

    if realtime:
        _d = getattr(settings, "SCOPE_BUFFER_BLOCKS", 6)
        print(f"[SCOPE] realtime: generated on a worker thread, {_d} x 256 "
              "sample ring")
    if realtime and not use_raster:
        print("[SCOPE] realtime applies to raster only (vector frames have no "
              "positional correspondence); ignoring.")
        realtime = False
    if mix_hz and realtime:
        print("[SCOPE] mix needs whole passes; ignoring realtime.")
        realtime = False
    if mix_hz:
        fps = int(round(mix_hz))        # the switch rate IS the trace rate

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
        # Calibrate HERE, before the generator is built: it needs the same
        # grid and levels frame mode uses, or the two modes render the same
        # content differently.
        try:
            cal = calibrate(main_libs, float_libs, n_pass,
                            density=density, trim=trim, rows=rows)
        except Exception as e:
            print(f"[SCOPE] calibration skipped ({e})")
            cal = {}
        if cal:
            _spc = n_pass / max(cal["grid_rows"] * cal["grid_cols"], 1)
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

    if (use_raster or mix_hz) and not cal:
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

    # Measure the real per-frame cost once, so a slow machine says so up front
    # instead of quietly dropping traces.  The budget is one index period.
    if use_raster or mix_hz:
        try:
            import time as _t
            ml0 = next((l for l in main_libs if l is not None), None)
            fl0 = next((l for l in float_libs if l is not None), None)
            if ml0 is not None:
                for _ in range(3):
                    raster_frame(ml0, 0, fl0, 0, scope.samples_per_frame,
                                 gamma=gamma, trim=trim, density=density,
                                 rows=rows, close=False, **cal)
                _t0 = _t.perf_counter()
                for _k in range(10):
                    raster_frame(ml0, _k, fl0, _k, scope.samples_per_frame,
                                 gamma=gamma, trim=trim, density=density,
                                 rows=rows, close=False, **cal)
                _ms = (_t.perf_counter() - _t0) / 10 * 1000.0
                _budget = 1000.0 / max(IPS, 1)
                print(f"[SCOPE] {_ms:.1f} ms per frame, budget {_budget:.1f} ms "
                      f"({_ms / _budget:.0%} of one index period)")
                if _ms > 0.7 * _budget:
                    print("[SCOPE] tight: raise --scope-density, lower "
                          "--scope-fps, or drop --scope-oversample")
        except Exception:
            pass

    # --- live controls ---
    # Everything below is adjustable while watching the scope; restarting to
    # try a different trim is useless when the thing you are judging is a beam.
    live_state = dict(trim=trim, density=density, gamma=gamma, rows=rows,
                      lowpass=lowpass, raster=use_raster, sweep=sweep_mode,
                      autofit=autofit)
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
    tick = 1.0 / max(2 * IPS, 2 * fps)
    prev_index = -1
    prev_key = None
    last_push = 0.0
    duty_acc = 0.0
    sweep = {"rev": False, "end": None}
    last_report = time.time()

    with scope:
      try:
        while True:
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
                    use_raster = live_state["raster"]
                    if use_raster or mix_hz:
                        try:
                            cal = calibrate(main_libs, float_libs,
                                            scope.samples_per_frame,
                                            density=density, trim=trim, rows=rows)
                            spc = scope.samples_per_frame / max(
                                cal["grid_rows"] * cal["grid_cols"], 1)
                            print(f"  grid {cal['grid_cols']}x{cal['grid_rows']} "
                                  f"({spc:.2f} samples/cell)"
                                  + ("  <-- below 0.3, expect flecking"
                                     if spc < 0.3 else ""), flush=True)
                        except Exception as e:
                            print(f"  recalibration failed: {e}", flush=True)
                    prev_key = None          # force a redraw with the new values
                gamma = live_state["gamma"]; lowpass = live_state["lowpass"]
                sweep_mode = live_state["sweep"]

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