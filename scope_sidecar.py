"""
scope_sidecar.py -- run the scope output FRAME-LOCKED to the video display,
in the same process, with zero modifications to the repo.

    python scope_sidecar.py --dir images --xy-dir images_xy --raster --ask

Why a sidecar rather than running scope_display.py alongside image_display.py:

    folder_selector.update_folder_selection() is stateful AND stochastic --
    a time-seeded RNG driving pre-shuffled deques.  Two processes each calling
    it advance two independent random sequences, so they pick DIFFERENT
    folders.  The two outputs would not merely drift, they would show
    different content.

    So the scope must never call it.  This module lets image_display own the
    clock and the selector exactly as it does today, and taps the result:
    update_folder_selection is wrapped so that every time the video advances an
    index and picks a folder pair, the same (index, main, float) is published
    to the scope.  One decision, two outputs.

Accuracy note: this locks to the index that DRIVES the video.  Under load
image_display may briefly display a compensated index instead
(compensator.get_compensated_index -> fifo.get), in which case the scope can
be a frame or two ahead of the screen.  For exact lock, image_display would
need a one-line tap on `d_idx` inside its loop -- see EXACT_LOCK below.

EXACT_LOCK: in image_display.run_display, right after
        d_idx, m_img, f_img, m_sbs, f_sbs = res
    add
        scope_sidecar.publish_displayed(d_idx)
    and pass --exact to this script.
"""
import argparse
import os
import sys
import threading
import time

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import settings                                                   # noqa: E402
from scope_bake import XYLibrary, merge, raster_frame, SweepSource  # noqa: E402
from scope_out import Scope, choose_device                        # noqa: E402

_state = {"main": None, "mi": 0, "float": None, "fi": 0}
_scope = None
_libs = {"main": [], "float": []}
_cfg = {}
_lock = threading.Lock()
_exact = False
_pending_folders = None


def publish_displayed(d_idx):
    """Optional exact-lock hook -- see EXACT_LOCK in the module docstring."""
    if _pending_folders is None:
        return
    mf, ff = _pending_folders
    _push(int(d_idx), mf, ff)


# ---------------------------------------------------------------- plumbing

def _folder_dir(image_path, xy_root):
    folder = os.path.dirname(image_path)
    rel = os.path.relpath(folder, settings.IMAGES_DIR)
    if rel.startswith(".."):
        from pathlib import Path
        parts = Path(folder).parts
        for key in ("face", "float"):
            if key in parts:
                rel = os.path.join(*parts[parts.index(key):])
                break
    return os.path.join(xy_root, rel)


def _push(index, mf, ff):
    """Publish one (index, folder pair) to the scope. Called from the video
    thread, so it must stay cheap."""
    global _scope
    if _scope is None:
        return
    mains, floats = _libs["main"], _libs["float"]
    ml = mains[mf % len(mains)] if mains else None
    fl = floats[ff % len(floats)] if floats else None
    if _cfg.get("realtime"):
        with _lock:
            _state.update(main=ml, mi=index, float=fl, fi=index)
        return
    n = _scope.samples_per_frame
    if _cfg["raster"]:
        frame = raster_frame(ml, index, fl, index, n,
                             gamma=_cfg["gamma"], trim=_cfg["trim"],
                             density=_cfg["density"], rows=_cfg["rows"],
                             reverse=_cfg["rev"], start=_cfg["end"],
                             close=False)
        if frame is not None:
            _cfg["rev"] = not _cfg["rev"]
            _cfg["end"] = frame[-1]
            _scope.show_frame(frame)
    else:
        _scope.show(merge(ml, index, fl, index,
                          min_feature=_cfg["min_feature"]))


def _wrap_selector(mod):
    """Wrap image_display's bound reference to update_folder_selection so the
    scope sees exactly the folder pair the video just chose."""
    original = mod.update_folder_selection

    def wrapped(index, float_count, main_count, *a, **kw):
        global _pending_folders
        result = original(index, float_count, main_count, *a, **kw)
        try:
            mf, ff = mod.folder_dictionary["Main_and_Float_Folders"]
            _pending_folders = (mf, ff)
            if not _exact:
                _push(int(index), mf, ff)
        except Exception as e:
            print(f"[SIDECAR] publish failed: {e}")
        return result

    mod.update_folder_selection = wrapped
    return original


# ---------------------------------------------------------------- main

def main():
    global _scope, _exact
    ap = argparse.ArgumentParser(
        description="Scope output frame-locked to the video display")
    ap.add_argument("--dir", help="image source folder")
    ap.add_argument("--xy-dir", help="baked XY libraries")
    ap.add_argument("--raster", action="store_true")
    ap.add_argument("--realtime", action="store_true",
                    help="stream continuously (raster only); costs CPU in the "
                         "audio thread, so only on a machine with headroom")
    ap.add_argument("--fps", type=int, help="scope trace rate (default: IPS)")
    ap.add_argument("--samples", type=int, help="samples per trace")
    ap.add_argument("--trim", type=float, default=0.02)
    ap.add_argument("--gamma", type=float, default=2.2)
    ap.add_argument("--density", type=float, default=1.0)
    ap.add_argument("--rows", type=int)
    ap.add_argument("--min-feature", type=float, default=0.02)
    ap.add_argument("--ask", action="store_true")
    ap.add_argument("--device")
    ap.add_argument("--exact", action="store_true",
                    help="lock to the frame actually displayed rather than the "
                         "driving index (requires the EXACT_LOCK hook)")
    args = ap.parse_args()
    _exact = args.exact

    if args.dir:
        p = os.path.abspath(args.dir)
        settings.IMAGES_DIR = p
        settings.MAIN_FOLDER_PATH = os.path.join(p, "face")
        settings.FLOAT_FOLDER_PATH = os.path.join(p, "float")
    xy_root = args.xy_dir or getattr(settings, "XY_DIR", "images_xy")

    # image_display owns cache naming via main.py normally; mirror it here
    source = os.path.basename(os.path.normpath(settings.IMAGES_DIR)).replace(" ", "_")
    suffix = f"{source}_scope_sidecar"
    settings.PROCESSED_DIR = os.path.join("_cache", f"folders_processed_{suffix}")
    settings.GENERATED_LISTS_DIR = os.path.join("_cache", f"generated_lists_{suffix}")

    import make_file_lists
    gen = os.path.join(ROOT_DIR, settings.GENERATED_LISTS_DIR)
    if not (os.path.isdir(gen) and os.listdir(gen)):
        print(">> Building file lists...")
        make_file_lists.process_files()

    _, main_paths, float_paths = make_file_lists.initialize_image_lists(
        settings.CLOCK_MODE)
    for key, paths in (("main", main_paths), ("float", float_paths)):
        for f in range(len(paths[0])):
            d = _folder_dir(paths[0][f], xy_root)
            try:
                _libs[key].append(XYLibrary(d))
            except Exception as e:
                print(f"[SIDECAR] ⚠️  {key} folder {f}: no library at {d} ({e})")
                _libs[key].append(None)
    if not any(l is not None for l in _libs["main"] + _libs["float"]):
        raise SystemExit(f"No XY libraries under {xy_root}")

    _cfg.update(raster=args.raster, realtime=args.realtime and args.raster,
                gamma=args.gamma, trim=args.trim, density=args.density,
                rows=args.rows, min_feature=args.min_feature,
                rev=False, end=None)

    fps = args.fps or getattr(settings, "IPS", 30)
    src = None
    if _cfg["realtime"]:
        probe = Scope(fps=fps, samples=args.samples,
                      device=choose_device(ask=args.ask, device=args.device))
        n_pass = probe.samples_per_frame
        probe.stream.close()
        src = SweepSource(lambda: dict(_state), n_pass, gamma=args.gamma,
                          trim=args.trim, density=args.density, rows=args.rows)
    # invert_y=False: everything out of scope_bake is ALREADY in scope
    # space (y up).  XYLibrary.frame() applies flip_y, and render_luma
    # builds its rows with ys = -linspace(...).  Scope.show()'s invert_y
    # is for callers handing it raw screen-space polylines; applying it
    # here flips a second time and stands the vector picture on its head.
    _scope = Scope(fps=fps, samples=args.samples, source=src, invert_y=False,
                   device=choose_device(ask=args.ask, device=args.device))
    _scope.stream.start()
    print(f"[SIDECAR] {_scope.samples_per_frame} samples/trace, "
          f"{_scope.samplerate} Hz, mode "
          f"{'RASTER' if args.raster else 'VECTOR'}"
          f"{' REALTIME' if _cfg['realtime'] else ''}, "
          f"audio latency ~{_scope.stream.latency * 1000:.0f} ms")

    import image_display
    _wrap_selector(image_display)
    print("[SIDECAR] frame-locked: image_display owns the clock and the "
          "folder selector; the scope follows its decisions.")
    try:
        image_display.run_display(settings.CLOCK_MODE)
    except KeyboardInterrupt:
        pass
    finally:
        _scope.stream.stop()
        _scope.stream.close()


if __name__ == "__main__":
    main()