"""
scope_tap.py -- frame-locked scope output driven by image_display.

image_display calls publish() once per displayed frame.  Until enable() is
called, ENABLED is False and publish() returns immediately, so the normal
display path pays nothing.

Why the video must drive this rather than the scope running its own loop:
folder_selector.update_folder_selection() is stateful AND stochastic -- its RNG
is seeded from time.time().  Two independent callers advance two independent
random sequences and pick DIFFERENT folders, so the two outputs would show
different content within seconds of the first switch, not merely drift.  So the
scope never touches the clock or the selector; it renders whatever the video
just decided.

publish() receives d_idx -- the frame actually displayed, which under load may
differ from the driving index -- so the scope follows the screen exactly.
"""
import os
import threading
from pathlib import Path

import settings
from scope_bake import XYLibrary, merge, raster_frame, SweepSource
from scope_out import Scope, choose_device

ENABLED = False

_scope = None
_source = None
_libs = {"main": [], "float": []}
_cfg = {}
_state = {"main": None, "mi": 0, "float": None, "fi": 0}
_lock = threading.Lock()


def _folder_dir(image_path, xy_root):
    """The baker mirrors the source tree, so a folder's path relative to
    IMAGES_DIR is its library's path relative to XY_DIR."""
    folder = os.path.dirname(image_path)
    rel = os.path.relpath(folder, settings.IMAGES_DIR)
    if rel.startswith(".."):
        parts = Path(folder).parts
        for key in ("face", "float"):
            if key in parts:
                rel = os.path.join(*parts[parts.index(key):])
                break
    return os.path.join(xy_root, rel)


def enable(main_paths, float_paths, xy_root=None, raster=False, realtime=False,
           fps=None, samples=None, trim=None, gamma=None, density=None,
           rows=None, min_feature=None, sweep=None, mix=None, mix_duty=None,
           device=None, ask=False):
    """
    Open the audio device and the baked libraries.  Call once at startup,
    before run_display().

    Every tuning argument defaults to None and falls back to settings.SCOPE_*,
    so an installation can be configured once in settings.py and launched with
    a bare --scope.
    """
    global ENABLED, _scope, _source

    if trim is None:
        trim = getattr(settings, "SCOPE_TRIM", 0.02)
    if gamma is None:
        gamma = getattr(settings, "SCOPE_GAMMA", 2.2)
    if density is None:
        density = getattr(settings, "SCOPE_DENSITY", 1.0)
    if min_feature is None:
        min_feature = getattr(settings, "SCOPE_MIN_FEATURE", 0.02)
    if sweep is None:
        sweep = getattr(settings, "SCOPE_SWEEP", "alternate")
    if rows is None:
        rows = getattr(settings, "SCOPE_ROWS", None)
    if mix_duty is None:
        mix_duty = getattr(settings, "SCOPE_MIX_DUTY", 0.5)
    if fps is None:
        fps = getattr(settings, "SCOPE_FPS", None)

    xy_root = xy_root or getattr(settings, "XY_DIR", "images_xy")
    for key, paths in (("main", main_paths), ("float", float_paths)):
        for f in range(len(paths[0])):
            d = _folder_dir(paths[0][f], xy_root)
            try:
                _libs[key].append(XYLibrary(d))
            except Exception as e:
                print(f"[SCOPE] warning: {key} folder {f}: no library at {d} ({e})")
                _libs[key].append(None)
    if not any(l is not None for l in _libs["main"] + _libs["float"]):
        raise RuntimeError(
            f"No XY libraries under '{xy_root}'. Bake first:\n"
            f"  python utilities/convert_to_xy.py -i {settings.IMAGES_DIR} "
            f"-o {xy_root}")

    if realtime and not raster:
        print("[SCOPE] --scope-realtime applies to raster only; ignoring.")
        realtime = False
    if mix and realtime:
        print("[SCOPE] --scope-mix and --scope-realtime are incompatible "
              "(mix needs whole passes); ignoring --scope-realtime.")
        realtime = False

    _cfg.update(raster=raster, realtime=realtime, trim=trim, gamma=gamma,
                density=density, rows=rows, min_feature=min_feature,
                sweep=sweep, mix=mix, mix_duty=min(1.0, max(0.0, mix_duty)),
                rev=False, end=None, duty_acc=0.0)

    # One trace per index by default: that maximises samples per trace, which
    # is the entire resolution budget (samples = rate / fps).
    fps = fps or getattr(settings, "IPS", 30)
    if mix:
        fps = int(round(mix))               # the switch rate IS the trace rate

    dev = choose_device(ask=ask, device=device)
    if realtime:
        probe = Scope(fps=fps, samples=samples, device=dev)
        n_pass = probe.samples_per_frame
        probe.stream.close()
        _source = SweepSource(lambda: dict(_state), n_pass, gamma=gamma,
                              trim=trim, density=density, rows=rows)
    _scope = Scope(fps=fps, samples=samples, device=dev, source=_source)
    _scope.stream.start()

    mode = "MIX" if mix else ("RASTER" if raster else "VECTOR")
    print(f"[SCOPE] frame-locked to the display | {mode}"
          f"{' REALTIME' if realtime else ''} | "
          f"{_scope.samples_per_frame} samples/trace @ {_scope.samplerate} Hz "
          f"({_scope.samplerate / _scope.samples_per_frame:.0f} passes/sec) | "
          f"latency ~{_scope.stream.latency * 1000:.0f} ms")
    ENABLED = True


def publish(d_idx, folders):
    """Called by image_display once per displayed frame.  Must stay cheap --
    this runs on the render thread."""
    if not ENABLED or _scope is None:
        return
    try:
        mf, ff = folders
        mains, floats = _libs["main"], _libs["float"]
        ml = mains[mf % len(mains)] if mains else None
        fl = floats[ff % len(floats)] if floats else None
        idx = int(d_idx)

        if _cfg["realtime"]:
            with _lock:
                _state.update(main=ml, mi=idx, float=fl, fi=idx)
            return

        use_raster = _cfg["raster"]
        if _cfg["mix"]:
            # Bresenham accumulator spreads the duty ratio evenly across
            # passes rather than clumping them into a visible beat.
            _cfg["duty_acc"] += _cfg["mix_duty"]
            if _cfg["duty_acc"] >= 1.0:
                use_raster = True
                _cfg["duty_acc"] -= 1.0
            else:
                use_raster = False

        n = _scope.samples_per_frame
        if use_raster:
            sweep = _cfg["sweep"]
            frame = raster_frame(
                ml, idx, fl, idx, n,
                gamma=_cfg["gamma"], trim=_cfg["trim"],
                density=_cfg["density"], rows=_cfg["rows"],
                palindrome=(sweep == "palindrome"),
                reverse=(sweep == "alternate" and _cfg["rev"]),
                start=_cfg["end"] if sweep == "alternate" else None,
                close=(sweep == "retrace"))
            if frame is not None:
                if sweep == "alternate":
                    _cfg["rev"] = not _cfg["rev"]
                    _cfg["end"] = frame[-1]
                _scope.show_frame(frame)
        else:
            _scope.show(merge(ml, idx, fl, idx,
                              min_feature=_cfg["min_feature"]))
    except Exception as e:
        print(f"[SCOPE] publish failed, disabling tap: {e}")
        shutdown()


def shutdown():
    global ENABLED, _scope
    ENABLED = False
    if _scope is not None:
        try:
            _scope.stream.stop()
            _scope.stream.close()
        except Exception:
            pass
        _scope = None
