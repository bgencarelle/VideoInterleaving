"""
test_scope_parity.py -- the two scope paths must stay one algorithm.

Run from the repo root:  python test_scope_parity.py

`--mode scope` and `scope_screen.py` drive the same hardware with the same
engine and differ only in where the picture comes from: one composites two
baked libraries, the other grabs the screen.  They have drifted twice.

  1. SweepSource and raster_frame were two implementations of one algorithm
     and produced visibly different output.
  2. scope_screen was still re-deriving its grid every frame long after mode
     scope had moved to a fixed one, and picked up `border` and `oversample`
     only when someone remembered to port them.

Extracting scope_bake.TraceEmitter fixes the second one.  It does not PREVENT
the next one: nothing stops a parameter being added to one caller and not the
other.  This test does.  If you add a tuning knob, it belongs on TraceEmitter,
and this file fails until both callers get it.
"""
import sys

import numpy as np


def _fake_audio():
    """scope_out imports sounddevice; this test never opens a device."""
    import types
    if "sounddevice" in sys.modules:
        return
    m = types.ModuleType("sounddevice")
    m.query_devices = lambda *a, **k: {"name": "stub", "default_samplerate": 96000}
    m.OutputStream = object
    m.default = types.SimpleNamespace(device=(0, 1))
    sys.modules["sounddevice"] = m


def _subject(h=128, w=96, cx=48, cy=64):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    head = np.exp(-(((xx - cx) / 22.) ** 2 + ((yy - cy) / 38.) ** 2))
    eyes = (np.exp(-(((xx - cx + 12) / 4.) ** 2 + ((yy - cy + 14) / 3.) ** 2))
            + np.exp(-(((xx - cx - 12) / 4.) ** 2 + ((yy - cy + 14) / 3.) ** 2)))
    lum = np.clip(head - 0.85 * eyes, 0, 1).astype(np.float32)
    lum[lum < 0.05] = 0.0
    return lum


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def main():
    _fake_audio()
    from scope_bake import TraceEmitter, render_luma, plan_grid
    ok = True
    lum = _subject()
    N, RATE = 3200, 96000

    print("TraceEmitter reproduces a hand-rolled render_luma call")
    e = TraceEmitter(RATE, N, gamma=2.2, trim=0.10, sweep="alternate")
    a = e.emit(lum)
    b = render_luma(lum, N, gamma=2.2, trim=0.10, fields=1, field=0,
                    reverse=False, start=None, close=False)
    ok &= check("first trace identical", np.array_equal(a, b),
                f"max delta {np.abs(a - b).max():.2e}" if a.shape == b.shape else "shape differs")

    print("\nchaining: each trace continues from the last")
    e2 = TraceEmitter(RATE, N, gamma=2.2, trim=0.10, sweep="alternate")
    prev, worst = None, 0.0
    for _ in range(8):
        f = e2.emit(lum)
        if prev is not None:
            worst = max(worst, float(np.hypot(*(f[0] - prev))))
        prev = f[-1]
    ok &= check("no flyback-sized jump between traces", worst < 0.25,
                f"largest seam {worst:.3f} (an unchained loop is ~1.3)")

    print("\ninterlace: fields partition the rows exactly")
    for fields in (2, 4):
        gr, gc = plan_grid(lum, N, trim=0.10)
        e3 = TraceEmitter(RATE, N // fields, trim=0.10, fields=fields,
                          grid=(gr, gc))
        seen = set()
        for _ in range(fields):
            seen |= set(np.round(e3.emit(lum)[:, 1], 6).tolist())
        one = TraceEmitter(RATE, N, trim=0.10, grid=(gr, gc)).emit(lum)
        base = len(set(np.round(one[:, 1], 6).tolist()))
        ok &= check(f"fields={fields} covers every scanline",
                    len(seen) >= base - 1, f"{len(seen)} vs {base} progressive")

    print("\nevery tuning knob is reachable through TraceEmitter")
    import inspect
    params = set(inspect.signature(TraceEmitter.__init__).parameters)
    for knob in ("gamma", "trim", "density", "rows", "fields", "border",
                 "oversample", "sweep", "dc_comp", "grid", "levels", "autofit"):
        ok &= check(f"knob '{knob}'", knob in params)

    print("\nboth callers go through TraceEmitter, not render_luma directly")
    import re
    for path in ("scope_display.py", "scope_screen.py"):
        try:
            src = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            print(f"  SKIP  {path} not present")
            continue
        # a bare render_luma/raster_frame call in a driver means a second
        # parameter list, which is exactly how the last two drifts happened
        direct = re.findall(r"^\s*(?:\w+\s*=\s*)?(render_luma|raster_frame)\(",
                            src, re.M)
        ok &= check(f"{path} has no direct render call", not direct,
                    f"found {direct}" if direct else "")

    print()
    print("PARITY OK" if ok else "PARITY BROKEN -- the two paths have diverged")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())