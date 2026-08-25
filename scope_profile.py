"""
scope_profile.py -- measure where the CPU actually goes, on YOUR machine.

Timings taken elsewhere do not transfer: screen capture in particular is wildly
platform-dependent, and a Retina panel is 4x the pixels of the logical size.
This times each stage separately and reports it as a share of one core, so the
expensive part is obvious rather than inferred.

    python scope_profile.py                  # everything except real capture
    python scope_profile.py --source screen  # include the real grab
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def timeit(fn, n=30, warm=5):
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--source", choices=("test", "screen"), default="test")
    ap.add_argument("--region", help="left,top,width,height")
    ap.add_argument("--fps", type=int, default=30, help="traces per second")
    ap.add_argument("--capture-fps", type=float, default=12.0)
    ap.add_argument("--rate", type=int, default=96000)
    ap.add_argument("--downto", type=int, default=160)
    args = ap.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location("ss", os.path.join(ROOT_DIR, "scope_screen.py"))
    ss = importlib.util.module_from_spec(spec)
    saved, sys.argv = sys.argv, ["x"]
    spec.loader.exec_module(ss)
    sys.argv = saved
    from scope_bake import render_luma

    n = args.rate // max(args.fps, 1)
    print(f"samples/trace {n}   traces/sec {args.fps}   rate {args.rate}\n")
    rows = []

    if args.source == "screen":
        import mss
        sct = mss.mss()
        region = [int(v) for v in args.region.split(",")] if args.region else None
        mon = sct.monitors[1] if region is None else {
            "left": region[0], "top": region[1],
            "width": region[2], "height": region[3]}
        raw_holder = {}

        def raw_grab():
            raw_holder["v"] = np.asarray(sct.grab(mon))[:, :, :3]
        t_grab = timeit(raw_grab, n=20)
        shape = raw_holder["v"].shape
        rows.append(("mss grab", t_grab, args.capture_fps,
                     f"{shape[1]}x{shape[0]}"))
        raw = raw_holder["v"]
    else:
        raw = np.random.randint(0, 255, (1200, 1920, 3), dtype=np.uint8)
        rows.append(("(synthetic source)", 0.0, args.capture_fps, "1920x1200"))

    t_shrink = timeit(lambda: ss.shrink(raw, args.downto), n=30)
    rows.append(("shrink to %d wide" % args.downto, t_shrink, args.capture_fps, ""))

    lum = ss.shrink(raw, args.downto)
    t_render = timeit(lambda: render_luma(lum, n, trim=0.02, close=False), n=30)
    rows.append(("render_luma (whole trace)", t_render, args.fps, ""))

    # audio callback overhead: a copy out of a ring, times callbacks per second
    from scope_out import BufferedSource
    src = BufferedSource(lambda k: np.zeros((k, 2), np.float32),
                         blocksize=256, depth=6)
    time.sleep(0.05)
    t_cb = timeit(lambda: src(256), n=200, warm=50)
    src.close()

    print(f"{'stage':30s} {'each':>9s} {'per sec':>9s} {'% core':>8s}  note")
    print("-" * 74)
    total = 0.0
    for name, t, rate, note in rows:
        share = t * rate * 100
        total += share
        print(f"{name:30s} {t*1000:8.2f}ms {rate:8.1f}x {share:7.1f}%  {note}")

    for bs in (64, 128, 256, 512, 1024):
        cps = args.rate / bs
        share = t_cb * cps * 100
        mark = "  <- PortAudio often picks this on CoreAudio" if bs == 64 else ""
        print(f"{'audio callback @ %d' % bs:30s} {t_cb*1000:8.3f}ms "
              f"{cps:8.1f}x {share:7.1f}%{mark}")

    print("-" * 74)
    print(f"{'pipeline total (excl. callback)':30s} {'':9s} {'':9s} {total:7.1f}%")
    print()
    print("If the callback row dominates, set SCOPE_BLOCKSIZE in settings.py")
    print("(or --blocksize) to 512 or 1024: same work, far fewer wake-ups.")


if __name__ == "__main__":
    main()