"""
bake_advisor.py -- what --thumb-width does YOUR content actually need?

    python bake_advisor.py --xy-dir images_xy
    python bake_advisor.py --xy-dir images_xy --rate 96000 --fps 30 --fields 2

The thumbnail is a hard ceiling on grid size, and storage scales with its
square: width 96 is about 2 GB for 32 folders x 2221 frames, width 128 about
3.5 GB.  So the temptation is to pick the smaller one.

The catch is that the right answer is content-dependent and not monotonic.
autofit grows the grid by 1/sqrt(the fraction of cells surviving trim), capped
at 2.5x, so how much ceiling you need depends on how much of the frame the
subject fills:

    subject fills 90% of frame  -> grid  49x73   width 96 is plenty
    subject fills 50%           -> grid  71x106  width 96 is plenty
    subject fills 30%           -> grid 115x172  width 96 CLIPS, 128 does not
    subject fills 15%           -> grid  46x69   width 96 is plenty

A mid-sized subject is the worst case: small enough that trim discards most of
the frame, big enough that autofit's growth is not wasted.  Guessing from the
average is exactly wrong -- it is the mid-sized frames that decide.

So this reads the bake you already have, runs the real sizing rule over real
frames, and reports what each width would give you.  It changes nothing.
"""
import argparse
import json
import os
import sys

import numpy as np


def _fake_audio():
    """scope_bake pulls in scope_out, which imports sounddevice."""
    import types
    if "sounddevice" in sys.modules:
        return
    m = types.ModuleType("sounddevice")
    m.query_devices = lambda *a, **k: {"name": "stub", "default_samplerate": 96000}
    m.OutputStream = object
    m.default = types.SimpleNamespace(device=(0, 1))
    sys.modules["sounddevice"] = m


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xy-dir", required=True, help="a baked library tree")
    ap.add_argument("--rate", type=int, default=96000, help="device sample rate")
    ap.add_argument("--fps", type=int, default=30, help="picture rate (IPS)")
    ap.add_argument("--fields", type=int, default=1, help="interlace fields")
    ap.add_argument("--trim", type=float, default=0.10)
    ap.add_argument("--density", type=float, default=1.0)
    ap.add_argument("--frames", type=int, default=40,
                    help="frames sampled per folder (default 40)")
    ap.add_argument("--format", action="store_true",
                    help="also report what the STORAGE format costs and what "
                         "the alternatives would give on this content")
    args = ap.parse_args()

    _fake_audio()
    from scope_bake import plan_grid

    n = args.rate // max(args.fps * max(args.fields, 1), 1)
    print(f"budget: {n} samples/trace at {args.rate} Hz / {args.fps} fps"
          f"{f' / {args.fields} fields' if args.fields > 1 else ''}\n")

    folders = []
    for root, _dirs, files in os.walk(args.xy_dir):
        if "thumbs.npy" in files:
            folders.append(root)
    if not folders:
        print(f"No thumbs.npy under {args.xy_dir} -- was this baked --no-thumbs?")
        return 1
    folders.sort()

    worst = {}          # width -> (rows, cols, clipped_count)
    baked_width = None
    total_bytes = {}
    n_frames_total = 0

    for fd in folders:
        try:
            T = np.load(os.path.join(fd, "thumbs.npy"), mmap_mode="r")
        except Exception as e:
            print(f"  skip {fd}: {e}")
            continue
        if T.ndim != 4:
            continue
        count = len(T)
        n_frames_total += count
        idx = np.unique(np.linspace(0, count - 1, min(args.frames, count)).astype(int))
        th, tw = T.shape[1], T.shape[2]
        baked_width = tw

        for i in idx:
            t = np.asarray(T[i], dtype=np.float32)
            lum = (t[..., 0] / 255.0) * (t[..., 1] / 255.0)
            if lum.max() <= 0.02:
                continue                       # blank frame, tells us nothing
            # Rescale the SAME picture to each candidate width, so the
            # comparison is about the ceiling and not about which frames the
            # existing bake happens to contain.
            for w in (64, 80, 96, 128, 160):
                h = max(1, int(round(th * w / float(tw))))
                ys = np.linspace(0, lum.shape[0] - 1, h).astype(int)
                xs = np.linspace(0, lum.shape[1] - 1, w).astype(int)
                small = lum[np.ix_(ys, xs)]
                r, c = plan_grid(small, n, density=args.density, trim=args.trim,
                                 fields=args.fields)
                clipped = (r >= h) or (c >= w)
                pr, pc, pcl = worst.get(w, (0, 0, 0))
                worst[w] = (max(pr, r), max(pc, c), pcl + int(clipped))
                total_bytes[w] = h * w * 2

    print(f"{len(folders)} folders, {n_frames_total} frames total, "
          f"up to {args.frames} sampled per folder\n")
    print(f"{'width':>6} {'ceiling':>10} {'largest grid':>14} {'clipped':>9} "
          f"{'bytes/frame':>12} {'whole bake':>11}")
    print("-" * 68)
    best = None
    for w in sorted(worst):
        r, c, cl = worst[w]
        h = total_bytes[w] // (w * 2)
        size = total_bytes[w] * n_frames_total
        flag = "CLIPPED" if cl else "ok"
        if not cl and best is None:
            best = w
        print(f"{w:>6} {f'{w}x{h}':>10} {f'{c}x{r}':>14} {flag:>9} "
              f"{total_bytes[w]:>12} {size / 1e9:>10.2f} GB")

    print()
    if best is None:
        print("Every width clipped. Your content wants a bigger thumbnail than")
        print("160 -- or trim is high enough that autofit is hitting its 2.5x")
        print("cap. Check --trim before rebaking larger.")
    else:
        cur = baked_width or max(worst)
        saving = (total_bytes.get(cur, 0) - total_bytes[best]) * n_frames_total
        print(f"You baked at --thumb-width {baked_width}.")
        print(f"Smallest width that never clips on this content: --thumb-width {best}")
        if best < cur and saving > 0:
            print(f"Going from {cur} to {best} saves {saving / 1e9:.2f} GB "
                  f"and loses no grid, because the sample budget runs out first.")
        else:
            print("Nothing smaller would do -- the ceiling is doing real work here.")
    print("\nRebake is required to change this: --thumb-width is baked in.")
    if args.format:
        _format_report(folders, n_frames_total, args)
    return 0


def _format_report(folders, n_frames_total, args):
    """What the current layout costs, and what the alternatives would give.

    Ratios are wholly content-dependent -- a soft portrait on black compresses
    an order of magnitude better than fabric or hair -- so this measures YOUR
    frames instead of quoting a number from somewhere else.
    """
    print("\n" + "=" * 68)
    print("STORAGE FORMAT")
    print("=" * 68)
    sample = []
    for fd in folders[:8]:
        try:
            T = np.load(os.path.join(fd, "thumbs.npy"), mmap_mode="r")
        except Exception:
            continue
        idx = np.unique(np.linspace(0, len(T) - 1,
                                    min(24, len(T))).astype(int))
        sample.append(np.asarray(T[idx]))
    if not sample:
        print("no thumbnails to measure")
        return
    S = np.concatenate(sample)
    lum, alpha = S[..., 0], S[..., 1]
    per_frame = S[0].nbytes
    total = per_frame * n_frames_total

    print(f"\nsampled {len(S)} frames from {len(sample)} folders")
    print(f"current: raw uint8 [luma, alpha] interleaved, {per_frame} B/frame")
    print(f"         whole bake {total / 1e9:.2f} GB\n")
    print(f"  alpha fully transparent : {100 * (alpha == 0).mean():5.1f}% of pixels")
    print(f"  alpha fully opaque      : {100 * (alpha == 255).mean():5.1f}%")
    print(f"  matte edge (soft alpha) : {100 * ((alpha > 0) & (alpha < 255)).mean():5.1f}%")
    invisible = (alpha == 0).mean() / 2.0
    print(f"  -> {100 * invisible:.0f}% of the file is luma behind transparent "
          "pixels: stored, then multiplied by zero at runtime")

    try:
        import zstandard as zstd
    except ImportError:
        print("\n(pip install zstandard to measure the compressed options)")
        return

    cc = zstd.ZstdCompressor(level=3)
    dc = zstd.ZstdDecompressor()
    prem = np.stack([(lum.astype(np.uint16) * alpha // 255).astype(np.uint8),
                     alpha], axis=-1)
    variants = [
        ("raw interleaved (current)", S.tobytes(), False),
        ("zstd-3, as-is", S.tobytes(), True),
        ("zstd-3, premultiplied", prem.tobytes(), True),
        ("zstd-3, premultiplied + planar",
         np.concatenate([prem[..., 0].ravel(), prem[..., 1].ravel()]).tobytes(), True),
    ]
    base = S.nbytes
    print(f"\n{'layout':34} {'ratio':>7} {'whole bake':>12}")
    print("-" * 56)
    for name, buf, comp in variants:
        size = len(cc.compress(buf)) if comp else len(buf)
        ratio = base / size
        print(f"{name:34} {ratio:>6.2f}x {total / ratio / 1e9:>11.2f} GB")

    one = cc.compress(prem[0].tobytes())
    import time
    for _ in range(5):
        dc.decompress(one)
    t = time.perf_counter()
    for _ in range(200):
        dc.decompress(one)
    ms = (time.perf_counter() - t) / 200 * 1000
    print(f"\ndecompress one frame: {ms:.3f} ms "
          f"-> {ms * args.fps / 10:.2f}% of a core at {args.fps} fps")
    print("Compression costs the mmap: frames would be seek-and-decompress")
    print("rather than zero-copy. At this cost that is affordable, and a bake")
    print("small enough to sit in page cache may well READ faster than a 3.5 GB")
    print("one that cannot.")


if __name__ == "__main__":
    sys.exit(main())