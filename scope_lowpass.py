"""
scope_lowpass.py -- put a deliberately aggressive low-pass in the XY path.

Why you would want this:

  * Emulate a worse output chain.  A Pi's headphone jack or a cheap USB codec
    has a lower, softer reconstruction filter than a good interface.  Filtering
    on the dev machine shows what the deployment target will actually draw.
  * Find out whether the resolution is real.  Detail finer than the output can
    pass is fiction -- the DAC smooths it away.  Sweep the cutoff down and the
    grid size where the picture stops changing is the grid size worth baking.
  * Audition a physical RC filter before building one.

Two filter implementations, because the two signal paths need different things:

  lowpass_circular()  for frame playback.  The frame LOOPS, so the filter has
                      to be periodic or the loop seam clicks and the beam
                      jumps.  Done in the frequency domain over the whole
                      frame, which is exact for a repeating signal, and
                      zero-phase so the image does not rotate or shift.

  CascadedOnePole     for --realtime streaming, where samples arrive in chunks
                      and the filter must carry state across them.

Works identically for hardware and virtual routing: the filter sits in the
sample path before the device, so BlackHole -> a virtual scope sees exactly
what a real interface -> a real scope sees.

    # audition on the bench pattern, cycling cutoffs
    python scope_lowpass.py --device BlackHole --sweep

    # one cutoff, your own baked content
    python scope_lowpass.py --xy-dir images_xy --bg 1 --fg 1 --cutoff 2500

    # no audio, just the picture comparison
    python scope_lowpass.py --xy-dir images_xy --bg 1 --fg 1 --png lp.png
"""
import argparse
import math
import os
import sys
import time

import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


# ---------------------------------------------------------------- filters

def lowpass_circular(frame, cutoff_hz, samplerate, order=4):
    """
    Zero-phase Butterworth-magnitude low-pass applied circularly.

    The frame repeats every len(frame) samples, so its spectrum is exactly the
    harmonics of samplerate/len(frame).  Shaping those bins filters the
    repeating signal with no seam and no phase shift -- an ordinary IIR would
    ring at the loop point and rotate the image.
    """
    frame = np.asarray(frame, dtype=np.float64)
    n = len(frame)
    if not cutoff_hz or cutoff_hz <= 0 or cutoff_hz >= samplerate / 2 or n < 4:
        return frame.astype(np.float32)
    freqs = np.fft.rfftfreq(n, d=1.0 / samplerate)
    with np.errstate(divide="ignore", over="ignore"):
        mag = 1.0 / np.sqrt(1.0 + (freqs / float(cutoff_hz)) ** (2 * order))
    out = np.empty_like(frame)
    for ch in range(frame.shape[1]):
        out[:, ch] = np.fft.irfft(np.fft.rfft(frame[:, ch]) * mag, n=n)
    return out.astype(np.float32)


class CascadedOnePole:
    """Stateful low-pass for streamed chunks (realtime mode).

    N one-pole sections in series: 6*N dB/octave.  Introduces a little phase
    lag, unlike the circular version, which shows up as a slight lean on fast
    strokes -- that is honest, since a real analog filter does the same.
    """

    def __init__(self, cutoff_hz, samplerate, order=4, channels=2):
        self.order = max(1, int(order))
        self.set_cutoff(cutoff_hz, samplerate)
        self.z = np.zeros((self.order, channels), dtype=np.float64)

    def set_cutoff(self, cutoff_hz, samplerate):
        self.enabled = bool(cutoff_hz and 0 < cutoff_hz < samplerate / 2)
        if self.enabled:
            self.a = math.exp(-2.0 * math.pi * float(cutoff_hz) / samplerate)
        else:
            self.a = 0.0

    def process(self, x):
        if not self.enabled:
            return x
        x = np.asarray(x, dtype=np.float64)
        a, b = self.a, 1.0 - self.a
        for s in range(self.order):
            zs = self.z[s]
            out = np.empty_like(x)
            for i in range(len(x)):          # sample loop: chunks are small
                zs = a * zs + b * x[i]
                out[i] = zs
            self.z[s] = zs
            x = out
        return x.astype(np.float32)


def describe(cutoff_hz, samplerate, samples_per_trace, cols):
    """Translate a cutoff into what it means for the picture."""
    trace_hz = samplerate / max(samples_per_trace, 1)
    cell_hz = samplerate / max(samples_per_trace / max(cols, 1), 1)
    return (f"cutoff {cutoff_hz:>6.0f} Hz | trace rate {trace_hz:5.1f} Hz | "
            f"cell rate {cell_hz / 1000:5.1f} kHz | "
            f"{'cells SURVIVE' if cutoff_hz >= cell_hz else 'cells SMEARED'}")


# ---------------------------------------------------------------- sources

def bench_frame(n):
    """Circle plus rotating square -- known geometry, so filter effects are
    unambiguous rather than confused with content."""
    from scope_out import rasterize
    th = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    circle = np.stack([0.7 * np.cos(th), 0.7 * np.sin(th)], axis=1)
    circle = np.vstack([circle, circle[:1]])
    s = 0.4
    square = np.array([[-s, -s], [s, -s], [s, s], [-s, s], [-s, -s]])
    spokes = [np.array([[0, 0], [0.7 * math.cos(a), 0.7 * math.sin(a)]])
              for a in np.linspace(0, np.pi, 5, endpoint=False)]
    return rasterize([circle, square] + spokes, n)


def baked_frame(xy_dir, bg, fg, index, n, raster, trim):
    from scope_bake import XYLibrary, merge, raster_frame
    from scope_out import rasterize
    from pathlib import Path
    root = Path(xy_dir)
    dirs = sorted(p.parent for p in root.rglob("thumbs.npy")) or \
        sorted(p.parent for p in root.rglob("verts.npy"))
    mains = [d for d in dirs if "float" not in d.relative_to(root).parts]
    floats = [d for d in dirs if "float" in d.relative_to(root).parts]
    if not mains or not floats:
        raise SystemExit(f"need main and float libraries under {root}")
    m = XYLibrary(mains[bg % len(mains)])
    f = XYLibrary(floats[fg % len(floats)])
    i = index % min(len(m), len(f))
    if raster:
        fr = raster_frame(m, i, f, i, n, trim=trim, close=False)
        if fr is None:
            raise SystemExit("no thumbnails in this bake; omit --raster")
        return fr
    return rasterize(merge(m, i, f, i), n)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutoff", type=float, default=3000.0,
                    help="low-pass corner in Hz (default 3000; try 800-8000)")
    ap.add_argument("--order", type=int, default=4,
                    help="filter order; 4 = 24 dB/octave (default)")
    ap.add_argument("--sweep", action="store_true",
                    help="cycle the cutoff downward while playing, so you can "
                         "watch the picture dissolve")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="play for this long (0 = until Ctrl+C, no audio if "
                         "--png given and no device)")
    ap.add_argument("--fps", type=int, default=30, help="trace rate")
    ap.add_argument("--samples", type=int, help="samples per trace (overrides --fps)")
    ap.add_argument("--device", help="audio output: index or name fragment")
    ap.add_argument("--ask", action="store_true", help="choose the output interactively")
    ap.add_argument("--no-audio", action="store_true", help="picture only")
    ap.add_argument("--png", help="write a before/after comparison image here")
    ap.add_argument("--xy-dir", help="use baked content instead of the bench pattern")
    ap.add_argument("--bg", type=int, default=1)
    ap.add_argument("--fg", type=int, default=1)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--raster", action="store_true", help="raster instead of vector")
    ap.add_argument("--trim", type=float, default=0.10)
    args = ap.parse_args()

    from scope_out import Scope, choose_device

    scope = None
    samplerate = 48000
    if not args.no_audio:
        try:
            scope = Scope(fps=args.fps, samples=args.samples,
                          device=choose_device(ask=args.ask, device=args.device))
            samplerate = scope.samplerate
        except Exception as e:
            print(f"[LP] no audio device ({e}); picture only")
    n = (scope.samples_per_frame if scope
         else (args.samples or int(48000 / max(args.fps, 1))))

    if args.xy_dir:
        frame = baked_frame(args.xy_dir, args.bg, args.fg, args.index, n,
                            args.raster, args.trim)
    else:
        frame = bench_frame(n)

    cols = 48
    print(f"[LP] {n} samples/trace @ {samplerate} Hz, order {args.order}")
    print(f"[LP] {describe(args.cutoff, samplerate, n, cols)}")

    if args.png:
        try:
            import cv2, importlib.util
            spec = importlib.util.spec_from_file_location("t", "test_scope_pair.py")
            t = importlib.util.module_from_spec(spec)
            saved, sys.argv = sys.argv, ["x"]
            spec.loader.exec_module(t)
            sys.argv = saved
            cuts = [None, args.cutoff * 4, args.cutoff, args.cutoff / 3]
            panels = []
            for c in cuts:
                fr = frame if c is None else lowpass_circular(frame, c, samplerate,
                                                              args.order)
                lab = "unfiltered" if c is None else f"{c:.0f} Hz"
                panels.append(t.annotate(t.render_trace(fr, 400, exposure=0.5), lab))
            cv2.imwrite(args.png, np.hstack(panels))
            print(f"[LP] wrote {args.png}")
        except Exception as e:
            print(f"[LP] could not render comparison: {e}")

    if scope is None:
        return

    scope.stream.start()
    print("[LP] playing -- Ctrl+C to stop")
    t0 = time.time()
    try:
        while True:
            if args.sweep:
                # walk the corner down over 20 s, then jump back up
                phase = ((time.time() - t0) % 20.0) / 20.0
                cut = args.cutoff * (4.0 ** (1.0 - 2.0 * phase))
            else:
                cut = args.cutoff
            scope.show_frame(lowpass_circular(frame, cut, samplerate, args.order))
            if args.sweep:
                print(f"\r[LP] {describe(cut, samplerate, n, cols)}", end="", flush=True)
            time.sleep(0.1)
            if args.seconds and time.time() - t0 > args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        print()
        scope.stream.stop()
        scope.stream.close()


if __name__ == "__main__":
    main()