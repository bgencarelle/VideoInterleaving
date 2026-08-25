"""
scope_screen.py -- use an oscilloscope as a (very) low resolution display.

Feeds any live source into the same dwell-modulated sweep the VideoInterleaving
scope mode uses, so anything you can put in a numpy array can go on the tube:

    python scope_screen.py --source screen --device BlackHole
    python scope_screen.py --source screen --region 0,0,800,600
    python scope_screen.py --source video --file clip.mp4
    python scope_screen.py --source test

BE REALISTIC ABOUT THE RESOLUTION.  The grid is bounded by samples per trace,
which is sample_rate / fps -- 3200 at 96 kHz and 30 fps.  A full-frame source
has no dark margins for trim to reclaim, so that is about 56x56 cells,
grayscale.  Good for silhouettes, large shapes, a clock, a moving figure.
Not for text and not for a desktop UI.

Lower --fps for a bigger grid at the cost of refresh:

    48 kHz, 30 fps -> 1600 samples -> ~40x40
    96 kHz, 30 fps -> 3200 samples -> ~56x56
    96 kHz, 15 fps -> 6400 samples -> ~80x80   (15 Hz flickers on short phosphor)

Screen capture needs `mss` (pip install mss).  On Wayland mss cannot grab the
screen; use X11, or feed frames another way with --source video.
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scope_bake import SweepSource, render_luma, plan_grid   # noqa: E402
from scope_out import (Scope, BufferedSource, choose_device,  # noqa: E402
                       precompensate_hpf)


# ---------------------------------------------------------------- sources

def shrink(raw, downto):
    """Downscale a captured frame to `downto` pixels wide, cheaply.

    cv2.resize with INTER_AREA straight from 3840 to 256 averages 15x15 blocks
    per output pixel and costs ~52 ms on a 4K frame -- more than a core at 30
    grabs/sec.  Striding first is a VIEW, so it costs nothing, and the area
    average then runs on an image 16x smaller.  Measured 12.8x faster, and the
    residual aliasing is invisible once the grid is ~56 cells wide.
    """
    import cv2
    h, w = raw.shape[:2]
    if w <= downto:
        src = raw
    else:
        # stride down to roughly 2x the target, then area-average the rest
        step = max(1, int(w // (downto * 2)))
        src = raw[::step, ::step]
        sh, sw = src.shape[:2]
        if sw > downto:
            src = cv2.resize(src, (downto, max(1, int(sh * downto / sw))),
                             interpolation=cv2.INTER_AREA)
    return (src[:, :, 2] * 0.2126 + src[:, :, 1] * 0.7152
            + src[:, :, 0] * 0.0722) / 255.0


def screen_source(region=None, downto=160):
    """Grab the screen (or a region) as luminance.

    np.asarray(shot) copies the whole BGRA buffer -- ~30 MB on a Retina panel,
    every grab.  np.frombuffer wraps the same bytes without copying, which is
    free.  Capture is by far the dominant cost of screen mode (measured 34 ms
    per grab on a Retina Mac against 1 ms for the trace build), so the copy is
    worth removing even though the win is only part of it.
    """
    try:
        from mss import MSS as _MSS          # newer API
    except ImportError:
        from mss import mss as _MSS

    sct = _MSS()
    mon = sct.monitors[1] if region is None else {
        "left": region[0], "top": region[1],
        "width": region[2], "height": region[3]}

    def grab():
        shot = sct.grab(mon)
        raw = np.frombuffer(shot.raw, dtype=np.uint8).reshape(
            shot.height, shot.width, 4)
        return shrink(raw[:, :, :3], downto)
    return grab


def ffmpeg_source(width=160, fps=12, region=None, input_spec=None,
                  display=None):
    """
    Capture via ffmpeg instead of in Python.

    mss goes through CoreGraphics on macOS and costs ~34 ms per grab at Retina
    backing resolution -- more than everything else in the pipeline combined.
    ffmpeg uses the platform's fast path (avfoundation / x11grab / gdigrab) AND
    does the downscale and grayscale conversion itself, so Python receives a
    ~160x104 single-channel frame and does no image work at all.

    The output size is pinned explicitly so each frame is a known number of
    bytes; guessing it from ffmpeg's stderr is fragile.
    """
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found. brew install ffmpeg / apt install ffmpeg")

    # source dimensions, so the scaled height is exact (no grab required)
    try:
        import mss as _mss
        with (_mss.MSS() if hasattr(_mss, "MSS") else _mss.mss()) as _s:
            mon = _s.monitors[1]
            sw, sh = mon["width"], mon["height"]
    except Exception:
        sw, sh = 1920, 1080
    if region:
        sw, sh = region[2], region[3]

    w = int(width)
    h = max(2, int(round(w * sh / float(sw))) // 2 * 2)

    if input_spec:
        fmt, src = input_spec.split(":", 1)
    elif sys.platform == "darwin":
        fmt, src = "avfoundation", f"{display if display is not None else 1}:none"
    elif sys.platform.startswith("win"):
        fmt, src = "gdigrab", "desktop"
    else:
        fmt, src = "x11grab", os.environ.get("DISPLAY", ":0.0")

    cmd = ["ffmpeg", "-loglevel", "error", "-f", fmt,
           "-framerate", str(int(max(fps, 1)))]
    if fmt == "x11grab" and region:
        cmd += ["-video_size", f"{region[2]}x{region[3]}",
                "-i", f"{src}+{region[0]},{region[1]}"]
    else:
        cmd += ["-i", src]
    cmd += ["-vf", f"scale={w}:{h}", "-pix_fmt", "gray",
            "-f", "rawvideo", "-an", "-sn", "-"]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, bufsize=0)
    nbytes = w * h
    last = [np.zeros((h, w), np.float32)]

    def grab():
        buf = proc.stdout.read(nbytes)
        if not buf or len(buf) < nbytes:
            return last[0]
        last[0] = (np.frombuffer(buf, np.uint8).reshape(h, w)
                   .astype(np.float32) / 255.0)
        return last[0]

    grab.proc = proc
    grab.size = (w, h)
    return grab


class Throttled:
    """Capture on a background thread at its own rate.

    The sweep asks for a frame every trace, but a ~56 cell display does not
    need the screen sampled 30 times a second, and capture is the expensive
    part.  Decoupling means the generator never waits on a grab -- it uses
    whatever the last one produced.
    """

    def __init__(self, grab, fps=12.0):
        import threading
        self._grab = grab
        self._latest = grab()
        self._stop = threading.Event()
        self.captures = 0
        self._period = 1.0 / max(fps, 0.5)
        self._t = threading.Thread(target=self._run, daemon=True,
                                   name="scope-capture")
        self._t.start()

    def _run(self):
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                self._latest = self._grab()
                self.captures += 1
            except Exception:
                pass
            time.sleep(max(0.0, self._period - (time.perf_counter() - t0)))

    def __call__(self):
        return self._latest

    def close(self):
        self._stop.set()
        self._t.join(timeout=0.5)


def video_source(path, downto=160, loop=True):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")

    def grab():
        ok, frame = cap.read()
        if not ok:
            if loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
            if not ok:
                return np.zeros((64, 64), np.float32)
        return shrink(frame, downto)
    return grab


def test_source():
    """Known geometry, so you can tell distortion from content."""
    t = [0]

    def grab():
        t[0] += 1
        y, x = np.mgrid[0:120, 0:160].astype(np.float32)
        cx, cy = 80 + 45 * np.sin(t[0] / 40.0), 60 + 30 * np.cos(t[0] / 55.0)
        img = 0.10 + 0.35 * ((((x // 20) + (y // 20)) % 2) == 0)
        img += 0.55 * (((x - cx) ** 2 + (y - cy) ** 2) < 260)
        return np.clip(img, 0, 1)
    return grab


# ---------------------------------------------------------------- main

def _profile(args, grab):
    """Report the real cost split on this machine.

    Worth doing rather than guessing: screen capture on macOS goes through
    CoreGraphics and can dwarf everything downstream, but it cannot be
    measured anywhere except on the machine doing it.
    """
    import cv2

    n = int(args.samples or (96000 // max(args.fps, 1)))

    inner = grab._grab if hasattr(grab, "_grab") else grab
    ffmpeg_mode = hasattr(grab, "proc") or hasattr(getattr(grab, "_grab", None), "proc")
    for _ in range(3):
        inner()
    t0 = time.perf_counter()
    for _ in range(20):
        inner()
    cap = (time.perf_counter() - t0) / 20 * 1000

    lum = np.asarray(inner(), dtype=np.float32)
    for _ in range(3):
        render_luma(lum, n, trim=args.trim, close=False)
    t0 = time.perf_counter()
    for _ in range(30):
        render_luma(lum, n, trim=args.trim, close=False)
    build = (time.perf_counter() - t0) / 30 * 1000

    cap_hz = args.capture_fps
    print()
    if ffmpeg_mode:
        print(f"  pipe read + convert : {cap:6.2f} ms x {cap_hz:.0f}/s "
              f"= {cap * cap_hz / 10:5.1f}% of a core   (Python side only;")
        print( "                        ffmpeg does the capture and scaling in "
               "its own process)")
    else:
        print(f"  capture + downscale : {cap:6.2f} ms x {cap_hz:.0f}/s "
              f"= {cap * cap_hz / 10:5.1f}% of a core")
    print(f"  trace build         : {build:6.2f} ms x {args.fps}/s "
          f"= {build * args.fps / 10:5.1f}% of a core")
    bs = args.blocksize or 256
    print(f"  audio callbacks     : {96000 / bs:6.0f}/s at blocksize {bs} "
          "(Python overhead per wake-up)")
    print()
    if cap * cap_hz / 10 > 5 and args.source == "screen":
        print("  capture dominates. How it scales with region size on THIS "
              "machine:")
        for wh in ((1920, 1200), (1280, 800), (800, 600), (640, 400)):
            try:
                g = screen_source((0, 0, wh[0], wh[1]), downto=args.downto)
                for _ in range(2):
                    g()
                t0 = time.perf_counter()
                for _ in range(10):
                    g()
                ms = (time.perf_counter() - t0) / 10 * 1000
                print(f"    --region 0,0,{wh[0]},{wh[1]:<5} {ms:6.2f} ms "
                      f"= {ms * cap_hz / 10:5.1f}% of a core at "
                      f"{cap_hz:.0f} grabs/s")
            except Exception as e:
                print(f"    --region 0,0,{wh[0]},{wh[1]}: {e}")
        print()
        print("  Also worth knowing: this is a macOS problem. Capture there "
              "goes through")
        print("  CoreGraphics and is slow. On Linux/X11 the same grab uses "
              "XShm and is")
        print("  typically 5-10x cheaper, so a Pi 5 may well beat the Mac at "
              "this.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[1],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("screen", "ffmpeg", "video", "test"),
                    default="test",
                    help="screen = mss (simple, slow on macOS); ffmpeg = the "
                         "platform fast path, captures and downscales natively")
    ap.add_argument("--ffmpeg-input", metavar="FMT:SRC",
                    help="override ffmpeg input, e.g. avfoundation:1:none")
    ap.add_argument("--display", type=int,
                    help="avfoundation screen index (see: ffmpeg -f "
                         "avfoundation -list_devices true -i \"\")")
    ap.add_argument("--file", help="video file for --source video")
    ap.add_argument("--region", help="screen region as left,top,width,height")
    ap.add_argument("--fps", type=int, default=30,
                    help="traces per second; lower = bigger grid, more flicker")
    ap.add_argument("--samples", type=int, help="samples per trace (overrides --fps)")
    ap.add_argument("--trim", type=float, default=0.02,
                    help="drop cells dimmer than this. Useful on dark content; "
                         "leave low for a full-frame source")
    ap.add_argument("--gamma", type=float, default=1.8)
    ap.add_argument("--fields", type=int, default=1, metavar="N",
                    help="Interlace: N=2 draws every other row per trace and "
                         "alternates, so the beam repaints at N x the picture "
                         "rate on the SAME grid. Refresh without paying "
                         "resolution for it. Use --fps N x capture-fps.")
    ap.add_argument("--density", type=float, default=1.0)
    ap.add_argument("--rows", type=int, help="scanline count (default: auto)")
    ap.add_argument("--adapt", type=float, default=4.0, metavar="SEC",
                    help="tone-mapping time constant. Live content cannot be "
                         "pre-scanned, but adapting fast makes cells flicker, "
                         "so this is deliberately slow. 0 = fixed 0..1")
    ap.add_argument("--capture-fps", type=float, default=12.0,
                    help="how often to grab the source, independent of the "
                         "trace rate. Capture is the expensive part and a ~56 "
                         "cell display does not need 30 grabs a second.")
    ap.add_argument("--downto", type=int, default=160,
                    help="width to shrink the source to before gridding "
                         "(default 160; the grid is ~56 wide, so this is "
                         "already oversampled)")
    ap.add_argument("--stream", action="store_true",
                    help="generate row by row for lowest latency. Costs ~3x "
                         "the CPU because each row is a handful of tiny numpy "
                         "calls; the default builds a whole trace at once, "
                         "which is what you want unless latency matters.")
    ap.add_argument("--buffer-blocks", type=int, default=6,
                    help="blocks queued ahead of the audio callback")
    ap.add_argument("--dc-comp", type=float, metavar="HZ",
                    help="cancel the output's AC coupling (try 20-50). The "
                         "vertical sweep runs at the trace rate, so a headphone "
                         "jack funnels the picture without this.")
    ap.add_argument("--blocksize", type=int, default=1024,
                    help="audio callback size. Bigger = fewer Python wake-ups "
                         "and less CPU, at the cost of latency (1024 at 96 kHz "
                         "is 10.7 ms). 0 lets the driver choose.")
    ap.add_argument("--profile", action="store_true",
                    help="measure where the time actually goes on THIS machine "
                         "and exit: capture, downscale, trace build")
    ap.add_argument("--device", help="audio output: index or name fragment")
    ap.add_argument("--ask", action="store_true")
    args = ap.parse_args()

    region = [int(v) for v in args.region.split(",")] if args.region else None
    if args.source == "ffmpeg":
        grab = ffmpeg_source(width=args.downto, fps=args.capture_fps,
                             region=region, input_spec=args.ffmpeg_input,
                             display=args.display)
    elif args.source == "screen":
        grab = screen_source(region, downto=args.downto)
    elif args.source == "video":
        if not args.file:
            raise SystemExit("--source video needs --file")
        grab = video_source(args.file, downto=args.downto)
    else:
        grab = test_source()
    if args.source != "ffmpeg":
        grab = Throttled(grab, fps=args.capture_fps)

    probe = grab()
    print(f"[SCREEN] source {args.source}, {probe.shape[1]}x{probe.shape[0]} "
          "after downscale")

    if args.profile:
        _profile(args, grab)
        return

    scope = Scope(fps=args.fps, samples=args.samples, blocksize=args.blocksize,
                  device=choose_device(ask=args.ask, device=args.device))
    n = scope.samples_per_frame

    # Tone mapping for unknown live content.  Adapting per frame is what makes
    # cells flicker, so this is a slow exponential average -- seconds, not
    # frames.
    lv = {"lo": None, "hi": None}

    def levels_for(g):
        if args.adapt <= 0:
            return None
        lit = g[g > 0.01]
        if lit.size < 16:
            return (lv["lo"], lv["hi"]) if lv["lo"] is not None else None
        lo_n, hi_n = float(np.percentile(lit, 2)), float(np.percentile(lit, 98))
        if lv["lo"] is None:
            lv["lo"], lv["hi"] = lo_n, hi_n
        else:
            a = min(1.0, (1.0 / max(args.fps, 1)) / args.adapt)
            lv["lo"] += a * (lo_n - lv["lo"])
            lv["hi"] += a * (hi_n - lv["hi"])
        return (lv["lo"], lv["hi"])

    # --- fix the grid once -------------------------------------------------
    # autofit re-derives rows/cols from the fraction of cells surviving trim.
    # On live screen content that fraction moves whenever the picture does, so
    # leaving autofit on per frame meant the grid resized under the image and
    # every cell boundary re-quantized -- cells popping in and out. Same reason
    # mode scope calls calibrate() at startup and holds the result.
    #
    # levels are handled separately above and are deliberately adaptive, but
    # slowly (--adapt, in seconds). Geometry cannot be adaptive at all.
    _probe = np.asarray(grab(), dtype=np.float32)
    _grid_rows, _grid_cols = plan_grid(_probe, n, density=args.density,
                                       trim=args.trim, rows=args.rows,
                                       fields=max(1, args.fields))
    print(f"[SCREEN] grid fixed at {_grid_cols}x{_grid_rows} "
          f"({n * max(1, args.fields) / max(_grid_rows * _grid_cols, 1):.2f} "
          f"samples/cell)"
          + (f", interlace x{args.fields}" if args.fields > 1 else ""))

    if args.stream:
        gen = SweepSource(lum_fn=grab, samples_per_pass=n, gamma=args.gamma,
                          trim=args.trim, density=args.density, rows=args.rows,
                          auto_levels=args.adapt)
        scope.source = BufferedSource(gen, blocksize=max(args.blocksize, 256),
                                      depth=args.buffer_blocks)
        gen(256)
        rws, cls = gen._dims
    else:
        # Whole-trace build: one vectorised pass instead of ~40 tiny per-row
        # ones.  Measured ~1.6 ms per trace against ~12% of a core streaming.
        sweep = {"rev": False, "end": None, "field": 0}
        _fields = max(1, args.fields)

        def push():
            # Gate on the callback having taken the last frame. Without it a
            # frame can be queued over an unconsumed one and dropped while
            # sweep["end"] advances anyway -- so the next trace starts from a
            # position the beam was never at, which with close=False is a
            # full-screen jump nothing budgeted samples for: a bright flyback.
            # It also guarantees interlaced fields arrive one per trace, in
            # order, instead of one silently replacing the other.
            if not scope.ready():
                return
            lum = np.asarray(grab(), dtype=np.float32)
            fr = render_luma(lum, n, gamma=args.gamma, trim=args.trim,
                             density=args.density,
                             grid_rows=_grid_rows, grid_cols=_grid_cols,
                             fields=_fields, field=sweep["field"] % _fields,
                             levels=levels_for(lum), reverse=sweep["rev"],
                             start=sweep["end"], close=False)
            if fr is not None:
                sweep["field"] += 1
                sweep["rev"] = not sweep["rev"]
                sweep["end"] = fr[-1]
                if args.dc_comp:
                    fr = precompensate_hpf(fr, args.dc_comp, scope.samplerate)
                scope.show_frame(fr)

        push()
        rws = cls = 0
        import threading

        def pump():
            period = 1.0 / max(args.fps, 1)
            while not stop.is_set():
                t0 = time.perf_counter()
                try:
                    push()
                except Exception:
                    pass
                time.sleep(max(0.0, period - (time.perf_counter() - t0)))

        stop = threading.Event()
        threading.Thread(target=pump, daemon=True, name="scope-frames").start()

    scope.stream.start()
    print("[SCREEN] running -- Ctrl+C to stop")
    try:
        while True:
            time.sleep(1.0)
            u = getattr(getattr(scope, "source", None), "underruns", 0)
            if u:
                print(f"[SCREEN] {u} underruns -- raise --buffer-blocks or "
                      "lower --fps")
                scope.source.underruns = 0
    except KeyboardInterrupt:
        pass
    finally:
        try:
            stop.set()
        except NameError:
            pass
        if hasattr(grab, "close"):
            grab.close()
        if hasattr(grab, "proc"):
            grab.proc.terminate()
        if getattr(scope, "source", None) is not None:
            scope.source.close()
        scope.stream.stop()
        scope.stream.close()


if __name__ == "__main__":
    main()