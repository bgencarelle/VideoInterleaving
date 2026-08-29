"""
test_scope_pair.py -- output one main/float pair over BOTH display and scope.

Picks a background (main) folder, a foreground (float) folder and a frame
index -- as CLI args or interactively -- then streams the composite to the
audio device while showing a simulated phosphor trace on screen.  The preview
is rendered from the same sample stream the DAC receives and obeys the same
trace timing, so what you see is what the scope draws.

    python test_scope_pair.py --xy-dir images_xy
    python test_scope_pair.py --xy-dir images_xy --bg 3 --fg 3 --raster \
           --fps 30 --ips 30 --exposure 0.4 --play

Keys:  space play/pause · left/right step · m occlusion cull · v cycle mode
       r vector<->raster
       + / - intensity · [ / ] gamma · d density · , / . speed · l pingpong<->loop
       q quit

Fallbacks: no audio device -> display only; no GUI -> writes a PNG.
"""
import argparse
import itertools
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scope_bake import (XYLibrary, merge, raster_frame, content_bbox,
                        composite_luma, composite_stipple_candidates,
                        StochasticEmitter, StippleEmitter,
                        TriangleMixScheduler,
                        PositionMultiplexer, apply_trace_border)    # noqa: E402
from scope_out import Scope, rasterize, FPS, choose_device           # noqa: E402

PNG_OUT = "test_scope_pair_output.png"


# ---------------------------------------------------------------- discovery

def discover(xy_root):
    root = Path(xy_root)
    if not root.is_dir():
        raise SystemExit(f"XY directory not found: {root}\n"
                         "Bake first: python utilities/convert_to_xy.py "
                         f"-i <images> -o {root}")
    # Use the live scope runtime's manifest verbatim.  In particular, this
    # applies the normal display rule of selecting the largest frame count
    # shared by both layers and preserves numeric-prefix folder order.  The
    # pair tester must not offer stale folders the real mode would exclude.
    from scope_display import _manifest_from_xy
    manifest = _manifest_from_xy(root)
    if manifest is None:
        raise SystemExit(
            f"Need matching main and float libraries under {root}; no shared "
            "positive frame count was found")
    _frames, mains, floats = manifest
    return mains, floats, root


def choose(name, options, root, given):
    if given is not None:
        return options[given % len(options)]
    print(f"\n{name} folders:")
    for i, d in enumerate(options):
        print(f"  [{i}] {d.relative_to(root)}")
    raw = input(f"{name} index [0]: ").strip()
    return options[(int(raw) if raw else 0) % len(options)]


# ---------------------------------------------------------------- timing

class TraceSim:
    """Mirrors Scope's timing so the preview cannot promise what the hardware
    can't deliver: a trace takes 1/fps seconds, a queued frame appears only at
    the next boundary, and a superseded frame is dropped rather than delayed."""

    def __init__(self, fps, now):
        self.period = 1.0 / max(fps, 0.01)
        self.next_boundary = now + self.period
        self.pending = None
        self.drawn = 0
        self.dropped = 0

    def queue(self, payload, index):
        if self.pending is not None:
            self.dropped += 1
        self.pending = (payload, index)

    def poll(self, now):
        if now < self.next_boundary:
            return None
        self.next_boundary = now + self.period
        self.drawn += 1
        if self.pending is None:
            return None
        out, self.pending = self.pending, None
        return out


def advance(index, direction, frames, pingpong=True):
    """One playback step.  Ping-pong reflects at both ends without repeating
    the end frame; loop wraps around."""
    if frames <= 1:
        return 0, direction
    nxt = index + direction
    if not pingpong:
        return nxt % frames, direction
    if nxt >= frames:
        return frames - 2, -1
    if nxt < 0:
        return 1, 1
    return nxt, direction


# ---------------------------------------------------------------- phosphor sim

def render_trace(samples, size=700, spot=None, exposure=1.0, budget=None):
    """Delegates to scope_bake.preview_frame -- the canonical phosphor model.

    Was a second copy of that algorithm. The web preview needs the same one,
    and two implementations of one algorithm is the mistake SweepSource and
    raster_frame already made.
    """
    from scope_bake import preview_frame
    # `budget` is accepted and ignored: subdivision is now per segment in
    # pixels rather than a global point budget. Kept in the signature so any
    # caller passing it does not break.
    return preview_frame(samples, size=size, spot=spot, exposure=exposure)


def annotate(img, text):
    import cv2
    cv2.putText(img, text, (12, img.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 200, 120), 1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------- main

def build_parser():
    ap = argparse.ArgumentParser(
        description="A/B one baked pair on display + scope",
        conflict_handler="resolve")
    ap.add_argument("--xy-dir", default="images_xy")
    ap.add_argument("--bg", type=int, help="background (main) folder number")
    ap.add_argument("--fg", type=int, help="foreground (float) folder number")
    ap.add_argument("--index", type=int, help="frame index")
    ap.add_argument("--size", type=int, default=700, help="preview window size")

    ap.add_argument("--raster", action="store_true", help="start in raster mode")
    ap.add_argument("--stipple", action="store_true",
                    help="start in stable luminance-weighted stipple mode")
    ap.add_argument("--fusion", choices=("vrs", "vr", "sv", "sr"),
                    help="start in fused-density mode with these components")
    ap.add_argument("--samples", type=int,
                    help="path length per trace -- the real detail budget. "
                         "Refresh follows as rate/samples. Overrides --fps.")
    ap.add_argument("--fps", type=int, default=FPS,
                    help="scope refresh; LOWER = more samples per trace "
                         "(30 -> 1470 at 44.1k, 50 -> 882)")
    ap.add_argument("--ips", type=float, default=30.0,
                    help="playback images per second (repo default IPS is 30)")

    ap.add_argument("--rows", type=int, help="raster scanline count (default auto)")
    ap.add_argument("--gamma", type=float, default=2.2, help="raster contrast")
    ap.add_argument("--stochastic-gamma", type=float, default=2.0,
                    help="stochastic luminance exponent (default 2.0)")
    ap.add_argument("--stipple-points", type=int, default=768,
                    help="stable weighted positions in stipple mode")
    ap.add_argument("--density", type=float, default=1.0,
                    help="raster samples per cell; 1.0 = finest, above 1 trades "
                         "detail for brighter solider lines")
    ap.add_argument("--fields", type=int, default=1,
                    help="raster interlacing: 2 draws alternate rows per trace. "
                         "Only meaningful when --fps is a multiple of --ips")
    ap.add_argument("--fit", action="store_true",
                    help="scale the subject to fill the screen using one stable "
                         "bbox for the whole sequence")
    ap.add_argument("--trim", type=float, default=0.02,
                    help="drop cells dimmer than this from the sweep. Higher "
                         "= fewer stray lines crossing dark areas, at the cost "
                         "of clipping faint detail. Try 0.08-0.16.")
    ap.add_argument("--no-trim", action="store_true",
                    help="sweep full width even across black margins")
    ap.add_argument("--border", type=float, default=0.0, metavar="F",
                    help="fraction of every image trace spent on a fixed "
                         "full-extent border (same as main --scope-border)")
    ap.add_argument("--mix", nargs="?", type=float, const=120.0, default=None,
                    metavar="HZ",
                    help="triangular VECTOR -> RASTER -> STOCHASTIC -> RASTER "
                         "whole-trace mix at this rate (default 120). The trace "
                         "rate is HZ, so each pass gets rate/HZ samples.")
    ap.add_argument("--mix-duty", type=float, default=0.5, metavar="F",
                    help="fraction of mixed passes spent on RASTER (default "
                         "0.5); the remainder splits equally between VECTOR "
                         "and STOCHASTIC")
    ap.add_argument("--sweep", choices=("alternate", "palindrome", "retrace"),
                    default="alternate",
                    help="alternate: each trace sweeps one way and the next "
                         "resumes from there with the NEXT index -- no flyback, "
                         "full density (needs a fresh frame per trace, so use "
                         "--fps == --ips). palindrome: down and back over the "
                         "same image, safe when traces repeat, half density. "
                         "retrace: visible CRT-style flyback.")
    ap.add_argument("--min-feature", type=float, default=0.02,
                    help="vector: shortest stroke kept by the occlusion cull")

    ap.add_argument("--exposure", type=float, default=1.0,
                    help="preview brightness; stands in for the scope's "
                         "intensity knob (try 0.4)")
    ap.add_argument("--play", action="store_true", help="start playing")
    ap.add_argument("--loop", action="store_true",
                    help="wrap at the end instead of ping-ponging")
    ap.add_argument("--ask", action="store_true",
                    help="choose the audio output interactively")
    ap.add_argument("--device", help="audio output: index or name fragment")
    return ap


def main():
    args = build_parser().parse_args()

    if args.mix is not None:
        if not math.isfinite(args.mix) or args.mix <= 0:
            raise SystemExit("--mix must be a finite rate greater than zero")
        if args.samples is not None:
            raise SystemExit("--samples cannot be combined with --mix; the "
                             "mix rate is the trace clock")
    if (not math.isfinite(args.mix_duty)
            or not 0.0 <= args.mix_duty <= 1.0):
        raise SystemExit("--mix-duty must be between 0 and 1")

    mains, floats, root = discover(args.xy_dir)
    bg_dir = choose("Background (main)", mains, root, args.bg)
    fg_dir = choose("Foreground (float)", floats, root, args.fg)
    mlib, flib = XYLibrary(bg_dir), XYLibrary(fg_dir)
    frames = min(len(mlib), len(flib))
    if len(mlib) != len(flib):
        print(f"⚠️  frame counts differ ({len(mlib)} vs {len(flib)}) -- "
              "registration is broken; clamping to the shorter.")
    if args.index is not None:
        index = args.index % frames
    else:
        raw = input(f"Frame index 0..{frames - 1} [0]: ").strip()
        index = (int(raw) if raw else 0) % frames

    if args.mix:
        args.fps = args.mix                 # switch rate == trace rate
    scope = None
    try:
        # invert_y=False: everything out of scope_bake is ALREADY in scope
        # space (y up).  XYLibrary.frame() applies flip_y, and render_luma
        # builds its rows with ys = -linspace(...).  Scope.show()'s invert_y
        # is for callers handing it raw screen-space polylines; applying it
        # here flips a second time and stands the vector picture on its head.
        scope = Scope(fps=args.fps, samples=args.samples, invert_y=False,
                      device=choose_device(ask=args.ask, device=args.device))
        scope.stream.start()
        print(f"[AUDIO] {scope.samplerate} Hz, {scope.samples_per_frame} samples/trace")
    except Exception as e:
        print(f"[AUDIO] unavailable, display only ({e})")
    n_samples = scope.samples_per_frame if scope else int(44100 / max(args.fps, 1))

    if args.fps < args.ips:
        print(f"[TIMING] {args.fps} fps < {args.ips:.0f} ips: only {args.fps} of "
              f"every {args.ips:.0f} indices can be shown. Beam time per index is "
              "sample_rate/ips regardless -- lowering fps buys fewer indices, "
              "not more samples each.")
    if args.fields > 1 and args.fps % max(int(args.ips), 1) != 0:
        print(f"[FIELDS] --fps {args.fps} is not a multiple of --ips "
              f"{args.ips:.0f}; interlacing will show partial images.")

    fit_bbox = content_bbox([mlib, flib]) if args.fit else None
    if fit_bbox:
        area = (fit_bbox[2] - fit_bbox[0]) * (fit_bbox[3] - fit_bbox[1])
        print(f"[FIT] subject occupies {area:.0%} of the frame")

    import cv2
    win = "scope pair test"
    gui = (sys.platform.startswith(("win", "darwin"))
           or bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")))
    culled, raster = True, args.raster
    exposure, gamma, density = args.exposure, args.gamma, args.density
    playing, pingpong, ips = args.play, not args.loop, max(0.5, args.ips)
    direction = 1
    field_counter = itertools.count()
    sweep = {"rev": False, "end": None}
    duty = min(1.0, max(0.0, args.mix_duty))
    mix_scheduler = TriangleMixScheduler(duty)
    stochastic = StochasticEmitter(
        scope.samplerate if scope else 44100, n_samples,
        gamma=args.stochastic_gamma, border=args.border)
    stipple = StippleEmitter(
        scope.samplerate if scope else 44100, n_samples,
        points=args.stipple_points, gamma=args.stochastic_gamma,
        trim=0.0 if args.no_trim else args.trim, border=args.border)
    fusion_multiplexer = PositionMultiplexer()
    beam_end = None
    last_mode = None
    if args.sweep == "alternate" and args.fps > args.ips:
        print(f"[SWEEP] --fps {args.fps} > --ips {args.ips:.0f}: traces will "
              "repeat and a repeated one-way sweep shows a flyback. "
              "Use --sweep palindrome or match fps to ips.")

    def build(idx, do_cull, mode):
        nonlocal beam_end, last_mode
        if mode == "raster":
            samp = raster_frame(mlib, idx, flib, idx, n_samples,
                                gamma=gamma, rows=args.rows, density=density,
                                trim=0.0 if args.no_trim else args.trim,
                                border=args.border,
                                bbox=fit_bbox, fields=args.fields,
                                field=next(field_counter),
                                palindrome=(args.sweep == "palindrome"),
                                reverse=(args.sweep == "alternate" and sweep["rev"]),
                                start=beam_end if args.sweep == "alternate" else None,
                                close=(args.sweep == "retrace"))
            if args.sweep == "alternate" and samp is not None:
                sweep["rev"] = not sweep["rev"]
                sweep["end"] = samp[-1]
            if samp is None:
                print("[RASTER] this bake has no thumbs.npy -- rebake:\n"
                      "  python utilities/convert_to_xy.py -i <images> "
                      "-o <xy_dir> --thumbs-only")
                return build(idx, do_cull, "vector")
            beam_end = samp[-1].copy()
            last_mode = mode
            if scope:
                scope.show_frame(samp)
            return samp, 0
        if mode == "stochastic":
            if beam_end is not None and last_mode != "stochastic":
                stochastic.start_at(beam_end)
            samp = stochastic.emit(composite_luma(
                mlib, idx, flib, idx, raw=True))
            if samp is None:
                return build(idx, do_cull, "vector")
            beam_end = samp[-1].copy()
            last_mode = mode
            if scope:
                scope.show_frame(samp)
            return samp, 0
        if mode == "stipple":
            if beam_end is not None:
                stipple.start_at(beam_end)
            cloud = composite_stipple_candidates(mlib, idx, flib, idx)
            samp = (stipple.emit_candidates(cloud) if cloud is not None
                    else stipple.emit(composite_luma(
                        mlib, idx, flib, idx, raw=True)))
            if samp is None:
                return build(idx, do_cull, "vector")
            beam_end = samp[-1].copy()
            last_mode = mode
            if scope:
                scope.show_frame(samp)
            return samp, 0
        if mode == "fusion":
            components = args.fusion or "vrs"
            vector_trace = (rasterize(
                merge(mlib, idx, flib, idx,
                      min_feature=args.min_feature), n_samples)
                if "v" in components else None)
            raster_trace = None
            if "r" in components:
                raster_trace = raster_frame(
                    mlib, idx, flib, idx, n_samples,
                    gamma=gamma, rows=args.rows, density=density,
                    trim=0.0 if args.no_trim else args.trim,
                    border=0.0,
                    bbox=fit_bbox, fields=args.fields,
                    field=next(field_counter),
                    palindrome=(args.sweep == "palindrome"),
                    reverse=(args.sweep == "alternate" and sweep["rev"]),
                    start=(beam_end if args.sweep == "alternate" else None),
                    close=(args.sweep == "retrace"))
                if args.sweep == "alternate" and raster_trace is not None:
                    sweep["rev"] = not sweep["rev"]
            stochastic_trace = None
            if "s" in components:
                if beam_end is not None:
                    stochastic.start_at(beam_end)
                old_border = stochastic.border
                stochastic.border = 0.0
                try:
                    stochastic_trace = stochastic.emit(composite_luma(
                        mlib, idx, flib, idx, raw=True))
                finally:
                    stochastic.border = old_border
            samp = fusion_multiplexer.emit(
                vector=vector_trace, raster=raster_trace,
                stochastic=stochastic_trace, components=components)
            if samp is None:
                return build(idx, do_cull, "vector")
            reference = composite_luma(mlib, idx, flib, idx, raw=False)
            if reference is not None:
                h, w = reference.shape
                samp = apply_trace_border(
                    samp, args.border, aspect=h / float(max(w, 1)),
                    level=stochastic.level)
            if beam_end is not None:
                samp[0] = beam_end
            beam_end = samp[-1].copy()
            stochastic.chain_from(beam_end)
            last_mode = mode
            if scope:
                scope.show_frame(samp)
            return samp, 0
        if do_cull:
            polys = merge(mlib, idx, flib, idx, min_feature=args.min_feature)
        else:
            mp, mf = mlib.frame(idx % len(mlib))
            fp, ff = flib.frame(idx % len(flib))
            polys = ([p for p, f in zip(mp, mf) if f != 2]
                     + [p for p, f in zip(fp, ff) if f != 2])
        samp = rasterize(polys, n_samples)
        beam_end = samp[-1].copy()
        last_mode = mode
        if scope:
            scope.show_frame(samp)
        return samp, len(polys)

    active_mode = (mix_scheduler.next_mode() if args.mix
                   else "fusion" if args.fusion
                   else "stipple" if args.stipple
                   else "raster" if raster else "vector")
    samples, npolys = build(index, culled, active_mode)
    shown_index = index
    dirty = True
    img = None
    headless_written = False
    headless_audio_announced = False
    last_step = time.perf_counter()
    last_push = last_step
    push_interval = (1.0 / args.mix) if args.mix else (1.0 / ips)
    sim = TraceSim(args.fps, last_step)
    if args.mix:
        outer = (1.0 - duty) * args.mix * 0.5
        print(f"[MIX] triangular V -> R -> S -> R at {args.mix:g} Hz "
              f"-> {outer:g} vector, {duty * args.mix:g} raster, "
              f"{outer:g} stochastic passes/sec")

    while True:
        now = time.perf_counter()
        if playing:
            # Audio playback must not depend on an OpenCV window.  A headless
            # OpenCV build used to strand the very first (vector) trace here
            # forever.  For a mix, also wait until the callback has accepted
            # the pending trace so V/R/S/R cannot collapse through last-write-
            # wins replacement into whichever component happened to survive.
            can_push = (not args.mix or scope is None or scope.ready())
            if now - last_push >= push_interval and can_push:
                last_push = now
                if now - last_step >= 1.0 / ips:
                    last_step = now
                    index, direction = advance(index, direction, frames, pingpong)
                if args.mix:
                    active_mode = mix_scheduler.next_mode()
                sim.queue(build(index, culled, active_mode), index)
            got = sim.poll(now)
            if got is not None:
                (samples, npolys), shown_index = got
                dirty = True
        else:
            shown_index = index

        # In headless audio mode the DAC still advances continuously, but
        # rendering and rewriting a PNG at the trace rate would only steal
        # time from it.  Save one representative preview and keep scheduling.
        if dirty and (gui or img is None):
            mode = ("RASTER" if active_mode == "raster"
                    else "STOCHASTIC" if active_mode == "stochastic"
                    else "STIPPLE" if active_mode == "stipple"
                    else f"FUSION {args.fusion.upper()}" if active_mode == "fusion"
                    else f"VECTOR paths {npolys} merge {'on' if culled else 'OFF'}")
            if args.mix:
                mode = (f"MIX {args.mix:g}Hz duty {duty:.2f} "
                        f"[{active_mode[0].upper()}]")
            state = (f"{'PLAY' if playing else 'PAUSE'} "
                     f"{'<>' if pingpong else '>>'}{'+' if direction > 0 else '-'} "
                     f"{ips:.0f}ips/{args.fps}fps")
            if playing and sim.dropped:
                state += f"  SKIPPED {sim.dropped / max(sim.dropped + sim.drawn, 1):.0%}"
            img = annotate(render_trace(samples, args.size, exposure=exposure),
                           f"{bg_dir.relative_to(root)} + {fg_dir.relative_to(root)}"
                           f"  frame {shown_index}/{frames - 1}  {mode}"
                           f"  {n_samples} samples  exp {exposure:.2f}"
                           + (f"  gamma {gamma:.1f} density {density:.0f}"
                              if active_mode == "raster" else "")
                           + f"  {state}")
            dirty = False

        if not gui:
            if not headless_written:
                cv2.imwrite(PNG_OUT, img)
                print(f"[DISPLAY] no GUI -- wrote {PNG_OUT}")
                headless_written = True
            if scope:
                if playing and args.mix:
                    message = "[AUDIO] mixed playback continues, Ctrl+C to stop"
                else:
                    message = "[AUDIO] looping to scope, Ctrl+C to stop"
                if not headless_audio_announced:
                    print(message)
                    headless_audio_announced = True
                try:
                    time.sleep(0.001 if playing else 0.05)
                except KeyboardInterrupt:
                    break
                continue
            break

        try:
            cv2.imshow(win, img)
            key = cv2.waitKey(1 if playing else 30) & 0xFF
        except cv2.error:
            gui = False
            continue

        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            playing = not playing
            last_step = time.perf_counter()
            dirty = True
        elif key in (81, ord("a")):
            index, direction = (index - 1) % frames, -1
            samples, npolys = build(index, culled, active_mode); dirty = True
        elif key in (83, ord("s")):
            index, direction = (index + 1) % frames, 1
            samples, npolys = build(index, culled, active_mode); dirty = True
        elif key == ord("."):
            ips = min(120.0, ips * 1.25); dirty = True
        elif key == ord(","):
            ips = max(0.5, ips / 1.25); dirty = True
        elif key == ord("l"):
            pingpong = not pingpong; dirty = True
        elif key == ord("m"):
            culled = not culled
            samples, npolys = build(index, culled, active_mode); dirty = True
        elif key == ord("r"):
            raster = not raster
            active_mode = "raster" if raster else "vector"
            samples, npolys = build(index, culled, active_mode); dirty = True
        elif key == ord("v"):
            order = ("vector", "raster", "stochastic", "stipple", "fusion")
            active_mode = order[(order.index(active_mode) + 1) % len(order)]
            if active_mode == "fusion" and not args.fusion:
                args.fusion = "vrs"
            samples, npolys = build(index, culled, active_mode); dirty = True
        elif key == ord("f"):
            order = ("vrs", "vr", "sv", "sr")
            current = args.fusion if args.fusion in order else "vrs"
            args.fusion = order[(order.index(current) + 1) % len(order)]
            active_mode = "fusion"
            samples, npolys = build(index, culled, active_mode); dirty = True
        elif key in (ord("+"), ord("=")):
            exposure = min(4.0, exposure * 1.3); dirty = True
        elif key in (ord("-"), ord("_")):
            exposure = max(0.05, exposure / 1.3); dirty = True
        elif key == ord("]"):
            gamma = min(5.0, gamma + 0.2)
            samples, npolys = build(index, culled, active_mode); dirty = True
        elif key == ord("["):
            gamma = max(0.6, gamma - 0.2)
            samples, npolys = build(index, culled, active_mode); dirty = True
        elif key == ord("d"):
            density = 1.0 if density >= 4.0 else density + 1.0
            samples, npolys = build(index, culled, active_mode); dirty = True

    if scope:
        scope.stream.stop()
        scope.stream.close()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


if __name__ == "__main__":
    main()
