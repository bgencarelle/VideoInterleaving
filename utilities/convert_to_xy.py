"""
XY Baker (utilities/convert_to_xy.py)

Converts matted RGBA image folders into vector path libraries for scope
mode.  Modeled on utilities/bake_assets.py: mirrors the source tree, one
library folder per image folder, parallel over folders.

    python utilities/convert_to_xy.py -i images -o images_xy --profile tiny

Offline tool only -- the library format, reader, and runtime merge live
in scope_bake.py at the repo root.
"""
import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

# --- repo imports, same trick as bake_assets.py ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scope_bake import Q, order_paths, fit_epsilon, subdivide, path_length  # noqa: E402
from make_file_lists import natural_sort_key  # noqa: E402

VALID = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# --- display profiles: sized for a tiny CRT / virtual scope --------------
# min_feature: normalized length below which a ~70 mm tube (or a desktop
# sample budget on a virtual scope) cannot resolve detail.  max_seg:
# subdivision length bounding the runtime occlusion-cull error.
# Vertices are allocated PER CONTOUR with a floor, not by one global epsilon.
# A single epsilon gives large contours plenty and starves small ones into
# 2-point chords -- and small contours (eyes, mouth, nostrils) are exactly what
# makes a face recognizable.  min_v is therefore the most important knob here.
PROFILES = {
    "tiny": dict(budget_main=150, budget_float=45, sil_boost=3.0, bands=3,
                 min_v=7, max_v=26, min_feature=0.025, max_seg=0.05),
    "std":  dict(budget_main=400, budget_float=90, sil_boost=3.0, bands=4,
                 min_v=9, max_v=40, min_feature=0.010, max_seg=0.03),
}

# Thumbnail baked alongside the vectors.  The runtime area-averages it down to
# whatever grid the sample budget supports, so this is a resolution-independent
# store like the vectors -- but its size is a hard ceiling on scanline count.
#
# Raster has a natural ceiling: cells = samples/density, so a normal trace runs
# out of DAC samples before a 128px thumbnail runs out of cells. The thumbnail
# is therefore a compact RAW source, not a second image archive. Raster applies
# its small horizontal compensation only after reducing this source to the real
# sweep grid; stochastic reads the same raw luminance directly.
#
# Stipple keeps source-resolution placement without a 256px image plane. The
# baker stores a small importance-sampled candidate cloud separately: 1024
# normalized positions plus luminance/alpha/edge values per frame. At runtime
# the requested (normally 768) targets are reweighted for the live gamma. This
# is the old baker's governing idea again: store what the beam can consume.
#
#   width   grid cap   per folder (2220 frames)   32 folders
#      48      48x64          14 MB                 0.44 GB
#      64      64x85          24 MB                 0.77 GB
#      96     96x128          55 MB                 1.75 GB
#     128    128x171          97 MB                 3.11 GB   <- default
#     256    256x341         388 MB                12.40 GB
#
# The 1024-point stipple cloud adds about 0.57 GB to a 32 x 2220-frame bake,
# putting the default complete library near 3.7 GB instead of 18-25 GB.
THUMB_W = 128
STIPPLE_CANDIDATES = 1024
STIPPLE_SOURCE_W = 256


def load_rgba(path):
    """RGBA sources give the real matte; SBS JPEGs are split as fallback."""
    with Image.open(path) as im:
        if im.mode in ("RGBA", "LA") or "transparency" in im.info:
            a = np.asarray(im.convert("RGBA"))
            return a[..., :3], a[..., 3]
        rgb = np.asarray(im.convert("RGB"))
    if path.suffix.lower() in (".jpg", ".jpeg"):        # SBS: colour | matte
        w = rgb.shape[1] // 2
        alpha = rgb[:, w:w * 2].mean(axis=2).astype(np.uint8)
        return rgb[:, :w], alpha
    return rgb, np.full(rgb.shape[:2], 255, np.uint8)


# Recommended grid-domain compensation. It is metadata, not a baked channel:
# scope_bake applies it horizontally after reducing to the actual sweep grid.
PRECONDITION = 0.45


def make_thumb(rgb, alpha, width=THUMB_W, precondition=PRECONDITION):
    """(h, w, 2) uint8: [raw luminance, alpha].

    ``precondition`` remains in the call signature for source compatibility,
    but is deliberately not applied here. Sharpening a large stored image and
    then reducing it made facial shadows look hollow while spending a complete
    extra channel. New bakes record the requested amount in format.json and
    apply it to raster's much smaller final grid.
    """
    import cv2
    h, w = alpha.shape
    tw = int(width)
    th = max(1, int(round(h * tw / float(w))))
    lum = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lum = cv2.resize(lum, (tw, th), interpolation=cv2.INTER_AREA)
    a = cv2.resize(alpha, (tw, th), interpolation=cv2.INTER_AREA)
    return np.stack([lum, a], axis=-1).astype(np.uint8)


def make_stipple_candidates(rgb, alpha, count=STIPPLE_CANDIDATES,
                            width=STIPPLE_SOURCE_W):
    """Compact source-detail store for stipple.

    Candidates are systematic samples of a broad proposal distribution, not
    the final points. Keeping raw luminance, alpha and edge values lets runtime
    change gamma/trim/edge gain and combine arbitrary main/float pairs.
    Coordinates are normalized uint16, so their precision is independent of
    the compact raster thumbnail.
    """
    import cv2
    count = max(8, int(count))
    h0, w0 = alpha.shape
    sw = max(16, int(width))
    sh = max(1, int(round(h0 * sw / float(w0))))
    lum = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lum = cv2.resize(lum, (sw, sh), interpolation=cv2.INTER_AREA)
    a = cv2.resize(alpha, (sw, sh), interpolation=cv2.INTER_AREA)
    L = lum.astype(np.float64) / 255.0
    A = a.astype(np.float64) / 255.0
    if min(L.shape) > 1:
        gy, gx = np.gradient(L)
        edge = np.hypot(gx, gy)
        peak = float(edge.max())
        if peak > 1e-12:
            edge /= peak
    else:
        edge = np.zeros_like(L)

    # A luminance floor keeps midtones available when live gamma is lowered;
    # luma and edge bias put more of the finite pool where detail is.
    proposal = A * (0.15 + 0.75 * L + 0.10 * edge)
    total = float(proposal.sum())
    xy = np.zeros((count, 2), dtype=np.uint16)
    lae = np.zeros((count, 3), dtype=np.uint8)
    if total <= 1e-12:
        return xy, lae, np.float32(0.0)

    marks = (np.arange(count, dtype=np.float64) + 0.5) * total / count
    flat = np.searchsorted(np.cumsum(proposal.ravel()), marks, side="left")
    flat = np.clip(flat, 0, proposal.size - 1)
    yy, xx = np.divmod(flat, sw)
    xy[:, 0] = np.round(xx * 65535.0 / max(sw - 1, 1)).astype(np.uint16)
    xy[:, 1] = np.round(yy * 65535.0 / max(sh - 1, 1)).astype(np.uint16)
    lae[:, 0] = lum[yy, xx]
    lae[:, 1] = a[yy, xx]
    lae[:, 2] = np.round(edge[yy, xx] * 255.0).astype(np.uint8)
    return xy, lae, np.float32(total / proposal.size)


def simplify_to(c, target, closed=True):
    """Simplify one contour to (at most) target vertices -- per-contour, so a
    small feature keeps enough points to stay legible."""
    import cv2
    lo, hi = 0.05, 60.0
    for _ in range(26):
        mid = 0.5 * (lo + hi)
        if len(cv2.approxPolyDP(c, mid, closed)) > target:
            lo = mid
        else:
            hi = mid
    return cv2.approxPolyDP(c, hi, closed).reshape(-1, 2).astype(np.float64)


def vectorize(path, budget, min_feature, max_seg, sil_boost=3.0, bands=3,
              min_v=7, max_v=26, min_area=0.001, thumb_width=THUMB_W,
              precondition=PRECONDITION, stipple_candidates=0,
              stipple_width=STIPPLE_SOURCE_W):
    """One frame -> (polylines, flags, thumbnail).

    Silhouette from the alpha matte, interior from posterized luminance BANDS
    (closed regions) rather than Canny edges -- Canny on a photograph yields
    dozens of disconnected fragments that no budget can render as anything but
    chords.  Contours are then ranked and given individual vertex allocations.
    """
    import cv2

    rgb, alpha = load_rgba(path)
    thumb = make_thumb(rgb, alpha, thumb_width, precondition)
    stipple = (make_stipple_candidates(
        rgb, alpha, stipple_candidates, stipple_width)
        if stipple_candidates else None)
    h, w = alpha.shape
    s = max(w, h) / 2.0
    _, mask = cv2.threshold(alpha, 128, 255, cv2.THRESH_BINARY)
    g = cv2.medianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 5)

    # candidates: (contour, score, flag)  flag 1 = silhouette, 0 = interior
    cand, border = [], []
    sil, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for c in sil:
        if cv2.arcLength(c, True) < 2 * min_feature * s:
            continue
        _, _, cw, ch = cv2.boundingRect(c)
        if cw >= 0.96 * w and ch >= 0.96 * h:
            border.append(c)          # full-bleed: occludes but never drawn
        else:
            cand.append((c, cv2.contourArea(c) * sil_boost, 1))

    inside = g[mask > 0]
    if inside.size and bands > 0:
        qs = np.percentile(inside, np.linspace(0, 100, bands + 2)[1:-1])
        k = np.ones((3, 3), np.uint8)
        for q in qs:
            bw = cv2.bitwise_and(cv2.inRange(g, 0, float(q)), mask)
            bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)
            bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)
            cs, _ = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            for c in cs:
                ar = cv2.contourArea(c)
                if ar >= min_area * w * h:
                    cand.append((c, ar, 0))

    # rank by score, then give each contour its own vertex allocation
    cand.sort(key=lambda x: -x[1])
    paths, flags, spent = [], [], 0
    for c, _score, fl in cand:
        v = int(np.clip(np.sqrt(max(cv2.contourArea(c), 1.0)) * 0.9, min_v, max_v))
        if spent + v > budget:
            continue
        p = simplify_to(c, v, True)
        if len(p) >= 3:
            paths.append(p)
            flags.append(fl)
            spent += v
    for c in border:
        p = cv2.approxPolyDP(c, 0.01 * s, True).reshape(-1, 2).astype(np.float64)
        if len(p) >= 3:
            paths.append(p)
            flags.append(2)

    if not paths:
        return ([], [], thumb, *stipple) if stipple is not None else ([], [], thumb)

    ordered, oflags, pos = [], [], np.zeros(2)
    for want in (1, 0):
        group = [p for p, f in zip(paths, flags) if f == want]
        if group:
            tour = order_paths(group, want == 1, start=pos)
            ordered.extend(tour)
            oflags.extend([want] * len(tour))
            pos = tour[-1][-1]
    for p, f in zip(paths, flags):
        if f == 2:
            ordered.append(np.vstack([p, p[:1]]))
            oflags.append(2)

    out_p, out_f = [], []
    for p, f in zip(ordered, oflags):
        p = np.stack([(p[:, 0] - w / 2.0) / s, (p[:, 1] - h / 2.0) / s], axis=1)
        if f != 2 and path_length(p) < min_feature:
            continue
        out_p.append(subdivide(p, max_seg) if f != 2 else p)
        out_f.append(f)
    return ((out_p, out_f, thumb, *stipple)
            if stipple is not None else (out_p, out_f, thumb))


def folder_allowed(path, root):
    """Same gate make_file_lists applies: main folders 0_..254_, float 255_.
    Anything else (999_backup, non-numeric) is ignored by the list builder, so
    baking it is wasted time and disk."""
    from pathlib import Path as _P
    parts = _P(path).relative_to(root).parts if _P(path) != _P(root) else ()
    if not parts:
        return False, "not a layer folder"
    layer = parts[0]
    name = parts[-1]
    prefix = name.partition("_")[0]
    if not prefix.isdigit():
        return False, f"non-numeric prefix ({name})"
    p = int(prefix)
    if layer == "float":
        return (p == 255, f"float folders must start 255_ ({name})")
    return (0 <= p <= 254, f"main folders must start 0_..254_ ({name})")


def process_folder(args):
    src_folder, dest_dir, prof = args
    files = sorted(
        (p for p in Path(src_folder).iterdir()
         if p.is_file() and p.suffix.lower() in VALID),
        key=lambda p: natural_sort_key(p.name),
    )
    if not files:
        return f"Skipped (no images): {src_folder}"

    thumbs_only = prof.get("thumbs_only", False)
    is_float = "float" in Path(src_folder).parts
    budget = prof["budget_float"] if is_float else prof["budget_main"]
    bands = 0 if is_float else prof["bands"]   # floats are mattes: silhouette only
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    thumb_temp = dest / f".thumbs-{os.getpid()}.npy"
    stipple_xy_temp = dest / f".stipple-xy-{os.getpid()}.npy"
    stipple_lae_temp = dest / f".stipple-lae-{os.getpid()}.npy"
    stipple_mass_temp = dest / f".stipple-mass-{os.getpid()}.npy"
    thumbs = None
    stipple_xy = stipple_lae = stipple_mass = None

    try:
        verts, poly_starts, frame_starts, flags, names = [], [0], [0], [], []
        # Write every fixed-size store directly into temporary .npy memmaps.
        # Completed files are atomically moved into place, so an interrupted
        # bake cannot replace a good library with a partial one.
        n_thumbs = 0
        candidate_count = int(prof.get(
            "stipple_candidates", STIPPLE_CANDIDATES))
        stipple_width = int(prof.get("stipple_width", STIPPLE_SOURCE_W))
        for frame_i, fp in enumerate(files):
            if thumbs_only:
                rgb_i, alpha_i = load_rgba(fp)
                polys, fl, thumb = [], [], make_thumb(
                    rgb_i, alpha_i, prof.get("thumb_width", THUMB_W),
                    prof.get("precondition", PRECONDITION))
                sxy, slae, smass = make_stipple_candidates(
                    rgb_i, alpha_i, candidate_count, stipple_width)
            else:
                baked = vectorize(
                    fp, budget, prof["min_feature"], prof["max_seg"],
                    sil_boost=prof["sil_boost"], bands=bands,
                    min_v=prof["min_v"], max_v=prof["max_v"],
                    thumb_width=prof.get("thumb_width", THUMB_W),
                    precondition=prof.get("precondition", PRECONDITION),
                    stipple_candidates=candidate_count,
                    stipple_width=stipple_width)
                if len(baked) == 6:
                    polys, fl, thumb, sxy, slae, smass = baked
                else:  # compatibility for third-party vectorize wrappers
                    polys, fl, thumb = baked
                    sxy = np.zeros((candidate_count, 2), np.uint16)
                    slae = np.zeros((candidate_count, 3), np.uint8)
                    smass = np.float32(0.0)
            if thumb is not None:
                if thumbs is None:
                    thumbs = np.lib.format.open_memmap(
                        thumb_temp, mode="w+", dtype=np.uint8,
                        shape=(len(files),) + thumb.shape)
                thumbs[n_thumbs] = thumb
                n_thumbs += 1
            if stipple_xy is None:
                stipple_xy = np.lib.format.open_memmap(
                    stipple_xy_temp, mode="w+", dtype=np.uint16,
                    shape=(len(files), candidate_count, 2))
                stipple_lae = np.lib.format.open_memmap(
                    stipple_lae_temp, mode="w+", dtype=np.uint8,
                    shape=(len(files), candidate_count, 3))
                stipple_mass = np.lib.format.open_memmap(
                    stipple_mass_temp, mode="w+", dtype=np.float32,
                    shape=(len(files),))
            stipple_xy[frame_i] = sxy
            stipple_lae[frame_i] = slae
            stipple_mass[frame_i] = smass
            for p, f in zip(polys, fl):
                q = np.clip(np.round(p * Q), -Q, Q).astype(np.int16)
                verts.append(q)
                poly_starts.append(poly_starts[-1] + len(q))
                flags.append(int(f))
            frame_starts.append(len(poly_starts) - 1)
            names.append(fp.name)

        V = np.vstack(verts) if verts else np.zeros((0, 2), np.int16)
        np.save(dest / "verts.npy", V)
        np.save(dest / "poly_starts.npy", np.array(poly_starts, np.int32))
        np.save(dest / "frame_starts.npy", np.array(frame_starts, np.int32))
        np.save(dest / "flags.npy", np.array(flags, np.uint8))
        if thumbs is not None:
            if n_thumbs != len(files):
                raise RuntimeError(
                    f"generated {n_thumbs} thumbnails for {len(files)} frames")
            thumbs.flush()
            thumbs = None
            os.replace(thumb_temp, dest / "thumbs.npy")
        if stipple_xy is not None:
            stipple_xy.flush()
            stipple_lae.flush()
            stipple_mass.flush()
            stipple_xy = stipple_lae = stipple_mass = None
            os.replace(stipple_xy_temp, dest / "stipple_xy.npy")
            os.replace(stipple_lae_temp, dest / "stipple_lae.npy")
            os.replace(stipple_mass_temp, dest / "stipple_mass.npy")
        (dest / "format.json").write_text(json.dumps({
            "version": 2,
            "thumbnail_channels": ["raw_luminance", "alpha"],
            "thumb_width": int(prof.get("thumb_width", THUMB_W)),
            "raster_precondition": float(
                prof.get("precondition", PRECONDITION)),
            "stipple_candidates": candidate_count,
            "stipple_source_width": stipple_width,
        }, indent=2))
        (dest / "names.json").write_text(json.dumps(names))
        return None
    except Exception as e:
        try:
            thumbs = None
            stipple_xy = stipple_lae = stipple_mass = None
            for temp in (thumb_temp, stipple_xy_temp, stipple_lae_temp,
                         stipple_mass_temp):
                if temp.exists():
                    temp.unlink()
        except Exception:
            pass
        return f"Error processing {src_folder}: {e}"


def main():
    ap = argparse.ArgumentParser(
        description="Bake matted image folders into XY path libraries.")
    ap.add_argument("-i", "--input-dir", required=True)
    ap.add_argument("-o", "--output-dir", default=None,
                    help="default: {input_dir}_xy")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="tiny")
    ap.add_argument("--budget", type=int, help="override main-layer budget")
    ap.add_argument("--bands", type=int, help="luminance bands for interior detail")
    ap.add_argument("--all-folders", action="store_true",
                    help="bake every folder, ignoring the 0_..254_ / 255_ naming "
                         "gate that make_file_lists uses (off by default, so "
                         "999_* backups are not baked for nothing)")
    ap.add_argument("--precondition", type=float, default=PRECONDITION,
                    metavar="AMOUNT",
                    help=f"recommended runtime raster compensation stored in "
                         f"format.json (default {PRECONDITION}; 0 disables). "
                         "Applied after reduction to the sweep grid, not baked "
                         "into another full-resolution channel.")
    ap.add_argument("--thumb-width", type=int, default=THUMB_W,
                    help=f"raw luminance/alpha width (default {THUMB_W}). "
                         "Raster uses it as a grid ceiling and stochastic as "
                         "its field. Stipple detail is stored separately as "
                         "source-resolution candidates.")
    ap.add_argument("--stipple-candidates", type=int,
                    default=STIPPLE_CANDIDATES, metavar="N",
                    help=f"source-detail candidates stored per frame (default "
                         f"{STIPPLE_CANDIDATES}; runtime normally selects 768)")
    ap.add_argument("--stipple-width", type=int, default=STIPPLE_SOURCE_W,
                    metavar="PX",
                    help=f"analysis width used to place stipple candidates "
                         f"(default {STIPPLE_SOURCE_W}; coordinate storage size "
                         "does not grow with this value)")
    ap.add_argument("--thumbs-only", action="store_true",
                    help="bake only raster/raw luminance + alpha thumbnails "
                         "(skip vectorizing) -- "
                         "much faster for raster/stochastic-only use")
    ap.add_argument("--min-verts", type=int,
                    help="floor on vertices per contour (raise for legibility, "
                         "lower to fit more contours)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    if args.thumb_width < 16:
        ap.error("--thumb-width must be at least 16 pixels")
    if args.stipple_candidates < 8:
        ap.error("--stipple-candidates must be at least 8")
    if args.stipple_width < 16:
        ap.error("--stipple-width must be at least 16 pixels")

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), 20),
                        format="%(asctime)s [%(levelname)s] %(message)s")

    prof = dict(PROFILES[args.profile])
    prof["thumb_width"] = args.thumb_width
    prof["precondition"] = args.precondition
    prof["stipple_candidates"] = args.stipple_candidates
    prof["stipple_width"] = args.stipple_width
    if args.thumbs_only:
        prof["thumbs_only"] = True
    if args.budget:
        prof["budget_main"] = args.budget
    if args.bands is not None:
        prof["bands"] = args.bands
    if args.min_verts:
        prof["min_v"] = args.min_verts

    input_root = Path(args.input_dir).expanduser().resolve()
    if not input_root.is_dir():
        logging.error("Invalid directory: %s", input_root)
        sys.exit(1)
    output_root = (Path(args.output_dir).expanduser().resolve()
                   if args.output_dir
                   else input_root.parent / f"{input_root.name}_xy")

    tasks, skipped = [], []
    for root, _dirs, files in os.walk(input_root):
        if any(f.lower().endswith(tuple(VALID)) for f in files):
            rel = Path(root).relative_to(input_root)
            if not args.all_folders:
                ok, why = folder_allowed(Path(root), input_root)
                if not ok:
                    skipped.append(why)
                    continue
            tasks.append((Path(root), output_root / rel, prof))
    for why in skipped:
        logging.info("Skipping (ignored by make_file_lists): %s", why)
    if not tasks:
        logging.warning("No image folders found.")
        sys.exit(0)

    logging.info("--- XY BAKER (profile: %s) ---", args.profile)
    logging.info("Source: %s  Dest: %s  Folders: %d",
                 input_root, output_root, len(tasks))

    start, errors = time.time(), []
    with ProcessPoolExecutor() as ex:
        for i, res in enumerate(ex.map(process_folder, tasks)):
            if res:
                errors.append(res)
            logging.info("Progress: %.1f%% (%d/%d)",
                         (i + 1) / len(tasks) * 100, i + 1, len(tasks))

    # registration check: matted folders must agree on frame count per layer
    counts = {}
    for _src, dest, _p in tasks:
        f = Path(dest) / "frame_starts.npy"
        if f.exists():
            layer = "float" if "float" in Path(dest).parts else "main"
            counts.setdefault(layer, set()).add(len(np.load(f)) - 1)
    registration_broken = False
    for layer, ns in counts.items():
        if len(ns) > 1:
            registration_broken = True
            logging.warning("Frame counts differ within %s layer: %s "
                            "-- index registration is broken!", layer, sorted(ns))

    logging.info("Done in %.1fs.", time.time() - start)
    for e in errors:
        logging.warning("  - %s", e)
    if errors or registration_broken:
        logging.error("Bake incomplete: %d folder error(s)%s", len(errors),
                      " plus frame-registration failures"
                      if registration_broken else "")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
