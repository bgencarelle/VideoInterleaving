#!/usr/bin/env python3
"""Compare V7 sampling grids using cropped frames from the first SBS face set.

The transmitted DCT shapes and packet geometry remain fixed. This is a
synthetic, profile-matched bench, not a production wire profile.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from animation_modem import v7
from animation_modem.imaging import values_image
from tools.measure_plane_survival import plane_metrics_arrays
from tools.v7_torture_matrix import CASES, RATE, TARGET, impair

SHAPES = v7.V7_SHAPES
GRIDS = {
    'current': v7.V7_GRIDS,
    '2x-grid': tuple((rows * 2, cols * 2) for rows, cols in v7.V7_GRIDS),
}


def crop_sbs(path):
    """Use the left eye/image half of a side-by-side source."""
    with Image.open(path) as im:
        im = im.convert('RGB')
        return im.crop((0, 0, im.width // 2, im.height))


def sample_values(image, grids, resample='LANCZOS'):
    method = getattr(Image.Resampling, resample)
    rows, cols = grids[0]
    sampled = image.convert('RGB').resize((cols, rows), method).convert('YCbCr')
    planes = sampled.split()
    return np.concatenate([
        np.asarray(plane.resize((shape[1], shape[0]), Image.Resampling.BOX),
                   dtype=np.float64).ravel()
        for plane, shape in zip(planes, grids)
    ]) / 127.5 - 1


def experimental_model(grids, source):
    """Derive profile tables locally, restoring canonical globals afterward."""
    if grids == v7.V7_GRIDS:
        return v7.load_model(TARGET, 'nearest')
    old_grids, old_prepare = v7.V7_GRIDS, v7.prepare_image
    try:
        v7.V7_GRIDS = grids
        out_size = (grids[0][1], grids[0][0])
        v7.prepare_image = lambda image, encode_filter='lanczos': image.convert(
            'RGB').resize(out_size, getattr(Image.Resampling, encode_filter.upper()))
        prepared = v7.prepare_image(source, 'nearest')
        tables = v7.derive_tables(prepared, 'nearest', v7.phase_table())
        return v7._assemble(tables, v7.phase_table(), TARGET, 'nearest')
    finally:
        v7.V7_GRIDS, v7.prepare_image = old_grids, old_prepare


def decode(model, audio, reference):
    results, _ = v7.decode_pulse_stream(model, audio)
    frames = []
    for result in results:
        if result.status == 'lost' or not result.diag.get('displayable', False):
            continue
        values = v7.values_from(model, result.coeffs)
        picture = values_image(values, model.coder.grids)
        frames.append(picture)
    scores = [plane_metrics_arrays(reference, frame.resize(reference.size,
              Image.Resampling.LANCZOS)) for frame in frames]
    metrics = None
    if scores:
        scores = np.asarray(scores)
        metrics = {plane: {
            'psnr_db': float(scores[:, i, 0].mean()),
            'ssim': float(scores[:, i, 1].mean()),
            'mae': float(scores[:, i, 2].mean()),
        } for i, plane in enumerate(('Y', 'Cb', 'Cr'))}
    return frames, {'decoded': len(results), 'displayable': len(frames),
                    'lost': sum(r.status == 'lost' for r in results),
                    'metrics_at_source_size': metrics}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path,
        default=ROOT / 'images_sbs/face/00_C_BG_faceSource_960')
    parser.add_argument('--out', type=Path, default=ROOT / 'tmp/v7-grid-comparison')
    parser.add_argument('--frames', type=int, default=5)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--only', action='append', default=[])
    parser.add_argument('--max-images', type=int, default=6)
    args = parser.parse_args(argv)
    if args.frames < 3 or args.max_images < 1:
        parser.error('--frames must be >= 3 and --max-images >= 1')
    cases_by_name = {case.name: case for case in CASES}
    unknown = set(args.only) - cases_by_name.keys()
    if unknown:
        parser.error('unknown cases: ' + ', '.join(sorted(unknown)))
    paths = sorted(p for p in args.source_dir.iterdir()
                   if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))[:args.max_images]
    if not paths:
        parser.error(f'no image files in {args.source_dir}')
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    chosen_cases = [cases_by_name[n] for n in args.only] if args.only else list(CASES)
    for path in paths:
        source = crop_sbs(path)
        stem = path.stem
        source.save(args.out / f'{stem}_left-source.png')
        # Compare each reconstruction against the very same cropped input,
        # viewed at the original SBS-half dimensions.
        reference = source
        models = {}
        audios = {}
        for name, grids in GRIDS.items():
            model = experimental_model(grids, source)
            prep = source.resize((grids[0][1], grids[0][0]), Image.Resampling.NEAREST)
            values = sample_values(prep, grids, 'NEAREST')
            audio48 = v7.encode_pulse_stream(model, [values] * args.frames,
                                             1, [0] * args.frames)
            audios[name] = resample_poly(audio48, 2, 1, axis=0).astype(np.float32)
            models[name] = model
        for case in chosen_cases:
            sheet_cols = [('Source (left SBS half)', source)]
            for name in GRIDS:
                damaged = impair(audios[name], case, seed=args.seed)
                frames, stats = decode(models[name], damaged, reference)
                row = {'image': path.name, 'case': case.name,
                       'profile': name, 'seed': args.seed, **stats}
                rows.append(row)
                picture = frames[len(frames)//2] if frames else None
                if picture is not None:
                    picture.save(args.out / f'{stem}_{case.name}_{name}.png')
                    display = picture.resize((480, 640), Image.Resampling.LANCZOS)
                else:
                    display = None
                sheet_cols.append((f'{name} ({stats["displayable"]} decoded)', display))
            tile_size = (480, 640)
            margin, heading = 16, 52
            sheet = Image.new('RGB', (margin + len(sheet_cols)*(tile_size[0]+margin),
                                      tile_size[1] + heading + margin*2), '#d8d8d8')
            draw = ImageDraw.Draw(sheet)
            for i, (label, image) in enumerate(sheet_cols):
                x = margin + i*(tile_size[0]+margin)
                draw.text((x, margin), label, fill='black')
                if image is None:
                    draw.text((x+20, margin+heading+20), 'NO DECODE', fill='black')
                else:
                    sheet.paste(image, (x, margin+heading))
            sheet.save(args.out / f'{stem}_{case.name}_comparison.png')
            print(json.dumps(rows[-2:]), flush=True)
    (args.out / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(f'Wrote {len(paths)} cropped sources and {len(chosen_cases)} cases to {args.out}')


if __name__ == '__main__':
    raise SystemExit(main())
