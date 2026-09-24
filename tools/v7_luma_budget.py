#!/usr/bin/env python3
"""Test V7 luma/chroma allocation changes at a fixed coefficient budget."""
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
from tools.v7_grid_comparison import crop_sbs, sample_values
from tools.v7_torture_matrix import CASES, TARGET, impair

CURRENT_SHAPES = v7.V7_SHAPES
# Same 2,880 transmitted coefficients: +420 (+21.9%) luma; -210 (-43.8%)
# coefficients from each chroma plane. Aspect ratios stay close to V7's grids.
LUMA_HEAVY_SHAPES = ((52, 45), (18, 15), (18, 15))
CHROMA_HEAVY_SHAPES = ((44, 36), (27, 24), (27, 24))
assert sum(r*c for r, c in CURRENT_SHAPES) == sum(
    r*c for r, c in LUMA_HEAVY_SHAPES) == 2880
assert sum(r*c for r, c in CHROMA_HEAVY_SHAPES) == 2880


def custom_shape_model(source, shapes):
    """Derive a matched experimental model without touching frozen V7 tables."""
    old_shapes = v7.V7_SHAPES
    try:
        v7.V7_SHAPES = shapes
        tables = v7.derive_tables(source, 'nearest', v7.phase_table())
        return v7._assemble(tables, v7.phase_table(), TARGET, 'nearest')
    finally:
        v7.V7_SHAPES = old_shapes


def decode(model, audio, reference):
    results, _ = v7.decode_pulse_stream(model, audio)
    frames = []
    scores = []
    for result in results:
        if result.status == 'lost' or not result.diag.get('displayable', False):
            continue
        pixels = model.coder.inverse(result.coeffs * model.coder.gains)
        frame = values_image(pixels, model.coder.grids)
        frames.append(frame)
        scores.append(plane_metrics_arrays(
            reference, frame.resize(reference.size, Image.Resampling.LANCZOS)))
    metrics = None
    if scores:
        score = np.asarray(scores)
        metrics = {plane: {
            'psnr_db': float(score[:, i, 0].mean()),
            'ssim': float(score[:, i, 1].mean()),
            'mae': float(score[:, i, 2].mean()),
        } for i, plane in enumerate(('Y', 'Cb', 'Cr'))}
    return frames, {
        'decoded': len(results),
        'displayable': len(frames),
        'lost': sum(result.status == 'lost' for result in results),
        'metrics': metrics,
    }


def comparison_sheet(path, source, current, candidate, title):
    tile = (480, 640)
    margin, heading = 16, 48
    sheet = Image.new('RGB', (3*tile[0] + 4*margin, tile[1] + heading + 2*margin),
                      '#dedede')
    draw = ImageDraw.Draw(sheet)
    for i, (label, image) in enumerate((('Source (left SBS half)', source),
                                        ('Current V7', current),
                                        (title, candidate))):
        x = margin + i*(tile[0]+margin)
        draw.text((x, margin), label, fill='black')
        if image is not None:
            sheet.paste(image.resize(tile, Image.Resampling.LANCZOS),
                        (x, margin+heading))
        else:
            draw.text((x+20, margin+heading+20), 'NO DECODE', fill='black')
    sheet.save(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path,
        default=ROOT / 'images_sbs/face/00_C_BG_faceSource_960')
    parser.add_argument('--out', type=Path, default=ROOT / 'tmp/v7-luma-budget')
    parser.add_argument('--frames', type=int, default=5)
    parser.add_argument('--max-images', type=int, default=6)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--only', action='append', default=[])
    parser.add_argument('--allocation', choices=('luma-heavy', 'chroma-heavy'),
                        default='luma-heavy')
    args = parser.parse_args(argv)
    if args.frames < 3 or args.max_images < 1:
        parser.error('--frames must be >= 3 and --max-images >= 1')
    by_name = {case.name: case for case in CASES}
    unknown = set(args.only) - by_name.keys()
    if unknown:
        parser.error('unknown cases: ' + ', '.join(sorted(unknown)))
    paths = sorted(p for p in args.source_dir.iterdir()
                   if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))[:args.max_images]
    if not paths:
        parser.error(f'no images in {args.source_dir}')
    cases = [by_name[name] for name in args.only] if args.only else list(CASES)
    args.out.mkdir(parents=True, exist_ok=True)
    candidate_shapes = (LUMA_HEAVY_SHAPES if args.allocation == 'luma-heavy'
                        else CHROMA_HEAVY_SHAPES)
    current_model = v7.load_model(TARGET, 'nearest')
    rows = []
    for path in paths:
        source = crop_sbs(path)
        stem = path.stem
        source.save(args.out / f'{stem}_source.png')
        # Match the production tables' training source; only the retained
        # luma/chroma shapes differ from the frozen canonical profile.
        with Image.open(v7.REFERENCE_FIXTURE) as fixture:
            candidate_model = custom_shape_model(fixture.convert('RGB'), candidate_shapes)
        values = sample_values(source, current_model.coder.grids, 'NEAREST')
        audio48 = v7.encode_pulse_stream(current_model, [values]*args.frames, 1,
                                         [0]*args.frames)
        stereo = resample_poly(audio48, 2, 1, axis=0).astype(np.float32)
        # Encode once per shape set; the candidate's source slots and transforms
        # are different even though the wire count and packet timing are equal.
        candidate_values = sample_values(source, candidate_model.coder.grids, 'NEAREST')
        candidate_audio48 = v7.encode_pulse_stream(
            candidate_model, [candidate_values]*args.frames, 1, [0]*args.frames)
        candidate_stereo = resample_poly(candidate_audio48, 2, 1, axis=0).astype(np.float32)
        for case in cases:
            selected = []
            candidate_name = args.allocation
            for profile, model, audio in (('current', current_model, stereo),
                                          (candidate_name, candidate_model, candidate_stereo)):
                damaged = impair(audio, case, seed=args.seed)
                images, stats = decode(model, damaged, source)
                picture = images[len(images)//2] if images else None
                if picture is not None:
                    picture.save(args.out / f'{stem}_{case.name}_{profile}.png')
                selected.append(picture)
                rows.append({'image': path.name, 'case': case.name,
                             'profile': profile, 'seed': args.seed,
                             'shapes': [list(s) for s in model.coder.shapes], **stats})
            comparison_sheet(args.out / f'{stem}_{case.name}_comparison.png',
                             source, selected[0], selected[1],
                             f'{candidate_name.replace("-", " ").title()} allocation')
            print(json.dumps(rows[-2:]), flush=True)
    (args.out / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(f'Wrote {len(paths)} cropped frames and {len(cases)} case comparisons to {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
