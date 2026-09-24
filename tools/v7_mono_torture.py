#!/usr/bin/env python3
"""Visual V7 mono fold-down of the synthetic tape torture matrix.

The wire geometry stays fixed.  Smaller sources are box-downsampled and
upsampled before encoding, so this tests pre-smoothing, not a new wire profile.
Each stereo wire is impaired before its two tracks are mixed into a mono WAV.
Only the resulting WAV samples are passed to the mono decoder.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.io import wavfile
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from animation_modem import v7
from animation_modem.imaging import values_image
from tools.measure_plane_survival import plane_metrics_arrays
from tools.v7_torture_matrix import (CASES, DEFAULT_FRAMES, FIXTURE, RATE,
                                     TARGET, image_quality, impair)

SIZES = ((80, 96), (40, 48), (20, 24))
TILE_SIZE = (320, 384)


def source_variant(reference, size):
    """Shrink the source detail without changing the modem's 80x96 input."""
    if size == reference.size:
        return reference.copy()
    return reference.resize(size, Image.Resampling.BOX).resize(
        reference.size, Image.Resampling.BICUBIC)


def mono_wav(path, stereo):
    """Write and reread an actual single-channel float WAV for each trial."""
    mono = np.ascontiguousarray(stereo.mean(axis=1), dtype=np.float32)
    wavfile.write(path, RATE, mono)
    rate, readback = wavfile.read(path)
    if rate != RATE or readback.ndim != 1:
        raise ValueError(f'{path} is not a {RATE} Hz mono WAV')
    return readback


def decode_trial(model, audio, reference, values, force_float32=False):
    results, info = v7.decode_pulse_stream(
        model, audio, force_float32=force_float32, sample_rate=RATE)
    usable = []
    quality_rows = []
    errors = []
    for result in results:
        if result.status == 'lost' or not result.diag.get('displayable', False):
            continue
        decoded = v7.values_from(model, result.coeffs)
        picture = values_image(decoded, model.coder.grids)
        usable.append((result.counter, picture))
        quality_rows.append(plane_metrics_arrays(reference, picture))
        errors.append(float(np.sqrt(np.mean((decoded - values)**2))))
    row = {
        'decoded_frames': len(results),
        'received': sum(result.status != 'lost' for result in results),
        'lost': sum(result.status == 'lost' for result in results),
        'metadata_valid': sum(bool(result.diag.get('metadata_valid')) for result in results),
        'displayable': len(usable),
        'mean_rmse': float(np.mean(errors)) if errors else None,
        'image_quality': image_quality(quality_rows) if quality_rows else None,
        'recovered': bool(info.get('recovered')),
    }
    return row, usable


def contact_sheet(path, columns):
    """Diagnostic sheet; missing decodes get a labelled grey tile, not a frame."""
    margin, heading = 12, 52
    width, height = TILE_SIZE
    sheet = Image.new('RGB', (margin + len(columns)*(width + margin),
                              height + heading + 2*margin), (215, 215, 215))
    draw = ImageDraw.Draw(sheet)
    for n, (label, image) in enumerate(columns):
        x = margin + n*(width + margin)
        draw.text((x, margin), label, fill=(0, 0, 0))
        if image is None:
            draw.text((x + 20, margin + heading + 20), 'NO DECODE', fill=(40, 40, 40))
        else:
            sheet.paste(image.resize(TILE_SIZE, Image.Resampling.NEAREST),
                        (x, margin + heading))
    sheet.save(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('tmp/v7-mono-torture'))
    parser.add_argument('--fixture', type=Path, default=FIXTURE)
    parser.add_argument('--frames', type=int, default=DEFAULT_FRAMES)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--only', action='append', default=[],
                        help='repeat to select cases from tools/v7_torture_matrix.py')
    parser.add_argument('--force-float32', action='store_true')
    args = parser.parse_args(argv)
    if args.frames < 3:
        parser.error('--frames must be at least 3 (the final packet needs a following header)')
    unknown = set(args.only) - {case.name for case in CASES}
    if unknown:
        parser.error('unknown case(s): ' + ', '.join(sorted(unknown)))

    args.out.mkdir(parents=True, exist_ok=True)
    model = v7.load_model(TARGET, 'nearest')
    with Image.open(args.fixture) as image:
        reference = v7.prepare_image(image, 'nearest')
    reference.save(args.out / 'reference.png')
    variants = {}
    for size in SIZES:
        name = f'{size[0]}x{size[1]}'
        source = source_variant(reference, size)
        source.save(args.out / f'source_{name}.png')
        values = v7.image_values(source, model.coder.grids, 'nearest')
        audio48 = v7.encode_pulse_stream(model, [values]*args.frames, 1,
                                          [0]*args.frames)
        variants[name] = (values, resample_poly(audio48, 2, 1, axis=0).astype(np.float32))

    rows = []
    cases = [case for case in CASES if not args.only or case.name in args.only]
    for case in cases:
        columns = [('Original 80x96', reference)]
        for name, (values, wire) in variants.items():
            damaged = impair(wire, case, seed=args.seed)
            # Stereo control from the same damaged full-resolution signal.
            if name == '80x96':
                control, images = decode_trial(model, damaged, reference, values,
                                               args.force_float32)
                control.update(case=case.name, variant='stereo-80x96')
                rows.append(control)
                picture = images[len(images)//2][1] if images else None
                if picture is not None:
                    picture.save(args.out / f'{case.name}_stereo-80x96.png')
                columns.append(('Stereo 80x96', picture))

            wav_path = args.out / f'{case.name}_mono-{name}.wav'
            mono = mono_wav(wav_path, damaged)
            row, images = decode_trial(model, mono, reference, values,
                                       args.force_float32)
            row.update(case=case.name, variant=f'mono-{name}', wav=wav_path.name)
            rows.append(row)
            picture = images[len(images)//2][1] if images else None
            if picture is not None:
                picture.save(args.out / f'{case.name}_mono-{name}.png')
            columns.append((f'Mono {name}  {row["displayable"]}/{args.frames-1}', picture))
            print(json.dumps(row), flush=True)
        contact_sheet(args.out / f'{case.name}_comparison.png', columns)
    (args.out / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(f'Wrote {len(cases)} comparison sheets, mono WAVs and results to {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
