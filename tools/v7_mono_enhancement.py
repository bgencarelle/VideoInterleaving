#!/usr/bin/env python3
"""Synthetic impairment check for the experimental V7 mono fold-off wire."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'test_modem_v7'))

from animation_modem import v7                                           # noqa: E402
from mono_wire import MonoWire                                            # noqa: E402
from tools.v7_torture_matrix import (CASES, RATE, TARGET, impair)         # noqa: E402


def run(args):
    if args.frames < v7.TAIL_PHASES+1:
        raise ValueError(
            f'--frames must be at least {v7.TAIL_PHASES+1} to cover every tail phase')
    args.out.mkdir(parents=True, exist_ok=True)
    model = v7.load_model(TARGET, args.encode_filter)
    with Image.open(args.fixture) as image:
        prepared = v7.prepare_image(image, args.encode_filter)
    values = v7.image_values(prepared, model.coder.grids, args.encode_filter)
    profile = MonoWire(model)
    mono_model = profile.model_for(model)
    wire48 = profile.encode(model, [values]*args.frames, start_counter=1)
    wire96 = resample_poly(wire48, RATE//v7.RATE, 1, axis=0).astype(np.float32)

    rows = []
    for case in CASES:
        damaged = impair(wire96, case, seed=args.seed)
        with profile.receiving():
            results, info = v7.decode_pulse_stream(
                mono_model, damaged, sample_rate=RATE,
                pilot_timing='tone-seeded', frame_boundary='eof')
        row = {
            'case': case.name,
            'results': len(results),
            'metadata_valid': sum(
                bool(result.diag.get('metadata_valid')) for result in results),
            'received': sum(result.status != 'lost' for result in results),
            'displayable': sum(
                bool(result.diag.get('displayable')) for result in results),
            'eof_validated': int(info.get('eof_markers_validated') or 0),
        }
        row['pass'] = bool(
            row['results'] == args.frames and
            row['metadata_valid'] == args.frames and
            row['displayable'] == args.frames and
            row['eof_validated'] == args.frames)
        rows.append(row)
        print(json.dumps(row), flush=True)

    report = {
        'profile': 'mono-fold-off',
        'encode_filter': args.encode_filter,
        'frames_per_case': args.frames,
        'capture_rate': RATE,
        'seed': args.seed,
        'accepted_cases': sum(row['pass'] for row in rows),
        'case_count': len(rows),
        'cases': rows,
        'note': 'Synthetic impairments only; not a real tape/deck result.',
    }
    (args.out/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f"Wrote {args.out/'results.json'}; "
          f"{report['accepted_cases']}/{report['case_count']} cases passed")
    return 0 if report['accepted_cases'] == report['case_count'] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('tmp/v7-mono-enhancement'))
    parser.add_argument('--fixture', type=Path, default=v7.REFERENCE_FIXTURE)
    parser.add_argument('--frames', type=int, default=12)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--encode-filter', choices=v7.ENCODING_FILTERS,
                        default='box')
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    raise SystemExit(main())
