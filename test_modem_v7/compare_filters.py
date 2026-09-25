#!/usr/bin/env python3
"""Encode filter through the real V7 modem: nearest (today's app default)
against box and lanczos. No wire change: the receiver already selects the
model from the packet's encoding_type.

For every scored frame (2, 4, 6, ...): send it as a still, impair, decode, score the last
decoded steady packet at 405x540 against the source.

    python test_modem_v7/compare_filters.py FRAME... [--quick]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from common import (CASES, RATE, REPO, STEADY_FROM, TARGET, Grids, contact_sheet, impair,
                    load_frames, reference, score, v7, v7_values)

FILTERS = ('nearest', 'box', 'lanczos')
DEFAULT_CASES = ('clean-96k', 'hiss-40', 'type-ii', 'type-i', 'wow-flutter',
                 'fast-flutter', 'lowpass-10k', 'dropouts')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('frames', nargs='+', type=Path)
    ap.add_argument('--cases', nargs='+', default=list(DEFAULT_CASES),
                    choices=[c.name for c in CASES], metavar='CASE')
    ap.add_argument('--packets', type=int, default=12)
    ap.add_argument('--quick', action='store_true', help='smoke run: one frame, clean only')
    ap.add_argument('--out', type=Path, default=REPO/'tmp'/'test_modem_v7'/'compare_filters')
    args = ap.parse_args(argv)
    _, test = load_frames(args.frames)
    if args.quick:
        test, args.cases = test[:1], ['clean-96k']
    args.out.mkdir(parents=True, exist_ok=True)
    cases = {c.name: c for c in CASES}
    grid = Grids(v7.V7_GRIDS)
    kept = grid.corner_positions(v7.V7_SHAPES)
    rows, sheet = [], [('source', reference(test[0]))]
    for encode_filter in FILTERS:
        model = v7.load_model(TARGET, encode_filter)
        for case in args.cases:
            scores = []
            for index, frame in enumerate(test):
                values = v7_values(frame, encode_filter)
                wire = resample_poly(v7.encode_pulse_stream(
                    model, [values]*args.packets, 1, [0]*args.packets), 2, 1,
                    axis=0).astype(np.float32)
                results, _ = v7.decode_pulse_stream(
                    model, impair(wire, cases[case], seed=2026), sample_rate=RATE)
                steady = [r for r in results if r.counter >= STEADY_FROM and r.status != 'lost']
                if not steady:
                    continue
                full = np.zeros(grid.off[-1]); full[kept] = steady[-1].coeffs
                picture = grid.picture(full)
                scores.append(score(reference(frame), picture))
                if index == 0 and case == args.cases[0]:
                    sheet.append((f'{encode_filter} ({case})', picture))
            row = {'filter': encode_filter, 'case': case, 'frames_scored': len(scores),
                   **{k: round(float(np.mean([s[k] for s in scores])), 2)
                      for k in ('ssimulacra2', 'psnr_y', 'de2000')}}
            rows.append(row)
            print(json.dumps(row), flush=True)
    contact_sheet(args.out/'filters.png', sheet, columns=len(sheet))
    (args.out/'results.json').write_text(json.dumps(rows, indent=1) + '\n')


if __name__ == '__main__':
    main()
