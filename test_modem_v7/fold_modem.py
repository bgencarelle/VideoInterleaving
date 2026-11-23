#!/usr/bin/env python3
"""Luma folding through the real V7 modem under warble, jitter, low-pass and
dropouts (1x, box encode filter, today's slot layout).

For every scored frame (2, 4, 6, ...): send it as a still (PACKETS packets), impair
the audio, decode it with the normal V7 receiver, and score every decoded
steady packet at 405x540 against the source.

  base        the normal, unfolded wire
  fold M      M luma slots folded 2:1, every symbol unfolded
  fold M+fb   the same, with the low-confidence fallback

    python test_modem_v7/fold_modem.py FRAME... [--folds 500 1000] [--quick]
"""
import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from common import (CASES, RATE, REPO, STEADY_FROM, TARGET, contact_sheet, impair,
                    jitter_warp, load_frames, reference, ssimulacra2, v7, v7_values)
from folding import FoldCodec, equaliser_capture

_C = {case.name: case for case in CASES}
IMPAIRMENTS = {
    'clean': lambda w: impair(w, _C['clean-96k'], seed=2026),
    'wow-flutter': lambda w: impair(w, _C['wow-flutter'], seed=2026),
    'fast-flutter': lambda w: impair(w, _C['fast-flutter'], seed=2026),
    'jitter 0.1%': lambda w: impair(jitter_warp(w), _C['clean-96k'], seed=2026),
    'jitter 0.3%': lambda w: impair(jitter_warp(w, .003), _C['clean-96k'], seed=2026),
    'lowpass-12k': lambda w: impair(w, _C['lowpass-12k'], seed=2026),
    'lowpass-10k': lambda w: impair(w, _C['lowpass-10k'], seed=2026),
    'dropouts': lambda w: impair(w, _C['dropouts'], seed=2026),
    'warble+lpf12k+dropouts': lambda w: impair(
        w, dataclasses.replace(_C['fast-flutter'], lowpass=12000, dropout_ms=12,
                               dropout_every=.7), seed=2026),
}


def decode(model, values, impairment, packets):
    """Steady decoded packets as (result, (xhat, conf))."""
    wire = resample_poly(v7.encode_pulse_stream(model, [values]*packets, 1, [0]*packets),
                         2, 1, axis=0).astype(np.float32)
    with equaliser_capture() as captured:
        results, _ = v7.decode_pulse_stream(model, IMPAIRMENTS[impairment](wire),
                                            sample_rate=RATE)
    equalised = [r for r in results if 'noise' in r.diag]
    if len(equalised) != len(captured):
        raise RuntimeError('equaliser capture out of step with the decoder')
    by_result = {id(r): cap for r, cap in zip(equalised, captured)}
    return [(r, by_result.get(id(r))) for r in results
            if r.counter >= STEADY_FROM and r.status != 'lost']


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('frames', nargs='+', type=Path)
    ap.add_argument('--folds', type=int, nargs='+', default=[500, 1000])
    ap.add_argument('--design-db', type=float, default=30.0,
                    help='channel SNR the fold step is chosen for')
    ap.add_argument('--conf-min', type=float, default=.9)
    ap.add_argument('--packets', type=int, default=20)
    ap.add_argument('--cases', nargs='+', default=list(IMPAIRMENTS), choices=list(IMPAIRMENTS),
                    metavar='CASE')
    ap.add_argument('--quick', action='store_true',
                    help='smoke run: one scored frame, clean + dropouts + jitter 0.1%%')
    ap.add_argument('--out', type=Path, default=REPO/'tmp'/'test_modem_v7'/'fold_modem')
    args = ap.parse_args(argv)
    train, test = load_frames(args.frames)
    if args.quick:
        test, args.cases, args.packets = test[:1], ['clean', 'dropouts', 'jitter 0.1%'], 12
    args.out.mkdir(parents=True, exist_ok=True)

    model = v7.load_model(TARGET, 'box')
    codecs = {M: FoldCodec(model, train, M, design_db=args.design_db, conf_min=args.conf_min)
              for M in args.folds}
    for M, codec in codecs.items():
        print(f'# fold {M}: step D={codec.D:.3f} (chosen for {args.design_db:g} dB)', flush=True)
    grid = next(iter(codecs.values())).grid if codecs else None
    rows = []
    for case in args.cases:
        scores, sheet = {'base': []}, []
        for index, frame in enumerate(test):
            ref, values = reference(frame), v7_values(frame, 'box')
            picture = None
            for r, _ in decode(model, values, case, args.packets):
                picture = grid.picture(codecs[args.folds[0]].plain(r.coeffs))
                scores['base'].append(ssimulacra2(ref, picture))
            if index == 0 and picture is not None:
                sheet += [('source', ref), (f'{case}: no fold', picture)]
            for M, codec in codecs.items():
                for label in (f'fold {M}', f'fold {M}+fb'):
                    scores.setdefault(label, [])
                picture = None
                for r, (xhat, conf) in decode(model, codec.encode(values), case, args.packets):
                    for fallback in (False, True):
                        picture = grid.picture(codec.decode(r.coeffs, xhat, conf, fallback))
                        label = f'fold {M}+fb' if fallback else f'fold {M}'
                        scores[label].append(ssimulacra2(ref, picture))
                if index == 0 and picture is not None:
                    sheet.append((f'{case}: fold {M}+fb', picture))
        row = {'case': case, **{k: round(float(np.mean(v)), 1) for k, v in scores.items()},
               'frames_scored': len(scores['base'])}
        rows.append(row)
        print(json.dumps(row), flush=True)
        if sheet:
            contact_sheet(args.out/f'{case.replace(" ", "_").replace("%", "pct")}.png', sheet)
    (args.out/'results.json').write_text(json.dumps(rows, indent=1) + '\n')


if __name__ == '__main__':
    main()
