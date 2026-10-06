"""Nested fold against stock through the V7 torture matrix.

Uses the repository's own impairment cases (``tools/v7_torture_matrix.py``:
filters, hiss, wow and flutter, azimuth, crosstalk, hum, bias, saturation,
dropouts, NR pumping, the two tape types, mono sum, one leg).  Each scored
unit is a stream of real ``aspect-fold-500`` packets of one picture, impaired
as a stream and decoded by the production receiver.

    .venv/bin/python tools/v7_sk_torture.py --out tmp/sk/torture

Synthetic channels are not tape validation.
"""
import argparse
import io
import json
import math
import sys
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from animation_modem import v7
from tools import v7_sk_study as study
from tools.v7_sk_adaptive import Adaptive, Leveled
from tools.v7_sk_wire import Link, matched, shipped_plane
from tools.v7_torture_matrix import CASES, impair

STOCK = 'stock as shipped'
PAD = 4800


@contextmanager
def equaliser_calls():
    """Equaliser output of every decoded packet, in decode order, with the
    equaliser's own packet phase (the counter modulo the tail rotation)."""
    captured = []
    real = v7._equalize_numba

    def capture(model, Z, H, noise, counter):
        out = real(model, Z, H, noise, counter)
        captured.append((int(counter), out[0].copy(), out[1].copy()))
        return out

    v7._equalize_numba = capture
    try:
        yield captured
    finally:
        v7._equalize_numba = real


def stream(link, rgb, symbols, frames, index):
    """Back-to-back packets of one picture, each at the stock packet's level."""
    packets = []
    for counter in range(1, frames+1):
        stock = link.packet(rgb, None, counter, index)
        packets.append(stock if symbols is None else
                       matched(link.packet(rgb, symbols, counter, index), stock)[0])
    silence = np.zeros((PAD, 2), np.float32)
    return np.concatenate([silence]+packets+[silence])


def decode_stream(link, audio):
    """{counter: (luma symbols or None, stock values or None)} for usable packets."""
    with equaliser_calls() as captured, redirect_stdout(io.StringIO()):
        packets, _, _ = link.wire.decode(np.asarray(audio, np.float64))
    out, at = {}, 0
    for result, values in packets:
        # Results and equaliser calls are both in stream order; a packet lost
        # before equalisation has no call.  Match on the packet phase.
        symbols = None
        phase = int(result.counter) % v7.TAIL_PHASES
        if link.g.mono:
            # The mono receiver attaches its equaliser output to each result.
            symbols = link.symbols_of(result)
        elif at < len(captured) and captured[at][0] == phase:
            _, xhat, confidence = captured[at]
            symbols = (xhat/np.maximum(confidence, 1e-3))[link.index]/link.sd
            at += 1
        if values is None or result.status == 'lost':
            continue
        out[int(result.counter)] = (symbols, values)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--out', default='tmp/sk/torture')
    parser.add_argument('--corpus', default='tmp/sk/corpus')
    parser.add_argument('--noise', default='tmp/sk/wire/NOISE.npz')
    parser.add_argument('--adaptive', default='tmp/sk/study/ADAPTIVE.json')
    parser.add_argument('--pictures', nargs='+', required=True,
                        help='pictures to score (any size; centre-cropped to 3:4)')
    parser.add_argument('--frames', type=int, default=12)
    parser.add_argument('--tables', nargs='+', default=['clean', 'hiss-45'],
                        help='noise conditions whose frozen fold tables to run')
    parser.add_argument('--only', nargs='+', default=[])
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--profile', default='aspect-fold-500', choices=('aspect-fold-500', 'aspect-mono-500'))
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    geometry = study.Geometry(args.profile)
    link = Link(geometry)
    train_files, _ = study.corpus(args.corpus)
    small, big = study.Analyzer(study.LATTICE), study.Analyzer(geometry.big)
    train = [small(study.portrait(f)) for f in train_files]
    stats = study.Statistics(train)
    noise_file = np.load(args.noise)
    chosen = json.loads(Path(args.adaptive).read_text())['conditions']
    schemes = {'stock mapping, retrained': study.Production(geometry, stats, noise_file['clean'])}
    for table in args.tables:
        schemes[f'fold, {table} table'] = Leveled(Adaptive(
            geometry, stats, noise_file[table], train, passes=2, **chosen[table]['parameters']))
    rgbs = [study.portrait(f) for f in args.pictures]
    planes = [small(rgb) for rgb in rgbs]
    references = [big(rgb) for rgb in rgbs]
    tail = study.ensemble_tail(geometry, study.Statistics(references).variance)
    # Clean streams are built once and reused for every case.
    streams = {}
    for i, (rgb, plane) in enumerate(zip(rgbs, planes)):
        streams[STOCK, i] = (stream(link, rgb, None, args.frames, 100+i), None)
        for name, scheme in schemes.items():
            symbols = scheme.encode(plane)
            streams[name, i] = (stream(link, rgb, symbols, args.frames, 100+i), scheme)
    names = [STOCK]+list(schemes)
    report = {'pictures': [Path(f).name for f in args.pictures], 'frames_per_stream': args.frames,
              'tables': args.tables, 'seed': args.seed, 'cases': {}}
    for case in CASES:
        if args.only and case.name not in args.only:
            continue
        row = {}
        for name in names:
            errors, energy, delivered, psnr = [], [], 0, []
            for i in range(len(rgbs)):
                audio, scheme = streams[name, i]
                try:
                    decoded = decode_stream(link, impair(audio, case, seed=args.seed+i))
                except Exception as error:                      # a case the receiver cannot take
                    row.setdefault('errors', {})[name] = repr(error)[:120]
                    decoded = {}
                for counter, (symbols, values) in decoded.items():
                    if scheme is None:
                        plane = shipped_plane(values)
                    elif symbols is None:
                        continue
                    else:
                        plane = scheme.decode(symbols)
                    score = study.score(geometry, references[i], plane)
                    errors.append(score['error'])
                    psnr.append(score['psnr'])
                    delivered += 1
            total = args.frames*len(rgbs)
            row[name] = {'delivered': delivered, 'sent': total,
                         'effective_coefficients': (study.ensemble_equivalent(tail, float(np.mean(errors)))
                                                    if errors else None),
                         'psnr_db': float(np.mean(psnr)) if psnr else None}
        report['cases'][case.name] = row
        print(case.name, json.dumps({n: (row[n]['delivered'], row[n]['effective_coefficients'])
                                     for n in names}), flush=True)
        (out/'TORTURE.json').write_text(json.dumps(report, indent=1)+'\n')
    lines = ['| Case | '+' | '.join(names)+' |', '|---|'+'---:|'*len(names)]
    for case, row in report['cases'].items():
        cells = []
        for name in names:
            r = row[name]
            cells.append('none decoded' if r['effective_coefficients'] is None else
                         f"{r['effective_coefficients']:,} ({r['delivered']}/{r['sent']})")
        lines.append(f'| {case} | '+' | '.join(cells)+' |')
    (out/'TORTURE.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
