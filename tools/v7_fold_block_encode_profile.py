#!/usr/bin/env python3
"""Paired production-path timing for resize and block-Fold projections.

The timer includes per-frame source preparation, Fold 500, coded-pilot
insertion, and packet synthesis. It is offline and does not open an audio
device or apply output-rate adaptation.
"""
import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from animation_modem import v7
from animation_modem.v7_fold import Fold500
from animation_modem.v7_source_dct import FoldBlockDCTProjector
from modem_v7_display import (
    _source_dct_coefficients, _source_values,
    encode_folded_coefficients_packet, encode_values_packet)


DEFAULT_SOURCE = ROOT/'images_sbs/face/00_C_BG_faceSource_960/benFaceSource0000.jpg'
DEFAULT_OUT = ROOT/'tmp/v7-fold-block-encode-profile.json'
TARGET = .1521/np.sqrt(1+10**(v7.CLOCK_REL_DB/10))


def _source_image(path):
    with Image.open(path) as opened:
        image = opened.convert('RGB')
    if image.width*2 == image.height*3:
        image = image.crop((0, 0, image.width//2, image.height))
    return image


def _stats(values):
    samples = np.asarray(values, dtype=np.float64)
    return {
        'median_ms': statistics.median(samples),
        'p10_ms': float(np.percentile(samples, 10)),
        'p90_ms': float(np.percentile(samples, 90)),
    }


def run(sources=None, out=DEFAULT_OUT, block_size=8, rounds=200, warmups=10,
        seed=20260929):
    if rounds <= 0 or warmups < 0:
        raise ValueError('rounds must be positive and warmups non-negative')
    sources = list(sources or [DEFAULT_SOURCE])
    model = v7.load_model(TARGET, 'box')
    fold = Fold500(model)
    loop = v7.LoopInfo(12, v7.LOOP_NO_CLOCK, pingpong=True)
    rng = random.Random(seed)
    source_results = []

    for path in sources:
        image = _source_image(path)
        rgb = np.asarray(image)
        aspect = v7.aspect_wire_code(image.size)
        setup_started = time.perf_counter_ns()
        projector = FoldBlockDCTProjector(
            rgb.shape[:2], model.coder.grids, fold.source_positions,
            block_size=block_size)
        projector_setup_ms = (time.perf_counter_ns()-setup_started)/1e6

        def resize_first(index):
            values = _source_values(model, image, encode_filter='box')
            return encode_values_packet(
                model, values, 1, index, aspect, loop=loop, direction=1,
                eof_marker=True, fold=fold)

        def direct_grid_dct(index):
            full = _source_dct_coefficients(model, image, 'box')
            coefficients = fold.encode_dct_coefficients(full)
            return encode_folded_coefficients_packet(
                model, coefficients, 1, index, aspect, loop=loop,
                direction=1, eof_marker=True)

        def fold_block(index):
            full = projector.project(rgb)
            coefficients = fold.encode_dct_coefficients(full)
            return encode_folded_coefficients_packet(
                model, coefficients, 1, index, aspect, loop=loop,
                direction=1, eof_marker=True)

        paths = {
            'resize_first': resize_first,
            'direct_grid_dct': direct_grid_dct,
            'fold_block': fold_block,
        }
        if not np.array_equal(resize_first(0), direct_grid_dct(0)):
            raise AssertionError(
                'direct grid-DCT baseline differs from resize-first packet')
        for index in range(warmups):
            for encode in paths.values():
                audio = encode(index % 12)
                if len(audio) != v7.PULSE_FRAME or not np.isfinite(audio).all():
                    raise RuntimeError('warm-up packet encode returned invalid audio')

        samples = {name: [] for name in paths}
        paired_deltas = []
        paired_ratios = []
        paired_grid_dct_deltas = []
        paired_grid_dct_ratios = []
        for index in range(rounds):
            names = list(paths)
            rng.shuffle(names)
            pair = {}
            for name in names:
                started = time.perf_counter_ns()
                if name == 'resize_first':
                    values = _source_values(model, image, encode_filter='box')
                    prepared = time.perf_counter_ns()
                    audio = encode_values_packet(
                        model, values, 1, index % 12, aspect, loop=loop,
                        direction=1, eof_marker=True, fold=fold)
                elif name == 'direct_grid_dct':
                    full = _source_dct_coefficients(model, image, 'box')
                    prepared = time.perf_counter_ns()
                    coefficients = fold.encode_dct_coefficients(full)
                    audio = encode_folded_coefficients_packet(
                        model, coefficients, 1, index % 12, aspect, loop=loop,
                        direction=1, eof_marker=True)
                else:
                    full = projector.project(rgb)
                    prepared = time.perf_counter_ns()
                    coefficients = fold.encode_dct_coefficients(full)
                    audio = encode_folded_coefficients_packet(
                        model, coefficients, 1, index % 12, aspect, loop=loop,
                        direction=1, eof_marker=True)
                ended = time.perf_counter_ns()
                if len(audio) != v7.PULSE_FRAME or not np.isfinite(audio).all():
                    raise RuntimeError(f'{name} packet encode returned invalid audio')
                row = {
                    'prepare_ms': (prepared-started)/1e6,
                    'encode_ms': (ended-prepared)/1e6,
                    'total_ms': (ended-started)/1e6,
                }
                samples[name].append(row)
                pair[name] = row['total_ms']
            paired_deltas.append(pair['fold_block']-pair['resize_first'])
            paired_ratios.append(pair['fold_block']/pair['resize_first'])
            paired_grid_dct_deltas.append(
                pair['fold_block']-pair['direct_grid_dct'])
            paired_grid_dct_ratios.append(
                pair['fold_block']/pair['direct_grid_dct'])

        source_results.append({
            'source': str(Path(path).resolve()),
            'source_size': list(image.size),
            'block_size': block_size,
            'projector_setup_ms': projector_setup_ms,
            'samples_per_path': rounds,
            'resize_first': {
                stage: _stats([row[stage] for row in samples['resize_first']])
                for stage in ('prepare_ms', 'encode_ms', 'total_ms')},
            'direct_grid_dct': {
                stage: _stats([row[stage]
                               for row in samples['direct_grid_dct']])
                for stage in ('prepare_ms', 'encode_ms', 'total_ms')},
            'fold_block': {
                stage: _stats([row[stage] for row in samples['fold_block']])
                for stage in ('prepare_ms', 'encode_ms', 'total_ms')},
            'paired_fold_block_minus_resize_median_ms': float(
                np.median(paired_deltas)),
            'paired_fold_block_over_resize_median_ratio': float(np.median(
                paired_ratios)),
            'paired_fold_block_minus_direct_grid_dct_median_ms': float(
                np.median(paired_grid_dct_deltas)),
            'paired_fold_block_over_direct_grid_dct_median_ratio': float(
                np.median(paired_grid_dct_ratios)),
        })

    report = {
        'profile': 'canonical Box model, production Fold-500 packet encoders',
        'warmups_per_path': warmups,
        'paired_rounds_per_source': rounds,
        'seed': seed,
        'timed_work': (
            'source preparation + Fold 500 + coded pilot + one complete packet; '
            'source/model setup and output-rate adaptation excluded'),
        'order': ('randomized resize-first, direct grid-DCT, and fold-block '
                  'path order within each pair'),
        'sources': source_results,
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2)+'\n')
    for result in source_results:
        base = result['resize_first']['total_ms']
        candidate = result['fold_block']['total_ms']
        print(Path(result['source']).name)
        grid = result['direct_grid_dct']['total_ms']
        print(f"  resize-first median/P90: {base['median_ms']:.3f}/"
              f"{base['p90_ms']:.3f} ms")
        print(f"  direct grid-DCT median/P90: {grid['median_ms']:.3f}/"
              f"{grid['p90_ms']:.3f} ms")
        print(f"  fold-block median/P90: {candidate['median_ms']:.3f}/"
              f"{candidate['p90_ms']:.3f} ms")
    print(f'Wrote {out}')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', action='append', type=Path,
                        help='native RGB image; repeat to profile more frames')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--block-size', type=int, default=8)
    parser.add_argument('--rounds', type=int, default=200)
    parser.add_argument('--warmups', type=int, default=10)
    parser.add_argument('--seed', type=int, default=20260929)
    args = parser.parse_args(argv)
    run(args.source, args.out, args.block_size, args.rounds,
        args.warmups, args.seed)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
