#!/usr/bin/env python3
"""Paired production-path timing for resize-first and grid-DCT Fold-500 sends.

The comparison includes image preparation, Fold 500, coded-pilot insertion,
and packet synthesis. It does not open an audio device or perform output-rate
adaptation.
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
from modem_v7_display import (
    _source_values, encode_image_dct_packet, encode_values_packet)


DEFAULT_SOURCE = (
    ROOT/'images_sbs/face/14_D_BG_First_Upscale_960/0000.jpg')
DEFAULT_OUT = ROOT/'tmp/v7-folded-grid-dct-timing.json'
TARGET = .1521/np.sqrt(1+10**(v7.CLOCK_REL_DB/10))


def _source_image(path):
    with Image.open(path) as opened:
        image = opened.convert('RGB')
    if image.width*2 == image.height*3:
        image = image.crop((0, 0, image.width//2, image.height))
    return image


def _measure(fn):
    started = time.perf_counter()
    fn()
    return (time.perf_counter()-started)*1000


def run(source, out, rounds=120, warmups=10, seed=20260929):
    if rounds <= 0 or warmups < 0:
        raise ValueError('rounds must be positive and warmups non-negative')
    image = _source_image(source)
    model = v7.load_model(TARGET, 'box')
    fold = Fold500(model)
    loop = v7.LoopInfo(12, v7.LOOP_NO_CLOCK, pingpong=True)
    aspect = v7.aspect_wire_code(image.size)

    def resize_first():
        values = _source_values(model, image, 'box')
        return encode_values_packet(
            model, values, 1, 0, aspect, loop=loop,
            eof_marker=True, fold=fold)

    def direct_grid_dct():
        return encode_image_dct_packet(
            model, image, 1, 0, aspect, loop=loop,
            eof_marker=True, fold=fold, encode_filter='box')

    expected = resize_first()
    actual = direct_grid_dct()
    packets_equal = bool(np.array_equal(actual, expected))
    if not packets_equal:
        raise AssertionError('direct grid-DCT packet differs from resize-first')
    for _ in range(warmups):
        resize_first()
        direct_grid_dct()

    rng = random.Random(seed)
    resize_ms, direct_ms, ratios, deltas = [], [], [], []
    for _ in range(rounds):
        pair = {}
        order = ['resize_first', 'direct_grid_dct']
        rng.shuffle(order)
        for name in order:
            fn = resize_first if name == 'resize_first' else direct_grid_dct
            pair[name] = _measure(fn)
        resize_ms.append(pair['resize_first'])
        direct_ms.append(pair['direct_grid_dct'])
        ratios.append(pair['direct_grid_dct']/pair['resize_first'])
        deltas.append(pair['direct_grid_dct']-pair['resize_first'])

    def stats(values):
        return {
            'median_ms': statistics.median(values),
            'p10_ms': float(np.percentile(values, 10)),
            'p90_ms': float(np.percentile(values, 90)),
        }

    result = {
        'source': str(Path(source).resolve()),
        'source_size': list(image.size),
        'rounds': rounds,
        'warmups': warmups,
        'seed': seed,
        'model_table_sha256': v7.MODEL_TABLES_SHA256,
        'resize_first': stats(resize_ms),
        'direct_grid_dct': stats(direct_ms),
        'paired_direct_over_resize_median_ratio': float(np.median(ratios)),
        'paired_direct_over_resize_p10_ratio': float(np.percentile(ratios, 10)),
        'paired_direct_over_resize_p90_ratio': float(np.percentile(ratios, 90)),
        'paired_median_delta_ms': float(np.median(deltas)),
        'packets_byte_identical': packets_equal,
        'timed_work': (
            'source preparation + Fold 500 + coded pilot + packet synthesis; '
            'no compositing, output-rate adaptation, or device output'),
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(f"resize-first median: {result['resize_first']['median_ms']:.3f} ms")
    print(f"direct grid-DCT median: {result['direct_grid_dct']['median_ms']:.3f} ms")
    ratio = result['paired_direct_over_resize_median_ratio']
    print(f'paired median ratio: {ratio:.4f} ({(ratio-1)*100:+.2f}%)')
    print(f"byte-identical packets: {packets_equal}; results: {out}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--rounds', type=int, default=120)
    parser.add_argument('--warmups', type=int, default=10)
    parser.add_argument('--seed', type=int, default=20260929)
    args = parser.parse_args(argv)
    run(args.source, args.out, args.rounds, args.warmups, args.seed)


if __name__ == '__main__':
    main()
