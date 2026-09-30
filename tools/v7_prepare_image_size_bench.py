#!/usr/bin/env python3
"""Compare generic BOX resize with V7's dimension-gated preparation path.

The script writes a deterministic set of RGB PNGs under tmp/, measures warmed
paired preparation calls, and records source/output differences. It is an
offline image-only benchmark; it does not open an audio device.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from animation_modem import v7


SIZE_CASES = (
    # Smaller fallback and eligible sizes; source ratios vary deliberately.
    (320, 240, 'small-4x-by-2_5x'),
    (400, 480, 'portrait-grid-5x'),
    (640, 480, '4x3-grid-8x-by-5x'),
    (800, 480, 'wide-grid-10x-by-5x'),
    (720, 960, 'face-3x4-grid-9x-by-10x'),
    (1280, 768, 'wide-grid-16x-by-8x'),
    (1440, 960, 'photo-grid-18x-by-10x'),
    (1920, 1152, 'large-grid-24x-by-12x'),
    # Common or deliberately awkward dimensions that must use resize fallback.
    (1280, 720, 'hd-video-width-only'),
    (1600, 900, 'hd-plus-width-only'),
    (1920, 1080, 'full-hd'),
    (2560, 1440, 'qhd'),
    (3840, 2160, '4k-uhd'),
    (721, 961, 'odd-near-face'),
    (1080, 1920, 'portrait-video-height-only'),
)
WARMUPS = 5
DEFAULT_ROUNDS = 40


def _patterned_rgb(width, height, seed):
    """Build deterministic color/edge/texture content at the requested size."""
    y, x = np.indices((height, width), dtype=np.uint32)
    checker = ((x//11 + y//17) & 1)*71
    diagonal = ((x ^ (y*3)) >> 3) & 63
    red = (x*13 + y*3 + checker + diagonal + seed*19) & 255
    green = (x*5 + y*17 + checker//2 + diagonal*2 + seed*31) & 255
    blue = (x*23 + y*7 + checker//3 + diagonal*3 + seed*47) & 255
    return np.stack((red, green, blue), axis=-1).astype(np.uint8)


def _generic_box(image):
    """The pre-optimization implementation, including RGB conversion."""
    return image.convert('RGB').resize(
        v7.PREPARED_SIZE, Image.Resampling.BOX)


def _percentile(values, percentile):
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _paired_timings(image, rounds, seed):
    functions = {
        'generic_resize': _generic_box,
        'prepare_image': lambda source: v7.prepare_image(source, 'box'),
    }
    for function in functions.values():
        for _ in range(WARMUPS):
            function(image)

    samples = {name: [] for name in functions}
    paired_delta = []
    order_rng = random.Random(seed)
    for _ in range(rounds):
        order = list(functions)
        order_rng.shuffle(order)
        pair = {}
        for name in order:
            started = time.perf_counter_ns()
            functions[name](image)
            elapsed = (time.perf_counter_ns()-started)/1e6
            samples[name].append(elapsed)
            pair[name] = elapsed
        paired_delta.append(pair['generic_resize']-pair['prepare_image'])

    summary = {}
    for name, values in samples.items():
        summary[name] = {
            'median_ms': float(np.median(values)),
            'p90_ms': _percentile(values, 90),
            'samples_ms': values,
        }
    summary['paired_generic_minus_prepare_median_ms'] = float(
        np.median(paired_delta))
    summary['paired_samples_generic_minus_prepare_ms'] = paired_delta
    return summary


def _pixel_metrics(actual, generic):
    actual = np.asarray(actual, dtype=np.int16)
    generic = np.asarray(generic, dtype=np.int16)
    delta = actual-generic
    return {
        'byte_identical': bool(np.array_equal(actual, generic)),
        'max_abs_rgb_code': int(np.max(np.abs(delta), initial=0)),
        'rms_rgb_code': float(np.sqrt(np.mean(delta.astype(np.float64)**2))),
        'changed_rgb_values': int(np.count_nonzero(delta)),
    }


def run(out, rounds):
    image_dir = out/'images'
    image_dir.mkdir(parents=True, exist_ok=True)
    report_cases = []

    for index, (width, height, name) in enumerate(SIZE_CASES):
        path = image_dir/f'{name}_{width}x{height}.png'
        Image.fromarray(_patterned_rgb(width, height, index+1), 'RGB').save(path)
        with Image.open(path) as opened:
            image = opened.convert('RGB')

        reduce_eligible = (
            width % v7.PREPARED_SIZE[0] == 0 and
            height % v7.PREPARED_SIZE[1] == 0)
        expected_factors = (
            [width//v7.PREPARED_SIZE[0], height//v7.PREPARED_SIZE[1]]
            if reduce_eligible else None)

        generic = _generic_box(image)
        prepared = v7.prepare_image(image, 'box')
        pixel_metrics = _pixel_metrics(prepared, generic)
        values_delta = (v7.image_values(prepared, v7.V7_SHAPES, 'box') -
                        v7.image_values(generic, v7.V7_SHAPES, 'box'))
        timing = _paired_timings(image, rounds, 20260930+index)
        old_median = timing['generic_resize']['median_ms']
        new_median = timing['prepare_image']['median_ms']
        report_cases.append({
            'name': name,
            'image': str(path.relative_to(out)),
            'source_dimensions': [width, height],
            'source_aspect': width/height,
            'prepared_dimensions': list(v7.PREPARED_SIZE),
            'width_divisible_by_80': width % 80 == 0,
            'height_divisible_by_96': height % 96 == 0,
            'uses_integer_reduce': reduce_eligible,
            'reduce_factors_xy': expected_factors,
            'output_pixels_vs_generic_box': pixel_metrics,
            'coder_values_vs_generic_box': {
                'max_abs': float(np.max(np.abs(values_delta), initial=0.0)),
                'rms': float(np.sqrt(np.mean(values_delta**2))),
                'changed_values': int(np.count_nonzero(values_delta)),
            },
            'timing': timing,
            'median_speedup_percent': float(
                (old_median-new_median)/old_median*100)
                if old_median else 0.0,
            'generic_over_prepare_median_ratio': (
                float(old_median/new_median) if new_median else None),
        })

        print(f'{width:4}x{height:<4} '
              f'{"reduce "+str(expected_factors) if reduce_eligible else "fallback":<18} '
              f'{old_median:.3f} -> {new_median:.3f} ms '
              f'({report_cases[-1]["median_speedup_percent"]:+.1f}%) '
              f'max pixel Δ {pixel_metrics["max_abs_rgb_code"]}',
              flush=True)

    report = {
        'title': 'V7 source preparation size/dimension suite',
        'benchmark': 'generic RGB resize(80x96, BOX) versus v7.prepare_image(..., BOX)',
        'prepared_dimensions': list(v7.PREPARED_SIZE),
        'eligibility_rule': 'source width divisible by 80 AND source height divisible by 96',
        'rounds_per_path_per_image': rounds,
        'warmups_per_path_per_image': WARMUPS,
        'measurement': 'paired randomized operation order, warmed, image load excluded, Pillow operations only, no audio device',
        'image_generation': 'deterministic patterned RGB gradients, checker edges, and diagonal texture; each generated PNG and its dimensions are retained under images/',
        'cases': report_cases,
    }
    report_path = out/'results.json'
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    print(f'saved {report_path}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--out', type=Path,
        default=Path('tmp/v7-prepare-image-size-suite-20260930'),
        help='directory for generated image suite and raw results')
    parser.add_argument('--rounds', type=int, default=DEFAULT_ROUNDS)
    args = parser.parse_args(argv)
    if args.rounds < 5:
        parser.error('--rounds must be at least 5')
    run(args.out, args.rounds)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
