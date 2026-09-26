#!/usr/bin/env python3
"""Paired CPU smoke timing for the standalone V7 live encode/decode paths.

Always report the selected feature beside its matched baseline. CPU numbers
without the baseline comparison are not useful for judging feature overhead.

Run from the repository root::

    .venv/bin/python test_modem_v7/cpu_smoke.py
    .venv/bin/python test_modem_v7/cpu_smoke.py --packets 12 --repeats 20

This is synthetic and opens no audio device. It includes source preparation,
packet encoding, and coded-pilot overlay on send; pulse decode is reported both
alone and with coefficient-to-pixel reconstruction on receive. Capture, audio
I/O, display, and optional log-stat scans are excluded.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
PROTOTYPE = ROOT/'test_modem_v7'
for path in (ROOT, PROTOTYPE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from animation_modem import v7                                     # noqa: E402
from tools import v7_live                                           # noqa: E402
from live_fold import LiveFold                                      # noqa: E402
from tone_code import (coded_pilot_timing, warmup_coded_decoder,
                       warmup_tone_templates)                       # noqa: E402


def _measure(function, repeats, units_per_run):
    wall_ms, cpu_ms = [], []
    for index in range(repeats):
        wall_start = time.perf_counter_ns()
        cpu_start = time.process_time_ns()
        result = function(index)
        completed_units = (int(result) if isinstance(result, (int, np.integer))
                           else units_per_run)
        cpu_elapsed = (time.process_time_ns()-cpu_start)/1e6
        wall_elapsed = (time.perf_counter_ns()-wall_start)/1e6
        if completed_units != units_per_run:
            raise RuntimeError(
                f'expected {units_per_run} decoded packets, got {completed_units}')
        wall_ms.append(wall_elapsed/units_per_run)
        cpu_ms.append(cpu_elapsed/units_per_run)

    def summary(samples):
        return {
            'median_ms_per_unit': round(statistics.median(samples), 3),
            'p95_ms_per_unit': round(float(np.percentile(samples, 95)), 3),
        }
    return {'wall': summary(wall_ms), 'cpu': summary(cpu_ms)}


def _comparison(baseline, feature):
    base = baseline['cpu']['median_ms_per_unit']
    current = feature['cpu']['median_ms_per_unit']
    return {
        'cpu_delta_ms_per_unit': round(current-base, 3),
        'cpu_delta_pct': round((current/base-1)*100, 1) if base else None,
    }


def run(args):
    fixture = v7_live.DEFAULT_FIXTURE
    with Image.open(fixture) as source:
        image = source.convert('RGB')

    base_model = v7_live._model(fixture, 'nearest')
    feature_model = v7_live._model(fixture, 'box')
    fold = LiveFold(500)
    fold.check(feature_model)
    warmup_tone_templates(fold.slots)
    v7.warmup_equalizer(base_model)
    v7.warmup_equalizer(feature_model)
    warmup_coded_decoder(feature_model)

    def encode_baseline(counter):
        values, aspect = v7_live._values(
            base_model, image, 'nearest', 1.05, 1.0)
        packet = v7.encode_pulse_frame(
            base_model, values, counter+1, aspect_code=aspect,
            source_index=counter, pilot_tones=True, eof_marker=True)
        return v7.speed_pulse_stream(packet, 1.0, rate=v7.RATE)

    def encode_feature(counter):
        values, aspect = v7_live._values(
            feature_model, image, 'box', 1.0, 1.0)
        coeffs = fold.encode_coefficients(feature_model, values)
        packet = v7_live._encode_pulse_frame_coeffs(
            feature_model, coeffs, counter+1, aspect_code=aspect,
            source_index=counter, eof_marker=True)
        packet = v7_live._add_coded_pilots(packet, counter+1, fold.slots)
        return v7.speed_pulse_stream(packet, 1.0, rate=v7.RATE)

    # Generate repeatable receiver inputs before timing either decode path.
    base_wire = np.concatenate([
        encode_baseline(index) for index in range(args.packets)])
    feature_wire = np.concatenate([
        encode_feature(index) for index in range(args.packets)])

    def decode_baseline(reconstruct):
        def decode(_iteration):
            results, _ = v7.decode_pulse_stream(
                base_model, base_wire, sample_rate=v7.RATE,
                pilot_timing='tone-seeded', frame_boundary='eof')
            if reconstruct:
                for result in results:
                    if (result.status in ('received', 'verified') or
                            result.diag.get('displayable')):
                        v7.values_from(base_model, result.coeffs)
            return len(results)
        return decode

    def decode_feature(reconstruct):
        def decode(_iteration):
            fold.install()
            try:
                with coded_pilot_timing():
                    results, _ = v7.decode_pulse_stream(
                        feature_model, feature_wire, sample_rate=v7.RATE,
                        pilot_timing='tone-seeded', frame_boundary='eof')
                if reconstruct:
                    for result in results:
                        if (result.status in ('received', 'verified') or
                                result.diag.get('displayable')):
                            fold.values(
                                feature_model, result,
                                metadata_confirmed=(
                                    v7_live._coded_mode_matches_fold(result,
                                                                     fold)))
                return len(results)
            finally:
                fold.uninstall()
        return decode

    # Warm Python/Numba and reconstruction paths before measuring.
    decode_baseline(False)(0)
    decode_feature(False)(0)
    decode_baseline(True)(0)
    decode_feature(True)(0)

    encode_base = _measure(encode_baseline, args.repeats, 1)
    encode_featured = _measure(encode_feature, args.repeats, 1)
    decode_base = _measure(decode_baseline(False), args.repeats, args.packets)
    decode_featured = _measure(decode_feature(False), args.repeats, args.packets)
    render_base = _measure(decode_baseline(True), args.repeats, args.packets)
    render_featured = _measure(decode_feature(True), args.repeats, args.packets)

    report = {
        'fixture': str(fixture.relative_to(ROOT)),
        'packets_per_decode_run': args.packets,
        'repeats': args.repeats,
        'baseline': 'nearest / brightness 1.05 / steady pilots',
        'feature': 'box / brightness 1.0 / coded M=500 fold',
        'audio_io_capture_display_and_log_stats': 'excluded',
        'encode': {
            'baseline': encode_base,
            'feature': encode_featured,
            'comparison': _comparison(encode_base, encode_featured),
        },
        'decode_core': {
            'baseline': decode_base,
            'feature': decode_featured,
            'comparison': _comparison(decode_base, decode_featured),
        },
        'decode_plus_reconstruction': {
            'baseline': render_base,
            'feature': render_featured,
            'comparison': _comparison(render_base, render_featured),
        },
    }
    print(json.dumps(report, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packets', type=int, default=8,
                        help='packets per decode pass (default: 8)')
    parser.add_argument('--repeats', type=int, default=15,
                        help='timed repetitions after warmup (default: 15)')
    args = parser.parse_args(argv)
    if args.packets < 1 or args.repeats < 2:
        parser.error('--packets must be positive and --repeats must be at least 2')
    run(args)


if __name__ == '__main__':
    main()
