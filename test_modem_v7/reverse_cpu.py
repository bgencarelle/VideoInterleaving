#!/usr/bin/env python3
"""Paired CPU check for automatic V7 direction acquisition and reverse decode.

Run from the repository root::

    .venv/bin/python test_modem_v7/reverse_cpu.py [--repeats 300]

The 4,048-sample input window contains one complete packet plus bounded
neighbor margins, matching an interior packet in LiveInput's rolling buffer.
It is synthetic and opens no audio device.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from animation_modem import v7                                      # noqa: E402


def _paired(first, second, repeats):
    batch_size = 5
    a, b = [], []
    for index in range(repeats):
        order = (first, second) if index % 2 == 0 else (second, first)
        elapsed = []
        for function in order:
            start = time.process_time_ns()
            for _ in range(batch_size):
                function()
            elapsed.append((time.process_time_ns()-start)/1e6/batch_size)
        if index % 2 == 0:
            a.append(elapsed[0])
            b.append(elapsed[1])
        else:
            b.append(elapsed[0])
            a.append(elapsed[1])

    def summary(values):
        return {
            'median_cpu_ms_per_packet': round(statistics.median(values), 4),
            'p95_cpu_ms_per_packet': round(
                float(np.percentile(values, 95)), 4),
        }
    return summary(a), summary(b)


def run(repeats):
    target = .1521/np.sqrt(1 + 10**(v7.CLOCK_REL_DB/10))
    model = v7.load_model(target, 'nearest')
    with Image.open(v7.REFERENCE_FIXTURE) as source:
        values = v7.image_values(
            v7.prepare_image(source, 'nearest'), model.coder.grids, 'nearest')
    wire = np.concatenate([
        v7.encode_pulse_frame(
            model, values, counter=index+1, aspect_code=6,
            source_index=index, pilot_tones=True, eof_marker=True)
        for index in range(3)]).astype(np.float32)
    reverse = wire[::-1].copy()
    window_start = v7.PULSE_FRAME-64
    forward_window = wire[window_start:window_start+v7.PULSE_FRAME+128]
    reverse_window = reverse[window_start:window_start+v7.PULSE_FRAME+128]

    forward_baseline_state = v7.PulseState()
    forward_auto_state = v7.PulseState()
    reverse_state = v7.PulseState()

    def decode_forward(state, hit):
        state.set_playback_direction(1)
        results, _ = v7.decode_pulse_stream(
            model, forward_window, latest_only=True, sample_rate=v7.RATE,
            pilot_timing='tone-seeded', frame_boundary='eof',
            pulse_starts=(hit,), state=state)
        return len(results)

    def forward_only():
        hits = v7.pulse_frame_starts(forward_window)
        if not hits:
            return 0
        return decode_forward(forward_baseline_state, hits[0])

    def forward_auto():
        hits = v7.pulse_frame_hits(forward_window, direction='auto')
        if not hits:
            return 0
        return decode_forward(forward_auto_state, hits[0][:3])

    def reverse_auto():
        hits = v7.pulse_frame_hits(reverse_window, direction='auto')
        if not hits:
            return 0
        hit = hits[0]
        results, _ = v7.decode_reverse_packet(
            model, reverse_window, hit[0], hit[1], state=reverse_state,
            sample_rate=v7.RATE, pilot_timing='tone-seeded')
        return len(results)

    # Compile the acquisition signatures and receiver paths before timing.
    v7.PULSE.warmup_pulse_kernels()
    v7.warmup_equalizer(model)
    if not all(function() == 1 for function in
               (forward_only, forward_auto, reverse_auto)):
        raise RuntimeError('benchmark packet did not decode in all paths')

    old, automatic = _paired(forward_only, forward_auto, repeats)
    automatic_forward, automatic_reverse = _paired(
        forward_auto, reverse_auto, repeats)
    detect_delta = (automatic['median_cpu_ms_per_packet'] /
                    old['median_cpu_ms_per_packet']-1)*100
    reverse_delta = (automatic_reverse['median_cpu_ms_per_packet'] /
                     automatic_forward['median_cpu_ms_per_packet']-1)*100
    print(json.dumps({
        'fixture': str(v7.REFERENCE_FIXTURE.relative_to(ROOT)),
        'window_samples': len(forward_window),
        'paired_batches': repeats,
        'packets_per_batch': 5,
        'forward_only_detect_plus_decode': old,
        'automatic_forward_detect_plus_decode': automatic_forward,
        'reverse_auto_detect_plus_decode': automatic_reverse,
        'automatic_forward_from_detection_comparison': automatic,
        'automatic_detection_delta_pct': round(detect_delta, 2),
        'reverse_decode_delta_pct': round(reverse_delta, 2),
        'targets_pct': {'automatic_detection_max': 3.0,
                        'reverse_decode_max': 5.0},
        'targets_met': detect_delta <= 3.0 and reverse_delta <= 5.0,
    }, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=300,
                        help='paired timed runs after warmup (default: 300)')
    args = parser.parse_args(argv)
    if args.repeats < 10:
        parser.error('--repeats must be at least 10')
    run(args.repeats)


if __name__ == '__main__':
    main()
