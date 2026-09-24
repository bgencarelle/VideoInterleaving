#!/usr/bin/env python3
"""Current V7 wire-only synthetic tape torture matrix.

This reconstructs the historical 19-case matrix, excluding worn-deck. It is
deliberately a synthetic regression matrix, not a claim to model a particular
deck; real tape captures remain authoritative. Outputs go under tmp/.
"""
import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import butter, resample_poly, sosfilt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from animation_modem import v7
from animation_modem.imaging import values_image
from tools.measure_plane_survival import plane_metrics_arrays

RATE = 96_000
DEFAULT_FRAMES = 12
DEFAULT_OUT = Path('tmp/v7-torture-current')
FIXTURE = v7.REFERENCE_FIXTURE
TARGET = .1521 / np.sqrt(1 + 10**(v7.CLOCK_REL_DB / 10))


@dataclass(frozen=True)
class Case:
    name: str
    lowpass: float = 0
    highpass: float = 0
    noise_dbfs: float | None = None
    wow: float = 0
    flutter: float = 0
    azimuth_us: float = 0
    crosstalk: float = 0
    right_gain_db: float = 0
    dc: float = 0
    hum_dbfs: float | None = None
    bias_hz: float = 0
    bias_dbfs: float | None = None
    saturation: float = 0
    dropout_ms: float = 0
    dropout_every: float = 0
    nr_pump: float = 0


CASES = (
    Case('clean-96k'),
    Case('lowpass-18k', lowpass=18000),
    Case('lowpass-15k', lowpass=15000),
    Case('lowpass-12k', lowpass=12000),
    Case('lowpass-10k', lowpass=10000),
    Case('hiss-45', noise_dbfs=-45),
    Case('hiss-40', noise_dbfs=-40),
    Case('hiss-35', noise_dbfs=-35),
    Case('wow-flutter', wow=.45, flutter=.12),
    Case('azimuth-12us', azimuth_us=12),
    Case('crosstalk-10pct', crosstalk=.10),
    Case('right-minus-4db', right_gain_db=-4),
    Case('dc-hum', dc=.025, hum_dbfs=-38),
    Case('bias-leak-30k', bias_hz=30000, bias_dbfs=-30),
    Case('soft-saturation', saturation=2.0),
    Case('dropouts', dropout_ms=12, dropout_every=.7),
    Case('nr-pumping', nr_pump=.55),
    Case('type-i', highpass=70, lowpass=12000, noise_dbfs=-39,
         wow=.35, flutter=.10, azimuth_us=7, crosstalk=.08,
         right_gain_db=-2, saturation=1.35),
    Case('type-ii', highpass=50, lowpass=16000, noise_dbfs=-44,
         wow=.22, flutter=.07, azimuth_us=4, crosstalk=.05,
         right_gain_db=-1, saturation=1.15),
)


def delay_channel(x, delay):
    at = np.arange(len(x), dtype=float) - delay
    return np.interp(at, np.arange(len(x)), x, left=0, right=0)


def time_warp(x, case):
    if not case.wow and not case.flutter:
        return x
    t = np.arange(len(x)) / RATE
    speed = ((case.wow / 100) * np.sin(2*np.pi*.55*t + .4) +
             (case.flutter / 100) * np.sin(2*np.pi*7.3*t + 1.1))
    position = np.arange(len(x), dtype=float) + np.cumsum(speed)
    base = np.arange(len(x), dtype=float)
    return np.column_stack([np.interp(position, base, x[:, c])
                            for c in range(2)])


def tape_noise(count, dbfs, rng):
    noise = rng.standard_normal((count, 2))
    high = sosfilt(butter(2, 1800, btype='highpass', fs=RATE, output='sos'),
                   noise, axis=0)
    high /= max(np.sqrt(np.mean(high * high)), 1e-12)
    return high * 10**(dbfs / 20)


def nr_pump(x, amount):
    low = sosfilt(butter(2, 2500, fs=RATE, output='sos'), x, axis=0)
    high = x - low
    envelope = np.maximum(np.abs(low[:, 0]), np.abs(low[:, 1]))
    alpha = math.exp(-1 / (.060 * RATE))
    tracked = np.empty_like(envelope)
    state = 0.0
    for i, value in enumerate(envelope):
        state = max(value, alpha * state + (1 - alpha) * value)
        tracked[i] = state
    reference = max(np.percentile(tracked, 90), 1e-6)
    gain = 1 - amount * (1 - np.clip(tracked / reference, 0, 1))
    return low + high * gain[:, None]


def impair(source, case, seed=2026):
    rng = np.random.default_rng(seed)
    x = np.asarray(source, float).copy()
    x[:, 1] *= 10**(case.right_gain_db / 20)
    if case.crosstalk:
        left, right = x[:, 0].copy(), x[:, 1].copy()
        x[:, 0] = (1-case.crosstalk)*left + case.crosstalk*right
        x[:, 1] = case.crosstalk*left + (1-case.crosstalk)*right
    x = time_warp(x, case)
    if case.azimuth_us:
        x[:, 1] = delay_channel(x[:, 1], case.azimuth_us * RATE / 1e6)
    filters = []
    if case.highpass:
        filters.append(butter(4, case.highpass, btype='highpass', fs=RATE, output='sos'))
    if case.lowpass:
        filters.append(butter(4, case.lowpass, fs=RATE, output='sos'))
    if filters:
        x = sosfilt(np.concatenate(filters), x, axis=0)
    if case.nr_pump:
        x = nr_pump(x, case.nr_pump)
    t = np.arange(len(x)) / RATE
    if case.dc:
        x += case.dc
    if case.hum_dbfs is not None:
        x += (10**(case.hum_dbfs / 20) * np.sin(2*np.pi*60*t + .2))[:, None]
    if case.bias_hz and case.bias_dbfs is not None:
        x += (10**(case.bias_dbfs / 20) *
              np.sin(2*np.pi*case.bias_hz*t))[:, None]
    if case.noise_dbfs is not None:
        x += tape_noise(len(x), case.noise_dbfs, rng)
    if case.saturation:
        x = np.tanh(case.saturation*x) / np.tanh(case.saturation)
    if case.dropout_ms and case.dropout_every:
        duration = max(1, round(case.dropout_ms * RATE / 1000))
        period = max(duration + 1, round(case.dropout_every * RATE))
        fade = min(duration // 3, round(.002 * RATE))
        for start in range(period, len(x), period):
            stop = min(len(x), start + duration)
            x[start:stop] = 0
            if fade:
                before = max(0, start-fade)
                x[before:start] *= np.linspace(1, 0, start-before)[:, None]
                after = min(len(x), stop+fade)
                x[stop:after] *= np.linspace(0, 1, after-stop)[:, None]
    return np.clip(x, -1, 1).astype(np.float32)


def image_quality(rows):
    values = np.asarray(rows, float)
    return {
        name: {
            'psnr_db': float(values[:, index, 0].mean()),
            'ssim': float(values[:, index, 1].mean()),
            'mae': float(values[:, index, 2].mean()),
            'normalized_rmse': float(values[:, index, 3].mean()),
        }
        for name, index in zip(('Y', 'Cb', 'Cr'), range(3))
    } | {
        'overall': {
            'psnr_db': float(values[:, :, 0].mean()),
            'ssim': float(values[:, :, 1].mean()),
            'mae': float(values[:, :, 2].mean()),
            'normalized_rmse': float(values[:, :, 3].mean()),
        }
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT,
                        help='output directory (default: tmp/v7-torture-current)')
    parser.add_argument('--frames', type=int, default=DEFAULT_FRAMES,
                        help='identical V7 frames to generate (default: 12)')
    parser.add_argument('--seed', type=int, default=2026,
                        help='repeatable synthetic-noise seed (default: 2026)')
    parser.add_argument('--only', action='append', default=[],
                        help='run only this case; may be repeated')
    parser.add_argument('--force-float32', action='store_true',
                        help='use the experimental float32/complex64 decoder')
    args = parser.parse_args(argv)
    if args.frames < 3:
        parser.error('--frames must be at least 3')
    selected = set(args.only)
    known = {case.name for case in CASES}
    unknown = selected - known
    if unknown:
        parser.error('unknown case(s): ' + ', '.join(sorted(unknown)))

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    model = v7.load_model(TARGET, 'nearest')
    with Image.open(FIXTURE) as image:
        reference = v7.prepare_image(image, 'nearest')
        values = v7.image_values(reference,
                                 model.coder.grids, 'nearest')
    audio48 = v7.encode_pulse_stream(model, [values] * args.frames, 1,
                                     [0] * args.frames)
    wire96 = resample_poly(audio48, 2, 1, axis=0).astype(np.float32)
    rows = []
    cases = [case for case in CASES if not selected or case.name in selected]
    failures = []
    for case in cases:
        damaged = impair(wire96, case, seed=args.seed)
        results, info = v7.decode_pulse_stream(
            model, damaged, force_float32=args.force_float32)
        errors = []
        quality_rows = []
        metadata = 0
        displayable = 0
        for result in results:
            metadata += int(result.diag.get('metadata_valid', False))
            displayable += int(result.diag.get('displayable', False))
            decoded = v7.values_from(model, result.coeffs)
            quality_rows.append(plane_metrics_arrays(
                reference, values_image(decoded, model.coder.grids)))
            if result.status != 'lost':
                errors.append(float(np.sqrt(np.mean((decoded - values)**2))))
        row = {
            'case': case.name,
            'input_samples': len(damaged),
            'decoded_frames': len(results),
            'received': sum(r.status != 'lost' for r in results),
            'lost': sum(r.status == 'lost' for r in results),
            'metadata_valid': metadata,
            'displayable': displayable,
            'mean_rmse': float(np.mean(errors)) if errors else None,
            'max_rmse': float(np.max(errors)) if errors else None,
            'image_quality': image_quality(quality_rows) if quality_rows else None,
            'recovered': bool(info.get('recovered')),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        expected = args.frames - 1  # the last packet has no following header
        if (row['decoded_frames'] != expected or
                row['metadata_valid'] != expected or
                row['displayable'] != expected):
            failures.append(case.name)
    (out / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
    if failures:
        print('FAILED cases: ' + ', '.join(failures), file=sys.stderr)
        return 1
    print(f'PASS: {len(rows)} cases, {args.frames} packets each')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
