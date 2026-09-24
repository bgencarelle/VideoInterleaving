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
import time
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
    fast_flutter_25: float = 0
    fast_flutter_60: float = 0
    mains_buzz_dbfs: float | None = None
    mono_sum: bool = False
    one_leg_only: bool = False


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
    Case('fast-flutter', wow=.45, flutter=.12,
         fast_flutter_25=.001, fast_flutter_60=.0005),
    Case('mains-buzz', mains_buzz_dbfs=-26),
    Case('highpass-300', highpass=300),
    Case('lowpass-4k', lowpass=4000),
    Case('mono-sum', mono_sum=True),
    Case('one-leg-only', one_leg_only=True),
)


def delay_channel(x, delay):
    at = np.arange(len(x), dtype=float) - delay
    return np.interp(at, np.arange(len(x)), x, left=0, right=0)


def time_warp(x, case):
    if not case.wow and not case.flutter:
        return x
    t = np.arange(len(x)) / RATE
    speed = ((case.wow / 100) * np.sin(2*np.pi*.55*t + .4) +
             (case.flutter / 100) * np.sin(2*np.pi*7.3*t + 1.1) +
             case.fast_flutter_25*np.sin(2*np.pi*25*t) +
             case.fast_flutter_60*np.sin(2*np.pi*60*t+.4))
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
    if case.mains_buzz_dbfs is not None:
        buzz = np.zeros(len(x))
        for fundamental in (50, 60):
            harmonic = 1
            while fundamental*harmonic <= 1500:
                phase = 2*np.pi*((harmonic*13+fundamental) % 97)/97
                buzz += np.sin(2*np.pi*fundamental*harmonic*t+phase)/harmonic
                harmonic += 1
        buzz *= (10**(case.mains_buzz_dbfs/20) /
                 max(float(np.sqrt(np.mean(buzz*buzz))), 1e-12))
        x += buzz[:, None]
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
    if case.mono_sum:
        x = ((x[:, 0]+x[:, 1])/np.sqrt(2))[:, None]
    elif case.one_leg_only:
        x[:, 1] = 0
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


def image_quality_by_frame(rows):
    output = []
    for frame, values in rows:
        planes = {
            name: {
                'psnr_db': float(metrics[0]),
                'ssim': float(metrics[1]),
                'mae': float(metrics[2]),
                'normalized_rmse': float(metrics[3]),
            }
            for name, metrics in zip(('Y', 'Cb', 'Cr'), values)
        }
        planes['overall'] = {
            key: float(np.mean([planes[name][key]
                                for name in ('Y', 'Cb', 'Cr')]))
            for key in ('psnr_db', 'ssim', 'mae', 'normalized_rmse')
        }
        output.append({'frame': int(frame), **planes})
    return output


def write_difference_maps(out, case_name, reference_frames, candidate_frames):
    """Save amplified per-plane differences for paired decoder outputs.

    Red, green, and blue encode 8x absolute Y, Cb, and Cr differences,
    respectively. The maps are diagnostic visualizations, not rendered images.
    """
    directory = out/'diffmaps'/case_name
    directory.mkdir(parents=True, exist_ok=True)
    for frame in sorted(reference_frames.keys() & candidate_frames.keys()):
        reference = Image.fromarray(reference_frames[frame]).convert('YCbCr')
        candidate = Image.fromarray(candidate_frames[frame]).convert('YCbCr')
        delta = np.abs(np.asarray(candidate, dtype=np.int16)-
                       np.asarray(reference, dtype=np.int16))
        amplified = np.clip(delta*8, 0, 255).astype(np.uint8)
        Image.fromarray(amplified, mode='RGB').save(
            directory/f'frame_{frame:04d}.png')


def paired_frame_quality(reference_rows, candidate_rows):
    reference = {row['frame']: row for row in reference_rows}
    candidate = {row['frame']: row for row in candidate_rows}
    frames = sorted(reference.keys() & candidate.keys())
    deltas = [
        {
            'frame': int(frame),
            **{plane: (candidate[frame][plane]['psnr_db']-
                       reference[frame][plane]['psnr_db'])
               for plane in ('Y', 'Cb', 'Cr')},
        }
        for frame in frames
    ]
    summary = {
        plane: {
            'mean_psnr_delta_db': float(np.mean(
                [row[plane] for row in deltas])) if deltas else None,
            'median_psnr_delta_db': float(np.median(
                [row[plane] for row in deltas])) if deltas else None,
            'min_psnr_delta_db': float(np.min(
                [row[plane] for row in deltas])) if deltas else None,
            'frames_below_minus_0_05_db': sum(
                row[plane] < -.05 for row in deltas),
        }
        for plane in ('Y', 'Cb', 'Cr')
    }
    return {'frames_compared': len(deltas), 'planes': summary}, deltas


def _decode_pilot_variant(model, reference, values, audio, mode,
                          force_float32=False, measure_speed=True,
                          retain_images=False):
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    results, info = v7.decode_pulse_stream(
        model, audio, force_float32=force_float32, sample_rate=RATE,
        pilot_timing=mode)
    cpu_elapsed = time.process_time()-cpu_started
    wall_elapsed = time.perf_counter()-wall_started
    errors, quality_rows, frame_quality = [], [], []
    frame_images = {}
    pilot_residuals, timing_residuals, tone_fit_residuals = [], [], []
    tone_snrs, tone_speeds = [], []
    tone_detected = tone_speed_locked = tone_applied = 0
    metadata_retries = metadata_recovered = 0
    tone_rejection_reasons = {}
    dual_tone_disagreements = []
    foundation_confidence, foundation_coverage, foundation_noise = [], [], []
    for result in results:
        decoded = v7.values_from(model, result.coeffs)
        decoded_image = values_image(decoded, model.coder.grids)
        image_metrics = plane_metrics_arrays(reference, decoded_image)
        quality_rows.append(image_metrics)
        frame_quality.append((result.counter, image_metrics))
        if retain_images:
            if isinstance(decoded_image, Image.Image):
                image = np.asarray(decoded_image.convert('RGB'), dtype=np.uint8)
            else:
                image = np.asarray(decoded_image, dtype=np.uint8)
                if image.ndim == 2:
                    image = np.repeat(image[..., None], 3, axis=2)
                elif image.shape[-1] == 4:
                    image = image[..., :3]
            frame_images[result.counter] = image.copy()
        if result.status != 'lost':
            errors.append(float(np.sqrt(np.mean((decoded-values)**2))))
        if result.diag.get('head_confidence') is not None:
            foundation_confidence.append(
                float(result.diag['head_confidence']))
        if result.diag.get('head_coverage') is not None:
            foundation_coverage.append(float(result.diag['head_coverage']))
        if result.diag.get('noise') is not None:
            foundation_noise.append(float(np.max(result.diag['noise'])))
        retry = result.diag.get('metadata_pilot_retry', {})
        metadata_retries += int(bool(retry.get('retried')))
        metadata_recovered += int(bool(retry.get('valid')))
        residual = result.diag.get('pilot_residual')
        if residual is not None:
            pilot_residuals.append(float(residual))
        timing = result.diag.get('pilot_timing', {})
        if timing.get('detected'):
            tone_detected += 1
            tone_snrs.extend(float(snr) for snr in timing.get('tone_snr_db', []))
        elif timing.get('reason'):
            reason = str(timing['reason'])
            tone_rejection_reasons[reason] = (
                tone_rejection_reasons.get(reason, 0)+1)
        disagreement = timing.get('dual_tone_timing_disagreement_samples')
        if disagreement is not None:
            dual_tone_disagreements.append(float(disagreement))
        if timing.get('mode_applied') == mode:
            tone_applied += 1
        timing_residual = result.diag.get('timing_residual_samples')
        if timing_residual is not None:
            timing_residuals.append(float(timing_residual))
        tone_fit_residual = timing.get('timing_sample_residual_rms')
        if tone_fit_residual is None:
            tone_fit_residual = timing.get('timing_residual_rms')
        if tone_fit_residual is not None:
            tone_fit_residuals.append(float(tone_fit_residual))
        if measure_speed and result.diag.get('frame_start') is not None:
            speed = v7.pilot_tone_speed(
                audio, RATE, result.diag['frame_start'],
                result.diag['frame_scale'])
            tone_snrs.extend(float(snr)
                             for snr in speed.get('tone_snr_db', []))
            if speed.get('detected'):
                tone_speed_locked += 1
                tone_speeds.append(float(speed['difference_pct']))
    expected_frames = len(results)
    return {
        'frames': expected_frames,
        'received': sum(result.status != 'lost' for result in results),
        'lost': sum(result.status == 'lost' for result in results),
        'metadata_valid': sum(bool(result.diag.get('metadata_valid'))
                              for result in results),
        'displayable': sum(bool(result.diag.get('displayable'))
                           for result in results),
        'mean_foundation_confidence': (
            float(np.mean(foundation_confidence))
            if foundation_confidence else None),
        'mean_foundation_coverage': (
            float(np.mean(foundation_coverage))
            if foundation_coverage else None),
        'mean_max_pilot_noise': (float(np.mean(foundation_noise))
                                 if foundation_noise else None),
        'metadata_retries': metadata_retries,
        'metadata_retry_recovered': metadata_recovered,
        'mean_rmse_received': float(np.mean(errors)) if errors else None,
        'quality': image_quality(quality_rows) if quality_rows else None,
        'quality_by_frame': image_quality_by_frame(frame_quality),
        'mean_pilot_residual': (float(np.mean(pilot_residuals))
                                if pilot_residuals else None),
        'mean_timing_residual_samples': (
            float(np.mean(timing_residuals)) if timing_residuals else None),
        'mean_tone_fit_residual_samples': (
            float(np.mean(tone_fit_residuals)) if tone_fit_residuals else None),
        'tone_detected_frames': tone_detected,
        'tone_speed_locked_frames': tone_speed_locked,
        'tone_applied_frames': tone_applied,
        'tone_rejection_reasons': tone_rejection_reasons,
        'mean_dual_tone_disagreement_samples': (
            float(np.mean(dual_tone_disagreements))
            if dual_tone_disagreements else None),
        'mean_tone_snr_db': float(np.mean(tone_snrs)) if tone_snrs else None,
        'mean_tone_speed_difference_pct': (
            float(np.mean(tone_speeds)) if tone_speeds else None),
        'max_abs_tone_speed_difference_pct': (
            float(np.max(np.abs(tone_speeds))) if tone_speeds else None),
        'cpu_elapsed_s': cpu_elapsed,
        'wall_elapsed_s': wall_elapsed,
        'ms_per_frame': 1000*cpu_elapsed/max(expected_frames, 1),
        'wall_ms_per_frame': 1000*wall_elapsed/max(expected_frames, 1),
        'recovered': bool(info.get('recovered')),
        '_frame_images': frame_images,
    }


def _waveform_metrics(audio):
    audio = np.asarray(audio)
    return {
        'peak': float(np.max(np.abs(audio))) if audio.size else 0.0,
        'rms': float(np.sqrt(np.mean(audio*audio))) if audio.size else 0.0,
        'samples_above_0_89': int(np.count_nonzero(np.abs(audio) > .89)),
        'samples_at_pcm_clip': int(np.count_nonzero(np.abs(audio) >= 1.0)),
        'limiter_packets_at_0_89': int(np.any(
            np.abs(audio).reshape(-1, v7.PULSE_FRAME, audio.shape[-1]) > .89,
            axis=(1, 2)).sum()) if audio.ndim == 2 and
            len(audio) % v7.PULSE_FRAME == 0 else 0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT,
                        help='output directory (default: tmp/v7-torture-current)')
    parser.add_argument('--frames', type=int, default=None,
                        help='identical V7 frames to generate (default: 12; '
                             'pilot A/B defaults to 40)')
    parser.add_argument('--pilot-ab', action='store_true',
                        help='40-packet tones-off/on and receiver-mode A/B')
    parser.add_argument('--diffmaps', action='store_true',
                        help='save paired tone-on baseline/joint per-plane '
                             'difference maps for representative cases')
    parser.add_argument('--seed', type=int, default=2026,
                        help='repeatable synthetic-noise seed (default: 2026)')
    parser.add_argument('--only', action='append', default=[],
                        help='run only this case; may be repeated')
    parser.add_argument('--force-float32', action='store_true',
                        help='use the experimental float32/complex64 decoder')
    args = parser.parse_args(argv)
    args.frames = args.frames if args.frames is not None else (
        40 if args.pilot_ab else DEFAULT_FRAMES)
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
    if args.pilot_ab:
        audio48_tones = v7.encode_pulse_stream(
            model, [values]*args.frames, 1, [0]*args.frames,
            pilot_tones=True)
        wire96_tones = resample_poly(
            audio48_tones, 2, 1, axis=0).astype(np.float32)
        variants = (
            ('off/baseline', wire96, 'baseline'),
            ('on/baseline', wire96_tones, 'baseline'),
            ('on/tone-seeded', wire96_tones, 'tone-seeded'),
            ('on/tone-joint', wire96_tones, 'tone-joint'),
            ('on/tone-replaced', wire96_tones, 'tone-replaced'),
        )
        rows = []
        failures = []
        print('packet tone metrics:', json.dumps({
            'off': _waveform_metrics(audio48),
            'on': _waveform_metrics(audio48_tones),
        }), flush=True)
        for case in CASES:
            if selected and case.name not in selected:
                continue
            row = {'case': case.name, 'expected_frames': args.frames-1,
                   'results': {}}
            damaged_by_wire = {}
            retained_images = {}
            diffmap_cases = (selected if selected else {
                'fast-flutter', 'lowpass-4k', 'type-i', 'mains-buzz'})
            for name, wire, mode in variants:
                key = 'on' if name.startswith('on/') else 'off'
                if key not in damaged_by_wire:
                    damaged_by_wire[key] = impair(wire, case, seed=args.seed)
                result = _decode_pilot_variant(
                    model, reference, values, damaged_by_wire[key], mode,
                    force_float32=args.force_float32,
                    measure_speed=key == 'on',
                    retain_images=(args.diffmaps and
                                   case.name in diffmap_cases and
                                   name in ('on/baseline', 'on/tone-joint')))
                retained_images[name] = result.pop('_frame_images', {})
                row['results'][name] = result
            if args.diffmaps and case.name in diffmap_cases:
                write_difference_maps(
                    out, case.name, retained_images['on/baseline'],
                    retained_images['on/tone-joint'])
            # Decoder A/B is paired on the identical tone-on waveform. The
            # tones-off result remains useful as a separate transmitter A/B,
            # but must not be mixed into decoder acceptance.
            base = row['results']['on/baseline']
            candidate = row['results']['on/tone-joint']
            plane_psnr_delta = {}
            if base['quality'] and candidate['quality']:
                plane_psnr_delta = {
                    plane: (candidate['quality'][plane]['psnr_db']-
                            base['quality'][plane]['psnr_db'])
                    for plane in ('Y', 'Cb', 'Cr')
                }
            frame_quality_summary, frame_quality_delta = paired_frame_quality(
                base['quality_by_frame'], candidate['quality_by_frame'])
            rx_ok = candidate['received'] >= base['received']
            rmse_ok = (candidate['mean_rmse_received'] is None or
                       base['mean_rmse_received'] is None or
                       candidate['mean_rmse_received'] <=
                       base['mean_rmse_received']+.0005)
            psnr_ok = (not plane_psnr_delta or
                       all(delta >= -.05
                           for delta in plane_psnr_delta.values()))
            foundation_ok = True
            for metric, better in (
                    ('mean_foundation_confidence', 'higher'),
                    ('mean_foundation_coverage', 'higher'),
                    ('mean_max_pilot_noise', 'lower')):
                before, after = base[metric], candidate[metric]
                if before is None or after is None:
                    continue
                if better == 'higher':
                    foundation_ok &= after >= before-.001
                else:
                    foundation_ok &= after <= before*1.01+1e-5
            cpu_delta = (candidate['ms_per_frame']-
                         base['ms_per_frame'])
            row['acceptance'] = {
                'received_not_worse': rx_ok,
                'rmse_not_worse': rmse_ok,
                'per_plane_psnr_delta_db': plane_psnr_delta,
                'per_plane_psnr_not_worse': psnr_ok,
                'foundation_not_worse': bool(foundation_ok),
                'cpu_delta_ms_per_frame': cpu_delta,
                'cpu_within_0_2ms': cpu_delta <= .2,
                'frame_quality_delta_summary': frame_quality_summary,
            }
            row['frame_quality_delta_db'] = frame_quality_delta
            if case.name in ('fast-flutter', 'wow-flutter'):
                before_timing = base['mean_timing_residual_samples']
                after_timing = candidate['mean_timing_residual_samples']
                row['acceptance']['flutter_timing_improved'] = bool(
                    before_timing is not None and after_timing is not None and
                    after_timing < before_timing)
            flutter_ok = row['acceptance'].get(
                'flutter_timing_improved', True)
            if not (rx_ok and rmse_ok and psnr_ok and
                    cpu_delta <= .2 and flutter_ok):
                failures.append(case.name)
            rows.append(row)
            summary = {
                'case': case.name,
                'acceptance': row['acceptance'],
                'results': {
                    name: {
                        'received': result['received'],
                        'lost': result['lost'],
                        'metadata_valid': result['metadata_valid'],
                        'metadata_retries': result['metadata_retries'],
                        'metadata_retry_recovered': result[
                            'metadata_retry_recovered'],
                        'rmse': result['mean_rmse_received'],
                        'psnr_y_cb_cr': (None if result['quality'] is None else
                                         [result['quality'][plane]['psnr_db']
                                          for plane in ('Y', 'Cb', 'Cr')]),
                        'pilot_residual': result['mean_pilot_residual'],
                        'timing_residual': result[
                            'mean_timing_residual_samples'],
                        'tone_fit_residual': result[
                            'mean_tone_fit_residual_samples'],
                        'foundation_confidence': result[
                            'mean_foundation_confidence'],
                        'foundation_coverage': result[
                            'mean_foundation_coverage'],
                        'max_pilot_noise': result['mean_max_pilot_noise'],
                        'tone_lock_frames': result['tone_detected_frames'],
                        'timing_applied_frames': result['tone_applied_frames'],
                        'tone_rejection_reasons': result[
                            'tone_rejection_reasons'],
                        'dual_tone_disagreement_samples': result[
                            'mean_dual_tone_disagreement_samples'],
                        'tone_speed_lock_frames': result[
                            'tone_speed_locked_frames'],
                        'tone_snr_db': result['mean_tone_snr_db'],
                        'tone_speed_diff_pct': result[
                            'mean_tone_speed_difference_pct'],
                        'max_abs_tone_speed_diff_pct': result[
                            'max_abs_tone_speed_difference_pct'],
                        'cpu_ms_per_frame': result['ms_per_frame'],
                        'wall_ms_per_frame': result['wall_ms_per_frame'],
                    }
                    for name, result in row['results'].items()
                },
            }
            print(json.dumps(summary), flush=True)
        (out/'results.json').write_text(json.dumps(rows, indent=2)+'\n')
        if failures:
            print('PILOT A/B acceptance failures: '+', '.join(failures),
                  file=sys.stderr)
            return 1
        print(f'PASS: {len(rows)} cases, {args.frames} packets each')
        return 0
    rows = []
    cases = [case for case in CASES if not selected or case.name in selected]
    failures = []
    for case in cases:
        damaged = impair(wire96, case, seed=args.seed)
        results, info = v7.decode_pulse_stream(
            model, damaged, force_float32=args.force_float32,
            sample_rate=RATE)
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
