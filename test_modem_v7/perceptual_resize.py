#!/usr/bin/env python3
"""Compare V7 sender resize ablations on real images_sbs source images.

All scoring inputs are held out from the frozen canonical box model and pinned
fold tables. The clean folded decode below is a coefficient-perfect baseline,
not a simulated tape or audio-loopback result. Send cost includes `_values`
preparation and coefficient folding; Numba compilation is reported separately.

    .venv/bin/python test_modem_v7/perceptual_resize.py \
        --frames images_sbs/face/00_C_BG_faceSource_960/a.jpg ...
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import numba
from PIL import Image

from common import (DISPLAY, REPO, reference, score)
from animation_modem.imaging import values_image

if str(REPO/'tools') not in sys.path:
    sys.path.insert(0, str(REPO/'tools'))
if str(REPO/'test_modem_v7') not in sys.path:
    sys.path.insert(0, str(REPO/'test_modem_v7'))

from animation_modem import v7
from live_fold import LiveFold
from tools import v7_live
from utilities.convert_to_xy import load_rgba
from animation_modem.perceptual_resize import warmup_resize
from tools.v7_torture_matrix import TARGET


def _quantiles(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    if not values.size:
        return {key: None for key in ('p50', 'p90', 'p95', 'p99', 'max')}
    q = np.percentile(values, (50, 90, 95, 99, 100))
    return dict(zip(('p50', 'p90', 'p95', 'p99', 'max'),
                    (float(value) for value in q)))


def _coefficient_statistics(model, coefficients):
    rank = np.empty(len(model.order), dtype=np.int32)
    rank[model.order] = np.arange(len(model.order), dtype=np.int32)
    plane_offsets = np.cumsum([0, *(r*c for r, c in v7.V7_SHAPES)])
    tier_ranges = (('head', 0, v7.HEAD),
                   ('body', v7.HEAD, v7.BODY_END),
                   ('tail', v7.BODY_END, len(model.order)))
    groups = {}
    flat_indices = np.arange(len(model.order))
    actual_mean = np.mean(coefficients, axis=0)
    actual_variance = np.var(coefficients, axis=0)
    for plane, start, end in ((p, int(plane_offsets[p]), int(plane_offsets[p+1]))
                              for p in range(len(v7.V7_SHAPES))):
        for tier, rank_start, rank_end in tier_ranges:
            members = flat_indices[(flat_indices >= start) & (flat_indices < end) &
                                   (rank >= rank_start) & (rank < rank_end)]
            if not len(members):
                continue
            mean_departure = np.abs(actual_mean[members]-model.mu[members]) / np.sqrt(
                np.maximum(model.lam[members], 1e-15))
            variance_ratio = actual_variance[members] / np.maximum(
                model.lam[members], 1e-15)
            mean_worst = int(members[np.argmax(mean_departure)])
            variance_worst = int(members[np.argmax(np.abs(np.log(
                np.maximum(variance_ratio, 1e-15))))])
            groups[f'{("Y", "Cb", "Cr")[plane]}_{tier}'] = {
                'coefficient_count': int(len(members)),
                'absolute_mean_departure_sigma': _quantiles(mean_departure),
                'variance_to_model_lam': _quantiles(variance_ratio),
                'largest_mean_departure': {
                    'coefficient': mean_worst,
                    'rank': int(rank[mean_worst]),
                    'departure_sigma': float(mean_departure[np.argmax(mean_departure)]),
                },
                'largest_variance_departure': {
                    'coefficient': variance_worst,
                    'rank': int(rank[variance_worst]),
                    'variance_ratio': float(actual_variance[variance_worst] /
                                            max(model.lam[variance_worst], 1e-15)),
                },
            }
    return groups


def _fold_statistics(model, codec, values, coefficients):
    packets = len(values)
    kept = coefficients[:, codec.kept]
    host_norm = ((kept[:, codec.hosts]-model.mu[codec.hosts]) /
                 codec.sd_host).ravel()
    signature = int(codec.signature)
    picture_guest_count = codec.M-signature
    guest_norm = (coefficients[:, codec.guests[:picture_guest_count]] /
                  codec.sd_guest[:picture_guest_count]).ravel()
    tail_energy, tail_slot_count = 0.0, 0
    rank = np.empty(len(model.order), dtype=np.int32)
    rank[model.order] = np.arange(len(model.order), dtype=np.int32)
    for frame_number in range(packets):
        ranks = np.asarray(model.rank_tables[(frame_number+1) % v7.TAIL_PHASES])
        valid = ranks >= 0
        tail = ranks[valid & (rank[np.maximum(ranks, 0)] >= v7.BODY_END)]
        if tail.size:
            amplitude = model.gain[tail]*(kept[frame_number, tail]-model.mu[tail])
            tail_energy += float(np.sum(amplitude*amplitude))
            tail_slot_count += int(tail.size)

    unfolded_packet_count = 0
    unfolded_picture_slots = 0
    decoded = []
    for frame_number, value in enumerate(values):
        slot_coefficients = codec.encode_coefficients(value)
        xhat = slot_coefficients-model.mu
        full = codec.decode(slot_coefficients, xhat,
                            np.ones(len(slot_coefficients)),
                            metadata_confirmed=True)
        picture_slots = max(0, int(codec.last_unfolded_slots)-signature)
        unfolded_picture_slots += picture_slots
        unfolded_packet_count += int(picture_slots > 0)
        decoded.append(codec.grid.inverse(full))

    data = {
        'packet_count': packets,
        'unfolded_packet_count': unfolded_packet_count,
        'unfolded_packet_rate': (unfolded_packet_count/packets if packets else None),
        'picture_slot_denominator': int(packets*picture_guest_count),
        'unfolded_picture_slot_count': unfolded_picture_slots,
        'unfolded_picture_slot_rate': (
            unfolded_picture_slots/(packets*picture_guest_count)
            if packets and picture_guest_count else None),
        'signature_slots_excluded_per_packet': signature,
        'host_normalized_magnitude': _quantiles(np.abs(host_norm)),
        'guest_normalized_magnitude_before_clip': _quantiles(np.abs(guest_norm)),
        'guest_fraction_above_2_sigma': float(np.mean(np.abs(guest_norm) > 2.0)),
        'guest_fraction_above_2_5_sigma': float(np.mean(np.abs(guest_norm) > 2.5)),
        'tail_transmitted_slot_count': tail_slot_count,
        'tail_energy_sum_per_packet': tail_energy/packets if packets else None,
        'tail_mean_slot_power': tail_energy/tail_slot_count if tail_slot_count else None,
    }
    return data, decoded


def _settings(strengths):
    result = [('off', None), ('linear-box', None)]
    for mode in ('gamma-detail', 'linear-detail'):
        result.extend((mode, float(strength)) for strength in strengths)
    return result


def _setting_name(mode, strength):
    return mode if strength is None else f'{mode}-s{strength:g}'


def _cpu_model():
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            key, separator, value = line.partition(':')
            if separator and key.strip().lower() in ('model name', 'hardware'):
                return value.strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def run(args):
    source_root = (REPO/'images_sbs').resolve()
    frames = []
    for path in args.frames:
        resolved = Path(path).resolve()
        try:
            resolved.relative_to(source_root)
        except ValueError as exc:
            raise ValueError(f'input must be a real source image under images_sbs/: {path}') from exc
        rgb, _alpha = load_rgba(resolved)
        source = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode='RGB')
        capture_height = max(1, int(round(source.height*args.capture_width/
                                          source.width)))
        capture = source.resize((args.capture_width, capture_height),
                                Image.Resampling.LANCZOS)
        frames.append((resolved, source, capture))
    if not frames:
        raise ValueError('at least one images_sbs frame is required')
    if args.repeats < 1:
        raise ValueError('--repeats must be positive')

    model = v7.load_model(TARGET, 'box')
    fold = LiveFold(args.fold)
    codec = fold.codec(model)
    settings = _settings(args.strengths)
    cold_start_ms = {}
    resize_cost_ms = {}
    send_cost_ms = {}
    send_cpu_ms = {}
    paired_baseline_ms = {}
    paired_baseline_cpu_ms = {}
    send_delta_percent = {}
    send_cpu_delta_percent = {}
    raw_coefficients = {}
    fold_metrics = {}
    quality = {}

    # The one-time call for each preprocessor shape performs Numba compilation
    # and warms cached footprints before steady-state CPU samples.
    for mode, strength in settings:
        name = _setting_name(mode, strength)
        if mode != 'off':
            started = time.perf_counter()
            warmup_resize(frames[0][2].size, mode,
                          0.25 if strength is None else strength)
            cold_start_ms[name] = (time.perf_counter()-started)*1000.0
        else:
            cold_start_ms[name] = 0.0
        values_for_mode, coeffs_for_mode = [], []
        per_frame_resize, per_frame_quality = [], []
        for path, image, capture in frames:
            selected_strength = 0.25 if strength is None else strength
            started = time.perf_counter()
            values, _ = v7_live._values(
                model, capture, 'box', brightness=1.0, gamma=1.0,
                perceptual_resize=mode,
                perceptual_detail_strength=selected_strength)
            per_frame_resize.append((time.perf_counter()-started)*1000.0)
            values_for_mode.append(values)
            coefficients = codec.grid.forward(values)
            coeffs_for_mode.append(coefficients)

            kept = codec.encode_coefficients(values)
            decoded_full = codec.decode(
                kept, kept-model.mu, np.ones(len(kept)), metadata_confirmed=True)
            decoded = codec.grid.inverse(decoded_full)
            display = values_image(decoded, v7.V7_GRIDS).resize(
                DISPLAY, Image.Resampling.BICUBIC)
            per_frame_quality.append({
                'image': path.relative_to(REPO).as_posix(),
                **score(reference(image), display),
            })

        raw_coefficients[name] = np.asarray(coeffs_for_mode)
        resize_cost_ms[name] = float(np.mean(per_frame_resize))
        fold_metrics[name], _ = _fold_statistics(
            model, codec, values_for_mode, raw_coefficients[name])
        quality[name] = per_frame_quality

    def measure_sender(capture, mode, strength):
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        values, _ = v7_live._values(
            model, capture, 'box', brightness=1.0, gamma=1.0,
            perceptual_resize=mode,
            perceptual_detail_strength=0.25 if strength is None else strength)
        codec.encode_coefficients(values)
        return ((time.perf_counter()-wall_start)*1000.0,
                (time.process_time()-cpu_start)*1000.0)

    def live_frame(capture):
        # The live sender receives each capture as a read-only RGB uint8 array
        # (np.frombuffer over the FFmpeg pipe), not a Pillow image; time both
        # paths from that same form.
        pixels = np.asarray(capture.convert('RGB'), dtype=np.uint8)
        return np.frombuffer(pixels.tobytes(), np.uint8).reshape(pixels.shape)

    for mode, strength in settings:
        name = _setting_name(mode, strength)
        frame_baselines, frame_candidates, frame_deltas = [], [], []
        frame_cpu_baselines, frame_cpu_candidates, frame_cpu_deltas = [], [], []
        for _, _, capture_image in frames:
            capture = live_frame(capture_image)
            baseline_samples, candidate_samples = [], []
            for repeat in range(args.repeats):
                order = (('off', None, baseline_samples),
                         (mode, strength, candidate_samples))
                if repeat % 2:
                    order = tuple(reversed(order))
                for timed_mode, timed_strength, samples in order:
                    samples.append(measure_sender(capture, timed_mode,
                                                  timed_strength))
            baseline_ms = float(np.median([sample[0] for sample in baseline_samples]))
            candidate_ms = float(np.median([sample[0] for sample in candidate_samples]))
            baseline_cpu_ms = float(np.median([sample[1] for sample in baseline_samples]))
            candidate_cpu_ms = float(np.median([sample[1] for sample in candidate_samples]))
            if mode == 'off':
                candidate_ms, candidate_cpu_ms = baseline_ms, baseline_cpu_ms
            frame_baselines.append(baseline_ms)
            frame_candidates.append(candidate_ms)
            frame_deltas.append(0.0 if mode == 'off' else
                                (candidate_ms/baseline_ms-1.0)*100.0
                                if baseline_ms else 0.0)
            frame_cpu_baselines.append(baseline_cpu_ms)
            frame_cpu_candidates.append(candidate_cpu_ms)
            frame_cpu_deltas.append(0.0 if mode == 'off' else
                                    (candidate_cpu_ms/baseline_cpu_ms-1.0)*100.0
                                    if baseline_cpu_ms else 0.0)
        paired_baseline_ms[name] = float(np.mean(frame_baselines))
        send_cost_ms[name] = float(np.mean(frame_candidates))
        send_delta_percent[name] = float(np.mean(frame_deltas))
        paired_baseline_cpu_ms[name] = float(np.mean(frame_cpu_baselines))
        send_cpu_ms[name] = float(np.mean(frame_cpu_candidates))
        send_cpu_delta_percent[name] = float(np.mean(frame_cpu_deltas))

    results = {}
    for mode, strength in settings:
        name = _setting_name(mode, strength)
        results[name] = {
            'mode': mode,
            'detail_strength': strength,
            'resize_ms_per_frame_mean': resize_cost_ms[name],
            'jit_and_first_call_ms': cold_start_ms[name],
            'warmed_sender_ms_per_frame_mean': send_cost_ms[name],
            'paired_off_box_ms_per_frame_mean': paired_baseline_ms[name],
            'warmed_sender_delta_percent_vs_off_box': send_delta_percent[name],
            'warmed_sender_cpu_ms_per_frame_mean': send_cpu_ms[name],
            'paired_off_box_cpu_ms_per_frame_mean': paired_baseline_cpu_ms[name],
            'warmed_sender_cpu_delta_percent_vs_off_box':
                send_cpu_delta_percent[name],
            'coefficient_statistics_vs_frozen_model': _coefficient_statistics(
                model, raw_coefficients[name][:, codec.kept]),
            **fold_metrics[name],
            'fidelity_per_image': quality[name],
            'mean_ssimulacra2': float(np.mean([
                item['ssimulacra2'] for item in quality[name]])),
            'mean_psnr_y_db': float(np.mean([
                item['psnr_y'] for item in quality[name]])),
            'mean_delta_e2000': float(np.mean([
                item['de2000'] for item in quality[name]])),
        }

    output = {
        'description': 'clean coefficient-perfect folded decode; not tape/audio evidence',
        'model': 'canonical frozen box',
        'model_tables_sha256': v7.MODEL_TABLES_SHA256,
        'fold': args.fold,
        'fold_table_sha256': fold.digest,
        'fold_signature_slots': codec.signature,
        'environment': {
            'platform': platform.platform(),
            'processor': _cpu_model(),
            'logical_cpu_count': os.cpu_count(),
            'python': platform.python_version(),
            'numba': numba.__version__,
        },
        'frames': [path.relative_to(REPO).as_posix()
                   for path, _, _ in frames],
        'capture_proxy': {
            'filter': 'Pillow Lanczos, held constant for all candidates',
            'width': args.capture_width,
            'sizes': [list(capture.size) for _, _, capture in frames],
            'source': 'SBS color panel via utilities.convert_to_xy.load_rgba',
            'purpose': 'live-resolution input proxy from actual bake sources',
        },
        'held_out_from_model_fit': True,
        'score_geometry': {
            'display_size': list(DISPLAY),
            'source_reference': 'Lanczos resize of each source to the fixed display size',
            'decoded_display': 'V7 values_image then Pillow bicubic to display size',
            'delta_e2000': 'skimage rgb2lab (sRGB input), mean over display pixels',
        },
        'timing_method': (
            'per frame, interleaved off-box and candidate order, alternating by '
            'repeat; records wall and process CPU time for sender _values plus '
            'pinned FoldCodec coefficient folding from a read-only RGB uint8 '
            'capture array (the live FFmpeg frame form), excludes packet '
            'waveform/audio I/O; medians per frame then means across frames'),
        'settings': results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True)+'\n')
    print(json.dumps(output, indent=2, sort_keys=True))
    print(f'results: {args.out}')
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', nargs='+', required=True, type=Path,
                        help='real held-out source images, all beneath images_sbs/')
    parser.add_argument('--fold', choices=(500,), type=int, default=500)
    parser.add_argument('--capture-width', type=int, default=160,
                        help='fixed simulated live-capture width (default: 160 px)')
    parser.add_argument('--strengths', nargs='+', type=float,
                        default=(0.0, 0.25, 0.5, 1.0),
                        help='fixed detail strengths for both detail-domain ablations')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--out', type=Path,
                        default=REPO/'tmp'/'test_modem_v7'/'perceptual_resize'/'results.json')
    args = parser.parse_args(argv)
    if args.capture_width < 1:
        parser.error('--capture-width must be positive')
    if any(not np.isfinite(s) or s < 0 or s > 1 for s in args.strengths):
        parser.error('--strengths values must be finite and in [0, 1]')
    try:
        run(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
