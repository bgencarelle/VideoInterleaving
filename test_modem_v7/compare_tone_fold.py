#!/usr/bin/env python3
"""Compare coded-pilot metadata selection with blind and oracle fold decoding.

For each pinned fold table, send identical folded values with no tones, the
current steady pilot tones, or the experimental coded pilot. Compare blind
decoding, probing both pinned fold signatures, coded-status selection/latching,
and an oracle supplied the sender's table. The signature gates ordinary table
probes; a valid coded mode may authorize signature-missing reconstruction.
The same V7 pulse receiver/timing mode is used on all paths.
``--include-current-v7`` and ``--include-box-v7`` add un-folded baselines.

    python test_modem_v7/compare_tone_fold.py [--packets 12]
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
from scipy.signal import resample_poly

from common import (CASES as CHANNEL_CASES, DISPLAY, RATE, REPO, STEADY_FROM,
                    TARGET, impair, reference, ssimulacra2, v7, v7_values)
from animation_modem.imaging import values_image
from fold_modem import IMPAIRMENTS
from live_fold import LiveFold
from tone_code import (FOLD_500, FOLD_1000, acquire_packet_starts,
                       add_tone_code, coded_pilot_timing, decode_status,
                       decode_tone_code, encode_status,
                       match_tone_results_to_frames)

TONE_VARIANTS = ('no-tone', 'steady', 'coded')
MODE_FOR_SLOTS = {500: FOLD_500, 1000: FOLD_1000}
STEADY_VARIANTS = ('steady', 'current-v7', 'box-v7')
DEFAULT_CASES = ('clean', 'dropouts', 'wow-flutter', 'fast-flutter',
                 'jitter 0.1%', 'jitter 0.3%', 'lowpass-10k',
                 'warble+lpf12k+dropouts', 'mono-sum', 'hiss-35',
                 'hiss-30')
_CHANNEL_CASE_BY_NAME = {case.name: case for case in CHANNEL_CASES}
IMPAIRMENTS = dict(IMPAIRMENTS)
for _extra_case in ('mono-sum', 'hiss-35'):
    _case = _CHANNEL_CASE_BY_NAME[_extra_case]
    IMPAIRMENTS[_extra_case] = (
        lambda wire, case=_case: impair(wire, case, seed=2026))
_hiss_30 = replace(_CHANNEL_CASE_BY_NAME['hiss-35'],
                   name='hiss-30', noise_dbfs=-30)
IMPAIRMENTS['hiss-30'] = lambda wire: impair(wire, _hiss_30, seed=2026)


def _encode(model, values, fold, slots, variant, packets, aspect_code):
    transmit_values = fold.encode(model, values) if fold is not None else values
    mode = FOLD_500 if slots == 500 else FOLD_1000 if slots == 1000 else None
    wire = []
    for index in range(packets):
        counter = index+1
        packet = v7.encode_pulse_frame(
            model, transmit_values, counter, aspect_code=aspect_code,
            source_index=index,
            pilot_tones=(variant in STEADY_VARIANTS), eof_marker=True)
        if variant == 'coded':
            packet = add_tone_code(packet, counter, encode_status(mode))
        wire.append(packet)
    return np.concatenate(wire)


def _receive(capture_fold, model, capture, pilot_timing):
    with coded_pilot_timing():
        if capture_fold is None:
            results, info = v7.decode_pulse_stream(
                model, capture, sample_rate=RATE, pilot_timing=pilot_timing,
                frame_boundary='eof')
        else:
            capture_fold.install()
            try:
                results, info = v7.decode_pulse_stream(
                    model, capture, sample_rate=RATE, pilot_timing=pilot_timing,
                    frame_boundary='eof')
            finally:
                capture_fold.uninstall()
    return results, info


def _fold_for_status(folds, status_result):
    if status_result is None or not status_result.get('valid'):
        return None
    status = status_result.get('status')
    if status is None:
        return None
    fold_slots = status['fold_slots']
    return folds.get(fold_slots) if fold_slots else None


def _probe_fold_signatures(folds, model, result):
    """Try each compatible pinned signature and require one unambiguous hit."""
    hits = []
    for fold in folds.values():
        try:
            codec = fold.codec(model)
            codec.last_score = codec.last_noise = None
            values = fold.values(model, result)
        except ValueError:
            continue
        if (codec.last_score is not None and codec.last_score >= .5 and
                codec.last_noise is not None and
                codec.last_noise <= codec.noise_max and
                codec.last_unfolded_slots > 0):
            hits.append((fold, values, codec.last_score, codec.last_noise))
    if len(hits) == 1:
        return hits[0]
    return None, None, None, None


def _score_run(ref, values, model, fold, slots, folds, variant, case,
               packets, pilot_timing, aspect_code):
    wire = resample_poly(
        _encode(model, values, fold, slots, variant, packets, aspect_code), 2, 1,
        axis=0).astype(np.float32)
    capture = IMPAIRMENTS[case](wire)
    # install() only records equaliser outputs for later fold reconstruction;
    # the capture hook's table is deliberately independent of the sent table.
    capture_fold = next(iter(folds.values())) if folds else None
    results, info = _receive(capture_fold, model, capture, pilot_timing)
    starts = acquire_packet_starts(capture, limit=packets)
    tone_results = [decode_tone_code(
        capture, start, scale, RATE) for start, scale, _ in starts]
    status_by_result, matched_status_frames = match_tone_results_to_frames(
        results, starts, tone_results)
    latched_status_by_result = {}
    latched_status = None
    have_latched_status = False
    for decoded_frame in results:
        status_result = status_by_result.get(id(decoded_frame))
        if status_result is not None and status_result.get('valid'):
            latched_status = status_result
            have_latched_status = True
        latched_status_by_result[id(decoded_frame)] = (
            latched_status if have_latched_status else None)

    image_paths = {
        'blind': {'scores': [], 'fold_packets': 0, 'fold_applied_frames': 0,
                  'metadata_packets': 0},
        'signature_probe': {'scores': [], 'fold_packets': 0,
                            'fold_applied_frames': 0, 'metadata_packets': 0,
                            'signature_scores': [], 'signature_noises': []},
        'status_selected': {'scores': [], 'fold_packets': 0,
                            'fold_applied_frames': 0, 'metadata_packets': 0},
        'latched_status': {'scores': [], 'fold_packets': 0,
                           'fold_applied_frames': 0, 'metadata_packets': 0},
        'oracle': {'scores': [], 'fold_packets': 0, 'fold_applied_frames': 0,
                   'metadata_packets': 0},
    }
    signature_scores, signature_noises = [], []
    displayable = 0
    for result in results:
        shown = (result.status in ('received', 'verified') or
                 result.diag.get('displayable'))
        if not shown:
            continue
        displayable += 1
        source_index = result.diag.get('source_index')
        if source_index is None or result.counter < STEADY_FROM:
            continue

        packet_status = status_by_result.get(id(result))
        packet_latched_status = latched_status_by_result.get(id(result))
        selected_fold = _fold_for_status(folds, packet_status)
        latched_fold = _fold_for_status(folds, packet_latched_status)
        signature_fold, signature_values, signature_score, signature_noise = (
            _probe_fold_signatures(folds, model, result))
        paths = {
            'blind': (None, False, None),
            'signature_probe': (signature_fold, False, signature_values),
            'status_selected': (selected_fold,
                                selected_fold is not None and packet_status is not None,
                                None),
            'latched_status': (latched_fold,
                               latched_fold is not None and
                               packet_latched_status is not None,
                               None),
            'oracle': (fold, fold is not None, None),
        }
        score_cache = {}
        for name, (selected, metadata_confirmed, precomputed) in paths.items():
            summary = image_paths[name]
            if (name == 'status_selected' and packet_status is not None and
                    packet_status.get('valid')):
                summary['metadata_packets'] += 1
            if name == 'latched_status' and packet_latched_status is not None:
                summary['metadata_packets'] += 1
            if name == 'signature_probe' and signature_fold is not None:
                summary['signature_scores'].append(signature_score)
                if signature_noise is not None:
                    summary['signature_noises'].append(signature_noise)
            if selected is not None:
                summary['fold_packets'] += 1
                cache_key = ('fold', selected.slots, metadata_confirmed)
            else:
                cache_key = ('plain', False)
            if cache_key not in score_cache:
                if selected is not None:
                    codec = selected.codec(model)
                    if precomputed is None:
                        codec.last_score = codec.last_noise = None
                        decoded_values = selected.values(
                            model, result,
                            metadata_confirmed=metadata_confirmed)
                    else:
                        decoded_values = precomputed
                    if name == 'oracle':
                        if codec.last_score is not None:
                            signature_scores.append(codec.last_score)
                        if codec.last_noise is not None:
                            signature_noises.append(codec.last_noise)
                    unfolded_slots = codec.last_unfolded_slots
                else:
                    decoded_values = v7.values_from(model, result.coeffs)
                    unfolded_slots = 0
                picture = values_image(decoded_values, v7.V7_GRIDS).resize(
                    DISPLAY, Image.Resampling.BICUBIC)
                score_cache[cache_key] = (ssimulacra2(ref, picture),
                                          unfolded_slots)
            score, unfolded_slots = score_cache[cache_key]
            if selected is not None and unfolded_slots > 0:
                summary['fold_applied_frames'] += 1
            summary['scores'].append(score)

    image_recovery = {}
    for name, summary in image_paths.items():
        scores = summary.pop('scores')
        image_recovery[name] = {
            'ssimulacra2': round(float(np.mean(scores)), 2) if scores else None,
            'scored_frames': len(scores),
            'fold_table_selected_frames': summary['fold_packets'],
            'fold_applied_frames': summary['fold_applied_frames'],
            'metadata_available_frames': summary['metadata_packets'],
        }
        if name == 'signature_probe':
            image_recovery[name]['mean_signature_score'] = (
                round(float(np.mean(summary['signature_scores'])), 3)
                if summary['signature_scores'] else None)
            image_recovery[name]['mean_signature_noise'] = (
                round(float(np.mean(summary['signature_noises'])), 4)
                if summary['signature_noises'] else None)
    oracle_score = image_recovery['oracle']['ssimulacra2']
    blind_score = image_recovery['blind']['ssimulacra2']
    signature_score = image_recovery['signature_probe']['ssimulacra2']
    for name in ('signature_probe', 'status_selected', 'latched_status'):
        score = image_recovery[name]['ssimulacra2']
        image_recovery[name]['delta_vs_blind'] = (
            round(score-blind_score, 2)
            if score is not None and blind_score is not None else None)
        image_recovery[name]['delta_vs_oracle'] = (
            round(score-oracle_score, 2)
            if score is not None and oracle_score is not None else None)
        image_recovery[name]['delta_vs_signature_probe'] = (
            round(score-signature_score, 2)
            if score is not None and signature_score is not None else None)

    expected = None
    if variant == 'coded':
        expected = decode_status(encode_status(MODE_FOR_SLOTS[slots]))
    valid_statuses = [result for result in tone_results if result['valid']]
    correct_statuses = (sum(result['status']['mode'] == expected['mode']
                             for result in valid_statuses)
                         if expected is not None else None)
    status = {
        'acquired': len(tone_results),
        'valid': len(valid_statuses),
        'correct': correct_statuses,
        'false_accepts': (len(valid_statuses)-correct_statuses
                          if expected is not None else len(valid_statuses)),
        'mean_pilot_score': round(float(np.mean(
            [result['pilot_score'] for result in tone_results])), 3)
            if tone_results else None,
    }
    timing_diagnostics = [result.diag.get('pilot_timing', {})
                          for result in results]
    coded_timing = [item for item in timing_diagnostics
                    if item.get('coded_chips_removed')]

    return {
        'case': case,
        'fold': slots,
        'tones': variant,
        'receiver_pilot_timing': pilot_timing,
        'decoded': len(results),
        'displayable': displayable,
        'scored_frames': image_recovery['oracle']['scored_frames'],
        'ssimulacra2': oracle_score,
        'image_recovery': image_recovery,
        'fold_signature_packets': len(signature_scores),
        'mean_fold_signature_score': round(float(np.mean(signature_scores)), 3)
            if signature_scores else None,
        'mean_fold_signature_noise': round(float(np.mean(signature_noises)), 4)
            if signature_noises else None,
        'metadata': status,
        'tone_frame_matches': matched_status_frames,
        'coded_timing_despread_frames': len(coded_timing),
        'coded_timing_applied_frames': sum(
            item.get('mode_applied') == pilot_timing for item in coded_timing),
        'receiver_info': info,
    }


def _current_v7_values(frame, model, brightness):
    prepared = v7.prepare_image(frame, 'nearest')
    if brightness != 1.0:
        prepared = ImageEnhance.Brightness(prepared).enhance(brightness)
    return v7.image_values(prepared, model.coder.grids, 'nearest')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', nargs='+', type=Path,
                        help='images to score (defaults to V7 reference fixture)')
    parser.add_argument('--folds', nargs='+', type=int, choices=(500, 1000),
                        default=[500, 1000])
    parser.add_argument('--packets', type=int, default=12)
    parser.add_argument('--cases', nargs='+', choices=tuple(IMPAIRMENTS),
                        default=list(DEFAULT_CASES))
    parser.add_argument('--tones', nargs='+', choices=TONE_VARIANTS,
                        default=list(TONE_VARIANTS))
    parser.add_argument('--include-current-v7', action='store_true',
                        help='also run live V7 defaults: nearest, brightness 1.05, steady tones')
    parser.add_argument('--include-box-v7', action='store_true',
                        help='also run un-folded box V7 as a matched fold-profile baseline')
    parser.add_argument('--current-brightness', type=float, default=1.05,
                        help='live V7 baseline brightness (default: 1.05)')
    parser.add_argument('--pilot-timing', choices=('baseline', 'tone-seeded'),
                        default='baseline',
                        help='same V7 receive timing mode for every comparison')
    parser.add_argument('--out', type=Path,
                        default=REPO/'tmp'/'test_modem_v7'/'tone_fold_compare')
    args = parser.parse_args(argv)
    if args.packets < STEADY_FROM:
        parser.error(f'--packets must be at least {STEADY_FROM} to score steady frames')

    model = v7.load_model(TARGET, 'box')
    nearest_model = (v7.load_model(TARGET, 'nearest')
                     if args.include_current_v7 else None)
    frames = args.frames or [v7.REFERENCE_FIXTURE]
    folds = {slots: LiveFold(slots) for slots in args.folds}
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in frames:
        with Image.open(path) as image:
            frame = image.convert('RGB')
        ref = reference(frame)
        values = v7_values(frame, 'box')
        aspect_code = v7.aspect_wire_code(frame.size)
        for case in args.cases:
            if args.include_current_v7:
                current_values = _current_v7_values(
                    frame, nearest_model, args.current_brightness)
                row = _score_run(ref, current_values, nearest_model, None, None,
                                 folds, 'current-v7', case, args.packets,
                                 args.pilot_timing, aspect_code)
                row['frame'] = str(path)
                row['encode_filter'] = 'nearest'
                row['brightness'] = args.current_brightness
                rows.append(row)
            if args.include_box_v7:
                row = _score_run(ref, values, model, None, None, folds,
                                 'box-v7', case, args.packets, args.pilot_timing,
                                 aspect_code)
                row['frame'] = str(path)
                row['encode_filter'] = 'box'
                row['brightness'] = 1.0
                rows.append(row)
            for slots, fold in folds.items():
                for variant in args.tones:
                    row = _score_run(ref, values, model, fold, slots, folds,
                                     variant, case, args.packets,
                                     args.pilot_timing, aspect_code)
                    row['frame'] = str(path)
                    row['encode_filter'] = 'box'
                    row['brightness'] = 1.0
                    rows.append(row)

    # Add paired quality deltas for each image/fold/impairment combination.
    by_key = {}
    for row in rows:
        by_key.setdefault((row['frame'], row['case'], row['fold']), {})[
            row['tones']] = row
    for variants in by_key.values():
        coded = variants.get('coded')
        if coded is None:
            continue
        for baseline in ('no-tone', 'steady'):
            score = variants.get(baseline, {}).get('ssimulacra2')
            coded[f'delta_ssimulacra2_vs_{baseline}'] = (
                round(coded['ssimulacra2']-score, 2)
                if coded['ssimulacra2'] is not None and score is not None else None)
        for baseline in ('current-v7', 'box-v7'):
            reference_row = next((row for row in rows
                                  if row['frame'] == coded['frame'] and
                                  row['case'] == coded['case'] and
                                  row['tones'] == baseline), None)
            baseline_score = (reference_row['ssimulacra2']
                              if reference_row else None)
            coded[f'delta_ssimulacra2_vs_{baseline}'] = (
                round(coded['ssimulacra2']-baseline_score, 2)
                if coded['ssimulacra2'] is not None and baseline_score is not None
                else None)
    for row in rows:
        print(json.dumps(row, sort_keys=True), flush=True)
    (args.out/'results.json').write_text(json.dumps(rows, indent=1) + '\n')


if __name__ == '__main__':
    main()
