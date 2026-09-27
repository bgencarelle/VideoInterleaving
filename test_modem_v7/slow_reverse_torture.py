#!/usr/bin/env python3
"""Repeatable reverse-playback speed, CPU and picture-quality benchmark.

Run from the repository root::

    .venv/bin/python test_modem_v7/slow_reverse_torture.py

The first run writes one fixed packet series under ``tmp/``. Every tested speed
is made from those same encoded packets; only their sample spacing changes.
The benchmark reports whole-series pulse acquisition, isolated packet decode,
and the chunked ``LiveInput`` path used by the standalone receiver.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from animation_modem import v7                                      # noqa: E402
from animation_modem.imaging import values_image                    # noqa: E402
from animation_modem.v7_live_input import LiveInput                # noqa: E402


SERIES_PATH = ROOT/'tmp/v7_slow_reverse_series.npz'
DEFAULT_SPEEDS = (.25, .2, .15, .1, .075, .05, .025, .0125, .01)
COUNTERS = (1, 2, 3, 4, 7, 8, 9, 10)  # avoid loop-field slices 5 and 6


def _fixed_series(path):
    """Load or create the one forward-time packet train used by all cases."""
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            wire = np.asarray(saved['wire'], dtype=np.float32)
            source_indices = np.asarray(saved['source_indices'], dtype=np.int32)
            digest = str(saved['sha256'].item())
        actual = hashlib.sha256(wire.tobytes()).hexdigest()
        if actual != digest:
            raise ValueError(f'{path} has a corrupt packet-series hash')
        if len(wire) != len(COUNTERS)*v7.PULSE_FRAME:
            raise ValueError(f'{path} has an unexpected packet count')
        return wire, source_indices, digest

    target = .1521/np.sqrt(1 + 10**(v7.CLOCK_REL_DB/10))
    model = v7.load_model(target, 'nearest')
    with Image.open(v7.REFERENCE_FIXTURE) as image:
        images = (image.convert('RGB'), ImageOps.mirror(image).convert('RGB'))
    values = [v7.image_values(v7.prepare_image(image, 'nearest'),
                              model.coder.grids, 'nearest')
              for image in images]
    packets = [v7.encode_pulse_frame(
        model, values[index % 2], counter=counter, aspect_code=6,
        source_index=index, pilot_tones=True, eof_marker=True)
        for index, counter in enumerate(COUNTERS)]
    wire = np.concatenate(packets).astype(np.float32)
    source_indices = np.arange(len(COUNTERS), dtype=np.int32)
    digest = hashlib.sha256(wire.tobytes()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, wire=wire, source_indices=source_indices,
                        sha256=np.asarray(digest))
    return wire, source_indices, digest


def _samples(wire, speed):
    """Apply speed to the fixed packets, then reverse packet order and time."""
    sped = v7.speed_pulse_stream(wire, speed, rate=v7.RATE)
    reverse = sped[::-1].copy()
    margin = int(np.ceil(v7.REVERSE_PACKET_MARGIN/speed))+8
    silence = np.zeros((margin, 2), dtype=np.float32)
    return np.concatenate((silence, reverse, silence))


def _quality_references(model):
    """Rebuild the two source-coded references used by the fixed series."""
    with Image.open(v7.REFERENCE_FIXTURE) as image:
        images = (image.convert('RGB'),
                  ImageOps.mirror(image.convert('RGB')))
    prepared = [v7.prepare_image(image, 'nearest') for image in images]
    vectors = [v7.image_values(image, model.coder.grids, 'nearest')
               for image in prepared]
    pictures = [values_image(values, model.coder.grids) for values in vectors]
    display_size = (405, 540)
    source_references = [image.resize(display_size, Image.Resampling.LANCZOS)
                         for image in images]
    return (vectors, [np.asarray(image, dtype=np.float64) for image in pictures],
            source_references)


def _ssimulacra2(reference, decoded):
    """Score the same fixed 405x540 source and decoded display images."""
    from ssimulacra2 import compute_ssimulacra2

    def png_bytes(image):
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        return buffer

    return float(compute_ssimulacra2(png_bytes(reference),
                                     png_bytes(decoded)))


def _quality_metrics(frames, reference_vectors, reference_pictures,
                     source_references, model):
    """Score decoded frames against encode-side and source-image references."""
    value_sse = 0.0
    value_count = 0
    plane_sse = np.zeros(len(model.coder.grids), dtype=np.float64)
    plane_counts = np.zeros(len(model.coder.grids), dtype=np.int64)
    rgb_sse = 0.0
    rgb_count = 0
    perceptual_scores = []
    for frame in frames:
        source_index = int(frame.diag['source_index'])
        reference_index = source_index % len(reference_vectors)
        expected = reference_vectors[reference_index]
        decoded = v7.values_from(model, frame.coeffs)
        error = decoded-expected
        value_sse += float(np.sum(np.square(error)))
        value_count += error.size
        offset = 0
        for plane, (rows, columns) in enumerate(model.coder.grids):
            count = rows*columns
            plane_error = error[offset:offset+count]
            plane_sse[plane] += float(np.sum(np.square(plane_error)))
            plane_counts[plane] += count
            offset += count
        decoded_picture = np.asarray(
            values_image(decoded, model.coder.grids), dtype=np.float64)
        rgb_error = decoded_picture-reference_pictures[reference_index]
        rgb_sse += float(np.sum(np.square(rgb_error)))
        rgb_count += rgb_error.size
        display_picture = values_image(decoded, model.coder.grids).resize(
            source_references[reference_index].size, Image.Resampling.BICUBIC)
        perceptual_scores.append(_ssimulacra2(
            source_references[reference_index], display_picture))
    value_rmse = float(np.sqrt(value_sse/value_count))
    rgb_rmse = float(np.sqrt(rgb_sse/rgb_count))
    return {
        'frames': len(frames),
        'normalized_value_rmse': round(value_rmse, 7),
        'normalized_value_rmse_by_plane': [
            round(float(np.sqrt(error/count)), 7)
            for error, count in zip(plane_sse, plane_counts)],
        'reconstructed_rgb_rmse_255': round(rgb_rmse, 4),
        'reconstructed_rgb_psnr_db': round(20*np.log10(255/rgb_rmse), 3),
        'ssimulacra2_vs_source_display_mean': round(
            statistics.mean(perceptual_scores), 3),
        'ssimulacra2_vs_source_display_min': round(min(perceptual_scores), 3),
        'ssimulacra2_vs_source_display_max': round(max(perceptual_scores), 3),
    }


def _measure(function, repeats):
    cpu_ms, wall_ms = [], []
    function()  # populate any one-time lazy caches before collecting samples
    for _ in range(repeats):
        cpu_start, wall_start = time.process_time_ns(), time.perf_counter_ns()
        function()
        cpu_ms.append((time.process_time_ns()-cpu_start)/1e6)
        wall_ms.append((time.perf_counter_ns()-wall_start)/1e6)
    return {
        'median_cpu_ms': round(statistics.median(cpu_ms), 4),
        'p95_cpu_ms': round(float(np.percentile(cpu_ms, 95)), 4),
        'median_wall_ms': round(statistics.median(wall_ms), 4),
    }


def _live_pass(model, samples, chunk_size=1024):
    """Run the production rolling-input policy over one captured packet train."""
    live = LiveInput(rate=v7.RATE, direction='auto')
    state = v7.PulseState(tail_memory=False)
    decoded, decode_cpu_ms, decode_wall_ms = [], [], []
    last_absolute_start = None
    live_cpu_start = time.process_time_ns()
    for offset in range(0, len(samples), chunk_size):
        block = samples[offset:offset+chunk_size]
        live.add(block.copy())
        audio = live.take(offset/v7.RATE)
        if audio is None:
            continue
        hits = live.pulse_hits(audio)
        reverse_hits = [hit for hit in hits if hit[3] < 0]
        if reverse_hits:
            packet_start, scale, _, direction = max(
                reverse_hits, key=lambda hit: hit[0])
            audio_start = live.total-len(audio)
            absolute_start = int(round(audio_start+packet_start))
            if absolute_start != last_absolute_start:
                state.set_playback_direction(direction)
                cpu_start, wall_start = time.process_time_ns(), time.perf_counter_ns()
                frames, info = v7.decode_reverse_packet(
                    model, audio, packet_start, scale, state=state,
                    sample_rate=v7.RATE, pilot_timing='tone-seeded')
                decode_cpu_ms.append((time.process_time_ns()-cpu_start)/1e6)
                decode_wall_ms.append((time.perf_counter_ns()-wall_start)/1e6)
                eof_count = int(info.get('eof_markers_validated') or 0)
                decoded.extend((int(frame.diag.get('source_index', -1)),
                                frame.status, eof_count)
                               for frame in frames)
                last_absolute_start = absolute_start
        live.decoded()
    live_cpu_ms = (time.process_time_ns()-live_cpu_start)/1e6
    return {
        'decoded': decoded,
        'live_cpu_ms': live_cpu_ms,
        'decode_cpu_ms': decode_cpu_ms,
        'decode_wall_ms': decode_wall_ms,
        'buffer_max_scale': live.max_scale,
        'buffer_final_scale': live.scale,
    }


def _time_live(model, samples, repeats):
    values = [_live_pass(model, samples) for _ in range(repeats)]
    first = values[0]
    return {
        'decoded': first['decoded'],
        'median_live_cpu_ms': round(statistics.median(
            value['live_cpu_ms'] for value in values), 3),
        'p95_live_cpu_ms': round(float(np.percentile(
            [value['live_cpu_ms'] for value in values], 95)), 3),
        'median_decode_cpu_ms_per_packet': round(statistics.median(
            [ms for value in values for ms in value['decode_cpu_ms']]), 4),
        'p95_decode_cpu_ms_per_packet': round(float(np.percentile(
            [ms for value in values for ms in value['decode_cpu_ms']], 95)), 4),
        'median_decode_wall_ms_per_packet': round(statistics.median(
            [ms for value in values for ms in value['decode_wall_ms']]), 4),
        'buffer_max_scale': first['buffer_max_scale'],
        'buffer_final_scale': first['buffer_final_scale'],
    }


def _silence_pass(seconds, chunk_size=1024):
    """Measure unknown-speed rolling-buffer upkeep without packet decodes."""
    block = np.zeros((chunk_size, 2), dtype=np.float32)
    count = int(np.ceil(seconds*v7.RATE/chunk_size))
    live = LiveInput(rate=v7.RATE, direction='auto')
    start = time.process_time_ns()
    for index in range(count):
        live.add(block.copy())
        audio = live.take(index*chunk_size/v7.RATE)
        if audio is not None:
            live.decoded()
    cpu_ms = (time.process_time_ns()-start)/1e6
    media_seconds = count*chunk_size/v7.RATE
    return {
        'cpu_ms': cpu_ms,
        'media_seconds': media_seconds,
        'buffered_samples': live.buffered(),
        'buffer_cap_samples': live.cap(),
    }


def _time_silence(seconds, repeats):
    values = [_silence_pass(seconds) for _ in range(repeats)]
    median_cpu = statistics.median(value['cpu_ms'] for value in values)
    media_seconds = values[0]['media_seconds']
    return {
        'media_seconds': round(media_seconds, 3),
        'median_cpu_ms': round(median_cpu, 3),
        'p95_cpu_ms': round(float(np.percentile(
            [value['cpu_ms'] for value in values], 95)), 3),
        'one_core_realtime_cpu_pct': round(median_cpu/(media_seconds*10), 2),
        'buffered_samples': values[0]['buffered_samples'],
        'buffer_cap_samples': values[0]['buffer_cap_samples'],
    }


def run(speeds, repeats, series_path, silence_seconds):
    wire, source_indices, digest = _fixed_series(series_path)
    target = .1521/np.sqrt(1 + 10**(v7.CLOCK_REL_DB/10))
    model = v7.load_model(target, 'nearest')
    (reference_vectors, reference_pictures,
     source_references) = _quality_references(model)
    v7.PULSE.warmup_pulse_kernels()
    v7.warmup_equalizer(model)
    silence_baseline = _time_silence(silence_seconds, repeats)
    print(f'unknown-speed silence: {silence_baseline["one_core_realtime_cpu_pct"]:.2f}% '
          f'core over {silence_baseline["media_seconds"]:.1f}s '
          f'({silence_baseline["buffer_cap_samples"]} samples retained)',
          flush=True)

    rows = []
    production_max = v7.PULSE_MAX_SCALE
    for speed in speeds:
        if not np.isfinite(speed) or speed <= 0 or speed > 4:
            raise ValueError(f'invalid playback speed {speed!r}')
        requested_scale = 1.0/speed
        if requested_scale > production_max*1.02:
            raise ValueError(
                f'{speed:g}x requires scale {requested_scale:g}, above the '
                f'production maximum {production_max:g}')
        samples = _samples(wire, speed)
        hits = v7.pulse_frame_hits(samples, sample_rate=v7.RATE,
                                   direction='auto')
        expected = list(reversed(source_indices.tolist()))
        if len(hits) != len(expected) or any(hit[3] != -1 for hit in hits):
            raise RuntimeError(f'{speed:g}x acquisition got {len(hits)} hits')

        def acquire():
            return v7.pulse_frame_hits(samples, sample_rate=v7.RATE,
                                       direction='auto')

        acquisition = _measure(acquire, repeats)

        decode_samples = []
        decode_results = []
        for hit in hits:
            packet_results, info = v7.decode_reverse_packet(
                model, samples, hit[0], hit[1],
                state=v7.PulseState(tail_memory=False),
                sample_rate=v7.RATE, pilot_timing='tone-seeded')
            decode_samples.append((hit, packet_results))
            if (len(packet_results) != 1 or
                    int(info.get('eof_markers_validated') or 0) != 1 or
                    packet_results[0].status != 'received' or
                    not packet_results[0].diag.get('metadata_valid')):
                raise RuntimeError(f'{speed:g}x packet failed EOF/decode preflight')
            decode_results.append(packet_results[0])
        decoded_indices = [int(frame.diag.get('source_index', -1))
                           for frame in decode_results]
        if decoded_indices != expected:
            raise RuntimeError(f'{speed:g}x source order {decoded_indices}')
        quality = _quality_metrics(decode_results, reference_vectors,
                                   reference_pictures, source_references, model)

        def decode_all():
            for hit, _ in decode_samples:
                v7.decode_reverse_packet(
                    model, samples, hit[0], hit[1],
                    state=v7.PulseState(tail_memory=False),
                    sample_rate=v7.RATE, pilot_timing='tone-seeded')

        decode_timing = _measure(decode_all, repeats)
        decode_timing['median_cpu_ms_per_packet'] = round(
            decode_timing.pop('median_cpu_ms')/len(hits), 4)
        decode_timing['p95_cpu_ms_per_packet'] = round(
            decode_timing.pop('p95_cpu_ms')/len(hits), 4)
        decode_timing['median_wall_ms_per_packet'] = round(
            decode_timing.pop('median_wall_ms')/len(hits), 4)

        live_timing = _time_live(model, samples, repeats)
        live_decoded = live_timing.pop('decoded')
        live_indices = [index for index, status, eof in live_decoded
                        if status in ('received', 'degraded') and eof]
        if live_indices != expected:
            raise RuntimeError(
                f'{speed:g}x live path received {live_indices}, expected {expected}')
        duration_s = len(samples)/v7.RATE
        live_timing['media_duration_s'] = round(duration_s, 3)
        live_timing['one_core_realtime_cpu_pct'] = round(
            live_timing['median_live_cpu_ms']/(duration_s*10), 2)
        live_timing['live_received_indices'] = live_indices
        live_timing['live_statuses'] = [status for _, status, _ in live_decoded]
        row = {
            'speed': speed,
            'scale': requested_scale,
            'sample_count': len(samples),
            'packet_count': len(hits),
            'acquisition': acquisition,
            'isolated_decode': decode_timing,
            'quality': quality,
            'live_input': live_timing,
            'all_live_packets_received': live_indices == expected,
        }
        rows.append(row)
        print(f'{speed:7.3f}x scale={requested_scale:6.2f} '
              f'samples={len(samples):8d} '
              f'detect={acquisition["median_cpu_ms"]:8.3f}ms/series '
              f'decode={decode_timing["median_cpu_ms_per_packet"]:7.3f}ms/pkt '
              f'RGB-RMSE={quality["reconstructed_rgb_rmse_255"]:6.3f}/255 '
              f'PSNR={quality["reconstructed_rgb_psnr_db"]:5.2f}dB '
              f'SSIMULACRA2={quality["ssimulacra2_vs_source_display_mean"]:7.3f} '
              f'live={live_timing["median_live_cpu_ms"]:8.2f}ms '
              f'({live_timing["one_core_realtime_cpu_pct"]:5.2f}% core) '
              f'frames={live_indices}', flush=True)

    report = {
        'series_file': str(series_path.relative_to(ROOT)),
        'series_sha256': digest,
        'packet_count': len(COUNTERS),
        'capture_rate_hz': v7.RATE,
        'repeats': repeats,
        'unknown_speed_silence_baseline': silence_baseline,
        'production_max_scale': production_max,
        'results': rows,
    }
    print(json.dumps(report, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--speeds', nargs='+', type=float, default=DEFAULT_SPEEDS,
                        help='playback speeds in x (supported range: .01 through 4)')
    parser.add_argument('--repeats', type=int, default=5,
                        help='timed runs per case (default: 5)')
    parser.add_argument('--silence-seconds', type=float, default=30.0,
                        help='unknown-speed silence baseline duration (default: 30)')
    parser.add_argument('--series', type=Path, default=SERIES_PATH,
                        help='fixed packet-series NPZ path under repository tmp/')
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    if not np.isfinite(args.silence_seconds) or args.silence_seconds <= 0:
        parser.error('--silence-seconds must be positive')
    if not args.series.is_absolute():
        args.series = ROOT/args.series
    try:
        args.series.relative_to(ROOT/'tmp')
    except ValueError:
        parser.error('--series must be located under the repository tmp/ directory')
    run(args.speeds, args.repeats, args.series, args.silence_seconds)


if __name__ == '__main__':
    main()
