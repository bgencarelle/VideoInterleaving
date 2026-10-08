#!/usr/bin/env python3
"""Benchmark color coding/detail allocation through real V7 audio packets.

Run from the repository root with .venv/bin/python. No audio device is opened.
See docs/V7_COLOR_RESOLUTION_BENCHMARK.md for the experimental design.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw
from scipy.io import wavfile

from animation_modem import v7
from animation_modem.v7_core import adapt_packet_for_output
from tools.v7_color_transforms import ColorTransform, NAMES
from tools.v7_color_wire import ColorWire, fit_statistics, SPLITS
from tools.v7_color_corpus import (
    ROOT, PARTITIONS, PACKET_SECONDS, exposures, face_image, face_path,
    indices, digest_file)
from tools.v7_color_metrics import (
    render, picture_metrics, bar_metrics, summarize, resize_rgb, raster_size, ROIS,
    display_defaults)
from tools.v7_torture_matrix import CASES, RATE, impair
from tools.v7_send_gui import default_kernel_for_profile, ENCODE_DEFAULTS


PROFILES = ('aspect-fold-500', 'aspect-mono-500')
KERNELS = ('viewer_solve', 'native_source', 'csf_peak', 'band_taper', 'antiring')
FINAL_CASES = ('clean-96k', 'hiss-45', 'hiss-40', 'hiss-35', 'lowpass-12k',
               'lowpass-10k', 'wow-flutter', 'fast-flutter', 'cassette-i-hot',
               'dropouts', 'azimuth-12us', 'crosstalk-10pct', 'right-minus-4db')
STEREO_CASES = frozenset(('azimuth-12us', 'crosstalk-10pct', 'right-minus-4db'))
SEEDS = (2026, 2027, 2028, 2029, 2030)
CASE_MAP = {case.name: case for case in CASES}
FORMAT = 'v7-color-resolution-1'


def json_write(path, data):
    def convert(x):
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(data, indent=2, default=convert, allow_nan=False)+'\n')


def code_identity():
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    diff = subprocess.run(['git', 'diff', 'HEAD'], cwd=ROOT,
                          capture_output=True, check=True).stdout
    files = [ROOT/'tools'/name for name in (
        'v7_color_resolution_bench.py', 'v7_color_wire.py', 'v7_color_transforms.py',
        'v7_color_metrics.py', 'v7_color_corpus.py', 'v7_live.py',
        'v7_gl_viewer.py', 'v7_send_gui.py', 'v7_torture_matrix.py')]
    files += sorted((ROOT/'animation_modem').glob('v7*.py'))
    files += sorted((ROOT/'dct_kernels').glob('*.py'))
    files += sorted((ROOT/'test_modem_v7').glob('*.py'))
    files += [ROOT/'animation_modem/transport3.py']
    return {'revision': revision, 'tracked_diff_sha256': hashlib.sha256(diff).hexdigest(),
            'benchmark_files': {str(p.relative_to(ROOT)): digest_file(p) for p in files}}


def config_id(config):
    return '-'.join((config['profile'], config['candidate'], config['kernel'], config['split'],
                     config.get('parameter_variant', 'default')))


def config(candidate, profile, kernel='viewer_solve', split='current', params=None):
    return {'candidate': candidate, 'profile': profile, 'kernel': kernel,
            'split': split, 'kernel_params': params or {},
            'parameter_variant': 'default' if not params else hashlib.sha256(
                json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]}


def build_models(args, out, parent=None):
    """Freeze fitted statistics and PCA training; tuning itself is wire-only."""
    folder = out/'models'
    folder.mkdir()
    if parent is not None:
        old = parent/'models'
        for file in old.iterdir():
            if file.is_file():
                shutil.copy2(file, folder/file.name)
        training = json.loads((folder/'training.json').read_text())
        for asset in training['assets']:
            if digest_file(ROOT/asset['path']) != asset['sha256']:
                raise ValueError('training asset changed since model fitting')
        return json.loads((folder/'catalog.json').read_text())
    training = indices('train', args.training_images)
    pixels = []
    for index in training:
        rgb = face_image(index).reshape(-1, 3)
        selection = np.rint(np.linspace(0, len(rgb)-1, min(4096, len(rgb)))).astype(int)
        pixels.append(rgb[selection]/255)
    pixels = np.concatenate(pixels)
    catalog = {}
    for name in args.transforms:
        if name == 'pillow-ycbcr':
            catalog[name] = {'transform': ColorTransform(name).record(), 'statistics': None}
            fitted_name = 'pillow-ycbcr-refit'
        else:
            fitted_name = name
        transform = (ColorTransform.fit_pca(name, pixels) if name.startswith('pca')
                     else ColorTransform(name))
        print(f'Fitting {fitted_name} on {len(training)} training images', flush=True)
        mean, variance = fit_statistics((face_image(i) for i in training), transform)
        filename = fitted_name+'.npz'
        np.savez_compressed(folder/filename, mean=mean, variance=variance)
        catalog[fitted_name] = {'transform': transform.record(), 'statistics': filename,
                                'sha256': digest_file(folder/filename)}
    json_write(folder/'catalog.json', catalog)
    json_write(folder/'training.json', {
        'indices': training, 'partition': PARTITIONS['train'],
        'pca_pixels_per_image': 4096, 'pca_whitening': False,
        'statistics_preparation': 'native transform, production block/decimation geometry, no kernel',
        'assets': [{'path': str(face_path(i).relative_to(ROOT)),
                    'sha256': digest_file(face_path(i))} for i in training]})
    return catalog


def make_wire(configuration, layout, catalog, out):
    entry = catalog[configuration['candidate']]
    transform = ColorTransform.from_record(entry['transform'])
    statistics = None
    if entry['statistics']:
        path = out/'models'/entry['statistics']
        if digest_file(path) != entry['sha256']:
            raise ValueError('frozen training statistics hash mismatch')
        with np.load(path, allow_pickle=False) as data:
            statistics = data['mean'], data['variance']
    wire = ColorWire(configuration['profile'], layout, transform,
                     configuration['kernel'], configuration['split'], statistics,
                     configuration['kernel_params'])
    identity = config_id(configuration)+'-'+layout.replace(':', 'x')
    json_write(out/'models'/(identity+'.json'), wire.record())
    if wire.tables is not None:
        np.savez_compressed(out/'models'/(identity+'.npz'), **wire.tables)
    return wire


def select_candidates(summary, count):
    """Validation screening order, not a final weighted quality score."""
    groups = {}
    for trial in summary:
        key = (trial['profile'], trial['candidate'], trial['configuration'])
        groups.setdefault(key, []).append(trial)
    ranked = {profile: [] for profile in PROFILES}
    for (profile, candidate, identifier), trials in groups.items():
        contrast, correlations, ssim = [], [], []
        shown = frames = 0
        for trial in trials:
            metrics = trial['steady'] or trial['all']
            shown += metrics['pictures_shown']
            frames += metrics['frames']
            for key, value in metrics.items():
                if key.endswith('_faithful_contrast_median') and '_pitch' in key:
                    contrast.append(value)
                if '_band' in key and key.endswith('_correlation_median'):
                    correlations.append(value)
            if metrics.get('ssim_median') is not None:
                ssim.append(metrics['ssim_median'])
        score = (shown/max(frames, 1), float(np.median(contrast)) if contrast else -1,
                 float(np.median(correlations)) if correlations else -1,
                 float(np.median(ssim)) if ssim else -1)
        ranked[profile].append((score, candidate, identifier))
    result = {}
    for profile, entries in ranked.items():
        entries.sort(reverse=True)
        seen = set()
        chosen = []
        for score, candidate, identifier in entries:
            if candidate in ('pillow-ycbcr', 'gray') or candidate in seen:
                continue
            seen.add(candidate)
            if len(chosen) < count:
                chosen.append({'candidate': candidate, 'configuration': identifier,
                               'validation_order_fields': score})
        result[profile] = chosen
    return result


def configurations(args, catalog, parent):
    baselines = [config('pillow-ycbcr', p, default_kernel_for_profile(p)) for p in args.profiles]
    if parent is None:
        return baselines+[config(name, p) for p in args.profiles
                          for name in catalog if name != 'pillow-ycbcr']
    manifest = json.loads((parent/'manifest.json').read_text())
    previous = json.loads((parent/'summary.json').read_text())
    lookup = {config_id(c): c for c in manifest['configurations']}
    if args.stage == 'tune':
        selected = select_candidates(previous, args.top)
        output = baselines.copy()
        for profile in args.profiles:
            names = {row['candidate'] for row in selected[profile]}
            names.add('pillow-ycbcr-refit')
            if 'gray' in catalog:
                names.add('gray')
            for name in sorted(names):
                if name not in catalog:
                    continue
                # Equal finite tuning grid for every candidate, including YCbCr.
                for kernel in args.kernels:
                    for split in args.splits:
                        output.append(config(name, profile, kernel, split))
                # Two declared viewer-solve variants, same budget for everyone.
                if 'viewer_solve' in args.kernels:
                    for lam in (.25, .75):
                        output.append(config(name, profile, 'viewer_solve', 'current',
                                             {'lam': lam}))
        return output
    selected = select_candidates(previous, args.top)
    output = baselines.copy()
    for profile in args.profiles:
        chosen = [row['configuration'] for row in selected[profile]]
        # Always include the equally tuned YCbCr comparison and grayscale.
        for name in ('pillow-ycbcr-refit', 'gray'):
            subset = [row for row in previous if row['profile'] == profile and row['candidate'] == name]
            if subset:
                forced = select_candidates([dict(row, candidate='forced') for row in subset], 1)
                if forced[profile]:
                    chosen.append(forced[profile][0]['configuration'])
        for identifier in chosen:
            if identifier in lookup and lookup[identifier] not in output:
                output.append(lookup[identifier])
    return output


def write_csv(path, rows):
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def as_image(rgb):
    return Image.fromarray(np.uint8(np.rint(np.clip(rgb, 0, 1)*255)))


def comparison_frame(source, baseline, received, label):
    images = [as_image(source)]
    for picture in (baseline, received):
        if picture is None:
            placeholder = Image.new('RGB', images[0].size, (70, 70, 70))
            ImageDraw.Draw(placeholder).text((10, 10), 'No decoded picture yet', fill='white')
            images.append(placeholder)
        else:
            images.append(as_image(picture))
    w, h = images[0].size
    canvas = Image.new('RGB', (w*3, h+32), (30, 30, 30))
    for i, image in enumerate(images):
        canvas.paste(image, (i*w, 32))
    ImageDraw.Draw(canvas).text((8, 8), 'SOURCE | DEFAULT BASELINE | CANDIDATE   '+label, fill='white')
    return canvas


class VideoWriter:
    def __init__(self, path, size):
        self.process = subprocess.Popen([
            'ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'rawvideo',
            '-pixel_format', 'rgb24', '-video_size', f'{size[0]}x{size[1]}',
            '-framerate', str(1/PACKET_SECONDS), '-i', 'pipe:0', '-an',
            '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libx264',
            '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', str(path)],
            stdin=subprocess.PIPE)

    def write(self, frame):
        self.process.stdin.write(np.asarray(frame, np.uint8).tobytes())

    def close(self):
        self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError('comparison movie encoding failed')


def encode_exposure(wire, exposure):
    # Compile/warm without including those calls in the timed audio stream.
    for _ in range(2):
        wire.encode_packet(exposure.image(0), 1, 0)
    packets, elapsed, levels = [], [], []
    count = len(exposure.source_indices)
    for packet in range(count+2):
        rgb = exposure.image(min(packet, count-1))
        started = time.perf_counter()
        audio = wire.encode_packet(rgb, packet+1, packet)
        elapsed.append((time.perf_counter()-started)*1000)
        packet_audio = adapt_packet_for_output(audio, RATE)
        packets.append(packet_audio)
        body = audio[v7.PULSE.SYNC_LEN:v7.PULSE.SYNC_LEN+v7.FRAME]
        ceiling = min(np.max(np.abs(audio[:v7.PULSE.SYNC_LEN])),
                      np.max(np.abs(audio[-24:])))*10**(-1.5/20)
        levels.append({'packet': packet, 'body_rms': float(np.sqrt(np.mean(body*body))),
                       'body_peak': float(np.max(np.abs(body))),
                       'body_ceiling_utilization': float(np.max(np.abs(body))/max(ceiling, 1e-12)),
                       'header_peak': float(np.max(np.abs(audio[:v7.PULSE.SYNC_LEN]))),
                       'packet_peak': float(np.max(np.abs(audio)))})
    return np.concatenate(packets), elapsed[:count], levels[:count]


def score_trial(wire, exposure, audio, case, seed, encode_ms, levels,
                directory, baseline_directory, args):
    damaged = impair(audio, case, seed=seed)
    received, info, decode_seconds = wire.decode(damaged, RATE)
    decoded = {}
    count = len(exposure.source_indices)
    unexpected = []
    for result, values in received:
        index = result.diag.get('source_index')
        if index is None or not 0 <= index < count:
            if values is not None and index is None:
                unexpected.append({'counter': result.counter, 'reason': 'missing source index'})
            continue
        # Never align quality by receiver iteration order: losses create gaps.
        decoded[int(index)] = result, values
    directory.mkdir(parents=True, exist_ok=True)
    size = raster_size(exposure.image(0), args.short_side)
    rows, frame_metrics = [], []
    last_values = None
    last_source = last_shown = None
    writer = None
    hold_start = None
    recoveries = []
    for packet in range(count):
        result, values = decoded.get(packet, (None, None))
        shown = values is not None
        if shown:
            last_values = values
            if hold_start is not None:
                recoveries.append(packet-hold_start)
                hold_start = None
        elif hold_start is None:
            hold_start = packet
        source = resize_rgb(exposure.image(packet)/255, size)
        available = last_values is not None
        row = {'packet': packet, 'source_index': exposure.source_indices[packet],
               'source_timestamp': packet*PACKET_SECONDS,
               'phase': 'settling' if packet < 7 else 'steady',
               'shown': shown, 'available': available,
               'valid': result is not None and result.status == 'received',
               'receiver_status': result.status if result is not None else 'missing',
               'encode_ms': encode_ms[packet], **levels[packet]}
        native = display = None
        if available:
            native = render(wire, last_values, size, display=False)
            display = render(wire, last_values, size, display=True)
            for name, image in (('native', native), ('display', display)):
                metrics = picture_metrics(source, image, exposure.movie is None)
                if exposure.name.startswith('v7_pixel_motion_'):
                    metrics.update(bar_metrics(source, image))
                row.update({name+'_'+key: value for key, value in metrics.items()})
            row.update({key.removeprefix('display_'): value
                        for key, value in row.items() if key.startswith('display_')})
            if shown:
                row['decoded_only_ssim'] = row['ssim']
                row['decoded_only_psnr'] = row['psnr']
            if last_source is not None and last_shown is not None:
                row['temporal_delta_mse'] = float(np.mean(((np.clip(display, 0, 1)-last_shown)-
                                                          (source-last_source))**2))
            last_shown = np.clip(display, 0, 1)
        last_source = source
        baseline = None
        baseline_file = baseline_directory/f'{packet:05d}.png'
        if wire.production:
            if display is not None:
                baseline_directory.mkdir(parents=True, exist_ok=True)
                as_image(display).save(baseline_file)
                baseline = display
        elif baseline_file.exists():
            with Image.open(baseline_file) as image:
                baseline = np.asarray(image, float)/255
        label = f'{exposure.name} packet={packet} {case.name} seed={seed}'
        if args.videos or packet in (0, 7, count//2, count-1):
            frame = comparison_frame(source, baseline, display, label)
            if packet in (0, 7, count//2, count-1):
                frame.save(directory/f'comparison-{packet:05d}.png')
            if args.videos:
                if writer is None:
                    writer = VideoWriter(directory/'comparison.mp4', frame.size)
                writer.write(frame)
        rows.append(row)
        frame_metrics.append((row.get('ssim', -1), packet))
    if writer is not None:
        writer.close()
    write_csv(directory/'frames.csv', rows)
    json_write(directory/'frames.json', rows)
    worst = [packet for _, packet in sorted(frame_metrics)[:3]]
    held_values = None
    for packet in range(count):
        values = decoded.get(packet, (None, None))[1]
        if values is not None:
            held_values = values
        if packet not in worst:
            continue
        source = resize_rgb(exposure.image(packet)/255, size)
        image = render(wire, held_values, size, True) if held_values is not None else None
        baseline_file = baseline_directory/f'{packet:05d}.png'
        baseline = None
        if baseline_file.exists():
            with Image.open(baseline_file) as baseline_image:
                baseline = np.asarray(baseline_image, float)/255
        comparison_frame(source, baseline, image, f'worst packet={packet}').save(
            directory/f'worst-{packet:05d}.png')
    curves = []
    for orientation in ('vertical', 'horizontal'):
        for pitch in (8, 6, 4, 3, 2):
            key = f'{orientation}_pitch{pitch}'
            selected = [row for row in rows if key+'_contrast' in row]
            if selected:
                curves.append({'orientation': orientation, 'pitch': pitch,
                               'cycles_per_picture': selected[0][key+'_cycles_per_picture'],
                               'contrast_median': float(np.median([row[key+'_contrast'] for row in selected])),
                               'faithful_contrast_median': float(np.median([
                                   row[key+'_faithful_contrast'] for row in selected])),
                               'resolved_fraction': sum(row.get(key+'_resolved', False)
                                                        for row in rows)/count})
    if curves:
        write_csv(directory/'resolution-curves.csv', curves)
        chart = Image.new('RGB', (720, 360), 'white')
        draw = ImageDraw.Draw(chart)
        draw.text((12, 8), 'Faithfully recovered contrast (failed fits = 0)', fill='black')
        draw.line((50, 35, 50, 310, 680, 310), fill='black', width=2)
        for orientation, color in (('vertical', 'blue'), ('horizontal', 'red')):
            data = [row for row in curves if row['orientation'] == orientation]
            points = [(50+int(630*(row['cycles_per_picture']-10)/38),
                       310-int(250*min(row['faithful_contrast_median'], 1.1))) for row in data]
            draw.line(points, fill=color, width=3)
            for point in points:
                draw.ellipse((point[0]-3, point[1]-3, point[0]+3, point[1]+3), fill=color)
            draw.text((480, 20 if orientation == 'vertical' else 40), orientation, fill=color)
        draw.text((180, 330), 'Spatial frequency: cycles per picture axis (10..48)', fill='black')
        chart.save(directory/'resolution-curves.png')
    json_write(directory/'diagnostics.json', {
        'decoder_info': info, 'unmatched_results': unexpected,
        'worst_packet_indices': worst, 'recovery_packets': recoveries,
        'decode_seconds': decode_seconds,
        'decode_ms_per_transmitted_packet': decode_seconds*1000/(count+2)})
    if args.audio:
        wavfile.write(directory/'emitted.wav', RATE, np.float32(audio))
        wavfile.write(directory/'received.wav', RATE, np.float32(damaged))
    return {'all': summarize(rows),
            'settling': summarize(rows[:7]),
            'steady': summarize(rows[7:]) if count > 7 else None,
            'encode_p50_ms': float(np.median(encode_ms)),
            'encode_p95_ms': float(np.percentile(encode_ms, 95)),
            'encode_deadline_misses': sum(ms > PACKET_SECONDS*1000 for ms in encode_ms),
            'decode_ms_per_transmitted_packet': decode_seconds*1000/(count+2),
            'recovery_packets': recoveries}


def paired_report(summary):
    baselines = {(r['profile'], r['exposure'], r['case'], r['seed']): r
                 for r in summary if r['candidate'] == 'pillow-ycbcr'}
    tuned = {}
    for profile in PROFILES:
        subset = [r for r in summary if r['profile'] == profile and r['candidate'] == 'pillow-ycbcr-refit']
        selected = select_candidates([dict(r, candidate='forced') for r in subset], 1)[profile]
        if selected:
            identifier = selected[0]['configuration']
            tuned[profile] = {(r['exposure'], r['case'], r['seed']): r for r in subset
                              if r['configuration'] == identifier}
    output = []
    for row in summary:
        if row['candidate'] == 'pillow-ycbcr':
            continue
        comparisons = [('default', baselines.get((row['profile'], row['exposure'], row['case'], row['seed']))),
                       ('tuned-ycbcr', tuned.get(row['profile'], {}).get(
                           (row['exposure'], row['case'], row['seed'])))]
        for kind, baseline in comparisons:
            if baseline is None or baseline['configuration'] == row['configuration']:
                continue
            a, b = row['steady'] or row['all'], baseline['steady'] or baseline['all']
            differences = {key: value-b[key] for key, value in a.items()
                           if key in b and isinstance(value, (int, float))}
            stability = all(row['all'][key] >= .95*baseline['all'][key]
                            for key in ('pictures_shown', 'valid_packets'))
            output.append({'configuration': row['configuration'], 'profile': row['profile'],
                           'candidate': row['candidate'], 'exposure': row['exposure'],
                           'case': row['case'], 'seed': row['seed'], 'baseline': kind,
                           'baseline_configuration': baseline['configuration'],
                           'within_5pct_delivery': stability,
                           'clean_interior_gap_free': row['all']['interior_picture_gaps'] == 0,
                           'deltas': differences})
    return output


def bootstrap_report(pairs):
    """Paired uncertainty over whole exposures, not adjacent frames."""
    grouped = {}
    for pair in pairs:
        grouped.setdefault((pair['configuration'], pair['case'], pair['baseline']), []).append(pair)
    report = []
    for (identifier, case, baseline), rows in grouped.items():
        blocks = {}
        for row in rows:
            blocks.setdefault(row['exposure'], []).append(row['deltas'])
        fields = set().union(*(r['deltas'].keys() for r in rows))
        statistics = {}
        rng = np.random.default_rng(2026)
        for field in sorted(fields):
            values = [np.mean([b[field] for b in block if field in b])
                      for block in blocks.values() if any(field in b for b in block)]
            if not values:
                continue
            values = np.asarray(values)
            estimate = np.mean(values)
            if len(values) >= 2:
                draws = np.mean(rng.choice(values, (1000, len(values))), axis=1)
                interval = np.percentile(draws, [2.5, 97.5]).tolist()
            else:
                interval = None
            statistics[field] = {'paired_mean_delta': float(estimate),
                                 'exposure_blocks': len(values), 'ci95': interval}
        report.append({'configuration': identifier, 'case': case, 'baseline': baseline, 'metrics': statistics})
    return report


def write_report(out, summary, errors, limited):
    lines = ['# V7 color-resolution wire benchmark', '',
             '**Limited/smoke run: no final resolution claim.**' if limited else
             'Wire-backed results; inspect detail curves, artifacts and delivery together.', '',
             '| Configuration | Exposure | Channel | Shown | Finest V/H pitch | Hair detail correlation | Display SSIM |',
             '|---|---|---|---:|---|---:|---:|']
    for row in summary:
        metrics = row['steady'] or row['all']
        ssim = metrics.get('ssim_median')
        score = f'{ssim:.4f}' if ssim is not None else 'unavailable'
        pitches = '/'.join(str(metrics.get(axis+'_finest_pitch_median', '—'))
                           for axis in ('vertical', 'horizontal'))
        detail = metrics.get('hair_band2_correlation_median')
        detail = f'{detail:.4f}' if detail is not None else '—'
        lines.append(f"| {row['configuration']} | {row['exposure']} | {row['case']} / {row['seed']} | "
                     f"{metrics['pictures_shown']}/{metrics['frames']} | {pitches} | {detail} | {score} |")
    if errors:
        lines += ['', '## Failed configurations', '']
        lines += [f"- `{row['configuration']}` / {row['exposure']}: {row['error']}" for row in errors]
    lines += ['', '## Interpretation', '',
              '- `frames.csv/json` contains both native and display detail metrics.',
              '- Paired delivery/detail deltas: `paired.json`; whole-exposure intervals: `uncertainty.json`.',
              '- Initial unavailable pictures have no fabricated image quality score.',
              '- Holds are scored against the source intended at that presentation time.',
              '- Display uses production edge/guided/DCT reconstruction and a CPU Mitchell shader model.',
              '- Alternatives use a declared RGB-to-display-YCbCr bridge; GPU grain/dither are omitted.',
              '- The receiver is preconfigured for the frozen profile; auto-profile switching is not measured.',
              '- Synthetic channels are not real tape validation.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('screen', 'tune', 'evaluate'))
    p.add_argument('--from-run', type=Path, help='screen input for tune; tune input for evaluate')
    p.add_argument('--out', type=Path)
    p.add_argument('--resume', type=Path, help='resume an interrupted run with identical options/code')
    p.add_argument('--transforms', nargs='+', choices=NAMES, default=list(NAMES))
    p.add_argument('--profiles', nargs='+', choices=PROFILES, default=list(PROFILES))
    p.add_argument('--assets', nargs='+', choices=('stills', 'motion', 'movies'),
                   default=['stills', 'motion', 'movies'])
    p.add_argument('--training-images', type=int, default=64)
    p.add_argument('--stills', type=int, default=24)
    p.add_argument('--repeats', type=int, default=24)
    p.add_argument('--max-packets', type=int, help='truncate each exposure; marks run as limited')
    p.add_argument('--short-side', type=int, default=480)
    p.add_argument('--cases', nargs='+', choices=tuple(CASE_MAP))
    p.add_argument('--seeds', nargs='+', type=int)
    p.add_argument('--top', type=int, help='candidate families: default six for tune, three for evaluate')
    p.add_argument('--kernels', nargs='+', choices=KERNELS, default=list(KERNELS))
    p.add_argument('--splits', nargs='+', choices=tuple(SPLITS), default=list(SPLITS))
    p.add_argument('--videos', action='store_true', help='save source/default/candidate comparison movies')
    p.add_argument('--audio', action='store_true', help='save emitted and impaired float WAVs')
    p.add_argument('--fail-fast', action='store_true')
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.resume is not None and args.out is not None:
        p.error('use either --resume or --out')
    if args.stage != 'screen' and args.from_run is None:
        p.error('tune/evaluate require --from-run with frozen models')
    if args.stage == 'screen' and args.from_run is not None:
        p.error('screen fits new training models; omit --from-run')
    if min(args.training_images, args.stills, args.repeats, args.short_side) <= 0:
        p.error('image counts, repeats and raster size must be positive')
    if args.max_packets is not None and args.max_packets <= 0:
        p.error('--max-packets must be positive')
    args.top = args.top or (6 if args.stage == 'tune' else 3)
    if args.top < 1:
        p.error('--top must be positive')
    if 'pillow-ycbcr' not in args.transforms:
        args.transforms.insert(0, 'pillow-ycbcr')
    parent = args.from_run.resolve() if args.from_run is not None else None
    parent_manifest = None
    if parent is not None:
        parent_manifest = json.loads((parent/'manifest.json').read_text())
        expected = 'screen' if args.stage == 'tune' else 'tune'
        if parent_manifest['format'] != FORMAT or parent_manifest['stage'] != expected:
            p.error(f'{args.stage} requires a {expected} run')
        if parent_manifest['status'] != 'complete':
            p.error('parent run did not complete')
        if code_identity()['benchmark_files'] != parent_manifest['code']['benchmark_files']:
            p.error('benchmark implementation changed since parent run; rerun screen')
    out = (args.resume or args.out or ROOT/'tmp/v7-color-resolution'/(
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+args.stage)).resolve()
    old_manifest = None
    if args.resume is not None:
        old_manifest = json.loads((out/'manifest.json').read_text())
        if old_manifest['code']['benchmark_files'] != code_identity()['benchmark_files']:
            p.error('cannot resume after benchmark/dependency code changes')
        catalog = json.loads((out/'models/catalog.json').read_text())
    else:
        out.mkdir(parents=True, exist_ok=False)
        catalog = build_models(args, out, parent)
    configurations_list = configurations(args, catalog, parent)
    partition = 'test' if args.stage == 'evaluate' else 'validation'
    dataset = exposures(partition, ROOT/'tmp/v7-color-resolution/cache', args.assets,
                       args.stills, args.repeats, args.max_packets)
    cases = args.cases or (list(FINAL_CASES) if args.stage == 'evaluate' else
                           ['clean-96k', 'hiss-40'])
    seeds = args.seeds or (list(SEEDS) if args.stage == 'evaluate' else [SEEDS[0]])
    limited = (args.max_packets is not None or args.stills < 24 or args.repeats < 24
               or set(args.assets) != {'stills', 'motion', 'movies'}
               or args.short_side != 480
               or (args.stage == 'evaluate' and (set(cases) != set(FINAL_CASES)
                                                or set(seeds) != set(SEEDS)))
               or (parent_manifest is not None and parent_manifest['limited']))
    manifest = {'format': FORMAT, 'stage': args.stage, 'status': 'running',
                'code': code_identity(), 'limited': limited,
                'parent': str(parent) if parent else None,
                'parent_manifest_sha256': digest_file(parent/'manifest.json') if parent else None,
                'partition': partition, 'partitions': PARTITIONS,
                'configurations': configurations_list, 'cases': cases, 'seeds': seeds,
                'capture_rate': RATE, 'packet_seconds': PACKET_SECONDS,
                'source_brightness': 1, 'source_gamma': 1,
                'sender_encode_defaults': ENCODE_DEFAULTS,
                'display': display_defaults(),
                'short_side': args.short_side, 'rois': ROIS,
                'selection_order': ['delivery', 'bar contrast', 'natural-band correlation', 'SSIM'],
                'dataset': [exposure.record() for exposure in dataset]}
    if old_manifest is not None:
        for field in ('stage', 'partition', 'configurations', 'cases', 'seeds',
                      'short_side', 'dataset', 'parent_manifest_sha256'):
            # JSON normalizes tuples to lists; compare canonical encodings.
            if json.dumps(manifest[field], sort_keys=True) != json.dumps(old_manifest[field], sort_keys=True):
                p.error(f'resume options differ for {field}; use original options')
    json_write(out/'manifest.json', manifest)
    if parent_manifest is not None and args.stage == 'tune':
        if json.dumps(manifest['dataset'], sort_keys=True) != json.dumps(
                parent_manifest['dataset'], sort_keys=True):
            p.error('tune must use the same assets/counts/limits as its screen run')
    summary = (json.loads((out/'summary.json').read_text())
               if old_manifest is not None and (out/'summary.json').exists() else [])
    errors = []
    completed = {(row['configuration'], row['exposure'], row['case'], row['seed']) for row in summary}
    wire_cache = {}
    try:
        for exposure in dataset:
            for configuration in configurations_list:
                identifier = config_id(configuration)
                key = identifier, exposure.layout
                jobs = []
                for case_name in cases:
                    if configuration['profile'] == 'aspect-mono-500' and case_name in STEREO_CASES:
                        continue
                    case = CASE_MAP[case_name]
                    for seed in (seeds if case.noise_dbfs is not None else seeds[:1]):
                        if (identifier, exposure.name, case_name, seed) not in completed:
                            jobs.append((case_name, seed))
                if not jobs:
                    continue
                print(f'{identifier}: {exposure.name} ({len(exposure.source_indices)} packets)', flush=True)
                try:
                    if key not in wire_cache:
                        wire_cache[key] = make_wire(configuration, exposure.layout, catalog, out)
                    wire = wire_cache[key]
                    wire.aspect = v7.aspect_wire_code(exposure.image(0).shape[1::-1])
                    audio, encode_ms, levels = encode_exposure(wire, exposure)
                    for case_name, seed in jobs:
                        case = CASE_MAP[case_name]
                        slug = f'{case_name}-{seed}'
                        directory = out/'trials'/identifier/exposure.name/slug
                        baseline_directory = out/'baseline-frames'/configuration['profile']/exposure.name/slug
                        trial = score_trial(wire, exposure, audio, case, seed, encode_ms,
                                            levels, directory, baseline_directory, args)
                        trial.update(configuration=identifier, candidate=configuration['candidate'],
                                     profile=configuration['profile'], exposure=exposure.name,
                                     case=case_name, seed=seed)
                        summary.append(trial)
                        completed.add((identifier, exposure.name, case_name, seed))
                        json_write(out/'summary.json', summary)
                except Exception as exc:
                    errors.append({'configuration': identifier, 'exposure': exposure.name,
                                   'error': f'{type(exc).__name__}: {exc}'})
                    json_write(out/'errors.json', errors)
                    print(f'FAILED: {errors[-1]["error"]}', file=sys.stderr, flush=True)
                    if args.fail_fast:
                        raise
        pairs = paired_report(summary)
        json_write(out/'paired.json', pairs)
        json_write(out/'uncertainty.json', bootstrap_report(pairs))
        json_write(out/'selection.json', select_candidates(summary, args.top))
        json_write(out/'errors.json', errors)
        manifest['status'] = 'complete' if not errors else 'complete-with-errors'
        json_write(out/'manifest.json', manifest)
        write_report(out, summary, errors, limited)
        from tools.v7_color_gain_loss_chart import write_outputs
        write_outputs(out)
    except BaseException:
        manifest['status'] = 'failed'
        json_write(out/'manifest.json', manifest)
        raise
    print(f'Results: {out}/REPORT.md', flush=True)
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
