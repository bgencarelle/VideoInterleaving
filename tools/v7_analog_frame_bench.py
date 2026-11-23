#!/usr/bin/env python3
"""Fresh-packet matched analog V7 trials; no audio device is opened."""
import argparse
from contextlib import redirect_stdout
import csv
import hashlib
import io
import itertools
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw
from scipy.io import wavfile
from animation_modem.v7_core import adapt_packet_for_output
from tools.v7_analog_frame import (
    AnalogFrameWire, ProductionFrameWire, GRID_FAMILIES, REPAIRS, fit, render, SPLITS)
from tools.v7_color_corpus import exposures, face_image, face_path, indices, PACKET_SECONDS, digest_file
from tools.v7_color_transforms import ColorTransform, NAMES
from tools.v7_color_metrics import picture_metrics, bar_metrics, raster_size, resize_rgb
from tools.v7_color_resolution_bench import json_write, code_identity
from tools.v7_torture_matrix import CASES, impair

PROFILES = ('aspect-fold-500', 'aspect-mono-500')
CASE_MAP = {case.name: case for case in CASES}


def first_picture(wire, audio, source_index):
    # decode() constructs and binds a new PulseState for each call.
    with redirect_stdout(io.StringIO()):
        packets, _, elapsed = wire.decode(audio)
    found = [(r, v) for r, v in packets if r.diag.get('source_index') == source_index]
    if not found:
        return None, None, elapsed
    result, values = found[0]
    return result, values, elapsed


def trial(wire, rgb, source_index, case, seed, folder, short_side, loops, save_audio,
          natural=True, resolution_target=False):
    # Warm source kernels, then measure actual encoding. Warm decoder is done
    # once before the trials. No decoded frame is reused by the measured call.
    wire.values(rgb)
    started = time.perf_counter()
    clean = adapt_packet_for_output(wire.encode_packet(rgb, 1, source_index), 96000)
    encode_ms = (time.perf_counter()-started)*1000
    audio = impair(clean, CASE_MAP[case], seed=seed)
    result, values, decode_s = first_picture(wire, audio, source_index)
    size = raster_size(rgb, short_side)
    source = resize_rgb(rgb/255, size)
    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(np.clip(source, 0, 1)*255)).save(folder/'source.png')
    row = {'shown': values is not None, 'valid': result is not None and result.status == 'received',
           'encode_ms': encode_ms, 'decode_ms': decode_s*1000,
           'encode_deadline_missed': encode_ms > PACKET_SECONDS*1000,
           'decode_deadline_missed': decode_s > PACKET_SECONDS,
           'packet_seconds': PACKET_SECONDS, 'samples': len(audio),
           'emitted_rms': float(np.sqrt(np.mean(clean*clean))),
           'emitted_peak': float(np.max(abs(clean))),
           'first_packet_only': True, 'source_index': source_index,
           'received_source_index': None if result is None else result.diag.get('source_index')}
    for display in (False, True):
        if values is None:
            continue
        received = render(wire, values, size, display)
        prefix = 'display_' if display else 'raw_'
        row.update({prefix+k: v for k, v in picture_metrics(source, received, natural).items()})
        row.update({prefix+k: v for k, v in bar_metrics(source, received).items()}
                   if resolution_target else {})
        Image.fromarray(np.uint8(np.clip(received, 0, 1)*255)).save(
            folder/('display.png' if display else 'decoded.png'))
    if save_audio:
        wavfile.write(folder/'emitted.wav', 96000, clean.astype(np.float32))
        wavfile.write(folder/'impaired.wav', 96000, audio.astype(np.float32))
    if loops:
        # A continuous loop AND fresh-state decoding of each impaired member.
        # Neither path has temporal fusion enabled.
        packets = [adapt_packet_for_output(wire.encode_packet(rgb, i+1, source_index), 96000)
                   for i in range(loops)]
        loop_audio = impair(np.concatenate(packets), CASE_MAP[case], seed=seed)
        with redirect_stdout(io.StringIO()):
            continuous, _, _ = wire.decode(loop_audio)
        independent = [first_picture(wire, loop_audio[i*len(clean):(i+1)*len(clean)],
                                     source_index)[1] for i in range(loops)]
        continuous_values = [v for r, v in continuous if v is not None and
                             r.diag.get('source_index') == source_index]
        row['loop_independent_shown'] = sum(v is not None for v in independent)
        row['loop_continuous_shown'] = len(continuous_values)
        row['loop_packets'] = loops
        row['loop_fresh_continuous_max_abs_difference'] = (
            max(float(np.max(abs(a-b))) for a, b in zip(independent, continuous_values))
            if len(continuous_values) == loops and all(v is not None for v in independent)
            else None)
    return row


def reports(out, rows):
    # Paired deltas retain per-frame evidence in rows.json and passing curves.
    controls = {(r['profile'], r['asset'], r['case']): r for r in rows
                if r['config'] == 'default' and 'error' not in r}
    metrics = {'SSIM': ('display_ssim', 1), 'color ΔE': ('display_delta_e76', -1),
               'detail corr.': ('display_eyes_band1_correlation', 1),
               'edge ringing': ('display_vertical_edge_overshoot', -1),
               'delivery': ('shown', 1), 'encode ms': ('encode_ms', -1)}
    summary = []
    for key in sorted({(r['profile'], r['case'], r['config']) for r in rows}):
        group = [r for r in rows if (r['profile'], r['case'], r['config']) == key]
        item = dict(zip(('profile', 'case', 'config'), key))
        item.update(trials=len(group), errors=sum('error' in r for r in group))
        for label, (metric, direction) in metrics.items():
            differences = []
            for r in group:
                base = controls.get((r['profile'], r['asset'], r['case']), {})
                if r.get(metric) is not None and base.get(metric) is not None:
                    differences.append(direction*(float(r[metric])-float(base[metric])))
            item[label] = float(np.mean(differences)) if differences else None
            item[label+' n'] = len(differences)
        summary.append(item)
    json_write(out/'summary.json', summary)
    if summary:
        with (out/'GAIN-LOSS.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=summary[0].keys())
            writer.writeheader()
            writer.writerows(summary)
    curves = []
    target_keys = sorted({k for r in rows for k in r
                          if k.startswith('display_') and k.endswith('_resolved')})
    for r in rows:
        if r['asset'].startswith('v7_pixel_motion'):
            for k in target_keys:
                # Delivery failures remain in the passing-fraction denominator.
                # Configuration errors are separately identified, not silently
                # omitted from an otherwise flattering resolution curve.
                v = bool(r.get(k, False))
                prefix = k[:-9]
                curves.append({**{x: r[x] for x in ('profile', 'case', 'config', 'asset')},
                               'target': prefix, 'passed': v,
                               'picture_available': r.get('shown', False),
                               'configuration_error': r.get('error'),
                               'contrast': r.get(prefix+'_contrast'),
                               'phase': r.get(prefix+'_phase'),
                               'residual': r.get(prefix+'_residual')})
    json_write(out/'resolution-curves.json', curves)
    from tools.v7_color_gain_loss_chart import font
    for profile, case in sorted({(r['profile'], r['case']) for r in summary}):
        group = [r for r in summary if r['profile'] == profile and r['case'] == case
                 and r['config'] != 'default']
        image = Image.new('RGB', (1260, 130+40*len(group)), '#f6f7f8')
        draw = ImageDraw.Draw(image)
        draw.text((16, 12), profile+' / '+case+' — paired mean gains', fill='#17212b', font=font(20, True))
        draw.text((16, 43), 'Green = gain; red = loss; each cell shows its paired measurement count (n).',
                  fill='#53606c', font=font(14))
        for j, label in enumerate(metrics):
            draw.text((365+j*145, 76), label, fill='#17212b', font=font(15, True))
        for i, r in enumerate(group):
            y = 105+i*40
            draw.text((16, y+8), r['config'], fill='#17212b', font=font(12))
            for j, label in enumerate(metrics):
                v = r[label]
                color = '#e7ebef' if v is None else '#d6eedf' if v > 0 else '#f2d6d4' if v < 0 else 'white'
                x = 365+j*145
                draw.rectangle((x, y, x+140, y+36), fill=color)
                text = 'unavailable' if v is None else f'{v:+.4f} (n={r[label+" n"]})'
                draw.text((x+5, y+10), text, fill='#17212b', font=font(12))
        image.save(out/f'GAIN-LOSS-{profile}-{case}.png')
        # Passing fractions stay pitch-specific: coarse-target wins cannot
        # conceal a failure to resolve the fine targets.
        subset = [r for r in curves if r['profile'] == profile and r['case'] == case]
        if subset:
            targets = sorted({r['target'] for r in subset},
                             key=lambda k: (k.split('_pitch')[0], int(k.split('pitch')[1])))
            configs = sorted({r['config'] for r in subset})
            image = Image.new('RGB', (365+125*len(targets), 120+40*len(configs)), '#f6f7f8')
            draw = ImageDraw.Draw(image)
            draw.text((16, 12), profile+' / '+case+' — resolution passing fractions',
                      fill='#17212b', font=font(20, True))
            draw.text((16, 44), 'Pitch: smaller is finer. Cells show passing percent and tested-frame count.',
                      fill='#53606c', font=font(14))
            for j, target in enumerate(targets):
                draw.text((365+125*j, 76), target.removeprefix('display_'), fill='#17212b', font=font(11))
            for i, config in enumerate(configs):
                y = 105+40*i
                draw.text((16, y+8), config, fill='#17212b', font=font(12))
                for j, target in enumerate(targets):
                    members = [r for r in subset if r['config'] == config and r['target'] == target]
                    fraction = np.mean([r['passed'] for r in members]) if members else None
                    x = 365+125*j
                    draw.rectangle((x, y, x+120, y+36), fill='#e7ebef' if fraction is None else
                                   (int(245-45*fraction), int(215+25*fraction), int(215+5*fraction)))
                    text = 'unavailable' if fraction is None else f'{fraction:.0%} (n={len(members)})'
                    draw.text((x+5, y+10), text, fill='#17212b', font=font(12))
            image.save(out/f'RESOLUTION-{profile}-{case}.png')
    lines = ['# Frame-independent analog V7 results', '',
             'Every scored picture is from one audio packet decoded with fresh state.',
             'Positive chart values mean improvement. Color uses ΔE units, delivery uses',
             'fractions, timing uses milliseconds; columns are not interchangeable.',
             'Paired measurement counts are in GAIN-LOSS.csv. Missing pictures are',
             'delivery failures, never black-frame substitutes. These are synthetic',
             'channel measurements, not tape validation.', '',
             f'Trials: {len(rows)}; configuration/trial errors: {sum("error" in r for r in rows)}.', '',
             'Resolution passing/contrast/phase evidence: `resolution-curves.json`.',
             'Per-frame data and deadline misses: `rows.json`.',
             'Source, decoded and display examples: `trials/`.', '']
    lines += [f'![{p.stem}]({p.name})' for pattern in ('GAIN-LOSS-*.png', 'RESOLUTION-*.png')
              for p in sorted(out.glob(pattern))]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--candidates', nargs='+', choices=NAMES, default=list(NAMES))
    parser.add_argument('--grids', nargs='+', choices=GRID_FAMILIES, default=list(GRID_FAMILIES))
    parser.add_argument('--splits', nargs='+', choices=SPLITS, default=list(SPLITS))
    parser.add_argument('--repairs', nargs='+', choices=REPAIRS, default=['off'])
    parser.add_argument('--profiles', nargs='+', choices=PROFILES, default=list(PROFILES))
    parser.add_argument('--cases', nargs='+', choices=CASE_MAP, default=['clean-96k', 'hiss-40'])
    parser.add_argument('--partition', choices=('validation', 'test'), default='validation')
    parser.add_argument('--training', type=int, default=16)
    parser.add_argument('--faces', type=int, default=2)
    parser.add_argument('--movie-entries', type=int, default=3)
    parser.add_argument('--short-side', type=int, default=240)
    parser.add_argument('--loops', type=int, default=0)
    parser.add_argument('--save-audio', action='store_true')
    args = parser.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out/'models').mkdir(exist_ok=True)
    identity = code_identity()
    identity['analog_files'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), ROOT/'tools/v7_analog_frame.py',
                                          *sorted((ROOT/'tools/v7_analog_kernels').glob('*.py'))]}
    manifest = {'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'code': identity, 'frame_independent': True, 'rate': 96000}
    if (out/'manifest.json').exists():
        if json.loads((out/'manifest.json').read_text()) != manifest:
            raise ValueError('run configuration/code changed; use a new output directory')
    else:
        json_write(out/'manifest.json', manifest)
    rows = json.loads((out/'rows.json').read_text()) if (out/'rows.json').exists() else []
    done = {(r['profile'], r['config'], r['asset'], r['case']) for r in rows}
    assets = exposures(args.partition, out/'cache', assets=('stills', 'movies'),
                       stills=args.faces, repeats=1)
    samples = []
    for exposure in assets:
        entries = [0] if exposure.movie is None else np.rint(np.linspace(
            0, len(exposure.source_indices)-1, args.movie_entries)).astype(int)
        for entry in sorted(set(entries)):
            samples.append((exposure, entry))
    json_write(out/'sources.json', [e.record() for e in assets])
    train_indices = indices('train', args.training)
    json_write(out/'training.json', {'indices': train_indices,
               'files': {str(face_path(i).relative_to(ROOT)): digest_file(face_path(i))
                         for i in train_indices}})
    pixels = np.concatenate([face_image(i).reshape(-1, 3)[::256]/255 for i in train_indices])
    transforms = {name: ColorTransform.fit_pca(name, pixels) if name.startswith('pca')
                  else ColorTransform(name) for name in args.candidates}
    statistics = {}
    models = {}
    configs = [('default', None, None, None, 'ordinary')]
    configs += [('production-'+repair, None, None, None, repair) for repair in ('off', 'linear', 'clip')]
    configs += [('-'.join((name, grid, split, repair)), name, grid, split, repair)
                for name, grid, split, repair in itertools.product(
                    args.candidates, args.grids, args.splits, args.repairs)]
    for config_id, name, grid, split, repair in configs:
        if name and (name, grid) not in statistics:
            print('Fitting', name, grid, flush=True)
            statistics[name, grid] = fit((face_image(i) for i in train_indices), transforms[name], GRID_FAMILIES[grid])
            (out/'models').mkdir(exist_ok=True)
            np.savez(out/'models'/f'{name}-{grid}.npz', mean=statistics[name, grid][0],
                     variance=statistics[name, grid][1])
            json_write(out/'models'/f'{name}.json', transforms[name].record())
        for profile in args.profiles:
            for exposure, entry in samples:
                asset = exposure.name+f'-entry{entry}'
                key = (config_id, profile, exposure.layout)
                if all((profile, config_id, asset, case) in done for case in args.cases):
                    continue
                try:
                    if key not in models:
                        wire = (ProductionFrameWire(profile, exposure.layout, repair) if name is None else
                                AnalogFrameWire(profile, exposure.layout, transforms[name], statistics[name, grid],
                                                grid=grid, split=split, repair=repair))
                        # Compile/warm with a packet, then discard receiver state.
                        first_picture(wire, adapt_packet_for_output(wire.encode_packet(exposure.image(entry), 1,
                                      exposure.source_indices[entry]), 96000), exposure.source_indices[entry])
                        json_write(out/'models'/('-'.join(key)+'.json'), wire.record())
                        models[key] = wire
                    wire = models[key]
                    construction_error = None
                except Exception as error:
                    construction_error = type(error).__name__+': '+str(error)
                for case in args.cases:
                    trial_key = (profile, config_id, asset, case)
                    if trial_key in done:
                        continue
                    print(profile, config_id, asset, case, flush=True)
                    row = dict(zip(('profile', 'config', 'asset', 'case'), trial_key))
                    try:
                        if construction_error:
                            raise ValueError(construction_error)
                        row.update(trial(wire, exposure.image(entry), exposure.source_indices[entry], case,
                                         2026+entry, out/'trials'/'-'.join(trial_key), args.short_side,
                                         args.loops if exposure.movie is None else 0, args.save_audio,
                                         natural=exposure.movie is None,
                                         resolution_target=exposure.name.startswith('v7_pixel_motion')))
                    except Exception as error:
                        row['error'] = type(error).__name__+': '+str(error)
                    rows.append(row)
                    done.add(trial_key)
                    json_write(out/'rows.json', rows)
    reports(out, rows)
    print('Report:', out/'REPORT.md', flush=True)


if __name__ == '__main__':
    main()
