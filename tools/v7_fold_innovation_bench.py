#!/usr/bin/env python3
"""Fixed production aspect/fold layout, testing analog conditional innovations."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from contextlib import redirect_stdout
import io
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.v7_fold_innovation import FoldInnovationWire, METHODS
from tools.v7_fold_surface import FoldSurfaceWire, DIMENSIONS
from tools.v7_analog_frame import ProductionFrameWire
from tools.v7_analog_frame_bench import trial, reports, first_picture, PROFILES
from tools.v7_color_corpus import exposures, face_image, face_path, indices, digest_file
from tools.v7_color_resolution_bench import json_write, code_identity
from animation_modem.v7_core import adapt_packet_for_output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--training', type=int, default=32)
    parser.add_argument('--faces', type=int, default=2)
    parser.add_argument('--movie-entries', type=int, default=3)
    parser.add_argument('--short-side', type=int, default=240)
    parser.add_argument('--loops', type=int, default=3)
    parser.add_argument('--partition', choices=('validation', 'test'), default='validation')
    parser.add_argument('--save-audio', action='store_true')
    parser.add_argument('--packing', action='store_true', help='Descending dimension-reducing clean-wire sweep')
    parser.add_argument('--dimensions', nargs='+', default=[f'{d}to{k}' for d,k in DIMENSIONS])
    parser.add_argument('--strips', nargs='+', type=int, default=[4,8])
    parser.add_argument('--normalization',choices=('fold-native','arctan','linear'),default='fold-native')
    parser.add_argument('--headroom',nargs='+',type=float,default=[1.],help='Packed guest amplitude factors; keep fold normalization fixed')
    parser.add_argument('--frozen-from',type=Path,help='Freeze training/mapping settings from a completed packing run')
    args = parser.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out/'models').mkdir(exist_ok=True)
    identity = code_identity()
    identity['innovation_files'] = {str(p.relative_to(ROOT)): digest_file(p) for p in (
        Path(__file__), ROOT/'tools/v7_fold_innovation.py', ROOT/'tools/v7_analog_frame.py',
        ROOT/'tools/v7_analog_frame_bench.py', ROOT/'tools/v7_fold_surface.py',
        ROOT/'tools/v7_fold_surface_chart.py',ROOT/'tools/v7_fold_confirmation.py')}
    manifest = {'code': identity, 'args': {k: str(v) if isinstance(v, Path) else v
                                         for k,v in vars(args).items()},
                'grids_ratios_ranks': 'unchanged production aspect-fold/mono-500',
                'cases': ['clean-96k'] if args.packing else ['clean-96k', 'hiss-40'], 'first_packet_only': True}
    path = out/'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('code/configuration changed; use a new output directory')
    json_write(path, manifest)
    training_indices = indices('train', args.training)
    if args.frozen_from:
        parent=json.loads((args.frozen_from/'manifest.json').read_text())['args']
        if parent['training']!=args.training or parent['normalization']!=args.normalization:
            raise ValueError('frozen parent training or normalization differs')
        parent_training=json.loads((args.frozen_from/'training.json').read_text())
        current_training={str(face_path(i).relative_to(ROOT)):digest_file(face_path(i)) for i in training_indices}
        if parent_training!=current_training:
            raise ValueError('frozen training sources differ')
    training = [face_image(i) for i in training_indices]
    json_write(out/'training.json', {str(face_path(i).relative_to(ROOT)): digest_file(face_path(i))
                                   for i in training_indices})
    assets = exposures(args.partition, out/'cache', assets=('stills', 'movies'),
                       stills=args.faces, repeats=1)
    json_write(out/'sources.json', [e.record() for e in assets])
    if args.packing:
        # A real transformed 1-to-1 fidelity gate, independent of successful
        # acquisition/finite output. Invalid controls stop scored sweeps.
        validation=[]
        control_models={}
        for profile in PROFILES:
            for exposure in assets:
                entries=[0] if exposure.movie is None else np.rint(np.linspace(
                    0,len(exposure.source_indices)-1,args.movie_entries)).astype(int)
                for entry in sorted(set(entries)):
                    key=(profile,exposure.layout)
                    if key not in control_models:
                        control_models[key]=(ProductionFrameWire(profile,exposure.layout),
                            FoldSurfaceWire(profile,exposure.layout,1,1,4,training,normalization=args.normalization))
                    baseline,control=control_models[key]
                    source=exposure.image(entry)
                    index=exposure.source_indices[entry]
                    a=adapt_packet_for_output(baseline.encode_packet(source,1,index),96000)
                    b=adapt_packet_for_output(control.encode_packet(source,1,index),96000)
                    _,x,_=first_picture(baseline,a,index)
                    _,y,_=first_picture(control,b,index)
                    audio_error=float(np.max(abs(a-b)))
                    value_error=None if x is None or y is None else float(np.max(abs(x-y)))
                    passed=audio_error<=1e-9 and value_error is not None and value_error<=1e-9
                    validation.append({'profile':profile,'asset':exposure.name,'entry':int(entry),
                                       'audio_max_abs_error':audio_error,'decoded_max_abs_error':value_error,
                                       'passed':passed})
        json_write(out/'CONTROL-VALIDATION.json',{'normalization':args.normalization,
                   'tolerance':1e-9,'passed':all(r['passed'] for r in validation),'trials':validation})
        if not all(r['passed'] for r in validation):
            raise RuntimeError('transformed 1-to-1 fidelity gate failed; scored capacity sweep stopped')
    rows_path = out/'rows.json'
    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    done = {(r['profile'], r['config'], r['asset'], r['case']) for r in rows}
    models = {}
    packing = {}
    if args.packing:
        dimensions = []
        for text in args.dimensions:
            d,k = map(int,text.split('to'))
            if not 1 <= k <= d:
                parser.error('dimensions require inputs >= outputs >= 1')
            dimensions.append((d,k))
        for d,k in sorted(set(dimensions),key=lambda x:x[0]/x[1],reverse=True):
            for strips in args.strips:
                if strips < 2:
                    parser.error('strips must be >= 2')
                for headroom in args.headroom:
                    if not np.isfinite(headroom) or not 0 < headroom <= 1:
                        parser.error('headroom must be in (0,1]')
                    suffix='' if headroom==1 else '-amp'+format(headroom,'g').replace('.','p')
                    packing[f'surface-{d}to{k}-strips{strips}'+suffix] = (d,k,strips,headroom)
    configs = ('default', *(['identity-control'] if args.packing else []), 'production-off', 'production-linear', 'production-clip',
               *(packing if args.packing else METHODS))
    for config in configs:
        for profile in PROFILES:
            for exposure in assets:
                entries = [0] if exposure.movie is None else np.rint(np.linspace(
                    0, len(exposure.source_indices)-1, args.movie_entries)).astype(int)
                for entry in sorted(set(entries)):
                    asset = exposure.name+f'-entry{entry}'
                    key = (config, profile, exposure.layout)
                    if all((profile, config, asset, case) in done for case in manifest['cases']):
                        continue
                    wire = None
                    try:
                        if key not in models:
                            with redirect_stdout(io.StringIO()):
                                wire = (FoldSurfaceWire(profile,exposure.layout,1,1,4,training,normalization='identity')
                                        if config=='identity-control' else FoldSurfaceWire(profile,exposure.layout,*packing[config][:3],training,
                                                        normalization=args.normalization,headroom=packing[config][3])
                                        if config in packing else FoldInnovationWire(profile, exposure.layout, config, training)
                                        if config in METHODS else ProductionFrameWire(
                                            profile, exposure.layout, 'ordinary' if config == 'default'
                                            else config.removeprefix('production-')))
                            first_picture(wire, adapt_packet_for_output(wire.encode_packet(
                                exposure.image(entry), 1, exposure.source_indices[entry]), 96000),
                                exposure.source_indices[entry])
                            if args.frozen_from:
                                previous=args.frozen_from/'models'/('-'.join(key)+'.json')
                                if previous.exists():
                                    record=json.loads(previous.read_text())
                                    current=wire.record()
                                    for field in ('model_digest','mapping_digest'):
                                        if record.get(field)!=current.get(field):
                                            raise ValueError('frozen model differs: '+field)
                            json_write(out/'models'/('-'.join(key)+'.json'), wire.record())
                            models[key] = wire
                        wire = models[key]
                        construction_error = None
                    except Exception as error:
                        construction_error = type(error).__name__+': '+str(error)
                    for case in manifest['cases']:
                        trial_key = (profile, config, asset, case)
                        if trial_key in done:
                            continue
                        print(*trial_key, flush=True)
                        row = dict(zip(('profile', 'config', 'asset', 'case'), trial_key))
                        try:
                            if construction_error:
                                raise ValueError(construction_error)
                            row.update(trial(wire, exposure.image(entry), exposure.source_indices[entry],
                                case, 2026+int(entry), out/'trials'/'-'.join(trial_key), args.short_side,
                                args.loops if exposure.movie is None else 0, args.save_audio,
                                natural=exposure.movie is None,
                                resolution_target=exposure.name.startswith('v7_pixel_motion')))
                            if config in packing:
                                row.update(wire.source_diagnostics(exposure.image(entry)))
                            # Reconstruction signal/error margin measures all
                            # source-coding + channel distortion, not noise alone.
                            clean_values = ProductionFrameWire.values(wire, exposure.image(entry))
                            emitted = adapt_packet_for_output(wire.encode_packet(
                                exposure.image(entry), 1, exposure.source_indices[entry]), 96000)
                            from tools.v7_analog_frame_bench import CASE_MAP
                            from tools.v7_torture_matrix import impair
                            _, values, _ = first_picture(wire, impair(emitted, CASE_MAP[case],
                                                                      seed=2026+int(entry)),
                                                         exposure.source_indices[entry])
                            if values is not None:
                                source_coeffs = wire.codec.grid.forward(clean_values)
                                received_coeffs = wire.codec.grid.forward(values)
                                positions = wire.codec.guests[:-wire.codec.signature]
                                signal = float(np.mean(source_coeffs[positions]**2))
                                error = float(np.mean((received_coeffs[positions]-source_coeffs[positions])**2))
                                row['guest_signal_to_reconstruction_error_db'] = float(
                                    10*np.log10(max(signal, 1e-15)/max(error, 1e-15)))
                        except Exception as error:
                            row['error'] = type(error).__name__+': '+str(error)
                        rows.append(row)
                        done.add(trial_key)
                        json_write(rows_path, rows)
    reports(out, rows)
    if args.packing:
        from tools.v7_fold_surface_chart import write_chart
        write_chart(out,rows)
    from tools.v7_fold_confirmation import write_outcome
    write_outcome(out,rows)
    print('Report:', out/'REPORT.md', flush=True)


if __name__ == '__main__':
    main()
