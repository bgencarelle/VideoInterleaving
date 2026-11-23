"""Explicit, matched-table experiments on the real V7 live path.

Both ends must use the same experimental table and, for serpentine, explicitly
select the same mapping; it is wire-incompatible with legacy and is not
autodetected. Stock launchers and frozen production tables are unaffected.

    python tools/v7_nested_experiment.py --tables tmp/candidate.npz --mapping legacy send ...
    python tools/v7_nested_experiment.py --tables tmp/candidate.npz --mapping legacy receive ...
    python tools/v7_nested_experiment.py --tables tmp/candidate.npz --mapping serpentine --fold-level 2 send ...
    python tools/v7_nested_experiment.py --tables tmp/candidate.npz --mapping serpentine receive ...
"""
import argparse
import hashlib
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def install(live, tables, set_default=True):
    """Select explicit tables and adapt source analysis to their plane grids."""
    import numpy as np
    live._ensure_test_modem_path()
    import nested_fold
    from animation_modem.v7_core import SourceCoder
    with np.load(tables) as data:
        shapes = {tuple(map(int, data[key])) for key in data.files
                  if key.endswith('/luma_grid')}
        missing_grid = any(key.endswith('/host_position') and
                           key.rsplit('/', 1)[0]+'/luma_grid' not in data.files
                           for key in data.files)
        if missing_grid:
            shapes.add((96, 80))
        chroma_shapes = {tuple(map(int, data[key])) for key in data.files
                         if key.endswith('/chroma_grid')}
    if len(shapes) != 1:
        raise ValueError('an experiment needs one consistent luma grid in its tables')
    shape = shapes.pop()
    if len(chroma_shapes) > 1:
        raise ValueError('an experiment needs one consistent chroma grid in its tables')
    chroma = chroma_shapes.pop() if chroma_shapes else (48, 40)
    if set_default:
        nested_fold.TABLES = Path(tables).resolve()
    grids = (shape, chroma, chroma)
    if set_default and (shape != (96, 80) or chroma != (48, 40)):
        models = {}
        def source_model(model):
            key = (id(model), grids)
            if key not in models:
                models[key] = replace(model, coder=SourceCoder(
                    model.coder.shapes, grids=grids))
            return models[key]
        real_values = getattr(live, '_v7_nested_base_values', live._values)
        real_warm = getattr(live, '_v7_nested_base_warm', live._warm_kernels)
        live._v7_nested_base_values = real_values
        live._v7_nested_base_warm = real_warm
        def values(model, *args, **kwargs):
            return real_values(source_model(model), *args, **kwargs)
        def warm(model, *args, **kwargs):
            return real_warm(source_model(model), *args, **kwargs)
        live._values, live._warm_kernels = values, warm
    selected = Path(tables).resolve()
    digest = hashlib.sha256(selected.read_bytes()).hexdigest()
    print(f'V7 nested experiment: selected_tables={selected} '
          f'table_sha256={digest} '
          f'default_tables={Path(nested_fold.TABLES).resolve()} '
          f'luma={shape[1]}x{shape[0]} chroma={chroma[1]}x{chroma[0]} '
          'matched tables required at both ends', flush=True)
    return grids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables', type=Path, required=True)
    parser.add_argument('--mapping', choices=('legacy', 'serpentine'), default='legacy',
                        help=('experimental mapping; both ends must select the same mapping. '
                              'Serpentine is incompatible with the legacy nested-fold mapping'))
    parser.add_argument('--fold-level', type=int, default=None,
                        help=('sender fixed ladder level (-3..11), required for a '
                              'serpentine sender; receiver reads level from packets'))
    parser.add_argument('--codec-report', type=Path, default=None,
                        help='optional metadata folder for the selected codec')
    parser.add_argument('--step-factor', type=float, default=None,
                        help='matched serpentine host stair multiplier (0, 1]')
    parser.add_argument('--guest-factor', type=float, default=1.0,
                        help='matched serpentine guest normalization multiplier >=1')
    parser.add_argument('--exact-inverse', action='store_true',
                        help='invert the forward compander table directly (both ends must agree)')
    parser.add_argument('--adaptive-guest-range', action='store_true',
                        help='opt in to per-frame factors 1,2,4,8,16 with fixed host level -1')
    parser.add_argument('--luma-range-passes', type=int, default=0,
                        help='opt-in receiver legal-range reconstruction iterations (0 disables)')
    parser.add_argument('--range-host-cells', action='store_true',
                        help='allow legal continuous quantization-cell movement of carried hosts')
    parser.add_argument('--chroma-detail', action='store_true',
                        help='enable the experimental slice chroma-detail plane at both ends')
    args, remaining = parser.parse_known_args()
    if args.step_factor is None:
        if args.adaptive_guest_range:
            from tools.v7_adaptive_pipeline import (
                SERPENTINE_STEP_FACTOR, EXACT_INVERSE)
            args.step_factor = SERPENTINE_STEP_FACTOR
            args.exact_inverse = EXACT_INVERSE
        else:
            args.step_factor = 1.0
    if args.luma_range_passes < 0 or (args.range_host_cells and args.luma_range_passes == 0):
        parser.error('--range-host-cells requires positive --luma-range-passes')
    if args.luma_range_passes and args.mapping != 'serpentine':
        parser.error('--luma-range-passes requires --mapping serpentine')
    if args.chroma_detail:
        import os
        os.environ['SLICE_CHROMA_DETAIL'] = '1'
    from tools import v7_live as live
    install(live, args.tables)
    options = live.parser().parse_args(remaining)
    # The normal CLI's run_send path applies the selected profile before
    # resolving profile-specific modes. This wrapper must do the same before
    # checking nested_fold, notably for commands assembled by the sender GUI.
    live._apply_profile_option(options)
    if args.mapping == 'legacy' and args.fold_level is not None:
        parser.error('--fold-level requires --mapping serpentine')
    if args.mapping == 'legacy' and (args.step_factor != 1.0 or
                                    args.guest_factor != 1.0 or args.exact_inverse or
                                    args.adaptive_guest_range):
        parser.error('codec factors and --exact-inverse require --mapping serpentine')
    if args.mapping == 'serpentine':
        if args.adaptive_guest_range and args.fold_level is not None:
            parser.error('--adaptive-guest-range fixes host level at -1; omit --fold-level')
        if args.adaptive_guest_range and args.guest_factor != 1.0:
            parser.error('--adaptive-guest-range selects its own per-frame guest factor')
        if options.mode == 'send' and args.fold_level is None and not args.adaptive_guest_range:
            parser.error('serpentine sending requires --fold-level; receiver reads level from each packet')
        if args.fold_level is not None:
            import nested_fold
            if not nested_fold.LOWEST <= args.fold_level <= nested_fold.HIGHEST:
                parser.error(f'--fold-level must be between {nested_fold.LOWEST} and {nested_fold.HIGHEST}')
    if options.mode == 'send':
        live._resolve_send_source(options)
        if not options.nested_fold:
            parser.error('experimental sending requires a nested profile')
    if args.mapping == 'serpentine':
        if args.adaptive_guest_range:
            from tools.v7_adaptive_serpentine import install as install_codec
            install_codec(args.codec_report, step_factor=args.step_factor,
                          exact_inverse=args.exact_inverse,
                          isolated_path=args.tables)
            options.adaptive_nested_installed = True
            options.adaptive_nested_table = str(args.tables.resolve())
        else:
            from tools.v7_serpentine_codec import install as install_codec
            install_codec(args.codec_report, fixed_level=args.fold_level,
                          step_factor=args.step_factor, guest_factor=args.guest_factor,
                          exact_inverse=args.exact_inverse)
        print(f'V7 nested mapping=serpentine'
              f'{" adaptive_guest_range=factors[1,2,4,8,16] host_level=-1" if args.adaptive_guest_range else ""}'
              f'{" fixed_level=" + str(args.fold_level) if args.fold_level is not None else ""}',
              flush=True)
    if args.luma_range_passes:
        from tools.v7_range_codec import install as install_range
        install_range(args.codec_report, args.luma_range_passes, args.range_host_cells)
        print(f'V7 legal-luma-range passes={args.luma_range_passes} '
              f'host_cells={args.range_host_cells}', flush=True)
    if options.mode == 'send':
        live.run_send(options)
    elif options.mode == 'receive':
        live.run_receive(options)
    else:
        parser.error('use send or receive for the experiment')


if __name__ == '__main__':
    main()
