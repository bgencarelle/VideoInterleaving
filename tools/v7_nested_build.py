"""Build the frozen nested-fold tables (``test_modem_v7/nested_tables.npz``).

One table per profile family ('mono': aspect-mono-500, 'slices': one channel
of stereo-slices) and aspect layout.  Statistics come from a folder of
general pictures, centre-cropped to each layout; nothing is fitted to one
kind of content or one link.

    .venv/bin/python tools/v7_nested_build.py --corpus tmp/sk/corpus

Each table is designed at one moderate noise level and decoded softly at
whatever noise a packet turns out to have, so there is one table per layout
and no noise setting.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from tools import v7_live, v7_sk_fold as sk, v7_sk_study as study

v7_live._ensure_test_modem_path()
from tools.v7_sk_adaptive import Adaptive

PROFILES = {'mono': 'aspect-mono-500', 'slices': 'stereo-slices'}
# Chosen on training pictures for the lowest error averaged over clean to
# heavy hiss with the soft receiver (see docs/V7_SK_VERDICT.md).
DESIGN = {'mono': dict(linear=64, kappa=20, noise=.020),
          'slices': dict(linear=600, kappa=28, noise=.020)}


def design_noise(geometry, level):
    """Design noise of each slot in units of its own signal: white wire
    noise of standard deviation ``level`` against the slot's design power.
    The receiver rescales this profile to each packet's measured noise."""
    return level/np.sqrt(geometry.power)


def fit(geometry, planes, noise, linear, kappa, legs):
    """Frozen arrays for one family and layout."""
    true_planes, paired = planes, None
    if geometry.slices:
        # Hosts are the stock slices slots: base plus a share of a detail
        # coefficient (minus on the other channel; the statistics are alike).
        slots = geometry.slots
        base, detail, ride = slots.base[0], slots.detail[0], slots.ride[0]
        flat = np.stack([plane.ravel() for plane in planes])
        # Smoothed training variances of the base and of the riding detail.
        variance = study.Statistics(planes).variance.ravel()
        paired = {'detail_position': np.asarray(detail), 'ride': np.asarray(ride, float),
                  'common': variance[base], 'differ': ride**2*variance[detail]}
        virtual = []
        for row, plane in zip(flat, planes):
            changed = row.copy()
            changed[base] = row[base]+ride*row[detail]
            changed[detail[ride > 0]] = 0.0
            virtual.append(changed.reshape(plane.shape))
        planes = virtual
    stats = study.Statistics(planes)
    candidates = geometry.guest_position
    if paired is not None:
        taken = np.zeros(geometry.rows*geometry.cols, bool)
        taken[paired['detail_position'][paired['ride'] > 0]] = True
        candidates = candidates[~taken[candidates]]
        stats_true = study.Statistics(true_planes)
    ranking = (stats_true if paired is not None else stats).variance.ravel()
    candidates = candidates[np.argsort(-ranking[candidates], kind='stable')]
    first = Adaptive(geometry, stats, noise, planes, passes=2, linear=linear, triple=0,
                     kappa=kappa, guard=1.0, guest_order=np.ascontiguousarray(candidates[0::legs]))
    fold = first.fold
    count = fold.guests
    sd = (stats_true if paired is not None else stats).sd.ravel()
    slot, _ = fold.guest_order()
    positions = np.stack([candidates[leg::legs][:count] for leg in range(legs)])
    rows, cols = geometry.rows, geometry.cols
    width, height = geometry.aspect
    u, v = positions//cols, positions % cols
    sector = np.minimum((np.arctan2(v*height, u*width)/(math.pi/2)*first.sectors).astype(np.int64),
                        first.sectors-1)
    active = first.active
    extra = {} if paired is None else {name: value[active] for name, value in paired.items()}
    return {
        **extra,
        'model_index': np.asarray(geometry.model_index)[active],
        'host_position': geometry.host_position[active],
        'host_mean': first.host_mean[active], 'host_sd': first.host_sd[active],
        'noise': np.asarray(noise)[active],
        'signature_noise': (np.asarray(noise)[first.reserved] if len(first.reserved)
                            else np.full(16, float(np.sqrt(np.mean(noise[active][-64:]**2))))),
        'level': fold.level, 'step': fold.step, 'scale': fold.scale, 'alpha': fold.alpha,
        'compander': fold.table,
        'density': sk.residual_density(np.ascontiguousarray(
            np.concatenate([first.guests(plane) for plane in planes])), fold.table),
        'member': first.member, 'host_sector': first.host_sector, 'law': first.law,
        'limits': np.asarray(first.limits, float),
        'guest_position': positions, 'guest_sd': sd[positions],
        'guest_slot': slot, 'guest_sector': sector,
        'guest_band': first.guest_band}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--corpus', default='tmp/sk/corpus')
    parser.add_argument('--out', default=str(ROOT/'test_modem_v7/nested_tables.npz'))
    parser.add_argument('--pictures', choices=('all', 'training'), default='all',
                        help="'training' keeps every second picture out, for held-out scoring")
    parser.add_argument('--layouts', nargs='+')
    parser.add_argument('--families', nargs='+', default=list(PROFILES))
    parser.add_argument('--design', nargs='+', default=[], metavar='FAMILY:KEY=VALUE',
                        help="override a design value for this build, e.g. slices:kappa=16 "
                             "mono:linear=200 (keys: linear, kappa, noise)")
    args = parser.parse_args(argv)
    for item in args.design:
        family, setting = item.split(':', 1)
        key, value = setting.split('=', 1)
        if family not in DESIGN or key not in DESIGN[family]:
            parser.error(f'unknown design setting {item!r}')
        DESIGN[family][key] = type(DESIGN[family][key])(float(value))
    from aspect_fold import LAYOUT_NAMES
    train_files, test_files = study.corpus(args.corpus)
    files = train_files+(test_files if args.pictures == 'all' else [])
    arrays, record = {}, {'pictures': [f.name for f in files], 'design': DESIGN, 'tables': {}}
    analyse = study.Analyzer(study.LATTICE)
    for family in args.families:
        for layout in args.layouts or LAYOUT_NAMES:
            geometry = study.Geometry(PROFILES[family], layout)
            planes = [analyse(study.portrait(f, geometry.aspect)) for f in files]
            design = DESIGN[family]
            table = fit(geometry, planes, design_noise(geometry, design['noise']),
                        design['linear'], design['kappa'], 2 if family == 'slices' else 1)
            digest = hashlib.sha256()
            for name in sorted(table):
                arrays[f'{family}/{layout}/{name}'] = table[name]
                digest.update(np.ascontiguousarray(table[name]).tobytes())
            record['tables'][f'{family}/{layout}'] = {
                'slots': int(len(table['host_position'])),
                'guests_per_channel': int(table['guest_position'].shape[1]),
                'channels': int(table['guest_position'].shape[0]),
                'sha256': digest.hexdigest()}
            print(family, layout, record['tables'][f'{family}/{layout}'], flush=True)
    np.savez_compressed(args.out, **arrays)
    Path(args.out).with_suffix('.json').write_text(json.dumps(record, indent=1)+'\n')


if __name__ == '__main__':
    main()
