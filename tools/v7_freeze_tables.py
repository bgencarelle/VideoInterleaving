#!/usr/bin/env python3
"""Regenerate the frozen V7 model tables (animation_modem/v7_model_tables.npz).

The canonical V7 profile is data: a per-cell phase table plus, for every
encode filter, the coefficient means, variance table, rank order, SoftCast
gains and unit-scale level. This tool derives them once from the reference
fixture with the documented seeds (PHASE_SEED, CROP_SEED, LEVEL_SEED in
animation_modem/v7.py) and writes the file. The modem itself only loads it.

Regenerating CHANGES THE WIRE if the libraries now produce different numbers.
Do it deliberately: bump the profile, update MODEL_TABLES_SHA256, and rerun the
V7 tests and bench. `--check` compares a fresh derivation with the frozen file
without writing anything.

    .venv/bin/python tools/v7_freeze_tables.py --check
    .venv/bin/python tools/v7_freeze_tables.py --write
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from animation_modem import v7                                          # noqa: E402

KEYS = ('mu', 'lam', 'order', 'gain', 'unit_rms')


def derive():
    phase = v7.phase_table()
    arrays = {'phase': phase}
    with Image.open(v7.REFERENCE_FIXTURE) as source:
        for name in v7.ENCODING_FILTERS:
            tables = v7.derive_tables(source, name, phase)
            for key in KEYS:
                arrays[f'{name}/{key}'] = np.asarray(tables[key])
    return arrays


def compare(fresh):
    with np.load(v7.MODEL_TABLES, allow_pickle=False) as frozen:
        worst = {}
        for key, value in fresh.items():
            old = frozen[key]
            if old.shape != value.shape:
                worst[key] = float('inf')
            elif old.dtype.kind in 'iu':
                worst[key] = float(np.count_nonzero(old != value))
            else:
                worst[key] = float(np.max(np.abs(old - value)))
    return worst


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--write', action='store_true', help='write the tables file')
    ap.add_argument('--check', action='store_true', help='compare a fresh derivation')
    args = ap.parse_args(argv)
    fresh = derive()
    if args.write:
        np.savez_compressed(v7.MODEL_TABLES, **fresh)
        digest = hashlib.sha256(v7.MODEL_TABLES.read_bytes()).hexdigest()
        print(f'wrote {v7.MODEL_TABLES.relative_to(ROOT)}  sha256 {digest}')
        print('update MODEL_TABLES_SHA256 in animation_modem/v7.py to match')
    if args.check or not args.write:
        worst = compare(fresh)
        drift = {k: v for k, v in worst.items() if v != 0}
        if drift:
            print('DRIFT: this environment derives different V7 tables '
                  '(the wire still uses the frozen file):')
            for key, value in sorted(drift.items()):
                print(f'  {key}: max difference {value:g}')
            return 1
        print('no drift: a fresh derivation matches the frozen tables exactly')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
