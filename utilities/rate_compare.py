#!/usr/bin/env python3
"""Definitive treble+noise rate comparison across wire geometries.

3+ wire layouts x 4 noise seeds x 32 frames. Encodes once per layout (32
distinct bake frames), reuses across seeds (EQ+noise vary). Compares verify
RATES so noise luck averages out despite different frame lengths. Slow
(~10 min): the number to re-run after any wire or decoder change touching
margins.

Run from the repo root:

    .venv/bin/python utilities/rate_compare.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from animation_modem import transport3 as V3
from animation_modem.core import Layout
from animation_modem.imaging import (image_values, plane_grids, plane_shapes,
                                     fit_shapes)
from modem_bake import ModemLibrary
from utilities.eq_phase_battery import minphase_eq

REPO = Path(__file__).resolve().parent.parent
TREBLE = [-12, -8, -4, 0, 2, 2, 0, -4, -10, -16]
FRAMES, SEEDS = 32, (0, 1, 2, 3)

LAYOUTS = [
    ('WIRE', V3.WIRE),
    # Historical counterfactuals, kept to show what each knob buys. (The old
    # WIRE-hw20 twin is gone: it is identical to WIRE since the retune.)
    ('widev3-geom', Layout(top_bin=54, image_symbols=15, name='w',
                           progressive=True, orthogonal_training=True)),
    ('WIRE-isym15', Layout(top_bin=54, image_symbols=15, name='y',
                           progressive=True, orthogonal_training=True,
                           spread_carriers=True, dense_header=True,
                           header_width=27)),
]


def build_coder(layout, profile):
    shapes = fit_shapes(plane_shapes(profile), layout.capacity)
    grids = plane_grids(profile)
    return shapes, grids, V3.SourceCoder(shapes, grids=grids)


def psnr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    err = np.mean((a-b)**2)
    return float('inf') if err <= 0 else 10*np.log10(1/err)


def main(argv=None):
    lib = ModemLibrary(REPO/'images_modem')
    # encode once per layout (content varies by frame, absolute by frame)
    encoded, srcs = {}, {}
    for tag, lay in LAYOUTS:
        shapes, grids, coder = build_coder(lay, lib.profile)
        stream = [np.asarray(V3.encode(
            image_values(lib.composite(n, 1, 0), grids), lay, coder,
            n+1, n+1, FRAMES, profile=0), np.float32) for n in range(FRAMES)]
        encoded[tag] = (lay, coder, np.concatenate(stream))
        srcs[tag] = image_values(lib.composite(FRAMES-1, 1, 0), grids)

    for tag, lay in LAYOUTS:
        lay, coder, clean = encoded[tag]
        rates, psnrs = [], []
        for seed in SEEDS:
            wire = minphase_eq(clean, TREBLE)
            rng = np.random.default_rng(100+seed)
            wire = (wire+rng.normal(0, 10**(-45/20), wire.shape)
                    ).astype(np.float32)
            rx = V3.Receiver(lay, coder, pulse_only=True, candidates=None,
                             fast=True)
            recs = []
            for s in range(0, len(wire), 8192):
                recs.extend(rx.feed(wire[s:s+8192]))
            recs.extend(rx.flush())
            ver = sum(1 for r in recs if r.identity == 'verified_header')
            good = [r for r in recs if r.values is not None]
            ps = float(np.mean([psnr(srcs[tag], r.values) for r in good])) \
                if good else None
            rates.append(ver)
            psnrs.append(ps)
        print(f'{tag:12s} verify {rates}  mean {np.mean(rates):.1f}/{FRAMES}  '
              f'psnr {[round(p, 1) if p else None for p in psnrs]}')


if __name__ == '__main__':
    main(sys.argv)
