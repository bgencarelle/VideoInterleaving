#!/usr/bin/env python3
"""Tape-deck emulation battery: realistic deck chains against a rendered WAV.

Each deck = min-phase head EQ + hiss + wow/flutter (time-varying resample) +
optional azimuth skew. Reports verify/tiers plus RGB PSNR vs source (which
separates decks the tier buckets tie), ranks worst-first, and saves review
PNGs to scratch/deck_review/.

Render the input first, then run (from the repo root):

    .venv/bin/python main.py --mode modem --modem-dir images_modem \\
        --modem-pair 1,0 --modem-wav eqtest.wav --modem-frames 12
    .venv/bin/python utilities/tape_decks.py [wav]
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from animation_modem import transport3 as V3
from animation_modem.imaging import values_image
from modem_bake import ModemLibrary
from utilities.eq_phase_battery import load_wav, minphase_eq
from utilities.modem_v3_check import coder_for

REPO = Path(__file__).resolve().parent.parent
OUT = REPO/'scratch'/'deck_review'
RATE = 48000
lay = V3.WIRE
coder, grids = coder_for('color-dct', None, lay)


# wav wobble: [(rate deviation, Hz)] sinusoidal FM, integrated to positions
def wobble(n, comps, seed=0):
    t = np.arange(n)/RATE
    dev = np.zeros(n)
    for i, (d, f) in enumerate(comps):
        dev += d*np.sin(2*np.pi*f*t+i*1.7)
    return np.arange(n)+np.cumsum(dev)


def apply_wow(wire, comps):
    n = len(wire)
    pos = wobble(n, comps)
    k = np.arange(n)
    return np.stack([np.interp(pos, k, wire[:, c])
                     for c in range(2)], axis=1).astype(np.float32)


def azimuth(wire, samples):
    # delay right channel only (head azimuth), fractional via linear interp
    n = len(wire)
    k = np.arange(n)
    out = wire.copy()
    out[:, 1] = np.interp(k-samples, k, wire[:, 1]).astype(np.float32)
    return out


DECKS = [
    ('clean', {}, None),
    ('good-chrome', dict(eq=[0, 0, 0, 0, 0, 0, 0, -1, -2, -3], hiss=-55,
                          wow=[(0.0008, 0.8), (0.0004, 7.0)]), None),
    ('mid-ferric', dict(eq=[1, 1, 0, 0, 0, 0, -1, -3, -5, -6], hiss=-48,
                         wow=[(0.0015, 0.7), (0.0006, 9.0)]), None),
    ('portable', dict(eq=[3, 2, 0, -1, -1, -2, -4, -8, -12, -14], hiss=-42,
                        wow=[(0.003, 0.6), (0.0012, 6.0), (0.0008, 11.0)]),
     None),
    ('worn-heads', dict(eq=[-2, -2, -2, -1, 0, 1, 0, -4, -12, -16],
                         hiss=-45, wow=[(0.002, 0.9)], azimuth=1.5), None),
    ('bad-azimuth', dict(eq=[0]*10, hiss=-50, wow=[(0.001, 0.8)],
                          azimuth=2.5), None),
    ('fast-2pct', dict(eq=[0]*10, hiss=-55, speed=1.02), None),
]


def run_deck(wire, spec):
    w = wire
    if spec.get('eq') is not None:
        w = minphase_eq(w, spec['eq'])
    if spec.get('speed') is not None:
        n = int(len(w)/spec['speed'])
        k = np.arange(len(w))
        w = np.stack([np.interp(np.arange(n)*spec['speed'], k, w[:, c])
                      for c in range(2)], axis=1).astype(np.float32)
    if spec.get('wow'):
        w = apply_wow(w, spec['wow'])
    if spec.get('hiss') is not None:
        rng = np.random.default_rng(7)
        w = (w+rng.normal(0, 10**(spec['hiss']/20), w.shape)).astype(
            np.float32)
    if spec.get('azimuth'):
        w = azimuth(w, spec['azimuth'])
    return w


def decode_all(wire):
    rx = V3.Receiver(lay, coder)
    out = []
    for i in range(0, len(wire), 1024):
        out += rx.feed(np.asarray(wire[i:i+1024], np.float32))
    out += rx.flush()
    return [r for r in out if r.values is not None]


TIER_W = {'poor': 0, 'good': 1, 'better': 2, 'best': 3, 'none': -1}


def frame_psnr(values, src_idx, lib):
    img = values_image(values, grids)
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.asarray(img))
    dec = np.asarray(img.convert('RGB').resize((80, 96)), float)/255
    src = np.asarray(lib.composite(src_idx, 1, 0).convert('RGB').resize(
        (80, 96), Image.LANCZOS), float)/255
    err = np.mean((dec-src)**2)
    return float('inf') if err <= 0 else 10*np.log10(1/err)


def frame_image(values):
    img = values_image(values, grids)
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.asarray(img))
    return img.convert('RGB')


def main(argv=None):
    OUT.mkdir(exist_ok=True)
    lib = ModemLibrary(REPO/'images_modem')
    wav = argv[1] if argv and len(argv) > 1 else 'eqtest.wav'
    base = load_wav(wav)
    n_frames = len(base)//lay.frame
    print(f'{len(base)} samples (~{n_frames} frames), peak '
          f'{np.max(np.abs(base)):.3f}\n')
    results = []
    for tag, spec, _ in DECKS:
        got = decode_all(run_deck(base, spec))
        ver = sum(1 for r in got if r.identity == 'verified_header')
        tiers = {}
        for r in got:
            tiers[r.tier] = tiers.get(r.tier, 0)+1
        skew = [round(float(r.extra.get('skew_samples', 0)), 3)
                for r in got[:3]]
        ps = [frame_psnr(r.values, r.source_index or 0, lib) for r in got
              if r.source_index is not None]
        mean_ps = float(np.mean(ps)) if ps else None
        tier_score = (sum(TIER_W.get(t, -1)*c for t, c in tiers.items())
                      / max(len(got), 1))
        results.append(dict(tag=tag, got=got, ver=ver, tiers=tiers,
                            skew=skew, psnr=mean_ps, tier_score=tier_score))
    # rank worst (0) -> best: verified, then tier mix, then psnr
    results.sort(key=lambda r: (r['ver'], r['tier_score'],
                                r['psnr'] if r['psnr'] else -1))
    strip = []
    for rank, r in enumerate(results):
        r['rank'] = rank
        ps = f"{r['psnr']:.1f}" if r['psnr'] is not None else '--'
        print(f"{rank} | {r['tag']:12s} verified {r['ver']:2d} "
              f"tiers {r['tiers']} psnr {ps} "
              f"skew~{r['skew'][0] if r['skew'] else None}")
        got = r['got']
        imgs = [frame_image(v.values) for v in got]
        while len(imgs) < n_frames:
            imgs.append(Image.new('RGB', imgs[0].size, (0, 0, 0)))
        w, h = imgs[0].size
        sheet = Image.new('RGB', (w*4, h*((len(imgs)+3)//4)), (0, 0, 0))
        for i, im in enumerate(imgs):
            sheet.paste(im, ((i % 4)*w, (i//4)*h))
        sheet.save(OUT/f"{rank}_{r['tag']}_{r['ver']}.png")
        strip.append(imgs[0].resize((80, 96)))
    strip_img = Image.new('RGB', (80*len(strip), 96), (0, 0, 0))
    for i, im in enumerate(strip):
        strip_img.paste(im, (i*80, 0))
    strip_img.save(OUT/'_strip_ranked_0worst.png')
    print('ranked order (0=worst):', ' | '.join(
        f"{r['rank']}={r['tag']}" for r in results))


if __name__ == '__main__':
    main(sys.argv)
