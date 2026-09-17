#!/usr/bin/env python3
"""Combined record->tape->playback chains: distort going onto the tape first
(record levels/EQ), then add the tape + playback weaknesses. Ranked 0-worst
with review PNGs in scratch/combined/.

Render the input first, then run (from the repo root):

    .venv/bin/python main.py --mode modem --modem-dir images_modem \\
        --modem-pair 1,0 --modem-wav eqtest.wav --modem-frames 12
    .venv/bin/python utilities/combined_chain.py [wav]
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
OUT = REPO/'scratch'/'combined'
RATE = 48000
lay = V3.WIRE
coder, grids = coder_for('color-dct', None, lay)
CENTERS = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]


def soft_clip(w, drive, peak=0.95):
    return (np.tanh(drive*w)/np.tanh(drive*peak)*peak).astype(np.float32)


def hard_clip(w, level=0.6):
    return np.clip(w, -level, level).astype(np.float32)


def pre_emphasis(w):
    # record EQ: gentle treble lift going onto tape
    return minphase_eq(w, [0, 0, 0, 0, 0, 1, 2, 4, 6, 7])


def hiss(w, dbfs, seed=7):
    rng = np.random.default_rng(seed)
    return (w+rng.normal(0, 10**(dbfs/20), w.shape)).astype(np.float32)


def wow(w, comps):
    n = len(w)
    t = np.arange(n)/RATE
    dev = np.zeros(n)
    for i, (d, f) in enumerate(comps):
        dev += d*np.sin(2*np.pi*f*t+i*1.7)
    pos = np.arange(n)+np.cumsum(dev)
    k = np.arange(n)
    return np.stack([np.interp(pos, k, w[:, c])
                     for c in range(2)], axis=1).astype(np.float32)


def azimuth(w, samples):
    n = len(w)
    k = np.arange(n)
    out = w.copy()
    out[:, 1] = np.interp(k-samples, k, w[:, 1]).astype(np.float32)
    return out


def dropouts(w, ms=20, every=2.0):
    out = w.copy()
    n = len(out)
    for start in range(int(0.5*RATE), n, int(every*RATE)):
        out[start:start+int(ms*RATE/1000)] = 0
    return out


GOOD_PB = dict(eq=[0, 0, 0, 0, 0, 0, 0, -1, -2, -3], hiss=-55,
               wow=[(0.0008, 0.8)])
MID_PB = dict(eq=[1, 1, 0, 0, 0, 0, -1, -3, -5, -6], hiss=-48,
              wow=[(0.0015, 0.7)])
WORN_PB = dict(eq=[-2, -2, -2, -1, 0, 1, 0, -4, -12, -16], hiss=-45,
               wow=[(0.002, 0.9)])
PORT_PB = dict(eq=[3, 2, 0, -1, -1, -2, -4, -8, -12, -14], hiss=-42,
               wow=[(0.003, 0.6)])


def playback(w, pb, azimuth_s=0.0, extra_hiss=None):
    w = minphase_eq(w, pb['eq'])
    w = wow(w, pb['wow'])
    w = hiss(w, extra_hiss if extra_hiss else pb['hiss'])
    if azimuth_s:
        w = azimuth(w, azimuth_s)
    return w


CHAINS = [
    ('clean', lambda w: w),
    ('hot+good', lambda w: playback(soft_clip(pre_emphasis(w), 3.0),
                                    GOOD_PB)),
    ('hot+worn', lambda w: playback(soft_clip(pre_emphasis(w), 3.0),
                                    WORN_PB)),
    ('slammed+mid', lambda w: playback(hard_clip(pre_emphasis(w), 0.6),
                                       MID_PB)),
    ('slammed+good', lambda w: playback(hard_clip(pre_emphasis(w), 0.6),
                                        GOOD_PB)),
    ('slammed+portable', lambda w: playback(
        dropouts(hard_clip(pre_emphasis(w), 0.6)), PORT_PB)),
    ('whine+worn', lambda w: playback(w, WORN_PB, azimuth_s=1.5,
                                      extra_hiss=None) + np.stack(
        [np.sin(2*np.pi*15000*np.arange(len(w))/RATE)]*2,
        axis=1).astype(np.float32)*10**(-50/20)),
]


def decode_all(wire):
    rx = V3.Receiver(lay, coder)
    out = []
    for i in range(0, len(wire), 1024):
        out += rx.feed(np.asarray(wire[i:i+1024], np.float32))
    out += rx.flush()
    return [r for r in out if r.values is not None]


TIER_W = {'poor': 0, 'good': 1, 'better': 2, 'best': 3, 'none': -1}


def frame_image(values):
    img = values_image(values, grids)
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.asarray(img))
    return img.convert('RGB')


def frame_psnr(values, src_idx, lib):
    dec = np.asarray(frame_image(values).resize((80, 96)), float)/255
    src = np.asarray(lib.composite(src_idx, 1, 0).convert('RGB').resize(
        (80, 96), Image.LANCZOS), float)/255
    err = np.mean((dec-src)**2)
    return float('inf') if err <= 0 else 10*np.log10(1/err)


def main(argv=None):
    OUT.mkdir(exist_ok=True)
    lib = ModemLibrary(REPO/'images_modem')
    wav = argv[1] if argv and len(argv) > 1 else 'eqtest.wav'
    base = load_wav(wav)
    n_frames = len(base)//lay.frame
    results = []
    for tag, fn in CHAINS:
        got = decode_all(fn(base))
        ver = sum(1 for r in got if r.identity == 'verified_header')
        tiers = {}
        for r in got:
            tiers[r.tier] = tiers.get(r.tier, 0)+1
        ps = [frame_psnr(r.values, r.source_index or 0, lib) for r in got
              if r.source_index is not None]
        mean_ps = float(np.mean(ps)) if ps else None
        tier_score = (sum(TIER_W.get(t, -1)*c for t, c in tiers.items())
                      / max(len(got), 1))
        results.append(dict(tag=tag, got=got, ver=ver, tiers=tiers,
                            psnr=mean_ps, tier_score=tier_score))
    results.sort(key=lambda r: (r['ver'], r['tier_score'],
                                r['psnr'] if r['psnr'] else -1))
    strip = []
    for rank, r in enumerate(results):
        r['rank'] = rank
        ps = f"{r['psnr']:.1f}" if r['psnr'] is not None else '--'
        print(f"{rank} | {r['tag']:16s} pictures {len(r['got']):2d} "
              f"verified {r['ver']:2d} tiers {r['tiers']} psnr {ps}")
        imgs = [frame_image(v.values) for v in r['got']]
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
