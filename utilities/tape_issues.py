#!/usr/bin/env python3
"""Tape-issue battery: stereo separation, random EQ spikes, filtered noise
and friends. Ranked 0-worst with review PNGs in scratch/tape_issues/.

Render the input first, then run (from the repo root):

    .venv/bin/python main.py --mode modem --modem-dir images_modem \\
        --modem-pair 1,0 --modem-wav eqtest.wav --modem-frames 12
    .venv/bin/python utilities/tape_issues.py [wav]
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import butter, sosfilt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from animation_modem import transport3 as V3
from animation_modem.imaging import values_image
from modem_bake import ModemLibrary
from utilities.eq_phase_battery import load_wav
from utilities.modem_v3_check import coder_for

REPO = Path(__file__).resolve().parent.parent
OUT = REPO/'scratch'/'tape_issues'
RATE = 48000
lay = V3.WIRE
coder, grids = coder_for('color-dct', lay)


def minphase_curve(wire, freqs, db):
    """Min-phase EQ from arbitrary (freq, dB) points (narrow spikes ok)."""
    n = len(wire)
    nfft = 1 << int(np.ceil(np.log2(n + 8192)))
    f = np.fft.rfftfreq(nfft, 1 / RATE)
    lm = np.interp(f, freqs, db, left=db[0], right=db[-1]) / 20 * np.log(10)
    c = np.fft.irfft(lm, n=nfft)
    m = nfft // 2
    cm = np.zeros(nfft)
    cm[0] = c[0]
    cm[1:m] = 2 * c[1:m]
    cm[m] = c[m]
    H = np.exp(np.fft.rfft(cm))
    out = np.fft.irfft(np.fft.rfft(wire, n=nfft, axis=0) * H[:, None],
                       n=nfft, axis=0)
    return out[:n].astype(np.float32)


def spike_eq(peaks, base=0.0):
    """peaks: [(center_hz, db, width_hz)] gaussian bumps on a flat base."""
    freqs = np.array([20, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000,
                      6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000,
                      22000], float)
    db = np.full_like(freqs, base)
    for f0, gdb, w in peaks:
        db = db+gdb*np.exp(-((freqs-f0)/w)**2)
    return freqs, db


def xtalk(wire, c):
    return np.stack([wire[:, 0]*(1-c)+wire[:, 1]*c,
                     wire[:, 1]*(1-c)+wire[:, 0]*c], axis=1).astype(np.float32)


def shaped_noise(n, kind, dbfs, seed=7):
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 1, (n, 2))
    if kind == 'white':
        pass
    elif kind == 'pinkish':
        w = sosfilt(butter(2, 3000/(RATE/2), 'low', output='sos'), w, axis=0)
    elif kind == 'rumble':
        w = sosfilt(butter(2, 250/(RATE/2), 'low', output='sos'), w, axis=0)
    w = w/np.sqrt(np.mean(w**2))*10**(dbfs/20)
    return w.astype(np.float32)


def hum(n, dbfs, base=50.0):
    t = np.arange(n)/RATE
    s = np.zeros(n)
    for k, a in enumerate([1.0, 0.4, 0.2, 0.1]):
        s = s+a*np.sin(2*np.pi*base*(k+1)*t)
    s = np.stack([s, s], axis=1)
    return (s/np.sqrt(np.mean(s**2))*10**(dbfs/20)).astype(np.float32)


def whistle(n, dbfs, f=15000.0):
    t = np.arange(n)/RATE
    s = np.sin(2*np.pi*f*t)
    s = np.stack([s, s], axis=1)
    return (s/np.sqrt(np.mean(s**2))*10**(dbfs/20)).astype(np.float32)


def dropouts(wire, ms=20, every=2.0):
    out = wire.copy()
    n = len(out)
    for start in range(int(0.5*RATE), n, int(every*RATE)):
        out[start:start+int(ms*RATE/1000)] = 0
    return out


fq, db = spike_eq([(2100, 10, 180), (5300, 8, 250)])
_, db2 = spike_eq([(900, -12, 150), (3400, 9, 200), (8700, -8, 300)])
ISSUES = [
    ('clean', lambda w: w),
    ('xtalk-05', lambda w: xtalk(w, 0.05)),
    ('xtalk-15', lambda w: xtalk(w, 0.15)),
    ('xtalk-30', lambda w: xtalk(w, 0.30)),
    ('weak-right', lambda w: (w*np.array([1.0, 0.1])).astype(np.float32)),
    ('spikes', lambda w: minphase_curve(w, fq, db)),
    ('spikes+notches', lambda w: minphase_curve(w, fq, db2)),
    ('pink-hiss-45', lambda w: (w+shaped_noise(len(w), 'pinkish', -45)).astype(np.float32)),
    ('rumble-40', lambda w: (w+shaped_noise(len(w), 'rumble', -40)).astype(np.float32)),
    ('hum-50db', lambda w: (w+hum(len(w), -50)).astype(np.float32)),
    ('whistle-15k', lambda w: (w+whistle(len(w), -50)).astype(np.float32)),
    ('dropouts', lambda w: dropouts(w)),
    ('hot-clip', lambda w: np.clip(w, -0.6, 0.6).astype(np.float32)),
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
    for tag, fn in ISSUES:
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
        print(f"{rank} | {r['tag']:14s} pictures {len(r['got']):2d} "
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
