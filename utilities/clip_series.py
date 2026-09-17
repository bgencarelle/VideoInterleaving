#!/usr/bin/env python3
"""Clipping series: hard levels, asymmetric, soft (tanh) saturation.
Reports clipped fraction (clip-discount gate), verify, pixel error and Cb
control; saves montages + strip to scratch/clip_review/.

Render the input first, then run (from the repo root):

    .venv/bin/python main.py --mode modem --modem-dir images_modem \\
        --modem-pair 1,0 --modem-wav eqtest.wav --modem-frames 12
    .venv/bin/python utilities/clip_series.py [wav]
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from animation_modem import transport3 as V3
from animation_modem.imaging import values_image
from modem_bake import ModemLibrary
from utilities.eq_phase_battery import load_wav
from utilities.modem_v3_check import coder_for

REPO = Path(__file__).resolve().parent.parent
OUT = REPO/'scratch'/'clip_review'
lay = V3.WIRE
coder, grids = coder_for('color-dct', lay)


def soft_clip(w, drive, peak=0.95):
    return (np.tanh(drive*w)/np.tanh(drive*peak)*peak).astype(np.float32)


CASES = [
    ('clean', lambda w: w),
    ('clip0.9', lambda w: np.clip(w, -0.9, 0.9).astype(np.float32)),
    ('clip0.8', lambda w: np.clip(w, -0.8, 0.8).astype(np.float32)),
    ('clip0.7', lambda w: np.clip(w, -0.7, 0.7).astype(np.float32)),
    ('clip0.6', lambda w: np.clip(w, -0.6, 0.6).astype(np.float32)),
    ('clip0.5', lambda w: np.clip(w, -0.5, 0.5).astype(np.float32)),
    ('clip0.4', lambda w: np.clip(w, -0.4, 0.4).astype(np.float32)),
    ('asym', lambda w: np.clip(w, -0.95, 0.65).astype(np.float32)),
    ('soft2', lambda w: soft_clip(w, 2.0)),
    ('soft3', lambda w: soft_clip(w, 3.0)),
]


def decode_all(wire):
    rx = V3.Receiver(lay, coder)
    out = []
    for i in range(0, len(wire), 1024):
        out += rx.feed(np.asarray(wire[i:i+1024], np.float32))
    out += rx.flush()
    return [r for r in out if r.values is not None]


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
    rng = np.random.default_rng(11)
    PX, PY = rng.integers(0, 80, 400), rng.integers(0, 96, 400)
    results = []
    for tag, fn in CASES:
        w = fn(base)
        peak = float(np.max(np.abs(w)))
        flat = float(np.mean(np.abs(w) > 0.98*peak)) if peak > 0 else 0.0
        got = decode_all(w)
        ver = sum(1 for r in got if r.identity == 'verified_header')
        errs, cbs = [], []
        for i, r in enumerate(got[:n_frames]):
            img = frame_image(r.values)
            dec = np.asarray(img.resize((80, 96)), float)/255
            src = np.asarray(lib.composite(i, 1, 0).convert('RGB').resize(
                (80, 96), Image.LANCZOS), float)/255
            errs.append(float(np.mean(np.abs(dec[PY, PX]-src[PY, PX]))))
            cbs.append(float(r.values[7680:9600].std()))
        results.append(dict(tag=tag, got=got, ver=ver,
                            px=float(np.mean(errs)) if errs else None,
                            cb=float(np.mean(cbs)) if cbs else None,
                            flat=flat))
    strip = []
    for r in results:
        print(f"{r['tag']:8s} flat={r['flat']:.2e} "
              f"{'GATE' if r['flat'] > 1e-3 else '----'} "
              f"verified {r['ver']:2d}/{n_frames} pixel {r['px']:.3f} "
              f"Cb {r['cb']:.3f}")
        imgs = [frame_image(v.values) for v in r['got']]
        while len(imgs) < n_frames:
            imgs.append(Image.new('RGB', imgs[0].size, (0, 0, 0)))
        w, h = imgs[0].size
        sheet = Image.new('RGB', (w*4, h*((len(imgs)+3)//4)), (0, 0, 0))
        for i, im in enumerate(imgs):
            sheet.paste(im, ((i % 4)*w, (i//4)*h))
        sheet.save(OUT/f"{r['tag']}_{r['ver']}.png")
        strip.append(imgs[0].resize((80, 96)))
    strip_img = Image.new('RGB', (80*len(strip), 96), (0, 0, 0))
    for i, im in enumerate(strip):
        strip_img.paste(im, (i*80, 0))
    strip_img.save(OUT/'_strip_clip.png')
    print('order:', ' | '.join(r['tag'] for r in results))


if __name__ == '__main__':
    main(sys.argv)
