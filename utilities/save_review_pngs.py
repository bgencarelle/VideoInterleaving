#!/usr/bin/env python3
"""Re-run the EQ+phase battery and save review PNGs.

Per case: a montage of the decoded frames. Plus one strip: frame 0 of every
case, in battery order. Filenames carry the verify count. Output:
scratch/eq_review/ (excluded from git -- review artifacts, not product).

Run (from the repo root) after rendering the input WAV (see eq_phase_battery):

    .venv/bin/python utilities/save_review_pngs.py [wav]
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from animation_modem import transport3 as V3
from animation_modem.imaging import values_image
from utilities.eq_phase_battery import CASES, lay, load_wav
from utilities.modem_v3_check import coder_for

REPO = Path(__file__).resolve().parent.parent
OUT = REPO/'scratch'/'eq_review'

coder, grids = coder_for('color-dct', None, lay)


def frame_image(values):
    img = values_image(values, grids)
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.asarray(img))
    return img.convert('RGB')


def decode_values(wire):
    rx = V3.Receiver(lay, coder)
    out = []
    for i in range(0, len(wire), 1024):
        out += rx.feed(np.asarray(wire[i:i+1024], np.float32))
    out += rx.flush()
    got = [r for r in out if r.values is not None]
    ver = sum(1 for r in out if r.identity == 'verified_header')
    return got, ver


def montage(imgs, cols=4):
    w, h = imgs[0].size
    rows = (len(imgs)+cols-1)//cols
    sheet = Image.new('RGB', (w*cols, h*rows), (0, 0, 0))
    for i, im in enumerate(imgs):
        sheet.paste(im, ((i % cols)*w, (i//cols)*h))
    return sheet


def main(argv=None):
    OUT.mkdir(exist_ok=True)
    wav = argv[1] if argv and len(argv) > 1 else 'eqtest.wav'
    base = load_wav(wav)
    n = len(base)//lay.frame
    strip_first = []
    results = []
    for tag, fn in CASES:
        got, ver = decode_values(fn(base))
        imgs = [frame_image(r.values) for r in got]
        while len(imgs) < n:
            imgs.append(Image.new('RGB', imgs[0].size, (0, 0, 0)))
        name = f'{tag}_{ver}-{n}.png'
        montage(imgs).save(OUT/name)
        strip_first.append(imgs[0].resize((80, 96)))
        results.append((tag, len(got), ver))
        print(f'{tag:14s} pictures {len(got):2d}/{n} verified {ver:2d}/{n} '
              f'-> {name}')

    strip = Image.new('RGB', (80*len(strip_first), 96), (0, 0, 0))
    for i, im in enumerate(strip_first):
        strip.paste(im, (i*80, 0))
    strip.save(OUT/'_strip_frame0_all_cases.png')
    print('\norder:', ' | '.join(t for t, _, _ in results))


if __name__ == '__main__':
    main(sys.argv)
