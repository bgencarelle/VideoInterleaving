"""Pictures and one chart for the nested-fold verdict (reads finished runs).

    .venv/bin/python tools/v7_sk_figures.py --out tmp/sk/figures
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw

from tools import v7_sk_fold as sk, v7_sk_study as study

SHOW = (576, 480)              # rows, cols: six display pixels per lattice sample


def show(plane):
    """Band-limited (DCT zero-padded) enlargement of a coefficient plane."""
    rows, cols = plane.shape
    padded = np.zeros(SHOW)
    r, c = min(rows, SHOW[0]), min(cols, SHOW[1])
    padded[:r, :c] = plane[:r, :c]*np.sqrt(SHOW[0]*SHOW[1]/(study.LATTICE[0]*study.LATTICE[1]))
    pixels = sk.separable(np.ascontiguousarray(sk.dct_matrix(SHOW[0]).T), padded,
                          np.ascontiguousarray(sk.dct_matrix(SHOW[1]).T))
    return np.uint8(np.rint(np.clip((pixels+1)/2, 0, 1)*255))


def panel(images, labels, path):
    height, width = images[0].shape
    canvas = Image.new('L', (width*len(images)+8*(len(images)-1), height+30), 255)
    draw = ImageDraw.Draw(canvas)
    for i, (image, label) in enumerate(zip(images, labels)):
        canvas.paste(Image.fromarray(image), (i*(width+8), 30))
        draw.text((i*(width+8)+6, 9), label, fill=0)
    canvas.save(path)


def chart(wire, bound, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    conditions = list(wire['conditions'])
    x = np.arange(len(conditions))
    ink, muted = '#1a1a19', '#6b6a63'
    fig, ax = plt.subplots(figsize=(7.6, 4.3), dpi=150)
    existing = ('linear 1,920', 'production fold 500', 'production as shipped')
    count = lambda c, name: wire['conditions'][c][name]['ensemble_equivalent_coefficients']
    # Baseline: whichever existing option is best in that condition.
    base = [max(count(c, name) for name in existing) for c in conditions]
    ceiling = [np.sqrt(bound['conditions'][c]['bound']['actual_power']['equivalent_coefficients']/b)
               for c, b in zip(conditions, base)]
    best = [np.sqrt(max(count(c, k) for k in wire['conditions'][c] if 'nested' in k)/b)
            for c, b in zip(conditions, base)]
    earlier = [np.sqrt(count(c, 'earlier surface 3to2, 3 strips')/b) for c, b in zip(conditions, base)]
    ax.axhline(1.5, color=muted, lw=1, ls=(0, (4, 3)))
    ax.text(len(x)-1, 1.515, 'requested 1.5×', ha='right', va='bottom', color=muted, fontsize=9)
    ax.axhline(1.0, color=muted, lw=1)
    ax.text(3.2, .975, 'best existing mapping = 1', ha='center', va='top', color=muted, fontsize=9)
    for values, colour, label in ((ceiling, '#2a78d6', 'Shannon ceiling (no mapping can exceed)'),
                                  (best, '#eb6834', 'adaptive nested fold, real wire'),
                                  (earlier, '#1baf7a', 'earlier 3→2 surface, real wire')):
        ax.plot(x, values, color=colour, lw=2, marker='o', ms=7, mec='white', mew=1.5, label=label)
        ax.annotate(f'{values[0]:.2f}×', (x[0], values[0]), textcoords='offset points',
                    xytext=(-8, 0), ha='right', va='center', fontsize=9, color=ink)
        ax.annotate(f'{values[-1]:.2f}×', (x[-1], values[-1]), textcoords='offset points',
                    xytext=(8, 0), ha='left', va='center', fontsize=9, color=ink)
    ax.set_xticks(x, [c.replace('hiss-', 'hiss −')+('' if c == 'clean' else ' dBFS') for c in conditions])
    ax.set_xlim(-.6, len(x)-.4)
    ax.set_ylim(0, 1.85)
    ax.set_ylabel('linear resolution vs best existing mapping', color=ink)
    ax.set_title('Detail per packet: what folding delivers and what it could', loc='left',
                 fontsize=11, color=ink)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.grid(axis='y', color='#e6e5df', lw=.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc='lower left')
    fig.tight_layout()
    fig.savefig(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--corpus', default='tmp/sk/corpus')
    parser.add_argument('--study', default='tmp/sk/study/STUDY.json')
    parser.add_argument('--wire', default='tmp/sk/wire')
    parser.add_argument('--out', default='tmp/sk/figures')
    parser.add_argument('--condition', default='clean')
    parser.add_argument('--pictures', nargs='+', type=int, default=[1, 6, 9])
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frozen = json.loads(Path(args.study).read_text())
    wire = json.loads((Path(args.wire)/'WIRE.json').read_text())
    planes = np.load(Path(args.wire)/'decoded_planes.npz')
    chart(wire, frozen, out/'resolution-vs-noise.png')
    _, test_files = study.corpus(args.corpus)
    big = study.Analyzer((study.LATTICE[0]*study.BIG, study.LATTICE[1]*study.BIG))
    names = ['production fold 500', 'earlier surface 3to2, 3 strips', 'adaptive nested fold']
    for index in args.pictures:
        reference = big(study.portrait(test_files[index]))
        images = [show(reference)]+[show(planes[f'{args.condition}|{name}|{index}']) for name in names]
        labels = ['source (fine reference)', 'production fold, decoded audio',
                  'earlier 3to2 surface, decoded audio', 'adaptive nested fold, decoded audio']
        panel(images, labels, out/f'{args.condition}-{test_files[index].stem}.png')
        y, x = SHOW[0]//3, SHOW[1]//4
        crops = [np.kron(image[y:y+192, x:x+160], np.ones((3, 3), np.uint8)) for image in images]
        panel(crops, labels, out/f'{args.condition}-{test_files[index].stem}-crop.png')


if __name__ == '__main__':
    main()
