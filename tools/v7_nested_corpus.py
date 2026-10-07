"""Write a training folder for tools/v7_nested_build.py from the photographs
bundled with scikit-image (nothing is downloaded).

    .venv/bin/python tools/v7_nested_corpus.py --out tmp/nested-corpus-natural

Each source gives several windows at random scales and places, half of them
mirrored; the builder centre-crops every window to each layout.  astronaut,
chelsea and coffee are left out so they stay held-out pictures for
tools/v7_nested_display_eval.py.  Fifty-odd windows of nine photographs is a
thin corpus: general video footage would be better.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

SOURCES = ('rocket', 'hubble_deep_field', 'immunohistochemistry', 'retina',
           'camera', 'moon', 'coins', 'clock')


def main(argv=None):
    from skimage import data
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', default='tmp/nested-corpus-natural')
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pictures = {name: np.asarray(getattr(data, name)()) for name in SOURCES}
    pictures['motorcycle'] = np.asarray(data.stereo_motorcycle()[0])
    rng = np.random.default_rng(args.seed)
    count = 0
    for name, picture in pictures.items():
        if picture.ndim == 2:
            picture = np.repeat(picture[..., None], 3, -1)
        picture = picture[..., :3].astype(np.uint8)
        height, width = picture.shape[:2]
        for index in range(6):
            scale = rng.uniform(.55, 1.0) if index else 1.0
            rows, cols = int(height*scale), int(width*scale)
            top = int(rng.integers(0, height-rows+1))
            left = int(rng.integers(0, width-cols+1))
            window = picture[top:top+rows, left:left+cols]
            if index % 2:
                window = window[:, ::-1]
            Image.fromarray(np.ascontiguousarray(window)).save(out/f'{name}_{index}.png')
            count += 1
    print(f'{count} pictures in {out}')


if __name__ == '__main__':
    main()
