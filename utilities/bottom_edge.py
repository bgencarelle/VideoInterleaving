#!/usr/bin/env python3
"""Bottom-edge margin probe: steep highpass + noise, verify counts.

Takes an optional tree path (default: this repo) so a baseline checkout can
be measured the same way. Synthetic gradient content, 8 packets per point --
a quick margin check, not a battery.

Run from anywhere:

    .venv/bin/python utilities/bottom_edge.py [tree-path]
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import butter, sosfiltfilt


def main(argv=None):
    tree = argv[1] if argv and len(argv) > 1 else str(
        Path(__file__).resolve().parent.parent)
    sys.path.insert(0, tree)
    from animation_modem import transport3 as v3
    from animation_modem.imaging import (fit_shapes, image_values,
                                         plane_shapes)
    layout = v3.WIRE
    tag = f'WIRE@{tree}'
    coder_shapes = fit_shapes(plane_shapes('color-dct'), layout.capacity)
    coder = v3.SourceCoder(coder_shapes)
    yy, xx = np.mgrid[:48, :40]
    img = Image.fromarray(np.uint8(np.stack([xx*6, yy*5, (xx+yy)*2], -1)))
    vals = image_values(img, coder_shapes)
    print(f'tree={tag}')
    for lo, order in ((300, 4), (300, 10), (375, 4), (375, 10)):
        ver = 0
        for n in range(8):
            audio = np.asarray(v3.encode(vals, layout, coder, n+1, 1, 3),
                               float)
            audio = sosfiltfilt(butter(order, lo/24000, 'high', output='sos'),
                                audio, axis=0)
            audio = (audio+np.random.default_rng(n).normal(0, .005,
                     audio.shape)).astype(np.float32)
            rx = v3.Receiver(layout, coder)
            got = []
            for i in range(0, len(audio), 1024):
                got += rx.feed(audio[i:i+1024])
            got += rx.flush()
            ver += sum(1 for g in got if g.identity == 'verified_header')
        print(f'  highpass {lo}Hz ord{order}: {ver}/24')


if __name__ == '__main__':
    main(sys.argv)
