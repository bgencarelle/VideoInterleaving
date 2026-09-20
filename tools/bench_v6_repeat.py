#!/usr/bin/env python3
"""Compare V6 foundation protection with its full-repeat control.

The full-repeat control intentionally spends nearly twice the image capacity
on diversity. It is a diagnostic, not the default wire.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import butter, sosfilt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from animation_modem import transport3 as V3                         # noqa: E402
from animation_modem.imaging import image_values, prepare_image       # noqa: E402
from animation_modem.v6 import coder_for, full_repeat_coder           # noqa: E402


def decode(audio, layout, coder, code):
    receiver = V3.Receiver(layout, coder, pulse_only=True, fast=True,
                           coders={code: coder},
                           candidates=[(layout, coder, {code: coder})])
    results = []
    for start in range(0, len(audio), 1024):
        results.extend(receiver.feed(audio[start:start+1024]))
    return results + receiver.flush()


def damage(audio, case):
    out = audio.copy()
    if case == 'mute-left':
        out[:, 0] = 0
    elif case == 'mute-right':
        out[:, 1] = 0
    elif case == 'lowpass-8k':
        sos = butter(4, 8000, fs=V3.REFERENCE_RATE, output='sos')
        out = sosfilt(sos, out, axis=0).astype(np.float32)
    elif case == 'hiss-35':
        rng = np.random.default_rng(35)
        out = (out + rng.normal(0, 10**(-35/20), out.shape)).astype(np.float32)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path,
                        default=Path('images_sbs/face/00_C_BG_faceSource_960/'
                                     'benFaceSource1110.jpg'))
    args = parser.parse_args(argv)
    source = Image.open(args.source).convert('RGB').crop((0, 0, 720, 960))
    prepared = prepare_image(source, 'auto', 'lanczos')
    print('control transform case acquired verified pictures rmse_vs_clean fps')
    for control, layout, factory in (
            ('foundation', V3.WIRE_V6, coder_for),
            ('full-repeat', V3.WIRE_V6_REPEAT, full_repeat_coder)):
        for code, transform in enumerate(('dct', 'wavelet')):
            coder = factory(transform)
            values = image_values(prepared, coder.grids)
            audio = V3.encode(values, layout, coder, 1, 1, 1, profile=code)
            clean = decode(audio, layout, coder, code)
            reference = next(r.values for r in clean
                             if r.identity == 'verified_header' and r.values is not None)
            for case in ('clean', 'mute-left', 'mute-right', 'lowpass-8k', 'hiss-35'):
                results = decode(damage(audio, case), layout, coder, code)
                pictures = [r for r in results if r.values is not None]
                verified = sum(r.identity == 'verified_header' for r in results)
                rmse = (float(np.mean([np.sqrt(np.mean((r.values-reference)**2))
                                       for r in pictures])) if pictures else None)
                print(control, transform, case, len(results), verified,
                      len(pictures), rmse, f'{layout.fps:.3f}')


if __name__ == '__main__':
    main()
