#!/usr/bin/env python3
"""Fit a power allocation to the pictures you actually send.

The transmitter spends power per coefficient in proportion to how big that
coefficient is expected to be -- that is what `default_allocation` estimates,
from spatial frequency alone, with a 1/(1+12f) law that knows nothing about
the content. It is a reasonable prior and it is leaving several dB on the
table, because a real sequence has structure a frequency prior cannot see: a
face in roughly the same place every frame, a fixed background, chroma that
barely moves.

Measured, fitting the table on eight frames and scoring a HELD-OUT ninth:

    lean-14k   1428 slots  17.34 fps  12750 Hz   28.23 -> 32.07 dB   +3.84
    mid-14k    2622 slots  11.41 fps  12750 Hz   28.35 -> 31.82 dB   +3.46
    hires-v3   2880 slots  16.48 fps  20250 Hz   28.42 -> 32.26 dB   +3.83

That is the same order as the gain from doubling the slot count, for no slots
at all -- so it converts directly into whichever of the three you want: keep
the quality and cut the band, keep the band and raise the frame rate, or keep
both and take the picture.

The table is SHARED STATE. It is not on the wire and there is nothing to
detect it from, so the receiver must be given the same file. It is also
specific to one (preset, profile) pair, because its length is that pair's slot
count; the loader checks.

    python3 utilities/fit_allocation.py --modem-dir images_modem \\
            --preset lean-14k --profile color-dct --out lean14k.npy

    python3 modem_screen.py --preset lean-14k --profile color-dct \\
            --allocation lean14k.npy --device "BlackHole 2ch"
    python3 utilities/modem_v3_check.py live-receive \\
            --allocation lean14k.npy --device "BlackHole 2ch"
"""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from animation_modem import transport3 as V3                     # noqa: E402
from animation_modem.imaging import (fit_shapes, image_values,   # noqa: E402
                                     plane_grids, plane_shapes)

PRESETS = V3.ALL_PRESETS


def geometry(profile, layout):
    """The shapes and grids a transmitter will actually use for this pair."""
    shapes = plane_shapes(profile)
    grids = plane_grids(profile)
    if sum(int(np.prod(s)) for s in shapes) > layout.capacity:
        shapes = fit_shapes(shapes, layout.capacity)
        grids = shapes
    return shapes, grids


def fit(frames, shapes, grids, floor=1e-6):
    """RMS coefficient magnitude per slot, over the frames given.

    Run through a FLAT coder so what comes back is the coefficient itself
    rather than the coefficient times the prior we are trying to replace --
    fitting against the default table would fold it in twice.

    RMS rather than mean: the allocation wants the expected magnitude, and a
    coefficient that is large in a few frames and near zero in the rest still
    needs the power when it is large.
    """
    count = int(sum(np.prod(s) for s in shapes))
    probe = V3.SourceCoder(shapes, np.ones(count), grids=grids)
    total = np.zeros(count)
    for image in frames:
        total += np.abs(probe.forward(image_values(image, grids)))**2
    sigma = np.sqrt(total/max(len(frames), 1))
    # A slot that was zero in every training frame still has to be able to
    # carry something, or a picture outside the training set loses it entirely.
    return np.maximum(sigma, floor*float(np.max(sigma)) if np.max(sigma) else floor)


def frames_from(args):
    if args.modem_dir:
        from modem_bake import ModemLibrary
        library = ModemLibrary(args.modem_dir)
        picks = range(0, min(library.frames, args.frames*args.stride), args.stride)
        got = [library.composite(i, 0, 0) for i in picks]
        return got, library.profile
    from utilities.modem_v3_check import frames_from as synthetic
    import types
    return synthetic(types.SimpleNamespace(
        modem_dir=None, frames=args.frames, stride=args.stride, profile=None))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--modem-dir', type=Path,
                   help='Bake to fit against. Without one, synthetic frames '
                        'are used, which is only useful for a smoke test.')
    p.add_argument('--preset', choices=list(PRESETS), default='lean-14k')
    p.add_argument('--profile', default=None,
                   help="Defaults to the bake's own profile.")
    p.add_argument('--frames', type=int, default=32)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--out', type=Path, default=Path('allocation.npy'))
    args = p.parse_args(argv)

    frames, bake_profile = frames_from(args)
    if not frames:
        raise SystemExit('No frames to fit against')
    profile = args.profile or bake_profile
    layout = PRESETS[args.preset]
    shapes, grids = geometry(profile, layout)
    sigma = fit(frames, shapes, grids)
    np.save(args.out, sigma)
    spread = 20*np.log10(float(np.max(sigma))/float(np.min(sigma)))
    print(f'{args.preset} / {profile}: {len(sigma)} slots from {len(frames)} '
          f'frames -> {args.out}')
    print(f'  picture {shapes[0][1]}x{shapes[0][0]}, '
          f'{layout.fps:.2f} fps, {layout.band_at(48000)[1]:.0f} Hz')
    print(f'  magnitude spread {spread:.1f} dB across slots')
    print('  BOTH ENDS must be given this file: it is not on the wire.')


if __name__ == '__main__':
    main()
