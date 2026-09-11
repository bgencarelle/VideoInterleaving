#!/usr/bin/env python3
"""Measure where image quality is actually lost on the modem link.

Reports, for the frames a real bake would transmit:

  * the source-coding ceiling -- what the profile's subsampling and 8-bit
    quantization cost before any audio is involved. No channel change can
    beat this number.
  * the channel's contribution under each impairment preset.
  * how much transmit power is spent on the constant image mean.
  * the waveform peak against the sync peak, i.e. level left unused.
  * the gain an analog power allocation (DCT + per-chunk gains) would give,
    both with per-frame metadata and with a fixed table computed from the
    bake itself -- which is free, because the image library never changes.

Every figure is content-dependent, so run this against the bake you ship.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.fft import dctn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from animation_modem import transport as T          # noqa: E402
from animation_modem import impairments as I        # noqa: E402


def sample_frames(root, count, stride):
    """Composite frames straight out of a modem bake, as modem_display would."""
    from modem_bake import ModemLibrary
    library = ModemLibrary(root)
    picks = range(0, min(library.frames, count * stride), stride)
    frames = [library.composite(i, 0, 0) for i in picks]
    return frames, library.profile


def load_images(paths, profile):
    size = T.PROFILES[profile][0]
    return [Image.open(p).convert('RGB').resize(size, Image.Resampling.LANCZOS)
            for p in paths]


def psnr(a, b):
    error = ((np.asarray(a, float) - np.asarray(b, float)) ** 2).mean()
    return float('inf') if error == 0 else 10 * np.log10(255 ** 2 / error)


def coefficients(values, profile):
    out, offset = [], 0
    for shape in T.plane_shapes(profile):
        n = int(np.prod(shape))
        out.append(dctn(values[offset:offset + n].reshape(shape), norm='ortho').ravel())
        offset += n
    return np.concatenate(out)


def chunk_ids(profile, k):
    ids, offset, cid = np.empty(0, int), 0, 0
    pieces = []
    for shape in T.plane_shapes(profile):
        grid = np.empty(shape, int)
        for ys in np.array_split(np.arange(shape[0]), k):
            for xs in np.array_split(np.arange(shape[1]), k):
                grid[np.ix_(ys, xs)] = cid
                cid += 1
        pieces.append(grid.ravel())
    return np.concatenate(pieces), cid


def allocation_mse(a, u):
    """Analog MSE for gains u under a unit transmit-power constraint."""
    u = u / np.sum(u * a ** 2)
    return float(np.sum(1.0 / u))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--modem-dir', type=Path, help='A bake from convert_to_modem.py')
    source.add_argument('--images', type=Path, nargs='+', help='Image files instead of a bake')
    p.add_argument('--profile', choices=T.PROFILES, default='color',
                   help='Only used with --images; a bake states its own')
    p.add_argument('--count', type=int, default=64, help='Frames to sample (default 64)')
    p.add_argument('--stride', type=int, default=7, help='Source-index stride (default 7)')
    p.add_argument('--presets', nargs='+', default=['clean', 'mild', 'rough'],
                   choices=list(I.PRESETS))
    p.add_argument('--chunks', type=int, default=8,
                   help='Chunk grid per plane edge for the allocation estimate (default 8)')
    args = p.parse_args(argv)

    if args.modem_dir:
        frames, profile = sample_frames(args.modem_dir, args.count, args.stride)
    else:
        frames, profile = load_images(args.images[:args.count], args.profile), args.profile
    if not frames:
        p.error('No frames to measure')
    width, height = T.PROFILES[profile][0]
    print(f'{len(frames)} frames, profile {profile} ({width}x{height}), '
          f'{15 * len(T.BINS) * 4} real values per frame\n')

    prepared = [T.prepare_image(f, 1, 1, 1, False, profile) for f in frames]
    values = np.stack([T.image_values(im, profile) for im in prepared])

    ceiling = np.mean([psnr(T.values_image(v, profile), im)
                       for v, im in zip(values, prepared)])
    print('SOURCE-CODING CEILING (subsampling + 8-bit, no audio involved)')
    print(f'  {ceiling:6.2f} dB   no channel change can beat this\n')

    print('END-TO-END THROUGH THE MODEM')
    peaks = []
    for preset in args.presets:
        got = []
        for im in prepared[:min(len(prepared), 16)]:
            audio, reference, _ = T.encode(im, 1, 1, 1, False, profile)
            peaks.append(np.abs(audio[T.SYNC_LEN:T.PACKET]).max())
            emulator = I.Emulator(I.Settings(**I.PRESETS[preset]))
            rx, results = T.Receiver(), []
            for i in range(0, len(audio), 256):
                results += rx.feed(emulator.process(audio[i:i + 256]))
            decoded = [r for r in results if r.image is not None]
            if decoded:
                got.append(psnr(decoded[0].image, reference))
        if got:
            print(f'  {preset:8s} {np.mean(got):6.2f} dB   channel costs '
                  f'{ceiling - np.mean(got):5.2f} dB')
        else:
            print(f'  {preset:8s}    ---     no frame decoded')

    print('\nTRANSMIT POWER')
    mean_power = values.mean() ** 2
    total = (values ** 2).mean()
    print(f'  constant image mean takes {100 * mean_power / total:.1f}% of the image-carrier power'
          f'  ({10 * np.log10(total / values.var()):.2f} dB)')
    print(f'  body peak {np.mean(peaks):.3f} vs sync peak {np.abs(T.SYNC).max():.3f} '
          f'and full scale 1.000')
    print(f'  body can rise {20 * np.log10(np.abs(T.SYNC).max() / np.mean(peaks)):.2f} dB '
          f'before it reaches the sync peak')

    print('\nANALOG POWER ALLOCATION (DCT, gain relative to today)')
    coeffs = np.stack([coefficients(v, profile) for v in values])
    n = coeffs.shape[1]
    base = np.mean([allocation_mse(v, np.ones(n)) for v in values])

    def report(name, per_frame):
        print(f'  {name:52s} {10 * np.log10(base / np.mean(per_frame)):+6.2f} dB')

    report('per-frame mean removed, no transform',
           [allocation_mse(v - v.mean(), np.ones(n)) for v in values])
    report('DCT, no allocation (orthonormal, so no gain)',
           [allocation_mse(c, np.ones(n)) for c in coeffs])
    ids, total_chunks = chunk_ids(profile, args.chunks)
    counts = np.bincount(ids)
    report(f'DCT + {total_chunks}-chunk allocation, per-frame metadata',
           [allocation_mse(c, 1 / np.maximum(np.sqrt(np.bincount(ids, c ** 2) / counts)[ids], 1e-9))
            for c in coeffs])
    half = max(1, len(coeffs) // 2)
    table = np.sqrt(np.bincount(ids, (coeffs[:half] ** 2).mean(0)) / counts)
    gains = 1 / np.maximum(table[ids], 1e-9)
    report(f'DCT + {total_chunks}-chunk FIXED table from the bake, no metadata',
           [allocation_mse(c, gains) for c in coeffs[half:]])
    report('DCT + per-coefficient optimum (unreachable bound)',
           [allocation_mse(c, 1 / np.maximum(abs(c), 1e-9)) for c in coeffs])
    print('\nAllocation gains only matter where the channel costs dB above.')


if __name__ == '__main__':
    main()
