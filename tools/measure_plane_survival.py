#!/usr/bin/env python3
"""Score decoded modem images separately in Y, Cb, and Cr.

The scorer compares rendered images directly.  It does not apply temporal
stabilisation or perceptual colour weighting; the per-plane values make it
clear whether a failure is brightness/detail or colour.
"""
import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image


def plane_metrics_arrays(reference, decoded):
    """Return PSNR, SSIM, MAE, and normalized RMSE for each YCbCr plane."""
    def ycbcr(value):
        if isinstance(value, Image.Image):
            return np.asarray(value.convert('YCbCr'), dtype=float)
        return np.asarray(Image.fromarray(np.asarray(value).astype(np.uint8))
                        .convert('YCbCr'), dtype=float)

    ref = ycbcr(reference)
    got = ycbcr(decoded)
    if ref.shape != got.shape:
        raise ValueError(f'image size mismatch: {ref.shape} vs {got.shape}')
    result = []
    for index in range(3):
        x, y = ref[..., index], got[..., index]
        error = x - y
        mse = float(np.mean(error * error))
        psnr = float('inf') if mse == 0 else 10 * math.log10(255**2 / mse)
        ux, uy = float(x.mean()), float(y.mean())
        vx, vy = float(x.var()), float(y.var())
        covariance = float(np.mean((x - ux) * (y - uy)))
        c1, c2 = (0.01 * 255)**2, (0.03 * 255)**2
        ssim = ((2*ux*uy + c1) * (2*covariance + c2) /
                ((ux*ux + uy*uy + c1) * (vx + vy + c2)))
        mae = float(np.mean(np.abs(error)))
        normalized = math.sqrt(mse / max(vx, 1e-12))
        result.append((psnr, float(ssim), mae, normalized))
    return result


def plane_metrics(reference, decoded):
    """Score two image paths using :func:`plane_metrics_arrays`."""
    with Image.open(reference) as ref, Image.open(decoded) as got:
        return plane_metrics_arrays(ref, got)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('decoded', type=Path,
                        help='directory containing frame_*.png')
    args = parser.parse_args(argv)
    refs = sorted(args.reference.glob('frame_*.png'))
    rows = []
    for ref in refs:
        got = args.decoded / ref.name
        if got.exists():
            rows.append(plane_metrics(ref, got))
    if not rows:
        raise SystemExit('no matching frame_*.png files')
    values = np.asarray(rows)
    print(f'frames={len(rows)}')
    print('plane,mean_psnr_db,mean_ssim,mean_mae,mean_normalized_rmse')
    for name, index in zip(('Y', 'Cb', 'Cr'), range(3)):
        print(f'{name},{values[:, index, 0].mean():.3f},'
              f'{values[:, index, 1].mean():.5f},'
              f'{values[:, index, 2].mean():.3f},'
              f'{values[:, index, 3].mean():.3f}')
    print('overall,'
          f'{values[:, :, 0].mean():.3f},'
          f'{values[:, :, 1].mean():.5f},'
          f'{values[:, :, 2].mean():.3f},'
          f'{values[:, :, 3].mean():.3f}')


if __name__ == '__main__':
    raise SystemExit(main())
