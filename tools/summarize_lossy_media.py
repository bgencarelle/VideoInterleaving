#!/usr/bin/env python3
"""Summarize lossy modem decode logs and basic decoded-color metrics."""
import argparse
import json
from pathlib import Path
import statistics

import numpy as np
from PIL import Image


def rows(path):
    out = []
    for line in path.read_text(errors='replace').splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and 'identity' in value:
            out.append(value)
    return out


def color_metrics(directory):
    unique, chroma_steps = [], []
    for path in sorted(directory.glob('frame_*.png')):
        rgb = np.asarray(Image.open(path).convert('RGB'))
        ycbcr = np.asarray(Image.fromarray(rgb).convert('YCbCr'), dtype=np.int16)
        unique.append(len(np.unique(ycbcr[:, :, 1:])) )
        dx = np.abs(np.diff(ycbcr[:, :, 1:], axis=1)).ravel()
        dy = np.abs(np.diff(ycbcr[:, :, 1:], axis=0)).ravel()
        chroma_steps.append(float(np.mean(np.concatenate((dx, dy)) == 0)))
    if not unique:
        return '-', '-'
    return f'{statistics.mean(unique):.0f}', f'{statistics.mean(chroma_steps):.3f}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--frames', type=int, default=60)
    args = parser.parse_args()
    print('case frames verified received mean_pilot mean_coverage chroma_unique chroma_flat')
    for log in sorted(args.directory.glob('*.log')):
        data = rows(log)
        if not data:
            continue
        verified = sum(r.get('identity') == 'verified_header' for r in data)
        received = sum(r.get('status') == 'received' for r in data)
        pilot = statistics.mean(r['pilot_error'] for r in data
                                if r.get('pilot_error') is not None)
        coverage = statistics.mean(r['coverage'] for r in data
                                   if r.get('coverage') is not None)
        label = log.stem
        unique, flat = color_metrics(args.directory / f'{label}-frames')
        print(f'{label} {len(data):5d} {verified:8d} {received:8d} '
              f'{pilot:11.3f} {coverage:13.3f} {unique:13} {flat:11}')


if __name__ == '__main__':
    main()
