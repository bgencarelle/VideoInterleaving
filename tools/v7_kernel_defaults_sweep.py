#!/usr/bin/env python3
"""Score every shipped DCT-kernel default and run one-parameter sweeps.

The reference pictures and receiver reconstruction come from
``tools.v7_kernel_bench``. Each kernel's declared parameters are independently
tested at their low, middle, and high values while the other parameters stay at
their defaults. This is a screening sweep, not a joint global optimizer.

Example (include representative video frames):

    .venv/bin/python tools/v7_kernel_defaults_sweep.py \
        --display ideal --image tmp/frame-000.png --csv tmp/ideal.csv

Repeat with ``--display bilinear`` to measure the upscaled display path.
"""
import argparse
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from animation_modem.v7_kernels import open_registry
from tools import v7_kernel_bench as bench


def candidate_values(param):
    """Return low/middle/high probes, preserving integer parameter values."""
    values = (param.low, (param.low + param.high) / 2.0, param.high)
    if param.integer:
        values = tuple(round(value) for value in values)
    result = []
    for value in values:
        value = float(value)
        if value not in result:
            result.append(value)
    return result


def score(selection, mask_set, images, adjust):
    row = bench.score_pictures([('candidate', selection)], mask_set,
                               adjust=adjust, extra=images)['candidate']
    built_in = [row[name] for name in
                ('astronaut', 'chelsea', 'coffee', 'rocket')]
    extra = [value for name, value in row.items()
             if name not in ('astronaut', 'chelsea', 'coffee', 'rocket')]
    built_in_mean = float(np.mean(built_in))
    extra_mean = float(np.mean(extra)) if extra else None
    # The fixed natural-image set is the tuning objective. Project images are
    # recorded as a separate hold-out so a synthetic/stress fixture cannot
    # dominate the defaults just because several adjacent frames were supplied.
    return row, built_in_mean, extra_mean


class AspectMonoBenchmark:
    """Aspect-aware sent-coefficient masks for the Aspect Mono 500 wire."""

    def __init__(self):
        from animation_modem import v7

        test_modem = str(ROOT / 'test_modem_v7')
        if test_modem not in sys.path:
            sys.path.insert(0, test_modem)
        from aspect_mono import AspectMonoWire

        target_rms = .1521 / np.sqrt(1 + 10**(v7.CLOCK_REL_DB/10))
        self.model = v7.load_model(target_rms, encode_filter='box')
        self.wire = AspectMonoWire(self.model, layout='auto')
        self.layouts = {}
        for code in range(8):
            layout = self.wire.layout_for(code)
            if layout not in self.layouts:
                self.layouts[layout] = self._layout_masks(layout)
        self.v7 = v7

    def _layout_masks(self, layout):
        from animation_modem import v7

        profile_model = self.wire.model_for(self.model, layout)
        codec = self.wire._codec(profile_model)
        luma = codec.sent_luma_mask().reshape(v7.V7_GRIDS[0])
        kept = codec.kept[np.asarray(codec.sent_model_indices, dtype=int)]
        masks = [luma]
        offsets = codec.grid.off
        for plane in (1, 2):
            start, stop = int(offsets[plane]), int(offsets[plane+1])
            mask = np.zeros(stop-start, dtype=bool)
            inside = kept[(kept >= start) & (kept < stop)]
            mask[inside-start] = True
            masks.append(mask.reshape(v7.V7_GRIDS[plane]))
        return masks

    def masks_for_size(self, size):
        code = self.v7.aspect_wire_code(size)
        layout = self.wire.layout_for(code)
        return layout, self.layouts[layout], self.v7.V7_ASPECT_RATIOS[code]

    def score_pictures(self, selection, images, adjust):
        from PIL import Image
        from skimage import data
        from ssimulacra2 import compute_ssimulacra2

        pictures = {name: Image.fromarray(getattr(data, name)())
                    for name in ('astronaut', 'chelsea', 'coffee', 'rocket')}
        for path in images:
            with Image.open(path) as image:
                pictures[Path(path).stem] = image.convert('RGB')
        results = {}
        target_height = bench.GRIDS[0][0] * 4
        display_filter = (bench.FILTERS.get(bench.DISPLAY,
                                            Image.Resampling.BICUBIC))

        def png(image):
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            buffer.seek(0)
            return buffer

        for name, image in pictures.items():
            image = image.convert('RGB')
            layout, mask_set, ratio = self.masks_for_size(image.size)
            width = max(1, round(target_height*ratio))
            size = (width, target_height)
            reference = image.resize(size, Image.Resampling.LANCZOS)
            values = bench.encode(np.asarray(image), selection, mask_set,
                                  adjust=adjust)
            reconstructed = Image.fromarray(
                bench.shown_rgb(values, mask_set, up=4), 'RGB')
            reconstructed = reconstructed.resize(size, display_filter)
            results[name] = float(compute_ssimulacra2(
                png(reference), png(reconstructed)))
        return results

    def mean_scores(self, selection, images, adjust):
        rows = self.score_pictures(selection, images, adjust)
        built_in = [rows[name] for name in
                    ('astronaut', 'chelsea', 'coffee', 'rocket')]
        extra = [value for name, value in rows.items()
                 if name not in ('astronaut', 'chelsea', 'coffee', 'rocket')]
        return rows, float(np.mean(built_in)), (
            float(np.mean(extra)) if extra else None)


class AspectStereoBenchmark(AspectMonoBenchmark):
    """Aspect-aware masks for Aspect Fold 500's default fixed tail."""

    def __init__(self):
        from animation_modem import v7

        test_modem = str(ROOT / 'test_modem_v7')
        if test_modem not in sys.path:
            sys.path.insert(0, test_modem)
        from aspect_fold import AspectFoldWire

        target_rms = .1521 / np.sqrt(1 + 10**(v7.CLOCK_REL_DB/10))
        self.model = v7.load_model(target_rms, encode_filter='box')
        self.wire = AspectFoldWire(layout='auto')
        self.layouts = {}
        for code in range(8):
            layout = self.wire.layout_for(code)
            if layout not in self.layouts:
                self.layouts[layout] = self._layout_masks(layout)
        self.v7 = v7

    def _layout_masks(self, layout):
        from animation_modem import v7

        profile_model = self.wire.model_for(self.model, layout)
        codec = self.wire.codec(profile_model)
        luma = codec.sent_luma_mask().reshape(v7.V7_GRIDS[0])
        kept = np.asarray(codec.kept, dtype=int)
        sent = getattr(codec, 'sent_model_indices', None)
        if sent is not None:
            kept = kept[np.asarray(sent, dtype=int)]
        masks = [luma]
        offsets = codec.grid.off
        for plane in (1, 2):
            start, stop = int(offsets[plane]), int(offsets[plane+1])
            mask = np.zeros(stop-start, dtype=bool)
            inside = kept[(kept >= start) & (kept < stop)]
            mask[inside-start] = True
            masks.append(mask.reshape(v7.V7_GRIDS[plane]))
        return masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--display', choices=('ideal', 'bilinear', 'bicubic',
                                               'nearest'), default='ideal')
    parser.add_argument('--image', action='append', default=[], metavar='FILE',
                        help='extra hold-out image; repeatable')
    parser.add_argument('--kernel', action='append', default=[],
                        help='limit to a kernel name; repeatable')
    parser.add_argument('--profile', choices=('aspect-mono-500',
                                               'aspect-fold-500'),
                        help='test with this profile and its aspect-aware masks')
    parser.add_argument('--defaults-only', action='store_true',
                        help='score defaults without parameter probes')
    parser.add_argument('--no-luma-adjust', action='store_true')
    parser.add_argument('--csv', metavar='FILE',
                        help='write the full sweep to a CSV file')
    args = parser.parse_args()

    bench.DISPLAY = args.display
    registry = open_registry()
    profile_bench = {
        'aspect-mono-500': AspectMonoBenchmark,
        'aspect-fold-500': AspectStereoBenchmark,
    }.get(args.profile)
    profile_bench = profile_bench() if profile_bench else None
    for path, message in registry.errors:
        print(f'skipped {path}: {message}', file=sys.stderr)
    names = args.kernel or registry.names()[1:]
    unknown = sorted(set(names) - set(registry.names()[1:]))
    if unknown:
        parser.error('unknown kernel(s): ' + ', '.join(unknown))
    mask_set = bench.masks(False)
    images = [Path(path) for path in args.image]
    rows = []

    if profile_bench:
        reference_scores, reference_mean, reference_extra = \
            profile_bench.mean_scores(None, images, not args.no_luma_adjust)
    else:
        reference_scores, reference_mean, reference_extra = score(
            None, mask_set, images, not args.no_luma_adjust)
    rows.append({
        'kernel': 'reference', 'kind': 'reference', 'parameter': '',
        'value': '', 'mean': reference_mean,
        'extra_mean': reference_extra,
        'scores': reference_scores, 'params': {},
    })
    holdout = (f' (hold-out {reference_extra:.2f})'
               if reference_extra is not None else '')
    print(f'{"reference":22s} {reference_mean:8.2f}{holdout}')

    for name in names:
        kernel = registry.get(name)
        defaults = kernel.defaults(args.profile)
        default_selection = registry.select(name, profile=args.profile)
        if profile_bench:
            scores, mean, extra_mean = profile_bench.mean_scores(
                default_selection, images, not args.no_luma_adjust)
        else:
            scores, mean, extra_mean = score(
                default_selection, mask_set, images, not args.no_luma_adjust)
        default_record = {
            'kernel': name,
            'kind': 'default',
            'parameter': '',
            'value': '',
            'mean': mean,
            'extra_mean': extra_mean,
            'scores': scores,
            'params': default_selection.params,
        }
        rows.append(default_record)
        holdout = f' (hold-out {extra_mean:.2f})' if extra_mean is not None else ''
        print(f'{name:22s} default {mean:8.2f}{holdout} '
              f'{default_selection.params}')

        if not args.defaults_only:
            best = (mean, defaults)
            module_params = getattr(kernel.module, 'PARAMS', {}) or {}
            parameters = list(module_params)
            if profile_bench:
                parameters.extend(('luma_mix', 'chroma_mix'))
            for parameter in parameters:
                param = kernel.params[parameter]
                for value in candidate_values(param):
                    if value == defaults[parameter]:
                        continue
                    selection = registry.select(
                        name, {parameter: value}, profile=args.profile)
                    if profile_bench:
                        values, candidate_mean, candidate_extra_mean = \
                            profile_bench.mean_scores(
                                selection, images, not args.no_luma_adjust)
                    else:
                        values, candidate_mean, candidate_extra_mean = score(
                            selection, mask_set, images,
                            not args.no_luma_adjust)
                    record = {
                        'kernel': name,
                        'kind': 'one-factor',
                        'parameter': parameter,
                        'value': value,
                        'mean': candidate_mean,
                        'extra_mean': candidate_extra_mean,
                        'scores': values,
                        'params': selection.params,
                    }
                    rows.append(record)
                    print(f'{name:22s} {parameter}={value:g} '
                          f'{candidate_mean:8.2f}')
                    if candidate_mean > best[0]:
                        best = (candidate_mean, selection.params)

            improvement = best[0] - mean
            print(f'{name:22s} best single-axis delta {improvement:+.2f} '
                  f'{best[1]}')

    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=(
                'kernel', 'kind', 'parameter', 'value', 'mean', 'extra_mean',
                'scores', 'params'))
            writer.writeheader()
            for row in rows:
                writer.writerow({**row,
                                 'scores': json.dumps(row['scores'], sort_keys=True),
                                 'params': json.dumps(row['params'], sort_keys=True)})
        print(f'wrote {path}')


if __name__ == '__main__':
    main()
