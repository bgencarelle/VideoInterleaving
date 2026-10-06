#!/usr/bin/env python3
"""Render a plain-language gain/loss chart from completed V7 wire trials."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROFILES = ('aspect-fold-500', 'aspect-mono-500')


METRICS = (
    ('horizontal_cycles', 'H resolution\nΔ cycles/frame', 12, True, 'Higher frequency passed'),
    ('vertical_cycles', 'V resolution\nΔ cycles/frame', 12, True, 'Higher frequency passed'),
    ('h_pitch2', 'H pitch-2\ncontrast Δ pp', 35, True, 'Finest bar contrast'),
    ('v_pitch2', 'V pitch-2\ncontrast Δ pp', 35, True, 'Finest bar contrast'),
    ('hair_detail', 'Hair detail\nΔ corr. pp', 10, True, 'Fine-band correlation'),
    ('delivery', 'Shown\nΔ pp', 5, True, 'Delivery fraction'),
    ('edge_halo', 'Edge halo\nΔ pp', 8, False, 'Overshoot / undershoot'),
    ('color_error', 'Color error\nΔ ΔE76', 8, False, 'Perceptual color error'),
)


def font(size, bold=False):
    paths = ([Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'),
              Path('/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf')]
             if bold else [Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
                           Path('/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf')])
    for path in paths:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def summary_metrics(trial):
    metrics = trial.get('steady') or trial.get('all') or {}
    output = {}
    for axis, dimension in (('horizontal', 80), ('vertical', 96)):
        pitch = metrics.get(axis+'_finest_pitch_median')
        key = axis+'_pitch2_faithful_contrast_median'
        if key in metrics:
            # No passing pitch means zero resolved cycles, not missing data.
            output[axis+'_cycles'] = dimension/float(pitch) if pitch else 0.0
            output['h_pitch2' if axis == 'horizontal' else 'v_pitch2'] = (
                100*float(metrics[key]))
    if 'hair_band2_correlation_median' in metrics:
        output['hair_detail'] = 100*float(metrics['hair_band2_correlation_median'])
    frames = metrics.get('frames', 0)
    if frames:
        output['delivery'] = 100*metrics.get('pictures_shown', 0)/frames
    edge = [metrics.get(key) for key in ('vertical_edge_overshoot_median',
                                          'vertical_edge_undershoot_median',
                                          'horizontal_edge_overshoot_median',
                                          'horizontal_edge_undershoot_median')]
    edge = [float(x) for x in edge if x is not None]
    if edge:
        output['edge_halo'] = 100*float(np.median(edge))
    if 'delta_e76_median' in metrics:
        output['color_error'] = float(metrics['delta_e76_median'])
    return output


def collect(run, baseline='pillow-ycbcr'):
    manifest = json.loads((run/'manifest.json').read_text())
    summary = json.loads((run/'summary.json').read_text())
    lookup = {}
    for trial in summary:
        key = (trial['profile'], trial['exposure'], trial['case'], trial['seed'])
        lookup.setdefault(key, []).append(trial)
    rows = []
    grouped = {}
    for trial in summary:
        if trial['candidate'] == baseline:
            continue
        grouped.setdefault((trial['profile'], trial['candidate'], trial['configuration']), []).append(trial)
    for (profile, candidate, configuration), trials in grouped.items():
        deltas = {name: [] for name, *_ in METRICS}
        pairs = 0
        reference_configs = set()
        for trial in trials:
            key = (profile, trial['exposure'], trial['case'], trial['seed'])
            matches = [row for row in lookup.get(key, ()) if row['candidate'] == baseline]
            if not matches:
                continue
            ref = matches[0]
            current, reference = summary_metrics(trial), summary_metrics(ref)
            for name, *_ in METRICS:
                if name in current and name in reference:
                    deltas[name].append(current[name]-reference[name])
            reference_configs.add(ref['configuration'])
            pairs += 1
        medians = {name: (float(np.median(values)) if values else None)
                   for name, values in deltas.items()}
        rows.append({'profile': profile, 'candidate': candidate,
                     'configuration': configuration, 'paired_trials': pairs,
                     'baseline_configuration': ', '.join(sorted(reference_configs)),
                     **medians})
    rows.sort(key=lambda row: (PROFILES.index(row['profile']), row['candidate'],
                               row['configuration']))
    return manifest, rows


def _color(value, scale, positive_good):
    if value is None:
        return (225, 228, 232)
    norm = max(-1., min(1., value/scale))
    if not positive_good:
        norm = -norm
    neutral = np.array([247, 247, 244], float)
    target = np.array([53, 145, 88] if norm >= 0 else [198, 76, 70], float)
    return tuple(np.uint8(np.rint(neutral*(1-abs(norm))+target*abs(norm))))


def _format(name, value):
    if value is None:
        return '—'
    if name.endswith('_cycles'):
        return f'{value:+.1f}'
    if name in ('h_pitch2', 'v_pitch2', 'hair_detail', 'delivery', 'edge_halo'):
        return f'{value:+.1f}'
    return f'{value:+.2f}'


def render_chart(rows, path, limited=False):
    name_width, cell_width, header_h, row_h = 300, 142, 112, 35
    profile_gap, margin = 24, 18
    groups = {profile: [r for r in rows if r['profile'] == profile]
              for profile in PROFILES}
    visible = [p for p in PROFILES if groups[p]]
    height = margin*2+header_h*2+sum(len(groups[p])*row_h+profile_gap+5 for p in visible)+18
    width = margin*2+name_width+len(METRICS)*cell_width
    image = Image.new('RGB', (width, height), '#f6f7f8')
    draw = ImageDraw.Draw(image)
    title = font(23, True)
    body = font(15)
    small = font(12)
    bold = font(13, True)
    draw.text((margin, margin), 'V7 wire benchmark — what improved, what regressed',
              fill='#17212b', font=title)
    subtitle = ('LIMITED VALIDATION RUN' if limited else 'FULL VALIDATION RUN')
    draw.text((margin, margin+32),
              subtitle+'  |  Changes are medians versus current YCbCr GUI default',
              fill='#59636e', font=small)
    x0, y = margin, margin+header_h-46
    draw.text((x0, y+14), 'COLOR / CONFIGURATION', fill='#34404b', font=bold)
    for i, (key, label, scale, positive, help_text) in enumerate(METRICS):
        x = margin+name_width+i*cell_width
        draw.rectangle((x, y, x+cell_width-3, y+header_h-12), fill='#e7ebef')
        lines = label.split('\n')
        for line_index, line in enumerate(lines):
            draw.text((x+5, y+7+line_index*18), line, fill='#26323c', font=bold)
        draw.text((x+5, y+54), 'Higher is better' if positive else 'Lower is better',
                  fill='#53606c', font=small)
        draw.text((x+5, y+72), help_text[:20], fill='#53606c', font=small)
    y += header_h-12
    for profile in visible:
        label = 'STEREO ASPECT FOLD 500' if profile == 'aspect-fold-500' else 'MONO ASPECT FOLD 500'
        draw.rectangle((margin, y, width-margin, y+profile_gap-2), fill='#344b5e')
        draw.text((margin+8, y+4), label, fill='white', font=bold)
        y += profile_gap
        for row in groups[profile]:
            label = row['candidate']
            configuration = row['configuration']
            # Keep full config as detail for tuned runs; screen rows stay uncluttered.
            if configuration != row['candidate'] and not configuration.endswith(
                    'viewer_solve-current-default'):
                label += ' · '+configuration.removeprefix('aspect-fold-500-').removeprefix(
                    'aspect-mono-500-')
            draw.rectangle((margin, y, margin+name_width-4, y+row_h-2), fill='white')
            draw.text((margin+6, y+4), label[:34], fill='#18232d', font=body)
            draw.text((margin+6, y+21), f"{row['paired_trials']} matched exposure/channel pairs",
                      fill='#69747f', font=small)
            for i, (key, _heading, scale, positive, _help) in enumerate(METRICS):
                value = row.get(key)
                x = margin+name_width+i*cell_width
                fill = _color(value, scale, positive)
                draw.rectangle((x, y, x+cell_width-3, y+row_h-2), fill=fill)
                text = _format(key, value)
                box = draw.textbbox((0, 0), text, font=bold)
                draw.text((x+(cell_width-(box[2]-box[0]))/2, y+8), text,
                          fill='#17212b', font=bold)
            y += row_h
        y += 5
    draw.text((margin, height-margin-31),
              'Resolution = highest built-in bar frequency faithfully recovered; higher cycles/picture is finer.',
              fill='#4e5963', font=small)
    draw.text((margin, height-margin-16),
              'GREEN = gain; RED = loss. Contrast, hair detail and delivery use percentage points; “—” = no matched measure.',
              fill='#4e5963', font=small)
    image.save(path)


def write_outputs(run):
    run = Path(run)
    manifest, rows = collect(run)
    json_path, csv_path = run/'gain-loss.json', run/'gain-loss.csv'
    json_path.write_text(json.dumps({'baseline': 'pillow-ycbcr', 'rows': rows},
                                    indent=2, allow_nan=False)+'\n')
    fields = ['profile', 'candidate', 'configuration', 'paired_trials',
              'baseline_configuration']+[metric[0] for metric in METRICS]
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    chart = run/'GAIN-LOSS.png'
    render_chart(rows, chart, manifest.get('limited', True))
    report = run/'REPORT.md'
    if report.exists():
        text = report.read_text()
        marker = '## Gains and losses'
        if marker in text:
            text = text[:text.index(marker)]
        text += (f'\n{marker}\n\n![Gain/loss chart](GAIN-LOSS.png)\n\n'
                 'The chart compares candidates with the current GUI-default YCbCr wire. '
                 'See `gain-loss.csv/json` for numeric paired medians and matched counts. '
                 'Green means improvement and red means regression in every column. '
                 'For halo/color error, a decrease is an improvement.\n')
        report.write_text(text)
    return chart, csv_path, json_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args(argv)
    chart, csv_path, json_path = write_outputs(args.run)
    print(f'Chart: {chart}\nData: {csv_path}\nData: {json_path}')


if __name__ == '__main__':
    main()
