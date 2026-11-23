"""Fine-line, aliasing and moire test patterns for V7 resolution work.

Every pattern is defined analytically in picture coordinates and rendered by
supersampling, so the files themselves are free of rendering aliases; any
moire in a decoded picture comes from the system under test.  Frequencies are
in cycles per picture height (c/ph).  For the 3:4 layout the production packet
keeps luma detail to about 28 c/ph, the fold guests reach about 32, and the
96x80 sampling lattice ends at 48 c/ph vertically (53 horizontally).

    .venv/bin/python tools/v7_test_patterns.py --out tmp/v7-test-patterns

Writes one PNG per pattern, ``MANIFEST.json`` and ``CONTACT.png``.
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from numba import njit
from PIL import Image, ImageDraw

LAYOUTS = {'3:4': (768, 1024), '4:3': (1024, 768), '16:9': (1280, 720), '5:6': (800, 960)}
SUPERSAMPLE = 4

# kind codes for the compiled renderer
(ZONE, HYPERBOLIC, SWEEP, WEDGE, STAR, BARS, SLANT, CHECKER, BEAT, RINGS,
 HATCH, LINES, DOTS, STAIRS, SLANT_EDGE) = range(15)


@njit(cache=True)
def _square(phase):
    """Square wave of unit period: 1 for the first half, 0 for the second."""
    return 1.0 if phase-math.floor(phase) < .5 else 0.0


@njit(cache=True)
def _value(kind, x, y, aspect, p0, p1, p2, p3):
    """Pattern luminance in [0, 1] at picture coordinates.

    ``y`` runs 0..1 down the picture height; ``x`` runs 0..aspect in the same
    units, so a frequency means the same pitch on both axes.
    """
    cx, cy = .5*aspect, .5
    dx, dy = x-cx, y-cy
    if kind == ZONE:
        # Circular zone plate: local frequency rises linearly to p0 c/ph at
        # the half-height radius.  p1 is contrast.
        r2 = dx*dx+dy*dy
        return .5+.5*p1*math.cos(2*math.pi*p0*r2)
    if kind == HYPERBOLIC:
        return .5+.5*p1*math.cos(2*math.pi*p0*2*dx*dy)
    if kind == SWEEP:
        # Linear sweep from p0 to p1 c/ph along the axis p2 (0 = vertical
        # detail, lines stacked down the picture).  p3: 0 sine, 1 square.
        t = y if p2 == 0 else x/aspect
        length = 1.0 if p2 == 0 else aspect
        phase = length*(p0*t+.5*(p1-p0)*t*t)
        if p3 > 0:
            return _square(phase)
        return .5+.5*math.cos(2*math.pi*phase)
    if kind == WEDGE:
        # Converging line wedge: p0 lines fanning from a point; local pitch
        # shrinks towards the apex.  p2 selects orientation.
        if p2 == 0:
            u, v = dx, y+.02
        else:
            u, v = dy, x/aspect+.02
        return _square(p0*u/v*.5+.25)
    if kind == STAR:
        # Siemens star with p0 spokes (sine if p1 == 0).
        angle = math.atan2(dy, dx)
        if p1 > 0:
            return _square(p0*angle/(2*math.pi))
        return .5+.5*math.cos(p0*angle)
    if kind == BARS:
        # Blocks of square-wave bars at listed frequencies; p0 = first c/ph,
        # p1 = step, p2 = number of blocks, p3 = axis.
        if p3 == 0:
            block = min(int(x/aspect*p2), int(p2)-1)
            return _square((p0+p1*block)*y)
        block = min(int(y*p2), int(p2)-1)
        return _square((p0+p1*block)*x)
    if kind == SLANT:
        # Grating of p0 c/ph tilted p1 degrees from horizontal lines.
        a = math.radians(p1)
        return _square(p0*(y*math.cos(a)+x*math.sin(a)))
    if kind == CHECKER:
        # Checkerboard blocks; cell pitch falls across p2 blocks.
        block = min(int(y*p2), int(p2)-1)
        f = p0+p1*block
        a, b = math.floor(2*f*x), math.floor(2*f*y)
        return 1.0 if (a+b) % 2 == 0 else 0.0
    if kind == BEAT:
        # Two sine gratings p0 and p1 c/ph, the second turned by p2 degrees:
        # their difference frequency is a designed-in beat, a reference for
        # what moire looks like when it is really in the source.
        a = math.radians(p2)
        g1 = math.cos(2*math.pi*p0*y)
        g2 = math.cos(2*math.pi*p1*(y*math.cos(a)+x*math.sin(a)))
        return .5+.25*(g1+g2)
    if kind == RINGS:
        # Concentric rings of constant pitch p0 c/ph (square if p1 > 0).
        r = math.sqrt(dx*dx+dy*dy)
        if p1 > 0:
            return _square(p0*r)
        return .5+.5*math.cos(2*math.pi*p0*r)
    if kind == HATCH:
        # Cross-hatch of thin dark lines: pitch 1/p0, line width p1 (picture
        # heights), at +-p2 degrees.
        a = math.radians(p2)
        u = y*math.cos(a)+x*math.sin(a)
        v = y*math.cos(a)-x*math.sin(a)
        fu = u*p0-math.floor(u*p0)
        fv = v*p0-math.floor(v*p0)
        return 0.0 if (fu < p1*p0 or fv < p1*p0) else 1.0
    if kind == LINES:
        # Isolated lines of growing width on a flat ground: p0 lines, widths
        # from p1 to p2 picture heights, p3: 0 horizontal lines, 1 vertical.
        t = y if p3 == 0 else x/aspect
        slot = min(int(t*p0), int(p0)-1)
        centre = (slot+.5)/p0
        width = p1+(p2-p1)*slot/max(p0-1, 1.0)
        length = 1.0 if p3 == 0 else aspect
        dark = abs(t-centre)*length < .5*width
        polarity = (x/aspect if p3 == 0 else y) < .5
        if dark:
            return 1.0 if polarity else 0.0
        return 0.0 if polarity else 1.0
    if kind == DOTS:
        # Grid of isolated dots whose radius grows down the picture.
        cell = 1.0/p0
        gx, gy = x/cell, y/cell
        fx, fy = gx-math.floor(gx)-.5, gy-math.floor(gy)-.5
        radius = (p1+(p2-p1)*y)/cell
        return 1.0 if fx*fx+fy*fy < radius*radius else 0.0
    if kind == STAIRS:
        # Near-horizontal and near-vertical thin lines: the classic jaggies
        # test.  p0 lines per fan, p1 width, slopes from p2 to p3 degrees.
        half = x < .5*aspect
        best = 1.0
        for i in range(int(p0)):
            a = math.radians(p2+(p3-p2)*i/max(p0-1, 1.0))
            offset = (i+.5)/p0
            if half:
                d = abs((y-offset)*math.cos(a)-(x-.25*aspect)*math.sin(a))
            else:
                d = abs((x/aspect-.5-.5*offset)*aspect*math.cos(a)-(y-.5)*math.sin(a))
            if d < .5*p1:
                best = 0.0
        return best
    # SLANT_EDGE: a dark rectangle turned p0 degrees, for edge-response work.
    a = math.radians(p0)
    u = dx*math.cos(a)+dy*math.sin(a)
    v = -dx*math.sin(a)+dy*math.cos(a)
    return p1 if (abs(u) < .28*aspect and abs(v) < .3) else p2


@njit(cache=True)
def render(kind, height, width, p0, p1, p2, p3, supersample):
    """Box-filtered rendering: mean of supersample^2 points per pixel."""
    out = np.empty((height, width))
    aspect = width/height
    step = 1.0/(height*supersample)
    for row in range(height):
        for col in range(width):
            total = 0.0
            for i in range(supersample):
                for j in range(supersample):
                    y = (row*supersample+i+.5)*step
                    x = (col*supersample+j+.5)*step
                    total += _value(kind, x, y, aspect, p0, p1, p2, p3)
            out[row, col] = total/(supersample*supersample)
    return out


# name, kind, parameters, what it shows
SUITE = [
    ('zone-plate-64', ZONE, (64, 1.0, 0, 0),
     'Circular zone plate to 64 c/ph at the half-height radius. False rings away from the centre are aliases.'),
    ('zone-plate-64-low-contrast', ZONE, (64, .2, 0, 0),
     'Same plate at 20% contrast: fine detail near the noise a fold adds.'),
    ('zone-plate-hyperbolic-48', HYPERBOLIC, (48, 1.0, 0, 0),
     'Hyperbolic plate: diagonal frequencies, which the radial coefficient order cuts first.'),
    ('sweep-vertical-sine-4-64', SWEEP, (4, 64, 0, 0),
     'Sine sweep 4 to 64 c/ph down the picture (horizontal lines). Read the limit where contrast dies.'),
    ('sweep-horizontal-sine-4-64', SWEEP, (4, 64, 1, 0),
     'Sine sweep 4 to 64 c/ph across the picture (vertical lines).'),
    ('sweep-vertical-square-4-48', SWEEP, (4, 48, 0, 1),
     'Square-wave sweep: harmonics above the limit fold back as moire.'),
    ('sweep-horizontal-square-4-48', SWEEP, (4, 48, 1, 1),
     'Square-wave sweep across the picture.'),
    ('bars-horizontal-lines-16-48', BARS, (16, 4, 9, 0),
     'Nine blocks of horizontal line pairs: 16, 20, ... 48 c/ph, left to right.'),
    ('bars-vertical-lines-16-48', BARS, (16, 4, 9, 1),
     'Nine blocks of vertical line pairs: 16, 20, ... 48 c/ph, top to bottom.'),
    ('wedge-vertical', WEDGE, (60, 0, 0, 0),
     'Converging wedge of 60 lines: pitch shrinks towards the top. Spurious bands near the apex are moire.'),
    ('wedge-horizontal', WEDGE, (60, 0, 1, 0),
     'Converging wedge turned a quarter.'),
    ('siemens-star-72', STAR, (72, 1, 0, 0),
     'Siemens star, 72 spokes: the blur circle radius gives the limit in every direction at once.'),
    ('siemens-star-72-sine', STAR, (72, 0, 0, 0),
     'Sine star: no harmonics, so anything beyond blur is the system.'),
    ('rings-30', RINGS, (30, 1, 0, 0),
     'Concentric square rings at 30 c/ph, just above the kept radius: all orientations at one pitch.'),
    ('rings-40-sine', RINGS, (40, 0, 0, 0),
     'Sine rings at 40 c/ph: carried only by folded guests.'),
    ('slant-30-at-3deg', SLANT, (30, 3, 0, 0),
     'Lines at 30 c/ph tilted 3 degrees: near-axis detail beats with the sampling lattice.'),
    ('slant-36-at-7deg', SLANT, (36, 7, 0, 0),
     'Lines at 36 c/ph tilted 7 degrees.'),
    ('slant-24-at-45deg', SLANT, (24, 45, 0, 0),
     'Diagonal lines at 24 c/ph: inside the limit on each axis, a test of diagonal handling.'),
    ('checker-8-40', CHECKER, (8, 4, 9, 0),
     'Checkerboards from 8 to 40 c/ph in nine bands: energy on the diagonals.'),
    ('beat-30-33-at-4deg', BEAT, (30, 33, 4, 0),
     'Two gratings 30 and 33 c/ph, 4 degrees apart: a real 3 c/ph beat is in the source. Extra beats are not.'),
    ('beat-40-44-at-2deg', BEAT, (40, 44, 2, 0),
     'Two gratings above the kept radius. If only the beat survives, the carriers were lost.'),
    ('hatch-20-fine', HATCH, (20, .004, 30, 0),
     'Cross-hatch of hairlines at 30 degrees, 20 per picture height.'),
    ('lines-horizontal-widths', LINES, (12, .001, .012, 0),
     'Twelve isolated horizontal lines, 1 to 12 thousandths of the height, both polarities.'),
    ('lines-vertical-widths', LINES, (12, .001, .012, 1),
     'Twelve isolated vertical lines, both polarities.'),
    ('dots-growing', DOTS, (24, .002, .012, 0),
     'Isolated dots growing down the picture: point response and ringing.'),
    ('stairs-near-axis', STAIRS, (9, .004, .5, 8),
     'Hairlines half a degree to eight degrees off each axis: stair-stepping and crawl.'),
    ('slant-edge-5deg', SLANT_EDGE, (5, .2, .8, 0),
     'Slanted edge, 5 degrees, 20% to 80% grey: for an edge-spread measurement without clipping.'),
]


def image(name, layout='3:4', supersample=SUPERSAMPLE):
    """Float luminance plane of one suite pattern."""
    width, height = LAYOUTS[layout]
    for entry in SUITE:
        if entry[0] == name:
            return render(entry[1], height, width, *map(float, entry[2]), supersample)
    raise KeyError(name)


def to_rgb(plane):
    return np.repeat(np.uint8(np.rint(np.clip(plane, 0, 1)*255))[..., None], 3, -1)


def write_suite(out, layout='3:4', supersample=SUPERSAMPLE):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    width, height = LAYOUTS[layout]
    manifest = {'layout': layout, 'size': [width, height], 'supersample': supersample,
                'frequency_unit': 'cycles per picture height',
                'reference_3x4': {'kept_radius': 28, 'production_guests_to': 32,
                                  'lattice_nyquist_vertical': 48, 'lattice_nyquist_horizontal': 53},
                'patterns': []}
    thumbs = []
    for index, (name, kind, parameters, note) in enumerate(SUITE):
        plane = render(kind, height, width, *map(float, parameters), supersample)
        file = f'{index+1:02d}-{name}.png'
        Image.fromarray(to_rgb(plane)).save(out/file)
        manifest['patterns'].append({'file': file, 'name': name, 'parameters': list(parameters),
                                     'shows': note, 'mean_level': float(plane.mean())})
        thumbs.append((file, Image.fromarray(to_rgb(plane)).resize((width//4, height//4), Image.Resampling.BOX)))
    (out/'MANIFEST.json').write_text(json.dumps(manifest, indent=1)+'\n')
    columns = 7
    tw, th = thumbs[0][1].size
    rows = -(-len(thumbs)//columns)
    sheet = Image.new('RGB', (columns*(tw+8)+8, rows*(th+26)+8), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, (file, thumb) in enumerate(thumbs):
        x, y = 8+(i % columns)*(tw+8), 8+(i//columns)*(th+26)
        sheet.paste(thumb, (x, y+18))
        draw.text((x, y+3), file[:-4][:30], fill='black')
    sheet.save(out/'CONTACT.png')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--out', default='tmp/v7-test-patterns')
    parser.add_argument('--layout', choices=LAYOUTS, default='3:4')
    parser.add_argument('--supersample', type=int, default=SUPERSAMPLE)
    args = parser.parse_args(argv)
    manifest = write_suite(args.out, args.layout, args.supersample)
    print(f"{len(manifest['patterns'])} patterns in {args.out}")


if __name__ == '__main__':
    main()
