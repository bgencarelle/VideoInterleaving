#!/usr/bin/env python3
"""Compare DCT-mode kernels on synthetic pictures, with no audio in the loop.

For every kernel (the reference and each file in dct_kernels/) it encodes the
same test pictures with the direct DCT encoder and reports what an ideal
receiver of the sent coefficients would show:

  overshoot   largest excursion past the two levels of a hard edge (percent)
  rise        10-90% width of that edge, in pixels of the sent picture
  ripple      RMS error away from the edge (percent): the mesh and halo
  mtf@.1-.4   retained amplitude of a sine grating at that many cycles per
              sent pixel (0.5 is the most the wire holds); 1.0 is perfect
  ms          encoder time per frame (the reference includes nothing extra)

`--guests` shows the band with the fold's guest coefficients, the default
drops them to the sent rectangle (the case under noise). Nothing here
validates a real link; it ranks kernels before a human looks at them, and
`--sheet FILE` writes a contact sheet of the same pictures for that.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from animation_modem import v7
from animation_modem.v7_kernels import open_registry
from animation_modem.v7_source_dct import direct_dct_values

GRIDS, SHAPES = v7.V7_GRIDS, v7.V7_SHAPES
SOURCE = (960, 800)           # rows, columns of the synthetic source
UP = 8                        # reconstruction upsampling over the grid


def luma_mask(guests):
    mask = np.zeros(GRIDS[0], bool)
    mask[:SHAPES[0][0], :SHAPES[0][1]] = True
    if guests:
        import json
        table = json.loads((ROOT/'animation_modem/v7_fold_table_500.json').read_text())
        flat = np.asarray(table['guests'])
        flat = flat[flat < GRIDS[0][0]*GRIDS[0][1]]
        mask.reshape(-1)[flat] = True
    return mask


def masks(guests):
    chroma = np.zeros(GRIDS[1], bool)
    chroma[:SHAPES[1][0], :SHAPES[1][1]] = True
    return [luma_mask(guests), chroma, chroma]


def encode(rgb, selection, mask_set, post=True, adjust=True):
    """The sender's path: direct encode, luma adjustment, then the kernel's
    refit. ``adjust`` is the GUI default (Luma adjustment on)."""
    frame, luminance = [], [] if adjust else None
    values = direct_dct_values(rgb, GRIDS, SHAPES, kernel=selection,
                               kernel_masks=mask_set, kernel_frame_out=frame,
                               luminance_out=luminance)
    if adjust:
        from animation_modem.v7_source_dct import luma_adjust
        values = luma_adjust(values, GRIDS, mask_set[1:], luminance[0])
    if frame and post:
        values = frame[0].post(values, GRIDS)
    return values


DISPLAY = 'ideal'             # how the viewer enlarges the picture (--display)
FILTERS = {'bilinear': Image.Resampling.BILINEAR,
           'bicubic': Image.Resampling.BICUBIC,
           'nearest': Image.Resampling.NEAREST}


def enlarge(plane01, up):
    """A native-grid plane as the viewer shows it: reduced to the sent
    pixel (the receiver's own shape) and enlarged with its upscaler."""
    rows, cols = plane01.shape
    sent = plane01.reshape(rows//2, 2, cols//2, 2).mean((1, 3))
    image = Image.fromarray(sent.astype(np.float32), 'F')
    big = image.resize((cols*up, rows*up), FILTERS[DISPLAY])
    return np.asarray(big, float)


def shown(values, mask, up=UP):
    """What the receiver draws from the luma it was sent (at up x the grid)."""
    rows, cols = GRIDS[0]
    coefficients = dctn(values[:rows*cols].reshape(rows, cols), norm='ortho')
    coefficients[~mask] = 0
    if DISPLAY != 'ideal':
        return enlarge(idctn(coefficients, norm='ortho')*.5 + .5, up)
    big = np.zeros((rows*up, cols*up))
    big[:rows, :cols] = coefficients*up
    return idctn(big, norm='ortho')*.5 + .5


def shown_rgb(values, mask_set, up=4):
    """The colour picture an ideal receiver of the sent coefficients shows."""
    planes, offset = [], 0
    for grid, mask in zip(GRIDS, mask_set):
        rows, cols = grid
        coefficients = dctn(values[offset:offset+rows*cols].reshape(rows, cols),
                            norm='ortho')
        offset += rows*cols
        coefficients[~mask] = 0
        # every plane is drawn at the luma grid's size times `up`
        height, width = GRIDS[0][0]*up, GRIDS[0][1]*up
        if DISPLAY != 'ideal':
            native = idctn(coefficients, norm='ortho')*.5 + .5
            if native.shape != GRIDS[0]:          # chroma: to the luma grid
                native = np.asarray(Image.fromarray(
                    native.astype(np.float32), 'F').resize(
                        (GRIDS[0][1], GRIDS[0][0]), Image.Resampling.BILINEAR))
            planes.append(enlarge(native, up))
            continue
        big = np.zeros((height, width))
        big[:rows, :cols] = coefficients*np.sqrt(height*width/(rows*cols))
        planes.append(idctn(big, norm='ortho')*.5+.5)
    y, cb, cr = planes[0], (planes[1]-128/255)*1.0, (planes[2]-128/255)*1.0
    # values were 2x-1 coded in [0, 1] with 128/255 neutral chroma
    cb, cr = planes[1]-.5, planes[2]-.5
    rgb = np.stack((y+1.402*cr, y-.344136*cb-.714136*cr, y+1.772*cb), -1)
    return np.uint8(np.clip(rgb, 0, 1)*255+.5)


def flat_rgb(plane01):
    return np.repeat(np.round(np.clip(plane01, 0, 1)*255).astype(np.uint8)[..., None], 3, 2)


def step_picture():
    h, w = SOURCE
    plane = np.full((h, w), .15)
    plane[:, w//2+7:] = .85           # off-centre phase: no lucky alignment
    return flat_rgb(plane), w//2+7


def grating(cycles_per_sent_pixel):
    h, w = SOURCE
    sent_width = SHAPES[0][1]
    x = (np.arange(w)+.5)/w*sent_width
    row = .5 + .25*np.sin(2*np.pi*cycles_per_sent_pixel*x)
    return flat_rgb(np.tile(row, (h, 1)))


def measure(selection, mask_set, post=True, adjust=True):
    mask = mask_set[0]
    out = {}
    rgb, edge = step_picture()
    image = shown(encode(rgb, selection, mask_set, post, adjust), mask)
    row = image[image.shape[0]//2]
    scale = row.size/SOURCE[1]
    centre = edge*scale
    levels = (.15, .85)
    out['overshoot'] = 100*max(row.max()-levels[1], levels[0]-row.min())/(levels[1]-levels[0])
    lo = np.argmax(row > levels[0]+.1*(levels[1]-levels[0]))
    hi = np.argmax(row > levels[0]+.9*(levels[1]-levels[0]))
    out['rise'] = (hi-lo)/(row.size/SHAPES[0][1])
    far = np.abs(np.arange(row.size)-centre) > 1.5*row.size/SHAPES[0][1]
    ideal = np.where(np.arange(row.size) < centre, levels[0], levels[1])
    out['ripple'] = 100*np.sqrt(np.mean((row-ideal)[far]**2))/(levels[1]-levels[0])
    for nu in (.1, .2, .3, .4):
        image = shown(encode(grating(nu), selection, mask_set, post, adjust), mask)
        line = image[image.shape[0]//2, image.shape[1]//8:-image.shape[1]//8]
        out[f'mtf@{nu}'] = (line.max()-line.min())/2/.25
    return out


def timing(selection, mask_set, repeats=20, adjust=True):
    rgb = np.random.default_rng(1).integers(0, 255, (SOURCE[0], SOURCE[1], 3), dtype=np.uint8)
    encode(rgb, selection, mask_set, adjust=adjust)
    start = time.perf_counter()
    for _ in range(repeats):
        encode(rgb, selection, mask_set, adjust=adjust)
    return (time.perf_counter()-start)/repeats*1000


def score_pictures(selections, mask_set, adjust=True, extra=()):
    """SSIMULACRA2 of what an ideal receiver of the sent coefficients shows,
    against the same picture resampled (Lanczos) to the same size. Natural
    pictures from scikit-image's bundled data, plus any in ``extra``.
    Higher is better; returns {kernel: {picture: score}}."""
    from skimage import data
    from ssimulacra2 import compute_ssimulacra2
    import io
    pictures = {}
    for name in ('astronaut', 'chelsea', 'coffee', 'rocket'):
        pictures[name] = Image.fromarray(getattr(data, name)())
    for path in extra:
        pictures[Path(path).stem] = Image.open(path).convert('RGB')
    rows, cols = GRIDS[0]
    up = 4

    def png(array):
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format='PNG')
        buffer.seek(0)
        return buffer
    results = {name: {} for name, _ in selections}
    for picture, image in pictures.items():
        target = cols/rows
        width = min(image.width, round(image.height*target))
        height = min(image.height, round(image.width/target))
        left, top = (image.width-width)//2, (image.height-height)//2
        crop = image.crop((left, top, left+width, top+height))
        reference = np.asarray(crop.resize((cols*up, rows*up), Image.Resampling.LANCZOS))
        source = np.asarray(crop)
        for name, selection in selections:
            values = encode(source, selection, mask_set, adjust=adjust)
            results[name][picture] = float(compute_ssimulacra2(
                png(reference), png(shown_rgb(values, mask_set, up))))
    return results


def contact_sheet(path, selections, mask_set, adjust=True, image=None):
    from PIL import ImageDraw
    rgb, _edge = step_picture()
    h, w = SOURCE
    test = np.zeros((h, w, 3), np.uint8)+30
    if True:
        test[:, w//2+13:] = 220
        yy, xx = np.mgrid[0:h, 0:w]
        zone = np.sin(((xx-w*.7)**2+(yy-h*.72)**2)*.0004) > 0
        patch = (slice(h//2, h-40), slice(w//2+30, w-10))
        test[patch][zone[patch]] = 245
        test[150:420, 90:300] = (230, 40, 40)
        test[560:700, 60:360] = (40, 90, 220)
    if image is not None:
        # a real picture, cropped to the wire's 5:6 shape from the centre
        picture = Image.open(image).convert('RGB')
        target = GRIDS[0][1]/GRIDS[0][0]
        width = min(picture.width, round(picture.height*target))
        height = min(picture.height, round(picture.width/target))
        left, top = (picture.width-width)//2, (picture.height-height)//2
        test = np.asarray(picture.crop((left, top, left+width, top+height)))
    tiles = []
    for name, selection in selections:
        values = encode(test, selection, mask_set, adjust=adjust)
        tile = Image.fromarray(shown_rgb(values, mask_set))
        ImageDraw.Draw(tile).text((4, 4), name, fill=(255, 220, 0))
        tiles.append(tile)
    columns = min(4, len(tiles))
    rows = -(-len(tiles)//columns)
    sheet = Image.new('RGB', (columns*tiles[0].width, rows*tiles[0].height))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns)*tile.width, (index//columns)*tile.height))
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('kernels', nargs='*', help='names (default: all)')
    parser.add_argument('--dct-kernel-dir', action='append', default=[])
    parser.add_argument('--param', action='append', default=[], metavar='KERNEL.NAME=VALUE')
    parser.add_argument('--guests', action='store_true')
    parser.add_argument('--display', choices=('ideal', *FILTERS), default='ideal',
                        help='how the viewer enlarges the picture (default: an '
                             'ideal band-limited enlargement)')
    parser.add_argument('--no-luma-adjust', action='store_true',
                        help='measure without luma adjustment (the GUI default is on)')
    parser.add_argument('--sheet', metavar='FILE')
    parser.add_argument('--score', action='store_true',
                        help='also score natural pictures with SSIMULACRA2 (needs the ssimulacra2 and scikit-image packages)')
    parser.add_argument('--image', metavar='FILE',
                        help='picture for the contact sheet (default: synthetic)')
    args = parser.parse_args()
    global DISPLAY
    DISPLAY = args.display
    registry = open_registry(args.dct_kernel_dir)
    for path, message in registry.errors:
        print(f'skipped {path}: {message}', file=sys.stderr)
    overrides = {}
    for item in args.param:
        key, value = item.split('=', 1)
        kernel, name = key.split('.', 1)
        overrides.setdefault(kernel, {})[name] = float(value)
    names = args.kernels or registry.names()
    mask_set = masks(args.guests)
    selections = [(name, registry.select(name, overrides.get(name))) for name in names]
    columns = ['overshoot', 'rise', 'ripple', 'mtf@0.1', 'mtf@0.2', 'mtf@0.3', 'mtf@0.4']
    print(f'{"kernel":22s}' + ''.join(f'{c:>10s}' for c in columns) + f'{"ms":>8s}')
    for name, selection in selections:
        result = measure(selection, mask_set, adjust=not args.no_luma_adjust)
        print(f'{name:22s}' + ''.join(f'{result[c]:10.2f}' for c in columns)
              + f'{timing(selection, mask_set, adjust=not args.no_luma_adjust):8.2f}')
    if args.score:
        scores = score_pictures(selections, mask_set, not args.no_luma_adjust,
                                [args.image] if args.image else [])
        pictures = list(next(iter(scores.values())))
        print()
        print(f'{"SSIMULACRA2":22s}' + ''.join(f'{p:>11s}' for p in pictures) + f'{"mean":>9s}')
        for name, row in scores.items():
            print(f'{name:22s}' + ''.join(f'{row[p]:11.2f}' for p in pictures)
                  + f'{np.mean(list(row.values())):9.2f}')
    if args.sheet:
        contact_sheet(args.sheet, selections, mask_set, not args.no_luma_adjust,
                      args.image)
        print('wrote', args.sheet)


if __name__ == '__main__':
    main()
