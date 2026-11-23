"""Received-picture detail measurements and deterministic display modeling."""
from functools import lru_cache

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from tools.v7_color_transforms import ColorTransform
from tools.v7_color_wire import split_planes
from tools import generate_v7_pixel_motion_test as targets


LUMA = np.array([.2126, .7152, .0722])
# Fixed source-relative regions. They are spatial boxes, not tracked anatomy.
ROIS = {'eyes': (.2, .25, .8, .48), 'hair': (.05, .02, .78, .25),
        'mouth': (.25, .48, .8, .7), 'background': (.82, .05, .98, .6)}


def display_defaults():
    from tools import v7_gl_viewer as viewer
    return {'dct': viewer.RECOMMENDED_DCT_RECONSTRUCTION,
            'edge': viewer.RECOMMENDED_EDGE_MODE,
            'edge_strength': viewer.RECOMMENDED_EDGE_STRENGTH,
            'chroma': viewer.RECOMMENDED_CHROMA_MODE,
            'upscaler': 'CPU Mitchell B=C=1/3', 'grain': 'off', 'dither': 'off',
            'alternative_bridge': 'RGB to BT.601 YCbCr on received grid'}


def raster_size(rgb, short_side=480):
    h, w = rgb.shape[:2]
    scale = short_side/min(w, h)
    return round(w*scale), round(h*scale)


def resize_rgb(rgb, size):
    return np.stack([np.asarray(Image.fromarray(np.float32(rgb[..., i]), 'F').resize(
        size, Image.Resampling.LANCZOS), float) for i in range(3)], -1)


@lru_cache(maxsize=48)
def _mitchell_matrix(n, count):
    # Exact four-tap Mitchell B=C=1/3 footprint used by the GUI float shader.
    pos = (np.arange(count)+.5)*n/count-.5
    base = np.floor(pos).astype(int)
    offsets = np.arange(-1, 3)
    x = np.abs(offsets[None, :]-(pos-base)[:, None])
    weights = np.where(x < 1, (7*x**3-12*x**2+16/3)/6,
                       np.where(x < 2, (-7/3*x**3+12*x**2-20*x+32/3)/6, 0))
    # First polynomial: 12-9B-6C=7, -18+12B+6C=-12.
    weights /= weights.sum(1, keepdims=True)
    return np.clip(base[:, None]+offsets, 0, n-1), weights


def mitchell_resize(plane, size):
    w, h = size
    ix, wx = _mitchell_matrix(plane.shape[1], w)
    result = np.sum(plane[:, ix]*wx[None], axis=2)
    iy, wy = _mitchell_matrix(plane.shape[0], h)
    return np.sum(result[iy]*wy[..., None], axis=1)


def render(wire, values, size, display=False):
    if not display:
        return resize_rgb(wire.reconstruct(values), size)
    from tools.v7_gl_viewer import dct_reconstruct_planes
    if wire.transform.base == 'pillow-ycbcr':
        planes = split_planes(values)
        transform = wire.transform
    else:
        # Shared physical brightness/color display bridge for alternatives.
        # This is explicit, not interpreting PCA/HSV component zero as luma.
        rgb = wire.reconstruct(values)
        transform = ColorTransform('ycbcr601')
        working = transform.forward(rgb)*2-1
        planes = [working[..., 0]]
        planes += [np.asarray(Image.fromarray(np.float32(working[..., i]), 'F').resize(
            (40, 48), Image.Resampling.BOX), float) for i in (1, 2)]
    defaults = display_defaults()
    planes = dct_reconstruct_planes(planes, defaults['dct'], edge=defaults['edge'],
                                    edge_strength=defaults['edge_strength'],
                                    chroma=defaults['chroma'])
    shown = np.stack([(mitchell_resize(p, size)+1)/2 for p in planes], -1)
    return transform.inverse(shown)


def crop_box(array, box, grid=(1, 1)):
    h, w = array.shape[:2]
    x0, y0, x1, y1 = box
    x0, x1 = (int(round(x*w/grid[0])) for x in (x0, x1))
    y0, y1 = (int(round(y*h/grid[1])) for y in (y0, y1))
    return array[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]


def _correlation(a, b):
    a, b = a.ravel()-a.mean(), b.ravel()-b.mean()
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/denominator) if denominator > 1e-12 else None


def _ssim(a, b):
    ma, mb = gaussian_filter(a, 1.5), gaussian_filter(b, 1.5)
    va = gaussian_filter(a*a, 1.5)-ma*ma
    vb = gaussian_filter(b*b, 1.5)-mb*mb
    covariance = gaussian_filter(a*b, 1.5)-ma*mb
    return float(np.mean((2*ma*mb+.01**2)*(2*covariance+.03**2)/
                         ((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))))


def picture_metrics(source, received, natural=False):
    clip_rate = float(np.mean((received < 0) | (received > 1)))
    image = np.clip(received, 0, 1)
    mse = float(np.mean((source-image)**2))
    y, yh = source@LUMA, image@LUMA
    result = {'psnr': -10*np.log10(max(mse, 1e-15)), 'ssim': _ssim(y, yh),
              'mse': mse, 'clipping_fraction': clip_rate}
    lab = ColorTransform('lab')
    a = lab.forward(source)*(lab.high-lab.low)+lab.low
    b = lab.forward(image)*(lab.high-lab.low)+lab.low
    result['delta_e76'] = float(np.mean(np.linalg.norm(a-b, axis=-1)))
    if natural:
        for name, box in ROIS.items():
            a, b = crop_box(y, box), crop_box(yh, box)
            for sigma in (1, 2, 4):
                band = gaussian_filter(a, sigma)-gaussian_filter(a, sigma*2)
                recovered = gaussian_filter(b, sigma)-gaussian_filter(b, sigma*2)
                result[f'{name}_band{sigma}_correlation'] = _correlation(band, recovered)
                result[f'{name}_band{sigma}_nmse'] = float(np.mean((band-recovered)**2)/
                                                        max(np.mean(band**2), 1e-8))
    return result


def _fundamental(profile, pitch_pixels):
    x = np.arange(len(profile))+.5
    angle = 2*np.pi*x/pitch_pixels
    design = np.stack((np.ones_like(x), np.cos(angle), np.sin(angle)), -1)
    fit = np.linalg.lstsq(design, profile, rcond=None)[0]
    return fit[1:]


def bar_metrics(source, received):
    result = {}
    source_y, received_y = source@LUMA, received@LUMA
    for orientation, spec, axis in (('vertical', targets.VERTICAL_BARS, 0),
                                    ('horizontal', targets.HORIZONTAL_BARS, 1)):
        passing = []
        for pitch, c0, c1 in targets.bar_groups(spec):
            box = (c0, targets.BAR_ROWS[0], c1, targets.BAR_ROWS[1])
            a = crop_box(source_y, box, (80, 96)).mean(axis=axis)
            b = crop_box(received_y, box, (80, 96)).mean(axis=axis)
            dimension = source.shape[1] if axis == 0 else source.shape[0]
            grid_dimension = 80 if axis == 0 else 96
            fa, fb = _fundamental(a, pitch*dimension/grid_dimension), _fundamental(
                b, pitch*dimension/grid_dimension)
            contrast = float(np.linalg.norm(fb)/max(np.linalg.norm(fa), 1e-9))
            phase = float(np.arctan2(fa[0]*fb[1]-fa[1]*fb[0], fa@fb))
            ac, bc = a-a.mean(), b-b.mean()
            scale = float(ac@bc/max(ac@ac, 1e-12))
            residual = float(np.linalg.norm(bc-scale*ac)/max(np.linalg.norm(ac), 1e-9))
            passed = contrast >= .2 and abs(phase) <= np.pi/8 and scale > 0 and residual <= .25
            key = f'{orientation}_pitch{pitch}'
            result.update({key+'_contrast': contrast, key+'_phase': phase,
                           key+'_residual': residual, key+'_resolved': bool(passed),
                           key+'_faithful_contrast': contrast if passed else 0.0,
                           key+'_cycles_per_picture': grid_dimension/pitch})
            if passed:
                passing.append(pitch)
        result[orientation+'_finest_pitch'] = min(passing) if passing else None
    for name, box in (('checker2', targets.CHECK_BOXES[0][:4]),
                      ('checker4', targets.CHECK_BOXES[1][:4]),
                      ('diagonal', targets.DIAGONAL_BOX), ('text', targets.TEXT_BOX)):
        a = crop_box(source_y, box, (80, 96))
        b = crop_box(received_y, box, (80, 96))
        result[name+'_correlation'] = _correlation(a, b)
        result[name+'_mse'] = float(np.mean((a-b)**2))
    for name, box, axis in (('vertical_edge', targets.VERTICAL_EDGE_BOX, 0),
                             ('horizontal_edge', targets.HORIZONTAL_EDGE_BOX, 1)):
        a = crop_box(source_y, box, (80, 96)).mean(axis=axis)
        b = crop_box(received_y, box, (80, 96)).mean(axis=axis)
        low, high = np.quantile(a, [.05, .95])
        span = high-low
        if span < .1:
            continue
        crossing = int(np.argmin(abs(a-(low+high)/2)))
        if crossing < 2 or crossing > len(a)-3:
            continue
        lo = max(0, crossing-len(a)//3)
        hi = min(len(a), crossing+len(a)//3+1)
        segment = b[lo:hi]
        p10, p90 = np.argmin(abs(segment-(low+.1*span))), np.argmin(abs(segment-(low+.9*span)))
        result[name+'_width_pixels'] = int(abs(p90-p10))
        result[name+'_overshoot'] = float(max(0, b.max()-high)/span)
        result[name+'_undershoot'] = float(max(0, low-b.min())/span)
        flat = abs(a-(low+high)/2) > .4*span
        result[name+'_flat_error'] = float(np.sqrt(np.mean((b[flat]-a[flat])**2)))
    return result


def summarize(rows):
    result = {'frames': len(rows), 'pictures_shown': sum(r['shown'] for r in rows),
              'valid_packets': sum(r['valid'] for r in rows),
              'initial_unavailable': sum(not r['available'] for r in rows)}
    longest = current = 0
    for row in rows:
        current = 0 if row['shown'] else current+1
        longest = max(longest, current)
    result['longest_hold_packets'] = longest
    shown_indices = [i for i, row in enumerate(rows) if row['shown']]
    result['interior_picture_gaps'] = (sum(not rows[i]['shown'] for i in range(
        shown_indices[1]+1, shown_indices[-1])) if len(shown_indices) >= 2 else 0)
    for key in sorted(set().union(*(r.keys() for r in rows))):
        if key.endswith('_resolved'):
            result[key+'_fraction'] = sum(bool(r.get(key)) for r in rows)/max(len(rows), 1)
        values = [r[key] for r in rows if isinstance(r.get(key), (int, float))
                  and not isinstance(r[key], bool) and np.isfinite(r[key])]
        if values and key not in ('packet', 'source_index'):
            result[key+'_median'] = float(np.median(values))
            result[key+'_p10'] = float(np.percentile(values, 10))
    return result
