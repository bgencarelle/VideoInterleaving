"""Minimal GLFW/ModernGL viewer for the standalone V7 receiver."""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from animation_modem.imaging import values_image


DISPLAY_MODES = (
    'nearest', 'bilinear', 'sharp-bilinear', 'bicubic',
    'spline36', 'robidoux', 'robidoux-sharp', 'cubic-bspline',
    'kaiser-sinc', 'hann-sinc', 'ewa-jinc',
)
FULLSCREEN_TOOLBAR_HIDE_SECONDS = 2.0
FULLSCREEN_TOOLBAR_EDGE = 14
DISPLAY_LABELS = {
    'nearest': 'Nearest',
    'bilinear': 'Bilinear',
    'sharp-bilinear': 'Sharp bilinear',
    'bicubic': 'Bicubic · Mitchell · recommended',
    'spline36': 'Spline36',
    'robidoux': 'Robidoux',
    'robidoux-sharp': 'Robidoux Sharp',
    'cubic-bspline': 'Cubic B-spline',
    'kaiser-sinc': 'Kaiser sinc · β=8.6',
    'hann-sinc': 'Hann sinc',
    'ewa-jinc': 'EWA Jinc · 3 lobes',
}
FLOAT_MODE_IDS = {
    'bilinear': 0,
    'sharp-bilinear': 1,
    'bicubic': 2,
    'ewa-jinc': 3,
    'spline36': 4,
    'robidoux': 5,
    'robidoux-sharp': 6,
    'kaiser-sinc': 7,
    'hann-sinc': 8,
    'cubic-bspline': 9,
}
FILTER_LUT_MODES = frozenset(('ewa-jinc', 'kaiser-sinc', 'hann-sinc'))
FILTER_PRECOMPUTE_MODES = frozenset((
    'spline36', 'robidoux', 'robidoux-sharp', 'cubic-bspline',
    'kaiser-sinc', 'hann-sinc',
))
FILTER_INTERMEDIATE_SCALE = 4
FILTER_LUT_SIZE = 2048
EWA_JINC_RADIUS = 3.2383154841662362
KAISER_SINC_RADIUS = 3.0
KAISER_SINC_BETA = 8.6
HANN_SINC_RADIUS = 3.0
DCT_RECONSTRUCTION_MODES = ('off', '2x', '4x', '8x', '16x', 'viewport', 'pixel')
# 'pixel' shows the picture on the grid the wire carries in full (the pixel
# grid an aspect-fold-500 packet names, else half the coder grid per axis:
# 48x40 luma, 24x20 chroma) as hard pixels, each
# repeated PIXEL_REPEAT times so the output has the 4x mode's size. With the
# sender's pixel encode and Fold 500 (whose sent luma is exactly that
# rectangle) a 40x48 picture arrives pixel for pixel, up to channel noise and
# half-resolution colour.
PIXEL_REPEAT = 8
# Display grain: fine noise in flat picture areas only, where it breaks up the
# regular ringing ripple of a band-limited picture. Textured areas and edges
# are left alone. Amplitude: about 2.5/255 standard deviation in luma.
GRAIN_MODES = ('off', 'flat')
GRAIN_LABELS = {'off': 'Off', 'flat': 'Flat areas · masks ringing'}
# Edge reconstruction: rebuild luma as the sharpest, flattest picture that
# still matches every received coefficient (see
# animation_modem.v7_dct_display.edge_consistent_plane). Removes the ringing
# ripple and sharpens edges. 'on' works on the coder grid (about 1.4 ms per
# new picture on one core); 'high' works at twice the grid (about 4 ms).
EDGE_MODES = ('on', 'high', 'off')
EDGE_LABELS = {'on': 'Consistent · sharp edges, no ripple · recommended',
               'high': 'Consistent, high · twice the grid, more CPU',
               'off': 'Off'}
# Recommended receiver display (measured; see docs/MODEM_MODE.md): edge
# reconstruction on the coder grid, 4x DCT reconstruction, then the
# shader's bicubic to the window. 4x + bicubic matches the exact viewport
# evaluation to 77 dB PSNR at 1080 lines (nearest: 43 dB) for about a fifth of
# the CPU time.
RECOMMENDED_DISPLAY_MODE = 'bicubic'
RECOMMENDED_DCT_RECONSTRUCTION = '4x'
RECOMMENDED_EDGE_MODE = 'on'
# Edge strength: how much of the edge reconstruction is mixed into the plain
# picture. Full strength suits flat-shaded pictures and looks painted on
# natural texture. Real modem, SSIMULACRA2 over no reconstruction at 25 / 50 /
# 75 / 100 %: clean cartoon +0.8 / +1.6 / +2.2 / +2.7, photos +1.1 / +1.7 /
# +1.9 / +1.6; MP3 320 cartoon +0.3 / +0.5 / +0.7 / +0.9, photos +0.3 / +0.4
# / +0.3 / -0.2.
EDGE_STRENGTHS = (1.0, .75, .5, .25)
EDGE_STRENGTH_LABELS = {1.0: '100% · flattest, sharpest',
                        .75: '75% · recommended',
                        .5: '50%', .25: '25% · most natural texture'}
RECOMMENDED_EDGE_STRENGTH = .75
GRAIN_AMOUNT = 0.024            # triangular +-amount; sigma = amount/sqrt(6)
GRAIN_FLAT_SIGMA = 1.2          # luma grid samples
GRAIN_FLAT_CONTRAST = 0.06      # local luma s.d. (code units) that stops grain
DCT_RECONSTRUCTION_LABELS = {
    'off': 'Off',
    '2x': '2×',
    '4x': '4× · recommended',
    '8x': '8×',
    '16x': '16× · fast PCs',
    'viewport': 'Viewport size',
    'pixel': 'Pixel · hard pixels on the sent grid',
}


def toolbar_layout(width, open_dropdown=None):
    """Logical-pixel hit regions for the compact in-viewer toolbar."""
    width = max(320, int(width))
    upscale = (12, 7, 204, 42)
    panel = (224, 7, 382, 42)
    fullscreen = (width-102, 7, width-12, 42)
    hits = {'upscale_button': upscale, 'panel_button': panel}
    if width >= 500:
        hits['fullscreen_button'] = fullscreen
    if width >= 720:
        hits['save_default_button'] = (width-226, 7, width-112, 42)
    if open_dropdown == 'upscale':
        for index, mode in enumerate(DISPLAY_MODES):
            y = 48 + index*29
            hits[f'mode:{mode}'] = (upscale[0], y, upscale[2], y+28)
    elif open_dropdown == 'panel':
        for index, visible in enumerate((True, False)):
            y = 48 + index*29
            hits[f'panel:{int(visible)}'] = (panel[0], y, panel[2], y+28)
    return hits


def _contains(rect, x, y):
    return rect[0] <= x < rect[2] and rect[1] <= y < rect[3]


def _preference_path():
    root = Path(os.environ.get('XDG_CONFIG_HOME') or Path.home()/'.config')
    return root/'modemTest'/'v7_display.json'


def _load_display_default():
    try:
        data = json.loads(_preference_path().read_text())
        if (isinstance(data, dict) and data.get('version') == 1 and
                data.get('mode') in DISPLAY_MODES):
            return data['mode']
    except (OSError, ValueError, TypeError):
        pass
    return 'nearest'


def _save_display_default(mode):
    if mode not in DISPLAY_MODES:
        raise ValueError(f'unsupported display mode: {mode}')
    path = _preference_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'version': 1, 'mode': mode},
                                    sort_keys=True)+'\n')
    os.replace(temporary, path)


def _toolbar_image(size, mode, show_details, open_dropdown, notice=''):
    """Build a small, dark toolbar and optional dropdown using the viewer style."""
    width = max(320, int(size[0]))
    height = 48
    if open_dropdown == 'upscale':
        height += len(DISPLAY_MODES)*29 + 4
    elif open_dropdown == 'panel':
        height += 2*29 + 4
    image = Image.new('RGBA', (width, height), (9, 16, 24, 246))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 14)
        small = ImageFont.truetype('DejaVuSans.ttf', 12)
    except OSError:
        try:
            font = ImageFont.load_default(size=14)
            small = ImageFont.load_default(size=12)
        except TypeError:
            font = small = ImageFont.load_default()

    draw.rectangle((0, 46, width, 47), fill=(50, 70, 89, 255))
    draw.rounded_rectangle((12, 7, 204, 41), radius=5,
                           fill=(25, 39, 52, 255),
                           outline=(66, 94, 116, 255), width=1)
    draw.text((22, 15), f'Upscale  {DISPLAY_LABELS[mode]}  ▾',
              fill=(232, 240, 246, 255), font=font)
    panel_label = 'Info panel  On' if show_details else 'Info panel  Off'
    draw.rounded_rectangle((224, 7, 382, 41), radius=5,
                           fill=(25, 39, 52, 255),
                           outline=(66, 94, 116, 255), width=1)
    draw.text((234, 15), f'{panel_label}  ▾',
              fill=(232, 240, 246, 255), font=small)
    if width >= 920:
        draw.text((404, 17), 'F fullscreen  ·  I info  ·  Esc exit',
                  fill=(137, 162, 184, 255), font=small)
    save_box = (width-226, 7, width-112, 41)
    if width >= 500:
        draw.rounded_rectangle((width-102, 7, width-12, 41), radius=5,
                               fill=(25, 39, 52, 255),
                               outline=(66, 94, 116, 255), width=1)
        draw.text((width-90, 15), 'Fullscreen',
                  fill=(232, 240, 246, 255), font=small)
    if width >= 720:
        draw.rounded_rectangle(save_box, radius=5,
                               fill=(25, 39, 52, 255),
                               outline=(66, 94, 116, 255), width=1)
        draw.text((save_box[0]+8, 15), notice or 'Save default',
                  fill=(232, 240, 246, 255), font=small)

    if open_dropdown:
        hits = toolbar_layout(width, open_dropdown)
        prefix = 'mode:' if open_dropdown == 'upscale' else 'panel:'
        rows = [key for key in hits if key.startswith(prefix)]
        active = (f'mode:{mode}' if open_dropdown == 'upscale' else
                  f'panel:{int(show_details)}')
        anchor = (12, 224) if open_dropdown == 'upscale' else (224, 382)
        draw.rounded_rectangle((anchor[0], 47, anchor[1], height-3), radius=4,
                               fill=(18, 29, 40, 255),
                               outline=(66, 94, 116, 255), width=1)
        for key in rows:
            box = hits[key]
            if key == active:
                draw.rectangle((box[0]+1, box[1], box[2]-1, box[3]),
                               fill=(45, 76, 98, 255))
            if key.startswith('mode:'):
                label = DISPLAY_LABELS[key.split(':', 1)[1]]
            else:
                label = 'Show diagnostics' if key.endswith(':1') else 'Hide diagnostics'
            draw.text((box[0]+10, box[1]+6), label,
                      fill=(232, 240, 246, 255), font=small)
    return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))


def float_planes(values, shapes):
    """Copy decoded Y/Cb/Cr values into owned, unclipped float32 planes."""
    values = np.asarray(values)
    shapes = tuple((int(rows), int(cols)) for rows, cols in shapes)
    if len(shapes) not in (1, 3) or any(rows <= 0 or cols <= 0
                                       for rows, cols in shapes):
        raise ValueError('display planes need one or three positive shapes')
    expected = sum(rows*cols for rows, cols in shapes)
    if values.ndim != 1 or values.size != expected:
        raise ValueError('decoded values do not match their display shapes')
    if not np.all(np.isfinite(values)):
        raise ValueError('decoded display values must be finite')
    planes = []
    offset = 0
    for rows, cols in shapes:
        count = rows*cols
        planes.append(np.array(values[offset:offset+count].reshape(rows, cols),
                               dtype=np.float32, order='C', copy=True))
        offset += count
    if len(planes) == 1:
        # Code 128 is the neutral chroma value in the 8-bit Pillow convention.
        neutral_chroma = np.full((1, 1), 1.0/255.0, np.float32)
        planes.extend((neutral_chroma.copy(), neutral_chroma))
    return tuple(planes)


def flat_area_mask(luma):
    """0..1 per luma grid sample: 1 where the decoded picture is flat.

    Local standard deviation over about one grid sample, mapped linearly to
    zero at GRAIN_FLAT_CONTRAST. Evaluated on the decoded grid (96x80), not at
    display size, so it costs well under a millisecond per frame.
    """
    from scipy.ndimage import gaussian_filter
    plane = np.asarray(luma, dtype=np.float32)
    mean = gaussian_filter(plane, GRAIN_FLAT_SIGMA, mode='nearest')
    square = gaussian_filter(plane*plane, GRAIN_FLAT_SIGMA, mode='nearest')
    deviation = np.sqrt(np.maximum(square-mean*mean, 0.0))
    return np.ascontiguousarray(
        np.clip(1.0-deviation/GRAIN_FLAT_CONTRAST, 0.0, 1.0), dtype=np.float32)


def dct_reconstruct_planes(planes, mode, viewport_size=None, edge=False,
                           edge_strength=1.0, pixel_shapes=None):
    """Resample decoded planes by evaluating their retained DCT spectrum.

    The input planes are already spatial-domain inverse-DCT output. Transforming
    them back recovers the transmitted low-frequency corner; padding or
    truncating that corner before the inverse transform evaluates the same
    cosine reconstruction on a larger or smaller display grid. The coefficient
    scale preserves pixel amplitude across the changed orthonormal grid size.

    With ``edge`` ('on', 'high' or True for 'on') the luma plane is first
    rebuilt by consistent reconstruction (EDGE_MODES); output sizes are
    unchanged except that with ``mode`` 'off' and 'high' the luma comes back at
    twice its grid. ``edge_strength`` (0 to 1) mixes that rebuild with the
    plain picture. ``pixel_shapes`` is the pixel grid the picture was sent
    on, when the wire names one; mode 'pixel' then shows that grid.
    """
    if mode not in DCT_RECONSTRUCTION_MODES:
        raise ValueError(f'unknown DCT reconstruction mode {mode!r}')
    planes = tuple(np.asarray(plane, dtype=np.float32) for plane in planes)
    if not planes or any(plane.ndim != 2 or min(plane.shape) <= 0
                         for plane in planes):
        raise ValueError('DCT reconstruction needs non-empty 2-D planes')
    if mode == 'pixel':
        # The sent grid itself, as hard pixels; edge reconstruction does not
        # apply (nothing is interpolated).
        from animation_modem.v7_dct_display import reconstruct_plane
        if pixel_shapes is None or len(pixel_shapes) != len(planes):
            pixel_shapes = tuple((max(1, plane.shape[0]//2),
                                  max(1, plane.shape[1]//2))
                                 for plane in planes)
        return tuple(np.ascontiguousarray(np.repeat(np.repeat(
            reconstruct_plane(plane, (int(rows), int(cols))),
            PIXEL_REPEAT, axis=0), PIXEL_REPEAT, axis=1))
            for plane, (rows, cols) in zip(planes, pixel_shapes))
    if edge is True:
        edge = 'on'
    if edge not in (False, None, 'off') and edge not in EDGE_MODES:
        raise ValueError(f'unknown edge reconstruction mode {edge!r}')
    if edge not in (False, None, 'off'):
        from animation_modem.v7_dct_display import (EDGE_HIGH,
                                                    edge_consistent_plane,
                                                    reconstruct_plane,
                                                    viewport_shapes)
        luma = edge_consistent_plane(
            planes[0], strength=edge_strength,
            **(EDGE_HIGH if edge == 'high' else {}))
        if mode == 'off':
            return (luma,) + planes[1:]
        if mode == 'viewport':
            if viewport_size is None or len(viewport_size) != 2:
                raise ValueError('viewport-size DCT reconstruction needs a size')
            shapes = viewport_shapes(tuple(plane.shape for plane in planes),
                                     viewport_size)
        else:
            scale = int(mode[:-1])
            shapes = tuple((plane.shape[0]*scale, plane.shape[1]*scale)
                           for plane in planes)
        rest = dct_reconstruct_planes(planes[1:], mode, viewport_size) \
            if mode != 'viewport' else tuple(
                reconstruct_plane(plane, shape)
                for plane, shape in zip(planes[1:], shapes[1:]))
        return (reconstruct_plane(luma, shapes[0]),) + tuple(rest)
    if mode == 'off':
        return planes

    if mode == 'viewport':
        if viewport_size is None or len(viewport_size) != 2:
            raise ValueError('viewport-size DCT reconstruction needs a size')
        # Evaluate the decoded spectrum directly on the display grid (only
        # its occupied part, separable, one thread): same result as the
        # padded full-size inverse DCT below, several times faster.
        from animation_modem.v7_dct_display import reconstruct_planes
        return reconstruct_planes(planes, viewport_size)

    scale = int(mode[:-1])
    target_shapes = tuple((plane.shape[0]*scale,
                           plane.shape[1]*scale) for plane in planes)
    if scale > 8:
        # 16x (1,536x1,280 luma) evaluates the decoded spectrum directly: the
        # padded inverse DCT below costs about twice as much at this size.
        from animation_modem.v7_dct_display import reconstruct_plane
        return tuple(reconstruct_plane(plane, shape)
                     for plane, shape in zip(planes, target_shapes))

    # Fixed 2x/4x/8x modes (4x is the receiver default) are unchanged.

    from scipy.fft import dctn, idctn

    output = []
    for plane, (target_height, target_width) in zip(planes, target_shapes):
        source_height, source_width = plane.shape
        coefficients = dctn(plane, norm='ortho')
        resized = np.zeros((target_height, target_width),
                           dtype=coefficients.dtype)
        copy_height = min(source_height, target_height)
        copy_width = min(source_width, target_width)
        resized[:copy_height, :copy_width] = coefficients[
            :copy_height, :copy_width]
        resized *= math.sqrt((target_height*target_width) /
                             (source_height*source_width))
        output.append(np.ascontiguousarray(
            idctn(resized, norm='ortho'), dtype=np.float32))
    return tuple(output)


def _bessel_j1_series(x):
    """J1 evaluated by its convergent power series for the short Jinc LUT."""
    half_x = .5*x
    term = half_x
    total = term
    for order in range(1, 40):
        term *= -(half_x*half_x)/(order*(order+1))
        total += term
        if abs(term) <= 1e-16*max(1.0, abs(total)):
            break
    return total


def _jinc(x):
    if abs(x) < 1e-12:
        return 1.0
    argument = math.pi*x
    return 2.0*_bessel_j1_series(argument)/argument


def build_filter_lut(mode, sample_count=FILTER_LUT_SIZE):
    """Return normalized 1-D or radial kernel weights for GLSL lookup."""
    sample_count = int(sample_count)
    if sample_count < 2:
        raise ValueError('filter LUT needs at least two samples')
    if mode == 'ewa-jinc':
        radius = EWA_JINC_RADIUS
        distances = np.linspace(0.0, radius, sample_count)
        window_scale = 1.2196698912665045/radius
        weights = np.asarray([
            _jinc(float(distance))*_jinc(float(distance)*window_scale)
            for distance in distances], dtype=np.float64)
    elif mode == 'kaiser-sinc':
        radius = KAISER_SINC_RADIUS
        distances = np.linspace(0.0, radius, sample_count)
        window = np.i0(KAISER_SINC_BETA*np.sqrt(
            np.maximum(0.0, 1.0-(distances/radius)**2)))/np.i0(
                KAISER_SINC_BETA)
        weights = np.sinc(distances)*window
    elif mode == 'hann-sinc':
        radius = HANN_SINC_RADIUS
        distances = np.linspace(0.0, radius, sample_count)
        window = .5+.5*np.cos(np.pi*distances/radius)
        weights = np.sinc(distances)*window
    else:
        weights = np.ones(1, dtype=np.float32)
    return np.ascontiguousarray(weights, dtype=np.float32)


def _separable_filter_radius(mode):
    return 3.0 if mode in ('spline36', 'kaiser-sinc', 'hann-sinc') else 2.0


def _separable_filter_weight(mode, distance):
    x = np.abs(np.asarray(distance, dtype=np.float64))
    if mode == 'spline36':
        result = np.zeros_like(x)
        mask = x < 1.0
        t = x[mask]
        result[mask] = ((13.0/11.0*t - 453.0/209.0)*t - 3.0/209.0)*t + 1.0
        mask = (x >= 1.0) & (x < 2.0)
        t = x[mask]-1.0
        result[mask] = ((-6.0/11.0*t + 270.0/209.0)*t - 156.0/209.0)*t
        mask = (x >= 2.0) & (x < 3.0)
        t = x[mask]-2.0
        result[mask] = ((1.0/11.0*t - 45.0/209.0)*t + 26.0/209.0)*t
        return result
    if mode in ('robidoux', 'robidoux-sharp', 'cubic-bspline'):
        if mode == 'robidoux':
            b, c = 0.37821575509399867, 0.31089212245300067
        elif mode == 'robidoux-sharp':
            b, c = 0.2620145123990142, 0.3689927438004929
        else:
            b, c = 1.0, 0.0
        result = np.zeros_like(x)
        mask = x < 1.0
        t = x[mask]
        result[mask] = ((12.0-9.0*b-6.0*c)*t**3
                        + (-18.0+12.0*b+6.0*c)*t**2
                        + 6.0-2.0*b)/6.0
        mask = (x >= 1.0) & (x < 2.0)
        t = x[mask]
        result[mask] = ((-b-6.0*c)*t**3
                        + (6.0*b+30.0*c)*t**2
                        + (-12.0*b-48.0*c)*t
                        + 8.0*b+24.0*c)/6.0
        return result
    if mode == 'kaiser-sinc':
        window = np.i0(KAISER_SINC_BETA*np.sqrt(
            np.maximum(0.0, 1.0-(x/KAISER_SINC_RADIUS)**2)))/np.i0(
                KAISER_SINC_BETA)
        return np.sinc(x)*window
    if mode == 'hann-sinc':
        window = .5+.5*np.cos(np.pi*x/HANN_SINC_RADIUS)
        return np.sinc(x)*window
    raise ValueError(f'no separable kernel for display mode {mode!r}')


def _resample_axis(values, output_length, axis, mode):
    values = np.asarray(values, dtype=np.float32)
    source_length = values.shape[axis]
    output_length = int(output_length)
    if output_length <= 0:
        raise ValueError('resampled plane dimensions must be positive')
    scale = max(source_length/output_length, 1.0)
    support = _separable_filter_radius(mode)*scale
    radius = int(math.ceil(support))
    offsets = np.arange(-radius, radius+1, dtype=np.int64)
    positions = ((np.arange(output_length, dtype=np.float64)+.5)
                 *source_length/output_length-.5)
    base = np.floor(positions).astype(np.int64)
    distances = (offsets[None, :] - (positions-base)[:, None])/scale
    weights = _separable_filter_weight(mode, distances)
    weights[np.abs(distances) >= _separable_filter_radius(mode)] = 0.0
    total = weights.sum(axis=1, keepdims=True)
    if np.any(np.abs(total) < 1e-12):
        raise ValueError(f'{mode} produced a zero-weight resampling footprint')
    weights /= total
    indexes = np.clip(base[:, None]+offsets[None, :], 0, source_length-1)
    samples = np.take(values, indexes, axis=axis)
    if axis == 0:
        result = np.sum(samples*weights[:, :, None], axis=1, dtype=np.float64)
    elif axis == 1:
        result = np.sum(samples*weights[None, :, :], axis=2, dtype=np.float64)
    else:
        raise ValueError('resampling axis must be 0 or 1')
    return np.ascontiguousarray(result, dtype=np.float32)


def resample_filter_planes(planes, mode,
                           scale=FILTER_INTERMEDIATE_SCALE):
    """Build bounded 4x separable reconstruction planes for the display GPU."""
    if mode not in FILTER_PRECOMPUTE_MODES:
        raise ValueError(f'{mode!r} is not a precomputed separable filter')
    scale = int(scale)
    if scale < 1:
        raise ValueError('intermediate scale must be positive')
    output = []
    for plane in planes:
        height, width = plane.shape
        horizontal = _resample_axis(plane, width*scale, 1, mode)
        output.append(_resample_axis(horizontal, height*scale, 0, mode))
    return tuple(output)


VERTEX_SHADER = '''#version 330
out vec2 uv;
void main() {
    vec2 position;
    if (gl_VertexID == 0) position = vec2(-1.0, -1.0);
    else if (gl_VertexID == 1) position = vec2(3.0, -1.0);
    else position = vec2(-1.0, 3.0);
    gl_Position = vec4(position, 0.0, 1.0);
    uv = vec2((position.x + 1.0) * 0.5, (1.0 - position.y) * 0.5);
}
'''

FRAGMENT_SHADER = '''#version 330
uniform sampler2D image;
in vec2 uv;
out vec4 color;
void main() {
    color = texture(image, uv);
}
'''


FLOAT_FRAGMENT_SHADER = '''#version 330
uniform sampler2D plane_y;
uniform sampler2D plane_cb;
uniform sampler2D plane_cr;
uniform sampler2D kernel_lut;
uniform sampler2D grain_mask;
uniform float grain_amount;
uniform int grain_seed;
uniform int reconstruction;
uniform int filtered_intermediate;
uniform vec2 output_size;
in vec2 uv;
out vec4 color;

float mitchell_weight(float distance) {
    float x = abs(distance);
    if (x < 1.0)
        return ((7.0*x - 12.0)*x*x + 16.0/3.0)/6.0;
    if (x < 2.0)
        return ((-7.0/3.0*x + 12.0)*x - 20.0)*x/6.0 + 16.0/9.0;
    return 0.0;
}

float sample_mitchell(sampler2D plane, vec2 coord) {
    ivec2 size = textureSize(plane, 0);
    vec2 sample_position = coord*vec2(size) - 0.5;
    vec2 fraction = fract(sample_position);
    ivec2 base = ivec2(floor(sample_position));
    float value = 0.0;
    float weight_sum = 0.0;
    for (int y = -1; y <= 2; ++y) {
        float wy = mitchell_weight(float(y) - fraction.y);
        for (int x = -1; x <= 2; ++x) {
            float weight = wy*mitchell_weight(float(x) - fraction.x);
            ivec2 at = clamp(base + ivec2(x, y), ivec2(0), size-1);
            value += texelFetch(plane, at, 0).r*weight;
            weight_sum += weight;
        }
    }
    return value/weight_sum;
}

float cubic_bc_weight(float x, float b, float c) {
    x = abs(x);
    if (x < 1.0)
        return ((12.0 - 9.0*b - 6.0*c)*x*x*x
                + (-18.0 + 12.0*b + 6.0*c)*x*x
                + (6.0 - 2.0*b))/6.0;
    if (x < 2.0)
        return ((-b - 6.0*c)*x*x*x
                + (6.0*b + 30.0*c)*x*x
                + (-12.0*b - 48.0*c)*x
                + (8.0*b + 24.0*c))/6.0;
    return 0.0;
}

float spline36_weight(float x) {
    x = abs(x);
    if (x < 1.0)
        return ((13.0/11.0*x - 453.0/209.0)*x - 3.0/209.0)*x + 1.0;
    if (x < 2.0) {
        float t = x - 1.0;
        return ((-6.0/11.0*t + 270.0/209.0)*t - 156.0/209.0)*t;
    }
    if (x < 3.0) {
        float t = x - 2.0;
        return ((1.0/11.0*t - 45.0/209.0)*t + 26.0/209.0)*t;
    }
    return 0.0;
}

float lut_weight(float distance, float support) {
    float count = float(textureSize(kernel_lut, 0).x);
    float position = clamp(abs(distance)/support, 0.0, 1.0);
    float texel = (position*(count-1.0) + 0.5)/count;
    return texture(kernel_lut, vec2(texel, 0.5)).r;
}

float separable_radius() {
    if (reconstruction == 4 || reconstruction == 7 || reconstruction == 8)
        return 3.0;
    if (reconstruction == 5 || reconstruction == 6 ||
        reconstruction == 9)
        return 2.0;
    return 0.0;
}

float separable_weight(float distance) {
    if (reconstruction == 4)
        return spline36_weight(distance);
    if (reconstruction == 5)
        return cubic_bc_weight(distance, 0.3782157551, 0.3108921225);
    if (reconstruction == 6)
        return cubic_bc_weight(distance, 0.2620145124, 0.3689927438);
    if (reconstruction == 7)
        return lut_weight(distance, 3.0);
    if (reconstruction == 8)
        return lut_weight(distance, 3.0);
    if (reconstruction == 9)
        return cubic_bc_weight(distance, 1.0, 0.0);
    return 0.0;
}

float sample_separable(sampler2D plane, vec2 coord) {
    ivec2 size = textureSize(plane, 0);
    vec2 sample_position = coord*vec2(size) - 0.5;
    vec2 fraction = fract(sample_position);
    ivec2 base = ivec2(floor(sample_position));
    float radius = separable_radius();
    float value = 0.0;
    float weight_sum = 0.0;
    for (int y = -3; y <= 3; ++y) {
        float dy = float(y) - fraction.y;
        if (abs(dy) >= radius)
            continue;
        float wy = separable_weight(dy);
        for (int x = -3; x <= 3; ++x) {
            float dx = float(x) - fraction.x;
            if (abs(dx) >= radius)
                continue;
            float weight = wy*separable_weight(dx);
            ivec2 at = clamp(base + ivec2(x, y), ivec2(0), size-1);
            value += texelFetch(plane, at, 0).r*weight;
            weight_sum += weight;
        }
    }
    return value/weight_sum;
}

float sample_ewa_jinc(sampler2D plane, vec2 coord) {
    const float radius = 3.2383154841662362;
    ivec2 size = textureSize(plane, 0);
    vec2 sample_position = coord*vec2(size) - 0.5;
    vec2 fraction = fract(sample_position);
    ivec2 base = ivec2(floor(sample_position));
    float value = 0.0;
    float weight_sum = 0.0;
    for (int y = -4; y <= 4; ++y) {
        float dy = float(y) - fraction.y;
        for (int x = -4; x <= 4; ++x) {
            float dx = float(x) - fraction.x;
            float distance = length(vec2(dx, dy));
            if (distance >= radius)
                continue;
            float weight = lut_weight(distance, radius);
            ivec2 at = clamp(base + ivec2(x, y), ivec2(0), size-1);
            value += texelFetch(plane, at, 0).r*weight;
            weight_sum += weight;
        }
    }
    return value/weight_sum;
}

vec2 sharp_bilinear_coord(sampler2D plane, vec2 coord) {
    vec2 size = vec2(textureSize(plane, 0));
    vec2 source_position = coord*size - 0.5;
    vec2 fraction = fract(source_position);
    vec2 scale = max(output_size/size, vec2(1.0));
    // Compress each bilinear transition to one output-pixel footprint. This
    // preserves flat cell interiors while softening only source-cell edges.
    vec2 remapped = clamp((fraction - 0.5)*scale + 0.5, 0.0, 1.0);
    return (floor(source_position) + remapped + 0.5)/size;
}

float sample_plane(sampler2D plane, vec2 coord) {
    if (filtered_intermediate == 1)
        return texture(plane, coord).r;
    if (reconstruction == 2)
        return sample_mitchell(plane, coord);
    if (reconstruction == 1)
        return texture(plane, sharp_bilinear_coord(plane, coord)).r;
    if (reconstruction == 3)
        return sample_ewa_jinc(plane, coord);
    if (reconstruction >= 4)
        return sample_separable(plane, coord);
    return texture(plane, coord).r;
}

float grain_hash(ivec2 pixel, int salt) {
    uint h = uint(pixel.x)*1973u + uint(pixel.y)*9277u +
             uint(grain_seed)*26699u + uint(salt)*104729u;
    h = (h ^ (h >> 16u))*0x45d9f3bu;
    h = (h ^ (h >> 16u))*0x45d9f3bu;
    h = h ^ (h >> 16u);
    return float(h & 0xffffffu)/16777216.0;
}

void main() {
    float y_code = sample_plane(plane_y, uv);
    float cb_code = sample_plane(plane_cb, uv);
    float cr_code = sample_plane(plane_cr, uv);
    // Values are code/127.5 - 1. Recover full-range Y and 8-bit BT.601 chroma
    // centered at code 128, matching Pillow's YCbCr conversion convention.
    float y = (y_code + 1.0)*0.5;
    if (grain_amount > 0.0) {
        ivec2 pixel = ivec2(gl_FragCoord.xy);
        float noise = grain_hash(pixel, 0) + grain_hash(pixel, 1) - 1.0;
        y += grain_amount*texture(grain_mask, uv).r*noise;
    }
    float cb = cb_code*0.5 - 0.5/255.0;
    float cr = cr_code*0.5 - 0.5/255.0;
    vec3 rgb = vec3(y + 1.402*cr,
                    y - 0.344136*cb - 0.714136*cr,
                    y + 1.772*cb);
    color = vec4(clamp(rgb, 0.0, 1.0), 1.0);
}
'''


def _float_texture_filter(mode, moderngl):
    return (moderngl.LINEAR if mode in (
        'bilinear', 'sharp-bilinear', *FILTER_PRECOMPUTE_MODES)
            else moderngl.NEAREST)


def fit_viewport(framebuffer_size, image_aspect):
    """Return a centered letterbox viewport preserving the transmitted aspect."""
    width, height = (max(0, int(value)) for value in framebuffer_size)
    if width == 0 or height == 0 or image_aspect <= 0:
        return 0, 0, 0, 0
    window_aspect = width/height
    if window_aspect > image_aspect:
        view_height = height
        view_width = max(1, round(height*image_aspect))
    else:
        view_width = width
        view_height = max(1, round(width/image_aspect))
    return ((width-view_width)//2, (height-view_height)//2,
            view_width, view_height)


def title_for_status(meter, details=False, display_mode='nearest'):
    """Keep receiver state visible without putting diagnostic widgets over video."""
    status = str(meter.get('status') or 'acquiring').upper()
    parts = ['V7 Receiver']
    parts.append(f'view {DISPLAY_LABELS.get(display_mode, display_mode)}')
    device = meter.get('device')
    if device is not None:
        rate = float(meter.get('capture_rate') or 0)/1000
        channels = meter.get('input_channels') or 1
        mode = meter.get('mode') or ('mono-input' if channels == 1 else 'M/S')
        parts.append(f'{device} · {rate:g} kHz · {channels} ch · {mode}')
    parts.append(status)
    index = meter.get('source_index')
    if index is not None:
        parts.append(f'index {index}')
    lag_ms = meter.get('lag_ms')
    if lag_ms is not None:
        parts.append(f'lag {lag_ms:+.0f} ms')
    if details:
        incoming = meter.get('input_fps') or 0.0
        gain = meter.get('auto_gain') or 1.0
        dropped = meter.get('dropped') or 0
        parts.extend((f'{incoming:.1f} fps', f'gain {gain:.1f}x',
                      f'drops {dropped}'))
    return '  |  '.join(parts)


def _diagnostic_image(size, diagnostics):
    """Build compact diagnostic cards at a low, fixed refresh rate."""
    width, height = size
    image = Image.new('RGBA', (max(1, width), max(1, height)),
                      (9, 16, 24, 224))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype('DejaVuSansMono.ttf', 14)
        heading_font = ImageFont.truetype('DejaVuSans.ttf', 13)
    except OSError:
        try:
            font = ImageFont.load_default(size=14)
            heading_font = ImageFont.load_default(size=13)
        except TypeError:
            font = heading_font = ImageFont.load_default()

    padding = 14
    gap = 12
    heading_height = 20
    status = diagnostics.get('status', ('ACQUIRING',))[0].upper()
    draw.text((padding, 3),
              f'{status}  ·  F full  ·  I info  ·  C setup  ·  Esc exit',
              fill=(132, 153, 173, 255), font=heading_font)
    specs = [
        ('SYNC  /  INDEX', 'sync'),
        ('DECODE  /  FLOW', 'decode'),
        ('INPUT  /  LEVEL', 'input'),
        ('PICTURE  /  SIGNAL', 'signal'),
    ]
    if 'audio' in diagnostics:
        specs.append(('AUDIO FIFO  /  XRUNS', 'audio'))
    if 'decode_cpu' in diagnostics:
        specs.append(('DECODE CPU', 'decode_cpu'))
    if 'resources' in diagnostics:
        specs.append(('GUI RESOURCES', 'resources'))
    columns = (2 if width < 600 else
               4 if width >= 720 and len(specs) > 6 else
               3 if len(specs) > 4 else 2)
    rows = (len(specs)+columns-1)//columns
    card_width = max(1, (width-2*padding-(columns-1)*gap)//columns)
    card_height = max(
        1, (height-2*padding-heading_height-(rows-1)*gap)//rows)
    for index, (heading, key) in enumerate(specs):
        col, row = index % columns, index // columns
        x = padding + col*(card_width+gap)
        y = padding+heading_height + row*(card_height+gap)
        box = (x, y, x+card_width, y+card_height)
        draw.rounded_rectangle(box, radius=5, fill=(19, 30, 41, 232),
                               outline=(50, 70, 89, 230), width=1)
        draw.text((x+10, y+7), heading, fill=(137, 162, 184, 255),
                  font=heading_font)
        line_y = y+26
        available_width = max(1, card_width-20)
        max_lines = max(0, (card_height-46)//17+1)
        lines = [str(line) for line in diagnostics.get(key, ())]
        visible_lines = lines[:max_lines]
        if len(lines) > max_lines and visible_lines:
            visible_lines[-1] = f'{visible_lines[-1]} …'
        for line in visible_lines:
            line = _fit_diagnostic_text(line, font, available_width)
            draw.text((x+10, line_y), line, fill=(227, 237, 245, 255),
                      font=font)
            line_y += 17
    return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))


def _fit_diagnostic_text(text, font, width):
    """Keep diagnostic lines inside their card at compact window sizes."""
    text = str(text)
    if font.getlength(text) <= width:
        return text
    ellipsis = '…'
    while text and font.getlength(text+ellipsis) > width:
        text = text[:-1]
    return text+ellipsis if text else ellipsis


def run(frame_source, status_source, aspect_ratios, fullscreen=False,
        show_diagnostics=True, diagnostics_source=None,
        profile_cpu=False, display_mode=None, image_only=False,
        stop_event=None):
    """Display new frames on a vsynced GL window, sleeping between events.

    GLFW and ModernGL are imported here so headless receive stays independent
    of the graphics stack. The context is owned by this thread; decoded frames
    arrive through the receiver's latest-frame mailbox.
    """
    import glfw
    import moderngl

    if not glfw.init():
        raise RuntimeError('GLFW initialization failed')

    image_only = bool(image_only)
    fullscreen = bool(fullscreen or image_only)

    window = None
    texture = None
    plane_textures = []
    plane_texture_shapes = None
    kernel_textures = {}
    overlay = None
    toolbar = None
    context = None
    program = None
    vertex_array = None
    float_program = None
    float_array = None
    overlay_program = None
    overlay_array = None
    try:
        glfw.default_window_hints()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
        if image_only:
            glfw.window_hint(glfw.DECORATED, glfw.FALSE)
        monitor = glfw.get_primary_monitor() if fullscreen else None
        if fullscreen:
            mode = glfw.get_video_mode(monitor) if monitor else None
            width = mode.size.width if mode else 1280
            height = mode.size.height if mode else 720
        else:
            width, height = 960, 720
        title = 'V7 · Image only' if image_only else 'V7 Receiver'
        window = glfw.create_window(width, height, title, monitor, None)
        if not window:
            raise RuntimeError('GLFW could not create the V7 receiver window')
        glfw.make_context_current(window)
        if image_only:
            glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_HIDDEN)
        glfw.swap_interval(1)

        context = moderngl.create_context(require=330)
        program = context.program(vertex_shader=VERTEX_SHADER,
                                  fragment_shader=FRAGMENT_SHADER)
        program['image'].value = 0
        vertex_array = context.vertex_array(program, [])
        overlay_program = context.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER)
        overlay_program['image'].value = 0
        overlay_array = context.vertex_array(overlay_program, [])

        windowed_bounds = {'position': (80, 80), 'size': (960, 720)}
        is_fullscreen = bool(fullscreen)
        show_details = bool(show_diagnostics) and not image_only
        last_frame_generation = None
        last_float_generation = None
        last_float_mode = None
        last_viewport = None
        last_title = None
        overlay_key = None
        toolbar_key = None
        display_mode = display_mode or _load_display_default()
        if display_mode not in DISPLAY_MODES:
            display_mode = 'nearest'

        def update_texture_filters():
            if texture is not None:
                texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
            filtering = _float_texture_filter(display_mode, moderngl)
            for plane_texture in plane_textures:
                plane_texture.filter = (filtering, filtering)

        def ensure_float_renderer():
            nonlocal float_program, float_array
            if float_program is not None:
                return
            float_program = context.program(
                vertex_shader=VERTEX_SHADER,
                fragment_shader=FLOAT_FRAGMENT_SHADER)
            float_program['plane_y'].value = 0
            float_program['plane_cb'].value = 1
            float_program['plane_cr'].value = 2
            float_program['kernel_lut'].value = 3
            float_array = context.vertex_array(float_program, [])

        def kernel_texture_for(mode):
            texture_for_mode = kernel_textures.get(mode)
            if texture_for_mode is None:
                weights = build_filter_lut(mode)
                texture_for_mode = context.texture(
                    (weights.size, 1), 1, weights.tobytes(), dtype='f4')
                texture_for_mode.filter = (moderngl.LINEAR, moderngl.LINEAR)
                texture_for_mode.repeat_x = False
                texture_for_mode.repeat_y = False
                kernel_textures[mode] = texture_for_mode
            return texture_for_mode

        open_dropdown = None
        save_notice = ''
        save_notice_until = 0.0
        toolbar_visible = not is_fullscreen
        last_ui_activity = time.monotonic()
        toolbar_hits = toolbar_layout(width)
        last_overlay_update = 0.0
        last_window_size = (width, height)
        profile_wall = time.monotonic()
        profile_process = time.process_time()
        profile_thread = time.thread_time()

        def toggle_fullscreen():
            nonlocal is_fullscreen, toolbar_visible, last_ui_activity
            nonlocal toolbar_key, dirty
            primary = glfw.get_primary_monitor()
            if primary is None:
                return
            if is_fullscreen:
                x, y = windowed_bounds['position']
                w, h = windowed_bounds['size']
                glfw.set_window_monitor(window, None, x, y, w, h,
                                        glfw.DONT_CARE)
                is_fullscreen = False
            else:
                windowed_bounds['position'] = glfw.get_window_pos(window)
                windowed_bounds['size'] = glfw.get_window_size(window)
                mode = glfw.get_video_mode(primary)
                glfw.set_window_monitor(window, primary, 0, 0,
                                        mode.size.width, mode.size.height,
                                        mode.refresh_rate)
                is_fullscreen = True
            toolbar_visible = not is_fullscreen
            last_ui_activity = time.monotonic()
            toolbar_key = None
            dirty = True

        def on_cursor_position(_window, _x, y):
            nonlocal toolbar_visible, last_ui_activity, toolbar_key, dirty
            if not is_fullscreen or image_only:
                return
            now = time.monotonic()
            if toolbar_visible:
                last_ui_activity = now
            elif y <= FULLSCREEN_TOOLBAR_EDGE:
                toolbar_visible = True
                last_ui_activity = now
                toolbar_key = None
                dirty = True

        def on_key(_window, key, _scancode, action, _mods):
            nonlocal show_details, last_title, display_mode, open_dropdown
            nonlocal toolbar_key, dirty, toolbar_visible, last_ui_activity
            if action != glfw.PRESS:
                return
            if image_only:
                if key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
                    glfw.set_window_should_close(window, True)
                return
            was_hidden = not toolbar_visible
            toolbar_visible = True
            last_ui_activity = time.monotonic()
            if was_hidden:
                toolbar_key = None
                dirty = True
            if key == glfw.KEY_ESCAPE and open_dropdown is not None:
                open_dropdown = None
                last_title = None
            elif key == glfw.KEY_F:
                toggle_fullscreen()
            elif key == glfw.KEY_I:
                show_details = not show_details
                last_title = None
                open_dropdown = None
            elif key == glfw.KEY_U:
                step = -1 if _mods & glfw.MOD_SHIFT else 1
                display_mode = DISPLAY_MODES[
                    (DISPLAY_MODES.index(display_mode)+step) % len(DISPLAY_MODES)]
                update_texture_filters()
                open_dropdown = None
                last_title = None
            elif (glfw.KEY_1 <= key <= glfw.KEY_9 and
                  key-glfw.KEY_1 < len(DISPLAY_MODES)):
                display_mode = DISPLAY_MODES[key-glfw.KEY_1]
                update_texture_filters()
                open_dropdown = None
                last_title = None
            elif key == glfw.KEY_ESCAPE:
                if is_fullscreen:
                    toggle_fullscreen()
                else:
                    glfw.set_window_should_close(window, True)
            toolbar_key = None
            dirty = True

        glfw.set_key_callback(window, on_key)
        glfw.set_cursor_pos_callback(window, on_cursor_position)
        dirty = True

        def on_mouse_button(_window, button, action, _mods):
            nonlocal show_details, display_mode, open_dropdown
            nonlocal toolbar_key, last_title, dirty
            nonlocal save_notice, save_notice_until
            nonlocal toolbar_visible, last_ui_activity
            if button != glfw.MOUSE_BUTTON_LEFT or action != glfw.PRESS:
                return
            if not image_only:
                was_hidden = not toolbar_visible
                toolbar_visible = True
                last_ui_activity = time.monotonic()
                if was_hidden:
                    toolbar_key = None
                    dirty = True
                    return
            x, y = glfw.get_cursor_pos(window)
            key = next((name for name, rect in toolbar_hits.items()
                        if _contains(rect, x, y)), None)
            if key == 'upscale_button':
                open_dropdown = None if open_dropdown == 'upscale' else 'upscale'
            elif key == 'panel_button':
                open_dropdown = None if open_dropdown == 'panel' else 'panel'
            elif key == 'fullscreen_button':
                open_dropdown = None
                toggle_fullscreen()
            elif key == 'save_default_button':
                try:
                    _save_display_default(display_mode)
                    save_notice = 'Saved'
                except OSError as exc:
                    print(f'Could not save V7 display preference: {exc}',
                          file=sys.stderr, flush=True)
                    save_notice = 'Save failed'
                save_notice_until = time.monotonic()+1.5
            elif key and key.startswith('mode:'):
                display_mode = key.split(':', 1)[1]
                update_texture_filters()
                open_dropdown = None
                last_title = None
            elif key and key.startswith('panel:'):
                show_details = key.endswith(':1')
                open_dropdown = None
                last_title = None
            elif open_dropdown is not None:
                open_dropdown = None
            toolbar_key = None
            dirty = True

        glfw.set_mouse_button_callback(window, on_mouse_button)

        def on_refresh(_window):
            nonlocal dirty
            dirty = True

        glfw.set_window_refresh_callback(window, on_refresh)
        while (not glfw.window_should_close(window) and
               not (stop_event is not None and stop_event.is_set())):
            # Wait for input/resize events; this caps polling at 60 Hz without
            # consuming a CPU core while the decoder has no new picture.
            glfw.wait_events_timeout(1/60)

            meter = status_source()
            title = title_for_status(meter, show_details, display_mode)
            if title != last_title:
                glfw.set_window_title(window, title)
                last_title = title

            frame = frame_source()
            if frame is not None and frame.generation != last_frame_generation:
                image = values_image(frame.values, frame.shapes)
                pixels = np.ascontiguousarray(np.asarray(image.convert('RGB'),
                                                         dtype=np.uint8))
                size = (pixels.shape[1], pixels.shape[0])
                if texture is None or texture.size != size:
                    if texture is not None:
                        texture.release()
                    texture = context.texture(size, 3, data=pixels.tobytes(),
                                              dtype='f1')
                    texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
                    texture.repeat_x = False
                    texture.repeat_y = False
                else:
                    texture.write(pixels.tobytes())

                last_frame_generation = frame.generation
                dirty = True

            if (frame is not None and display_mode != 'nearest' and
                    (frame.generation != last_float_generation or
                     display_mode != last_float_mode)):
                planes = float_planes(frame.values, frame.shapes)
                filtered_intermediate = (
                    display_mode in FILTER_PRECOMPUTE_MODES)
                if filtered_intermediate:
                    planes = resample_filter_planes(planes, display_mode)
                plane_sizes = tuple((plane.shape[1], plane.shape[0])
                                    for plane in planes)
                if (plane_texture_shapes != plane_sizes or
                        len(plane_textures) != len(planes)):
                    for plane_texture in plane_textures:
                        plane_texture.release()
                    plane_textures = [
                        context.texture(plane_size, 1, plane.tobytes(),
                                        dtype='f4')
                        for plane_size, plane in zip(plane_sizes, planes)]
                    for plane_texture in plane_textures:
                        plane_texture.repeat_x = False
                        plane_texture.repeat_y = False
                    plane_texture_shapes = plane_sizes
                else:
                    for plane_texture, plane in zip(plane_textures, planes):
                        plane_texture.write(plane.tobytes())
                update_texture_filters()
                last_float_generation = frame.generation
                last_float_mode = display_mode
                dirty = True

            now = time.monotonic()
            if is_fullscreen and not image_only:
                cursor_y = glfw.get_cursor_pos(window)[1]
                toolbar_was_visible = toolbar_visible
                if open_dropdown is not None or cursor_y <= FULLSCREEN_TOOLBAR_EDGE:
                    toolbar_visible = True
                elif (toolbar_visible and
                      now-last_ui_activity >= FULLSCREEN_TOOLBAR_HIDE_SECONDS):
                    toolbar_visible = False
                if toolbar_visible != toolbar_was_visible:
                    toolbar_key = None
                    dirty = True
            details_visible = (show_details and
                               (not is_fullscreen or toolbar_visible))
            window_size = glfw.get_window_size(window)
            if details_visible and diagnostics_source is not None and (
                    now-last_overlay_update >= .2 or
                    window_size != last_window_size):
                last_overlay_update = now
                last_window_size = window_size
                diagnostics = diagnostics_source()
                key = tuple((name, tuple(lines))
                            for name, lines in diagnostics.items())
                if key != overlay_key:
                    logical_w = max(320, int(window_size[0]))
                    logical_h = max(160, round(window_size[1]*.36))
                    overlay_pixels = _diagnostic_image(
                        (logical_w, logical_h), diagnostics)
                    if overlay is not None:
                        overlay.release()
                    overlay = context.texture(
                        (logical_w, logical_h), 4,
                        data=overlay_pixels.tobytes(), dtype='f1')
                    overlay.filter = (moderngl.LINEAR, moderngl.LINEAR)
                    overlay.repeat_x = False
                    overlay.repeat_y = False
                    overlay_key = key
                    dirty = True
            elif not details_visible and overlay is not None:
                overlay.release()
                overlay = None
                overlay_key = None
                dirty = True

            fb_size = glfw.get_framebuffer_size(window)
            window_size = glfw.get_window_size(window)
            if save_notice and time.monotonic() >= save_notice_until:
                save_notice = ''
                toolbar_key = None
            toolbar_fb_height = 0
            show_toolbar = (not image_only and
                            (not is_fullscreen or toolbar_visible or
                             open_dropdown is not None))
            if show_toolbar:
                toolbar_state = (window_size, display_mode,
                                 show_details, open_dropdown, save_notice,
                                 toolbar_visible, is_fullscreen)
                dropdown_height = (
                    len(DISPLAY_MODES)*29+4 if open_dropdown == 'upscale'
                    else 62 if open_dropdown == 'panel' else 0)
                toolbar_height = 48+dropdown_height
                toolbar_size = (max(320, int(window_size[0])), toolbar_height)
                if (toolbar is None or toolbar.size != toolbar_size or
                        toolbar_key != toolbar_state):
                    toolbar_pixels = _toolbar_image(
                        window_size, display_mode, show_details, open_dropdown,
                        save_notice)
                    if toolbar is not None:
                        toolbar.release()
                    toolbar = context.texture(
                        toolbar_size, 4, data=toolbar_pixels.tobytes(), dtype='f1')
                    toolbar.filter = (moderngl.LINEAR, moderngl.LINEAR)
                    toolbar.repeat_x = False
                    toolbar.repeat_y = False
                    toolbar_key = toolbar_state
                    toolbar_hits = toolbar_layout(window_size[0], open_dropdown)
                    dirty = True
                toolbar_fb_height = (
                    round(fb_size[1]*toolbar_size[1]/window_size[1])
                    if window_size[1] else 0)
            else:
                toolbar_hits = {}
            ratio = (aspect_ratios[frame.aspect & 7]
                     if frame is not None else 4/3)
            panel_height = (round(fb_size[1]*.36)
                            if details_visible and overlay is not None else 0)
            picture_area = (fb_size[0], max(
                0, fb_size[1]-panel_height-toolbar_fb_height))
            picture_viewport = fit_viewport(picture_area, ratio)
            viewport = ((picture_viewport[0],
                         panel_height+picture_viewport[1],
                         picture_viewport[2], picture_viewport[3]))
            if viewport != last_viewport:
                last_viewport = viewport
                dirty = True
            if dirty and viewport[2] and viewport[3]:
                context.viewport = (0, 0, *fb_size)
                context.clear(.025, .032, .045, 1.0)
                if texture is not None:
                    context.viewport = viewport
                    if display_mode == 'nearest':
                        texture.use(location=0)
                        vertex_array.render(mode=moderngl.TRIANGLES, vertices=3)
                    else:
                        ensure_float_renderer()
                        for unit, plane_texture in enumerate(plane_textures):
                            plane_texture.use(location=unit)
                        kernel_texture_for(display_mode).use(location=3)
                        float_program['reconstruction'].value = FLOAT_MODE_IDS[
                            display_mode]
                        float_program['output_size'].value = (
                            float(viewport[2]), float(viewport[3]))
                        float_program['filtered_intermediate'].value = int(
                            display_mode in FILTER_PRECOMPUTE_MODES)
                        float_array.render(mode=moderngl.TRIANGLES, vertices=3)
                if details_visible and overlay is not None and panel_height:
                    context.enable(moderngl.BLEND)
                    context.blend_func = (moderngl.SRC_ALPHA,
                                          moderngl.ONE_MINUS_SRC_ALPHA)
                    context.viewport = (0, 0, fb_size[0], panel_height)
                    overlay.use(location=0)
                    overlay_array.render(mode=moderngl.TRIANGLES, vertices=3)
                    context.disable(moderngl.BLEND)
                if show_toolbar and toolbar is not None and toolbar_fb_height:
                    context.enable(moderngl.BLEND)
                    context.blend_func = (moderngl.SRC_ALPHA,
                                          moderngl.ONE_MINUS_SRC_ALPHA)
                    context.viewport = (0, fb_size[1]-toolbar_fb_height,
                                        fb_size[0], toolbar_fb_height)
                    toolbar.use(location=0)
                    overlay_array.render(mode=moderngl.TRIANGLES, vertices=3)
                    context.disable(moderngl.BLEND)
                glfw.swap_buffers(window)
                dirty = False

            if profile_cpu:
                elapsed = time.monotonic()-profile_wall
                if elapsed >= 5.0:
                    process_cpu = time.process_time()-profile_process
                    thread_cpu = time.thread_time()-profile_thread
                    print(
                        'V7 viewer CPU over '
                        f'{elapsed:.1f}s: process {100*process_cpu/elapsed:.1f}% '
                        f'(all threads), UI {100*thread_cpu/elapsed:.1f}% '
                        '(viewer thread; event waits excluded)',
                        file=sys.stderr, flush=True)
                    profile_wall = time.monotonic()
                    profile_process = time.process_time()
                    profile_thread = time.thread_time()
    finally:
        if overlay_array is not None:
            overlay_array.release()
        if overlay_program is not None:
            overlay_program.release()
        if vertex_array is not None:
            vertex_array.release()
        if float_array is not None:
            float_array.release()
        if float_program is not None:
            float_program.release()
        if program is not None:
            program.release()
        if overlay is not None:
            overlay.release()
        if toolbar is not None:
            toolbar.release()
        if texture is not None:
            texture.release()
        for plane_texture in plane_textures:
            plane_texture.release()
        for kernel_texture in kernel_textures.values():
            kernel_texture.release()
        if context is not None:
            context.release()
        if window is not None:
            glfw.destroy_window(window)
        glfw.terminate()
