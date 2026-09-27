"""Deterministic sender-side area resize experiments for V7.

The detail kernel is DPID-inspired, not a claim of compatibility with a
particular DPID implementation. It starts with exact pixel-area weights, then
reweights contributors by their RGB distance from the area mean. All three
channels share those scalar weights. The hot contributor loops are Numba
compiled; footprint geometry is cached by source and destination size.
"""
from functools import lru_cache

import numpy as np
from numba import njit
from PIL import Image


RESIZE_MODES = ('linear-box', 'gamma-detail', 'linear-detail')
DEFAULT_DETAIL_STRENGTH = 0.25
PREPARED_SIZE = (80, 96)  # width, height; V7's prepared RGB canvas


@lru_cache(maxsize=32)
def _axis_footprints(source_size, target_size):
    """Return exact 1D pixel-overlap geometry for one resampled axis."""
    source_size, target_size = int(source_size), int(target_size)
    if source_size <= 0 or target_size <= 0:
        raise ValueError('resize dimensions must be positive')
    max_count = int(np.ceil(source_size/target_size))+1
    indexes = np.zeros((target_size, max_count), np.int32)
    weights = np.zeros((target_size, max_count), np.float64)
    counts = np.zeros(target_size, np.int32)
    scale = source_size/target_size
    for out_index in range(target_size):
        start = out_index*scale
        end = (out_index+1)*scale
        first = int(np.floor(start))
        last = min(source_size, int(np.ceil(end)))
        count = 0
        for source_index in range(first, last):
            overlap = min(end, source_index+1.0)-max(start, float(source_index))
            if overlap > 0:
                indexes[out_index, count] = source_index
                weights[out_index, count] = overlap
                count += 1
        if count == 0:
            raise RuntimeError('empty area-resize footprint')
        counts[out_index] = count
    indexes.setflags(write=False)
    weights.setflags(write=False)
    counts.setflags(write=False)
    return indexes, weights, counts


@njit(nogil=True)
def _resize_area_kernel(source, yi, yw, yc, xi, xw, xc, strength):
    height, width, _ = source.shape
    out_height, out_width = len(yc), len(xc)
    output = np.empty((out_height, out_width, 3), np.float64)
    for oy in range(out_height):
        for ox in range(out_width):
            area = 0.0
            mean0 = 0.0
            mean1 = 0.0
            mean2 = 0.0
            for yk in range(yc[oy]):
                sy = yi[oy, yk]
                wy = yw[oy, yk]
                for xk in range(xc[ox]):
                    sx = xi[ox, xk]
                    weight = wy*xw[ox, xk]
                    area += weight
                    mean0 += weight*source[sy, sx, 0]
                    mean1 += weight*source[sy, sx, 1]
                    mean2 += weight*source[sy, sx, 2]
            mean0 /= area
            mean1 /= area
            mean2 /= area
            if strength == 0.0:
                output[oy, ox, 0] = mean0
                output[oy, ox, 1] = mean1
                output[oy, ox, 2] = mean2
                continue

            weighted_distance2 = 0.0
            for yk in range(yc[oy]):
                sy = yi[oy, yk]
                wy = yw[oy, yk]
                for xk in range(xc[ox]):
                    sx = xi[ox, xk]
                    weight = wy*xw[ox, xk]
                    d0 = source[sy, sx, 0]-mean0
                    d1 = source[sy, sx, 1]-mean1
                    d2 = source[sy, sx, 2]-mean2
                    distance2 = (d0*d0+d1*d1+d2*d2)/3.0
                    weighted_distance2 += weight*distance2
            scale = max(np.sqrt(weighted_distance2/area), 1.0/255.0)
            weighted_total = 0.0
            accum0 = 0.0
            accum1 = 0.0
            accum2 = 0.0
            for yk in range(yc[oy]):
                sy = yi[oy, yk]
                wy = yw[oy, yk]
                for xk in range(xc[ox]):
                    sx = xi[ox, xk]
                    base_weight = wy*xw[ox, xk]
                    d0 = source[sy, sx, 0]-mean0
                    d1 = source[sy, sx, 1]-mean1
                    d2 = source[sy, sx, 2]-mean2
                    distance = np.sqrt((d0*d0+d1*d1+d2*d2)/3.0)
                    detail = min(distance/scale, 2.0)
                    weight = base_weight*(1.0+strength*detail)
                    weighted_total += weight
                    accum0 += weight*source[sy, sx, 0]
                    accum1 += weight*source[sy, sx, 1]
                    accum2 += weight*source[sy, sx, 2]
            output[oy, ox, 0] = accum0/weighted_total
            output[oy, ox, 1] = accum1/weighted_total
            output[oy, ox, 2] = accum2/weighted_total
    return output


@njit(nogil=True)
def _input_domain(pixels, linear):
    height, width, _ = pixels.shape
    output = np.empty((height, width, 3), np.float64)
    for y in range(height):
        for x in range(width):
            for channel in range(3):
                if linear:
                    output[y, x, channel] = _SRGB8_TO_LINEAR[pixels[y, x, channel]]
                else:
                    output[y, x, channel] = pixels[y, x, channel]/255.0
    return output


@njit(nogil=True)
def _quantize_output(resized, linear):
    height, width, _ = resized.shape
    output = np.empty((height, width, 3), np.uint8)
    for y in range(height):
        for x in range(width):
            for channel in range(3):
                value = resized[y, x, channel]
                if linear:
                    value = min(max(value, 0.0), 1.0)
                    low, high = 0, len(_LINEAR_ROUND_THRESHOLDS)
                    while low < high:
                        middle = (low+high)//2
                        if _LINEAR_ROUND_THRESHOLDS[middle] < value:
                            low = middle+1
                        else:
                            high = middle
                    code = low
                    # Match round-to-nearest, ties-to-even at transfer-space
                    # half-code boundaries without a second nonlinear transfer.
                    if (code < 255 and value == _LINEAR_ROUND_THRESHOLDS[code]
                            and code % 2):
                        code += 1
                    output[y, x, channel] = np.uint8(code)
                else:
                    value = min(max(value, 0.0), 1.0)
                    output[y, x, channel] = np.uint8(np.rint(value*255.0))
    return output


def _srgb_to_linear(rgb):
    rgb = np.asarray(rgb, dtype=np.float64)
    return np.where(rgb <= 0.04045, rgb/12.92,
                    ((rgb+0.055)/1.055)**2.4)


def _linear_to_srgb(rgb):
    rgb = np.asarray(rgb, dtype=np.float64)
    return np.where(rgb <= 0.0031308, rgb*12.92,
                    1.055*np.maximum(rgb, 0.0)**(1.0/2.4)-0.055)


_SRGB8 = np.arange(256, dtype=np.float64)/255.0
_SRGB8_TO_LINEAR = np.where(
    _SRGB8 <= 0.04045, _SRGB8/12.92,
    ((_SRGB8+0.055)/1.055)**2.4)
_HALF_SRGB_CODES = (np.arange(255, dtype=np.float64)+0.5)/255.0
_LINEAR_ROUND_THRESHOLDS = np.where(
    _HALF_SRGB_CODES <= 0.04045, _HALF_SRGB_CODES/12.92,
    ((_HALF_SRGB_CODES+0.055)/1.055)**2.4)
_SRGB8_TO_LINEAR.setflags(write=False)
_LINEAR_ROUND_THRESHOLDS.setflags(write=False)


def resize_rgb(rgb, mode, detail_strength=DEFAULT_DETAIL_STRENGTH,
               target_size=PREPARED_SIZE):
    """Resize an RGB uint8 array to ``(width, height)`` using one V7 ablation."""
    if mode not in RESIZE_MODES:
        raise ValueError(f'perceptual resize mode must be one of {RESIZE_MODES}')
    detail_strength = float(detail_strength)
    if not np.isfinite(detail_strength) or not 0.0 <= detail_strength <= 1.0:
        raise ValueError('perceptual detail strength must be finite and in [0, 1]')
    target_width, target_height = map(int, target_size)
    if target_width <= 0 or target_height <= 0:
        raise ValueError('resize dimensions must be positive')
    pixels = np.asarray(rgb)
    if pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.size == 0:
        raise ValueError('resize input must be a non-empty HxWx3 RGB array')
    if pixels.dtype != np.uint8:
        raise ValueError('resize input must use uint8 RGB pixels')

    linear = mode.startswith('linear-')
    source = _input_domain(pixels, linear)
    strength = 0.0 if mode == 'linear-box' else detail_strength
    yi, yw, yc = _axis_footprints(pixels.shape[0], target_height)
    xi, xw, xc = _axis_footprints(pixels.shape[1], target_width)
    resized = _resize_area_kernel(source, yi, yw, yc, xi, xw, xc, strength)
    return _quantize_output(resized, linear)


def resize_image(image, mode, detail_strength=DEFAULT_DETAIL_STRENGTH,
                 target_size=PREPARED_SIZE):
    """Pillow adapter used by the live sender before brightness and gamma."""
    rgb = np.asarray(image.convert('RGB'), dtype=np.uint8)
    return Image.fromarray(resize_rgb(rgb, mode, detail_strength, target_size),
                           mode='RGB')


def warmup_resize(source_size, mode, detail_strength=DEFAULT_DETAIL_STRENGTH,
                  target_size=PREPARED_SIZE):
    """Compile the resize kernel on a small image with the requested geometry."""
    width, height = map(int, source_size)
    sample = np.zeros((height, width, 3), np.uint8)
    resize_rgb(sample, mode, detail_strength, target_size)
