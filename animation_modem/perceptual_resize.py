"""Deterministic sender-side area resize experiments for V7.

The detail kernel is DPID-inspired, not a claim of compatibility with a
particular DPID implementation. It starts with exact pixel-area weights, then
reweights contributors by their RGB distance from the area mean. All three
channels share those scalar weights. The hot contributor loops are Numba
compiled; footprint geometry is cached by source and destination size.

Performance without changing a byte. The kernels are arranged for speed, but
every output byte equals the direct per-pixel implementation this module
started with; modem_tests/test_v7_perceptual_resize.py pins that against a
copy of it.

- uint8 codes become floats through a 256-entry table: code/255 for gamma,
  the sRGB transfer for linear light. These are the same float64 values the
  per-pixel division and transfer produced, without a divide per sample.
- The box path averages each footprint in one pass.
- The detail path gathers each output row's contributors once into
  contiguous rows, then runs its three passes (area mean, distance spread,
  reweighted mean) as unit-stride loops over the row, which LLVM vectorizes.
  Footprints shorter than the widest one are padded with zero weights. Every
  padded term adds an exact +0.0 (all values are non-negative), so each
  pixel's arithmetic and accumulation order are unchanged.
- Linear-light quantization finds its code with a bin lookup and a step or
  two of correction instead of a binary search, whose unpredictable branches
  made it the costliest stage.
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


@lru_cache(maxsize=32)
def _padded_footprints(source_size, target_size):
    """The x-axis footprints transposed to (contributor, output) rows and
    padded with zero weights to the widest footprint (see module docstring).
    Padded slots repeat a real index so every gather stays in bounds."""
    indexes, weights, counts = _axis_footprints(source_size, target_size)
    width = indexes.shape[1]
    padded_indexes = np.empty((width, len(counts)), np.int64)
    padded_weights = np.zeros((width, len(counts)), np.float64)
    for out_index, count in enumerate(counts):
        padded_indexes[:, out_index] = indexes[out_index, 0]
        padded_indexes[:count, out_index] = indexes[out_index, :count]
        padded_weights[:count, out_index] = weights[out_index, :count]
    padded_indexes.setflags(write=False)
    padded_weights.setflags(write=False)
    return padded_indexes, padded_weights


@njit(nogil=True, cache=True)
def _to_float(pixels, table):
    """uint8 HxWx3 codes to float64 through a 256-entry table."""
    height, width, _ = pixels.shape
    output = np.empty((height, width, 3), np.float64)
    for y in range(height):
        for x in range(width):
            for channel in range(3):
                output[y, x, channel] = table[pixels[y, x, channel]]
    return output


@njit(nogil=True, cache=True)
def _to_planes(pixels, table):
    """uint8 HxWx3 codes to three float64 planes through a 256-entry table."""
    height, width, _ = pixels.shape
    planes = np.empty((3, height, width), np.float64)
    for y in range(height):
        for x in range(width):
            for channel in range(3):
                planes[channel, y, x] = table[pixels[y, x, channel]]
    return planes


@njit(nogil=True, cache=True, error_model='numpy')
def _area_mean_kernel(source, yi, yw, yc, xi, xw, xc):
    """Exact pixel-area mean of each output footprint."""
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
            output[oy, ox, 0] = mean0/area
            output[oy, ox, 1] = mean1/area
            output[oy, ox, 2] = mean2/area
    return output


@njit(nogil=True, cache=True, error_model='numpy')
def _detail_kernel(planes, yi, yw, yc, xidx, xwt, strength):
    """Detail-preserving reweighting, one output row at a time.

    Per pixel: the area mean; the area-weighted RMS RGB distance of the
    contributors from it (floored at one code); then each contributor's
    weight is multiplied by 1 + strength*min(distance/spread, 2).
    """
    out_height = len(yc)
    count, out_width = xidx.shape
    rows = yi.shape[1]
    output = np.empty((out_height, out_width, 3), np.float64)
    gathered = np.empty((rows, count, 3, out_width))
    distance2 = np.empty((rows, count, out_width))
    area = np.empty(out_width)
    mean0 = np.empty(out_width)
    mean1 = np.empty(out_width)
    mean2 = np.empty(out_width)
    spread = np.empty(out_width)
    total = np.empty(out_width)
    accum0 = np.empty(out_width)
    accum1 = np.empty(out_width)
    accum2 = np.empty(out_width)
    for oy in range(out_height):
        used = yc[oy]
        for yk in range(used):
            sy = yi[oy, yk]
            for channel in range(3):
                row = planes[channel, sy]
                for j in range(count):
                    index = xidx[j]
                    destination = gathered[yk, j, channel]
                    for ox in range(out_width):
                        destination[ox] = row[index[ox]]

        area[:] = 0.0
        mean0[:] = 0.0
        mean1[:] = 0.0
        mean2[:] = 0.0
        for yk in range(used):
            wy = yw[oy, yk]
            for j in range(count):
                wx = xwt[j]
                v0 = gathered[yk, j, 0]
                v1 = gathered[yk, j, 1]
                v2 = gathered[yk, j, 2]
                for ox in range(out_width):
                    weight = wy*wx[ox]
                    area[ox] += weight
                    mean0[ox] += weight*v0[ox]
                    mean1[ox] += weight*v1[ox]
                    mean2[ox] += weight*v2[ox]
        for ox in range(out_width):
            mean0[ox] /= area[ox]
            mean1[ox] /= area[ox]
            mean2[ox] /= area[ox]

        spread[:] = 0.0
        for yk in range(used):
            wy = yw[oy, yk]
            for j in range(count):
                wx = xwt[j]
                v0 = gathered[yk, j, 0]
                v1 = gathered[yk, j, 1]
                v2 = gathered[yk, j, 2]
                squared = distance2[yk, j]
                for ox in range(out_width):
                    d0 = v0[ox]-mean0[ox]
                    d1 = v1[ox]-mean1[ox]
                    d2 = v2[ox]-mean2[ox]
                    value = (d0*d0+d1*d1+d2*d2)/3.0
                    squared[ox] = value
                    spread[ox] += wy*wx[ox]*value
        for ox in range(out_width):
            spread[ox] = max(np.sqrt(spread[ox]/area[ox]), 1.0/255.0)

        total[:] = 0.0
        accum0[:] = 0.0
        accum1[:] = 0.0
        accum2[:] = 0.0
        for yk in range(used):
            wy = yw[oy, yk]
            for j in range(count):
                wx = xwt[j]
                v0 = gathered[yk, j, 0]
                v1 = gathered[yk, j, 1]
                v2 = gathered[yk, j, 2]
                squared = distance2[yk, j]
                for ox in range(out_width):
                    detail = min(np.sqrt(squared[ox])/spread[ox], 2.0)
                    weight = wy*wx[ox]*(1.0+strength*detail)
                    total[ox] += weight
                    accum0[ox] += weight*v0[ox]
                    accum1[ox] += weight*v1[ox]
                    accum2[ox] += weight*v2[ox]
        for ox in range(out_width):
            output[oy, ox, 0] = accum0[ox]/total[ox]
            output[oy, ox, 1] = accum1[ox]/total[ox]
            output[oy, ox, 2] = accum2[ox]/total[ox]
    return output


@njit(nogil=True, cache=True)
def _quantize_kernel(resized, linear, thresholds, bin_base, bins):
    height, width, _ = resized.shape
    output = np.empty((height, width, 3), np.uint8)
    last = len(thresholds)
    for y in range(height):
        for x in range(width):
            for channel in range(3):
                value = min(max(resized[y, x, channel], 0.0), 1.0)
                if linear:
                    # The code is the count of half-code thresholds below the
                    # value. The bin gives it to within a step; the loops
                    # then enforce thresholds[code-1] < value <= thresholds[code]
                    # exactly, as a binary search would.
                    code = bin_base[int(value*bins)]
                    while code > 0 and thresholds[code-1] >= value:
                        code -= 1
                    while code < last and thresholds[code] < value:
                        code += 1
                    # Match round-to-nearest, ties-to-even at transfer-space
                    # half-code boundaries without a second nonlinear transfer.
                    if (code < 255 and value == thresholds[code]
                            and code % 2):
                        code += 1
                    output[y, x, channel] = np.uint8(code)
                else:
                    output[y, x, channel] = np.uint8(np.rint(value*255.0))
    return output


def _quantize_output(resized, linear):
    """Resized float RGB back to uint8 codes in the domain it was averaged in."""
    return _quantize_kernel(resized, linear, _LINEAR_ROUND_THRESHOLDS,
                            _LINEAR_BIN_BASE, _LINEAR_BINS)


def _srgb_to_linear(rgb):
    rgb = np.asarray(rgb, dtype=np.float64)
    return np.where(rgb <= 0.04045, rgb/12.92,
                    ((rgb+0.055)/1.055)**2.4)


def _linear_to_srgb(rgb):
    rgb = np.asarray(rgb, dtype=np.float64)
    return np.where(rgb <= 0.0031308, rgb*12.92,
                    1.055*np.maximum(rgb, 0.0)**(1.0/2.4)-0.055)


_SRGB8 = np.arange(256, dtype=np.float64)/255.0
_SRGB8_TO_GAMMA = _SRGB8.copy()
_SRGB8_TO_LINEAR = np.where(
    _SRGB8 <= 0.04045, _SRGB8/12.92,
    ((_SRGB8+0.055)/1.055)**2.4)
_HALF_SRGB_CODES = (np.arange(255, dtype=np.float64)+0.5)/255.0
_LINEAR_ROUND_THRESHOLDS = np.where(
    _HALF_SRGB_CODES <= 0.04045, _HALF_SRGB_CODES/12.92,
    ((_HALF_SRGB_CODES+0.055)/1.055)**2.4)
# Quantizer bins over [0, 1] of linear light: the thresholds below each bin's
# lower edge. At 2**14 bins no bin holds more than one threshold.
_LINEAR_BINS = 1 << 14
_LINEAR_BIN_BASE = np.searchsorted(
    _LINEAR_ROUND_THRESHOLDS,
    np.arange(_LINEAR_BINS+1, dtype=np.float64)/_LINEAR_BINS,
    side='left').astype(np.int64)
for _table in (_SRGB8_TO_GAMMA, _SRGB8_TO_LINEAR, _LINEAR_ROUND_THRESHOLDS,
               _LINEAR_BIN_BASE):
    _table.setflags(write=False)
del _table


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
    table = _SRGB8_TO_LINEAR if linear else _SRGB8_TO_GAMMA
    strength = 0.0 if mode == 'linear-box' else detail_strength
    yi, yw, yc = _axis_footprints(pixels.shape[0], target_height)
    if strength == 0.0:
        xi, xw, xc = _axis_footprints(pixels.shape[1], target_width)
        resized = _area_mean_kernel(_to_float(pixels, table),
                                    yi, yw, yc, xi, xw, xc)
    else:
        xidx, xwt = _padded_footprints(pixels.shape[1], target_width)
        resized = _detail_kernel(_to_planes(pixels, table),
                                 yi, yw, yc, xidx, xwt, strength)
    return _quantize_output(resized, linear)


def resize_image(image, mode, detail_strength=DEFAULT_DETAIL_STRENGTH,
                 target_size=PREPARED_SIZE):
    """Pillow adapter used by the live sender before brightness and gamma."""
    rgb = np.asarray(image.convert('RGB'), dtype=np.uint8)
    return Image.fromarray(resize_rgb(rgb, mode, detail_strength, target_size),
                           mode='RGB')


def warmup_resize(source_size, mode, detail_strength=DEFAULT_DETAIL_STRENGTH,
                  target_size=PREPARED_SIZE):
    """Compile the resize kernels for the requested mode and strength.

    Numba compiles read-only arrays separately from writable ones, and live
    frames are read-only (np.frombuffer captures, Pillow arrays), so both are
    warmed; otherwise the first live frame would stall on compilation.
    """
    width, height = map(int, source_size)
    sample = np.zeros((height, width, 3), np.uint8)
    resize_rgb(sample, mode, detail_strength, target_size)
    sample.setflags(write=False)
    resize_rgb(sample, mode, detail_strength, target_size)
