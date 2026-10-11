"""Shared production configuration and source conversion for adaptive V7."""
from __future__ import annotations

import numpy as np
from numba import njit

# This is the geometry stored in chroma50.npz (rows, columns), not a preview
# size. Sender, receiver, and the end-to-end tests import this same config.
LUMA_SHAPE = (120, 100)
CHROMA_SHAPE = (60, 50)
SOURCE_SHAPES = np.ascontiguousarray(
    (LUMA_SHAPE, CHROMA_SHAPE, CHROMA_SHAPE), dtype=np.int64)
SERPENTINE_STEP_FACTOR = 0.25
ADAPTIVE_HOST_LEVEL = -1
ADAPTIVE_GUEST_FACTORS = (1, 2, 4, 8, 16)
EXACT_INVERSE = True
# Short FIR candidate shared by adaptive wire IDs. Integration is not a claim
# of completed mono/stereo live fidelity acceptance; see the integration note.
BODY_FILTER_TAPS = 17
BODY_LEVEL_BELOW_EOF_DB = 0.8


def sample_sender_settings(mode):
    """Exact public source/codec preset used by the short-FIR sample feed.

    Device, source and OS sample rate are supplied by the caller. This preset
    deliberately selects Reference rather than silently discarding a kernel.
    """
    if mode not in ('mono', 'stereo'):
        raise ValueError('sample mode must be mono or stereo')
    return {
        'profile': f'adaptive-{mode}-nested',
        'dct_encode': True, 'dct_kernel': 'reference', 'luma_adjust': False,
        'capture_width': '768', 'brightness': '1', 'gamma': '1',
    }


@njit(cache=True)
def _area_ycbcr(rgb, shapes):
    height, width = rgb.shape[0], rgb.shape[1]
    count = 0
    for plane in range(3):
        count += shapes[plane, 0] * shapes[plane, 1]
    values = np.empty(count, np.float64)
    offset = 0
    for plane in range(3):
        rows, cols = shapes[plane, 0], shapes[plane, 1]
        sy, sx = height / rows, width / cols
        for y in range(rows):
            y0, y1 = y * sy, (y + 1) * sy
            for x in range(cols):
                x0, x1 = x * sx, (x + 1) * sx
                red = green = blue = 0.0
                for yy in range(int(y0), min(height, int(np.ceil(y1)))):
                    wy = min(y1, yy + 1.0) - max(y0, float(yy))
                    for xx in range(int(x0), min(width, int(np.ceil(x1)))):
                        weight = wy * (min(x1, xx + 1.0) - max(x0, float(xx)))
                        red += weight * rgb[yy, xx, 0]
                        green += weight * rgb[yy, xx, 1]
                        blue += weight * rgb[yy, xx, 2]
                scale = 1.0 / (sx * sy * 255.0)
                red, green, blue = red * scale, green * scale, blue * scale
                if plane == 0:
                    value = .299 * red + .587 * green + .114 * blue
                elif plane == 1:
                    value = .5 - .168736 * red - .331264 * green + .5 * blue
                else:
                    value = .5 + .5 * red - .418688 * green - .081312 * blue
                values[offset + y * cols + x] = 2.0 * value - 1.0
        offset += rows * cols
    return values


def source_values(rgb):
    """Area-integrate source RGB directly onto the table's Y/Cb/Cr grids."""
    image = np.asarray(rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('adaptive source conversion requires uint8 RGB')
    return _area_ycbcr(np.ascontiguousarray(image), SOURCE_SHAPES)
