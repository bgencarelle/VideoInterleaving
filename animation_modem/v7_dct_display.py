"""Direct DCT reconstruction of decoded V7 planes at an arbitrary size.

The decoder's planes are band-limited: each is an inverse DCT of a low-
frequency block of coefficients on its coder grid. Displaying one at another
size is then exact by evaluating the same cosine series on the new pixel
grid, which is what padding the spectrum and running a full inverse DCT at
display size computes. Here only the occupied part of the spectrum is
evaluated, as two separable products with cached float32 bases, in one thread
(no FFT or BLAS pools left spinning in a receiver that also plays audio).
"""
from functools import lru_cache
import math

import numpy as np
from numba import njit
from scipy.fft import dctn

# Coefficients below this fraction of a plane's largest one are rounding
# residue of the decoder's inverse DCT (float32 planes), not signal.
SUPPORT_RELATIVE_FLOOR = 1e-6


@lru_cache(maxsize=64)
def _basis(size, modes):
    """Orthonormal DCT-II synthesis basis (size x modes), float32."""
    samples = np.arange(size, dtype=np.float64)[:, None]
    orders = np.arange(modes, dtype=np.float64)[None, :]
    basis = np.cos(np.pi*(2*samples+1)*orders/(2*size))*math.sqrt(2.0/size)
    basis[:, 0] /= math.sqrt(2.0)
    basis = np.ascontiguousarray(basis, dtype=np.float32)
    basis.setflags(write=False)
    return basis


@lru_cache(maxsize=64)
def _basis_rows(size, modes):
    rows = np.ascontiguousarray(_basis(size, modes).T)
    rows.setflags(write=False)
    return rows


@njit(cache=True, fastmath=True, nogil=True)
def _synthesize(left, coefficients, right_rows):
    """``left @ coefficients @ right_rows`` for a tall/wide output.

    ``left`` is H x r, ``coefficients`` r x c, ``right_rows`` c x W. The
    output row loop is blocked four modes at a time so each output row is
    read and written c/4 times instead of c.
    """
    height = left.shape[0]
    rows = coefficients.shape[0]
    cols = coefficients.shape[1]
    width = right_rows.shape[1]
    partial = np.zeros((height, cols), dtype=np.float32)
    for y in range(height):
        for u in range(rows):
            weight = left[y, u]
            for v in range(cols):
                partial[y, v] += weight*coefficients[u, v]
    out = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        row = out[y]
        v = 0
        while v+4 <= cols:
            a0 = partial[y, v]
            a1 = partial[y, v+1]
            a2 = partial[y, v+2]
            a3 = partial[y, v+3]
            r0 = right_rows[v]
            r1 = right_rows[v+1]
            r2 = right_rows[v+2]
            r3 = right_rows[v+3]
            for x in range(width):
                row[x] += a0*r0[x]+a1*r1[x]+a2*r2[x]+a3*r3[x]
            v += 4
        while v < cols:
            weight = partial[y, v]
            basis_row = right_rows[v]
            for x in range(width):
                row[x] += weight*basis_row[x]
            v += 1
    return out


def spectral_support(coefficients, floor=SUPPORT_RELATIVE_FLOOR):
    """Rows and columns that hold every coefficient above the floor."""
    magnitude = np.abs(coefficients)
    peak = float(magnitude.max(initial=0.0))
    if peak == 0.0:
        return 1, 1
    occupied = magnitude > peak*floor
    rows = int(np.flatnonzero(occupied.any(axis=1))[-1])+1
    cols = int(np.flatnonzero(occupied.any(axis=0))[-1])+1
    return rows, cols


def reconstruct_plane(plane, target_shape):
    """Evaluate a decoded plane's DCT series on a target_shape pixel grid.

    Equivalent to zero-padding (or truncating) the plane's orthonormal DCT
    to the target size, scaling by sqrt(target area / source area) and
    inverse transforming there.
    """
    plane = np.asarray(plane, dtype=np.float64)
    if plane.ndim != 2 or min(plane.shape) <= 0:
        raise ValueError('DCT reconstruction needs a non-empty 2-D plane')
    target_height, target_width = (int(value) for value in target_shape)
    if target_height <= 0 or target_width <= 0:
        raise ValueError('target dimensions must be positive')
    source_height, source_width = plane.shape
    coefficients = dctn(plane, norm='ortho')
    rows, cols = spectral_support(coefficients)
    rows = min(rows, target_height)
    cols = min(cols, target_width)
    block = coefficients[:rows, :cols]*math.sqrt(
        (target_height*target_width)/(source_height*source_width))
    return _synthesize(_basis(target_height, rows),
                       np.ascontiguousarray(block, dtype=np.float32),
                       _basis_rows(target_width, cols))


def viewport_shapes(plane_shapes, viewport_size):
    """Luma at the viewport's (width, height); chroma at the same scale."""
    target_width, target_height = (int(value) for value in viewport_size)
    if target_width <= 0 or target_height <= 0:
        raise ValueError('viewport dimensions must be positive')
    luma_height, luma_width = plane_shapes[0]
    scale_x = target_width/luma_width
    scale_y = target_height/luma_height
    return ((target_height, target_width),) + tuple(
        (max(1, round(rows*scale_y)), max(1, round(cols*scale_x)))
        for rows, cols in plane_shapes[1:])


def reconstruct_planes(planes, viewport_size):
    """Decoded Y[/Cb/Cr] planes straight to a viewport-size display."""
    shapes = tuple(np.shape(plane) for plane in planes)
    return tuple(reconstruct_plane(plane, shape) for plane, shape in zip(
        planes, viewport_shapes(shapes, viewport_size)))
