"""Compiled solves for the DCT downscale kernels (dct_kernels/).

A kernel that fits sent coefficients to a model of the viewer minimises
``|M c - T c0|^2 + lam |c - c0|^2`` over the coefficients the wire carries,
where ``M`` (coefficients -> the picture the viewer shows) and ``T``
(coefficients -> the picture it should show) are separable: one matrix per
axis.  Applying ``M`` and its transpose in every conjugate-gradient step
costs the size of the *viewer's* picture, which for a wire whose sent
coefficients reach the edge of the grid (the nested folds) is sixteen times
the grid.  The step only ever needs ``M'M`` and ``M'T``, which are
grid-sized and fixed per layout, so they are formed once and the solve runs
entirely on the grid.
"""
import math
from functools import lru_cache

import numpy as np
from numba import njit


@njit(cache=True)
def _masked_product(left, plane, right, mask, out):
    """``out = (left @ plane @ right) * mask``."""
    product = np.dot(np.dot(left, plane), right)
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            out[r, c] = product[r, c]*mask[r, c]


@njit(cache=True)
def masked_gram_solve(start, mask, gram_rows, gram_cols, cross_rows, cross_cols,
                      lam, iterations):
    """Conjugate-gradient solution of
    ``(G_r (c*mask) G_c)*mask + lam*c*mask = (K_r start K_c)*mask + lam*start``
    from ``start`` (already zero outside ``mask``).

    ``gram_rows``/``gram_cols``: ``M'M`` per axis; ``cross_rows``: ``M'T``
    for the row axis; ``cross_cols``: ``T'M`` for the column axis.
    """
    rows, cols = start.shape
    target = np.empty((rows, cols))
    _masked_product(cross_rows, start, cross_cols, mask, target)
    current = start.copy()
    applied = np.empty((rows, cols))
    _masked_product(gram_rows, current*mask, gram_cols, mask, applied)
    residual = np.empty((rows, cols))
    for r in range(rows):
        for c in range(cols):
            residual[r, c] = (target[r, c]+lam*start[r, c] -
                              applied[r, c]-lam*current[r, c]*mask[r, c])
    direction = residual.copy()
    power = 0.0
    for r in range(rows):
        for c in range(cols):
            power += residual[r, c]*residual[r, c]
    for _ in range(iterations):
        if power < 1e-14:
            break
        _masked_product(gram_rows, direction*mask, gram_cols, mask, applied)
        along = 0.0
        for r in range(rows):
            for c in range(cols):
                applied[r, c] += lam*direction[r, c]*mask[r, c]
                along += direction[r, c]*applied[r, c]
        alpha = power/max(along, 1e-30)
        following = 0.0
        for r in range(rows):
            for c in range(cols):
                current[r, c] += alpha*direction[r, c]
                residual[r, c] -= alpha*applied[r, c]
                following += residual[r, c]*residual[r, c]
        ratio = following/power
        for r in range(rows):
            for c in range(cols):
                direction[r, c] = residual[r, c]+ratio*direction[r, c]
        power = following
    return current


# ------------------------------------------------------------ shared pieces
# What the kernels' per-frame ``post`` steps are made of.  Transforms are
# orthonormal DCT-II matrices (``dct_basis``), formed once per size by the
# caller, so a transform is two matrix products.

@njit(cache=True)
def dct_basis(n):
    """Orthonormal DCT-II matrix: coefficients = basis @ samples."""
    out = np.empty((n, n))
    for k in range(n):
        scale = math.sqrt((1.0 if k == 0 else 2.0)/n)
        for i in range(n):
            out[k, i] = scale*math.cos(math.pi*(2*i+1)*k/(2.0*n))
    return out


@njit(cache=True)
def forward_plane(plane, rows_basis, cols_transposed):
    return np.dot(np.dot(rows_basis, plane), cols_transposed)


@njit(cache=True)
def inverse_plane(coefficients, rows_transposed, cols_basis):
    return np.dot(np.dot(rows_transposed, coefficients), cols_basis)


@njit(cache=True)
def project(plane, mask, rows_basis, rows_transposed, cols_basis, cols_transposed):
    """The picture the receiver can show: only the coefficients sent
    (``mask`` is 1 where sent, 0 elsewhere)."""
    coefficients = forward_plane(plane, rows_basis, cols_transposed)
    for r in range(coefficients.shape[0]):
        for c in range(coefficients.shape[1]):
            coefficients[r, c] *= mask[r, c]
    return inverse_plane(coefficients, rows_transposed, cols_basis)


@njit(cache=True)
def clip_into(plane, low, high):
    out = np.empty(plane.shape)
    for r in range(plane.shape[0]):
        for c in range(plane.shape[1]):
            value = plane[r, c]
            if value < low[r, c]:
                value = low[r, c]
            elif value > high[r, c]:
                value = high[r, c]
            out[r, c] = value
    return out


@njit(cache=True)
def clip_project(plane, mask, low, high, rounds, rows_basis, rows_transposed,
                 cols_basis, cols_transposed):
    """``rounds`` times: show what the receiver can, move it back inside
    [low, high]; then show what the receiver can."""
    shown = plane
    for _ in range(rounds):
        shown = clip_into(project(shown, mask, rows_basis, rows_transposed,
                                  cols_basis, cols_transposed), low, high)
    return project(shown, mask, rows_basis, rows_transposed, cols_basis, cols_transposed)


@njit(cache=True)
def neighbourhood_extreme(plane, radius, largest, offset):
    """Largest (or smallest) value within ``radius`` cells, edges repeated,
    plus ``offset``."""
    rows, cols = plane.shape
    out = np.empty((rows, cols))
    for r in range(rows):
        for c in range(cols):
            best = plane[r, c]
            for dr in range(-radius, radius+1):
                rr = min(max(r+dr, 0), rows-1)
                for dc in range(-radius, radius+1):
                    value = plane[rr, min(max(c+dc, 0), cols-1)]
                    if (largest and value > best) or (not largest and value < best):
                        best = value
            out[r, c] = best+offset
    return out


@njit(cache=True)
def _cells(size, count):
    """First and one-past-last source index of each of ``count`` cells along
    an axis of ``size`` (neighbouring cells may share a boundary pixel)."""
    first, last = np.empty(count, np.int64), np.empty(count, np.int64)
    for i in range(count):
        low = size*i/count
        high = size*(i+1)/count
        first[i] = min(int(math.floor(low+1e-9)), size-1)
        last[i] = max(min(int(math.ceil(high-1e-9)), size), first[i]+1)
    last[count-1] = size
    return first, last


@njit(cache=True)
def reduce_plane(plane, rows, cols, how):
    """A pixel-domain plane brought down to (rows, cols): ``how`` 0 mean,
    1 smallest, 2 largest over the source pixels each cell covers (rows
    first, then columns)."""
    height, width = plane.shape
    first, last = _cells(height, rows)
    tall = np.empty((rows, width))
    for i in range(rows):
        for c in range(width):
            value = plane[first[i], c]
            for r in range(first[i]+1, last[i]):
                if how == 0:
                    value += plane[r, c]
                elif how == 1:
                    value = min(value, plane[r, c])
                else:
                    value = max(value, plane[r, c])
            tall[i, c] = value/(last[i]-first[i]) if how == 0 else value
    first, last = _cells(width, cols)
    out = np.empty((rows, cols))
    for i in range(rows):
        for j in range(cols):
            value = tall[i, first[j]]
            for c in range(first[j]+1, last[j]):
                if how == 0:
                    value += tall[i, c]
                elif how == 1:
                    value = min(value, tall[i, c])
                else:
                    value = max(value, tall[i, c])
            out[i, j] = value/(last[j]-first[j]) if how == 0 else value
    return out


@njit(cache=True)
def shock_rounds(coefficients, mask, low, high, rounds, amount, up,
                 tall_rows, tall_rows_transposed, wide_cols, wide_cols_transposed):
    """Osher-Rudin shock filter on the band enlarged ``up`` times.

    ``coefficients``: the sent band (masked).  ``tall_rows``: the first
    ``rows`` rows of the enlarged row basis (rows x rows*up) and its
    transpose; ``wide_cols`` likewise for columns.  ``low``/``high``: the
    source's range on the enlarged lattice.  Returns the band after
    ``rounds`` of: enlarge, erode where convex and dilate where concave,
    clip, re-project.
    """
    rows, cols = coefficients.shape
    current = coefficients.copy()
    for _ in range(rounds):
        x = np.dot(np.dot(tall_rows_transposed, current*up), wide_cols)
        height, width = x.shape
        moved = np.empty((height, width))
        for r in range(height):
            above, below = (r-1) % height, (r+1) % height
            for c in range(width):
                left, right = (c-1) % width, (c+1) % width
                laplacian = x[above, c]+x[below, c]+x[r, left]+x[r, right]-4.0*x[r, c]
                # Gradient as numpy.gradient takes it: central inside,
                # one-sided at the edges.
                if r == 0:
                    gy = x[1, c]-x[0, c]
                elif r == height-1:
                    gy = x[r, c]-x[r-1, c]
                else:
                    gy = .5*(x[r+1, c]-x[r-1, c])
                if c == 0:
                    gx = x[r, 1]-x[r, 0]
                elif c == width-1:
                    gx = x[r, c]-x[r, c-1]
                else:
                    gx = .5*(x[r, c+1]-x[r, c-1])
                sign = 1.0 if laplacian > 0 else (-1.0 if laplacian < 0 else 0.0)
                value = x[r, c]-amount*sign*math.sqrt(gx*gx+gy*gy)
                bound_low, bound_high = low[r//up, c//up], high[r//up, c//up]
                if value < bound_low:
                    value = bound_low
                elif value > bound_high:
                    value = bound_high
                moved[r, c] = value
        back = np.dot(np.dot(tall_rows, moved), wide_cols_transposed)
        for r in range(rows):
            for c in range(cols):
                current[r, c] = back[r, c]/up*mask[r, c]
    return current


@njit(cache=True)
def total_variation_flatten(image, weight, iterations):
    """Chambolle's projection algorithm for the ROF model (wrapping edges)."""
    rows, cols = image.shape
    px, py = np.zeros((rows, cols)), np.zeros((rows, cols))
    u = np.empty((rows, cols))
    tau = 0.125
    for step in range(iterations+1):
        for r in range(rows):
            for c in range(cols):
                divergence = (px[r, c]-px[r, (c-1) % cols])+(py[r, c]-py[(r-1) % rows, c])
                u[r, c] = image[r, c]-weight*divergence
        if step == iterations:
            break
        for r in range(rows):
            for c in range(cols):
                gx = u[r, (c+1) % cols]-u[r, c]
                gy = u[(r+1) % rows, c]-u[r, c]
                norm = 1.0+tau/weight*math.sqrt(gx*gx+gy*gy)
                px[r, c] = (px[r, c]-tau/weight*gx)/norm
                py[r, c] = (py[r, c]-tau/weight*gy)/norm
    return u


@njit(cache=True)
def blend(a, b, keep):
    """``a + keep*(b - a)``."""
    out = np.empty(a.shape)
    for r in range(a.shape[0]):
        for c in range(a.shape[1]):
            out[r, c] = a[r, c]+keep*(b[r, c]-a[r, c])
    return out


@lru_cache(maxsize=32)
def transforms(rows, cols):
    """(row basis, its transpose, column basis, its transpose) for a grid;
    formed once per size."""
    row, col = dct_basis(int(rows)), dct_basis(int(cols))
    return (row, np.ascontiguousarray(row.T), col, np.ascontiguousarray(col.T))


@lru_cache(maxsize=32)
def enlarged_transforms(rows, cols, up):
    """The first ``rows``/``cols`` rows of the bases ``up`` times larger, and
    their transposes: a band zero-padded to the enlarged lattice."""
    tall = np.ascontiguousarray(dct_basis(int(rows)*int(up))[:int(rows)])
    wide = np.ascontiguousarray(dct_basis(int(cols)*int(up))[:int(cols)])
    return (tall, np.ascontiguousarray(tall.T), wide, np.ascontiguousarray(wide.T))
