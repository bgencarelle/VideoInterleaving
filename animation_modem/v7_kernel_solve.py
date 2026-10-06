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
