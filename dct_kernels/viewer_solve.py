"""Choose the coefficients that look best *through* the viewer's upscaler."""
from functools import lru_cache

import numpy as np
from PIL import Image
from scipy.fft import idct          # set-up only: the model matrices

from animation_modem.v7_kernel_solve import (forward_plane, inverse_plane,
                                             masked_gram_solve)

LABEL = 'Viewer-model solve'
HELP = ('Plain truncation is the best choice only for an ideal receiver. '
        'This models the viewer (the receiver shows its sent-size picture '
        'through an upscaler) and solves, by a few conjugate-gradient steps, '
        'for the sent coefficients whose upscaled picture is closest to the '
        'ideal enlargement of the band: a regularised deconvolution of the '
        'viewer, limited to the coefficients the wire carries. Then halos '
        'are moved back inside the source range. `viewer` must match the '
        'receiver\'s upscaler: 0 bilinear, 1 bicubic, 2 nearest, or 3 for '
        'the frequency-space enlargement the receiver uses by default '
        '(animation_modem/v7_viewer_model.py). Against that one there is '
        'nothing to correct and the plain coefficients are sent.')
PARAMS = {
    'viewer': (0, 0, 3, 1, '0 bilinear, 1 bicubic, 2 nearest, '
               "3 the receiver's own (frequency space)", True),
    'iterations': (10, 1, 40, 1, 'conjugate-gradient steps', True),
    'lam': (0.5, 0.0, 1.0, 0.005, 'pull toward the plain coefficients (bigger = gentler)'),
    'up': (4, 2, 6, 1, 'model lattice: viewer pixels per sent pixel', True),
    'tame': (4, 0, 20, 1, 'halo clean-up rounds after the solve (0 = off)', True),
}
PROFILE_DEFAULTS = {
    'aspect-mono-500': {'luma_mix': 0.25, 'tame': 0},
    'aspect-fold-500': {'luma_mix': 0.2, 'tame': 0},
}
_FILTERS = (Image.Resampling.BILINEAR, Image.Resampling.BICUBIC,
            Image.Resampling.NEAREST)
RECEIVER_VIEWER = 3     # enlarges in frequency space: see _model


def viewer_from(display):
    """The `viewer` setting for a receiver display description
    (animation_modem.v7_viewer_model.RECEIVER_DISPLAY)."""
    if display.get('enlarge') == 'dct':
        return RECEIVER_VIEWER
    return {'bilinear': 0, 'bicubic': 1, 'nearest': 2}[display['draw']]


@lru_cache(maxsize=64)
def _resample_matrix(n, m, method):
    """(m x n) matrix of PIL's own resize along one axis."""
    matrix = np.empty((m, n), np.float64)
    for index in range(n):
        line = np.zeros((1, n), np.float32)
        line[0, index] = 1.0
        out = Image.fromarray(line, 'F').resize((m, 1), method)
        matrix[:, index] = np.asarray(out)[0]
    matrix.setflags(write=False)
    return matrix


@lru_cache(maxsize=32)
def _model(viewer, rows, cols, sent_rows, sent_cols, up):
    """My, Mx: coefficient -> viewer-shown picture, per axis."""
    if int(viewer) == RECEIVER_VIEWER:
        # The receiver zero-pads the coefficients it was sent and inverts:
        # that is the ideal enlargement itself.
        return _ideal(rows, cols, sent_rows, sent_cols, up)

    def axis(n, sent):
        reduce_ = _resample_matrix(n, sent, Image.Resampling.BOX)
        enlarge = _resample_matrix(sent, sent*up, _FILTERS[int(viewer)])
        basis = idct(np.eye(n), axis=0, norm='ortho')
        return np.ascontiguousarray(enlarge @ reduce_ @ basis)
    return axis(rows, sent_rows), axis(cols, sent_cols)


@lru_cache(maxsize=32)
def _ideal(rows, cols, sent_rows, sent_cols, up):
    """Per-axis map: coefficient -> ideal band-limited enlargement."""
    def axis(n, sent):
        target = sent*up
        padded = np.zeros((target, n))
        padded[:n, :] = np.eye(n)*np.sqrt(target/n)
        return np.ascontiguousarray(idct(padded, axis=0, norm='ortho'))
    return axis(rows, sent_rows), axis(cols, sent_cols)


@lru_cache(maxsize=32)
def _normal(viewer, rows, cols, sent_rows, sent_cols, up):
    """The solve's fixed matrices, on the coefficient grid: M'M per axis and
    M'T / T'M (viewer model against ideal enlargement).  The viewer-sized
    pictures themselves are never formed per frame."""
    my, mx = _model(viewer, rows, cols, sent_rows, sent_cols, up)
    iy, ix = _ideal(rows, cols, sent_rows, sent_cols, up)
    return tuple(np.ascontiguousarray(matrix, np.float64) for matrix in
                 (my.T @ my, mx.T @ mx, my.T @ iy, ix.T @ mx))


def post(grid, ctx, viewer, iterations, lam, up, tame):
    if ctx.plane:
        return grid
    rows, cols = grid.shape
    sent_rows, sent_cols = ctx.sent
    mask = ctx.weights
    transforms = ctx.transforms
    c0 = forward_plane(np.ascontiguousarray(grid, np.float64), transforms[0],
                       transforms[3])*mask
    # The picture an ideal receiver shows is what the viewer should show:
    # minimise |M c - T c0|^2 + lam |c - c0|^2 over the sent coefficients.
    c = masked_gram_solve(c0, mask, *_normal(int(viewer), rows, cols, sent_rows,
                                             sent_cols, int(up)),
                          float(lam), int(iterations))
    shown = inverse_plane(c, transforms[1], transforms[2])
    if int(tame) > 0 and ctx.reference is not None:
        shown = ctx.tame(shown, tame)
    return shown
