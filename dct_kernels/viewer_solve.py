"""Choose the coefficients that look best *through* the viewer's upscaler."""
from functools import lru_cache

import numpy as np
from PIL import Image
from scipy.fft import dctn, idct, idctn
from scipy.ndimage import maximum_filter, minimum_filter

LABEL = 'Viewer-model solve'
HELP = ('Plain truncation is the best choice only for an ideal receiver. '
        'This models the viewer (the receiver shows its sent-size picture '
        'through an upscaler) and solves, by a few conjugate-gradient steps, '
        'for the sent coefficients whose upscaled picture is closest to the '
        'ideal enlargement of the band: a regularised deconvolution of the '
        'viewer, limited to the coefficients the wire carries. Then halos '
        'are moved back inside the source range. `viewer` must match the '
        'receiver\'s upscaler (0 bilinear, 1 bicubic, 2 nearest).')
PARAMS = {
    'viewer': (0, 0, 2, 1, '0 bilinear, 1 bicubic, 2 nearest', True),
    'iterations': (10, 1, 40, 1, 'conjugate-gradient steps', True),
    'lam': (0.02, 0.0, 1.0, 0.005, 'pull toward the plain coefficients (bigger = gentler)'),
    'up': (4, 2, 6, 1, 'model lattice: viewer pixels per sent pixel', True),
    'tame': (4, 0, 20, 1, 'halo clean-up rounds after the solve (0 = off)', True),
}
_FILTERS = (Image.Resampling.BILINEAR, Image.Resampling.BICUBIC,
            Image.Resampling.NEAREST)


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


def post(grid, ctx, viewer, iterations, lam, up, tame):
    if ctx.plane:
        return grid
    rows, cols = grid.shape
    sent_rows, sent_cols = ctx.sent
    up = int(up)
    mask = ctx.mask.astype(np.float64)
    my, mx = _model(int(viewer), rows, cols, sent_rows, sent_cols, up)
    iy, ix = _ideal(rows, cols, sent_rows, sent_cols, up)
    c0 = dctn(grid, norm='ortho')*mask
    # the picture an ideal receiver shows is what the viewer should show
    target = iy @ c0 @ ix.T

    def forward(c):
        return my @ (c*mask) @ mx.T

    def adjoint(r):
        return (my.T @ r @ mx)*mask

    def normal(c):
        return adjoint(forward(c)) + lam*c*mask

    b = adjoint(target) + lam*c0
    c = c0.copy()
    r = b - normal(c)
    p = r.copy()
    rs = float((r*r).sum())
    for _ in range(int(iterations)):
        if rs < 1e-14:
            break
        hp = normal(p)
        alpha = rs/max(float((p*hp).sum()), 1e-30)
        c += alpha*p
        r -= alpha*hp
        rs_new = float((r*r).sum())
        p = r + (rs_new/rs)*p
        rs = rs_new
    shown = idctn(c, norm='ortho')
    if int(tame) > 0 and ctx.reference is not None:
        low = minimum_filter(ctx.reduce(ctx.reference, 'min'), size=3, mode='nearest')
        high = maximum_filter(ctx.reduce(ctx.reference, 'max'), size=3, mode='nearest')
        for _ in range(int(tame)):
            shown = np.clip(ctx.project(shown), low - 0.02, high + 0.02)
        shown = ctx.project(shown)
    return shown
