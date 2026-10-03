"""Steepen edges the way they will look after the picture is blown up."""
import numpy as np
from scipy.fft import dctn, idctn
from scipy.ndimage import maximum_filter, minimum_filter

LABEL = 'Shock-filter edges (upscaled domain)'
HELP = ('Blow the sent band up the way an ideal viewer would, apply an '
        'Osher-Rudin shock filter there (it pushes each side of an edge '
        'toward its own plateau, so the edge narrows without a halo), then '
        'keep only the coefficients the wire carries and repeat. The result '
        'is the band-limited picture whose upscaled edges are as steep and '
        'clean as that band allows. About 3 ms per round for luma.')
PARAMS = {
    'rounds': (3, 1, 8, 1, 'shock then re-project rounds (about 3 ms each)', True),
    'strength': (0.6, 0.0, 1.5, 0.05, 'size of each shock step'),
    'up': (3, 2, 4, 1, 'upscaling used inside the filter', True),
    'chroma': (0, 0, 1, 1, '1 = steepen colour edges too', True),
}


def _gradient_magnitude(x):
    gy, gx = np.gradient(x)
    return np.hypot(gx, gy)


def post(grid, ctx, rounds, strength, up, chroma):
    if ctx.reference is None or (ctx.plane and not chroma):
        return grid
    up = int(up)
    rows, cols = grid.shape
    mask = ctx.mask
    coefficients = dctn(grid.astype(np.float32), norm='ortho') * mask
    # the range the source had nearby, on the upscaled pixel lattice
    low = minimum_filter(ctx.reduce(ctx.reference, 'min'), size=3, mode='nearest')
    high = maximum_filter(ctx.reduce(ctx.reference, 'max'), size=3, mode='nearest')
    low = np.kron(low, np.ones((up, up), np.float32)) - 0.02
    high = np.kron(high, np.ones((up, up), np.float32)) + 0.02
    big = np.zeros((rows*up, cols*up), np.float32)
    for _ in range(int(rounds)):
        big[:] = 0
        big[:rows, :cols] = coefficients*up
        x = idctn(big, norm='ortho')
        lap = (np.roll(x, 1, 0) + np.roll(x, -1, 0) + np.roll(x, 1, 1)
               + np.roll(x, -1, 1) - 4*x)
        # u_t = -sign(laplacian) |grad u|: erode where convex, dilate where concave
        x = x - (0.25*strength/up)*np.sign(lap)*_gradient_magnitude(x)*up
        x = np.clip(x, low, high)
        coefficients = dctn(x, norm='ortho')[:rows, :cols]/up * mask
    return idctn(coefficients, norm='ortho')
