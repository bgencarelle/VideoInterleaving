"""Steepen edges the way they will look after the picture is blown up."""
import numpy as np

from animation_modem.v7_kernel_solve import (enlarged_transforms, forward_plane,
                                             inverse_plane, shock_rounds)

LABEL = 'Shock-filter edges (upscaled domain)'
HELP = ('Blow the sent band up the way an ideal viewer would, apply an '
        'Osher-Rudin shock filter there (it pushes each side of an edge '
        'toward its own plateau, so the edge narrows without a halo), then '
        'keep only the coefficients the wire carries and repeat. The result '
        'is the band-limited picture whose upscaled edges are as steep and '
        'clean as that band allows. About 3 ms per round for luma.')
PARAMS = {
    'rounds': (2, 1, 8, 1, 'shock then re-project rounds (about 3 ms each)', True),
    'strength': (0.6, 0.0, 1.5, 0.05, 'size of each shock step'),
    'up': (3, 2, 4, 1, 'upscaling used inside the filter', True),
    'chroma': (0, 0, 1, 1, '1 = steepen colour edges too', True),
}
PROFILE_DEFAULTS = {
    'aspect-mono-500': {'luma_mix': 0.25, 'strength': 0.9},
}


def post(grid, ctx, rounds, strength, up, chroma):
    if ctx.reference is None or (ctx.plane and not chroma):
        return grid
    up = int(up)
    rows, cols = grid.shape
    transforms = ctx.transforms
    coefficients = forward_plane(np.ascontiguousarray(grid, np.float64), transforms[0],
                                 transforms[3])*ctx.weights
    # the range the source had nearby (looked up per enlarged pixel)
    low, high = ctx.source_range(1, 0.02)
    # u_t = -sign(laplacian) |grad u|: erode where convex, dilate where concave
    coefficients = shock_rounds(coefficients, ctx.weights, low, high, int(rounds),
                                0.25*float(strength), up,
                                *enlarged_transforms(rows, cols, up))
    return inverse_plane(coefficients, transforms[1], transforms[2])
