"""Total-variation flattening: clean plateaus and sharp edges when enlarged."""
import numpy as np

from animation_modem.v7_kernel_solve import blend, total_variation_flatten

LABEL = 'TV cartoon refit'
HELP = ('Total-variation (Rudin-Osher-Fatemi) smoothing keeps edges and '
        'flattens everything else, so when the small picture is enlarged '
        'its regions are clean and its outlines crisp, and there is less '
        'texture for noise to corrupt. The flattened plane is re-fitted to '
        'the coefficients the wire carries. A stylised look: use `keep` to '
        'bring texture back.')
PARAMS = {
    'weight': (0.03, 0.0, 0.3, 0.005, 'smoothing strength (picture units)'),
    'iterations': (25, 1, 80, 1, 'Chambolle steps (about 0.03 ms each)', True),
    'keep': (0.5, 0.0, 1.0, 0.05, 'fraction of the original detail kept'),
    'chroma': (1, 0, 1, 1, '1 = flatten colour planes too', True),
}
PROFILE_DEFAULTS = {
    'aspect-mono-500': {'keep': 0.75, 'luma_mix': 0.25},
    'aspect-fold-500': {'keep': 0.75, 'luma_mix': 0.05},
}


def post(grid, ctx, weight, iterations, keep, chroma):
    if weight <= 0 or (ctx.plane and not chroma):
        return grid
    grid = np.ascontiguousarray(grid, np.float64)
    flat = total_variation_flatten(grid, float(weight), int(iterations))
    return ctx.project(blend(flat, grid, float(keep)))
