"""Total-variation flattening: clean plateaus and sharp edges when enlarged."""
import numpy as np

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


def _tv(image, weight, iterations):
    """Chambolle's projection algorithm for the ROF model."""
    px = np.zeros_like(image)
    py = np.zeros_like(image)
    tau = 0.125
    for _ in range(int(iterations)):
        div = (px - np.roll(px, 1, 1)) + (py - np.roll(py, 1, 0))
        u = image - weight*div
        gx = np.roll(u, -1, 1) - u
        gy = np.roll(u, -1, 0) - u
        norm = 1.0 + tau/weight*np.hypot(gx, gy)
        px = (px - tau/weight*gx)/norm
        py = (py - tau/weight*gy)/norm
    div = (px - np.roll(px, 1, 1)) + (py - np.roll(py, 1, 0))
    return image - weight*div


def post(grid, ctx, weight, iterations, keep, chroma):
    if weight <= 0 or (ctx.plane and not chroma):
        return grid
    flat = _tv(grid.astype(np.float32), weight, iterations)
    blended = flat + keep*(grid - flat)
    return ctx.project(blended)
