"""Edge clean-up: keep the sent picture inside what the source contained."""
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

LABEL = 'Anti-ringing refit'
HELP = ('The wire holds a limited band, so a hard edge overshoots into a '
        'halo. This refit alternates between "only the coefficients that are '
        'sent" and "no darker or lighter than the source was nearby", which '
        'moves the halo out of the picture while keeping the edge as sharp '
        'as the band allows. No window: pair it with a low mix of another '
        'kernel by editing this file.')
PARAMS = {
    'iterations': (10, 0, 40, 1, 'projection rounds (about 0.15 ms each)', True),
    'radius': (1, 0, 3, 1, 'pixels of slack around the source range', True),
    'margin': (0.01, 0.0, 0.1, 0.005, 'extra slack, in picture units'),
    'chroma': (0, 0, 1, 1, '1 = refit colour planes too (about 2x the time)', True),
}
PROFILE_DEFAULTS = {
    'aspect-mono-500': {'luma_mix': 0.25, 'margin': 0.1},
    'aspect-fold-500': {'luma_mix': 0.05, 'margin': 0.1},
}


def post(grid, ctx, iterations, radius, margin, chroma):
    if ctx.reference is None or int(iterations) < 1:
        return grid
    if ctx.plane and not chroma:
        return grid
    low = ctx.reduce(ctx.reference, 'min')
    high = ctx.reduce(ctx.reference, 'max')
    size = 2*int(radius) + 1
    if size > 1:
        low = minimum_filter(low, size=size, mode='nearest')
        high = maximum_filter(high, size=size, mode='nearest')
    low, high = low - margin, high + margin
    shown = grid
    for _ in range(int(iterations)):
        shown = np.clip(ctx.project(shown), low, high)
    return ctx.project(shown)
