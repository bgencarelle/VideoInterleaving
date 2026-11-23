"""Windowed-sinc resampling, the classic sharp downscaler."""
import numpy as np

LABEL = 'Lanczos'
HELP = ('Windowed sinc. Sharp edges with slight ringing; fewer lobes are '
        'softer. width above 1 softens, below 1 sharpens (and rings).')
SUPPORT = 5.0
PARAMS = {
    'lobes': (3, 2, 5, 1, 'sinc lobes kept (a)', True),
    'width': (0.9, 0.5, 1.6, 0.05, 'kernel width in sent pixels'),
}
PROFILE_DEFAULTS = {
    'aspect-mono-500': {'luma_mix': 0.25, 'width': 0.5},
    'aspect-fold-500': {'luma_mix': 0.5, 'width': 0.5},
}


def kernel(x, lobes, width):
    lobes = int(round(lobes))
    t = np.asarray(x, np.float64) / width
    return np.where(np.abs(t) < lobes, np.sinc(t) * np.sinc(t / lobes), 0.0)
