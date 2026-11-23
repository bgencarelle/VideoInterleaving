"""The window that leaks the least energy out of the band (Slepian / DPSS)."""
import numpy as np
from scipy.signal.windows import dpss

LABEL = 'Slepian (DPSS) window'
HELP = ('Among all windows of this length, the discrete prolate spheroidal '
        'sequence concentrates the most energy inside the band, so the least '
        'of an edge spills into ringing. `nw` is the time-bandwidth product: '
        'about 1 is nearly flat, 4 is a Gaussian-like taper. A cousin of '
        'Kaiser, derived rather than approximated.')
PARAMS = {
    'nw': (0.5, 0.5, 6.0, 0.1, 'time-bandwidth product'),
    'floor': (0.15, 0.0, 0.5, 0.01, 'minimum gain at the band edge'),
}
PROFILE_DEFAULTS = {
    'aspect-mono-500': {'floor': 0.5, 'luma_mix': 0.25},
    'aspect-fold-500': {'luma_mix': 0.05},
}


def response(nu, nw, floor):
    nu = np.asarray(nu, float)
    window = dpss(1025, nw)
    window = window[512:]/window[512]                  # centre = DC, edge = 0.5
    position = np.clip(np.abs(nu)/0.5, 0.0, 1.0)*(len(window) - 1)
    gain = np.interp(position, np.arange(len(window)), window)
    return floor + (1.0 - floor)*gain
