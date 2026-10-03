"""Costella's Magic Kernel with its Sharp 2013 follow-up filter."""
import numpy as np

LABEL = 'Magic Kernel Sharp'
HELP = ('Quadratic B-spline (no ringing, no aliasing of its own) followed by '
        'the small [-1 6 -1] sharpening filter. sharp = 0 is the plain '
        'magic kernel, 1 the published Sharp, above 1 pushes further.')
PARAMS = {'sharp': (1.0, 0.0, 2.0, 0.05, 'amount of the [-1 6 -1] filter')}


def response(nu, sharp):
    smooth = np.sinc(nu) ** 3
    sharpen = (6.0 - 2.0*np.cos(2*np.pi*nu)) / 4.0
    return smooth * (1.0 + sharp*(sharpen - 1.0))
