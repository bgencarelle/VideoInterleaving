"""Gaussian low-pass: no ringing at all, at the cost of edge width."""
import numpy as np

LABEL = 'Gaussian'
HELP = 'Gaussian blur of the given sigma in sent pixels. No overshoot.'
PARAMS = {'sigma': (0.45, 0.1, 1.2, 0.025, 'sigma, in sent pixels')}
PROFILE_DEFAULTS = {
    'aspect-mono-500': {'luma_mix': 0.25, 'sigma': 0.1},
    'aspect-fold-500': {'luma_mix': 0.05, 'sigma': 0.1},
}


def response(nu, sigma):
    return np.exp(-2*(np.pi*sigma*nu)**2)
