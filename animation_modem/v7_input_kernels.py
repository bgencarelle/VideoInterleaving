"""Persistent Numba kernels for V7's live audio-input path."""
import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def _leg_correlation_sums(x):
    """Mean-removed left/right powers and cross product of a stereo block."""
    count = x.shape[0]
    left_mean = 0.0
    right_mean = 0.0
    for i in range(count):
        left_mean += x[i, 0]
        right_mean += x[i, 1]
    left_mean /= count
    right_mean /= count
    left_power = 0.0
    right_power = 0.0
    cross = 0.0
    for i in range(count):
        left = x[i, 0]-left_mean
        right = x[i, 1]-right_mean
        left_power += left*left
        right_power += right*right
        cross += left*right
    return left_power, right_power, cross


@njit(cache=True, fastmath=False)
def _mono_gain(samples, gain):
    """Apply capture gain and mono-mix a one- or two-channel scan window."""
    out = np.empty(samples.shape[0], dtype=samples.dtype)
    scale = samples.dtype.type(gain)
    half = samples.dtype.type(.5)
    for i in range(samples.shape[0]):
        if samples.ndim == 1:
            out[i] = samples[i]*scale
        elif samples.shape[1] == 1:
            out[i] = samples[i, 0]*scale
        else:
            left = samples[i, 0]*scale
            right = samples[i, 1]*scale
            out[i] = (left+right)*half
    return out
