"""Persistent Numba kernels for V7's live audio-input path."""
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
