"""Wavelet source coder: the v4 alternative to the DCT in ``core.SourceCoder``.

Same wire, same slot count, same power-allocation machinery, same header --
only the transform inside the coder changes. ``dctn``/``idctn`` become an
orthonormal 2-D DWT, and the coder keeps the top-N coefficients by average
magnitude across the bake (measured fixed mask). This closes the clean-quality
gap while preserving the noise-robustness advantage.

The Haar wavelet is used because it is orthonormal (Parseval holds, so the
allocation and Wiener de-bias inherited from ``SourceCoder`` still apply) and
trivial to invert exactly with periodic boundaries.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from .core import SourceCoder, _slot_order

SQRT2 = float(np.sqrt(2))

# Masks for luma (Y) plane: grid 96x80 -> DWT 48x40 subbands -> keep 1920/7680
MASK_LL_Y = np.array([
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,True,True,True,True,True,True,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,True,True,True,True,True,True,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,False,True,True,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,False,False,False,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,False,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,True,True,True,True,False,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,True,True,True,True,True,True,True,True,True],
    [True,False,True,True,False,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,True,True,True,True,True,False,True,True,True],
    [False,False,True,True,False,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,True,True,True,True,True,False,True,True,True],
    [False,True,True,False,False,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,True,True,True,True,True,True,False,True,True],
    [False,True,True,False,False,True,True,False,False,False,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,True,True,True,True,True,True,True,False,True],
    [False,True,True,False,False,True,True,False,False,False,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,True,True,True,True,True,True,True,True,True],
    [True,True,True,False,False,True,True,False,False,False,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,False,True,True,True,True,True,True,True,True],
    [True,True,True,False,False,True,True,False,False,False,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,False,True,True,True,True,True,True,True,True],
    [True,True,True,False,False,True,True,False,False,False,False,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,False,True,True,True,True,True,True,True,True],
    [True,True,False,False,False,True,True,False,False,False,False,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,False,True,True,True,True,True,True,True,True],
    [True,False,False,False,False,True,True,False,False,False,False,True,True,True,True,False,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,False,True,True,True,True,True,True,True,True],
    [True,False,False,False,False,True,True,False,False,False,False,False,True,True,True,False,False,True,True,True,True,True,True,True,True,True,True,True,True,True,False,False,True,True,True,True,True,True,True,True],
    [False,False,False,False,False,True,True,False,False,False,False,False,True,True,True,True,False,False,True,True,True,True,True,True,True,True,True,True,True,True,False,False,False,True,True,True,True,True,True,True],
    [False,False,False,False,False,True,True,False,False,False,False,False,False,True,True,True,True,False,True,True,True,True,True,True,True,True,True,True,True,True,False,False,False,True,True,True,True,True,True,True],
    [False,False,False,False,False,True,True,True,False,False,False,False,False,True,True,True,True,True,False,True,True,True,True,True,True,True,True,True,True,True,False,False,False,True,True,True,True,True,True,True],
    [True,True,True,False,False,True,True,True,False,False,False,False,False,False,True,True,True,True,True,False,True,True,True,True,True,True,True,True,True,True,False,False,False,False,True,True,True,True,True,True],
    [True,True,True,False,False,True,True,True,False,False,False,False,False,False,True,True,True,True,True,True,False,True,True,True,True,True,True,True,True,True,True,False,False,False,True,True,True,True,True,True]
], dtype=bool)

MASK_LH_Y = np.array([
    [False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,True,False,False,True,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,True,False,False],
    [False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,True,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,True,False,False,False,False,False,False,True,True,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,True,False,False,False,False,False,False,False],
    [False,False,False,False,True,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,True,True,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,True,True,True,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,True,True,True,False],
    [False,True,True,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,True,True,True],
    [False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,True,True],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,True],
    [False,False,False,False,False,False,False,False,False,False,False,True,True,True,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,True],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,True,True],
    [False,False,False,False,False,False,False,False,False,False,False,False,True,True,False,False,False,False,False,False,False,False,False,True,True,True,False,False,False,False,False,False,False,False,False,False,False,False,True,True],
    [False,False,False,False,False,False,False,False,False,False,False,True,True,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,True],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,True,False,False,True,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,True,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True],
    [True,True,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,True,False,False,True,True,False,False],
    [False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,True,True,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,True,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,True,True,True,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,True,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,True,True,True,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False]
], dtype=bool)

MASK_HL_Y = np.array([
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False],
    [False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,True,False,False,False,False,False,False],
    [False,False,False,True,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,True,False,False,False],
    [True,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,True,False,False,False,False],
    [False,False,True,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,True,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False],
    [False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,True,False,False,False,False,False,False],
    [False,False,False,False,False,True,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,True,False,True,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,True,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False],
    [False,False,False,True,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False]
], dtype=bool)

MASK_HH_Y = np.array([
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,True,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False]
], dtype=bool)

# Masks for chroma (Cb/Cr) planes: grid 48x40 -> DWT 24x20 subbands -> keep 480/1920
MASK_LL_C = np.array([
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,True,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,True,True,True,True],
    [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,True,True,True,True],
    [True,True,True,True,False,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True],
    [True,False,True,True,False,True,True,True,True,True,True,True,True,True,True,False,True,True,True,True],
    [True,True,False,True,False,True,True,True,True,True,True,True,True,True,True,False,True,True,True,True],
    [True,False,False,True,False,False,True,True,True,True,True,True,True,True,True,False,True,True,True,True],
    [True,False,False,True,False,False,True,True,True,True,True,True,True,True,True,False,True,True,True,True],
    [False,True,False,True,False,False,True,True,True,True,True,True,True,True,True,False,False,True,True,True],
    [True,True,False,True,False,False,False,True,True,True,True,True,True,True,True,False,False,True,True,True]
], dtype=bool)

MASK_LH_C = np.array([
    [False,False,False,False,False,True,False,False,False,False,False,False,True,False,False,False,False,False,False,False],
    [False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,True,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,True],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,True],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,True],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,True,False,True,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,True,True],
    [False,True,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False]
], dtype=bool)

MASK_HL_C = np.array([
    [False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False],
    [False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,True,True,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,True,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False]
], dtype=bool)

MASK_HH_C = np.array([
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False]
], dtype=bool)

# Total: Y=1920 (1752 LL + 108 LH + 57 HL + 3 HH), C=480 (446 LL + 20 LH + 14 HL + 0 HH)

def haar_level(plane):
    """One level of the orthonormal 2-D Haar DWT.

    Returns ``(LL, LH, HL, HH)``, each half the input's rows and columns.
    ``plane`` must have even dimensions.
    """
    low = (plane[:, 0::2] + plane[:, 1::2]) / SQRT2
    high = (plane[:, 0::2] - plane[:, 1::2]) / SQRT2
    ll = (low[0::2, :] + low[1::2, :]) / SQRT2
    lh = (low[0::2, :] - low[1::2, :]) / SQRT2
    hl = (high[0::2, :] + high[1::2, :]) / SQRT2
    hh = (high[0::2, :] - high[1::2, :]) / SQRT2
    return ll, lh, hl, hh


def ihaar_level(ll, lh, hl, hh):
    """Inverse of :func:`haar_level`."""
    low_even = (ll + lh) / SQRT2
    low_odd = (ll - lh) / SQRT2
    high_even = (hl + hh) / SQRT2
    high_odd = (hl - hh) / SQRT2

    low = np.empty((2 * low_even.shape[0], low_even.shape[1]), dtype=low_even.dtype)
    low[0::2] = low_even
    low[1::2] = low_odd
    high = np.empty_like(low)
    high[0::2] = high_even
    high[1::2] = high_odd

    even = (low + high) / SQRT2
    odd = (low - high) / SQRT2
    out = np.empty((even.shape[0], 2 * even.shape[1]), dtype=even.dtype)
    out[:, 0::2] = even
    out[:, 1::2] = odd
    return out


def _levels(grid_rows, grid_cols, rows, cols):
    """How many DWT levels make the coarsest LL land on ``rows x cols``.

    The standard profiles halve each axis once (grid is 2x the wire shape), so
    this is 1. It is a hard requirement, not a hint: a grid that does not land
    exactly on the shape is a coder bug.
    """
    lv = 0
    while (grid_rows, grid_cols) != (rows, cols):
        if grid_rows % 2 or grid_cols % 2 or grid_rows // 2 < rows or grid_cols // 2 < cols:
            raise ValueError('Grid does not halve cleanly to the wire shape')
        grid_rows, grid_cols = grid_rows // 2, grid_cols // 2
        lv += 1
    return lv


def _get_masks(grid_rows, grid_cols):
    """Return the correct mask set for the given grid size."""
    if grid_rows == 96 and grid_cols == 80:
        return MASK_LL_Y, MASK_LH_Y, MASK_HL_Y, MASK_HH_Y
    elif grid_rows == 48 and grid_cols == 40:
        return MASK_LL_C, MASK_LH_C, MASK_HL_C, MASK_HH_C
    else:
        raise ValueError(f'Unknown grid size: {grid_rows}x{grid_cols}')


def haar_level(plane):
    """One level of the orthonormal 2-D Haar DWT.

    Returns ``(LL, LH, HL, HH)``, each half the input's rows and columns.
    ``plane`` must have even dimensions.
    """
    low = (plane[:, 0::2] + plane[:, 1::2]) / SQRT2
    high = (plane[:, 0::2] - plane[:, 1::2]) / SQRT2
    ll = (low[0::2, :] + low[1::2, :]) / SQRT2
    lh = (low[0::2, :] - low[1::2, :]) / SQRT2
    hl = (high[0::2, :] + high[1::2, :]) / SQRT2
    hh = (high[0::2, :] - high[1::2, :]) / SQRT2
    return ll, lh, hl, hh


def ihaar_level(ll, lh, hl, hh):
    """Inverse of :func:`haar_level`."""
    low_even = (ll + lh) / SQRT2
    low_odd = (ll - lh) / SQRT2
    high_even = (hl + hh) / SQRT2
    high_odd = (hl - hh) / SQRT2

    low = np.empty((2 * low_even.shape[0], low_even.shape[1]), dtype=low_even.dtype)
    low[0::2] = low_even
    low[1::2] = low_odd
    high = np.empty_like(low)
    high[0::2] = high_even
    high[1::2] = high_odd

    even = (low + high) / SQRT2
    odd = (low - high) / SQRT2
    out = np.empty((even.shape[0], 2 * even.shape[1]), dtype=even.dtype)
    out[:, 0::2] = even
    out[:, 1::2] = odd
    return out


def _levels(grid_rows, grid_cols, rows, cols):
    """How many DWT levels make the coarsest LL land on ``rows x cols``.

    The standard profiles halve each axis once (grid is 2x the wire shape), so
    this is 1. It is a hard requirement, not a hint: a grid that does not land
    exactly on the shape is a coder bug.
    """
    lv = 0
    while (grid_rows, grid_cols) != (rows, cols):
        if grid_rows % 2 or grid_cols % 2 or grid_rows // 2 < rows or grid_cols // 2 < cols:
            raise ValueError('Grid does not halve cleanly to the wire shape')
        grid_rows, grid_cols = grid_rows // 2, grid_cols // 2
        lv += 1
    return lv


def _apply_mask_and_pack(ll, lh, hl, hh, mask_ll, mask_lh, mask_hl, mask_hh, rows, cols):
    """Apply masks and pack coefficients into wire shape."""
    ll = ll * mask_ll
    lh = lh * mask_lh
    hl = hl * mask_hl
    hh = hh * mask_hh
    out = np.empty((rows, cols), dtype=ll.dtype)
    idx = 0
    for arr, mask in [(ll, mask_ll), (lh, mask_lh), (hl, mask_hl), (hh, mask_hh)]:
        vals = arr[mask]
        n = len(vals)
        out.ravel()[idx:idx+n] = vals
        idx += n
    return out


def _unpack_and_apply_mask(corner, rows, cols, grid, mask_ll, mask_lh, mask_hl, mask_hh):
    """Unpack coefficients and apply masks for inverse transform."""
    flat = np.asarray(corner).ravel()
    idx = 0
    ll = np.zeros_like(mask_ll, dtype=corner.dtype)
    lh = np.zeros_like(mask_lh, dtype=corner.dtype)
    hl = np.zeros_like(mask_hl, dtype=corner.dtype)
    hh = np.zeros_like(mask_hh, dtype=corner.dtype)
    for arr, mask in [(ll, mask_ll), (lh, mask_lh), (hl, mask_hl), (hh, mask_hh)]:
        n = int(mask.sum())
        arr[mask] = flat[idx:idx+n]
        idx += n
    return ihaar_level(ll, lh, hl, hh)


class WaveletCoder(SourceCoder):
    """A ``SourceCoder`` that replaces the DCT with a 2-D Haar DWT.

    The transform is a full 1-level DWT with measured fixed masks keeping
    top-N coefficients by average magnitude across the bake (1/4 keep ratio).
    """
    def _forward_transform(self, plane, rows, cols):
        ll, lh, hl, hh = haar_level(plane)
        # Get correct masks for this plane's grid size
        grid_rows, grid_cols = plane.shape[0], plane.shape[1]
        mask_ll, mask_lh, mask_hl, mask_hh = _get_masks(grid_rows, grid_cols)
        return _apply_mask_and_pack(ll, lh, hl, hh, mask_ll, mask_lh, mask_hl, mask_hh, rows, cols)

    def _inverse_transform(self, corner, rows, cols, grid):
        mask_ll, mask_lh, mask_hl, mask_hh = _get_masks(grid[0], grid[1])
        return _unpack_and_apply_mask(corner, rows, cols, grid, mask_ll, mask_lh, mask_hl, mask_hh)


# --------------------------------------------------------------------------
# CDF 9/7 Wavelet (JPEG2000) - Lifting Scheme
# --------------------------------------------------------------------------
# Lifting coefficients for CDF 9/7 (Daubechies 9/7)
# From JPEG2000 Part 1, Annex A
_CDF_ALPHA = -1.586134342059924
_CDF_BETA  = -0.052980118572961
_CDF_GAMMA =  0.882911075530934
_CDF_DELTA =  0.443506852043971
_CDF_K     =  1.230174104914001
_CDF_KINV  =  1.0 / _CDF_K

def _cdf97_predict_lift(odd, even, coeff):
    """Predict step: odd[k] += coeff*(even[k] + even[k+1]).

    The lifted pair must MIX the two polyphase components -- odd[k]'s
    neighbours in the interleaved signal are even[k] and even[k+1].

    Symmetric boundary (JPEG2000 whole-sample symmetric extension): past the
    last sample the signal mirrors, so the missing even[k+1] of the last odd
    sample is even[k] itself. The periodic boundary this replaces wrapped the
    frame's right edge into its left: every edge coefficient mixed opposite
    sides of the picture, and a truncated pyramid smeared that across the
    border -- measured 8 dB worse than the DCT on a smooth frame, 0.7 dB with
    this. Still exactly invertible: the inverse lifts with the same rule.
    Works along the LAST axis, so a 2-D array lifts every row at once.
    """
    nxt = np.concatenate([even[..., 1:], even[..., -1:]], axis=-1)
    return odd + coeff * (even + nxt)

def _cdf97_update_lift(even, odd, coeff):
    """Update step: even[k] += coeff*(odd[k-1] + odd[k]).

    Mirror of :func:`_cdf97_predict_lift`, same symmetric boundary: before
    the first sample odd[-1] reflects to odd[0]. Last axis, like predict.
    """
    prv = np.concatenate([odd[..., :1], odd[..., :-1]], axis=-1)
    return even + coeff * (prv + odd)

def cdf97_forward_1d(x, axis=-1):
    """1D CDF 9/7 forward transform using lifting (periodic boundary).

    Returns (low, high) each half the length along `axis`. Any other axes are
    transformed independently in one vectorised pass -- the 2-D transform
    below relies on that. The per-row Python loop it replaces cost ~25 ms a
    frame for hd-dwt, more than the sender's 10 ms prepare lead, so every live
    packet missed its deadline and nothing was ever sent.
    """
    x = np.moveaxis(np.asarray(x, float), axis, -1)
    n = x.shape[-1]
    if n % 2 != 0:
        raise ValueError("Length must be even")
    even = x[..., ::2]
    odd = x[..., 1::2]

    # Predict 1
    odd = _cdf97_predict_lift(odd, even, _CDF_ALPHA)
    # Update 1
    even = _cdf97_update_lift(even, odd, _CDF_BETA)
    # Predict 2
    odd = _cdf97_predict_lift(odd, even, _CDF_GAMMA)
    # Update 2
    even = _cdf97_update_lift(even, odd, _CDF_DELTA)

    # Scale
    low = even * _CDF_K
    high = odd * _CDF_KINV
    return np.moveaxis(low, -1, axis), np.moveaxis(high, -1, axis)

def cdf97_inverse_1d(low, high, axis=-1):
    """1D CDF 9/7 inverse transform using lifting, along `axis`."""
    low = np.moveaxis(np.asarray(low), axis, -1)
    high = np.moveaxis(np.asarray(high), axis, -1)
    # Inverse scale
    even = low * _CDF_KINV
    odd = high * _CDF_K

    # Inverse Update 2
    even = _cdf97_update_lift(even, odd, -_CDF_DELTA)
    # Inverse Predict 2
    odd = _cdf97_predict_lift(odd, even, -_CDF_GAMMA)
    # Inverse Update 1
    even = _cdf97_update_lift(even, odd, -_CDF_BETA)
    # Inverse Predict 1
    odd = _cdf97_predict_lift(odd, even, -_CDF_ALPHA)

    # Interleave
    x = np.empty(even.shape[:-1] + (even.shape[-1] * 2,), dtype=even.dtype)
    x[..., ::2] = even
    x[..., 1::2] = odd
    return np.moveaxis(x, -1, axis)

def cdf97_forward_2d(plane, levels=3):
    """Multi-level 2D CDF 9/7 forward transform.

    Returns dict with subbands: {level: {LL, LH, HL, HH}}. Rows, then the
    columns of each half, each as one vectorised 1-D pass.
    """
    current = np.asarray(plane, float)
    subbands = {}
    for level in range(levels):
        if current.shape[0] < 2 or current.shape[1] < 2:
            break
        # Transform rows
        rows_low, rows_high = cdf97_forward_1d(current, axis=1)
        # Transform columns of low, then of high
        cols_low, cols_high = cdf97_forward_1d(rows_low, axis=0)
        cols_low_h, cols_high_h = cdf97_forward_1d(rows_high, axis=0)

        subbands[level] = {
            'LL': cols_low,
            'HL': cols_high,   # horizontal detail
            'LH': cols_low_h,  # vertical detail
            'HH': cols_high_h  # diagonal detail
        }
        current = cols_low
    return subbands

def cdf97_inverse_2d(subbands, levels=3, out_shape=None):
    """Multi-level 2D CDF 9/7 inverse transform.

    Reconstructs from a subbands dict. Each level's LL is the RECONSTRUCTED
    output of the coarser level ('current'), never a transmitted LL -- only
    the coarsest level supplies one. That is the pyramid identity: a
    non-redundant dict (deeper LL replaces shallower LL) inverts exactly like
    a full one, and detail bands missing from the wire zero-fill as
    (soft, low-passed) content instead of collapsing the output.
    """
    deepest = max(subbands)  # coarsest present level (forward may early-out)
    if out_shape is None:
        ll = subbands[deepest]['LL']
        h = ll.shape[0] * (2 ** (deepest + 1))
        w = ll.shape[1] * (2 ** (deepest + 1))
    else:
        h, w = out_shape

    current = subbands[deepest]['LL']
    for level in range(deepest, -1, -1):
        sb = subbands[level]
        # Inverse transform columns: low side is 'current' (this level's LL),
        # high side is the transmitted HL/LH/HH details.
        rows_low = cdf97_inverse_1d(current, sb['HL'], axis=0)
        rows_high = cdf97_inverse_1d(sb['LH'], sb['HH'], axis=0)
        # Inverse transform rows
        current = cdf97_inverse_1d(rows_low, rows_high, axis=1)
    return current[:h, :w]


def _cdf97_subband_dims(grid_rows, grid_cols, levels):
    """Subband size at each DWT depth, matching ``cdf97_forward_2d``.

    dims[0] is the finest level's subband size (grid halved once), dims[k]
    the size after k+1 halvings. The loop mirrors forward's early-out when a
    side would drop below 2 samples.
    """
    dims = []
    rows, cols = grid_rows, grid_cols
    for _ in range(levels):
        if rows < 2 or cols < 2:
            break
        rows, cols = rows // 2, cols // 2
        dims.append((rows, cols))
    return dims


def _cdf97_pack_order(dims):
    """Coefficient order shared by forward and inverse.

    Non-redundant pyramid, JPEG2000-style: the deepest level carries all four
    bands, every shallower level carries only its three detail bands (its LL
    is represented by the deeper levels). Packed that way the total is exactly
    the plane size and a full pyramid round-trips losslessly; a truncated set
    zero-fills whatever the wire could not carry. The inverse must unpack in
    exactly this order or the subbands swap.
    """
    order = []
    deepest = len(dims) - 1
    for level in range(deepest, -1, -1):
        rows, cols = dims[level]
        bands = ('LL', 'LH', 'HL', 'HH') if level == deepest else ('LH', 'HL', 'HH')
        for band in bands:
            order.append((level, band, rows, cols))
    return order


# Per-band RMS of the packed pyramid, per plane (Y, Cb, Cr), for the value
# scale image_values produces ([-1, 1]). Measured on 48 synthetic frames with
# a natural-image 1/f spectrum (seed 7) at the baked 80x96 grid, 2 levels,
# symmetric boundary; test_hd_dwt re-measures it. The bake is fixed, so the
# table is a constant and costs no header bits -- the same bargain as
# default_allocation, but for the wavelet's actual bands. It sets each
# value's transmit power (sqrt of its RMS, the analog optimum) and is the
# Wiener prior on decode.
HD_BAND_RMS = (
    {'LL': 1.80, 'LH': .457, 'HL': .457, 'HH': .303},
    {'LL': .871, 'LH': .093, 'HL': .093, 'HH': .058},
    {'LL': .891, 'LH': .135, 'HL': .135, 'HH': .085},
)
# Carrier the most important values should ride: low-mid, where tape keeps
# its response (~3.75 kHz at 375 Hz/bin). The bottom two bins are avoided
# (rumble, the guard carrier) rather than forbidden.
TAPE_CENTRE_BIN = 10


class Cdf97Coder(SourceCoder):
    """v5 source coder: 2-level CDF 9/7 on the baked grid, protected by repeats.

    What rides the wire, per plane: the coarsest level's four bands and the
    finest level's detail is dropped. That is the whole half-resolution
    pyramid -- luma 1920 values, each chroma 480, 2880 in all -- cut at a
    band boundary. (Cutting mid-band, as v5 first did, spent the leftover on
    the first rows of one detail band: a sharp strip across the top of the
    frame and nothing anywhere else.)

    The wire holds `capacity` values (3680 on WIRE_HD). The 800 left over do
    NOT buy resolution -- 800 more values cannot come close to a full 80x96
    pyramid, 7680 luma values alone. They buy survival: the highest-energy
    values (every LL coefficient of all three planes, then the strongest
    detail) are sent twice. Each copy rides the OTHER stereo channel, at
    least MIN_COPY_SPREAD carriers away from its original, preferring the
    low-mid band tape keeps. The decoder combines the two with the
    equaliser's per-slot reliability (maximum-ratio, then Wiener), so a copy
    on a dead carrier costs nothing and one on a live carrier rescues it.

    Measured against v3 on the same synthetic frames (tools/compare_codecs.py;
    run it with --modem-dir for real content): -0.7 dB on a perfect channel;
    +4.0 dB cassette (10 kHz, -45 dB); +3.1 dB at -30 dB
    hiss; +6.9 dB through a 6 kHz lowpass; +3.4 dB with one stereo leg dead
    (v3 gained most there from the erasure fix in decode_packet); +3.3 dB
    under a 1 kHz bass cut; +6.4 dB clipped at half scale.
    """
    MIN_COPY_SPREAD = 10

    def __init__(self, grids, capacity, levels=2):
        self.levels = int(levels)
        self.grids = [tuple(g) for g in grids]
        step = 2**self.levels
        if any(r % step or c % step for r, c in self.grids):
            raise ValueError(f'Every grid must divide by {step} for {levels} levels')
        # The kept region of each plane is its coarsest-level pyramid, which
        # packs to exactly (rows/2)x(cols/2) values: reported as `shapes`.
        self.shapes = [(r//2, c//2) for r, c in self.grids]
        self.keep = [r*c//4 for r, c in self.grids]
        self.n_orig = sum(self.keep)
        self.source_count = sum(r*c for r, c in self.grids)
        self.truncated = True
        if capacity < self.n_orig:
            raise ValueError(f'{capacity} slots cannot hold the {self.n_orig}-value '
                             'half-resolution pyramid')
        tables = HD_BAND_RMS if len(self.grids) == 3 else HD_BAND_RMS[:1]
        sigma = np.concatenate([
            [tables[p][band] for _, band, rows, cols in
             _cdf97_pack_order(_cdf97_subband_dims(r, c, self.levels))
             for _ in range(rows*cols)][:k]
            for p, ((r, c), k) in enumerate(zip(self.grids, self.keep))])
        # Importance rank: position in the plane's packed order over the
        # plane's size, so every plane's LL precedes every plane's detail.
        self.rank = np.concatenate([np.arange(k)/(r*c)
                                    for (r, c), k in zip(self.grids, self.keep)])
        spare = int(capacity) - self.n_orig
        order = np.lexsort((self.rank, -sigma))       # strongest first
        self.copy_of = np.sort(order[:spare])
        self.count = self.n_orig + len(self.copy_of)
        full = np.concatenate([sigma, sigma[self.copy_of]])
        gains = np.sqrt(full/np.mean(full))
        self.gains = gains/np.sqrt(np.mean(gains**2))
        self.variance = sigma**2
        # Posterior-confidence floors for tape recovery. Luma LL carries the
        # broad brightness and shape, so it survives at very low confidence.
        # Detail needs stronger evidence, and false chroma is more objectionable
        # than missing chroma detail. The gate acts after repeat fusion and does
        # not alter coefficients whose posterior confidence is at least .85.
        floors = []
        for plane, ((rows, cols), k) in enumerate(zip(self.grids, self.keep)):
            ll = (rows >> self.levels) * (cols >> self.levels)
            if plane == 0:
                floors.extend([.05]*ll + [.45]*(k-ll))
            else:
                floors.extend([.25]*ll + [.65]*(k-ll))
        self.recovery_floor = np.asarray(floors)
        self._slots = {}

    def recovery_gate(self, confidence):
        """Luma-first coefficient gate for a noisy tape posterior.

        Below each band's floor the observation is an erasure. Between the
        floor and .85 it fades in; above .85 it is untouched. This preserves
        the current wire and turns uncertain detail into softness instead of
        synthesizing high-resolution noise.
        """
        confidence = np.asarray(confidence, float)
        return np.clip((confidence - self.recovery_floor) /
                       (.85 - self.recovery_floor), 0.0, 1.0)

    def forward(self, values):
        """Pixels in (source_count), wire values out (count): originals, then copies."""
        planes = self._split(np.asarray(values, float), self.grids)
        kept = np.concatenate([self._pack(plane)[:k]
                               for plane, k in zip(planes, self.keep)])
        return np.concatenate([kept, kept[self.copy_of]])*self.gains

    def inverse(self, sent, reliability=None, noise_variance=None):
        """Wire values in, full-grid pixels out; each repeat combined with its original.

        With reliability: the slot arrives as y = gain*rel*x + n. Stacking the
        original and its copy, the MMSE estimate of x under the prior variance
        is var*sum(a*y/n) / (1 + var*sum(a^2/n)), a = gain*rel -- a single
        observation reduces to SourceCoder's Wiener exactly. Without it, the
        copies are simply averaged.
        """
        y = np.asarray(sent, float)
        n0, cp = self.n_orig, self.copy_of
        if reliability is None:
            x = y/self.gains
            kept = x[:n0].copy()
            kept[cp] = (kept[cp] + x[n0:])/2
        else:
            a = self.gains*np.maximum(np.asarray(reliability, float), 1e-6)
            nv = np.maximum(np.asarray(noise_variance, float), 1e-20)
            num, den = a*y/nv, a*a/nv
            top, bottom = num[:n0].copy(), den[:n0].copy()
            np.add.at(top, cp, num[n0:])
            np.add.at(bottom, cp, den[n0:])
            kept = self.variance*top/(1 + self.variance*bottom)
            confidence = self.variance*bottom/(1 + self.variance*bottom)
            kept *= self.recovery_gate(confidence)
        out, offset = [], 0
        for (rows, cols), k in zip(self.grids, self.keep):
            flat = np.zeros(rows*cols)
            flat[:k] = kept[offset:offset+k]
            offset += k
            out.append(self._unpack(flat, rows, cols).ravel())
        return np.concatenate(out)

    def slots(self, layout):
        """Wire slot of every value (core.coder_slots calls this).

        Originals, most important first, take the image slots nearest the
        tape centre carrier. Copies then go, strongest first, to the free slot
        on the other channel, far enough in frequency from their original,
        nearest the tape centre. Deterministic, so both ends agree; cached.
        """
        if layout in self._slots:
            return self._slots[layout]
        if self.count > layout.capacity:
            raise ValueError(f'{self.count} values exceed {layout.name} capacity '
                             f'{layout.capacity}')
        bins, chans = _slot_carriers(layout)
        order = _slot_order(layout)
        mapping = np.empty(self.count, int)
        mine = np.sort(order[:self.n_orig])
        health = np.abs(bins[mine] - TAPE_CENTRE_BIN) + 40*(bins[mine] < 3)
        mapping[np.argsort(self.rank, kind='stable')] = \
            mine[np.argsort(health, kind='stable')]
        pool = np.sort(order[self.n_orig:self.count])
        health = np.abs(bins[pool] - TAPE_CENTRE_BIN) + 40*(bins[pool] < 3)
        taken = np.zeros(len(pool), bool)
        for j in np.lexsort((self.rank[self.copy_of], -self.variance[self.copy_of])):
            home = mapping[self.copy_of[j]]
            score = (health + 100*(chans[pool] == chans[home])
                     + 100*(np.abs(bins[pool] - bins[home]) < self.MIN_COPY_SPREAD)
                     + 1e6*taken)
            pick = int(np.argmin(score))
            taken[pick] = True
            mapping[self.n_orig + j] = pool[pick]
        mapping.setflags(write=False)
        self._slots[layout] = mapping
        return mapping

    def _pack(self, plane):
        subbands = cdf97_forward_2d(plane, self.levels)
        order = _cdf97_pack_order(_cdf97_subband_dims(*plane.shape, self.levels))
        return np.concatenate([subbands[level][band].ravel()
                               for level, band, _, _ in order])

    def _unpack(self, flat, rows, cols):
        order = _cdf97_pack_order(_cdf97_subband_dims(rows, cols, self.levels))
        subbands, idx = {}, 0
        for level, band, sr, sc in order:
            subbands.setdefault(level, {})[band] = flat[idx:idx+sr*sc].reshape(sr, sc)
            idx += sr*sc
        return cdf97_inverse_2d(subbands, self.levels, out_shape=(rows, cols))


class StereoRepeatCoder(SourceCoder):
    """DCT coder whose complete compact picture is repeated on the other leg."""

    MIN_COPY_SPREAD = 8

    def __init__(self, shapes, grids):
        super().__init__(shapes, grids=grids)
        self.n_orig = self.count
        self.base_gains = self.gains.copy()
        self.copy_of = np.arange(self.n_orig)
        self.count = 2*self.n_orig
        self.gains = np.concatenate([self.base_gains, self.base_gains])
        self._slots = {}

    def forward(self, values):
        planes = self._split(np.asarray(values, float), self.grids)
        kept = [self._forward_transform(plane, rows, cols).ravel()
                for plane, (rows, cols) in zip(planes, self.shapes)]
        base = np.concatenate(kept)*self.base_gains
        return np.concatenate([base, base])

    def inverse(self, sent, reliability=None, noise_variance=None):
        y = np.asarray(sent, float)
        if reliability is None:
            x = y/self.gains
            kept = (x[:self.n_orig] + x[self.n_orig:])/2
        else:
            a = self.gains*np.maximum(np.asarray(reliability, float), 1e-6)
            nv = np.maximum(np.asarray(noise_variance, float), 1e-20)
            num = a*y/nv
            den = a*a/nv
            top = num[:self.n_orig] + num[self.n_orig:]
            bottom = den[:self.n_orig] + den[self.n_orig:]
            prior = self.variance
            kept = prior*top/(1 + prior*bottom)
        out = []
        for corner, (rows, cols), grid in zip(
                self._split(kept, self.shapes), self.shapes, self.grids):
            out.append(self._inverse_transform(corner, rows, cols, grid).ravel())
        return np.concatenate(out)

    def slots(self, layout):
        if layout in self._slots:
            return self._slots[layout]
        if self.count > layout.capacity:
            raise ValueError(f'{self.count} values exceed {layout.name} capacity')
        bins, chans = _slot_carriers(layout)
        order = _slot_order(layout)
        by_channel = [order[chans[order] == channel] for channel in (0, 1)]
        per_leg = self.n_orig//2
        # Alternate coefficient homes across legs so luma and chroma are not
        # concentrated on one track. Leave enough opposite-leg slots for every
        # copy before considering the layout's unused tail.
        originals = np.column_stack([by_channel[0][:per_leg],
                                     by_channel[1][:per_leg]]).ravel()
        pools = [by_channel[0][per_leg:], by_channel[1][per_leg:]]
        mapping = np.empty(self.count, int)
        mapping[:self.n_orig] = originals
        # Solve each cross-leg assignment globally. A greedy nearest-free pick
        # strands edge carriers at the end and forces a few adjacent copies even
        # though a fully frequency-diverse matching exists.
        for other, pool in enumerate(pools):
            home_indices = np.flatnonzero(chans[originals] != other)
            home = originals[home_indices]
            distance = np.abs(bins[home, None]-bins[pool][None, :])
            cost = (1e6*(distance < self.MIN_COPY_SPREAD) +
                    np.abs(bins[pool]-TAPE_CENTRE_BIN)[None, :])
            rows, columns = linear_sum_assignment(cost)
            mapping[self.n_orig+home_indices[rows]] = pool[columns]
        mapping.setflags(write=False)
        self._slots[layout] = mapping
        return mapping


def tape_80x60_coder():
    return StereoRepeatCoder([(15, 20), (5, 6), (5, 6)],
                             [(60, 80), (30, 40), (30, 40)])


def tape_80x96_coder():
    coder = StereoRepeatCoder([(28, 24), (8, 8), (8, 8)],
                              [(96, 80), (48, 40), (48, 40)])
    coder.profile_name = 'tape-80x96'
    return coder


def _slot_carriers(layout):
    """Carrier bin and stereo channel of every flat wire slot.

    Mirrors the flat layout encode/decode_packet use: the header symbols'
    spare carriers first, then the image symbols, each cell four values
    (channel 0 re/im, channel 1 re/im).
    """
    cells = []
    if layout.dense_header:
        cells += [layout.spare_bins]*layout.header_symbols
    cells += [layout.data_bins]*layout.image_symbols
    bins = np.repeat(np.concatenate(cells), 4)
    chans = np.tile(np.array([0, 0, 1, 1]), len(bins)//4)
    return bins, chans


def hd_dwt_coder():
    """The one v5 coder. Every sender, receiver and tool builds it here."""
    from .imaging import HD_MONO_CAPACITY, plane_grids
    return Cdf97Coder(plane_grids('hd-dwt'), HD_MONO_CAPACITY, levels=2)
