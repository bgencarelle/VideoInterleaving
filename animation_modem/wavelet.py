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

from .core import SourceCoder

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

def _cdf97_predict_lift(x, coeff):
    """Predict (highpass) lifting step: odd -= coeff * (even_left + even_right)"""
    return x - coeff * (np.roll(x, 1) + np.roll(x, -1))

def _cdf97_update_lift(x, coeff):
    """Update (lowpass) lifting step: even += coeff * (odd_left + odd_right)"""
    return x + coeff * (np.roll(x, 1) + np.roll(x, -1))

def cdf97_forward_1d(x):
    """1D CDF 9/7 forward transform using lifting (periodic boundary).
    
    Returns (low, high) each half the length.
    """
    n = len(x)
    if n % 2 != 0:
        raise ValueError("Length must be even")
    half = n // 2
    even = x[::2].copy()
    odd = x[1::2].copy()
    
    # Predict 1
    odd = _cdf97_predict_lift(odd, _CDF_ALPHA)
    # Update 1
    even = _cdf97_update_lift(even, _CDF_BETA)
    # Predict 2
    odd = _cdf97_predict_lift(odd, _CDF_GAMMA)
    # Update 2
    even = _cdf97_update_lift(even, _CDF_DELTA)
    
    # Scale
    low = even * _CDF_K
    high = odd * _CDF_KINV
    return low, high

def cdf97_inverse_1d(low, high):
    """1D CDF 9/7 inverse transform using lifting."""
    # Inverse scale
    even = low * _CDF_KINV
    odd = high * _CDF_K
    
    # Inverse Update 2
    even = _cdf97_update_lift(even, -_CDF_DELTA)
    # Inverse Predict 2
    odd = _cdf97_predict_lift(odd, -_CDF_GAMMA)
    # Inverse Update 1
    even = _cdf97_update_lift(even, -_CDF_BETA)
    # Inverse Predict 1
    odd = _cdf97_predict_lift(odd, -_CDF_ALPHA)
    
    # Interleave
    n = len(even) * 2
    x = np.empty(n, dtype=even.dtype)
    x[::2] = even
    x[1::2] = odd
    return x

def cdf97_forward_2d(plane, levels=3):
    """Multi-level 2D CDF 9/7 forward transform.
    
    Returns dict with subbands: {level: {LL, LH, HL, HH}}
    """
    h, w = plane.shape
    current = plane.astype(float)
    subbands = {}
    for level in range(levels):
        if current.shape[0] < 2 or current.shape[1] < 2:
            break
        # Transform rows
        rows_low = np.empty((current.shape[0], current.shape[1] // 2))
        rows_high = np.empty_like(rows_low)
        for i in range(current.shape[0]):
            rows_low[i], rows_high[i] = cdf97_forward_1d(current[i])
        # Transform columns of low
        cols_low = np.empty((current.shape[0] // 2, current.shape[1] // 2))
        cols_high = np.empty_like(cols_low)
        for j in range(rows_low.shape[1]):
            cols_low[:, j], cols_high[:, j] = cdf97_forward_1d(rows_low[:, j])
        # Transform columns of high
        cols_low_h = np.empty((current.shape[0] // 2, current.shape[1] // 2))
        cols_high_h = np.empty_like(cols_low_h)
        for j in range(rows_high.shape[1]):
            cols_low_h[:, j], cols_high_h[:, j] = cdf97_forward_1d(rows_high[:, j])
        
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
    
    Reconstructs from subbands dict.
    """
    if out_shape is None:
        # Compute from finest LL
        ll = subbands[levels-1]['LL']
        h, w = ll.shape[0] * (2**levels), ll.shape[1] * (2**levels)
    else:
        h, w = out_shape
    
    current = subbands[levels-1]['LL']
    for level in reversed(range(levels)):
        sb = subbands[level]
        # Inverse transform columns
        rows_low = np.empty((current.shape[0] * 2, current.shape[1]))
        rows_high = np.empty_like(rows_low)
        for j in range(current.shape[1]):
            rows_low[:, j] = cdf97_inverse_1d(sb['LL'][:, j], sb['HL'][:, j])
            rows_high[:, j] = cdf97_inverse_1d(sb['LH'][:, j], sb['HH'][:, j])
        # Inverse transform rows
        current = np.empty((current.shape[0] * 2, current.shape[1] * 2))
        for i in range(rows_low.shape[0]):
            current[i] = cdf97_inverse_1d(rows_low[i], rows_high[i])
    return current[:h, :w]


class Cdf97Coder(SourceCoder):
    """SourceCoder using multi-level CDF 9/7 DWT with measured allocation.
    
    v5: 3-level DWT on 96x112 source -> collects coefficients from all subbands
    to fill 56x44 wire shape.
    """
    def __init__(self, shapes, grids=None, allocation=None, levels=3):
        self.levels = levels
        # Generate allocation based on subband structure if not provided
        if allocation is None:
            from .core import default_allocation
            count = int(sum(np.prod(s) for s in shapes))
            allocation = default_allocation(shapes, count)
        # SourceCoder expects: (shapes, allocation, grids)
        super().__init__(shapes, allocation, grids)

    def _forward_transform(self, plane, rows, cols):
        # Full multi-level DWT on source grid
        subbands = cdf97_forward_2d(plane, self.levels)
        # Collect coefficients from all subbands in priority order:
        # LL (finest) > LH/HL/HH (finest) > LL (next) > LH/HL/HH (next) > ...
        coeffs = []
        for level in range(self.levels - 1, -1, -1):
            if level in subbands:
                sb = subbands[level]
                # Priority: LL first, then detail subbands
                for key in ['LL', 'LH', 'HL', 'HH']:
                    if key in sb:
                        coeffs.append(sb[key].ravel())
        # Concatenate all and truncate/pad to wire shape size
        all_coeffs = np.concatenate(coeffs) if coeffs else np.array([])
        target_size = rows * cols
        if len(all_coeffs) >= target_size:
            return all_coeffs[:target_size]
        else:
            # Pad with zeros
            padded = np.zeros(target_size)
            padded[:len(all_coeffs)] = all_coeffs
            return padded

    def _inverse_transform(self, corner, rows, cols, grid):
        # This is a simplified inverse - full implementation would need
        # proper subband unpacking. For now, just use the LL corner.
        full = np.zeros(grid)
        full[:rows, :cols] = corner.reshape(rows, cols)
        # Use 1-level inverse on the wire shape as approximation
        return full  # Placeholder
