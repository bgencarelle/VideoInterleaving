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

def _cdf97_predict_lift(odd, even, coeff):
    """Predict step: odd[k] += coeff*(even[k] + even[k+1]), k+1 wraps.

    The lifted pair must MIX the two polyphase components -- odd[k]'s
    neighbours in the interleaved signal are even[k] and even[k+1]. Lifting the
    odd sequence against itself (an np.roll on the SAME sequence) is not the
    CDF 9/7 predict step and is not invertible. Periodic boundary: the
    successor of the last even sample is the first one.

    Works along the LAST axis, so a 2-D array lifts every row at once.
    """
    return odd + coeff * (even + np.roll(even, -1, axis=-1))

def _cdf97_update_lift(even, odd, coeff):
    """Update step: even[k] += coeff*(odd[k-1] + odd[k]), k-1 wraps.

    Mirror of :func:`_cdf97_predict_lift`: even[k]'s neighbours are odd[k-1]
    (periodically the last odd sample) and odd[k]. Last axis, like predict.
    """
    return even + coeff * (np.roll(odd, 1, axis=-1) + odd)

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
        # Full multi-level DWT on source grid, packed in shared order and
        # truncated/padded to the wire shape. Truncation lands mid-subband:
        # the inverse zero-fills dropped tails (JPEG2000 truncation semantics).
        subbands = cdf97_forward_2d(plane, self.levels)
        dims = _cdf97_subband_dims(plane.shape[0], plane.shape[1], self.levels)
        order = _cdf97_pack_order(dims)
        out = np.concatenate([subbands[level][band].ravel()
                              for level, band, _, _ in order])
        target = rows * cols
        if len(out) >= target:
            return out[:target]
        padded = np.zeros(target)
        padded[:len(out)] = out
        return padded

    def _inverse_transform(self, corner, rows, cols, grid):
        # Unpack the flat wire corner back into subbands in the shared order
        # (zero-filling every dropped tail), then run the true multi-level
        # inverse. The output is the full source grid, soft where the wire
        # could not carry the detail bands.
        dims = _cdf97_subband_dims(grid[0], grid[1], self.levels)
        order = _cdf97_pack_order(dims)
        flat = np.asarray(corner).ravel()
        subbands = {}
        idx = 0
        for level, band, sr, sc in order:
            full = np.zeros((sr, sc), dtype=flat.dtype)
            n = sr * sc
            take = min(n, len(flat) - idx)
            if take > 0:
                full.ravel()[:take] = flat[idx:idx + take]
                idx += take
            subbands.setdefault(level, {})[band] = full
        return cdf97_inverse_2d(subbands, self.levels, out_shape=tuple(grid))
