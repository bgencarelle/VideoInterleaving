"""Opt-in bounded adaptive guest range for the live serpentine nested fold."""
from __future__ import annotations

import math

import numpy as np
from tools import v7_sk_fold as sk


FACTORS = (1.0, 2.0, 4.0, 8.0, 16.0)
# Explicit, symmetric codebook. The remaining signature levels are reserved.
PILOT_LEVEL_TO_FACTOR = {-3: 1.0, -2: 2.0, -1: 4.0, 0: 8.0, 1: 16.0}
FACTOR_TO_PILOT_LEVEL = {value: key for key, value in PILOT_LEVEL_TO_FACTOR.items()}
HOST_LEVEL = -1


def select_guest_factor(flat, guest_positions, guest_sd, guest_scale, host_factor):
    """Choose the smallest bounded factor across all actual guest legs.

    Arrays are setup-compatible contiguous arrays: flat is the source frame;
    the remaining arrays have shape ``(legs, actual_guest_count)`` and include
    only positions selected by the table's guest map (never host or unused
    slots). ``host_factor`` is required and comes from the active ``nf.RATIO``.
    A frame that is non-finite or needs more than 16 is rejected, never clipped.
    """
    # Require arrays prepared by table/codec setup: no per-frame conversion or
    # numpy arithmetic is hidden in this decision path.
    flat = np.asarray(flat)
    positions = np.asarray(guest_positions)
    sd = np.asarray(guest_sd)
    scale = np.asarray(guest_scale)
    if (flat.dtype != np.float64 or not flat.flags.c_contiguous
            or positions.dtype != np.int64 or not positions.flags.c_contiguous
            or sd.dtype != np.float64 or not sd.flags.c_contiguous
            or scale.dtype != np.float64 or not scale.flags.c_contiguous):
        raise ValueError('adaptive factor inputs must be contiguous float64/int64 setup arrays')
    flat = flat.ravel()
    if positions.ndim != 2 or positions.shape != sd.shape or positions.shape != scale.shape:
        raise ValueError('guest positions, standard deviations and scales must match (legs, slots)')
    peak = 0.0
    for leg in range(positions.shape[0]):
        candidate, status = sk.adaptive_guest_peak(
            flat, positions[leg], sd[leg], scale[leg], float(host_factor))
        if status:
            raise ValueError('adaptive factor inputs contain invalid/non-finite values')
        peak = max(peak, candidate)
    factor = float(sk.adaptive_guest_factor_from_peak(peak))
    if factor == 0.0:
        raise OverflowError('frame exceeds the maximum adaptive guest factor 16')
    return factor


def host_factor_for_ratio(ratio):
    """Validate the actual nested-fold ratio and return its fixed -1 scale."""
    ratio = float(ratio)
    if not math.isfinite(ratio) or ratio <= 1.0:
        raise ValueError('nested-fold ratio must be finite and greater than one')
    return ratio**HOST_LEVEL


def pilot_level_for_factor(factor):
    """Map an exact supported factor to its explicitly assigned pilot level."""
    try:
        return FACTOR_TO_PILOT_LEVEL[float(factor)]
    except (KeyError, TypeError, ValueError):
        raise ValueError(f'unsupported adaptive guest factor: {factor!r}') from None


def factor_for_pilot_level(level):
    """Resolve only assigned rows; reserved rows are invalid, not stale state."""
    try:
        numeric = float(level)
        if not math.isfinite(numeric) or numeric != math.floor(numeric):
            raise ValueError
        return PILOT_LEVEL_TO_FACTOR[int(numeric)]
    except (KeyError, TypeError, ValueError, OverflowError):
        raise ValueError(f'reserved adaptive guest pilot level: {level!r}') from None


def pilot_pattern(level, hadamard, dithered=False):
    """Return the assigned signature row with the existing dither sign rule."""
    factor_for_pilot_level(level)  # fail closed for the ten reserved rows
    row = int(level)-(-3)+1  # row zero remains the stock signature
    if row <= 0 or row >= len(hadamard):
        raise ValueError(f'adaptive guest pilot level has no signature row: {level!r}')
    return sk.adaptive_pilot_pattern(hadamard, row, bool(dithered))


def install(out=None, step_factor=1.0, exact_inverse=False,
            isolated_path=None):
    """Install live adaptive serpentine hooks with host level fixed at -1."""
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    test_modem = repo/'test_modem_v7'
    if str(test_modem) not in sys.path:
        sys.path.insert(0, str(test_modem))
    import nested_fold as nf
    # Derive this at runtime from the actual loaded module; callers pass the
    # resulting value explicitly to select_guest_factor (no baked-in ratio).
    host_factor_for_ratio(nf.RATIO)
    from tools.v7_serpentine_codec import install as install_serpentine

    options = dict(out=out, fixed_level=HOST_LEVEL,
                   step_factor=step_factor, guest_factor=1.0,
                   exact_inverse=exact_inverse,
                   adaptive_guest_range=True)
    if isolated_path is not None:
        options['isolated_path'] = isolated_path
    return install_serpentine(**options)
