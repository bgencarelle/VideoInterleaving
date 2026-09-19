"""V6 analog tape candidates on one shared pulse-counted wire.

V6 changes source coding and protection, not acquisition or modulation.  Both
candidate transforms carry 2,880 analog-valued coefficients on ``WIRE_V6``.
The 720-coefficient color foundation (20x24 Y and 10x12 Cb/Cr) is repeated on
the opposite stereo track and a separated carrier; refinement is sent once.
This keeps DCT versus CDF 9/7 comparisons honest: source grids, slot budget,
bandwidth, timing, copy budget, and reconstruction geometry are identical.
"""
from functools import lru_cache

import numpy as np
from scipy.optimize import linear_sum_assignment

from .core import SourceCoder, _slot_order
from .wavelet import Cdf97Coder, TAPE_CENTRE_BIN, _cdf97_subband_dims, _slot_carriers


V6_GRIDS = ((96, 80), (48, 40), (48, 40))
V6_SHAPES = ((48, 40), (24, 20), (24, 20))
FOUNDATION_SHAPES = ((24, 20), (12, 10), (12, 10))
ORIGINAL_VALUES = 2880
FOUNDATION_VALUES = 720
WIRE_VALUES = ORIGINAL_VALUES + FOUNDATION_VALUES


def _plane_offsets(shapes):
    offset = 0
    for rows, cols in shapes:
        yield offset
        offset += rows*cols


def _dct_foundation():
    """Indices of the low-frequency color foundation in the packed DCT."""
    out = []
    for offset, (rows, cols), (keep_rows, keep_cols) in zip(
            _plane_offsets(V6_SHAPES), V6_SHAPES, FOUNDATION_SHAPES):
        yy, xx = np.mgrid[:keep_rows, :keep_cols]
        out.extend(offset + yy.ravel()*cols + xx.ravel())
    return np.asarray(out, int)


def _dct_rank():
    """Comparable normalized spatial-frequency rank for all three planes."""
    ranks = []
    for rows, cols in V6_SHAPES:
        yy, xx = np.mgrid[:rows, :cols]
        ranks.extend(np.hypot(yy/rows, xx/cols).ravel())
    return np.asarray(ranks)


def _wavelet_foundation():
    """Indices of each plane's deepest LL band in CDF packed order."""
    out = []
    offset = 0
    for rows, cols in V6_GRIDS:
        deepest_rows, deepest_cols = _cdf97_subband_dims(rows, cols, 2)[-1]
        count = deepest_rows*deepest_cols
        out.extend(range(offset, offset+count))
        offset += rows*cols//4
    return np.asarray(out, int)


def slot_symbols(layout):
    """OFDM symbol number of every flat image slot (for diversity tests)."""
    symbols = []
    if layout.dense_header:
        for symbol in range(layout.header_symbols):
            symbols.extend([2+symbol] * (len(layout.spare_bins)*4))
    first = 2 + layout.header_symbols
    for symbol in range(layout.image_symbols):
        symbols.extend([first+symbol] * (len(layout.data_bins)*4))
    return np.asarray(symbols, int)


class ProtectedAnalogCoder:
    """Add unequal, soft-combined stereo protection to an analog source coder."""

    MIN_COPY_SPREAD = 8

    def __init__(self, base, copy_of, rank, profile_name):
        self.base = base
        self.shapes = list(base.shapes)
        self.grids = list(base.grids)
        self.source_count = base.source_count
        self.n_orig = base.count
        self.copy_of = np.asarray(copy_of, int)
        self.count = self.n_orig + len(self.copy_of)
        self.rank = np.asarray(rank, float)
        self.profile_name = profile_name
        if self.n_orig != ORIGINAL_VALUES or len(self.copy_of) != FOUNDATION_VALUES:
            raise ValueError('V6 requires 2880 originals and 720 foundation copies')
        if self.rank.shape != (self.n_orig,):
            raise ValueError('V6 requires one importance rank per original')
        if len(np.unique(self.copy_of)) != len(self.copy_of) or np.any(
                (self.copy_of < 0) | (self.copy_of >= self.n_orig)):
            raise ValueError('Foundation copies must name unique originals')

        sigma = np.sqrt(np.asarray(base.variance, float))
        full = np.concatenate([sigma, sigma[self.copy_of]])
        gains = np.sqrt(full/np.mean(full))
        self.gains = gains/np.sqrt(np.mean(gains*gains))
        self.variance = np.asarray(base.variance, float)
        self._slots = {}

        # Confidence thresholds are unequal protection, not temporal smoothing.
        # Coarse luma and chroma remain available at low confidence; uncertain
        # unprotected detail fades to zero (softness) rather than false color.
        plane = np.empty(self.n_orig, int)
        for p, (offset, shape) in enumerate(zip(_plane_offsets(self.shapes),
                                                self.shapes)):
            plane[offset:offset+shape[0]*shape[1]] = p
        protected = np.zeros(self.n_orig, bool)
        protected[self.copy_of] = True
        self.recovery_floor = np.where(
            protected, np.where(plane == 0, .05, .15),
            np.where(plane == 0, .45, .60))

    def forward(self, values):
        raw = self.base.forward(values)/self.base.gains
        return np.concatenate([raw, raw[self.copy_of]])*self.gains

    def inverse(self, sent, reliability=None, noise_variance=None):
        y = np.asarray(sent, float)
        if reliability is None:
            raw = y/self.gains
            kept = raw[:self.n_orig].copy()
            kept[self.copy_of] = (kept[self.copy_of] + raw[self.n_orig:])/2
        else:
            rel = np.maximum(np.asarray(reliability, float), 1e-6)
            noise = np.maximum(np.asarray(noise_variance, float), 1e-20)
            a = self.gains*rel
            numerator = a*y/noise
            precision = a*a/noise
            top = numerator[:self.n_orig].copy()
            bottom = precision[:self.n_orig].copy()
            np.add.at(top, self.copy_of, numerator[self.n_orig:])
            np.add.at(bottom, self.copy_of, precision[self.n_orig:])
            kept = self.variance*top/(1 + self.variance*bottom)
            confidence = self.variance*bottom/(1 + self.variance*bottom)
            gate = np.clip((confidence-self.recovery_floor) /
                           (.85-self.recovery_floor), 0.0, 1.0)
            kept *= gate
        # Feed the base coder its own wire convention.  Reliability has already
        # been fused above, so this call only performs the inverse transform.
        return self.base.inverse(kept*self.base.gains)

    def slots(self, layout):
        """Place every copy on the opposite track and a separated carrier."""
        if layout in self._slots:
            return self._slots[layout]
        if self.count > layout.capacity:
            raise ValueError(f'{self.count} values exceed {layout.name} capacity '
                             f'{layout.capacity}')
        bins, channels = _slot_carriers(layout)
        symbols = slot_symbols(layout)
        order = _slot_order(layout)

        originals = order[:self.n_orig]
        mapping = np.empty(self.count, int)
        # _slot_order is already middle-out and alternates carrying dimensions.
        # Keeping that order also balances the protected set across tracks; a
        # second sort by health grouped channels and could leave too few slots
        # on the opposite track for all copies.
        mapping[np.argsort(self.rank, kind='stable')] = originals

        remaining = order[self.n_orig:]
        for target in (0, 1):
            copy_indices = np.flatnonzero(
                channels[mapping[self.copy_of]] == 1-target)
            homes = mapping[self.copy_of[copy_indices]]
            pool = remaining[channels[remaining] == target]
            distance = np.abs(bins[homes, None]-bins[pool][None, :])
            cost = (1e6*(distance < self.MIN_COPY_SPREAD) +
                    1e4*(symbols[homes, None] == symbols[pool][None, :]) +
                    np.abs(bins[pool]-TAPE_CENTRE_BIN)[None, :])
            rows, columns = linear_sum_assignment(cost)
            if len(rows) != len(copy_indices):
                raise ValueError('Wire has no opposite-track slots for foundation')
            mapping[self.n_orig+copy_indices[rows]] = pool[columns]

        mapping.setflags(write=False)
        self._slots[layout] = mapping
        return mapping


@lru_cache(maxsize=1)
def dct_coder():
    base = SourceCoder(V6_SHAPES, grids=V6_GRIDS)
    return ProtectedAnalogCoder(base, _dct_foundation(), _dct_rank(), 'v6-dct')


@lru_cache(maxsize=1)
def wavelet_coder():
    # Capacity equal to the original count gives a non-repeating CDF base. V6
    # then applies exactly the same 720-copy policy used by the DCT candidate.
    base = Cdf97Coder(V6_GRIDS, ORIGINAL_VALUES, levels=2)
    return ProtectedAnalogCoder(base, _wavelet_foundation(), base.rank,
                                'v6-wavelet')


def coder_for(transform):
    if transform == 'dct':
        return dct_coder()
    if transform == 'wavelet':
        return wavelet_coder()
    raise ValueError("V6 transform must be 'dct' or 'wavelet'")
