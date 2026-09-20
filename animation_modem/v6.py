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

from .core import N, REFERENCE_RATE, SourceCoder, _slot_order
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


def slot_fields(layout):
    """Carrier bin, stereo channel, real/imag part and symbol of every slot."""
    bins, channels = _slot_carriers(layout)
    parts = np.tile(np.array([0, 1]), len(bins)//2)
    return bins, channels, parts, slot_symbols(layout)


def _interleave_step(count):
    """A stride coprime with `count` near its golden fraction."""
    step = max(1, int(round(count*.382)))
    while np.gcd(step, count) != 1:
        step += 1
    return step


# Tape placement. Spacing loss grows linearly with frequency in dB, so on tape
# (and on low-bitrate lossy codecs, which also shed treble first) the safest
# carriers are the lowest ones. Bin 1 (375 Hz) is demoted rather than forbidden:
# it sits on the declared band edge and on 50/60 Hz hum harmonics.
TAPE_DEMOTED_BINS = (1,)
TAPE_DEMOTED_HEALTH = 28
TAPE_COPY_SPREAD = 7        # bins; 2.6 kHz, ample against narrowband hum/beats


def _tape_health(bins):
    health = np.asarray(bins, float).copy()
    health[np.isin(bins, TAPE_DEMOTED_BINS)] = TAPE_DEMOTED_HEALTH
    return health


def tape_slot_order(layout):
    """Slots from lowest (safest) carrier upward, interleaved in time.

    Within a carrier, successive values alternate track, then real/imag, and
    walk the symbols with a coprime stride so that consecutive-importance
    values are ~40% of a packet apart: a burst takes a scattered subset of
    each importance tier rather than a contiguous block of it.
    """
    bins, channels, parts, symbols = slot_fields(layout)
    total = int(symbols.max())+1
    walk = (symbols*_interleave_step(total)) % total
    return np.lexsort((channels, parts, walk, _tape_health(bins)))


def slot_report(coder, layout, tiers=((0, 100), (0, 720), (720, 2880))):
    """Where each importance tier actually rides (frequency, time, track).

    Returns one dict per tier of originals (by rank) plus one for copies. The
    point is to make placement auditable: priority is only as good as the
    carriers and symbols it lands on.
    """
    bins, channels, _, symbols = slot_fields(layout)
    hz = bins*(REFERENCE_RATE/N)
    slots = coder.slots(layout)
    by_rank = np.argsort(coder.rank, kind='stable')
    rows = []

    def describe(label, where):
        pct = lambda x: np.percentile(x, [0, 10, 50, 90, 100]).round(0).tolist()
        return {'tier': label, 'count': int(len(where)), 'hz': pct(hz[where]),
                'symbols': pct(symbols[where]),
                'distinct_symbols': int(len(np.unique(symbols[where]))),
                'left_share': round(float(np.mean(channels[where] == 0)), 3),
                'header_spare': int(np.sum(where < layout.header_capacity))}

    for lo, hi in tiers:
        rows.append(describe(f'rank {lo}-{hi}', slots[by_rank[lo:hi]]))
    if len(coder.copy_of):
        home, copy = slots[coder.copy_of], slots[coder.n_orig:]
        row = describe('copies', copy)
        row['copy_symbol_gap'] = np.percentile(
            np.abs(symbols[home]-symbols[copy]), [0, 50, 100]).tolist()
        row['copy_bin_gap'] = np.percentile(
            np.abs(bins[home]-bins[copy]), [0, 50, 100]).tolist()
        row['both_above_6k'] = int(np.sum((hz[home] > 6000) & (hz[copy] > 6000)))
        row['both_below_4k'] = int(np.sum((hz[home] < 4000) & (hz[copy] < 4000)))
        rows.append(row)
    return rows


class ProtectedAnalogCoder:
    """Add unequal, soft-combined stereo protection to an analog source coder.

    ``placement='middle-out'`` is the original V6 mapping. ``placement='tape'``
    puts the foundation on the lowest carriers and interleaves every tier in
    time (see ``_tape_slots``). ``copy_of`` may be empty for a no-copy control.
    """

    MIN_COPY_SPREAD = 8

    def __init__(self, base, copy_of, rank, profile_name, placement='middle-out'):
        if placement not in ('middle-out', 'tape'):
            raise ValueError("placement must be 'middle-out' or 'tape'")
        self.placement = placement
        if placement == 'tape':
            self.MIN_COPY_SPREAD = TAPE_COPY_SPREAD
        self.base = base
        self.shapes = list(base.shapes)
        self.grids = list(base.grids)
        self.source_count = base.source_count
        self.n_orig = base.count
        self.copy_of = np.asarray(copy_of, int)
        self.count = self.n_orig + len(self.copy_of)
        self.rank = np.asarray(rank, float)
        self.profile_name = profile_name
        if self.n_orig != ORIGINAL_VALUES or len(self.copy_of) not in (
                0, FOUNDATION_VALUES, ORIGINAL_VALUES):
            raise ValueError('V6 requires 2880 originals and 0, 720 or 2880 copies')
        if placement == 'tape' and len(self.copy_of) == ORIGINAL_VALUES:
            raise ValueError('tape placement protects a foundation, not everything')
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
        if self.placement == 'tape':
            mapping = self._tape_slots(layout)
            mapping.setflags(write=False)
            self._slots[layout] = mapping
            return mapping
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

    def _tape_slots(self, layout):
        """Foundation low in frequency and spread in time; detail above it.

        The lowest usable carriers form a foundation zone split into a lower
        half (homes) and an upper half (copies). Each home carrier is paired
        with one copy carrier; a home at (symbol s, track t, part p) has its
        copy at (s + half a packet, other track, same part), so every copy is
        cross-track, >= TAPE_COPY_SPREAD bins away and ~half a packet later.
        Foundation values fill the homes in rank order, lowest carrier first.
        Everything else follows ``tape_slot_order`` by rank, so the header
        symbols' high spare carriers take the least important detail.
        """
        bins, channels, parts, symbols = slot_fields(layout)
        order = tape_slot_order(layout)
        mapping = np.empty(self.count, int)
        by_rank = np.argsort(self.rank, kind='stable')
        if not len(self.copy_of):
            mapping[by_rank] = order[:self.n_orig]
            return mapping

        foundation = self.copy_of[np.argsort(self.rank[self.copy_of],
                                             kind='stable')]
        carriers = sorted(set(bins.tolist())-set(TAPE_DEMOTED_BINS))
        per_bin = {b: int(np.sum(bins == b)) for b in carriers}
        zone = []
        for b in carriers:
            zone.append(b)
            half = len(zone)//2
            if len(zone) % 2 == 0 and sum(per_bin[z] for z in zone[:half]) >= \
                    len(foundation):
                break
        else:
            raise ValueError(f'{layout.name} has no room for a tape foundation')
        low, high = zone[:len(zone)//2], zone[len(zone)//2:]

        lookup = {key: slot for slot, key in enumerate(
            zip(bins.tolist(), symbols.tolist(), channels.tolist(), parts.tolist()))}
        position_in_order = np.empty(len(order), int)
        position_in_order[order] = np.arange(len(order))
        homes, copies = [], []
        for home_bin, copy_bin in zip(low, high):
            if abs(copy_bin-home_bin) < self.MIN_COPY_SPREAD:
                raise ValueError('tape foundation zone is too narrow for copy spread')
            here = np.flatnonzero(bins == home_bin)
            here = here[np.argsort(position_in_order[here], kind='stable')]
            carried = np.unique(symbols[bins == copy_bin])
            for slot in here:
                s = int(symbols[slot])
                at = int(np.searchsorted(carried, s))
                shifted = int(carried[(at + len(carried)//2) % len(carried)])
                homes.append(slot)
                copies.append(lookup[(copy_bin, shifted, 1-int(channels[slot]),
                                      int(parts[slot]))])
        homes = np.asarray(homes[:len(foundation)])
        copies = np.asarray(copies[:len(foundation)])
        mapping[foundation] = homes
        position = {int(c): j for j, c in enumerate(self.copy_of)}
        for value, copy in zip(foundation, copies):
            mapping[self.n_orig + position[int(value)]] = copy

        used = np.zeros(len(bins), bool)
        used[homes] = used[copies] = True
        free = order[~used[order]]
        rest = by_rank[~np.isin(by_rank, foundation)]
        mapping[rest] = free[:len(rest)]
        return mapping


@lru_cache(maxsize=4)
def tape_coder(transform, copies=True):
    """V6 with tape placement (low-carrier foundation, time-interleaved).

    ``copies=False`` is the matching no-copy control on the same wire.
    """
    if transform == 'dct':
        base = SourceCoder(V6_SHAPES, grids=V6_GRIDS)
        foundation, rank = _dct_foundation(), _dct_rank()
    elif transform == 'wavelet':
        base = Cdf97Coder(V6_GRIDS, ORIGINAL_VALUES, levels=2)
        foundation, rank = _wavelet_foundation(), base.rank
    else:
        raise ValueError("V6 transform must be 'dct' or 'wavelet'")
    copy_of = foundation if copies else np.empty(0, int)
    name = f"v6-tape{'' if copies else '-nocopy'}-{transform}"
    return ProtectedAnalogCoder(base, copy_of, rank, name, placement='tape')


@lru_cache(maxsize=2)
def nocopy_coder(transform):
    """Original middle-out V6 placement with the foundation copies removed."""
    if transform == 'dct':
        base = SourceCoder(V6_SHAPES, grids=V6_GRIDS)
        rank = _dct_rank()
    elif transform == 'wavelet':
        base = Cdf97Coder(V6_GRIDS, ORIGINAL_VALUES, levels=2)
        rank = base.rank
    else:
        raise ValueError("V6 transform must be 'dct' or 'wavelet'")
    return ProtectedAnalogCoder(base, np.empty(0, int), rank,
                                f'v6-nocopy-{transform}')


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


@lru_cache(maxsize=2)
def full_repeat_coder(transform):
    """Full-repeat control: every analog coefficient has a second copy."""
    if transform == 'dct':
        base = SourceCoder(V6_SHAPES, grids=V6_GRIDS)
        rank = _dct_rank()
    elif transform == 'wavelet':
        base = Cdf97Coder(V6_GRIDS, ORIGINAL_VALUES, levels=2)
        rank = base.rank
    else:
        raise ValueError("V6 transform must be 'dct' or 'wavelet'")
    return ProtectedAnalogCoder(base, np.arange(ORIGINAL_VALUES), rank,
                                f'v6-repeat-{transform}')


def coder_for(transform):
    if transform == 'dct':
        return dct_coder()
    if transform == 'wavelet':
        return wavelet_coder()
    raise ValueError("V6 transform must be 'dct' or 'wavelet'")
