"""V7 pulse acquisition compatibility module.

The V7 wire uses the established edge-counted preamble.  This module retains
only that acquisition path and its pulse geometry; legacy V1-V6 packet
encoders, OFDM decoders, and receiver state machines are intentionally gone.
"""
import math

import numpy as np
from numba import njit

from .v7_core import _sample_at

SYNC_LEN = 288
HALF = 8
PREAMBLE_BITS = (0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0)
PREAMBLE_AMPLITUDE = 0.55
SHORT, LONG = HALF, 2 * HALF
EDGE_HYSTERESIS = 0.2

# The six current coded-profile IDs use constant-weight biphase words.  Every
# word keeps the same first/last edge and the same edge count, so pulse timing
# geometry is unchanged; the interval pattern carries the profile ID.  ID 1 is
# the historical V7 word (Fold-500), preserving its existing pulse signature.
# Words have minimum Hamming distance six after short/long interval parsing.
PROFILE_PREAMBLE_BITS = {
    0: (0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0),  # Fold-off / aspect-fold-500
    1: PREAMBLE_BITS,                                        # Fold-500
    2: (0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0),  # Fold-1000
    3: (0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0),  # Mono-off / aspect-mono-500
    4: (0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0),  # Mono-500
    5: (0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0),  # Mono-1000
}


def _biphase(bits=PREAMBLE_BITS, half=HALF, amplitude=PREAMBLE_AMPLITUDE):
    """Build the countable biphase-mark acquisition preamble."""
    level, out = 1.0, []
    for bit in bits:
        level = -level
        out += [level] * half
        if bit:
            level = -level
        out += [level] * half
    wave = np.asarray(out, float)
    return wave * (amplitude / np.max(np.abs(wave)))


PREAMBLE = _biphase()
NOMINAL_EDGES = np.flatnonzero(np.diff(np.signbit(PREAMBLE)))
NOMINAL_SPAN = float(NOMINAL_EDGES[-1] - NOMINAL_EDGES[0])
NOMINAL_GAPS = np.diff(NOMINAL_EDGES).astype(float)
REVERSED_PREAMBLE = PREAMBLE[::-1].copy()
REVERSED_EDGES = np.flatnonzero(np.diff(np.signbit(REVERSED_PREAMBLE)))
REVERSED_SPAN = float(REVERSED_EDGES[-1] - REVERSED_EDGES[0])
REVERSED_GAPS = np.diff(REVERSED_EDGES).astype(float)
MIN_RUN = len(NOMINAL_GAPS) - 4


def profile_preamble(profile_code):
    """Return the read-only Schmitt-coded synchronization word for a profile."""
    try:
        return _PROFILE_PREAMBLES[int(profile_code)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'unknown pulse profile code {profile_code!r}') from exc


_PROFILE_CODES = np.asarray(tuple(PROFILE_PREAMBLE_BITS), dtype=np.int64)
_PROFILE_PREAMBLES = {
    code: _biphase(bits) for code, bits in PROFILE_PREAMBLE_BITS.items()
}
for _wave in _PROFILE_PREAMBLES.values():
    _wave.setflags(write=False)

_PROFILE_EDGES = np.asarray([
    np.flatnonzero(np.diff(np.signbit(_PROFILE_PREAMBLES[int(code)])))
    for code in _PROFILE_CODES], dtype=np.int64)
_PROFILE_REVERSED_EDGES = np.asarray([
    np.flatnonzero(np.diff(np.signbit(_PROFILE_PREAMBLES[int(code)][::-1])))
    for code in _PROFILE_CODES], dtype=np.int64)
if (not np.all(_PROFILE_EDGES.shape == (len(_PROFILE_CODES), len(NOMINAL_EDGES))) or
        not np.all(_PROFILE_REVERSED_EDGES.shape == _PROFILE_EDGES.shape) or
        not np.all(_PROFILE_EDGES[:, -1]-_PROFILE_EDGES[:, 0] == NOMINAL_SPAN) or
        not np.all(_PROFILE_REVERSED_EDGES[:, -1]-
                   _PROFILE_REVERSED_EDGES[:, 0] == NOMINAL_SPAN)):
    raise RuntimeError('pulse profile words must preserve the timing geometry')
_PROFILE_GAPS = np.diff(_PROFILE_EDGES, axis=1).astype(np.float64)
_PROFILE_REVERSED_GAPS = np.diff(_PROFILE_REVERSED_EDGES, axis=1).astype(np.float64)
_PROFILE_NOMINAL = _PROFILE_EDGES.astype(np.float64)+.5
_PROFILE_REVERSED_NOMINAL = _PROFILE_REVERSED_EDGES.astype(np.float64)+.5
for _array in (_PROFILE_CODES, _PROFILE_EDGES, _PROFILE_REVERSED_EDGES,
               _PROFILE_GAPS, _PROFILE_REVERSED_GAPS, _PROFILE_NOMINAL,
               _PROFILE_REVERSED_NOMINAL):
    _array.setflags(write=False)


def edge_intervals(samples, hysteresis=EDGE_HYSTERESIS * PREAMBLE_AMPLITUDE):
    state = np.where(samples > hysteresis, 1,
                     np.where(samples < -hysteresis, -1, 0))
    live = np.flatnonzero(state)
    if len(live) < 2:
        return np.empty(0, int)
    changes = np.flatnonzero(np.diff(state[live]))
    return live[changes + 1]


def _runs(gaps, tolerance=0.28):
    out = []
    start = 0
    n = len(gaps)
    while start < n:
        unit = float(np.min(gaps[start:start + MIN_RUN])) if start < n else 0.0
        if unit <= 0:
            start += 1
            continue
        end = start
        while end < n:
            gap = gaps[end]
            near_short = abs(gap - unit) <= tolerance * unit
            near_long = abs(gap - 2 * unit) <= tolerance * 2 * unit
            if not (near_short or near_long):
                break
            end += 1
        if end - start >= MIN_RUN:
            out.append((start, end, unit))
            start = end
        else:
            start += 1
    return out


@njit(cache=True, fastmath=False)
def _pulse_word_kernel(samples, hysteresis, profile_gaps,
                       profile_reversed_gaps, nominal_span, min_scale,
                       max_scale, match_both, requested_direction,
                       positions, starts,
                       profile_ids, fit_profiles):
    """Schmitt edges, their interpolated zero crossings, and the start of
    every edge run whose spacing matches a coded preamble.

    Same rules as measure_pulses_numpy: an edge is the first sample of a new
    Schmitt state; its time is the last sign change before that sample,
    interpolated linearly. Profile words share their edge count and span; the
    short/long interval pattern is decoded by matching those edge counts.
    """
    count = samples.shape[0]
    edges = 0
    state = 0
    last_crossing = -1
    previous_negative = False
    for i in range(count):
        value = samples[i]
        negative = math.copysign(1.0, value) < 0
        if i >= 1 and negative != previous_negative:
            last_crossing = i-1
        previous_negative = negative
        now = 1 if value > hysteresis else (-1 if value < -hysteresis else 0)
        if now == 0:
            continue
        if state != 0 and now != state:
            left = last_crossing
            denominator = samples[left]-samples[left+1]
            if denominator == 0.0:
                # Opposite signed zeros can mark a sign-bit transition inside
                # a dropout. Its only finite sample-time estimate is the
                # midpoint; the subsequent word matcher rejects the spurious
                # edge unless the surrounding pulse intervals also fit.
                positions[edges] = left+.5
            else:
                positions[edges] = left+samples[left]/denominator
            edges += 1
        state = now
    width = profile_gaps.shape[1]+1
    valid = 0
    for j in range(edges-width+1):
        scale = (positions[j+width-1]-positions[j])/nominal_span
        if scale < .98*min_scale or scale > 1.02*max_scale:
            continue
        match_count = 0
        matched_profile = -1
        matched_direction = 0
        best_profile = -1
        best_direction = 0
        best_score = 1e30
        for profile in range(profile_gaps.shape[0]):
            for orientation in range(2 if match_both else 1):
                direction = 1 if orientation == 0 else -1
                if (requested_direction != 0 and
                        direction != requested_direction):
                    continue
                gaps = (profile_gaps[profile] if direction > 0 else
                        profile_reversed_gaps[profile])
                score = 0.0
                matched = True
                for g in range(width-1):
                    observed = positions[j+g+1]-positions[j+g]
                    expected = gaps[g]*scale
                    tolerance = max(1.2, .45*expected)
                    error = abs(observed-expected)
                    if error > tolerance:
                        matched = False
                        break
                    normalized = error/tolerance
                    score += normalized*normalized
                if matched:
                    match_count += 1
                    if match_count == 1:
                        matched_profile = profile
                        matched_direction = direction
                    else:
                        if (matched_profile >= 0 and
                                profile != matched_profile):
                            matched_profile = -2
                        if direction != matched_direction:
                            matched_direction = 0
                    if score < best_score:
                        best_score = score
                        best_profile = profile
                        best_direction = direction
                    elif (profile != best_profile or
                          direction != best_direction):
                        # Keep timing from the best template, but fail closed
                        # on a profile/direction match that is not unique.
                        pass
        if match_count:
            # Signed, one-based edge indices leave zero for an ambiguous
            # direction. Multiple matching IDs get profile_id=-1.
            starts[valid] = (best_direction*(j+1)
                             if matched_direction != 0 else 0)
            if matched_profile >= 0 and matched_direction != 0:
                profile_ids[valid] = best_profile
            else:
                profile_ids[valid] = -1
            fit_profiles[valid] = best_profile
            valid += 1
    return edges, valid


def _measure_pulse_profiles(samples, min_scale, max_scale, match_both,
                            requested_direction=0):
    """Compiled edge matcher; return earliest fitted hit plus profile ID."""
    samples = np.asarray(samples)
    if samples.ndim != 1 or samples.dtype not in (np.float32, np.float64):
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    positions = np.empty(len(samples))
    starts = np.empty(len(samples), np.int64)
    profile_ids = np.empty(len(samples), np.int64)
    fit_profiles = np.empty(len(samples), np.int64)
    edges, valid = _pulse_word_kernel(
        samples, np.float64(EDGE_HYSTERESIS*PREAMBLE_AMPLITUDE),
        _PROFILE_GAPS, _PROFILE_REVERSED_GAPS, NOMINAL_SPAN,
        float(min_scale), float(max_scale), bool(match_both),
        int(requested_direction), positions,
        starts, profile_ids, fit_profiles)
    if edges < len(NOMINAL_EDGES) or not valid:
        return None
    first = {1: None, -1: None}
    count = len(NOMINAL_EDGES)
    for candidate in range(valid):
        signed_start = int(starts[candidate])
        if signed_start == 0:
            continue
        direction = 1 if signed_start > 0 else -1
        if first[direction] is not None:
            continue
        profile = int(fit_profiles[candidate])
        index = abs(signed_start)-1
        nominal = (_PROFILE_NOMINAL[profile] if direction > 0 else
                   _PROFILE_REVERSED_NOMINAL[profile])
        hit = _fit_pulse_words(
            samples, (positions[index:index+count],), nominal)
        if hit is not None:
            profile_index = int(profile_ids[candidate])
            profile_code = (int(_PROFILE_CODES[profile_index])
                            if profile_index >= 0 else None)
            first[direction] = (*hit, direction, profile_code)
        if first[1] is not None and first[-1] is not None:
            break
    forward, reverse = first[1], first[-1]
    if not match_both:
        return forward
    if forward is not None and reverse is not None:
        if _opposite_words_overlap(forward, reverse):
            return None
    candidates = [hit for hit in (forward, reverse) if hit is not None]
    return min(candidates, key=lambda hit: hit[0]) if candidates else None


def measure_pulses(samples, min_scale=0.5, max_scale=4.0):
    """Acquire one pulse word as ``(position, scale, confidence)``.

    This is the designated V7 timing path: edge/pulse counted, not FFT
    correlated.  The returned scale is relative to the reference preamble.
    Edge finding and word matching run compiled; measure_pulses_numpy is the
    reference implementation.
    """
    hit = _measure_pulse_profiles(samples, min_scale, max_scale, False)
    return hit[:3] if hit is not None else None


def measure_pulses_profile(samples, min_scale=0.5, max_scale=4.0):
    """Acquire a forward pulse word and return ``(position, scale, conf, id)``."""
    hit = _measure_pulse_profiles(samples, min_scale, max_scale, False)
    return hit if hit is None else (hit[0], hit[1], hit[2], hit[4])


# Opposite-direction words closer than one preamble length overlap: they read
# the same stretch of signal both ways, so neither orientation is trusted. Two
# genuine opposite words can sit closer than a whole packet in one case only: a
# reversed packet followed by a forward one (a reverse-to-forward turn on a
# packet boundary, as in a sampler's ping-pong loop). The reversed preamble
# then ends 16 samples before the boundary and the forward one starts 16 after
# it, so their template starts are exactly SYNC_LEN apart. Testing against
# SYNC_LEN itself made that pair a coin toss on scale-fit noise of a few
# hundredths of a sample. One preamble length leaves half a lead-in (32
# samples at 1x) of margin on each side.
OPPOSITE_WORD_SPACING = len(PREAMBLE)


def _opposite_words_overlap(first, second):
    """True if two ``(position, scale, ...)`` hits of opposite direction are
    closer than one preamble at the smaller of their scales."""
    return abs(first[0]-second[0]) < OPPOSITE_WORD_SPACING*min(first[1],
                                                               second[1])


def measure_pulses_both(samples, min_scale=0.5, max_scale=4.0,
                        direction='auto'):
    """Acquire the earliest unambiguous pulse word in either direction.

    Returns ``(template_start, scale, confidence, direction)``, with direction
    ``+1`` for the ordinary word and ``-1`` for its reversed gap pattern.
    Schmitt edges are extracted once; both gap templates are tested against
    that same edge list. A candidate matching both patterns is discarded.
    """
    hit = measure_pulses_profile_both(
        samples, min_scale, max_scale, direction=direction)
    return None if hit is None else hit[:4]


def measure_pulses_profile_both(samples, min_scale=0.5, max_scale=4.0,
                                direction='auto'):
    """Acquire either direction and return ``(pos, scale, conf, dir, id)``."""
    if direction not in ('auto', 'forward', 'reverse'):
        raise ValueError(f'unknown pulse direction {direction!r}')
    requested = {'auto': 0, 'forward': 1, 'reverse': -1}[direction]
    return _measure_pulse_profiles(
        samples, min_scale, max_scale, True,
        requested_direction=requested)


def measure_pulses_numpy(samples, min_scale=0.5, max_scale=4.0):
    """Reference edge/word search for measure_pulses."""
    samples = np.asarray(samples)
    edges = edge_intervals(samples)
    count = len(NOMINAL_EDGES)
    if len(edges) < count:
        return None
    crossings = np.flatnonzero(np.diff(np.signbit(samples)))
    left = crossings[np.searchsorted(crossings, edges - 1, side='right') - 1]
    values = samples[left]
    positions = left + values / (values - samples[left + 1])
    words = np.lib.stride_tricks.sliding_window_view(positions, count)
    scales = (words[:, -1] - words[:, 0]) / NOMINAL_SPAN
    valid = (scales >= 0.98 * min_scale) & (scales <= 1.02 * max_scale)
    expected = NOMINAL_GAPS[None, :] * scales[:, None]
    valid &= np.all(
        np.abs(np.diff(words, axis=1) - expected)
        <= np.maximum(1.2, 0.45 * expected), axis=1)
    return _fit_pulse_words(samples, words[valid], _FIT_NOMINAL)


def warmup_pulse_kernels():
    """Compile both acquisition signatures before a live input is opened."""
    silence = np.zeros(2048, dtype=np.float32)
    measure_pulses(silence)
    measure_pulses_both(silence)


def measure_pulses_both_numpy(samples, min_scale=0.5, max_scale=4.0,
                              direction='auto'):
    """Reference implementation for bidirectional pulse acquisition."""
    if direction not in ('auto', 'forward', 'reverse'):
        raise ValueError(f'unknown pulse direction {direction!r}')
    samples = np.asarray(samples)
    edges = edge_intervals(samples)
    count = len(NOMINAL_EDGES)
    if len(edges) < count:
        return None
    crossings = np.flatnonzero(np.diff(np.signbit(samples)))
    crossing_indexes = np.searchsorted(crossings, edges-1, side='right')-1
    crossing_indexes = np.clip(crossing_indexes, 0, max(len(samples)-2, 0))
    left = crossings[crossing_indexes]
    values = samples[left]
    positions = left + values/(values-samples[left+1])
    words = np.lib.stride_tricks.sliding_window_view(positions, count)
    scales = (words[:, -1]-words[:, 0])/NOMINAL_SPAN
    allowed = (scales >= .98*min_scale) & (scales <= 1.02*max_scale)
    observed = np.diff(words, axis=1)
    forward = allowed & np.all(
        np.abs(observed-NOMINAL_GAPS[None, :]*scales[:, None]) <=
        np.maximum(1.2, .45*NOMINAL_GAPS[None, :]*scales[:, None]), axis=1)
    reverse = allowed & np.all(
        np.abs(observed-REVERSED_GAPS[None, :]*scales[:, None]) <=
        np.maximum(1.2, .45*REVERSED_GAPS[None, :]*scales[:, None]), axis=1)
    candidates = []
    for way, mask, nominal in (
            (1, forward & ~reverse, _FIT_NOMINAL),
            (-1, reverse & ~forward, _FIT_REVERSED_NOMINAL)):
        if direction == 'forward' and way != 1:
            continue
        if direction == 'reverse' and way != -1:
            continue
        hit = _fit_pulse_words(samples, words[mask], nominal)
        if hit is not None:
            candidates.append((*hit, way))
    if len(candidates) == 2 and _opposite_words_overlap(*candidates):
        return None
    return min(candidates, key=lambda hit: hit[0]) if candidates else None


@njit(cache=True, fastmath=False)
def _fit_pulse_word(word, nominal):
    """Least-squares scale/position of one edge word and its confidence."""
    count = word.shape[0]
    word_mean = 0.0
    nominal_mean = 0.0
    for i in range(count):
        word_mean += word[i]
        nominal_mean += nominal[i]
    word_mean /= count
    nominal_mean /= count
    cross = 0.0
    spread = 0.0
    for i in range(count):
        centered = nominal[i]-nominal_mean
        cross += (word[i]-word_mean)*centered
        spread += centered*centered
    scale = cross/spread
    position = word_mean-scale*nominal_mean
    sq = 0.0
    for i in range(count):
        error = word[i]-position-scale*nominal[i]
        sq += error*error
    residual = math.sqrt(sq/count)
    return position, scale, max(0.0, 1-residual/(SHORT*scale))


_FIT_NOMINAL = NOMINAL_EDGES.astype(float) + 0.5
_FIT_REVERSED_NOMINAL = REVERSED_EDGES.astype(float) + 0.5


def _fit_pulse_words(samples, words, nominal):
    """Refine and fit candidate edge words; first confident fit wins.

    A word from slowed playback (span below nominal) is first refined on the
    signal with 16-tap reads; the fit itself runs compiled.
    """
    for word in words:
        if (word[-1] - word[0]) / NOMINAL_SPAN < 0.999:
            refined = word.copy()
            mono = samples[:, None]
            for _ in range(3):
                probes = np.concatenate([refined, refined - 0.05, refined + 0.05])
                amplitudes = _sample_at(mono, probes, taps=16)[:, 0]
                center, minus, plus = np.split(amplitudes, 3)
                derivative = (plus - minus) / 0.1
                delta = np.divide(
                    center, derivative, out=np.zeros_like(center),
                    where=np.abs(derivative) > 1e-6)
                refined = np.clip(
                    refined - np.clip(delta, -0.25, 0.25),
                    word - 0.5, word + 0.5)
            word = refined
        position, scale, confidence = _fit_pulse_word(
            np.ascontiguousarray(word, dtype=np.float64), nominal)
        if confidence >= 0.45:
            return float(position), float(scale), float(confidence)
    return None
