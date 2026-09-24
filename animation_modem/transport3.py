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
EDGE_HYSTERESIS = 0.12


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
MIN_RUN = len(NOMINAL_GAPS) - 4


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
def _pulse_word_kernel(samples, hysteresis, nominal_gaps, nominal_span,
                       min_scale, max_scale, positions, starts):
    """Schmitt edges, their interpolated zero crossings, and the start of
    every edge run whose spacing matches the preamble word.

    Same rules as measure_pulses_numpy: an edge is the first sample of a new
    Schmitt state; its time is the last sign change before that sample,
    interpolated linearly. Returns (edge count, valid word count).
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
            positions[edges] = left + samples[left]/(samples[left]-samples[left+1])
            edges += 1
        state = now
    width = nominal_gaps.shape[0]+1
    valid = 0
    for j in range(edges-width+1):
        scale = (positions[j+width-1]-positions[j])/nominal_span
        if scale < .98*min_scale or scale > 1.02*max_scale:
            continue
        ok = True
        for g in range(width-1):
            expected = nominal_gaps[g]*scale
            if abs(positions[j+g+1]-positions[j+g]-expected) > max(1.2, .45*expected):
                ok = False
                break
        if ok:
            starts[valid] = j
            valid += 1
    return edges, valid


def measure_pulses(samples, min_scale=0.5, max_scale=4.0):
    """Acquire one pulse word as ``(position, scale, confidence)``.

    This is the designated V7 timing path: edge/pulse counted, not FFT
    correlated.  The returned scale is relative to the reference preamble.
    Edge finding and word matching run compiled; measure_pulses_numpy is the
    reference implementation.
    """
    samples = np.asarray(samples)
    if samples.ndim != 1 or samples.dtype not in (np.float32, np.float64):
        return measure_pulses_numpy(samples, min_scale, max_scale)
    count = len(NOMINAL_EDGES)
    positions = np.empty(len(samples))
    starts = np.empty(len(samples), np.int64)
    edges, valid = _pulse_word_kernel(
        samples, np.float64(EDGE_HYSTERESIS*PREAMBLE_AMPLITUDE), NOMINAL_GAPS,
        NOMINAL_SPAN, float(min_scale), float(max_scale), positions, starts)
    if edges < count or not valid:
        return None
    words = (positions[start:start+count] for start in starts[:valid])
    return _fit_pulse_words(samples, words)


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
    return _fit_pulse_words(samples, words[valid])


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


def _fit_pulse_words(samples, words):
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
            np.ascontiguousarray(word, dtype=np.float64), _FIT_NOMINAL)
        if confidence >= 0.45:
            return float(position), float(scale), float(confidence)
    return None
