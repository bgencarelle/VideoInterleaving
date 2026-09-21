"""V7 pulse acquisition compatibility module.

The V7 wire uses the established edge-counted preamble.  This module retains
only that acquisition path and its pulse geometry; legacy V1-V6 packet
encoders, OFDM decoders, and receiver state machines are intentionally gone.
"""
import numpy as np

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


def measure_pulses(samples, min_scale=0.5, max_scale=4.0):
    """Acquire one pulse word as ``(position, scale, confidence)``.

    This is the designated V7 timing path: edge/pulse counted, not FFT
    correlated.  The returned scale is relative to the reference preamble.
    """
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
    nominal = NOMINAL_EDGES.astype(float) + 0.5
    centered = nominal - nominal.mean()
    for word in words[valid]:
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
        scale = float(np.dot(word - word.mean(), centered) /
                      np.dot(centered, centered))
        position = float(word.mean() - scale * nominal.mean())
        residual = float(np.sqrt(np.mean(
            (word - position - scale * nominal) ** 2)))
        confidence = max(0.0, 1 - residual / (SHORT * scale))
        if confidence >= 0.45:
            return position, scale, confidence
    return None
