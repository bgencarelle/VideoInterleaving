"""Reference parity for the Numba raster weighted-walk sampler."""
import numpy as np
import pytest

import scope_bake
from scope_bake import TraceEmitter, render_luma


def _numpy_walk_raw(points, weights, samples):
    """Pre-Numba implementation retained here as an independent oracle."""
    weights = np.maximum(np.asarray(weights, np.float64), 1e-12)
    cumulative = np.concatenate([[0.0], np.cumsum(weights)])
    t = np.linspace(0.0, cumulative[-1], samples, endpoint=False)
    segment = np.clip(
        np.searchsorted(cumulative, t, side="right") - 1,
        0, len(weights) - 1)
    fraction = ((t - cumulative[segment]) / weights[segment])[:, None]
    return points[segment] + fraction * (
        points[segment + 1] - points[segment])


@pytest.mark.parametrize("point_dtype", (np.float32, np.float64))
def test_weighted_walk_matches_numpy_at_boundaries_and_clamped_weights(
        point_dtype):
    points = np.array([
        [-0.91, 0.42], [-0.7, 0.1], [-0.2, 0.1], [0.0, -0.4],
        [0.38, -0.4], [0.81, 0.7], [0.97, -0.2],
    ], dtype=point_dtype)
    exact_boundary_weights = np.array([0.25, 0.5, 0.125, 0.375])
    clamped_weights = np.array([0.0, 0.25, 0.5, -0.2, 0.125, 0.375])

    # The first total is exactly representable and n=5 samples land on
    # cumulative segment boundaries, pinning searchsorted(side="right").
    for weights, sample_counts in (
            (exact_boundary_weights, (0, 1, 4, 5, 97, 2048)),
            (clamped_weights, (1, 73, 2048))):
        for samples in sample_counts:
            expected = _numpy_walk_raw(points, weights, samples)
            actual = scope_bake._walk_raw(points, weights, samples)
            assert np.array_equal(actual, expected)


def test_weighted_walk_accepts_strided_input_without_changing_samples():
    points = np.arange(28, dtype=np.float32).reshape(14, 2)[::2]
    weights = np.array([0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.25])[::2]
    expected = _numpy_walk_raw(points, weights, 333)
    actual = scope_bake._walk_raw(points, weights, 333)
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("oversample", (1, 2))
@pytest.mark.parametrize("fields", (1, 2))
def test_trace_emitter_matches_reference_sampler(
        monkeypatch, oversample, fields):
    yy, xx = np.mgrid[:96, :72].astype(np.float32)
    lum = np.exp(-(((xx - 36) / 19.) ** 2 + ((yy - 48) / 31.) ** 2))
    lum -= 0.6 * np.exp(-(((xx - 36) / 8.) ** 2 + ((yy - 42) / 5.) ** 2))
    lum = np.clip(lum, 0, 1).astype(np.float32)
    lum[lum < 0.04] = 0
    kwargs = dict(
        gamma=2.2, trim=0.02, fields=fields, oversample=oversample,
        border=0.04, sweep="alternate")

    with monkeypatch.context() as patcher:
        patcher.setattr(scope_bake, "_walk_raw", _numpy_walk_raw)
        reference = TraceEmitter(48000, 800, **kwargs)
        expected = [reference.emit(lum) for _ in range(3)]

    actual_emitter = TraceEmitter(48000, 800, **kwargs)
    actual = [actual_emitter.emit(lum) for _ in range(3)]
    for got, want in zip(actual, expected):
        assert np.array_equal(got, want)


def test_fixed_yt_trace_matches_reference_sampler(monkeypatch):
    yy, xx = np.mgrid[:48, :32].astype(np.float32)
    lum = np.zeros((48, 32), dtype=np.float32)
    lum[((xx - 16) / 11.) ** 2 + ((yy - 24) / 20.) ** 2 < 1] = 0.8
    kwargs = dict(grid=(24, 16), levels=(0.0, 1.0), yt_timing="fixed")

    with monkeypatch.context() as patcher:
        patcher.setattr(scope_bake, "_walk_raw", _numpy_walk_raw)
        expected = TraceEmitter(48000, 800, **kwargs).emit(lum)

    actual = TraceEmitter(48000, 800, **kwargs).emit(lum)
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize(
    ("yt_timing", "point_type"),
    ((None, "float32"), ("fixed", "float64")))
def test_trace_emitter_warms_its_sampler_signature_before_render(
        yt_timing, point_type):
    emitter = TraceEmitter(48000, 800, yt_timing=yt_timing)
    signatures_before = tuple(scope_bake._walk_raw_numba_kernel.signatures)
    assert any(
        str(signature[0]) == f"array({point_type}, 2d, C)"
        and str(signature[1]) == "array(float64, 1d, C)"
        for signature in signatures_before)

    lum = np.zeros((24, 18), dtype=np.float32)
    lum[3:21, 4:14] = 0.75
    frame = emitter.emit(lum)
    assert frame.shape == (800, 2)
    assert tuple(scope_bake._walk_raw_numba_kernel.signatures) == signatures_before


def test_render_luma_reference_entry_point_still_returns_float32():
    lum = np.zeros((24, 18), dtype=np.float32)
    lum[3:21, 4:14] = 0.75
    frame = render_luma(lum, 800)
    assert frame.dtype == np.float32
    assert frame.shape == (800, 2)
