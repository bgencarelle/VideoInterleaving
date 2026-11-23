"""State and output parity for the Numba causal scope low-pass."""
import numpy as np

from scope_lowpass import (CascadedOnePole,
                           _cascaded_one_pole_kernel)


def _reference_process(samples, state, a, enabled=True):
    if not enabled:
        return samples
    current = np.asarray(samples, dtype=np.float64)
    for section in range(state.shape[0]):
        zs = state[section]
        output = np.empty_like(current)
        for sample in range(len(current)):
            zs = a * zs + (1.0 - a) * current[sample]
            output[sample] = zs
        state[section] = zs
        current = output
    return current.astype(np.float32)


def test_cascaded_filter_matches_reference_output_state_and_endpoints():
    rng = np.random.default_rng(417)
    samples = rng.normal(0, 0.3, (2307, 2)).astype(np.float32)
    expected_state = np.zeros((4, 2), dtype=np.float64)
    actual_filter = CascadedOnePole(3200, 44100, order=4, channels=2)
    a = actual_filter.a

    expected = _reference_process(samples, expected_state, a)
    actual = actual_filter.process(samples)

    assert np.array_equal(actual, expected)
    assert np.array_equal(actual_filter.z, expected_state)
    assert np.array_equal(actual[0], expected[0])
    assert np.array_equal(actual[-1], expected[-1])


def test_cascaded_filter_is_invariant_to_chunk_partitioning():
    rng = np.random.default_rng(992)
    samples = rng.normal(0, 0.25, (4097, 2)).astype(np.float32)
    whole = CascadedOnePole(2700, 48000, order=4, channels=2)
    chunked = CascadedOnePole(2700, 48000, order=4, channels=2)

    expected = whole.process(samples)
    cuts = (0, 1, 13, 500, 2011, len(samples))
    actual = np.concatenate([
        chunked.process(samples[begin:end])
        for begin, end in zip(cuts[:-1], cuts[1:])
    ])

    assert np.array_equal(actual, expected)
    assert np.array_equal(chunked.z, whole.z)


def test_disabled_filter_is_identity_and_preserves_state():
    disabled = CascadedOnePole(None, 44100, order=4, channels=2)
    samples = np.arange(24, dtype=np.float32).reshape(12, 2)
    state = disabled.z.copy()

    output = disabled.process(samples)

    assert output is samples
    assert np.array_equal(disabled.z, state)


def test_filter_signature_is_warmed_before_processing():
    filtered = CascadedOnePole(3000, 44100, order=4, channels=2)
    signatures_before = tuple(_cascaded_one_pole_kernel.signatures)
    assert any(
        str(signature[0]) == "array(float64, 2d, C)"
        and str(signature[1]) == "array(float64, 2d, C)"
        for signature in signatures_before)

    filtered.process(np.ones((256, 2), dtype=np.float32))
    assert tuple(_cascaded_one_pole_kernel.signatures) == signatures_before


def test_cutoff_changes_preserve_the_reference_state_transition():
    rng = np.random.default_rng(52)
    first = rng.normal(0, 0.2, (317, 2)).astype(np.float32)
    second = rng.normal(0, 0.2, (283, 2)).astype(np.float32)
    actual_filter = CascadedOnePole(3500, 96000, order=3, channels=2)
    expected_state = np.zeros((3, 2), dtype=np.float64)

    expected_first = _reference_process(first, expected_state, actual_filter.a)
    actual_first = actual_filter.process(first)
    actual_filter.set_cutoff(2100, 96000)
    expected_second = _reference_process(second, expected_state, actual_filter.a)
    actual_second = actual_filter.process(second)

    assert np.array_equal(actual_first, expected_first)
    assert np.array_equal(actual_second, expected_second)
    assert np.array_equal(actual_filter.z, expected_state)
