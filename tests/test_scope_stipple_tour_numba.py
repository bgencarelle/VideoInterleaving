"""Parity and warm-signature checks for the stipple nearest-neighbour tour."""
import unittest

import numpy as np

from scope_bake import (
    StippleEmitter,
    _greedy_nearest_order,
    _greedy_nearest_order_kernel,
)


def reference_order(points, start):
    """Original NumPy implementation, retained here as an independent oracle."""
    points = np.asarray(points, dtype=np.float64)
    if not len(points):
        return np.empty(0, dtype=np.int64)
    remaining = np.ones(len(points), dtype=bool)
    dx = np.empty(len(points), dtype=np.float64)
    dy = np.empty(len(points), dtype=np.float64)
    distance = np.empty(len(points), dtype=np.float64)
    squared_y = np.empty(len(points), dtype=np.float64)
    order = np.empty(len(points), dtype=np.int64)
    pos = np.asarray(start, dtype=np.float64)
    for i in range(len(points)):
        np.subtract(points[:, 0], pos[0], out=dx)
        np.square(dx, out=distance)
        np.subtract(points[:, 1], pos[1], out=dy)
        np.square(dy, out=squared_y)
        np.add(distance, squared_y, out=distance)
        distance[~remaining] = np.inf
        chosen = int(np.argmin(distance))
        order[i] = chosen
        remaining[chosen] = False
        pos = points[chosen]
    return order


class StippleTourNumbaTests(unittest.TestCase):
    def assert_parity_and_immutability(self, points, start):
        before = np.array(points, copy=True)
        expected = reference_order(points, start)
        actual = _greedy_nearest_order(points, start)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(points, before)
        self.assertEqual(actual.dtype, np.dtype(np.int64))
        np.testing.assert_array_equal(np.sort(actual), np.arange(len(points)))

    def test_duplicates_and_equal_distance_ties_choose_lowest_remaining_index(self):
        points = np.array([
            [1.0, 0.0], [-1.0, 0.0], [1.0, 0.0],
            [0.0, 1.0], [0.0, -1.0], [2.0, 0.0],
        ])
        self.assert_parity_and_immutability(points, np.array([0.0, 0.0]))
        np.testing.assert_array_equal(
            _greedy_nearest_order(points, [0.0, 0.0]),
            np.array([0, 2, 5, 3, 1, 4], dtype=np.int64))

    def test_start_position_selects_nearest_point_first(self):
        points = np.array([[0.0, 0.0], [9.0, 1.0], [2.0, 1.0]])
        actual = _greedy_nearest_order(points, [8.5, 1.0])
        np.testing.assert_array_equal(actual, reference_order(points, [8.5, 1.0]))
        self.assertEqual(actual[0], 1)

    def test_strided_points_and_start_match_numpy_reference(self):
        storage = np.arange(96, dtype=np.float64).reshape(12, 8)
        points = storage[::2, 1::4]
        start_storage = np.array([99.0, 3.0, 88.0, -4.0])
        start = start_storage[1::2]
        self.assertFalse(points.flags.c_contiguous)
        self.assertFalse(start.flags.c_contiguous)
        self.assert_parity_and_immutability(points, start)

    def test_candidate_coordinate_bounds_and_deterministic_cloud(self):
        rng = np.random.default_rng(719)
        points = rng.uniform(0.0, 1.0, size=(96, 2))
        points[4] = [0.0, 0.0]
        points[5] = [1.0, 1.0]
        self.assert_parity_and_immutability(points, np.array([0.5, 0.5]))

    def test_empty_returns_empty_int64_permutation(self):
        result = _greedy_nearest_order(np.empty((0, 2)), [0.0, 0.0])
        self.assertEqual(result.shape, (0,))
        self.assertEqual(result.dtype, np.dtype(np.int64))

    def test_stipple_emitter_setup_warms_contiguous_float64_signature(self):
        StippleEmitter(48000, 1600, points=256)
        self.assertTrue(any(
            len(signature) == 2
            and signature[0].ndim == 2 and signature[0].layout == "C"
            and signature[1].ndim == 1 and signature[1].layout == "C"
            and "float64" in str(signature)
            for signature in _greedy_nearest_order_kernel.signatures))

    def test_read_only_cache_arrays_use_the_prewarmed_signature(self):
        StippleEmitter(48000, 1600, points=256)
        points = np.array([[0., 0.], [1., 0.], [0., 1.]])
        start = np.array([0.5, 0.5])
        points.setflags(write=False)
        start.setflags(write=False)
        signatures = tuple(_greedy_nearest_order_kernel.signatures)
        self.assert_parity_and_immutability(points, start)
        self.assertEqual(tuple(_greedy_nearest_order_kernel.signatures), signatures)

    def test_invalid_points_are_rejected_before_the_native_kernel(self):
        for points in (np.zeros((2, 1)), np.array([[np.nan, 0.]])):
            with self.assertRaises(ValueError):
                _greedy_nearest_order(points, [0., 0.])
        with self.assertRaises(ValueError):
            _greedy_nearest_order(np.zeros((2, 2)), [np.inf, 0.])


if __name__ == "__main__":
    unittest.main()
