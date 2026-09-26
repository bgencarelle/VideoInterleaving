"""Exactness checks for the allocation-reduced stipple route builder."""
import unittest

import numpy as np

from scope_bake import _greedy_nearest_order


def _reference_order(points, start):
    remaining = np.ones(len(points), dtype=bool)
    pos = np.asarray(start, dtype=np.float64)
    order = []
    for _ in range(len(points)):
        delta = points - pos
        distance = np.einsum("ij,ij->i", delta, delta)
        distance[~remaining] = np.inf
        chosen = int(np.argmin(distance))
        order.append(chosen)
        remaining[chosen] = False
        pos = points[chosen]
    return np.asarray(order, dtype=np.int64)


class StippleRouteTests(unittest.TestCase):
    def test_order_is_identical_to_previous_greedy_selection(self):
        rng = np.random.default_rng(4821)
        points = rng.random((257, 2))
        points[:4] = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
        for start in (np.array([0.5, 0.5]), np.array([-0.2, 0.7])):
            np.testing.assert_array_equal(
                _greedy_nearest_order(points, start),
                _reference_order(points, start))

    def test_empty_route_returns_an_empty_index_vector(self):
        order = _greedy_nearest_order(np.empty((0, 2)), (0.0, 0.0))
        self.assertEqual(order.dtype, np.int64)
        self.assertEqual(order.shape, (0,))


if __name__ == "__main__":
    unittest.main()
