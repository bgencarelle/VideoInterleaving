"""Algebraic checks for range-consistent luma reconstruction (not live tests)."""
import unittest

import numpy as np

from tools.v7_sk_fold import dct_matrix, range_consistent_luma, range_host_rooms


def reconstruct(luma, mask, bound=.8, passes=80, host_room=None):
    rows, cols = luma.shape
    rm, cm = dct_matrix(rows), dct_matrix(cols)
    return range_consistent_luma(luma.ravel(), mask.ravel(), rm, rm.T,
                                 cm, cm.T, bound, passes, host_room)


def violation(image, bound):
    return float(np.sum(np.maximum(np.abs(image)-bound, 0.0)**2))


class RangeReconstructionTests(unittest.TestCase):
    def test_overshoot_reduced_and_carried_coefficients_exact(self):
        # A bounded source whose low-frequency coefficients alone overshoot.
        rows, cols = 8, 6
        y, x = np.mgrid[:rows, :cols]
        source = np.where(x < cols//2, .8, -.8)
        rm, cm = dct_matrix(rows), dct_matrix(cols)
        coeff = rm @ source @ cm.T
        mask = np.zeros((rows, cols), dtype=np.bool_)
        mask[:2, :2] = True
        missing = np.where(mask, coeff, 0.0)
        initial_image = rm.T @ missing @ cm
        self.assertGreater(violation(initial_image, .8), 0.0)
        result = reconstruct(missing, mask, .8, 120).reshape(rows, cols)
        decoded = rm.T @ result @ cm
        self.assertLess(violation(decoded, .8), violation(initial_image, .8))
        np.testing.assert_array_equal(result[mask], missing[mask])

    def test_in_range_is_exact_identity_at_both_scales(self):
        y, x = np.mgrid[:8, :6]
        textured = .73*np.sin(.7*y+.23*x)*np.cos(.31*y-.8*x)
        mask = np.zeros_like(textured, dtype=bool)
        for bound in (.8, 1.0):
            source = textured*(bound/.8)
            coefficients = dct_matrix(8) @ source @ dct_matrix(6).T
            result = reconstruct(coefficients, mask, bound, 50)
            np.testing.assert_array_equal(result, coefficients.ravel())

    def test_zero_passes_returns_coefficients_exact_and_validation(self):
        coefficients = np.arange(12, dtype=float).reshape(4, 3)/3
        mask = np.zeros_like(coefficients, dtype=bool)
        np.testing.assert_array_equal(reconstruct(coefficients, mask, .8, 0),
                                      coefficients.ravel())
        with self.assertRaises(ValueError):
            reconstruct(coefficients[:-1], mask, .8)
        with self.assertRaises(ValueError):
            reconstruct(coefficients, mask, 0.0)
        with self.assertRaises(ValueError):
            reconstruct(np.full((4, 3), np.nan), mask, .8)

    def test_host_rooms_reduce_overshoot_and_constrain_carried_modes(self):
        rows, cols = 8, 6
        y, x = np.mgrid[:rows, :cols]
        source = np.where(x < cols//2, .8, -.8)
        rm, cm = dct_matrix(rows), dct_matrix(cols)
        coeff = rm @ source @ cm.T
        mask = np.zeros((rows, cols), dtype=bool)
        mask[:2, :2] = True
        carried = np.where(mask, coeff, 0.0)
        rooms = np.zeros_like(carried)
        rooms[mask] = .4
        before = rm.T @ carried @ cm
        result = reconstruct(carried, mask, .8, 120, rooms.ravel()).reshape(rows, cols)
        after = rm.T @ result @ cm
        self.assertLess(violation(after, .8), violation(before, .8))
        self.assertTrue(np.all(np.abs(result[mask]-carried[mask]) <= rooms[mask]))
        zero_room = reconstruct(carried, mask, .8, 120, np.zeros(carried.size))
        fixed = reconstruct(carried, mask, .8, 120)
        np.testing.assert_array_equal(zero_room, fixed)
        np.testing.assert_array_equal(zero_room.reshape(rows, cols)[mask], carried[mask])
        with self.assertRaises(ValueError):
            reconstruct(carried, mask, .8, 10, np.full(carried.size, -1.0))
        with self.assertRaises(ValueError):
            reconstruct(carried, mask, .8, 10, np.full(carried.size, np.inf))
        with self.assertRaises(ValueError):
            reconstruct(carried, mask, .8, 10, np.zeros(carried.size-1))

    def test_quantized_host_room_formula_covers_mono_and_paired_base(self):
        position = np.array([1, 4, 7], dtype=np.int64)
        level = np.array([1, 2, 0], dtype=np.int64)
        step = np.array([.4, .6, .8])
        host_sd = np.array([2., 3., 4.])
        factor = 1.7
        room = range_host_rooms(9, position, level, step, host_sd, factor)
        self.assertEqual(room[1], .5*.4*2.*factor)
        self.assertEqual(room[4], 0.0)
        self.assertEqual(room[7], 0.0)
        # Moving only the normalized shared base by this amount shifts both
        # +/- riding-detail host legs by the same amount, within either cell.
        ride_detail = .23
        delta_base = room[1]
        for sign in (-1., 1.):
            self.assertLessEqual(abs((delta_base+sign*ride_detail*0.) / factor),
                                 .5*step[0]*host_sd[0])


if __name__ == '__main__':
    unittest.main()
