"""Mathematical kernel checks; these are not live modem validation."""
import unittest

import numpy as np

from tools import v7_sk_fold as sk


def objective(image, edge):
    xx = np.zeros_like(image)
    yy = np.zeros_like(image)
    xy = np.zeros_like(image)
    xx[:, :-2] = np.diff(image, n=2, axis=1)
    yy[:-2, :] = np.diff(image, n=2, axis=0)
    xy[:-1, :-1] = np.diff(np.diff(image, axis=0), axis=1)
    return np.sqrt(xx*xx+yy*yy+2*xy*xy+edge*edge).sum()


class CurvatureTests(unittest.TestCase):
    def test_gradient_matches_directional_derivative_at_boundaries(self):
        rng = np.random.default_rng(721)
        for shape in ((1, 1), (2, 2), (3, 7), (8, 5)):
            image = rng.normal(size=shape)
            direction = rng.normal(size=shape)
            gradient = np.empty_like(image)
            sk.curvature_gradient(image, .04, gradient)
            step = 1e-5
            numeric = (objective(image+step*direction, .04)-
                       objective(image-step*direction, .04))/(2*step)
            self.assertAlmostEqual(float(np.sum(gradient*direction)),
                                   float(numeric), places=6)

    def test_affine_ramp_has_no_curvature_force(self):
        y, x = np.mgrid[:9, :13]
        image = np.ascontiguousarray(.1*x+.2*y-.4, np.float64)
        gradient = np.empty_like(image)
        sk.curvature_gradient(image, .04, gradient)
        self.assertLess(float(np.max(np.abs(gradient))), 1e-12)

    def test_solve_respects_fixed_and_finite_coefficient_bounds(self):
        rng = np.random.default_rng(127)
        source = rng.normal(size=80)
        original = source.copy()
        room = np.full(80, .02)
        room[:10] = 0.
        room[60:] = np.inf
        a, b = sk.dct_matrix(8), sk.dct_matrix(10)
        result = sk.curvature_within_room(
            source, room, 8, a, np.ascontiguousarray(a.T), b,
            np.ascontiguousarray(b.T), 64, .04)
        np.testing.assert_array_equal(source, original)
        np.testing.assert_array_equal(result[:10], source[:10])
        self.assertLessEqual(float(np.max(np.abs(result[10:60]-source[10:60]))), .02+1e-14)
        initial = a.T@source.reshape(8, 10)@b
        solved = a.T@result.reshape(8, 10)@b
        self.assertLess(objective(solved, .04), objective(initial, .04))


if __name__ == '__main__':
    unittest.main()
