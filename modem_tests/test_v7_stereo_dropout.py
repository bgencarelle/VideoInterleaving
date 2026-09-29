"""Unit checks for per-leg pilot erasure classification."""
import unittest

import numpy as np

from animation_modem import v7


class V7StereoDropoutTests(unittest.TestCase):
    def test_one_leg_outlier_is_erased_but_common_mode_is_not(self):
        clean = np.full((v7.F, 2), .001)
        left_burst = clean.copy()
        left_burst[5, 0] = .2
        common_mode = clean.copy()
        common_mode[5] = .2

        np.testing.assert_array_equal(
            v7._stereo_erasure_mask(left_burst)[5], [True, False])
        np.testing.assert_array_equal(
            v7._stereo_erasure_mask(common_mode),
            np.zeros_like(common_mode, bool))

    def test_persistent_one_leg_erasure_uses_paired_leg_contrast(self):
        noise = np.full((v7.F, 2), .001)
        noise[:, 1] = .3

        mask = v7._stereo_erasure_mask(noise)

        self.assertTrue(np.all(mask[:, 1]))
        self.assertFalse(np.any(mask[:, 0]))


if __name__ == '__main__':
    unittest.main()
