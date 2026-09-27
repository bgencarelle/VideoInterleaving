"""Compatibility checks for the V7 torture-matrix reference wire."""
import unittest

import numpy as np
from scipy.signal import resample_poly

from tools.v7_torture_matrix import RATE, _playback_wire


class V7TortureMatrixPlaybackTests(unittest.TestCase):
    def test_default_forward_wire_keeps_legacy_96khz_resampling(self):
        rng = np.random.default_rng(781)
        audio48 = rng.uniform(-.55, .55, (3*3920, 2)).astype(np.float32)
        expected = resample_poly(audio48, 2, 1, axis=0).astype(np.float32)

        actual = _playback_wire(audio48, 1.0, 'forward')

        self.assertEqual(RATE, 96000)
        np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
