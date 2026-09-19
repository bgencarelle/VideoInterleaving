import unittest
import numpy as np

from animation_modem.imaging import stabilize_chroma


class ChromaRecoveryTests(unittest.TestCase):
    SHAPES = [(4, 4), (2, 2), (2, 2)]

    def test_clean_result_is_unchanged(self):
        current = np.arange(24, dtype=float)
        got = stabilize_chroma(current, np.zeros(24), self.SHAPES,
                                pilot_error=.05, coverage=1.0)
        np.testing.assert_array_equal(got, current)

    def test_only_chroma_blends_on_degraded_result(self):
        current = np.ones(24, dtype=float)
        previous = np.zeros(24, dtype=float)
        got = stabilize_chroma(current, previous, self.SHAPES,
                               pilot_error=1.0, coverage=.5)
        np.testing.assert_array_equal(got[:16], current[:16])
        self.assertTrue(np.all((got[16:] > 0) & (got[16:] < 1)))

    def test_degraded_chroma_smoothing_reduces_spatial_variation(self):
        rng = np.random.default_rng(4)
        current = np.ones(24, dtype=float)
        current[16:] += rng.normal(0, .4, 8)
        previous = np.zeros(24, dtype=float)
        got = stabilize_chroma(current, previous, self.SHAPES,
                               pilot_error=1.0, coverage=.5)
        self.assertLess(np.std(got[16:]), np.std(current[16:]))

    def test_missing_or_different_previous_is_unchanged(self):
        current = np.ones(24, dtype=float)
        np.testing.assert_array_equal(
            stabilize_chroma(current, None, self.SHAPES, 1.0, .2), current)
        np.testing.assert_array_equal(
            stabilize_chroma(current, np.zeros(8), self.SHAPES, 1.0, .2), current)


if __name__ == '__main__':
    unittest.main()
