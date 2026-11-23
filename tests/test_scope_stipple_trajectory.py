import unittest

import numpy as np

from scope_bake import StippleEmitter


class StippleTrajectoryTests(unittest.TestCase):
    @staticmethod
    def render(rate, samples):
        lum = np.zeros((20, 24), dtype=np.float32)
        lum[2:18, 3:21] = np.linspace(
            0.2, 1.0, 16 * 18, dtype=np.float32).reshape(16, 18)
        emitter = StippleEmitter(
            rate, samples, points=96, geometry_samples=512,
            traversal_hz=30.0)
        pieces = []
        for _ in range(4):
            pieces.append(emitter.emit(lum).copy())
            emitter.accept(pieces[-1][-1])
        return np.concatenate(pieces)

    def test_timed_stipple_path_matches_across_dac_rates(self):
        at_48k = self.render(48_000, 800)
        at_96k = self.render(96_000, 1_600)
        np.testing.assert_allclose(at_48k, at_96k[::2], atol=2e-6, rtol=0)

    def test_rejected_candidate_does_not_commit_time(self):
        lum = np.ones((12, 12), dtype=np.float32)
        emitter = StippleEmitter(
            48_000, 800, points=64, geometry_samples=256,
            traversal_hz=20.0)
        checkpoint = emitter.checkpoint()
        discarded = emitter.emit(lum)
        emitter.restore(checkpoint)
        retried = emitter.emit(lum)
        np.testing.assert_array_equal(discarded, retried)
        self.assertEqual(emitter._trajectory_phase, 0.0)


if __name__ == "__main__":
    unittest.main()
