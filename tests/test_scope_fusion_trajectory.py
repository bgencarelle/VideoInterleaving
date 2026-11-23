import unittest

import numpy as np

from scope_bake import PositionMultiplexer
from scope_display import _validate_independent_trajectory


class FusionTrajectoryPolicyTests(unittest.TestCase):
    def test_fusion_default_clock_path_remains_allowed(self):
        self.assertIsNone(_validate_independent_trajectory(
            "bake", "fusion", None, None, True))

    def test_independent_clock_controls_are_explicitly_rejected_for_fusion(self):
        for geometry_samples, traversal_hz in ((512, None), (None, 30.0)):
            with self.subTest(geometry_samples=geometry_samples,
                              traversal_hz=traversal_hz):
                with self.assertRaisesRegex(ValueError, "component clocks"):
                    _validate_independent_trajectory(
                        "bake", "fusion", geometry_samples,
                        traversal_hz, True)

    def test_default_mux_output_and_weighted_transaction_rollback(self):
        vector = np.arange(24, dtype=np.float32).reshape(12, 2)
        raster = vector + 100
        stochastic = vector + 200
        mux = PositionMultiplexer()

        # The no-weight legacy selector is exact round-robin.
        default = mux.emit(vector, raster, stochastic)
        expected = np.stack((vector, raster, stochastic))[np.arange(12) % 3,
                                                            np.arange(12)]
        np.testing.assert_array_equal(default, expected)

        weighted = PositionMultiplexer()
        checkpoint = weighted.checkpoint()
        luminance_weights = {
            "v": np.linspace(1.0, 0.0, 12),
            "r": np.linspace(0.0, 1.0, 12),
            "s": np.full(12, 0.2),
        }
        candidate = weighted.emit(
            vector, raster, stochastic, weights=luminance_weights)
        weighted.restore(checkpoint)  # rejected output candidate
        retry = weighted.emit(
            vector, raster, stochastic, weights=luminance_weights)
        np.testing.assert_array_equal(candidate, retry)


if __name__ == "__main__":
    unittest.main()
