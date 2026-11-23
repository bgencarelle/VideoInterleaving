import unittest

import numpy as np

from scope_out import Scope, redistribute_dwell, marker_window


class PhysicalDwellTests(unittest.TestCase):
    def test_zero_strength_is_exact_and_nonzero_preserves_budget_bounds(self):
        frame = np.array([
            [-0.8, -0.5], [-0.4, -0.5], [-0.2, -0.5],
            [-0.2, -0.1], [-0.2, 0.3], [0.2, 0.3], [0.8, 0.5],
        ], dtype=np.float32)
        np.testing.assert_array_equal(redistribute_dwell(frame), frame)
        adjusted = redistribute_dwell(frame, 0.8)
        self.assertEqual(adjusted.shape, frame.shape)
        np.testing.assert_allclose(adjusted.min(axis=0), frame.min(axis=0))
        np.testing.assert_allclose(adjusted.max(axis=0), frame.max(axis=0))
        self.assertFalse(np.array_equal(adjusted, frame))

    def test_x_only_uses_only_emitted_axis_and_trigger_is_separate(self):
        frame = np.array([
            [-0.8, -0.9], [-0.2, 0.9], [-0.1, -0.9],
            [0.1, 0.9], [0.8, -0.9],
        ], dtype=np.float32)
        adjusted = redistribute_dwell(frame, 0.7, x_only=True)
        self.assertEqual(len(adjusted), len(frame))
        self.assertEqual(float(adjusted[:, 0].min()), float(frame[:, 0].min()))
        self.assertEqual(float(adjusted[:, 0].max()), float(frame[:, 0].max()))
        marker = marker_window(8)
        picture_with_trigger = np.vstack((marker, adjusted))
        np.testing.assert_array_equal(picture_with_trigger[:8], marker)
        np.testing.assert_array_equal(picture_with_trigger[8:], adjusted)

    def test_invalid_strength_rejected(self):
        with self.assertRaises(ValueError):
            redistribute_dwell(np.zeros((4, 2)), 1.1)

    def test_scope_applies_dwell_only_to_picture_before_trigger(self):
        frame = np.array([
            [-0.8, -0.4], [-0.4, -0.4], [-0.2, -0.4],
            [-0.2, 0.0], [-0.2, 0.4], [0.2, 0.4], [0.8, 0.4],
        ], dtype=np.float32)
        with Scope(device="null", samples=len(frame), trigger=True,
                    physical_dwell=0.8) as scope:
            scope.show_frame(frame)
            output, _identity = scope._pending_record
            self.assertEqual(len(output), len(frame) + len(scope._marker))
            np.testing.assert_array_equal(output[:len(scope._marker)],
                                          scope._marker)
            self.assertFalse(np.array_equal(output[len(scope._marker):], frame))


if __name__ == "__main__":
    unittest.main()
