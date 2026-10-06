import unittest
from unittest import mock

import numpy as np

from scope_trajectory import VectorTrajectory


class VectorTrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.path = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]],
                             dtype=np.float32)

    def test_timed_waveform_is_invariant_to_chunking_and_dac_rate(self):
        a = VectorTrajectory(geometry_samples=256, traversal_hz=2.0)
        b = VectorTrajectory(geometry_samples=256, traversal_hz=2.0)
        a0 = a.emit(self.path, 480, 48000)
        a.accept()
        a1 = a.emit(self.path, 480, 48000)
        b0 = b.emit(self.path, 960, 96000)
        b.accept()
        b1 = b.emit(self.path, 960, 96000)
        np.testing.assert_allclose(np.concatenate((a0, a1)),
                                   np.concatenate((b0, b1))[::2], atol=2e-6)

    def test_traversal_speed_is_independent_of_geometry_and_output_budget(self):
        slow = VectorTrajectory(geometry_samples=128, traversal_hz=1.0)
        fast = VectorTrajectory(geometry_samples=128, traversal_hz=3.0)
        a = slow.emit(self.path, 2400, 48000)
        b = fast.emit(self.path, 800, 48000)
        np.testing.assert_allclose(b, a[::3], atol=2e-6)
        self.assertEqual(slow.geometry_samples, fast.geometry_samples)
        self.assertAlmostEqual((slow._candidate or 0), 2400 / 48000)
        self.assertAlmostEqual((fast._candidate or 0), 3 * 800 / 48000)

    def test_retry_reset_and_acceptance_control_phase(self):
        trajectory = VectorTrajectory(geometry_samples=64, traversal_hz=4.0)
        candidate = trajectory.emit(self.path, 600, 48000)
        trajectory.reject()
        self.assertEqual(trajectory.phase, 0.0)
        np.testing.assert_array_equal(candidate,
                                      trajectory.emit(self.path, 600, 48000))
        trajectory.accept()
        self.assertAlmostEqual(trajectory.phase, 0.05)
        trajectory.emit(self.path, 100, 48000)
        trajectory.reset()
        self.assertEqual(trajectory.phase, 0.0)
        self.assertFalse(trajectory.accept())

    def test_partitioned_chunks_match_one_whole_duration(self):
        whole = VectorTrajectory(256, 2.0)
        parts = VectorTrajectory(256, 2.0)
        expected = whole.emit(self.path, 960, 48000)
        first = parts.emit(self.path, 480, 48000)
        parts.accept()
        second = parts.emit(self.path, 480, 48000)
        np.testing.assert_allclose(np.concatenate((first, second)), expected,
                                   atol=2e-6)

    def test_geometry_only_draws_full_cycle_instead_of_parked_point(self):
        trajectory = VectorTrajectory(64)
        frame = trajectory.emit(self.path, 256, 48000)
        np.testing.assert_allclose(frame[::64], self.path, atol=2e-6)
        self.assertGreater(float(np.ptp(frame[:, 0])), 1.9)

    def test_vector_application_transaction_retries_without_handoff_deformation(self):
        from scope_display import _emit, _execute_frame_transaction
        from scope_out import rasterize

        class QueueScope:
            samplerate = 48000
            samples_per_frame = 480
            frames_accepted = 0
            accept_output = False

            def show_frame(self, frame, **kwargs):
                self.frame = frame.copy()
                self.kwargs = kwargs
                if self.accept_output:
                    self.frames_accepted += 1
                return frame[-1].copy()

        scope = QueueScope()
        trajectory = VectorTrajectory(64, 2.0)
        def render():
            return _emit(scope, None, None, 0, "vector", {}, "alternate",
                         2.2, .02, 1., None, 0., True, 0., 1, None,
                         beam_start=np.array([.2, -.4]),
                         vector_trajectory=trajectory)
        def commit(endpoint):
            trajectory.accept()
        with mock.patch("scope_display.merge", return_value=[self.path]):
            _execute_frame_transaction(scope, render, [], lambda state: None, commit)
            first = scope.frame.copy()
            self.assertEqual(trajectory.phase, 0.)
            scope.accept_output = True
            _execute_frame_transaction(scope, render, [], lambda state: None, commit)
        np.testing.assert_array_equal(scope.frame, first)
        expected = VectorTrajectory(64, 2.).emit(rasterize([self.path], 64), 480, 48000)
        np.testing.assert_array_equal(scope.frame, expected)
        self.assertNotIn("handoff", scope.kwargs)
        self.assertAlmostEqual(trajectory.phase, .02)


if __name__ == "__main__":
    unittest.main()
