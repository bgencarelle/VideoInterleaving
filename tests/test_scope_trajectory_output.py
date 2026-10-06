"""Check time-based raster samples survive Scope's output boundary."""
import unittest

import numpy as np

from scope_bake import TraceEmitter
from scope_out import Scope, clip_for_trigger


class TrajectoryOutputTests(unittest.TestCase):
    def test_speed_changes_traversal_without_changing_canonical_detail(self):
        y, x = np.mgrid[:24, :32]
        luminance = ((x > 3) & (x < 28) & (y > 2) & (y < 21)).astype(np.float32)
        slow = TraceEmitter(48000, 800, geometry_samples=512,
                            traversal_hz=6.0, levels=(0, 1), grid=(12, 16))
        fast = TraceEmitter(48000, 400, geometry_samples=512,
                            traversal_hz=12.0, levels=(0, 1), grid=(12, 16))
        slow_frame = slow.emit(luminance)
        fast_frame = fast.emit(luminance)
        # Twice the speed covers the same path interval in half the time.
        np.testing.assert_allclose(fast_frame, slow_frame[::2], atol=2e-6)
        self.assertAlmostEqual(fast._phase, slow._phase)
        self.assertEqual(fast.geometry_samples, slow.geometry_samples)

    def test_trigger_output_preserves_timed_geometry_and_sample_budget(self):
        y, x = np.mgrid[:24, :32]
        luminance = ((x > 3) & (x < 28) & (y > 2) & (y < 21)).astype(np.float32)
        emitter = TraceEmitter(48000, 800, geometry_samples=512,
                               traversal_hz=12.0, levels=(0, 1),
                               grid=(12, 16))
        scope = Scope(device="null", samplerate=48000, samples=800,
                      trigger=True, blocksize=256)
        try:
            candidate = emitter.emit(luminance, commit=False)
            self.assertEqual(emitter._phase, 0.0)
            endpoint = scope.show_frame(candidate)
            emitter.accept(endpoint)
            buffer = np.empty((scope.trace_samples, 2), dtype=np.float32)
            # Scope adopts only after the previously active idle trace ends.
            scope._callback(buffer, len(buffer), None, None)
            scope._callback(buffer, len(buffer), None, None)
            np.testing.assert_array_equal(
                buffer[scope.yt_trigger_samples:], clip_for_trigger(candidate))
            self.assertEqual(len(buffer), 800 + scope.yt_trigger_samples)
            self.assertAlmostEqual(emitter._phase, 12.0 * 800 / 48000)
            self.assertEqual(scope.frames_adopted, 1)
        finally:
            scope.stream.close()


if __name__ == "__main__":
    unittest.main()
