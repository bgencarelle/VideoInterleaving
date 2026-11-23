"""Synthetic PortAudio stream-clock mapping and callback adoption coverage."""
from types import SimpleNamespace
import unittest

import numpy as np

from scope_out import Scope, predict_dac_monotonic_ns


class DacTimePredictionTests(unittest.TestCase):
    def test_maps_stream_dac_clock_to_host_monotonic_with_sample_offset(self):
        info = SimpleNamespace(currentTime=12.5, outputBufferDacTime=12.51)
        predicted = predict_dac_monotonic_ns(info, 1_000_000_000, 480, 48_000)
        self.assertEqual(predicted, 1_020_000_000)

    def test_missing_or_invalid_portaudio_timestamps_are_unavailable(self):
        self.assertIsNone(predict_dac_monotonic_ns(None, 10, 0, 48_000))
        info = SimpleNamespace(currentTime=float("nan"), outputBufferDacTime=2)
        self.assertIsNone(predict_dac_monotonic_ns(info, 10, 0, 48_000))
        outlier = SimpleNamespace(currentTime=1, outputBufferDacTime=5)
        self.assertIsNone(predict_dac_monotonic_ns(outlier, 10, 0, 48_000))

    def test_callback_records_synthetic_dac_reservation_at_adoption_boundary(self):
        scope = Scope(device="null", samplerate=48_000, samples=64,
                      trigger=False, channel_pair=(24, 25))
        frame = np.zeros((64, 2), dtype=np.float32)
        scope.show_frame(frame, identity=(0, "test", 1, 0, 0, 0, 1,
                                          "synthetic", 1, 2, 3, 4, None))
        out = np.zeros((64, 2), dtype=np.float32)
        info = SimpleNamespace(currentTime=40.0, outputBufferDacTime=40.01)
        scope._callback(out, 64, info, None)
        self.assertEqual(scope.frames_adopted, 1)
        self.assertIsNotNone(scope.last_adopted_dac_prediction_ns)
        self.assertEqual(len(scope.dac_adoption_snapshot()), 1)
        self.assertIsNotNone(scope.dac_adoption_snapshot()[0][2])
        # The output-block offset contributes another full 64-sample trace.
        self.assertGreaterEqual(
            scope.last_adopted_dac_schedule_offset_ms, 11.3)
        self.assertLessEqual(
            scope.last_adopted_dac_schedule_offset_ms, 12.0)
        scope.stream.close()


if __name__ == "__main__":
    unittest.main()
