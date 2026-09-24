"""Pulse-anchored within-packet timing-map prototype."""
from pathlib import Path
import unittest

import numpy as np
from PIL import Image
from scipy.signal import butter, resample_poly, sosfilt

from animation_modem import v7


FIXTURE = Path(__file__).parent/'fixtures/v7_reference_face.png'
TARGET = .1521/np.sqrt(1 + 10**(v7.CLOCK_REL_DB/10))


class V7PulseWarpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = v7.build_model(FIXTURE, TARGET, encode_filter='nearest')
        with Image.open(FIXTURE) as image:
            cls.values = v7.image_values(
                v7.prepare_image(image, 'nearest'),
                cls.model.coder.grids, 'nearest')

    def _fast_flutter_capture(self, packets=8):
        wire = v7.encode_pulse_stream(self.model, [self.values]*packets)
        source = resample_poly(wire, 2, 1, axis=0).astype(np.float32)
        source_indexes = np.arange(len(source), dtype=float)
        time = source_indexes/96_000
        speed_error = (
            .0045*np.sin(2*np.pi*.55*time+.4) +
            .0012*np.sin(2*np.pi*7.3*time+1.1) +
            .001*np.sin(2*np.pi*25*time) +
            .0005*np.sin(2*np.pi*60*time+.4))
        source_position = source_indexes+np.cumsum(speed_error)
        capture = np.column_stack([
            np.interp(source_position, source_indexes, source[:, channel],
                      left=0, right=0)
            for channel in range(2)]).astype(np.float32)
        return capture, source_position, source_indexes

    def test_map_is_linear_when_playback_speed_is_steady(self):
        start, scale = 123.25, 2.0
        following = start+v7.PULSE_FRAME*scale
        positions = np.asarray((v7.PULSE.SYNC_LEN,
                                v7.PULSE.SYNC_LEN+v7.FRAME-1))
        mapped, local_scale = v7._pulse_warp_map(
            start, following, scale, scale, 1.0, 1.0, positions)
        np.testing.assert_allclose(mapped, start+positions*scale, atol=1e-9)
        np.testing.assert_allclose(local_scale, scale, atol=1e-12)

    def test_pulse_anchors_reduce_known_fast_flutter_position_error(self):
        capture, source_position, capture_indexes = \
            self._fast_flutter_capture()
        starts = v7.pulse_frame_starts(capture, sample_rate=96_000)
        self.assertGreaterEqual(len(starts), 7)
        symbol_centers = (np.arange(v7.F)*v7.SYM + v7.WIN +
                          (v7.N-1)/2)
        packet_coordinates = v7.PULSE.SYNC_LEN+symbol_centers
        linear_errors, warped_errors = [], []
        for packet, (current, following) in enumerate(zip(starts, starts[1:])):
            start, start_scale, start_confidence = current
            next_start, next_scale, next_confidence = following
            truth = np.interp(
                2*(packet*v7.PULSE_FRAME+packet_coordinates),
                source_position, capture_indexes)
            average_scale = (next_start-start)/v7.PULSE_FRAME
            linear = (start+v7.PULSE.SYNC_LEN*start_scale+
                      symbol_centers*average_scale)
            warped, _ = v7._pulse_warp_map(
                start, next_start, start_scale, next_scale,
                start_confidence, next_confidence, packet_coordinates)
            self.assertIsNotNone(warped)
            linear_errors.extend((linear-truth).tolist())
            warped_errors.extend((warped-truth).tolist())

        linear_rms = float(np.sqrt(np.mean(np.square(linear_errors))))
        warped_rms = float(np.sqrt(np.mean(np.square(warped_errors))))
        self.assertLess(warped_rms, .8*linear_rms)

    def test_receiver_keeps_warp_opt_in_and_decodes_flutter_capture(self):
        capture, _, _ = self._fast_flutter_capture()
        baseline, _ = v7.decode_pulse_stream(
            self.model, capture, sample_rate=96_000)
        warped, _ = v7.decode_pulse_stream(
            self.model, capture, sample_rate=96_000,
            pulse_timing='pulse-warp')

        self.assertTrue(all('pulse_timing' not in result.diag
                            for result in baseline))
        self.assertEqual(len(warped), len(baseline))
        self.assertGreater(
            sum(result.diag['pulse_timing']['mode_applied'] == 'pulse-warp'
                for result in warped), 0)
        self.assertGreaterEqual(
            sum(result.status != 'lost' for result in warped),
            sum(result.status != 'lost' for result in baseline))

    def test_steady_speed_skips_unneeded_warp(self):
        wire = v7.encode_pulse_stream(self.model, [self.values]*6)
        capture = v7.speed_pulse_stream(wire, 1.0, rate=96_000)
        decoded, _ = v7.decode_pulse_stream(
            self.model, capture, sample_rate=96_000,
            pulse_timing='pulse-warp')

        self.assertGreater(len(decoded), 0)
        self.assertTrue(all(
            result.diag['pulse_timing']['mode_applied'] == 'baseline' and
            result.diag['pulse_timing']['reason'] == 'pulse_scales_near_average'
            for result in decoded))

    def test_static_filter_bias_falls_back_to_packet_average_scale(self):
        wire = v7.encode_pulse_stream(self.model, [self.values]*8)
        capture = resample_poly(wire, 2, 1, axis=0).astype(np.float32)
        capture = sosfilt(
            butter(4, 300, btype='highpass', fs=96_000, output='sos'),
            capture, axis=0).astype(np.float32)
        decoded, _ = v7.decode_pulse_stream(
            self.model, capture, sample_rate=96_000,
            pulse_timing='pulse-warp')

        self.assertGreater(len(decoded), 0)
        self.assertTrue(all(
            result.diag['pulse_timing']['mode_applied'] == 'baseline' and
            result.diag['pulse_timing']['reason'] ==
            'local_scale_interval_mismatch'
            for result in decoded))


if __name__ == '__main__':
    unittest.main()
