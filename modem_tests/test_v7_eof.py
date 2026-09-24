"""Opt-in V7 end-marker wire and receiver path."""
import unittest

import numpy as np
from PIL import Image
from scipy.signal import resample_poly

from animation_modem import v7


FRAME_COUNT = 4


class V7EOFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        target = .1521/np.sqrt(1 + 10**(v7.CLOCK_REL_DB/10))
        cls.model = v7.load_model(target, 'nearest')
        with Image.open(v7.REFERENCE_FIXTURE) as source:
            cls.values = v7.image_values(
                v7.prepare_image(source, 'nearest'), cls.model.coder.grids,
                'nearest')
        cls.baseline_wire = v7.encode_pulse_stream(
            cls.model, [cls.values]*FRAME_COUNT,
            pilot_tones=False, eof_marker=False)
        cls.eof_wire = v7.encode_pulse_stream(
            cls.model, [cls.values]*FRAME_COUNT,
            pilot_tones=False, eof_marker=True)

    def _packet_starts(self, stream):
        return v7.pulse_frame_starts(stream)

    def test_schmitt_marker_matches_the_fitted_marker(self):
        # The counted (Schmitt + run counter) marker must find the same EOF,
        # at the same place, as the earlier least-squares fit -- on clean,
        # noisy, saturated, polarity-inverted and slowed/sped streams.
        rng = np.random.default_rng(7)
        clean = self.eof_wire.astype(np.float32)
        streams = {
            'clean': clean,
            'hiss': clean+rng.normal(0, 10**(-38/20), clean.shape).astype(np.float32),
            'saturated': (np.tanh(2*clean)/np.tanh(2)).astype(np.float32),
            'inverted': -clean,
            'slow': v7.speed_pulse_stream(self.eof_wire, .8),
            'fast': v7.speed_pulse_stream(self.eof_wire, 1.5),
        }
        for name, stream in streams.items():
            mono = v7._mono(stream)
            starts = self._packet_starts(stream)
            self.assertGreaterEqual(len(starts), FRAME_COUNT-1, name)
            for frame_start, scale, _ in starts:
                fitted = v7._measure_eof_marker_fit(mono, frame_start, scale)
                counted = v7._measure_eof_marker(mono, frame_start, scale)
                with self.subTest(stream=name, frame_start=round(frame_start)):
                    self.assertEqual(fitted is None, counted is None)
                    if fitted is None:
                        continue
                    for key in ('start', 'end', 'scale', 'packet_scale'):
                        self.assertAlmostEqual(counted[key], fitted[key],
                                               places=6)
                    self.assertEqual(counted['polarity'], fitted['polarity'])

    def test_counted_marker_ignores_chatter_inside_the_schmitt_band(self):
        # Small ringing that crosses zero near an edge breaks runs of raw
        # sign changes but never leaves the hysteresis band.
        stream = self.eof_wire[:, 0].astype(np.float64).copy()
        frame_start, scale, _ = self._packet_starts(self.eof_wire)[0]
        edge = int(frame_start+v7.EOF_MARKER_OFFSET*scale+v7.EOF_MARKER_EDGES[1])
        stream[edge-1:edge+2] = (.01, -.01, .01)
        marker = v7._measure_eof_marker(stream, frame_start, scale)
        self.assertIsNotNone(marker)
        self.assertAlmostEqual(marker['packet_scale'], 1.0, places=3)

    def test_latest_only_validates_each_committed_marker_once(self):
        calls = []
        original = v7._measure_eof_marker

        def counting(samples, frame_start, scale):
            result = original(samples, frame_start, scale)
            calls.append(result is not None)
            return result
        v7._measure_eof_marker = counting
        try:
            window = self.eof_wire[:int(2.3*v7.PULSE_FRAME)].astype(np.float32)
            results, _ = v7.decode_pulse_stream(
                self.model, window, latest_only=True, frame_boundary='eof')
        finally:
            v7._measure_eof_marker = original
        self.assertEqual(len(results), 1)
        self.assertEqual(sum(calls), 1)          # one successful validation

    def test_marker_reuses_the_guard_and_is_opt_in(self):
        self.assertEqual(len(self.eof_wire), len(self.baseline_wire))
        guard = self.baseline_wire[-v7.EOF_MARKER_LENGTH:]
        self.assertEqual(float(np.max(np.abs(guard))), 0.0)
        expected = np.concatenate([
            np.full(run, level, np.float32)
            for run, level in zip(v7.EOF_MARKER_RUNS, v7.EOF_MARKER_LEVELS)
        ])*v7.EOF_MARKER_LEVEL
        np.testing.assert_array_equal(
            self.eof_wire[-v7.EOF_MARKER_LENGTH:, 0], expected)
        np.testing.assert_array_equal(
            self.eof_wire[-v7.EOF_MARKER_LENGTH:, 1], expected)
        self.assertEqual(float(np.max(np.abs(
            self.eof_wire[-32:-v7.EOF_MARKER_LENGTH]))), 0.0)

    def test_eof_receiver_commits_final_packet_without_next_header(self):
        baseline, _ = v7.decode_pulse_stream(
            self.model, self.baseline_wire, pilot_timing='baseline',
            frame_boundary='baseline')
        eof, info = v7.decode_pulse_stream(
            self.model, self.eof_wire, frame_boundary='eof')

        self.assertEqual(len(baseline), FRAME_COUNT-1)
        self.assertEqual(len(eof), FRAME_COUNT)
        self.assertEqual(info['eof_markers_validated'], FRAME_COUNT)
        self.assertEqual([result.counter for result in eof],
                         list(range(1, FRAME_COUNT+1)))
        self.assertTrue(all(result.diag['metadata_valid'] for result in eof))
        for result in eof:
            decoded = v7.values_from(self.model, result.coeffs)
            self.assertLess(float(np.sqrt(np.mean(
                np.square(decoded-self.values)))), .10)

    def test_eof_receiver_rejects_missing_or_damaged_final_marker(self):
        no_marker, _ = v7.decode_pulse_stream(
            self.model, self.baseline_wire[:v7.PULSE_FRAME],
            frame_boundary='eof')
        self.assertEqual(no_marker, [])

        damaged = self.eof_wire.copy()
        damaged[-v7.EOF_MARKER_LENGTH:] = 0
        results, info = v7.decode_pulse_stream(
            self.model, damaged, frame_boundary='eof')
        self.assertEqual(len(results), FRAME_COUNT-1)
        self.assertEqual(info['eof_markers_validated'], FRAME_COUNT-1)

    def test_eof_receiver_stops_at_a_damaged_interior_marker(self):
        damaged = self.eof_wire.copy()
        second_end = 2*v7.PULSE_FRAME
        damaged[second_end-v7.EOF_MARKER_LENGTH:second_end] = 0
        results, info = v7.decode_pulse_stream(
            self.model, damaged, frame_boundary='eof')

        self.assertEqual([result.counter for result in results], [1])
        self.assertEqual(info['eof_markers_validated'], 1)

    def test_truncated_marker_does_not_commit_final_packet(self):
        truncated = self.eof_wire[:-16]
        results, info = v7.decode_pulse_stream(
            self.model, truncated, frame_boundary='eof')
        self.assertEqual(len(results), FRAME_COUNT-1)
        self.assertEqual(info['eof_markers_validated'], FRAME_COUNT-1)

    def test_latest_only_can_commit_a_single_eof_packet(self):
        results, info = v7.decode_pulse_stream(
            self.model, self.eof_wire[:v7.PULSE_FRAME], latest_only=True,
            frame_boundary='eof')
        self.assertEqual(len(results), 1)
        self.assertEqual(info['eof_markers_validated'], 1)
        self.assertTrue(results[0].diag['metadata_valid'])

    def test_eof_clock_works_at_device_rates_and_playback_speeds(self):
        for sample_rate, speed in ((48000, .75), (48000, 1.5),
                                   (96000, 1.0), (96000, 1.5)):
            with self.subTest(sample_rate=sample_rate, speed=speed):
                capture = resample_poly(
                    self.eof_wire, sample_rate, v7.RATE, axis=0)
                capture = v7.speed_pulse_stream(
                    capture, speed, rate=sample_rate)
                results, info = v7.decode_pulse_stream(
                    self.model, capture, sample_rate=sample_rate,
                    frame_boundary='eof')
                self.assertEqual(len(results), FRAME_COUNT)
                self.assertEqual(info['eof_markers_validated'], FRAME_COUNT)
                self.assertTrue(all(
                    result.diag['metadata_valid'] for result in results))

    def test_eof_tone_reference_equalization_is_wired_through_receiver(self):
        toned_wire = v7.encode_pulse_stream(
            self.model, [self.values]*FRAME_COUNT,
            pilot_tones=True, eof_marker=True)
        results, info = v7.decode_pulse_stream(
            self.model, toned_wire, frame_boundary='eof',
            pilot_timing='tone-seeded', tone_equalization='m-reference')

        self.assertEqual(len(results), FRAME_COUNT)
        self.assertEqual(info['eof_markers_validated'], FRAME_COUNT)
        self.assertTrue(all(result.diag['metadata_valid'] for result in results))
        self.assertTrue(all(
            result.diag['tone_equalization']['mode'] == 'm-reference'
            for result in results))


if __name__ == '__main__':
    unittest.main()
