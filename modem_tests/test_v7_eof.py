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
        cls.eof_wire = v7.encode_pulse_stream(
            cls.model, [cls.values]*FRAME_COUNT,
            pilot_tones=False)

    def _packet_starts(self, stream):
        return v7.pulse_frame_starts(stream)

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
                self.model, window, latest_only=True)
        finally:
            v7._measure_eof_marker = original
        self.assertEqual(len(results), 1)
        self.assertEqual(sum(calls), 1)          # one successful validation

    def test_marker_sits_in_the_guard_of_every_packet(self):
        self.assertEqual(len(self.eof_wire), FRAME_COUNT*v7.PULSE_FRAME)
        # The mark is shaped like the header and peaks at the header's level.
        expected = v7._shaped_eof_marker()[-v7.EOF_MARKER_LENGTH:, 0]
        for packet in self.eof_wire.reshape(FRAME_COUNT, v7.PULSE_FRAME, 2):
            for channel in (0, 1):
                np.testing.assert_allclose(
                    packet[-v7.EOF_MARKER_LENGTH:, channel], expected,
                    atol=.02)
        self.assertAlmostEqual(float(np.max(np.abs(expected))),
                               v7.emitted_pulse_level(), delta=1e-6)

    def test_eof_receiver_commits_final_packet_without_next_header(self):
        eof, info = v7.decode_pulse_stream(
            self.model, self.eof_wire)

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
        one = self.eof_wire[:v7.PULSE_FRAME].copy()
        one[-32:] = 0                         # the guard with no marker in it
        no_marker, _ = v7.decode_pulse_stream(self.model, one)
        self.assertEqual(no_marker, [])

        damaged = self.eof_wire.copy()
        damaged[-v7.EOF_MARKER_LENGTH:] = 0
        results, info = v7.decode_pulse_stream(
            self.model, damaged)
        self.assertEqual(len(results), FRAME_COUNT-1)
        self.assertEqual(info['eof_markers_validated'], FRAME_COUNT-1)

    def _toned_wire(self):
        return v7.encode_pulse_stream(
            self.model, [self.values]*FRAME_COUNT, pilot_tones=True)

    def test_a_packet_without_its_mark_ends_where_its_tones_say(self):
        damaged = self._toned_wire()
        second_end = 2*v7.PULSE_FRAME
        damaged[second_end-v7.EOF_MARKER_LENGTH:second_end] = 0
        results, info = v7.decode_pulse_stream(
            self.model, damaged, pilot_timing='tone-seeded')

        self.assertEqual([result.counter for result in results],
                         list(range(1, FRAME_COUNT+1)))
        self.assertEqual(info['eof_markers_validated'], FRAME_COUNT-1)
        self.assertEqual(info['tone_witnesses'], 1)
        witness = results[1].diag['eof_marker']
        self.assertEqual(witness['witness'], 'tones')
        self.assertAlmostEqual(witness['end'], second_end, delta=.5)
        decoded = v7.values_from(self.model, results[1].coeffs)
        self.assertLess(float(np.sqrt(np.mean(
            np.square(decoded-self.values)))), .10)

    def test_the_tone_end_needs_nothing_after_the_packet(self):
        # One packet, its mark removed, and no audio after its last sample.
        one = self._toned_wire()[:v7.PULSE_FRAME].copy()
        one[-v7.EOF_MARKER_LENGTH:] = 0
        for speed in (1.0, .8, 1.5):
            stream = one if speed == 1.0 else v7.speed_pulse_stream(one, speed)
            results, info = v7.decode_pulse_stream(
                self.model, stream, pilot_timing='tone-seeded')
            with self.subTest(speed=speed):
                self.assertEqual(len(results), 1)
                self.assertEqual(info['tone_witnesses'], 1)
                self.assertTrue(results[0].diag['metadata_valid'])

    def test_no_mark_and_no_tones_is_no_packet_even_before_a_header(self):
        # Without tones the next header used to end the packet. It no longer
        # does: a packet ends at its own mark or by its own tones.
        damaged = self.eof_wire.copy()
        second_end = 2*v7.PULSE_FRAME
        damaged[second_end-v7.EOF_MARKER_LENGTH:second_end] = 0
        results, info = v7.decode_pulse_stream(self.model, damaged)
        self.assertEqual(info['tone_witnesses'], 0)
        self.assertEqual(info['eof_markers_validated'], FRAME_COUNT-1)
        self.assertEqual(results[1].status, 'lost')

    def test_packet_without_any_endpoint_is_held_and_decoding_resumes(self):
        damaged = self.eof_wire.copy()
        second_end = 2*v7.PULSE_FRAME
        # The second packet's mark and the third packet's header are gone.
        damaged[second_end-v7.EOF_MARKER_LENGTH:second_end+300] = 0
        results, _info = v7.decode_pulse_stream(
            self.model, damaged)

        self.assertEqual(results[0].status, 'received')
        # Either no endpoint is found, or the only one is a later packet's,
        # which the body never reaches; the packet is held either way.
        self.assertEqual(results[1].status, 'lost')
        self.assertTrue(results[1].diag['held'])
        self.assertEqual(results[-1].status, 'received')

    def test_truncated_marker_does_not_commit_final_packet(self):
        truncated = self.eof_wire[:-16]
        results, info = v7.decode_pulse_stream(
            self.model, truncated)
        self.assertEqual(len(results), FRAME_COUNT-1)
        self.assertEqual(info['eof_markers_validated'], FRAME_COUNT-1)

    def test_latest_only_can_commit_a_single_eof_packet(self):
        results, info = v7.decode_pulse_stream(
            self.model, self.eof_wire[:v7.PULSE_FRAME], latest_only=True)
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
                    self.model, capture, sample_rate=sample_rate)
                self.assertEqual(len(results), FRAME_COUNT)
                self.assertEqual(info['eof_markers_validated'], FRAME_COUNT)
                self.assertTrue(all(
                    result.diag['metadata_valid'] for result in results))

    def test_eof_tone_reference_equalization_is_wired_through_receiver(self):
        toned_wire = v7.encode_pulse_stream(
            self.model, [self.values]*FRAME_COUNT,
            pilot_tones=True)
        results, info = v7.decode_pulse_stream(
            self.model, toned_wire,
            pilot_timing='tone-seeded', tone_equalization='m-reference')

        self.assertEqual(len(results), FRAME_COUNT)
        self.assertEqual(info['eof_markers_validated'], FRAME_COUNT)
        self.assertTrue(all(result.diag['metadata_valid'] for result in results))
        self.assertTrue(all(
            result.diag['tone_equalization']['mode'] == 'm-reference'
            for result in results))


if __name__ == '__main__':
    unittest.main()
