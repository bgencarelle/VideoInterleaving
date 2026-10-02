"""Emitted levels: the header's pulses are the loudest part of every packet,
close to full scale, and the picture body stays below them."""
import unittest

import numpy as np

from animation_modem import v7

TARGET = .1521/np.sqrt(1 + 10**(v7.CLOCK_REL_DB/10))
BODY = slice(v7.PULSE.SYNC_LEN, v7.PULSE.SYNC_LEN+v7.FRAME)


def _db(value):
    return 20*np.log10(value)


class EmittedLevelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = v7.load_model(TARGET, 'box')
        rng = np.random.default_rng(4)
        deviation = np.sqrt(cls.model.lam)
        # A quiet picture, an ordinary one and one far louder than the model.
        cls.packets = [
            v7.encode_pulse_frame_coeffs(
                cls.model, cls.model.mu+scale*deviation *
                rng.standard_normal(len(deviation)), counter,
                pilot_tones=True, eof_marker=True)
            for counter, scale in enumerate((.3, 1.0, 2.0, 4.0), 1)]

    def test_header_is_at_one_level_near_full_scale(self):
        peaks = [float(np.max(np.abs(packet[:v7.PULSE.SYNC_LEN])))
                 for packet in self.packets]
        for peak in peaks:
            self.assertLess(_db(peak), -.5)
            self.assertGreater(_db(peak), -1.3)
        self.assertLess(_db(max(peaks))-_db(min(peaks)), .5)

    def test_body_stays_below_the_header_and_nothing_passes_full_scale(self):
        for packet in self.packets:
            header = float(np.max(np.abs(packet[:v7.PULSE.SYNC_LEN])))
            body = float(np.max(np.abs(packet[BODY])))
            self.assertGreater(_db(header)-_db(body), 1.0)
            self.assertLess(float(np.max(np.abs(packet))), 1.0)

    def test_only_a_loud_body_is_scaled_down(self):
        def body_peak(scale):
            rng = np.random.default_rng(4)
            coefficients = self.model.mu+scale*np.sqrt(self.model.lam) * \
                rng.standard_normal(len(self.model.mu))
            packet = v7.encode_pulse_frame_coeffs(self.model, coefficients, 1)
            return float(np.max(np.abs(packet[BODY])))
        # A quiet picture's body goes out as coded, under the ceiling; loud
        # ones are held at it however loud they are.
        self.assertLess(body_peak(.3), .8*v7.BODY_PEAK)
        self.assertAlmostEqual(body_peak(4.0), v7.BODY_PEAK, delta=.01)
        self.assertAlmostEqual(body_peak(8.0), v7.BODY_PEAK, delta=.01)

    def test_end_marker_is_at_the_header_pulse_level(self):
        level = v7.emitted_pulse_level()
        quiet = v7.encode_pulse_frame_coeffs(
            self.model, self.model.mu, 1, eof_marker=True)
        marker = quiet[-v7.EOF_MARKER_LENGTH:, 0]
        np.testing.assert_allclose(np.abs(marker), level, rtol=.02)


if __name__ == '__main__':
    unittest.main()
