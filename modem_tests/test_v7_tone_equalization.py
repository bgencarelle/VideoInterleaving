"""Known low-bin tones as an opt-in M-path equalization reference."""
import unittest

import numpy as np

from animation_modem import v7


class V7ToneEqualizationTests(unittest.TestCase):
    def test_reference_recovers_packet_relative_m_gain_track(self):
        symbols = np.arange(v7.F)
        expected = .08*np.sin(2*np.pi*symbols/(v7.F-1))
        spectrum = np.zeros((v7.F, 65, 2), dtype=complex)
        spectrum[:, (0, 2), :] = 1e-4
        for tone_index, bin_index in enumerate(v7.PILOT_TONE_BINS):
            phase = .4*symbols + tone_index*.7
            spectrum[:, bin_index, :] = (
                np.exp(expected+1j*phase)[:, None])

        correction, diagnostic = v7._tone_reference_gain(
            spectrum, np.zeros((v7.F, 2)), np.zeros((v7.F, 2)))

        self.assertTrue(diagnostic['used'])
        for channel in range(2):
            np.testing.assert_allclose(
                correction[:, channel], expected-np.median(expected),
                atol=.002)
        self.assertTrue(all(
            entry['used'] for entry in diagnostic['channels']))

    def test_unavailable_tones_fall_back_without_gain_correction(self):
        spectrum = np.zeros((v7.F, 65, 2), dtype=complex)
        correction, diagnostic = v7._tone_reference_gain(
            spectrum, np.zeros((v7.F, 2)), np.zeros((v7.F, 2)))

        self.assertFalse(diagnostic['used'])
        np.testing.assert_array_equal(correction, np.zeros((v7.F, 2)))
        self.assertTrue(all(
            channel['reason'] == 'insufficient_tone_reference'
            for channel in diagnostic['channels']))


if __name__ == '__main__':
    unittest.main()
