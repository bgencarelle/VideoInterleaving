"""The receiver's quality gates must not depend on the input level."""
import unittest

import numpy as np

from animation_modem import v7

TARGET = .1521/np.sqrt(1 + 10**(v7.CLOCK_REL_DB/10))


class LevelIndependenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = v7.load_model(TARGET, 'box')
        rng = np.random.default_rng(3)
        coeffs = cls.model.mu+rng.standard_normal(len(cls.model.mu)) * \
            np.sqrt(cls.model.lam)
        values = v7.values_from(cls.model, coeffs)
        packets = [v7.encode_pulse_frame(cls.model, values, counter)
                   for counter in range(1, 7)]
        audio = np.concatenate(packets).astype(np.float64)
        rng = np.random.default_rng(11)
        # Hiss scaled with the signal: the same link at every level.
        cls.audio = audio+.03*rng.standard_normal(audio.shape)

    def _decode(self, gain):
        results, _ = v7.decode_pulse_stream(
            self.model, (self.audio*gain).astype(np.float32),
            state=v7.PulseState(), sample_rate=48000)
        return [(result.status, max(result.diag['noise']))
                for result in results]

    def test_status_and_pilot_noise_do_not_follow_the_input_level(self):
        reference = self._decode(1.0)
        self.assertGreaterEqual(len(reference), 4)
        for gain in (.25, 4.0):
            with self.subTest(gain=gain):
                scaled = self._decode(gain)
                self.assertEqual([status for status, _ in scaled],
                                 [status for status, _ in reference])
                np.testing.assert_allclose(
                    [noise for _, noise in scaled],
                    [noise for _, noise in reference], rtol=.05)

    def test_pilot_reference_power_is_one_for_a_unity_link(self):
        H = np.zeros((v7.F, 65, 2, 2), complex)
        H[..., 0, :] = np.array([1, 1])/np.sqrt(2)
        H[..., 1, :] = np.array([1, -1])/np.sqrt(2)
        np.testing.assert_allclose(v7._pilot_reference_power(H), [1.0, 1.0])
        np.testing.assert_allclose(v7._pilot_reference_power(3*H), [9.0, 9.0])


if __name__ == '__main__':
    unittest.main()
