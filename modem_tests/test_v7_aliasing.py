"""Harmonic alias folding and polyphase anti-alias regression checks."""
import unittest

import numpy as np
from scipy.signal import resample_poly


class V7AliasingTests(unittest.TestCase):
    def test_harmonics_fold_to_their_nyquist_aliases(self):
        input_rate, output_rate = 192_000, 48_000
        harmonics = ((28_000, 1.0, 20_000),
                     (56_000, .25, 8_000),
                     (84_000, .125, 12_000))
        time = np.arange(input_rate, dtype=np.float64)/input_rate
        signal = sum(amplitude*np.sin(2*np.pi*frequency*time)
                     for frequency, amplitude, _alias in harmonics)
        naive = signal[::input_rate//output_rate]
        naive_fft = np.fft.rfft(naive)
        frequencies = np.fft.rfftfreq(len(naive), 1/output_rate)

        for _frequency, amplitude, alias in harmonics:
            index = int(np.argmin(np.abs(frequencies-alias)))
            measured = 2*abs(naive_fft[index])/len(naive)
            with self.subTest(alias_hz=alias):
                self.assertAlmostEqual(measured, amplitude, places=5)

    def test_polyphase_resampling_suppresses_folded_harmonics(self):
        input_rate, output_rate = 192_000, 48_000
        harmonics = ((28_000, 1.0, 20_000),
                     (56_000, .25, 8_000),
                     (84_000, .125, 12_000))
        time = np.arange(input_rate, dtype=np.float64)/input_rate
        signal = sum(amplitude*np.sin(2*np.pi*frequency*time)
                     for frequency, amplitude, _alias in harmonics)
        filtered = resample_poly(signal, output_rate, input_rate)
        filtered_fft = np.fft.rfft(filtered)
        frequencies = np.fft.rfftfreq(len(filtered), 1/output_rate)

        for _frequency, amplitude, alias in harmonics:
            index = int(np.argmin(np.abs(frequencies-alias)))
            measured = 2*abs(filtered_fft[index])/len(filtered)
            with self.subTest(alias_hz=alias):
                self.assertLess(measured, amplitude*.01)


if __name__ == '__main__':
    unittest.main()
