"""Inter-channel phase, taken for free from the estimate we already solve.

The equaliser fits a 2x2 channel matrix per carrier and then inverts it away.
Before that inverse runs, the matrix says exactly how the path treated one
channel against the other: a phase ramp against frequency is a delay -- head
azimuth on tape, a long leg on a cable -- and the off-diagonal energy says how
far the two channels have been mixed toward each other.

Nothing here changes the wire. The measurement was always present in the
estimate; only the reading was missing. Note it has to be taken at the
estimate: once the 2x2 inverse has been applied, both channels read zero
residual and the relationship is gone.
"""
import unittest

import numpy as np

from animation_modem import transport3 as v3
from animation_modem.imaging import plane_shapes

LAYOUT = v3.WIRE


def delayed(audio, tau, channel=0):
    """An exact bandlimited fractional delay on one channel.

    A pure phase ramp. Linear interpolation is NOT usable here: it is a delay
    plus a lowpass, its group delay varies across the band, and the estimator
    correctly refuses to call that a single delay -- which looks like a failing
    test when it is the test's own signal that is wrong.
    """
    out = np.asarray(audio, float).copy()
    n = len(out)
    ramp = np.exp(-2j*np.pi*np.fft.rfftfreq(n)*tau)
    out[:, channel] = np.fft.irfft(np.fft.rfft(out[:, channel])*ramp, n)
    return out


class ChannelSkewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.coder = v3.SourceCoder(plane_shapes('lean-dct'))
        cls.values = np.random.default_rng(3).uniform(-.2, .2, cls.coder.count)
        cls.audio = np.concatenate([
            v3.encode(cls.values, LAYOUT, cls.coder, n, n, 4)
            for n in range(1, 5)])

    def decode(self, audio):
        rx = v3.Receiver(LAYOUT, self.coder)
        return rx.feed(np.asarray(audio, np.float32))+rx.flush()

    def first(self, audio):
        out = self.decode(audio)
        self.assertTrue(out, 'nothing decoded')
        return out[0].extra

    def test_azimuth_is_recovered_to_a_hundredth_of_a_sample(self):
        """Sub-sample delays are the point: tape azimuth is a fraction."""
        for tau in (0.0, .05, .10, .25, .50, 1.0, 3.0, -.35, -1.5):
            with self.subTest(tau=tau):
                extra = self.first(delayed(self.audio, tau))
                self.assertAlmostEqual(extra['skew_samples'], tau, delta=.01)
                self.assertLess(extra['skew_spread'], .2)

    def test_sign_says_which_channel_is_late(self):
        ahead = self.first(delayed(self.audio, .75, channel=0))['skew_samples']
        behind = self.first(delayed(self.audio, .75, channel=1))['skew_samples']
        self.assertGreater(ahead, .7)
        self.assertLess(behind, -.7)

    def test_a_clean_path_reads_zero_on_every_number(self):
        extra = self.first(self.audio)
        self.assertAlmostEqual(extra['skew_samples'], 0, delta=.001)
        self.assertAlmostEqual(extra['skew_spread'], 0, delta=.001)
        self.assertAlmostEqual(extra['crosstalk'], 0, delta=.001)

    def test_crosstalk_measures_how_far_the_path_has_collapsed(self):
        """0 is stereo, 0.5 is mono, 1 is fully swapped."""
        mono = self.audio.copy()
        mono[:, 0] = mono[:, 1] = (self.audio[:, 0]+self.audio[:, 1])/2
        self.assertAlmostEqual(self.first(mono)['crosstalk'], .5, delta=.01)
        swapped = self.audio[:, ::-1].copy()
        self.assertAlmostEqual(self.first(swapped)['crosstalk'], 1., delta=.01)
        bleed = self.audio.copy()
        bleed[:, 0] = .93*self.audio[:, 0] + .37*self.audio[:, 1]
        # Off-diagonal energy over the total: .37**2 / (.93**2 + .37**2 + 1).
        self.assertAlmostEqual(self.first(bleed)['crosstalk'], .0684, delta=.002)

    def test_mono_collapse_is_visible_before_the_picture_explains_why(self):
        """The diagnostic that earns its keep: a rank-1 path cannot carry the
        image at all, and this says so without having to infer it from noise."""
        mono = self.audio.copy()
        mono[:, 0] = mono[:, 1] = (self.audio[:, 0]+self.audio[:, 1])/2
        out = self.decode(mono)
        error = np.median([np.sqrt(np.mean((r.values-self.values)**2))
                           for r in out if r.values is not None])
        self.assertGreater(error, .05)            # picture is gone
        self.assertGreater(out[0].extra['crosstalk'], .4)   # and here is why

    def test_spread_flags_a_path_that_is_not_one_delay(self):
        """A single number is only honest if it admits when it does not fit."""
        clean = self.first(delayed(self.audio, .25))
        noisy = self.first(self.audio + np.random.default_rng(1)
                           .normal(0, .02, self.audio.shape))
        self.assertLess(clean['skew_spread'], .2)
        self.assertGreater(noisy['skew_spread'], 1.)

    def test_a_dead_channel_reports_nothing_rather_than_a_number(self):
        dead = self.audio.copy()
        dead[:, 1] = 0
        extra = self.first(dead)
        for key in ('skew_samples', 'skew_spread', 'crosstalk'):
            self.assertNotIn(key, extra)

    def test_measuring_it_changes_no_decoded_value(self):
        """It is a reading, not a correction. The picture must be untouched."""
        out = self.decode(self.audio)
        self.assertEqual([r.absolute for r in out], [1, 2, 3, 4])
        for r in out:
            self.assertLess(np.sqrt(np.mean((r.values-self.values)**2)), 1e-5)
        # And it survives a real delay rather than only tolerating one.
        for r in self.decode(delayed(self.audio, .5)):
            self.assertEqual(r.identity, 'verified_header')


if __name__ == '__main__':
    unittest.main()
