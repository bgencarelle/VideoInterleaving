"""One damaged stereo leg must not take the clean leg down with it.

The joint 2x2 equaliser reads energetic garbage on one input (a pitch- or
frequency-shifted leg, one-way leakage) as crosstalk and blends it into the
other; before the single-input retry and the corrected MMSE expansion, those
channels froze the picture: zero verified packets. Silence (a muted leg) was
always survivable, but decoded v3 to values of +-70.
"""
import unittest

import numpy as np
from scipy.signal import hilbert

from animation_modem import core
from animation_modem import engines as ENG
from animation_modem.imaging import image_values
from utilities.modem_v3_check import candidates_for, coders_for, frames_from


def shifted(x, hz, rate=48000):
    """Single-sideband frequency shift: every carrier lands on a wrong bin."""
    t = np.arange(len(x))/rate
    return np.real(hilbert(x)*np.exp(2j*np.pi*hz*t)).astype(np.float32)


class OneLegDamageTests(unittest.TestCase):
    PACKETS = 6

    def stream(self, codec, damage):
        engine = ENG.get_engine(codec)
        profile = engine.profiles[0]
        layout = engine.wire
        coder, _ = engine.coder_for(profile, layout)

        class Args:
            modem_dir = None
            frames = self.PACKETS
            stride = 1
        images, _ = frames_from(Args)
        audio = np.concatenate(
            [np.zeros((500, 2), np.float32)] +
            [engine.encode(image_values(im, coder.grids), coder, n+1, n+1,
                           self.PACKETS, profile=engine.profile_code(profile))
             for n, im in enumerate(images)])
        audio = damage(audio.copy())
        rx = engine.receiver(layout, coder, coders=coders_for(layout),
                             candidates=candidates_for(engine))
        out = [r for i in range(0, len(audio), 256) for r in rx.feed(audio[i:i+256])]
        return out + rx.flush()

    def verified(self, out):
        return sum(r.identity == 'verified_header' for r in out)

    def test_a_frequency_shifted_leg_no_longer_freezes_the_picture(self):
        for codec in ('v3', 'v5'):
            for leg in (0, 1):
                with self.subTest(codec=codec, leg=leg):
                    def damage(a, leg=leg):
                        a[:, leg] = shifted(a[:, leg], 90)
                        return a
                    out = self.stream(codec, damage)
                    self.assertEqual(self.verified(out), self.PACKETS)
                    self.assertTrue(all(r.extra.get('single_input') == 1-leg
                                        for r in out))

    def test_one_way_leakage_is_equalised(self):
        """Left bleeds into right at 0.6, right into left not at all. The
        old expansion swapped h01/h10, which only symmetric leakage hides."""
        def damage(a):
            a[:, 1] += .6*a[:, 0]
            return a
        for codec in ('v3', 'v5'):
            with self.subTest(codec=codec):
                self.assertEqual(self.verified(self.stream(codec, damage)),
                                 self.PACKETS)

    def test_a_muted_leg_decodes_to_sane_values(self):
        def damage(a):
            a[:, 0] = 0
            return a
        for codec in ('v3', 'v5'):
            with self.subTest(codec=codec):
                out = self.stream(codec, damage)
                self.assertEqual(self.verified(out), self.PACKETS)
                values = np.concatenate([r.values for r in out])
                self.assertLess(np.abs(values).max(), 2.0)

    def test_equaliser_matches_the_textbook_mmse_solve(self):
        """Drive _channel_equalizer with a known asymmetric 2x2 channel."""
        layout = ENG.get_engine('v3').wire
        coder, _ = ENG.get_engine('v3').coder_for('color-dct', layout)
        values = np.random.default_rng(0).uniform(-.8, .8, coder.source_count)
        from animation_modem import transport3 as V3
        audio = V3.encode(values, layout, coder, 1, 1, 1)
        body = audio[core.SYNC_LEN:layout.packet].reshape(
            layout.symbols, core.SYMBOL, 2)[:, core.CP-4:core.CP-4+core.N]
        mix = np.array([[1., 0.], [.5, .8]])            # [receive, transmit]
        spectrum = np.fft.rfft(body @ mix.T, n=core.N, axis=1)
        clean = np.fft.rfft(body, n=core.N, axis=1)
        equal, *_ = core._equalise_spectrum(spectrum, layout)
        want, *_ = core._equalise_spectrum(clean, layout)
        self.assertLess(np.sqrt(np.mean(np.abs(equal - want)**2)), .01)


if __name__ == '__main__':
    unittest.main()
