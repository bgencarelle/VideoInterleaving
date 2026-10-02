"""Companded fold guests on Fold 500 and the mono colour profiles.

Aspect Fold 500's test is in test_v7_aspect_fold.py. These pin the same
behaviour for the stereo Fold 500 table, mono-colour-500 and
aspect-mono-500: guests far past the old +-2.5 clip survive a clean round
trip, a little symbol noise only shrinks them, past guest_noise_max the
guests are dropped while the hosts are still read as steps, and past
noise_max the hosts are read plainly too.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT/'test_modem_v7', ROOT/'tools'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from animation_modem import v7                                           # noqa: E402
from animation_modem.v7_fold import Fold500                              # noqa: E402
from common import TARGET                                                # noqa: E402
import folding                                                           # noqa: E402
import live_fold                                                         # noqa: E402
from live_fold import LiveFold                                           # noqa: E402
from aspect_mono import AspectMonoWire                                   # noqa: E402
import mono_video                                                        # noqa: E402
from mono_video import MonoColourFoldWire                                # noqa: E402


def _round_trip(codec, values, sigma, seed=3):
    """Decode `values` after noise of `sigma` (symbol units) on the hosts."""
    model = codec.model
    coeffs = codec.encode_coefficients(values)
    rng = np.random.default_rng(seed)
    symbol = (coeffs[codec.hosts]-model.mu[codec.hosts])/codec.sd_host
    conf = np.ones(len(model.mu))
    xhat = np.zeros(len(model.mu))
    xhat[codec.hosts] = codec.sd_host*(
        symbol+sigma/np.sqrt(codec.power)*rng.standard_normal(codec.M))
    return codec.decode(coeffs, xhat, conf, fallback=True,
                        metadata_confirmed=True)


class CompandedFoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = v7.load_model(TARGET, 'box')

    def _check(self, codec, settings):
        self.assertEqual(codec.compand, (settings['limit'], settings['mu']))
        self.assertEqual(codec.D, settings['step'])
        self.assertEqual(codec.guest_noise_max, settings['guest_noise_max'])
        table = codec.table()
        self.assertEqual(table['compand'], [settings['limit'], settings['mu']])
        self.assertEqual(table['guest_noise_max'], settings['guest_noise_max'])
        model = codec.model
        # Guests of 0.5 to 10 model standard deviations, hosts on the steps.
        full = np.zeros(codec.grid.off[-1])
        full[codec.kept] = model.mu
        data = slice(0, codec.M-codec.signature)
        sizes = np.linspace(.5, 10, codec.M)*np.where(
            np.arange(codec.M) % 2, 1, -1)
        self.assertGreater(float(np.mean(np.abs(sizes) > folding.U_CLIP)), .7)
        full[codec.guests] = sizes*codec.sd_guest
        full[codec.kept[codec.hosts]] += codec.sd_host*codec.D*(
            np.arange(codec.M) % 5-2)
        values = codec.grid.inverse(full)
        hosts = codec.kept[codec.hosts]
        clean = _round_trip(codec, values, 0.0)
        np.testing.assert_allclose(clean[codec.guests][data],
                                   full[codec.guests][data], atol=1e-6)
        np.testing.assert_allclose(clean[hosts][data], full[hosts][data],
                                   atol=1e-6)
        # A little noise: guests shrink slightly and stay close.
        noisy = _round_trip(codec, values, .03)
        error = (noisy[codec.guests][data]-full[codec.guests][data]) / \
            codec.sd_guest[data]
        self.assertLess(float(np.sqrt(np.mean(error**2))), 1.2)
        self.assertLess(codec.last_noise*codec.D, codec.guest_noise_max)
        # Past guest_noise_max the guests are dropped (the display fills
        # them in) while the hosts are still read as steps.
        sigma = .5*(codec.guest_noise_max+codec.noise_max*codec.D)
        rough = _round_trip(codec, values, sigma)
        self.assertGreater(codec.last_noise*codec.D, codec.guest_noise_max)
        self.assertLess(codec.last_noise, codec.noise_max)
        np.testing.assert_array_equal(rough[codec.guests], 0.0)
        self.assertEqual(codec.last_unfolded_slots, codec.M)
        exact = np.isclose(rough[hosts][data], full[hosts][data], atol=1e-6)
        self.assertGreater(float(np.mean(exact)), .8)
        # Past noise_max (in steps) the hosts are read plainly as well.
        _round_trip(codec, values, 1.6*codec.noise_max*codec.D)
        self.assertEqual(codec.last_unfolded_slots, 0)
        return values

    def test_fold_500_table_carries_companded_guests(self):
        fold = LiveFold(500)
        settings = live_fold.TABLE_COMPAND[500]
        self.assertEqual(fold.table['compand'],
                         [settings['limit'], settings['mu']])
        self.assertEqual(fold.table['D'], settings['step'])
        codec = fold.codec(self.base)
        values = self._check(codec, settings)
        # The application sender folds the same companded symbols.
        np.testing.assert_array_equal(
            Fold500(self.base).encode_coefficients(values),
            codec.encode_coefficients(values))

    def test_fold_1000_table_keeps_linear_guests(self):
        self.assertNotIn(1000, live_fold.TABLE_COMPAND)
        self.assertIsNone(LiveFold(1000).codec(self.base).compand)

    def test_mono_colour_500_carries_companded_guests(self):
        wire = MonoColourFoldWire(self.base)
        codec = wire._codec(wire.model_for(self.base))
        self._check(codec, mono_video.MONO_COLOUR_COMPAND)
        self.assertEqual(codec.identity,
                         mono_video.MONO_COLOUR_FOLD_TABLE_SHA256)

    def test_aspect_mono_500_carries_companded_guests(self):
        wire = AspectMonoWire(self.base)
        for layout in ('16:9', '3:4'):
            with self.subTest(layout=layout):
                codec = wire._codec(wire.model_for(self.base, layout))
                self._check(codec, mono_video.MONO_COLOUR_COMPAND)


if __name__ == '__main__':
    unittest.main()
