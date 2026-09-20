"""Full-repeat V6 robustness control."""
import unittest

import numpy as np

from animation_modem import transport3 as V3
from animation_modem.core import decode_packet
from animation_modem.v6 import ORIGINAL_VALUES, full_repeat_coder
from animation_modem.wavelet import _slot_carriers


class V6FullRepeatTests(unittest.TestCase):
    def test_budget_and_wire_tradeoff(self):
        self.assertEqual(V3.WIRE_V6_REPEAT.band, (375.0, 12750.0))
        self.assertEqual(V3.WIRE_V6_REPEAT.wire_magic, b'VR')
        self.assertEqual(V3.WIRE_V6_REPEAT.capacity, 5800)
        self.assertGreaterEqual(V3.WIRE_V6_REPEAT.capacity, 2*ORIGINAL_VALUES)
        self.assertAlmostEqual(V3.WIRE_V6_REPEAT.fps, 6.0362, places=3)
        for transform in ('dct', 'wavelet'):
            coder = full_repeat_coder(transform)
            self.assertEqual(coder.n_orig, ORIGINAL_VALUES)
            self.assertEqual(len(coder.copy_of), ORIGINAL_VALUES)
            self.assertEqual(coder.count, 2*ORIGINAL_VALUES)

    def test_every_coefficient_is_cross_track_and_frequency_diverse(self):
        bins, channels = _slot_carriers(V3.WIRE_V6_REPEAT)
        for transform in ('dct', 'wavelet'):
            coder = full_repeat_coder(transform)
            slots = coder.slots(V3.WIRE_V6_REPEAT)
            home, copy = slots[:coder.n_orig], slots[coder.n_orig:]
            with self.subTest(transform=transform):
                self.assertTrue(np.all(channels[home] != channels[copy]))
                self.assertTrue(np.all(np.abs(bins[home]-bins[copy]) >=
                                       coder.MIN_COPY_SPREAD))

    def test_clean_and_one_track_loss_round_trip(self):
        values = np.random.default_rng(15).uniform(-.7, .7, 11520)
        for code, transform in enumerate(('dct', 'wavelet')):
            coder = full_repeat_coder(transform)
            audio = V3.encode(values, V3.WIRE_V6_REPEAT, coder, 1, 1, 1,
                              profile=code)
            for leg in (None, 0, 1):
                damaged = audio.copy()
                if leg is not None:
                    damaged[:, leg] = 0
                result = decode_packet(damaged, V3.WIRE_V6_REPEAT, coder,
                                       coders={code: coder})
                with self.subTest(transform=transform, muted=leg):
                    self.assertEqual(result.identity, 'verified_header')
                    self.assertIsNotNone(result.values)
                    self.assertLess(np.sqrt(np.mean((result.values-values)**2)),
                                    .60)


if __name__ == '__main__':
    unittest.main()
