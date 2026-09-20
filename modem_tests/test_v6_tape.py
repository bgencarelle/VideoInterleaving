"""V6 tape placement: foundation low in frequency, spread in time, cross-track."""
import unittest

import numpy as np

from animation_modem import transport3 as V3
from animation_modem.core import N, REFERENCE_RATE, decode_packet
from animation_modem.v6 import (FOUNDATION_VALUES, ORIGINAL_VALUES,
                                TAPE_COPY_SPREAD, WIRE_VALUES, coder_for,
                                slot_fields, slot_report, tape_coder)

HZ_PER_BIN = REFERENCE_RATE/N


class V6TapePlacementTests(unittest.TestCase):
    def coders(self):
        return tape_coder('dct'), tape_coder('wavelet')

    def test_budget_and_wire_match_foundation_v6(self):
        for coder in self.coders():
            with self.subTest(transform=coder.profile_name):
                self.assertEqual(coder.n_orig, ORIGINAL_VALUES)
                self.assertEqual(len(coder.copy_of), FOUNDATION_VALUES)
                self.assertEqual(coder.count, WIRE_VALUES)
                np.testing.assert_array_equal(np.sort(coder.copy_of),
                                              np.sort(coder_for(
                                                  coder.profile_name[-3:] if
                                                  coder.profile_name.endswith('dct')
                                                  else 'wavelet').copy_of))
                slots = coder.slots(V3.WIRE_V6)
                self.assertEqual(len(np.unique(slots)), coder.count)

    def test_foundation_rides_low_carriers_and_never_header_spares(self):
        bins, _, _, _ = slot_fields(V3.WIRE_V6)
        for coder in self.coders():
            with self.subTest(transform=coder.profile_name):
                slots = coder.slots(V3.WIRE_V6)
                home, copy = slots[coder.copy_of], slots[coder.n_orig:]
                self.assertLessEqual(bins[home].max()*HZ_PER_BIN, 4000)
                self.assertLessEqual(bins[copy].max()*HZ_PER_BIN, 7000)
                self.assertTrue(np.all(home >= V3.WIRE_V6.header_capacity))
                self.assertTrue(np.all(copy >= V3.WIRE_V6.header_capacity))
                self.assertFalse(np.any(bins[home] == 1))

    def test_foundation_is_interleaved_across_the_packet(self):
        _, _, _, symbols = slot_fields(V3.WIRE_V6)
        for coder in self.coders():
            with self.subTest(transform=coder.profile_name):
                slots = coder.slots(V3.WIRE_V6)
                ranked = np.argsort(coder.rank, kind='stable')
                top = symbols[slots[ranked[:100]]]
                self.assertGreaterEqual(len(np.unique(top)), 20)
                found = symbols[slots[coder.copy_of]]
                self.assertEqual(len(np.unique(found)), V3.WIRE_V6.image_symbols)

    def test_every_copy_is_cross_track_spread_and_half_a_packet_away(self):
        bins, channels, _, symbols = slot_fields(V3.WIRE_V6)
        for coder in self.coders():
            with self.subTest(transform=coder.profile_name):
                slots = coder.slots(V3.WIRE_V6)
                home, copy = slots[coder.copy_of], slots[coder.n_orig:]
                self.assertTrue(np.all(channels[home] != channels[copy]))
                self.assertTrue(np.all(np.abs(bins[home]-bins[copy]) >=
                                       TAPE_COPY_SPREAD))
                self.assertTrue(np.all(np.abs(symbols[home]-symbols[copy]) >=
                                       V3.WIRE_V6.image_symbols//2))

    def test_report_describes_every_tier(self):
        rows = slot_report(tape_coder('dct'), V3.WIRE_V6)
        self.assertEqual([row['tier'] for row in rows],
                         ['rank 0-100', 'rank 0-720', 'rank 720-2880', 'copies'])
        self.assertEqual(rows[1]['header_spare'], 0)
        self.assertEqual(rows[-1]['both_above_6k'], 0)

    def test_round_trip_and_either_track_loss_keep_color(self):
        y = np.linspace(-.7, .7, 96*80).reshape(96, 80)
        values = np.concatenate([y.ravel(), np.full(48*40, .15),
                                 np.full(48*40, -.10)])
        for coder in self.coders():
            audio = V3.encode(values, V3.WIRE_V6, coder, 1, 1, 1)
            clean = decode_packet(audio, V3.WIRE_V6, coder)
            self.assertEqual(clean.identity, 'verified_header')
            self.assertLess(np.sqrt(np.mean((clean.values-values)**2)), .02)
            for leg in (0, 1):
                with self.subTest(transform=coder.profile_name, muted=leg):
                    damaged = audio.copy()
                    damaged[:, leg] = 0
                    got = decode_packet(damaged, V3.WIRE_V6, coder)
                    self.assertEqual(got.identity, 'verified_header')
                    self.assertLess(np.sqrt(np.mean((got.values-clean.values)**2)),
                                    .02)
                    self.assertAlmostEqual(np.mean(got.values[7680:9600]), .15,
                                           delta=.01)
                    self.assertAlmostEqual(np.mean(got.values[-1920:]), -.10,
                                           delta=.01)

    def test_nocopy_control_uses_the_same_wire(self):
        coder = tape_coder('dct', copies=False)
        self.assertEqual(coder.count, ORIGINAL_VALUES)
        values = np.random.default_rng(3).uniform(-.7, .7, coder.source_count)
        got = decode_packet(V3.encode(values, V3.WIRE_V6, coder, 1, 1, 1),
                            V3.WIRE_V6, coder)
        self.assertEqual(got.identity, 'verified_header')
        self.assertLess(np.sqrt(np.mean((got.values-values)**2)), .55)


if __name__ == '__main__':
    unittest.main()
