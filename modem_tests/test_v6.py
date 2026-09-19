"""V6's two transforms must remain a controlled, protected analog comparison."""
import unittest

import numpy as np

from animation_modem import transport3 as V3
from animation_modem.core import decode_packet
from animation_modem.v6 import (FOUNDATION_VALUES, ORIGINAL_VALUES, V6_GRIDS,
                                WIRE_VALUES, dct_coder, slot_symbols,
                                wavelet_coder)
from animation_modem.wavelet import _slot_carriers


class V6Tests(unittest.TestCase):
    def test_candidates_share_source_wire_and_protection_budgets(self):
        dct, wavelet = dct_coder(), wavelet_coder()
        for coder in (dct, wavelet):
            self.assertEqual(tuple(coder.grids), V6_GRIDS)
            self.assertEqual(coder.source_count, 11520)
            self.assertEqual(coder.n_orig, ORIGINAL_VALUES)
            self.assertEqual(len(coder.copy_of), FOUNDATION_VALUES)
            self.assertEqual(coder.count, WIRE_VALUES)
            self.assertLessEqual(coder.count, V3.WIRE_V6.capacity)
        self.assertEqual(V3.WIRE_V6.band, (375.0, 12750.0))
        self.assertEqual(V3.WIRE_V6.emission_ceiling, 14000)
        self.assertEqual(V3.WIRE_V6.wire_magic, b'V6')
        self.assertNotEqual(V3.WIRE_V6.wire_magic, V3.WIRE_TAPE.wire_magic)

    def test_every_foundation_copy_is_frequency_and_track_diverse(self):
        bins, channels = _slot_carriers(V3.WIRE_V6)
        symbols = slot_symbols(V3.WIRE_V6)
        for coder in (dct_coder(), wavelet_coder()):
            with self.subTest(transform=coder.profile_name):
                slots = coder.slots(V3.WIRE_V6)
                home = slots[coder.copy_of]
                copy = slots[coder.n_orig:]
                self.assertTrue(np.all(channels[home] != channels[copy]))
                self.assertTrue(np.all(np.abs(bins[home]-bins[copy]) >=
                                       coder.MIN_COPY_SPREAD))
                self.assertTrue(np.all(symbols[home] != symbols[copy]))

    def test_both_transforms_round_trip_on_the_same_wire(self):
        values = np.random.default_rng(6).uniform(-.7, .7, 11520)
        for code, coder in enumerate((dct_coder(), wavelet_coder())):
            with self.subTest(transform=coder.profile_name):
                audio = V3.encode(values, V3.WIRE_V6, coder, 1, 1, 1,
                                  profile=code)
                got = decode_packet(audio, V3.WIRE_V6, coder)
                self.assertEqual(got.identity, 'verified_header')
                self.assertIsNotNone(got.values)
                self.assertTrue(np.isfinite(got.values).all())
                self.assertLess(np.sqrt(np.mean((got.values-values)**2)), .55)

    def test_foundation_and_color_survive_either_track_loss(self):
        y = np.linspace(-.7, .7, 96*80).reshape(96, 80)
        cb = np.full((48, 40), .15)
        cr = np.full((48, 40), -.10)
        values = np.concatenate([y.ravel(), cb.ravel(), cr.ravel()])
        for code, coder in enumerate((dct_coder(), wavelet_coder())):
            audio = V3.encode(values, V3.WIRE_V6, coder, 1, 1, 1,
                              profile=code)
            clean = decode_packet(audio, V3.WIRE_V6, coder).values
            for leg in (0, 1):
                with self.subTest(transform=coder.profile_name, muted=leg):
                    damaged = audio.copy()
                    damaged[:, leg] = 0
                    got = decode_packet(damaged, V3.WIRE_V6, coder)
                    self.assertEqual(got.identity, 'verified_header')
                    self.assertIsNotNone(got.values)
                    self.assertLess(np.sqrt(np.mean((got.values-clean)**2)), .02)
                    self.assertAlmostEqual(np.mean(got.values[96*80:96*80+48*40]),
                                           .15, delta=.01)
                    self.assertAlmostEqual(np.mean(got.values[-48*40:]),
                                           -.10, delta=.01)

    def test_complete_waveform_obeys_tape_ceiling(self):
        coder = dct_coder()
        values = np.random.default_rng(8).uniform(-.7, .7, coder.source_count)
        audio = V3.encode(values, V3.WIRE_V6, coder, 1, 1, 1)
        spectrum = np.sum(np.abs(np.fft.rfft(audio, axis=0))**2, axis=1)
        frequency = np.fft.rfftfreq(len(audio), 1/V3.REFERENCE_RATE)
        self.assertLess(float(spectrum[frequency > 14000].sum()/spectrum.sum()),
                        1e-4)

    def test_stream_survives_loss_and_return_of_either_track(self):
        coder = dct_coder()
        values = np.linspace(-.5, .5, coder.source_count)
        clean = [V3.encode(values, V3.WIRE_V6, coder, frame, frame, 4)
                 for frame in range(1, 5)]
        for leg in (0, 1):
            with self.subTest(interrupted=leg):
                packets = [packet.copy() for packet in clean]
                packets[1][:, leg] = 0
                packets[2][:, leg] = 0
                stream = np.concatenate(packets)
                receiver = V3.Receiver(V3.WIRE_V6, coder, pulse_only=True,
                                       candidates=[(V3.WIRE_V6, coder, {0: coder})],
                                       coders={0: coder})
                out = []
                for start in range(0, len(stream), 257):
                    out.extend(receiver.feed(stream[start:start+257]))
                out.extend(receiver.flush())
                self.assertEqual([result.absolute for result in out], [1, 2, 3, 4])
                self.assertTrue(all(result.identity == 'verified_header'
                                    and result.values is not None for result in out))


if __name__ == '__main__':
    unittest.main()
