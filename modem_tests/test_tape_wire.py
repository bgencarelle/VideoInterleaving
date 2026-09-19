"""The tape encoder carries the complete HD picture inside its stated band."""
import unittest

import numpy as np
from scipy.signal import resample_poly

from animation_modem import transport3 as V3
from animation_modem.core import decode_packet
from animation_modem.wavelet import hd_dwt_coder, tape_80x96_coder
from animation_modem.engines import coder_for
from animation_modem.imaging import source_size


class TapeWireTests(unittest.TestCase):
    def setUp(self):
        self.layout = V3.WIRE_TAPE
        self.coder = tape_80x96_coder()
        self.values = np.random.default_rng(44).uniform(
            -.7, .7, self.coder.source_count)

    def test_redundant_80x96_payload_fits(self):
        self.assertGreaterEqual(self.layout.capacity, self.coder.count)
        self.assertEqual(self.coder.source_count, 96*80 + 2*48*40)
        self.assertEqual(self.coder.n_orig, 800)
        self.assertEqual(self.coder.count, 1600)
        self.assertEqual(self.layout.band, (375.0, 12750.0))
        self.assertGreater(self.layout.fps, 16)

    def test_every_80x96_value_has_a_diverse_opposite_leg_copy(self):
        from animation_modem.wavelet import _slot_carriers
        slots = self.coder.slots(self.layout)
        bins, channels = _slot_carriers(self.layout)
        home, copy = slots[:self.coder.n_orig], slots[self.coder.n_orig:]
        self.assertTrue(np.all(channels[home] != channels[copy]))
        self.assertTrue(np.all(np.abs(bins[home]-bins[copy]) >=
                               self.coder.MIN_COPY_SPREAD))

    def test_packet_round_trips_on_the_narrow_wire(self):
        audio = V3.encode(self.values, self.layout, self.coder, 1, 1, 1,
                          profile=2)
        got = decode_packet(audio, self.layout, self.coder,
                            coders={2: self.coder})
        self.assertEqual(got.identity, 'verified_header')
        want = self.coder.inverse(self.coder.forward(self.values))
        # Random pixels are the worst possible transform source. The mandatory
        # short emission filter costs some orthogonality, but the complete
        # payload must verify and remain well below the error of an erased band.
        self.assertLess(np.sqrt(np.mean((got.values-want)**2)), .07)

    def test_whole_waveform_is_emission_limited(self):
        audio = V3.encode(self.values, self.layout, self.coder, 1, 1, 1,
                          profile=2)
        spectrum = np.sum(np.abs(np.fft.rfft(audio, axis=0))**2, axis=1)
        frequencies = np.fft.rfftfreq(len(audio), 1/V3.REFERENCE_RATE)
        total = float(spectrum.sum())
        above = float(spectrum[frequencies > self.layout.emission_ceiling].sum())
        self.assertLess(above/total, 1e-4)


class FastTapeWireTests(unittest.TestCase):
    def setUp(self):
        self.layout = V3.WIRE_TAPE_25
        self.coder, _ = coder_for('tape-80x60', self.layout)
        self.values = np.random.default_rng(45).uniform(
            -.7, .7, self.coder.source_count)

    def test_budget_reconstructs_80x60_above_25_fps(self):
        self.assertEqual(source_size('tape-80x60'), (80, 60))
        self.assertEqual(self.coder.shapes, [(15, 20), (5, 6), (5, 6)])
        self.assertEqual(self.coder.n_orig, 360)
        self.assertEqual(self.coder.count, 720)
        self.assertEqual(self.layout.capacity, 760)
        self.assertGreaterEqual(self.layout.fps, 25)

    def test_every_value_has_an_opposite_leg_frequency_diverse_copy(self):
        from animation_modem.wavelet import _slot_carriers
        slots = self.coder.slots(self.layout)
        bins, channels = _slot_carriers(self.layout)
        home, copy = slots[:self.coder.n_orig], slots[self.coder.n_orig:]
        self.assertTrue(np.all(channels[home] != channels[copy]))
        self.assertTrue(np.all(np.abs(bins[home]-bins[copy]) >=
                               self.coder.MIN_COPY_SPREAD))

    def test_fast_profile_round_trips_and_declares_itself(self):
        audio = V3.encode(self.values, self.layout, self.coder, 1, 1, 1,
                          profile=3)
        got = decode_packet(audio, self.layout, self.coder,
                            coders={3: self.coder})
        self.assertEqual(got.identity, 'verified_header')
        self.assertEqual(got.extra['profile'], 'tape-80x60')
        want = self.coder.inverse(self.coder.forward(self.values))
        self.assertLess(np.sqrt(np.mean((got.values-want)**2)), .08)

    def test_fast_whole_waveform_is_emission_limited(self):
        audio = V3.encode(self.values, self.layout, self.coder, 1, 1, 1,
                          profile=3)
        spectrum = np.sum(np.abs(np.fft.rfft(audio, axis=0))**2, axis=1)
        frequencies = np.fft.rfftfreq(len(audio), 1/V3.REFERENCE_RATE)
        above = spectrum[frequencies > self.layout.emission_ceiling].sum()
        self.assertLess(float(above/spectrum.sum()), 1e-4)


class CleanRateTests(unittest.TestCase):
    def test_both_tape_profiles_decode_cleanly_at_48_and_96_khz(self):
        cases = [(V3.WIRE_TAPE, tape_80x96_coder(), 2),
                 (V3.WIRE_TAPE_25, coder_for('tape-80x60', V3.WIRE_TAPE_25)[0], 3)]
        for layout, coder, profile in cases:
            with self.subTest(wire=layout.name):
                values = np.random.default_rng(profile).uniform(
                    -.5, .5, coder.source_count)
                source = np.concatenate([
                    V3.encode(values, layout, coder, n, n, 2, profile=profile)
                    for n in (1, 2)])
                for rate, audio in ((48000, source),
                                    (96000, resample_poly(source, 2, 1, axis=0))):
                    receiver = V3.Receiver(layout, coder, input_rate=rate,
                                           coders={profile: coder})
                    got = receiver.feed(np.asarray(audio, np.float32)) + receiver.flush()
                    self.assertEqual([result.identity for result in got],
                                     ['verified_header', 'verified_header'])


if __name__ == '__main__':
    unittest.main()
