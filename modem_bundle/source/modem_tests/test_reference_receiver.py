"""Short deterministic checks; target-hardware timings belong to the operator."""
import unittest
from unittest.mock import patch
import numpy as np
from scipy.signal import resample_poly
from animation_modem import transport2 as v
from animation_modem.imaging import plane_shapes
from animation_modem.reference_receiver import Receiver


class ReferenceReceiverTests(unittest.TestCase):
    def setUp(self):
        self.layout = v.PRESETS['wide']
        self.coder = v.SourceCoder(plane_shapes())
        self.values = np.random.default_rng(14).uniform(-.2, .2, self.coder.count)
        self.audio = v.encode(self.values, self.layout, self.coder, 1, 1, 2)

    def collect(self, audio, chunk=256, receiver=None):
        rx = receiver or Receiver(self.layout, self.coder)
        results = []
        for at in range(0, len(audio), chunk):
            results.extend(rx.feed(audio[at:at+chunk]))
        return rx, results+rx.flush()

    def test_one_pass_without_any_search_or_packet_decode(self):
        with patch.object(v, '_coarse_bank', side_effect=AssertionError('bank')), \
             patch.object(v.Receiver, '_acquire', side_effect=AssertionError('search')), \
             patch.object(v, '_fit_sync', side_effect=AssertionError('iterative fit')), \
             patch.object(v, 'decode_packet', side_effect=AssertionError('packet replay')):
            rx, out = self.collect(self.audio)
        self.assertFalse(out[0].extra['complete'])
        self.assertTrue(out[-1].extra['complete'])
        self.assertEqual(out[-1].absolute, 1)
        self.assertEqual(out[-1].extra['fft_symbols'], self.layout.symbols)
        self.assertEqual(rx.reference_fits, 1)
        np.testing.assert_allclose(out[-1].values, self.values, atol=.0001)

    def test_slow_reference_clock(self):
        for factor in (2, 4):
            with self.subTest(factor=factor):
                rx, out = self.collect(resample_poly(self.audio, factor, 1, axis=0))
                self.assertTrue(out[-1].extra['complete'])
                self.assertEqual(out[-1].absolute, 1)
                self.assertEqual(rx.reference_fits, 1)
                self.assertAlmostEqual(out[-1].rate_error+1, factor, delta=.002)
                self.assertLess(np.sqrt(np.mean((out[-1].values-self.values)**2)), .01)

    def test_feed_boundaries(self):
        _, expected = self.collect(self.audio)
        for chunk in (37, len(self.audio)):
            _, out = self.collect(self.audio, chunk)
            self.assertEqual(out[-1].absolute, 1)
            np.testing.assert_allclose(out[-1].values, expected[-1].values, atol=.0001)

    def test_gaps_gain_and_fresh_frame(self):
        rx, first = self.collect(self.audio)
        before = first[-1].values.copy()
        self.assertEqual(rx.feed(np.zeros((6000, 2), np.float32)), [])
        following = v.encode(-self.values, self.layout, self.coder, 2, 2, 2)*.3
        _, out = self.collect(following, receiver=rx)
        _, alone = self.collect(following)
        self.assertEqual(out[-1].absolute, 2)
        np.testing.assert_allclose(out[0].values, alone[0].values, atol=.0001)
        np.testing.assert_array_equal(first[-1].values, before)

    def test_bad_header_can_still_deliver_current_image(self):
        audio = self.audio.copy()
        at = v.SYNC_LEN+2*v.SYMBOL
        audio[at:at+self.layout.header_symbols*v.SYMBOL] = 0
        _, out = self.collect(audio)
        self.assertTrue(out[-1].extra['complete'])
        self.assertIsNone(out[-1].absolute)
        self.assertIsNotNone(out[-1].values)

    def test_noise_and_reset_emit_no_blank_picture(self):
        rx, out = self.collect(np.random.default_rng(11).normal(0, .02, (4000, 2)))
        self.assertEqual(out, [])
        rx.reset(preserve_timing=True)
        self.assertEqual(rx.feed(np.zeros((4000, 2))), [])
        _, out = self.collect(self.audio, receiver=rx)
        self.assertEqual(out[-1].absolute, 1)


if __name__ == '__main__':
    unittest.main()
