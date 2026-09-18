"""Separate decoder resampling error from input-path anti-alias filtering."""
from fractions import Fraction
import unittest
from unittest.mock import patch

import numpy as np
from scipy.signal import firwin, resample_poly
from animation_modem import transport3 as v
from animation_modem import core
from animation_modem.imaging import plane_shapes


class SpeedCompensationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layout = v.WIRE
        cls.coder = v.SourceCoder(plane_shapes('color-dct'))
        cls.values = np.random.default_rng(3).uniform(-.2, .2, cls.coder.count)
        cls.audio = np.concatenate([
            v.encode(cls.values, cls.layout, cls.coder, n, n, 6)
            for n in range(1, 7)])

    def speed_audio(self, speed):
        ratio = Fraction(1/speed).limit_denominator(1000)
        maximum = max(ratio.numerator, ratio.denominator)
        # A deliberately narrow transition band keeps the reference generator
        # from removing the top carriers BEFORE they reach the decoder.
        filt = firwin(2*256*maximum+1, 1/maximum, window=('kaiser', 12))
        return resample_poly(self.audio, ratio.numerator, ratio.denominator, window=filt)

    def decode(self, audio):
        rx = v.Receiver(self.layout, self.coder)
        return rx.feed(audio)+rx.flush()

    def error(self, results):
        return float(np.median([np.sqrt(np.mean((r.values-self.values)**2)) for r in results]))

    def test_faster_playback_preserves_identity_and_speed(self):
        for speed in (1.05, 1.10, 1.15, 1.18):
            with self.subTest(speed=speed):
                out = self.decode(self.speed_audio(speed))
                self.assertEqual([r.absolute for r in out], list(range(1, 7)))
                self.assertTrue(all(r.identity == 'verified_header' for r in out))
                for r in out:
                    self.assertAlmostEqual(r.extra['playback_speed'], speed, delta=.0003)
                    self.assertEqual(r.extra['resample_taps'], 32)
                self.assertLess(self.error(out), .035)

    def test_longer_interpolation_reduces_high_speed_error(self):
        audio = self.speed_audio(1.15)
        fixed = self.decode(audio)
        # Restore only the old short payload filter; keep pulse timing equal.
        def short_payload(samples, positions, taps=8):
            return core._sample_at(samples, positions, taps=8 if len(positions) > 500 else taps)
        with patch.object(v, '_sample_at', side_effect=short_payload):
            short = self.decode(audio)
        self.assertLess(self.error(fixed), .4*self.error(short))

    def test_unity_speed_remains_exact(self):
        out = self.decode(self.audio)
        self.assertEqual(len(out), 6)
        self.assertLess(self.error(out), 1e-5)
        self.assertTrue(all(r.extra['playback_speed'] == 1 for r in out))

    def test_speed_estimation_does_not_call_a_template_search(self):
        with patch.object(v, '_fit_preamble', side_effect=AssertionError('template fit')), \
             patch.object(v.Receiver, '_correlate', side_effect=AssertionError('speed sweep')):
            out = self.decode(self.speed_audio(1.15))
        self.assertEqual(len(out), 6)
