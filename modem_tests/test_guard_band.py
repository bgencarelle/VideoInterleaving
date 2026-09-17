"""The input filter's guard band is derived from the live layout, so pin it.

transport3 owns the wire, but it imports audio_common, so audio_common cannot
read it off transport3.WIRE. It hand-copies the geometry instead, and nothing
but this file keeps the copy in step: a wire edit would otherwise leave
receivers filtering for a band the signal had left, silently.
"""
import unittest

import numpy as np

from animation_modem.audio_common import (GUARD_BINS, N, REFERENCE_RATE,
                                          WIRE as GUARD_WIRE, InputFilter)
from animation_modem.transport3 import WIRE


class GuardBandTests(unittest.TestCase):
    def test_the_hand_copied_layout_is_the_live_default(self):
        self.assertEqual(GUARD_WIRE, WIRE)

    def test_the_guard_reaches_past_the_bottom_carrier(self):
        """Regression: BAND was (200, 22000), which left the bottom carrier
        unprotected -- a 250 Hz tone one bin away leaks into it with a rotating
        phase the training cannot equalise, and identity is lost. The guard has
        to sit one carrier up, and no further, so the equalizer only has to
        restore the bottom carrier."""
        f = InputFilter(rate=REFERENCE_RATE)
        lowest = GUARD_WIRE.carriers[0]*REFERENCE_RATE/N
        second = GUARD_WIRE.carriers[1]*REFERENCE_RATE/N
        self.assertEqual(f.high, lowest + GUARD_BINS*REFERENCE_RATE/N)
        self.assertGreater(f.high, lowest)
        self.assertLessEqual(f.high, second)
        self.assertGreater(f.low, GUARD_WIRE.carriers[-1]*REFERENCE_RATE/N)

    def test_the_guard_follows_the_arrival_clock(self):
        for rate in (44100, 48000, 96000):
            f = InputFilter(rate=rate)
            lowest = GUARD_WIRE.carriers[0]*rate/N
            self.assertEqual(f.high, lowest + GUARD_BINS*rate/N)
            self.assertEqual(f.rate, rate)

    def test_a_sub_band_tone_is_attenuated(self):
        f = InputFilter(rate=REFERENCE_RATE)
        n = 8192
        t = np.arange(n)/REFERENCE_RATE
        tone = np.stack([np.sin(2*np.pi*250*t)]*2, axis=1).astype(np.float32)
        out = f.process(tone)
        self.assertLess(float(np.sqrt(np.mean(out[2048:]**2))), 0.1)


if __name__ == '__main__':
    unittest.main()
