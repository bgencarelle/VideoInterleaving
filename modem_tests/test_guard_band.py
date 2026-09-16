"""The input filter's guard band is spelled out by hand, so pin it.

transport3 owns the preset table, but it imports audio_common, so this module
cannot read the live default's carriers off ALL_PRESETS. It hand-copies the
geometry instead, and nothing but this file keeps the copy in step: a preset
edit would otherwise leave receivers filtering for a band the signal had
left, silently.
"""
import unittest

import numpy as np

from animation_modem.audio_common import BAND, WIDE_V3, InputFilter
from animation_modem.transport3 import ALL_PRESETS


class GuardBandTests(unittest.TestCase):
    def test_the_hand_copied_layout_is_the_live_default(self):
        live = ALL_PRESETS['wide-v3']
        np.testing.assert_array_equal(WIDE_V3.carriers, live.carriers)

    def test_the_default_band_spans_the_carriers(self):
        """Regression: BAND was (600, 22000) against carriers that started at
        1125 Hz, then the band moved down to 375 Hz without it -- so
        InputFilter() with its own defaults raised ValueError on construction,
        and nothing noticed because no caller used the default."""
        from animation_modem.audio_common import REFERENCE_RATE, N
        high, low = BAND
        self.assertLess(high, WIDE_V3.carriers[0]*REFERENCE_RATE/N)
        self.assertGreater(low, WIDE_V3.carriers[-1]*REFERENCE_RATE/N)
        InputFilter()      # must construct on its own defaults


if __name__ == '__main__':
    unittest.main()
