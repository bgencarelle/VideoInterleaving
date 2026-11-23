"""Coded status words and the mono profiles' M-only pilot values."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT/'test_modem_v7', ROOT/'tools'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from animation_modem import v7                                      # noqa: E402
from tone_code import (MONO_OFF, STATUS_WORD_BY_MODE,
                       decode_status)                                    # noqa: E402
from mono_wire import MONO_PILOT_VALUES                                   # noqa: E402


class V7MonoWireTests(unittest.TestCase):
    def test_six_status_words_are_balanced_and_distance_six(self):
        words = list(STATUS_WORD_BY_MODE.values())
        self.assertEqual(len(words), 6)
        for word in words:
            self.assertEqual(sum(word), 6)
            self.assertEqual(decode_status(word)['mode'],
                             next(mode for mode, candidate in
                                  STATUS_WORD_BY_MODE.items()
                                  if candidate == word))
        for index, left in enumerate(words):
            for right in words[index+1:]:
                self.assertGreaterEqual(
                    sum(a != b for a, b in zip(left, right)), 6)
        self.assertEqual(decode_status(STATUS_WORD_BY_MODE[MONO_OFF])['mode'],
                         MONO_OFF)

    def test_mono_pilots_are_on_m_only(self):
        np.testing.assert_array_equal(MONO_PILOT_VALUES[..., 1], 0)
        np.testing.assert_array_equal(MONO_PILOT_VALUES[..., 0],
                                      v7.SCATTERED_PILOTS[..., 0])


if __name__ == '__main__':
    unittest.main()
