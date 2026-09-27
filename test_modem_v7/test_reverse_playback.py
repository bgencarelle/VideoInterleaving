"""Reverse EOF packet decode on the live fold-500 coded-pilot profile."""
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
PROTOTYPE = Path(__file__).resolve().parent
for path in (ROOT, PROTOTYPE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from animation_modem import v7                                          # noqa: E402
from tools import v7_live                                               # noqa: E402
from live_fold import LiveFold                                          # noqa: E402
from tone_code import (FOLD_500, add_tone_code, coded_pilot_timing,
                       encode_status)                                    # noqa: E402


TARGET = .1521/np.sqrt(1 + 10**(v7.CLOCK_REL_DB/10))


class ReverseCodedProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = v7.load_model(TARGET, 'box')
        with Image.open(v7.REFERENCE_FIXTURE) as source:
            cls.values = v7.image_values(
                v7.prepare_image(source, 'box'), cls.model.coder.grids, 'box')

    def test_reverse_packets_keep_fold_and_coded_timing(self):
        fold = LiveFold(500)
        fold.check(self.model)
        fold.install()
        try:
            for counter in (2, 3, 4, 7):
                packet = v7_live._encode_pulse_frame_coeffs(
                    self.model, fold.encode_coefficients(self.model, self.values),
                    counter, aspect_code=6, source_index=counter-1,
                    eof_marker=True)
                packet = add_tone_code(
                    packet, counter, encode_status(FOLD_500))
                reverse = packet[::-1].copy()
                hit = v7.pulse_frame_hits(reverse)[0]
                with coded_pilot_timing():
                    results, info = v7.decode_reverse_packet(
                        self.model, reverse, hit[0], hit[1],
                        state=v7.PulseState(), sample_rate=v7.RATE,
                        pilot_timing='tone-seeded')
                with self.subTest(counter=counter):
                    self.assertEqual(info['eof_markers_validated'], 1)
                    self.assertEqual(len(results), 1)
                    result = results[0]
                    self.assertEqual(result.status, 'received')
                    self.assertTrue(result.diag['metadata_valid'])
                    self.assertEqual(result.diag['playback_direction'], -1)
                    self.assertTrue(result.diag['pilot_timing'].get(
                        'coded_chips_removed'))
                    self.assertEqual(result.diag['pilot_timing'].get(
                        'coded_status_mode'), FOLD_500)
                    values = fold.values(
                        self.model, result, metadata_confirmed=True)
                    self.assertEqual(values.shape, self.values.shape)
                    self.assertLess(float(np.sqrt(np.mean(
                        np.square(values-self.values)))), .10)
        finally:
            fold.uninstall()


if __name__ == '__main__':
    unittest.main()
