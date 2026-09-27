"""tools/v7_wire_profile.py: the bench tools' wire must be the one that is sent.

The default profile has to stay byte-identical to the live sender's packets
(fold 500, coded pilots, EOF marker), decode every packet with the live
receiver settings, and unfold. The fold-off variant keeps the live framing for
benches whose models the fold tables do not cover. The baseline profile is the
low-level encoder's older framing.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from animation_modem import v7                                          # noqa: E402
from tools import v7_live                                               # noqa: E402
from tools.v7_wire_profile import WireProfile                           # noqa: E402

TARGET = .1521/np.sqrt(1 + 10**(v7.CLOCK_REL_DB/10))
PACKETS = 3


def fixture_values(model, encode_filter):
    with Image.open(v7.REFERENCE_FIXTURE) as image:
        return v7.image_values(v7.prepare_image(image, encode_filter),
                               model.coder.grids, encode_filter)


class WireProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.box = v7.load_model(TARGET, 'box')
        cls.nearest = v7.load_model(TARGET, 'nearest')
        cls.box_values = fixture_values(cls.box, 'box')
        cls.nearest_values = fixture_values(cls.nearest, 'nearest')

    def test_default_profile_is_byte_identical_to_the_live_sender(self):
        profile = WireProfile('default')
        wire = profile.encode(self.box, [self.box_values]*PACKETS, 5,
                              [6]*PACKETS, [10, 11, 12])
        fold = profile._fold
        sent = np.concatenate([
            v7_live._add_coded_pilots(v7_live._encode_pulse_frame_coeffs(
                self.box, fold.encode_coefficients(self.box, self.box_values),
                5+i, aspect_code=6, source_index=10+i, eof_marker=True),
                5+i, 500)
            for i in range(PACKETS)])
        np.testing.assert_array_equal(wire, sent)

    def test_default_profile_decodes_every_packet_and_unfolds(self):
        profile = WireProfile('default')
        wire = profile.encode(self.box, [self.box_values]*PACKETS)
        hooks = (v7._equalize_numba, v7._equalize_numpy, v7.decode_frame,
                 v7.pilot_tone_timing)
        results, _ = profile.decode(self.box, wire, sample_rate=v7.RATE)
        # The fold and coded-pilot hooks are installed for the call only.
        self.assertEqual((v7._equalize_numba, v7._equalize_numpy,
                          v7.decode_frame, v7.pilot_tone_timing), hooks)
        self.assertEqual(len(results), PACKETS)          # EOF commits the last one
        self.assertTrue(all(r.diag.get('metadata_valid') for r in results))
        codec = profile._fold.codec(self.box)
        for result in results:
            self.assertEqual(result.diag['pilot_timing'].get('coded_status_mode'), 1)
            values = profile.values(self.box, result)
            self.assertEqual(codec.last_unfolded_slots, codec.M)
            self.assertEqual(values.shape, self.box_values.shape)

    def test_fold_off_keeps_the_live_framing(self):
        profile = WireProfile('default', fold=False)
        self.assertEqual(profile.encode_filter, 'nearest')
        wire = profile.encode(self.nearest, [self.nearest_values]*PACKETS)
        results, _ = profile.decode(self.nearest, wire, sample_rate=v7.RATE)
        self.assertEqual(len(results), PACKETS)
        for result in results:
            self.assertEqual(result.diag['pilot_timing'].get('coded_status_mode'), 0)
            np.testing.assert_array_equal(
                profile.values(self.nearest, result),
                v7.values_from(self.nearest, result.coeffs))

    def test_folded_profile_refuses_another_model(self):
        with self.assertRaises(ValueError):
            WireProfile('default').encode(self.nearest, [self.nearest_values])

    def test_baseline_profile_is_the_low_level_default(self):
        profile = WireProfile('baseline')
        wire = profile.encode(self.nearest, [self.nearest_values]*PACKETS)
        np.testing.assert_array_equal(
            wire, v7.encode_pulse_stream(self.nearest, [self.nearest_values]*PACKETS))
        results, _ = profile.decode(self.nearest, wire, sample_rate=v7.RATE)
        self.assertEqual(len(results), PACKETS-1)        # no following header


if __name__ == '__main__':
    unittest.main()
