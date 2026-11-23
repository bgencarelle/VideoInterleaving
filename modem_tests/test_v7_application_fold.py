"""Application V7 Fold 500 sender runtime stays on the pinned wire profile."""
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from animation_modem import v7
from animation_modem.v7_coded_pilot import add_fold500_coded_pilot
from animation_modem.v7_fold import Fold500
from modem_v7_display import encode_folded_coefficients_packet


ROOT = Path(__file__).resolve().parents[1]
TARGET = .1521/np.sqrt(1 + 10**(v7.CLOCK_REL_DB/10))


class ApplicationFoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = v7.load_model(TARGET, 'box')
        with Image.open(v7.REFERENCE_FIXTURE) as image:
            cls.values = v7.image_values(
                v7.prepare_image(image, 'box'), cls.model.coder.grids, 'box')

    def test_folded_coefficients_match_the_pinned_live_table_runtime(self):
        test_runtime = str(ROOT/'test_modem_v7')
        if test_runtime not in sys.path:
            sys.path.insert(0, test_runtime)
        from live_fold import LiveFold

        actual = Fold500(self.model).encode_coefficients(self.values)
        expected = LiveFold(500).encode_coefficients(self.model, self.values)
        np.testing.assert_array_equal(actual, expected)

    def test_live_fold_applies_deferred_gain_to_the_existing_dct(self):
        test_runtime = str(ROOT/'test_modem_v7')
        if test_runtime not in sys.path:
            sys.path.insert(0, test_runtime)
        from live_fold import LiveFold

        fold = LiveFold(500)
        codec = fold.codec(self.model)
        gains = [np.ones(grid, dtype=float) for grid in self.model.coder.grids]
        gains[0][1:, :] = 1.1
        gains[0][0, 0] = 1.0
        full = codec.grid.forward(self.values)
        luma_count = np.prod(self.model.coder.grids[0])
        luma = full[:luma_count].reshape(self.model.coder.grids[0])
        luma *= gains[0]
        expected = codec.encode_coefficients(self.values, full=full)
        actual = fold.encode_coefficients(
            self.model, self.values, coefficient_gains=tuple(gains))
        np.testing.assert_array_equal(actual, expected)

    def test_coded_pilot_matches_the_standalone_receiver_wire(self):
        test_runtime = str(ROOT/'test_modem_v7')
        if test_runtime not in sys.path:
            sys.path.insert(0, test_runtime)
        import tone_code

        fold = Fold500(self.model)
        coefficients = fold.encode_coefficients(self.values)
        packet = v7.encode_pulse_frame_coeffs(
            self.model, coefficients, 5, aspect_code=3, source_index=2,
            pilot_tones=False)
        actual = add_fold500_coded_pilot(packet, 5)
        expected = tone_code.add_tone_code(
            packet, 5, tone_code.encode_status(tone_code.FOLD_500))
        np.testing.assert_array_equal(actual, expected)

    def test_production_fold_path_mixes_coded_pilot_inside_level_fitting(self):
        coefficients = Fold500(self.model).encode_coefficients(self.values)
        actual = encode_folded_coefficients_packet(
            self.model, coefficients, 5, 2, 3)
        expected = v7.encode_pulse_frame_coeffs(
            self.model, coefficients, 5, aspect_code=3, source_index=2,
            pilot_tones=False,
            extra_tone_mixer=lambda packet: add_fold500_coded_pilot(packet, 5))
        np.testing.assert_array_equal(actual, expected)

    def test_fold_table_rejects_the_nearest_encoding_model(self):
        nearest = v7.load_model(TARGET, 'nearest')
        with self.assertRaisesRegex(ValueError, 'canonical Box'):
            Fold500(nearest)


if __name__ == '__main__':
    unittest.main()
