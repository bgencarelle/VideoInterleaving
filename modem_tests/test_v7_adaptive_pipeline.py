import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.signal import filtfilt, firwin, lfilter_zi

from animation_modem.v7_core import shape_emission
from tools.v7_adaptive_pipeline import SOURCE_SHAPES, source_values
from tools.v7_sk_fold import adaptive_body_filtfilt


class AdaptivePipelineTests(unittest.TestCase):
    def test_gui_profile_selection_and_old_preferences_match_sample(self):
        from tools.v7_send_gui import SenderGui, _restore_sender_settings
        from tools.v7_adaptive_pipeline import sample_sender_settings
        for mode in ('mono', 'stereo'):
            expected = sample_sender_settings(mode)
            gui = SenderGui(restore_preferences=False)
            gui._assign('profile', expected['profile'])
            for key, value in expected.items():
                self.assertEqual(gui.settings[key], value, key)
            restored = {}
            _restore_sender_settings(restored, {
                'profile': expected['profile'], 'dct_kernel': 'viewer_solve',
                'capture_width': '160'})
            for key, value in expected.items():
                self.assertEqual(restored[key], value, key)
            restored = {}
            _restore_sender_settings(restored, {
                **expected, 'capture_width': '512'})
            self.assertEqual(restored['capture_width'], '512')

    def test_sample_feed_uses_existing_adaptive_modes_and_public_cli(self):
        from tools.v7_sample_feed import command_for
        from tools.v7_live import parser
        for mode in ('mono', 'stereo'):
            command = command_for('send', mode, 'pulse', 'picture.mkv')
            args = parser().parse_args(command[2:])
            self.assertEqual(args.profile, f'adaptive-{mode}-nested')
            self.assertEqual(args.capture_width, 768)
            self.assertEqual(args.dct_kernel, 'reference')
            self.assertEqual(args.brightness, 1)
            self.assertFalse(args.luma_adjust)
            self.assertTrue(args.dct_encode)
        args = parser().parse_args(command_for('receive', 'stereo', 'pulse')[2:])
        self.assertFalse(args.headless)
        self.assertEqual(args.nested_smooth, 0)

    def test_adaptive_source_does_not_discard_selected_kernel_values(self):
        from tools import v7_live
        rgb = np.zeros((120, 100, 3), dtype=np.uint8)
        carried = np.full(18000, .125)
        kernel = SimpleNamespace(name='viewer_solve', defer_gain_to_codec=False)
        selection = SimpleNamespace(kernel=kernel)
        with patch.object(v7_live, '_picture_values', return_value=(carried, 0)):
            values, _ = v7_live._values(
                None, rgb, dct_encode=True,
                dct_options={'kernel': selection}, adaptive_source_area=True)
        self.assertIs(values, carried)

    def test_adaptive_reference_uses_public_area_conversion(self):
        from tools import v7_live
        rgb = np.full((120, 100, 3), 128, dtype=np.uint8)
        kernel = SimpleNamespace(name='reference', defer_gain_to_codec=False)
        with patch.object(v7_live, '_picture_values',
                          return_value=(np.zeros(18000), 0)):
            values, _ = v7_live._values(
                None, rgb, brightness=1, dct_encode=True,
                dct_options={'kernel': SimpleNamespace(kernel=kernel)},
                adaptive_source_area=True)
        np.testing.assert_array_equal(values, source_values(rgb))

    def test_source_area_conversion_emits_exact_table_geometry(self):
        y, x = np.mgrid[:48, :64]
        rgb = np.stack((x * 4, y * 5, (x + y) * 2), axis=2).clip(0, 255)
        values = source_values(np.ascontiguousarray(rgb, dtype=np.uint8))
        expected = int(np.sum(SOURCE_SHAPES[:, 0] * SOURCE_SHAPES[:, 1]))
        self.assertEqual(values.shape, (expected,))
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue(np.all(values >= -1.0))
        self.assertTrue(np.all(values <= 1.0))

    def test_adaptive_short_fir_matches_reference_and_public_shaper(self):
        rng = np.random.default_rng(701)
        signal = np.ascontiguousarray(
            rng.normal(0, .1, (3920, 2)).astype(np.float32))
        taps = firwin(17, 14000, fs=48000, window=('kaiser', 8.6))
        expected = filtfilt(taps, [1.0], signal, axis=0)
        actual = adaptive_body_filtfilt(signal, taps, lfilter_zi(taps, [1.0]))
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
        public = shape_emission(signal, 14000, rate=48000, taps=17)
        np.testing.assert_allclose(public, expected, rtol=2e-6, atol=2e-7)
        self.assertTrue(adaptive_body_filtfilt.nopython_signatures)


if __name__ == '__main__':
    unittest.main()
