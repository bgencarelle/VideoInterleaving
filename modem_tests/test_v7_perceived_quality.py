"""Clip-aware encode, the direct-encode small-frame fallback and display grain."""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.fft import dctn, idctn

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT/'test_modem_v7', ROOT/'tools'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from animation_modem import v7                                           # noqa: E402
from animation_modem.v7_source_dct import (CLIP_AWARE_ITERATIONS,        # noqa: E402
                                           clip_aware_luma)
from common import TARGET                                                # noqa: E402
from aspect_fold import AspectFoldWire                                   # noqa: E402
from aspect_mono import AspectMonoWire                                   # noqa: E402
from live_fold import LiveFold                                           # noqa: E402
from mono_video import FRESH_SLOTS, MonoColourFoldWire                   # noqa: E402
from tools import v7_live                                                # noqa: E402
from tools.v7_gl_viewer import (FLOAT_FRAGMENT_SHADER, GRAIN_MODES,      # noqa: E402
                                flat_area_mask)

ROWS, COLS = v7.V7_GRIDS[0]


def _bright_text_values():
    """Grid values: white bars on black, as the direct encode clips them."""
    luma = np.full((ROWS, COLS), -1.0)
    for x in range(10, 70, 9):
        luma[30:60, x:x+3] = 1.0
    chroma = np.zeros(sum(r*c for r, c in v7.V7_GRIDS[1:]))
    return np.concatenate((luma.ravel(), chroma))


def _visible_error(values, sent):
    """RMS error the receiver shows: sent coefficients only, then the clip."""
    luma = values[:ROWS*COLS].reshape(ROWS, COLS)
    shown = idctn(np.where(sent, dctn(luma, norm='ortho'), 0.0), norm='ortho')
    target = _bright_text_values()[:ROWS*COLS].reshape(ROWS, COLS)
    return float(np.sqrt(np.mean((np.clip(shown, -1, 1)-target)**2)))


class ClipAwareEncodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = v7.load_model(TARGET, 'box')
        wire = AspectFoldWire('16:9', 'chroma')
        cls.codec = wire.codec(wire.model_for(cls.base, '16:9'))
        cls.sent = cls.codec.sent_luma_mask().reshape(ROWS, COLS)

    def test_only_sent_coefficients_change_and_visible_error_drops(self):
        values = _bright_text_values()
        fitted = clip_aware_luma(values, (ROWS, COLS), self.sent)
        before = dctn(values[:ROWS*COLS].reshape(ROWS, COLS), norm='ortho')
        after = dctn(fitted[:ROWS*COLS].reshape(ROWS, COLS), norm='ortho')
        np.testing.assert_allclose(after[~self.sent], before[~self.sent],
                                   atol=1e-12)
        np.testing.assert_array_equal(fitted[ROWS*COLS:], values[ROWS*COLS:])
        self.assertLess(_visible_error(fitted, self.sent),
                        .9*_visible_error(values, self.sent))
        np.testing.assert_array_equal(
            clip_aware_luma(values, (ROWS, COLS), self.sent, iterations=0),
            values)
        self.assertEqual(CLIP_AWARE_ITERATIONS, 10)

    def test_mid_grey_pictures_are_left_alone(self):
        count = sum(rows*cols for rows, cols in v7.V7_GRIDS)
        values = np.random.default_rng(1).uniform(-.3, .3, count)
        luma = values[:ROWS*COLS].reshape(ROWS, COLS)
        sent_only = idctn(np.where(self.sent, dctn(luma, norm='ortho'), 0),
                          norm='ortho')
        # Nothing reaches black or white, so the fit is the least-squares
        # band-limited picture: the sent coefficients are unchanged.
        self.assertLess(float(np.abs(sent_only).max()), .99)
        fitted = clip_aware_luma(values, (ROWS, COLS), self.sent)
        np.testing.assert_allclose(fitted, values, atol=1e-9)

    def test_sent_masks_follow_each_profile(self):
        stereo = LiveFold(500).codec(self.base)
        self.assertEqual(int(stereo.sent_luma_mask().sum()), 48*40+500)
        self.assertEqual(int(self.codec.sent_luma_mask().sum()), 1920+500)
        for wire in (MonoColourFoldWire(self.base), AspectMonoWire(self.base)):
            model = wire._packet_model(self.base, 3)
            codec = wire._codec(model)
            plane = np.asarray(codec.model.plane)
            fresh_luma = int(np.count_nonzero(
                plane[codec.sent_model_indices] == 0))
            with self.subTest(wire=wire.wire_profile):
                self.assertEqual(len(codec.sent_model_indices), FRESH_SLOTS)
                self.assertEqual(int(codec.sent_luma_mask().sum()),
                                 fresh_luma+500)

    def test_cli_flag_is_opt_in(self):
        args = v7_live.parser().parse_args(
            ['send', '--device', 'null', '--source', 'test'])
        self.assertFalse(args.clip_aware_encode)
        args = v7_live.parser().parse_args(
            ['send', '--device', 'null', '--source', 'test',
             '--clip-aware-encode'])
        self.assertTrue(args.clip_aware_encode)


class DirectEncodeFallbackTests(unittest.TestCase):
    def test_frames_smaller_than_the_grid_use_the_box_resize(self):
        model = v7_live._model(v7_live.DEFAULT_FIXTURE, 'box')
        small = np.full((15, 20, 3), 200, np.uint8)
        values, aspect = v7_live._values(model, small, 'box', brightness=1.0,
                                         dct_encode=True)
        expected, _ = v7_live._values(model, small, 'box', brightness=1.0)
        np.testing.assert_array_equal(values, expected)
        self.assertEqual(aspect, v7.aspect_wire_code((20, 15)))


class DisplayGrainTests(unittest.TestCase):
    def test_flat_mask_is_one_on_flat_and_zero_on_detail(self):
        luma = np.full((ROWS, COLS), -.2, np.float32)
        luma[:, 40:] += np.where(np.arange(40) % 2, .5, 0)[None, :]
        mask = flat_area_mask(luma)
        self.assertEqual(mask.shape, luma.shape)
        self.assertEqual(mask.dtype, np.float32)
        self.assertTrue(mask.flags.c_contiguous)
        self.assertGreater(float(mask[:, :30].min()), .99)
        self.assertLess(float(mask[:, 45:].max()), .01)

    def test_shader_grain_is_off_unless_enabled(self):
        self.assertEqual(GRAIN_MODES[0], 'off')
        for declaration in ('uniform sampler2D grain_mask;',
                            'uniform float grain_amount;',
                            'uniform int grain_seed;'):
            self.assertIn(declaration, FLOAT_FRAGMENT_SHADER)
        self.assertIn('if (grain_amount > 0.0)', FLOAT_FRAGMENT_SHADER)


if __name__ == '__main__':
    unittest.main()
