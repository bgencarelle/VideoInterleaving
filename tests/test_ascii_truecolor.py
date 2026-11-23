"""Truecolour preserves tint gradients without changing character exposure."""
import re
import unittest
from unittest.mock import patch

import numpy as np

import settings
import ascii_converter
from ascii_encoding import encode_rgb_rows


def glyphs(text):
    return re.sub(r'\x1b\[[\d;]*m', '', text)


def tints(text):
    return [tuple(map(int, match)) for match in
            re.findall(r'\x1b\[38;2;(\d+);(\d+);(\d+)m', text)]


class TruecolorTests(unittest.TestCase):
    def setUp(self):
        for name, value in (('ASCII_COLOR', True), ('ASCII_TRUECOLOR', True),
                            ('ASCII_COLOR_BLUR', 0), ('ASCII_COLOR_BRIGHTNESS', 1.0),
                            ('ASCII_CONTRAST', 1.0), ('ASCII_SATURATION', .9),
                            ('ASCII_BRIGHTNESS', 1.4)):
            patcher = patch.object(settings, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_skin_gradient_keeps_shading_and_same_glyphs(self):
        shade = np.linspace(.45, 1.15, 101)
        rgb = np.clip(shade[:, None] * [190,135,110],0,255).astype(np.uint8)[None,:,:]
        text = ascii_converter.to_ascii(rgb,width=101,height=1,sampled=True)
        with patch.object(settings,'ASCII_TRUECOLOR',False):
            palette = ascii_converter.to_ascii(rgb,width=101,height=1,sampled=True)
        self.assertEqual(glyphs(text),glyphs(palette))
        self.assertGreater(len(set(tints(text))),90)
        # The old clipped highlights made these two shades identical.
        light = np.array([[[191,136,111],[218,155,126]]],np.uint8)
        colors = tints(ascii_converter.to_ascii(light,width=2,height=1,sampled=True))
        self.assertEqual(len(colors),2)
        self.assertNotEqual(colors[0],colors[1])
        self.assertTrue(all(r<255 for r,g,b in colors))

    def test_opaque_white_remains_white_and_near_white_is_not_dimmed(self):
        rgb = np.array([[[255,255,255],[254,254,254],[255,250,240]]],np.uint8)
        colors = tints(ascii_converter.to_ascii(rgb,width=3,height=1,sampled=True))
        self.assertEqual(colors[:2],[(255,255,255),(254,254,254)])
        self.assertGreaterEqual(min(colors[2]),240)

    def test_glyph_brightness_and_tint_brightness_are_independent(self):
        rgb = np.array([[[100,70,50]]],np.uint8)
        baseline = ascii_converter.to_ascii(rgb,width=1,height=1,sampled=True)
        with patch.object(settings,'ASCII_BRIGHTNESS',2.0):
            bright_glyph = ascii_converter.to_ascii(rgb,width=1,height=1,sampled=True)
        self.assertEqual(tints(baseline),tints(bright_glyph))
        self.assertNotEqual(glyphs(baseline),glyphs(bright_glyph))
        with patch.object(settings,'ASCII_COLOR_BRIGHTNESS',1.2):
            bright_tint = ascii_converter.to_ascii(rgb,width=1,height=1,sampled=True)
        self.assertEqual(glyphs(baseline),glyphs(bright_tint))
        self.assertNotEqual(tints(baseline),tints(bright_tint))

    def test_contrast_and_gamma_remain_connected(self):
        rgb = np.array([[[100,70,50]]],np.uint8)
        baseline = ascii_converter.to_ascii(rgb,width=1,height=1,sampled=True)
        with patch.object(settings,'ASCII_CONTRAST',1.8):
            contrast = ascii_converter.to_ascii(rgb,width=1,height=1,sampled=True)
        self.assertNotEqual(glyphs(baseline),glyphs(contrast))
        self.assertNotEqual(tints(baseline),tints(contrast))
        with patch.object(settings,'ASCII_GAMMA',2.0):
            gamma = ascii_converter.to_ascii(rgb,width=1,height=1,sampled=True)
        self.assertNotEqual(glyphs(baseline),glyphs(gamma))
        self.assertEqual(tints(baseline),tints(gamma))

    def test_encoder_preserves_utf8_runs_rows_and_decimal_channels(self):
        rng = np.random.default_rng(167)
        for palette in ('MW .','Mé█漢😀 '):
            chars = np.array(list(palette),dtype='U1')[rng.integers(0,len(palette),(8,12))]
            rgb = rng.integers(0,256,(8,12,3),dtype=np.uint8)
            rgb[0,:] = (0,9,255)
            rgb[1,:] = (10,99,100)
            rows = []
            for row in range(8):
                parts = []; previous = None
                for col in range(12):
                    color = tuple(map(int,rgb[row,col]))
                    if color != previous:
                        parts.append(f'\x1b[38;2;{color[0]};{color[1]};{color[2]}m')
                        previous = color
                    parts.append(chars[row,col])
                rows.append(''.join(parts))
            self.assertEqual(encode_rgb_rows(rgb,chars),'\r\n'.join(rows))
