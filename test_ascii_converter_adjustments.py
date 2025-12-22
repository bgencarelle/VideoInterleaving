import sys
import types
import unittest

import numpy as np

try:  # pragma: no cover - exercised implicitly through ascii_converter import
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - fallback for environments without libGL
    def _resize(image, size, interpolation=None):
        new_w, new_h = size
        h, w = image.shape[:2]
        y_idx = np.linspace(0, h - 1, new_h).astype(int)
        x_idx = np.linspace(0, w - 1, new_w).astype(int)
        return image[np.ix_(y_idx, x_idx)]

    def _cvt_color(image, code):
        if code == 0:  # COLOR_RGB2HSV
            hsv = np.zeros_like(image, dtype=float)
            value = image.mean(axis=2)
            hsv[..., 2] = value
            return hsv
        if code == 1:  # COLOR_HSV2RGB
            value = image[..., 2]
            return np.stack([value, value, value], axis=2)
        if code == 2:  # COLOR_RGB2GRAY
            return image.mean(axis=2)
        raise ValueError("Unsupported color conversion code")

    def _lut(src, table):
        return table[src.astype(int)]

    cv2 = types.SimpleNamespace(
        COLOR_RGB2HSV=0,
        COLOR_HSV2RGB=1,
        COLOR_RGB2GRAY=2,
        INTER_NEAREST=0,
        resize=_resize,
        cvtColor=_cvt_color,
        LUT=_lut,
    )
    sys.modules['cv2'] = cv2
else:  # pragma: no cover - real cv2 path
    pass

import ascii_converter
import settings


class ToAsciiPreHsvAdjustmentsTest(unittest.TestCase):
    def setUp(self):
        self.original_settings = {
            'ASCII_WIDTH': settings.ASCII_WIDTH,
            'ASCII_HEIGHT': settings.ASCII_HEIGHT,
            'ASCII_FONT_RATIO': settings.ASCII_FONT_RATIO,
            'ASCII_CONTRAST': settings.ASCII_CONTRAST,
            'ASCII_SATURATION': settings.ASCII_SATURATION,
            'ASCII_BRIGHTNESS': settings.ASCII_BRIGHTNESS,
            'ASCII_COLOR': settings.ASCII_COLOR,
        }

        settings.ASCII_WIDTH = 1
        settings.ASCII_HEIGHT = 1
        settings.ASCII_FONT_RATIO = 1.0
        settings.ASCII_CONTRAST = 1.0
        settings.ASCII_SATURATION = 1.0
        settings.ASCII_BRIGHTNESS = 1.5
        settings.ASCII_COLOR = False

        self.original_chars = ascii_converter.CHARS
        self.original_gamma = ascii_converter.GAMMA_LUT

        ascii_converter.CHARS = np.array(list("abcde"))
        ascii_converter.GAMMA_LUT = np.arange(256, dtype=np.uint8)

    def tearDown(self):
        for key, value in self.original_settings.items():
            setattr(settings, key, value)

        ascii_converter.CHARS = self.original_chars
        ascii_converter.GAMMA_LUT = self.original_gamma

    def test_pre_hsv_brightness_boosts_ascii_tone(self):
        frame = np.array([[[100, 100, 100]]], dtype=np.uint8)
        ascii_frame = ascii_converter.to_ascii(frame)

        # Brightness should lighten the mapped character (toward index 0) even without
        # the legacy HSV value multiplier. For a mid-gray pixel (100) with a 1.5x
        # brightness boost, the output should shift from the middle of the palette to
        # the next brighter glyph.
        self.assertEqual(ascii_frame, f"b{ascii_converter.RESET_CODE}")


if __name__ == "__main__":
    unittest.main()
