import sys
import types
import unittest

import numpy as np


def _make_stub_cv2():
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

    return types.SimpleNamespace(
        COLOR_RGB2HSV=0,
        COLOR_HSV2RGB=1,
        COLOR_RGB2GRAY=2,
        INTER_NEAREST=0,
        resize=_resize,
        cvtColor=_cvt_color,
        LUT=_lut,
    )


try:  # pragma: no cover - exercised implicitly through ascii_converter import
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - fallback for environments without libGL
    cv2 = _make_stub_cv2()
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
        settings.ASCII_BRIGHTNESS = 1.0
        settings.ASCII_COLOR = False

        self.original_chars = ascii_converter.CHARS
        self.original_gamma = ascii_converter.GAMMA_LUT
        self.original_cv2 = ascii_converter.cv2

        ascii_converter.CHARS = np.array(list("abcde"))
        ascii_converter.GAMMA_LUT = np.arange(256, dtype=np.uint8)
        ascii_converter.cv2 = _make_stub_cv2()

    def tearDown(self):
        for key, value in self.original_settings.items():
            setattr(settings, key, value)

        ascii_converter.CHARS = self.original_chars
        ascii_converter.GAMMA_LUT = self.original_gamma
        ascii_converter.cv2 = self.original_cv2

    def test_pre_hsv_brightness_boosts_ascii_tone(self):
        frame = np.array([[[100, 100, 100]]], dtype=np.uint8)
        baseline = ascii_converter.to_ascii(frame)

        settings.ASCII_BRIGHTNESS = 1.5
        boosted = ascii_converter.to_ascii(frame)

        reset = ascii_converter.RESET_CODE
        base_char = baseline[:-len(reset)]
        boosted_char = boosted[:-len(reset)]

        chars = ascii_converter.CHARS.tolist()
        self.assertLess(chars.index(boosted_char), chars.index(base_char))


class ToAsciiColorParityTest(unittest.TestCase):
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
        settings.ASCII_BRIGHTNESS = 1.0
        settings.ASCII_COLOR = True

        self.original_chars = ascii_converter.CHARS
        self.original_gamma = ascii_converter.GAMMA_LUT
        self.original_contrast_lut = ascii_converter.CONTRAST_LUT
        self.original_rgb_flag = ascii_converter._apply_rgb_brightness
        self.original_rgb_gain = ascii_converter._rgb_brightness
        self.original_cv2 = ascii_converter.cv2

        ascii_converter.CHARS = np.array(list("Xyz"))
        ascii_converter.GAMMA_LUT = np.arange(256, dtype=np.uint8)
        ascii_converter.cv2 = _make_stub_cv2()

    def tearDown(self):
        for key, value in self.original_settings.items():
            setattr(settings, key, value)

        ascii_converter.CHARS = self.original_chars
        ascii_converter.GAMMA_LUT = self.original_gamma
        ascii_converter.CONTRAST_LUT = self.original_contrast_lut
        ascii_converter._apply_rgb_brightness = self.original_rgb_flag
        ascii_converter._rgb_brightness = self.original_rgb_gain
        ascii_converter.cv2 = self.original_cv2

    def test_pre_hsv_rgb_brightness_skips_hsv_value_boost(self):
        ascii_converter.CONTRAST_LUT = None
        ascii_converter._apply_rgb_brightness = True
        ascii_converter._rgb_brightness = np.array([2.0, 1.0, 1.0])

        frame = np.array([[[30, 40, 50]]], dtype=np.uint8)
        settings.ASCII_BRIGHTNESS = 2.0
        bright_frame = ascii_converter.to_ascii(frame)

        settings.ASCII_BRIGHTNESS = 1.0
        baseline = ascii_converter.to_ascii(frame)

        expected = f"\033[38;5;235my{ascii_converter.RESET_CODE}"
        self.assertEqual(bright_frame, baseline)
        self.assertEqual(baseline, expected)

    def test_contrast_lut_disables_hsv_value_multiplier(self):
        ascii_converter.CONTRAST_LUT = np.arange(256, dtype=np.uint8)
        ascii_converter._apply_rgb_brightness = False
        ascii_converter._rgb_brightness = np.ones(3, dtype=float)
        ascii_converter.CHARS = np.array(list("abc"))

        frame = np.array([[[100, 100, 100]]], dtype=np.uint8)
        settings.ASCII_COLOR = False
        settings.ASCII_BRIGHTNESS = 1.8
        bright_frame = ascii_converter.to_ascii(frame)

        settings.ASCII_BRIGHTNESS = 1.0
        baseline = ascii_converter.to_ascii(frame)

        expected = f"b{ascii_converter.RESET_CODE}"
        self.assertEqual(bright_frame, baseline)
        self.assertEqual(baseline, expected)


if __name__ == "__main__":
    unittest.main()
