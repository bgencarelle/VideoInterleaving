import unittest

import numpy as np
from PIL import Image

from animation_modem import v7


class V7PrepareImageTests(unittest.TestCase):
    def test_integer_grid_dimensions_reduce_with_at_most_one_code_error(self):
        rng = np.random.default_rng(20260930)
        for width, height in ((400, 480), (640, 480), (720, 960),
                              (1280, 768), (1440, 960)):
            with self.subTest(size=(width, height)):
                pixels = rng.integers(
                    0, 256, size=(height, width, 3), dtype=np.uint8)
                source = Image.fromarray(pixels)

                actual = np.asarray(
                    v7.prepare_image(source, 'box'), dtype=np.int16)
                generic = np.asarray(source.resize(
                    v7.PREPARED_SIZE, Image.Resampling.BOX), dtype=np.int16)

                self.assertEqual(actual.shape, (96, 80, 3))
                self.assertLessEqual(int(np.max(np.abs(actual-generic))), 1)

    def test_non_integer_box_and_other_filters_use_regular_resize_results(self):
        rng = np.random.default_rng(20260930)
        cases = []
        for width, height in ((317, 251), (1280, 720), (1920, 1080),
                              (1080, 1920)):
            pixels = rng.integers(
                0, 256, size=(height, width, 3), dtype=np.uint8)
            cases.append(('box', Image.fromarray(pixels)))
        exact_pixels = rng.integers(
            0, 256, size=(960, 720, 3), dtype=np.uint8)
        cases.append(('lanczos', Image.fromarray(exact_pixels)))

        for name, source in cases:
            with self.subTest(encode_filter=name, size=source.size):
                expected = source.convert('RGB').resize(
                    v7.PREPARED_SIZE,
                    getattr(Image.Resampling, name.upper()))
                actual = v7.prepare_image(source, name)
                np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
