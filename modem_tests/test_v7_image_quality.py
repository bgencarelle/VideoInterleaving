import unittest

import numpy as np
from PIL import Image

from tools.measure_plane_survival import plane_metrics_arrays


class V7ImageQualityTests(unittest.TestCase):
    def test_identical_images_have_perfect_metrics(self):
        image = Image.fromarray(np.full((8, 8, 3), 127, dtype=np.uint8), 'RGB')
        metrics = plane_metrics_arrays(image, image)
        for psnr, ssim, mae, normalized in metrics:
            self.assertEqual(psnr, float('inf'))
            self.assertAlmostEqual(ssim, 1.0)
            self.assertEqual(mae, 0.0)
            self.assertEqual(normalized, 0.0)

    def test_metrics_detect_image_change(self):
        reference = Image.fromarray(np.full((8, 8, 3), 127, dtype=np.uint8), 'RGB')
        decoded = Image.fromarray(np.full((8, 8, 3), 127, dtype=np.uint8), 'RGB')
        decoded.putpixel((0, 0), (255, 0, 0))
        metrics = plane_metrics_arrays(reference, decoded)
        self.assertTrue(any(np.isfinite(row[0]) and row[2] > 0 for row in metrics))
        self.assertTrue(any(row[1] < 1.0 for row in metrics))


if __name__ == '__main__':
    unittest.main()
