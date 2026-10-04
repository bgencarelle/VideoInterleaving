"""Runtime image thumbnails used by the optional live scope source."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scope_bake import calibrate, composite_luma
from scope_image_source import RuntimeScopeImageSource


class RuntimeScopeImageSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.main_path = root / "main.png"
        self.float_path = root / "float.png"
        main = np.zeros((8, 16, 4), dtype=np.uint8)
        main[..., :3] = 255
        main[..., 3] = 255
        main[:, :8, 3] = 0
        front = np.zeros((8, 16, 4), dtype=np.uint8)
        Image.fromarray(main, "RGBA").save(self.main_path)
        Image.fromarray(front, "RGBA").save(self.float_path)
        self.source = RuntimeScopeImageSource(
            [[str(self.main_path)]], [[str(self.float_path)]], width=32)
        self.addCleanup(self.source.close)

    def test_live_luminance_uses_lazy_rgba_thumbnails(self):
        main = self.source.main_libs[0]
        front = self.source.float_libs[0]
        self.assertEqual(main.thumbs.shape, (1, 16, 32, 2))
        lum = composite_luma(main, 0, front, 0)
        self.assertEqual(lum.shape, (16, 32))
        self.assertGreater(float(lum.max()), 0.95)
        self.assertEqual(float(lum.min()), 0.0)

    def test_live_libraries_work_with_fixed_grid_calibration(self):
        result = calibrate(self.source.main_libs, self.source.float_libs,
                           1600, rows=12)
        self.assertGreaterEqual(result["grid_rows"], 12)
        self.assertLessEqual(result["grid_rows"], 16)
        self.assertGreaterEqual(result["grid_cols"], 8)
        self.assertLessEqual(result["grid_cols"], 32)

    def test_prefetch_decodes_the_next_index_without_changing_layer_pair(self):
        self.source.prefetch(0, 0, 0)
        first = self.source.main_libs[0].thumb(0)
        second = self.source.float_libs[0].thumb(0)
        self.assertFalse(first.flags.writeable)
        self.assertFalse(second.flags.writeable)

    def test_live_source_splits_side_by_side_jpeg_luminance_and_matte(self):
        path = Path(self.temp.name) / "side_by_side.jpg"
        pixels = np.empty((8, 32, 3), dtype=np.uint8)
        pixels[:, :16] = 255
        pixels[:, 16:] = 128
        Image.fromarray(pixels, "RGB").save(path, quality=100, subsampling=0)
        source = RuntimeScopeImageSource([[str(path)]], [[str(path)]], width=32)
        try:
            thumb = source.main_libs[0].thumb(0)
            self.assertEqual(thumb.shape, (16, 32, 2))
            self.assertGreater(float(thumb[..., 0].mean()), 0.98 * 255)
            self.assertAlmostEqual(float(thumb[..., 1].mean()), 128, delta=3)
        finally:
            source.close()


if __name__ == "__main__":
    unittest.main()
