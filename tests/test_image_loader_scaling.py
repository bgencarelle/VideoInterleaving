"""Web decode scaling is passed through and guarded by each JPEG's size."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import image_loader
from turbojpeg import TJPF_RGB


class _FakeJPEG:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.decoded_scale = "unset"

    def decode_header(self, _data):
        return self.width, self.height, 0, 0

    def decode(self, _data, pixel_format, scaling_factor=None):
        assert pixel_format == TJPF_RGB
        self.decoded_scale = scaling_factor
        return np.zeros((2, 3, 3), dtype=np.uint8)


class ImageLoaderScalingTests(unittest.TestCase):
    def test_web_scale_is_used_only_when_each_jpeg_covers_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.jpg"
            path.write_bytes(b"jpeg")
            for dimensions, expected in (((1440, 960), (1, 2)),
                                         ((1200, 960), None),
                                         ((1440, 760), None)):
                with self.subTest(dimensions=dimensions):
                    decoder = _FakeJPEG(*dimensions)
                    with patch.object(image_loader, "jpeg", decoder):
                        image, is_sbs = image_loader.ImageLoader().read_image(
                            str(path), jpeg_scaling_factor=(1, 2),
                            jpeg_min_size=(1280, 800))
                    self.assertEqual(decoder.decoded_scale, expected)
                    self.assertTrue(is_sbs)
                    self.assertEqual(image.shape, (2, 3, 3))


if __name__ == "__main__":
    unittest.main()
