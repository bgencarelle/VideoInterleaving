"""Exact selection and lossless reuse with small decoded working sets."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from image_loader import FIFOImageBuffer, ImageLoader
from modem_image_source import RuntimeImageLibrary
from scope_image_source import RuntimeScopeImageSource


class WorkingSetTests(unittest.TestCase):
    def test_scope_jpeg_content_reuse_preserves_thumbnail_pixels(self):
        rng = np.random.default_rng(91)
        with tempfile.TemporaryDirectory(dir='tmp') as directory:
            a, b = Path(directory)/'a.jpg', Path(directory)/'b.jpg'
            Image.fromarray(rng.integers(0, 256, (31, 80, 3), np.uint8)).save(a)
            b.write_bytes(a.read_bytes())
            source = RuntimeScopeImageSource([[str(a)]], [[str(b)]], width=32)
            try:
                first = source._decode(str(a))
                self.assertIs(source._decode(str(b)), first)
                self.assertFalse(first.flags.writeable)
                # Changing bytes at a previously used path must replace it.
                Image.fromarray(np.zeros((31, 80, 3), np.uint8)).save(a)
                changed = source._decode(str(a))
                self.assertIsNot(first, changed)
                self.assertTrue(np.any(first != changed))
                self.assertLessEqual(len(source._decoded_assets), 2)
            finally:
                source.close()

    def test_exact_lookup_never_substitutes_nearby_picture_and_releases_old(self):
        fifo = FIFOImageBuffer(2)
        layers = (object(), None, False, False)
        fifo.update(5, layers, (0, 0))
        self.assertIsNone(fifo.get(6, (0, 0), exact=True))
        fifo.update(6, layers, (0, 0))
        fifo.retain({(6, (0, 0))})
        self.assertEqual(fifo.current_depth(), 1)
        self.assertIsNone(fifo.get(5, (0, 0), exact=True))

    def test_content_cache_is_bounded_and_invalidates_on_file_or_scale_change(self):
        decoder = Mock()
        decoder.decode.side_effect = lambda *a, **k: np.zeros((2, 4, 3), np.uint8)
        loader = ImageLoader()
        with tempfile.TemporaryDirectory(dir='tmp') as directory, \
                patch('image_loader.jpeg', decoder):
            a, b = Path(directory)/'a.jpg', Path(directory)/'b.jpg'
            a.write_bytes(b'same'); b.write_bytes(b'same')
            first, _ = loader.read_image(str(a))
            other, _ = loader.read_image(str(b))
            self.assertIs(first, other)
            self.assertFalse(first.flags.writeable)
            self.assertEqual(decoder.decode.call_count, 1)
            loader.read_image(str(a), jpeg_scaling_factor=(1, 2))
            b.write_bytes(b'new content')
            changed, _ = loader.read_image(str(b))
            self.assertIsNot(first, changed)
            self.assertEqual(decoder.decode.call_count, 3)
            self.assertEqual(len(loader._jpeg_cache), 2)

    def test_fused_full_resolution_composite_matches_pillow_for_multiple_pictures(self):
        rng = np.random.default_rng(610)
        source = RuntimeImageLibrary.__new__(RuntimeImageLibrary)
        source.frames = 1
        for channels in (3, 4):
            for sbs in (False, True):
                for seed in range(3):
                    main = rng.integers(0, 256, (21, 32, channels), np.uint8)
                    front = rng.integers(0, 256, (17, 26, channels), np.uint8)
                    source._layers_for = lambda *a: (main, front, sbs, sbs)
                    m = source._rgba(main, sbs)
                    f = source._rgba(front, sbs)
                    expected = Image.new('RGBA', m.size, (4, 7, 9, 255))
                    expected.alpha_composite(m)
                    expected.alpha_composite(f)
                    for rotation in (0, 90, 37):
                        transformed = expected.rotate(rotation, expand=True).convert('RGB')
                        transformed = transformed.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                        actual = source.composite(0, 0, 0, (4, 7, 9), rotation, True)
                        np.testing.assert_array_equal(actual, transformed)
