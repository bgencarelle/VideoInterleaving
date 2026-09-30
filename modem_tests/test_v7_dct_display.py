"""Direct DCT reconstruction of decoded planes at an arbitrary size."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from animation_modem import v7
from animation_modem.v7_dct_display import (
    reconstruct_plane, reconstruct_planes, spectral_support, viewport_shapes)
from tools.v7_gl_viewer import dct_reconstruct_planes, float_planes


def _padded_idct(plane, target_shape):
    """The previous viewport reconstruction: pad the DCT, inverse at size."""
    plane = np.asarray(plane, dtype=np.float32)
    height, width = target_shape
    coefficients = dctn(plane, norm='ortho')
    resized = np.zeros((height, width), dtype=coefficients.dtype)
    rows = min(plane.shape[0], height)
    cols = min(plane.shape[1], width)
    resized[:rows, :cols] = coefficients[:rows, :cols]
    resized *= math.sqrt(height*width/plane.size)
    return idctn(resized, norm='ortho')


def _decoded_face_values():
    """Grid values as the receiver's coder hands them to the display."""
    model = v7.load_model(.1521/np.sqrt(1+10**(v7.CLOCK_REL_DB/10)), 'box')
    with Image.open(v7.REFERENCE_FIXTURE) as image:
        values = v7.image_values(v7.prepare_image(image, 'box'),
                                 model.coder.grids, 'box')
    return model.coder.inverse(model.coder.forward(values)), model.coder.grids


class DirectDCTDisplayTests(unittest.TestCase):
    def test_matches_the_padded_inverse_dct_at_any_viewport(self):
        values, grids = _decoded_face_values()
        planes = float_planes(values, grids)
        rng = np.random.default_rng(3)
        # Full-grid spectrum, as fold guests can occupy beyond the corner.
        dense = tuple(rng.normal(0, .3, plane.shape).astype(np.float32)
                      for plane in planes)
        for source in (planes, dense):
            for size in ((675, 900), (1920, 1080), (641, 333), (80, 96),
                         (40, 50)):
                with self.subTest(size=size, dense=source is dense):
                    actual = reconstruct_planes(source, size)
                    shapes = viewport_shapes(
                        tuple(p.shape for p in source), size)
                    self.assertEqual(tuple(p.shape for p in actual), shapes)
                    for plane, result, shape in zip(source, actual, shapes):
                        expected = _padded_idct(plane, shape)
                        np.testing.assert_allclose(result, expected,
                                                   rtol=0, atol=2e-5)

    def test_decoded_planes_only_evaluate_the_sent_corner(self):
        values, grids = _decoded_face_values()
        planes = float_planes(values, grids)
        for plane, shape in zip(planes, v7.V7_SHAPES):
            coefficients = dctn(plane.astype(np.float64), norm='ortho')
            self.assertEqual(spectral_support(coefficients), shape)

    def test_viewer_viewport_mode_uses_the_direct_path(self):
        values, grids = _decoded_face_values()
        planes = float_planes(values, grids)
        via_viewer = dct_reconstruct_planes(planes, 'viewport', (675, 900))
        direct = reconstruct_planes(planes, (675, 900))
        for a, b in zip(via_viewer, direct):
            np.testing.assert_array_equal(a, b)
            self.assertEqual(a.dtype, np.float32)
            self.assertTrue(a.flags.c_contiguous)

    def test_default_4x_mode_is_unchanged(self):
        values, grids = _decoded_face_values()
        planes = float_planes(values, grids)
        for plane, result in zip(planes,
                                 dct_reconstruct_planes(planes, '4x')):
            shape = (plane.shape[0]*4, plane.shape[1]*4)
            np.testing.assert_allclose(
                result, _padded_idct(plane, shape).astype(np.float32),
                rtol=0, atol=0)

    def test_flat_and_empty_planes(self):
        flat = np.full((48, 40), -.3, np.float32)
        np.testing.assert_allclose(reconstruct_plane(flat, (7, 1000)), -.3,
                                   atol=1e-6)
        zero = np.zeros((48, 40), np.float32)
        np.testing.assert_array_equal(reconstruct_plane(zero, (5, 6)), 0.0)
        with self.assertRaisesRegex(ValueError, 'positive'):
            reconstruct_plane(flat, (0, 4))


if __name__ == '__main__':
    unittest.main()
