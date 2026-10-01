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
    edge_consistent_plane, reconstruct_plane, reconstruct_planes,
    spectral_support, viewport_shapes)
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

    def test_16x_mode_matches_the_padded_inverse_dct(self):
        values, grids = _decoded_face_values()
        planes = float_planes(values, grids)
        enlarged = dct_reconstruct_planes(planes, '16x')
        for plane, result in zip(planes, enlarged):
            shape = (plane.shape[0]*16, plane.shape[1]*16)
            self.assertEqual(result.shape, shape)
            self.assertEqual(result.dtype, np.float32)
            self.assertTrue(result.flags.c_contiguous)
            np.testing.assert_allclose(result, _padded_idct(plane, shape),
                                       rtol=0, atol=2e-5)

    def test_flat_and_empty_planes(self):
        flat = np.full((48, 40), -.3, np.float32)
        np.testing.assert_allclose(reconstruct_plane(flat, (7, 1000)), -.3,
                                   atol=1e-6)
        zero = np.zeros((48, 40), np.float32)
        np.testing.assert_array_equal(reconstruct_plane(zero, (5, 6)), 0.0)
        with self.assertRaisesRegex(ValueError, 'positive'):
            reconstruct_plane(flat, (0, 4))


def _bars_plane(rows=96, cols=80, kept=(48, 40)):
    """A band-limited plane of hard bars: what the decoder hands the display
    for flat-shaded content (rings along every edge)."""
    plane = np.full((rows, cols), -.8)
    plane[20:76, 10:18] = .8
    plane[20:76, 30:34] = .8
    plane[40:44, :] = .8
    coefficients = dctn(plane, norm='ortho')
    coefficients[kept[0]:, :] = 0
    coefficients[:, kept[1]:] = 0
    return idctn(coefficients, norm='ortho').astype(np.float32), plane


class EdgeReconstructionTests(unittest.TestCase):
    def test_received_coefficients_are_kept(self):
        decoded, _ = _bars_plane()
        rebuilt = edge_consistent_plane(decoded)
        self.assertEqual(rebuilt.shape, (192, 160))
        received = dctn(decoded.astype(np.float64), norm='ortho')[:48, :40]
        kept = dctn(rebuilt.astype(np.float64), norm='ortho')[:48, :40]/2
        # Exact up to float32 rounding of the working grid.
        np.testing.assert_allclose(kept, received,
                                   atol=1e-3*float(np.abs(received).max()))

    def test_ringing_goes_and_edges_sharpen(self):
        decoded, truth = _bars_plane()
        plain = reconstruct_plane(decoded, (192, 160))
        rebuilt = edge_consistent_plane(decoded)
        from scipy.ndimage import binary_erosion
        target = np.kron(truth, np.ones((2, 2)))
        flat = binary_erosion(target < 0, iterations=4)
        # Ripple in the flat background (measured: -65%), and error against
        # the hard bars (-32%).
        self.assertLess(float(np.std(rebuilt[flat])), .5*float(np.std(plain[flat])))
        self.assertLess(float(np.mean(np.abs(rebuilt-target))),
                        .8*float(np.mean(np.abs(plain-target))))

    def test_flat_planes_stay_flat(self):
        flat = np.full((96, 80), .25, dtype=np.float32)
        np.testing.assert_allclose(edge_consistent_plane(flat), .25, atol=1e-4)

    def test_viewer_keeps_output_sizes(self):
        values, grids = _decoded_face_values()
        planes = float_planes(values, grids)
        for mode, size in (('4x', None), ('viewport', (640, 768))):
            plain = dct_reconstruct_planes(planes, mode, size)
            edged = dct_reconstruct_planes(planes, mode, size, edge=True)
            with self.subTest(mode=mode):
                self.assertEqual([p.shape for p in edged], [p.shape for p in plain])
                np.testing.assert_array_equal(edged[1], plain[1])
        off = dct_reconstruct_planes(planes, 'off', edge=True)
        self.assertEqual(off[0].shape, (192, 160))
        self.assertEqual(off[1].shape, planes[1].shape)


if __name__ == '__main__':
    unittest.main()
