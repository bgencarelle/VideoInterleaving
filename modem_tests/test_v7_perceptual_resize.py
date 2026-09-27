"""Sender-side V7 perceptual resize ablations and compatibility boundary."""
import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image

from animation_modem import v7
from animation_modem.perceptual_resize import (
    RESIZE_MODES, _axis_footprints, _linear_to_srgb, _srgb_to_linear,
    resize_rgb,
)
from tools import v7_live


def reference_area_resize(rgb, mode, strength, target_size):
    """Small independent implementation of the specified overlap equation."""
    tw, th = target_size
    source = np.asarray(rgb, np.float64)/255.0
    linear = mode.startswith('linear-')
    if linear:
        source = _srgb_to_linear(source)
    height, width, _ = source.shape
    output = np.zeros((th, tw, 3), np.float64)
    for oy in range(th):
        y0, y1 = oy*height/th, (oy+1)*height/th
        ys = [(sy, max(0.0, min(y1, sy+1)-max(y0, sy)))
              for sy in range(int(np.floor(y0)), min(height, int(np.ceil(y1))))]
        for ox in range(tw):
            x0, x1 = ox*width/tw, (ox+1)*width/tw
            xs = [(sx, max(0.0, min(x1, sx+1)-max(x0, sx)))
                  for sx in range(int(np.floor(x0)), min(width, int(np.ceil(x1))))]
            contributors = [(source[sy, sx], wy*wx)
                            for sy, wy in ys for sx, wx in xs if wy*wx > 0]
            area = sum(weight for _, weight in contributors)
            mean = sum(pixel*weight for pixel, weight in contributors)/area
            actual_strength = 0.0 if mode == 'linear-box' else strength
            if actual_strength:
                distances = [np.sqrt(np.mean((pixel-mean)**2))
                             for pixel, _ in contributors]
                scale = max(np.sqrt(sum(weight*distance**2
                                        for (_, weight), distance in
                                        zip(contributors, distances))/area), 1/255)
                weights = [weight*(1+actual_strength*min(distance/scale, 2))
                           for (_, weight), distance in zip(contributors, distances)]
                output[oy, ox] = sum(pixel*weight for (pixel, _), weight
                                     in zip(contributors, weights))/sum(weights)
            else:
                output[oy, ox] = mean
    if linear:
        output = _linear_to_srgb(output)
    return np.uint8(np.rint(np.clip(output, 0, 1)*255))


class PerceptualResizeTests(unittest.TestCase):
    def test_standard_srgb_transfer_reference_points(self):
        self.assertAlmostEqual(float(_srgb_to_linear(np.array([0.04045]))[0]),
                               0.04045/12.92, places=8)
        self.assertAlmostEqual(float(_linear_to_srgb(np.array([0.0031308]))[0]),
                               0.0031308*12.92, places=8)
        np.testing.assert_allclose(
            _linear_to_srgb(_srgb_to_linear(np.linspace(0, 1, 257))),
            np.linspace(0, 1, 257), atol=1e-12)

    def test_footprints_cover_each_output_area_for_unusual_sizes(self):
        for source_size, target_size in ((1, 7), (7, 3), (11, 4), (13, 13)):
            indexes, weights, counts = _axis_footprints(source_size, target_size)
            self.assertEqual(indexes.shape, weights.shape)
            np.testing.assert_allclose(
                np.sum(weights, axis=1), source_size/target_size, atol=1e-12)
            for out_index, count in enumerate(counts):
                self.assertTrue(np.all(indexes[out_index, :count] < source_size))
                self.assertTrue(np.all(weights[out_index, :count] > 0))

    def test_each_resize_matches_direct_area_equation(self):
        y, x = np.mgrid[:7, :11]
        rgb = np.stack(((x*29+y*7) % 256, (x*3+y*31) % 256,
                        ((x+y)*19) % 256), axis=-1).astype(np.uint8)
        for mode in RESIZE_MODES:
            strength = 0.5
            actual = resize_rgb(rgb, mode, strength, (4, 3))
            expected = reference_area_resize(rgb, mode, strength, (4, 3))
            np.testing.assert_array_equal(actual, expected, err_msg=mode)

    def test_zero_detail_and_constant_images_preserve_area_means(self):
        y, x = np.mgrid[:9, :13]
        rgb = np.stack(((x*17+y*9) % 256, (x*5+y*23) % 256,
                        (x*11+y*13) % 256), axis=-1).astype(np.uint8)
        zero = resize_rgb(rgb, 'gamma-detail', 0.0, (5, 4))
        np.testing.assert_array_equal(
            zero, reference_area_resize(rgb, 'gamma-detail', 0.0, (5, 4)))
        constant = np.full((9, 13, 3), (40, 127, 230), np.uint8)
        for mode in RESIZE_MODES:
            np.testing.assert_array_equal(
                resize_rgb(constant, mode, 1.0, (5, 4)),
                np.broadcast_to(np.array([40, 127, 230], np.uint8), (4, 5, 3)))

    def test_identity_dimensions_are_exact_and_results_are_deterministic(self):
        image = np.random.default_rng(13).integers(0, 256, (96, 80, 3), np.uint8)
        for mode in RESIZE_MODES:
            first = resize_rgb(image, mode, 0.25)
            np.testing.assert_array_equal(first, image)
            np.testing.assert_array_equal(first, resize_rgb(image, mode, 0.25))

    def test_bounds_and_input_validation(self):
        rgb = np.zeros((2, 3, 3), np.uint8)
        for bad in (-0.01, 1.01, float('nan')):
            with self.subTest(strength=bad), self.assertRaises(ValueError):
                resize_rgb(rgb, 'linear-detail', bad, (2, 2))
        with self.assertRaises(ValueError):
            resize_rgb(rgb, 'off', 0.25, (2, 2))
        with self.assertRaises(ValueError):
            resize_rgb(np.zeros((2, 3), np.uint8), 'linear-box', 0.25, (2, 2))

    def test_sender_off_path_is_byte_identical(self):
        class Model:
            coder = SimpleNamespace(grids=v7.V7_GRIDS)

        image = Image.fromarray(np.random.default_rng(7).integers(
            0, 256, (131, 73, 3), np.uint8))
        for encode_filter in ('nearest', 'box'):
            expected_prepared = v7.prepare_image(image, encode_filter)
            expected = v7.image_values(expected_prepared, v7.V7_GRIDS,
                                       encode_filter=encode_filter)
            actual, _ = v7_live._values(
                Model(), image, encode_filter, brightness=1.0, gamma=1.0)
            np.testing.assert_array_equal(actual, expected)

    def test_sender_opt_in_preprocesses_before_existing_value_conversion(self):
        class Model:
            coder = SimpleNamespace(grids=v7.V7_GRIDS)

        image = Image.fromarray(np.random.default_rng(8).integers(
            0, 256, (127, 79, 3), np.uint8))
        values, aspect = v7_live._values(
            Model(), image, 'box', brightness=1.0, gamma=1.0,
            perceptual_resize='linear-detail',
            perceptual_detail_strength=0.5)
        self.assertEqual(values.shape, (sum(r*c for r, c in v7.V7_GRIDS),))
        self.assertEqual(aspect, v7.aspect_wire_code(image.size))
        prepared = Image.fromarray(resize_rgb(
            np.asarray(image), 'linear-detail', 0.5))
        expected = v7.image_values(prepared, v7.V7_GRIDS, encode_filter='box')
        np.testing.assert_array_equal(values, expected)
        with self.assertRaisesRegex(ValueError, 'requires --encode-filter box'):
            v7_live._values(Model(), image, 'nearest', perceptual_resize='linear-box')

    def test_sender_rejects_tone_controls_that_can_emit_black_or_nan(self):
        class Model:
            coder = SimpleNamespace(grids=v7.V7_GRIDS)

        image = Image.new('RGB', (8, 8), (40, 120, 220))
        for brightness in (0.0, -1.0, float('nan'), float('inf')):
            with self.subTest(brightness=brightness), self.assertRaisesRegex(
                    ValueError, '--brightness must be finite and positive'):
                v7_live._values(Model(), image, brightness=brightness)
        for gamma in (0.0, -1.0, float('nan'), float('inf')):
            with self.subTest(gamma=gamma), self.assertRaisesRegex(
                    ValueError, '--gamma must be finite and positive'):
                v7_live._values(Model(), image, gamma=gamma)

    def test_profile_validation_rejects_invalid_tone_controls_before_capture(self):
        args = v7_live.parser().parse_args([
            'send', '--device', 'loopback', '--source', 'test'])
        for attribute, bad_values in (
                ('brightness', (0.0, -1.0, float('nan'), float('inf'))),
                ('gamma', (0.0, -1.0, float('nan'), float('inf')))):
            for value in bad_values:
                args.brightness = None
                args.gamma = 1.0
                setattr(args, attribute, value)
                message = f'--{attribute} must be finite and positive'
                with self.subTest(option=attribute, value=value), \
                        self.assertRaisesRegex(ValueError, message):
                    v7_live._send_profile(args, 0)
        args.brightness = None
        args.gamma = 1.0
        self.assertEqual(v7_live._send_profile(args, 0), ('nearest', 1.05))

    def test_perceptual_sender_requires_canonical_pinned_fold_model(self):
        parser = v7_live.parser()
        args = parser.parse_args([
            'send', '--device', 'loopback', '--source', 'test',
            '--perceptual-resize', 'linear-box'])
        self.assertEqual(v7_live._send_profile(args, 500), ('box', 1.0))
        with self.assertRaisesRegex(ValueError, 'requires.*fold'):
            v7_live._send_profile(args, 0)
        args.encode_filter = 'nearest'
        with self.assertRaisesRegex(ValueError, 'requires --encode-filter box'):
            v7_live._send_profile(args, 500)
        args.encode_filter = 'box'
        args.fixture = v7.REFERENCE_FIXTURE.parent/'other.png'
        with self.assertRaisesRegex(ValueError, 'canonical V7 fixture'):
            v7_live._send_profile(args, 500)


if __name__ == '__main__':
    unittest.main()
