"""Sender-side V7 perceptual resize ablations and compatibility boundary."""
import unittest
from types import SimpleNamespace

import numpy as np
from numba import njit
from PIL import Image

from animation_modem import v7
from animation_modem import perceptual_resize
from animation_modem.perceptual_resize import (
    RESIZE_MODES, _LINEAR_ROUND_THRESHOLDS, _SRGB8_TO_LINEAR, _axis_footprints,
    _linear_to_srgb, _quantize_output, _srgb_to_linear, resize_rgb,
    warmup_resize,
)
from tools import v7_live


# The first, direct per-pixel implementation, kept verbatim as the byte-exact
# reference for the faster kernels in animation_modem/perceptual_resize.py.
@njit(nogil=True, cache=True)
def _legacy_kernel(source, yi, yw, yc, xi, xw, xc, strength):
    height, width, _ = source.shape
    out_height, out_width = len(yc), len(xc)
    output = np.empty((out_height, out_width, 3), np.float64)
    for oy in range(out_height):
        for ox in range(out_width):
            area = 0.0
            mean0 = 0.0
            mean1 = 0.0
            mean2 = 0.0
            for yk in range(yc[oy]):
                sy = yi[oy, yk]
                wy = yw[oy, yk]
                for xk in range(xc[ox]):
                    sx = xi[ox, xk]
                    weight = wy*xw[ox, xk]
                    area += weight
                    mean0 += weight*source[sy, sx, 0]
                    mean1 += weight*source[sy, sx, 1]
                    mean2 += weight*source[sy, sx, 2]
            mean0 /= area
            mean1 /= area
            mean2 /= area
            if strength == 0.0:
                output[oy, ox, 0] = mean0
                output[oy, ox, 1] = mean1
                output[oy, ox, 2] = mean2
                continue

            weighted_distance2 = 0.0
            for yk in range(yc[oy]):
                sy = yi[oy, yk]
                wy = yw[oy, yk]
                for xk in range(xc[ox]):
                    sx = xi[ox, xk]
                    weight = wy*xw[ox, xk]
                    d0 = source[sy, sx, 0]-mean0
                    d1 = source[sy, sx, 1]-mean1
                    d2 = source[sy, sx, 2]-mean2
                    distance2 = (d0*d0+d1*d1+d2*d2)/3.0
                    weighted_distance2 += weight*distance2
            scale = max(np.sqrt(weighted_distance2/area), 1.0/255.0)
            weighted_total = 0.0
            accum0 = 0.0
            accum1 = 0.0
            accum2 = 0.0
            for yk in range(yc[oy]):
                sy = yi[oy, yk]
                wy = yw[oy, yk]
                for xk in range(xc[ox]):
                    sx = xi[ox, xk]
                    base_weight = wy*xw[ox, xk]
                    d0 = source[sy, sx, 0]-mean0
                    d1 = source[sy, sx, 1]-mean1
                    d2 = source[sy, sx, 2]-mean2
                    distance = np.sqrt((d0*d0+d1*d1+d2*d2)/3.0)
                    detail = min(distance/scale, 2.0)
                    weight = base_weight*(1.0+strength*detail)
                    weighted_total += weight
                    accum0 += weight*source[sy, sx, 0]
                    accum1 += weight*source[sy, sx, 1]
                    accum2 += weight*source[sy, sx, 2]
            output[oy, ox, 0] = accum0/weighted_total
            output[oy, ox, 1] = accum1/weighted_total
            output[oy, ox, 2] = accum2/weighted_total
    return output


@njit(nogil=True, cache=True)
def _legacy_input_domain(pixels, linear, table):
    height, width, _ = pixels.shape
    output = np.empty((height, width, 3), np.float64)
    for y in range(height):
        for x in range(width):
            for channel in range(3):
                if linear:
                    output[y, x, channel] = table[pixels[y, x, channel]]
                else:
                    output[y, x, channel] = pixels[y, x, channel]/255.0
    return output


@njit(nogil=True, cache=True)
def _legacy_quantize(resized, linear, thresholds):
    height, width, _ = resized.shape
    output = np.empty((height, width, 3), np.uint8)
    for y in range(height):
        for x in range(width):
            for channel in range(3):
                value = resized[y, x, channel]
                if linear:
                    value = min(max(value, 0.0), 1.0)
                    low, high = 0, len(thresholds)
                    while low < high:
                        middle = (low+high)//2
                        if thresholds[middle] < value:
                            low = middle+1
                        else:
                            high = middle
                    code = low
                    if (code < 255 and value == thresholds[code]
                            and code % 2):
                        code += 1
                    output[y, x, channel] = np.uint8(code)
                else:
                    value = min(max(value, 0.0), 1.0)
                    output[y, x, channel] = np.uint8(np.rint(value*255.0))
    return output


def legacy_resize_rgb(rgb, mode, strength, target_size=(80, 96)):
    linear = mode.startswith('linear-')
    pixels = np.asarray(rgb)
    source = _legacy_input_domain(pixels, linear, _SRGB8_TO_LINEAR)
    yi, yw, yc = _axis_footprints(pixels.shape[0], target_size[1])
    xi, xw, xc = _axis_footprints(pixels.shape[1], target_size[0])
    resized = _legacy_kernel(source, yi, yw, yc, xi, xw, xc,
                             0.0 if mode == 'linear-box' else float(strength))
    return _legacy_quantize(resized, linear, _LINEAR_ROUND_THRESHOLDS)


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

    def test_fast_kernels_match_the_direct_implementation_byte_for_byte(self):
        rng = np.random.default_rng(2026)
        with Image.open(v7.REFERENCE_FIXTURE) as image:
            face = image.convert('RGB')
        for size in ((160, 213), (160, 90), (131, 73), (320, 427), (7, 11),
                     (1, 1)):
            smooth = np.asarray(face.resize(size, Image.Resampling.BILINEAR))
            grain = np.clip(smooth.astype(int)+rng.integers(-3, 4, smooth.shape),
                            0, 255).astype(np.uint8)
            contents = {
                'face': np.asarray(face.resize(size, Image.Resampling.LANCZOS)),
                'noise': rng.integers(0, 256, (size[1], size[0], 3), np.uint8),
                'grain': grain,
            }
            for content, rgb in contents.items():
                for mode in RESIZE_MODES:
                    for strength in (0.0, 0.25, 1.0):
                        for target in ((80, 96), (4, 3), (13, 7)):
                            with self.subTest(size=size, content=content,
                                              mode=mode, strength=strength,
                                              target=target):
                                np.testing.assert_array_equal(
                                    resize_rgb(rgb, mode, strength, target),
                                    legacy_resize_rgb(rgb, mode, strength,
                                                      target))

    def test_fast_kernels_match_the_direct_floats_exactly(self):
        # Stricter than the bytes: any reordering of the per-pixel arithmetic
        # shows up here even when it rounds to the same code.
        rng = np.random.default_rng(17)
        for size, target in (((160, 213), (80, 96)), ((131, 73), (13, 7))):
            rgb = rng.integers(0, 256, (size[1], size[0], 3), np.uint8)
            yi, yw, yc = _axis_footprints(size[1], target[1])
            xi, xw, xc = _axis_footprints(size[0], target[0])
            xidx, xwt = perceptual_resize._padded_footprints(size[0], target[0])
            for linear in (False, True):
                table = (_SRGB8_TO_LINEAR if linear
                         else perceptual_resize._SRGB8_TO_GAMMA)
                source = _legacy_input_domain(rgb, linear, _SRGB8_TO_LINEAR)
                np.testing.assert_array_equal(
                    perceptual_resize._to_float(rgb, table), source)
                np.testing.assert_array_equal(
                    perceptual_resize._area_mean_kernel(
                        source, yi, yw, yc, xi, xw, xc),
                    _legacy_kernel(source, yi, yw, yc, xi, xw, xc, 0.0))
                planes = perceptual_resize._to_planes(rgb, table)
                for strength in (0.25, 1.0):
                    with self.subTest(size=size, linear=linear,
                                      strength=strength):
                        np.testing.assert_array_equal(
                            perceptual_resize._detail_kernel(
                                planes, yi, yw, yc, xidx, xwt, strength),
                            _legacy_kernel(source, yi, yw, yc, xi, xw, xc,
                                           strength))

    def test_linear_quantizer_matches_binary_search_at_every_threshold(self):
        rng = np.random.default_rng(5)
        values = np.concatenate((
            _LINEAR_ROUND_THRESHOLDS,
            np.nextafter(_LINEAR_ROUND_THRESHOLDS, 0.0),
            np.nextafter(_LINEAR_ROUND_THRESHOLDS, 1.0),
            rng.random(30000), rng.random(3000)*1e-3,
            [0.0, 1.0, np.nextafter(1.0, 0.0), 5e-324, -0.5, 1.5, -0.0]))
        values = values[:len(values)//3*3].reshape(1, -1, 3)
        for linear in (True, False):
            np.testing.assert_array_equal(
                _quantize_output(values, linear),
                _legacy_quantize(values, linear, _LINEAR_ROUND_THRESHOLDS))

    def test_warmup_covers_read_only_capture_frames(self):
        # Live frames are read-only (np.frombuffer over the FFmpeg pipe) and
        # Numba compiles those separately; warmup must leave nothing to
        # compile on the first live frame.
        kernels = (perceptual_resize._to_float, perceptual_resize._to_planes,
                   perceptual_resize._area_mean_kernel,
                   perceptual_resize._detail_kernel,
                   perceptual_resize._quantize_kernel)
        for mode, strength in (('linear-box', 0.25), ('gamma-detail', 0.0),
                               ('gamma-detail', 0.25), ('linear-detail', 1.0)):
            warmup_resize((2, 2), mode, strength)
            compiled = [len(kernel.signatures) for kernel in kernels]
            frame = np.frombuffer(
                np.random.default_rng(3).integers(
                    0, 256, 213*160*3, np.uint8).tobytes(),
                np.uint8).reshape(213, 160, 3)
            self.assertFalse(frame.flags.writeable)
            resize_rgb(frame, mode, strength)
            resize_rgb(frame.copy(), mode, strength)
            with self.subTest(mode=mode, strength=strength):
                self.assertEqual(
                    [len(kernel.signatures) for kernel in kernels], compiled)

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

    def test_sender_array_frames_match_the_pillow_path(self):
        class Model:
            coder = SimpleNamespace(grids=v7.V7_GRIDS)

        rng = np.random.default_rng(11)
        pixels = rng.integers(0, 256, (131, 97, 3), np.uint8)
        read_only = np.frombuffer(pixels.tobytes(), np.uint8).reshape(
            pixels.shape)
        bgra = rng.integers(0, 256, (131, 97, 4), np.uint8)
        frames = {
            'ffmpeg read-only': read_only,
            'writable': pixels.copy(),
            'mss strided view': bgra[:, :, 2::-1],
            'rgba fallback': bgra,
            'gray fallback': pixels[:, :, 0].copy(),
        }
        for name, frame in frames.items():
            for mode in RESIZE_MODES:
                with self.subTest(frame=name, mode=mode):
                    expected = v7_live._values(
                        Model(), Image.fromarray(frame), 'box', brightness=1.0,
                        gamma=1.0, perceptual_resize=mode,
                        perceptual_detail_strength=0.5)
                    actual = v7_live._values(
                        Model(), frame, 'box', brightness=1.0, gamma=1.0,
                        perceptual_resize=mode, perceptual_detail_strength=0.5)
                    np.testing.assert_array_equal(actual[0], expected[0])
                    self.assertEqual(actual[1], expected[1])

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
