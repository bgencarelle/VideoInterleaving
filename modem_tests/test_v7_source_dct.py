"""Tests for opt-in source-resolution V7 DCT preparation."""
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from animation_modem.v7_source_dct import (
    FoldBlockDCTProjector, _aggregate, _axis_frequency_gain,
    _frequency_weights, direct_dct_coefficients, direct_dct_values,
    source_dct_values,
    source_fold_block_dct_coefficients, source_fold_dct_coefficients)
from animation_modem import v7
from animation_modem.v7_fold import Fold500
from tools.v7_capture import CapturedFrame
from tools import v7_live
from tools.v7_source_dct_bench import (
    DISPLAY_SIZE, VARIANTS as DCT_BENCH_VARIANTS, _crop_viewport,
    _viewport_size)
from tools.v7_source_dct_bench import _select_variants
from modem_v7_display import (
    encode_folded_coefficients_packet, encode_folded_source_packet)


class SourceDCTTests(unittest.TestCase):
    def test_viewport_crop_excludes_letterbox_bars(self):
        aspect = .75
        width, height = _viewport_size(aspect)
        canvas = Image.new('RGB', DISPLAY_SIZE, '#777777')
        left = (DISPLAY_SIZE[0]-width)//2
        top = (DISPLAY_SIZE[1]-height)//2
        canvas.paste(Image.new('RGB', (width, height), (12, 34, 56)),
                     (left, top))

        viewport = _crop_viewport(canvas, aspect)

        self.assertEqual(viewport.size, (675, 900))
        self.assertEqual(viewport.getpixel((0, 0)), (12, 34, 56))
        self.assertEqual(viewport.getpixel((width-1, height-1)), (12, 34, 56))

    def test_matching_source_and_coder_grids_round_trip_float_luma(self):
        y, x = np.mgrid[:8, :10]
        rgb = np.stack((x/9, y/7, (x+y)/16), axis=-1)
        values, stats = source_dct_values(
            rgb, ((8, 10),), ((8, 10),), clip_values=False)

        expected_y = (.299*rgb[..., 0] + .587*rgb[..., 1] +
                      .114*rgb[..., 2])
        np.testing.assert_allclose(values.reshape(8, 10), 2*expected_y-1,
                                   rtol=0, atol=2e-14)
        self.assertEqual(stats['source_size'], [10, 8])
        self.assertEqual(stats['grid_clip_fraction'], 0.0)

    def test_gaussian_spectral_window_preserves_flat_input(self):
        rgb = np.full((120, 160, 3), (92, 132, 177), dtype=np.uint8)
        grids = ((12, 16), (6, 8), (6, 8))
        shapes = ((6, 8), (3, 4), (3, 4))
        direct, _ = source_dct_values(
            rgb, grids, shapes, clip_values=False)
        weighted, stats = source_dct_values(
            rgb, grids, shapes, aggregation='weighted-gaussian',
            clip_values=False)

        np.testing.assert_allclose(weighted, direct, atol=3e-14, rtol=0)
        self.assertEqual(stats['aggregation'], 'weighted-gaussian')

    def test_weighted_windows_multiply_matching_whole_frame_modes(self):
        coefficients = np.ones((20, 30), dtype=np.float64)
        target_shape = (4, 5)
        for aggregation in ('weighted-tent', 'weighted-cosine',
                            'weighted-gaussian'):
            with self.subTest(aggregation=aggregation):
                actual = _aggregate(coefficients, aggregation, target_shape)
                row_gain = _axis_frequency_gain(target_shape[0], aggregation)
                col_gain = _axis_frequency_gain(target_shape[1], aggregation)
                expected = row_gain[:, None]*col_gain[None, :]
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)
                self.assertEqual(float(actual[0, 0]), 1.0)

        tent = _axis_frequency_gain(4, 'weighted-tent')
        cosine = _axis_frequency_gain(4, 'weighted-cosine')
        gaussian = _axis_frequency_gain(4, 'weighted-gaussian')
        self.assertFalse(np.array_equal(tent, cosine))
        self.assertFalse(np.array_equal(cosine, gaussian))

    def test_weighted_reducers_do_not_fold_out_of_band_modes_into_the_target(self):
        source = np.zeros((20, 30), dtype=np.float64)
        source[2, 3] = 1.0
        source[15, 24] = 2.0
        for mode in ('weighted-tent', 'weighted-cosine', 'weighted-gaussian'):
            with self.subTest(mode=mode):
                reduced = _aggregate(source, mode, (4, 5))
                self.assertEqual(reduced.shape, (4, 5))
                expected = np.zeros((4, 5), dtype=np.float64)
                expected[2, 3] = (
                    _axis_frequency_gain(4, mode)[2]*
                    _axis_frequency_gain(5, mode)[3])
                np.testing.assert_allclose(
                    reduced, expected, rtol=0, atol=1e-15)

    def test_area_box_transform_matches_spatial_pixel_area_resampling(self):
        gray = np.arange(12*16, dtype=np.uint8).reshape(12, 16)*1
        rgb = np.repeat(gray[..., None], 3, axis=2)
        values, stats = source_dct_values(
            rgb, ((6, 8),), ((6, 8),), aggregation='area-box',
            clip_values=False)

        expected = 2*(gray.reshape(6, 2, 8, 2).mean(axis=(1, 3))/255.0)-1
        np.testing.assert_allclose(values.reshape(6, 8), expected,
                                   rtol=0, atol=3e-14)
        self.assertEqual(stats['aggregation'], 'area-box')
        self.assertGreater(float(values.std()), 0.1)

    def test_weighted_reducers_produce_noncollapsed_values(self):
        height, width = 120, 160
        y, x = np.mgrid[:height, :width]
        gray = (.5+.3*np.sin(2*np.pi*y/height)+
                .15*np.cos(2*np.pi*x/width))
        rgb = np.repeat(gray[..., None], 3, axis=2)
        grids, shapes = ((12, 16),), ((6, 8),)
        for mode in ('weighted-tent', 'weighted-cosine', 'weighted-gaussian'):
            values, stats = source_dct_values(
                rgb, grids, shapes, aggregation=mode, clip_values=False)
            self.assertGreater(float(values.std()), 0.2, msg=mode)
            self.assertEqual(stats['aggregation'], mode)

    def test_fold_projection_matches_exact_requested_source_dct_modes(self):
        height, width = 36, 48
        y, x = np.mgrid[:height, :width]
        rgb = np.stack((.2+.6*x/(width-1), .1+.7*y/(height-1),
                        .25+.4*(x+y)/(height+width-2)), axis=-1)
        grids = ((12, 16), (6, 8), (6, 8))
        offsets = np.cumsum([0] + [rows*cols for rows, cols in grids])
        positions = np.asarray([0, 7*16+11, offsets[1], offsets[1]+5*8+7,
                                offsets[2], offsets[2]+4*8+6], dtype=np.int64)
        actual = source_fold_dct_coefficients(rgb, grids, positions)

        rgb = rgb.astype(np.float64)
        red, green, blue = (rgb[..., index] for index in range(3))
        ycc = (
            .299*red+.587*green+.114*blue,
            128/255-.168736*red-.331264*green+.5*blue,
            128/255+.5*red-.418688*green-.081312*blue,
        )
        for plane, grid in enumerate(grids):
            expected = dctn(2*ycc[plane]-1, norm='ortho')
            expected *= np.sqrt(np.prod(grid)/(height*width))
            for position in positions[(positions >= offsets[plane]) &
                                      (positions < offsets[plane+1])]:
                local = int(position-offsets[plane])
                row, col = divmod(local, grid[1])
                self.assertAlmostEqual(
                    float(actual[position]), float(expected[row, col]), places=5)

    def test_block_projection_matches_native_modes_for_block_constant_rgb(self):
        rng = np.random.default_rng(12)
        reduced = rng.integers(0, 256, (12, 16, 3), dtype=np.uint8)
        rgb = np.repeat(np.repeat(reduced, 2, axis=0), 2, axis=1)
        grids = ((8, 8), (4, 4), (4, 4))
        positions = np.arange(sum(rows*cols for rows, cols in grids))

        native = source_fold_dct_coefficients(rgb, grids, positions)
        blocked = source_fold_block_dct_coefficients(
            rgb, grids, positions, block_size=2)
        projector = FoldBlockDCTProjector(
            rgb.shape[:2], grids, positions, block_size=2)
        planned = projector.project(rgb)

        np.testing.assert_allclose(blocked, native, rtol=0, atol=2e-5)
        np.testing.assert_array_equal(planned, blocked)

    def test_block_projection_rejects_nondivisible_source_dimensions(self):
        rgb = np.zeros((25, 32, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, 'divisible by block size'):
            source_fold_block_dct_coefficients(
                rgb, ((8, 8), (4, 4), (4, 4)), np.arange(96),
                block_size=2)

    def test_fold_source_projection_covers_the_actual_consumed_positions(self):
        model = v7.load_model(
            .1521/np.sqrt(1+10**(v7.CLOCK_REL_DB/10)), 'box')
        fold = Fold500(model)
        offsets = np.cumsum([0] + [r*c for r, c in v7.V7_GRIDS])
        extents = []
        for plane, (rows, cols) in enumerate(v7.V7_GRIDS):
            local = fold.source_positions[
                (fold.source_positions >= offsets[plane]) &
                (fold.source_positions < offsets[plane+1])]-offsets[plane]
            extents.append((int((local//cols).max())+1,
                            int((local%cols).max())+1))
        self.assertEqual(extents, [(60, 50), (24, 20), (24, 20)])

        rgb = np.full((120, 160, 3), (92, 132, 177), dtype=np.uint8)
        projected = source_fold_dct_coefficients(
            rgb, v7.V7_GRIDS, fold.source_positions)
        folded = fold.encode_dct_coefficients(projected)
        aspect = v7.aspect_wire_code((160, 120))
        direct = encode_folded_source_packet(
            model, rgb, 1, 0, aspect, fold=fold)
        replay = encode_folded_coefficients_packet(
            model, folded, 1, 0, aspect)
        np.testing.assert_array_equal(direct, replay)

    def test_default_bench_includes_corrected_weighted_reducers(self):
        names = {variant['name'] for variant in DCT_BENCH_VARIANTS}
        for name in ('weighted-tent', 'weighted-cosine', 'weighted-gaussian'):
            self.assertIn(name, names)
            self.assertEqual(_select_variants([name])[0]['name'], name)
        self.assertNotIn('area-box-resample', names)
        self.assertEqual(
            _select_variants(['area-box-resample'])[0]['name'],
            'area-box-resample')
        self.assertNotIn('fold-native-projection', names)
        self.assertEqual(
            _select_variants(['fold-native-projection'])[0]['name'],
            'fold-native-projection')
        for name in ('fold-block-4x', 'fold-block-5x',
                     'fold-block-6x', 'fold-block-8x',
                     'fold-block-10x',
                     'fold-block-12x'):
            self.assertNotIn(name, names)
            self.assertEqual(_select_variants([name])[0]['name'], name)

    def test_band_profiles_preserve_dc_and_separate_luma_from_chroma(self):
        y_shape, c_shape = (96, 80), (48, 40)
        common = ((1080, 900), 600.0, 96.0)
        mid_y = _frequency_weights(
            y_shape, 0, 'mid-luma', *common, viewport_aspect=.75)
        mid_cb = _frequency_weights(
            c_shape, 1, 'mid-luma', *common, viewport_aspect=.75)
        csf_y = _frequency_weights(
            y_shape, 0, 'perceptual-color', *common, viewport_aspect=.75)
        csf_cb = _frequency_weights(
            c_shape, 1, 'perceptual-color', *common, viewport_aspect=.75)

        self.assertEqual(mid_y[0, 0], 1.0)
        self.assertTrue(np.all(mid_cb == 1.0))
        self.assertEqual(csf_y[0, 0], 1.0)
        self.assertEqual(csf_cb[0, 0], 1.0)
        self.assertLessEqual(float(csf_cb.max()), 1.0)
        self.assertLessEqual(float(csf_y.max()), 1.2)

    def test_small_source_and_unknown_reducer_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'smaller than'):
            source_dct_values(np.zeros((7, 10, 3), np.uint8),
                              ((8, 10),), ((8, 10),))
        with self.assertRaisesRegex(ValueError, 'aggregation'):
            source_dct_values(np.zeros((12, 16, 3), np.uint8),
                              ((8, 8),), ((4, 4),), aggregation='box')

    def test_live_dct_path_preserves_native_geometry_and_rejects_prepared(self):
        model = v7.load_model(.1521/np.sqrt(1+10**(v7.CLOCK_REL_DB/10)), 'box')
        rgb = np.full((120, 160, 3), 128, dtype=np.uint8)
        values, aspect = v7_live._values(
            model, rgb, 'box', brightness=1.0, dct_encode=True)

        self.assertEqual(values.shape, (model.coder.source_count,))
        self.assertEqual(aspect, v7.aspect_wire_code((160, 120)))
        with self.assertRaisesRegex(ValueError, 'unprepared source'):
            v7_live._values(
                model, CapturedFrame(rgb, (160, 120), prepared=True), 'box',
                brightness=1.0, dct_encode=True)

    def test_explicit_noop_dct_option_still_requires_opt_in(self):
        args = v7_live.parser().parse_args([
            'send', '--device', 'null', '--source', 'test',
            '--dct-sharpen-strength', '0.25'])

        self.assertTrue(v7_live._dct_options_explicit(args))
        with self.assertRaisesRegex(ValueError, 'require --dct-encode'):
            v7_live._send_profile(args, 0)

    def test_live_sender_accepts_corrected_weighting(self):
        args = v7_live.parser().parse_args([
            'send', '--device', 'null', '--source', 'test', '--dct-encode',
            '--dct-aggregation', 'weighted-tent'])
        self.assertEqual(v7_live._dct_encode_options(args)['aggregation'],
                         'weighted-tent')

    def test_live_sender_accepts_explicit_area_box_transform(self):
        model = v7.load_model(
            .1521/np.sqrt(1+10**(v7.CLOCK_REL_DB/10)), 'box')
        y, x = np.mgrid[:120, :160]
        rgb = np.stack((x/159, y/119, .5*x/159+.5*y/119), axis=-1)
        values, aspect = v7_live._values(
            model, rgb, 'box', brightness=1.0, gamma=1.0, dct_encode=True,
            dct_options={'aggregation': 'area-box'})

        self.assertEqual(values.shape, (model.coder.source_count,))
        self.assertTrue(np.isfinite(values).all())
        self.assertGreater(float(values.std()), 0.1)
        self.assertEqual(aspect, v7.aspect_wire_code((160, 120)))


def _direct_reference(rgb, grids, shapes, brightness=1.0, gamma=1.0,
                      sharpen='off', strength=.25, clarity=0.0,
                      chroma_gain=1.0):
    """The direct-encode spec written out with dctn/idctn, for comparison."""
    from scipy.fft import idctn
    from scipy.ndimage import gaussian_filter
    x = np.clip(np.asarray(rgb, float)/255.0*brightness, 0, 1)**(1/gamma)
    height, width = x.shape[:2]
    luma_rows, luma_cols = grids[0]
    block_y = max(1, height//(2*luma_rows))
    block_x = max(1, width//(2*luma_cols))
    rows, cols = height//block_y, width//block_x
    x = x[:rows*block_y, :cols*block_x].reshape(
        rows, block_y, cols, block_x, 3).mean(axis=(1, 3))
    red, green, blue = x[..., 0], x[..., 1], x[..., 2]
    y = .299*red+.587*green+.114*blue
    cb = np.clip(.5+chroma_gain*(-.168736*red-.331264*green+.5*blue), 0, 1)
    cr = np.clip(.5+chroma_gain*(.5*red-.418688*green-.081312*blue), 0, 1)
    unit = (rows/luma_rows, cols/luma_cols)
    if clarity:
        y = y+clarity*(y-gaussian_filter(
            y, (6*unit[0], 6*unit[1]), mode='reflect'))
    if sharpen == 'usm':
        y = y+strength*(y-gaussian_filter(
            y, (.8*unit[0], .8*unit[1]), mode='reflect'))
    out = []
    for index, (plane, (grid_rows, grid_cols), (sent_rows, sent_cols)) in \
            enumerate(zip((y, cb, cr), grids, shapes)):
        coeff = dctn(plane, norm='ortho')[:grid_rows, :grid_cols]
        coeff *= np.sqrt(grid_rows*grid_cols/(rows*cols))
        if index == 0 and sharpen == 'taper':
            radius = np.hypot(np.arange(sent_rows)[:, None]/sent_rows,
                              np.arange(sent_cols)[None, :]/sent_cols)
            coeff[:sent_rows, :sent_cols] *= 1+strength*np.where(
                radius < 1, 27/4*radius**2*(1-radius), 0)
        out.append(idctn(coeff, norm='ortho').ravel())
    return np.clip(np.concatenate(out)*2-1, -1, 1)


def _textured_frame(width, height, seed=0):
    with Image.open(v7.REFERENCE_FIXTURE) as image:
        base = np.asarray(image.convert('RGB').resize(
            (width, height), Image.Resampling.BILINEAR), float)
    noise = np.random.default_rng(seed).normal(0, 6, base.shape)
    return np.clip(base+noise, 0, 255).astype(np.uint8)


class DirectDCTEncodeTests(unittest.TestCase):
    grids, shapes = v7.V7_GRIDS, v7.V7_SHAPES

    def test_matches_the_spec_pipeline_for_every_option(self):
        cases = ({}, {'brightness': 1.05, 'gamma': 1.2},
                 {'sharpen': 'taper', 'strength': .5},
                 {'sharpen': 'usm', 'strength': .5},
                 {'clarity': .3, 'chroma_gain': 1.2},
                 {'sharpen': 'usm', 'strength': .5, 'clarity': .3})
        for width, height in ((160, 96), (400, 480), (641, 333)):
            rgb = _textured_frame(width, height)
            for case in cases:
                with self.subTest(size=(width, height), **case):
                    actual = direct_dct_values(
                        rgb, self.grids, self.shapes,
                        brightness=case.get('brightness', 1.0),
                        gamma=case.get('gamma', 1.0),
                        sharpen=case.get('sharpen', 'off'),
                        sharpen_strength=case.get('strength', .25),
                        clarity=case.get('clarity', 0.0),
                        chroma_gain=case.get('chroma_gain', 1.0))
                    expected = _direct_reference(
                        rgb, self.grids, self.shapes, **case)
                    np.testing.assert_allclose(actual, expected, atol=1e-9)

    def test_flat_colour_encodes_to_exact_constants_with_any_option(self):
        rgb = np.full((300, 410, 3), (92, 132, 177), dtype=np.uint8)
        red, green, blue = np.array((92, 132, 177))/255.0
        y = .299*red+.587*green+.114*blue
        for options in ({}, {'sharpen': 'taper', 'sharpen_strength': 1.0},
                        {'sharpen': 'usm', 'sharpen_strength': 1.0},
                        {'clarity': 1.0}):
            values = direct_dct_values(rgb, self.grids, self.shapes, **options)
            luma = values[:96*80]
            np.testing.assert_allclose(luma, 2*y-1, atol=1e-12)
        grey = np.full((300, 410, 3), 128, dtype=np.uint8)
        values = direct_dct_values(grey, self.grids, self.shapes,
                                   chroma_gain=1.3)
        np.testing.assert_allclose(values[96*80:], 0.0, atol=1e-12)

    def test_source_at_grid_size_passes_through(self):
        y, x = np.mgrid[:96, :80]
        rgb = np.stack((x/79, y/95, .5*(x/79+y/95)), axis=-1)
        values = direct_dct_values(rgb, ((96, 80),), ((48, 40),))
        luma = .299*rgb[..., 0]+.587*rgb[..., 1]+.114*rgb[..., 2]
        np.testing.assert_allclose(values, (2*luma-1).ravel(), atol=1e-12)

    def test_taper_changes_only_the_sent_luma_corner(self):
        rgb = _textured_frame(400, 480, seed=2)
        plain = direct_dct_coefficients(rgb, self.grids, self.shapes)
        taper = direct_dct_coefficients(rgb, self.grids, self.shapes,
                                        sharpen='taper', sharpen_strength=.5)
        outside = np.ones((96, 80), bool)
        outside[:48, :40] = False
        np.testing.assert_array_equal(taper[0][outside], plain[0][outside])
        self.assertFalse(np.allclose(taper[0][:48, :40], plain[0][:48, :40]))
        self.assertEqual(taper[0][0, 0], plain[0][0, 0])
        for plane in (1, 2):
            np.testing.assert_array_equal(taper[plane], plain[plane])

    def test_coefficients_are_the_coders_transform_and_feed_the_fold(self):
        model = v7.load_model(.1521/np.sqrt(1+10**(v7.CLOCK_REL_DB/10)), 'box')
        rgb = _textured_frame(720, 960, seed=4)
        values = direct_dct_values(rgb, self.grids, self.shapes)
        self.assertLess(float(np.abs(values).max()), 1.0)  # nothing clipped
        coefficients = direct_dct_coefficients(rgb, self.grids, self.shapes)
        offset = 0
        for plane, (rows, cols) in zip(coefficients, self.grids):
            grid = values[offset:offset+rows*cols].reshape(rows, cols)
            np.testing.assert_allclose(plane, dctn(grid, norm='ortho'),
                                       atol=1e-10)
            offset += rows*cols
        fold = Fold500(model)
        np.testing.assert_allclose(
            fold.encode_dct_coefficients(np.concatenate(
                [plane.ravel() for plane in coefficients])),
            fold.encode_coefficients(values), atol=1e-9)

    def test_uint8_and_float_frames_agree(self):
        rgb = _textured_frame(400, 480, seed=6)
        np.testing.assert_allclose(
            direct_dct_values(rgb, self.grids, self.shapes, gamma=1.3),
            direct_dct_values(rgb/255.0, self.grids, self.shapes, gamma=1.3),
            atol=1e-12)

    def test_live_sender_uses_the_direct_encoder_by_default(self):
        model = v7.load_model(.1521/np.sqrt(1+10**(v7.CLOCK_REL_DB/10)), 'box')
        rgb = _textured_frame(400, 480, seed=8)
        values, _aspect = v7_live._values(
            model, rgb, 'box', brightness=1.05, dct_encode=True,
            dct_options={'sharpen': 'taper', 'sharpen_strength': .25,
                         'clarity': 0.0, 'chroma_gain': 1.0,
                         'aggregation': 'off', 'band_profile': 'off'})
        np.testing.assert_array_equal(values, direct_dct_values(
            rgb, model.coder.grids, model.coder.shapes, brightness=1.05,
            sharpen='taper', sharpen_strength=.25))

    def test_small_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'smaller than'):
            direct_dct_values(np.zeros((95, 200, 3), np.uint8),
                              self.grids, self.shapes)


if __name__ == '__main__':
    unittest.main()
