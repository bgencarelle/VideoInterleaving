"""Tests for the pluggable DCT downscale kernels (dct_kernels/)."""
import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from animation_modem import v7
from animation_modem import v7_kernels as K
from animation_modem.v7_source_dct import (
    KernelFrame, _dct_matrix, _direct_plan, _direct_planes,
    _linear_to_srgb, _separable, _srgb_to_linear_extended,
    direct_dct_values)
from tools import v7_live

GRIDS, SHAPES = v7.V7_GRIDS, v7.V7_SHAPES


def _frame(seed=3, rows=480, cols=400):
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 255, (rows//8, cols//8, 3), dtype=np.uint8)
    return np.kron(frame, np.ones((8, 8, 1), np.uint8))


def _write(folder, name, body):
    path = Path(folder)/f'{name}.py'
    path.write_text(textwrap.dedent(body))
    return path


TRIANGLE = '''
    import numpy as np
    SUPPORT = 1.0
    def kernel(x):
        return np.clip(1.0 - np.abs(x), 0.0, None)
'''


class EnlargementKernelTests(unittest.TestCase):
    """The kernels written for a picture that is enlarged afterwards."""

    NAMES = ('upscale_precomp', 'shock_edge', 'csf_peak', 'slepian', 'tv_cartoon')

    def setUp(self):
        self.registry = K.open_registry()
        self.frame = _frame()
        self.masks = []
        for index, grid in enumerate(GRIDS):
            mask = np.zeros(grid, bool)
            mask[:SHAPES[index][0], :SHAPES[index][1]] = True
            self.masks.append(mask)

    def encode(self, name, **params):
        frame = []
        values = direct_dct_values(
            self.frame, GRIDS, SHAPES, kernel=self.registry.select(name, params),
            kernel_masks=self.masks, kernel_frame_out=frame)
        return frame[0].post(values, GRIDS) if frame else values

    def test_each_one_keeps_flat_pictures_flat_and_stays_finite(self):
        flat = np.full((480, 400, 3), 120, np.uint8)
        reference = direct_dct_values(flat, GRIDS, SHAPES)
        for name in self.NAMES:
            with self.subTest(kernel=name):
                frame = []
                values = direct_dct_values(
                    flat, GRIDS, SHAPES, kernel=self.registry.select(name, {}),
                    kernel_masks=self.masks, kernel_frame_out=frame)
                if frame:
                    values = frame[0].post(values, GRIDS)
                self.assertTrue(np.isfinite(values).all())
                np.testing.assert_allclose(values, reference, atol=2e-3)

    def test_each_one_changes_the_picture(self):
        reference = direct_dct_values(self.frame, GRIDS, SHAPES)
        for name in self.NAMES:
            with self.subTest(kernel=name):
                self.assertGreater(
                    np.abs(self.encode(name, **(
                        {'amount': 0.6} if name == 'csf_peak' else {}))
                           - reference).max(), 1e-3)

    def test_precompensation_lifts_what_the_upscaler_would_soften(self):
        kernel = self.registry.get('upscale_precomp')
        from animation_modem.v7_kernels import KernelContext
        ctx = KernelContext(0, GRIDS[0], SHAPES[0], self.masks[0], None)
        gains = {}
        for upscaler in (0, 1, 2, 3):
            gain = kernel.gain(ctx, kernel.resolve({'upscaler': upscaler}))
            gains[upscaler] = gain[0, SHAPES[0][1]*3//4]        # nu = 0.375
        self.assertGreater(gains[0], gains[1])          # bilinear needs most
        self.assertGreater(gains[1], 1.0)
        self.assertAlmostEqual(gains[3], 1.0, places=6)  # ideal: nothing to undo
        self.assertLessEqual(max(gains.values()), 6.0)

    def test_the_shock_filter_does_not_leave_the_source_range(self):
        step = np.zeros((480, 400, 3), np.uint8)
        step[:, 200:] = 255
        frame = []
        values = direct_dct_values(
            step, GRIDS, SHAPES, kernel=self.registry.select('shock_edge', {}),
            kernel_masks=self.masks, kernel_frame_out=frame)
        values = frame[0].post(values, GRIDS)
        luma = values[:GRIDS[0][0]*GRIDS[0][1]].reshape(GRIDS[0])
        self.assertTrue(np.isfinite(luma).all())
        self.assertLess(np.abs(luma).max(), 1.15)

    def test_tv_cartoon_flattens_texture(self):
        rng = np.random.default_rng(5)
        noisy = np.clip(128 + rng.normal(0, 20, (480, 400, 3)), 0, 255).astype(np.uint8)
        frame = []
        values = direct_dct_values(
            noisy, GRIDS, SHAPES,
            kernel=self.registry.select('tv_cartoon', {'keep': 0.0, 'weight': 0.08}),
            kernel_masks=self.masks, kernel_frame_out=frame)
        values = frame[0].post(values, GRIDS)
        plain = []
        untouched = direct_dct_values(
            noisy, GRIDS, SHAPES,
            kernel=self.registry.select('tv_cartoon', {'weight': 0.0}),
            kernel_masks=self.masks, kernel_frame_out=plain)
        untouched = plain[0].post(untouched, GRIDS)

        def variation(vector):
            luma = vector[:GRIDS[0][0]*GRIDS[0][1]].reshape(GRIDS[0])
            return np.abs(np.diff(luma, axis=0)).sum() + np.abs(np.diff(luma, axis=1)).sum()
        self.assertLess(variation(values), 0.8*variation(untouched))


class ViewerAndPrefilterTests(unittest.TestCase):
    def setUp(self):
        self.registry = K.open_registry()

    def test_the_shipped_pre_shrink_factor_is_the_default(self):
        frame = _frame(rows=1080, cols=1920)
        plain = direct_dct_values(frame, GRIDS, SHAPES)
        four = direct_dct_values(
            frame, GRIDS, SHAPES,
            preshrink=4.0)
        np.testing.assert_array_equal(plain, four)

    def test_a_finer_pre_shrink_changes_a_detailed_picture(self):
        rng = np.random.default_rng(4)
        stripes = np.zeros((1080, 1920, 3), np.uint8)
        stripes[:, ::2] = 200
        stripes += rng.integers(0, 40, stripes.shape, dtype=np.uint8)
        plain = direct_dct_values(stripes, GRIDS, SHAPES)
        fine = direct_dct_values(stripes, GRIDS, SHAPES, preshrink=8.0)
        self.assertGreater(np.abs(plain - fine).max(), 1e-3)
        self.assertTrue(np.isfinite(fine).all())

    def test_a_kernel_prefilter_overrides_the_setting(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/'coarse.py').write_text(textwrap.dedent('''
                def prefilter():
                    return {'preshrink': 2.0}
            '''))
            registry = K.KernelRegistry([folder])
            registry.scan()
            self.assertEqual(registry.errors, [])
            frame = _frame(rows=1080, cols=1920)
            by_kernel = direct_dct_values(frame, GRIDS, SHAPES,
                                          kernel=registry.select('coarse', {}),
                                          preshrink=8.0)
            by_setting = direct_dct_values(frame, GRIDS, SHAPES, preshrink=2.0)
            np.testing.assert_array_equal(by_kernel, by_setting)

    def test_a_prefilter_outside_the_limits_is_refused_and_bypassed(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/'wild.py').write_text(textwrap.dedent('''
                def prefilter():
                    return {'preshrink': 50}
            '''))
            registry = K.KernelRegistry([folder])
            registry.scan()
            # refused at load: the self-test runs the hook
            self.assertTrue(registry.errors)
            self.assertIn('preshrink', registry.errors[0][1])

    def test_the_viewer_solve_gets_closer_to_the_ideal_enlargement(self):
        module = self.registry.get('viewer_solve').module
        grid, sent, up = (96, 80), (48, 40), 4
        mask = np.zeros(grid, bool)
        mask[:sent[0], :sent[1]] = True
        rng = np.random.default_rng(9)
        plane = np.clip(np.cumsum(np.cumsum(rng.normal(0, .3, grid), 0), 1)/40
                        + .5, 0, 1)
        ctx = K.KernelContext(0, grid, sent, mask, None)
        from scipy.fft import dctn, idctn
        solved = module.post(plane, ctx, viewer=0, iterations=14, lam=0.02,
                             up=up, tame=0)
        my, mx = module._model(0, *grid, *sent, up)
        iy, ix = module._ideal(*grid, *sent, up)
        base = dctn(plane, norm='ortho')*mask
        ideal = iy @ base @ ix.T
        seen = lambda x: my @ (dctn(x, norm='ortho')*mask) @ mx.T
        plain_error = np.linalg.norm(seen(plane) - ideal)
        solved_error = np.linalg.norm(seen(solved) - ideal)
        self.assertLess(solved_error, 0.6*plain_error)
        self.assertTrue(np.isfinite(solved).all())

    def test_the_viewer_solve_leaves_flat_and_chroma_planes_alone(self):
        module = self.registry.get('viewer_solve').module
        grid, sent = (96, 80), (48, 40)
        flat = np.full(grid, .4)
        ctx = K.KernelContext(0, grid, sent, None, None)
        out = module.post(flat, ctx, viewer=1, iterations=10, lam=.02, up=4, tame=0)
        np.testing.assert_allclose(out, flat, atol=1e-6)
        chroma = K.KernelContext(1, (48, 40), (24, 20), None, None)
        plane = np.random.default_rng(1).random((48, 40))
        np.testing.assert_array_equal(
            module.post(plane, chroma, viewer=1, iterations=10, lam=.02,
                        up=4, tame=0), plane)


class RegistryTests(unittest.TestCase):
    def test_shipped_kernels_all_load(self):
        registry = K.open_registry()
        self.assertEqual(registry.errors, [])
        for name in ('lanczos', 'mitchell', 'magic_kernel_sharp', 'gaussian',
                     'band_taper', 'antiring', 'upscale_precomp', 'shock_edge',
                     'csf_peak', 'slepian', 'tv_cartoon', 'viewer_solve'):
            self.assertIn(name, registry.names())
        self.assertEqual(registry.names()[0], K.REFERENCE)

    def test_files_are_added_and_removed_by_rescanning_the_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            registry = K.KernelRegistry([folder])
            self.assertEqual(registry.scan(), ([], []))
            path = _write(folder, 'triangle', TRIANGLE)
            self.assertEqual(registry.scan(), (['triangle'], []))
            kept = registry.get('triangle')
            registry.scan()
            self.assertIs(registry.get('triangle'), kept)    # unchanged: reused
            path.unlink()
            self.assertEqual(registry.scan(), ([], ['triangle']))
            with self.assertRaises(K.KernelError):
                registry.get('triangle')

    def test_an_edited_file_is_reloaded(self):
        with tempfile.TemporaryDirectory() as folder:
            path = _write(folder, 'edit', TRIANGLE)
            registry = K.KernelRegistry([folder])
            registry.scan()
            first = registry.get('edit')
            _write(folder, 'edit', TRIANGLE.replace('SUPPORT = 1.0',
                                                    'SUPPORT = 2.0'))
            registry.scan()
            self.assertIsNot(registry.get('edit'), first)
            self.assertEqual(registry.get('edit').support, 2.0)

    def test_a_broken_file_is_reported_and_the_rest_still_load(self):
        with tempfile.TemporaryDirectory() as folder:
            _write(folder, 'good', TRIANGLE)
            _write(folder, 'syntax', 'def kernel(x:\n')
            _write(folder, 'nohooks', 'X = 1\n')
            _write(folder, 'explodes', '''
                def gain(ctx):
                    raise RuntimeError('boom')
            ''')
            _write(folder, 'badshape', '''
                import numpy as np
                def gain(ctx):
                    return np.ones((3, 3))
            ''')
            _write(folder, 'badparam', '''
                PARAMS = {'x': (5, 0, 1, 0.1, 'default outside range')}
                def response(nu, x): return nu*0+1
            ''')
            _write(folder, 'reserved', '''
                PARAMS = {'luma_mix': (1, 0, 1, 0.1, 'taken')}
                def response(nu, luma_mix): return nu*0+1
            ''')
            _write(folder, '_ignored', TRIANGLE)
            registry = K.KernelRegistry([folder])
            registry.scan()
            self.assertEqual(registry.names(), [K.REFERENCE, 'good'])
            self.assertEqual({Path(path).stem for path, _ in registry.errors},
                             {'syntax', 'nohooks', 'explodes', 'badshape',
                              'badparam', 'reserved'})

    def test_later_folders_override_earlier_ones(self):
        with tempfile.TemporaryDirectory() as first, \
                tempfile.TemporaryDirectory() as second:
            _write(first, 'same', TRIANGLE)
            _write(second, 'same', TRIANGLE.replace('SUPPORT = 1.0',
                                                    'SUPPORT = 2.0'))
            registry = K.KernelRegistry([first, second])
            registry.scan()
            self.assertEqual(registry.get('same').support, 2.0)

    def test_the_environment_folder_is_searched(self):
        import os
        with tempfile.TemporaryDirectory() as folder:
            _write(folder, 'fromenv', TRIANGLE)
            old = os.environ.get(K.ENV_DIRS)
            os.environ[K.ENV_DIRS] = folder
            try:
                self.assertIn('fromenv', K.open_registry().names())
            finally:
                if old is None:
                    del os.environ[K.ENV_DIRS]
                else:
                    os.environ[K.ENV_DIRS] = old

    def test_parameters_are_clamped_and_unknown_ones_refused(self):
        registry = K.open_registry()
        selection = registry.select('lanczos', {'width': 99, 'lobes': 3.4})
        self.assertEqual(selection.params['width'], 1.6)
        self.assertEqual(selection.params['lobes'], 3.0)
        with self.assertRaises(K.KernelError):
            registry.select('lanczos', {'nope': 1})
        with self.assertRaises(K.KernelError):
            registry.select('lanczos', {'width': float('nan')})
        self.assertIsNone(registry.select(K.REFERENCE))

    def test_kernel_host_defaults_are_applied_and_explicit_values_win(self):
        with tempfile.TemporaryDirectory() as folder:
            _write(folder, 'host_defaults', '''
                import numpy as np
                PARAMS = {'strength': (0.2, 0.0, 1.0, 0.1, 'strength')}
                HOST_DEFAULTS = {'luma_mix': 0.25, 'chroma_mix': 0.5}
                PROFILE_DEFAULTS = {
                    'aspect-mono-500': {'strength': 0.8, 'luma_mix': 0.4}
                }
                def response(nu): return np.ones_like(nu)
            ''')
            registry = K.KernelRegistry([folder])
            registry.scan()
            kernel = registry.get('host_defaults')
            self.assertEqual(kernel.defaults()['luma_mix'], 0.25)
            self.assertEqual(kernel.defaults()['chroma_mix'], 0.5)
            self.assertEqual(kernel.defaults('aspect-mono-500')['strength'], 0.8)
            self.assertEqual(kernel.defaults('aspect-mono-500')['luma_mix'], 0.4)
            selection = registry.select('host_defaults', {'luma_mix': 1.0})
            self.assertEqual(selection.params['luma_mix'], 1.0)
            self.assertEqual(selection.params['chroma_mix'], 0.5)
            selection = registry.select(
                'host_defaults', {'strength': 0.6}, profile='aspect-mono-500')
            self.assertEqual(selection.params['strength'], 0.6)
            self.assertEqual(selection.params['luma_mix'], 0.4)


class GainTests(unittest.TestCase):
    def setUp(self):
        self.registry = K.open_registry()
        self.luma = K.KernelContext(0, GRIDS[0], SHAPES[0])
        self.chroma = K.KernelContext(1, GRIDS[1], SHAPES[1])

    def test_dc_gain_is_one_for_every_gain_kernel(self):
        for name in self.registry.names()[1:]:
            kernel = self.registry.get(name)
            if kernel.has_gain:
                gain = kernel.gain(self.luma, kernel.defaults())
                self.assertEqual(gain[0, 0], 1.0, name)
                self.assertTrue(np.isfinite(gain).all(), name)

    def test_spatial_kernel_response_matches_the_analytic_one(self):
        # a one-pixel box has the response sinc(nu)
        with tempfile.TemporaryDirectory() as folder:
            _write(folder, 'box', '''
                import numpy as np
                SUPPORT = 0.5
                def kernel(x):
                    return (np.abs(x) < 0.5).astype(float)
            ''')
            registry = K.KernelRegistry([folder])
            registry.scan()
            kernel = registry.get('box')
            gain = kernel.gain(self.luma, kernel.defaults())
            nu_rows, nu_cols = self.luma.nu()
            expected = np.sinc(nu_rows)*np.sinc(nu_cols)
            np.testing.assert_allclose(gain, expected, atol=2e-3)

    def test_mix_zero_switches_a_plane_off_and_one_is_as_written(self):
        kernel = self.registry.get('gaussian')
        params = kernel.defaults()
        full = kernel.gain(self.luma, params)
        half = kernel.gain(self.luma, dict(params, luma_mix=0.5))
        np.testing.assert_allclose(half, (1+full)/2)
        self.assertIsNone(kernel.gain(self.luma, dict(params, luma_mix=0.0)))
        self.assertIsNotNone(kernel.gain(self.chroma, dict(params, luma_mix=0.0)))

    def test_gain_arrays_are_cached_and_read_only(self):
        kernel = self.registry.get('lanczos')
        first = kernel.gain(self.luma, kernel.defaults())
        self.assertIs(first, kernel.gain(self.luma, kernel.defaults()))
        self.assertIsNot(first, kernel.gain(
            self.luma, dict(kernel.defaults(), width=1.2)))
        with self.assertRaises(ValueError):
            first[1, 1] = 0

    def test_csf_diamond_windows_chroma_by_default_with_opt_out(self):
        kernel = self.registry.get('csf_diamond')
        params = kernel.defaults()
        self.assertAlmostEqual(params['luma_mix'], 0.25)
        self.assertEqual(params['color_planes'], 1)
        self.assertIsNotNone(kernel.gain(self.luma, params))
        self.assertIsNotNone(kernel.gain(self.chroma, params))
        self.assertIsNone(kernel.gain(
            self.chroma, dict(params, color_planes=0)))

    def test_csf_luma_target_matches_the_matrix_reference(self):
        selection = self.registry.select('csf_diamond')
        planes = [np.zeros(grid) for grid in GRIDS]
        frame = KernelFrame(selection, GRIDS, SHAPES, None, planes)
        target = np.random.default_rng(8).random(GRIDS[0])
        gain = frame.gain(0)
        basis_y = _dct_matrix(GRIDS[0][0], GRIDS[0][0])
        basis_x = _dct_matrix(GRIDS[0][1], GRIDS[0][1])
        coded = _linear_to_srgb(target)
        coefficients = basis_y.T @ coded @ basis_x
        expected = _srgb_to_linear_extended(
            basis_y @ (coefficients*gain) @ basis_x.T)
        np.testing.assert_allclose(frame.filter_target(target), expected,
                                   rtol=0.0, atol=3e-13)

    def test_a_kernel_may_depend_on_the_plane_it_is_given(self):
        with tempfile.TemporaryDirectory() as folder:
            _write(folder, 'planewise', '''
                import numpy as np
                def response(nu, plane):
                    return np.exp(-nu*(8.0 if plane else 0.0))
            ''')
            registry = K.KernelRegistry([folder])
            registry.scan()
            kernel = registry.get('planewise')
            self.assertTrue(np.all(kernel.gain(self.luma, kernel.defaults()) == 1))
            self.assertLess(kernel.gain(self.chroma, kernel.defaults())[12, 10], 0.9)


class EncoderTests(unittest.TestCase):
    def setUp(self):
        self.registry = K.open_registry()
        self.frame = _frame()

    def test_no_kernel_and_a_switched_off_kernel_are_the_shipped_bytes(self):
        reference = direct_dct_values(self.frame, GRIDS, SHAPES)
        np.testing.assert_array_equal(
            reference, direct_dct_values(self.frame, GRIDS, SHAPES, kernel=None))
        for name in ('lanczos', 'band_taper', 'antiring'):
            off = self.registry.select(name, {'luma_mix': 0, 'chroma_mix': 0})
            np.testing.assert_array_equal(reference, direct_dct_values(
                self.frame, GRIDS, SHAPES, kernel=off), name)

    def test_taper_and_a_kernel_window_multiply(self):
        selection = self.registry.select('gaussian')
        both = direct_dct_values(self.frame, GRIDS, SHAPES, sharpen='taper',
                                 kernel=selection)
        taper = direct_dct_values(self.frame, GRIDS, SHAPES, sharpen='taper')
        window = direct_dct_values(self.frame, GRIDS, SHAPES, kernel=selection)
        self.assertGreater(np.abs(both-taper).max(), 1e-3)
        self.assertGreater(np.abs(both-window).max(), 1e-3)

    def test_csf_diamond_shapes_chroma_by_default_but_can_skip_it(self):
        reference = direct_dct_values(self.frame, GRIDS, SHAPES)
        selection = self.registry.select('csf_diamond')
        shaped = direct_dct_values(self.frame, GRIDS, SHAPES, kernel=selection)
        luma_count = GRIDS[0][0]*GRIDS[0][1]
        self.assertGreater(np.max(np.abs(shaped[luma_count:] -
                                         reference[luma_count:])), 1e-3)
        luma_only = self.registry.select('csf_diamond', {'color_planes': 0})
        unshaped = direct_dct_values(self.frame, GRIDS, SHAPES,
                                     kernel=luma_only)
        np.testing.assert_array_equal(unshaped[luma_count:],
                                      reference[luma_count:])

    def test_live_deferred_csf_returns_reference_values_and_gain_sidecar(self):
        selection = self.registry.select('csf_diamond')
        deferred = []
        model = v7.load_model(.1521, 'box')
        values, _aspect, preview = v7_live._values(
            model, self.frame, 'box', 1.0, 1.0, dct_encode=True,
            dct_options={'kernel': selection}, return_resized=True,
            defer_kernel_gain=True,
            deferred_gains_out=deferred)
        np.testing.assert_array_equal(
            values, direct_dct_values(self.frame, GRIDS, SHAPES))
        self.assertEqual(preview.mode, 'RGB')
        self.assertEqual(len(deferred), 1)
        self.assertIsNotNone(deferred[0][0])
        self.assertIsNotNone(deferred[0][1])
        self.assertIsNotNone(deferred[0][2])

    def test_deferred_gain_matches_exact_path_when_clamping_is_inactive(self):
        from scipy.fft import dctn, idctn

        yy, xx = np.indices((480, 400))
        wave = 127.0 + 25.0*np.sin(2*np.pi*xx/31)*np.cos(2*np.pi*yy/37)
        rgb = np.stack((wave, wave+4.0, wave-3.0), axis=-1).clip(0, 255).astype(
            np.uint8)
        selection = self.registry.select('csf_diamond')
        exact = direct_dct_values(rgb, GRIDS, SHAPES, kernel=selection)
        frames = []
        values = direct_dct_values(
            rgb, GRIDS, SHAPES, kernel=selection, kernel_frame_out=frames,
            defer_kernel_gain=True)
        self.assertLess(float(np.max(np.abs(exact))), .9)
        gained, offset = [], 0
        for gain, (rows, cols) in zip(frames[0].coefficient_gains(), GRIDS):
            count = rows*cols
            plane = values[offset:offset+count].reshape(rows, cols)
            coefficients = dctn(plane, norm='ortho')
            if gain is not None:
                coefficients *= gain
            filtered = idctn(coefficients, norm='ortho')
            gained.append(filtered.ravel())
            offset += count
        np.testing.assert_allclose(np.concatenate(gained), exact,
                                   rtol=0.0, atol=3e-13)

    def test_fft_window_path_matches_the_source_dct_reference(self):
        selection = self.registry.select('csf_diamond')
        planes, grids, shapes, _taper, _target, blocks, source = _direct_planes(
            self.frame, GRIDS, SHAPES, 1.0, 1.0, 'off', .25, 0.0, 1.0)
        kernel_frame = KernelFrame(selection, grids, shapes, None, planes)
        expected = []
        for index, (plane, (rows, cols), sent) in enumerate(
                zip(planes, grids, shapes)):
            (analysis_y, analysis_x, synthesis_y, synthesis_x,
             left, right) = _direct_plan(
                 *source, rows, cols, *blocks[index])
            gain = kernel_frame.gain(index)
            if gain is None:
                grid = _separable(left, plane, right)
            else:
                coefficients = _separable(analysis_y, plane, analysis_x)
                coefficients *= gain
                grid = _separable(synthesis_y, coefficients, synthesis_x)
            expected.append(grid.ravel())
        expected = np.concatenate(expected)*2.0-1.0
        np.clip(expected, -1.0, 1.0, out=expected)
        actual = direct_dct_values(self.frame, GRIDS, SHAPES,
                                   kernel=selection)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=3e-13)

    def test_flat_pictures_stay_flat_with_any_kernel(self):
        flat = np.full((240, 200, 3), 97, np.uint8)
        reference = direct_dct_values(flat, GRIDS, SHAPES)
        for name in self.registry.names()[1:]:
            frames = []
            values = direct_dct_values(
                flat, GRIDS, SHAPES, kernel=self.registry.select(name),
                kernel_frame_out=frames)
            if frames:
                values = frames[0].post(values, GRIDS)
            np.testing.assert_allclose(values, reference, atol=2e-3, err_msg=name)

    def test_a_softer_window_removes_high_frequency_energy(self):
        luma = GRIDS[0][0]*GRIDS[0][1]
        reference = direct_dct_values(self.frame, GRIDS, SHAPES)[:luma]
        soft = direct_dct_values(self.frame, GRIDS, SHAPES,
                                 kernel=self.registry.select('gaussian'))[:luma]
        self.assertLess(np.abs(np.diff(soft.reshape(GRIDS[0]), axis=1)).mean(),
                        np.abs(np.diff(reference.reshape(GRIDS[0]), axis=1)).mean())

    def test_the_window_is_applied_to_the_luma_adjust_target(self):
        selection = self.registry.select('gaussian')
        plain, shaped = [], []
        direct_dct_values(self.frame, GRIDS, SHAPES, luminance_out=plain)
        direct_dct_values(self.frame, GRIDS, SHAPES, luminance_out=shaped,
                          kernel=selection)
        self.assertEqual(plain[0].shape, shaped[0].shape)
        self.assertGreater(np.abs(plain[0]-shaped[0]).max(), 1e-3)
        # the window works on the picture as shown (gamma coded), where DC is
        # untouched: the coded mean of the goal does not move
        from animation_modem.v7_source_dct import _linear_to_srgb
        self.assertAlmostEqual(_linear_to_srgb(plain[0]).mean(),
                               _linear_to_srgb(shaped[0]).mean(), delta=2e-3)

    def test_antiring_pulls_edge_overshoot_inside_the_source_range(self):
        step = np.full((480, 400, 3), 40, np.uint8)
        step[:, 213:] = 215
        selection = self.registry.select('antiring')
        frames = []
        values = direct_dct_values(step, GRIDS, SHAPES, kernel=selection,
                                   kernel_frame_out=frames)
        refit = frames[0].post(values, GRIDS)
        mask = np.zeros(GRIDS[0], bool)
        mask[:SHAPES[0][0], :SHAPES[0][1]] = True

        def shown_range(vector):
            from scipy.fft import dctn, idctn
            luma = vector[:GRIDS[0][0]*GRIDS[0][1]].reshape(GRIDS[0])
            coefficients = dctn(luma, norm='ortho')
            coefficients[~mask] = 0
            image = idctn(coefficients, norm='ortho')*.5+.5
            return image.min(), image.max()
        low, high = 40/255, 215/255
        ref_low, ref_high = shown_range(direct_dct_values(step, GRIDS, SHAPES))
        new_low, new_high = shown_range(refit)
        self.assertGreater(max(high-ref_low*0 - high, ref_high-high,
                               low-ref_low), 0.03)          # the reference rings
        self.assertLess(max(new_high-high, low-new_low), 0.5*max(
            ref_high-high, low-ref_low))

    def test_a_kernel_that_fails_is_bypassed_and_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            _write(folder, 'flaky', '''
                import numpy as np
                PARAMS = {'trouble': (0, 0, 1, 1, 'raises when 1', True)}
                def gain(ctx, trouble):
                    if trouble:
                        raise RuntimeError('went wrong')
                    return np.ones(ctx.grid)
            ''')
            registry = K.KernelRegistry([folder])
            registry.scan()
            kernel = registry.get('flaky')
            selection = registry.select('flaky', {'trouble': 1})
            reference = direct_dct_values(self.frame, GRIDS, SHAPES)
            for _ in range(3):
                values = direct_dct_values(self.frame, GRIDS, SHAPES,
                                           kernel=selection)
            np.testing.assert_allclose(values, reference, atol=1e-9)
            self.assertIn('went wrong', kernel.take_error())
            self.assertIsNone(kernel.take_error())

    def test_masks_of_the_wrong_size_fall_back_to_the_sent_rectangle(self):
        selection = self.registry.select('band_taper', {'use_guests': 1})
        wrong = [np.ones(10, bool)]*3
        np.testing.assert_array_equal(
            direct_dct_values(self.frame, GRIDS, SHAPES, kernel=selection,
                              kernel_masks=wrong),
            direct_dct_values(self.frame, GRIDS, SHAPES, kernel=selection))


class CostTests(unittest.TestCase):
    def test_a_slow_refit_is_reported_once(self):
        with tempfile.TemporaryDirectory() as folder:
            _write(folder, 'slowpoke', '''
                import time
                def post(grid, ctx):
                    time.sleep(0.03)
                    return grid
            ''')
            registry = K.KernelRegistry([folder])
            registry.scan()
            controls = v7_live.LiveKernelControls(registry, 'slowpoke')
            frame = _frame()
            for _ in range(2):
                frames = []
                values = direct_dct_values(
                    frame, GRIDS, SHAPES, kernel=controls.options({})['kernel'],
                    kernel_frame_out=frames)
                frames[0].post(values, GRIDS)
            notices = controls.take_notices()
            self.assertEqual(len(notices), 1)
            self.assertIn('slowpoke is slow', notices[0])
            self.assertEqual(controls.take_notices(), [])


class ContextTests(unittest.TestCase):
    def test_reduce_covers_the_pixels_each_cell_touches(self):
        rng = np.random.default_rng(1)
        array = rng.random((385, 321))
        ctx = K.KernelContext(0, (96, 80), (48, 40))
        lows, highs = ctx.reduce(array, 'min'), ctx.reduce(array, 'max')
        self.assertEqual(lows.shape, (96, 80))
        self.assertTrue(np.all(lows <= highs))
        self.assertEqual(lows.min(), array.min())
        self.assertEqual(highs.max(), array.max())
        np.testing.assert_allclose(ctx.reduce(np.ones((385, 321)), 'mean'), 1.0)

    def test_project_keeps_only_the_masked_coefficients(self):
        ctx = K.KernelContext(0, (96, 80), (48, 40))
        rng = np.random.default_rng(2)
        grid = rng.random((96, 80))
        once = ctx.project(grid)
        np.testing.assert_allclose(ctx.project(once), once, atol=1e-12)
        from scipy.fft import dctn
        spectrum = dctn(once, norm='ortho')
        self.assertLess(np.abs(spectrum[48:]).max(), 1e-10)
        self.assertLess(np.abs(spectrum[:, 40:]).max(), 1e-10)

    def test_extent_includes_the_guests(self):
        mask = np.zeros((96, 80), bool)
        mask[:48, :40] = True
        mask[:59, :49] = True
        ctx = K.KernelContext(0, (96, 80), (48, 40), mask)
        self.assertEqual(ctx.extent, (59, 49))
        self.assertGreater(ctx.band()[55, 0], 1.0)
        self.assertLess(ctx.band(extent=True)[55, 0], 1.0)


class LiveControlTests(unittest.TestCase):
    def setUp(self):
        self.registry = K.open_registry()
        self.controls = v7_live.LiveKernelControls(self.registry)

    def send(self, **message):
        return self.controls.update(json.dumps(message))

    def test_starts_on_the_reference(self):
        self.assertIsNone(self.controls.options({})['kernel'])

    def test_profile_defaults_follow_the_selected_wire_profile(self):
        aspect_mono = v7_live.LiveKernelControls(
            self.registry, 'lanczos', profile='aspect-mono-500')
        mono = aspect_mono.options({})['kernel']
        self.assertEqual(mono.params['width'], 0.5)
        self.assertEqual(mono.params['luma_mix'], 0.25)

        aspect_stereo = v7_live.LiveKernelControls(
            self.registry, 'lanczos', profile='aspect-fold-500')
        stereo = aspect_stereo.options({})['kernel']
        self.assertEqual(stereo.params['width'], 0.5)
        self.assertEqual(stereo.params['luma_mix'], 0.5)

        stereo_viewer = self.registry.select(
            'viewer_solve', profile='aspect-fold-500')
        self.assertEqual(stereo_viewer.params['luma_mix'], 0.2)
        self.assertEqual(stereo_viewer.params['tame'], 0)

    def test_choosing_and_tuning_a_kernel(self):
        self.assertTrue(self.send(kernel='lanczos', kernel_params={'width': 0.8}))
        selection = self.controls.options({})['kernel']
        self.assertEqual(selection.kernel.name, 'lanczos')
        self.assertEqual(selection.params['width'], 0.8)
        self.assertTrue(self.send(kernel_params={'width': 1.1}))
        self.assertEqual(self.controls.options({})['kernel'].params['width'], 1.1)

    def test_each_kernel_remembers_its_own_values(self):
        self.send(kernel='lanczos', kernel_params={'width': 0.7})
        self.send(kernel='gaussian')
        self.send(kernel='lanczos')
        self.assertEqual(self.controls.options({})['kernel'].params['width'], 0.7)

    def test_bad_requests_keep_the_running_kernel_and_say_why(self):
        self.send(kernel='lanczos')
        self.assertTrue(self.send(kernel='nonesuch'))
        self.assertTrue(self.send(kernel_params={'nonesuch': 1}))
        self.assertEqual(self.controls.options({})['kernel'].kernel.name, 'lanczos')
        self.assertEqual(len(self.controls.take_notices()), 2)
        self.assertFalse(self.controls.update('not json'))
        self.assertFalse(self.controls.update('[1]'))

    def test_dct_values_are_validated_and_merged(self):
        self.send(dct={'sharpen': 'taper', 'sharpen_strength': 0.4,
                       'clarity': 7, 'chroma_gain': 'x'})
        options = self.controls.options({'sharpen': 'off', 'clarity': 0.0,
                                         'chroma_gain': 1.0})
        self.assertEqual(options['sharpen'], 'taper')
        self.assertEqual(options['sharpen_strength'], 0.4)
        self.assertEqual(options['clarity'], 0.0)       # 7 refused
        self.assertEqual(options['chroma_gain'], 1.0)   # 'x' refused

    def test_reload_picks_up_a_file_dropped_in_while_running(self):
        with tempfile.TemporaryDirectory() as folder:
            registry = K.KernelRegistry([folder])
            registry.scan()
            controls = v7_live.LiveKernelControls(registry)
            _write(folder, 'newcomer', TRIANGLE)
            self.assertTrue(controls.update('{"kernels": "reload"}'))
            self.assertIn('+newcomer', controls.take_notices()[0])
            self.assertTrue(controls.update('{"kernel": "newcomer"}'))
            self.assertEqual(controls.options({})['kernel'].kernel.name, 'newcomer')
            (Path(folder)/'newcomer.py').unlink()
            controls.update('{"kernels": "reload"}')
            self.assertIsNone(controls.options({})['kernel'])      # fell back

    def test_the_reader_feeds_the_controller(self):
        import io
        import threading
        stream = io.StringIO('{"kernel": "mitchell"}\n{"brightness": 1.2}\n')
        tones = v7_live.LiveToneControls(1.0, 1.0)
        v7_live._read_live_tone_controls(stream, tones, threading.Event(),
                                         None, self.controls)
        self.assertEqual(self.controls.options({})['kernel'].kernel.name, 'mitchell')
        self.assertEqual(tones.snapshot()['brightness'], 1.2)

    def test_live_encode_uses_the_chosen_kernel(self):
        model = v7.load_model(.1521/np.sqrt(1+10**(v7.CLOCK_REL_DB/10)), 'box')
        frame = _frame()
        base = {'sharpen': 'off', 'sharpen_strength': .25, 'clarity': 0.0,
                'chroma_gain': 1.0, 'aggregation': 'off', 'band_profile': 'off'}
        reference, _ = v7_live._values(
            model, frame, 'box', 1.0, 1.0, dct_encode=True,
            dct_options=self.controls.options(base))
        np.testing.assert_array_equal(reference, direct_dct_values(
            frame, model.coder.grids, model.coder.shapes))
        self.send(kernel='gaussian')
        shaped, _ = v7_live._values(
            model, frame, 'box', 1.0, 1.0, dct_encode=True,
            dct_options=self.controls.options(base))
        self.assertGreater(np.abs(shaped-reference).max(), 1e-3)

    def test_the_preview_shows_what_is_sent_once_a_kernel_is_chosen(self):
        model = v7.load_model(.1521/np.sqrt(1+10**(v7.CLOCK_REL_DB/10)), 'box')
        frame = _frame()
        base = {'sharpen': 'off', 'sharpen_strength': .25, 'clarity': 0.0,
                'chroma_gain': 1.0, 'aggregation': 'off', 'band_profile': 'off'}
        masks_for = lambda aspect: [None, None, None]
        shots = []
        for name in ('reference', 'gaussian'):
            self.send(kernel=name)
            _values, _aspect, preview = v7_live._values(
                model, frame, 'box', 1.0, 1.0, dct_encode=True,
                dct_options=self.controls.options(base), return_resized=True,
                kernel_masks_for=masks_for)
            shots.append(np.asarray(preview, float))
        # the reference keeps the toned-source preview; a kernel shows the
        # reconstruction, which is a different (and smaller-detail) picture.
        self.assertNotEqual(shots[0].shape, shots[1].shape)
        self.assertEqual(shots[1].shape[2], 3)

    def test_post_runs_after_luma_adjust_with_the_wire_masks(self):
        model = v7.load_model(.1521/np.sqrt(1+10**(v7.CLOCK_REL_DB/10)), 'box')
        frame = _frame()
        self.send(kernel='antiring')
        seen = []

        def masks_for(aspect):
            seen.append(aspect)
            mask = np.zeros(GRIDS[0], bool)
            mask[:SHAPES[0][0], :SHAPES[0][1]] = True
            chroma = np.zeros(GRIDS[1], bool)
            chroma[:SHAPES[1][0], :SHAPES[1][1]] = True
            return [mask.ravel(), chroma.ravel(), chroma.ravel()]
        base = {'sharpen': 'off', 'sharpen_strength': .25, 'clarity': 0.0,
                'chroma_gain': 1.0, 'aggregation': 'off', 'band_profile': 'off'}
        chroma_masks = masks_for(0)[1:]
        seen.clear()
        values, aspect = v7_live._values(
            model, frame, 'box', 1.0, 1.0, dct_encode=True,
            dct_options=self.controls.options(base),
            chroma_sent_for=lambda a: chroma_masks, kernel_masks_for=masks_for)
        self.assertEqual(seen, [aspect])
        self.assertTrue(np.isfinite(values).all())


class CommandLineTests(unittest.TestCase):
    def parse(self, *extra):
        return v7_live.parser().parse_args(
            ['send', '--device', 'null', '--source', 'test', *extra])

    def test_a_kernel_needs_direct_dct_encode(self):
        args = self.parse('--dct-kernel', 'lanczos')
        self.assertTrue(v7_live._dct_options_explicit(args))
        with self.assertRaisesRegex(ValueError, 'require --dct-encode'):
            v7_live._send_profile(args, 0)

    def test_reference_is_not_an_enhancement(self):
        self.assertFalse(v7_live._dct_options_explicit(
            self.parse('--dct-kernel', 'reference')))

    def test_parameters_parse(self):
        args = self.parse('--dct-kernel-param', 'width=0.8',
                          '--dct-kernel-param', 'lobes=2')
        self.assertEqual(v7_live._kernel_cli_values(args.dct_kernel_param),
                         {'width': 0.8, 'lobes': 2.0})
        with self.assertRaises(ValueError):
            v7_live._kernel_cli_values(['width'])
        with self.assertRaises(ValueError):
            v7_live._kernel_cli_values(['width=wide'])

    def test_extra_folders_are_collected(self):
        args = self.parse('--dct-kernel-dir', '/a', '--dct-kernel-dir', '/b')
        self.assertEqual(args.dct_kernel_dir, ['/a', '/b'])


if __name__ == '__main__':
    unittest.main()
