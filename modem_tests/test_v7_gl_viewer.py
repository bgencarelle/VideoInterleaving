"""Layout, status, and float reconstruction tests for the optional GL viewer."""
import unittest

import numpy as np
from PIL import Image

from tools.v7_gl_viewer import (DISPLAY_MODES, fit_viewport, float_planes,
                                title_for_status,
                                toolbar_layout, _toolbar_image,
                                _diagnostic_image)
from tools.v7_viewer import main as preview_main


class GLViewerHelperTests(unittest.TestCase):
    def test_viewport_letterboxes_without_distorting_aspect(self):
        self.assertEqual(fit_viewport((1920, 1080), 4/3),
                         (240, 0, 1440, 1080))
        self.assertEqual(fit_viewport((800, 600), 16/9),
                         (0, 75, 800, 450))

    def test_zero_sized_framebuffer_has_empty_viewport(self):
        self.assertEqual(fit_viewport((0, 720), 4/3), (0, 0, 0, 0))

    def test_toolbar_dropdowns_expose_only_supported_choices(self):
        upscale = toolbar_layout(960, 'upscale')
        self.assertIn('mode:nearest', upscale)
        self.assertIn('mode:bilinear', upscale)
        self.assertIn('mode:sharp-bilinear', upscale)
        self.assertIn('mode:bicubic', upscale)
        self.assertNotIn('mode:lanczos3', upscale)
        panel = toolbar_layout(960, 'panel')
        self.assertIn('panel:1', panel)
        self.assertIn('panel:0', panel)
        self.assertIn('fullscreen_button', toolbar_layout(960))
        self.assertIn('save_default_button', toolbar_layout(960))

    def test_toolbar_draws_dropdown_in_viewer_style(self):
        pixels = _toolbar_image((960, 720), 'bilinear', True, 'upscale')
        self.assertEqual(pixels.shape,
                         (48+len(DISPLAY_MODES)*29+4, 960, 4))

    def test_float_planes_preserve_unclipped_values_and_leave_input_unchanged(self):
        values = np.array([-1.2, 0.0, 1.3, -.5, .25, .75, .1, .2, .3],
                          np.float64)
        original = values.copy()
        planes = float_planes(values, ((1, 3), (1, 3), (1, 3)))

        self.assertEqual([plane.dtype for plane in planes], [
            np.dtype(np.float32)]*3)
        self.assertEqual([plane.shape for plane in planes], [(1, 3)]*3)
        np.testing.assert_allclose(planes[0], [[-1.2, 0.0, 1.3]],
                                   rtol=0.0, atol=1e-7)
        np.testing.assert_array_equal(planes[1], [[-.5, .25, .75]])
        np.testing.assert_array_equal(values, original)

    def test_grayscale_float_planes_get_neutral_chroma(self):
        y, cb, cr = float_planes(np.array([-1.0, 0.0, 1.0]), ((1, 3),))

        self.assertEqual(y.shape, (1, 3))
        neutral = np.full((1, 1), 1.0/255.0, np.float32)
        np.testing.assert_array_equal(cb, neutral)
        np.testing.assert_array_equal(cr, neutral)

    def test_float_plane_validation_rejects_mismatched_or_nonfinite_values(self):
        with self.assertRaisesRegex(ValueError, 'do not match'):
            float_planes(np.zeros(2), ((2, 2),))
        with self.assertRaisesRegex(ValueError, 'must be finite'):
            float_planes(np.array([0, np.nan]), ((1, 2),))

    def test_diagnostic_cards_expand_for_decode_cpu_and_gui_resources(self):
        diagnostics = {
            'status': ('RECEIVED',),
            'sync': ('shown 1 / 10',),
            'decode': ('good 1 · lost 0',),
            'input': ('peak mono -2 dBFS',),
            'signal': ('aspect 4:3',),
            'decode_cpu': ('last 3.1 ms/frame', 'average 8% / core'),
            'resources': ('thread 2% · proc 12%', 'RSS 256 MiB'),
        }
        pixels = _diagnostic_image((960, 194), diagnostics)
        self.assertEqual(pixels.shape, (194, 960, 4))

    def test_preview_rejects_nonpositive_or_nonfinite_fps_before_loading(self):
        for fps in ('0', '-1', 'nan', 'inf', '-inf'):
            with self.subTest(fps=fps), self.assertRaisesRegex(
                    SystemExit, '--fps must be finite and positive'):
                preview_main(['missing-image.png', f'--fps={fps}'])

    def test_preview_cli_accepts_every_display_mode(self):
        for mode in DISPLAY_MODES:
            with self.subTest(mode=mode), self.assertRaisesRegex(
                    SystemExit, 'no readable image paths'):
                preview_main(['missing-image.png', '--display-mode', mode])

    def test_title_keeps_status_and_optional_live_metrics(self):
        meter = {'status': 'received', 'source_index': 42, 'lag_ms': -35.0,
                 'input_fps': 29.97, 'auto_gain': 1.5, 'dropped': 2,
                 'device': 'BlackHole', 'capture_rate': 48000,
                 'input_channels': 2, 'mode': 'M/S'}
        title = title_for_status(meter)
        self.assertIn('RECEIVED', title)
        self.assertIn('BlackHole · 48 kHz · 2 ch · M/S', title)
        self.assertIn('index 42', title)
        self.assertIn('lag -35 ms', title)
        detailed = title_for_status(meter, details=True)
        self.assertIn('30.0 fps', detailed)
        self.assertIn('gain 1.5x', detailed)
        self.assertIn('drops 2', detailed)


class FloatShaderReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import moderngl
        except ImportError as exc:
            raise unittest.SkipTest(f'ModernGL is unavailable: {exc}')
        try:
            cls.context = moderngl.create_standalone_context(
                require=330, backend='egl')
        except Exception as exc:
            raise unittest.SkipTest(f'headless OpenGL is unavailable: {exc}')
        cls.moderngl = moderngl
        from tools.v7_gl_viewer import (FLOAT_FRAGMENT_SHADER,
                                        FLOAT_MODE_IDS, VERTEX_SHADER)
        cls.mode_ids = FLOAT_MODE_IDS
        cls.program = cls.context.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FLOAT_FRAGMENT_SHADER)
        cls.program['plane_y'].value = 0
        cls.program['plane_cb'].value = 1
        cls.program['plane_cr'].value = 2
        cls.vertex_array = cls.context.vertex_array(cls.program, [])
        cls.output = cls.context.texture((1, 1), 4, dtype='f4')
        cls.framebuffer = cls.context.framebuffer(
            color_attachments=[cls.output])

    @classmethod
    def tearDownClass(cls):
        for name in ('framebuffer', 'output', 'vertex_array', 'program',
                     'context'):
            resource = getattr(cls, name, None)
            if resource is not None:
                resource.release()

    def _render(self, rgb, mode):
        from tools.v7_gl_viewer import _float_texture_filter, float_planes

        ycbcr = np.asarray(
            Image.new('RGB', (1, 1), rgb).convert('YCbCr').getpixel((0, 0)),
            dtype=np.uint8)
        values = ycbcr.astype(np.float32)/127.5-1.0
        planes = float_planes(values, ((1, 1), (1, 1), (1, 1)))
        textures = []
        try:
            filtering = _float_texture_filter(mode, self.moderngl)
            for unit, plane in enumerate(planes):
                texture = self.context.texture(
                    (1, 1), 1, plane.tobytes(), dtype='f4')
                texture.filter = (filtering, filtering)
                texture.repeat_x = False
                texture.repeat_y = False
                texture.use(location=unit)
                textures.append(texture)
            self.program['reconstruction'].value = self.mode_ids[mode]
            self.program['output_size'].value = (1.0, 1.0)
            self.framebuffer.use()
            self.context.viewport = (0, 0, 1, 1)
            self.vertex_array.render(
                mode=self.moderngl.TRIANGLES, vertices=3)
            rendered = np.frombuffer(
                self.framebuffer.read(components=4, dtype='f4'),
                dtype=np.float32).copy()
        finally:
            for texture in textures:
                texture.release()
        reference = np.asarray(
            Image.fromarray(ycbcr.reshape(1, 1, 3), 'YCbCr')
            .convert('RGB').getpixel((0, 0)), dtype=np.float32)/255.0
        return rendered[:3], reference

    def _render_luma_ramp(self, mode):
        from tools.v7_gl_viewer import _float_texture_filter, float_planes

        planes = float_planes(np.array([-1.0, 1.0]), ((1, 2),))
        textures = []
        output = self.context.texture((4, 1), 4, dtype='f4')
        framebuffer = self.context.framebuffer(color_attachments=[output])
        try:
            filtering = _float_texture_filter(mode, self.moderngl)
            for unit, plane in enumerate(planes):
                texture = self.context.texture(
                    (plane.shape[1], plane.shape[0]), 1,
                    plane.tobytes(), dtype='f4')
                texture.filter = (filtering, filtering)
                texture.repeat_x = False
                texture.repeat_y = False
                texture.use(location=unit)
                textures.append(texture)
            self.program['reconstruction'].value = self.mode_ids[mode]
            self.program['output_size'].value = (4.0, 1.0)
            framebuffer.use()
            self.context.viewport = (0, 0, 4, 1)
            self.vertex_array.render(
                mode=self.moderngl.TRIANGLES, vertices=3)
            rendered = np.frombuffer(
                framebuffer.read(components=4, dtype='f4'),
                dtype=np.float32).reshape(1, 4, 4)[0, :, :3].copy()
        finally:
            for texture in textures:
                texture.release()
            framebuffer.release()
            output.release()
        return rendered

    def test_float_reconstruction_color_matches_pillow_ycbcr_reference(self):
        colors = ((0, 0, 0), (255, 255, 255), (128, 128, 128),
                  (255, 0, 0), (0, 255, 0), (0, 0, 255))
        for mode in ('bilinear', 'sharp-bilinear', 'bicubic'):
            for color in colors:
                with self.subTest(mode=mode, color=color):
                    rendered, reference = self._render(color, mode)
                    self.assertLessEqual(
                        float(np.max(np.abs(rendered-reference))),
                        1.0/255.0+1e-6)

    def test_reconstruction_modes_have_distinct_expected_edge_profiles(self):
        bilinear = self._render_luma_ramp('bilinear')
        sharp = self._render_luma_ramp('sharp-bilinear')
        bicubic = self._render_luma_ramp('bicubic')

        np.testing.assert_allclose(bilinear[:, 0], [0.0, .25, .75, 1.0],
                                   atol=1e-6)
        np.testing.assert_allclose(sharp[:, 0], [0.0, 0.0, 1.0, 1.0],
                                   atol=1e-6)
        np.testing.assert_allclose(bilinear, np.repeat(bilinear[:, :1], 3,
                                                        axis=1), atol=1e-6)
        np.testing.assert_allclose(sharp, np.repeat(sharp[:, :1], 3,
                                                     axis=1), atol=1e-6)
        self.assertGreater(bicubic[1, 0], .20)
        self.assertLess(bicubic[1, 0], .25)
        self.assertAlmostEqual(float(bicubic[1, 0]+bicubic[2, 0]), 1.0,
                               places=6)
        np.testing.assert_allclose(bicubic, np.repeat(bicubic[:, :1], 3,
                                                       axis=1), atol=1e-6)


if __name__ == '__main__':
    unittest.main()
