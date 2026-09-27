"""Pure layout and status formatting tests for the optional GL viewer."""
import unittest

from tools.v7_gl_viewer import (fit_viewport, title_for_status,
                                toolbar_layout, _toolbar_image)
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
        self.assertNotIn('mode:lanczos3', upscale)
        panel = toolbar_layout(960, 'panel')
        self.assertIn('panel:1', panel)
        self.assertIn('panel:0', panel)
        self.assertIn('fullscreen_button', toolbar_layout(960))
        self.assertIn('save_default_button', toolbar_layout(960))

    def test_toolbar_draws_dropdown_in_viewer_style(self):
        pixels = _toolbar_image((960, 720), 'bilinear', True, 'upscale')
        self.assertEqual(pixels.shape, (110, 960, 4))

    def test_preview_rejects_nonpositive_or_nonfinite_fps_before_loading(self):
        for fps in ('0', '-1', 'nan', 'inf', '-inf'):
            with self.subTest(fps=fps), self.assertRaisesRegex(
                    SystemExit, '--fps must be finite and positive'):
                preview_main(['missing-image.png', f'--fps={fps}'])

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

if __name__ == '__main__':
    unittest.main()
