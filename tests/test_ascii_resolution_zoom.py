"""Client zoom is local; the server renders a single shared maximum grid."""
import shutil
import io
import subprocess
import unittest
from unittest import mock

import ascii_render
import settings


class ZoomTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'Node.js required for browser logic tests')
    def test_browser_resolution_and_input_controller(self):
        result = subprocess.run(['node', 'tests/ascii_resolution.test.js'],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(shutil.which('node'), 'Node.js required for browser renderer tests')
    def test_persistent_browser_grid_renderer(self):
        result = subprocess.run(['node', 'tests/ascii_grid_renderer.test.js'],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(shutil.which('node'), 'Node.js required for browser atlas tests')
    def test_batched_browser_renderer(self):
        result = subprocess.run(['node', 'tests/ascii_atlas_renderer.test.js'],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_one_master_capture_target_not_one_per_viewer(self):
        context = mock.Mock()
        capture = ascii_render.AsciiGridCapture(context, sizes=(settings.ASCII_WEB_MAX_SIZE,))
        context.simple_framebuffer.assert_called_once_with((120, 80), components=3)
        self.assertEqual(tuple(capture.targets), ((120, 80),))
        capture.close()

    def test_normal_modes_default_not_overridden(self):
        self.assertIsNone(settings.ASCII_WEB_SOURCE_SIZE)
        self.assertEqual((settings.ASCII_WIDTH, settings.ASCII_HEIGHT), (60, 40))
        self.assertEqual(settings.ASCII_WEB_MAX_SIZE, (120, 80))

    def test_namespaced_assets_and_traversal_protection(self):
        import web_service
        from pathlib import Path
        for path in ('/static/ascii_resolution.js', '/ascii/static/ascii_resolution.js?v=1',
                     '/asciistatic/ascii_resolution.js'):
            handler = mock.Mock(path=path, wfile=io.BytesIO())
            web_service._serve_static_file(handler)
            handler.send_response.assert_called_once_with(200)
            self.assertEqual(handler.wfile.getvalue(), Path('static/ascii_resolution.js').read_bytes())
        for name in ('ascii_atlas_renderer.js', 'ascii_grid_renderer.js'):
            handler = mock.Mock(path='/ascii/static/' + name + '?v=atlas-1', wfile=io.BytesIO())
            web_service._serve_static_file(handler)
            handler.send_response.assert_called_once_with(200)
            self.assertEqual(handler.wfile.getvalue(), Path('static', name).read_bytes())
        for path in ('/ascii/static/%2e%2e/%2e%2e/settings.py',
                     '/ascii/static/../../settings.py', '/asciistatic/../../settings.py'):
            handler = mock.Mock(path=path, wfile=io.BytesIO())
            web_service._serve_static_file(handler)
            handler.send_error.assert_called_once_with(403)
            self.assertEqual(handler.wfile.getvalue(), b'')

    def test_monospaced_font_and_corresponding_source_are_served(self):
        import web_service
        from pathlib import Path
        root = 'fonts/schoolbook-mono/'
        for name in ('InterleaveSchoolbookMono-Regular.woff2',
                     'InterleaveSchoolbookMono-Bold.woff2', 'schoolbook-mono-source.zip',
                     'source/LICENSE', 'source/COPYING'):
            for prefix in ('/ascii/static/', '/asciistatic/'):
                handler = mock.Mock(path=prefix + root + name + '?v=mono-1', wfile=io.BytesIO())
                web_service._serve_static_file(handler)
                handler.send_response.assert_called_once_with(200)
                self.assertEqual(handler.wfile.getvalue(), Path('static', root, name).read_bytes())


if __name__ == '__main__':
    unittest.main()
