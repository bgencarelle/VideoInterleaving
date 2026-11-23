import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import web_service
from lightweight_monitor import HTML_TEMPLATE
from server_config import MODE_SCOPE


class ScopeDashboardRouteTests(unittest.TestCase):
    def request_root(self, mode):
        handler = web_service.MonitorHandler.__new__(web_service.MonitorHandler)
        handler.path = "/"
        handler.wfile = io.BytesIO()
        handler.send_response = Mock()
        handler.send_header = Mock()
        handler.end_headers = Mock()
        config = SimpleNamespace(get_mode=lambda: mode)
        with patch("web_service.get_config", return_value=config):
            handler.do_GET()
        return handler, handler.wfile.getvalue().decode("utf-8")

    def test_scope_mode_root_serves_live_scope_dashboard(self):
        handler, body = self.request_root(MODE_SCOPE)
        handler.send_response.assert_called_once_with(200)
        self.assertIn('id="scope-img"', body)
        self.assertIn("scope/stream.mjpg", body)

    def test_other_modes_keep_the_general_monitor(self):
        _handler, body = self.request_root("web")
        self.assertEqual(body, HTML_TEMPLATE)

    def test_scope_and_scope2_aliases_serve_the_same_promoted_page(self):
        _handler, expected = self.request_root(MODE_SCOPE)
        self.assertIn('Scope Monitor — Stream & Audio', expected)
        for path in ('/scope', '/scope2', '/scope.html', '/scope2.html'):
            with self.subTest(path=path):
                handler = web_service.MonitorHandler.__new__(web_service.MonitorHandler)
                handler.path = path
                handler.wfile = io.BytesIO()
                handler.send_response = Mock()
                handler.send_header = Mock()
                handler.end_headers = Mock()
                handler.do_GET()
                self.assertEqual(handler.wfile.getvalue().decode(), expected)


if __name__ == "__main__":
    unittest.main()
