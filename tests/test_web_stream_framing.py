import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import web_service

from web_service import (_MjpegWriter, HEADER_BOUNDARY, HEADER_CTYPE_JPEG,
                         HEADER_CTYPE_WEBP)


class FlushedBytes(io.BytesIO):
    def __init__(self):
        super().__init__()
        self.flushes = 0

    def flush(self):
        self.flushes += 1


class MultipartFrameTests(unittest.TestCase):
    def test_first_picture_is_delimited_without_waiting_for_next_picture(self):
        stream = FlushedBytes()
        writer = _MjpegWriter(stream)
        writer.write(b'picture')
        self.assertEqual(stream.getvalue(),
            b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: 7\r\n\r\n'
            b'picture\r\n--frame\r\n')
        self.assertEqual(stream.flushes, 1)

    def test_consecutive_pictures_have_no_duplicate_or_empty_parts(self):
        stream = FlushedBytes()
        writer = _MjpegWriter(stream)
        writer.write(b'jpeg', HEADER_CTYPE_JPEG)
        writer.write(b'webp\x00data', HEADER_CTYPE_WEBP)
        parts = stream.getvalue().split(HEADER_BOUNDARY)
        self.assertEqual(len(parts), 4)
        self.assertEqual(parts[0], b'')
        self.assertEqual(parts[-1], b'')
        self.assertEqual(parts[1],
            b'Content-Type: image/jpeg\r\nContent-Length: 4\r\n\r\njpeg\r\n')
        self.assertEqual(parts[2],
            b'Content-Type: image/webp\r\nContent-Length: 9\r\n\r\nwebp\x00data\r\n')
        self.assertEqual(stream.flushes, 2)


def scope_stream_bytes(kind):
    """Exercise each real HTTP handler, ending after its first picture."""
    import numpy as np
    handler = web_service.MonitorHandler.__new__(web_service.MonitorHandler)
    handler.path = '/scope/stream.mjpg?size=256' if kind == 'preview' else '/scope/luma.mjpg'
    handler.connection = Mock()
    handler.wfile = FlushedBytes()
    handler.send_response = Mock()
    handler.send_header = Mock()
    handler.end_headers = Mock()
    scope = SimpleNamespace(want_tap=Mock(),
                            read_tap=Mock(return_value=(1, object())),
                            read_luma=Mock(return_value=(1, np.full((8, 8), 128, np.uint8))))
    with patch.dict('sys.modules', {
            'scope_out': SimpleNamespace(Scope=scope),
            'scope_bake': SimpleNamespace(PreviewWorkspace=lambda size: None,
                                          _warm_preview_kernels=lambda: None)}), \
            patch.object(web_service, '_scope_jpeg', return_value=b'preview'), \
            patch.object(web_service.time, 'sleep', side_effect=OSError('test complete')):
        if kind == 'preview':
            handler._scope_mjpeg()
        else:
            handler._scope_luma_stream()
    return handler.wfile.getvalue()


class ScopeMultipartTests(unittest.TestCase):
    def test_both_actual_scope_handlers_emit_complete_length_framed_parts(self):
        for kind in ('preview', 'luma'):
            with self.subTest(kind=kind):
                wire = scope_stream_bytes(kind)
                headers, rest = wire.split(b'\r\n\r\n', 1)
                length = int(headers.split(b'Content-Length: ')[1])
                self.assertTrue(wire.startswith(HEADER_BOUNDARY))
                self.assertEqual(rest[length:], b'\r\n' + HEADER_BOUNDARY)
                self.assertGreater(length, 0)
