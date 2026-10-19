import io
import unittest

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
