"""Public terminal bounds, protocol fragmentation, and non-scrolling redraws."""
import re
import unittest
from unittest.mock import patch

import numpy as np

from ascii_terminal import WindowSize, TerminalPictures, SCALES, SYNC_BEGIN, SYNC_END, LEAVE
import ascii_converter
import settings


def naws(cols, rows):
    body = cols.to_bytes(2, 'big') + rows.to_bytes(2, 'big')
    return b'\xff\xfa\x1f' + body.replace(b'\xff', b'\xff\xff') + b'\xff\xf0'


class TerminalTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(settings, 'ASCII_TRUECOLOR', False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_raw_client_can_choose_each_fixed_scale(self):
        window = WindowSize()
        for key, size in zip(b'1234', SCALES):
            self.assertEqual(window.feed(bytes([key]) + b'\r\n'), b'')
            self.assertEqual(window.size, size)
        # Protocol payload bytes that happen to be digits are not commands.
        window.feed(naws(49, 50))
        self.assertEqual(window.size, (49, 50))

    def test_four_scales_select_largest_that_fits(self):
        pictures = TerminalPictures()
        pictures.publish(np.zeros((40, 60, 3), np.uint8))
        # A non-space grid makes its occupied rectangle measurable.
        with patch('ascii_terminal.ascii_converter.to_ascii',
                   side_effect=lambda frame, width, height, **kwargs: '\r\n'.join(['X' * width] * height)):
            for size, expected in (((40,24),SCALES[0]), ((67,38),SCALES[1]),
                                   ((93,46),SCALES[2]), ((160,60),SCALES[3])):
                _, data = pictures.picture(size)
                rows = re.split(rb'\x1b\[\d+;1H', data)[1:]
                plain = [re.sub(rb'\x1b\[[\d;]*m', b'', row) for row in rows]
                occupied = [row.count(b'X') for row in plain if b'X' in row]
                self.assertEqual((max(occupied),len(occupied)),expected)

    def test_sampled_glyphs_and_colours_survive_terminal_framing(self):
        frame = np.random.default_rng(51).integers(0, 256, (480, 360, 3), dtype=np.uint8)
        expected = ascii_converter.to_ascii(frame, width=60, height=40)
        pictures = TerminalPictures()
        pictures.publish(frame)
        _, actual = pictures.picture((60, 40))
        # Ignore cursor placement and background setup; compare every glyph
        # together with the exact foreground SGR active at that glyph.
        def cells(text):
            color = None
            result = []
            for token in re.findall(r'\x1b\[[0-9;?]*[mHhl]|[^\r\n]', text):
                if token.startswith('\x1b['):
                    if token.endswith('m') and token != '\x1b[40m':
                        color = token
                else:
                    result.append((token, color))
            return result
        self.assertEqual(cells(expected), cells(actual.decode()))

    def test_large_scale_resolves_detail_not_enlarged_from_small_grid(self):
        # Fine vertical bars have structure between the small grid's samples.
        y, x = np.indices((480,360))
        frame = np.repeat((((x // 3) % 2) * 255).astype(np.uint8)[:, :, None], 3, axis=2)
        pictures = TerminalPictures()
        pictures.publish(frame)
        with patch.object(settings, 'ASCII_COLOR', False):
            _, low = pictures.picture((60,40))
            _, high = pictures.picture((90,60))
        def grid(data):
            rows = re.split(rb'\x1b\[\d+;1H',data)[1:]
            return np.array([list(re.sub(rb'\x1b\[[\d;?]*[mhl]',b'',row).decode()) for row in rows])
        low, high = grid(low), grid(high)
        enlarged = low[np.arange(60)*40//60][:, np.arange(90)*60//90]
        self.assertEqual(high.shape,(60,90))
        self.assertGreater(np.count_nonzero(high != enlarged), 100)

    def test_each_scale_converts_its_own_rendered_rgb(self):
        frames = {size: np.full((size[1], size[0], 3), i * 60, np.uint8)
                  for i, size in enumerate(SCALES)}
        pictures = TerminalPictures()
        pictures.publish(frames)
        with patch('ascii_terminal.ascii_converter.to_ascii',
                   side_effect=lambda rgb, width, height, **kwargs: '\r\n'.join(['X'*width]*height)) as convert:
            for size in SCALES:
                pictures.picture(size)
                self.assertIs(convert.call_args.args[0], frames[size])
                self.assertEqual(convert.call_args.kwargs,
                                 dict(width=size[0], height=size[1], sampled=True))

    def test_sampled_grid_is_not_resized_or_cropped_again(self):
        frame = np.random.default_rng(71).integers(0,256,(30,45,3),dtype=np.uint8)
        with patch.object(ascii_converter.cv2, 'resize', side_effect=AssertionError('resized grid')):
            text = ascii_converter.to_ascii(frame,width=45,height=30,sampled=True)
        plain = re.sub(r'\x1b\[[\d;]*m','',text).split('\r\n')
        self.assertEqual(len(plain),30)
        self.assertTrue(all(len(row)==45 for row in plain))
        with self.assertRaises(ValueError):
            ascii_converter.to_ascii(frame,width=60,height=40,sampled=True)

    def test_shipped_colour_does_not_blur_source_tints(self):
        self.assertEqual(settings.ASCII_COLOR_BLUR,0)
        frame = np.zeros((8,12,3),np.uint8)
        frame[:,:4] = (255,0,0)
        frame[:,4:8] = (0,255,0)
        frame[:,8:] = (0,0,255)
        with patch.object(ascii_converter.cv2,'GaussianBlur',side_effect=AssertionError('blurred tint')):
            text = ascii_converter.to_ascii(frame,width=12,height=8,sampled=True)
        for index in (196,46,21):
            self.assertIn(f'\x1b[38;5;{index}m',text)

    def test_near_white_and_warm_highlights_do_not_fall_to_dull_grey(self):
        with patch.object(settings,'ASCII_BRIGHTNESS',1.0), \
                patch.object(settings,'ASCII_CONTRAST',1.0), \
                patch.object(settings,'ASCII_SATURATION',1.0):
            for rgb in ((255,255,255),(254,254,254),(250,248,245),(255,250,240)):
                text = ascii_converter.to_ascii(np.array([[rgb]],np.uint8),
                                                width=1,height=1,sampled=True)
                self.assertIn('\x1b[38;5;231m',text,rgb)

    def test_palette_levels_are_nearest_actual_xterm_values(self):
        # Palette selection must not systematically understate brightness.
        for rgb, expected in ((95,59),(135,102),(175,145),(215,188),(255,231)):
            with patch.object(settings,'ASCII_BRIGHTNESS',1.0), \
                    patch.object(settings,'ASCII_CONTRAST',1.0):
                text = ascii_converter.to_ascii(np.full((1,1,3),rgb,np.uint8),
                                                width=1,height=1,sampled=True)
            self.assertIn(f'\x1b[38;5;{expected}m',text)

    def test_fragmented_negotiation_and_escaped_iac(self):
        window = WindowSize()
        for byte in b'\xff\xfb\x1f' + naws(255, 32):
            self.assertEqual(window.feed(bytes([byte])), b'')
        self.assertEqual(window.size, (160, 32))
        window.feed(naws(12, 5))
        self.assertEqual(window.size, (12, 5))

    def test_malformed_and_oversize_input_is_bounded(self):
        window = WindowSize()
        window.feed(b'\xff\xfa\x1f' + b'X' * 100000 + b'\xff\xf0')
        self.assertEqual(window.size, (60, 24))
        self.assertLessEqual(len(window.data), 64)
        window.feed(naws(0, 0))
        self.assertEqual(window.size, (60, 24))
        window.feed(naws(65535, 65535))
        self.assertEqual(window.size, (160, 60))
        window.feed(naws(80, 24))
        self.assertEqual(window.size, (80, 24))

    def test_latest_picture_join_resize_and_no_scroll_controls(self):
        pictures = TerminalPictures(variants=2)
        pictures.publish(np.zeros((40, 60, 3), np.uint8))
        identity, payload = pictures.picture((20, 8))
        self.assertIsNotNone(payload)  # Picture published before subscriber joined.
        self.assertNotIn(b'\n', payload)
        self.assertNotIn(b'\r', payload)
        lines = re.split(rb'\x1b\[\d+;1H', payload)[1:]
        self.assertEqual(len(lines), 8)
        for line in lines:
            self.assertEqual(len(re.sub(rb'\x1b\[[\d;?]*[mhl]', b'', line)), 20)
        self.assertIsNone(pictures.picture((20, 8), identity)[1])
        self.assertIsNotNone(pictures.picture((10, 4), identity)[1])
        pictures.picture((30, 12))
        self.assertEqual(len(pictures.cache), 2)
        pictures.publish(np.full((40, 60, 3), 255, np.uint8))
        self.assertIsNotNone(pictures.picture((20, 8), identity)[1])
        self.assertEqual(len(pictures.cache), 1)

    def test_same_size_conversion_shared_and_invalid_size_rejected(self):
        pictures = TerminalPictures()
        pictures.publish(np.zeros((40, 60, 3), np.uint8))
        with patch('ascii_terminal.ascii_converter.to_ascii', return_value='x') as convert:
            for _ in range(5):
                pictures.picture((1, 1))
            self.assertEqual(convert.call_count, 1)
        for size in ((0, 1), (161, 1), (1, 61)):
            with self.assertRaises(ValueError):
                pictures.picture(size)

    def test_complete_frame_is_one_synchronised_paint_batch(self):
        pictures = TerminalPictures()
        pictures.publish(np.zeros((40,60,3),np.uint8))
        _, payload = pictures.picture((60,40))
        self.assertTrue(payload.startswith(SYNC_BEGIN.encode()))
        self.assertTrue(payload.endswith(SYNC_END.encode()))
        self.assertEqual(payload.count(SYNC_BEGIN.encode()),1)
        self.assertEqual(payload.count(SYNC_END.encode()),1)
        self.assertTrue(LEAVE.startswith(SYNC_END.encode()))

    def test_missing_size_requests_source_grid_instead_of_resizing_another(self):
        pictures = TerminalPictures()
        self.assertEqual(pictures.requested_scales(),((60,40),))
        pictures.publish({(60,40):np.zeros((40,60,3),np.uint8)})
        identity, payload = pictures.picture((90,60))
        self.assertIsNone(identity)
        self.assertIsNone(payload)
        self.assertEqual(pictures.requested_scales(),((90,60),))
        pictures.publish({(90,60):np.zeros((60,90,3),np.uint8)})
        identity, payload = pictures.picture((90,60),identity)
        self.assertIsNotNone(payload)
        with patch('ascii_terminal.time.monotonic',return_value=10**20):
            self.assertEqual(pictures.requested_scales(),((60,40),))
