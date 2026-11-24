"""--printer: the ASCII frame on screen as ESC/POS for the TM-T88V."""
import threading
import unittest

import numpy as np

from thermal_printer import (DoubleTap, FramePrinter, blank_unseen, fit_to_paper,
                             frame_to_escpos, ink_to_escpos, paper_indices, screen_to_ink)

GLYPHS = np.array(list('M8a. ') + ['C'])       # palette 'M8a. ' (densest first) + a phrase letter
PALETTE = 5


class PaperTests(unittest.TestCase):
    def test_paper_carries_the_characters_on_screen(self):
        base = np.array([[0, 4]])                  # brightest, darkest (blank) on screen
        shown = np.array([[5, 4]])                 # 'C' put over the first cell
        self.assertEqual(''.join(GLYPHS[paper_indices(base, shown, PALETTE)][0]), 'C ')

    def test_inverted_paper_flips_tones_and_keeps_phrase_letters(self):
        base = np.array([[0, 4]])
        self.assertEqual(''.join(GLYPHS[paper_indices(base, base, PALETTE, True)][0]), ' M')
        shown = np.array([[5, 4]])
        self.assertEqual(''.join(GLYPHS[paper_indices(base, shown, PALETTE, True)][0]), 'CM')

    def test_characters_too_dark_to_see_on_screen_print_blank(self):
        shown = np.array([[0, 3]])                 # 'M' bright, '.' in near-black
        colours = np.array([[[200, 90, 40], [20, 18, 30]]], np.uint8)
        self.assertEqual(''.join(GLYPHS[blank_unseen(shown, colours, PALETTE, 48)][0]), 'M ')
        self.assertEqual(''.join(GLYPHS[blank_unseen(shown, colours, PALETTE, 0)][0]), 'M.')

    def test_narrow_frames_are_magnified_to_fill_the_paper(self):
        grid = np.zeros((10, 15), int)
        self.assertEqual(fit_to_paper(grid, 64)[1], 4)          # 15 x 4 = 60 of 64
        self.assertEqual(fit_to_paper(np.zeros((40, 60), int), 64)[1], 1)

    def test_wide_frames_are_resampled_keeping_their_shape(self):
        grid, scale = fit_to_paper(np.zeros((80, 128), int), 64)
        self.assertEqual((grid.shape, scale), ((40, 64), 1))

    def test_a_wide_frame_is_spelled_again_after_shrinking(self):
        base = np.zeros((2, 128), np.int64)
        spelled = []

        def respell(grid):
            spelled.append(grid.shape)
            return np.full(grid.shape, 5)              # the phrase letter everywhere
        data = frame_to_escpos(base, base, GLYPHS, PALETTE, 64, respell=respell)
        self.assertEqual(spelled, [(1, 64)])
        self.assertIn(b'C' * 64, data)

    def test_escpos_resets_picks_font_b_prints_and_cuts_after_feeding(self):
        base = np.full((2, 3), 0)
        data = frame_to_escpos(base, base, GLYPHS, PALETTE, 64)
        self.assertTrue(data.startswith(b'\x1b@'))
        self.assertIn(b'\x1bM\x01', data)
        self.assertIn(b'\x1b3\x90', data)                      # 18 dots x 8: 3 columns, 8x size
        self.assertIn(b'\x1d!\x77', data)
        self.assertIn(b'MMM\nMMM\n', data)
        self.assertTrue(data.endswith(b'\x1bd\x0a\x1dV\x00'))   # 10 lines, then the cut


class ScreenTests(unittest.TestCase):
    def test_dark_on_screen_is_ink_and_bright_is_paper(self):
        rgb = np.zeros((100, 60, 3), np.uint8)
        rgb[:, 30:] = 255                          # left half black, right half white
        ink = screen_to_ink(rgb, dots=96)
        self.assertEqual(ink.shape, (160, 96))      # portrait kept, 96 dots across
        self.assertAlmostEqual(ink[:, :40].mean(), .85, delta=.03)   # black, capped at 85% of dots
        self.assertLess(ink[:, 56:].mean(), .05)

    def test_landscape_prints_across_the_full_width_unless_asked_to_turn(self):
        across = screen_to_ink(np.zeros((60, 100, 3), np.uint8), dots=96)
        self.assertEqual(across.shape, (58, 96))
        along = screen_to_ink(np.zeros((60, 100, 3), np.uint8), dots=96, rotate='auto')
        self.assertEqual(along.shape, (160, 96))

    def test_white_stays_paper(self):
        self.assertLess(screen_to_ink(np.full((40, 20, 3), 255, np.uint8), dots=32).mean(), .01)

    def test_greys_dither(self):
        ink = screen_to_ink(np.full((50, 50, 3), 128, np.uint8), dots=96)
        self.assertAlmostEqual(ink.mean(), .5, delta=.08)

    def test_raster_is_sent_slowly_in_small_bands_then_cut(self):
        ink = np.zeros((300, 576), bool)
        ink[0, 0] = True                           # the very first dot
        data = ink_to_escpos(ink)
        speed = b'\x1d(K\x02\x002\x02'          # GS ( K fn 50: speed level 2
        self.assertTrue(data.startswith(b'\x1b@' + speed + b'\x1dv0\x00\x48\x00\x20\x00'))  # 72 bytes x 32 rows
        self.assertEqual(data[2 + 7 + 8], 0x80)     # leftmost dot is the high bit
        self.assertEqual(data.count(b'\x1dv0\x00\x48\x00'), 10)   # 9 bands of 32 + one of 12
        self.assertIn(b'\x1dv0\x00\x48\x00\x0c\x00', data)
        self.assertTrue(data.endswith(b'\x1bd\x0a\x1dV\x00'))   # 10 lines, then the cut
        self.assertEqual(len(data), 2 + 7 + 10 * 8 + 300 * 72 + 6)

    def test_speed_can_be_left_to_the_printer(self):
        self.assertTrue(ink_to_escpos(np.zeros((8, 576), bool), speed=None)
                        .startswith(b'\x1b@\x1dv0'))

    def test_a_band_lost_to_a_reset_costs_at_most_one_band(self):
        from thermal_printer import RASTER_BAND
        self.assertLessEqual(RASTER_BAND * 72, 2304)   # <= 2.3 KB of garbage, then a fresh header


class SendTests(unittest.TestCase):
    def test_a_slow_printer_is_never_given_up_on_mid_picture(self):
        from unittest import mock
        import thermal_printer
        sock = mock.MagicMock()
        sock.__enter__.return_value = sock
        with mock.patch.object(thermal_printer.socket, 'create_connection',
                               return_value=sock) as connect:
            FramePrinter('printer')._send_tcp(b'data')
        self.assertEqual(connect.call_args.kwargs['timeout'], 5)     # connecting: short
        sock.settimeout.assert_called_once_with(300)                 # sending: long
        sock.sendall.assert_called_once_with(b'data')


class SchedulingTests(unittest.TestCase):
    def setUp(self):
        self.now = [0.0]
        self.sent = []
        self.done = threading.Event()

        def send(data):
            self.sent.append(data)
            self.done.set()
        self.printer = FramePrinter('printer', every=(60, 60), clock=lambda: self.now[0], send=send)

    def test_random_time_asks_for_the_next_drawn_frame(self):
        self.printer.tick()
        self.assertFalse(self.printer.take())
        self.now[0] = 61
        self.printer.tick()
        self.assertTrue(self.printer.take())
        self.assertFalse(self.printer.take())          # once per request
        self.assertEqual(self.printer.next_at, 121)

    def test_without_printer_flag_only_p_prints(self):
        printer = FramePrinter('printer', every=(60, 60), clock=lambda: self.now[0],
                               send=lambda d: None, random_prints=False)
        self.now[0] = 1000
        printer.tick()
        self.assertFalse(printer.take())
        printer.request()
        self.assertTrue(printer.take())

    def test_p_and_double_tap_ask_at_once(self):
        self.printer.request()
        self.assertTrue(self.printer.take())

    def test_screen_pixels_print_on_their_own_thread(self):
        self.assertTrue(self.printer.print_screen(np.zeros((20, 10, 3), np.uint8)))
        self.assertTrue(self.done.wait(2))
        self.assertTrue(self.sent[0].startswith(b'\x1b@\x1d(K'))

    def test_one_print_at_a_time(self):
        self.printer._busy.acquire()
        self.printer.request()
        self.assertFalse(self.printer.take())          # waits while one prints
        self.printer._busy.release()
        self.assertTrue(self.printer.take())

    def test_text_mode_prints_characters(self):
        base = np.full((2, 3), 0)
        self.assertTrue(self.printer.print_text((base, base, GLYPHS, PALETTE)))
        self.assertTrue(self.done.wait(2))
        self.assertIn(b'MMM', self.sent[0])


class DoubleTapTests(unittest.TestCase):
    def test_two_quick_taps_in_one_place(self):
        tap = DoubleTap()
        self.assertFalse(tap.press(10.0, 100, 100))
        self.assertTrue(tap.press(10.3, 110, 95))

    def test_slow_or_distant_taps_are_not_a_double_tap(self):
        tap = DoubleTap()
        tap.press(10.0, 100, 100)
        self.assertFalse(tap.press(10.6, 100, 100))             # too slow
        self.assertFalse(tap.press(10.8, 300, 100))             # too far from the last
        self.assertFalse(tap.press(10.9, 100, 100))

    def test_a_third_tap_starts_afresh(self):
        tap = DoubleTap()
        tap.press(0.0, 0, 0)
        self.assertTrue(tap.press(.2, 0, 0))
        self.assertFalse(tap.press(.3, 0, 0))


if __name__ == '__main__':
    unittest.main()
