"""Local ASCII display: size mapping, zoom control and the random driver.

No window or GL: drawing is checked by running the app (see the commit).
"""
import unittest
from unittest import mock

import local_ascii
import numpy as np

from local_ascii import (Zoom, decode_scale, dimensions, glow_strength, mip_levels,
                         update_zoom_selection)

COARSE, DEFAULT, MASTER = (15, 10), (60, 40), (120, 80)


class DimensionsTests(unittest.TestCase):
    def test_anchors_match_the_browser(self):
        self.assertEqual(dimensions(10, DEFAULT, MASTER), (15, 10))
        self.assertEqual(dimensions(40, DEFAULT, MASTER), (60, 40))
        self.assertEqual(dimensions(80, DEFAULT, MASTER), (120, 80))

    def test_halves_round_up_like_math_round(self):
        # 16.5 columns: JavaScript gives 17, Python's round() would give 16.
        self.assertEqual(dimensions(11, DEFAULT, MASTER), (17, 11))

    def test_rows_are_clamped(self):
        self.assertEqual(dimensions(3, DEFAULT, MASTER), (15, 10))
        self.assertEqual(dimensions(500, DEFAULT, MASTER), (120, 80))

    def test_multiplied_master_extends_the_fine_end(self):
        self.assertEqual(dimensions(160, DEFAULT, (240, 160)), (240, 160))
        self.assertEqual(dimensions(40, DEFAULT, (240, 160)), (60, 40))


class SmallLetterTests(unittest.TestCase):
    def test_glow_ramps_in_as_cells_shrink(self):
        self.assertEqual(glow_strength(40, .6, 32, 8), 0.0)
        self.assertEqual(glow_strength(32, .6, 32, 8), 0.0)
        self.assertAlmostEqual(glow_strength(20, .6, 32, 8), .3)
        self.assertEqual(glow_strength(8, .6, 32, 8), .6)
        self.assertEqual(glow_strength(3, .6, 32, 8), .6)
        self.assertEqual(glow_strength(3, 0, 32, 8), 0.0)

    def test_strokes_keep_thin_lines_bright(self):
        atlas = np.zeros((8, 8), np.uint8)
        atlas[:, 3] = 255                      # a one-pixel stroke
        averaged = mip_levels(atlas, 0.0)
        kept = mip_levels(atlas, 1.0)
        self.assertEqual(averaged[0].max(), 128)
        self.assertEqual(kept[0].max(), 255)
        self.assertEqual(kept[-1].shape, (1, 1))
        self.assertEqual(kept[-1][0, 0], 255)


class PictureRectTests(unittest.TestCase):
    def rect(self, area):
        display = local_ascii.LocalAsciiDisplay.__new__(local_ascii.LocalAsciiDisplay)
        display.picture_area = area
        return display.picture_rect()

    def test_unturned_picture_is_where_it_was_drawn(self):
        self.assertEqual(self.rect((100, 50, 600, 800, 800, 900, 0)), (100, 50, 600, 800, 0))

    def test_turned_pictures_map_onto_the_screen(self):
        # canvas 900 x 800 (screen 800 x 900 at a quarter turn)
        area = (100, 50, 600, 300, 900, 800, 1)
        self.assertEqual(self.rect(area), (50, 900 - 100 - 600, 300, 600, 1))
        area = (100, 50, 600, 300, 900, 800, 3)
        self.assertEqual(self.rect(area), (800 - 50 - 300, 100, 300, 600, 3))
        area = (100, 50, 600, 300, 900, 800, 2)
        self.assertEqual(self.rect(area), (900 - 100 - 600, 800 - 50 - 300, 600, 300, 2))


class DecodeScaleTests(unittest.TestCase):
    def scale(self, size, **values):
        defaults = dict(LOCAL_ASCII_DECODE_ROW_PX=4, LOCAL_ASCII_GRID=None,
                        ASCII_WEB_MAX_SIZE=(120, 80), LOCAL_ASCII_MULTIPLIER=1)
        defaults.update(values)
        with mock.patch.multiple(local_ascii.settings, create=True, **defaults):
            return decode_scale(size)

    def test_largest_reduction_that_keeps_four_pixels_a_row(self):
        # 80 rows x 4 px = 320 px high at least.
        self.assertEqual(self.scale((1920, 2160)), ((1, 4), (1, 1280)))
        self.assertEqual(self.scale((1920, 1080)), ((1, 2), (1, 640)))
        self.assertEqual(self.scale((480, 640)), ((1, 2), (1, 640)))
        self.assertEqual(self.scale((3840, 4320)), ((1, 8), (1, 2560)))

    def test_small_pictures_and_off_decode_at_full_size(self):
        self.assertEqual(self.scale((480, 600)), (None, None))
        self.assertEqual(self.scale((1920, 2160), LOCAL_ASCII_DECODE_ROW_PX=0), (None, None))
        self.assertEqual(self.scale(None), (None, None))

    def test_finer_grids_keep_more_pixels(self):
        self.assertEqual(self.scale((1920, 1080), LOCAL_ASCII_GRID=(240, 160)), (None, None))
        self.assertEqual(self.scale((1920, 2160), LOCAL_ASCII_MULTIPLIER=2), ((1, 2), (1, 1280)))


def settings(**values):
    defaults = dict(LOCAL_ASCII_ROWS=(None, None), LOCAL_ASCII_HOME_ROWS=None,
                    LOCAL_ASCII_ZOOM='random', LOCAL_ASCII_WALK_RATE=0,
                    LOCAL_ASCII_RANDOM_TRANSITION=(0, 0))
    defaults.update(values)
    return mock.patch.multiple(local_ascii.settings, create=True, **defaults)


class ZoomTests(unittest.TestCase):
    def make(self, **values):
        with settings(**values):
            return Zoom(COARSE, DEFAULT, MASTER)

    def test_range_and_home_come_from_settings(self):
        zoom = self.make(LOCAL_ASCII_ROWS=(20, 60), LOCAL_ASCII_HOME_ROWS=30)
        self.assertEqual((zoom.low, zoom.high, zoom.home), (20, 60, 30))
        self.assertEqual(zoom.grid(), dimensions(30, DEFAULT, MASTER))

    def test_unset_range_is_the_whole_master(self):
        zoom = self.make()
        self.assertEqual((zoom.low, zoom.high, zoom.home), (10, 80, 40))

    def test_manual_steps_take_over_and_stay_in_range(self):
        zoom = self.make(LOCAL_ASCII_ROWS=(20, 60))
        zoom.nudge(-100)
        self.assertEqual((zoom.driver, zoom.target), ('manual', 20))
        zoom.nudge(+100)
        self.assertEqual(zoom.target, 60)

    def test_reset_returns_home_and_to_the_random_driver(self):
        zoom = self.make()
        zoom.nudge(5)
        with settings():
            zoom.reset()
        self.assertEqual((zoom.driver, zoom.target), ('random', 40))

    def test_walk_moves_at_the_set_rate(self):
        clock = [100.0]
        with mock.patch.object(local_ascii.time, 'monotonic', lambda: clock[0]):
            zoom = self.make(LOCAL_ASCII_WALK_RATE=30, LOCAL_ASCII_ZOOM='manual')
            zoom.nudge(30)                     # target 70 rows
            clock[0] += 0.5                    # 15 rows in half a second
            self.assertTrue(zoom.update(0))
            self.assertAlmostEqual(zoom.rows, 55.0)
            clock[0] += 5
            zoom.update(0)
            self.assertEqual(zoom.rows, 70.0)

    def test_filling_the_window_keeps_its_shape_at_every_size(self):
        zoom = self.make()
        zoom.cols_for = lambda r: round(r * 269 / 80)     # a 16:10 window
        zoom.finest = (269, 80)
        for rows in (10, 25, 40, 80):
            zoom.rows = float(rows)
            cols, got = zoom.grid()
            self.assertEqual(got, rows)
            self.assertAlmostEqual(cols / rows, 269 / 80, delta=.06)

    def test_wheel_scales_by_the_same_proportion_at_every_size(self):
        zoom = self.make(LOCAL_ASCII_ZOOM='manual')
        zoom.rows = 20.0
        zoom.scale(1.5)
        self.assertEqual((zoom.driver, zoom.target), ('manual', 30.0))
        zoom.rows = 40.0
        zoom.scale(1.5)
        self.assertEqual(zoom.target, 60.0)
        zoom.scale(100)
        self.assertEqual(zoom.target, 80.0)

    def test_keys_move_one_whole_row_from_a_fractional_size(self):
        zoom = self.make(LOCAL_ASCII_ZOOM='manual')
        zoom.rows = 40.4
        zoom.nudge(-1)
        self.assertEqual(zoom.target, 40)
        zoom.nudge(1)
        self.assertEqual(zoom.target, 41)

    def test_random_takes_over_again_after_the_wheel_is_left_alone(self):
        clock = [100.0]
        with mock.patch.object(local_ascii.time, 'monotonic', lambda: clock[0]), \
                mock.patch.object(local_ascii, 'update_zoom_selection', lambda *a: None):
            with settings(LOCAL_ASCII_MANUAL_TIMEOUT=30):
                zoom = Zoom(COARSE, DEFAULT, MASTER)
                zoom.scale(1.5)
                clock[0] += 29
                zoom.update(0)
                self.assertEqual(zoom.driver, 'manual')
                zoom.nudge(1)                      # touching it again restarts the wait
                clock[0] += 29
                zoom.update(0)
                self.assertEqual(zoom.driver, 'manual')
                clock[0] += 2
                zoom.update(0)
                self.assertEqual(zoom.driver, 'random')

    def test_no_timeout_stays_manual(self):
        clock = [100.0]
        with mock.patch.object(local_ascii.time, 'monotonic', lambda: clock[0]):
            with settings(LOCAL_ASCII_MANUAL_TIMEOUT=0):
                zoom = Zoom(COARSE, DEFAULT, MASTER)
                zoom.scale(1.5)
                clock[0] += 1000
                zoom.update(0)
                self.assertEqual(zoom.driver, 'manual')

    def test_manual_setting_never_goes_random(self):
        clock = [100.0]
        with mock.patch.object(local_ascii.time, 'monotonic', lambda: clock[0]):
            with settings(LOCAL_ASCII_ZOOM='manual', LOCAL_ASCII_MANUAL_TIMEOUT=30):
                zoom = Zoom(COARSE, DEFAULT, MASTER)
                zoom.scale(1.5)
                clock[0] += 1000
                zoom.update(0)
                self.assertEqual(zoom.driver, 'manual')

    def test_zero_walk_rate_jumps(self):
        zoom = self.make(LOCAL_ASCII_ZOOM='manual')
        zoom.nudge(-30)
        zoom.update(0)
        self.assertEqual(zoom.rows, 10.0)


class RandomTransitionTests(unittest.TestCase):
    """Random changes take a random time, ease in and out, and the next pick
    waits for the change to land."""

    def setUp(self):
        self.clock = [100.0]
        self.wall = [1000.0]
        self.picks = []
        patches = [mock.patch.object(local_ascii.time, 'monotonic', lambda: self.clock[0]),
                   mock.patch.object(local_ascii.time, 'time', lambda: self.wall[0]),
                   mock.patch.object(local_ascii, 'update_zoom_selection',
                                     lambda index, zoom, state: self.picks.pop(0) if self.picks else None)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        with settings(LOCAL_ASCII_RANDOM_TRANSITION=(4.0, 4.0), LOCAL_ASCII_WALK_RATE=30):
            self.zoom = Zoom(COARSE, DEFAULT, MASTER)       # home 40 rows

    def step(self, seconds):
        self.clock[0] += seconds
        self.wall[0] += seconds
        self.zoom.update(0)

    def test_change_takes_the_drawn_time_and_eases(self):
        self.picks.append(80)
        self.step(0)
        self.step(1.0)                         # quarter way: smootherstep(.25) = .103515625
        self.assertAlmostEqual(self.zoom.rows, 40 * 2 ** .103515625)
        self.step(1.0)                         # halfway in time is halfway in ratio
        self.assertAlmostEqual(self.zoom.rows, 40 * 2 ** .5)
        self.step(2.0)
        self.assertEqual(self.zoom.rows, 80.0)

    def test_a_new_pick_mid_change_carries_on_from_where_the_size_is(self):
        self.picks.append(80)
        self.step(0)
        self.step(2.0)                         # halfway: 40 * 2 ** .5
        here = self.zoom.rows
        self.picks.append(20)
        self.step(0)
        self.assertEqual(self.zoom._move[0], here)
        self.step(4.0)
        self.assertEqual(self.zoom.rows, 20.0)

    def test_hand_control_drops_the_slow_change(self):
        self.picks.append(80)
        self.step(0)
        self.step(1.0)
        self.zoom.nudge(-1)                    # wheel: walk at the manual rate from here
        self.assertIsNone(self.zoom._move)

    def test_mid_change_the_neighbouring_grids_cross_fade(self):
        self.zoom.crossfade = True
        self.zoom.rows = 50.25
        (low, w_low), (high, w_high) = self.zoom.layers()
        self.assertEqual((low, high), (dimensions(50, DEFAULT, MASTER), dimensions(51, DEFAULT, MASTER)))
        self.assertAlmostEqual(w_low, .75)
        self.assertAlmostEqual(w_high, .25)
        self.zoom.rows = 50.0
        self.assertEqual(self.zoom.layers(), [(dimensions(50, DEFAULT, MASTER), 1.0)])

    def test_cross_fade_redraws_every_step_of_a_change(self):
        self.zoom.crossfade = True
        self.picks.append(80)
        self.step(0)
        self.clock[0] += .01                   # less than a row's worth
        self.assertTrue(self.zoom.update(0))

    def test_cross_fade_is_off_by_default(self):
        with settings():
            self.assertFalse(Zoom(COARSE, DEFAULT, MASTER).crossfade)

    def test_without_cross_fade_one_grid_is_drawn(self):
        with settings(LOCAL_ASCII_CROSSFADE=False):
            zoom = Zoom(COARSE, DEFAULT, MASTER)
        zoom.rows = 50.25
        self.assertEqual(zoom.layers(), [(dimensions(50, DEFAULT, MASTER), 1.0)])

    def test_range_is_drawn_from(self):
        with settings(LOCAL_ASCII_RANDOM_TRANSITION=(0.5, 3.0)):
            zoom = Zoom(COARSE, DEFAULT, MASTER)
        self.assertEqual(zoom.transition, (0.5, 3.0))


class RandomDriverTests(unittest.TestCase):
    def setUp(self):
        with settings(LOCAL_ASCII_ROWS=(20, 29)):
            self.zoom = Zoom(COARSE, DEFAULT, MASTER)
        self.zoom.home = 25
        self.state = {}
        self.now = [1000.0]
        self.folders = {'active_cycle': False}
        patches = [mock.patch.object(local_ascii.time, 'time', lambda: self.now[0]),
                   mock.patch('folder_selector.folder_dictionary', self.folders),
                   mock.patch('index_calculator.midi_mode', False, create=True)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def pick(self, index=0):
        return update_zoom_selection(index, self.zoom, self.state)

    def test_waits_while_the_folders_rest_at_startup(self):
        self.assertIsNone(self.pick())

    def test_the_folders_rest_does_not_pull_the_size_home(self):
        self.folders['active_cycle'] = True
        self.pick()
        self.folders['active_cycle'] = False       # the pass comes round
        self.now[0] += 30
        self.assertIsNone(self.pick())
        self.folders['active_cycle'] = True        # and leaves rest: a new size
        self.assertIn(self.pick(), range(20, 30))

    def test_leaving_rest_soon_after_a_pick_keeps_the_hold(self):
        self.zoom.hold = (6.0, 20.0)
        self.folders['active_cycle'] = True
        self.pick()
        self.folders['active_cycle'] = False
        self.now[0] += 2
        self.pick()
        self.folders['active_cycle'] = True
        self.assertIsNone(self.pick())

    def test_first_pick_when_the_folders_leave_rest(self):
        self.pick()
        self.folders['active_cycle'] = True
        first = self.pick()
        self.assertIn(first, range(20, 30))
        self.assertIsNone(self.pick())         # not again straight away

    def test_a_size_is_held_at_least_the_minimum(self):
        self.zoom.hold = (6.0, 20.0)
        self.folders['active_cycle'] = True
        self.pick()
        for index in range(400):               # every modulus slot, 5 s after
            self.now[0] = 1005.0
            self.assertIsNone(self.pick(index))

    def test_long_quiet_spell_forces_a_pick(self):
        self.folders['active_cycle'] = True
        self.pick()
        self.now[0] += 25                      # past the ~20 s cap
        self.assertIn(self.pick(), range(20, 30))

    def test_every_size_comes_up_before_any_repeats(self):
        self.folders['active_cycle'] = True
        seen = [self.pick()]
        for _ in range(9):
            self.now[0] += 25
            seen.append(self.pick())
        self.assertEqual(sorted(seen), list(range(20, 30)))

    def test_midi_clock_holds_the_size(self):
        self.folders['active_cycle'] = True
        with mock.patch('index_calculator.midi_mode', True, create=True):
            self.assertIsNone(self.pick())


if __name__ == '__main__':
    unittest.main()
