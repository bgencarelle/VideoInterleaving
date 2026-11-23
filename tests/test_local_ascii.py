"""Local ASCII display: size mapping, zoom control and the random driver.

No window or GL: drawing is checked by running the app (see the commit).
"""
import unittest
from unittest import mock

import local_ascii
from local_ascii import Zoom, dimensions, update_zoom_selection

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


def settings(**values):
    defaults = dict(LOCAL_ASCII_ROWS=(None, None), LOCAL_ASCII_HOME_ROWS=None,
                    LOCAL_ASCII_ZOOM='random', LOCAL_ASCII_WALK_RATE=0)
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

    def test_zero_walk_rate_jumps(self):
        zoom = self.make(LOCAL_ASCII_ZOOM='manual')
        zoom.nudge(-30)
        zoom.update(0)
        self.assertEqual(zoom.rows, 10.0)


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

    def test_rests_at_home_while_the_folders_rest(self):
        self.assertEqual(self.pick(), 25)

    def test_first_pick_when_the_folders_leave_rest(self):
        self.pick()
        self.folders['active_cycle'] = True
        first = self.pick()
        self.assertIn(first, range(20, 30))
        self.assertIsNone(self.pick())         # not again straight away

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
