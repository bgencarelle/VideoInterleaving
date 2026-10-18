import unittest

import numpy as np

from ascii_color_sharing import ColorSharingEngine
from ascii_temporal import TemporalRunPlanner, TemporalTerminalEncoder
from tests.test_ascii_delta import picture, apply


class TemporalTests(unittest.TestCase):
    def test_current_reference_bounds_do_not_accumulate(self):
        planner = TemporalRunPlanner(tolerance=1)
        chars = np.full((3, 9), 'M', dtype='U1')
        old = np.full((3, 9, 3), 80, np.uint8)
        for value in range(80, 120):
            current = np.full_like(old, value)
            out = planner.plan(current, chars, old, chars)
            error = np.sqrt(np.sum((planner.engine.perceptual(current)-
                                    planner.engine.perceptual(out))**2, axis=2))
            self.assertLessEqual(float(error.max()), 1+1e-9)
            old = out
        self.assertGreater(int(old[0, 0, 0]), 110)

    def test_zero_tolerance_is_exact_and_new_glyph_gets_its_tint(self):
        rng = np.random.default_rng(7301)
        for width in (1, 7, 60, 160):
            chars = np.full((3, width), 'M', dtype='U1')
            old_chars = np.full_like(chars, ' ')
            old = rng.integers(0, 256, (3, width, 3), np.uint8)
            current = rng.integers(0, 256, old.shape, np.uint8)
            out = TemporalRunPlanner(tolerance=0).plan(current, chars, old, old_chars)
            np.testing.assert_array_equal(out, current)

    def test_outputs_use_only_real_current_or_previous_row_tints(self):
        rng = np.random.default_rng(8301)
        for _ in range(20):
            current = rng.integers(70, 110, (4, 13, 3), np.uint8)
            old = rng.integers(70, 110, current.shape, np.uint8)
            chars = np.full((4, 13), 'M', dtype='U1')
            out = TemporalRunPlanner(tolerance=2).plan(current, chars, old, chars)
            error = np.sqrt(np.sum((ColorSharingEngine.perceptual(current)-
                                    ColorSharingEngine.perceptual(out))**2, axis=2))
            self.assertLessEqual(float(error.max()), 2+1e-9)
            for row in range(4):
                candidates = {tuple(v) for v in np.concatenate((old[row], current[row]))}
                self.assertTrue(all(tuple(v) in candidates for v in out[row]))

    def test_adapter_reconstructs_current_glyphs_and_bounded_colours(self):
        encoder = TemporalTerminalEncoder(tolerance=1)
        screen = None
        for value in range(80, 120):
            full = picture(['M. 8M', 'M 8.M'], [[(value,90,100)]*5]*2)
            encoded = encoder.encode(full, (5,2))
            self.assertLessEqual(len(encoded), len(full))
            screen = apply(encoded, (5,2), screen)
            reference = apply(full, (5,2))
            self.assertEqual([g for g,_ in screen], [g for g,_ in reference])
            actual, _ = encoder._arrays(screen, 5, 2)
            expected, chars = encoder._arrays(reference, 5, 2)
            error = np.sqrt(np.sum((ColorSharingEngine.perceptual(actual)-
                                    ColorSharingEngine.perceptual(expected))**2, axis=2))
            self.assertLessEqual(float(error[chars!=' '].max()), 1+1e-9)

    def test_phase_switch_reset_resize_and_unsupported_fallback(self):
        encoder = TemporalTerminalEncoder()
        full = picture(['MMMM'])
        encoder.encode(full, (4,1))
        self.assertEqual(encoder.encode(full, (4,1), optimize=False), full)
        self.assertLess(len(encoder.encode(full, (4,1))), len(full))
        encoder.reset()
        self.assertEqual(encoder.encode(full, (4,1)), full)
        small = picture(['X'])
        self.assertEqual(encoder.encode(small, (1,1)), small)
        bad = b'\x1b[2J'
        self.assertEqual(encoder.encode(bad, (1,1)), bad)
        self.assertIsNone(encoder.delta.cells)
        indexed = full.replace(b'\x1b[38;2;50;90;130m', b'\x1b[38;5;100m')
        encoder.encode(indexed, (4,1))
        self.assertEqual(encoder.encode(indexed, (4,1)), indexed)

    def test_invalid_parameters_and_grids(self):
        for bad in (-1, float('nan'), float('inf'), 101):
            with self.assertRaises(ValueError):
                TemporalRunPlanner(tolerance=bad)
        planner = TemporalRunPlanner()
        rgb = np.zeros((1,2,3), np.uint8)
        chars = np.full((1,2), 'M', dtype='U1')
        for bad in (chars.astype('S1'), np.full((2,2), 'M'), np.full((1,2), '\x00')):
            with self.assertRaises(ValueError):
                planner.plan(rgb,bad,rgb,chars)
        with self.assertRaises(ValueError):
            planner.plan(rgb,chars,rgb[:,:1],chars)


if __name__ == '__main__':
    unittest.main()
