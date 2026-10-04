"""Regression checks for the kernel benchmark's display reconstruction."""
import unittest

import numpy as np

from animation_modem.v7_source_dct import direct_dct_values
from tools import v7_kernel_bench as bench
from tools.v7_kernel_defaults_sweep import (AspectMonoBenchmark,
                                            AspectStereoBenchmark)


class KernelBenchDisplayTests(unittest.TestCase):
    def test_bilinear_reconstruction_uses_each_colour_plane(self):
        rgb = np.empty((240, 320, 3), dtype=np.uint8)
        rgb[:] = (220, 40, 20)
        masks = [np.ones(grid, dtype=bool) for grid in bench.GRIDS]
        values = direct_dct_values(rgb, bench.GRIDS, bench.SHAPES,
                                   kernel_masks=masks)
        previous = bench.DISPLAY
        try:
            for display in ('ideal', 'bilinear'):
                bench.DISPLAY = display
                shown = bench.shown_rgb(values, masks)
                expected = np.broadcast_to((220, 40, 20), shown.shape)
                np.testing.assert_allclose(shown, expected, atol=1)
        finally:
            bench.DISPLAY = previous


class AspectMonoBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = AspectMonoBenchmark()

    def test_each_aspect_uses_its_layout_masks_and_fold_guests(self):
        self.assertEqual(len(self.profile.layouts), 7)
        for size, expected in (((854, 480), '16:9'),
                               ((480, 854), '9:16'),
                               ((512, 512), '1:1')):
            layout, masks, ratio = self.profile.masks_for_size(size)
            self.assertEqual(layout, expected)
            expected_ratio = {
                '16:9': 16/9, '9:16': 9/16, '1:1': 1.0,
            }[expected]
            self.assertAlmostEqual(ratio, expected_ratio)
            self.assertEqual([mask.shape for mask in masks], list(bench.GRIDS))
            # 1,264 fresh coefficients plus the 500 luma fold guests.
            self.assertEqual(sum(int(mask.sum()) for mask in masks), 1764)
        self.assertFalse(np.array_equal(
            self.profile.layouts['1:1'][0], self.profile.layouts['16:9'][0]))


class AspectStereoBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = AspectStereoBenchmark()

    def test_each_aspect_uses_fixed_tail_masks_and_fold_guests(self):
        self.assertEqual(self.profile.wire.tail, 'fixed')
        self.assertEqual(len(self.profile.layouts), 7)
        for size, expected in (((854, 480), '16:9'),
                               ((480, 854), '9:16'),
                               ((512, 512), '1:1')):
            layout, masks, ratio = self.profile.masks_for_size(size)
            self.assertEqual(layout, expected)
            self.assertAlmostEqual(ratio, {
                '16:9': 16/9, '9:16': 9/16, '1:1': 1.0,
            }[expected])
            self.assertEqual([mask.shape for mask in masks], list(bench.GRIDS))
            # The fixed tail carries 2,320 kept coefficients; the 500 luma
            # guests complete the Aspect Fold allocation.
            self.assertEqual(sum(int(mask.sum()) for mask in masks), 2820)
            self.assertEqual(int(masks[0].sum()), 2420)
        self.assertFalse(np.array_equal(
            self.profile.layouts['1:1'][0], self.profile.layouts['16:9'][0]))


if __name__ == '__main__':
    unittest.main()
