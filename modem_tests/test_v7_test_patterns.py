"""Fine-line / aliasing / moire pattern suite: geometry and frequency checks."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools import v7_test_patterns as patterns


def peak_cycles(column):
    """Dominant frequency of a 1-D profile, in cycles over its length."""
    spectrum = abs(np.fft.rfft(column-column.mean()))
    return int(np.argmax(spectrum))


class PatternTests(unittest.TestCase):
    def test_names_are_unique_and_every_pattern_renders_in_range(self):
        names = [entry[0] for entry in patterns.SUITE]
        self.assertEqual(len(names), len(set(names)))
        for name, kind, parameters, note in patterns.SUITE:
            plane = patterns.render(kind, 96, 72, *map(float, parameters), 2)
            self.assertEqual(plane.shape, (96, 72), name)
            self.assertTrue(np.all((plane >= 0) & (plane <= 1)), name)
            self.assertGreater(plane.std(), .01, name)
            self.assertTrue(note)

    def test_rendering_is_deterministic(self):
        a = patterns.image('zone-plate-64')
        b = patterns.image('zone-plate-64')
        np.testing.assert_array_equal(a, b)

    def test_frequencies_are_cycles_per_picture_height(self):
        rings = patterns.image('rings-40-sine')
        height, width = rings.shape
        # Down the centre column the ring pitch is 1/40 of the height.
        self.assertEqual(peak_cycles(rings[:, width//2]), 40)
        slant = patterns.image('slant-24-at-45deg')
        self.assertAlmostEqual(peak_cycles(slant[:, width//2]), 24*np.cos(np.pi/4), delta=1)
        bars = patterns.image('bars-horizontal-lines-16-48')
        self.assertEqual(peak_cycles(bars[:, 5]), 16)
        self.assertEqual(peak_cycles(bars[:, width-5]), 48)
        vertical = patterns.image('bars-vertical-lines-16-48')
        # 16 c/ph across a 3:4 picture is 12 cycles over the width.
        self.assertEqual(peak_cycles(vertical[5]), 12)

    def test_zone_plate_reaches_its_stated_frequency_at_half_height(self):
        plate = patterns.image('zone-plate-64')
        height, width = plate.shape
        column = plate[:, width//2]
        top = column[:height//16]                      # outermost sixteenth
        crossings = np.sum(np.diff(np.sign(top-.5)) != 0)
        # Local frequency there is 64*(1-1/16) on average: about 60 c/ph.
        self.assertAlmostEqual(crossings/2*16, 60, delta=6)

    def test_supersampling_removes_rendering_aliases(self):
        # 200 c/ph is far beyond a 96-row raster: a point-sampled render
        # shows a strong false pattern, the box-filtered one is nearly flat.
        point = patterns.render(patterns.SLANT, 96, 72, 200., 0., 0., 0., 1)
        filtered = patterns.render(patterns.SLANT, 96, 72, 200., 0., 0., 0., 16)
        self.assertGreater(point.std(), 4*filtered.std())

    def test_suite_writes_files_manifest_and_contact_sheet(self):
        with tempfile.TemporaryDirectory() as folder:
            original = dict(patterns.LAYOUTS)
            patterns.LAYOUTS['tiny'] = (60, 80)
            try:
                manifest = patterns.write_suite(folder, 'tiny', 1)
            finally:
                patterns.LAYOUTS.clear()
                patterns.LAYOUTS.update(original)
            files = sorted(Path(folder).glob('*.png'))
            self.assertEqual(len(files), len(patterns.SUITE)+1)
            self.assertEqual(len(manifest['patterns']), len(patterns.SUITE))
            self.assertTrue((Path(folder)/'MANIFEST.json').is_file())


if __name__ == '__main__':
    unittest.main()
