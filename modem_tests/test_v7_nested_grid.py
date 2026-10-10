"""Grid normalization and coefficient-addressing checks, not picture scoring."""
import unittest
import tempfile
from pathlib import Path

import numpy as np

from tools import v7_live
v7_live._ensure_test_modem_path()
from nested_grid import FoldGrid
import nested_fold


class NestedGridTests(unittest.TestCase):
    def test_smaller_grid_discards_outside_frequencies_without_wrapping_rows(self):
        grid = FoldGrid((72, 60))
        old = np.zeros(grid.old_size)
        old[0] = .3*np.sqrt(7680)
        old[59] = 2.0
        old[60] = 9.0  # must not wrap into the next row of the narrower grid
        old[72*80] = 7.0
        old[7680] = -.2*np.sqrt(1920)
        contracted = grid.contract(grid.expand(old))
        self.assertEqual(contracted[59], 2.0)
        self.assertEqual(contracted[60], 0.0)
        self.assertEqual(contracted[72*80], 0.0)
        self.assertEqual(contracted[7680], old[7680])
        flat = np.zeros(grid.size)
        flat[0] = old[0]
        np.testing.assert_allclose(grid.inverse(flat)[:4320], .3, atol=1e-12)

    def test_uniform_picture_keeps_legacy_coefficient_units(self):
        grid = FoldGrid((120, 100))
        values = np.ones(grid.size)
        coefficients = grid.forward(values)
        self.assertAlmostEqual(coefficients[0], np.sqrt(96*80), places=11)
        self.assertAlmostEqual(coefficients[12000], np.sqrt(48*40), places=11)
        np.testing.assert_allclose(grid.inverse(coefficients), values, atol=1e-12)

    def test_legacy_coefficients_embed_without_changing_brightness(self):
        grid = FoldGrid((120, 100))
        old = np.zeros(grid.old_size)
        old[0] = .3*np.sqrt(96*80)
        old[7680] = -.2*np.sqrt(48*40)
        old[7680+1920] = .1*np.sqrt(48*40)
        expanded = grid.expand(old)
        np.testing.assert_array_equal(grid.contract(expanded), old)
        pixels = grid.inverse(expanded)
        np.testing.assert_allclose(pixels[:12000], .3, atol=1e-12)
        np.testing.assert_allclose(pixels[12000:13920], -.2, atol=1e-12)

    def test_new_horizontal_frequency_survives_the_enlarged_grid(self):
        grid = FoldGrid((120, 100))
        coefficients = np.zeros(grid.size)
        coefficients[90] = 1.0  # outside the old grid's 80 columns
        decoded = grid.forward(grid.inverse(coefficients))
        np.testing.assert_allclose(decoded, coefficients, atol=1e-12)
        np.testing.assert_allclose(grid.contract(decoded), 0, atol=1e-12)

    def test_enlarged_chroma_maps_legacy_wire_positions_without_budget_growth(self):
        grid = FoldGrid((120, 100), (60, 50))
        self.assertEqual(grid.grids, ((120, 100), (60, 50), (60, 50)))
        self.assertEqual(grid.size, 12000 + 2*3000)
        self.assertEqual(grid.old_size, 7680 + 2*1920)
        old = np.zeros(grid.old_size)
        old[7680] = .25*np.sqrt(1920)
        old[7680+1920] = -.5*np.sqrt(1920)
        expanded = grid.expand(old)
        decoded = grid.inverse(expanded)
        np.testing.assert_allclose(decoded[:12000], 0, atol=1e-12)
        np.testing.assert_allclose(decoded[12000:15000], .25, atol=1e-12)
        np.testing.assert_allclose(decoded[15000:], -.5, atol=1e-12)
        np.testing.assert_array_equal(grid.contract(expanded), old)

    def test_compiled_plane_contraction_matches_coordinate_map(self):
        grid = FoldGrid((120, 100), (60, 50))
        planes = [np.arange(rows*cols, dtype=np.float64)
                  for rows, cols in grid.grids]
        expected = grid.contract(np.concatenate(planes))
        actual = nested_fold.sk.contract_grid_planes(
            *planes, grid.positions, grid.old_off, grid.off, grid.old_size)
        np.testing.assert_array_equal(actual, expected)

    def test_enlarged_fold_uses_full_grid_for_decode_and_bounds(self):
        stock = nested_fold.table('mono', '3:4')
        grid = FoldGrid((120, 100))
        arrays = {name: getattr(stock, name).copy() for name in stock.FIELDS}
        arrays['host_position'] = grid.positions[arrays['host_position']]
        arrays['guest_position'] = grid.positions[arrays['guest_position']]
        arrays['luma_grid'] = np.array([120, 100])
        fold = nested_fold.FrozenFold(arrays)
        sigma = np.full(fold.slots, .01)
        decoded = fold.decode({0: (np.zeros(fold.slots), sigma)}, 0)
        self.assertEqual(decoded.shape, (12000,))
        self.assertEqual(fold.sent_mask.shape, (12000,))
        self.assertEqual(fold.room(0, sigma).shape, (12000,))

    def test_chroma_grid_metadata_reaches_nested_fold_setup(self):
        arrays = {name: getattr(nested_fold.table('mono', '3:4'), name).copy()
                  for name in nested_fold.FrozenFold.FIELDS}
        arrays['luma_grid'] = np.asarray((120, 100))
        arrays['chroma_grid'] = np.asarray((60, 50))
        fold = nested_fold.FrozenFold(arrays)
        self.assertEqual(fold.grids, ((120, 100), (60, 50), (60, 50)))
        self.assertEqual(fold.grid.size, 12000+2*3000)

    def test_table_loader_preserves_chroma_grid_metadata(self):
        original = nested_fold.table('mono', '3:4')
        arrays = {f'mono/3:4/{name}': getattr(original, name)
                  for name in original.FIELDS}
        arrays['mono/3:4/luma_grid'] = np.asarray((120, 100), np.int64)
        arrays['mono/3:4/chroma_grid'] = np.asarray((60, 50), np.int64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'tables.npz'
            np.savez(path, **arrays)
            loaded = nested_fold.table('mono', '3:4', path=path)
        self.assertEqual(loaded.grids, ((120, 100), (60, 50), (60, 50)))


if __name__ == '__main__':
    unittest.main()
