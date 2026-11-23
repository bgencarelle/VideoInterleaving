"""Experimental nested image grids; wire coefficients retain legacy units."""
import math

import numpy as np

from tools import v7_sk_fold as sk

LEGACY = ((96, 80), (48, 40), (48, 40))


class FoldGrid:
    def __init__(self, shape, chroma_shape=LEGACY[1]):
        """Map frozen legacy coefficient planes onto an experimental grid.

        Luma and chroma extents are independent; no coefficient slots are
        added to the packet by enlarging either transform.
        """
        self.grids = (tuple(map(int, shape)), tuple(map(int, chroma_shape)),
                      tuple(map(int, chroma_shape)))
        self.off = np.cumsum([0]+[r*c for r, c in self.grids])
        self.old_off = np.cumsum([0]+[r*c for r, c in LEGACY])
        self.forward_matrices, self.inverse_matrices = [], []
        self.scales = []
        positions, old_start, new_start = [], 0, 0
        for old, new in zip(LEGACY, self.grids):
            rows, cols = new
            if rows <= 0 or cols <= 0:
                raise ValueError('grid dimensions must be positive')
            left, right = sk.dct_matrix(rows), sk.dct_matrix(cols)
            self.forward_matrices.append((left, np.ascontiguousarray(right.T)))
            self.inverse_matrices.append((np.ascontiguousarray(left.T), right))
            self.scales.append(math.sqrt(old[0]*old[1]/(rows*cols)))
            u, v = np.mgrid[:old[0], :old[1]]
            mapped = new_start+u*cols+v
            mapped[(u >= rows) | (v >= cols)] = -1
            positions.append(mapped.ravel())
            old_start += old[0]*old[1]
            new_start += rows*cols
        self.positions = np.ascontiguousarray(np.concatenate(positions), np.int64)
        self.old_size, self.size = old_start, new_start
        self.luma_size = self.grids[0][0]*self.grids[0][1]

    def forward(self, values):
        out, start = np.empty(self.size), 0
        for matrices, scale in zip(self.forward_matrices, self.scales):
            start = sk.grid_plane(values, out, start, *matrices, scale)
        return out

    def inverse(self, values):
        out, start = np.empty(self.size), 0
        for matrices, scale in zip(self.inverse_matrices, self.scales):
            start = sk.grid_plane(values, out, start, *matrices, 1.0/scale)
        return out

    def expand(self, values):
        return sk.remap_grid(values, self.positions, self.size, True)

    def contract(self, values):
        return sk.remap_grid(values, self.positions, self.old_size, False)
