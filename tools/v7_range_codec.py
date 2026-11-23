"""Opt-in legal-luma-range reconstruction adapter for nested V7 experiments.

This changes receiver-side reconstruction only. It preserves the compiled
range kernels and the carried-coefficient/host-cell constraints.
"""
import json
from pathlib import Path

import numpy as np


def install(out=None, passes=0, host_cells=False):
    """Install the receiver adapter; ``passes`` must be positive when enabled."""
    passes = int(passes)
    host_cells = bool(host_cells)
    if passes <= 0:
        raise ValueError('range reconstruction requires positive passes')
    import nested_fold as nf
    from tools import v7_sk_fold as sk

    config = (passes, host_cells)
    previous = getattr(nf, '_range_codec_configuration', None)
    if previous is not None:
        if previous != config:
            raise ValueError('range reconstruction is already installed with different parameters')
        return nf
    if nf._cache:
        raise RuntimeError('install range reconstruction before any target fold is cached')

    matrix = np.eye(2, dtype=np.float64)
    mask = np.zeros(4, dtype=np.bool_)
    mask.setflags(write=False)
    sk.range_consistent_luma(np.zeros(4), mask, matrix, matrix, matrix, matrix, 1.0, 1,
                             np.zeros(4) if host_cells else None)
    if host_cells:
        sk.range_host_rooms(4, np.array([0], dtype=np.int64),
                            np.ones(1, dtype=np.int64), np.ones(1), np.ones(1), 1.0)

    original_decode = nf.FrozenFold.decode
    if host_cells:
        def decode(self, readings, index, *args, **kwargs):
            self._range_factor = nf.RATIO ** int(index)
            return original_decode(self, readings, index, *args, **kwargs)
        nf.FrozenFold.decode = decode

    def clean(self, luma, room, passes=None):
        if room is None:
            return luma
        bound = 1.0 if self.grid is None else self.grid.scales[0]
        reach = (sk.range_host_rooms(self.luma, self.host_position, self.level,
                                     self.step, self.host_sd, self._range_factor)
                 if host_cells else None)
        return sk.range_consistent_luma(
            luma, self.sent_mask, *self._transforms, bound,
            passes if passes is not None else config[0], reach)

    if out is not None:
        original_init = nf.FrozenFold.__init__
        metadata_path = Path(out) / 'range-reconstruction.json'

        def init(self, arrays):
            original_init(self, arrays)
            data = json.loads(metadata_path.read_text(encoding='utf-8'))
            data.setdefault('actual_bounds', {})[
                f"{'slices' if self.paired else 'mono'}:{self.rows}x{self.cols}"] = (
                    1.0 if self.grid is None else float(self.grid.scales[0]))
            metadata_path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')

        nf.FrozenFold.__init__ = init
    nf.FrozenFold.clean = clean
    nf.SMOOTH_PASSES = passes
    nf._range_codec_configuration = config
    if out is not None:
        path = Path(out)
        path.mkdir(parents=True, exist_ok=True)
        (path / 'range-reconstruction.json').write_text(json.dumps({
            'method': 'legal-luma-range projected descent',
            'passes': passes,
            'host_cells': host_cells,
            'carried_coefficients': ('hosts within continuous quantization cells; other carried fixed'
                                     if host_cells else 'all fixed exactly'),
            'source_access': False,
            'white_region_mask': False,
            'smoothness_penalty': False,
            'canonical_bound': 'FoldGrid.scales[0], or 1 on legacy grid',
            'actual_bound': 'FoldGrid.scales[0] when available, otherwise 1',
            'host_cell_factor': 'RATIO**decoded host ladder level, not adaptive pilot row or source index',
        }, indent=2) + '\n', encoding='utf-8')
    return nf
