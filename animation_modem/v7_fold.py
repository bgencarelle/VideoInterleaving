"""Pinned sender-side stereo Fold 500 profile for the V7 wire."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.fft import dctn

from animation_modem import v7


TABLE_SHA256 = '1b3ef13f3457a4e4c02a92e149b3bb0eac3c0e233ed1f18eefb84459e24d8ded'
U_CLIP = 2.5
SIGNATURE_STEPS = 3


def _table_text(table):
    return json.dumps(table, sort_keys=True, separators=(',', ':')) + '\n'


def _model_digest(model):
    digest = hashlib.sha256(f'encoding {int(model.encoding_type)}'.encode())
    for array in (model.mu, model.lam, model.gain):
        digest.update(np.ascontiguousarray(array, dtype='<f8').tobytes())
    digest.update(np.ascontiguousarray(model.order, dtype='<i8').tobytes())
    return digest.hexdigest()


def _corner_positions():
    offsets = np.cumsum([0] + [rows*cols for rows, cols in v7.V7_GRIDS])
    return np.concatenate([
        offsets[plane] +
        (np.arange(rows)[:, None]*grid_cols +
         np.arange(cols)[None, :]).ravel()
        for plane, ((rows, cols), (grid_rows, grid_cols)) in enumerate(
            zip(v7.V7_SHAPES, v7.V7_GRIDS))])


def _full_grid_dct(values):
    values = np.asarray(values, dtype=float)
    expected = sum(rows*cols for rows, cols in v7.V7_GRIDS)
    if values.shape != (expected,) or not np.all(np.isfinite(values)):
        raise ValueError(f'Fold 500 expects {expected} finite source values')
    offsets = np.cumsum([0] + [rows*cols for rows, cols in v7.V7_GRIDS])
    return np.concatenate([
        dctn(values[start:end].reshape(shape), norm='ortho').ravel()
        for start, end, shape in zip(offsets, offsets[1:], v7.V7_GRIDS)])


class Fold500:
    """Fold the canonical Box-model source into the pinned stereo M=500 wire."""

    def __init__(self, model):
        path = Path(__file__).with_name('v7_fold_table_500.json')
        blob = path.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        if digest != TABLE_SHA256:
            raise ValueError('Fold 500 table hash does not match the pinned wire table')
        table = json.loads(blob)
        if _table_text(table).encode() != blob:
            raise ValueError('Fold 500 table is not in canonical form')
        if table.get('format') != 'v7-fold-2' or table.get('M') != 500:
            raise ValueError('Unsupported Fold 500 table format')
        if table.get('model_digest') != _model_digest(model):
            raise ValueError('Fold 500 requires the canonical Box encoding model')

        self.model = model
        self.table = table
        self.identity = hashlib.sha256(_table_text(table).encode()).hexdigest()
        self.hosts = np.asarray(table['hosts'], dtype=int)
        self.guests = np.asarray(table['guests'], dtype=int)
        self.kept = _corner_positions()
        self.sd_host = np.sqrt(model.lam[self.hosts])
        self.sd_guest = np.asarray(table['sd_guest'], dtype=float)
        # Guests add support outside the ordinary sent corners; the reserved
        # signature carriers are already among the kept host positions.
        self.source_positions = np.unique(
            np.concatenate((self.kept, self.guests))).astype(np.int64)
        self.D = float(table['D'])
        self.beta = .8*self.D/(2*U_CLIP)
        self.power = 1 + self.D**2/12 + self.beta**2
        self.signature = int(table['signature'])
        rng = np.random.default_rng(int(self.identity[:16], 16))
        self.pattern = rng.choice([-1.0, 1.0], self.signature)

    def encode_coefficients(self, values):
        """Return V7 corner coefficients with the guest luma slots folded in."""
        return self.encode_dct_coefficients(_full_grid_dct(values))

    def encode_dct_coefficients(self, full):
        """Fold a precomputed full-grid V7 DCT vector into transmitted slots.

        Source-domain encoders can map their DCT directly onto the sampling-grid
        basis and enter the exact production fold here, without synthesizing a
        pixel grid only to transform it again.
        """
        full = np.asarray(full, dtype=float)
        expected = sum(rows*cols for rows, cols in v7.V7_GRIDS)
        if full.shape != (expected,) or not np.all(np.isfinite(full)):
            raise ValueError(
                f'Fold 500 expects {expected} finite full-grid DCT values')
        coeffs = full[self.kept].copy()
        host = (coeffs[self.hosts] - self.model.mu[self.hosts])/self.sd_host
        guest = np.clip(full[self.guests]/self.sd_guest, -U_CLIP, U_CLIP)
        symbol = self.D*np.round(host/self.D) + self.beta*guest
        if self.signature:
            symbol[-self.signature:] = (
                SIGNATURE_STEPS*self.D*self.pattern)
        coeffs[self.hosts] = (
            self.model.mu[self.hosts] + self.sd_host*symbol/np.sqrt(self.power))
        return coeffs
