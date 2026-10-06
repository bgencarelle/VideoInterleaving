"""Time-parameterized sampling for canonical baked vector paths.

The emitter deliberately owns only phase. Call :meth:`accept` after the
surrounding output transaction accepts the returned candidate; retries then
reproduce the same waveform rather than advancing the path.
"""
import math

import numpy as np


class VectorTrajectory:
    """Sample a closed vector path independently of DAC trace length."""

    def __init__(self, geometry_samples=None, traversal_hz=None):
        if geometry_samples is not None and int(geometry_samples) < 2:
            raise ValueError("geometry_samples must be at least 2")
        if traversal_hz is not None and (
                not math.isfinite(float(traversal_hz))
                or float(traversal_hz) <= 0):
            raise ValueError("traversal_hz must be finite and positive")
        self.geometry_samples = (None if geometry_samples is None
                                 else int(geometry_samples))
        self.traversal_hz = (None if traversal_hz is None
                             else float(traversal_hz))
        self.phase = 0.0
        self._candidate = None
        from scope_bake import _trajectory_sample_kernel
        _trajectory_sample_kernel(np.zeros((2, 2), dtype=np.float32),
                                  0.0, 0.5, 2)

    def emit(self, path, count, samplerate):
        """Return one candidate waveform, without changing committed phase."""
        from scope_bake import _trajectory_sample_kernel

        count, samplerate = int(count), float(samplerate)
        if count < 1 or not math.isfinite(samplerate) or samplerate <= 0:
            raise ValueError("count and samplerate must be positive")
        points = np.ascontiguousarray(path, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError("canonical vector path must contain XY points")
        if not np.isfinite(points).all():
            raise ValueError("canonical vector path must be finite")
        if not points.flags.writeable:
            points = points.copy()
        phase = self.phase
        step = (1.0 / count if self.traversal_hz is None else
                self.traversal_hz / samplerate)
        if self.geometry_samples is not None:
            # Canonical path geometry is resampled to this fixed budget before
            # time traversal; output count/rate cannot alter its detail.
            from scope_bake import _trajectory_sample_kernel
            points = _trajectory_sample_kernel(
                points, 0.0, 1.0 / self.geometry_samples, self.geometry_samples)
        candidate = _trajectory_sample_kernel(points, phase, step, count)
        self._candidate = (phase + step * count) % 1.0
        return candidate

    def accept(self):
        """Commit the most recently emitted candidate after queue acceptance."""
        if self._candidate is None:
            return False
        self.phase = self._candidate
        self._candidate = None
        return True

    def reject(self):
        """Discard a failed/retried candidate without advancing the clock."""
        self._candidate = None

    def reset(self):
        self.phase = 0.0
        self._candidate = None
