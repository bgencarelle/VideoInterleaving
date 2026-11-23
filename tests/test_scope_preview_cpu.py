import unittest

import cv2
import numpy as np

from scope_bake import PreviewWorkspace, preview_frame


def _legacy_preview(samples, size, spot, exposure=1.0, max_split=192):
    samples = np.asarray(samples, dtype=np.float32)
    px = (samples[:, 0] + 1.0) * 0.5 * (size - 1)
    py = (1.0 - samples[:, 1]) * 0.5 * (size - 1)
    x0, x1, y0, y1 = px[:-1], px[1:], py[:-1], py[1:]
    lengths = np.hypot(x1 - x0, y1 - y0)
    pieces = np.clip(np.ceil(lengths), 1, max_split).astype(np.int64)
    segment = np.repeat(np.arange(len(pieces)), pieces)
    starts = np.concatenate(([0], np.cumsum(pieces)[:-1]))
    local = np.arange(int(pieces.sum())) - np.repeat(starts, pieces)
    t = (local + 0.5) / pieces[segment]
    xs = np.clip(x0[segment] + (x1[segment] - x0[segment]) * t,
                 0, size - 1).astype(np.int32)
    ys = np.clip(y0[segment] + (y1[segment] - y0[segment]) * t,
                 0, size - 1).astype(np.int32)
    weights = (1.0 / pieces[segment]).astype(np.float32)
    acc = np.bincount(ys * size + xs, weights=weights,
                      minlength=size * size).reshape(size, size)
    acc = cv2.GaussianBlur(acc.astype(np.float32), (0, 0), spot)
    lit = acc[acc > 0]
    gain = exposure * 2.5 / max(float(np.percentile(lit, 75)), 1e-6)
    value = 1.0 - np.exp(-acc * gain)
    rgb = np.stack((value * 0.35, value, value * 0.25), axis=-1) + 0.03
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


class ScopePreviewCpuTests(unittest.TestCase):
    def setUp(self):
        t = np.linspace(0.0, 1.0, 513, dtype=np.float32)
        self.points = np.column_stack((
            0.88 * np.sin(t * 21.0), 0.78 * np.cos(t * 13.0)))

    def test_compiled_reusable_workspace_matches_legacy_preview_pixels(self):
        size = 96
        rows = max(2, len(np.unique(np.round(self.points[:, 1], 5))))
        spot = max(0.6, 0.40 * size / rows)
        expected = _legacy_preview(self.points, size, spot, exposure=1.3)
        workspace = PreviewWorkspace(size)
        rgb_id = id(workspace.rgb)
        for _ in range(2):
            actual = preview_frame(self.points, size=size, spot=spot,
                                   exposure=1.3, workspace=workspace)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(id(actual), rgb_id)

    def test_auto_spot_exposure_and_workspace_reuse(self):
        workspace = PreviewWorkspace(80)
        base = preview_frame(self.points, size=80, workspace=workspace).copy()
        brighter = preview_frame(self.points, size=80, exposure=2.0,
                                 workspace=workspace)
        self.assertIs(brighter, workspace.rgb)
        self.assertGreater(float(brighter.mean()), float(base.mean()))

    def test_large_preview_uses_bounded_half_scale_filter_with_small_rgb_delta(self):
        size = 1024
        # Wide raster-like spots benefit from lower-resolution filtering; fine
        # spots stay full-resolution to avoid losing thin vector features.
        spot = 12.0
        expected = _legacy_preview(self.points, size, spot)
        actual = preview_frame(self.points, size=size, spot=spot)
        delta = np.abs(expected.astype(np.int16) - actual.astype(np.int16))
        self.assertLess(float(delta.mean()), 0.5)
        self.assertLessEqual(float(np.percentile(delta, 95)), 2.0)

    def test_large_preview_keeps_fine_spots_full_resolution(self):
        size = 1024
        spot = 2.3
        expected = _legacy_preview(self.points, size, spot)
        actual = preview_frame(self.points, size=size, spot=spot)
        delta = np.abs(expected.astype(np.int16) - actual.astype(np.int16))
        self.assertLessEqual(int(delta.max()), 1)
        self.assertLessEqual(float(np.count_nonzero(delta)) / delta.size,
                             1e-5)

    def test_empty_and_invalid_workspace_inputs(self):
        self.assertEqual(preview_frame(np.empty((0, 2)), size=12).shape,
                         (12, 12, 3))
        with self.assertRaisesRegex(ValueError, "workspace size"):
            preview_frame(self.points, size=16, workspace=PreviewWorkspace(17))


if __name__ == "__main__":
    unittest.main()
