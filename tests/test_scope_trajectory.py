import unittest
from unittest import mock

import numpy as np

import scope_bake
from scope_bake import TraceEmitter


class TraceTrajectoryTests(unittest.TestCase):
    def setUp(self):
        y, x = np.mgrid[:32, :48]
        self.lum = (((x > 5) & (x < 42) & (y > 4) & (y < 28))
                    * (0.35 + 0.65 * ((x + y) % 11) / 10)).astype(np.float32)

    def emitter(self, rate=48000, samples=800, **kwargs):
        return TraceEmitter(rate, samples, grid=(16, 24), levels=(0, 1),
                            sweep="alternate", **kwargs)

    def test_full_physical_sample_sequences_match_across_sample_rates(self):
        a = self.emitter(traversal_hz=2.0, geometry_samples=512)
        b = self.emitter(rate=96000, samples=1600, traversal_hz=2.0,
                         geometry_samples=512)
        samples_a = np.concatenate([a.emit(self.lum), a.emit(self.lum)])
        samples_b = np.concatenate([b.emit(self.lum), b.emit(self.lum)])
        np.testing.assert_allclose(samples_a, samples_b[::2], atol=2e-6)

    def test_chunk_partition_does_not_change_trajectory(self):
        whole = self.emitter(traversal_hz=1.5, geometry_samples=600)
        chunks = self.emitter(samples=400, traversal_hz=1.5,
                              geometry_samples=600)
        expected = whole.emit(self.lum)
        actual = np.concatenate([chunks.emit(self.lum), chunks.emit(self.lum)])
        np.testing.assert_allclose(actual, expected, atol=2e-6)

    def test_candidate_rejection_does_not_advance_phase_or_endpoint(self):
        emitter = self.emitter(traversal_hz=1.25, geometry_samples=400)
        baseline = self.emitter(traversal_hz=1.25, geometry_samples=400)
        candidate = emitter.emit(self.lum, commit=False)
        accepted_baseline = baseline.emit(self.lum)
        np.testing.assert_array_equal(candidate, accepted_baseline)
        self.assertEqual(emitter._phase, 0.0)
        self.assertIsNone(emitter._end)
        accepted = emitter.emit(self.lum)
        np.testing.assert_array_equal(accepted, accepted_baseline)
        self.assertAlmostEqual(emitter._phase, 1.25 * 800 / 48000)
        self.assertIsNotNone(emitter._end)
        emitter.emit(self.lum, commit=False)
        emitter.reset()
        self.assertEqual(emitter._phase, 0.0)
        self.assertIsNone(emitter._candidate_phase)
        self.assertIsNone(emitter._end)

    def test_geometry_budget_is_independent_of_output_budget(self):
        emitter = self.emitter(samples=800, geometry_samples=333,
                               traversal_hz=1.0)
        original = scope_bake.render_luma
        with mock.patch("scope_bake.render_luma", wraps=original) as render:
            frame = emitter.emit(self.lum, commit=False)
        self.assertEqual(render.call_args.args[1], 333)
        self.assertIsNone(render.call_args.kwargs["prepared_grid"])
        self.assertEqual(frame.shape, (800, 2))

    def test_time_traversal_rejects_unsupported_modes(self):
        with self.assertRaisesRegex(ValueError, "interlace"):
            self.emitter(fields=2, traversal_hz=1.0)
        with self.assertRaisesRegex(ValueError, "fixed Y-T"):
            self.emitter(yt_timing="fixed", traversal_hz=1.0)
        with self.assertRaisesRegex(ValueError, "positive"):
            self.emitter(traversal_hz=0)
        for bad_rate in (float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "finite"):
                self.emitter(traversal_hz=bad_rate)
        with self.assertRaisesRegex(ValueError, "samplerate"):
            TraceEmitter(float("inf"), 800, traversal_hz=1.0)
        with self.assertRaisesRegex(ValueError, "fixed Y-T"):
            self.emitter(yt_timing="fixed", geometry_samples=3200)
        with self.assertRaisesRegex(ValueError, "close_frame"):
            self.emitter(traversal_hz=1.0, close_frame=True)

    def test_geometry_only_sampler_is_prewarmed(self):
        emitter = self.emitter(geometry_samples=333)
        self.assertGreaterEqual(len(scope_bake._trajectory_sample_kernel.signatures), 1)
        self.assertEqual(emitter.geometry_samples, 333)

    def test_legacy_default_path_is_unchanged(self):
        emitter = self.emitter()
        reference = scope_bake.render_luma(
            self.lum, 800, gamma=emitter.gamma, trim=emitter.trim,
            density=emitter.density, rows=emitter.rows,
            autofit=emitter.autofit, oversample=emitter.oversample,
            border=emitter.border, row_bias=emitter.row_bias,
            precondition=emitter.precondition, fields=emitter.fields,
            field=0, levels=emitter.levels, palindrome=False, reverse=False,
            start=None, close=False, loop_anchor=False, yt_fixed=False,
            yt_trigger_samples=0, prepared_grid=None, grid_rows=16,
            grid_cols=24)
        np.testing.assert_array_equal(emitter.emit(self.lum), reference)


if __name__ == "__main__":
    unittest.main()
