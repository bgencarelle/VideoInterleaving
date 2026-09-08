"""One signal, two displays: the always-on X trigger.

Run with: python -m unittest test_scope_trigger

The marker used to be a mode -- --scope-yt -- that implied raster, forced
retrace, and refused to start alongside realtime, mix, stochastic or stipple.
It is a property of the output now, on by default, stamped on the finished
frame in Scope.show_frame that every renderer already passes through.

Two claims have to hold for that to be safe, and these tests pin both:

  1. A single-channel scope in Y-T still locks: exactly one rising crossing of
     +0.95 per trace, at one trace period, in every renderer.
  2. An XY scope shows no sign of it: the marker is monotonic (so the beam is
     moving, never dwelling) and parked outside the +-0.9 picture box, which
     is the same off-screen-excursion trick apply_overscan() uses.
"""
from pathlib import Path
import unittest

import numpy as np

from scope_bake import TraceEmitter, render_luma
from scope_out import (LEVEL, Scope, clip_for_trigger, trigger_frame,
                       yt_trigger_frame)

RATE = 96000
N = 3200
MARKER = 48


def rising(x, level=0.95):
    return np.flatnonzero((x[:-1] < level) & (x[1:] >= level)) + 1


def subject(rows=32, cols=32):
    g = np.tile(np.linspace(0.15, 0.95, cols), (rows, 1))
    g[:4] = 0.0
    return g


class TriggerUniquenessTests(unittest.TestCase):
    def test_one_crossing_per_trace_in_both_shapes(self):
        frame = render_luma(subject(), N, grid_rows=32, grid_cols=32)
        for shape in ("ramp", "step"):
            with self.subTest(shape=shape):
                out = trigger_frame(clip_for_trigger(frame), MARKER,
                                    shape=shape)
                self.assertEqual(len(rising(out[:, 0])), 1)

    def test_the_marker_is_periodic_across_block_boundaries(self):
        frame = render_luma(subject(), N, grid_rows=32, grid_cols=32)
        chunks, cursor = [], 0
        for count in (37, 1500, 91, 3200, 1372):
            idx = (np.arange(count) + cursor) % N
            chunks.append(trigger_frame(clip_for_trigger(frame[idx]), MARKER,
                                        offset=cursor, period=N))
            cursor = (cursor + count) % N
        out = np.vstack(chunks)
        edges = rising(out[:, 0])
        self.assertGreater(len(edges), 1)
        # Every gap is exactly one trace: that is what a timebase locks to.
        np.testing.assert_array_equal(np.diff(edges), N)

    def test_a_bright_edge_cannot_forge_a_second_crossing(self):
        # A filter that rings above +0.95 was the one way picture content could
        # look like a trigger. clip_for_trigger runs before the marker for
        # exactly this reason.
        frame = render_luma(subject(), N, grid_rows=32, grid_cols=32)
        frame[1000:1010, 0] = 0.999          # overshoot, as a lowpass would
        self.assertEqual(len(rising(frame[:, 0])), 1)   # the forged one
        out = trigger_frame(clip_for_trigger(frame), MARKER)
        self.assertEqual(len(rising(out[:, 0])), 1)     # only the marker's
        self.assertLess(int(rising(out[:, 0])[0]), MARKER)


class XyCompatibilityTests(unittest.TestCase):
    """The claim that lets the marker default to on."""

    def setUp(self):
        frame = render_luma(subject(), N, grid_rows=32, grid_cols=32)
        self.ramp = trigger_frame(clip_for_trigger(frame), MARKER)
        self.step = trigger_frame(clip_for_trigger(frame), MARKER, shape="step")

    def test_the_ramp_never_dwells(self):
        # A stationary run is the brightest thing on a scope. The step shape
        # has two of them by design; the ramp must have none.
        ramp_still = np.count_nonzero(
            np.abs(np.diff(self.ramp[:MARKER, 0])) <= 1e-6)
        step_still = np.count_nonzero(
            np.abs(np.diff(self.step[:MARKER, 0])) <= 1e-6)
        self.assertLessEqual(ramp_still, 2 * max(1, MARKER // 8))
        self.assertGreater(step_still, MARKER // 2)

    def test_the_ramp_is_monotonic_so_no_threshold_is_crossed_twice(self):
        self.assertTrue(np.all(np.diff(self.ramp[:MARKER, 0]) >= -1e-6))
        for level in (-0.5, 0.0, 0.5, 0.95):
            self.assertEqual(len(rising(self.ramp[:MARKER, 0], level)), 1, level)

    def test_the_ramp_sits_outside_the_picture_box(self):
        # Set the scope so +-0.9 fills the screen and the marker deflects past
        # the phosphor on BOTH axes -- there is nothing left to see.
        self.assertTrue(np.all(np.abs(self.ramp[:MARKER, 1]) > LEVEL))
        # X sweeps the full width, so it is inside the box for part of the
        # marker -- but only ever at Y on the rail, which is off the screen.
        self.assertGreater(float(np.abs(self.ramp[:MARKER, 0]).max()), LEVEL)
        self.assertEqual(float(np.abs(self.ramp[:MARKER, 1]).min()),
                         float(np.abs(self.ramp[:MARKER, 1]).max()),
                         "Y must stay pinned for the whole marker")

    def test_the_step_shape_is_still_available_and_leaves_y_alone(self):
        frame = render_luma(subject(), N, grid_rows=32, grid_cols=32)
        np.testing.assert_array_equal(self.step[:, 1],
                                      clip_for_trigger(frame)[:, 1])

    def test_the_picture_after_the_marker_is_untouched(self):
        frame = clip_for_trigger(
            render_luma(subject(), N, grid_rows=32, grid_cols=32))
        for out in (self.ramp, self.step):
            np.testing.assert_array_equal(out[MARKER:], frame[MARKER:])


class EveryRendererTests(unittest.TestCase):
    """It used to be raster-only. show_frame does not know what drew the frame."""

    def emit_through_scope(self, frame, **kw):
        scope = Scope(device="null", samplerate=RATE, samples=len(frame), **kw)
        try:
            scope.show_frame(frame)
            return scope._pending
        finally:
            scope.stream.close()

    def test_dwell_and_fixed_row_timing_both_carry_the_marker(self):
        for timing in ("dwell", "fixed"):
            with self.subTest(timing=timing):
                emitter = TraceEmitter(
                    RATE, N, grid=(32, 32), levels=(0.0, 1.0), border=0.09,
                    sweep="retrace",
                    yt_timing=timing,
                    yt_trigger_samples=MARKER if timing == "fixed" else 0)
                out = self.emit_through_scope(emitter.emit(subject()),
                                              yt_trigger_us=500.0)
                self.assertEqual(len(rising(out[:, 0])), 1)

    def test_an_arbitrary_non_raster_waveform_still_triggers(self):
        # Stands in for stochastic/stipple/vector: show_frame is shape-blind.
        t = np.linspace(0, 2 * np.pi, N, endpoint=False)
        walk = np.column_stack((0.8 * np.cos(3 * t), 0.8 * np.sin(5 * t)))
        out = self.emit_through_scope(walk.astype(np.float32),
                                      yt_trigger_us=500.0)
        self.assertEqual(len(rising(out[:, 0])), 1)

    def test_switching_the_trigger_off_leaves_the_waveform_alone(self):
        frame = render_luma(subject(), N, grid_rows=32, grid_cols=32)
        out = self.emit_through_scope(frame, trigger=False)
        np.testing.assert_allclose(out, frame, atol=1e-6)
        self.assertEqual(len(rising(out[:, 0])), 0)


class MeasurementToolsTests(unittest.TestCase):
    """Tools that MEASURE the waveform must not have it stamped on.

    The marker is a deliberate corruption of the signal for a triggering
    scope's benefit. On a calibration square or a filter audition it is
    indistinguishable from the distortion those tools exist to reveal, so
    both construct their Scope with trigger=False and these tests keep it
    that way -- the default flipping to on is exactly how it would come back.
    """

    def source_of(self, filename):
        return Path(__file__).with_name(filename).read_text()

    def test_the_calibration_bench_disables_the_marker(self):
        src = self.source_of("scope_out.py")
        bench = src[src.index('if __name__ == "__main__":'):]
        call = bench[bench.index("scope = Scope("):]
        self.assertIn("trigger=False", call[:call.index(")")],
                      "the calibration square must not carry a trigger marker")

    def test_the_lowpass_audition_tool_disables_the_marker(self):
        src = self.source_of("scope_lowpass.py")
        call = src[src.index("scope = Scope("):]
        self.assertIn("trigger=False", call[:call.index("\n            samplerate")],
                      "a filter audition must show the filter, not the marker")


class BackwardCompatibilityTests(unittest.TestCase):
    def test_the_old_function_name_still_resolves(self):
        self.assertIs(yt_trigger_frame, trigger_frame)

    def test_yt_mode_still_switches_the_trigger(self):
        for value, expected in ((True, True), (False, False)):
            scope = Scope(device="null", samplerate=RATE, samples=N,
                          yt_mode=value)
            try:
                self.assertEqual(scope.trigger, expected)
                self.assertEqual(scope.yt_mode, expected)   # legacy attribute
            finally:
                scope.stream.close()

    def test_an_unknown_shape_is_rejected_rather_than_guessed(self):
        with self.assertRaises(ValueError):
            trigger_frame(np.zeros((64, 2), np.float32), 8, shape="square")
        with self.assertRaises(ValueError):
            Scope(device="null", samples=64, trigger_shape="square")


if __name__ == "__main__":
    unittest.main()
