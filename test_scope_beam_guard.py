"""The beam must never park: enforcement tests. No audio hardware needed.

Run with: python -m unittest test_scope_beam_guard

Four places in this codebase say the beam must never sit still -- rasterize()
idles on a circle, BufferedSource holds its last sample, StochasticGen rings a
one-pixel source, scope_display falls back to a safe idle circle. None of them
checked that the frame actually reaching the DAC obeys it. These tests pin the
single enforcement point and the three sites that used to violate it.
"""
import threading
import unittest

import numpy as np

from scope_out import (LEVEL, PARK_PTP, BufferedSource, Scope, beam_is_parked,
                       precompensate_hpf, unpark_frame)


def constant_frame(n=3200, at=(0.0, 0.0)):
    return np.tile(np.asarray([at], dtype=np.float32), (n, 1))


def moving_frame(n=3200):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack((0.8 * np.cos(t), 0.8 * np.sin(t))).astype(np.float32)


class BeamParkDetectionTests(unittest.TestCase):
    def test_constant_frames_are_parked_wherever_they_sit(self):
        for at in ((0.0, 0.0), (0.9, -0.9), (-0.3, 0.42)):
            self.assertTrue(beam_is_parked(constant_frame(at=at)), at)

    def test_a_real_picture_is_not_parked(self):
        self.assertFalse(beam_is_parked(moving_frame()))

    def test_motion_on_one_axis_alone_still_counts_as_moving(self):
        # A single scanline is a legitimate picture: X sweeps, Y does not.
        f = constant_frame()
        f[:, 0] = np.linspace(-0.9, 0.9, len(f))
        self.assertFalse(beam_is_parked(f))

    def test_threshold_sits_below_anything_a_picture_produces(self):
        f = constant_frame()
        f[:, 0] += np.linspace(0.0, PARK_PTP * 0.5, len(f))
        self.assertTrue(beam_is_parked(f))
        f[:, 0] = np.linspace(0.0, PARK_PTP * 10.0, len(f))
        self.assertFalse(beam_is_parked(f))

    def test_unpark_rings_the_point_without_moving_the_picture(self):
        f = constant_frame(at=(0.4, -0.2))
        out = unpark_frame(f)
        self.assertEqual(out.shape, f.shape)
        self.assertFalse(beam_is_parked(out))
        # It stays where the content was: this is "a dot there", not a jump.
        self.assertLess(float(np.abs(out.mean(axis=0) - [0.4, -0.2]).max()), 0.05)
        self.assertLessEqual(float(np.abs(out).max()), LEVEL + 1e-6)

    def test_consecutive_parked_frames_do_not_retrace_one_ring(self):
        f = constant_frame(at=(0.1, 0.1))
        self.assertFalse(np.array_equal(unpark_frame(f, phase=0),
                                        unpark_frame(f, phase=7)))


class ShowFrameEnforcementTests(unittest.TestCase):
    def setUp(self):
        self.scope = Scope(device="null", samplerate=96000, samples=3200)
        self.addCleanup(self.scope.stream.close)

    def test_a_parked_frame_never_reaches_the_dac(self):
        self.scope.show_frame(constant_frame())
        self.assertFalse(beam_is_parked(self.scope._pending))
        self.assertEqual(self.scope.beams_unparked, 1)

    def test_a_normal_frame_passes_through_untouched(self):
        f = moving_frame()
        self.scope.show_frame(f)
        np.testing.assert_allclose(self.scope._pending, f, atol=1e-6)
        self.assertEqual(self.scope.beams_unparked, 0)

    def test_the_guard_runs_before_the_yt_marker_can_fake_motion(self):
        # The marker moves X by itself, so a check placed after it would
        # declare a collapsed picture healthy.
        scope = Scope(device="null", samplerate=96000, samples=3200,
                      yt_mode=True, yt_trigger_us=500.0)
        self.addCleanup(scope.stream.close)
        scope.show_frame(constant_frame())
        self.assertEqual(scope.beams_unparked, 1)
        # The marker is still intact: exactly one rising crossing per trace.
        x = scope._pending[:, 0]
        self.assertEqual(int(np.count_nonzero((x[:-1] < 0.95) & (x[1:] >= 0.95))), 1)

    def test_callback_output_is_never_silent_after_a_source_failure(self):
        def exploding_source(_frames):
            raise RuntimeError("producer died")

        scope = Scope(device="null", samplerate=96000, samples=3200,
                      source=exploding_source)
        self.addCleanup(scope.stream.close)
        buf = np.empty((512, 2), dtype=np.float32)
        scope._callback(buf, 512, None, None)
        self.assertFalse(np.array_equal(buf, np.zeros_like(buf)),
                         "a dead source must hold position, not park at centre")
        self.assertEqual(scope.dac_dropouts, 1)

    def test_reported_dropouts_track_portaudio_status(self):
        buf = np.empty((512, 2), dtype=np.float32)
        self.scope._callback(buf, 512, None, None)
        self.assertEqual(self.scope.dac_dropouts, 0)
        self.scope._callback(buf, 512, None, "output underflow")
        self.assertEqual(self.scope.dac_dropouts, 1)


class BufferedSourceColdStartTests(unittest.TestCase):
    def test_an_underrun_before_the_first_block_does_not_hold_centre(self):
        # A producer that never yields a block, so the read underruns and the
        # held value is whatever the constructor seeded.
        stalled = threading.Event()

        def never_produces(_n):
            stalled.wait(5.0)
            raise RuntimeError("producer never started")

        source = BufferedSource(never_produces, blocksize=256)
        self.addCleanup(source.close)
        self.addCleanup(stalled.set)
        out = source(64)
        self.assertEqual(source.underruns, 1)
        self.assertFalse(np.allclose(out, 0.0),
                         "cold-start hold parked the beam at screen centre")


class PrecompensateHpfTests(unittest.TestCase):
    def test_a_constant_frame_is_not_amplified_into_noise(self):
        f = constant_frame(at=(0.3, -0.5))
        out = precompensate_hpf(f, 20.0, 96000)
        # Was 0.9 peak of pure FFT round-off before the floor was added.
        np.testing.assert_allclose(out, f, atol=1e-6)

    def test_a_whisper_of_motion_is_not_given_thousands_of_x_gain(self):
        f = constant_frame(at=(0.3, -0.5))
        f[len(f) // 2:, 0] += 1e-4
        out = precompensate_hpf(f, 20.0, 96000)
        self.assertLess(float(np.abs(out).max()), 0.6)

    def test_a_real_frame_is_still_corrected_and_renormalised(self):
        f = moving_frame()
        out = precompensate_hpf(f, 20.0, 96000)
        self.assertAlmostEqual(float(np.abs(out).max()), LEVEL, places=5)
        self.assertFalse(np.allclose(out, f))


if __name__ == "__main__":
    unittest.main()
