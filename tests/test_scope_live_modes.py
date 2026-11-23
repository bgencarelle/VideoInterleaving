"""Captured-frame render modes and output-queue rollback, without devices."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from scope_bake import StochasticEmitter, StippleEmitter, TraceEmitter, PositionMultiplexer
from scope_live_renderers import LiveVectorEmitter, LiveFusionEmitter, LiveCaptureTone
from tools.scope_screen import build_parser, live_capture_modes, _push_live_image


class LiveCaptureModeTests(unittest.TestCase):
    def test_tone_pins_version_and_rebuilds_when_inverted(self):
        tone = LiveCaptureTone()
        source = np.array([[0., .25], [.75, 1.]], dtype=np.float32)
        first, identity = tone.prepare(source, 7)
        repeated, repeated_identity = tone.prepare(source, 7)
        self.assertIs(first, repeated)
        self.assertEqual(identity, repeated_identity)
        tone.invert = True
        inverted, inverted_identity = tone.prepare(source, 7)
        np.testing.assert_allclose(inverted, 1-source)
        self.assertNotEqual(identity, inverted_identity)
        np.testing.assert_array_equal(first, source)

    def test_supported_modes_and_raster_only_configurations(self):
        parser = build_parser()
        self.assertEqual(live_capture_modes(parser.parse_args([])),
                         ("vector", "raster", "stochastic", "stipple", "fusion"))
        for options in (["--stream"], ["--fields", "2"],
                        ["--geometry-samples", "2400"], ["--traversal-hz", "30"]):
            with self.subTest(options=options):
                self.assertEqual(live_capture_modes(parser.parse_args(options)),
                                 ("raster",))

    def test_rejected_trace_restores_trajectory_and_random_state(self):
        lum = np.ones((16, 24), np.float32)
        lum[:, :8] = .2
        captured = {"lum": lum, "source_metadata": {"source_sequence": 7}}
        for mode, factory in (("vector", LiveVectorEmitter), ("fusion", LiveFusionEmitter),
                              ("stochastic", StochasticEmitter), ("stipple", StippleEmitter)):
            with self.subTest(mode=mode):
                emitter = factory(96000, 3200)
                reference = factory(96000, 3200)
                scope = Mock()
                scope.frames_accepted = 0
                scope.show_frame.return_value = None
                self.assertFalse(_push_live_image(scope, emitter, captured, mode))
                first = scope.show_frame.call_args.args[0].copy()
                self.assertIsNone(emitter._end)

                def accept(frame, **_kwargs):
                    scope.frames_accepted += 1
                    return frame[-1]

                scope.show_frame.side_effect = accept
                self.assertTrue(_push_live_image(scope, emitter, captured, mode))
                second = scope.show_frame.call_args.args[0]
                np.testing.assert_array_equal(first, second)
                np.testing.assert_array_equal(second, reference.emit(lum))
                identity = scope.show_frame.call_args.kwargs["identity"]
                self.assertEqual(identity[1:3], (f"screen-{mode}", 7))
                self.assertTrue(np.isfinite(second).all())
                self.assertEqual(second.shape, (3200, 2))

    def test_vector_tracks_geometry_of_changed_capture_and_tone(self):
        emitter = LiveVectorEmitter(96000, 3200)
        left = np.zeros((40, 60), np.float32)
        left[10:30, 5:20] = 1
        right = np.zeros_like(left)
        right[10:30, 40:55] = 1
        first = emitter.emit(left)
        second = emitter.emit(right)
        self.assertLess(float(first[:, 0].mean()), -.2)
        self.assertGreater(float(second[:, 0].mean()), .2)
        self.assertGreater(float(np.ptp(first[:, 1])), .4)
        emitter.trim = 1.01
        np.testing.assert_array_equal(emitter.emit(right), emitter._idle)

    def test_application_runtime_thumbnails_render_vector_and_fusion(self):
        from scope_display import _emit
        thumb = np.zeros((40, 60, 2), np.uint8)
        thumb[:, :, 1] = 255
        thumb[10:30, 5:20, 0] = 255
        frames = []

        def show(frame, **_kwargs):
            frames.append(frame)
            return frame[-1]

        scope = SimpleNamespace(samples_per_frame=3200, samplerate=96000, show_frame=show,
                                _runtime_vector=LiveVectorEmitter(96000, 3200))
        raster = TraceEmitter(96000, 3200)
        walk = StochasticEmitter(96000, 3200)
        for mode in ("vector", "fusion"):
            with self.subTest(mode=mode):
                endpoint = _emit(scope, None, None, 0, mode, {"rev": False, "end": None},
                                 "alternate", 1.8, .02, 1., None, .02,
                                 emitter=raster, stochastic_emitter=walk,
                                 fusion_multiplexer=PositionMultiplexer(),
                                 source_thumbnails=(thumb, None))
                self.assertEqual(frames[-1].shape, (3200, 2))
                self.assertTrue(np.isfinite(frames[-1]).all())
                self.assertTrue(np.any(frames[-1]))
                self.assertEqual(endpoint.shape, (2,))
