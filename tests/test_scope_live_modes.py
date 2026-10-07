"""Captured-frame render modes and output-queue rollback, without devices."""
import unittest
from unittest.mock import Mock

import numpy as np

from scope_bake import StochasticEmitter, StippleEmitter
from tools.scope_screen import build_parser, live_capture_modes, _push_live_image


class LiveCaptureModeTests(unittest.TestCase):
    def test_supported_modes_and_raster_only_configurations(self):
        parser = build_parser()
        self.assertEqual(live_capture_modes(parser.parse_args([])),
                         ("raster", "stochastic", "stipple"))
        for options in (["--stream"], ["--fields", "2"],
                        ["--geometry-samples", "2400"], ["--traversal-hz", "30"]):
            with self.subTest(options=options):
                self.assertEqual(live_capture_modes(parser.parse_args(options)),
                                 ("raster",))

    def test_rejected_trace_restores_trajectory_and_random_state(self):
        lum = np.ones((16, 24), np.float32)
        lum[:, :8] = .2
        captured = {"lum": lum, "source_metadata": {"source_sequence": 7}}
        for mode, factory in (("stochastic", StochasticEmitter), ("stipple", StippleEmitter)):
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
