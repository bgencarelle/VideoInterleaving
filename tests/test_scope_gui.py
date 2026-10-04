import unittest

from scope_controls import KeyMap
from scope_gui import (fit_square_image, make_slider_spec,
                       slider_default_x, slider_fraction,
                       slider_value_at)


class ScopeSliderMathTests(unittest.TestCase):
    def test_default_tick_marks_the_startup_value(self):
        spec = make_slider_spec("density", 1.0)
        left, right = 40, 440
        marker = slider_default_x(left, right, spec)
        self.assertEqual(marker, round(left + slider_fraction(1.0, spec)
                                       * (right - left)))
        self.assertGreater(marker, left)
        self.assertLess(marker, right)

    def test_pointer_values_clamp_and_quantize_to_slider_steps(self):
        spec = make_slider_spec("fields", 1)
        self.assertEqual(slider_value_at(-10, 100, 300, spec), 1)
        self.assertEqual(slider_value_at(400, 100, 300, spec), 4)
        self.assertEqual(slider_value_at(200, 100, 300, spec), 3)

    def test_slider_range_expands_to_keep_a_high_startup_default_visible(self):
        spec = make_slider_spec("fps", 150)
        self.assertEqual(spec.default, 150)
        self.assertGreater(spec.maximum, spec.default)

    def test_preview_scales_to_fit_the_smaller_gui_dimension(self):
        self.assertEqual(fit_square_image(800, 540), 540)
        self.assertEqual(fit_square_image(800, 900), 680)


class ScopeGuiControlTests(unittest.TestCase):
    def setUp(self):
        self.state = {
            "trim": 0.02, "density": 1.0, "gamma": 2.2,
            "raster_gamma": 2.2, "stochastic_gamma": 2.0,
            "lowpass": None, "rows": None, "mode": "raster",
            "raster": True, "fusion_components": "vrs",
            "mode_locked": False,
        }
        self.controls = KeyMap(self.state)

    def test_dac_mute_is_applied_at_output_stage(self):
        import numpy as np
        from scope_out import Scope

        scope = Scope(device="null", fps=30, trigger=False)
        frame = np.array(((0.25, -0.5), (0.5, 0.75)), dtype=np.float32)
        output = np.empty_like(frame)
        scope.set_output_audio(muted=True)
        scope._write_output(output, frame)
        np.testing.assert_array_equal(output, np.zeros_like(frame))
        scope.set_output_audio(muted=False)
        scope._write_output(output, frame)
        np.testing.assert_allclose(output, frame)

    def test_live_dac_mute_fades_over_one_callback_block(self):
        import numpy as np
        from scope_out import Scope

        scope = Scope(device="null", fps=30, trigger=False)

        class ActiveStream:
            active = True

        scope.stream = ActiveStream()
        frame = np.ones((5, 2), dtype=np.float32)
        output = np.empty_like(frame)
        scope.set_output_audio(muted=True)
        scope._write_output(output, frame)
        np.testing.assert_allclose(output[:, 0], (1.0, 0.75, 0.5, 0.25, 0.0))
        np.testing.assert_allclose(output[:, 1], output[:, 0])

    def test_realtime_error_marker_cannot_bypass_dac_mute(self):
        import numpy as np
        from scope_out import Scope

        def broken_source(_frames):
            raise RuntimeError("source failed")

        scope = Scope(device="null", fps=30, source=broken_source,
                      trigger=True)
        scope.set_output_audio(muted=True)
        output = np.empty((16, 2), dtype=np.float32)
        scope._callback(output, len(output), None, None)
        np.testing.assert_array_equal(output, np.zeros_like(output))

    def test_grid_sliders_request_recalibration(self):
        self.controls.set_value("trim", 0.12)
        self.assertAlmostEqual(self.state["trim"], 0.12)
        self.assertTrue(self.controls.dirty)
        self.controls.dirty = False
        self.controls.set_value("density", 1.5)
        self.assertAlmostEqual(self.state["density"], 1.5)
        self.assertTrue(self.controls.dirty)

    def test_tone_and_lowpass_sliders_update_without_grid_recalibration(self):
        self.controls.set_value("gamma", 3.1)
        self.assertAlmostEqual(self.state["raster_gamma"], 3.1)
        self.assertFalse(self.controls.dirty)
        self.controls.set_value("lowpass", 6000)
        self.assertEqual(self.state["lowpass"], 6000.0)
        self.assertFalse(self.controls.dirty)
        self.controls.set_value("lowpass", 0)
        self.assertIsNone(self.state["lowpass"])

    def test_auto_rows_and_direct_mode_selection(self):
        self.controls.set_value("rows", 0)
        self.assertIsNone(self.state["rows"])
        self.assertTrue(self.controls.set_mode("stipple"))
        self.assertEqual(self.state["mode"], "stipple")
        self.assertFalse(self.state["raster"])
        self.assertTrue(self.controls.dirty)

    def test_mode_selection_respects_the_active_source_capabilities(self):
        self.state["available_modes"] = ("raster", "stochastic", "stipple")
        self.assertFalse(self.controls.set_mode("vector"))
        self.assertEqual(self.state["mode"], "raster")
        self.controls.feed("v")
        self.assertEqual(self.state["mode"], "stochastic")


if __name__ == "__main__":
    unittest.main()
