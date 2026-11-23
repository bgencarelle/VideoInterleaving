import unittest

from scope_gui import ScopeGUI


class ScopeGuiLivePipelineTests(unittest.TestCase):
    def test_live_pipeline_can_disable_unsupported_sliders(self):
        gui = ScopeGUI.__new__(ScopeGUI)

        self.assertTrue(gui._slider_disabled(
            "density", {"disabled_sliders": ("density",)}))
        self.assertFalse(gui._slider_disabled(
            "gamma", {"disabled_sliders": ("density",)}))


if __name__ == "__main__":
    unittest.main()
