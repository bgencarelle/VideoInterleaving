"""The macOS borderless fullscreen position path is testable without GLFW."""
import ast
from pathlib import Path
import types
import unittest


DISPLAY_MANAGER = Path(__file__).resolve().parents[1] / 'display_manager.py'


class MacFullscreenPositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(DISPLAY_MANAGER.read_text(), filename=str(DISPLAY_MANAGER))
        cls.position_helper = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == '_position_macos_fullscreen')
        cls.display_init = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == 'display_init')

    def test_existing_window_toggle_positions_on_primary_monitor(self):
        calls = []
        glfw = types.SimpleNamespace(
            get_monitor_pos=lambda monitor: (-1440, 0),
            set_window_pos=lambda window, x, y: calls.append((window, x, y)))
        namespace = {'_is_macos': lambda: True, 'glfw': glfw}
        module = ast.Module(body=[self.position_helper], type_ignores=[])
        exec(compile(module, str(DISPLAY_MANAGER), 'exec'), namespace)
        namespace['_position_macos_fullscreen']('existing-window', 'monitor')
        self.assertEqual(calls, [('existing-window', -1440, 0)])

        # The in-place fullscreen path must keep invoking the positioning
        # helper; initial window creation alone does not cover F-key toggles.
        toggled_window_calls = [
            node for node in ast.walk(self.display_init)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == '_position_macos_fullscreen'
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == 'window']
        self.assertTrue(toggled_window_calls)


if __name__ == '__main__':
    unittest.main()
