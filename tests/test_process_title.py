import ast
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


class ProcessTitleTests(unittest.TestCase):
    def test_each_cli_mode_gets_a_vi_title(self):
        source = Path(__file__).resolve().parents[1] / "main.py"
        tree = ast.parse(source.read_text(), filename=str(source))
        helper = next(node for node in tree.body
                      if isinstance(node, ast.FunctionDef)
                      and node.name == "_set_process_title")
        namespace = {}
        exec(compile(ast.Module(body=[helper], type_ignores=[]),
                     str(source), "exec"), namespace)

        seen = []
        fake_module = types.SimpleNamespace(
            setproctitle=lambda title: seen.append(title))
        with patch.dict(sys.modules, {"setproctitle": fake_module}):
            for mode in ("web", "ascii", "asciiweb", "local", "scope", "modem"):
                namespace["_set_process_title"](mode)

        self.assertEqual(seen, [
            "vi.web", "vi.ascii", "vi.asciiweb",
            "vi.local", "vi.scope", "vi.modem",
        ])


if __name__ == "__main__":
    unittest.main()
