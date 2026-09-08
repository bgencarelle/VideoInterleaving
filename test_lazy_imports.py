"""A mode must not pay for the dependencies of the modes that are not running.

Run with: python -m unittest test_lazy_imports

main.py used to import every mode's entry point at module level, so one
missing library took down modes that had no use for it: a box without
SimpleWebSocketServer could not start scope, local or web, and a box without
a MIDI stack could not run a free-running clock. image_display was wrapped in
try/except for this reason, which deferred the FAILURE but still ran the
import and still required the stack to be installed.

These tests pin the invariant at the two levels it can break: the import
statements themselves (fast, exact) and the behaviour of the one chain that
was hidden two modules deep (mido, behind index_calculator -> midi_control).
"""
import ast
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


def module_level_imports(filename):
    """Every name imported at module scope -- not inside a def, class or try.

    A try/except around an import still runs it, so it counts here.
    """
    tree = ast.parse(Path(__file__).with_name(filename).read_text(),
                     filename=filename)
    names = set()
    for node in tree.body:                       # top level only, by construction
        stack = [node] if isinstance(node, (ast.Import, ast.ImportFrom)) else []
        if isinstance(node, ast.Try):
            stack = [n for n in ast.walk(node)
                     if isinstance(n, (ast.Import, ast.ImportFrom))]
        for imp in stack:
            if isinstance(imp, ast.Import):
                names.update(a.name.split(".")[0] for a in imp.names)
            elif imp.module:
                names.add(imp.module.split(".")[0])
    return names


class MainDefersModeEntryPoints(unittest.TestCase):
    # module -> the mode(s) that actually need it
    MODE_ONLY = {
        "image_display": "local/web/ascii/asciiweb (TurboJPEG, glfw, moderngl)",
        "ascii_server": "ascii",
        "ascii_stats_server": "ascii",
        "ascii_web_server": "asciiweb (SimpleWebSocketServer)",
        "scope_display": "scope (numpy, sounddevice)",
        "web_service": "local/web/asciiweb/scope",
        "make_file_lists": "every mode except scope reading a baked manifest",
    }

    def test_no_mode_entry_point_is_imported_at_module_level(self):
        top = module_level_imports("main.py")
        for name, owner in self.MODE_ONLY.items():
            self.assertNotIn(
                name, top,
                f"main.py imports {name} at module level; it belongs to "
                f"{owner} and every other mode would pay for it")

    def test_main_still_imports_what_every_mode_needs(self):
        # The counterpart: this is not an argument for deferring everything.
        top = module_level_imports("main.py")
        for name in ("settings", "server_config", "argparse", "os", "sys"):
            self.assertIn(name, top)


class MidiIsNotARequirementOfEveryClock(unittest.TestCase):
    def test_index_calculator_does_not_import_midi_control_at_module_level(self):
        self.assertNotIn("midi_control", module_level_imports("index_calculator.py"))

    def test_make_file_lists_does_not_import_pillow_at_module_level(self):
        # Scope reads its manifest out of a baked .npy through numpy and never
        # opens an image, so Pillow belongs in the one function that does.
        self.assertNotIn("PIL", module_level_imports("make_file_lists.py"))

    def test_a_free_clock_runs_with_no_midi_stack_installed(self):
        """The behavioural half: block `mido` and drive the free clock."""
        script = textwrap.dedent("""
            import sys
            class Block:
                def find_spec(self, name, path=None, target=None):
                    if name.split(".")[0] in ("mido", "rtmidi"):
                        raise ImportError("no MIDI stack on this box")
                    return None
            sys.meta_path.insert(0, Block())
            import index_calculator
            index_calculator.midi_mode = False
            index_calculator.png_paths_len = 8
            i, d = index_calculator.update_index(8, pingpong=True)
            assert isinstance(i, int) and 0 <= i < 8, (i, d)
            assert "midi_control" not in sys.modules, "midi_control was imported anyway"
            print("OK")
        """)
        r = subprocess.run([sys.executable, "-c", script],
                           cwd=str(Path(__file__).parent),
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("OK", r.stdout)


if __name__ == "__main__":
    unittest.main()
