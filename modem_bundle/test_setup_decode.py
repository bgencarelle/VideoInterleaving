"""Focused installer checks; no package installs, browser or device access."""
import contextlib
import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import setup_decode as setup


class SetupTests(unittest.TestCase):
    def test_native_macports_matches_python_and_mode(self):
        args = type('Args', (), dict(wav=True, headless=True, non_interactive=False))()
        with patch.object(setup.sys, 'platform', 'darwin'), \
             patch.object(setup.sys, 'base_prefix', '/opt/local/Library/Frameworks/Python.framework/Versions/3.12'), \
             patch.object(setup.sys, 'version_info', type('Version', (), dict(major=3, minor=12))()), \
             patch.object(setup, 'executable', side_effect=['/opt/local/bin/port', None]), \
             patch.object(setup.sys.stdin, 'isatty', return_value=True), \
             patch('builtins.input', return_value='yes'), patch.object(setup, 'run') as run:
            setup.native_dependencies(args)
            run.assert_called_once_with(['sudo', '/opt/local/bin/port', 'install',
                                         'py312-numpy', 'py312-scipy', 'py312-Pillow'])

    def test_native_refuses_unrelated_python(self):
        args = type('Args', (), dict(wav=False, headless=False, non_interactive=False))()
        with patch.object(setup.sys, 'platform', 'darwin'), \
             patch.object(setup.sys, 'base_prefix', '/unrelated/python'), \
             patch.object(setup, 'executable', side_effect=['/opt/local/bin/port', None]), \
             patch.object(setup, 'run') as run:
            with self.assertRaises(RuntimeError):
                setup.native_dependencies(args)
            run.assert_not_called()

    def test_missing_manager_noninteractive(self):
        with patch.object(setup.sys, 'platform', 'win32'), \
             patch.object(setup, 'executable', return_value=None), \
             patch('builtins.input', side_effect=AssertionError('prompt')), \
             patch.object(setup.webbrowser, 'open', side_effect=AssertionError('browser')):
            setup.manager_help(True)

    def test_install_failure_is_nonzero_and_visible(self):
        with patch.object(setup.sys, 'argv', ['setup_decode.py', '--install', '--non-interactive']), \
             patch.object(setup, 'environment', return_value=Path('/selected/python')), \
             patch.object(setup, 'run', side_effect=subprocess.CalledProcessError(9, ['pip'])), \
             patch.object(setup, 'manager_help'), contextlib.redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(setup.main(), 1)
            self.assertIn('FAILED', errors.getvalue())

    def test_foreign_environment_is_not_modified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / '.venv' / ('Scripts/python.exe' if setup.os.name == 'nt' else 'bin/python')
            python.parent.mkdir(parents=True)
            python.touch()
            (root / '.venv/pyvenv.cfg').write_text('sentinel')
            with patch.object(setup, 'ROOT', root), \
                 patch.object(setup.subprocess, 'run', return_value=type('Result', (), dict(stdout='["/other/python", [3, 11]]'))()), \
                 patch.object(setup.venv, 'EnvBuilder') as builder:
                with self.assertRaises(RuntimeError):
                    setup.environment()
                builder.assert_not_called()
                self.assertEqual((root / '.venv/pyvenv.cfg').read_text(), 'sentinel')

    def test_success_uses_environment_for_install_and_checks(self):
        with patch.object(setup.sys, 'argv', ['setup_decode.py', '--install', '--wav', '--headless']), \
             patch.object(setup, 'environment', return_value=Path('/selected/python')), \
             patch.object(setup, 'run') as run:
            self.assertEqual(setup.main(), 0)
            self.assertEqual(run.call_count, 2)
            self.assertTrue(all(call.args[0][0] == '/selected/python' for call in run.call_args_list))
            self.assertEqual(run.call_args.args[0][-2:], ['--wav', '--headless'])


if __name__ == '__main__':
    unittest.main()
