import subprocess
import unittest
from unittest.mock import patch

from tools import v7_native_file_dialog as dialog


class NativeDirectoryDialogTests(unittest.TestCase):
    def test_macos_uses_native_dialog_with_arguments_not_tk(self):
        result = subprocess.CompletedProcess([], 0, '/Users/ben/Pictures/Test folder/\n', '')
        with patch.object(dialog.sys, 'platform', 'darwin'), \
                patch.object(dialog.subprocess, 'run', return_value=result) as run:
            selected = dialog.pick_directory(title='Save "decoded" images')
        self.assertEqual(selected, '/Users/ben/Pictures/Test folder/')
        command = run.call_args.args[0]
        self.assertEqual(command[0], '/usr/bin/osascript')
        self.assertEqual(command[-1], 'Save "decoded" images')

    def test_macos_cancel_is_not_a_gui_error(self):
        result = subprocess.CompletedProcess([], 1, '', 'User canceled. (-128)')
        with patch.object(dialog.sys, 'platform', 'darwin'), \
                patch.object(dialog.subprocess, 'run', return_value=result):
            self.assertIsNone(dialog.pick_directory())

    def test_other_platforms_isolate_tk_in_a_child_process(self):
        result = subprocess.CompletedProcess([], 0, '"/home/user/captures"\n', '')
        with patch.object(dialog.sys, 'platform', 'linux'), \
                patch.object(dialog.subprocess, 'run', return_value=result) as run:
            self.assertEqual(dialog.pick_directory(), '/home/user/captures')
        command = run.call_args.args[0]
        self.assertEqual(command[0], dialog.sys.executable)
        self.assertTrue(command[1].endswith('v7_native_file_dialog.py'))

    def test_child_process_failure_is_reported_without_killing_gui(self):
        result = subprocess.CompletedProcess([], -6, '', 'dialog initialization failed')
        with patch.object(dialog.sys, 'platform', 'linux'), \
                patch.object(dialog.subprocess, 'run', return_value=result):
            with self.assertRaisesRegex(RuntimeError, 'dialog initialization failed'):
                dialog.pick_directory()

    def test_child_cancel_and_invalid_output(self):
        with patch.object(dialog.sys, 'platform', 'linux'), \
                patch.object(dialog.subprocess, 'run') as run:
            run.return_value = subprocess.CompletedProcess([], 0, 'null\n', '')
            self.assertIsNone(dialog.pick_directory())
            run.return_value = subprocess.CompletedProcess([], 0, '{}\n', '')
            with self.assertRaisesRegex(RuntimeError, 'picker result'):
                dialog.pick_directory()


if __name__ == '__main__':
    unittest.main()
