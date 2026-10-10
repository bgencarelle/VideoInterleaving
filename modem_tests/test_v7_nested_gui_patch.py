import tempfile
import unittest
from pathlib import Path

from tools.v7_nested_gui_patch import (DEFAULT_TABLE_PATH, launcher_prefix,
                                      resolve_table_path, wrap_sender)
from tools.v7_send_gui import OutputDevice, SenderGui, build_command
from tools import v7_live
from tools.v7_receiver_gui import ReceiverGui, _receive_parser


class MatchedNestedGuiPatchTests(unittest.TestCase):
    class SoundDeviceStub:
        @staticmethod
        def check_output_settings(**_kwargs):
            return None

    def test_disabled_sender_command_is_unchanged(self):
        command = ['python', '/repo/tools/v7_live.py', 'send']
        self.assertIs(wrap_sender(command, '', False), command)

    def test_requires_an_existing_explicit_table(self):
        with self.assertRaisesRegex(ValueError, 'Matched nested table not found'):
            launcher_prefix('tmp/nonexistent-v7-table.npz', True,
                            'stereo-nested')

    def test_matched_profile_flags_and_chroma_selection(self):
        command = ['python', '/repo/tools/v7_live.py', 'send',
                   '--profile', 'stereo-nested']
        with tempfile.TemporaryDirectory() as folder:
            tables = Path(folder)/'matched.npz'
            tables.touch()
            stereo = wrap_sender(command, tables, True, 'stereo-nested')
            mono = wrap_sender(command, tables, True, 'aspect-mono-nested')

        self.assertTrue(stereo[1].endswith('v7_nested_experiment.py'))
        for flag in ('--mapping', 'serpentine', '--adaptive-guest-range',
                     '--step-factor', '0.25', '--exact-inverse'):
            self.assertIn(flag, stereo)
            self.assertIn(flag, mono)
        self.assertIn('--chroma-detail', stereo)
        self.assertNotIn('--chroma-detail', mono)

    def test_default_table_is_the_bundled_validated_candidate(self):
        self.assertEqual(str(DEFAULT_TABLE_PATH),
                         'test_modem_v7/experimental_tables/chroma50.npz')
        self.assertTrue(resolve_table_path().is_file())

    def test_rejects_non_nested_sender_profile(self):
        with self.assertRaisesRegex(ValueError, 'requires a nested wire profile'):
            launcher_prefix(DEFAULT_TABLE_PATH, True, 'aspect-fold-500')

    def test_sender_gui_builds_ready_candidate_command(self):
        device = OutputDevice(0, 'pulse', 32, 44100, 'ALSA')
        gui = SenderGui((device,))
        gui.settings.update(device=0, source='test', profile='stereo-nested',
                            luma_adjust=True)
        gui._assign('matched_nested_test', True)
        self.assertFalse(gui.settings['luma_adjust'])
        command = build_command(gui.settings, (device,), self.SoundDeviceStub(),
                                python='python')
        self.assertIn('--chroma-detail', command)
        self.assertEqual(command[1], str(Path(__file__).resolve().parents[1]/
                                         'tools'/'v7_nested_experiment.py'))
        self.assertIn(str(resolve_table_path()), command)

    def test_receiver_gui_builds_matching_candidate_options(self):
        root_parser, receive_parser = _receive_parser(v7_live)
        gui = ReceiverGui(v7_live, root_parser, receive_parser,
                          (('pulse · 25 in · 96 kHz', 0),))
        gui._field('device').value = 0
        self.assertFalse(gui._field(
            'Matched nested test patch · opt-in').value)
        gui._field('Matched nested test patch · opt-in').value = True
        gui._field('Matched nested profile').value = 'stereo-nested'
        arguments = gui._build_arguments()
        self.assertTrue(arguments.matched_nested_test)
        self.assertEqual(arguments.nested_test_profile, 'stereo-nested')
        self.assertEqual(arguments.nested_test_tables,
                         str(resolve_table_path()))


if __name__ == '__main__':
    unittest.main()
