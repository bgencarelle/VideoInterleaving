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
                            'adaptive-stereo-nested')

    def test_matched_profile_flags_and_chroma_selection(self):
        command = ['python', '/repo/tools/v7_live.py', 'send',
                   '--profile', 'adaptive-stereo-nested']
        with tempfile.TemporaryDirectory() as folder:
            tables = Path(folder)/'matched.npz'
            tables.touch()
            stereo = wrap_sender(command, tables, True, 'adaptive-stereo-nested')
            mono = wrap_sender(command, tables, True, 'adaptive-mono-nested')

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

    def test_adaptive_codec_registration_is_scoped_to_its_table(self):
        import nested_fold
        from tools.v7_nested_gui_patch import install_receiver
        default_path = Path(nested_fold.TABLES).resolve()
        legacy_before = nested_fold.table('mono', '1:1', default_path)
        legacy_stereo_before = nested_fold.table('slices', '1:1', default_path)
        table, _ = install_receiver(v7_live, DEFAULT_TABLE_PATH)
        legacy = nested_fold.table('mono', '1:1', default_path)
        adaptive = nested_fold.table('mono', '1:1', table)
        legacy_stereo = nested_fold.table('slices', '1:1', default_path)
        adaptive_stereo = nested_fold.table('slices', '1:1', table)
        self.assertEqual(Path(nested_fold.TABLES).resolve(), default_path)
        self.assertIs(legacy, legacy_before)
        self.assertIs(legacy_stereo, legacy_stereo_before)
        self.assertIsNot(type(legacy), type(adaptive))
        self.assertIsNot(type(legacy_stereo), type(adaptive_stereo))
        self.assertFalse(getattr(legacy, 'adaptive_guest_range', False))
        self.assertFalse(getattr(legacy_stereo, 'adaptive_guest_range', False))
        self.assertTrue(adaptive.adaptive_guest_range)
        self.assertTrue(adaptive_stereo.adaptive_guest_range)

    def test_rejects_non_nested_sender_profile(self):
        with self.assertRaisesRegex(ValueError, 'requires a nested wire profile'):
            launcher_prefix(DEFAULT_TABLE_PATH, True, 'aspect-fold-500')

    def test_sender_gui_builds_ready_candidate_command(self):
        device = OutputDevice(0, 'pulse', 32, 44100, 'ALSA')
        gui = SenderGui((device,))
        gui.settings.update(device=0, source='test', profile='adaptive-stereo-nested',
                            luma_adjust=True)
        command = build_command(gui.settings, (device,), self.SoundDeviceStub(),
                                python='python')
        self.assertNotIn('--luma-adjust', command)
        self.assertIn('--chroma-detail', command)
        self.assertNotIn('--modem-channels', command)
        self.assertNotIn('--rate', command)
        self.assertEqual(command[1], str(Path(__file__).resolve().parents[1]/
                                         'tools'/'v7_nested_experiment.py'))
        self.assertIn(str(resolve_table_path()), command)

    def test_receiver_gui_builds_matching_candidate_options(self):
        root_parser, receive_parser = _receive_parser(v7_live)
        gui = ReceiverGui(v7_live, root_parser, receive_parser,
                          (('pulse · 25 in · 96 kHz', 0),))
        gui._field('device').value = 0
        self.assertNotIn('Matched nested test patch · opt-in',
                         [field.label for field in gui.fields])
        arguments = gui._build_arguments()
        self.assertEqual(arguments.adaptive_nested_table,
                         str(resolve_table_path()))
        self.assertEqual(arguments.nested_test_tables,
                         str(resolve_table_path()))


if __name__ == '__main__':
    unittest.main()
