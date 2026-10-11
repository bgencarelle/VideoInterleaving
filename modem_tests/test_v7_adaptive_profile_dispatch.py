import unittest
from pathlib import Path
import sys
from unittest.mock import patch

from animation_modem.transport3 import (ADAPTIVE_MONO_MODE,
                                       ADAPTIVE_STEREO_MODE,
                                       PROFILE_PREAMBLE_BITS)
TEST_MODEM_PATH = str(Path(__file__).resolve().parents[1]/'test_modem_v7')
if TEST_MODEM_PATH not in sys.path:
    sys.path.insert(0, TEST_MODEM_PATH)
from tone_code import (ADAPTIVE_MONO, ADAPTIVE_STEREO, FOLD_MODE_TO_SLOTS,
                       STATUS_WORD_BY_MODE, encode_status)
from tools import v7_live
from tools.v7_live import _AdaptiveProfileDecoder
from tools.v7_send_gui import OutputDevice, SenderGui, build_command
from tools.v7_profile_registry import ADAPTIVE_PROFILES
from tools.v7_receiver_audio import ReceiverChannelRouter, passthrough_audio_side


class AdaptiveProfileDispatchTests(unittest.TestCase):
    def test_adaptive_mono_and_stereo_have_distinct_retired_wire_ids(self):
        self.assertEqual(ADAPTIVE_PROFILES['adaptive-mono-nested']['mode'], 2)
        self.assertEqual(ADAPTIVE_PROFILES['adaptive-stereo-nested']['mode'], 4)
        self.assertEqual(ADAPTIVE_MONO, 2)
        self.assertEqual(ADAPTIVE_STEREO, 4)
        for mode in (ADAPTIVE_MONO, ADAPTIVE_STEREO):
            self.assertIn(mode, PROFILE_PREAMBLE_BITS)
            self.assertEqual(encode_status(mode), STATUS_WORD_BY_MODE[mode])
            self.assertNotIn(mode, FOLD_MODE_TO_SLOTS)

    def test_retired_mono500_sender_switch_cannot_emit_adaptive_stereo_id(self):
        args = v7_live.parser().parse_args([
            'send', '--device', 'null', '--source', 'test',
            '--experimental-mono-fold'])
        with self.assertRaisesRegex(ValueError, 'retired status ID 4'):
            v7_live._apply_profile_option(args)

    @staticmethod
    def make_dispatcher():
        decoder = object.__new__(_AdaptiveProfileDecoder)
        decoder.active_mode = 1
        decoder.base_mode = 1
        decoder.active_side = None
        decoder.candidate = None
        decoder.streak = 0
        decoder.candidate_last_seen = None
        decoder.candidate_scale = None
        decoder.last_packet = None
        decoder.last_decoded_packet = None
        decoder.state = None
        decoder.stereo_tail_memory = True
        decoder.generation = 0
        decoder.on_switch = None
        decoder.adaptive_mono_mode = ADAPTIVE_MONO_MODE
        decoder.adaptive_stereo_mode = ADAPTIVE_STEREO_MODE
        decoder.adaptive_mode = ADAPTIVE_MONO_MODE
        decoder.aspect_mono_mode = 3
        decoder.mono_status_modes = frozenset((2, 3, 5))
        decoder.stereo_slice_modes = frozenset((4,))
        decoder.stereo_status_modes = frozenset((0, 1, 4))
        decoder.supported_modes = frozenset((0, 1, 2, 3, 4, 5))
        decoder._mode_names = {
            0: 'aspect-fold-500', 1: 'fold-500',
            2: 'adaptive-mono-nested', 3: 'aspect-mono-500',
            4: 'adaptive-stereo-nested', 5: 'mono-colour-500'}
        decoder._reset_aspect_wires = lambda: None
        decoder._last_layouts = {}
        decoder.dispatch_mode = decoder.dispatch_side = None
        decoder.dispatch_slice_metadata = decoder.aspect_mono_slices = False
        decoder.preferred_side = 'auto'
        return decoder

    def test_decoder_switches_mono_stereo_aspect500_mono_by_wire_identity(self):
        decoder = self.make_dispatcher()
        switches = []
        decoder.on_switch = lambda name, side: switches.append((name, side))
        position = 0

        def confirm(mode, side=None):
            nonlocal position
            for _ in range(decoder.REQUIRED_STREAK):
                decision = decoder._observe(position, 1.0, mode, side,
                                            position/10000)
                position += 5000
            self.assertTrue(decision['confirmed'])
            self.assertEqual(decoder.active_mode, mode)
            return decision

        self.assertEqual(confirm(ADAPTIVE_MONO_MODE, 1)['side_index'], 1)
        self.assertEqual(decoder.profile_name, 'adaptive-mono-nested')
        self.assertEqual(confirm(ADAPTIVE_STEREO_MODE)['side_index'], None)
        self.assertEqual(decoder.profile_name, 'adaptive-stereo-nested')
        self.assertEqual(confirm(0)['mode'], 0)
        self.assertEqual(decoder.profile_name, 'aspect-fold-500')
        self.assertEqual(confirm(ADAPTIVE_MONO_MODE, 0)['side_index'], 0)
        self.assertEqual(decoder.profile_name, 'adaptive-mono-nested')
        self.assertEqual([name for name, _side in switches], [
            'adaptive-mono-nested', 'adaptive-stereo-nested',
            'aspect-fold-500', 'adaptive-mono-nested'])

    def test_saved_profile_is_only_a_hint_and_restart_clears_stale_mode(self):
        decoder = self.make_dispatcher()
        self.assertTrue(decoder.start_with('adaptive-stereo-nested', side=0))
        self.assertEqual(decoder.startup_hint, (ADAPTIVE_STEREO_MODE, 0))
        self.assertEqual(decoder.active_mode, 1)
        decoder.active_mode, decoder.active_side = ADAPTIVE_STEREO_MODE, 0
        decoder.dispatch_mode, decoder.dispatch_side = ADAPTIVE_STEREO_MODE, 0
        decoder._reset_aspect_wires = lambda: None
        decoder._reset_capture_positions = lambda: None
        decoder.reset_capture_timeline()
        self.assertEqual(decoder.active_mode, 1)
        self.assertIsNone(decoder.active_side)
        self.assertIsNone(decoder.dispatch_mode)
        self.assertIsNone(decoder.dispatch_side)
        self.assertFalse(decoder.aspect_mono_slices)

    def test_persisted_stereo_preference_does_not_select_stale_decoder_or_side(self):
        decoder = self.make_dispatcher()
        payload = ('{"version": 1, "profile": "adaptive-stereo-nested", '
                   '"side": 0}')
        class Saved:
            def read_text(self):
                return payload
        with patch('tools.v7_live._sticky_profile_path', return_value=Saved()):
            self.assertEqual(v7_live._load_sticky_profile(decoder),
                             'adaptive-stereo-nested')
        self.assertEqual(decoder.startup_hint, (ADAPTIVE_STEREO_MODE, 0))
        self.assertEqual(decoder.active_mode, 1)
        self.assertIsNone(decoder.active_side)

    def test_adaptive_stereo_dispatch_requests_the_other_input_leg(self):
        decoder = object.__new__(_AdaptiveProfileDecoder)
        decoder.dispatch_mode = ADAPTIVE_STEREO_MODE
        decoder.stereo_slice_modes = frozenset((ADAPTIVE_STEREO_MODE,))
        decoder.aspect_mono_mode = 3
        self.assertTrue(decoder.may_use_stereo_slices())
        decoder.dispatch_mode = 3
        decoder.dispatch_slice_metadata = True
        self.assertTrue(decoder.may_use_stereo_slices())
        decoder.dispatch_mode = 1
        self.assertFalse(decoder.may_use_stereo_slices())

    def test_adaptive_gui_does_not_override_os_channels_or_sample_rate(self):
        class SoundDevice:
            calls = []

            @classmethod
            def check_output_settings(cls, **kwargs):
                cls.calls.append(kwargs)

        device = OutputDevice(0, 'pulse', 2, 44100, 'ALSA')
        gui = SenderGui((device,), restore_preferences=False)
        gui._assign('profile', 'adaptive-stereo-nested')
        gui.settings.update(device=0, source='test')
        command = build_command(gui.settings, (device,), SoundDevice,
                                python='python')
        self.assertEqual(SoundDevice.calls[-1]['channels'], 2)
        self.assertEqual(SoundDevice.calls[-1]['samplerate'], 44100)
        self.assertEqual(command[command.index('--profile')+1],
                         'adaptive-stereo-nested')
        self.assertIn('--adaptive-guest-range', command)
        self.assertIn('--exact-inverse', command)
        self.assertIn('--chroma-detail', command)
        self.assertNotIn('--rate', command)
        self.assertNotIn('--modem-channels', command)


if __name__ == '__main__':
    unittest.main()
