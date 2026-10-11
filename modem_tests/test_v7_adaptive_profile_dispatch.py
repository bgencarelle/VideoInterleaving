import unittest
from pathlib import Path
import sys

from animation_modem.transport3 import PROFILE_PREAMBLE_BITS
TEST_MODEM_PATH = str(Path(__file__).resolve().parents[1]/'test_modem_v7')
if TEST_MODEM_PATH not in sys.path:
    sys.path.insert(0, TEST_MODEM_PATH)
from tone_code import (ADAPTIVE_NESTED, FOLD_MODE_TO_SLOTS,
                       STATUS_WORD_BY_MODE, encode_status)
from tools.v7_live import _AdaptiveProfileDecoder
from tools.v7_send_gui import OutputDevice, SenderGui, build_command
from tools.v7_profile_registry import ADAPTIVE_NESTED_MODE


class AdaptiveProfileDispatchTests(unittest.TestCase):
    def test_id_two_is_a_distinct_adaptive_wire_identity_not_fold_1000(self):
        self.assertEqual(ADAPTIVE_NESTED_MODE, 2)
        self.assertEqual(ADAPTIVE_NESTED, ADAPTIVE_NESTED_MODE)
        self.assertIn(2, PROFILE_PREAMBLE_BITS)
        self.assertEqual(encode_status(2), STATUS_WORD_BY_MODE[2])
        self.assertNotIn(2, FOLD_MODE_TO_SLOTS)

    def test_receiver_switches_between_new_and_legacy_ids(self):
        decoder = object.__new__(_AdaptiveProfileDecoder)
        decoder.active_mode = 1
        decoder.active_side = None
        decoder.candidate = None
        decoder.streak = 0
        decoder.candidate_last_seen = None
        decoder.candidate_scale = None
        decoder.last_packet = None
        decoder.last_decoded_packet = None
        decoder.state = None
        decoder.stereo_tail_memory = True
        decoder.on_switch = None
        decoder._mode_names = {1: 'fold-500', 2: 'adaptive-mono-nested',
                               3: 'aspect-mono-500'}
        decoder.mono_status_modes = frozenset((2, 3))
        decoder._last_slices = False
        decoder._reset_aspect_wires = lambda: None
        decoder.generation = 0

        for packet in range(3):
            decision = decoder._observe(packet, 1.0, 2, 1, float(packet))
        self.assertTrue(decision['confirmed'])
        self.assertEqual(decoder.active_mode, ADAPTIVE_NESTED_MODE)

        for packet in range(3, 6):
            decision = decoder._observe(packet, 1.0, 3, 1, float(packet))
        self.assertTrue(decision['confirmed'])
        self.assertEqual(decoder.active_mode, 3)

    def test_adaptive_gui_does_not_override_os_channels_or_sample_rate(self):
        class SoundDevice:
            calls = []

            @classmethod
            def check_output_settings(cls, **kwargs):
                cls.calls.append(kwargs)

        device = OutputDevice(0, 'pulse', 2, 44100, 'ALSA')
        gui = SenderGui((device,), restore_preferences=False)
        gui.settings.update(device=0, source='test',
                            profile='adaptive-stereo-nested')
        command = build_command(gui.settings, (device,), SoundDevice,
                                python='python')
        self.assertEqual(SoundDevice.calls[-1]['channels'], 2)
        self.assertEqual(SoundDevice.calls[-1]['samplerate'], 44100)
        self.assertNotIn('--rate', command)
        self.assertNotIn('--modem-channels', command)


if __name__ == '__main__':
    unittest.main()
