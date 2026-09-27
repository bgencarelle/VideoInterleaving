"""Experimental V7 mono fold-off layout and coded receiver gating."""
import sys
import types
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT/'test_modem_v7', ROOT/'tools'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from animation_modem import v7                                      # noqa: E402
from tone_code import (FOLD_OFF, MONO_OFF, STATUS_WORD_BY_MODE,
                       decode_status)                                    # noqa: E402
from mono_wire import (FRESH, MONO_GROUPS, MONO_PILOT_VALUES, MonoWire,
                       mono_rank_tables)                                  # noqa: E402
from tools import v7_live                                                  # noqa: E402


TARGET = .1521/np.sqrt(1 + 10**(v7.CLOCK_REL_DB/10))
PACKETS = 8


class V7MonoWireTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = v7.load_model(TARGET, 'nearest')
        with Image.open(v7.REFERENCE_FIXTURE) as image:
            cls.values = v7.image_values(
                v7.prepare_image(image, 'nearest'), cls.model.coder.grids,
                'nearest')
        cls.wire = MonoWire(cls.model)
        cls.mono_model = cls.wire.model_for(cls.model)
        cls.audio = cls.wire.encode(cls.model, [cls.values]*PACKETS)

    def _decode(self, audio):
        with self.wire.receiving():
            return v7.decode_pulse_stream(
                self.mono_model, audio, sample_rate=v7.RATE,
                pilot_timing='tone-seeded', frame_boundary='eof')

    def test_six_status_words_are_balanced_and_distance_six(self):
        words = list(STATUS_WORD_BY_MODE.values())
        self.assertEqual(len(words), 6)
        for word in words:
            self.assertEqual(sum(word), 6)
            self.assertEqual(decode_status(word)['mode'],
                             next(mode for mode, candidate in
                                  STATUS_WORD_BY_MODE.items()
                                  if candidate == word))
        for index, left in enumerate(words):
            for right in words[index+1:]:
                self.assertGreaterEqual(
                    sum(a != b for a, b in zip(left, right)), 6)
        self.assertEqual(decode_status(STATUS_WORD_BY_MODE[MONO_OFF])['mode'],
                         MONO_OFF)

    def test_rank_tables_cover_all_corner_coefficients_in_seven_phases(self):
        tables = mono_rank_tables(self.model)
        self.assertEqual(len(tables), 7)
        self.assertTrue(all(stream == 'M'
                            for _, stream, _ in v7.GROUPS[:MONO_GROUPS]))
        for phase, ranks in enumerate(tables):
            self.assertEqual(int(np.count_nonzero(ranks >= 0)),
                             1264 if phase < 6 else 1248)
            self.assertTrue(np.array_equal(
                np.sort(ranks[ranks >= 0]),
                np.unique(ranks[ranks >= 0])))
            np.testing.assert_array_equal(
                np.sort(ranks[:v7.N_HEAD_G][ranks[:v7.N_HEAD_G] >= 0]),
                np.sort(self.model.order[:v7.HEAD]))
        covered = np.unique(np.concatenate([
            ranks[ranks >= 0] for ranks in tables]))
        np.testing.assert_array_equal(np.sort(covered),
                                      np.arange(len(self.model.order)))
        self.assertEqual(FRESH, 992)
        self.assertTrue(np.all(tables[0][MONO_GROUPS:] == -1))

    def test_mono_encoder_duplicates_legs_and_places_pilots_on_m_only(self):
        np.testing.assert_array_equal(MONO_PILOT_VALUES[..., 1], 0)
        packet = self.audio[:v7.PULSE_FRAME]
        np.testing.assert_array_equal(packet[:, 0], packet[:, 1])
        self.assertEqual(len(self.audio), PACKETS*v7.PULSE_FRAME)

    def test_receiver_decodes_stereo_mono_sum_and_either_leg(self):
        mono_sum = self.audio[:, 0]
        for label, capture in (
                ('stereo', self.audio), ('mono-sum', mono_sum),
                ('left', self.audio[:, 0]), ('right', self.audio[:, 1])):
            with self.subTest(path=label):
                results, info = self._decode(capture)
                self.assertEqual(len(results), PACKETS, info)
                self.assertTrue(all(result.status != 'lost' for result in results))
                self.assertTrue(all(result.diag.get('metadata_valid')
                                    for result in results))
                self.assertTrue(all(result.diag.get('wire_profile') ==
                                    'mono-fold-off' for result in results))
                self.assertTrue(all(result.diag.get('pilot_timing', {}).get(
                    'coded_status_mode') == MONO_OFF for result in results))

    def test_reverse_packets_keep_the_mono_profile_and_metadata_gate(self):
        reverse = self.audio[::-1].copy()
        hits = v7.pulse_frame_hits(reverse)
        self.assertEqual(len(hits), PACKETS)
        state = v7.PulseState()
        received = 0
        with self.wire.receiving():
            for start, scale, _confidence, direction in hits:
                self.assertEqual(direction, -1)
                results, info = v7.decode_reverse_packet(
                    self.mono_model, reverse, start, scale, state=state,
                    sample_rate=v7.RATE, pilot_timing='tone-seeded')
                self.assertEqual(info.get('eof_markers_validated'), 1)
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0].diag.get('pilot_timing', {}).get(
                    'coded_status_mode'), MONO_OFF)
                self.assertTrue(results[0].diag.get('metadata_valid'))
                if results[0].status == 'lost':
                    self.assertEqual(
                        results[0].diag.get('reverse_rejected'),
                        'metadata_not_independently_valid')
                    self.assertIn(results[0].diag.get('tail_slice'), (5, 6))
                else:
                    received += 1
                    self.assertEqual(results[0].diag.get('wire_profile'),
                                     'mono-fold-off')
        self.assertGreaterEqual(received, PACKETS-2)

    def test_fold_off_stereo_status_is_rejected_before_picture_decode(self):
        from tools.v7_wire_profile import WireProfile
        stereo = WireProfile('default', fold=False)
        normal_model = v7.load_model(TARGET, 'nearest')
        wire = stereo.encode(normal_model, [self.values]*3)
        results, _ = self._decode(wire)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(result.status == 'lost' for result in results))
        self.assertTrue(all(not result.diag.get('displayable', False)
                            for result in results))
        self.assertTrue(all(result.diag.get('mono_profile_rejected') ==
                            'unknown_or_non_mono_status'
                            for result in results))

    def test_live_receiver_selects_mono_profile_and_restores_hooks(self):
        args = v7_live.parser().parse_args([
            'receive', '--device', 'null', '--experimental-mono'])
        original_decode = v7.decode_frame
        original_timing = v7.pilot_tone_timing
        with mock.patch.object(v7_live, '_run_receive', return_value='stub') as run:
            self.assertEqual(v7_live.run_receive(args), 'stub')
        self.assertIs(v7.decode_frame, original_decode)
        self.assertIs(v7.pilot_tone_timing, original_timing)
        self.assertIsNone(run.call_args.args[1])
        self.assertIsInstance(run.call_args.args[2], MonoWire)

    def test_sender_profile_uses_neutral_mono_fold_off_defaults(self):
        args = v7_live.parser().parse_args([
            'send', '--device', 'null', '--experimental-mono',
            '--source', 'test'])
        self.assertEqual(v7_live._fold_slots(args), 0)
        self.assertEqual(v7_live._send_profile(args, 0), ('nearest', 1.05))

    def test_live_sender_emits_the_mono_profile_without_an_audio_device(self):
        written = []

        class OutputStream:
            samplerate = float(v7.RATE)

            def __init__(self, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

            def write(self, audio):
                written.append(np.array(audio, dtype=np.float32, copy=True))

        fake_sounddevice = types.ModuleType('sounddevice')
        fake_sounddevice.OutputStream = OutputStream
        with Image.open(v7.REFERENCE_FIXTURE) as image:
            still = np.asarray(image.convert('RGB'))

        previous_sounddevice = sys.modules.get('sounddevice')
        previous_capture = v7_live._capture
        sys.modules['sounddevice'] = fake_sounddevice
        v7_live._capture = lambda _args: (lambda: still)
        try:
            args = v7_live.parser().parse_args([
                'send', '--device', 'memory', '--source', 'test',
                '--seconds', '0.25', '--no-log', '--experimental-mono'])
            v7_live.run_send(args)
        finally:
            v7_live._capture = previous_capture
            if previous_sounddevice is None:
                sys.modules.pop('sounddevice', None)
            else:
                sys.modules['sounddevice'] = previous_sounddevice

        self.assertTrue(written)
        audio = np.concatenate(written)
        np.testing.assert_allclose(audio[:, 0], audio[:, 1], atol=1e-7)
        results, _ = self._decode(audio)
        self.assertTrue(results)
        self.assertTrue(all(result.diag.get('wire_profile') == 'mono-fold-off'
                            for result in results if result.status != 'lost'))


if __name__ == '__main__':
    unittest.main()
