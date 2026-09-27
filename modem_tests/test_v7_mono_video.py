"""All-fresh mono video layout and 500-class fold profile."""
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT/'test_modem_v7', ROOT/'tools'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from animation_modem import v7                                       # noqa: E402
from animation_modem.v7_live_input import LiveInput                    # noqa: E402
from common import TARGET                                                # noqa: E402
from mono_video import (FRESH_SLOTS, FOLDED_GUESTS, FOLD_SLOTS, HEAD_GROUPS,
                        MONO_VIDEO_MODE, MonoFreshFoldOffWire,
                        MonoFreshFoldWire, fresh_rank_tables)            # noqa: E402
from mono_wire import MonoWire                                            # noqa: E402
from tools import v7_live                                                  # noqa: E402


PACKETS = 8


class MonoVideoWireTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = v7.load_model(TARGET, 'box')
        with Image.open(v7.REFERENCE_FIXTURE) as image:
            cls.image = image.convert('RGB')
        cls.values, _ = v7_live._values(
            cls.model, cls.image, 'box', brightness=1.0)
        cls.wire = MonoFreshFoldWire(cls.model)
        cls.mono_model = cls.wire.model_for(cls.model)
        cls.audio = cls.wire.encode(cls.model, [cls.values]*PACKETS)

    def _decode(self, audio):
        with self.wire.receiving():
            return v7.decode_pulse_stream(
                self.mono_model, audio, sample_rate=v7.RATE,
                pilot_timing='tone-seeded', frame_boundary='eof',
                state=v7.PulseState(tail_memory=False))

    def test_rank_map_is_fixed_and_every_slot_is_m_only(self):
        tables = fresh_rank_tables(self.model)
        self.assertEqual(len(tables), v7.TAIL_PHASES)
        for table in tables:
            np.testing.assert_array_equal(table, tables[0])
            self.assertEqual(int(np.count_nonzero(table >= 0)), FRESH_SLOTS)
            np.testing.assert_array_equal(
                np.sort(table[table >= 0]), np.sort(self.model.order[:FRESH_SLOTS]))
            np.testing.assert_array_equal(
                np.sort(table[:HEAD_GROUPS].ravel()),
                np.sort(self.model.order[:v7.HEAD]))
            np.testing.assert_array_equal(
                np.sort(table[HEAD_GROUPS:FRESH_SLOTS//8].ravel()),
                np.sort(self.model.order[v7.HEAD:FRESH_SLOTS]))
        self.assertTrue(all(stream == 'M'
                            for _, stream, _ in v7.GROUPS[:FRESH_SLOTS//8]))

    def test_fold_uses_current_fresh_hosts_and_next_corner_ranks(self):
        codec = self.wire._codec(self.mono_model)
        rank_by_index = np.empty(len(self.model.order), dtype=int)
        rank_by_index[self.model.order] = np.arange(len(self.model.order))

        self.assertEqual(codec.M, FOLD_SLOTS)
        self.assertEqual(codec.signature, 16)
        self.assertEqual(FOLD_SLOTS-codec.signature, FOLDED_GUESTS)
        self.assertEqual(FRESH_SLOTS-codec.signature+FOLDED_GUESTS, 1732)
        np.testing.assert_array_equal(
            np.sort(rank_by_index[codec.hosts]), np.arange(764, 1264))
        np.testing.assert_array_equal(
            np.sort(rank_by_index[codec.guest_model_indices]),
            np.arange(1264, 1764))
        self.assertEqual(codec.table()['layout'], 'mono-fresh-500')

    def test_packets_are_identical_legs_and_require_the_new_status(self):
        packet = self.audio[:v7.PULSE_FRAME]
        np.testing.assert_array_equal(packet[:, 0], packet[:, 1])
        self.assertEqual(len(self.audio), PACKETS*v7.PULSE_FRAME)

        from tone_code import decode_tone_spectrum
        windows = packet[v7.PULSE.SYNC_LEN:v7.PULSE.SYNC_LEN+v7.FRAME]
        windows = windows.reshape(v7.F, v7.SYM, 2)[:, v7.WIN:v7.WIN+v7.N]
        spectrum = np.fft.rfft(windows, axis=1)*v7._receive_rotation(
            self.mono_model)
        status = decode_tone_spectrum(spectrum, self.mono_model)
        self.assertTrue(status['valid'])
        self.assertEqual(status['status']['mode'], MONO_VIDEO_MODE)

    def test_receiver_decodes_folded_video_packets_and_unfolds_guests(self):
        results, info = self._decode(self.audio)

        self.assertEqual(len(results), PACKETS, info)
        self.assertEqual(info.get('eof_markers_validated'), PACKETS)
        self.assertTrue(all(result.status != 'lost' for result in results))
        self.assertTrue(all(result.diag.get('wire_profile') == 'mono-fresh-500'
                            for result in results))
        self.assertTrue(all(result.diag.get('mono_fold_eq') is not None
                            for result in results))
        values = self.wire.values(self.mono_model, results[-1])
        self.assertEqual(values.shape, (self.mono_model.coder.source_count,))
        self.assertTrue(np.all(np.isfinite(values)))

    def test_receiver_rejects_previous_rotating_mono_layout(self):
        previous = MonoWire(self.model)
        old_audio = previous.encode(self.model, [self.values]*3)
        results, _info = self._decode(old_audio)

        self.assertEqual(len(results), 3)
        self.assertTrue(all(result.status == 'lost' for result in results))
        self.assertTrue(all(result.diag.get('mono_profile_rejected') ==
                            'unknown_or_non_mono_status'
                            for result in results))

    def test_receiver_rejects_stereo_fold_status(self):
        from tools.v7_wire_profile import WireProfile
        stereo = WireProfile('default').encode(
            self.model, [self.values]*2, source_indices=[0, 1])
        results, _info = self._decode(stereo)

        self.assertEqual(len(results), 2)
        self.assertTrue(all(result.status == 'lost' for result in results))
        self.assertTrue(all(result.diag.get('mono_profile_rejected') ==
                            'unknown_or_non_mono_status'
                            for result in results))

    def test_all_fresh_foldoff_control_has_a_distinct_coded_status(self):
        control = MonoFreshFoldOffWire(self.model)
        audio = control.encode(self.model, [self.values]*3)
        mono_model = control.model_for(self.model)
        with control.receiving():
            results, info = v7.decode_pulse_stream(
                mono_model, audio, sample_rate=v7.RATE,
                pilot_timing='tone-seeded', frame_boundary='eof',
                state=v7.PulseState(tail_memory=False))

        self.assertEqual(len(results), 3, info)
        self.assertTrue(all(result.status != 'lost' for result in results))
        self.assertTrue(all(result.diag['pilot_timing']['coded_status_mode'] ==
                            control.status_mode for result in results))
        self.assertEqual(control.fold_slots, 0)

    def test_video_profile_refuses_missing_eof(self):
        with self.assertRaisesRegex(ValueError, 'requires EOF markers'):
            self.wire.encode(self.model, [self.values], eof_marker=False)

    def test_selected_output_side_leaves_the_other_leg_silent_and_decodes(self):
        for side, index, silent in (('left', 0, 1), ('right', 1, 0)):
            with self.subTest(side=side):
                sender = MonoFreshFoldWire(self.model, side=side)
                audio = sender.encode(self.model, [self.values]*3)
                self.assertGreater(float(np.max(np.abs(audio[:, index]))), 0.0)
                np.testing.assert_array_equal(audio[:, silent], 0.0)

                live_input = LiveInput(rate=v7.RATE, direction='forward')
                live_input.add(audio[:, index:index+1].copy())
                buffered = live_input.take(1.0)
                self.assertIsNotNone(buffered)
                self.assertGreaterEqual(len(live_input.pulse_hits(buffered)), 1)

                receiver = MonoFreshFoldWire(self.model, side=side)
                mono_model = receiver.model_for(self.model)
                with receiver.receiving():
                    results, info = v7.decode_pulse_stream(
                        mono_model, audio[:, index], sample_rate=v7.RATE,
                        pilot_timing='tone-seeded', frame_boundary='eof',
                        state=v7.PulseState(tail_memory=False))
                self.assertEqual(len(results), 3, info)
                self.assertTrue(all(result.status != 'lost'
                                    for result in results))

    def test_sender_and_receiver_profile_flags_are_explicit(self):
        send = v7_live.parser().parse_args([
            'send', '--device', 'null', '--source', 'test',
            '--experimental-mono-fold'])
        recv = v7_live.parser().parse_args([
            'receive', '--device', 'null', '--experimental-mono-fold'])
        self.assertTrue(send.experimental_mono_fold)
        self.assertTrue(recv.experimental_mono_fold)
        self.assertEqual(send.mono_video_side, 'left')
        self.assertEqual(recv.mono_video_side, 'left')
        self.assertEqual(v7_live._fold_slots(send), 0)
        self.assertEqual(v7_live._fold_slots(recv), 0)
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            v7_live._fold_slots(type('Args', (), {
                'experimental_mono': True,
                'experimental_mono_fold': True,
                'experimental_fold': None,
                'baseline': False,
            })())


if __name__ == '__main__':
    unittest.main()
