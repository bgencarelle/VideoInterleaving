"""Regression tests for the isolated time-coded pilot prototype."""
from itertools import combinations
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from animation_modem import v7                                         # noqa: E402
from common import CASES, RATE, STEADY_FROM, impair                      # noqa: E402
from live_fold import LiveFold                                            # noqa: E402
from tone_code import (FOLD_500, FOLD_1000, FOLD_OFF, acquire_packet_starts,
                       add_tone_code, coded_pilot_timing, decode_status,
                       decode_tone_code, encode_packet, encode_status,
                       make_packet_stream,
                       match_tone_results_to_frames)                         # noqa: E402


TARGET = .1521/np.sqrt(1 + 10**(v7.CLOCK_REL_DB/10))


class ToneCodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = v7.load_model(TARGET, 'box')
        with Image.open(v7.REFERENCE_FIXTURE) as image:
            cls.values = v7.image_values(
                v7.prepare_image(image.convert('RGB'), 'box'), v7.V7_GRIDS, 'box')

    def test_distance_six_status_code_corrects_two_errors_and_five_erasures(self):
        expected_slots = {FOLD_OFF: 0, FOLD_500: 500, FOLD_1000: 1000}
        words = [encode_status(mode) for mode in expected_slots]
        distance = min(sum(left != right for left, right in zip(a, b))
                       for index, a in enumerate(words)
                       for b in words[index+1:])
        self.assertEqual(distance, 6)

        for mode, slots in expected_slots.items():
            word = encode_status(mode)
            with self.subTest(mode=mode):
                decoded = decode_status(word)
                self.assertEqual(decoded['mode'], mode)
                self.assertEqual(decoded['fold_slots'], slots)
                self.assertEqual(decoded['errors_corrected'], 0)
                self.assertEqual(decoded['erasures_filled'], 0)

                for count in (1, 2):
                    for indexes in combinations(range(12), count):
                        damaged = list(word)
                        for index in indexes:
                            damaged[index] ^= 1
                        recovered = decode_status(damaged)
                        self.assertIsNotNone(recovered)
                        self.assertEqual(recovered['mode'], mode)
                        self.assertEqual(recovered['errors_corrected'], count)

                for count in range(1, 6):
                    for indexes in combinations(range(12), count):
                        erased = list(word)
                        for index in indexes:
                            erased[index] = None
                        recovered = decode_status(erased)
                        self.assertIsNotNone(recovered)
                        self.assertEqual(recovered['mode'], mode)
                        self.assertEqual(recovered['erasures_filled'], count)

    def test_status_registry_fails_closed(self):
        with self.assertRaises(ValueError):
            encode_status(3)
        self.assertIsNone(decode_status((0, 1, .5) + (0,)*9))
        self.assertIsNone(decode_status((0,)*11))
        self.assertIsNone(decode_status((None,)*6 + encode_status(FOLD_500)[6:]))

    def test_clean_stream_decodes_every_status_and_existing_v7_frames(self):
        statuses = [FOLD_OFF, FOLD_500, FOLD_1000]
        wire = make_packet_stream(self.model, [self.values]*len(statuses), statuses)
        capture = resample_poly(wire, 2, 1, axis=0).astype(np.float32)
        starts = acquire_packet_starts(capture, limit=len(statuses))
        self.assertEqual(len(starts), len(statuses))
        decoded = [decode_tone_code(capture, start, scale, RATE)
                   for start, scale, _ in starts]
        for result, mode in zip(decoded, statuses):
            expected = decode_status(encode_status(mode))
            self.assertTrue(result['valid'], result['reason'])
            self.assertEqual(result['status'], expected)
            self.assertAlmostEqual(result['pilot_score'], 1.0, places=2)

        # The ordinary pulse receiver must still accept packets carrying the
        # experimental tone overlay; tone decoding does not hook or alter it.
        received, _ = v7.decode_pulse_stream(
            self.model, wire[:4*v7.PULSE_FRAME], sample_rate=v7.RATE)
        self.assertGreater(len(received), 0)
        self.assertTrue(all(frame.status == 'received' for frame in received))

    def test_status_pairing_uses_frame_position_when_a_scan_misses_a_packet(self):
        decoded = [SimpleNamespace(diag={'frame_start': float(start)})
                   for start in (1000, 5000, 9000)]
        starts = [(1000.0, 2.0, .9), (9000.0, 2.0, .9)]
        statuses = [{'status': {'mode': FOLD_500}},
                    {'status': {'mode': FOLD_1000}}]
        paired, matched = match_tone_results_to_frames(decoded, starts, statuses)
        self.assertEqual(matched, 2)
        self.assertEqual(paired[id(decoded[0])]['status']['mode'], FOLD_500)
        self.assertNotIn(id(decoded[1]), paired)
        self.assertEqual(paired[id(decoded[2])]['status']['mode'], FOLD_1000)

    def test_coded_chip_signs_are_removed_before_v7_tone_timing(self):
        packets = [encode_packet(self.model, self.values, index+1, FOLD_500,
                                 source_index=index)
                   for index in range(5)]
        wire = np.concatenate(packets)
        with coded_pilot_timing():
            received, _ = v7.decode_pulse_stream(
                self.model, wire, sample_rate=v7.RATE,
                pilot_timing='tone-seeded')
        timing = [result.diag.get('pilot_timing', {}) for result in received]
        self.assertTrue(any(metrics.get('coded_chips_removed') for metrics in timing))
        self.assertTrue(any(
            metrics.get('coded_chips_removed') and
            metrics.get('mode_applied') == 'tone-seeded'
            for metrics in timing))

    def test_steady_and_absent_tones_do_not_decode_as_the_coded_channel(self):
        for pilot_tones in (False, True):
            packet = v7.encode_pulse_frame(
                self.model, self.values, 1, pilot_tones=pilot_tones,
                eof_marker=True)
            capture = resample_poly(packet, 2, 1, axis=0).astype(np.float32)
            starts = acquire_packet_starts(capture, limit=1)
            self.assertEqual(len(starts), 1)
            result = decode_tone_code(capture, starts[0][0], starts[0][1], RATE)
            self.assertFalse(result['valid'])
            self.assertIsNone(result['status'])

    def test_bin1_timing_correction_recovers_status_under_flutter(self):
        packet = encode_packet(self.model, self.values, 1, FOLD_500)
        wire = resample_poly(packet, 2, 1, axis=0).astype(np.float32)
        case = next(case for case in CASES if case.name == 'wow-flutter')
        capture = impair(wire, case)
        starts = acquire_packet_starts(capture, limit=1)
        self.assertEqual(len(starts), 1)
        result = decode_tone_code(capture, starts[0][0], starts[0][1], RATE)
        self.assertTrue(result['valid'], result['reason'])
        self.assertEqual(result['status']['mode'], FOLD_500)
        self.assertIn(result['timing_window'], (7, 9))
        self.assertGreaterEqual(result['pilot_score'], .60)

    def test_distance_code_corrects_mains_buzz_errors(self):
        packets = [encode_packet(self.model, self.values, index+1, FOLD_500,
                                 source_index=index)
                   for index in range(8)]
        wire = resample_poly(np.concatenate(packets), 2, 1,
                             axis=0).astype(np.float32)
        case = next(case for case in CASES if case.name == 'mains-buzz')
        capture = impair(wire, case)
        starts = acquire_packet_starts(capture, limit=8)
        self.assertEqual(len(starts), 8)
        results = [decode_tone_code(capture, start, scale, RATE)
                   for start, scale, _ in starts]
        self.assertTrue(all(
            result['valid'] and result['status']['mode'] == FOLD_500
            for result in results))
        self.assertTrue(any(result['status_errors_corrected'] for result in results))

    def test_missing_data_tone_is_recovered_as_a_codeword_erasure(self):
        packet = encode_packet(self.model, self.values, 1, FOLD_500)
        clean = decode_tone_code(packet)
        self.assertTrue(clean['valid'], clean['reason'])
        missing_symbol = 1
        start = v7.PULSE.SYNC_LEN+missing_symbol*v7.SYM
        packet[start:start+v7.SYM] = 0
        result = decode_tone_code(packet, frame_start=0, frame_scale=1,
                                  sample_rate=v7.RATE)
        self.assertTrue(result['valid'], result['reason'])
        self.assertEqual(result['status']['mode'], FOLD_500)
        self.assertEqual(result['status_erasures_filled'], 1)
        self.assertIn(missing_symbol, result['erasure_symbols'])

    def test_coded_tones_leave_pinned_fold_signature_and_noise_measurement(self):
        for slots, mode in ((500, FOLD_500), (1000, FOLD_1000)):
            with self.subTest(fold=slots):
                fold = LiveFold(slots)
                folded = fold.encode(self.model, self.values)
                codec = fold.codec(self.model)

                def receive(coded):
                    packets = []
                    for index in range(12):
                        counter = index+1
                        packet = v7.encode_pulse_frame(
                            self.model, folded, counter, source_index=index,
                            eof_marker=True)
                        if coded:
                            packet = add_tone_code(
                                packet, counter, encode_status(mode))
                        packets.append(packet)
                    capture = resample_poly(
                        np.concatenate(packets), 2, 1, axis=0).astype(np.float32)
                    fold.install()
                    try:
                        received, _ = v7.decode_pulse_stream(
                            self.model, capture, sample_rate=RATE,
                            pilot_timing='baseline', frame_boundary='eof')
                    finally:
                        fold.uninstall()
                    scores, noises = [], []
                    for result in received:
                        if (result.counter < STEADY_FROM or
                                result.status not in ('received', 'verified')):
                            continue
                        codec.last_score = codec.last_noise = None
                        fold.values(self.model, result)
                        if codec.last_score is not None:
                            scores.append(codec.last_score)
                        if codec.last_noise is not None:
                            noises.append(codec.last_noise)
                    return received, scores, noises, capture

                baseline, base_scores, base_noises, _ = receive(False)
                coded, code_scores, code_noises, capture = receive(True)
                starts = acquire_packet_starts(capture, limit=12)
                status_results = [decode_tone_code(capture, start, scale, RATE)
                                  for start, scale, _ in starts]
                self.assertEqual(len(baseline), 12)
                self.assertEqual(len(coded), 12)
                self.assertEqual(len(base_scores), 12-STEADY_FROM+1)
                self.assertEqual(len(code_scores), len(base_scores))
                self.assertGreater(min(base_scores+code_scores), .9)
                self.assertLess(abs(float(np.mean(base_noises))-float(
                    np.mean(code_noises))), .01)
                self.assertEqual(len(status_results), 12)
                for index, result in enumerate(status_results):
                    self.assertTrue(result['valid'], result['reason'])
                    self.assertEqual(result['status']['mode'], mode)


if __name__ == '__main__':
    unittest.main()
