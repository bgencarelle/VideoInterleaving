"""Splice-aware packet timing: digital time-stretch and pitch-shift."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT, ROOT/'test_modem_v7'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from animation_modem import v7                                          # noqa: E402
import tone_code                                                        # noqa: E402
from tools import v7_timing_bench as bench                              # noqa: E402

P = v7.PULSE_FRAME
BODY = 288                     # transport3.SYNC_LEN: body origin in a packet


def _cut(audio, at, count):
    return np.concatenate((audio[:at], audio[at+count:])).astype(np.float32)


def _repeat(audio, at, count):
    return np.concatenate((audio[:at], audio[at-count:at],
                           audio[at:])).astype(np.float32)


class SpliceTimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wire, _base, cls.model, cls.audio, _values = bench.encode(6)
        cls.reference = cls._decode(cls.audio)

    @classmethod
    def _decode(cls, audio, latest_only=False):
        with bench._Capture(), tone_code.coded_pilot_timing():
            results, info = v7.decode_pulse_stream(
                cls.model, audio, latest_only=latest_only,
                models={cls.model.encoding_type: cls.model},
                state=v7.PulseState(tail_memory=False), sample_rate=v7.RATE,
                pilot_timing='tone-seeded')
        return results

    def _quality(self, result):
        clean = self.wire.values(self.model, self.reference[-1])
        return bench._psnr(self.wire.values(self.model, result), clean)

    def test_clean_and_resampled_streams_never_use_the_splice_map(self):
        for audio in (self.audio, bench.resample(self.audio, 1.02),
                      bench.resample(self.audio, .97)):
            results = self._decode(audio)
            self.assertEqual([result.status for result in results],
                             ['received']*6)
            self.assertFalse(any('splice' in result.diag
                                 for result in results))

    def test_a_cut_inside_the_body_is_bridged(self):
        for count, symbol in ((300, 2), (480, 10), (480, 17)):
            at = 2*P+BODY+symbol*v7.SYM+40
            with self.subTest(count=count, symbol=symbol):
                results = self._decode(_cut(self.audio, at, count))
                self.assertEqual([result.status for result in results],
                                 ['received']*6)
                splice = results[2].diag['splice']
                self.assertAlmostEqual(splice['jump'], -count, delta=2)
                self.assertIn(len(splice['cuts']), (1, 2))
                # The cut lands somewhere in the symbols the splice damaged.
                self.assertLessEqual(abs(splice['cuts'][0]-symbol), 4)
                self.assertGreaterEqual(splice['damaged_symbols'], 1)
                self.assertLessEqual(splice['damaged_symbols'], 6)
                # The cut symbols' coefficients are missing for this frame;
                # everything else matches the clean decode.
                self.assertGreater(self._quality(results[2]), 15.0)
                self.assertGreater(self._quality(results[3]), 50.0)

    def test_a_repeated_chunk_loses_nothing(self):
        results = self._decode(_repeat(self.audio, 2*P+BODY+10*v7.SYM+40, 480))
        self.assertEqual([result.status for result in results],
                         ['received']*6)
        splice = results[2].diag['splice']
        self.assertAlmostEqual(splice['jump'], 480, delta=2)
        self.assertEqual(splice['damaged_symbols'], 0)
        self.assertGreater(self._quality(results[2]), 50.0)

    def test_pitch_without_tempo_reads_the_header_scale(self):
        shifted = bench.pitch(self.audio, 1.04)
        results = self._decode(shifted)
        good = [result for result in results if result.status != 'lost']
        self.assertGreaterEqual(len(good), 5)
        for result in good:
            self.assertAlmostEqual(result.diag['pulse_scale'], 1/1.04,
                                   delta=.002)

    def test_live_latest_only_decodes_the_spliced_packet(self):
        damaged = _cut(self.audio, 2*P+BODY+10*v7.SYM+40, 480)
        start = int(P*1.2)
        end = start+int(2.2*P)
        results = self._decode(damaged[start:end], latest_only=True)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, 'received')
        self.assertIn('splice', results[0].diag)

    def test_metadata_cut_is_predicted_from_its_neighbours_in_a_recording(self):
        # A repeat that straddles the metadata symbol leaves the body intact.
        damaged = _repeat(self.audio, 2*P+BODY+v7.FRAME+60, 300)
        results = self._decode(damaged)
        statuses = [result.status for result in results]
        self.assertEqual(statuses, ['received']*6)
        third = results[2]
        if third.diag.get('metadata_predicted'):
            self.assertEqual(third.diag['source_index'], 2)
            self.assertEqual(third.diag['tail_slice'],
                             results[1].diag['tail_slice']+1)

    def test_erased_symbols_are_excluded_from_the_channel_fit(self):
        captured = {}
        real = v7.decode_frame

        def decode(*args, **kwargs):
            if kwargs.get('erased_symbols') is not None:
                captured['kwargs'] = kwargs
            return real(*args, **kwargs)
        v7.decode_frame = decode
        try:
            self._decode(_cut(self.audio, BODY+12*v7.SYM+40, 400)[:3*P])
        finally:
            v7.decode_frame = real
        erased = captured.get('kwargs', {}).get('erased_symbols')
        self.assertIsNotNone(erased)
        self.assertEqual(erased.shape, (v7.F,))
        self.assertTrue(erased[12:15].any())
        self.assertFalse(erased[:10].any())

    def test_pilots_tell_a_symbol_from_its_neighbours(self):
        # A whole-symbol shift reads another symbol whose cyclic prefix
        # correlates just as well; only its pilots give it away.
        start = 2*P+BODY+v7.WIN
        windows = np.stack([self.audio[start+j*v7.SYM:start+j*v7.SYM+v7.N]
                            for j in range(v7.F)])
        symbols = np.arange(v7.F)
        channel = v7._pilot_channel(v7._pilot_observations(
            self.model, symbols[:6], windows[:6]))
        right = v7._pilot_identity_residual(self.model, symbols, windows,
                                            channel)
        self.assertLess(float(np.max(right)), .05)
        for shift in (1, 2, 3):
            wrong = v7._pilot_identity_residual(
                self.model, symbols[shift:], windows[:-shift], channel)
            with self.subTest(shift=shift):
                self.assertGreater(float(np.median(wrong)),
                                   v7.SPLICE_IDENTITY_MAX)

    def test_wsola_pitch_shift_decodes_some_packets(self):
        results = self._decode(bench.wsola_pitch(self.audio, 1.25))
        self.assertTrue(any('splice' in result.diag for result in results))
        self.assertTrue(any(result.status != 'lost' or
                            result.diag.get('displayable')
                            for result in results))


class TimingBenchTests(unittest.TestCase):
    def test_synthetic_channels_are_deterministic(self):
        audio = np.random.default_rng(0).standard_normal((20000, 2)).astype(
            np.float32)
        self.assertEqual(len(bench.splice(audio, .1)), len(bench.splice(audio, .1)))
        np.testing.assert_array_equal(bench.splice(audio, .1),
                                      bench.splice(audio, .1))
        self.assertLess(len(bench.splice(audio, .1)), len(audio))
        self.assertGreater(len(bench.splice(audio, -.1)), len(audio))
        self.assertAlmostEqual(len(bench.resample(audio, 2.0)), len(audio)/2,
                               delta=1)
        self.assertAlmostEqual(len(bench.pitch(audio, 1.05)), len(audio),
                               delta=600)
        names = [name for name, _channel in bench.conditions()]
        self.assertIn('clean', names)
        self.assertEqual(len(names), len(set(names)))



class SpliceRecoveryTests(unittest.TestCase):
    """Cut headers and live metadata, built on the same six-packet stream.

    A packet is its header to its end marker: one whose header is gone is
    not a packet, and nothing after the previous end marker is taken for
    one."""

    setUpClass = classmethod(SpliceTimingTests.setUpClass.__func__)
    _decode = classmethod(SpliceTimingTests._decode.__func__)
    _quality = SpliceTimingTests._quality

    def test_a_packet_with_a_cut_header_is_not_shown_and_the_rest_are(self):
        damaged = self.audio.copy()
        damaged[2*P+16:2*P+250] = 0          # the third packet's header
        results = self._decode(damaged)
        self.assertEqual([(result.status, result.diag['source_index'])
                          for result in results],
                         [('received', index) for index in (0, 1, 3, 4, 5)])

    def test_live_window_does_not_take_a_cut_header_for_a_packet(self):
        damaged = self.audio.copy()
        damaged[2*P+16:2*P+250] = 0
        window = damaged[int(.9*P):int(3.98*P)]
        results = self._decode(window, latest_only=True)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, 'received')
        self.assertEqual(results[0].diag['source_index'], 1)

    def test_live_window_predicts_metadata_from_the_previous_packet(self):
        damaged = _repeat(self.audio, 2*P+BODY+v7.FRAME+60, 300)
        window = damaged[int(1.1*P):int(3.3*P)+300]
        results = self._decode(window, latest_only=True)
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.status, 'received')
        self.assertEqual(result.diag['source_index'], 2)


if __name__ == '__main__':
    unittest.main()
