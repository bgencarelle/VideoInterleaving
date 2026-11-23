"""Capture-cache invalidation and fused callback ownership/metering checks."""
from types import SimpleNamespace
import unittest

import numpy as np

from animation_modem.v7_input_kernels import select_and_measure
from animation_modem.v7_channel_align import ChannelAligner
from tools.v7_frame_cache import PreparedFrameCache, preparation_settings
from tools.v7_capture import FilePlayback


class FrameCacheTests(unittest.TestCase):
    def test_paused_playback_keeps_picture_generation_and_seek_replaces_it(self):
        def open_reader(start):
            return lambda: np.full((2, 3, 3), int(start)+1, np.uint8)
        playback = FilePlayback(open_reader, duration=20, fps=1, idle=0)
        try:
            first = playback()
            generation = playback.picture_generation
            playback.pause()
            self.assertIs(playback(), first)
            self.assertEqual(playback.picture_generation, generation)
            playback.seek(5)
            changed = playback()
            self.assertGreater(playback.picture_generation, generation)
            self.assertIsNot(changed, first)
            np.testing.assert_array_equal(changed, 6)
        finally:
            playback.close()

    def test_generation_settings_and_source_identity_invalidate_and_replay_gains(self):
        cache = PreparedFrameCache()
        source = np.zeros((4, 4, 3), np.uint8)
        calls = []

        def compute(gains):
            calls.append(True)
            gains.append(np.array([len(calls)], float))
            return np.array([len(calls)], float), 2

        def prepare(frame=source, token=1, setting=(1.0,)):
            gains = []
            result = cache.prepare(frame, token, setting, compute, gains)
            return result, gains

        original, gains = prepare()
        repeated, replayed = prepare()
        self.assertIs(original, repeated)
        np.testing.assert_array_equal(gains[0], replayed[0])
        self.assertEqual(len(calls), 1)
        prepare(token=2)                    # source may reuse the same array
        prepare(token=2, setting=(1.1,))   # tone changes on a paused picture
        prepare(frame=source.copy(), token=2, setting=(1.1,))
        prepare(token=None)
        prepare(token=None)
        self.assertEqual(len(calls), 6)

    def test_kernel_reload_parameters_geometry_and_preview_are_in_key(self):
        kernel = object()
        selection = SimpleNamespace(kernel=kernel, values=(('gain', 1.0),))
        options = {'kernel': selection, 'clarity': 0.0}
        key = preparation_settings(1.0, 1.0, options)
        self.assertEqual(key, preparation_settings(1.0, 1.0, dict(options)))
        changes = [
            preparation_settings(1.1, 1.0, options),
            preparation_settings(1.0, .9, options),
            preparation_settings(1.0, 1.0, options, rotation=90),
            preparation_settings(1.0, 1.0, options, mirror=True),
            preparation_settings(1.0, 1.0, options, preview=True),
            preparation_settings(1.0, 1.0, dict(options, clarity=.2)),
            preparation_settings(1.0, 1.0, dict(options, kernel=SimpleNamespace(
                kernel=object(), values=selection.values))),
            preparation_settings(1.0, 1.0, dict(options, kernel=SimpleNamespace(
                kernel=kernel, values=(('gain', .8),)))),
        ]
        for changed in changes:
            self.assertNotEqual(key, changed)


class InputMeterTests(unittest.TestCase):
    def test_selected_samples_are_owned_and_metering_matches_float64_reference(self):
        rng = np.random.default_rng(3141)
        for count in (1, 128, 1024):
            raw = rng.normal(size=(count, 8)).astype(np.float32)
            for channels in ((0,), (0, 1), (5, 2)):
                with self.subTest(count=count, channels=channels):
                    block, peak, rms, active = select_and_measure(raw, channels)
                    expected = raw[:, channels].astype(float)
                    np.testing.assert_array_equal(block, expected.astype(np.float32))
                    np.testing.assert_array_equal(peak, np.max(np.abs(expected), axis=0))
                    np.testing.assert_allclose(rms, np.sqrt(np.mean(expected*expected, axis=0)),
                                               rtol=2e-15, atol=0)
                    self.assertEqual(active, bool(np.max(np.abs(expected)) > 1e-5))
                    saved = block.copy()
                    raw[:] += 1
                    np.testing.assert_array_equal(block, saved)

    def test_activity_threshold_and_nonfinite_samples_keep_existing_policy(self):
        for value, expected in ((0, False), (1e-6, False), (.01, True),
                                (np.nan, False), (np.inf, True)):
            block = np.zeros((16, 2), np.float32)
            block[0, 0] = value
            _, peak, rms, active = select_and_measure(block, (0, 1))
            self.assertEqual(active, expected)
            if np.isnan(value):
                self.assertTrue(np.isnan(peak[0]))
                self.assertTrue(np.isnan(rms[0]))


class AlignmentFastPathTests(unittest.TestCase):
    def test_zero_offset_retains_float32_quantization_and_owned_history(self):
        raw = np.arange(20, dtype=float).reshape(10, 2)/7
        expected = raw.astype(np.float32).astype(float)
        state = ChannelAligner(48000)
        np.testing.assert_array_equal(state.process(raw), expected)
        raw[:] = 0
        np.testing.assert_array_equal(state.history, expected.astype(np.float32))

    def test_signed_delay_crosses_blocks_and_zero_offset_discards_pending(self):
        raw = np.arange(20, dtype=np.float32).reshape(10, 2)
        for offset in (-2, 2):
            with self.subTest(offset=offset):
                state = ChannelAligner(48000)
                state.offset = offset
                actual = np.concatenate([state.process(raw[:3]),
                                         state.process(raw[3:7]),
                                         state.process(raw[7:])])
                expected = raw.copy()
                lead = 0 if offset > 0 else 1
                expected[:2, lead] = 0
                expected[2:, lead] = raw[:-2, lead]
                np.testing.assert_array_equal(actual, expected)
                state.offset = 0
                np.testing.assert_array_equal(state.process(raw), raw)
                self.assertEqual(len(state.pending), 0)


if __name__ == '__main__':
    unittest.main()
