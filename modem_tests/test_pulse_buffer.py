"""Pulse-only timing and bounded capture/decoder handoff."""
import threading
import unittest
from unittest.mock import patch

import numpy as np

from animation_modem import transport3 as v3
from animation_modem.audio_buffer import AudioBuffer
from animation_modem.imaging import plane_shapes


class PulseTimingTests(unittest.TestCase):
    def test_speed_changes_and_dropouts_need_no_correlation(self):
        layout = v3.WIRE
        coder = v3.SourceCoder(plane_shapes('color-dct'))
        values = np.random.default_rng(15).uniform(-.2, .2, coder.count)
        speeds = (.5, .8, 1.0, 1.1, .75, 1.0)
        blocks = []
        for n, speed in enumerate(speeds, 1):
            packet = v3.encode(values, layout, coder, n, n, len(speeds))
            positions = np.arange(int(len(packet)/speed))*speed
            warped = np.stack([np.interp(positions, np.arange(len(packet)), packet[:, c])
                               for c in range(2)], axis=1)
            blocks.extend([np.zeros((600, 2)), warped])
        rx = v3.Receiver(layout, coder)
        with patch.object(v3, '_fit_preamble', side_effect=AssertionError('waveform fitting')), \
             patch.object(v3, 'preamble_correlation', side_effect=AssertionError('correlation')), \
             patch.object(rx, '_correlate', side_effect=AssertionError('speed sweep')):
            got = rx.feed(np.concatenate(blocks)) + rx.flush()
        self.assertEqual([r.absolute for r in got], list(range(1, 7)))
        for r, speed in zip(got, speeds):
            self.assertEqual(r.extra['timing_method'], 'pulse')
            self.assertAlmostEqual(r.extra['playback_speed'], speed, delta=.003)
            self.assertEqual(r.identity, 'verified_header')

    def test_silence_and_noise_never_trigger_a_speed_search(self):
        layout = v3.WIRE
        rx = v3.Receiver(layout, v3.SourceCoder(plane_shapes('color-dct')))
        noise = np.random.default_rng(12).normal(0, .03, (8000, 2))
        with patch.object(rx, '_correlate', side_effect=AssertionError('speed sweep')):
            self.assertEqual(rx.feed(noise)+rx.flush(), [])
        self.assertEqual(rx.correlation_hits, 0)


class AudioBufferTests(unittest.TestCase):
    def test_batches_preserve_samples_and_own_their_memory(self):
        buffer = AudioBuffer(100, 40)
        audio = np.ones((20, 2), np.float32)
        buffer.put(audio)
        audio[:] = 2
        self.assertIsNone(buffer.take(timeout=0))
        buffer.put(audio)
        batch, gap = buffer.take(timeout=0)
        np.testing.assert_array_equal(batch[:20], 1)
        np.testing.assert_array_equal(batch[20:], 2)
        self.assertFalse(gap)

    def test_slow_decoder_has_bounded_backlog_and_explicit_gap(self):
        buffer = AudioBuffer(60, 20)
        for n in range(4):
            buffer.put(np.full((20, 2), n))
        batch, gap = buffer.take(timeout=0)
        self.assertTrue(gap)
        self.assertEqual(buffer.dropped_samples, 20)
        np.testing.assert_array_equal(batch[:, 0], np.repeat([1, 2, 3], 20))
        buffer.put(np.full((20, 2), 4))
        self.assertFalse(buffer.take(timeout=0)[1])

    def test_device_overflow_discards_audio_before_the_gap(self):
        buffer = AudioBuffer(100, 40)
        buffer.put(np.zeros((20, 2)))
        buffer.put(np.ones((40, 2)), overflowed=True)
        batch, gap = buffer.take(timeout=0)
        self.assertTrue(gap)
        self.assertEqual(buffer.input_overflows, 1)
        np.testing.assert_array_equal(batch, 1)

    def test_close_wakes_waiting_decoder_and_releases_partial_batch(self):
        buffer = AudioBuffer(100, 40)
        buffer.put(np.zeros((10, 2)))
        got = []
        worker = threading.Thread(target=lambda: got.append(buffer.take(timeout=5)))
        worker.start()
        buffer.close()
        worker.join(timeout=1)
        self.assertFalse(worker.is_alive())
        self.assertEqual(got[0][0].shape, (10, 2))
        self.assertIsNone(buffer.take(timeout=0))


class FreshnessTests(unittest.TestCase):
    def test_long_decoder_pause_retains_only_recent_frames(self):
        layout = v3.WIRE
        coder = v3.SourceCoder(plane_shapes('color-dct'))
        buffer = AudioBuffer(3*layout.frame, layout.frame)
        audio = np.concatenate([
            *[v3.encode(np.zeros(coder.count), layout, coder, n, n, 30)
              for n in range(1, 31)],
            np.zeros((137, 2))]).astype(np.float32)
        # Capture continues while the decoder does no work for 30 frames.
        for offset in range(0, len(audio), 256):
            buffer.put(audio[offset:offset+256])
        recent, gap = buffer.take(timeout=0)
        self.assertTrue(gap)
        np.testing.assert_array_equal(recent, audio[-3*layout.frame:])
        rx = v3.Receiver(layout, coder)
        decoded = rx.feed(recent)+rx.flush()
        self.assertEqual([r.absolute for r in decoded], [29, 30])
        self.assertEqual(buffer.dropped_samples, len(audio)-len(recent))

    def test_queue_trims_part_of_an_old_capture_block(self):
        buffer = AudioBuffer(10, 4)
        buffer.put(np.arange(8)[:, None])
        buffer.put(np.arange(8, 15)[:, None])
        audio, gap = buffer.take(timeout=0)
        np.testing.assert_array_equal(audio[:, 0], np.arange(5, 15))
        self.assertTrue(gap)
        self.assertEqual(buffer.dropped_samples, 5)
