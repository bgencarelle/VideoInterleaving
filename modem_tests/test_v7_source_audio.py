"""Source-audio buffering and mono routing helpers stay device-independent."""
import io
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from animation_modem import v7
from tools.v7_source_audio import (DeviceSourceAudio, FFmpegSourceAudio,
                                   ClockMatchedReader, PacketAudioDelay,
                                   SampleBuffer, SharedVideoAudioSource)

ROOT = Path(__file__).resolve().parents[1]


class SourceAudioTests(unittest.TestCase):
    def test_bounded_fifo_returns_zeros_on_underflow_and_drops_oldest(self):
        fifo = SampleBuffer(5)
        fifo.push(np.arange(5, dtype=np.float32))
        fifo.push(np.arange(5, 8, dtype=np.float32))

        np.testing.assert_array_equal(fifo.read(5), np.arange(3, 8))
        np.testing.assert_array_equal(fifo.read(3), np.zeros(3))
        self.assertEqual(fifo.dropped, 3)

    def test_packet_delay_tracks_actual_packet_sample_count(self):
        delay = PacketAudioDelay(extra_delay_ms=1)
        source = np.arange(1, 13, dtype=np.float32)
        first = delay.apply(source[:4], packet_samples=4, sample_rate=1000)
        second = delay.apply(source[4:8], packet_samples=4, sample_rate=1000)
        third = delay.apply(source[8:], packet_samples=4, sample_rate=1000)

        np.testing.assert_array_equal(first, np.zeros(4))
        np.testing.assert_array_equal(second, np.zeros(1).tolist()+[1, 2, 3])
        np.testing.assert_array_equal(third, [4, 5, 6, 7])

    def test_packet_delay_does_not_reset_when_emitted_blocks_round_differently(self):
        delay = PacketAudioDelay()
        source = np.arange(1, 14, dtype=np.float32)
        blocks = (source[:4], source[4:9], source[9:])
        output = np.concatenate([
            delay.apply(block, packet_samples=len(block), sample_rate=1000)
            for block in blocks
        ])

        np.testing.assert_array_equal(output[:4], 0.0)
        np.testing.assert_array_equal(output[4:], source[:-4])

    def test_delay_uses_actual_packet_samples_at_rate_and_speed(self):
        counts = []
        for sample_rate, speed in ((44_100, 1.0), (48_000, 2.0),
                                   (96_000, .5)):
            packet_samples = len(v7.speed_pulse_stream(
                np.zeros(v7.PULSE_FRAME, dtype=np.float32), speed,
                rate=sample_rate))
            counts.append(packet_samples)
            delay = PacketAudioDelay()
            first = delay.apply(np.ones(packet_samples), packet_samples,
                                sample_rate)
            second = delay.apply(np.full(packet_samples, 2.0),
                                 packet_samples, sample_rate)
            self.assertEqual(delay.delay_samples, packet_samples)
            np.testing.assert_array_equal(first, 0.0)
            np.testing.assert_array_equal(second, 1.0)

        self.assertLess(counts[1], counts[0])
        self.assertLess(counts[0], counts[2])

    def test_clock_match_tracks_a_synthetic_external_clock_drift(self):
        target = 2000
        fifo = SampleBuffer(max_samples=target*2)
        initial = np.arange(target, dtype=np.float32)
        fifo.push(initial)
        reader = ClockMatchedReader(fifo, target)
        next_sample = target
        source_phase = 0.0
        levels = []

        # 40,000 x 100 samples models over 80 seconds at 48 kHz with a
        # persistent +500 ppm capture-clock error.
        for _ in range(40_000):
            output = reader.read(100)
            self.assertEqual(output.shape, (100,))
            self.assertTrue(np.all(np.isfinite(output)))
            source_phase += 100.05
            count = int(source_phase)
            source_phase -= count
            fifo.push(np.arange(next_sample, next_sample+count,
                                dtype=np.float32))
            next_sample += count
            levels.append(fifo.available+len(reader._pending))

        self.assertLess(max(levels)-min(levels), 40)
        self.assertEqual(fifo.dropped, 0)
        self.assertGreater(reader.max_correction, 0.0)
        self.assertLessEqual(reader.max_correction, 600e-6)
        self.assertLessEqual(reader.max_correction,
                             ClockMatchedReader.MAX_CORRECTION)

    def test_device_capture_selects_or_downmixes_input_channels(self):
        class InputStream:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def start(self):
                pass

            def stop(self):
                pass

            def close(self):
                pass

        SoundDevice = type('SoundDevice', (), {
            'InputStream': InputStream,
            'query_devices': staticmethod(
                lambda _device, _kind: {'max_input_channels': 2}),
        })

        for side, expected in (
                ('left', [1, 2]), ('right', [5, 6]), ('mix', [3, 4])):
            with self.subTest(side=side):
                source = DeviceSourceAudio(
                    7, 48_000, input_side=side,
                    target_samples=4, sounddevice_module=SoundDevice)
                source.stream.kwargs['callback'](
                    np.array([[1, 5], [2, 6], [3, 7], [4, 8]],
                             dtype=np.float32),
                    4, None, None)
                np.testing.assert_array_equal(source.read(2), expected)
                source.close()

    def test_embedded_audio_command_requests_optional_mono_f32le(self):
        class Process:
            def __init__(self):
                self.stdout = io.BytesIO()

            def poll(self):
                return 0

            def terminate(self):
                pass

        process = Process()
        with tempfile.NamedTemporaryFile(dir=ROOT/'tmp') as source, \
                mock.patch('subprocess.Popen', return_value=process) as popen:
            audio = FFmpegSourceAudio(source.name, 48_000)
            self.assertFalse(audio.wait_for_samples(1, .2))
            command = popen.call_args.args[0]
            audio.close()

        self.assertIn('-map', command)
        self.assertIn('0:a:0?', command)
        self.assertIn('pcm_f32le', command)
        self.assertIn('48000', command)
        self.assertIn('aresample=async=1:first_pts=0', command)
        self.assertIn('-re', command)
        self.assertIn('-stream_loop', command)

    def test_live_hls_audio_is_paced_to_its_source_timestamps(self):
        class Process:
            def __init__(self):
                self.stdout = io.BytesIO()

            def poll(self):
                return 0

            def terminate(self):
                pass

        process = Process()
        with mock.patch('subprocess.Popen', return_value=process) as popen:
            audio = FFmpegSourceAudio(
                'https://media.example/master.m3u8', 48_000, live=True)
            command = popen.call_args.args[0]
            audio.close()

        self.assertIn('-re', command)
        self.assertIn('-rw_timeout', command)
        self.assertNotIn('-stream_loop', command)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe') and
                         hasattr(os, 'mkfifo'),
                         'shared A/V smoke test needs POSIX FFmpeg and ffprobe')
    def test_shared_ffmpeg_video_preserves_audio_offset_and_no_track_silence(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp',
                                         prefix='v7-shared-av-test-') as folder:
            folder = Path(folder)
            with_audio = folder/'offset.mkv'
            without_audio = folder/'silent.mkv'
            subprocess.run([
                'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                '-f', 'lavfi', '-i',
                'color=c=black:s=64x64:r=10:d=0.5',
                '-itsoffset', '0.2', '-f', 'lavfi', '-i',
                'sine=frequency=1000:sample_rate=48000:duration=0.3',
                '-c:v', 'mpeg4', '-c:a', 'pcm_s16le', '-shortest',
                str(with_audio),
            ], check=True)
            source = SharedVideoAudioSource(
                str(with_audio), 48_000, width=64, target_samples=1024)
            try:
                frame = source.video_grab()
                self.assertEqual(frame.shape, (64, 64, 3))
                self.assertTrue(source.has_audio)
                self.assertTrue(source.wait_for_samples(12_000, 1.0))
                samples = source.read(12_000)
                rms = np.array([
                    np.sqrt(np.mean(samples[index:index+480]**2))
                    for index in range(0, len(samples), 480)
                ])
                self.assertTrue(np.all(rms[:19] < 1e-4), rms.tolist())
                self.assertTrue(np.all(rms[21:] > .05), rms.tolist())
            finally:
                source.close()

            source = SharedVideoAudioSource(
                str(with_audio), 48_000, width=32, target_samples=1024,
                preserve_size=True)
            try:
                self.assertEqual(source.video_grab().shape, (64, 64, 3))
            finally:
                source.close()

            subprocess.run([
                'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                '-f', 'lavfi', '-i',
                'color=c=black:s=64x64:r=10:d=0.5',
                '-c:v', 'mpeg4', str(without_audio),
            ], check=True)
            source = SharedVideoAudioSource(
                str(without_audio), 48_000, width=64, target_samples=1024)
            try:
                self.assertFalse(source.has_audio)
                self.assertEqual(source.video_grab().shape, (64, 64, 3))
                np.testing.assert_array_equal(source.read(1024), 0.0)
            finally:
                source.close()


if __name__ == '__main__':
    unittest.main()
