"""Offline capture tests: no audio hardware, sleeping, or wall-clock waits."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.signal import upfirdn, resample_poly

from animation_modem import decoder
from animation_modem.capture import CaptureResampler, open_input
from animation_modem.channel_scan import AutoChannels
from animation_modem.transport2 import encode, Receiver


class FakeSD:
    class PortAudioError(Exception):pass

    def __init__(self):
        self.events = []
        self.interrupt = False

    def check_input_settings(self, **kw):
        if kw['samplerate'] > 192000:
            raise self.PortAudioError('unsupported')

    def InputStream(self, **kw):
        sd, rate = self, kw['samplerate']
        class Stream:
            samplerate = rate
            def start(self):
                sd.events.append(('start', rate))
                if sd.interrupt:raise KeyboardInterrupt()
                if rate > 96000:raise sd.PortAudioError('clock busy')
            def stop(self):sd.events.append(('stop', rate))
            def close(self):sd.events.append(('close', rate))
        return Stream()


class CaptureTests(unittest.TestCase):
    def test_rate_fallback_and_body_error_close(self):
        sd = FakeSD()
        with self.assertRaisesRegex(ValueError, 'body'):
            with open_input(sd, 1, 8, 48000) as stream:
                self.assertEqual(stream.samplerate, 96000)
                raise ValueError('body')
        self.assertIn(('close', 192000), sd.events)
        self.assertEqual(sd.events[-2:], [('stop', 96000), ('close', 96000)])

    def test_interrupted_start_closes(self):
        sd = FakeSD(); sd.interrupt = True
        with self.assertRaises(KeyboardInterrupt):
            with open_input(sd, 1, 8, 48000):pass
        self.assertEqual(sd.events[-1], ('close', 192000))

    def test_resampling_chunk_boundaries_and_reset(self):
        x = np.random.default_rng(2).normal(size=(8099, 2)).astype(np.float32)
        for rate in (44100, 48000, 88200, 96000, 192000, 768000):
            with self.subTest(rate=rate):
                r = CaptureResampler(rate, 2)
                got = np.concatenate([r.process(x[i:i+173]) for i in range(0, len(x), 173)])
                ref = upfirdn(r.taps, x, r.up, r.down, axis=0)[:len(got)]
                np.testing.assert_allclose(got, ref, atol=1e-6, rtol=1e-5)
                self.assertEqual(len(got), len(x)//r.down*r.up)
                r.reset()
                np.testing.assert_array_equal(r.process(x), got)

    def test_image_identity_and_channel_selection_after_conversion(self):
        layout, coder = decoder.build()
        values = np.random.default_rng(3).uniform(-.1, .1, coder.count)
        audio = encode(values, layout, coder, 42, 7, 20)
        for rate in (44100, 96000, 192000):
            with self.subTest(rate=rate):
                converter = CaptureResampler(rate, 4)
                captured = resample_poly(np.concatenate([audio, np.zeros((1000, 2))]),
                                          converter.down, converter.up, axis=0)
                device = np.zeros((len(captured), 4), np.float32)
                device[:, 3], device[:, 1] = captured[:, 0], captured[:, 1]
                receiver = AutoChannels(layout, coder, 4)
                out = []
                for i in range(0, len(device), 173):
                    out += receiver.feed(converter.process(device[i:i+173]))
                out += receiver.flush()
                self.assertTrue(out)
                self.assertEqual(out[0].absolute, 42)
                self.assertEqual(set(out[0].extra['input_channels']), {2, 4})
                self.assertLess(np.sqrt(np.mean((out[0].values-values)**2)), .04)
                receiver.reset()
                self.assertEqual(set(receiver.selected), {1, 3})

    def test_stalled_stream_closes_before_reopen(self):
        clock = [0.0]; events = []; reports = []
        class Stop:
            stopped = False
            def is_set(self):return self.stopped
            def wait(self, seconds):clock[0] += 15.0
        stop = Stop()
        class Input:
            latency = .005
            samplerate = 48000
            def start(self):
                self.number = sum(e == 'start' for e in events)+1
                events.append('start')
            def stop(self):events.append('stop')
            def close(self):events.append('close')
            @property
            def read_available(self):
                if self.number == 2:stop.stopped = True
                return 0
        def query(*args):
            events.append('query')
            return dict(max_input_channels=4, default_samplerate=48000)
        def check(**kw):
            if kw['samplerate'] != 48000:raise FakeSD.PortAudioError('unsupported')
        sd = SimpleNamespace(query_devices=query, check_input_settings=check,
                             PortAudioError=FakeSD.PortAudioError,
                             InputStream=lambda **kw:Input())
        with patch.object(decoder.time, 'monotonic', side_effect=lambda:clock[0]), \
             patch.object(decoder, 'AutoChannels') as auto, patch.object(decoder, 'Emulator'):
            self.assertEqual(list(decoder.live_results(sd, 1, None, None, None, None,
                                                      stop, reports.append)), [])
            self.assertEqual(auto.call_count, 2)
        self.assertEqual(events, ['query', 'start', 'stop', 'close']*2)
        self.assertEqual(sum(r.get('status') == 'input_reopen' for r in reports), 1)


if __name__ == '__main__':unittest.main()
