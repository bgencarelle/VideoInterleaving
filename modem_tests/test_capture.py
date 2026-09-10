"""Offline capture tests: no audio hardware, sleeping, or wall-clock waits."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.signal import upfirdn, resample_poly

from animation_modem import decoder
from animation_modem.capture import BufferedInput, CaptureResampler, open_input
from animation_modem.channel_scan import AutoChannels
from animation_modem.transport2 import encode, Receiver


class FakeSD:
    class PortAudioError(Exception):pass

    def __init__(self):
        self.events = []
        self.interrupt = False

    def check_input_settings(self, **kw):
        raise AssertionError('Sample-rate probing must not be called')

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
    def test_capture_queue_is_bounded_and_marks_gaps(self):
        class Stream:
            samplerate=48000
            read_available=256
            n=0
            def read(self,size):
                self.n+=1
                return np.full((size,2),self.n,np.float32), self.n==4
        buffer=BufferedInput(Stream(),256,max_seconds=3*256/48000)
        for _ in range(5):buffer._poll()
        self.assertEqual(len(buffer.queue),3)
        audio,overflow,skipped=buffer.read()
        self.assertEqual((audio[0,0],overflow,skipped),(3,False,2))
        self.assertEqual(buffer.read()[1:],(True,0))
        buffer.error=RuntimeError('driver stopped')
        self.assertEqual(buffer.read()[0][0,0],5)
        with self.assertRaisesRegex(RuntimeError,'driver stopped'):buffer.read()

    def test_utility_uses_shared_input_and_preserves_channel_override(self):
        import contextlib
        import io
        import json
        from utilities import modem_v2_check as utility
        result=SimpleNamespace(values=None, absolute=42, index=7, source_index=6,
            count=20, face_folder=1, float_folder=2, tier='best', coverage=1.,
            status='received', identity='verified_header', pilot_error=0.,
            rate_error=0., extra={'input_channels':[4,2]})
        for options, expected in (([],None),(['--channels','4,2'],(3,1))):
            closed=[]; calls=[]
            def live(sd,device,channels,layout,coder,settings,stop,report):
                calls.append((device,channels))
                try:yield result
                finally:closed.append(True)
            output=io.StringIO()
            with patch.object(utility,'_sounddevice',return_value=object()), \
                 patch.object(utility,'live_results',live), \
                 contextlib.redirect_stdout(output),contextlib.redirect_stderr(io.StringIO()):
                utility.main(['live-receive','--device','18','--headless',*options])
            self.assertEqual(calls,[(18,expected)])
            self.assertEqual(closed,[True])
            self.assertEqual(json.loads(output.getvalue())['frame'],42)

    def test_utility_propagates_capture_failure(self):
        import contextlib
        import io
        from utilities import modem_v2_check as utility
        def broken(*args):
            raise RuntimeError('device disconnected')
            yield
        with patch.object(utility,'_sounddevice',return_value=object()), \
             patch.object(utility,'live_results',broken), \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'device disconnected'):
                utility.main(['live-receive','--headless'])

    def test_device_default_rate_and_body_error_close(self):
        sd = FakeSD()
        with self.assertRaisesRegex(ValueError, 'body'):
            with open_input(sd, 1, 8, 48000) as stream:
                self.assertEqual(stream.samplerate, 48000)
                raise ValueError('body')
        self.assertEqual(sd.events, [('start', 48000), ('stop', 48000), ('close', 48000)])

    def test_failed_start_does_not_try_other_rates(self):
        sd = FakeSD()
        with self.assertRaisesRegex(sd.PortAudioError, 'clock busy'):
            with open_input(sd, 1, 8, 192000):pass
        self.assertEqual(sd.events, [('start', 192000), ('close', 192000)])

    def test_interrupted_start_closes(self):
        sd = FakeSD(); sd.interrupt = True
        with self.assertRaises(KeyboardInterrupt):
            with open_input(sd, 1, 8, 48000):pass
        self.assertEqual(sd.events[-1], ('close', 48000))

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
        class ImmediateBuffer:
            def __init__(self, stream, size):self.stream=stream
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self):
                self.stream.read_available
                return None
        with patch.object(decoder, 'BufferedInput', ImmediateBuffer), \
             patch.object(decoder.time, 'monotonic', side_effect=lambda:clock[0]), \
             patch.object(decoder, 'AutoChannels') as auto, patch.object(decoder, 'Emulator'):
            self.assertEqual(list(decoder.live_results(sd, 1, None, None, None, None,
                                                      stop, reports.append)), [])
            self.assertEqual(auto.call_count, 2)
        self.assertEqual(events, ['query', 'start', 'stop', 'close']*2)
        self.assertEqual(sum(r.get('status') == 'input_reopen' for r in reports), 1)


if __name__ == '__main__':unittest.main()
