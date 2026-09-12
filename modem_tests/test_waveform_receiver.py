import unittest
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from scipy.signal import resample_poly

from animation_modem import transport2 as v
from animation_modem.imaging import plane_shapes
from animation_modem.waveform_receiver import Receiver


class WaveformReceiverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layout = v.PRESETS['wide']
        cls.coder = v.SourceCoder(plane_shapes())
        cls.truth = np.random.default_rng(14).uniform(-.2, .2, (3, cls.coder.count))
        cls.audio = np.concatenate([
            v.encode(values, cls.layout, cls.coder, i+1, i+1, len(cls.truth))
            for i, values in enumerate(cls.truth)])

    def receive(self, audio):
        rx = Receiver(self.layout, self.coder)
        out = []
        for at in range(0, len(audio), 257):
            out.extend(rx.feed(audio[at:at+257]))
        return out + rx.flush()

    def test_native_48k_and_96k_are_packet_clocked(self):
        for rate_audio, expected_scale in ((self.audio, 1.),
                                            (resample_poly(self.audio, 2, 1, axis=0), 2.)):
            out = self.receive(rate_audio)
            self.assertEqual([r.absolute for r in out], list(range(1, len(self.truth)+1)))
            self.assertAlmostEqual(float(out[0].extra['sample_scale']), expected_scale, delta=.01)
            for result in out:
                self.assertTrue(np.isfinite(result.values).all())
                error = np.sqrt(np.mean((result.values-self.truth[result.absolute-1])**2))
                self.assertLess(error, .003)
                self.assertEqual(result.extra['fft_symbols'], self.layout.symbols)

    def test_silence_and_single_frame_loss_do_not_poison_next_frame(self):
        first = self.audio[:self.layout.frame]
        second = self.audio[self.layout.frame:2*self.layout.frame]
        rx = Receiver(self.layout, self.coder)
        self.assertEqual(len(rx.feed(np.zeros((5000, 2), np.float32))), 0)
        self.assertEqual(len(rx.feed(first)), 1)
        self.assertEqual(len(rx.feed(np.zeros((5000, 2), np.float32))), 0)
        out = rx.feed(second)+rx.flush()
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].absolute, 2)

    def test_native_live_startup_without_resampler(self):
        from animation_modem import decoder
        stop = threading.Event()
        stream = SimpleNamespace(samplerate=96000., latency=.01)
        capture = Mock(capacity_ms=100.)
        capture.__enter__ = Mock(return_value=capture)
        capture.__exit__ = Mock(return_value=False)
        calls = []
        def read():
            if not calls:
                calls.append(True)
                return np.ones((256, 2), np.float32), False, 0
            stop.set()
            return None
        capture.read.side_effect = read
        sd = Mock()
        sd.query_devices.return_value = {'max_input_channels': 2, 'default_samplerate': 96000.}
        factory = Mock(native_waveform=True)
        received = SimpleNamespace(rate_error=1., extra={
            'packet_duration_samples': 200., 'packet_age_samples': 220.})
        factory.return_value.feed.return_value = [received]
        report = Mock()
        with patch.object(decoder, 'open_input') as opening, \
             patch.object(decoder, 'BufferedInput', return_value=capture), \
             patch.object(decoder, 'CaptureResampler') as resampler, \
             patch.object(decoder, 'AudioGate') as gate, \
             patch.object(decoder, 'Emulator') as emulator:
            opening.return_value.__enter__.return_value = stream
            gate.return_value.process.side_effect = lambda a: a
            emulator.return_value.process.side_effect = lambda a: a
            self.assertEqual(list(decoder.live_results(
                sd, None, (0, 1), self.layout, self.coder, None,
                stop, report, receiver_factory=factory)), [received])
            resampler.assert_not_called()
        startup = next(call.args[0] for call in report.call_args_list
                       if 'resample_filter_delay_ms' in call.args[0])
        self.assertIsNone(startup['resample_filter_delay_ms'])
        self.assertEqual(startup['modem_clock'], 'packet_reference')
        self.assertEqual(received.rate_error, 0.)
        self.assertEqual(received.extra['playback_speed'], 1.)
        self.assertEqual(received.extra['packet_duration_samples'], 100.)
        self.assertEqual(received.extra['packet_age_samples'], 110.)


if __name__ == '__main__':
    unittest.main()
