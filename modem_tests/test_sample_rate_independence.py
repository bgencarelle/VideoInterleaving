"""The decode engine asks no device for a rate, and needs none to decode.

Timing comes from counting preamble edges, so a capture at 44.1, 96 or 192 kHz
differs from a 48 kHz one only by a constant factor on every interval -- the
same thing a deck running slow or fast does, which the decoder already
corrects. These tests hold that line in three places:

* the demodulator works with no rate at all, and produces bit-identical values
  whether or not one is supplied;
* when a rate IS supplied -- always because an open device or a file header
  reported it, never because we asked for it -- speed, duration and carrier
  frequency come back in real units instead of being silently wrong;
* nothing on the live paths passes samplerate= to PortAudio.
"""
import inspect
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.signal import resample_poly

from animation_modem import transport3 as v3
from animation_modem.core import REFERENCE_RATE
from animation_modem.imaging import plane_shapes

LAYOUT = v3.ALL_PRESETS['lean-v3']


def captured_at(audio, rate):
    """The same signal, sampled by a device running at `rate`.

    Not a speed change: the transmission is identical and takes the same number
    of seconds. Only the number of samples that lands in the buffer differs,
    which is precisely the thing the decoder must not care about.
    """
    if rate == REFERENCE_RATE:
        return audio
    from fractions import Fraction
    ratio = Fraction(int(rate), REFERENCE_RATE).limit_denominator(1000)
    return resample_poly(audio, ratio.numerator, ratio.denominator)


class RateIndependentDecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.coder = v3.SourceCoder(plane_shapes('lean-dct'))
        cls.values = np.random.default_rng(11).uniform(-.2, .2, cls.coder.count)
        cls.audio = np.concatenate([
            v3.encode(cls.values, LAYOUT, cls.coder, n, n, 4)
            for n in range(1, 5)])

    def decode(self, audio, **kw):
        rx = v3.Receiver(LAYOUT, self.coder, **kw)
        return rx.feed(np.asarray(audio, np.float32))+rx.flush()

    def error(self, results):
        return float(np.median([np.sqrt(np.mean((r.values-self.values)**2))
                                for r in results]))

    def test_decodes_without_being_told_any_rate(self):
        out = self.decode(self.audio)
        self.assertEqual([r.absolute for r in out], [1, 2, 3, 4])
        self.assertLess(self.error(out), 1e-5)
        for r in out:
            # Sample-domain facts only: no hertz, no seconds, nothing invented.
            self.assertAlmostEqual(r.extra['timing_scale'], 1.0, places=6)
            self.assertNotIn('input_rate', r.extra)
            self.assertNotIn('input_top_hz', r.extra)
            self.assertNotIn('frame_seconds', r.extra)

    def test_the_rate_changes_no_decoded_value(self):
        """Reporting is the only thing `input_rate` touches."""
        blind = self.decode(self.audio)
        told = self.decode(self.audio, input_rate=REFERENCE_RATE)
        self.assertEqual(len(blind), len(told))
        for a, b in zip(blind, told):
            np.testing.assert_array_equal(a.values, b.values)
            self.assertEqual(a.extra['timing_scale'], b.extra['timing_scale'])

    def test_every_common_device_rate_decodes(self):
        """44.1 kHz up to 192 kHz, with the picture and the numbers intact.

        The top carrier sits at 20250 Hz, so 44.1 kHz is the lowest rate that
        can carry this layout at all -- below that the band is simply gone, and
        no amount of rate agility helps.
        """
        for rate in (44100, 48000, 96000, 192000):
            with self.subTest(rate=rate):
                out = self.decode(captured_at(self.audio, rate), input_rate=rate)
                self.assertEqual([r.absolute for r in out], [1, 2, 3, 4])
                self.assertLess(self.error(out), .05)
                for r in out:
                    # The transport is running normally at every one of these.
                    self.assertAlmostEqual(r.extra['playback_speed'], 1.0, delta=.005)
                    # Scale, by contrast, IS the rate ratio.
                    self.assertAlmostEqual(r.extra['timing_scale'],
                                           rate/REFERENCE_RATE, delta=.005)
                    self.assertEqual(r.extra['input_rate'], rate)
                    self.assertAlmostEqual(r.extra['input_top_hz'], 20250, delta=120)
                    self.assertAlmostEqual(r.extra['frame_seconds'],
                                           LAYOUT.frame/REFERENCE_RATE, delta=.001)

    def test_speed_window_follows_the_capture_clock(self):
        """Otherwise the rate quietly eats the speed range.

        A 96 kHz capture is scale 2.0 all by itself, which is the far edge of
        the old fixed 0.25-2.0x window: every real playback speed above 1.0x
        would fall outside it, and 192 kHz would be rejected outright. Anchored
        to the device clock, the window still means transport speed.
        """
        blind = v3.Receiver(LAYOUT, self.coder)
        self.assertEqual((blind.min_scale, blind.max_scale), (.5, 4.))
        fast = v3.Receiver(LAYOUT, self.coder, input_rate=192000)
        self.assertEqual((fast.min_scale, fast.max_scale), (2., 16.))
        # A 192 kHz device, decoding a deck running at half speed: scale 8,
        # which the old window could not express at any rate.
        slow = np.repeat(captured_at(self.audio, 192000), 2, axis=0)
        out = self.decode(slow, input_rate=192000)
        self.assertEqual([r.absolute for r in out], [1, 2, 3, 4])
        for r in out:
            self.assertAlmostEqual(r.extra['playback_speed'], .5, delta=.01)

    def test_a_rate_is_never_invented(self):
        self.assertIsNone(v3.Receiver(LAYOUT, self.coder).input_rate)
        self.assertEqual(v3.Receiver(LAYOUT, self.coder).clock, 1.)
        for bad in (0, -48000, float('nan'), float('inf')):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    v3.Receiver(LAYOUT, self.coder, input_rate=bad)

    def test_the_timing_measurements_take_no_rate(self):
        """A rate argument here would be a rate the caller had to obtain."""
        for fn in (v3.measure_pulses, v3.measure_speed, v3.edge_intervals):
            with self.subTest(fn=fn.__name__):
                names = set(inspect.signature(fn).parameters)
                self.assertEqual(names & {'rate', 'sample_rate', 'samplerate',
                                          'fs', 'input_rate'}, set())


class NoRateIsRequestedTests(unittest.TestCase):
    """PortAudio picks the device's own rate when none is passed."""

    def test_packet_output_opens_without_a_rate_and_follows_the_device(self):
        from animation_modem import playback
        stream = Mock(samplerate=44100, latency=.005, time=0.)
        fake = Mock(OutputStream=Mock(return_value=stream))
        with patch.object(playback, 'sounddevice', return_value=fake):
            out = playback.PacketOutput(frame=LAYOUT.frame, packet=LAYOUT.packet)
        self.assertNotIn('samplerate', fake.OutputStream.call_args.kwargs)
        self.assertEqual(out.rate, 44100)
        self.assertAlmostEqual(out.fps, 44100/LAYOUT.frame)

    def test_live_send_opens_without_a_rate(self):
        from utilities import modem_v3_check as check
        stream = Mock(samplerate=44100)
        sd = Mock(OutputStream=Mock(return_value=stream))
        with patch.object(check, 'sounddevice', return_value=sd), \
             patch.object(check.time, 'sleep', side_effect=KeyboardInterrupt):
            check.main(['live-send', '--device', '0', '--frames', '1'])
        self.assertNotIn('samplerate', sd.OutputStream.call_args.kwargs)
        stream.start.assert_called_once()
        stream.close.assert_called_once()


class BandLimitedSenderTests(unittest.TestCase):
    """A transmitter adapts its signal to the device, not the device to it.

    Receiving is clock-agnostic for free. Sending is not: the emitted band is
    the device's clock times the carrier geometry, so an unadapted packet out
    of a 96 kHz device puts the picture at 750-40500 Hz -- past a cheap DAC's
    reconstruction filter and past any sane receiver's Nyquist.
    """

    @classmethod
    def setUpClass(cls):
        cls.coder = v3.SourceCoder(plane_shapes('lean-dct'))
        cls.values = np.random.default_rng(23).uniform(-.2, .2, cls.coder.count)
        cls.packets = [v3.encode(cls.values, LAYOUT, cls.coder, n, n, 4)
                       for n in range(1, 5)]

    def emitted(self, rate):
        return np.concatenate([v3.band_limited(p, rate) for p in self.packets])

    def share_above(self, audio, rate, hz):
        """Fraction of emitted energy above `hz`, in dB. Measured, not assumed.

        An absolute 'highest live bin' threshold does not work here: a packet
        is a sequence of hard-edged OFDM frames with a silent guard, and that
        discontinuity alone splatters to -20 dB near Nyquist AT THE REFERENCE
        RATE. That is a property of the format, not of any resampling, so the
        only meaningful question is whether the band-limited output differs
        from the reference -- which is what this measures.
        """
        spectrum = np.abs(np.fft.rfft(audio[:, 0]))**2
        freqs = np.fft.rfftfreq(len(audio), 1/rate)
        return 10*np.log10(max(spectrum[freqs > hz].sum(), 1e-30)/spectrum.sum())

    def test_carriers_never_ride_the_clock_up(self):
        """Above the reference Nyquist there must be nothing worth hearing."""
        ceiling = REFERENCE_RATE/2
        for rate in (88200, 96000, 176400, 192000):
            with self.subTest(rate=rate):
                held = self.share_above(self.emitted(rate), rate, ceiling)
                self.assertLess(held, -45, 'emitted band rode the device clock')
                # Unadapted, the carriers themselves are up there. The top one
                # is the link-killer -- it lands past 37 kHz -- while the share
                # of ENERGY stranded is a smaller number only because the power
                # allocation weights the low and middle carriers.
                self.assertGreater(LAYOUT.band_at(rate)[1], ceiling)
                loose = self.share_above(np.concatenate(self.packets), rate, ceiling)
                self.assertGreater(loose, -15)
                self.assertGreater(loose-held, 30)

    def test_the_band_that_is_kept_matches_the_reference(self):
        """Holding the band must not mean quietly filtering the carriers."""
        reference = self.share_above(np.concatenate(self.packets),
                                     REFERENCE_RATE, 21000)
        for rate in (88200, 96000, 192000):
            with self.subTest(rate=rate):
                self.assertAlmostEqual(
                    self.share_above(self.emitted(rate), rate, 21000),
                    reference, delta=.5)

    def test_nothing_is_resampled_at_or_below_the_reference(self):
        """Below the reference the band scales DOWN, which is already safe."""
        for rate in (44100, 48000):
            with self.subTest(rate=rate):
                self.assertIsNone(v3.emit_ratio(rate))
                np.testing.assert_array_equal(
                    v3.band_limited(self.packets[0], rate), self.packets[0])

    def test_resampling_does_not_break_the_encoded_level(self):
        """Interpolation reconstructs intersample peaks and would clip."""
        for rate in (88200, 96000, 192000):
            with self.subTest(rate=rate):
                peak = float(np.max(np.abs(v3.band_limited(self.packets[0], rate))))
                self.assertAlmostEqual(peak, float(np.max(np.abs(self.packets[0]))),
                                       places=5)
                self.assertLessEqual(peak, 1.0)

    def test_every_sender_receiver_rate_pair_decodes(self):
        """The point of the band limit: the whole matrix works, not a diagonal.

        Unadapted, a 96 kHz sender is unreachable by a 48 kHz receiver because
        its carriers sit above that receiver's Nyquist. Held at the reference
        band, any of these senders reaches any of these receivers.
        """
        for send in (44100, 48000, 96000, 192000):
            audio = self.emitted(send)
            for listen in (44100, 48000, 96000):
                with self.subTest(send=send, listen=listen):
                    heard = captured_at_device(audio, send, listen)
                    rx = v3.Receiver(LAYOUT, self.coder, input_rate=listen)
                    out = rx.feed(np.asarray(heard, np.float32))+rx.flush()
                    self.assertEqual([r.absolute for r in out], [1, 2, 3, 4])
                    self.assertTrue(all(r.identity == 'verified_header' for r in out))
                    error = np.median([np.sqrt(np.mean((r.values-self.values)**2))
                                       for r in out])
                    self.assertLess(error, .05)


def captured_at_device(audio, sent_at, heard_at):
    """Resample as an independent receiving clock would see the same sound."""
    if sent_at == heard_at:
        return audio
    from fractions import Fraction
    ratio = Fraction(int(heard_at), int(sent_at)).limit_denominator(1000)
    return resample_poly(audio, ratio.numerator, ratio.denominator)


class SenderNoticeTests(unittest.TestCase):
    """Receiving is rate-agnostic; transmitting moves the band, so say so."""

    def test_reference_rate_says_nothing(self):
        from animation_modem.audio_common import wire_notice
        self.assertIsNone(wire_notice(LAYOUT, REFERENCE_RATE))

    def test_a_faster_device_reports_the_band_held_and_the_one_avoided(self):
        from animation_modem.audio_common import wire_notice
        notice = wire_notice(LAYOUT, 96000)
        self.assertIn('375-20250 Hz', notice)      # where the carriers stay
        self.assertIn('40500', notice)             # where they would have gone
        self.assertIn('2/1', notice)
        self.assertIn('17.34 fps', notice)

    def test_a_slower_device_reports_a_band_that_only_scaled_down(self):
        from animation_modem.audio_common import wire_notice
        notice = wire_notice(LAYOUT, 44100)
        self.assertIn('345-18605 Hz', notice)
        self.assertIn('15.93 fps', notice)

    def test_a_coarse_ratio_admits_its_own_offset(self):
        from animation_modem.audio_common import wire_notice
        notice = wire_notice(LAYOUT, 88200)
        self.assertIn('11/6', notice)
        self.assertIn('0.23%', notice)


class AnyRateWavTests(unittest.TestCase):
    def test_read_takes_the_rate_from_the_header(self):
        import json
        import wave
        import tempfile
        from pathlib import Path
        from animation_modem.audio_common import pcm, wav_rate
        from utilities import modem_v3_check as check

        coder = v3.SourceCoder(plane_shapes('lean-dct'))
        values = np.random.default_rng(7).uniform(-.2, .2, coder.count)
        audio = np.concatenate([v3.encode(values, LAYOUT, coder, n, n, 3)
                                for n in range(1, 4)])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'at_44k.wav'
            with wave.open(str(path), 'wb') as sink:
                # A real 44.1 kHz recording of a 48 kHz transmission. This used
                # to be refused as malformed input.
                sink.setparams((2, 2, 44100, 0, 'NONE', 'not compressed'))
                sink.writeframesraw(pcm(captured_at(audio, 44100)))
            self.assertEqual(wav_rate(path), 44100)
            printed = []
            with patch('builtins.print', side_effect=lambda *a, **kw: printed.append(a)):
                check.main(['read', '--wav', str(path)])
        records = [json.loads(a[0]) for a in printed
                   if a and isinstance(a[0], str) and a[0].startswith('{')]
        self.assertEqual([r['frame'] for r in records], [1, 2, 3])
        for r in records:
            self.assertEqual(r['identity'], 'verified_header')
            self.assertAlmostEqual(r['playback_speed'], 1.0, delta=.005)
            self.assertEqual(r['input_rate'], 44100)


if __name__ == '__main__':
    unittest.main()
