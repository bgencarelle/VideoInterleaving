"""Stereo slices: each channel its own mono wire, each alone a whole picture."""
import sys
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from scipy.fft import dctn

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT/'test_modem_v7', ROOT/'tools'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from animation_modem import v7                                           # noqa: E402
from common import TARGET                                                # noqa: E402
import slice_wire                                                        # noqa: E402
from slice_wire import (CHROMA_BASE, DETAIL, FOLD_GUESTS, LUMA_BASE,      # noqa: E402
                        MARKER_SLOTS, SIGNATURE_SLOTS, Half, SliceWire,
                        layout_tables, slice_positions)
from aspect_fold import LAYOUT_NAMES                                     # noqa: E402
from mono_video import FRESH_SLOTS                                       # noqa: E402
import tone_code                                                         # noqa: E402
from tools import v7_live                                                # noqa: E402

LUMA = 96*80


def _frame(seed=2, width=960, height=540):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (height//60, width//60, 3)).astype(np.uint8)
    frame = np.repeat(np.repeat(small, 60, 0), 60, 1).astype(float)
    y, x = np.mgrid[:height, :width]
    frame += 20*np.sin(x/23.0)[..., None]+20*np.cos(y/17.0)[..., None]
    return np.uint8(np.clip(frame, 0, 255))


def _luma_spectrum(values):
    return dctn(np.asarray(values)[:LUMA].reshape(96, 80), norm='ortho').ravel()


class _Receiver:
    """The live dispatcher, decoding one input channel at a time."""

    def __init__(self, base):
        self.base = base
        self.profile = v7_live._AdaptiveProfileDecoder(
            v7_live._experimental_fold(500), base)

    def halves(self, mono, channel):
        profile = self.profile
        profile.install()
        try:
            with tone_code.coded_pilot_timing():
                profile.active_mode = profile.dispatch_mode = (
                    profile.aspect_mono_mode)
                profile.active_side = 0
                results = v7.decode_pulse_stream(
                    self.base, np.asarray(mono, np.float32).reshape(-1, 1),
                    state=v7.PulseState(tail_memory=False),
                    sample_rate=v7.RATE, pilot_timing='tone-seeded')[0]
        finally:
            profile.uninstall()
        return [(result, profile.slice_half(self.base, result, channel))
                for result in results if result.status != 'lost']


def _perfect(wire, model, coefficients):
    """A Half as a noiseless channel would deliver ``coefficients``."""
    result = types.SimpleNamespace(
        coeffs=coefficients, status='received',
        diag={'head_confidence': 1.0, 'noise': [0.0],
              'mono_fold_eq': (coefficients-model.mu,
                               np.ones(len(coefficients)))})
    return wire.half(model, result)


class SliceWireTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = v7.load_model(TARGET, 'box')
        cls.wire = SliceWire('auto')
        cls.values, cls.code = v7_live._values(
            cls.base, _frame(), 'box', brightness=1.0, dct_encode=True)[:2]
        cls.layout = cls.wire.layout_for(cls.code)
        cls.model = cls.wire.model_for(cls.base, cls.layout)
        cls.slots = cls.wire.slots(cls.layout)
        cls.spectra = cls.wire.spectra(cls.values)
        cls.audio = cls.wire.encode(cls.base, [cls.values]*5,
                                    aspect_codes=[cls.code]*5)

    def _errors(self, values):
        """RMS error of the shown luma base and detail coefficients."""
        shown = _luma_spectrum(values)
        sent = self.spectra[0]
        return tuple(float(np.sqrt(np.mean((shown[index]-sent[index])**2)))
                     for index in (self.slots.base[0], self.slots.detail[0]))

    def test_every_layout_fills_one_channel_and_tables_are_pinned(self):
        self.assertEqual(slice_wire.TABLES_SHA256, '991c133d81816ee77f5b6591210fd589c05b290ddf0566a78f1f704e726fe041')
        self.assertEqual(LUMA_BASE+2*CHROMA_BASE+MARKER_SLOTS+SIGNATURE_SLOTS,
                         FRESH_SLOTS)
        for layout in LAYOUT_NAMES:
            with self.subTest(layout=layout):
                tables = layout_tables(layout)
                self.assertEqual(len(tables['lam']), FRESH_SLOTS)
                self.assertTrue(np.all(tables['lam'] > 0))
                self.assertEqual(len(set(tables['positions'].tolist())),
                                 FRESH_SLOTS)
                planes, ring = slice_positions(layout)
                for plane, (base, detail) in enumerate(planes):
                    carried = np.concatenate(
                        (tables[f'base_{plane}'], tables[f'detail_{plane}']))
                    self.assertEqual(len(set(carried.tolist())), len(carried))
                    np.testing.assert_array_equal(tables[f'base_{plane}'], base)
                    self.assertEqual(sorted(tables[f'detail_{plane}']),
                                     sorted(detail))
                # As built there is no fold: only the signature's slots.
                self.assertEqual(FOLD_GUESTS, 0)
                self.assertEqual(len(tables['ring']), 0)
                self.assertEqual(len(tables['fold_hosts']), SIGNATURE_SLOTS)
                # Colour is the same on both channels.
                self.assertFalse(np.any(tables['ride'][LUMA_BASE:]))
                # Strongest base with weakest detail: the base's variance
                # falls along the slots as the detail's part rises.
                luma = slice(0, LUMA_BASE)
                order = np.argsort(-tables['lam_sum'][luma], kind='stable')
                rest = (tables['lam'][luma]-tables['lam_sum'][luma])[order]
                self.assertTrue(np.all(np.diff(rest) >= -1e-12))

    def test_each_channel_is_a_mono_wire_at_the_stereo_level(self):
        self.assertEqual(int(self.model.encoding_type),
                         v7.ENCODING_FILTER_CODES['nearest'])
        self.assertEqual(self.audio.shape[1], 2)
        left, right = self.audio[:, 0], self.audio[:, 1]
        self.assertGreater(float(np.std(left-right)), .02)     # not copies
        tables = layout_tables(self.layout)
        self.assertAlmostEqual(
            float(self.model.scale)*float(tables['unit_rms']),
            float(self.base.scale) *
            float(v7._frozen_tables()['box/unit_rms']), places=9)
        nearest = v7.load_model(TARGET, 'nearest')
        np.testing.assert_allclose(
            SliceWire('auto').model_for(nearest, self.layout).scale,
            self.model.scale, rtol=1e-12)

    def test_slots_are_base_plus_and_minus_the_detail(self):
        left = self.wire.channel_coefficients(self.model, self.spectra, 'left')
        right = self.wire.channel_coefficients(self.model, self.spectra, 'right')
        plain = np.ones(LUMA_BASE, bool)
        plain[self.slots.fold_hosts[:self.slots.guests]] = False
        base = self.spectra[0][self.slots.base[0]]
        detail = self.spectra[0][self.slots.detail[0]]
        luma = self.slots.parts[0]
        np.testing.assert_allclose(((left+right)/2)[luma][plain], base[plain])
        np.testing.assert_allclose(((left-right)/2)[luma][plain],
                                   DETAIL*detail[plain])
        with self.assertRaises(ValueError):
            self.wire.channel_coefficients(self.model, self.spectra, 'sum')

    def test_a_perfect_channel_gives_whole_base_and_sum_pictures(self):
        left = self.wire.channel_coefficients(self.model, self.spectra, 'left')
        right = self.wire.channel_coefficients(self.model, self.spectra, 'right')
        halves = [_perfect(self.wire, self.model, left),
                  _perfect(self.wire, self.model, right)]
        self.assertEqual([half.kind for half in halves], ['left', 'right'])
        sd = float(np.std(self.spectra[0][self.slots.detail[0]]))
        base_error, detail_error = self._errors(self.wire.values(halves))
        self.assertLess(base_error, .05*sd)
        self.assertLess(detail_error, .1*sd)
        # A mono sum: the base, and nothing of the detail.
        summed = _perfect(self.wire, self.model, (left+right)/2)
        self.assertEqual(summed.kind, 'sum')
        self.assertAlmostEqual(summed.score, 0.0, places=6)
        values = self.wire.values([summed])
        shown = _luma_spectrum(values)
        self.assertLess(self._errors(values)[0], .05*sd)
        np.testing.assert_allclose(shown[self.slots.detail[0]], 0, atol=1e-9)
        # One channel: the base, disturbed by less than the detail it carries.
        base_error = self._errors(self.wire.values(halves[:1]))[0]
        self.assertLess(base_error, DETAIL*sd)
        self.assertGreater(base_error, 0.0)

    def test_both_channels_join_and_either_alone_is_a_whole_picture(self):
        receiver = _Receiver(self.base)
        left = receiver.halves(self.audio[:, 0], 0)
        right = receiver.halves(self.audio[:, 1], 1)
        self.assertGreaterEqual(min(len(left), len(right)), 3)
        self.assertEqual(left[-1][0].diag['wire_profile'], 'stereo-slices')
        self.assertEqual((left[-1][1].kind, right[-1][1].kind),
                         ('left', 'right'))
        self.assertAlmostEqual(left[-1][1].score, 1.0, delta=.05)
        self.assertAlmostEqual(right[-1][1].score, -1.0, delta=.05)
        profile = receiver.profile
        sd = float(np.std(self.spectra[0][self.slots.detail[0]]))
        both = profile.slice_values([left[-1][1], right[-1][1]], left[-1][0])
        self.assertEqual(left[-1][0].diag['slices_shown'], 'left+right')
        base_error, detail_error = self._errors(both)
        self.assertLess(base_error, .15*sd)
        self.assertLess(detail_error, .3*sd)
        for result, half in (left[-1], right[-1]):
            alone = profile.values(self.base, result)
            self.assertEqual(result.diag['slices_shown'], half.kind)
            self.assertLess(self._errors(alone)[0], DETAIL*sd)
        # Swapped cables: the markers, not the inputs, say which is which.
        swapped = self.wire.values([right[-1][1], left[-1][1]])
        np.testing.assert_allclose(swapped, both)

    def test_a_mono_sum_is_read_as_the_base_picture(self):
        receiver = _Receiver(self.base)
        mono = self.audio.sum(axis=1)/np.sqrt(2)
        halves = receiver.halves(mono, 0)
        self.assertGreaterEqual(len(halves), 3)
        self.assertEqual(halves[-1][1].kind, 'sum')
        self.assertAlmostEqual(halves[-1][1].score, 0.0, delta=.05)
        values = receiver.profile.values(self.base, halves[-1][0])
        sd = float(np.std(self.spectra[0][self.slots.detail[0]]))
        self.assertLess(self._errors(values)[0], .15*sd)
        np.testing.assert_allclose(
            _luma_spectrum(values)[self.slots.detail[0]], 0, atol=1e-9)
        # Heard on both inputs, the two sums are averaged.
        twice = self.wire.values([halves[-1][1], halves[-1][1]])
        np.testing.assert_allclose(twice, values, atol=2e-3)

    def test_crosstalk_is_read_from_the_markers_and_undone(self):
        receiver = _Receiver(self.base)
        mixed = self.audio @ np.array([[1.0, .1], [.1, 1.0]])
        left = receiver.halves(mixed[:, 0], 0)[-1][1]
        right = receiver.halves(mixed[:, 1], 1)[-1][1]
        self.assertAlmostEqual(self.wire.crosstalk([left, right]), .1, delta=.02)
        sd = float(np.std(self.spectra[0][self.slots.detail[0]]))
        undone = self._errors(self.wire.values([left, right]))[1]
        self.assertLess(undone, .4*sd)
        left.steady, right.steady = 1.0, -1.0           # as if none were seen
        self.assertEqual(self.wire.crosstalk([left, right]), 0.0)
        self.assertGreater(self._errors(self.wire.values([left, right]))[1],
                           1.2*undone)

    def test_a_damaged_channel_is_not_joined_with_a_clean_one(self):
        good = Half('left', 1.0, '16:9')
        damaged = Half('right', -1.0, '16:9', damaged=True)
        noisy = Half('right', -1.0, '16:9', good=False, noise=.2)
        self.assertEqual(SliceWire.choose([good, damaged]), [good])
        self.assertEqual(SliceWire.choose([damaged, good]), [good])
        self.assertEqual(SliceWire.choose([good, noisy]), [good, noisy])
        self.assertEqual(SliceWire.choose([damaged]), [damaged])

    def test_one_bad_packet_does_not_turn_a_channel_into_a_sum(self):
        wire = SliceWire('auto')
        model = wire.model_for(self.base, self.layout)
        coeffs = wire.channel_coefficients(model, self.spectra, 'left')
        marker = np.asarray(layout_tables(self.layout)['marker'])

        def result(scale, **diag):
            seen = coeffs.copy()
            seen[marker] *= scale
            return types.SimpleNamespace(
                coeffs=seen, status='received',
                diag={'head_confidence': 1.0, 'noise': [.01], **diag})

        self.assertEqual(wire.half(model, result(1.0), 0).kind, 'left')
        self.assertEqual(wire.half(model, result(0.0), 0).kind, 'left')
        self.assertEqual(wire.half(model, result(0.0)).kind, 'sum')
        half = wire.half(model, result(0.0, head_confidence=.5), 0)
        self.assertTrue(half.damaged)
        self.assertEqual(half.kind, 'left')
        for _ in range(12):                 # a lasting change is followed
            half = wire.half(model, result(-1.0), 0)
        self.assertEqual(half.kind, 'right')
        wire.reset()
        self.assertEqual(wire.half(model, result(0.0), 0).kind, 'sum')


def _send(argv, still, seconds='0.5'):
    written = []

    class OutputStream:
        def __init__(self, *args, **kwargs):
            self.samplerate = kwargs.get('samplerate', v7.RATE)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

        def write(self, audio):
            written.append(np.array(audio, dtype=np.float32, copy=True))

    fake = types.ModuleType('sounddevice')
    fake.OutputStream = OutputStream
    with mock.patch.dict(sys.modules, {'sounddevice': fake}), \
            mock.patch.object(v7_live, '_capture',
                              lambda _args: (lambda: still)):
        # The run is timed by the wall clock; a first run that spends it
        # compiling kernels writes nothing, so give it one longer try.
        for duration in (seconds, '4'):
            args = v7_live.parser().parse_args([
                'send', '--device', 'memory', '--source', 'test',
                '--seconds', duration, '--no-log', '--dct-encode', *argv])
            v7_live.run_send(args)
            if written:
                break
    return np.concatenate(written)


class SliceSenderTests(unittest.TestCase):
    def test_sender_emits_two_independent_channels(self):
        base = v7.load_model(TARGET, 'box')
        audio = _send(['--profile', 'stereo-slices', '--luma-adjust'], _frame())
        self.assertEqual(audio.shape[1], 2)
        receiver = _Receiver(base)
        left = receiver.halves(audio[:, 0], 0)
        right = receiver.halves(audio[:, 1], 1)
        self.assertTrue(left and right)
        self.assertEqual((left[-1][1].kind, right[-1][1].kind),
                         ('left', 'right'))
        self.assertEqual(left[-1][1].layout, '16:9')

    def test_stereo_slices_refuses_a_mono_output_and_pixel_encode(self):
        fake = {'sounddevice': types.ModuleType('sounddevice')}
        args = v7_live.parser().parse_args([
            'send', '--device', 'memory', '--profile', 'stereo-slices'])
        args.mono_sum = True
        with self.assertRaisesRegex(ValueError, 'two output channels'), \
                mock.patch.dict(sys.modules, fake):
            v7_live.run_send(args)
        args = v7_live.parser().parse_args([
            'send', '--device', 'memory', '--profile', 'stereo-slices',
            '--dct-encode', '--pixel-encode'])
        with self.assertRaisesRegex(ValueError, 'no pixel grid'), \
                mock.patch.dict(sys.modules, fake):
            v7_live.run_send(args)


class SliceLiveReceiverTests(unittest.TestCase):
    def _receive(self, audio):
        """Run the real receive loop on ``audio``; return published frames."""
        audio = np.concatenate([np.zeros((4800, 2), np.float32),
                                np.asarray(audio, np.float32),
                                np.zeros((9600, 2), np.float32)])
        stop_event = threading.Event()
        frames = []

        class SoundDevice:
            devices = [{'name': 'in', 'hostapi': 0, 'max_input_channels': 2,
                        'max_output_channels': 0,
                        'default_samplerate': 48000.0}]

            @classmethod
            def query_devices(cls, device=None, _kind=None):
                return (list(cls.devices) if device is None else
                        cls.devices[int(device)])

            @staticmethod
            def query_hostapis():
                return [{'name': 'Test API'}]

            class InputStream:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs
                    self.samplerate = float(kwargs['samplerate'])
                    self.active = True

                def start(self):
                    def feed():
                        callback = self.kwargs['callback']
                        seen = 0
                        for start in range(0, len(audio), 1024):
                            callback(audio[start:start+1024], 1024, None, None)
                            time.sleep(.012)
                            frame = v7_live.FRAME_BUFFER.snapshot()
                            if frame is not None and frame.generation != seen:
                                seen = frame.generation
                                frames.append(frame)
                        time.sleep(.4)
                        stop_event.set()
                    threading.Thread(target=feed, daemon=True).start()

                def stop(self):
                    self.active = False

                def close(self):
                    self.active = False

        args = v7_live.parser().parse_args(
            ['receive', '--device', '0', '--headless', '--no-log'])
        args.stop_event = stop_event
        with mock.patch.dict(sys.modules, {'sounddevice': SoundDevice}):
            v7_live.run_receive(args)
        return frames

    def test_live_receiver_joins_both_channels_and_survives_losing_one(self):
        base = v7.load_model(TARGET, 'box')
        wire = SliceWire('auto')
        values, code = v7_live._values(base, _frame(), 'box', brightness=1.0,
                                       dct_encode=True)[:2]
        slots = wire.slots(wire.layout_for(code))
        sent = wire.spectra(values)[0]
        sd = float(np.std(sent[slots.detail[0]]))
        audio = wire.encode(base, [values]*24, aspect_codes=[code]*24)

        def errors(frame):
            shown = _luma_spectrum(frame.values)
            return tuple(float(np.sqrt(np.mean((shown[index]-sent[index])**2)))
                         for index in (slots.base[0], slots.detail[0]))

        frames = self._receive(audio)
        self.assertGreaterEqual(len(frames), 8)
        base_error, detail_error = errors(frames[-1])
        self.assertLess(base_error, .15*sd)
        self.assertLess(detail_error, .3*sd)
        one = audio.copy()
        one[:, 1] = 0
        frames = self._receive(one)
        self.assertGreaterEqual(len(frames), 8)
        base_error, detail_error = errors(frames[-1])
        self.assertLess(base_error, DETAIL*sd)
        self.assertGreater(detail_error, .3*sd)       # the detail is not there


if __name__ == '__main__':
    unittest.main()
