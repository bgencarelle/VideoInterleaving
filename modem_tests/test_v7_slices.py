"""Stereo slices: each channel its own mono wire carrying half the picture."""
import sys
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT/'test_modem_v7', ROOT/'tools'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from animation_modem import v7                                           # noqa: E402
from common import TARGET                                                # noqa: E402
import slice_wire                                                        # noqa: E402
from slice_wire import (MARKER_SLOTS, SLICE_GRIDS, Half, SliceMonoWire,   # noqa: E402
                        SliceWire, _corner, fill, join, layout_tables,
                        slice_positions, slice_shapes, split, widen)
from aspect_fold import LAYOUT_NAMES                                     # noqa: E402
from mono_video import FRESH_SLOTS                                       # noqa: E402
import tone_code                                                         # noqa: E402
from tools import v7_live                                                # noqa: E402


def _frame(seed=2, width=960, height=540):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (height//60, width//60, 3)).astype(np.uint8)
    frame = np.repeat(np.repeat(small, 60, 0), 60, 1).astype(float)
    y, x = np.mgrid[:height, :width]
    frame += 20*np.sin(x/23.0)[..., None]+20*np.cos(y/17.0)[..., None]
    return np.uint8(np.clip(frame, 0, 255))


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
                    sample_rate=v7.RATE, pilot_timing='tone-seeded',
                    frame_boundary='eof')[0]
        finally:
            profile.uninstall()
        return [(result, profile.slice_half(self.base, result, channel))
                for result in results if result.status != 'lost']


class SliceWireTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = v7.load_model(TARGET, 'box')
        cls.wire = SliceWire('auto')
        cls.values, cls.code = v7_live._values(
            cls.base, _frame(), 'box', brightness=1.0, dct_encode=True)[:2]
        cls.layout = cls.wire.layout_for(cls.code)
        cls.sent = cls.wire.pictures(cls.values, cls.layout)
        cls.audio = cls.wire.encode(cls.base, [cls.values]*5,
                                    aspect_codes=[cls.code]*5)

    def _error(self, values):
        shown = _corner(np.asarray(values)[:96*80].reshape(96, 80),
                        slice_shapes(self.layout)[0])
        return float(np.sqrt(np.mean((shown-self.sent[0])**2)))

    def test_halves_split_join_fill_and_widen(self):
        picture = np.random.default_rng(0).standard_normal((6, 8))
        first, second = split(picture)
        self.assertEqual(first.shape, (6, 4))
        np.testing.assert_array_equal(join(first, second), picture)
        # The first half holds the samples with row+column even.
        self.assertEqual(first[0, 0], picture[0, 0])
        self.assertEqual(first[1, 0], picture[1, 1])
        filled = fill(first, True)
        np.testing.assert_array_equal(split(filled)[0], first)
        np.testing.assert_allclose(fill(np.ones((6, 4)), False), 1.0)
        ramp = np.add.outer(np.arange(6.0), np.arange(8.0))
        inner = np.s_[1:-1, 1:-1]
        np.testing.assert_allclose(fill(split(ramp)[0], True)[inner],
                                   ramp[inner])
        np.testing.assert_allclose(widen(np.full((4, 3), 2.5)), 2.5)

    def test_every_layout_fits_one_channel_and_tables_are_pinned(self):
        self.assertEqual(slice_wire.TABLES_SHA256, '01881891d69d33a09ebc0777eed7459055ddf1ee4b64b246704b5a14f71953ea')
        for layout in LAYOUT_NAMES:
            with self.subTest(layout=layout):
                (rows, cols), (chroma_rows, chroma_cols) = SLICE_GRIDS[layout]
                self.assertEqual(cols % 2, 0)
                self.assertEqual(chroma_cols % 2, 0)
                positions, marker = slice_positions(layout)
                count = rows*cols//2+chroma_rows*chroma_cols+MARKER_SLOTS
                self.assertEqual(len(positions), count)
                self.assertLessEqual(count, FRESH_SLOTS)
                self.assertEqual(len(set(positions.tolist())), count)
                tables = layout_tables(layout)
                self.assertEqual(len(tables['lam']), count)
                self.assertTrue(np.all(tables['lam'] > 0))
                np.testing.assert_array_equal(tables['marker'], marker)
        self.assertEqual(SLICE_GRIDS['9:16'][0], SLICE_GRIDS['16:9'][0][::-1])

    def test_each_channel_is_a_mono_wire_at_the_stereo_level(self):
        model = self.wire.model_for(self.base, self.layout)
        self.assertEqual(int(model.encoding_type),
                         v7.ENCODING_FILTER_CODES['nearest'])
        self.assertEqual(self.audio.shape[1], 2)
        left, right = self.audio[:, 0], self.audio[:, 1]
        self.assertGreater(float(np.std(left-right)), .05)     # not copies
        # Model pictures leave each channel at the stereo wire's level.
        tables = layout_tables(self.layout)
        self.assertAlmostEqual(
            float(model.scale)*float(tables['unit_rms']),
            float(self.base.scale) *
            float(v7._frozen_tables()['box/unit_rms']), places=9)
        for channel in (left, right):
            self.assertLess(float(np.max(np.abs(channel))), 1.6)
        # A nearest base model (what a live receiver loads for the model bit)
        # gives the same wire.
        nearest = v7.load_model(TARGET, 'nearest')
        np.testing.assert_allclose(
            SliceWire('auto').model_for(nearest, self.layout).scale,
            model.scale, rtol=1e-12)

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
        both = profile.slice_values([left[-1][1], right[-1][1]], left[-1][0])
        self.assertEqual(left[-1][0].diag['slices_shown'], 'left+right')
        self.assertEqual(tuple(left[-1][0].diag['pixel_shapes'][0]),
                         SLICE_GRIDS[self.layout][0])
        self.assertLess(self._error(both), .01)
        # One channel: complete, a little softer; the single-result path too.
        for result, half in (left[-1], right[-1]):
            alone = profile.values(self.base, result)
            self.assertEqual(result.diag['slices_shown'], half.kind)
            self.assertLess(self._error(alone), .12)
            self.assertGreater(self._error(alone), self._error(both))
        # Swapped cables: the markers, not the inputs, say which half is which.
        swapped = self.wire.values([right[-1][1], left[-1][1]])
        np.testing.assert_allclose(swapped, both)

    def test_a_mono_sum_is_the_pair_averaged_picture(self):
        receiver = _Receiver(self.base)
        mono = self.audio.sum(axis=1)/np.sqrt(2)
        halves = receiver.halves(mono, 0)
        self.assertGreaterEqual(len(halves), 3)
        self.assertEqual(halves[-1][1].kind, 'sum')
        self.assertAlmostEqual(halves[-1][1].score, 0.0, delta=.05)
        values = receiver.profile.values(self.base, halves[-1][0])
        first, second = split(self.sent[0])
        shown = _corner(values[:96*80].reshape(96, 80),
                        slice_shapes(self.layout)[0])
        self.assertLess(float(np.sqrt(np.mean(
            (shown-widen((first+second)/2))**2))), .01)
        # Heard on both inputs, the two sums are averaged, not joined.
        twice = self.wire.values([halves[-1][1], halves[-1][1]])
        np.testing.assert_allclose(twice, values, atol=1e-9)

    def test_crosstalk_is_read_from_the_markers_and_undone(self):
        receiver = _Receiver(self.base)
        mixed = self.audio @ np.array([[1.0, .1], [.1, 1.0]])
        left = receiver.halves(mixed[:, 0], 0)[-1][1]
        right = receiver.halves(mixed[:, 1], 1)[-1][1]
        leak = self.wire.crosstalk([left, right])
        self.assertAlmostEqual(leak, .1, delta=.02)
        self.assertLess(self._error(self.wire.values([left, right])), .015)
        plain = join(left.planes[0], right.planes[0])
        self.assertGreater(float(np.sqrt(np.mean(
            (plain-self.sent[0])**2))), .02)
        # No crosstalk: nothing is undone.
        clean = [Half('left', 1.0, left.planes, self.layout),
                 Half('right', -1.0, right.planes, self.layout)]
        self.assertEqual(self.wire.crosstalk(clean), 0.0)

    def test_a_damaged_channel_is_not_joined_with_a_clean_one(self):
        planes = [np.zeros((34, 30)), np.zeros((10, 10)), np.zeros((10, 10))]
        good = Half('left', 1.0, planes, '16:9')
        damaged = Half('right', -1.0, planes, '16:9', damaged=True)
        noisy = Half('right', -1.0, planes, '16:9', good=False, noise=.2)
        self.assertEqual(SliceWire.choose([good, damaged]), [good])
        self.assertEqual(SliceWire.choose([damaged, good]), [good])
        self.assertEqual(SliceWire.choose([good, noisy]), [good, noisy])
        self.assertEqual(SliceWire.choose([damaged]), [damaged])

    def test_one_bad_packet_does_not_turn_a_channel_into_a_sum(self):
        wire = SliceWire('auto')
        model = wire.model_for(self.base, self.layout)
        pictures = wire.pictures(self.values, self.layout)
        coeffs = wire.channel_coefficients(model, pictures, 'left')
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

    def test_mono_slices_leaves_the_other_channel_free(self):
        base = v7.load_model(TARGET, 'box')
        audio = _send(['--profile', 'mono-slices', '--mono-video-side', 'left',
                       '--source-audio', 'off'], _frame())
        np.testing.assert_array_equal(audio[:, 1], 0.0)
        halves = _Receiver(base).halves(audio[:, 0], 0)
        self.assertTrue(halves)
        self.assertEqual(halves[-1][1].kind, 'left')
        wire = SliceMonoWire(SliceWire('auto'), 'right')
        self.assertEqual((wire.carrier_index, wire._sides), (1, (None, 'left')))

    def test_stereo_slices_refuses_a_mono_output(self):
        with self.assertRaisesRegex(ValueError, 'two output channels'):
            args = v7_live.parser().parse_args([
                'send', '--device', 'memory', '--profile', 'stereo-slices'])
            args.mono_sum = True
            with mock.patch.dict(sys.modules, {
                    'sounddevice': types.ModuleType('sounddevice')}):
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
        layout = wire.layout_for(code)
        sent = wire.pictures(values, layout)[0]
        audio = wire.encode(base, [values]*24, aspect_codes=[code]*24)

        def error(frame):
            shown = _corner(np.asarray(frame.values)[:96*80].reshape(96, 80),
                            slice_shapes(layout)[0])
            return float(np.sqrt(np.mean((shown-sent)**2)))

        frames = self._receive(audio)
        self.assertGreaterEqual(len(frames), 8)
        self.assertEqual(tuple(frames[-1].pixel_shapes[0]),
                         SLICE_GRIDS[layout][0])
        self.assertLess(error(frames[-1]), .02)
        one = audio.copy()
        one[:, 1] = 0
        frames = self._receive(one)
        self.assertGreaterEqual(len(frames), 8)
        self.assertLess(error(frames[-1]), .15)


if __name__ == '__main__':
    unittest.main()
