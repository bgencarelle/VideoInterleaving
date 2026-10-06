"""A fixed sender aspect layout is signalled in the metadata (screen bit)."""
import sys
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT/'test_modem_v7', ROOT/'tools'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from animation_modem import v7                                           # noqa: E402
import tone_code                                                         # noqa: E402
from tools import v7_live                                                # noqa: E402
from tools.v7_capture import CapturedFrame                               # noqa: E402

GREY = 200


def _still(width=640, height=480):
    frame = np.full((height, width, 3), GREY, np.uint8)
    frame[::16] = 120                       # a little structure to decode
    return frame


def _send(argv, still, seconds='0.4'):
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
    previous_sounddevice = sys.modules.get('sounddevice')
    previous_capture = v7_live._capture
    sys.modules['sounddevice'] = fake
    v7_live._capture = lambda _args: (lambda: still)
    try:
        args = v7_live.parser().parse_args([
            'send', '--device', 'memory', '--source', 'test',
            '--seconds', seconds, '--no-log', '--dct-kernel', 'reference', *argv])
        v7_live.run_send(args)
    finally:
        v7_live._capture = previous_capture
        if previous_sounddevice is None:
            sys.modules.pop('sounddevice', None)
        else:
            sys.modules['sounddevice'] = previous_sounddevice
    return np.concatenate(written)


class FitToAspectTests(unittest.TestCase):
    def test_a_narrower_picture_is_pillarboxed_in_place(self):
        fitted = v7_live._fit_to_aspect(_still(), 3)            # 4:3 in 16:9
        self.assertEqual(fitted.shape, (480, 640, 3))
        bar = round(640*(1-(4/3)/(16/9))/2)
        self.assertTrue(np.all(fitted[:, :bar-1] == 0))
        self.assertTrue(np.all(fitted[:, -(bar-1):] == 0))
        self.assertTrue(np.all(fitted[1:-1, bar+2:-bar-2].max(axis=-1) > 0))

    def test_a_wider_picture_is_letterboxed_using_the_capture_size(self):
        # A prepared capture: grid-sized pixels standing for a 16:9 source.
        frame = CapturedFrame(np.full((96, 80, 3), GREY, np.uint8),
                              (1920, 1080), prepared=True)
        fitted = v7_live._fit_to_aspect(frame, 1)                # 16:9 in 4:3
        self.assertEqual(fitted.source_size, (1920, 1080))
        self.assertTrue(fitted.prepared)
        bar = round(96*(1-(4/3)/(16/9))/2)
        self.assertTrue(np.all(fitted.rgb[:bar-1] == 0))
        self.assertTrue(np.all(fitted.rgb[bar+1:-bar-1] == GREY))

    def test_values_carry_the_screen_flag(self):
        model = v7_live._model(v7_live.DEFAULT_FIXTURE, 'box')
        _values, plain = v7_live._values(model, _still(), 'box', 1.0)
        _values, fitted = v7_live._values(model, _still(), 'box', 1.0,
                                          fit_aspect=3)
        self.assertEqual(plain, 1)
        self.assertEqual(fitted, 3 | v7.ASPECT_SCREEN)

    def test_live_source_index_wraps_at_the_wire_field(self):
        self.assertEqual(v7_live._wire_index(1), 0)
        self.assertEqual(v7_live._wire_index(v7.MAX_SOURCE_INDEX+1),
                         v7.MAX_SOURCE_INDEX)
        self.assertEqual(v7_live._wire_index(v7.MAX_SOURCE_INDEX+2), 0)
        v7.metadata_word(0, 1, 0, v7_live._wire_index(10**7))


class SignalledLayoutTests(unittest.TestCase):
    def _receive(self, audio):
        args = v7_live.parser().parse_args(['receive', '--device', '0'])
        self.assertEqual(args.aspect_layout, 'auto')
        decoder = v7_live._make_auto_profile_decoder(
            args, v7_live._experimental_fold(500))
        # The live loop confirms the coded status first; aspect-fold-500's
        # is the former fold-off word.
        decoder.active_mode = decoder.dispatch_mode = tone_code.FOLD_OFF
        decoder.install()
        try:
            with tone_code.coded_pilot_timing():
                results, _ = v7.decode_pulse_stream(
                    decoder.base_model, audio, sample_rate=v7.RATE,
                    state=v7.PulseState(), pilot_timing='tone-seeded',
                    frame_boundary='eof')
            good = [result for result in results
                    if result.status in ('received', 'verified')]
            self.assertTrue(good)
            return good[-1], decoder.values(decoder.base_model, good[-1])
        finally:
            decoder.uninstall()

    def test_auto_receiver_follows_a_fixed_sender_layout(self):
        audio = _send(['--profile', 'aspect-fold-500',
                       '--aspect-layout', '16:9'], _still())
        result, values = self._receive(audio)
        self.assertEqual(result.diag['aspect_code'], 3)
        self.assertTrue(result.diag['aspect_screen'])
        self.assertEqual(result.diag['aspect_layout'], '16:9')
        # The 4:3 picture sits pillarboxed in the 16:9 frame.
        rows, cols = v7.V7_GRIDS[0]
        luma = np.asarray(values[:rows*cols]).reshape(rows, cols)
        edge = max(1, round(cols*(1-(4/3)/(16/9))/2)-3)
        middle = luma[:, cols//2-8:cols//2+8].mean()
        self.assertLess(luma[:, :edge].mean(), .5*middle)
        self.assertLess(luma[:, -edge:].mean(), .5*middle)

    def test_auto_layout_sends_the_picture_aspect(self):
        audio = _send(['--profile', 'aspect-fold-500'], _still())
        result, _values = self._receive(audio)
        self.assertEqual(result.diag['aspect_code'], 1)
        self.assertFalse(result.diag['aspect_screen'])
        self.assertEqual(result.diag['aspect_layout'], '4:3')


if __name__ == '__main__':
    unittest.main()
