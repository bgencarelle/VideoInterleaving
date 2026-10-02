"""Reverse playback of every live wire profile through the torture matrix.

Each profile's packets go through a channel model, are played backwards, and
are decoded with the receiver's profile dispatcher. A reversed packet must
show the same picture as the same packet played forwards.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT/'test_modem_v7'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from animation_modem import v7                                         # noqa: E402
from tools import v7_live                                              # noqa: E402
from tools.v7_torture_matrix import CASES, impair                       # noqa: E402
import tone_code                                                       # noqa: E402
from aspect_fold import AspectFoldWire                                 # noqa: E402
from aspect_mono import AspectMonoWire                                 # noqa: E402
from live_fold import LiveFold                                         # noqa: E402

TARGET = .1521/np.sqrt(1 + 10**(v7.CLOCK_REL_DB/10))
PACKETS = 9
SITUATIONS = ('clean-96k', 'hiss-40', 'wow-flutter', 'type-i', 'dropouts')
# A fresh receiver cannot verify the two packets in seven whose CRC field
# carries a loop value it has not learned yet, and reverse playback shows
# only independently verified packets.
UNVERIFIED = 3
# RMS difference allowed between the reversed and the forward decode of one
# packet, in grid value units (the picture spans -1..1). Under noise the two
# directions measure timing from opposite ends of the packet, so a few folded
# coefficients land on different steps (measured: up to .032).
SAME_PICTURE = .02
SAME_NOISY_PICTURE = .05
QUIET = ('clean-96k', 'dropouts')
# Playback speed -> the resampling that plays a recording at that speed.
SPEEDS = {.5: (2, 1), 1.5: (2, 3), 2.0: (1, 2)}


class ReverseTortureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = v7.load_model(TARGET, 'box')
        cls.cases = {case.name: case for case in CASES}
        with Image.open(v7.REFERENCE_FIXTURE) as image:
            cls.rgb = np.asarray(image.convert('RGB'))
        cls.dispatcher = v7_live._AdaptiveProfileDecoder(
            v7_live._experimental_fold(500), cls.base,
            aspect_tail=v7_live.DEFAULT_ASPECT_TAIL)

    def _values(self, masks=None):
        return v7_live._values(self.base, self.rgb, 'box', 1.0,
                               dct_encode=True, chroma_sent_for=masks)[:2]

    def _packets(self, model, coefficients, code, mode, profile_code):
        return np.concatenate([
            tone_code.add_tone_code(
                v7.encode_pulse_frame_coeffs(
                    model, coefficients, counter, aspect_code=code,
                    source_index=counter-1, pilot_tones=False,
                    eof_marker=True, pulse_profile_code=profile_code),
                counter, tone_code.encode_status(mode))
            for counter in range(1, PACKETS+1)]).astype(np.float64)

    def _aspect_fold(self):
        wire = AspectFoldWire('auto', v7_live.DEFAULT_ASPECT_TAIL)
        values, code = self._values(lambda c: v7_live._chroma_sent_masks(
            wire.codec(wire.model_for(self.base, wire.layout_for(c)))))
        model, coefficients = wire.encode_coefficients(self.base, values, code)
        return (self._packets(model, coefficients, code, wire.status_mode,
                              wire.pulse_profile_code), wire.status_mode, None)

    def _fold_500(self):
        fold = LiveFold(500)
        values, code = self._values()
        return (self._packets(self.base,
                              fold.encode_coefficients(self.base, values),
                              code, tone_code.FOLD_500, tone_code.FOLD_500),
                tone_code.FOLD_500, None)

    def _aspect_mono(self):
        wire = AspectMonoWire(self.base, side='right')
        values, code = self._values(lambda c: v7_live._chroma_sent_masks(
            wire._codec(wire._packet_model(self.base, c))))
        audio = np.concatenate([
            wire.encode_packet(self.base, values, counter, aspect_code=code,
                               source_index=counter-1)
            for counter in range(1, PACKETS+1)]).astype(np.float64)
        return audio, wire.status_mode, 1

    def _stereo_slices(self):
        wire = self.dispatcher.slice_wire
        for code in range(8):
            wire.model_for(self.base, wire.layout_for(code))
        values, code = self._values(
            lambda c: wire.chroma_sent_masks(wire.layout_for(c)))
        audio = wire.encode(self.base, [values]*PACKETS,
                            aspect_codes=[code]*PACKETS)
        # Reverse playback reads one slices channel (a known limit).
        return audio.astype(np.float64), wire.status_mode, 0

    def _decode(self, audio, rate, mode, leg, reverse):
        dispatcher = self.dispatcher
        dispatcher._last_slices = False
        dispatcher._last_layouts.clear()
        dispatcher.slice_wire.reset()
        if leg is not None:
            audio = audio[:, leg:leg+1]
        audio = np.ascontiguousarray(audio[::-1] if reverse else audio,
                                     dtype=np.float32)
        shown = []
        dispatcher.install()
        try:
            with tone_code.coded_pilot_timing():
                dispatcher.active_mode = dispatcher.dispatch_mode = mode
                if leg is not None:
                    dispatcher.active_side = 0
                state = v7.PulseState(tail_memory=False)
                if reverse:
                    state.set_playback_direction(-1)
                    results = []
                    for start, scale, _confidence, way in v7.pulse_frame_hits(
                            audio, sample_rate=rate):
                        if way < 0:
                            results += v7.decode_reverse_packet(
                                self.base, audio, start, scale, state=state,
                                sample_rate=rate,
                                pilot_timing='tone-seeded')[0]
                else:
                    results, _ = v7.decode_pulse_stream(
                        self.base, audio, state=state, sample_rate=rate,
                        pilot_timing='tone-seeded', frame_boundary='eof')
            for result in results:
                index = result.diag.get('source_index')
                if index is None or (result.status == 'lost' and
                                     not result.diag.get('displayable')):
                    continue
                shown.append((int(index), np.asarray(
                    dispatcher.values(self.base, result), float)))
        finally:
            dispatcher.uninstall()
        return shown

    def _profiles(self):
        return {'aspect-fold-500': self._aspect_fold,
                'fold-500': self._fold_500,
                'aspect-mono-500': self._aspect_mono,
                'stereo-slices': self._stereo_slices}

    def test_other_playback_speeds_show_the_same_picture_both_ways(self):
        for name, build in self._profiles().items():
            audio, mode, leg = build()
            wide = resample_poly(audio, 2, 1, axis=0)
            normal = dict(self._decode(wide, 96000, mode, leg, reverse=False))
            for speed, (up, down) in SPEEDS.items():
                capture = resample_poly(wide, up, down, axis=0)
                for reverse in (False, True):
                    with self.subTest(profile=name, speed=speed,
                                      reverse=reverse):
                        shown = self._decode(capture, 96000, mode, leg,
                                             reverse=reverse)
                        order = [index for index, _values in shown]
                        self.assertEqual(order,
                                         sorted(order, reverse=reverse))
                        self.assertGreaterEqual(
                            len(order), PACKETS-UNVERIFIED-1)
                        for index, values in shown:
                            if index in normal:
                                self.assertLess(float(np.sqrt(np.mean(
                                    (values-normal[index])**2))),
                                    SAME_NOISY_PICTURE, (name, speed, index))

    def test_reversed_packets_show_the_forward_picture(self):
        for name, build in self._profiles().items():
            audio, mode, leg = build()
            wide = resample_poly(audio, 2, 1, axis=0)
            for situation in SITUATIONS:
                with self.subTest(profile=name, situation=situation):
                    capture = impair(wide, self.cases[situation], seed=2026)
                    forward = dict(self._decode(capture, 96000, mode, leg,
                                                reverse=False))
                    backward = self._decode(capture, 96000, mode, leg,
                                            reverse=True)
                    order = [index for index, _values in backward]
                    self.assertEqual(order, sorted(order, reverse=True))
                    self.assertGreaterEqual(len(order),
                                            len(forward)-UNVERIFIED)
                    self.assertGreaterEqual(len(order), PACKETS-UNVERIFIED-1)
                    compared = 0
                    for index, values in backward:
                        if index in forward:
                            compared += 1
                            difference = float(np.sqrt(np.mean(
                                (values-forward[index])**2)))
                            self.assertLess(difference, SAME_PICTURE if situation in QUIET else SAME_NOISY_PICTURE,
                                            (name, situation, index))
                    self.assertGreaterEqual(compared, PACKETS-UNVERIFIED-2)


if __name__ == '__main__':
    unittest.main()
