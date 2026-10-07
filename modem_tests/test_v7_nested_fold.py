"""Nested-fold modes through the live sender wires and receiver dispatcher.

Covers the two experimental options (mono nested fold, stereo redundant
nested fold): shipped tables, stock packets left untouched, more detail on a
clean link, gentle fallback under noise, one channel and mono sum of the
stereo mode, other playback speeds, reversed waveforms, the live sender CLI
and the sender GUI.
"""
import hashlib
import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT/'test_modem_v7'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from animation_modem import v7                                         # noqa: E402
from tools import v7_live                                              # noqa: E402
from tools.v7_nested_eval import Rig, wide                              # noqa: E402
from tools.v7_torture_matrix import CASES, Case, impair                 # noqa: E402
import nested_fold                                                     # noqa: E402
from aspect_fold import LAYOUT_NAMES                                   # noqa: E402

LUMA = 96*80
PACKETS = 9
SPEEDS = {.5: (2, 1), 1.5: (2, 3), 2.0: (1, 2)}


def textured(seed=7, height=512, width=384):
    """A 3:4 picture with detail at every scale (1/f noise), unlike the small
    smooth fixture face, so there is fine detail for a mode to carry."""
    rng = np.random.default_rng(seed)
    fy, fx = np.meshgrid(np.fft.fftfreq(height), np.fft.fftfreq(width), indexing='ij')
    spectrum = (rng.standard_normal((height, width))+1j*rng.standard_normal((height, width)))
    spectrum /= np.maximum(np.hypot(fy, fx), 1/height)**1.2
    field = np.real(np.fft.ifft2(spectrum))
    field = (field-field.mean())/field.std()
    grey = np.uint8(np.clip(128+48*field, 0, 255))
    return np.ascontiguousarray(np.repeat(grey[..., None], 3, -1))


def luma_error(values, sent):
    return float(np.sqrt(np.mean((np.asarray(values)[:LUMA]-sent[:LUMA])**2)))


class TableTests(unittest.TestCase):
    def test_every_layout_has_both_tables_and_they_match_the_record(self):
        record = json.loads(nested_fold.TABLES.with_suffix('.json').read_text())
        with np.load(nested_fold.TABLES) as data:
            for family in ('mono', 'slices'):
                for layout in LAYOUT_NAMES:
                    fold = nested_fold.table(family, layout)
                    self.assertIsNotNone(fold, (family, layout))
                    self.assertEqual(fold.legs, 2 if family == 'slices' else 1)
                    self.assertEqual(fold.paired, family == 'slices')
                    prefix = f'{family}/{layout}/'
                    digest = hashlib.sha256()
                    for name in sorted(n[len(prefix):] for n in data.files if n.startswith(prefix)):
                        digest.update(np.ascontiguousarray(data[prefix+name]).tobytes())
                    self.assertEqual(digest.hexdigest(),
                                     record['tables'][f'{family}/{layout}']['sha256'])

    def test_a_layout_without_a_table_is_left_on_the_stock_mapping(self):
        self.assertIsNone(nested_fold.table('mono', '3:4', path='/nonexistent/tables.npz'))

    def test_level_patterns_are_orthogonal_to_each_other_and_to_stock(self):
        rows = np.array([nested_fold.level_row(i)
                         for i in range(nested_fold.LOWEST, nested_fold.HIGHEST+1)])
        np.testing.assert_allclose(rows@rows.T, 16*np.eye(len(rows)))
        np.testing.assert_allclose(rows.sum(axis=1), 0)            # stock is all ones
        rng = np.random.default_rng(1)
        for index in range(nested_fold.LOWEST, nested_fold.HIGHEST+1):
            noisy = nested_fold.level_row(index)+.8*rng.standard_normal(16)
            score, level, _, stock, dithered = nested_fold.read_signature(noisy)
            self.assertEqual(level, index)
            self.assertGreater(score, stock)
            self.assertFalse(dithered)
        score, _, _, stock, _ = nested_fold.read_signature(np.ones(16))
        self.assertGreater(stock, score)

    def test_a_dithered_packet_carries_its_level_pattern_negated(self):
        rng = np.random.default_rng(2)
        for index in range(nested_fold.LOWEST, nested_fold.HIGHEST+1):
            row = nested_fold.level_row(index, dithered=True)
            np.testing.assert_array_equal(row, -nested_fold.level_row(index))
            score, level, residual, stock, dithered = nested_fold.read_signature(
                row+.8*rng.standard_normal(16))
            self.assertEqual(level, index)
            self.assertTrue(dithered)
            self.assertGreater(score, max(stock, nested_fold.SCORE_MIN))
            self.assertLess(nested_fold.read_signature(row)[2], 1e-12)

    def test_dithered_stairs_average_out_over_the_dither_cycle(self):
        """A plain staircase makes the same error in every packet; shifted
        stairs make a different one each time, and the sum of a channel pair's
        offsets is nothing, so a mono sum reads as before."""
        phases = nested_fold.DITHER_PHASES
        for family in ('mono', 'slices'):
            fold = nested_fold.table(family, '3:4')
            folded = fold.level > 0
            np.testing.assert_array_equal(fold.offset(3, 0)+fold.offset(3, 1), 0)
            self.assertIsNone(fold.offset(None))
            self.assertLessEqual(float(np.max(np.abs(fold.offset(3)/fold.step))), .5)
            self.assertTrue(np.all(fold.offset(3)[~folded] == 0))
            plane = np.zeros(LUMA)
            rng = np.random.default_rng(5)
            plane[fold.host_position] = fold.host_mean+fold.host_sd*rng.laplace(
                scale=np.sqrt(.5), size=fold.slots)
            sigma = .5*fold.noise

            def hosts(phase):
                got = fold.decode({leg: (fold.encode(plane, 0, leg, phase), sigma)
                                   for leg in range(fold.legs)}, 0, phase, averaged=True)
                return (got-plane)[fold.host_position][folded]/fold.host_sd[folded]

            plain = hosts(None)
            each = np.array([hosts(phase) for phase in range(phases)])
            single = float(np.mean(each**2))
            self.assertLess(single, 1.25*float(np.mean(plain**2)), family)
            self.assertLess(float(np.mean(each.mean(axis=0)**2)), single/2.5, family)


class LiveTests(unittest.TestCase):
    """What the fold itself carries: the decoded values, as they are
    (BoundedDecodeTests covers the picture shown from them)."""

    @classmethod
    def setUpClass(cls):
        patcher = mock.patch.object(nested_fold, 'SMOOTH_PASSES', 0)
        patcher.start()
        cls.addClassCleanup(patcher.stop)
        cls.rig = Rig()
        cls.rgb = textured()
        cls.sent = v7_live._values(cls.rig.base, cls.rgb, 'box', 1.0, dct_encode=True)[0]
        cls.audio = {mode: wide(cls.rig.audio(mode, cls.rgb, PACKETS))
                     for mode in ('aspect-mono-500', 'mono nested', 'stereo-slices',
                                  'stereo nested')}
        cls.cases = {case.name: case for case in CASES}

    def error(self, mode, capture, **kwargs):
        shown = self.rig.shown(mode, capture, **kwargs)
        self.assertGreaterEqual(len(shown), PACKETS-2, mode)
        return float(np.mean([luma_error(values, self.sent) for _, values, _ in shown]))

    def test_nested_packets_are_auto_levelled_under_the_header(self):
        size, body = v7.PULSE_FRAME, slice(v7.PULSE.SYNC_LEN, v7.PULSE.SYNC_LEN+v7.FRAME)
        for mode in ('mono nested', 'stereo nested'):
            audio = self.rig.audio(mode, self.rgb, 3)
            for start in range(0, len(audio), size):
                packet = audio[start:start+size]
                framing_peak = min(
                    float(np.max(np.abs(packet[:v7.PULSE.SYNC_LEN]))),
                    float(np.max(np.abs(packet[v7.EOF_MARKER_OFFSET:]))))
                ceiling = framing_peak*10**(-v7.BODY_BELOW_EOF_DB/20)
                for channel in packet.T:
                    if not np.any(channel):
                        continue
                    header = float(np.max(np.abs(channel[:v7.PULSE.SYNC_LEN])))
                    peak = float(np.max(np.abs(channel[body])))
                    # The header stays the loudest part, so it clips first...
                    self.assertGreater(20*np.log10(header/peak), 1.0, mode)
                    self.assertLess(float(np.max(np.abs(channel))), 1.0)
                    # EOF packets use the lower framing ceiling instead of
                    # historical BODY_PEAK. These standalone wires add their
                    # status-tone overlay afterwards; the header-order checks
                    # above also cover that final overlay.
                    self.assertGreater(peak, .85*ceiling, mode)
        # Outside the nested senders a quiet body goes out as coded; with
        # auto-level it is raised to the ceiling its framing sets.
        model = self.rig.base
        quiet = model.mu+.3*np.sqrt(model.lam)*np.random.default_rng(4).standard_normal(len(model.mu))
        plain = v7.encode_pulse_frame_coeffs(model, quiet, 1)
        with v7.body_auto_level():
            raised = v7.encode_pulse_frame_coeffs(model, quiet, 1)
        framing = min(float(np.max(np.abs(raised[:v7.PULSE.SYNC_LEN]))),
                      float(np.max(np.abs(raised[v7.EOF_MARKER_OFFSET:]))))
        ceiling = framing*10**(-v7.BODY_BELOW_EOF_DB/20)
        self.assertLess(float(np.max(np.abs(plain[body]))), .8*ceiling)
        self.assertAlmostEqual(float(np.max(np.abs(raised[body]))), ceiling, delta=.01)

    def test_stock_packets_are_still_decoded_by_the_stock_decoders(self):
        self.error('aspect-mono-500', self.audio['aspect-mono-500'])
        codec = self.rig.dispatcher.aspect_mono_wire._codec(
            self.rig.dispatcher.aspect_mono_wire._packet_model(self.rig.base, 5))
        self.assertFalse(codec.last_nested)
        self.error('stereo-slices', self.audio['stereo-slices'])
        self.assertFalse(self.rig.dispatcher.slice_wire.last_nested)

    def test_nested_packets_are_recognised_and_carry_more_detail(self):
        stock = self.error('aspect-mono-500', self.audio['aspect-mono-500'])
        nested = self.error('mono nested', self.audio['mono nested'])
        codec = self.rig.dispatcher.aspect_mono_wire._codec(
            self.rig.dispatcher.aspect_mono_wire._packet_model(self.rig.base, 5))
        self.assertTrue(codec.last_nested)
        self.assertLess(nested, .98*stock)
        slices = self.error('stereo-slices', self.audio['stereo-slices'])
        stereo = self.error('stereo nested', self.audio['stereo nested'])
        self.assertTrue(self.rig.dispatcher.slice_wire.last_nested)
        self.assertLess(stereo, .98*slices)
        # Stereo's job: better than the mono mode on the same link.
        self.assertLess(stereo, nested)

    def test_noise_fades_the_picture_gently_with_no_collapse(self):
        previous = {'mono nested': 0.0, 'stereo nested': 0.0}
        for level in (-50, -40, -35, -30):
            case = Case(f'hiss{level}', noise_dbfs=level)
            for mode, stock in (('mono nested', 'aspect-mono-500'),
                                ('stereo nested', 'stereo-slices')):
                with self.subTest(mode=mode, level=level):
                    nested = self.error(mode, impair(self.audio[mode], case, seed=3))
                    plain = self.error(stock, impair(self.audio[stock], case, seed=3))
                    # Never far behind the stock mapping, however bad the link.
                    self.assertLess(nested, 1.25*plain)
                    # And worse links never look better.
                    self.assertGreater(nested, .9*previous[mode])
                    previous[mode] = nested

    def test_either_stereo_channel_alone_and_a_mono_sum_are_whole_pictures(self):
        both = self.error('stereo nested', self.audio['stereo nested'])
        stock_one = self.error('stereo-slices', self.audio['stereo-slices'], legs=[0])
        for leg in (0, 1):
            one = self.error('stereo nested', self.audio['stereo nested'], legs=[leg])
            self.assertGreater(one, both)
            self.assertLess(one, 1.1*stock_one)
        summed = impair(self.audio['stereo nested'], self.cases['mono-sum'])
        self.assertLess(self.error('stereo nested', summed),
                        1.2*self.error('stereo-slices',
                                       impair(self.audio['stereo-slices'], self.cases['mono-sum'])))

    def test_stereo_survives_level_phase_and_crosstalk_differences(self):
        clean = self.error('stereo nested', self.audio['stereo nested'])
        for case in (Case('right', right_gain_db=-12), Case('azimuth', azimuth_us=25),
                     Case('crosstalk', crosstalk=.25)):
            with self.subTest(case=case.name):
                self.assertLess(self.error('stereo nested',
                                           impair(self.audio['stereo nested'], case)), 1.1*clean)

    def test_other_speeds_and_reversed_waveforms_show_the_same_picture(self):
        for mode in ('mono nested', 'stereo nested'):
            normal = dict((index, values) for index, values, _ in
                          self.rig.shown(mode, self.audio[mode]))
            captures = {1.0: self.audio[mode]}
            for speed, (up, down) in SPEEDS.items():
                captures[speed] = resample_poly(self.audio[mode], up, down, axis=0)
            for speed, capture in captures.items():
                for reverse in (False, True):
                    if speed == 1.0 and not reverse:
                        continue
                    with self.subTest(mode=mode, speed=speed, reverse=reverse):
                        shown = self.rig.shown(mode, capture, reverse=reverse)
                        order = [index for index, _, _ in shown]
                        self.assertEqual(order, sorted(order, reverse=reverse))
                        # A fresh reverse receiver holds its first packets,
                        # exactly as it does for the stock profiles.
                        self.assertGreaterEqual(len(order), PACKETS-4)
                        for index, values, _ in shown:
                            if index in normal:
                                self.assertLess(float(np.sqrt(np.mean(
                                    (values-normal[index])**2))), .05, (mode, speed, index))


class HeldPictureTests(unittest.TestCase):
    """Dithered stairs through the live sender and receiver."""

    @classmethod
    def setUpClass(cls):
        # Dither and the held average are off by default; these tests are of
        # what they do when switched on.
        for name, value in (('SMOOTH_PASSES', 0), ('DITHER', True), ('HOLD', True)):
            patcher = mock.patch.object(nested_fold, name, value)
            patcher.start()
            cls.addClassCleanup(patcher.stop)
        cls.rig = Rig()
        cls.rgb = textured()
        cls.sent = v7_live._values(cls.rig.base, cls.rgb, 'box', 1.0, dct_encode=True)[0]

    def test_the_stair_work_arounds_are_off_unless_asked_for(self):
        import importlib
        import os
        with mock.patch.dict(os.environ, {}, clear=False):
            for name in ('NESTED_FOLD_DITHER', 'NESTED_FOLD_HOLD', 'NESTED_FOLD_SMOOTH',
                         'NESTED_FOLD_DISPLAY_ROOM'):
                os.environ.pop(name, None)
            fresh = importlib.util.module_from_spec(importlib.util.spec_from_file_location(
                'nested_fold_defaults', nested_fold.__file__))
            fresh.__spec__.loader.exec_module(fresh)
        self.assertFalse(fresh.DITHER)
        self.assertFalse(fresh.HOLD)
        self.assertEqual(fresh.SMOOTH_PASSES, 0)
        self.assertEqual(fresh.DISPLAY_ROOM, 0.0)

    def shown(self, mode, audio):
        """Luma error of each picture shown in the band where that mode's
        stair error is seen (cycles per picture height): the mono fold's
        below 12, the stereo fold's from 12 to 24."""
        from tools.v7_sk_wire import shipped_plane
        rows, cols = np.arange(96)[:, None], np.arange(80)[None, :]
        radius = np.hypot(rows, cols*96/80)/2
        low = (radius >= 12) & (radius < 24) if 'stereo' in mode else radius < 12
        sent = shipped_plane(self.sent)
        return [float(np.sqrt(np.sum((shipped_plane(values)-sent)[low]**2)))
                for _, values, _ in self.rig.shown(mode, audio)]

    def test_a_held_picture_gains_from_the_dither_cycle_and_plain_stairs_do_not(self):
        packets = 2*nested_fold.DITHER_PHASES
        for mode in ('mono nested', 'stereo nested'):
            dithered = self.shown(mode, wide(self.rig.audio(mode, self.rgb, packets)))
            with mock.patch.object(nested_fold, 'DITHER', False):
                plain = self.shown(mode, wide(self.rig.audio(mode, self.rgb, packets)))
            self.assertGreaterEqual(min(len(dithered), len(plain)), packets-2, mode)
            # The first packet of a picture is shown as decoded, at about the
            # plain-stair error; once the cycle is in, the error is lower.
            self.assertLess(dithered[0], 1.1*plain[0], mode)
            self.assertLess(dithered[-1], .7*plain[-1], mode)
            self.assertAlmostEqual(plain[0], plain[-1], delta=.02*plain[0], msg=mode)

    def test_the_average_can_be_switched_off(self):
        audio = wide(self.rig.audio('mono nested', self.rgb, nested_fold.DITHER_PHASES+2))
        held = self.shown('mono nested', audio)
        with mock.patch.object(nested_fold, 'HOLD', False):
            each = self.shown('mono nested', audio)
        self.assertLess(held[-1], .8*each[-1])
        self.assertAlmostEqual(held[0], each[0], places=9)

    def test_a_changed_picture_is_shown_as_decoded(self):
        fold = nested_fold.table('mono', '3:4')
        held = nested_fold.Held()
        rng = np.random.default_rng(3)
        sigma = .5*fold.noise

        def picture(seed):
            plane = np.zeros(LUMA)
            plane[fold.host_position] = fold.host_mean+fold.host_sd*np.random.default_rng(
                seed).laplace(scale=np.sqrt(.5), size=fold.slots)
            return plane

        def decoded(plane, phase):
            return fold.decode({0: (fold.encode(plane, 0, 0, phase), sigma)}, 0, phase)

        first, second = picture(1), picture(2)
        for phase in range(4):
            out = held.update(fold, decoded(first, phase), 0, phase, sigma)
        self.assertEqual(held.last_count, 4)
        moved = decoded(second, 4)
        np.testing.assert_array_equal(held.update(fold, moved, 0, 4, sigma), moved)
        self.assertEqual(held.last_count, 1)
        # A packet out of sequence, and plain stairs, start again too.
        held.update(fold, decoded(second, 5), 0, 5, sigma)
        self.assertEqual(held.last_count, 2)
        held.update(fold, decoded(second, 1), 0, 1, sigma)
        self.assertEqual(held.last_count, 1)
        plain = decoded(second, None)
        np.testing.assert_array_equal(held.update(fold, plain, 0, None, sigma), plain)
        self.assertEqual(held.last_count, 0)
        del out, rng


class BoundedDecodeTests(unittest.TestCase):
    """Decoding inside the bounds: the cleanest picture that fits."""

    def test_room_is_half_a_stair_for_folded_hosts_and_nothing_for_guests(self):
        for family in ('mono', 'slices'):
            fold = nested_fold.table(family, '3:4')
            sigma = np.zeros(fold.slots)
            room = fold.room(0, sigma)
            folded = fold.level > 0
            np.testing.assert_allclose(room[fold.host_position][folded],
                                       (.5*nested_fold.SMOOTH_ROOM*fold.step*fold.host_sd)[folded])
            self.assertTrue(np.all(room[fold.host_position][~folded] == 0))
            self.assertTrue(np.all(room[fold.guest_position.ravel()] == 0))
            self.assertTrue(np.all(np.isinf(room[~fold.sent_mask])))
            self.assertTrue(np.all(np.isfinite(room[fold.sent_mask])))
            # A level up the ladder is 1.5 times the room; seven averaged
            # dithered packets leave less of it; noise adds to it.
            np.testing.assert_allclose(fold.room(1, sigma)[fold.host_position],
                                       1.5*room[fold.host_position])
            self.assertLess(fold.room(0, sigma, count=7)[fold.host_position][folded].max(),
                            room[fold.host_position][folded].max())
            self.assertTrue(np.all(fold.room(0, sigma+.1)[fold.host_position] >
                                   room[fold.host_position]))
        # One channel of the two-channel table bounds only base plus or
        # minus detail, so the picture is shown as decoded.
        self.assertIsNone(fold.room(0, sigma, joined=False))
        luma = np.random.default_rng(1).standard_normal(LUMA)
        self.assertIs(fold.clean(luma, None), luma)
        self.assertIs(fold.clean(luma, room, passes=0), luma)

    def test_the_picture_shown_is_cleaner_and_no_further_from_the_source(self):
        rig = Rig()
        # White blocks on black: flat areas, where stair error shows most.
        rgb = np.zeros((512, 384, 3), np.uint8)
        rgb[120:300, 60:150] = 255
        rgb[200:420, 220:330] = 255
        sent = v7_live._values(rig.base, rgb, 'box', 1.0, dct_encode=True)[0]
        flat = np.abs(sent[:LUMA]) > .98          # black and white areas away from edges

        def shown(mode):
            values = rig.shown(mode, wide(rig.audio(mode, rgb, 3)))[-1][1]
            return (float(np.sqrt(np.mean((values[:LUMA]-sent[:LUMA])[flat]**2))),
                    luma_error(values, sent))

        for mode in ('mono nested', 'stereo nested'):
            with mock.patch.object(nested_fold, 'HOLD', False):
                with mock.patch.object(nested_fold, 'SMOOTH_PASSES', 16):
                    cleaned = shown(mode)
                with mock.patch.object(nested_fold, 'SMOOTH_PASSES', 0):
                    decoded = shown(mode)
            self.assertLess(cleaned[0], .8*decoded[0], mode)
            self.assertLess(cleaned[1], decoded[1], mode)


class DamagedPacketTests(unittest.TestCase):
    """Slot-by-slot damage: what the equaliser doubts is not spread over the
    packet, and a packet whose signature cannot be read keeps the stream's
    mapping."""

    def test_packet_noise_leaves_out_what_the_equaliser_attributes_to_those_slots(self):
        hiss = nested_fold.packet_noise(.3, .1)
        self.assertAlmostEqual(hiss, 3.0)
        own = np.full(16, .3/nested_fold.SLOT_NOISE_SHARE)
        self.assertAlmostEqual(nested_fold.packet_noise(.3, .1, own), 0.0)
        sigma = nested_fold.slot_noise(0.0, np.full(4, .02),
                                       confidence=np.array([1., .9, .5, .01]))
        self.assertAlmostEqual(sigma[0], .01)                 # floor: half the design
        self.assertTrue(np.all(np.diff(sigma) > 0))            # doubt raises a slot's noise

    def test_an_unreadable_signature_keeps_the_streams_mapping(self):
        seen = nested_fold.Recognition()
        self.assertEqual(seen.decide(.9, 4, .0), (True, 4))    # a nested packet, level 4
        self.assertEqual(seen.decide(.1, -2, .1), (True, 4))   # unreadable: still nested, level 4
        self.assertEqual(seen.decide(.0, 0, .95), (False, 4))  # a stock packet
        self.assertFalse(seen.decide(.1, 3, .1)[0])            # unreadable: still stock
        for _ in range(nested_fold.STICKY_PACKETS+1):
            seen.decide(.1, 3, .1)
        self.assertIsNone(seen.kind)                           # long silence: forgotten
        fresh = nested_fold.Recognition()
        self.assertFalse(fresh.decide(.1, 3, .1)[0])           # never seen anything: not nested


class LiveReceiverReverseTests(unittest.TestCase):
    def _receive(self, audio):
        """The real receive loop on ``audio``; the luma of published frames."""
        import threading
        import time
        audio = np.concatenate([np.zeros((4800, 2), np.float32), np.asarray(audio, np.float32),
                                np.zeros((9600, 2), np.float32)])
        stop_event, frames = threading.Event(), []

        class SoundDevice:
            devices = [{'name': 'in', 'hostapi': 0, 'max_input_channels': 2,
                        'max_output_channels': 0, 'default_samplerate': 48000.0}]

            @classmethod
            def query_devices(cls, device=None, _kind=None):
                return list(cls.devices) if device is None else cls.devices[int(device)]

            @staticmethod
            def query_hostapis():
                return [{'name': 'Test API'}]

            class InputStream:
                def __init__(self, **kwargs):
                    self.kwargs, self.samplerate, self.active = kwargs, float(kwargs['samplerate']), True

                def start(self):
                    def feed():
                        callback, seen = self.kwargs['callback'], 0
                        for start in range(0, len(audio), 1024):
                            callback(audio[start:start+1024], 1024, None, None)
                            time.sleep(.012)
                            frame = v7_live.FRAME_BUFFER.snapshot()
                            if frame is not None and frame.generation != seen:
                                seen = frame.generation
                                frames.append(np.asarray(frame.values, float)[:LUMA])
                        time.sleep(.4)
                        stop_event.set()
                    threading.Thread(target=feed, daemon=True).start()

                def stop(self):
                    self.active = False

                def close(self):
                    self.active = False

        args = v7_live.parser().parse_args(['receive', '--device', '0', '--headless', '--no-log'])
        args.stop_event = stop_event
        with mock.patch.dict(sys.modules, {'sounddevice': SoundDevice}):
            v7_live.run_receive(args)
        return frames

    def test_a_reversed_stereo_waveform_joins_both_channels_like_forward(self):
        rig = Rig()
        rgb = textured()
        sent = v7_live._values(rig.base, rgb, 'box', 1.0, dct_encode=True)[0][:LUMA]
        audio = rig.audio('stereo nested', rgb, 24).astype(np.float32)

        def error(frames):
            self.assertGreaterEqual(len(frames), 8)
            return float(np.mean([np.sqrt(np.mean((f-sent)**2)) for f in frames[len(frames)//2:]]))

        forward = error(self._receive(audio))
        backward = error(self._receive(audio[::-1].copy()))
        one = audio.copy()
        one[:, 1] = 0
        single = error(self._receive(one[::-1].copy()))
        # Reversed in memory and read as usual, both channels joined: the
        # forward picture, and clearly better than one channel alone.
        self.assertLess(abs(backward-forward), .05*forward)
        self.assertLess(backward, .95*single)


class SenderTests(unittest.TestCase):
    def _send(self, profile):
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

        with Image.open(v7.REFERENCE_FIXTURE) as image:
            still = np.asarray(image.convert('RGB'))
        fake = types.ModuleType('sounddevice')
        fake.OutputStream = OutputStream
        with mock.patch.dict(sys.modules, {'sounddevice': fake}), \
                mock.patch.object(v7_live, '_capture', lambda _args: (lambda: still)):
            for duration in ('0.8', '5'):
                args = v7_live.parser().parse_args([
                    'send', '--device', 'memory', '--source', 'test', '--seconds', duration,
                    '--no-log', '--dct-encode', '--profile', profile])
                v7_live.run_send(args)
                if written:
                    break
        return np.concatenate(written)

    def test_the_live_sender_emits_nested_packets_for_both_options(self):
        rig = Rig()
        for profile, mode in (('aspect-mono-nested', 'mono nested'),
                              ('stereo-nested', 'stereo nested')):
            with self.subTest(profile=profile):
                audio = self._send(profile)
                self.assertEqual(audio.shape[1], 2)
                capture = audio
                if profile == 'aspect-mono-nested':
                    # The mono sender uses the right channel by default.
                    capture = audio[:, ::-1]
                shown = rig.shown(mode, np.asarray(capture, np.float64), rate=v7.RATE)
                self.assertTrue(shown)
                if profile == 'stereo-nested':
                    self.assertTrue(rig.dispatcher.slice_wire.last_nested)
                else:
                    wire = rig.dispatcher.aspect_mono_wire
                    self.assertTrue(wire._codec(wire._packet_model(rig.base, 5)).last_nested)

    def test_profile_names_map_to_their_wires(self):
        for profile, flags in (('aspect-mono-nested', ('experimental_mono_fold', 'aspect_mono')),
                               ('stereo-nested', ('slices',))):
            args = v7_live.parser().parse_args(['send', '--device', 'memory',
                                                '--profile', profile])
            v7_live._apply_profile_option(args)
            self.assertTrue(args.nested_fold)
            for flag in flags:
                self.assertTrue(getattr(args, flag))
            self.assertEqual(v7_live._kernel_defaults_profile(args),
                             v7_live.NESTED_BASE_PROFILES[profile])

    def test_the_sender_gui_offers_both_options(self):
        from tools import v7_send_gui as gui
        names = dict((value, label) for label, value in gui.PROFILE_CHOICES)
        for profile in ('aspect-mono-nested', 'stereo-nested'):
            self.assertIn(profile, names)
            self.assertIn('experimental', names[profile])
            self.assertIn(profile, gui.FOLDED_PROFILES)
            self.assertIn(profile, gui.ASPECT_PROFILES)
            # Each fold's default downscale kernel is the benchmark winner
            # through its own chain; the sender CLI agrees with the GUI.
            expected = {'aspect-mono-nested': 'viewer_solve',
                        'stereo-nested': 'upscale_precomp'}[profile]
            self.assertEqual(gui.default_kernel_for_profile(profile), expected)
            self.assertEqual(v7_live._default_kernel_for_profile(profile), expected)
            # Kernel values shown, saved and sent are the base wire's.
            for kernel in ('viewer_solve', 'csf_peak'):
                self.assertEqual(
                    gui._kernel_values({}, gui.kernel_registry().get(kernel), profile),
                    gui.kernel_registry().get(kernel).defaults(gui.NESTED_BASE_PROFILES[profile]))
        self.assertIn('aspect-mono-nested', gui.MONO_PROFILES)
        self.assertNotIn('stereo-nested', gui.MONO_PROFILES)


if __name__ == '__main__':
    unittest.main()
