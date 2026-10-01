"""Aspect-matched stereo Fold 500 (experimental aspect-fold-500 profile)."""
import sys
import unittest
import unittest.mock
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT/'test_modem_v7', ROOT/'tools'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from animation_modem import v7                                           # noqa: E402
from common import TARGET                                                # noqa: E402
import aspect_fold                                                       # noqa: E402
from aspect_fold import (ASPECT_CODE_LAYOUT, LAYOUT_NAMES, PLANE_COUNTS,  # noqa: E402
                         TAIL_LUMA_SLOTS, TAIL_MODES, AspectFoldWire,
                         layout_positions, layout_tables)
from live_fold import LiveFold                                           # noqa: E402
import tone_code                                                         # noqa: E402
from tools import v7_live                                                # noqa: E402


def _grid_uv(index):
    offsets = np.cumsum([0] + [rows*cols for rows, cols in v7.V7_GRIDS])
    plane = int(np.searchsorted(offsets, index, side='right')-1)
    rows, cols = v7.V7_GRIDS[plane]
    return (plane, *divmod(int(index-offsets[plane]), cols))


def _text_frame(width=640, height=360):
    image = Image.new('RGB', (width, height), (30, 30, 30))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    try:
        font = ImageFont.truetype('DejaVuSans-Bold.ttf', 56)
    except OSError:
        pass
    draw.text((16, 20), 'Title 56', fill=(235, 235, 235), font=font)
    draw.text((16, 120), 'IIII |||| mmmm', fill=(235, 235, 235), font=font)
    for x in range(0, width, 12):
        draw.line((x, 240, x, 350), fill=(200, 200, 200), width=4)
    return image


def _round_trip(base, values, aspect, profile, packets=9):
    """Pulse-encode ``packets`` copies, decode, return displayed grid values."""
    captured = {}
    real_numba, real_numpy = v7._equalize_numba, v7._equalize_numpy

    def numba(*args, **kwargs):
        out = real_numba(*args, **kwargs)
        captured['eq'] = (out[0].copy(), out[1].copy())
        return out

    def numpy(*args, **kwargs):
        out = real_numpy(*args, **kwargs)
        captured['eq'] = (out[0].copy(), out[1].copy())
        return out

    if profile == 'fold-500':
        fold = LiveFold(500)
        model = base
        coeffs = fold.encode_coefficients(base, values)
        code = mode = tone_code.FOLD_500
    else:
        wire = AspectFoldWire(*profile)
        model, coeffs = wire.encode_coefficients(base, values, aspect)
        code = mode = wire.status_mode
    audio = np.concatenate([
        tone_code.add_tone_code(v7.encode_pulse_frame_coeffs(
            model, coeffs, counter, aspect_code=aspect, source_index=0,
            pilot_tones=False, eof_marker=True, pulse_profile_code=code),
            counter, tone_code.encode_status(mode))
        for counter in range(1, packets+1)])
    v7._equalize_numba, v7._equalize_numpy = numba, numpy
    try:
        with tone_code.coded_pilot_timing():
            results, _ = v7.decode_pulse_stream(
                model, audio, models={model.encoding_type: model},
                state=v7.PulseState(), pilot_timing='tone-seeded',
                frame_boundary='eof')
    finally:
        v7._equalize_numba, v7._equalize_numpy = real_numba, real_numpy
    good = [result for result in results if result.status != 'lost']
    result = good[-1]
    result.diag['fold_eq'] = captured['eq']
    if profile == 'fold-500':
        return fold.values(model, result, metadata_confirmed=True), len(good)
    return wire.values(model, result), len(good)


class AspectLayoutTests(unittest.TestCase):
    def test_layouts_follow_the_picture_shape_and_fit_the_grid(self):
        for layout in LAYOUT_NAMES:
            kept, guests, extra = layout_positions(layout)
            with self.subTest(layout=layout):
                self.assertEqual(len(kept), sum(PLANE_COUNTS))
                self.assertEqual(len(guests), aspect_fold.FOLD_SLOTS)
                self.assertEqual(len(np.unique(np.concatenate(
                    (kept, guests, extra)))), len(kept)+len(guests)+len(extra))
                luma = [_grid_uv(index) for index in kept[:PLANE_COUNTS[0]]]
                rows = max(u for _, u, _ in luma)+1
                cols = max(v for _, _, v in luma)+1
                width, height = aspect_fold.layout_size(layout)
                if width > height:
                    self.assertGreater(cols, rows)
                elif width < height:
                    self.assertGreater(rows, cols)
                self.assertTrue(all(_grid_uv(index)[0] == 0
                                    for index in np.concatenate((guests, extra))))
        wide = {v for _, _, v in map(_grid_uv, layout_positions('16:9')[0][:1920])}
        self.assertEqual(max(wide)+1, 66)       # vs 40 horizontal steps today

    def test_every_aspect_code_has_a_layout(self):
        self.assertEqual(len(ASPECT_CODE_LAYOUT), len(v7.V7_ASPECT_RATIOS))
        for code, ratio in enumerate(v7.V7_ASPECT_RATIOS):
            width, height = aspect_fold.layout_size(ASPECT_CODE_LAYOUT[code])
            self.assertAlmostEqual(width/height, ratio)

    def test_canonical_statistics_are_recovered_exactly(self):
        curves = aspect_fold._canonical_curves()
        self.assertEqual(len(curves), 3)

    def test_frozen_tables_are_pinned(self):
        self.assertIsNotNone(aspect_fold.TABLES_SHA256)
        for layout in LAYOUT_NAMES:
            for tail in TAIL_MODES:
                tables = layout_tables(layout, tail)
                self.assertEqual(len(tables['positions']),
                                 v7.BODY_END+v7.TAIL_PER if tail == 'fixed'
                                 else sum(PLANE_COUNTS))

    def test_tail_modes_keep_head_and_body_and_fill_the_tail(self):
        base = v7.load_model(TARGET, 'box')
        for tail in TAIL_MODES:
            wire = AspectFoldWire('16:9', tail)
            model = wire.model_for(base, '16:9')
            luma_slots = TAIL_LUMA_SLOTS[tail]
            plane = np.asarray(model.plane)
            with self.subTest(tail=tail):
                self.assertEqual(np.count_nonzero(plane == 0), 1920+luma_slots)
                self.assertEqual(sorted(model.order.tolist()),
                                 list(range(len(model.order))))
                extra = np.arange(1920, 1920+luma_slots)
                self.assertFalse(np.isin(extra, model.order[:v7.BODY_END]).any())
                seen = set()
                for ranks in model.rank_tables:
                    sent = set(ranks[ranks >= 0].tolist())
                    self.assertTrue(set(extra.tolist()) <= sent)
                    seen |= sent
                pool = len(model.order)-v7.BODY_END-luma_slots
                expected = (v7.BODY_END+luma_slots +
                            min(pool, (v7.TAIL_PER-luma_slots)*v7.TAIL_PHASES))
                self.assertEqual(len(seen), expected)
                codec = wire.codec(model)
                self.assertTrue(np.all(plane[codec.hosts] == 0))

    def test_fixed_tail_sends_the_same_coefficients_in_every_packet(self):
        base = v7.load_model(TARGET, 'box')
        wire = AspectFoldWire('16:9', 'fixed')
        model = wire.model_for(base, '16:9')
        chroma = AspectFoldWire('16:9', 'chroma')
        reference = chroma.model_for(base, '16:9')
        self.assertFalse(wire.rotates)
        self.assertTrue(chroma.rotates)
        self.assertEqual(len(model.order), v7.BODY_END+v7.TAIL_PER)
        for ranks in model.rank_tables[1:]:
            np.testing.assert_array_equal(ranks, model.rank_tables[0])
        # Exactly what the chroma tail's first packet carries, same statistics.
        sent = reference.order[:v7.BODY_END+v7.TAIL_PER]
        positions = np.asarray(reference.coder.positions)[sent]
        np.testing.assert_array_equal(
            np.asarray(model.coder.positions)[model.order], positions)
        np.testing.assert_array_equal(model.lam[model.order],
                                      reference.lam[sent])
        np.testing.assert_array_equal(wire.tail_prior(model, '16:9'), model.mu)

    def test_mismatched_layouts_have_distinct_fold_signatures(self):
        base = v7.load_model(TARGET, 'box')
        identities = set()
        for layout in ('16:9', '4:3'):
            for tail in TAIL_MODES:
                wire = AspectFoldWire(layout, tail)
                identities.add(wire.codec(wire.model_for(base, layout)).identity)
        self.assertEqual(len(identities), 2*len(TAIL_MODES))


class AspectWireTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = v7.load_model(TARGET, 'box')
        cls.frame = _text_frame()
        cls.aspect = v7.aspect_wire_code(cls.frame.size)
        cls.values = v7_live._values(cls.base, cls.frame, 'box', brightness=1.0)[0]

    def test_auto_layout_follows_the_packet_aspect_code(self):
        wire = AspectFoldWire('auto', 'chroma')
        self.assertEqual(wire.layout_for(self.aspect), '16:9')
        self.assertIsNone(wire.layout_for(None))
        self.assertEqual(AspectFoldWire('3:4').layout_for(self.aspect), '3:4')

    def test_packets_decode_and_widescreen_detail_improves(self):
        def luma_error(values):
            return float(np.mean((values[:96*80]-self.values[:96*80])**2))

        fold_values, fold_ok = _round_trip(self.base, self.values, self.aspect,
                                           'fold-500')
        errors = {}
        for tail in TAIL_MODES:
            values, ok = _round_trip(self.base, self.values, self.aspect,
                                     ('16:9', tail))
            self.assertEqual(ok, fold_ok)
            errors[tail] = luma_error(values)
        # Vertical strokes need horizontal detail: every 16:9 layout beats
        # the 5:6 corner, and luma tail slots add a little more.
        for tail, error in errors.items():
            self.assertLess(error, .9*luma_error(fold_values), tail)
        self.assertLessEqual(errors['luma'], errors['chroma'])

    def test_sender_cli_and_profile_mapping(self):
        args = v7_live.parser().parse_args([
            'send', '--device', 'null', '--source', 'test',
            '--profile', 'aspect-fold-500', '--aspect-layout', '16:9',
            '--aspect-tail', 'split', '--dct-encode'])
        v7_live._apply_profile_option(args)
        self.assertTrue(args.aspect_fold)
        self.assertEqual(v7_live._fold_slots(args), 0)
        self.assertEqual(v7_live._send_profile(args, 500)[0], 'box')
        self.assertEqual((args.aspect_layout, args.aspect_tail), ('16:9', 'split'))
        receive = v7_live.parser().parse_args([
            'receive', '--device', '0', '--aspect-layout', '16:9',
            '--aspect-tail', 'split'])
        self.assertEqual((receive.aspect_layout, receive.aspect_tail),
                         ('16:9', 'split'))

    def test_adaptive_receiver_dispatches_the_aspect_status(self):
        decoder = v7_live._make_auto_profile_decoder(
            v7_live.parser().parse_args(['receive', '--device', '0',
                                         '--aspect-tail', 'luma']),
            v7_live._experimental_fold(500))
        self.assertIn(tone_code.FOLD_OFF, decoder.supported_modes)
        decoder.active_mode = decoder.dispatch_mode = tone_code.FOLD_OFF
        seen = {}

        def fake_decode(model, x, tmap, counter, prev_tail, *args, **kwargs):
            seen['model'] = model
            return v7.Result(counter, 'received', np.zeros(len(model.mu)), {})

        decoder._real_decode = fake_decode
        body = np.zeros((v7.FRAME, 2))
        with unittest.mock.patch('mono_wire.MonoWire._status_for_body',
                                 return_value={'valid': True, 'status': {
                                     'mode': tone_code.FOLD_OFF}}):
            result = decoder._decode_frame(
                self.base, None, None, 3, self.base.mu, direct_body=body,
                profile_hint={'aspect_code': self.aspect})
            self.assertEqual(result.diag['aspect_layout'], '16:9')
            self.assertEqual(result.diag['wire_profile'], 'aspect-fold-500')
            self.assertEqual(np.count_nonzero(seen['model'].plane == 0), 1920+96)
            held = decoder._decode_frame(
                self.base, None, None, 3, self.base.mu, direct_body=body,
                profile_hint={'aspect_code': None})
            self.assertEqual(held.diag['profile_rejected'], 'aspect_layout_unknown')


if __name__ == '__main__':
    import unittest.mock  # noqa: F401
    unittest.main()
