"""Pixel grids of the aspect-fold-500 profile (Pixel encode / Pixel display)."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT/'test_modem_v7', ROOT/'tools'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from animation_modem import v7                                           # noqa: E402
from animation_modem.v7_dct_display import reconstruct_plane             # noqa: E402
from common import TARGET                                                # noqa: E402
import aspect_fold                                                       # noqa: E402
from aspect_fold import (LAYOUT_NAMES, PIXEL_GRID_NAMES, PIXEL_GRIDS,     # noqa: E402
                         SIGNATURE_SLOTS, AspectFoldWire, layout_tables,
                         pixel_positions, pixel_shapes)
import tone_code                                                         # noqa: E402
from tools import v7_live                                                # noqa: E402
from tools.v7_gl_viewer import PIXEL_REPEAT, dct_reconstruct_planes      # noqa: E402

HOSTS = 1920-SIGNATURE_SLOTS


def _art(rows, cols, seed=2):
    """Hard pixel art with one-pixel diagonals: the worst case for a grid."""
    rng = np.random.default_rng(seed)
    palette = rng.integers(0, 256, (8, 3))
    index = rng.integers(0, 8, (rows//4+1, cols//4+1))
    image = palette[np.repeat(np.repeat(index, 4, 0), 4, 1)[:rows, :cols]]
    image = image.astype(np.uint8)
    y, x = np.mgrid[:rows, :cols]
    image[(y+x) % 7 == 0] = 255-image[(y+x) % 7 == 0]
    return image


def _luma(rgb):
    red, green, blue = (rgb[..., i].astype(float)/255 for i in range(3))
    return .299*red+.587*green+.114*blue


def _stream(base, grid, layout, code, scale=10, packets=6, start=1):
    """(audio, sent luma picture) of ``packets`` pixel-encoded packets."""
    wire = AspectFoldWire('auto', 'fixed', pixel=grid)
    rows, cols = PIXEL_GRIDS[grid][layout]
    small = _art(rows, cols)
    frame = np.repeat(np.repeat(small, scale, 0), scale, 1)
    shapes = tuple(wire.pixel_shapes(wire.layout_for(c)) for c in range(8))
    values, aspect = v7_live._values(
        base, frame, 'box', brightness=1.0, dct_encode=True,
        dct_options={'pixel': True, 'pixel_shapes': shapes})[:2]
    assert aspect == code, (aspect, code)
    model, coeffs = wire.encode_coefficients(base, values, code)
    audio = np.concatenate([tone_code.add_tone_code(
        v7.encode_pulse_frame_coeffs(
            model, coeffs, counter, aspect_code=code, source_index=counter-1,
            pilot_tones=False,
            pulse_profile_code=wire.pulse_profile_code),
        counter, tone_code.encode_status(wire.status_mode))
        for counter in range(start, start+packets)]).astype(np.float32)
    return audio, _luma(small)


def _decode(profile, base, audio):
    profile.install()
    try:
        with tone_code.coded_pilot_timing():
            profile.active_mode = profile.dispatch_mode = profile.aspect_mode
            results = v7.decode_pulse_stream(
                base, audio, state=v7.PulseState(), sample_rate=v7.RATE,
                pilot_timing='tone-seeded')[0]
    finally:
        profile.uninstall()
    return [result for result in results if result.status != 'lost']


class PixelGridTests(unittest.TestCase):
    def test_grids_fit_their_slots_and_carry_every_pixel_coefficient(self):
        rows, cols = v7.V7_GRIDS[0]
        for grid in PIXEL_GRID_NAMES:
            for layout in LAYOUT_NAMES:
                with self.subTest(grid=grid, layout=layout):
                    rect_rows, rect_cols = PIXEL_GRIDS[grid][layout]
                    self.assertEqual((rect_rows % 2, rect_cols % 2), (0, 0))
                    self.assertEqual(pixel_shapes(layout, grid)[1],
                                     (rect_rows//2, rect_cols//2))
                    kept, guests = pixel_positions(layout, grid)
                    self.assertEqual(len(kept), 2880)
                    area = rect_rows*rect_cols
                    if grid == 'robust':
                        self.assertLessEqual(area, HOSTS)
                        self.assertEqual(len(guests), SIGNATURE_SLOTS)
                    else:
                        self.assertEqual(len(guests),
                                         area-HOSTS+SIGNATURE_SLOTS)
                    carried = np.concatenate(
                        (kept[:1920], guests[:-SIGNATURE_SLOTS]))
                    self.assertEqual(len(set(carried.tolist())), len(carried))
                    u, v = np.divmod(carried, cols)
                    inside = (u < rect_rows) & (v < rect_cols)
                    self.assertEqual(int(inside.sum()), area)
                    # The signature's hosts and guests are outside filler.
                    for index in (*kept[HOSTS:1920], *guests[-SIGNATURE_SLOTS:]):
                        self.assertFalse(index//cols < rect_rows and
                                         index % cols < rect_cols)
        # Portrait layouts mirror the landscape ones; large is exact aspect.
        for grid in PIXEL_GRID_NAMES:
            self.assertEqual(PIXEL_GRIDS[grid]['9:16'],
                             PIXEL_GRIDS[grid]['16:9'][::-1])
        self.assertEqual(PIXEL_GRIDS['large']['16:9'], (36, 64))
        self.assertEqual(PIXEL_GRIDS['large']['3:2'], (40, 60))

    def test_tables_are_pinned_and_only_two_tails_exist(self):
        self.assertEqual(aspect_fold.PIXEL_TABLES_SHA256,
                         'f5f1f8b76d52e653d477ba1fe2092aa3560ab6354f7a5d116a100e9a7b436f4f')
        tables = layout_tables('16:9', 'fixed', 'robust')
        self.assertEqual(len(tables['positions']), v7.BODY_END+v7.TAIL_PER)
        plane = aspect_fold._plane_of(tables['positions'])
        self.assertEqual(int((plane == 0).sum()), 1920)
        with self.assertRaises(ValueError):
            layout_tables('16:9', 'luma', 'robust')
        with self.assertRaises(ValueError):
            AspectFoldWire('auto', 'split', pixel='large')
        with self.assertRaises(ValueError):
            AspectFoldWire('auto', 'fixed', pixel='huge')

    def test_models_send_the_nearest_bit_and_grids_have_opposed_signatures(self):
        base = v7.load_model(TARGET, 'box')
        codecs = {}
        for grid in PIXEL_GRID_NAMES:
            wire = AspectFoldWire('auto', 'fixed', pixel=grid)
            model = wire.model_for(base, '3:2')
            self.assertEqual(int(model.encoding_type),
                             v7.ENCODING_FILTER_CODES['nearest'])
            codecs[grid] = wire.codec(model)
        self.assertEqual(float(np.dot(codecs['robust'].pattern,
                                      codecs['large'].pattern)), 0.0)
        self.assertNotEqual(codecs['robust'].identity,
                            codecs['large'].identity)
        # A nearest base model (what a live receiver loads for that bit)
        # gives the same wire.
        nearest = v7.load_model(TARGET, 'nearest')
        other = AspectFoldWire('auto', 'fixed', pixel='robust')
        np.testing.assert_allclose(
            other.model_for(nearest, '3:2').scale,
            AspectFoldWire('auto', 'fixed', pixel='robust').model_for(
                base, '3:2').scale, rtol=1e-12)

    def test_live_receiver_follows_either_grid_pixel_for_pixel(self):
        base = v7.load_model(TARGET, 'box')
        profile = v7_live._AdaptiveProfileDecoder(
            v7_live._experimental_fold(500), base)
        # One receiver, three streams in a row: it must follow each change.
        for grid, layout, code, limit in (('large', '16:9', 3, 16),
                                          ('robust', '16:9', 3, 4),
                                          ('large', '3:2', 2, 16),
                                          ('robust', '4:3', 1, 4)):
            with self.subTest(grid=grid, layout=layout):
                audio, sent = _stream(base, grid, layout, code)
                good = _decode(profile, base, audio)
                self.assertGreaterEqual(len(good), 4)
                # Followed from the first packet of each stream.
                self.assertEqual({result.diag['aspect_pixel']
                                  for result in good}, {grid})
                last = good[-1]
                shown = profile.values(base, last)
                shapes = last.diag['pixel_shapes']
                self.assertEqual(tuple(shapes[0]), PIXEL_GRIDS[grid][layout])
                rows, cols = v7.V7_GRIDS[0]
                picture = (reconstruct_plane(
                    shown[:rows*cols].reshape(rows, cols).astype(np.float32),
                    shapes[0])+1)/2
                error = np.abs(picture-sent)*255
                self.assertGreater(float(np.mean(error <= limit)), .99)
                # Nothing is shown outside the rectangle.
                from scipy.fft import dctn
                spectrum = dctn(shown[:rows*cols].reshape(rows, cols),
                                norm='ortho')
                spectrum[:shapes[0][0], :shapes[0][1]] = 0
                self.assertLess(float(np.abs(spectrum).max()), 1e-9)

    def test_ordinary_packets_are_not_taken_for_a_pixel_grid(self):
        base = v7.load_model(TARGET, 'box')
        frame = np.repeat(np.repeat(_art(36, 64), 10, 0), 10, 1)
        values = v7_live._values(base, frame, 'box', brightness=1.0)[0]
        wire = AspectFoldWire('auto', 'fixed')
        model, coeffs = wire.encode_coefficients(base, values, 3)
        audio = np.concatenate([tone_code.add_tone_code(
            v7.encode_pulse_frame_coeffs(
                model, coeffs, counter, aspect_code=3, source_index=counter-1,
                pilot_tones=False,
                pulse_profile_code=wire.pulse_profile_code),
            counter, tone_code.encode_status(wire.status_mode))
            for counter in range(1, 6)]).astype(np.float32)
        profile = v7_live._AdaptiveProfileDecoder(
            v7_live._experimental_fold(500), base)
        good = _decode(profile, base, audio)
        self.assertGreaterEqual(len(good), 3)
        self.assertIsNone(good[-1].diag['aspect_pixel'])
        profile.values(base, good[-1])
        self.assertNotIn('pixel_shapes', good[-1].diag)

    def test_pixel_display_shows_the_grid_the_packet_names(self):
        planes = tuple(np.zeros(shape, np.float32) for shape in v7.V7_GRIDS)
        shapes = pixel_shapes('16:9', 'robust')
        shown = dct_reconstruct_planes(planes, 'pixel', pixel_shapes=shapes)
        self.assertEqual(tuple(plane.shape for plane in shown),
                         tuple((rows*PIXEL_REPEAT, cols*PIXEL_REPEAT)
                               for rows, cols in shapes))
        # Without a named grid: half the planes, as before.
        shown = dct_reconstruct_planes(planes, 'pixel')
        self.assertEqual(shown[0].shape, (48*PIXEL_REPEAT, 40*PIXEL_REPEAT))

    def test_sender_cli_takes_the_grid(self):
        args = v7_live.parser().parse_args(['send', '--device', '0', '--pixel-grid', 'large'])
        self.assertEqual(args.pixel_grid, 'large')
        self.assertEqual(v7_live.parser().parse_args(['send', '--device', '0']).pixel_grid,
                         v7_live.DEFAULT_PIXEL_GRID)


if __name__ == '__main__':
    unittest.main()
