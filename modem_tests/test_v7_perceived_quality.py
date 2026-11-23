"""Clip-aware encode, the direct-encode small-frame fallback and display grain."""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.fft import dctn, idctn

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT/'test_modem_v7', ROOT/'tools'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from animation_modem import v7                                           # noqa: E402
from animation_modem.v7_source_dct import (CLIP_AWARE_ITERATIONS,        # noqa: E402
                                           _shown_luminance, clip_aware_luma,
                                           direct_dct_values, luma_adjust,
                                           received_chroma)
from common import TARGET                                                # noqa: E402
from aspect_fold import AspectFoldWire                                   # noqa: E402
from aspect_mono import AspectMonoWire                                   # noqa: E402
from live_fold import LiveFold                                           # noqa: E402
from mono_video import FRESH_SLOTS, MonoColourFoldWire                   # noqa: E402
from tools import v7_live                                                # noqa: E402
from tools.v7_gl_viewer import (FLOAT_FRAGMENT_SHADER, GRAIN_MODES,      # noqa: E402
                                flat_area_mask)

ROWS, COLS = v7.V7_GRIDS[0]


def _bright_text_values():
    """Grid values: white bars on black, as the direct encode clips them."""
    luma = np.full((ROWS, COLS), -1.0)
    for x in range(10, 70, 9):
        luma[30:60, x:x+3] = 1.0
    chroma = np.zeros(sum(r*c for r, c in v7.V7_GRIDS[1:]))
    return np.concatenate((luma.ravel(), chroma))


def _visible_error(values, sent):
    """RMS error the receiver shows: sent coefficients only, then the clip."""
    luma = values[:ROWS*COLS].reshape(ROWS, COLS)
    shown = idctn(np.where(sent, dctn(luma, norm='ortho'), 0.0), norm='ortho')
    target = _bright_text_values()[:ROWS*COLS].reshape(ROWS, COLS)
    return float(np.sqrt(np.mean((np.clip(shown, -1, 1)-target)**2)))


class ClipAwareEncodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = v7.load_model(TARGET, 'box')
        wire = AspectFoldWire('16:9')
        cls.codec = wire.codec(wire.model_for(cls.base, '16:9'))
        cls.sent = cls.codec.sent_luma_mask().reshape(ROWS, COLS)

    def test_only_sent_coefficients_change_and_visible_error_drops(self):
        values = _bright_text_values()
        fitted = clip_aware_luma(values, (ROWS, COLS), self.sent)
        before = dctn(values[:ROWS*COLS].reshape(ROWS, COLS), norm='ortho')
        after = dctn(fitted[:ROWS*COLS].reshape(ROWS, COLS), norm='ortho')
        np.testing.assert_allclose(after[~self.sent], before[~self.sent],
                                   atol=1e-12)
        np.testing.assert_array_equal(fitted[ROWS*COLS:], values[ROWS*COLS:])
        self.assertLess(_visible_error(fitted, self.sent),
                        .9*_visible_error(values, self.sent))
        np.testing.assert_array_equal(
            clip_aware_luma(values, (ROWS, COLS), self.sent, iterations=0),
            values)
        self.assertEqual(CLIP_AWARE_ITERATIONS, 10)

    def test_mid_grey_pictures_are_left_alone(self):
        count = sum(rows*cols for rows, cols in v7.V7_GRIDS)
        values = np.random.default_rng(1).uniform(-.3, .3, count)
        luma = values[:ROWS*COLS].reshape(ROWS, COLS)
        sent_only = idctn(np.where(self.sent, dctn(luma, norm='ortho'), 0),
                          norm='ortho')
        # Nothing reaches black or white, so the fit is the least-squares
        # band-limited picture: the sent coefficients are unchanged.
        self.assertLess(float(np.abs(sent_only).max()), .99)
        fitted = clip_aware_luma(values, (ROWS, COLS), self.sent)
        np.testing.assert_allclose(fitted, values, atol=1e-9)

    def test_sent_masks_follow_each_profile(self):
        stereo = LiveFold(500).codec(self.base)
        self.assertEqual(int(stereo.sent_luma_mask().sum()), 48*40+500)
        self.assertEqual(int(self.codec.sent_luma_mask().sum()), 1920+500)
        for wire in (MonoColourFoldWire(self.base), AspectMonoWire(self.base)):
            model = wire._packet_model(self.base, 3)
            codec = wire._codec(model)
            plane = np.asarray(codec.model.plane)
            fresh_luma = int(np.count_nonzero(
                plane[codec.sent_model_indices] == 0))
            with self.subTest(wire=wire.wire_profile):
                self.assertEqual(len(codec.sent_model_indices), FRESH_SLOTS)
                self.assertEqual(int(codec.sent_luma_mask().sum()),
                                 fresh_luma+500)

    def test_cli_flag_is_opt_in(self):
        args = v7_live.parser().parse_args(
            ['send', '--device', 'null', '--source', 'test'])
        self.assertFalse(args.clip_aware_encode)
        args = v7_live.parser().parse_args(
            ['send', '--device', 'null', '--source', 'test',
             '--clip-aware-encode'])
        self.assertTrue(args.clip_aware_encode)


def _colour_edge_frame():
    """Saturated red and blue bars on green: the brightness of each colour
    edge lives partly in chroma."""
    frame = np.zeros((480, 400, 3), np.uint8)
    frame[:] = (40, 170, 60)
    for x in range(20, 380, 48):
        frame[60:420, x:x+14] = (230, 30, 40)
        frame[60:420, x+24:x+30] = (40, 60, 230)
    return frame


class LumaAdjustTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = v7.load_model(TARGET, 'box')
        wire = AspectFoldWire('1:1')
        cls.codec = wire.codec(wire.model_for(cls.base, '1:1'))
        cls.masks = v7_live._chroma_sent_masks(cls.codec)
        cls.grids = v7.V7_GRIDS

    def _encode(self, frame):
        target = []
        values = direct_dct_values(frame, self.grids, v7.V7_SHAPES,
                                   luminance_out=target)
        return values, target[0]

    def _luminance_error(self, values, target):
        rows, cols = self.grids[0]
        cb, cr = received_chroma(values, self.grids, self.masks)
        luma = values[:rows*cols].reshape(rows, cols)
        return float(np.sqrt(np.mean((_shown_luminance(luma, cb, cr)-target)**2)))

    def test_chroma_masks_are_the_layouts_sent_colour(self):
        # The fixed tail: head, body and tail colour of one packet.
        plane = np.asarray(self.codec.model.plane)
        self.assertEqual([int(mask.sum()) for mask in self.masks],
                         [int((plane == 1).sum()), int((plane == 2).sum())])
        self.assertLess(sum(int(mask.sum()) for mask in self.masks), 960)

    def test_coloured_edges_keep_the_source_luminance(self):
        values, target = self._encode(_colour_edge_frame())
        self.assertEqual(target.shape, self.grids[0])
        adjusted = luma_adjust(values, self.grids, self.masks, target)
        count = int(np.prod(self.grids[0]))
        np.testing.assert_array_equal(adjusted[count:], values[count:])
        before = self._luminance_error(values, target)
        after = self._luminance_error(adjusted, target)
        self.assertLess(after, .25*before)
        self.assertTrue(np.all(np.abs(adjusted) <= 1.0))

    def test_smooth_grey_pictures_are_left_alone(self):
        ramp = np.linspace(20, 235, 400)[None, :, None]*np.ones((480, 1, 3))
        values, target = self._encode(np.uint8(np.rint(ramp)))
        adjusted = luma_adjust(values, self.grids, self.masks, target)
        self.assertLess(float(np.max(np.abs(adjusted-values))), .01)

    def test_sender_value_path_applies_it_only_when_asked(self):
        frame = _colour_edge_frame()
        plain = v7_live._values(self.base, frame, 'box', 1.0, dct_encode=True)[0]
        adjusted = v7_live._values(self.base, frame, 'box', 1.0, dct_encode=True,
                                   chroma_sent_for=lambda code: self.masks)[0]
        count = int(np.prod(self.grids[0]))
        self.assertGreater(float(np.max(np.abs(adjusted[:count]-plain[:count]))), .02)
        np.testing.assert_array_equal(adjusted[count:], plain[count:])

    def test_cli_flag_requires_direct_dct(self):
        args = v7_live.parser().parse_args(
            ['send', '--device', 'null', '--source', 'test', '--luma-adjust'])
        self.assertTrue(args.luma_adjust)
        with self.assertRaises(ValueError):
            v7_live.run_send(args)


class DirectEncodeFallbackTests(unittest.TestCase):
    def test_frames_smaller_than_the_grid_use_the_box_resize(self):
        model = v7_live._model(v7_live.DEFAULT_FIXTURE, 'box')
        small = np.full((15, 20, 3), 200, np.uint8)
        values, aspect = v7_live._values(model, small, 'box', brightness=1.0,
                                         dct_encode=True)
        expected, _ = v7_live._values(model, small, 'box', brightness=1.0)
        np.testing.assert_array_equal(values, expected)
        self.assertEqual(aspect, v7.aspect_wire_code((20, 15)))


class DisplayGrainTests(unittest.TestCase):
    def test_flat_mask_is_one_on_flat_and_zero_on_detail(self):
        luma = np.full((ROWS, COLS), -.2, np.float32)
        luma[:, 40:] += np.where(np.arange(40) % 2, .5, 0)[None, :]
        mask = flat_area_mask(luma)
        self.assertEqual(mask.shape, luma.shape)
        self.assertEqual(mask.dtype, np.float32)
        self.assertTrue(mask.flags.c_contiguous)
        self.assertGreater(float(mask[:, :30].min()), .99)
        self.assertLess(float(mask[:, 45:].max()), .01)

    def test_shader_grain_is_off_unless_enabled(self):
        self.assertEqual(GRAIN_MODES[0], 'off')
        for declaration in ('uniform sampler2D grain_mask;',
                            'uniform float grain_amount;',
                            'uniform int grain_seed;'):
            self.assertIn(declaration, FLOAT_FRAGMENT_SHADER)
        self.assertIn('if (grain_amount > 0.0)', FLOAT_FRAGMENT_SHADER)


def _offscreen_context():
    """A software or hardware GL context, or None where there is none."""
    try:
        import moderngl
    except ImportError:
        return None
    for options in ({'backend': 'egl'}, {}):
        try:
            return moderngl.create_standalone_context(**options)
        except Exception:
            continue
    return None


class DetailGrainAndDitherTests(unittest.TestCase):
    """Grain sized to the picture, and dither into the 8-bit framebuffer."""

    @classmethod
    def setUpClass(cls):
        cls.context = _offscreen_context()

    @classmethod
    def tearDownClass(cls):
        if cls.context is not None:
            cls.context.release()

    def _render(self, luma, size, **uniforms):
        import moderngl
        from tools.v7_gl_viewer import VERTEX_SHADER
        context = self.context
        program = context.program(vertex_shader=VERTEX_SHADER,
                                  fragment_shader=FLOAT_FRAGMENT_SHADER)
        array = context.vertex_array(program, [])
        neutral = np.full((1, 1), 1/255, np.float32)
        mask = np.ones((1, 1), np.float32)
        textures = []
        for unit, plane in enumerate((luma, neutral, neutral, mask, mask)):
            plane = np.ascontiguousarray(plane, np.float32)
            texture = context.texture(plane.shape[::-1], 1, plane.tobytes(),
                                      dtype='f4')
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            texture.repeat_x = texture.repeat_y = False
            texture.use(unit)
            textures.append(texture)
        values = {'plane_y': 0, 'plane_cb': 1, 'plane_cr': 2, 'kernel_lut': 3,
                  'grain_mask': 4, 'reconstruction': 0,
                  'filtered_intermediate': 0, 'grain_seed': 5,
                  'output_size': (float(size[0]), float(size[1]))}
        values.update(uniforms)
        for name, value in values.items():
            if name in program:
                program[name].value = value
        target = context.framebuffer([context.texture(size, 4)])
        target.use()
        context.viewport = (0, 0, *size)
        array.render(mode=moderngl.TRIANGLES, vertices=3)
        pixels = np.frombuffer(target.read(components=3), np.uint8).reshape(
            size[1], size[0], 3)[::-1].astype(float)   # top row first
        for item in textures+[target, array, program]:
            item.release()
        return pixels[..., 1]

    def test_modes_and_shader_declarations(self):
        from tools.v7_gl_viewer import GRAIN_LABELS
        self.assertEqual(GRAIN_MODES, ('off', 'flat', 'detail'))
        self.assertEqual(set(GRAIN_LABELS), set(GRAIN_MODES))
        for declaration in ('uniform int grain_kind;',
                            'uniform vec2 grain_cells;',
                            'uniform float dither_amount;'):
            self.assertIn(declaration, FLOAT_FRAGMENT_SHADER)
        self.assertIn('if (dither_amount > 0.0)', FLOAT_FRAGMENT_SHADER)

    def test_detail_mask_follows_picture_detail(self):
        from tools.v7_gl_viewer import (GRAIN_DETAIL_FLOOR,
                                        grain_detail_mask)
        luma = np.zeros((ROWS, COLS), np.float32)
        luma[:, 40:] = np.random.default_rng(3).uniform(-.5, .5, (ROWS, 40))
        mask = grain_detail_mask(luma)
        self.assertEqual((mask.shape, mask.dtype), ((ROWS, COLS), np.float32))
        np.testing.assert_allclose(mask[:, :30], GRAIN_DETAIL_FLOOR, atol=1e-3)
        self.assertGreater(float(mask[:, 50:].mean()), .9)
        self.assertLessEqual(float(mask.max()), 1.0)

    def test_dither_and_grain_default_to_off_in_the_shader(self):
        if self.context is None:
            self.skipTest('no offscreen GL context')
        ramp = np.linspace(-.9, -.7, ROWS, dtype=np.float32)[:, None]*np.ones(
            (1, COLS), np.float32)
        plain = self._render(ramp, (160, 384))
        again = self._render(ramp, (160, 384), grain_seed=9)
        np.testing.assert_array_equal(plain, again)

    def test_dither_removes_bands_and_keeps_the_local_mean(self):
        if self.context is None:
            self.skipTest('no offscreen GL context')
        from scipy.ndimage import uniform_filter
        from tools.v7_gl_viewer import DITHER_AMOUNT
        size = (320, 1536)
        ramp = np.linspace(-.9, -.7, ROWS, dtype=np.float32)[:, None]*np.ones(
            (1, COLS), np.float32)
        truth = np.interp((np.arange(size[1])+.5)/size[1]*ROWS-.5,
                          np.arange(ROWS), (ramp[:, 0]+1)*127.5)[:, None]
        truth = np.clip(truth, (ramp.min()+1)*127.5, (ramp.max()+1)*127.5)

        def measure(pixels):
            column = pixels[:, 7]
            runs = np.diff(np.flatnonzero(np.diff(column) != 0))
            error = uniform_filter(pixels-truth, 16)[32:-32]
            return int(runs.max()), float(np.sqrt(np.mean(error**2)))

        plain_run, plain_error = measure(self._render(ramp, size))
        dithered = self._render(ramp, size, dither_amount=DITHER_AMOUNT)
        dither_run, dither_error = measure(dithered)
        self.assertGreater(plain_run, 40)             # bands tens of pixels tall
        self.assertLess(dither_run, plain_run//2)
        self.assertLess(dither_error, .5*plain_error)
        self.assertLessEqual(float(np.abs(dithered-truth).max()), 2.01)

    def test_detail_grain_keeps_its_size_relative_to_the_picture(self):
        if self.context is None:
            self.skipTest('no offscreen GL context')
        from tools.v7_gl_viewer import (GRAIN_AMOUNT, GRAIN_DETAIL_AMOUNT,
                                        GRAIN_DETAIL_CYCLES)
        grey = np.zeros((ROWS, COLS), np.float32)
        cells = (COLS*GRAIN_DETAIL_CYCLES, ROWS*GRAIN_DETAIL_CYCLES)

        def mean_frequency(size, **uniforms):
            """Mean frequency of the grain, cycles per luma grid sample."""
            pixels = self._render(grey, size, **uniforms)
            power = np.abs(np.fft.fft2(pixels-pixels.mean()))**2
            radius = np.hypot(
                np.fft.fftfreq(size[1])[:, None]*size[1]/ROWS,
                np.fft.fftfreq(size[0])[None, :]*size[0]/COLS)
            return float((power*radius).sum()/power.sum())

        small, large = (400, 480), (1200, 1440)
        detail = [mean_frequency(size, grain_kind=1, grain_cells=cells,
                                 grain_amount=GRAIN_DETAIL_AMOUNT)
                  for size in (small, large)]
        pixel = [mean_frequency(size, grain_kind=0, grain_amount=GRAIN_AMOUNT)
                 for size in (small, large)]
        # Detail grain: the same place in the picture's spectrum at any size,
        # just above the band the wire carries. Per-pixel grain: three times
        # further out on a window three times as large.
        self.assertAlmostEqual(detail[0], detail[1], delta=.1*detail[0])
        self.assertGreater(detail[0], .5)
        self.assertLess(detail[0], 2.5)
        self.assertGreater(pixel[1], 2.5*pixel[0])

    def test_detail_grain_leaves_black_alone(self):
        if self.context is None:
            self.skipTest('no offscreen GL context')
        from tools.v7_gl_viewer import (GRAIN_DETAIL_AMOUNT,
                                        GRAIN_DETAIL_CYCLES)
        black = np.full((ROWS, COLS), -1.0, np.float32)
        pixels = self._render(
            black, (240, 288), grain_kind=1, grain_amount=GRAIN_DETAIL_AMOUNT,
            grain_cells=(COLS*GRAIN_DETAIL_CYCLES, ROWS*GRAIN_DETAIL_CYCLES))
        self.assertEqual(float(pixels.max()), 0.0)


if __name__ == '__main__':
    unittest.main()
