"""V7 send-path optimisations must not change the wire.

Each fast path is checked against the computation it replaced: the cached
emission filter against scipy's filtfilt, the cached resample filter against
resample_poly's own design, the direct body scatter against np.add.at, the
periodic pilot-tone table against the cosines, the compiled bake composite
against its NumPy reference, and nearest-sampled runtime compositing against
compositing the full-size layers.
"""
import unittest

import numpy as np
from PIL import Image
from scipy.signal import filtfilt, resample_poly

from animation_modem import v7, v7_core
from modem_bake import _alpha_over, _alpha_over_numpy
from modem_image_source import RuntimeImageLibrary

TARGET = .1521/np.sqrt(1 + 10**(v7.CLOCK_REL_DB/10))


class V7EncodeSpeedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = v7.load_model(TARGET, 'nearest')
        with Image.open(v7.REFERENCE_FIXTURE) as source:
            cls.values = v7.image_values(
                v7.prepare_image(source, encode_filter='nearest'),
                cls.model.coder.grids, encode_filter='nearest')
        cls.rng = np.random.default_rng(11)

    def test_cached_emission_filter_is_scipy_filtfilt(self):
        window, zi = v7_core._emission_design(v7_core.EMIT_TAPS, 14000.0, 48000.0)
        packet = v7.encode_pulse_frame(self.model, self.values, 3)
        for x in (packet, self.rng.standard_normal((1000, 2)).astype(np.float32),
                  self.rng.standard_normal((4000, 1))):
            np.testing.assert_array_equal(
                v7_core._fir_filtfilt(window, zi, x),
                filtfilt(window, [1.0], x, axis=0))
        self.assertFalse(window.flags.writeable or zi.flags.writeable)

    def test_cached_speed_filter_is_resample_polys_own(self):
        packet = v7.encode_pulse_frame(self.model, self.values, 2)
        for rate in (32000, 44100, 88200, 96000):
            for speed in (.8, 1.0, 1.5, 3.0):
                target = v7_core.speed_length(len(packet), rate, speed)
                if target == len(packet):
                    continue
                from fractions import Fraction
                ratio = Fraction(target, len(packet)).limit_denominator(256)
                with self.subTest(rate=rate, speed=speed):
                    np.testing.assert_array_equal(
                        resample_poly(packet, ratio.numerator, ratio.denominator,
                                      axis=0),
                        resample_poly(packet, ratio.numerator, ratio.denominator,
                                      axis=0, window=v7_core._speed_filter(
                                          ratio.numerator, ratio.denominator,
                                          packet.dtype.str)))

    def test_direct_scatter_is_add_at(self):
        coeffs = self.model.coder.forward(self.values)/self.model.coder.gains
        for counter in range(v7.TAIL_PHASES):
            got = v7.encode_frame_coeffs(self.model, coeffs, counter, return_X=True)
            idx = self.model.rank_tables[counter]
            vals = self.model.gain[np.maximum(idx, 0)]*(coeffs-self.model.mu)[np.maximum(idx, 0)]
            vals[idx < 0] = 0
            tx = vals @ v7.H8.T
            want = np.zeros((v7.F, 65, 2), complex)
            np.add.at(want, (v7.GROUP_SYMBOLS.ravel(), np.repeat(v7.GROUP_BINS, 8),
                             np.repeat(v7.GROUP_STREAMS, 8)),
                      (tx*v7.GROUP_QMULT[:, None]).ravel())
            want += v7.SCATTERED_PILOTS
            np.testing.assert_array_equal(got, want)

    def test_pilot_tone_table_is_the_cosines_reduced_exactly(self):
        packet = v7.encode_pulse_frame(self.model, self.values, 4)
        for counter in (1, 2, 7, 1000, 1048575):
            got = v7._add_pilot_tones(packet, counter).astype(float) - packet
            body = packet[v7.PULSE.SYNC_LEN:v7.PULSE.SYNC_LEN+v7.FRAME].astype(float)
            amplitude = (np.sqrt(2)*np.sqrt(np.mean(body*body)) *
                         10**(v7.PILOT_TONE_REL_DB/20))
            # The phase origin reduced modulo N first, as exact integers.
            k = ((counter-1)*v7.PILOT_TONE_DURATION + np.arange(len(packet))) % v7.N
            want = sum(amplitude*np.cos(2*np.pi*b*k/v7.N + v7.PILOT_TONE_PHASES[b])
                       for b in v7.PILOT_TONE_BINS)
            with self.subTest(counter=counter):
                np.testing.assert_allclose(got, np.repeat(want[:, None], 2, axis=1),
                                           rtol=0, atol=2e-7)

    def test_compiled_bake_composite_matches_numpy(self):
        values = np.arange(256, dtype=np.uint8)
        grid = np.zeros((256, 256, 4), np.uint8)
        grid[..., :3] = values[:, None, None]
        grid[..., 3] = values[None, :]
        clear, opaque = np.zeros_like(grid), np.full_like(grid, 255)
        for background in ((4, 4, 4), (0, 0, 0), (255, 255, 255), (128, 7, 251)):
            for main, front in ((grid, clear), (opaque, grid),
                                (self.rng.integers(0, 256, (96, 80, 4), dtype=np.uint8),
                                 self.rng.integers(0, 256, (96, 80, 4), dtype=np.uint8))):
                np.testing.assert_array_equal(_alpha_over(main, front, background),
                                              _alpha_over_numpy(main, front, background))

    def _library(self, main, front, main_sbs, front_sbs):
        library = RuntimeImageLibrary.__new__(RuntimeImageLibrary)
        library.frames = 4
        library._layers_for = lambda index, folders: (main, front, main_sbs, front_sbs)
        return library

    def _layer(self, height, width, sbs, channels):
        array = self.rng.integers(0, 256, (height, width*(2 if sbs else 1),
                                           3 if sbs else channels), dtype=np.uint8)
        if sbs:
            array[:, width:, 0] = self.rng.choice([0, 255, 90], (height, width))
        elif channels == 4:
            pick = self.rng.random((height, width))
            array[..., 3] = np.where(pick < .3, 0, np.where(pick < .6, 255, array[..., 3]))
        return array

    def test_nearest_composite_matches_full_size_composite(self):
        layouts = (((96, 80), (96, 80)), ((960, 540), (960, 540)),
                   ((45, 80), (45, 80)), ((300, 200), (250, 260)),
                   ((37, 23), (37, 23)))
        for (mh, mw), (fh, fw) in layouts:
            for main_sbs, front_sbs, channels in ((False, False, 4), (True, True, 3),
                                                  (True, False, 3), (False, False, 3)):
                main = self._layer(mh, mw, main_sbs, channels)
                front = self._layer(fh, fw, front_sbs, 4)
                library = self._library(main, front, main_sbs, front_sbs)
                for rotation, mirror in ((0, False), (90, False), (180, True),
                                         (270, False), (30, False), (-90, True)):
                    full = library.composite(1, 0, 0, (4, 4, 4), rotation, mirror)
                    fast = library.composite_nearest(1, 0, 0, v7.PREPARED_SIZE,
                                                     (4, 4, 4), rotation, mirror)
                    with self.subTest(size=(mh, mw), front=(fh, fw), sbs=main_sbs,
                                      rotation=rotation, mirror=mirror):
                        np.testing.assert_array_equal(
                            np.asarray(v7.prepare_image(fast, 'nearest')),
                            np.asarray(v7.prepare_image(full, 'nearest')))
                        self.assertEqual(fast.info['source_dimensions'], full.size)


if __name__ == '__main__':
    unittest.main()
