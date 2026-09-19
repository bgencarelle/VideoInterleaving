"""v5 (hd-dwt) source coder: symmetric CDF 9/7, band-cut pyramid, repeat copies.

Pins the properties the design depends on: the transform is exactly
invertible; the coder fills WIRE_HD exactly; every copy rides the other
stereo channel well away from its original; a dead leg or dead carriers are
rescued by the copies; the band RMS table still matches what it claims to
measure; and a packet round trips through the real encode/decode.
"""
import unittest

import numpy as np
from PIL import Image

from animation_modem import transport3 as V3
from animation_modem.core import decode_packet
from animation_modem.imaging import HD_MONO_CAPACITY, image_values, plane_grids
from animation_modem.wavelet import (Cdf97Coder, HD_BAND_RMS, _cdf97_pack_order,
                                     _cdf97_subband_dims, _slot_carriers,
                                     cdf97_forward_2d, cdf97_inverse_2d,
                                     hd_dwt_coder)


def natural_frames(count, seed):
    """Synthetic frames with a natural-image 1/f spectrum, 80x96 RGB."""
    rng = np.random.default_rng(seed)
    fy, fx = np.meshgrid(np.fft.fftfreq(96), np.fft.fftfreq(80), indexing='ij')
    f = np.hypot(fy, fx)
    f[0, 0] = 1
    out = []
    for _ in range(count):
        ch = [np.real(np.fft.ifft2((rng.standard_normal((96, 80))
                                    + 1j*rng.standard_normal((96, 80)))/f))
              for _ in range(3)]
        rgb = np.stack([ch[0] + .5*ch[1], ch[0] + .3*ch[2],
                        ch[0] - .2*ch[1] + .4*ch[2]], -1)
        rgb = (rgb - rgb.mean())/rgb.std()*55 + rng.uniform(70, 180, 3)
        out.append(Image.fromarray(np.uint8(np.clip(rgb, 0, 255))))
    return out


class TransformTests(unittest.TestCase):
    def test_symmetric_cdf97_is_exactly_invertible(self):
        rng = np.random.default_rng(1)
        for shape, levels in (((96, 80), 2), ((48, 40), 2), ((64, 64), 3)):
            x = rng.standard_normal(shape)
            back = cdf97_inverse_2d(cdf97_forward_2d(x, levels), levels, shape)
            self.assertLess(np.abs(back - x).max(), 1e-12)

    def test_no_wraparound_between_opposite_edges(self):
        """The periodic boundary mixed the right edge into the left. A ramp
        (dark left, bright right) must keep its coarse band monotone."""
        ramp = np.tile(np.linspace(-1, 1, 80), (96, 1))
        ll = cdf97_forward_2d(ramp, 2)[1]['LL']
        self.assertTrue(np.all(np.diff(ll[10]) > 0))


class CoderGeometryTests(unittest.TestCase):
    def setUp(self):
        self.coder = hd_dwt_coder()

    def test_fills_wire_hd_exactly(self):
        self.assertEqual(HD_MONO_CAPACITY, V3.WIRE_HD.capacity)
        self.assertEqual(self.coder.count, V3.WIRE_HD.capacity)
        self.assertEqual(self.coder.n_orig, 2880)       # half-res pyramid
        self.assertEqual(self.coder.grids, [(96, 80), (48, 40), (48, 40)])

    def test_every_ll_coefficient_is_repeated(self):
        copied = set(self.coder.copy_of.tolist())
        offset = 0
        for (rows, cols), keep in zip(self.coder.grids, self.coder.keep):
            ll = (rows >> 2)*(cols >> 2)           # coarsest LL leads the pack
            self.assertTrue(set(range(offset, offset+ll)) <= copied)
            offset += keep

    def test_slots_are_a_valid_diverse_placement(self):
        slots = self.coder.slots(V3.WIRE_HD)
        self.assertEqual(len(set(slots.tolist())), len(slots))
        self.assertLess(slots.max(), V3.WIRE_HD.capacity)
        bins, chans = _slot_carriers(V3.WIRE_HD)
        home = slots[self.coder.copy_of]
        copy = slots[self.coder.n_orig:]
        self.assertTrue(np.all(chans[home] != chans[copy]))
        self.assertTrue(np.all(np.abs(bins[home] - bins[copy])
                               >= Cdf97Coder.MIN_COPY_SPREAD))

    def test_refuses_a_layout_too_small(self):
        with self.assertRaises(ValueError):
            self.coder.slots(V3.WIRE)


class CoderRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.coder = hd_dwt_coder()
        self.values = image_values(natural_frames(1, 5)[0], self.coder.grids)
        self.sent = self.coder.forward(self.values)
        self.clean = self.coder.inverse(self.sent)

    def _decode(self, dead):
        sent = self.sent.copy()
        rel = np.ones(self.coder.count)
        sent[dead], rel[dead] = 0, 1e-6
        return self.coder.inverse(sent, rel, np.full(self.coder.count, 1e-6))

    def test_clean_reliability_path_matches_plain_inverse(self):
        out = self._decode(np.zeros(self.coder.count, bool))
        self.assertLess(np.abs(out - self.clean).max(), 1e-3)

    def test_dead_stereo_leg_keeps_the_whole_half_res_colour_picture(self):
        """Every LL value has a copy on the other channel, so the coarse
        picture -- colour included -- survives either leg dying."""
        _, chans = _slot_carriers(V3.WIRE_HD)
        on_leg = chans[self.coder.slots(V3.WIRE_HD)]
        for leg in (0, 1):
            out = self._decode(on_leg == leg)
            offset = 0
            for (rows, cols) in self.coder.grids:
                got = cdf97_forward_2d(out[offset:offset+rows*cols].reshape(rows, cols), 2)
                want = cdf97_forward_2d(self.clean[offset:offset+rows*cols].reshape(rows, cols), 2)
                self.assertLess(np.abs(got[1]['LL'] - want[1]['LL']).max(), 1e-3)
                offset += rows*cols

    def test_copies_beat_no_copies_when_upper_carriers_die(self):
        """A 6 kHz lowpass kills bins above 16. Copies must recover more."""
        bins, _ = _slot_carriers(V3.WIRE_HD)
        dead = bins[self.coder.slots(V3.WIRE_HD)] > 16
        with_copies = self._decode(dead)
        dead_both = dead.copy()
        dead_both[self.coder.n_orig:] = True           # as if never sent
        without = self._decode(dead_both)
        err = lambda v: np.mean((v - self.clean)**2)
        self.assertLess(err(with_copies), err(without)*.8)

    def test_luma_first_gate_keeps_shape_before_detail_and_color(self):
        """At one confidence, broad luma must be the last thing erased."""
        confidence = np.full(self.coder.n_orig, .50)
        gate = self.coder.recovery_gate(confidence)
        y_ll = 0
        y_detail = (96 >> 2)*(80 >> 2)
        cb_detail = self.coder.keep[0] + (48 >> 2)*(40 >> 2)
        self.assertGreater(gate[y_ll], gate[y_detail])
        self.assertGreater(gate[y_detail], gate[cb_detail])
        self.assertEqual(gate[cb_detail], 0.0)

    def test_high_confidence_coefficients_are_untouched(self):
        gate = self.coder.recovery_gate(np.full(self.coder.n_orig, .85))
        np.testing.assert_array_equal(gate, np.ones(self.coder.n_orig))

    def test_unreliable_luma_detail_becomes_soft_not_a_false_edge(self):
        """A large low-confidence tape error must not become image detail."""
        index = (96 >> 2)*(80 >> 2) + 17       # luma LH, not protected LL
        sent = self.sent.copy()
        rel = np.ones(self.coder.count)
        noise = np.full(self.coder.count, 1e-6)
        sent[index] += 8.0
        rel[index] = .02
        noise[index] = .5
        # Remove any repeat so the test represents one untrustworthy reading.
        copies = np.flatnonzero(self.coder.copy_of == index) + self.coder.n_orig
        rel[copies] = 1e-6
        noise[copies] = .5
        got = self.coder.inverse(sent, rel, noise)
        quiet = self.coder.inverse(self.sent, rel, noise)
        false_detail = np.mean((got[:96*80] - quiet[:96*80])**2)

        # The former Wiener-only result, retained here as the regression
        # baseline: posterior weighting alone leaves a visible random edge.
        def old_luma(observed):
            a = self.coder.gains*np.maximum(rel, 1e-6)
            num, den = a*observed/noise, a*a/noise
            top = num[:self.coder.n_orig].copy()
            bottom = den[:self.coder.n_orig].copy()
            np.add.at(top, self.coder.copy_of, num[self.coder.n_orig:])
            np.add.at(bottom, self.coder.copy_of, den[self.coder.n_orig:])
            kept = self.coder.variance*top/(1 + self.coder.variance*bottom)
            flat = np.zeros(96*80)
            flat[:self.coder.keep[0]] = kept[:self.coder.keep[0]]
            return self.coder._unpack(flat, 96, 80).ravel()
        old_false_detail = np.mean((old_luma(sent) - old_luma(self.sent))**2)
        self.assertLess(false_detail, old_false_detail*.01)


class BandTableTests(unittest.TestCase):
    def test_band_rms_table_matches_its_measurement(self):
        grids = plane_grids('hd-dwt')
        acc = {}
        frames = natural_frames(48, 7)
        for im in frames:
            v = image_values(im, grids)
            offset = 0
            for p, (rows, cols) in enumerate(grids):
                bands = cdf97_forward_2d(v[offset:offset+rows*cols].reshape(rows, cols), 2)
                offset += rows*cols
                for level, band, _, _ in _cdf97_pack_order(_cdf97_subband_dims(rows, cols, 2)):
                    if level == 1:
                        acc[p, band] = acc.get((p, band), 0) + np.mean(bands[level][band]**2)
        for (p, band), total in acc.items():
            measured = np.sqrt(total/len(frames))
            if band in ('LH', 'HL'):                # the table pools the pair
                measured = np.sqrt((acc[p, 'LH'] + acc[p, 'HL'])/2/len(frames))
            self.assertAlmostEqual(HD_BAND_RMS[p][band], measured,
                                   delta=.1*measured, msg=f'plane {p} {band}')


class WireRoundTripTests(unittest.TestCase):
    def test_packet_round_trips_as_verified_hd_dwt(self):
        coder = hd_dwt_coder()
        values = image_values(natural_frames(1, 11)[0], coder.grids)
        audio = V3.encode(values, V3.WIRE_HD, coder, 7, 7, 9, profile=2)
        got = decode_packet(audio, V3.WIRE_HD, coder, coders={2: coder})
        self.assertEqual(got.identity, 'verified_header')
        self.assertEqual(got.extra['profile'], 'hd-dwt')
        want = coder.inverse(coder.forward(values))
        psnr = 10*np.log10(4/np.mean((got.values - want)**2))
        self.assertGreater(psnr, 40)


if __name__ == '__main__':
    unittest.main()
