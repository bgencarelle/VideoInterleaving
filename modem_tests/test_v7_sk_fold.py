"""Nested fold kernels, rate-distortion bound and a real-wire round trip."""
import math
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn

from tools import v7_sk_fold as sk

FIXTURE = Path(__file__).with_name('fixtures')/'v7_reference_face.png'


def laplace(shape, seed):
    return np.random.default_rng(seed).laplace(scale=math.sqrt(.5), size=shape)


class KernelTests(unittest.TestCase):
    def test_dct_matches_reference_transform(self):
        plane = np.random.default_rng(1).standard_normal((12, 10))
        ours = sk.separable(sk.dct_matrix(12), plane, sk.dct_matrix(10))
        np.testing.assert_allclose(ours, dctn(plane, norm='ortho'), atol=1e-12)

    def test_area_matrix_preserves_means_for_fractional_ratios(self):
        weights = sk.area_matrix(500, 96)
        np.testing.assert_allclose(weights.sum(1), 1, atol=1e-12)
        np.testing.assert_allclose(weights.sum(0), 96/500, atol=1e-12)

    def test_compander_is_odd_monotone_and_spans_unit_range(self):
        table, power = sk.fit_compander(laplace(20000, 2), 1/3)
        x = np.linspace(-sk.B_LIMIT, sk.B_LIMIT, 2001)
        y = np.array([sk.compress(table, v) for v in x])
        self.assertTrue(np.all(np.diff(y) > 0))
        np.testing.assert_allclose(y, -y[::-1], atol=1e-12)
        self.assertAlmostEqual(y[-1], 1.0)
        self.assertTrue(0 < power < 1)

    def test_shifted_stairs_reduce_to_plain_stairs_at_zero_offset(self):
        rng = np.random.default_rng(11)
        n = 400
        hosts, guests = laplace(n, 12), .4*laplace(n, 13)
        level = (np.arange(n) % 4 > 0).astype(np.int64)
        step = rng.uniform(.3, 1.2, n)
        scale = rng.uniform(.7, 1.3, n)
        alpha = np.array([0.0, .35, .35])
        table, _ = sk.fit_compander(guests, 1/3)
        inverse = sk.expand_table(table)
        density = sk.residual_density(guests, table)
        zero = np.zeros(n)
        for index, width in ((0, .7), (1, .7), (-3, .4), (5, 1.1)):
            mass, mean = sk.laplace_cell((index-.5)*width, (index+.5)*width)
            self.assertAlmostEqual(mean, sk.laplace_centroid(index, width), places=12)
            self.assertGreater(mass, 0)
        total = sum(sk.laplace_cell((k-.5)*.7+.2, (k+.5)*.7+.2)[0] for k in range(-60, 61))
        self.assertAlmostEqual(total, 1.0, places=9)
        plain = sk.encode_frame(hosts, guests, zero, level, step, scale, alpha, 3, .25, table, 1.0)
        np.testing.assert_allclose(
            sk.encode_frame_dithered(hosts, guests, level, step, scale, alpha, table, zero), plain)
        np.testing.assert_allclose(sk.stair_values_dithered(hosts, level, step, zero),
                                   sk.stair_values(hosts, level, step), atol=1e-12)
        sigma = np.full(n, .02)
        received = plain+.02*rng.standard_normal(n)
        for ours, theirs in zip(
                sk.soft_decode_frame_dithered(received, level, step, scale, alpha, sigma,
                                              density, inverse, zero),
                sk.soft_decode_frame(received, level, step, scale, alpha, sigma, density, inverse)):
            np.testing.assert_allclose(ours, theirs, atol=1e-9)

    def test_a_shifted_staircase_is_read_back_on_the_same_shift(self):
        rng = np.random.default_rng(14)
        n = 400
        hosts, guests = laplace(n, 15), .4*laplace(n, 16)
        level = np.ones(n, np.int64)
        step = np.full(n, .8)
        scale = np.ones(n)
        alpha = np.array([0.0, .35, .35])
        table, _ = sk.fit_compander(guests, 1/3)
        inverse = sk.expand_table(table)
        density = sk.residual_density(guests, table)
        offset = rng.uniform(-.5, .5, n)*step
        sent = sk.encode_frame_dithered(hosts, guests, level, step, scale, alpha, table, offset)
        sigma = np.full(n, 1e-3)
        got_hosts, got_guests = sk.soft_decode_frame_dithered(
            sent, level, step, scale, alpha, sigma, density, inverse, offset)
        # Hosts land in their own cell of the shifted stairs; guests survive.
        self.assertLessEqual(float(np.max(np.abs(got_hosts-hosts))), .8+1e-9)
        np.testing.assert_allclose(
            got_hosts, sk.stair_values_dithered(hosts, level, step, offset), atol=1e-6)
        self.assertLess(float(np.mean((got_guests-guests)**2)), .1*float(np.mean(guests**2)))
        wrong, _ = sk.soft_decode_frame_dithered(
            sent, level, step, scale, alpha, sigma, density, inverse, np.zeros(n))
        self.assertGreater(float(np.mean((wrong-hosts)**2)),
                           float(np.mean((got_hosts-hosts)**2)))

    def test_water_filling_matches_closed_forms(self):
        # Equal variances: D = n*lambda*2^(-2R/n).
        theta, distortion, active = sk.reverse_water_fill(np.full(50, 2.0), 100.0)
        self.assertEqual(active, 50)
        self.assertAlmostEqual(distortion, 50*2.0*2**(-4.0), places=9)
        # Too few bits for the weak half: it is left at its variance.
        lam = np.concatenate((np.full(10, 16.0), np.full(10, 1e-3)))
        theta, distortion, active = sk.reverse_water_fill(lam, 10.0)
        self.assertEqual(active, 10)
        self.assertAlmostEqual(distortion, 10*16.0*2**(-2.0)+10*1e-3, places=9)


class NestedFoldTests(unittest.TestCase):
    def make(self, noise=.01, **kwargs):
        options = dict(linear=20, triple=60, kappa=24, guard=3.0, kappa2=15)
        options.update(kwargs)
        fold = sk.NestedFold(np.full(200, noise), **options)
        return fold.fit(laplace((40, fold.guests), 3))

    def test_layout_counts(self):
        fold = self.make()
        self.assertEqual(fold.guests, 120+2*60)
        slot, layer = fold.guest_order()
        self.assertEqual(len(slot), fold.guests)
        self.assertTrue(np.all(fold.level[slot[layer == 1]] == 2))

    def test_emitted_power_is_the_design_power(self):
        fold = self.make()
        frames = [fold.encode(laplace(200, 10+i), laplace((1, fold.guests), 50+i)) for i in range(300)]
        power = np.mean(np.square(frames), axis=0)
        for level in (0, 1, 2):
            self.assertAlmostEqual(float(power[fold.level == level].mean()), 1.0, delta=.08)

    def test_noiseless_round_trip_never_slips(self):
        fold = self.make()
        hosts, guests = laplace(200, 4), laplace((1, fold.guests), 5)
        a, b = fold.decode(fold.encode(hosts, guests))
        folded = fold.level > 0
        np.testing.assert_allclose(a[~folded], hosts[~folded]*fold.shrink[~folded], atol=1e-12)
        # A host is recovered to its own stair; the centroid lies inside it.
        self.assertTrue(np.all(abs(a[folded]-hosts[folded]) <= fold.step[folded]))
        slot, layer = fold.guest_order()
        single = fold.level[slot] == 1
        self.assertGreater(np.corrcoef(b[single], guests[0, single])[0, 1], .97)
        fine = layer == 1
        # The innermost layer is decoded with design-noise conditional means.
        self.assertGreater(np.corrcoef(b[fine], guests[0, fine])[0, 1], .8)

    def test_guard_holds_at_design_noise(self):
        fold = self.make()
        rng = np.random.default_rng(6)
        slips = total = 0
        for i in range(60):
            hosts = laplace(200, 100+i)
            sent = fold.encode(hosts, laplace((1, fold.guests), 200+i))
            a, _ = fold.decode(sent+fold.noise*rng.standard_normal(200))
            clean, _ = fold.decode(sent)
            folded = fold.level > 0
            slips += int(np.sum(a[folded] != clean[folded]))
            total += int(folded.sum())
        # Three-sigma guard each side: well under one percent of stairs slip.
        self.assertLess(slips/total, .01)

    def test_guests_improve_on_sending_nothing(self):
        fold = self.make()
        rng = np.random.default_rng(7)
        error = energy = 0.0
        for i in range(60):
            guests = laplace((1, fold.guests), 300+i)
            sent = fold.encode(laplace(200, 400+i), guests)
            _, b = fold.decode(sent+fold.noise*rng.standard_normal(200))
            error += np.sum((b-guests[0])**2)
            energy += np.sum(guests**2)
        self.assertLess(error/energy, .5)

    def test_invalid_layouts_are_rejected(self):
        with self.assertRaises(ValueError):
            sk.NestedFold(np.full(10, .01), linear=8, triple=4, kappa=20)
        with self.assertRaises(ValueError):
            sk.NestedFold(np.full(10, .01), linear=0, triple=0, kappa=4, guard=3.0)


class EarlierSurfaceDiagnosisTests(unittest.TestCase):
    """Pins the defects found in tools/v7_fold_surface.py."""

    def test_three_strip_extra_coordinate_decodes_to_zero_or_a_huge_value(self):
        from tools.v7_fold_surface import surface_decode, surface_encode
        limit, mu = 12.0, 4.0
        compress = lambda u: np.sign(u)*np.log1p(mu*np.minimum(abs(u), limit)/limit)/np.log1p(mu)
        expand = lambda c: np.sign(c)*limit*np.expm1(abs(c)*np.log1p(mu))/mu
        extra = np.array([0., .5, 1., 2., 2.2, 3., 6.])            # standard deviations
        source = np.stack((np.zeros(7), np.zeros(7), extra), -1)
        decoded = surface_decode(surface_encode(.5+.5*compress(source), 2, 3), 3, 3)
        recovered = expand(decoded[:, 2]*2-1)
        np.testing.assert_allclose(recovered[:4], 0, atol=1e-9)    # up to 2 sigma vanishes
        self.assertTrue(np.all(recovered[4:] > 5.5))                # then jumps past 5 sigma

    def test_existing_coordinate_is_squeezed_by_the_strip_count(self):
        from tools.v7_fold_surface import surface_encode
        a = surface_encode(np.array([[.50, .5, .5]]), 2, 3)[0, 0]
        b = surface_encode(np.array([[.56, .5, .5]]), 2, 3)[0, 0]
        self.assertAlmostEqual(abs(b-a), .06/3, places=12)


class RealWireTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tools import v7_sk_study as study, v7_sk_wire as wire
        cls.study, cls.wire = study, wire
        cls.geometry = study.Geometry()
        cls.link = wire.Link(cls.geometry)
        cls.rgb = np.asarray(Image.open(FIXTURE).convert('RGB'))
        analyse = study.Analyzer(study.LATTICE)
        # Functional statistics only: shifted and mirrored copies of the fixture.
        variants = [cls.rgb, cls.rgb[:, ::-1], np.roll(cls.rgb, 17, 0), np.roll(cls.rgb, 23, 1),
                    np.roll(cls.rgb[:, ::-1], 31, 0), cls.rgb[::-1]]
        cls.planes = [analyse(np.ascontiguousarray(v)) for v in variants]
        cls.stats = study.Statistics(cls.planes)
        cls.noise = np.full(len(cls.geometry.power), .012)/np.sqrt(
            cls.geometry.power/cls.geometry.power.mean())

    def test_geometry_matches_the_production_fold_table(self):
        g = self.geometry
        self.assertEqual(len(g.host_position), 1920)
        self.assertEqual(len(g.fold_hosts), 500)
        self.assertEqual(len(np.intersect1d(g.host_position, g.guest_position)), 0)
        self.assertEqual(set(g.guest_position[:500]), set(g.fold_guests))

    def test_nested_packet_survives_the_production_receiver(self):
        study, link = self.study, self.link
        scheme = study.Nested(self.geometry, self.stats, self.noise, self.planes,
                              linear=208, triple=300, kappa=24, guard=3.0, kappa2=15, guard2=3.0)
        symbols = scheme.encode(self.planes[0])
        reference = link.packet(self.rgb, None, 1, 9)
        audio, gain = self.wire.matched(link.packet(self.rgb, symbols, 1, 9), reference)
        self.assertLessEqual(gain, 1.0)
        ours = self.wire.audio_levels(audio)
        theirs = self.wire.audio_levels(reference)
        self.assertLessEqual(ours[0], theirs[0]*(1+1e-6))
        self.assertLessEqual(ours[1], theirs[1]*(1+1e-6))
        self.assertEqual(audio.shape, reference.shape)              # same packet length
        received, values, status = link.receive(audio)
        self.assertEqual(status, 'received')
        self.assertIsNotNone(values)
        error = received-symbols
        self.assertLess(float(np.sqrt(np.mean(error**2))), .05)
        active = scheme.active
        sent_host, _ = scheme.fold.decode(symbols[active])
        got_host, _ = scheme.fold.decode(received[active])
        folded = scheme.fold.level > 0
        self.assertLess(float(np.mean(sent_host[folded] != got_host[folded])), .02)
        decoded = scheme.decode(received)
        truth = self.planes[0]
        carried = np.zeros(truth.size, bool)
        carried[self.geometry.host_position] = True
        carried[scheme.q] = True
        carried[0] = False
        t, d = truth.ravel()[carried], decoded.ravel()[carried]
        self.assertGreater(float(np.dot(t, d)/math.sqrt(np.dot(t, t)*np.dot(d, d))), .98)

    def test_adaptive_scale_is_the_same_at_both_ends_of_the_real_wire(self):
        from tools.v7_sk_adaptive import Adaptive
        scheme = Adaptive(self.geometry, self.stats, self.noise, self.planes, passes=2,
                          linear=0, triple=300, kappa=24, guard=1.5, kappa2=20, guard2=1.5,
                          mid_gain=1.5, guest_order='variance')
        self.assertFalse(np.allclose(scheme.law[..., 1], 0))       # a law was fitted
        plane = self.planes[0]
        symbols = scheme.encode(plane)
        reference = self.link.packet(self.rgb, None, 1, 9)
        audio, _ = self.wire.matched(self.link.packet(self.rgb, symbols, 1, 9), reference)
        received, _, status = self.link.receive(audio)
        self.assertEqual(status, 'received')
        sender = scheme.scale(scheme.activity(scheme.sent_hosts(plane)))
        hosts, _ = scheme.fold.decode(received[scheme.active])
        receiver = scheme.scale(scheme.activity(hosts))
        # No side information: the receiver derives the scale from decoded hosts.
        np.testing.assert_allclose(receiver, sender, rtol=.05)
        decoded = scheme.decode(received)
        carried = np.zeros(plane.size, bool)
        carried[self.geometry.host_position] = True
        carried[scheme.q] = True
        carried[0] = False
        t, d = plane.ravel()[carried], decoded.ravel()[carried]
        self.assertGreater(float(np.dot(t, d)/math.sqrt(np.dot(t, t)*np.dot(d, d))), .98)

    def test_levelled_fold_reads_its_level_from_the_signature_slots(self):
        from tools.v7_sk_adaptive import Adaptive, Leveled
        base = Adaptive(self.geometry, self.stats, self.noise, self.planes, passes=2,
                        linear=0, triple=300, kappa=24, guard=1.5, kappa2=20, guard2=1.5)
        scheme = Leveled(base)
        mean = self.stats.mean
        loud = mean+(self.planes[0]-mean)*4.0                     # a much busier picture
        # Four times the detail is between three and four ladder steps of 1.5.
        self.assertGreaterEqual(scheme.level(loud), scheme.level(self.planes[0])+2)
        for plane in (self.planes[0], loud):
            symbols = scheme.encode(plane)
            self.assertEqual(scheme.read_level(symbols), scheme.level(plane))
        for index in range(scheme.LOWEST, scheme.HIGHEST+1):         # every level is readable
            marked = np.zeros(len(scheme.noise))
            marked[scheme.reserved] = (self.study.SIGNATURE_SYMBOL*scheme.pattern *
                                       scheme.marks(index))
            self.assertEqual(scheme.read_level(marked), index)
            # Neither a gain error nor heavy noise may change it.
            self.assertEqual(scheme.read_level(marked*.8), index)
            noisy = marked+np.random.default_rng(index+50).standard_normal(len(marked))
            self.assertEqual(scheme.read_level(noisy), index)
            self.assertGreater(scheme.signature_score(marked), .95)
        reference = self.link.packet(self.rgb, None, 1, 9)
        symbols = scheme.encode(loud)
        audio, _ = self.wire.matched(self.link.packet(self.rgb, symbols, 1, 9), reference)
        received, _, status = self.link.receive(audio)
        self.assertEqual(status, 'received')
        self.assertEqual(scheme.read_level(received), scheme.level(loud))
        decoded = scheme.decode(received)
        carried = np.zeros(loud.size, bool)
        carried[self.geometry.host_position] = True
        carried[base.q] = True
        carried[0] = False
        t, d = loud.ravel()[carried], decoded.ravel()[carried]
        self.assertGreater(float(np.dot(t, d)/math.sqrt(np.dot(t, t)*np.dot(d, d))), .98)


if __name__ == '__main__':
    unittest.main()
