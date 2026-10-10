import unittest

import numpy as np

from tools.v7_sk_fold import (
    B_LIMIT,
    serpentine_decode_frame,
    serpentine_decode_frame_exact,
    serpentine_encode_frame,
    compress,
    expand_exact,
    serpentine_place_pair,
    serpentine_stair_values,
)


class SerpentineMathTests(unittest.TestCase):
    def setUp(self):
        # Identity compander on the supported source range, independent of
        # any fitted training corpus.
        self.table = np.linspace(0.0, 1.0, 1025)
        self.inverse = np.linspace(0.0, B_LIMIT, 1025)

    def _roundtrip(self, hosts, guests, level, step, scale, offset):
        sent = serpentine_encode_frame(hosts, guests, level, step, scale,
                                       self.table, offset)
        return serpentine_decode_frame(sent, level, step, scale, self.inverse, offset)

    def test_matched_guest_inverse_on_positive_and_negative_stairs(self):
        hosts = np.array([2.0, -2.0, 3.0, -3.0])
        guests = np.array([1.25, -2.5, -3.0, 4.0])
        level = np.ones(4, np.int64)
        step = np.full(4, 1.0)
        h, g = self._roundtrip(hosts, guests, level, step, np.ones(4), np.zeros(4))
        np.testing.assert_allclose(h, hosts)
        np.testing.assert_allclose(g, guests, atol=1e-9)

    def test_plain_exact_nonunit_scale_and_dither(self):
        hosts = np.array([-.73, .41, 2.25])
        zeros = np.zeros(3)
        got, guest = self._roundtrip(hosts, np.array([2., 3., 4.]), zeros,
                                     np.ones(3), np.array([.5, 2., 3.]),
                                     np.array([.2, -.1, .3]))
        np.testing.assert_allclose(got, hosts)
        np.testing.assert_array_equal(guest, np.zeros(3))

    def test_host_bound_and_midpoint_activity_offsets(self):
        hosts = np.array([-2.31, -.41, .32, 2.29])
        step = np.ones(4)
        offset = np.array([.2, -.1, .15, -.25])
        level = np.ones(4, np.int64)
        got, _ = self._roundtrip(hosts, np.zeros(4), level, step, np.ones(4), offset)
        self.assertTrue(np.all(np.abs(got-hosts) <= .5*step+1e-12))
        np.testing.assert_allclose(serpentine_stair_values(hosts, level, step, offset), got)

    def test_adjacent_stairs_join_guest_orientation_continuously(self):
        # Same endpoint: +1 on an even stair and -1 on its odd neighbour.
        hosts = np.array([0.0, 1.0])
        guests = np.array([B_LIMIT, B_LIMIT])
        level = np.ones(2, np.int64)
        sent = serpentine_encode_frame(hosts, guests, level, np.ones(2),
                                       np.ones(2), self.table, np.zeros(2))
        np.testing.assert_allclose(sent, [0.5, 0.5], atol=1e-10)
        got_hosts, got_guests = serpentine_decode_frame(
            sent, level, np.ones(2), np.ones(2), self.inverse, np.zeros(2))
        np.testing.assert_allclose(got_guests, [B_LIMIT, B_LIMIT], atol=1e-6)
        np.testing.assert_array_equal(got_hosts, hosts)

    def test_pair_placement_is_direct_algebra(self):
        out = np.zeros(4)
        serpentine_place_pair(out, np.array([2., -1.]), np.array([0., 3.]),
                              np.array([0, 2]), np.array([10., 20.]),
                              np.array([2., 4.]), np.array([.5, 2.]),
                              np.array([1, 3]), 3.)
        np.testing.assert_allclose(out, [16., 12., 32., -12.])

    def test_invalid_level_rejected(self):
        with self.assertRaises(ValueError):
            serpentine_encode_frame(np.zeros(1), np.zeros(1), np.array([2]),
                                    np.ones(1), np.ones(1), self.table, np.zeros(1))
        with self.assertRaises(ValueError):
            serpentine_decode_frame(np.zeros(1), np.array([2]), np.ones(1),
                                    np.ones(1), self.inverse, np.zeros(1))

    def test_exact_inverse_of_nonlinear_forward_table(self):
        x = np.linspace(0.0, 1.0, 1025)
        table = x**1.7
        sources = np.array([1e-5, 2.0, 6.0, 12.0, 15.9,
                            -1e-5, -2.0, -6.0, -12.0, -15.9])
        codes = np.array([compress(table, value) for value in sources])
        recovered = np.array([expand_exact(table, code) for code in codes])
        np.testing.assert_allclose(recovered, sources, atol=1e-10, rtol=0.0)
        old_uniform_inverse = np.abs(codes)*B_LIMIT
        self.assertGreater(np.max(np.abs(old_uniform_inverse-np.abs(sources))), 1e-6)
        self.assertTrue(np.isfinite(expand_exact(table, 1.0)))
        self.assertEqual(expand_exact(table, 0.0), 0.0)

    def test_exact_decoder_matches_guest_across_signed_stairs(self):
        x = np.linspace(0.0, 1.0, 1025)
        table = x**1.7
        hosts = np.array([2.0, -2.0, 3.0, -3.0, .2, -.2])
        guests = np.array([2.0, -2.0, 6.0, -6.0, 12.0, -15.9])
        level = np.ones(len(hosts), np.int64)
        step = np.array([1., 1., .7, .7, 1.2, 1.2])
        scale = np.array([1., 1., .5, 2., 1.5, .75])
        offset = np.array([.1, -.1, .13, -.13, .2, -.2])
        sent = serpentine_encode_frame(hosts, guests, level, step, scale, table, offset)
        got_hosts, got_guests = serpentine_decode_frame_exact(
            sent, level, step, scale, table, offset)
        np.testing.assert_allclose(got_hosts, offset+np.round((hosts-offset)/step)*step,
                                   atol=1e-12)
        np.testing.assert_allclose(got_guests, guests, atol=1e-10, rtol=0.0)


if __name__ == '__main__':
    unittest.main()
