"""Carry image on the carriers the header symbols leave idle.

The header occupies header_width carriers. The rest of every header symbol would
be silence; the one wire fills them (dense_header=True), reclaiming capacity at
the same frame length, band and level. There is no sparse preset to compare
against any more, so the tests pair the shipped wire with a local sparse twin
(dense_header=False) to keep the delta pinned.

What dense_header is NOT is a straight resolution win. Capacity is not
resolution: the profile has to fit for a reason, and a fixed power budget spread
over more coefficients leaves less power on each.
"""
import unittest
from dataclasses import replace

import numpy as np

from animation_modem import transport3 as v3
from animation_modem.imaging import fit_shapes, plane_shapes

DENSE = v3.WIRE
SPARSE = replace(v3.WIRE, dense_header=False)


def round_trip(layout, shapes, profile='color-dct', seed=4, frames=2):
    coder = v3.SourceCoder(shapes)
    values = np.random.default_rng(seed).uniform(-.2, .2, coder.count)
    audio = np.concatenate([v3.encode(values, layout, coder, n, n, frames,
                                      profile=v3.profile_code(profile))
                            for n in range(1, frames+1)])
    rx = v3.Receiver(layout, coder)
    out = rx.feed(np.asarray(audio, np.float32))+rx.flush()
    return out, values, np.asarray(audio)


class DenseHeaderTests(unittest.TestCase):
    def test_the_idle_carriers_are_real_and_counted(self):
        self.assertEqual(len(SPARSE.spare_bins),
                         len(SPARSE.data_bins)-len(SPARSE.header_bins))
        self.assertEqual(SPARSE.header_capacity, 0)
        self.assertEqual(DENSE.header_capacity,
                         DENSE.header_symbols*len(DENSE.spare_bins)*4)
        self.assertEqual(DENSE.capacity, SPARSE.capacity+DENSE.header_capacity)

    def test_nothing_about_the_frame_changes(self):
        """The whole point: no extra samples, no extra bandwidth, no fewer
        frames per second."""
        self.assertEqual(DENSE.frame, SPARSE.frame)
        self.assertEqual(DENSE.packet, SPARSE.packet)
        self.assertEqual(DENSE.top_bin, SPARSE.top_bin)
        self.assertEqual(DENSE.fps, SPARSE.fps)

    def test_the_level_is_untouched(self):
        """Filling the header symbols raises no peak, because the preamble
        holds it. If it did, everything would renormalise down and the extra
        capacity would have been bought with SNR."""
        # SPARSE capacity is 2800, color-dct is 2880 - use fitted shapes for SPARSE
        sparse_shapes = _fit(plane_shapes('color-dct'), SPARSE.capacity)
        dense_shapes = plane_shapes('color-dct')
        _, _, sparse = round_trip(SPARSE, sparse_shapes, profile='color-dct')
        _, _, dense = round_trip(DENSE, dense_shapes, profile='color-dct')
        self.assertAlmostEqual(float(np.max(np.abs(dense))),
                               float(np.max(np.abs(sparse))), places=6)

    def test_the_reclaimed_slots_round_trip(self):
        """Sized to need them: the color-dct 2880 does not fit the twin."""
        shapes = plane_shapes('color-dct')
        self.assertGreater(sum(h*w for h, w in shapes), SPARSE.capacity)
        self.assertLessEqual(sum(h*w for h, w in shapes), DENSE.capacity)
        out, values, _ = round_trip(DENSE, shapes)
        self.assertEqual([r.absolute for r in out], [1, 2])
        for r in out:
            self.assertEqual(r.identity, 'verified_header')
            self.assertLess(np.sqrt(np.mean((r.values-values)**2)), 1e-5)

    def test_dense_and_sparse_are_not_confusable(self):
        """The failure this nearly shipped with.

        A sparse and a dense packet can have the same length, the same top_bin
        and a BIT-IDENTICAL header -- only the image carriers differ. The CRC
        cannot separate them, so a receiver trying candidates would verify the
        dense packet against the sparse layout and rebuild the picture at the
        wrong geometry: garbage, silently. Their magics differ for this reason.
        """
        self.assertEqual(DENSE.packet, SPARSE.packet)
        self.assertTrue(np.array_equal(DENSE.header_bins, SPARSE.header_bins))
        self.assertNotEqual(DENSE.wire_magic, SPARSE.wire_magic)

        def coders(layout):
            return {v3.profile_code(p): v3.SourceCoder(
                _fit(plane_shapes(p), layout.capacity)) for p in v3.PROFILE_CODES}
        cands = sorted([(l, coders(l)[v3.profile_code('color-dct')], coders(l))
                        for l in (SPARSE, DENSE)], key=lambda c: c[0].packet)
        # Use fitted shapes for SPARSE since color-dct (2880) doesn't fit in SPARSE (2800)
        sparse_shapes = _fit(plane_shapes('color-dct'), SPARSE.capacity)
        for layout in (SPARSE, DENSE):
            with self.subTest(sent=layout.name):
                shapes = sparse_shapes if layout is SPARSE else plane_shapes('color-dct')
                coder = v3.SourceCoder(shapes)
                values = np.random.default_rng(8).uniform(-.2, .2, coder.count)
                audio = v3.encode(values, layout, coder, 1, 1, 1,
                                  profile=v3.profile_code('color-dct'))
                rx = v3.Receiver(SPARSE, v3.SourceCoder(sparse_shapes), candidates=cands)
                out = rx.feed(np.asarray(audio, np.float32))+rx.flush()
                self.assertEqual(rx.detected, layout.name)
                self.assertLess(
                    np.sqrt(np.mean((out[0].values-values)**2)), 1e-5)


def _fit(shapes, capacity):
    if sum(int(np.prod(s)) for s in shapes) > capacity:
        return fit_shapes(shapes, capacity)
    return shapes


if __name__ == '__main__':
    unittest.main()
