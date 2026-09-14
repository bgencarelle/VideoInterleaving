"""Carry image on the carriers the header symbols leave idle.

The header occupies header_width carriers. The rest of every header symbol was
silence: on lean-v3 that is 30 of 50 data carriers across all four header
symbols, 480 slots, 2200 -> 2680. Same frame, same band, same frame rate, and
free in level terms because the preamble holds the packet peak either way.

What it is NOT is a straight resolution win. Capacity is not resolution: the
live profile already fits in 2200, so it gains nothing, and spending the slots
on a bigger plane gains only a little, because a fixed power budget spread over
more coefficients leaves less power on each.
"""
import unittest

import numpy as np

from animation_modem import core, transport3 as v3
from animation_modem.imaging import plane_shapes

LEAN = v3.ALL_PRESETS['lean-v3']
DENSE = v3.ALL_PRESETS['lean-v3-dense']


def round_trip(layout, shapes, seed=4, frames=2):
    coder = v3.SourceCoder(shapes)
    values = np.random.default_rng(seed).uniform(-.2, .2, coder.count)
    audio = np.concatenate([v3.encode(values, layout, coder, n, n, frames,
                                      profile=v3.profile_code('color-lean'))
                            for n in range(1, frames+1)])
    rx = v3.Receiver(layout, coder)
    out = rx.feed(np.asarray(audio, np.float32))+rx.flush()
    return out, values, np.asarray(audio)


class DenseHeaderTests(unittest.TestCase):
    def test_the_idle_carriers_are_real_and_counted(self):
        self.assertEqual(len(LEAN.spare_bins),
                         len(LEAN.data_bins)-len(LEAN.header_bins))
        self.assertEqual(LEAN.header_capacity, 0)
        self.assertEqual(DENSE.header_capacity,
                         DENSE.header_symbols*len(DENSE.spare_bins)*4)
        self.assertEqual(DENSE.capacity, LEAN.capacity+480)

    def test_nothing_about_the_frame_changes(self):
        """The whole point: no extra samples, no extra bandwidth, no fewer
        frames per second."""
        self.assertEqual(DENSE.frame, LEAN.frame)
        self.assertEqual(DENSE.packet, LEAN.packet)
        self.assertEqual(DENSE.top_bin, LEAN.top_bin)
        self.assertEqual(DENSE.fps, LEAN.fps)

    def test_the_level_is_untouched(self):
        """Filling the header symbols raises no peak, because the preamble
        holds it. If it did, everything would renormalise down and the extra
        capacity would have been bought with SNR."""
        shapes = plane_shapes('color-lean')
        _, _, lean = round_trip(LEAN, shapes)
        _, _, dense = round_trip(DENSE, shapes)
        self.assertAlmostEqual(float(np.max(np.abs(dense))),
                               float(np.max(np.abs(lean))), places=6)

    def test_the_reclaimed_slots_round_trip(self):
        """Sized to need them: 2628 coefficients do not fit in lean-v3."""
        shapes = [(54, 45), (11, 9), (11, 9)]
        self.assertGreater(sum(h*w for h, w in shapes), LEAN.capacity)
        self.assertLessEqual(sum(h*w for h, w in shapes), DENSE.capacity)
        out, values, _ = round_trip(DENSE, shapes)
        self.assertEqual([r.absolute for r in out], [1, 2])
        for r in out:
            self.assertEqual(r.identity, 'verified_header')
            self.assertLess(np.sqrt(np.mean((r.values-values)**2)), 1e-5)

    def test_dense_and_sparse_are_not_confusable(self):
        """The failure this nearly shipped with.

        lean-v3 and lean-v3-dense have the same packet length, the same
        top_bin, and a BIT-IDENTICAL header -- only the image carriers differ.
        The CRC cannot separate them, so a receiver trying candidates verified
        a dense packet against the sparse layout and rebuilt the picture at the
        wrong geometry: garbage, silently. Their magics differ for this reason.
        """
        self.assertEqual(DENSE.packet, LEAN.packet)
        self.assertTrue(np.array_equal(DENSE.header_bins, LEAN.header_bins))
        self.assertNotEqual(DENSE.wire_magic, LEAN.wire_magic)

        def coders(layout):
            return {v3.profile_code(p): v3.SourceCoder(
                _fit(plane_shapes(p), layout.capacity)) for p in v3.PROFILE_CODES}
        cands = sorted([(l, coders(l)[v3.profile_code('color-lean')], coders(l))
                        for l in (LEAN, DENSE)], key=lambda c: c[0].packet)
        for layout in (LEAN, DENSE):
            with self.subTest(sent=layout.name):
                shapes = plane_shapes('color-lean')
                coder = v3.SourceCoder(shapes)
                values = np.random.default_rng(8).uniform(-.2, .2, coder.count)
                audio = v3.encode(values, layout, coder, 1, 1, 1,
                                  profile=v3.profile_code('color-lean'))
                rx = v3.Receiver(LEAN, v3.SourceCoder(shapes), candidates=cands)
                out = rx.feed(np.asarray(audio, np.float32))+rx.flush()
                self.assertEqual(rx.detected, layout.name)
                self.assertLess(
                    np.sqrt(np.mean((out[0].values-values)**2)), 1e-5)

    def test_the_slot_order_is_unchanged_for_every_existing_preset(self):
        """Reclaiming slots meant rewriting the mapping that decides which
        coefficient rides which carrier. It must be byte-identical wherever
        dense_header is off, or every existing preset silently degrades."""
        for name, layout in v3.ALL_PRESETS.items():
            if layout.dense_header:
                continue
            for profile in v3.PROFILE_CODES:
                with self.subTest(preset=name, profile=profile):
                    shapes = _fit(plane_shapes(profile), layout.capacity)
                    got = core.coefficient_slots(layout, tuple(shapes))
                    self.assertTrue(np.array_equal(got, _legacy(layout, shapes)))


def _fit(shapes, capacity):
    from animation_modem.imaging import fit_shapes
    if sum(int(np.prod(s)) for s in shapes) > capacity:
        return fit_shapes(shapes, capacity)
    return shapes


def _legacy(layout, shapes):
    """The mapping exactly as it was before the two-region rewrite."""
    count = sum(h*w for h, w in shapes)
    if not layout.progressive:
        return np.arange(count)
    ranks = []
    for h, w in shapes:
        yy, xx = np.mgrid[:h, :w]
        ranks.extend(np.hypot(yy/h, xx/w).ravel())
    order = np.argsort(ranks, kind='stable')
    slots = np.arange(layout.capacity).reshape(
        layout.image_symbols, len(layout.data_bins), 2, 2)
    if layout.spread_carriers:
        lanes = len(layout.data_bins)
        middle = np.argsort(np.abs(np.arange(lanes)-(lanes-1)/2), kind='stable')
        low_to_high = slots[:, middle].transpose(0, 2, 3, 1).ravel()[:count]
    else:
        low_to_high = slots.transpose(1, 0, 2, 3).ravel()[:count]
    mapping = np.empty(count, int)
    mapping[order] = low_to_high
    return mapping


if __name__ == '__main__':
    unittest.main()
