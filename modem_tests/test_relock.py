"""A running receiver has to survive the transmitter changing.

Preset detection is paid once per lock -- 1.4 ms an attempt, thirteen attempts
-- and then short-circuited, which is right while one transmission runs. It was
also permanent: `detected` was set and never cleared, so when a SECOND
transmission started on a different preset the receiver kept decoding it
against the first one and produced nothing at all. The only cure was stopping
and restarting the receiver, which is exactly how it was found.
"""
import unittest

import numpy as np

from animation_modem import transport3 as v3
from animation_modem.imaging import fit_shapes, plane_shapes


def coder_for(profile, layout):
    shapes = plane_shapes(profile)
    if sum(int(np.prod(s)) for s in shapes) > layout.capacity:
        shapes = fit_shapes(shapes, layout.capacity)
    return v3.SourceCoder(shapes)


def stream(preset, frames=6, seed=6, profile='color-lean'):
    layout = v3.ALL_PRESETS[preset]
    coder = coder_for(profile, layout)
    values = np.random.default_rng(seed).uniform(-.2, .2, coder.count)
    audio = np.concatenate([
        v3.encode(values, layout, coder, n, n, frames,
                  profile=v3.profile_code(profile))
        for n in range(1, frames+1)])
    return layout, coder, values, audio


def candidates(names=('lean-v3', 'mid-14k', 'tape-v3', 'wide-v3')):
    out = []
    for name in names:
        layout = v3.ALL_PRESETS[name]
        coder = coder_for('color-lean', layout)
        out.append((layout, coder, None))
    return sorted(out, key=lambda c: c[0].packet)


def run(receiver, signal, block=256):
    out = []
    signal = np.asarray(signal, np.float32)
    for i in range(0, len(signal), block):
        out += receiver.feed(signal[i:i+block])
    return out


def verified(results):
    return [r for r in results if r.identity == 'verified_header']


class RelockTests(unittest.TestCase):
    def receiver(self):
        start = v3.ALL_PRESETS['lean-v3']
        return v3.Receiver(start, coder_for('color-lean', start),
                           candidates=candidates())

    def test_a_second_transmission_on_another_preset_is_found(self):
        """The regression, as the sequence that produced it: play one preset,
        stop, start another, never touch the receiver."""
        _, _, _, first = stream('lean-v3')
        _, _, _, second = stream('mid-14k')
        gap = np.zeros((8000, 2))
        rx = self.receiver()
        got_first = verified(run(rx, np.concatenate([first, gap])))
        self.assertGreaterEqual(len(got_first), 4)
        self.assertEqual(rx.detected, 'lean-v3')

        got_second = verified(run(rx, np.concatenate([second, gap])))
        rx.flush()
        self.assertGreaterEqual(len(got_second), 3,
                                'the receiver stayed locked to the old preset')
        self.assertEqual(rx.detected, 'mid-14k')

    def test_it_comes_back_the_other_way_too(self):
        """Not one-directional: the wide preset has the longer packet, so the
        candidate search reaches it on a different path."""
        _, _, _, first = stream('mid-14k')
        _, _, _, second = stream('wide-v3')
        gap = np.zeros((8000, 2))
        rx = self.receiver()
        run(rx, np.concatenate([first, gap]))
        self.assertEqual(rx.detected, 'mid-14k')
        got = verified(run(rx, np.concatenate([second, gap])))
        self.assertGreaterEqual(len(got), 3)
        self.assertEqual(rx.detected, 'wide-v3')

    def test_three_transmissions_in_a_row(self):
        gap = np.zeros((8000, 2))
        rx = self.receiver()
        for name in ('tape-v3', 'lean-v3', 'mid-14k'):
            _, _, _, audio = stream(name)
            got = verified(run(rx, np.concatenate([audio, gap])))
            self.assertGreaterEqual(len(got), 3, f'{name} was not picked up')
            self.assertEqual(rx.detected, name)

    def test_the_same_preset_twice_does_not_thrash(self):
        """Relocking must not undo the thing detection exists for. A second
        run of the SAME preset keeps the lock, so the search is still paid
        once rather than per transmission."""
        _, _, _, audio = stream('mid-14k')
        gap = np.zeros((8000, 2))
        rx = self.receiver()
        run(rx, np.concatenate([audio, gap]))
        self.assertEqual(rx.detected, 'mid-14k')
        got = verified(run(rx, np.concatenate([audio, gap])))
        self.assertGreaterEqual(len(got), 5)
        self.assertEqual(rx.detected, 'mid-14k')

    def test_a_lock_survives_isolated_bad_packets(self):
        """RELOCK_AFTER is not 1 on purpose. A single unverified header is
        ordinary on a marginal channel, and giving the lock up on one would
        re-run the candidate search constantly and risk adopting a wrong
        layout off a lucky CRC."""
        layout, coder, _, audio = stream('mid-14k', frames=8)
        rx = self.receiver()
        run(rx, audio[:layout.packet*2])
        self.assertEqual(rx.detected, 'mid-14k')
        rubbish = np.random.default_rng(3).normal(0, .02, (layout.packet, 2))
        run(rx, rubbish)
        self.assertEqual(rx.detected, 'mid-14k', 'gave up the lock too early')

    def test_a_gap_alone_does_not_drop_the_lock(self):
        """Silence between frames is normal -- the sender is not obliged to be
        contiguous -- and must not by itself force a re-search."""
        _, _, _, audio = stream('mid-14k', frames=3)
        rx = self.receiver()
        run(rx, audio)
        self.assertEqual(rx.detected, 'mid-14k')
        run(rx, np.zeros((40000, 2)))
        self.assertEqual(rx.detected, 'mid-14k')


if __name__ == '__main__':
    unittest.main()
