"""V7 mono compatibility: mono-aware slot order, the 2x2 MMSE prior side,
polarity-inverted legs and bare 1-D mono input (spec §6.5, §8.3, §9.6)."""
import unittest

import numpy as np
from PIL import Image

from animation_modem import v7

TARGET = .1521/np.sqrt(1 + 10**(v7.CLOCK_REL_DB/10))
FRAMES = 6


class V7MonoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = v7.load_model(TARGET, 'nearest')
        with Image.open(v7.REFERENCE_FIXTURE) as source:
            cls.values = v7.image_values(
                v7.prepare_image(source, encode_filter='nearest'),
                cls.model.coder.grids, encode_filter='nearest')
        cls.stereo = v7.encode_pulse_stream(cls.model, [cls.values]*FRAMES, 1, [6]*FRAMES)

    def _decode(self, audio):
        results, info = v7.decode_pulse_stream(self.model, audio)
        self.assertGreaterEqual(len(results), FRAMES-2)
        errors = [np.sqrt(np.mean((v7.values_from(self.model, r.coeffs)-self.values)**2))
                  for r in results]
        return results, info, float(np.median(errors))

    def test_slot_order_puts_every_s_slot_after_every_m_slot(self):
        head = v7.GROUPS[:v7.N_HEAD_G]
        self.assertTrue(all(stream == 'M' for _, stream, _ in head))
        streams = [stream for _, stream, _ in v7.GROUPS]
        first_s = streams.index('S')
        self.assertTrue(all(stream == 'S' for stream in streams[first_s:]))
        for stream in 'MS':
            carriers = [blk for blk, s, _ in v7.GROUPS[v7.N_HEAD_G:] if s == stream]
            self.assertEqual(carriers, sorted(carriers))
        self.assertTrue(all(stream == 'S' for _, stream, _ in v7.GROUPS[-v7.N_TAIL_G:]))
        self.assertEqual(sorted(v7.GROUPS), sorted(
            [(b, 'M', q) for b in v7.MONO for q in 'IQ'] +
            [(b, c, q) for b in v7.STEREO for c in 'MS' for q in 'IQ']))

    def test_mono_keeps_a_prefix_of_the_rank_order(self):
        """M carries ranks 0-1231 and half of the 64-rank window that straddles
        the M/S boundary, in every tail phase; nothing above it."""
        on_m = np.zeros(len(self.model.mu), bool)
        for table in self.model.rank_tables:
            for group, (_, stream, _) in enumerate(v7.GROUPS):
                if stream == 'M':
                    on_m[table[group][table[group] >= 0]] = True
        order = self.model.order
        self.assertTrue(on_m[order[:1232]].all())
        self.assertEqual(int(on_m[order[1232:1296]].sum()), 32)
        self.assertFalse(on_m[order[1296:]].any())

    def test_mono_downmix_keeps_the_important_half(self):
        mono = self.stereo.mean(axis=1, keepdims=True)
        results, _, error = self._decode(np.repeat(mono, 2, axis=1))
        self.assertTrue(all(r.status != 'lost' for r in results))
        # 0.113 with M/S interleaved, 0.091 with S ordered as if on b+6,
        # 0.082 with every S slot last.
        self.assertLess(error, .085)

    def test_unequal_tracks_decode_like_clean(self):
        """The prior belongs left of the inverse; the reversed order failed
        whenever a cell's M/S priors differ and the channel is not symmetric."""
        _, _, clean = self._decode(self.stereo)
        weak_right = self.stereo*np.float32([1, 10**(-4/20)])
        _, _, error = self._decode(weak_right)
        self.assertLess(error, clean*1.02)

    def test_polarity_inverted_leg_is_recovered(self):
        _, _, clean = self._decode(self.stereo)
        results, info, error = self._decode(self.stereo*np.float32([1, -1]))
        self.assertTrue(info.get('polarity_inverted'))
        self.assertTrue(all(r.diag.get('polarity_inverted') for r in results))
        self.assertAlmostEqual(error, clean, places=6)

    def test_normal_stream_is_not_flagged(self):
        results, info, _ = self._decode(self.stereo)
        self.assertNotIn('polarity_inverted', info)
        self.assertFalse(any(r.diag.get('polarity_inverted') for r in results))

    def test_leg_polarity_decision(self):
        frame = self.stereo[-v7.PULSE_FRAME:]
        self.assertEqual(v7.leg_polarity(frame), 1)
        self.assertEqual(v7.leg_polarity(frame*[1, -1]), -1)
        self.assertEqual(v7.leg_polarity(frame*[1, -1] + [.2, -.3]), -1)   # DC does not vote
        self.assertEqual(v7.leg_polarity(frame*[1, -.3]), -1)             # weak inverted leg
        for previous in (1, -1):
            self.assertEqual(v7.leg_polarity(0*frame, previous), previous)          # silence
            self.assertEqual(v7.leg_polarity(frame*[1, 0], previous), previous)     # dead leg
        self.assertEqual(v7.leg_polarity(frame.mean(axis=1)), 1)                   # mono

    def test_per_channel_gain_corrects_polarity_upstream(self):
        _, _, clean = self._decode(self.stereo)
        inverted = self.stereo*np.float32([1, -1])
        results, info = v7.decode_pulse_stream(self.model, inverted,
                                               input_gain=np.float32([1, -1]))
        self.assertNotIn('polarity_inverted', info)       # no retry was needed
        errors = [np.sqrt(np.mean((v7.values_from(self.model, r.coeffs)-self.values)**2))
                  for r in results]
        self.assertAlmostEqual(float(np.median(errors)), clean, places=6)
        mono = self.stereo.mean(axis=1, keepdims=True)
        results, _ = v7.decode_pulse_stream(self.model, mono, input_gain=np.float32([1, -1]))
        self.assertGreaterEqual(len(results), FRAMES-2)

    def test_bare_1d_mono_array(self):
        mono = self.stereo.mean(axis=1)
        _, _, flat = self._decode(mono)
        _, _, column = self._decode(mono[:, None])
        self.assertEqual(flat, column)


if __name__ == '__main__':
    unittest.main()
