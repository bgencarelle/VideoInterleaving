"""V7 loop metadata: tail slice, rotating CRC field (p, N), tail store and the
epoch-free loop clock that lets any receiver measure how late its picture is."""
import unittest

import numpy as np
from PIL import Image

from animation_modem import v7

TARGET = .1521/np.sqrt(1 + 10**(v7.CLOCK_REL_DB/10))
BIRTH_NS = 280152660*10**9          # 1978-11-17 07:11 America/New_York


class V7LoopClockTests(unittest.TestCase):
    def test_matches_the_application_clock(self):
        """loop_index(ticks, p, N) is index_calculator's free clock with the
        epoch reduced to p -- exact for an epoch on a whole second."""
        import index_calculator
        rng = np.random.default_rng(7)
        for frames in (2221, 1, 2, 500):
            for pingpong in (True, False):
                loop = v7.LoopInfo.from_epoch(frames, BIRTH_NS, pingpong)
                saved = index_calculator.launch_time
                index_calculator.launch_time = BIRTH_NS
                try:
                    for t in rng.integers(1.7e18, 1.9e18, 200):
                        expected, _ = index_calculator.calculate_free_clock_index(
                            frames, pingpong, at_time_ns=int(t), publish=False)
                        self.assertEqual(v7.loop_index(v7.loop_ticks(int(t)), loop),
                                         expected)
                finally:
                    index_calculator.launch_time = saved

    def test_lag_is_recovered_across_the_ping_pong_turns(self):
        loop = v7.LoopInfo.from_epoch(2221, BIRTH_NS)
        period = v7.loop_period(loop.frames)
        base = v7.loop_ticks(1_800_000_000*10**9)
        for offset in range(0, period, 97):
            sent_at = base + offset
            index = v7.loop_index(sent_at, loop)
            for lag in (-5, 0, 1, 3, 40, 900):
                with self.subTest(offset=offset, lag=lag):
                    # With the recent lag as the expectation (as the receiver
                    # passes it) every lag is recovered, through the turns.
                    got = v7.loop_lag_ticks(index, sent_at + lag, loop,
                                            expected=lag + 2)
                    self.assertEqual(got, lag)
            # With no history, small lags are still right away from a turn.
            turn = min(offset % loop.frames, loop.frames - 1 - offset % loop.frames)
            if turn > 10:
                self.assertEqual(v7.loop_lag_ticks(index, sent_at + 3, loop), 3)

    def test_direction_rides_in_the_index_field(self):
        for direction in (1, -1):
            raw = v7.metadata_word(3, 2, 4, source_index=12345, direction=direction)
            meta = v7.parse_metadata_word(raw)
            with self.subTest(direction=direction):
                self.assertEqual(meta.source_index, 12345)
                self.assertEqual(meta.direction, direction)
        with self.assertRaises(ValueError):
            v7.metadata_word(0, source_index=v7.MAX_SOURCE_INDEX + 1)

    def test_direction_removes_the_turn_ambiguity(self):
        loop = v7.LoopInfo.from_epoch(2221, BIRTH_NS)
        period = v7.loop_period(loop.frames)
        base = v7.loop_ticks(1_800_000_000*10**9)
        turn = base + (loop.frames - 3 - (base - loop.phase) % period)   # just before a turn
        for lag in (1, 5, 40, 200):
            for step in (0, 6):            # before and after the turn
                sent_at = turn + step
                index = v7.loop_index(sent_at, loop)
                way = v7.loop_direction(sent_at, loop)
                with self.subTest(lag=lag, step=step):
                    self.assertEqual(
                        v7.loop_lag_ticks(index, sent_at + lag, loop, direction=way),
                        lag)

    def test_loop_fields(self):
        loop = v7.LoopInfo(2221, 418, True)
        self.assertEqual(v7.LoopInfo.from_fields(loop.frames_field, loop.phase_field), loop)
        one_way = v7.LoopInfo(300, 17, False)
        self.assertEqual(one_way.frames_field, 300 | v7.LOOP_ONE_WAY)
        self.assertEqual(v7.LoopInfo.from_fields(one_way.frames_field, 17), one_way)
        self.assertFalse(v7.LoopInfo(2221, v7.LOOP_NO_CLOCK).clocked)
        self.assertTrue(loop.clocked)


class V7LoopLockTests(unittest.TestCase):
    def _meta(self, tail_slice, mask):
        return v7.Metadata(0, 0, tail_slice, 10, mask)

    def test_plain_slices_need_a_zero_mask(self):
        lock = v7.LoopLock()
        self.assertTrue(lock.check(self._meta(0, 0)))
        self.assertFalse(lock.check(self._meta(3, 5)))

    def test_values_lock_after_repeating_and_are_then_checked(self):
        lock = v7.LoopLock()
        self.assertIsNone(lock.check(self._meta(v7.TAIL_SLICE_P, 418)))
        self.assertIsNone(lock.check(self._meta(v7.TAIL_SLICE_N, 2221)))
        self.assertTrue(lock.check(self._meta(v7.TAIL_SLICE_P, 418)))
        self.assertTrue(lock.check(self._meta(v7.TAIL_SLICE_N, 2221)))
        self.assertEqual(lock.loop, v7.LoopInfo(2221, 418, True))
        self.assertFalse(lock.check(self._meta(v7.TAIL_SLICE_P, 9999)))   # damage
        self.assertTrue(lock.check(self._meta(v7.TAIL_SLICE_P, 418)))
        # A new sender run: the new value, twice in a row, replaces the old.
        self.assertFalse(lock.check(self._meta(v7.TAIL_SLICE_P, 100)))
        self.assertTrue(lock.check(self._meta(v7.TAIL_SLICE_P, 100)))
        self.assertEqual(lock.loop.phase, 100)

    def test_a_damaged_value_does_not_lock(self):
        lock = v7.LoopLock()
        for mask in (1234, 418, 777, 418):
            lock.check(self._meta(v7.TAIL_SLICE_P, mask))
        self.assertIsNone(lock.values[v7.TAIL_SLICE_P])


class V7LoopStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = v7.load_model(TARGET, 'nearest')
        with Image.open(v7.REFERENCE_FIXTURE) as source:
            cls.values = v7.image_values(v7.prepare_image(source, encode_filter='nearest'),
                                         cls.model.coder.grids, encode_filter='nearest')
        cls.coeffs = cls.model.coder.forward(cls.values)/cls.model.coder.gains
        rank = np.empty(len(cls.coeffs), int)
        rank[cls.model.order] = np.arange(len(cls.coeffs))
        cls.tail = rank >= v7.BODY_END
        cls.loop = v7.LoopInfo.from_epoch(2221, BIRTH_NS)

    def _stream(self, frames, start_index=100, loop=None):
        return v7.encode_pulse_stream(self.model, [self.values]*frames, 1, [6]*frames,
                                      source_indices=list(range(start_index,
                                                                start_index+frames)),
                                      loop=self.loop if loop is None else loop)

    def _tail_error(self, result):
        return np.sqrt(np.mean((result.coeffs[self.tail] - self.coeffs[self.tail])**2))

    def test_direction_survives_the_wire(self):
        ways = [1 if i % 2 else -1 for i in range(8)]
        audio = v7.encode_pulse_stream(self.model, [self.values]*8, 1, [6]*8,
                                       source_indices=list(range(100, 108)),
                                       loop=self.loop, directions=ways)
        results, _ = v7.decode_pulse_stream(self.model, audio)
        self.assertEqual([r.diag['direction'] for r in results], ways[:len(results)])

    def test_every_packet_is_accepted_and_the_loop_is_learned(self):
        results, _ = v7.decode_pulse_stream(self.model, self._stream(16))
        self.assertEqual([r.diag['source_index'] for r in results], list(range(100, 115)))
        self.assertEqual([r.diag['tail_slice'] for r in results],
                         [c % v7.TAIL_PHASES for c in range(1, 16)])
        self.assertTrue(all(r.status == 'received' for r in results))
        self.assertEqual(results[-1].diag['loop'], self.loop)
        # After the lock nothing is provisional any more.
        self.assertFalse(any(r.diag['metadata_provisional'] for r in results[-3:]))

    def test_tail_store_fills_the_tail(self):
        results, _ = v7.decode_pulse_stream(self.model, self._stream(12))
        at_mean = np.sqrt(np.mean((self.model.mu[self.tail] - self.coeffs[self.tail])**2))
        self.assertLess(self._tail_error(results[-1]), .1*at_mean)
        state = v7.PulseState(tail_memory=False)
        results, _ = v7.decode_pulse_stream(self.model, self._stream(12), state=state)
        self.assertGreater(self._tail_error(results[-1]), .5*at_mean)

    def test_tail_values_expire_after_one_rotation(self):
        store = v7.TailStore()
        store.update(self.model, self.coeffs, 3)
        idx = self.model.rank_tables[3]
        slice3 = np.intersect1d(idx[idx >= 0], np.flatnonzero(self.tail))
        others = [s for s in range(v7.TAIL_PHASES) if s != 3]
        for tail_slice in others:                      # six more packets
            np.testing.assert_array_equal(store.prior(self.model)[slice3],
                                          self.coeffs[slice3])
            store.update(self.model, self.model.mu, tail_slice)
        store.update(self.model, self.model.mu, others[0])    # a seventh: expired
        np.testing.assert_array_equal(store.prior(self.model)[slice3],
                                      self.model.mu[slice3])

    def test_live_lag_from_received_loop(self):
        """A picture chosen for time t and shown 100 ms later reads ~3 ticks."""
        state = v7.PulseState()
        results, _ = v7.decode_pulse_stream(self.model, self._stream(16), state=state)
        loop = results[-1].diag['loop']
        t = 1_800_000_000*10**9
        index = v7.loop_index(v7.loop_ticks(t), loop)
        lags = [v7.loop_lag_ticks(index, v7.loop_ticks(t + 100_000_000 + k), loop)
                for k in range(0, 33_000_000, 1_000_000)]
        self.assertAlmostEqual(np.mean(lags)*1000/v7.LOOP_IPS, 100, delta=34)

    def test_no_loop_information(self):
        empty = v7.LoopInfo(0, 0)
        results, _ = v7.decode_pulse_stream(self.model, self._stream(16, loop=empty))
        self.assertTrue(all(r.status == 'received' for r in results))
        self.assertFalse(results[-1].diag['loop'].clocked)


if __name__ == '__main__':
    unittest.main()
