"""Deterministic deadline checks: no sleeps, audio devices or timing benchmarks."""
import unittest
import numpy as np
from animation_modem.presentation import DeadlinePresentationBuffer
from animation_modem.transport2 import Decoded


class PresentationDeadlineTests(unittest.TestCase):
    def setUp(self):
        self.queue = DeadlinePresentationBuffer(1000, 100, 4)
        seed = self.result(0, age=100, complete=True, received=4)
        self.queue.put(seed, 100_000_000, now_ns=100_000_000)
        self.assertIs(self.queue.pop_due(100_000_000), seed)

    def result(self, packet=1, age=20, complete=False, received=1, duration=100, target=None):
        result = Decoded('received', values=np.zeros(4))
        result.extra.update(packet_id=packet, packet_age_samples=age,
                            packet_duration_samples=duration, complete=complete,
                            received_values=received)
        result.target_time_ns = target
        return result

    def put(self, result, ms):
        now = ms*1_000_000
        self.queue.put(result, result.target_time_ns or now, now_ns=now)

    def test_complete_beats_deadline_and_partial_cannot_reappear(self):
        self.put(self.result(), 1020)
        self.assertIsNone(self.queue.pop_due(1030_000_000))
        full = self.result(age=40, complete=True, received=4)
        self.put(full, 1040)
        self.assertIs(self.queue.pop_due(1040_000_000), full)
        self.assertIsNone(self.queue.pop_due(1060_000_000))

    def test_best_partial_appears_at_deadline_even_without_more_input(self):
        self.put(self.result(), 1020)
        best = self.result(age=40, received=2)
        self.put(best, 1040)
        self.assertEqual(len(self.queue.items), 1)
        self.assertIsNone(self.queue.pop_due(1049_999_999))
        self.assertIs(self.queue.pop_due(1050_000_000), best)
        refined = self.result(age=60, received=3)
        self.put(refined, 1060)
        self.assertIs(self.queue.pop_due(1060_000_000), refined)

    def test_shared_time_overrides_fallback_and_early_completion(self):
        target = 1100_000_000
        self.put(self.result(target=target), 1020)
        full = self.result(age=40, complete=True, received=4, target=target)
        self.put(full, 1040)
        self.assertIsNone(self.queue.pop_due(1099_999_999))
        self.assertIs(self.queue.pop_due(target), full)

    def test_previous_duration_excludes_gap_and_current_speed_change(self):
        # Last packet was 100 ms; a long gap and current 200-ms packet must
        # still give this packet 50 ms from its start, not from this update.
        partial = self.result(age=20, duration=200)
        self.put(partial, 10020)
        self.assertIsNone(self.queue.pop_due(10049_999_999))
        self.assertIs(self.queue.pop_due(10050_000_000), partial)

    def test_damaged_completion_is_still_a_partial_picture(self):
        damaged = self.result(age=20, complete=True, received=1)
        self.put(damaged, 1020)
        self.assertIsNone(self.queue.pop_due(1049_999_999))
        self.assertIs(self.queue.pop_due(1050_000_000), damaged)

    def test_new_unscheduled_packet_discards_old_future_preview(self):
        self.put(self.result(), 1020)
        full = self.result(packet=2, complete=True, received=4)
        self.put(full, 1030)
        self.assertIs(self.queue.pop_due(1030_000_000), full)
        self.assertIsNone(self.queue.pop_due(1100_000_000))

    def test_no_history_uses_current_duration_and_errors_remain_immediate(self):
        self.queue = DeadlinePresentationBuffer(1000, 100, 4)
        partial = self.result(age=20, duration=200)
        self.put(partial, 1020)
        self.assertIsNone(self.queue.pop_due(1099_999_999))
        self.assertIs(self.queue.pop_due(1100_000_000), partial)
        self.queue.put('input error', 1101_000_000)
        self.assertEqual(self.queue.pop_due(1101_000_000), 'input error')


if __name__ == '__main__':
    unittest.main()
