"""main-style lookahead loading: a filled window of upcoming pictures."""
from concurrent.futures import Future
import unittest

from image_preparation import LookaheadPreparer


class ManualExecutor:
    def __init__(self):
        self.queue = []

    def submit(self, fn, *args):
        future = Future()
        self.queue.append((future, fn, args))
        return future

    def complete(self, index=None):
        """Run the first queued job, or the one for picture ``index``."""
        pos = 0 if index is None else next(
            i for i, (_, _, args) in enumerate(self.queue) if args[0][0] == index)
        future, fn, args = self.queue.pop(pos)
        if future.set_running_or_notify_cancel():
            try:
                future.set_result(fn(*args))
            except Exception as exc:
                future.set_exception(exc)
        return future

    def complete_all(self):
        while self.queue:
            self.complete()

    def shutdown(self, **kwargs):
        for future, _, _ in self.queue:
            future.cancel()


FOLDERS = ('main', 'float')
IPS = 30          # pictures per second
PERIOD = 1/60     # loop period


class LookaheadTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.pool = ManualExecutor()
        self.prep = LookaheadPreparer(lambda index, folders: (index, folders),
                                      clock=lambda: self.now, period=PERIOD,
                                      executor=self.pool, capacity=15)
        self.addCleanup(self.prep.close)

    @staticmethod
    def index_at(deadline):
        return int(deadline*IPS + 1e-9)

    def keys(self):
        return sorted(index for index, _ in self.prep._jobs)

    def test_keeps_the_next_fifteen_pictures_loading(self):
        self.prep.plan(FOLDERS, 0, self.index_at, now=0.0)
        self.assertEqual(self.keys(), list(range(16)))      # current + 15 ahead
        self.assertEqual(len(self.pool.queue), 16)

    def test_window_slides_and_drops_passed_pictures(self):
        self.prep.plan(FOLDERS, 0, self.index_at, now=0.0)
        self.pool.complete_all()
        self.now = 5/IPS
        self.prep.plan(FOLDERS, 5, self.index_at, now=self.now)
        self.assertEqual(self.keys(), list(range(5, 21)))
        # Only the five new pictures are requested; the rest were kept.
        self.assertEqual(self.prep.stats['requested'], 21)
        self.assertEqual(self.prep.stats['expired'], 5)

    def test_one_slow_picture_does_not_block_the_ones_after_it(self):
        self.prep.plan(FOLDERS, 0, self.index_at, now=0.0)
        for index in range(16):
            if index != 1:              # picture 1 is stuck on a slow read
                self.pool.complete(index)
        self.assertIsNone(self.prep.get(1, FOLDERS))
        for index in range(2, 16):
            self.assertEqual(self.prep.get(index, FOLDERS), (index, FOLDERS))

    def test_loop_slower_than_picture_rate_still_loads_every_picture(self):
        # Headless web loops at 15/s while pictures advance at 30/s: sampling
        # only at the loop period would load every other picture.
        prep = LookaheadPreparer(lambda index, folders: (index, folders),
                                 clock=lambda: self.now, period=1/15,
                                 executor=self.pool, capacity=15)
        self.addCleanup(prep.close)
        prep.plan(FOLDERS, 0, self.index_at, now=0.0)
        self.assertEqual(sorted(i for i, _ in prep._jobs), list(range(16)))

    def test_pingpong_turn_is_followed(self):
        frames = 10                     # 0..9, 9..0
        def pingpong(deadline):
            step = int(deadline*IPS + 1e-9) % (2*frames)
            return step if step < frames else 2*frames - 1 - step
        self.now = 7/IPS
        self.prep.plan(FOLDERS, 7, pingpong, now=self.now)
        # 7,8,9 then back down 8..0; distinct pictures only.
        self.assertEqual(self.keys(), list(range(10)))

    def test_folder_change_restarts_the_window(self):
        self.prep.plan(FOLDERS, 0, self.index_at, now=0.0)
        self.prep.plan(('other', 'float'), 0, self.index_at, now=0.0)
        self.assertTrue(all(folders == ('other', 'float')
                            for _, folders in self.prep._jobs))
        self.assertEqual(len(self.prep._jobs), 16)

    def test_held_clock_keeps_only_the_current_picture(self):
        self.prep.plan(FOLDERS, 4, None, now=0.0)
        self.assertEqual(self.keys(), [4])

    def test_seed_and_get_count_hits_and_misses(self):
        self.prep.seed(0, FOLDERS, 'first')
        self.assertEqual(self.prep.get(0, FOLDERS), 'first')
        self.assertIsNone(self.prep.get(99, FOLDERS))
        self.assertEqual((self.prep.stats['hits'], self.prep.stats['misses']), (1, 1))

    def test_upcoming_lists_ready_future_pictures_in_order(self):
        self.prep.plan(FOLDERS, 0, self.index_at, now=0.0)
        self.pool.complete(3)
        self.pool.complete(2)
        self.assertEqual([key[0] for key, _, _ in self.prep.upcoming(now=0.0)], [2, 3])

    def test_failed_picture_is_reported_and_not_retried_at_once(self):
        errors = []
        prep = LookaheadPreparer(lambda index, folders: 1/0, clock=lambda: self.now,
                                 period=PERIOD, executor=self.pool, capacity=2,
                                 on_error=lambda key, exc: errors.append(key))
        self.addCleanup(prep.close)
        prep.plan(FOLDERS, 0, self.index_at, now=0.0)
        self.pool.complete_all()
        prep.plan(FOLDERS, 0, self.index_at, now=0.0)
        self.assertEqual(len(errors), 3)
        self.assertEqual(self.pool.queue, [])

    def test_default_workers_match_main(self):
        import os
        prep = LookaheadPreparer(lambda *a: None, executor=self.pool)
        self.addCleanup(prep.close)
        self.assertEqual(prep.workers, min(8, (os.cpu_count() or 1) + 2))


if __name__ == '__main__':
    unittest.main()
