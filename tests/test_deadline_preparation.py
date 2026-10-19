"""Deadline lead, useful-job ownership and exact identity under slow work."""
from concurrent.futures import Future
import unittest
from unittest.mock import patch, Mock

from image_preparation import DeadlinePreparer


class ManualExecutor:
    def __init__(self):
        self.queue = []

    def submit(self, fn, *args):
        future = Future()
        self.queue.append((future, fn, args))
        return future

    def complete(self):
        future, fn, args = self.queue.pop(0)
        if future.set_running_or_notify_cancel():
            try:
                future.set_result(fn(*args))
            except Exception as exc:
                future.set_exception(exc)
        return future

    def shutdown(self, **kwargs):
        for future, _, _ in self.queue:
            future.cancel()


class DeadlinePreparationTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.pool = ManualExecutor()
        self.deadlines = []
        def make(index, folders):
            self.now += .085
            return (index, folders)
        self.prep = DeadlinePreparer(make, clock=lambda:self.now, period=.033,
                                     executor=self.pool, capacity=4)
        self.addCleanup(self.prep.close)

    def index_at(self, deadline):
        self.deadlines.append(deadline)
        return int(deadline/.033 + .00001)

    def test_slow_completion_moves_preparation_ahead_without_moving_clock(self):
        self.prep.plan((0,0), 0, self.index_at)
        self.pool.complete()
        self.assertIsNone(self.prep.get(0,(0,0)))
        self.assertGreaterEqual(self.prep.lead, .085*1.25)
        self.deadlines.clear()
        now = self.now
        self.prep.plan((0,0), 2, self.index_at)
        self.assertGreaterEqual(min(self.deadlines)-now, .085*1.25)

    def test_later_request_cannot_cancel_a_still_useful_reserved_picture(self):
        self.prep.plan((0,0), 0, self.index_at)
        reserved = list(self.pool.queue)
        self.now = .01
        self.prep.plan((0,0), 0, self.index_at)
        self.assertTrue(all(not job[0].cancelled() for job in reserved))

    def test_expired_inflight_work_still_trains_preparation_lead(self):
        self.prep.plan((0,0), 0, self.index_at)
        future = self.pool.queue[0][0]
        future.set_running_or_notify_cancel()
        self.now = .2
        self.prep.plan((0,0), 6, self.index_at)
        future.set_result(('expired', .2))
        self.assertIsNone(self.prep.get(0,(0,0)))
        self.assertGreaterEqual(self.prep.lead, .25)

    def test_ready_storage_capacity_does_not_become_an_extra_worker_queue(self):
        self.prep.plan((0,0),0,self.index_at)
        reserved=list(self.pool.queue)
        self.now=.04
        self.prep.plan((0,0),1,self.index_at)
        self.assertEqual(len(self.pool.queue),2)
        self.assertTrue(all(not job[0].cancelled() for job in reserved))

    def test_result_survives_until_its_future_deadline_and_folder_change_invalidates(self):
        self.prep._durations.append(.2)
        self.prep.plan((0,0), 0, self.index_at)
        expected = int(self.deadlines[0]/.033 + .00001)
        self.pool.complete()
        self.prep.plan((0,0), 3, self.index_at)
        self.assertEqual(self.prep.get(expected,(0,0)), (expected,(0,0)))
        self.assertIsNone(self.prep.get(expected+1,(0,0)))
        self.prep.plan((1,0), expected, self.index_at)
        self.assertIsNone(self.prep.get(expected,(0,0)))

    def test_held_midi_index_is_retried_without_guessing_future_events(self):
        self.prep.plan((0,0), 12)
        self.pool.complete()
        self.assertEqual(self.prep.get(12,(0,0)), (12,(0,0)))
        self.assertIsNone(self.prep.get(13,(0,0)))
        self.prep.plan((0,0), 14)
        self.assertIsNone(self.prep.get(12,(0,0)))

    def test_seeded_startup_is_available_without_reloading_held_clock(self):
        self.prep.seed(0,(0,0),'startup')
        self.prep.plan((0,0),0)
        self.assertEqual(self.prep.get(0,(0,0)),'startup')
        self.assertEqual(len(self.pool.queue),0)

    def test_failed_source_is_reported_and_retry_is_bounded(self):
        errors=[]
        def fail(*args):
            raise ValueError('bad source')
        self.prep.make=fail
        self.prep.on_error=lambda key,exc:errors.append((key,str(exc)))
        self.prep.plan((0,0),12)
        self.pool.complete()
        self.assertIsNone(self.prep.get(12,(0,0)))
        self.prep.plan((0,0),12)
        self.assertEqual(len(errors),1)
        self.assertEqual(len(self.pool.queue),0)
        self.now=1.1
        self.prep.plan((0,0),12)
        self.assertEqual(len(self.pool.queue),1)


class TextureStagingTests(unittest.TestCase):
    def test_visible_slot_is_not_overwritten_and_shared_layer_skips_upload(self):
        import renderer
        a,b,c,blank = object(),object(),object(),object()
        with patch.object(renderer,'create_texture',side_effect=lambda image:Mock()) as create, \
                patch.object(renderer,'update_texture',side_effect=lambda texture,image:texture) as update:
            stage=renderer.TexturePairStage()
            visible=stage.select('a',(a,blank))
            prepared=stage.stage('b',(b,blank))
            self.assertEqual(stage.active,'a')
            self.assertIsNot(visible[0],prepared[0])
            self.assertEqual(stage.select('b',(b,blank)),prepared)
            stage.stage('c',(c,blank))
            self.assertEqual(len(stage.slots),2)
            self.assertEqual(create.call_count,4)
            self.assertEqual(update.call_count,1)
            self.assertEqual(stage.uploads,5)
            stage.close()


class LocalPacingTests(unittest.TestCase):
    def test_blocking_swap_does_not_add_another_frame_wait(self):
        from image_preparation import advance_tick
        period=1/60
        deadline,wait=advance_tick(0.0,period,period+.002,swap_consumed=True)
        self.assertAlmostEqual(deadline,period)
        self.assertEqual(wait,0)
        deadline,wait=advance_tick(0.0,period,.071,swap_consumed=True)
        self.assertAlmostEqual(deadline,4*period)
        self.assertEqual(wait,0)

    @staticmethod
    def _iterations_per_second(work, period, *, swap_consumed, seconds=2.0):
        """Simulate the image loop: do `work` seconds, then pace."""
        from image_preparation import advance_tick
        now = deadline = 0.0
        count = 0
        while now < seconds:
            now += work
            deadline, wait = advance_tick(deadline, period, now,
                                          swap_consumed=swap_consumed)
            now += wait
            count += 1
        return count / seconds

    def test_slightly_late_window_loop_keeps_its_rate(self):
        # A 60 fps loop whose iterations take 19 ms (Cherry Trail tablet,
        # where the refresh wait lands in the texture upload). Holding for
        # the next tick after each overrun halves it to the 30 fps the tablet
        # measured; presenting at once keeps it near main's ~52.
        period = 1/60
        held = self._iterations_per_second(.019, period, swap_consumed=False)
        visible = self._iterations_per_second(.019, period, swap_consumed=True)
        self.assertAlmostEqual(held, 30, delta=1)
        self.assertGreater(visible, 50)

    def test_regular_pacing_sleeps_and_overload_skips_to_future_without_swap(self):
        from image_preparation import advance_tick
        period=1/60
        deadline,wait=advance_tick(0.0,period,.005)
        self.assertAlmostEqual(deadline,period)
        self.assertAlmostEqual(wait,period-.005)
        deadline,wait=advance_tick(0.0,period,period+.002)
        self.assertAlmostEqual(deadline,2*period)
        self.assertGreater(wait,0)
