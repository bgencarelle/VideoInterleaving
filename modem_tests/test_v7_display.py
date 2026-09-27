import unittest

from tools.v7_display import LatestFrame


class LatestFrameTests(unittest.TestCase):
    def test_snapshot_returns_newest_frame_and_notifies_viewer(self):
        frames = LatestFrame()
        notifications = []
        frames.set_notifier(lambda: notifications.append(frames.snapshot()))

        frames.publish('first', 'shapes', 1)
        frames.publish('newest', 'shapes', 2)

        self.assertEqual(len(notifications), 2)
        self.assertEqual(notifications[-1].values, 'newest')
        snapshot = frames.snapshot()
        self.assertEqual(snapshot.generation, 2)
        self.assertEqual(snapshot.aspect, 2)
        self.assertGreater(snapshot.published_at, 0)

    def test_viewer_shutdown_callback_cannot_break_frame_publication(self):
        frames = LatestFrame()
        frames.set_notifier(lambda: (_ for _ in ()).throw(RuntimeError('closed')))

        frames.publish('frame', 'shapes')

        self.assertEqual(frames.snapshot().values, 'frame')


if __name__ == '__main__':
    unittest.main()
