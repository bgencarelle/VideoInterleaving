"""Selection correctness under out-of-order loads, folder changes and priming."""
import threading
import unittest

from image_loader import FIFOImageBuffer
from modem_v7_display import PicturesAhead


class ImageSelectionTests(unittest.TestCase):
    @staticmethod
    def layers(value):
        return value, value, False, False

    def test_old_completion_does_not_hide_exact_picture(self):
        fifo = FIFOImageBuffer(4)
        fifo.update(1, self.layers('old'))
        fifo.update(100, self.layers('exact'))
        self.assertEqual(fifo.get(100)[1], 'exact')

    def test_folder_pairs_are_not_interchangeable(self):
        fifo = FIFOImageBuffer(4)
        fifo.update(10, self.layers('previous'), (0, 0))
        self.assertIsNone(fifo.get(10, (1, 0)))
        fifo.update(10, self.layers('current'), (1, 0))
        self.assertEqual(fifo.get(10, (1, 0))[1], 'current')

    def test_one_way_wrap_uses_modular_distance_but_exact_wins(self):
        fifo = FIFOImageBuffer(4)
        fifo.update(99, self.layers('near'))
        self.assertEqual(fifo.get(0, total_images=100)[0], 99)
        fifo.update(0, self.layers('exact'))
        self.assertEqual(fifo.get(0, total_images=100)[0], 0)
        self.assertIsNone(fifo.get(50, total_images=100))

    def test_priming_waits_for_exact_job_despite_ready_neighbor(self):
        release = threading.Event()
        started = threading.Event()
        def make(index, folders):
            if index == 2:
                started.set()
                release.wait(2)
            return index
        ahead = PicturesAhead(make, 10)
        try:
            ahead.wait_exact(1, (0, 0))
            ahead.request(2, (0, 0))
            self.assertTrue(started.wait(1))
            self.assertEqual(ahead.take(2, (0, 0)), (1, 1))
            self.assertEqual(ahead.take(2, (0, 0), wait_seconds=.001), (1, 1))
            release.set()
            self.assertEqual(ahead.wait_exact(2, (0, 0)), (2, 2))
        finally:
            release.set()
            ahead.close()

    def test_prepared_folder_change_never_returns_previous_pair(self):
        ahead = PicturesAhead(lambda i, pair: (i, pair), 20)
        try:
            ahead.wait_exact(4, (0, 0))
            index, result = ahead.take(4, (1, 0))
            self.assertEqual((index, result), (4, (4, (1, 0))))
        finally:
            ahead.close()


if __name__ == '__main__':
    unittest.main()
