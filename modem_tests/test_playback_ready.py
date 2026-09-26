import threading
import unittest
from types import SimpleNamespace

import numpy as np

from animation_modem.playback import PacketOutput


class PacketOutputReadinessTests(unittest.TestCase):
    def test_callback_wakes_waiter_when_it_takes_the_pending_packet(self):
        class ObservedEvent(threading.Event):
            def __init__(self):
                super().__init__()
                self.entered_wait = threading.Event()

            def wait(self, timeout=None):
                self.entered_wait.set()
                return super().wait(timeout)

        output = PacketOutput.__new__(PacketOutput)
        output.error = None
        output.started = False
        output.done = threading.Event()
        output.ready_event = ObservedEvent()
        output.ready_event.clear()
        output.callback_progress = threading.Event()
        output.lock = threading.Lock()
        output.pending = np.ones((4, 2), dtype=np.float32)
        output.pending_start = None
        output.current = None
        output.position = 0
        output.finishing = False
        output.scheduled = False
        output.underflows = 0
        output.starvations = 0
        output.completed = 0

        result = []
        waiter = threading.Thread(target=lambda: result.append(output.wait_ready()),
                                  daemon=True)
        waiter.start()
        self.assertTrue(output.ready_event.entered_wait.wait(timeout=1))
        self.assertTrue(waiter.is_alive())

        output._callback(
            np.zeros((2, 2), dtype=np.float32), 2,
            SimpleNamespace(), SimpleNamespace(output_underflow=False))

        waiter.join(timeout=1)
        self.assertFalse(waiter.is_alive())
        self.assertEqual(result, [True])
        self.assertIsNotNone(output.current)
        self.assertIsNone(output.pending)


if __name__ == '__main__':
    unittest.main()
