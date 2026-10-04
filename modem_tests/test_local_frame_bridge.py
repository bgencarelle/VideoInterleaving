"""Loopback frame transport shared by local mode and the live sender."""
import socket
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

from local_frame_bridge import LatestFramePublisher, read_frame, write_frame
from tools.v7_capture import LocalModeSource


class LocalFrameBridgeTests(unittest.TestCase):
    def test_rgb_frame_round_trips_over_a_socket(self):
        left, right = socket.socketpair()
        frame = np.arange(7*5*3, dtype=np.uint8).reshape(5, 7, 3)
        try:
            write_frame(left, frame)
            received = read_frame(right)
        finally:
            left.close()
            right.close()
        np.testing.assert_array_equal(received, frame)

    def test_publisher_connects_to_loopback_receiver(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(('127.0.0.1', 0))
        listener.listen(1)
        listener.settimeout(2.0)
        publisher = LatestFramePublisher(listener.getsockname()[1])
        frame = np.full((9, 11, 3), (17, 83, 201), dtype=np.uint8)
        try:
            publisher.publish(frame)
            connection, _address = listener.accept()
            with connection:
                connection.settimeout(2.0)
                received = read_frame(connection)
        finally:
            publisher.close()
            listener.close()
        np.testing.assert_array_equal(received, frame)

    def test_mid_frame_disconnect_is_rejected(self):
        left, right = socket.socketpair()
        def send_partial():
            left.sendall(b'VIF1\x00\x00\x00\x02\x00\x00\x00\x02x')
            left.shutdown(socket.SHUT_WR)

        thread = threading.Thread(target=send_partial)
        thread.start()
        try:
            with self.assertRaisesRegex(ConnectionError, 'mid-frame'):
                read_frame(right)
        finally:
            thread.join(timeout=1.0)
            left.close()
            right.close()

    def test_local_mode_capture_launches_main_and_receives_its_frame(self):
        process = Mock()
        process.poll.return_value = None
        frame = np.full((4, 6, 3), (200, 90, 13), dtype=np.uint8)
        with patch('tools.v7_capture.subprocess.Popen', return_value=process) as popen:
            source = LocalModeSource()
            try:
                command = popen.call_args.args[0]
                self.assertEqual(command[command.index('--mode')+1], 'local')
                port = int(command[command.index('--local-frame-port')+1])
                with socket.create_connection(('127.0.0.1', port), timeout=2) as client:
                    write_frame(client, frame)
                    received = source()
            finally:
                source.close()
        np.testing.assert_array_equal(received, frame)
        process.send_signal.assert_called_once()


if __name__ == '__main__':
    unittest.main()
