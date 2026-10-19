"""The paced producer must keep broadcasting beyond its first frame."""
from collections import deque
from unittest import TestCase, mock
import gzip
import json
import threading
from types import SimpleNamespace

import ascii_web_server


def viewer():
    client = ascii_web_server.AsciiWebSocket(None, mock.Mock(), ('test', 1234))
    client.frame_encoding = 'text'
    return client


class BroadcastTests(TestCase):
    def test_websocket_handoff_does_not_use_the_100ms_library_poll_default(self):
        config = mock.Mock()
        config.get_ascii_websocket_port.return_value = 2424
        with mock.patch.object(ascii_web_server, 'get_config', return_value=config), \
                mock.patch.object(ascii_web_server.threading, 'Thread'), \
                mock.patch.object(ascii_web_server, 'LatestWebSocketServer') as server:
            ascii_web_server.start_server()
        self.assertEqual(server.call_args.kwargs['selectInterval'], .01)

    def test_multiple_frames_and_bounded_slow_client_backlog(self):
        client = viewer()
        client.sendMessage('important control')
        stop = RuntimeError('finished test input')
        with mock.patch.object(ascii_web_server, 'clients', [client]), \
                mock.patch.object(ascii_web_server.exchange_ascii, 'get_frame',
                                  side_effect=['one', b'two', stop]):
            with self.assertRaisesRegex(RuntimeError, 'finished test input'):
                ascii_web_server.broadcast_loop()
        # A viewer keeps no picture older than the newest one it is handed.
        self.assertEqual(client._pending_picture, ('\033[Htwo', None))
        self.assertEqual(list(client._controls), ['important control'])
        self.assertEqual(list(client.sendq), [])

    def test_shared_gzip_is_encoded_once_per_frame_not_per_client(self):
        plain = mock.Mock(frame_encoding='text')
        compressed = [mock.Mock(frame_encoding='gzip') for _ in range(3)]
        stop = RuntimeError('finished test input')
        original = gzip.compress
        with mock.patch.object(ascii_web_server, 'clients', [plain] + compressed), \
                mock.patch.object(ascii_web_server.exchange_ascii, 'get_frame',
                                  side_effect=['one', b'two', stop]), \
                mock.patch.object(ascii_web_server.gzip, 'compress', wraps=original) as encode:
            with self.assertRaisesRegex(RuntimeError, 'finished test input'):
                ascii_web_server.broadcast_loop()
        self.assertEqual(encode.call_count, 2)
        self.assertEqual([call.args[0] for call in plain.queue_picture.call_args_list],
                         ['\x1b[Hone', '\x1b[Htwo'])
        for index, text in enumerate(('one', 'two')):
            messages = [client.queue_picture.call_args_list[index].args[1] for client in compressed]
            self.assertTrue(all(message is messages[0] for message in messages))
            self.assertEqual(messages[0][:1], b'g')
            self.assertEqual(gzip.decompress(messages[0][1:]).decode(), '\x1b[H' + text)

    def test_zoom_messages_do_not_change_global_dimensions(self):
        client = SimpleNamespace(data=json.dumps({'type':'resize', 'cols':15, 'rows':10}),
                                 frame_encoding='text')
        import settings
        before = (settings.ASCII_WIDTH, settings.ASCII_HEIGHT)
        ascii_web_server.AsciiWebSocket.handleMessage(client)
        self.assertEqual((settings.ASCII_WIDTH, settings.ASCII_HEIGHT), before)
        client.data = json.dumps({'type':'encoding', 'encoding':'gzip'})
        ascii_web_server.AsciiWebSocket.handleMessage(client)
        self.assertEqual(client.frame_encoding, 'gzip')
        for data in ('[]', 'null', 'not json', 'x'*4097, b'bytes'):
            client.data = data
            ascii_web_server.AsciiWebSocket.handleMessage(client)

    def test_new_viewer_receives_master_dimensions_not_its_default_grid(self):
        import settings
        client = viewer()
        with mock.patch.object(ascii_web_server, 'clients', []), \
                mock.patch.object(settings, 'ASCII_WEB_SOURCE_SIZE', (120,80)):
            ascii_web_server.AsciiWebSocket.handleConnected(client)
            messages = list(client._controls)
            self.assertEqual(json.loads(messages[1]), {'type':'resize', 'cols':120, 'rows':80})
            self.assertEqual(client.frame_encoding, 'text')
            ascii_web_server.AsciiWebSocket.handleClose(client)


class LatestOnlyQueueTests(TestCase):
    def test_producers_only_replace_mailbox_and_never_mutate_partial_send(self):
        client = viewer()
        client.sendq.append((1, b'partial'))
        def produce():
            client.sendMessage('resize control')
            for i in range(100):
                client.queue_picture(str(i))
        thread = threading.Thread(target=produce)
        thread.start(); thread.join()
        self.assertEqual(list(client.sendq), [(1, b'partial')])
        self.assertEqual(client._pending_picture, ('99', None))
        client.flush_delivery()
        self.assertEqual(client.sendq[0], (1, b'partial'))
        self.assertEqual(len(client.sendq), 2)  # partial and preserved control
        self.assertEqual(client._pending_picture, ('99', None))

    def test_controls_precede_newest_picture_without_losing_initial_dimensions(self):
        client = viewer()
        client.sendMessage('clear')
        client.sendMessage('dimensions')
        client.queue_picture('old')
        client.queue_picture('newest')
        with mock.patch.object(ascii_web_server.WebSocket, 'sendMessage') as send:
            client.flush_delivery()
        self.assertEqual([call.args[0] for call in send.call_args_list],
                         ['clear', 'dimensions', 'newest'])

    def test_flow_waits_for_matching_ack_and_only_sends_latest_after_pause(self):
        client = viewer()
        def control(**message):
            client.data = json.dumps(message)
            client.handleMessage()
        control(type='flow')
        with mock.patch.object(ascii_web_server.WebSocket, 'sendMessage') as send:
            client.queue_picture('first')
            client.flush_delivery()
            self.assertEqual(client._awaiting_ack, 1)
            for i in range(90):
                client.queue_picture(str(i))
                client.flush_delivery()
            control(type='flow')  # repeated negotiation is not an acknowledgement
            control(type='ack', id=999)
            control(type='ack', id=True)
            client.flush_delivery()
            self.assertEqual(send.call_count, 2)  # frame ID, picture
            control(type='ack', id=1)
            client.flush_delivery()
            self.assertEqual(send.call_args.args[0], '89')
            self.assertEqual(client._awaiting_ack, 2)
            control(type='ack', id=1)  # a duplicate cannot release the next frame
            self.assertEqual(client._awaiting_ack, 2)

    def test_encoding_is_selected_when_the_socket_thread_admits_the_picture(self):
        client = viewer()
        client.queue_picture('plain', b'gzipped')
        client.data = json.dumps({'type': 'encoding', 'encoding': 'gzip'})
        client.handleMessage()
        with mock.patch.object(ascii_web_server.WebSocket, 'sendMessage') as send:
            client.flush_delivery()
        send.assert_called_once_with(b'gzipped')
