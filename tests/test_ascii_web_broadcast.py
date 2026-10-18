"""The paced producer must keep broadcasting beyond its first frame."""
from collections import deque
from unittest import TestCase, mock
import gzip
import json
from types import SimpleNamespace

import ascii_web_server


class BroadcastTests(TestCase):
    def test_multiple_frames_and_bounded_slow_client_backlog(self):
        client = mock.Mock()
        client.sendq = deque(['old1', 'old2', 'old3'])
        client.sendMessage.side_effect = client.sendq.append
        stop = RuntimeError('finished test input')
        with mock.patch.object(ascii_web_server, 'clients', [client]), \
                mock.patch.object(ascii_web_server.exchange_ascii, 'get_frame',
                                  side_effect=['one', b'two', stop]):
            with self.assertRaisesRegex(RuntimeError, 'finished test input'):
                ascii_web_server.broadcast_loop()
        self.assertEqual(list(client.sendq), ['\033[Hone', '\033[Htwo'])
        self.assertEqual(client.sendMessage.call_count, 2)

    def test_shared_gzip_is_encoded_once_per_frame_not_per_client(self):
        plain = mock.Mock(frame_encoding='text', sendq=deque())
        compressed = [mock.Mock(frame_encoding='gzip', sendq=deque()) for _ in range(3)]
        stop = RuntimeError('finished test input')
        original = gzip.compress
        with mock.patch.object(ascii_web_server, 'clients', [plain] + compressed), \
                mock.patch.object(ascii_web_server.exchange_ascii, 'get_frame',
                                  side_effect=['one', b'two', stop]), \
                mock.patch.object(ascii_web_server.gzip, 'compress', wraps=original) as encode:
            with self.assertRaisesRegex(RuntimeError, 'finished test input'):
                ascii_web_server.broadcast_loop()
        self.assertEqual(encode.call_count, 2)
        self.assertEqual([call.args[0] for call in plain.sendMessage.call_args_list],
                         ['\x1b[Hone', '\x1b[Htwo'])
        for index, text in enumerate(('one', 'two')):
            messages = [client.sendMessage.call_args_list[index].args[0] for client in compressed]
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
        client = mock.Mock(address=('test', 1234))
        with mock.patch.object(ascii_web_server, 'clients', []), \
                mock.patch.object(settings, 'ASCII_WEB_SOURCE_SIZE', (120,80)):
            ascii_web_server.AsciiWebSocket.handleConnected(client)
            messages = [call.args[0] for call in client.sendMessage.call_args_list]
            self.assertEqual(json.loads(messages[1]), {'type':'resize', 'cols':120, 'rows':80})
            self.assertEqual(client.frame_encoding, 'text')
            ascii_web_server.AsciiWebSocket.handleClose(client)
