"""The plain WebSocket import must not require removed SSL protocols."""
import subprocess
import sys
import unittest


class WebSocketImportCompatibilityTests(unittest.TestCase):
    def test_import_without_legacy_tls_constant_does_not_modify_ssl(self):
        code = '''
import ssl
if hasattr(ssl, 'PROTOCOL_TLSv1_2'):
    del ssl.PROTOCOL_TLSv1_2
import ascii_web_server
assert callable(ascii_web_server.SimpleWebSocketServer)
assert issubclass(ascii_web_server.AsciiWebSocket, ascii_web_server.WebSocket)
assert not hasattr(ssl, 'PROTOCOL_TLSv1_2')
server = ascii_web_server.SimpleWebSocketServer('127.0.0.1', 0,
                                               ascii_web_server.AsciiWebSocket)
server.close()
'''
        result = subprocess.run([sys.executable, '-c', code],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_existing_tls_constant_is_preserved(self):
        code = '''
import ssl
original = getattr(ssl, 'PROTOCOL_TLSv1_2', None)
import ascii_web_server
assert getattr(ssl, 'PROTOCOL_TLSv1_2', None) is original
'''
        result = subprocess.run([sys.executable, '-c', code],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
