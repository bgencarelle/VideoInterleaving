"""Live receiver configuration and startup failures, without audio hardware."""
import unittest
from unittest.mock import patch, Mock

from utilities import modem_v3_check as check


class LiveReceiveTests(unittest.TestCase):
    def test_live_receiver_accepts_matching_profile(self):
        with patch.object(check, 'do_live_receive') as receive:
            check.main(['live-receive', '--device', '0', '--headless'])
        args = receive.call_args.args[0]
        self.assertEqual((args.device, args.on_loss, args.buffer_frames), (0, 'hold', 2))

    def test_device_query_failure_stops_receiver(self):
        sd = Mock()
        sd.query_devices.side_effect = ValueError('Input device unavailable')
        self.assert_startup_fails(sd, 'Input device unavailable')

    def test_insufficient_channels_stops_receiver(self):
        sd = Mock()
        sd.query_devices.return_value = {'name': 'mono', 'max_input_channels': 1}
        self.assert_startup_fails(sd, 'Selected input channels unavailable')

    def assert_startup_fails(self, sd, message):
        def immediate_thread(*, target, **kwargs):
            worker = Mock(ident=None)
            worker.start.side_effect = target if target.__name__ != 'report' else None
            worker.is_alive.return_value = False
            return worker

        with patch.object(check, 'sounddevice', return_value=sd), \
             patch.object(check.time, 'sleep', side_effect=AssertionError('Receiver did not stop')), \
             patch('threading.Thread', side_effect=immediate_thread):
            with self.assertRaisesRegex(SystemExit, message):
                check.main(['live-receive', '--device', '0', '--headless'])



class ReceiverWindowTests(unittest.TestCase):
    def test_window_updates_and_quits_with_q(self):
        self.run_window('<q>')

    def test_window_quits_with_escape(self):
        self.run_window('<Escape>')

    def test_window_quits_with_ctrl_c(self):
        self.run_window('SIGINT')

    def test_slow_logging_does_not_block_picture_updates(self):
        self.run_window('<q>', slow_logging=True)

    def run_window(self, quit_key, slow_logging=False):
        import signal
        import threading
        log_started = threading.Event()
        release_log = threading.Event()
        previous_sigint = signal.getsignal(signal.SIGINT)
        import time
        import types
        import numpy as np
        from PIL import Image
        from animation_modem.imaging import image_values

        layout = check.PRESETS['lean-v3']
        coder, _ = check.coder_for('color-lean', None, layout)
        packets = [check.V3.encode(
            image_values(Image.new('RGB', (40, 48), (n*12, 30, 180)), coder.shapes),
            layout, coder, n+1, n+1, 20) for n in range(20)]
        audio = np.concatenate(packets + [np.zeros((1024, 2))]).astype(np.float32)
        shown = []

        class Root:
            pending = None
            closed = False

            def title(self, text):
                pass

            def protocol(self, name, callback):
                self.close = callback

            bindings = {}

            def bind(self, key, callback):
                self.bindings[key] = callback

            def after(self, delay, callback):
                self.pending = callback

            def destroy(self):
                self.closed = True

            def mainloop(self):
                deadline = time.monotonic()+5
                while not self.closed and time.monotonic() < deadline:
                    callback, self.pending = self.pending, None
                    if callback:
                        try:
                            callback()
                        except Exception as exc:
                            self.report_callback_exception(type(exc), exc, exc.__traceback__)
                    if len(shown) >= 3:
                        release_log.set()
                        if quit_key == 'SIGINT':
                            signal.raise_signal(signal.SIGINT)
                        else:
                            self.bindings[quit_key](None)
                    time.sleep(.001)
                self.destroy()

        root = Root()

        class Photo:
            def __init__(self, image):
                pass

            def paste(self, image):
                shown.append(np.asarray(image).copy())

        class Stream:
            pos = 0

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self, count):
                time.sleep(.001)
                chunk = audio[self.pos:self.pos+count]
                self.pos += count
                if len(chunk) < count:
                    chunk = np.pad(chunk, ((0, count-len(chunk)), (0, 0)))
                return chunk, False

        sd = Mock()
        sd.query_devices.return_value = {'name': 'test stereo', 'max_input_channels': 2}
        sd.InputStream.return_value = Stream()
        tk = types.SimpleNamespace(Tk=lambda: root, Label=lambda *a, **kw: Mock())
        image_tk = types.ModuleType('PIL.ImageTk')
        image_tk.PhotoImage = Photo
        def delayed_record(result):
            log_started.set()
            release_log.wait(2)
            return {}

        with patch.object(check, 'record', side_effect=delayed_record), \
             patch.object(check, 'sounddevice', return_value=sd), \
             patch.dict('sys.modules', {'tkinter': tk, 'PIL.ImageTk': image_tk}), \
             patch('PIL.ImageTk', image_tk, create=True):
            check.main(['live-receive', '--device', '0', '--verbose' if slow_logging else '--silent'])
        if slow_logging:
            self.assertTrue(log_started.is_set())
            self.assertTrue(release_log.is_set())
        self.assertIs(signal.getsignal(signal.SIGINT), previous_sigint)
        self.assertTrue(root.closed)
        self.assertGreaterEqual(len(shown), 3)
        self.assertFalse(np.array_equal(shown[0], shown[-1]))


class HeadlessShutdownTests(unittest.TestCase):
    def test_ctrl_c_closes_audio_stream_and_restores_signal_handler(self):
        import signal
        import numpy as np
        old_handler = signal.getsignal(signal.SIGINT)
        sd = Mock()
        sd.query_devices.return_value = {'name': 'test', 'max_input_channels': 2}
        stream = Mock()
        sd.InputStream.return_value.__enter__ = Mock(return_value=stream)
        sd.InputStream.return_value.__exit__ = Mock(return_value=False)

        def read(count):
            signal.raise_signal(signal.SIGINT)
            return np.zeros((count, 2), np.float32), False

        stream.read.side_effect = read
        with patch.object(check, 'sounddevice', return_value=sd):
            check.main(['live-receive', '--device', '0', '--headless', '--silent'])
        sd.InputStream.return_value.__exit__.assert_called_once()
        self.assertIs(signal.getsignal(signal.SIGINT), old_handler)
