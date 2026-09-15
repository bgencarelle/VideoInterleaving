"""Live receiver configuration and startup failures, without audio hardware."""
import unittest
from unittest.mock import patch, Mock

from PIL import Image

from utilities import modem_v3_check as check


class LiveReceiveTests(unittest.TestCase):
    def test_live_receiver_accepts_matching_profile(self):
        with patch.object(check, 'do_live_receive') as receive:
            check.main(['live-receive', '--device', '0', '--headless'])
        args = receive.call_args.args[0]
        # 'damaged' is the default: a picture whose header did not verify has
        # still decoded, and holding the last good frame hides it.
        self.assertEqual((args.device, args.on_loss, args.buffer_frames),
                         (0, 'damaged', 2))

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

    def test_window_updates_on_a_device_that_is_not_at_48k(self):
        """The receiver never asks for a rate, so 44.1 kHz is just a device."""
        self.run_window('<q>', device_rate=44100)

    def run_window(self, quit_key, slow_logging=False, device_rate=48000):
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
            # PortAudio reports the rate the device is running at, whether or
            # not one was requested. Nothing under test may request one.
            samplerate = device_rate

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
        self.addCleanup(lambda: self.assertNotIn(
            'samplerate', sd.InputStream.call_args.kwargs,
            'live-receive must open the device at whatever rate it is set to'))
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
        stream = Mock(samplerate=48000)
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


class DisplayOptionTests(unittest.TestCase):
    """Window size and how the picture is scaled up into it."""

    def parsed(self, *extra):
        with patch.object(check, 'do_live_receive') as receive:
            check.main(['live-receive', '--device', '0', '--headless', *extra])
        return receive.call_args.args[0]

    def test_raw_is_the_default(self):
        """Nearest-neighbour, deliberately. This doubles as a diagnostic
        display, and smoothing hides the artefacts worth seeing."""
        args = self.parsed()
        self.assertEqual(args.scaling, 'raw')
        self.assertIs(check.SCALING[args.scaling], Image.Resampling.NEAREST)

    def test_smooth_is_lanczos(self):
        args = self.parsed('--scaling', 'smooth')
        self.assertEqual(args.scaling, 'smooth')
        self.assertIs(check.SCALING[args.scaling], Image.Resampling.LANCZOS)

    def test_an_unknown_scaling_is_refused(self):
        with self.assertRaises(SystemExit):
            self.parsed('--scaling', 'bilinear')

    def test_the_window_has_a_stated_default(self):
        args = self.parsed()
        self.assertEqual((args.width, args.height), check.WINDOW)

    def test_the_default_window_is_a_whole_multiple_of_both_pictures(self):
        """40x48 and color-dct's 80x96 both have to land on an integer scale,
        or the window resamples the picture before `--scaling` ever sees it."""
        for w, h in ((40, 48), (80, 96)):
            self.assertEqual(check.WINDOW[0] % w, 0, f'{w} does not divide')
            self.assertEqual(check.WINDOW[1] % h, 0, f'{h} does not divide')
            self.assertEqual(check.WINDOW[0]//w, check.WINDOW[1]//h,
                             'the two axes must scale by the same factor')

    def test_the_window_can_be_overridden(self):
        args = self.parsed('--width', '320', '--height', '384')
        self.assertEqual((args.width, args.height), (320, 384))


class DisplayScalingTests(unittest.TestCase):
    def test_the_chosen_filter_is_what_reaches_the_window(self):
        """Pins the wiring, not just the flag: raw and smooth must actually
        produce different pixels, and raw must be the one that does not
        invent intermediate values."""
        import numpy as np
        from PIL import Image as _Image, ImageOps
        picture = _Image.fromarray(np.uint8([[0, 255], [255, 0]]*1).repeat(
            24, 0).repeat(20, 1)[:48, :40][..., None].repeat(3, 2))
        raw = ImageOps.contain(picture, check.WINDOW, check.SCALING['raw'])
        smooth = ImageOps.contain(picture, check.WINDOW, check.SCALING['smooth'])
        self.assertNotEqual(list(raw.getdata()), list(smooth.getdata()))
        levels = {p[0] for p in raw.getdata()}
        self.assertTrue(levels <= {0, 255},
                        'raw scaling must not create intermediate values')


class PictureSizeReportTests(unittest.TestCase):
    """How big the decoded image is, BEFORE the window scales it up."""

    def test_the_size_comes_from_the_luma_plane(self):
        self.assertEqual(check.picture_size([[48, 40], [12, 10], [12, 10]]), '40x48')
        self.assertEqual(check.picture_size([[96, 80], [48, 40], [48, 40]]), '80x96')
        self.assertIsNone(check.picture_size(None))
        self.assertIsNone(check.picture_size([]))

    def test_it_is_reported_as_its_own_field(self):
        """Not left buried in `shapes`, which is a nested list of (rows, cols)
        and reads backwards from the size anyone wants."""
        from animation_modem.core import Decoded
        r = Decoded(status='received', identity='verified_header')
        r.extra = {'shapes': [[96, 80], [48, 40], [48, 40]], 'preset': 'hires-v3'}
        self.assertEqual(check.record(r)['picture'], '80x96')

    def test_a_shrunk_picture_reports_what_it_actually_is(self):
        """The case this exists for. fit_shapes quietly reduces a profile that
        does not fit its preset, and the only previous sign was a nested list
        that still said 'color'."""
        from animation_modem.core import Decoded
        r = Decoded(status='received', identity='verified_header')
        r.extra = {'shapes': [[46, 38], [23, 19], [23, 19]], 'profile': 'color'}
        line = check.record(r)
        self.assertEqual(line['picture'], '38x46')
        self.assertEqual(line['profile'], 'color')

    def test_the_window_status_names_both_sizes(self):
        """Native and scaled, so it is clear which number is the picture and
        which is the window."""
        import inspect
        source = inspect.getsource(check.do_live_receive)
        self.assertIn('picture_size(', source)
        self.assertIn("native", source)
        self.assertIn('scaled.width', source)
