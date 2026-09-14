"""Live-source encoder: capture -> profile-sized picture -> v3 packets."""
import io
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import modem_screen
from animation_modem import transport3 as v3
from animation_modem.core import SourceCoder
from animation_modem.imaging import PROFILES, fit_shapes, image_values, plane_shapes


class ReadExactTests(unittest.TestCase):
    """subprocess with bufsize=0 gives a raw stream whose read(n) may return
    fewer than n bytes. A colour frame is hundreds of kB, so short reads are
    normal and a single read() silently yields a black picture."""

    class Dribble:
        def __init__(self, data, chunk):
            self.data, self.chunk, self.pos = data, chunk, 0

        def read(self, want):
            take = min(want, self.chunk, len(self.data)-self.pos)
            out = self.data[self.pos:self.pos+take]
            self.pos += take
            return out

    def test_assembles_a_frame_from_short_reads(self):
        payload = bytes(range(256))*400
        stream = self.Dribble(payload, 1000)
        self.assertEqual(modem_screen._read_exact(stream, len(payload)), payload)

    def test_returns_none_at_end_of_stream(self):
        stream = self.Dribble(b'abc', 10)
        self.assertIsNone(modem_screen._read_exact(stream, 100))

    def test_a_single_read_would_have_truncated(self):
        payload = b'x'*50000
        self.assertLess(len(self.Dribble(payload, 4096).read(50000)), len(payload))


class FitterTests(unittest.TestCase):
    def test_output_is_exactly_the_profile_size(self):
        for profile in ('color', 'color-lean', 'mono'):
            with self.subTest(profile=profile):
                prepare = modem_screen.fitter(profile)
                out = prepare(np.zeros((480, 640, 3), np.uint8))
                self.assertEqual(out.size, PROFILES[profile][0])

    def test_letterbox_preserves_aspect_and_crop_fills(self):
        raw = np.zeros((100, 400, 3), np.uint8)
        raw[:, :, 0] = 255
        boxed = np.asarray(modem_screen.fitter('color', letterbox=True)(raw))
        filled = np.asarray(modem_screen.fitter('color', letterbox=False)(raw))
        self.assertLess(boxed[:, :, 0].mean(), filled[:, :, 0].mean())

    def test_rotation_and_mirror_apply(self):
        raw = np.zeros((64, 48, 3), np.uint8)
        raw[:, :10] = 255
        plain = np.asarray(modem_screen.fitter('color')(raw))
        flipped = np.asarray(modem_screen.fitter('color', mirror=True)(raw))
        self.assertFalse(np.array_equal(plain, flipped))
        turned = modem_screen.fitter('color', rotate=90)(raw)
        self.assertEqual(turned.size, PROFILES['color'][0])


class PrescaleTests(unittest.TestCase):
    """A 40x48 target from a Retina grab is a 96:1 reduction. Going straight
    to LANCZOS costs more per frame than the whole packet budget."""

    def test_large_frames_cost_about_the_same_as_small_ones(self):
        import time
        prepare = modem_screen.fitter('color-lean')

        def cost(w, h):
            raw = np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8)
            prepare(raw)
            began = time.perf_counter()
            for _ in range(4):
                prepare(raw)
            return (time.perf_counter()-began)/4

        small, large = cost(320, 426), cost(2560, 1600)
        # Without the stride prescale this ratio was ~25x.
        self.assertLess(large, small*4)

    def test_prescale_keeps_the_exact_output_size(self):
        for profile in ('color', 'color-lean', 'mono'):
            with self.subTest(profile=profile):
                prepare = modem_screen.fitter(profile)
                for w, h in ((320, 426), (1920, 1080), (3840, 2400)):
                    self.assertEqual(
                        prepare(np.zeros((h, w, 3), np.uint8)).size,
                        PROFILES[profile][0])

    def test_prescale_preserves_the_picture(self):
        """Striding must not shift or blank the image."""
        yy, xx = np.mgrid[:1600, :2560]
        raw = np.zeros((1600, 2560, 3), np.uint8)
        raw[:, :, 0] = np.uint8(128+120*np.sin(xx/200))
        raw[:, :, 1] = np.uint8(128+120*np.cos(yy/160))
        out = np.asarray(modem_screen.fitter('color-lean')(raw), float)
        self.assertGreater(out.std(), 10.0)
        self.assertGreater(out[:, :, 0].std(), 5.0)


class TestSourceTests(unittest.TestCase):
    def test_test_source_moves(self):
        grab = modem_screen.test_source()
        first = grab().copy()
        import time
        time.sleep(.05)
        self.assertFalse(np.array_equal(first, grab()))

    def test_throttled_serves_the_newest_frame(self):
        grab = modem_screen.Throttled(modem_screen.test_source(), 50)
        try:
            self.assertEqual(grab().shape[2], 3)
        finally:
            grab.close()


class RoundTripTests(unittest.TestCase):
    def decode(self, path, preset, profile):
        from utilities.modem_v3_check import main
        import contextlib
        import json
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            main(['read',
                  '--wav', str(path)])
        return [json.loads(l) for l in out.getvalue().splitlines()
                if l.startswith('{')]

    def test_write_then_read(self):
        with tempfile.TemporaryDirectory() as d:
            wav = Path(d)/'live.wav'
            modem_screen.main(['--source', 'test', '--write', str(wav),
                               '--frames', '6'])
            records = self.decode(wav, 'lean-v3', 'color-lean')
            self.assertEqual([r['frame'] for r in records], [1, 2, 3, 4, 5, 6])
            self.assertTrue(all(r['identity'] == 'verified_header' for r in records))

    def test_pictures_are_not_blank(self):
        """Guards the pacing bug: a source that outruns its decoder used to
        hand back the zero-filled placeholder and still decode perfectly."""
        with tempfile.TemporaryDirectory() as d:
            wav = Path(d)/'live.wav'
            frames = Path(d)/'out'
            modem_screen.main(['--source', 'test', '--write', str(wav),
                               '--frames', '4'])
            from utilities.modem_v3_check import main
            import contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                main(['read',
                      '--wav', str(wav), '--save-frames', str(frames)])
            pngs = sorted(frames.glob('*.png'))
            self.assertEqual(len(pngs), 4)
            for png in pngs:
                arr = np.asarray(Image.open(png).convert('RGB'), float)
                self.assertGreater(arr.std(), 5.0, f'{png.name} is flat')

    def test_mismatched_preset_warns_instead_of_shrinking_silently(self):
        """fit_shapes will quietly scale a picture down to fit any preset."""
        import contextlib
        err = io.StringIO()
        with tempfile.TemporaryDirectory() as d:
            with contextlib.redirect_stderr(err), \
                 contextlib.redirect_stdout(io.StringIO()):
                modem_screen.main(['--source', 'test', '--preset', 'lofi',
                                   '--profile', 'mono',
                                   '--write', str(Path(d)/'x.wav'),
                                   '--frames', '1'])
        self.assertIn('WARNING', err.getvalue())
        self.assertIn('shrunk', err.getvalue())

    def test_matched_preset_is_silent(self):
        import contextlib
        err = io.StringIO()
        with tempfile.TemporaryDirectory() as d:
            with contextlib.redirect_stderr(err), \
                 contextlib.redirect_stdout(io.StringIO()):
                modem_screen.main(['--source', 'test', '--preset', 'lean-v3',
                                   '--profile', 'color-lean',
                                   '--write', str(Path(d)/'x.wav'),
                                   '--frames', '1'])
        self.assertNotIn('WARNING', err.getvalue())


class VideoSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import shutil
        if not shutil.which('ffmpeg'):
            raise unittest.SkipTest('ffmpeg not installed')
        cls.dir = tempfile.TemporaryDirectory()
        root = Path(cls.dir.name)
        for n in range(8):
            a = np.zeros((120, 100, 3), np.uint8)
            a[:, :, 0] = 40+n*20
            a[30+n*5:70+n*5, 20:80, 1] = 220
            Image.fromarray(a).save(root/f'f{n:03d}.png')
        cls.clip = root/'clip.mp4'
        subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-framerate', '5',
                        '-i', str(root/'f%03d.png'), '-c:v', 'libx264',
                        '-pix_fmt', 'yuv420p', str(cls.clip)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def test_video_frames_differ(self):
        grab = modem_screen.video_source(str(self.clip), loop=True, realtime=False)
        try:
            frames = [grab().copy() for _ in range(4)]
        finally:
            grab.proc.terminate()
        self.assertGreater(max(float(np.abs(f.astype(int)-frames[0]).mean())
                               for f in frames[1:]), 1.0)

    def test_render_from_video_is_not_blank(self):
        with tempfile.TemporaryDirectory() as d:
            wav = Path(d)/'v.wav'
            modem_screen.main(['--source', 'video', '--file', str(self.clip),
                               '--write', str(wav), '--frames', '5'])
            layout = v3.ALL_PRESETS['lean-v3']
            shapes = fit_shapes(plane_shapes('color-lean'), layout.capacity)
            coder = SourceCoder(shapes)
            from animation_modem.audio_common import wav_blocks
            rx = v3.Receiver(layout, coder)
            got = []
            for block in wav_blocks(str(wav), (0, 1), 1024):
                got += rx.feed(block)
            got += rx.flush()
            self.assertEqual(len(got), 5)
            for g in got:
                self.assertIsNotNone(g.values)
                self.assertGreater(float(np.std(g.values)), .01)


if __name__ == '__main__':
    unittest.main()


class ArgumentSurfaceTests(unittest.TestCase):
    """to_device read args.prepare_ms and args.receive_margin_ms, which the
    parser never defined. Every test used --write, which never reaches that
    branch, so the live path died on its first packet with the suite green."""

    def test_every_args_attribute_is_defined(self):
        import ast
        import inspect
        source = inspect.getsource(modem_screen)
        used = {n.attr for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.Attribute)
                and isinstance(n.value, ast.Name) and n.value.id == 'args'}
        defined = set(vars(modem_screen.parser().parse_args(['--source', 'test'])))
        self.assertEqual(used-defined, set(),
                         'read from args but never added to the parser')

    def test_no_dead_arguments(self):
        import ast
        import inspect
        source = inspect.getsource(modem_screen)
        used = {n.attr for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.Attribute)
                and isinstance(n.value, ast.Name) and n.value.id == 'args'}
        defined = set(vars(modem_screen.parser().parse_args(['--source', 'test'])))
        self.assertEqual(defined-used, set(), 'defined but never read')


class DeviceLoopTests(unittest.TestCase):
    """Exercise to_device against a fake PacketOutput.

    The real one needs an audio device, which CI does not have -- which is
    exactly why this branch went untested and shipped broken.
    """

    class FakeOutput:
        def __init__(self, *a, **kw):
            self.sent = []
            self.deadline_misses = 0
            self.starvations = 0
            self.finished = False
            self._ready = True

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def ready(self):
            return self._ready

        def reserve(self, prepare_ms, receive_margin_ms):
            self.reserved = (prepare_ms, receive_margin_ms)
            import types
            return types.SimpleNamespace(start_time=0., target_time_ns=1_000_000)

        def submit(self, audio, slot=None):
            self.sent.append(np.asarray(audio))
            return True

        def finish(self):
            self.finished = True

    def run_loop(self, argv, output):
        import contextlib
        real = modem_screen.PacketOutput
        modem_screen.PacketOutput = lambda *a, **kw: output
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                modem_screen.main(argv)
        finally:
            modem_screen.PacketOutput = real

    def test_live_loop_sends_packets(self):
        out = self.FakeOutput()
        self.run_loop(['--source', 'test', '--frames', '5'], out)
        layout = v3.ALL_PRESETS['lean-v3']
        self.assertEqual(len(out.sent), 5)
        self.assertTrue(out.finished)
        for audio in out.sent:
            self.assertEqual(audio.shape, (layout.frame, 2))
            self.assertTrue(np.isfinite(audio).all())

    def test_live_loop_passes_the_timing_arguments(self):
        out = self.FakeOutput()
        self.run_loop(['--source', 'test', '--frames', '2',
                       '--prepare-ms', '22', '--receive-margin-ms', '33'], out)
        self.assertEqual(out.reserved, (22.0, 33.0))

    def test_live_packets_decode(self):
        out = self.FakeOutput()
        self.run_loop(['--source', 'test', '--frames', '4'], out)
        layout = v3.ALL_PRESETS['lean-v3']
        shapes = fit_shapes(plane_shapes('color-lean'), layout.capacity)
        coder = SourceCoder(shapes)
        stream = np.concatenate(out.sent).astype(np.float32)
        rx = v3.Receiver(layout, coder)
        got = []
        for i in range(0, len(stream), 1024):
            got += rx.feed(stream[i:i+1024])
        got += rx.flush()
        self.assertEqual(len(got), 4)
        self.assertTrue(all(g.identity == 'verified_header' for g in got))
        for g in got:
            self.assertGreater(float(np.std(g.values)), .01)


class LiveCaptureRegressionTests(unittest.TestCase):
    def test_device_parser_accepts_already_parsed_indices(self):
        from animation_modem.audio_common import device
        for value, expected in [('02', 2), ('0', 0), (2, 2), (0, 0),
                                ('BlackHole 2ch', 'BlackHole 2ch'), (None, None)]:
            with self.subTest(value=value):
                self.assertEqual(device(device(value)), expected)

    def test_numeric_device_reaches_output(self):
        from unittest.mock import patch
        out = DeviceLoopTests.FakeOutput()
        with patch.object(modem_screen, 'PacketOutput', return_value=out) as output:
            modem_screen.main(['--source', 'test', '--device', '02', '--frames', '1'])
        self.assertEqual(output.call_args.args[0], 2)
        self.assertEqual(len(out.sent), 1)

    def test_screen_capture_and_cleanup_share_worker_thread(self):
        import threading
        import types
        from unittest.mock import patch
        calls = []
        captured = threading.Event()

        class FakeMSS:
            monitors = [None, {}]

            def __init__(self):
                calls.append(('open', threading.get_ident()))

            def grab(self, mon):
                calls.append(('grab', threading.get_ident()))
                if sum(name == 'grab' for name, _ in calls) >= 2:
                    captured.set()
                return types.SimpleNamespace(raw=bytes([1, 2, 3, 255]), height=1, width=1)

            def close(self):
                calls.append(('close', threading.get_ident()))

        with patch.dict('sys.modules', {'mss': types.SimpleNamespace(MSS=FakeMSS)}):
            grab = modem_screen.Throttled(modem_screen.screen_source(), 100)
            try:
                self.assertTrue(captured.wait(2))
                np.testing.assert_array_equal(grab(), [[[3, 2, 1]]])
            finally:
                grab.close()
        self.assertEqual(calls[-1][0], 'close')
        self.assertEqual(len({ident for _, ident in calls}), 1)
        self.assertNotEqual(calls[0][1], threading.get_ident())

    def test_startup_failure_is_reported(self):
        def fail():
            raise ValueError('camera unavailable')
        with self.assertRaisesRegex(RuntimeError, 'camera unavailable'):
            modem_screen.Throttled(fail, 30)

    def test_background_failure_is_reported_instead_of_freezing(self):
        import threading
        release = threading.Event()
        calls = 0

        def source():
            nonlocal calls
            calls += 1
            if calls == 1:
                return np.zeros((1, 1, 3), np.uint8)
            release.wait(2)
            raise ValueError('capture disconnected')

        grab = modem_screen.Throttled(source, 100)
        try:
            release.set()
            grab._thread.join(2)
            with self.assertRaisesRegex(RuntimeError, 'capture disconnected'):
                grab()
        finally:
            grab.close()

    def test_ffmpeg_failure_does_not_generate_black_frames(self):
        import types
        from unittest.mock import patch
        proc = types.SimpleNamespace(stdout=io.BytesIO(b''), poll=lambda: 0)
        with patch.object(modem_screen.shutil, 'which', return_value='/usr/bin/ffmpeg'), \
             patch.object(modem_screen.subprocess, 'Popen', return_value=proc) as popen:
            grab = modem_screen.camera_source()
        self.assertTrue(popen.call_args.kwargs['start_new_session'])
        try:
            with self.assertRaisesRegex(RuntimeError, 'FFmpeg capture ended'):
                grab()
        finally:
            grab.close()


class CameraRateTests(unittest.TestCase):
    def test_camera_uses_supported_default_and_honors_override(self):
        from unittest.mock import patch
        for flags, expected in [([], 30), (['--capture-fps', '15'], 15)]:
            args = modem_screen.parser().parse_args(['--source', 'camera'] + flags)
            with patch.object(modem_screen, 'camera_source') as camera:
                modem_screen.source_for(args, 17.34)
            self.assertEqual(camera.call_args.args[1], expected)

    def test_camera_pipe_is_drained_at_capture_rate(self):
        from unittest.mock import patch
        frame = np.zeros((10, 10, 3), np.uint8)
        for flags, expected in [([], 30), (['--capture-fps', '15'], 15)]:
            with patch.object(modem_screen, 'source_for', return_value=lambda: frame), \
                 patch.object(modem_screen, 'Throttled') as throttle, \
                 patch.object(modem_screen, 'to_device'):
                modem_screen.main(['--source', 'camera', '--frames', '1'] + flags)
            self.assertEqual(throttle.call_args.args[1], expected)


class CapturePipeTests(unittest.TestCase):
    def test_ppm_short_reads_and_frame_boundaries(self):
        class ShortReads(io.BytesIO):
            def read(self, count=-1):
                return super().read(min(count, 7))

        frames = [np.full((24, 32, 3), n, np.uint8) for n in (10, 32, 255)]
        stream = ShortReads(b''.join(b'P6\n32 24\n255\n'+f.tobytes() for f in frames))
        for expected in frames:
            np.testing.assert_array_equal(modem_screen._read_ppm(stream), expected)
        with self.assertRaisesRegex(RuntimeError, 'FFmpeg capture ended'):
            modem_screen._read_ppm(stream)

    def test_stalled_capture_is_not_silently_repeated(self):
        from unittest.mock import patch
        grab = modem_screen.Throttled(lambda: np.zeros((2, 2, 3), np.uint8), 30)
        grab.close()
        with patch.object(modem_screen.time, 'monotonic', return_value=grab._updated+6):
            with self.assertRaisesRegex(RuntimeError, 'Capture stalled'):
                grab()

    def test_ffmpeg_preserves_aspect_and_delivers_changing_frames(self):
        import shutil
        if not shutil.which('ffmpeg'):
            self.skipTest('ffmpeg not installed')
        with tempfile.TemporaryDirectory() as d:
            for width, height in ((80, 60), (60, 80), (96, 54)):
                for n in range(12):
                    frame = np.full((height, width, 3), 20+n*15, np.uint8)
                    Image.fromarray(frame).save(Path(d)/f'f{n:03d}.png')
                grab = modem_screen.ffmpeg_source(f'image2:{d}/f%03d.png', 30, width=96)
                try:
                    frames = [grab() for _ in range(12)]
                    for n, frame in enumerate(frames):
                        self.assertAlmostEqual(frame.shape[1]/frame.shape[0], width/height,
                                               delta=.02)
                        self.assertAlmostEqual(float(frame.mean()), 20+n*15, delta=1)
                    with self.assertRaisesRegex(RuntimeError, 'FFmpeg capture ended'):
                        grab()
                finally:
                    grab.close()
                    grab.proc.stdout.close()


class CaptureTimingTests(unittest.TestCase):
    def test_timestamp_gaps_do_not_duplicate_the_first_frame(self):
        import shutil
        from unittest.mock import patch
        if not shutil.which('ffmpeg'):
            self.skipTest('ffmpeg not installed')
        real_popen = subprocess.Popen

        def gapped_input(cmd, **kwargs):
            cmd = list(cmd)
            at = cmd.index('-vf')+1
            cmd[at] += ',setpts=100*PTS'
            return real_popen(cmd, **kwargs)

        with tempfile.TemporaryDirectory() as d:
            for n in range(8):
                Image.new('RGB', (32, 24), (n*25, 0, 0)).save(Path(d)/f'f{n:03d}.png')
            with patch.object(modem_screen.subprocess, 'Popen', side_effect=gapped_input):
                grab = modem_screen.ffmpeg_source(f'image2:{d}/f%03d.png', 30, width=32)
            try:
                for n in range(8):
                    self.assertAlmostEqual(float(grab()[:, :, 0].mean()), n*25, delta=1)
                with self.assertRaisesRegex(RuntimeError, 'FFmpeg capture ended'):
                    grab()
            finally:
                grab.close()
                grab.proc.stdout.close()
            self.assertIsNotNone(grab.proc.poll())

    def test_shutdown_unblocks_capture_and_reaps_child(self):
        from unittest.mock import patch
        import sys
        real_popen = subprocess.Popen
        # Emit one complete image, then block forever like a stalled camera.
        code = "import sys,time; sys.stdout.buffer.write(b'P6\\n1 1\\n255\\nabc'); sys.stdout.buffer.flush(); time.sleep(60)"

        def blocked_camera(cmd, **kwargs):
            return real_popen([sys.executable, '-c', code], **kwargs)

        with patch.object(modem_screen.shutil, 'which', return_value='ffmpeg'), \
             patch.object(modem_screen.subprocess, 'Popen', side_effect=blocked_camera):
            raw = modem_screen.camera_source()
        capture = modem_screen.Throttled(raw, 30)
        capture.close()
        capture.close()  # cleanup is idempotent
        self.assertFalse(capture._thread.is_alive())
        self.assertIsNotNone(raw.proc.poll())
        self.assertTrue(raw.proc.stdout.closed)


class LiveCaptureRegressionTests(unittest.TestCase):
    def test_device_parser_accepts_already_parsed_indices(self):
        from animation_modem.audio_common import device
        for value, expected in [('02', 2), ('0', 0), (2, 2), (0, 0),
                                ('BlackHole 2ch', 'BlackHole 2ch'), (None, None)]:
            with self.subTest(value=value):
                self.assertEqual(device(device(value)), expected)

    def test_numeric_device_reaches_output(self):
        from unittest.mock import patch
        out = DeviceLoopTests.FakeOutput()
        with patch.object(modem_screen, 'PacketOutput', return_value=out) as output:
            modem_screen.main(['--source', 'test', '--device', '02', '--frames', '1'])
        self.assertEqual(output.call_args.args[0], 2)
        self.assertEqual(len(out.sent), 1)

    def test_screen_capture_and_cleanup_share_worker_thread(self):
        import threading
        import types
        from unittest.mock import patch
        calls = []
        captured = threading.Event()

        class FakeMSS:
            monitors = [None, {}]

            def __init__(self):
                calls.append(('open', threading.get_ident()))

            def grab(self, mon):
                calls.append(('grab', threading.get_ident()))
                if sum(name == 'grab' for name, _ in calls) >= 2:
                    captured.set()
                return types.SimpleNamespace(raw=bytes([1, 2, 3, 255]), height=1, width=1)

            def close(self):
                calls.append(('close', threading.get_ident()))

        with patch.dict('sys.modules', {'mss': types.SimpleNamespace(MSS=FakeMSS)}):
            grab = modem_screen.Throttled(modem_screen.screen_source(), 100)
            try:
                self.assertTrue(captured.wait(2))
                np.testing.assert_array_equal(grab(), [[[3, 2, 1]]])
            finally:
                grab.close()
        self.assertEqual(calls[-1][0], 'close')
        self.assertEqual(len({ident for _, ident in calls}), 1)
        self.assertNotEqual(calls[0][1], threading.get_ident())

    def test_startup_failure_is_reported(self):
        def fail():
            raise ValueError('camera unavailable')
        with self.assertRaisesRegex(RuntimeError, 'camera unavailable'):
            modem_screen.Throttled(fail, 30)

    def test_background_failure_is_reported_instead_of_freezing(self):
        import threading
        release = threading.Event()
        calls = 0

        def source():
            nonlocal calls
            calls += 1
            if calls == 1:
                return np.zeros((1, 1, 3), np.uint8)
            release.wait(2)
            raise ValueError('capture disconnected')

        grab = modem_screen.Throttled(source, 100)
        try:
            release.set()
            grab._thread.join(2)
            with self.assertRaisesRegex(RuntimeError, 'capture disconnected'):
                grab()
        finally:
            grab.close()

    def test_ffmpeg_failure_does_not_generate_black_frames(self):
        import types
        from unittest.mock import patch
        proc = types.SimpleNamespace(stdout=io.BytesIO(b''))
        with patch.object(modem_screen.shutil, 'which', return_value='/usr/bin/ffmpeg'), \
             patch.object(modem_screen.subprocess, 'Popen', return_value=proc) as popen:
            grab = modem_screen.camera_source()
        self.assertIsNone(popen.call_args.kwargs['stderr'])
        with self.assertRaisesRegex(RuntimeError, 'FFmpeg capture ended'):
            grab()


class CameraRateTests(unittest.TestCase):
    def test_camera_uses_supported_default_and_honors_override(self):
        from unittest.mock import patch
        for flags, expected in [([], 30), (['--capture-fps', '15'], 15)]:
            args = modem_screen.parser().parse_args(['--source', 'camera'] + flags)
            with patch.object(modem_screen, 'camera_source') as camera:
                modem_screen.source_for(args, 17.34)
            self.assertEqual(camera.call_args.args[1], expected)

    def test_camera_pipe_is_drained_at_capture_rate(self):
        from unittest.mock import patch
        frame = np.zeros((10, 10, 3), np.uint8)
        for flags, expected in [([], 30), (['--capture-fps', '15'], 15)]:
            with patch.object(modem_screen, 'source_for', return_value=lambda: frame), \
                 patch.object(modem_screen, 'Throttled') as throttle, \
                 patch.object(modem_screen, 'to_device'):
                modem_screen.main(['--source', 'camera', '--frames', '1'] + flags)
            self.assertEqual(throttle.call_args.args[1], expected)


class CapturePipeTests(unittest.TestCase):
    def test_ppm_short_reads_and_frame_boundaries(self):
        class ShortReads(io.BytesIO):
            def read(self, count=-1):
                return super().read(min(count, 7))

        frames = [np.full((24, 32, 3), n, np.uint8) for n in (10, 32, 255)]
        stream = ShortReads(b''.join(b'P6\n32 24\n255\n'+f.tobytes() for f in frames))
        for expected in frames:
            np.testing.assert_array_equal(modem_screen._read_ppm(stream), expected)
        with self.assertRaisesRegex(RuntimeError, 'FFmpeg capture ended'):
            modem_screen._read_ppm(stream)

    def test_stalled_capture_is_not_silently_repeated(self):
        from unittest.mock import patch
        grab = modem_screen.Throttled(lambda: np.zeros((2, 2, 3), np.uint8), 30)
        grab.close()
        with patch.object(modem_screen.time, 'monotonic', return_value=grab._updated+6):
            with self.assertRaisesRegex(RuntimeError, 'Capture stalled'):
                grab()

    def test_ffmpeg_preserves_aspect_and_delivers_changing_frames(self):
        import shutil
        if not shutil.which('ffmpeg'):
            self.skipTest('ffmpeg not installed')
        with tempfile.TemporaryDirectory() as d:
            for width, height in ((80, 60), (60, 80), (96, 54)):
                for n in range(12):
                    frame = np.full((height, width, 3), 20+n*15, np.uint8)
                    Image.fromarray(frame).save(Path(d)/f'f{n:03d}.png')
                grab = modem_screen.ffmpeg_source(f'image2:{d}/f%03d.png', 30, width=96)
                try:
                    frames = [grab() for _ in range(12)]
                    for n, frame in enumerate(frames):
                        self.assertAlmostEqual(frame.shape[1]/frame.shape[0], width/height,
                                               delta=.02)
                        self.assertAlmostEqual(float(frame.mean()), 20+n*15, delta=1)
                    with self.assertRaisesRegex(RuntimeError, 'FFmpeg capture ended'):
                        grab()
                finally:
                    grab.proc.terminate()
                    grab.proc.wait(timeout=5)
                    grab.proc.stdout.close()
