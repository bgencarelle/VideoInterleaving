"""One preview setting and the video-file transport of the V7 sender.

Covers the GUI (preference migration, the command line per preview choice,
the transport controls and saved resume position), the file playback wrapper
(pause, seek, restart with fake readers and a fake clock), the soundtrack
readers, and the live sender (packets keep flowing with a repeated picture
while paused). No audio device is opened.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import v7_live                                              # noqa: E402
from tools.v7_capture import (CaptureEndOfStream, FilePlayback,       # noqa: E402
                              Throttled, is_file_video_source,
                              probe_duration, probe_frames, video_source)
from tools.v7_preview_popout import PopoutFrames                       # noqa: E402
from tools import v7_preview_popout                                    # noqa: E402
from tools.v7_preview_protocol import pack_preview_datagram            # noqa: E402
from tools.v7_send_gui import (OutputDevice, SenderGui, build_command, # noqa: E402
                               _restore_sender_settings, resume_position)
from tools.v7_source_audio import (FFmpegSourceAudio,                  # noqa: E402
                                   SharedVideoAudioSource)


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class FakeReaders:
    """open_reader for FilePlayback: frames are (start, index) pairs."""

    def __init__(self, tail_frames=3):
        self.opened = []
        self.closed = []
        self.released = []
        self.tail_frames = tail_frames

    def __call__(self, start):
        number = len(self.opened)
        self.opened.append(start)
        state = {'index': 0, 'closed': False}

        def reader():
            if state['closed']:
                raise RuntimeError('reader closed')
            if start > 0 and state['index'] >= self.tail_frames:
                raise CaptureEndOfStream
            state['index'] += 1
            return (start, state['index'])

        def close():
            state['closed'] = True
            self.closed.append(number)

        reader.close = close
        reader.release = lambda: self.released.append(number)
        return reader


class FilePlaybackTests(unittest.TestCase):
    FPS = 10.0

    def setUp(self):
        self.clock = FakeClock()
        self.readers = FakeReaders()
        self.playback = FilePlayback(self.readers, duration=60.0,
                                     clock=self.clock, idle=.001,
                                     fps=self.FPS)

    def read(self, count, playback=None):
        frame = None
        for _ in range(count):
            frame = (playback or self.playback)()
        return frame

    def test_playing_reads_the_looping_reader_and_counts_its_frames(self):
        self.assertEqual(self.readers.opened, [0.0])
        self.assertEqual(self.playback.position, 0.0)
        self.assertEqual(self.playback(), (0.0, 1))
        self.clock.now += 2.5                    # wall time is not position
        self.assertEqual(self.playback(), (0.0, 2))
        self.assertAlmostEqual(self.playback.position, .2)
        self.assertEqual(self.playback.status(), {
            'position': .2, 'duration': 60.0, 'paused': False})
        self.read(23)
        self.assertAlmostEqual(self.playback.position, 2.5)
        # The looping reader wraps at the end of the file.
        self.read(600)
        self.assertAlmostEqual(self.playback.position, 2.5)

    def test_pause_repeats_the_held_frame_and_freezes_the_position(self):
        held = self.read(40)
        self.playback.pause()
        self.assertEqual(self.readers.closed, [0])
        self.clock.now += 30.0
        for _ in range(5):
            self.assertEqual(self.playback(), held)
        self.assertTrue(self.playback.paused)
        self.assertAlmostEqual(self.playback.position, 4.0)
        # Nothing is read while paused.
        self.assertEqual(self.readers.opened, [0.0])
        self.assertEqual(self.readers.released, [0])

    def test_play_resumes_from_the_paused_position(self):
        self.read(40)
        self.playback.pause()
        self.playback()
        self.clock.now += 30.0
        self.playback.play()
        self.assertEqual(self.playback(), (4.0, 1))
        self.assertEqual(self.readers.opened, [0.0, 4.0])
        self.readers.tail_frames = 100
        self.read(14)
        self.assertAlmostEqual(self.playback.position, 5.5)
        self.assertFalse(self.playback.paused)

    def test_seek_while_playing_reopens_at_the_position(self):
        self.playback()
        self.playback.seek(42.0)
        self.assertAlmostEqual(self.playback.position, 42.0)
        self.assertEqual(self.playback(), (42.0, 1))
        self.assertAlmostEqual(self.playback.position, 42.1)
        self.assertEqual(self.readers.opened, [0.0, 42.0])
        self.assertEqual(set(self.readers.closed), {0})

    def test_seek_while_paused_holds_one_picture_from_the_new_position(self):
        self.playback()
        self.playback.pause()
        self.playback.seek(20.0)
        self.assertEqual(self.playback(), (20.0, 1))
        self.clock.now += 9.0
        for _ in range(3):
            self.assertEqual(self.playback(), (20.0, 1))
        self.assertAlmostEqual(self.playback.position, 20.0)
        self.assertTrue(self.playback.paused)
        self.playback.play()
        self.assertEqual(self.playback(), (20.0, 1))   # a new reader at 20 s
        self.assertEqual(self.readers.opened, [0.0, 20.0, 20.0])

    def test_restart_goes_to_the_beginning_with_the_looping_reader(self):
        self.playback()
        self.playback.seek(30.0)
        self.playback()
        self.playback.restart()
        self.assertAlmostEqual(self.playback.position, 0.0)
        self.assertEqual(self.playback(), (0.0, 1))
        self.assertEqual(self.readers.opened, [0.0, 30.0, 0.0])
        self.assertAlmostEqual(self.playback.position, .1)

    def test_end_of_a_seek_pass_repeats_from_the_beginning(self):
        self.playback.seek(58.0)
        for index in range(1, 4):
            self.assertEqual(self.playback(), (58.0, index))
        self.assertAlmostEqual(self.playback.position, 58.3)
        # The pass from 58 s ended: the looping reader opens at 0.
        self.assertEqual(self.playback(), (0.0, 1))
        self.assertEqual(self.readers.opened, [0.0, 58.0, 0.0])
        self.clock.now += 1.0
        self.assertAlmostEqual(self.playback.position, .1)

    def test_position_never_follows_a_slow_or_jittery_clock(self):
        # One script of play, pause, seek, restart and a loop wrap, run
        # against clocks that stand still, crawl, race and jump backwards:
        # the positions are the same, start + frames/fps, every time.
        def stopped():
            return 5.0

        def crawling(state=[0.0]):
            state[0] += 1e-6
            return state[0]

        def racing(state=[0.0]):
            state[0] += 977.0
            return state[0]

        def jittery(state=[0]):
            state[0] += 1
            return (state[0]*7919) % 613-300.0

        expected = [('start', 0.0), ('played 30', 1.2), ('paused', 1.2),
                    ('held', 1.2), ('resumed 10', 1.6), ('seek', 30.0),
                    ('played 5', 30.2), ('restart', 0.0),
                    ('played 100', 4.0), ('wrapped', 10.0),
                    ('seek near end', 59.9), ('tail', 59.98),
                    ('repeat from 0', .04), ('paused seek', 7.0),
                    ('still', 7.0), ('resumed 3', 7.12)]
        for clock in (stopped, crawling, racing, jittery):
            with self.subTest(clock=clock.__name__):
                readers = FakeReaders(tail_frames=1000)
                playback = FilePlayback(readers, duration=60.0, clock=clock,
                                        idle=.001, fps=25.0, loop_frames=1500)
                seen = [('start', playback.position)]
                self.read(30, playback)
                seen.append(('played 30', playback.position))
                playback.pause()
                seen.append(('paused', playback.position))
                self.read(4, playback)
                seen.append(('held', playback.position))
                playback.play()
                self.read(10, playback)
                seen.append(('resumed 10', playback.position))
                playback.seek(30.0)
                seen.append(('seek', playback.position))
                self.read(5, playback)
                seen.append(('played 5', playback.position))
                playback.restart()
                seen.append(('restart', playback.position))
                self.read(100, playback)
                seen.append(('played 100', playback.position))
                self.read(1650, playback)        # 1750 frames: a second pass
                seen.append(('wrapped', playback.position))
                readers.tail_frames = 2
                playback.seek(59.9)
                seen.append(('seek near end', playback.position))
                self.read(2, playback)
                seen.append(('tail', playback.position))
                self.read(1, playback)           # the pass ended: from 0
                seen.append(('repeat from 0', playback.position))
                playback.pause()
                playback.seek(7.0)
                seen.append(('paused seek', playback.position))
                self.read(3, playback)
                seen.append(('still', playback.position))
                readers.tail_frames = 1000
                playback.play()
                self.read(3, playback)
                seen.append(('resumed 3', playback.position))
                playback.close()
                self.assertEqual([name for name, _value in seen],
                                 [name for name, _value in expected])
                for (name, value), (_name, wanted) in zip(seen, expected):
                    self.assertAlmostEqual(value, wanted, places=9, msg=name)

    def test_loop_wrap_uses_the_frame_count_or_else_the_duration(self):
        # 250 frames a pass at 25 fps: after 260 frames the picture is the
        # tenth of the second pass, whatever the container says its length is.
        counted = FilePlayback(FakeReaders(), duration=10.4, fps=25.0,
                               loop_frames=250, clock=self.clock)
        self.read(260, counted)
        self.assertAlmostEqual(counted.position, .4)
        # No frame count (Matroska): wrap on the probed duration.
        timed = FilePlayback(FakeReaders(), duration=10.0, fps=25.0,
                             clock=self.clock)
        self.read(260, timed)
        self.assertAlmostEqual(timed.position, .4)

    def test_without_a_frame_rate_the_position_falls_back_to_wall_time(self):
        playback = FilePlayback(self.readers, duration=60.0,
                                clock=self.clock, idle=.001)
        self.assertIsNone(playback.fps)
        playback()
        self.clock.now += 2.5
        playback()
        self.assertAlmostEqual(playback.position, 2.5)

    def test_a_start_or_seek_past_the_end_begins_at_the_beginning(self):
        readers = FakeReaders()
        playback = FilePlayback(readers, duration=60.0, start=75.0,
                                clock=self.clock)
        self.assertEqual(readers.opened, [0.0])
        self.playback.seek(60.0)
        self.assertEqual(self.playback.position, 0.0)
        self.playback.seek(float('nan'))
        self.assertEqual(self.playback.position, 0.0)
        playback.close()

    def test_resume_start_opens_the_first_reader_at_the_position(self):
        readers = FakeReaders()
        playback = FilePlayback(readers, duration=60.0, start=12.5,
                                clock=self.clock)
        self.assertEqual(readers.opened, [12.5])
        self.assertEqual(playback(), (12.5, 1))
        self.assertAlmostEqual(playback.position, 12.5)

    def test_a_reader_error_that_is_not_a_transport_action_is_raised(self):
        def broken(_start):
            def reader():
                raise RuntimeError('FFmpeg failed')
            reader.close = lambda: None
            return reader

        playback = FilePlayback(broken, clock=self.clock)
        with self.assertRaisesRegex(RuntimeError, 'FFmpeg failed'):
            playback()

    def test_close_closes_the_reader_and_runs_the_hook_once(self):
        closed = []
        playback = FilePlayback(self.readers, clock=self.clock,
                                on_close=lambda: closed.append(True))
        playback()
        playback.close()
        playback.close()
        self.assertEqual(closed, [True])
        self.assertIn(1, self.readers.closed)
        with self.assertRaises(RuntimeError):
            playback()

    def test_capture_thread_keeps_a_fresh_frame_while_paused(self):
        # Throttled reports a stall when no frame arrives; a paused file
        # must keep handing it the held picture.
        playback = FilePlayback(self.readers, idle=.005)
        capture = Throttled(playback, 200)
        try:
            playback.pause()
            time.sleep(.05)
            held = capture()
            before = capture.grabs
            time.sleep(.08)
            self.assertEqual(capture(), held)
            self.assertGreater(capture.grabs, before)
        finally:
            capture.close()


class VideoSourceStartTests(unittest.TestCase):
    class Process:
        def __init__(self):
            self.stdout = io.BytesIO()

        def poll(self):
            return 0

    def _command(self, **kwargs):
        (ROOT/'tmp').mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=ROOT/'tmp', suffix='.mp4') as clip, \
                mock.patch('tools.v7_capture.shutil.which',
                           side_effect=lambda name: None if name == 'ffprobe'
                           else '/usr/bin/'+name), \
                mock.patch('tools.v7_capture.subprocess.Popen',
                           return_value=self.Process()) as popen:
            grab = video_source(clip.name, **kwargs)
            grab.close()
            return popen.call_args.args[0], clip.name

    def test_a_start_position_seeks_and_plays_to_the_end_once(self):
        command, path = self._command(start=12.5)
        self.assertNotIn('-stream_loop', command)
        self.assertIn('-re', command)
        seek = command.index('-ss')
        self.assertEqual(command[seek+1], '12.500')
        self.assertLess(seek, command.index('-i'))
        self.assertEqual(command[command.index('-i')+1], path)

    def test_without_a_start_the_command_is_the_looping_one(self):
        command, _path = self._command()
        self.assertIn('-stream_loop', command)
        self.assertNotIn('-ss', command)

    def test_only_local_files_are_file_sources(self):
        self.assertTrue(is_file_video_source('clip.mp4'))
        self.assertTrue(is_file_video_source('/movies/a clip.mkv'))
        for source in ('rtsp://camera.example/live', '',
                       'https://media.example/clip.mp4', None):
            self.assertFalse(is_file_video_source(source))

    def test_duration_probe_reads_ffprobe_and_tolerates_failure(self):
        with tempfile.NamedTemporaryFile() as clip, \
                mock.patch('tools.v7_capture.shutil.which',
                           return_value='/usr/bin/ffprobe'):
            good = mock.Mock(return_value=SimpleNamespace(
                returncode=0, stdout='63.250000\n'))
            self.assertEqual(probe_duration(clip.name, run=good), 63.25)
            for stdout, code in (('N/A\n', 0), ('', 0), ('10\n', 1)):
                bad = mock.Mock(return_value=SimpleNamespace(
                    returncode=code, stdout=stdout))
                self.assertIsNone(probe_duration(clip.name, run=bad))
            self.assertIsNone(probe_duration(clip.name+'.missing', run=good))
        self.assertIsNone(probe_duration('rtsp://camera.example/live'))

    def test_frame_probe_reads_the_rate_and_count_and_tolerates_failure(self):
        def result(stdout, code=0):
            return mock.Mock(return_value=SimpleNamespace(
                returncode=code, stdout=stdout))

        with tempfile.NamedTemporaryFile() as clip, \
                mock.patch('tools.v7_capture.shutil.which',
                           return_value='/usr/bin/ffprobe'):
            for stdout, expected in (
                    ('avg_frame_rate=30000/1001\nnb_frames=1798\n',
                     (30000/1001, 1798)),
                    ('avg_frame_rate=25/1\nnb_frames=N/A\n', (25.0, None)),
                    ('avg_frame_rate=24\n', (24.0, None)),
                    ('avg_frame_rate=0/0\nnb_frames=0\n', (None, None)),
                    ('', (None, None))):
                self.assertEqual(probe_frames(clip.name, run=result(stdout)),
                                 expected, stdout)
            self.assertEqual(
                probe_frames(clip.name, run=result('avg_frame_rate=25/1', 1)),
                (None, None))
            self.assertEqual(probe_frames(clip.name, run=mock.Mock(
                side_effect=OSError('no ffprobe'))), (None, None))
            self.assertEqual(
                probe_frames(clip.name+'.missing', run=result('x')),
                (None, None))
        self.assertEqual(probe_frames('rtsp://camera.example/live'),
                         (None, None))


class SoundtrackTransportTests(unittest.TestCase):
    class Process:
        def __init__(self):
            self.stdout = io.BytesIO(
                np.full(4800, .5, dtype='<f4').tobytes())

        def poll(self):
            return 0

        def terminate(self):
            pass

    def test_separate_soundtrack_is_silent_when_halted_and_restarts(self):
        (ROOT/'tmp').mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=ROOT/'tmp') as clip, \
                mock.patch('subprocess.Popen',
                           side_effect=lambda *a, **k: self.Process()) as popen:
            audio = FFmpegSourceAudio(clip.name, 48_000, target_samples=480)
            try:
                self.assertTrue(audio.wait_for_samples(4800, 1.0))
                self.assertGreater(float(np.max(np.abs(audio.read(480)))), 0)
                audio.halt()
                for _ in range(4):
                    np.testing.assert_array_equal(audio.read(480), 0.0)
                self.assertEqual(audio.buffer.available, 0)
                audio.restart(7.25)
                command = popen.call_args.args[0]
                self.assertEqual(command[command.index('-ss')+1], '7.250')
                self.assertLess(command.index('-ss'), command.index('-i'))
                self.assertNotIn('-stream_loop', command)
                self.assertTrue(audio.wait_for_samples(4800, 1.0))
                self.assertGreater(float(np.max(np.abs(audio.read(480)))), 0)
            finally:
                audio.close()
        first = popen.call_args_list[0].args[0]
        self.assertIn('-stream_loop', first)
        self.assertNotIn('-ss', first)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe') and
                         hasattr(os, 'mkfifo'),
                         'shared A/V smoke test needs POSIX FFmpeg and ffprobe')
    def test_shared_soundtrack_is_silent_while_paused_and_resumes(self):
        (ROOT/'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp',
                                         prefix='v7-transport-test-') as folder:
            clip = str(Path(folder)/'tone.mkv')
            subprocess.run([
                'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                '-f', 'lavfi', '-i', 'color=c=gray:s=32x32:r=10:d=3',
                '-f', 'lavfi', '-i',
                'sine=frequency=1000:sample_rate=48000:duration=3',
                '-c:v', 'mpeg4', '-c:a', 'pcm_s16le', clip,
            ], check=True)
            self.assertAlmostEqual(probe_duration(clip), 3.0, delta=.2)
            source = SharedVideoAudioSource(
                clip, 48_000, width=32, target_samples=2400)
            fps, _frame_count = probe_frames(clip)
            playback = FilePlayback(
                source.reopen, duration=probe_duration(clip),
                reader=source.playback_reader(), on_close=source.close,
                fps=fps)
            capture = Throttled(playback, 20)

            def level(count=2400):
                samples = source.read(count)
                return float(np.sqrt(np.mean(samples*samples)))

            try:
                self.assertTrue(source.wait_for_samples(4800, 2.0))
                self.assertGreater(level(), .05)
                first_process = source.proc

                playback.pause()
                self.assertIsNotNone(first_process.poll())   # reader stopped
                held = np.array(capture(), copy=True)
                for _ in range(5):
                    self.assertEqual(level(), 0.0)
                    time.sleep(.02)
                np.testing.assert_array_equal(capture(), held)
                position = playback.position
                self.assertGreater(position, 0.0)

                playback.play()
                deadline = time.monotonic()+3.0
                heard = 0.0
                while time.monotonic() < deadline and heard < .05:
                    heard = level()
                    time.sleep(.02)
                self.assertGreater(heard, .05)
                self.assertIsNot(source.proc, first_process)
                self.assertIsNone(source.proc.poll())
                self.assertGreaterEqual(playback.position, position)
                self.assertEqual(np.asarray(capture()).shape, (32, 32, 3))
                # The real file's rate and frame count are read (10 fps,
                # 3 s), and the position is a whole number of its frames.
                self.assertEqual(playback.fps, 10.0)
                frames = playback.position*10.0
                self.assertAlmostEqual(frames, round(frames), places=6)
            finally:
                capture.close()
            self.assertTrue(source._closed)
            self.assertIsNotNone(source.proc.poll())


class TransportControlChannelTests(unittest.TestCase):
    def test_transport_commands_reach_the_attached_playback(self):
        playback = mock.Mock()
        playback.status.return_value = {'position': 1.0, 'duration': 9.0,
                                        'paused': False}
        transport = v7_live.LiveTransportControls()
        self.assertFalse(transport.update('{"transport": "pause"}'))
        self.assertIsNone(transport.status())
        transport.attach(playback)
        self.assertTrue(transport.update('{"transport": "pause"}'))
        self.assertTrue(transport.update('{"transport": "play"}'))
        self.assertTrue(transport.update('{"transport": "restart"}'))
        self.assertTrue(transport.update(
            '{"transport": "seek", "position": 12.5}'))
        self.assertEqual(playback.method_calls, [
            mock.call.pause(), mock.call.play(), mock.call.restart(),
            mock.call.seek(12.5)])
        self.assertEqual(transport.status()['duration'], 9.0)
        for bad in ('not json', '[]', '{"transport": "close"}',
                    '{"transport": "seek"}', '{"brightness": 1.2}',
                    '{"transport": "seek", "position": -1}',
                    '{"transport": "seek", "position": "x"}'):
            self.assertFalse(transport.update(bad), bad)
        self.assertEqual(len(playback.method_calls), 5)   # + one status()

    def test_one_channel_carries_tone_and_transport_updates(self):
        playback = mock.Mock()
        transport = v7_live.LiveTransportControls()
        transport.attach(playback)
        tones = v7_live.LiveToneControls(1.0, 1.0)
        stream = io.StringIO('{"gamma": 0.8}\n{"transport": "pause"}\n')
        v7_live._read_live_tone_controls(
            stream, tones, threading.Event(), transport)
        self.assertEqual(tones.snapshot(), {'brightness': 1.0, 'gamma': 0.8})
        playback.pause.assert_called_once_with()
        # The tone-only form still works (no transport given).
        v7_live._read_live_tone_controls(
            io.StringIO('{"gamma": 0.7}\n'), tones, threading.Event())
        self.assertEqual(tones.snapshot()['gamma'], 0.7)


class ControlPipe:
    """A stand-in for the sender's stdin that the test feeds line by line."""

    def __init__(self):
        self.lines = queue.Queue()

    def readline(self):
        return self.lines.get()

    def send(self, **message):
        self.lines.put(json.dumps(message)+'\n')


class SenderPauseTests(unittest.TestCase):
    def test_pause_keeps_packets_flowing_with_the_held_picture(self):
        written = []
        frames_seen = []
        source_indices = []
        opened = []

        class OutputStream:
            samplerate = 48000.0

            def __init__(self, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

            def write(self, samples):
                written.append(len(samples))

        def fake_capture(_args, start=0.0):
            # Levels 1.. from the beginning, 101.. from a later position.
            opened.append(start)
            state = {'level': 100 if start > 0 else 0, 'closed': False}

            def reader():
                if state['closed']:
                    raise RuntimeError('reader closed')
                time.sleep(.01)
                state['level'] = min(250, state['level']+1)
                return np.full((96, 80, 3), state['level'], dtype=np.uint8)

            reader.close = lambda: state.update(closed=True)
            reader.paced = True
            return reader

        real_values = v7_live._values
        real_encode = v7_live._encode_pulse_frame_coeffs

        def recording_values(model, frame, *args, **kwargs):
            frames_seen.append(int(np.asarray(frame)[0, 0, 0]))
            return real_values(model, frame, *args, **kwargs)

        def recording_encode(*args, **kwargs):
            source_indices.append(kwargs['source_index'])
            return real_encode(*args, **kwargs)

        def wait_for(count, timeout=20.0):
            deadline = time.monotonic()+timeout
            while len(frames_seen) < count and time.monotonic() < deadline:
                time.sleep(.005)
            return len(frames_seen) >= count

        pipe = ControlPipe()
        marks = {}

        def operator():
            # frames_seen[0] is the sender's warm-up frame (level 0).
            if not wait_for(5):
                return
            pipe.send(transport='pause')
            marks['pause_sent'] = len(frames_seen)
            wait_for(marks['pause_sent']+9)
            marks['paused_until'] = len(frames_seen)
            marks['packets_at_pause_end'] = len(written)
            pipe.send(transport='play')
            wait_for(marks['paused_until']+6)
            marks['played_until'] = len(frames_seen)
            pipe.send(transport='restart')
            wait_for(marks['played_until']+6)
            marks['done'] = len(frames_seen)
            stop.set()

        stop = threading.Event()
        fake_sounddevice = types.ModuleType('sounddevice')
        fake_sounddevice.OutputStream = OutputStream
        output = io.StringIO()
        args = v7_live.parser().parse_args([
            'send', '--device', 'memory', '--source', 'video',
            '--video-source', 'clip.mp4', '--seconds', '30', '--no-log',
            '--profile', 'fold-500', '--gui-control'])

        with mock.patch.dict(sys.modules, {'sounddevice': fake_sounddevice}), \
                mock.patch.object(v7_live, '_capture', fake_capture), \
                mock.patch.object(v7_live, '_values', recording_values), \
                mock.patch.object(v7_live, '_encode_pulse_frame_coeffs',
                                  recording_encode), \
                mock.patch.object(v7_live.sys, 'stdin', pipe), \
                contextlib.redirect_stdout(output):
            worker = threading.Thread(target=operator, daemon=True)
            worker.start()

            # End the send when the operator is done (not after 30 s).
            def watch():
                stop.wait(60)
                args.seconds = .001

            threading.Thread(target=watch, daemon=True).start()
            v7_live.run_send(args)
            worker.join(timeout=5)
        pipe.lines.put('')

        self.assertIn('done', marks, (marks, frames_seen))
        # While paused the sender went on encoding and writing packets...
        paused = frames_seen[marks['pause_sent']+3:marks['paused_until']]
        self.assertGreaterEqual(len(paused), 5)
        self.assertEqual(len(set(paused)), 1, frames_seen)   # ...of one frame
        self.assertGreater(marks['packets_at_pause_end'],
                           marks['pause_sent']-1)
        # ...then played on from where it was, with a reader at a position.
        self.assertGreater(len(opened), 2)
        self.assertEqual(opened[0], 0.0)
        self.assertGreater(opened[1], 0.0)
        resumed = frames_seen[marks['paused_until']:marks['played_until']]
        self.assertTrue(any(level > 100 for level in resumed), frames_seen)
        # Restart reads from the beginning again (levels below 100).
        self.assertEqual(opened[-1], 0.0)
        restarted = frames_seen[marks['played_until']+3:marks['done']]
        self.assertTrue(restarted and restarted[-1] < 100, frames_seen)
        # The picture index in the packets counts on through all of it.
        sent = source_indices[1:]                # [0] is the warm-up packet
        self.assertGreaterEqual(len(sent), marks['done']-2)
        self.assertEqual(sent, list(range(sent[0], sent[0]+len(sent))))
        # Position, duration and the paused state are reported as status.
        reports = [json.loads(line) for line in output.getvalue().splitlines()
                   if line.startswith('{"status": "playback"')]
        self.assertTrue(any(report['paused'] for report in reports))
        self.assertFalse(reports[0]['paused'])
        self.assertFalse(reports[-1]['paused'])
        self.assertTrue(all(report['position'] >= 0 for report in reports))

    def test_other_sources_get_no_transport(self):
        for source, video in (('test', None), ('camera', None),
                              ('video', 'rtsp://camera.example/live'),
                              ('video', 'https://media.example/clip.mp4')):
            args = SimpleNamespace(source=source, video_source=video,
                                   video_live=False)
            self.assertIsNone(v7_live._file_playback(
                args, 0.0, mock.Mock(side_effect=AssertionError)), source)
        args = SimpleNamespace(source='video', video_source='clip.mp4',
                               video_live=False)
        reader = mock.Mock()
        playback = v7_live._file_playback(args, 0.0, None, reader=reader)
        self.assertIsInstance(playback, FilePlayback)
        self.assertIsNone(playback.duration)     # no such file to probe
        self.assertIsNone(playback.fps)
        # The file's frame rate and count drive the position.
        with mock.patch('tools.v7_capture.probe_frames',
                        return_value=(25.0, 250)), \
                mock.patch('tools.v7_capture.probe_duration',
                           return_value=10.0):
            playback = v7_live._file_playback(args, 0.0, None, reader=reader)
        self.assertEqual((playback.fps, playback.duration), (25.0, 10.0))
        for _ in range(260):
            playback()
        self.assertAlmostEqual(playback.position, .4)


class GuiPreviewSettingTests(unittest.TestCase):
    def setUp(self):
        self.devices = (OutputDevice(3, 'Test output', 2, 48000),)
        self.sd = mock.Mock()
        self.sd.check_output_settings.return_value = None

    def _restored(self, saved):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'sender.json'
            path.write_text(json.dumps({'version': 5, 'settings': saved}),
                            encoding='utf-8')
            return SenderGui(self.devices, preference_path=path,
                             restore_preferences=True)

    def test_old_preview_switches_migrate_to_the_one_setting(self):
        for saved, expected in (
                ({'image_preview': True, 'video_preview': True}, 'window'),
                ({'image_preview': True, 'video_preview': False}, 'window'),
                ({'image_preview': False, 'video_preview': True}, 'popout'),
                ({'video_preview': True}, 'popout'),
                ({'image_preview': False, 'video_preview': False}, 'off'),
                ({'encoded_preview': True}, 'window'),
                ({'encoded_preview': True, 'video_preview': True}, 'window'),
                ({'encoded_preview': False, 'video_preview': True},
                 'popout'),
                ({'image_preview': 'yes', 'video_preview': 1}, 'off'),
                ({}, 'off'),
                # The old External player value is the pop-out now.
                ({'preview': 'external'}, 'popout'),
                ({'preview': 'external', 'image_preview': True}, 'popout'),
                ({'preview': 'external', 'video_preview': False}, 'popout'),
                ({'preview': 'popout'}, 'popout'),
                ({'preview': 'window'}, 'window'),
                ({'preview': 'sideways', 'video_preview': True}, 'popout')):
            with self.subTest(saved=saved):
                gui = self._restored(saved)
                self.assertEqual(gui.settings['preview'], expected)
                self.assertNotIn('image_preview', gui.settings)
                self.assertNotIn('video_preview', gui.settings)
                self.assertNotIn('encoded_preview', gui.settings)
                # It renders and saves without error.
                gui._canvas((960, 720))
                gui._persist_preferences()

    def test_migration_also_applies_to_a_plain_settings_dict(self):
        target = {'preview': 'off', 'preview_stage': 'resized'}
        _restore_sender_settings(target, {'video_preview': True})
        self.assertEqual(target['preview'], 'popout')
        _restore_sender_settings(target, {'preview': 'external'})
        self.assertEqual(target['preview'], 'popout')

    def test_the_setting_is_saved_and_restored(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'sender.json'
            gui = SenderGui(self.devices, preference_path=path)
            gui._assign('preview', 'popout')
            saved = json.loads(path.read_text(encoding='utf-8'))['settings']
            self.assertEqual(saved['preview'], 'popout')
            self.assertNotIn('image_preview', saved)
            self.assertNotIn('video_preview', saved)
            restored = SenderGui(self.devices, preference_path=path,
                                 restore_preferences=True)
        self.assertEqual(restored.settings['preview'], 'popout')

    def test_each_choice_builds_the_one_preview_channel_or_none(self):
        gui = SenderGui(self.devices)
        expected = {'off': False, 'window': True, 'popout': True}
        self.assertEqual({value for _label, value in gui._choices('preview')},
                         set(expected))
        self.assertEqual(dict((value, label) for label, value in
                              gui._choices('preview'))['popout'],
                         'Pop-out window')
        for source in ('video', 'test'):
            gui.settings.update(device=3, source=source, profile='fold-500',
                                video_source='clip.mp4')
            for choice, channel in expected.items():
                with self.subTest(source=source, choice=choice):
                    gui.settings['preview'] = choice
                    command = build_command(gui.settings, self.devices,
                                            self.sd, image_preview_port=5005)
                    # The separate player is gone for every choice.
                    self.assertNotIn('--preview', command)
                    self.assertFalse(any('ffplay' in str(part)
                                         for part in command))
                    self.assertEqual('--image-preview-port' in command,
                                     channel)
                    args = v7_live.parser().parse_args(command[2:])
                    self.assertEqual(args.image_preview_port,
                                     5005 if channel else None)
                    self.assertFalse(hasattr(args, 'preview'))

    def test_the_sender_no_longer_has_a_player_flag(self):
        with contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit):
            v7_live.parser().parse_args(
                ['send', '--device', '3', '--source', 'video',
                 '--video-source', 'clip.mp4', '--preview'])
        self.assertFalse((ROOT/'tools'/'v7_video_preview.py').exists())

    def _started(self, choice, popen):
        gui = SenderGui(self.devices)
        gui.settings.update(device=3, source='test', preview=choice)
        with mock.patch.object(gui, '_build_command',
                               return_value=['python', 'send']) as build, \
                mock.patch('tools.v7_send_gui.subprocess.Popen', popen), \
                mock.patch('tools.v7_send_gui.threading.Thread'):
            gui._start()
        self.addCleanup(gui._close_preview_socket)
        return gui, build

    def test_start_launches_one_window_at_most_and_never_a_player(self):
        for choice, has_socket, popout in (('off', False, False),
                                           ('window', True, False),
                                           ('popout', True, True)):
            with self.subTest(choice=choice):
                children = [mock.Mock(name='sender'), mock.Mock(name='popout')]
                popen = mock.Mock(side_effect=list(children))
                gui, build = self._started(choice, popen)
                self.assertEqual('image_preview_port' in build.call_args.kwargs,
                                 has_socket)
                self.assertEqual(gui._preview_socket is not None, has_socket)
                commands = [call.args[0] for call in popen.call_args_list]
                self.assertEqual(commands[0], ['python', 'send'])
                self.assertEqual(len(commands), 2 if popout else 1)
                self.assertIs(gui.process, children[0])
                self.assertFalse(any('ffplay' in str(part)
                                     for command in commands
                                     for part in command))
                if popout:
                    self.assertEqual(commands[1], [
                        sys.executable,
                        str(ROOT/'tools'/'v7_preview_popout.py')])
                    self.assertIs(gui.popout, children[1])
                    # Never the pane as well.
                    gui.page = 'live'
                    with mock.patch('PIL.ImageDraw.ImageDraw.text') as text:
                        gui._canvas((960, 720))
                    self.assertNotIn('Preview', [call.args[1] for call in
                                                 text.call_args_list])
                else:
                    self.assertIsNone(gui.popout)

    def test_a_popout_that_cannot_start_does_not_stop_the_send(self):
        sender = mock.Mock(name='sender')
        popen = mock.Mock(side_effect=[sender, OSError('no interpreter')])
        gui, _build = self._started('popout', popen)
        self.assertIs(gui.process, sender)
        self.assertIsNone(gui.popout)
        self.assertIn('Pop-out window unavailable', gui.notice)
        self.assertIn('sending without a preview', gui.notice)
        sender.send_signal.assert_not_called()
        sender.terminate.assert_not_called()

    def test_a_popout_without_a_display_reports_and_sending_goes_on(self):
        sender, window = mock.Mock(name='sender'), mock.Mock(name='popout')
        gui, _build = self._started('popout',
                                    mock.Mock(side_effect=[sender, window]))
        window.stdout = [json.dumps({
            'status': 'popout_error',
            'message': 'Pop-out window unavailable: no display'})+'\n']
        gui._read_popout(window)             # the reader thread's work
        gui._drain_events()
        self.assertIsNone(gui.popout)
        self.assertIsNone(gui._popout_address)
        self.assertIs(gui.process, sender)
        self.assertEqual(gui.notice, 'Pop-out window unavailable: no display')
        sender.send_signal.assert_not_called()
        # Pictures that still arrive are dropped quietly.
        gui.preview_datagrams['resized'] = (1, b'datagram')
        gui._forward_popout()
        gui.page = 'live'
        gui._canvas((960, 720))

    def test_closing_the_popout_window_leaves_the_sender_running(self):
        sender, window = mock.Mock(name='sender'), mock.Mock(name='popout')
        gui, _build = self._started('popout',
                                    mock.Mock(side_effect=[sender, window]))
        window.stdout = [json.dumps({'status': 'popout_ready',
                                     'port': 5006})+'\n']
        gui._read_popout(window)             # ready, then the window closed
        self.assertEqual(gui._popout_address, ('127.0.0.1', 5006))
        gui._drain_events()
        self.assertIsNone(gui.popout)
        self.assertIs(gui.process, sender)
        self.assertEqual(gui.notice, 'Pop-out window closed; still sending.')

    def test_the_popout_closes_with_the_sender_and_reopens_on_start(self):
        first = [mock.Mock(name='sender'), mock.Mock(name='popout')]
        second = [mock.Mock(name='sender 2'), mock.Mock(name='popout 2')]
        popen = mock.Mock(side_effect=first+second)
        gui, _build = self._started('popout', popen)
        with mock.patch('tools.v7_send_gui.threading.Thread') as thread:
            gui.events.put(('exit', 0))
            gui._drain_events()
        first[1].stdin.close.assert_called_once()    # it ends on end of file
        self.assertEqual(thread.call_args.kwargs['args'], (first[1],))
        self.assertIsNone(gui.popout)
        self.assertIsNone(gui._preview_socket)
        # Its own exit report afterwards changes nothing.
        gui.events.put(('popout_exit', (first[1], None)))
        gui._drain_events()
        self.assertEqual(gui.notice, 'Sender stopped.')
        with mock.patch.object(gui, '_build_command',
                               return_value=['python', 'send']), \
                mock.patch('tools.v7_send_gui.subprocess.Popen', popen), \
                mock.patch('tools.v7_send_gui.threading.Thread'):
            gui._start()
        self.assertIs(gui.popout, second[1])
        self.assertIs(gui.process, second[0])

    def test_end_popout_waits_then_terminates_a_stuck_window(self):
        window = mock.Mock()
        SenderGui._end_popout(window)
        window.terminate.assert_not_called()
        window.wait.side_effect = subprocess.TimeoutExpired('popout', 2)
        SenderGui._end_popout(window)
        window.terminate.assert_called_once()

    def test_preview_is_one_dropdown_row_on_the_setup_page(self):
        gui = SenderGui(self.devices)
        gui.settings.update(device=3, source='video', video_source='clip.mp4')
        self.assertIn('preview', gui.BASIC_FIELDS)
        self.assertIn('preview', gui.DROPDOWN_FIELDS)
        self.assertEqual(gui._visible_fields().count('preview'), 1)
        gui._canvas((960, 720))
        self.assertIn('field:preview', gui.hits)
        self.assertNotIn('field:video_preview', gui.hits)
        self.assertNotIn('field:image_preview', gui.hits)
        self.assertEqual(gui._value_label('preview'), 'Off')
        gui._open_dropdown('preview')
        gui._canvas((960, 720))
        self.assertEqual(
            len([key for key in gui.hits if key.startswith('option:')]), 3)
        gui._select_option('preview', 'window')
        self.assertEqual(gui._value_label('preview'), 'In the window')
        gui._select_option('preview', 'popout')
        self.assertEqual(gui._value_label('preview'), 'Pop-out window')


def jpeg_of(color, size=(64, 80)):
    from PIL import Image
    output = io.BytesIO()
    Image.new('RGB', size, color).save(output, format='JPEG', quality=90)
    return output.getvalue()


class PopoutFramesTests(unittest.TestCase):
    """The pop-out's picture handling, with no window."""

    def test_it_shows_the_newest_datagram_and_keeps_it_over_bad_ones(self):
        frames = PopoutFrames()
        self.assertIn('waiting', frames.title())
        self.assertFalse(frames.feed(b'not a preview datagram'))
        self.assertIsNone(frames.image)
        red = pack_preview_datagram(7, 3, 1000, 'source', jpeg_of((200, 0, 0)))
        self.assertTrue(frames.feed(red))
        self.assertEqual((frames.counter, frames.aspect, frames.stage),
                         (7, 3, 'source'))
        self.assertEqual(frames.image.size, (64, 80))
        self.assertIn('Source', frames.title())
        self.assertIn('packet 7', frames.title())
        self.assertFalse(frames.feed(
            pack_preview_datagram(8, 3, 2000, 'resized', b'broken jpeg')))
        self.assertEqual(frames.counter, 7)
        self.assertEqual(frames.received, 1)

    def test_the_picture_is_scaled_to_fit_and_centred(self):
        frames = PopoutFrames()
        empty = frames.compose((300, 200))
        self.assertEqual(empty.size, (300, 200))
        self.assertEqual(empty.getpixel((150, 100)),
                         v7_preview_popout.BACKGROUND)
        frames.feed(pack_preview_datagram(1, 0, 1, 'resized',
                                          jpeg_of((0, 180, 0))))
        for size, picture in (((300, 200), (160, 200)),   # bars at the sides
                              ((160, 400), (160, 200)),   # bars above, below
                              ((640, 800), (640, 800))):  # ten times larger
            with self.subTest(size=size):
                canvas = frames.compose(size)
                self.assertEqual(canvas.size, size)
                pixels = np.asarray(canvas)
                green = (pixels[..., 1] > 150) & (pixels[..., 0] < 40)
                rows = np.flatnonzero(green.any(axis=1))
                columns = np.flatnonzero(green.any(axis=0))
                self.assertEqual((len(columns), len(rows)), picture)
                # Centred: the bars on both sides are equal (within 1).
                self.assertLessEqual(
                    abs(columns[0]-(size[0]-1-columns[-1])), 1)
                self.assertLessEqual(abs(rows[0]-(size[1]-1-rows[-1])), 1)

    def test_no_display_is_reported_and_the_process_ends(self):
        for glfw in (SimpleNamespace(init=lambda: False,
                                     terminate=lambda: None),
                     SimpleNamespace(
                         init=mock.Mock(side_effect=RuntimeError('X11'))),
                     None):                    # the import itself fails
            output = io.StringIO()
            with mock.patch.dict(sys.modules, {'glfw': glfw,
                                               'moderngl': mock.Mock()}):
                code = v7_preview_popout.run(control=io.StringIO(''),
                                             output=output)
            self.assertEqual(code, 2)
            report = json.loads(output.getvalue())
            self.assertEqual(report['status'], 'popout_error')
            self.assertIn('Pop-out window unavailable', report['message'])


class PopoutProtocolTests(unittest.TestCase):
    """The pop-out is sent the datagrams the pane would decode."""

    def setUp(self):
        self.devices = (OutputDevice(3, 'Test output', 2, 48000),)
        self.sockets = []
        self.sender = self.socket()
        self.window = self.socket()          # stands in for the pop-out
        self.window.settimeout(2.0)

    def socket(self):
        import socket
        opened = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        opened.bind(('127.0.0.1', 0))
        self.sockets.append(opened)
        self.addCleanup(opened.close)
        return opened

    def gui(self, preview):
        gui = SenderGui(self.devices)
        gui.settings.update(device=3, source='test', preview=preview,
                            preview_stage='resized')
        gui.process = SimpleNamespace(stdin=mock.Mock(), poll=lambda: None)
        channel = self.socket()
        channel.settimeout(.05)
        gui._preview_socket = channel
        thread = threading.Thread(target=gui._read_image_preview,
                                  args=(gui.process, channel), daemon=True)
        thread.start()

        def stop():
            gui.process = None
            thread.join(timeout=2)

        self.addCleanup(stop)
        return gui, channel.getsockname()

    def wait(self, condition):
        deadline = time.monotonic()+2.0
        while not condition() and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertTrue(condition())

    def test_the_popout_gets_the_same_pictures_as_the_pane(self):
        pane, pane_port = self.gui('window')
        popout, popout_port = self.gui('popout')
        popout._popout_address = self.window.getsockname()
        frames = PopoutFrames()
        colours = {1: (200, 30, 30), 2: (30, 200, 30), 3: (30, 30, 200)}
        for counter, colour in colours.items():
            wanted = pack_preview_datagram(counter, 2, counter*1000,
                                           'resized', jpeg_of(colour))
            other = pack_preview_datagram(counter, 2, counter*1000,
                                          'source', jpeg_of((9, 9, 9)))
            for port in (pane_port, popout_port):
                self.sender.sendto(other, port)
                self.sender.sendto(wanted, port)
            # Only the selected stage is forwarded, byte for byte.
            received = self.window.recv(65507)
            self.assertEqual(received, wanted)
            self.assertTrue(frames.feed(received))
            self.wait(lambda: pane.preview_counter == counter)
            self.assertEqual(frames.counter, pane.preview_counter)
            self.assertEqual(frames.aspect, pane.preview_aspect)
            self.assertEqual(frames.stage, pane.preview_stage)
            self.assertEqual(frames.image.tobytes(),
                             pane.preview_image.tobytes())
        # The pop-out GUI decodes nothing for a pane it does not show.
        self.assertIsNone(popout.preview_image)
        # The stage switch sends the other stage's newest picture at once.
        self.wait(lambda: 'source' in popout.preview_datagrams)
        popout._assign('preview_stage', 'source')
        received = self.window.recv(65507)
        self.assertEqual(received, other)
        frames.feed(received)
        self.assertEqual((frames.stage, frames.counter), ('source', 3))
        # An older picture arriving late does not replace a newer one.
        self.sender.sendto(pack_preview_datagram(
            2, 2, 2000, 'source', jpeg_of((1, 1, 1))), popout_port)
        newest = pack_preview_datagram(4, 2, 4000, 'source',
                                       jpeg_of((250, 250, 0)))
        time.sleep(.1)
        self.sender.sendto(newest, popout_port)
        self.assertEqual(self.window.recv(65507), newest)

    def test_the_held_picture_keeps_arriving_while_paused(self):
        # Paused, the sender repeats one picture under rising packet
        # numbers: the pop-out shows each of them, as the pane does.
        popout, port = self.gui('popout')
        popout._popout_address = self.window.getsockname()
        frames = PopoutFrames()
        held = jpeg_of((120, 120, 120))
        for counter in (10, 11, 12):
            self.sender.sendto(pack_preview_datagram(
                counter, 2, counter*1000, 'resized', held), port)
            frames.feed(self.window.recv(65507))
            self.assertEqual(frames.counter, counter)
        self.assertEqual(frames.received, 3)

    def test_nothing_is_forwarded_before_the_window_reports_its_port(self):
        popout, port = self.gui('popout')
        first = pack_preview_datagram(1, 2, 1000, 'resized',
                                      jpeg_of((5, 5, 5)))
        self.sender.sendto(first, port)
        self.wait(lambda: 'resized' in popout.preview_datagrams)
        self.window.settimeout(.1)
        with self.assertRaises(OSError):
            self.window.recv(65507)
        # Once it is ready it gets the newest picture straight away.
        window = mock.Mock()
        window.stdout = [json.dumps({
            'status': 'popout_ready',
            'port': self.window.getsockname()[1]})+'\n']
        popout.popout = window
        popout._read_popout(window)
        self.window.settimeout(2.0)
        self.assertEqual(self.window.recv(65507), first)


def overlap(first, second):
    return (first[0] < second[2] and second[0] < first[2] and
            first[1] < second[3] and second[1] < first[3])


class GuiTransportTests(unittest.TestCase):
    SIZES = ((960, 720), (720, 480), (1280, 800))

    def setUp(self):
        self.devices = (OutputDevice(3, 'Test output', 2, 48000),)
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.clip = str(Path(self.folder.name)/'clip.mp4')
        Path(self.clip).write_bytes(b'video'*100)
        self.preferences = Path(self.folder.name)/'sender.json'

    def gui(self, restore=False, **settings):
        gui = SenderGui(self.devices, preference_path=self.preferences,
                        restore_preferences=restore)
        gui.settings.update(device=3, source='video', video_source=self.clip,
                            preview='window', profile='fold-500')
        gui.settings.update(settings)
        gui.page = 'live'
        return gui

    def running(self, **settings):
        gui = self.gui(**settings)
        gui.process = SimpleNamespace(stdin=mock.Mock(),
                                      poll=lambda: None)
        return gui

    def click(self, gui, key, fraction=.5):
        rect = gui.hits[key]
        position = (rect[0]+(rect[2]-rect[0])*fraction, (rect[1]+rect[3])/2)
        glfw = SimpleNamespace(MOUSE_BUTTON_LEFT=1, PRESS=1, RELEASE=0,
                               get_cursor_pos=lambda _window: position)
        gui._on_mouse(glfw, None, 1, 1, 0)
        gui._on_mouse(glfw, None, 1, 0, 0)

    def mouse(self, gui, action, fraction):
        """Press (1) or release (0) on the seek bar's last drawn track."""
        rect = self.track
        position = (rect[0]+(rect[2]-rect[0])*fraction, (rect[1]+rect[3])/2)
        glfw = SimpleNamespace(MOUSE_BUTTON_LEFT=1, PRESS=1, RELEASE=0,
                               get_cursor_pos=lambda _window: position)
        gui._on_mouse(glfw, None, 1, action, 0)

    def drag(self, gui, fraction):
        rect = self.track
        gui._on_cursor(None, rect[0]+(rect[2]-rect[0])*fraction, rect[1]+3)

    def sent(self, gui):
        return [json.loads(call.args[0])
                for call in gui.process.stdin.write.call_args_list]

    def report(self, gui, position, duration=60.0, paused=False):
        gui.events.put(('line', json.dumps({
            'status': 'playback', 'position': position,
            'duration': duration, 'paused': paused})))
        gui._drain_events()

    TRANSPORT = ('transport:play_pause', 'transport:restart',
                 'transport:seek')

    def test_controls_are_shown_only_for_a_video_file(self):
        hidden = (
            dict(source='camera'), dict(source='screen'),
            dict(source='test'), dict(source='mouse-follow'),
            dict(source='video', video_source='rtsp://camera.example/live'),
            dict(source='video', video_source='https://media.example/a.mp4',
                 video_live=True),
            dict(source='video', video_source='https://media.example/a.mp4'),
            dict(source='video', video_source=''),
        )
        for preview in ('window', 'off', 'popout'):
            for settings in hidden:
                with self.subTest(preview=preview, **settings):
                    gui = self.gui(preview=preview, **settings)
                    gui._canvas((960, 720))
                    for key in self.TRANSPORT:
                        self.assertNotIn(key, gui.hits)
            with self.subTest(preview=preview, source='file'):
                gui = self.gui(preview=preview)
                gui._canvas((960, 720))
                for key in self.TRANSPORT:
                    self.assertIn(key, gui.hits)
                # Not on the Setup page.
                gui.page = 'setup'
                gui._canvas((960, 720))
                for key in self.TRANSPORT:
                    self.assertNotIn(key, gui.hits)

    def test_the_stage_switch_stays_on_the_pane(self):
        gui = self.gui()
        gui._canvas((960, 720))
        self.assertIn('preview_stage:source', gui.hits)
        self.assertIn('preview_stage:resized', gui.hits)
        self.click(gui, 'preview_stage:source')
        self.assertEqual(gui.settings['preview_stage'], 'source')
        gui = self.gui(preview='off')
        gui._canvas((960, 720))
        self.assertNotIn('preview_stage:source', gui.hits)

    def test_the_popout_has_the_same_stage_switch_on_the_live_page(self):
        gui = self.running(preview='popout')
        gui._popout_address = ('127.0.0.1', 5006)
        gui._preview_socket = mock.Mock()
        gui.preview_datagrams = {'source': (5, b'source picture'),
                                 'resized': (5, b'encoder input')}
        gui._canvas((960, 720))
        self.click(gui, 'preview_stage:source')
        self.assertEqual(gui.settings['preview_stage'], 'source')
        gui._preview_socket.sendto.assert_called_once_with(
            b'source picture', ('127.0.0.1', 5006))
        gui._canvas((960, 720))
        self.click(gui, 'preview_stage:resized')
        self.assertEqual(gui._preview_socket.sendto.call_args.args[0],
                         b'encoder input')
        gui._preview_socket = None

    def test_hit_rectangles_fit_the_window_and_do_not_overlap(self):
        for size in self.SIZES:
            for preview in ('window', 'off', 'popout'):
                with self.subTest(size=size, preview=preview):
                    gui = self.running(preview=preview)
                    self.report(gui, 20.0)
                    image = gui._canvas(size)
                    width, height = image.size
                    keys = list(self.TRANSPORT)
                    if preview != 'off':
                        keys += ['preview_stage:source',
                                 'preview_stage:resized']
                    rects = [gui.hits[key] for key in keys]
                    for key, rect in zip(keys, rects):
                        self.assertGreater(rect[2]-rect[0], 20, key)
                        self.assertGreaterEqual(rect[3]-rect[1], 24, key)
                        self.assertGreaterEqual(rect[0], 0, key)
                        self.assertLessEqual(rect[2], width-16, key)
                        # Above the footer, below the toolbar.
                        self.assertLessEqual(rect[3], height-34, key)
                        self.assertGreater(rect[1], gui.TOOLBAR, key)
                    for index, first in enumerate(rects):
                        for second in rects[index+1:]:
                            self.assertFalse(overlap(first, second),
                                             (first, second))
                    # The three controls share one row, left to right.
                    play, restart, seek = rects[:3]
                    self.assertEqual(play[1], restart[1])
                    self.assertLess(play[2], restart[0]+1)
                    self.assertLess(restart[2], seek[0])
                    if preview == 'window':
                        self.assertGreaterEqual(play[0], int(width*.51))

    def test_play_pause_click_sends_pause_then_play(self):
        gui = self.running()
        self.report(gui, 3.0)
        gui._canvas((960, 720))
        self.click(gui, 'transport:play_pause')
        self.assertEqual(self.sent(gui), [{'transport': 'pause'}])
        self.assertIn('still sending', gui.notice)
        self.report(gui, 3.2, paused=True)
        image = gui._canvas((960, 720))
        self.assertEqual(image.size, (960, 720))
        self.click(gui, 'transport:play_pause')
        self.assertEqual(self.sent(gui)[-1], {'transport': 'play'})
        gui.process.stdin.flush.assert_called()

    def test_restart_click_sends_restart_and_forgets_the_position(self):
        gui = self.running()
        self.report(gui, 30.0)
        gui._canvas((960, 720))
        self.click(gui, 'transport:restart')
        self.assertEqual(self.sent(gui), [{'transport': 'restart'}])
        self.assertEqual(gui._transport_state()[0], 0.0)
        self.assertEqual(resume_position(gui.resume_positions, self.clip), 0.0)

    def test_seek_bar_click_sends_the_position_under_the_pointer(self):
        gui = self.running()
        self.report(gui, 3.0, duration=80.0)
        gui._canvas((960, 720))
        self.click(gui, 'transport:seek', .25)
        self.assertEqual(self.sent(gui),
                         [{'transport': 'seek', 'position': 20.0}])
        self.click(gui, 'transport:seek', 0.0)
        self.assertEqual(self.sent(gui)[-1]['position'], 0.0)
        # Without a known length there is nothing to seek against.
        self.report(gui, 3.0, duration=None)
        gui._canvas((960, 720))
        self.click(gui, 'transport:seek', .5)
        self.assertEqual(len(self.sent(gui)), 2)

    def test_dragging_the_seek_bar_sends_one_seek_on_release(self):
        gui = self.running()
        self.report(gui, 8.0, duration=80.0)
        gui._canvas((960, 720))
        self.track = gui.hits['transport:seek']
        self.mouse(gui, 1, .25)                  # press
        self.assertEqual(self.sent(gui), [])
        self.assertAlmostEqual(gui._transport_state()[0], 20.0, delta=.2)
        for fraction, position in ((.5, 40.0), (.9, 72.0), (.75, 60.0)):
            gui.dirty = False
            self.drag(gui, fraction)
            self.assertTrue(gui.dirty)           # the bar is redrawn
            self.assertAlmostEqual(gui._transport_state()[0], position,
                                   delta=.2)
            # The sender's reports do not pull the bar away from the pointer.
            self.report(gui, 9.0, duration=80.0)
            self.assertAlmostEqual(gui._transport_state()[0], position,
                                   delta=.2)
            with mock.patch('PIL.ImageDraw.ImageDraw.text') as text:
                gui._canvas((960, 720))
            from tools.v7_send_gui import _clock_text
            shown = _clock_text(gui._transport_state()[0])+' / 1:20'
            self.assertIn(shown,
                          [call.args[1] for call in text.call_args_list])
        self.assertEqual(self.sent(gui), [])
        self.mouse(gui, 0, .75)                  # release
        sent = self.sent(gui)
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]['transport'], 'seek')
        self.assertAlmostEqual(sent[0]['position'], 60.0, delta=.2)
        self.assertIsNone(gui._seek_drag)
        # Afterwards the bar follows the sender again, and moving the
        # pointer or releasing again does nothing.
        self.report(gui, 61.0, duration=80.0)
        self.drag(gui, .1)
        self.mouse(gui, 0, .1)
        self.assertEqual(gui._transport_state()[0], 61.0)
        self.assertEqual(len(self.sent(gui)), 1)

    def test_a_drag_past_the_ends_of_the_bar_is_clamped(self):
        gui = self.running()
        self.report(gui, 8.0, duration=80.0)
        gui._canvas((960, 720))
        self.track = gui.hits['transport:seek']
        self.mouse(gui, 1, .5)
        self.drag(gui, -3.0)
        self.assertEqual(gui._transport_state()[0], 0.0)
        self.drag(gui, 4.0)
        self.assertEqual(gui._transport_state()[0], 80.0)
        self.mouse(gui, 0, 4.0)
        self.assertEqual(self.sent(gui),
                         [{'transport': 'seek', 'position': 79.75}])

    def test_a_drag_ends_when_the_sender_stops(self):
        gui = self.running()
        self.report(gui, 8.0, duration=80.0)
        gui._canvas((960, 720))
        self.track = gui.hits['transport:seek']
        self.mouse(gui, 1, .5)
        gui.events.put(('exit', 0))
        gui._drain_events()
        self.assertIsNone(gui._seek_drag)

    def test_controls_send_nothing_when_the_sender_is_not_running(self):
        gui = self.gui()
        gui._canvas((960, 720))
        self.click(gui, 'transport:play_pause')
        self.assertIn('Start', gui.notice)
        self.assertIsNone(gui.process)

    def test_playback_reports_update_the_bar_and_stay_out_of_the_log(self):
        gui = self.running()
        gui.events.put(('line', 'V7 send ready'))
        gui._drain_events()
        self.report(gui, 75.0, duration=300.0, paused=True)
        self.assertEqual(gui.lines, ['V7 send ready'])
        self.assertEqual(gui.notice, 'V7 send ready')
        self.assertEqual(gui._transport_state(), (75.0, 300.0, True))
        from tools.v7_send_gui import _clock_text
        self.assertEqual(_clock_text(75.0), '1:15')
        self.assertEqual(_clock_text(3725), '1:02:05')

    def test_position_is_saved_per_file_and_start_resumes_there(self):
        gui = self.running()
        self.report(gui, 12.5, duration=60.0)
        gui.events.put(('exit', 0))
        gui._drain_events()
        self.assertIsNone(gui.playback)
        saved = json.loads(self.preferences.read_text(encoding='utf-8'))
        entry = saved['resume'][os.path.abspath(self.clip)]
        self.assertEqual(entry['position'], 12.5)
        self.assertEqual(entry['duration'], 60.0)

        restored = self.gui(restore=True)
        self.assertEqual(restored._resume_position(), 12.5)
        # The bar shows where Start will resume.
        self.assertEqual(restored._transport_state(), (12.5, 60.0, False))
        child = mock.Mock()
        child.poll.return_value = None
        with mock.patch.object(restored, '_sounddevice') as sounddevice, \
                mock.patch('tools.v7_send_gui.subprocess.Popen',
                           return_value=child) as popen, \
                mock.patch('tools.v7_send_gui.threading.Thread'):
            sounddevice.return_value.check_output_settings.return_value = None
            restored._start()
        command = popen.call_args.args[0]
        self.assertEqual(command[command.index('--video-start')+1], '12.500')
        args = v7_live.parser().parse_args(command[2:])
        self.assertEqual(args.video_start, 12.5)
        restored._close_preview_socket()

        # Another file has its own (no) position.
        other = str(Path(self.folder.name)/'other.mp4')
        Path(other).write_bytes(b'other')
        self.assertEqual(
            resume_position(restored.resume_positions, other), 0.0)

    def test_restart_while_stopped_makes_start_begin_at_the_beginning(self):
        gui = self.running()
        self.report(gui, 12.5)
        gui.events.put(('exit', 0))
        gui._drain_events()
        gui._canvas((960, 720))
        self.click(gui, 'transport:restart')
        self.assertEqual(gui._resume_position(), 0.0)
        saved = json.loads(self.preferences.read_text(encoding='utf-8'))
        self.assertEqual(saved['resume'], {})
        child = mock.Mock()
        with mock.patch.object(gui, '_build_command',
                               return_value=['python', 'send']) as build, \
                mock.patch('tools.v7_send_gui.subprocess.Popen',
                           return_value=child), \
                mock.patch('tools.v7_send_gui.threading.Thread'):
            gui._start()
        self.assertNotIn('video_start', build.call_args.kwargs)
        gui._close_preview_socket()

    def test_seek_while_stopped_sets_where_start_resumes(self):
        gui = self.running()
        self.report(gui, 12.5, duration=60.0)
        gui.events.put(('exit', 0))
        gui._drain_events()
        gui._canvas((960, 720))
        self.click(gui, 'transport:seek', .5)
        self.assertAlmostEqual(gui._resume_position(), 30.0, delta=.01)

    def test_changed_missing_or_finished_files_start_at_the_beginning(self):
        gui = self.running()
        self.report(gui, 12.5, duration=60.0)
        positions = gui.resume_positions
        self.assertEqual(resume_position(positions, self.clip), 12.5)
        # At or past the end.
        key = os.path.abspath(self.clip)
        for position in (60.0, 75.0):
            ended = {key: dict(positions[key], position=position)}
            self.assertEqual(resume_position(ended, self.clip), 0.0)
        # Changed: another size, or another modification time.
        Path(self.clip).write_bytes(b'a different video file')
        self.assertEqual(resume_position(positions, self.clip), 0.0)
        Path(self.clip).write_bytes(b'video'*100)
        stat = os.stat(self.clip)
        os.utime(self.clip, (stat.st_atime, stat.st_mtime+100))
        self.assertEqual(resume_position(positions, self.clip), 0.0)
        # Missing.
        os.unlink(self.clip)
        self.assertEqual(resume_position(positions, self.clip), 0.0)
        self.assertEqual(gui._resume_position(), 0.0)
        # Malformed saved data is ignored on load.
        self.preferences.write_text(json.dumps({
            'version': 5, 'settings': {},
            'resume': {'a': 'b', 'c': {'position': 'x'}, 'd': {
                'position': float('1e999')}},
        }), encoding='utf-8')
        loaded = SenderGui(self.devices, preference_path=self.preferences,
                           restore_preferences=True)
        self.assertEqual(loaded.resume_positions, {})

    def test_stream_urls_never_get_a_start_position(self):
        gui = self.gui(video_source='https://media.example/clip.mp4')
        sd = mock.Mock()
        sd.check_output_settings.return_value = None
        command = build_command(gui.settings, self.devices, sd,
                                video_start=12.5)
        self.assertNotIn('--video-start', command)
        self.assertEqual(gui._resume_position(), 0.0)

    def test_paused_state_is_shown_in_the_status(self):
        gui = self.running()
        self.report(gui, 5.0, paused=True)
        with mock.patch('PIL.ImageDraw.ImageDraw.text') as text:
            gui._canvas((960, 720))
        drawn = [call.args[1] for call in text.call_args_list]
        self.assertIn('SENDING · PAUSED', drawn)
        self.assertTrue(any('0:05 / 1:00' == value for value in drawn))


if __name__ == '__main__':
    unittest.main()
