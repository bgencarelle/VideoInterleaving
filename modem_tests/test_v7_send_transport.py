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
                              probe_duration, video_source)
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
    def setUp(self):
        self.clock = FakeClock()
        self.readers = FakeReaders()
        self.playback = FilePlayback(self.readers, duration=60.0,
                                     clock=self.clock, idle=.001)

    def test_playing_reads_the_looping_reader_and_follows_the_clock(self):
        self.assertEqual(self.readers.opened, [0.0])
        self.assertEqual(self.playback(), (0.0, 1))
        self.clock.now += 2.5
        self.assertEqual(self.playback(), (0.0, 2))
        self.assertAlmostEqual(self.playback.position, 2.5)
        self.assertEqual(self.playback.status(), {
            'position': 2.5, 'duration': 60.0, 'paused': False})
        # The looping reader wraps at the end of the file.
        self.clock.now += 60.0
        self.assertAlmostEqual(self.playback.position, 2.5)

    def test_pause_repeats_the_held_frame_and_freezes_the_position(self):
        self.playback()
        self.clock.now += 4.0
        held = self.playback()
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
        self.playback()
        self.clock.now += 4.0
        self.playback.pause()
        self.playback()
        self.clock.now += 30.0
        self.playback.play()
        self.assertEqual(self.playback(), (4.0, 1))
        self.assertEqual(self.readers.opened, [0.0, 4.0])
        self.clock.now += 1.5
        self.assertAlmostEqual(self.playback.position, 5.5)
        self.assertFalse(self.playback.paused)

    def test_seek_while_playing_reopens_at_the_position(self):
        self.playback()
        self.playback.seek(42.0)
        self.assertAlmostEqual(self.playback.position, 42.0)
        self.assertEqual(self.playback(), (42.0, 1))
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
        self.assertEqual(self.playback(), (0.0, 1))
        self.assertEqual(self.readers.opened, [0.0, 30.0, 0.0])
        self.assertAlmostEqual(self.playback.position, 0.0)

    def test_end_of_a_seek_pass_repeats_from_the_beginning(self):
        self.playback.seek(58.0)
        for index in range(1, 4):
            self.assertEqual(self.playback(), (58.0, index))
        # The pass from 58 s ended: the looping reader opens at 0.
        self.assertEqual(self.playback(), (0.0, 1))
        self.assertEqual(self.readers.opened, [0.0, 58.0, 0.0])
        self.clock.now += 1.0
        self.assertAlmostEqual(self.playback.position, 1.0)

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
            playback = FilePlayback(
                source.reopen, duration=probe_duration(clip),
                reader=source.playback_reader(), on_close=source.close)
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
                ({'image_preview': False, 'video_preview': True}, 'external'),
                ({'video_preview': True}, 'external'),
                ({'image_preview': False, 'video_preview': False}, 'off'),
                ({'encoded_preview': True}, 'window'),
                ({'encoded_preview': True, 'video_preview': True}, 'window'),
                ({'encoded_preview': False, 'video_preview': True},
                 'external'),
                ({'image_preview': 'yes', 'video_preview': 1}, 'off'),
                ({}, 'off'),
                ({'preview': 'external', 'image_preview': True}, 'external'),
                ({'preview': 'window'}, 'window'),
                ({'preview': 'sideways', 'video_preview': True}, 'external')):
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
        self.assertEqual(target['preview'], 'external')

    def test_the_setting_is_saved_and_restored(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'sender.json'
            gui = SenderGui(self.devices, preference_path=path)
            gui._assign('preview', 'external')
            saved = json.loads(path.read_text(encoding='utf-8'))['settings']
            self.assertEqual(saved['preview'], 'external')
            self.assertNotIn('image_preview', saved)
            self.assertNotIn('video_preview', saved)
            restored = SenderGui(self.devices, preference_path=path,
                                 restore_preferences=True)
        self.assertEqual(restored.settings['preview'], 'external')

    def test_each_choice_builds_at_most_one_preview(self):
        gui = SenderGui(self.devices)
        gui.settings.update(device=3, source='video', profile='fold-500',
                            video_source='clip.mp4')
        expected = {'off': (False, False), 'window': (False, True),
                    'external': (True, False)}
        self.assertEqual({value for _label, value in gui._choices('preview')},
                         set(expected))
        for choice, (player, pane) in expected.items():
            with self.subTest(choice=choice):
                gui.settings['preview'] = choice
                # Even when a preview port is offered.
                command = build_command(gui.settings, self.devices, self.sd,
                                        image_preview_port=5005)
                self.assertEqual('--preview' in command, player)
                self.assertEqual('--image-preview-port' in command, pane)
                self.assertFalse('--preview' in command and
                                 '--image-preview-port' in command)
                args = v7_live.parser().parse_args(command[2:])
                self.assertEqual(bool(args.preview), player)
                self.assertEqual(args.image_preview_port,
                                 5005 if pane else None)

    def test_external_player_is_for_video_sources_only(self):
        gui = SenderGui(self.devices)
        gui.settings.update(device=3, source='test', preview='external')
        command = build_command(gui.settings, self.devices, self.sd,
                                image_preview_port=5005)
        self.assertNotIn('--preview', command)
        self.assertNotIn('--image-preview-port', command)

    def test_start_opens_the_preview_socket_only_for_the_window_choice(self):
        for choice, has_socket in (('off', False), ('window', True),
                                   ('external', False)):
            with self.subTest(choice=choice):
                gui = SenderGui(self.devices)
                gui.settings.update(device=3, source='test', preview=choice)
                child = mock.Mock()
                child.poll.return_value = None
                with mock.patch.object(gui, '_build_command',
                                       return_value=['python', 'send']) as build, \
                        mock.patch('tools.v7_send_gui.subprocess.Popen',
                                   return_value=child), \
                        mock.patch('tools.v7_send_gui.threading.Thread'):
                    gui._start()
                self.assertEqual('image_preview_port' in build.call_args.kwargs,
                                 has_socket)
                self.assertEqual(gui._preview_socket is not None, has_socket)
                gui._close_preview_socket()

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
        glfw = SimpleNamespace(MOUSE_BUTTON_LEFT=1, PRESS=1,
                               get_cursor_pos=lambda _window: position)
        gui._on_mouse(glfw, None, 1, 1, 0)

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
        for preview in ('window', 'off', 'external'):
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
        for preview in ('off', 'external'):
            gui = self.gui(preview=preview)
            gui._canvas((960, 720))
            self.assertNotIn('preview_stage:source', gui.hits)

    def test_hit_rectangles_fit_the_window_and_do_not_overlap(self):
        for size in self.SIZES:
            for preview in ('window', 'off'):
                with self.subTest(size=size, preview=preview):
                    gui = self.running(preview=preview)
                    self.report(gui, 20.0)
                    image = gui._canvas(size)
                    width, height = image.size
                    keys = list(self.TRANSPORT)
                    if preview == 'window':
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
