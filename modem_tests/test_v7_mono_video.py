"""All-fresh mono video layout and 500-class fold profile."""
import sys
import unittest
from pathlib import Path
import time
import types
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT/'test_modem_v7', ROOT/'tools'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from animation_modem import v7                                       # noqa: E402
from animation_modem.v7_live_input import LiveInput                    # noqa: E402
from common import TARGET                                                # noqa: E402
from mono_video import (FRESH_SLOTS, FOLDED_GUESTS, FOLD_SLOTS, HEAD_GROUPS,
                        MONO_VIDEO_MODE, MonoFreshFoldOffWire,
                        MonoFreshFoldWire, fresh_rank_tables)            # noqa: E402
from mono_wire import MonoWire                                            # noqa: E402
from tools import v7_live                                                  # noqa: E402
from tools.v7_source_audio import PacketAudioDelay                          # noqa: E402


PACKETS = 8


class MonoVideoWireTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = v7.load_model(TARGET, 'box')
        with Image.open(v7.REFERENCE_FIXTURE) as image:
            cls.image = image.convert('RGB')
        cls.values, _ = v7_live._values(
            cls.model, cls.image, 'box', brightness=1.0)
        cls.wire = MonoFreshFoldWire(cls.model)
        cls.mono_model = cls.wire.model_for(cls.model)
        cls.audio = cls.wire.encode(cls.model, [cls.values]*PACKETS)

    def _decode(self, audio):
        with self.wire.receiving():
            return v7.decode_pulse_stream(
                self.mono_model, audio, sample_rate=v7.RATE,
                pilot_timing='tone-seeded', frame_boundary='eof',
                state=v7.PulseState(tail_memory=False))

    def test_rank_map_is_fixed_and_every_slot_is_m_only(self):
        tables = fresh_rank_tables(self.model)
        self.assertEqual(len(tables), v7.TAIL_PHASES)
        for table in tables:
            np.testing.assert_array_equal(table, tables[0])
            self.assertEqual(int(np.count_nonzero(table >= 0)), FRESH_SLOTS)
            np.testing.assert_array_equal(
                np.sort(table[table >= 0]), np.sort(self.model.order[:FRESH_SLOTS]))
            np.testing.assert_array_equal(
                np.sort(table[:HEAD_GROUPS].ravel()),
                np.sort(self.model.order[:v7.HEAD]))
            np.testing.assert_array_equal(
                np.sort(table[HEAD_GROUPS:FRESH_SLOTS//8].ravel()),
                np.sort(self.model.order[v7.HEAD:FRESH_SLOTS]))
        self.assertTrue(all(stream == 'M'
                            for _, stream, _ in v7.GROUPS[:FRESH_SLOTS//8]))

    def test_fold_uses_current_fresh_hosts_and_next_corner_ranks(self):
        codec = self.wire._codec(self.mono_model)
        rank_by_index = np.empty(len(self.model.order), dtype=int)
        rank_by_index[self.model.order] = np.arange(len(self.model.order))

        self.assertEqual(codec.M, FOLD_SLOTS)
        self.assertEqual(codec.signature, 16)
        self.assertEqual(FOLD_SLOTS-codec.signature, FOLDED_GUESTS)
        self.assertEqual(FRESH_SLOTS-codec.signature+FOLDED_GUESTS, 1732)
        np.testing.assert_array_equal(
            np.sort(rank_by_index[codec.hosts]), np.arange(764, 1264))
        np.testing.assert_array_equal(
            np.sort(rank_by_index[codec.guest_model_indices]),
            np.arange(1264, 1764))
        self.assertEqual(codec.table()['layout'], 'mono-fresh-500')

    def test_packets_are_identical_legs_and_require_the_new_status(self):
        packet = self.audio[:v7.PULSE_FRAME]
        np.testing.assert_array_equal(packet[:, 0], packet[:, 1])
        self.assertEqual(len(self.audio), PACKETS*v7.PULSE_FRAME)

        from tone_code import decode_tone_spectrum
        windows = packet[v7.PULSE.SYNC_LEN:v7.PULSE.SYNC_LEN+v7.FRAME]
        windows = windows.reshape(v7.F, v7.SYM, 2)[:, v7.WIN:v7.WIN+v7.N]
        spectrum = np.fft.rfft(windows, axis=1)*v7._receive_rotation(
            self.mono_model)
        status = decode_tone_spectrum(spectrum, self.mono_model)
        self.assertTrue(status['valid'])
        self.assertEqual(status['status']['mode'], MONO_VIDEO_MODE)

    def test_receiver_decodes_folded_video_packets_and_unfolds_guests(self):
        results, info = self._decode(self.audio)

        self.assertEqual(len(results), PACKETS, info)
        self.assertEqual(info.get('eof_markers_validated'), PACKETS)
        self.assertTrue(all(result.status != 'lost' for result in results))
        self.assertTrue(all(result.diag.get('wire_profile') == 'mono-fresh-500'
                            for result in results))
        self.assertTrue(all(result.diag.get('mono_fold_eq') is not None
                            for result in results))
        values = self.wire.values(self.mono_model, results[-1])
        self.assertEqual(values.shape, (self.mono_model.coder.source_count,))
        self.assertTrue(np.all(np.isfinite(values)))

    def test_receiver_rejects_previous_rotating_mono_layout(self):
        previous = MonoWire(self.model)
        old_audio = previous.encode(self.model, [self.values]*3)
        results, _info = self._decode(old_audio)

        self.assertEqual(len(results), 3)
        self.assertTrue(all(result.status == 'lost' for result in results))
        self.assertTrue(all(result.diag.get('mono_profile_rejected') ==
                            'unknown_or_non_mono_status'
                            for result in results))

    def test_receiver_rejects_stereo_fold_status(self):
        from tools.v7_wire_profile import WireProfile
        stereo = WireProfile('default').encode(
            self.model, [self.values]*2, source_indices=[0, 1])
        results, _info = self._decode(stereo)

        self.assertEqual(len(results), 2)
        self.assertTrue(all(result.status == 'lost' for result in results))
        self.assertTrue(all(result.diag.get('mono_profile_rejected') ==
                            'unknown_or_non_mono_status'
                            for result in results))

    def test_all_fresh_foldoff_control_has_a_distinct_coded_status(self):
        control = MonoFreshFoldOffWire(self.model)
        audio = control.encode(self.model, [self.values]*3)
        mono_model = control.model_for(self.model)
        with control.receiving():
            results, info = v7.decode_pulse_stream(
                mono_model, audio, sample_rate=v7.RATE,
                pilot_timing='tone-seeded', frame_boundary='eof',
                state=v7.PulseState(tail_memory=False))

        self.assertEqual(len(results), 3, info)
        self.assertTrue(all(result.status != 'lost' for result in results))
        self.assertTrue(all(result.diag['pilot_timing']['coded_status_mode'] ==
                            control.status_mode for result in results))
        self.assertEqual(control.fold_slots, 0)

    def test_video_profile_refuses_missing_eof(self):
        with self.assertRaisesRegex(ValueError, 'requires EOF markers'):
            self.wire.encode(self.model, [self.values], eof_marker=False)

    def test_source_audio_is_delayed_one_packet_on_left_of_right_video(self):
        class AudioSource:
            def __init__(self):
                self.next = .1

            def read(self, count):
                values = np.full(count, self.next, dtype=np.float32)
                self.next += .1
                return values

        modem = np.zeros((8, 2), dtype=np.float32)
        modem[:, 1] = .25
        mixed = v7_live._mix_mono_video_audio(
            modem, frames=2, source=AudioSource(),
            delay=PacketAudioDelay(), video_side=1, sample_rate=48_000)

        np.testing.assert_array_equal(mixed[:4, 0], 0.0)
        np.testing.assert_allclose(mixed[4:, 0], np.full(4, .1))
        np.testing.assert_array_equal(mixed[:, 1], .25)

    def test_burst_offset_is_one_speed_scaled_packet_from_mono_header(self):
        class AudioBurst:
            def __init__(self, samples):
                self.samples = samples
                self.position = 0

            def read(self, count):
                start = self.position
                self.position += count
                return self.samples[start:self.position]

        speed = 2.0
        video = MonoFreshFoldWire(self.model, side='right').encode(
            self.model, [self.values]*2)
        modem = v7.speed_pulse_stream(video, speed)
        packet_samples = len(modem)//2
        burst_offset = 128
        burst = np.zeros(2*packet_samples, dtype=np.float32)
        burst[burst_offset:burst_offset+24] = .25
        mixed = v7_live._mix_mono_video_audio(
            modem, 2, AudioBurst(burst), PacketAudioDelay(), video_side=1,
            sample_rate=v7.RATE)

        headers = v7.pulse_frame_hits(
            mixed[:, 1:2], sample_rate=v7.RATE, direction='forward')
        decoded, info = self._decode(mixed[:, 1:2])
        burst_samples = np.flatnonzero(np.abs(mixed[:, 0]) > .2)
        self.assertEqual(len(headers), 2)
        self.assertEqual(info.get('eof_markers_validated'), 2)
        self.assertTrue(decoded[0].diag.get('displayable'))
        self.assertEqual(len(burst_samples), 24)
        offset_from_second_header = burst_samples[0]-round(headers[1][0])
        self.assertEqual(offset_from_second_header, burst_offset)
        picture_available = decoded[0].diag['eof_marker']['end']
        self.assertGreaterEqual(burst_samples[0], picture_available)
        self.assertLess(burst_samples[0]-picture_available, burst_offset+2)
        self.assertEqual(packet_samples, round(v7.PULSE_FRAME/speed))

    def test_opposite_leg_probe_flags_video_and_dual_mono_routing(self):
        wire = MonoFreshFoldWire(self.model, side='right')
        stereo = wire.encode(self.model, [self.values]*PACKETS)

        def probe_streak(samples):
            probe = v7_live._MonoChannelProbe(v7.RATE, 'forward')
            for start in range(0, len(samples), 1024):
                probe.add(samples[start:start+1024])
                probe.scan(time.monotonic())
            return probe.streak

        right_streak = probe_streak(stereo[:, 1:2])
        left_streak = probe_streak(stereo[:, 0:1])
        duplicated = np.repeat(stereo[:, 1:2], 2, axis=1)
        duplicate_streaks = [
            probe_streak(duplicated[:, index:index+1]) for index in (0, 1)
        ]

        self.assertGreaterEqual(right_streak, 2)
        self.assertLess(left_streak, 2)
        self.assertTrue(all(streak >= 2 for streak in duplicate_streaks))

    def test_auto_side_waits_for_mono_status_and_keeps_right_preferred(self):
        from tone_code import FOLD_500
        result = self._decode(self.audio[:, 1])[0][-1]

        self.assertEqual(v7_live._mono_packet_status_mode(result),
                         MONO_VIDEO_MODE)
        self.assertEqual(v7_live._coded_status_mode(
            self.audio[:, 1:2], 0, 1.0, v7.RATE), MONO_VIDEO_MODE)
        # Right starts active and is retained once its distinct mono status is
        # decoded, even when the opposite leg also has a pulse train.
        self.assertFalse(v7_live._should_try_other_mono_leg(
            MONO_VIDEO_MODE, MONO_VIDEO_MODE, 8,
            active_has_packet=True, switch_count=0))
        # A pulse-only opposite-leg candidate can be tried when the preferred
        # right leg is silent; selection is not locked until MONO_500 arrives.
        self.assertTrue(v7_live._should_try_other_mono_leg(
            None, MONO_VIDEO_MODE, 2,
            active_has_packet=False, switch_count=0))
        # A decoded non-mono status allows testing the other leg, while a weak
        # pulse streak or a packet with no decoded status does not.
        self.assertTrue(v7_live._should_try_other_mono_leg(
            FOLD_500, MONO_VIDEO_MODE, 2,
            active_has_packet=True, switch_count=0))
        self.assertFalse(v7_live._should_try_other_mono_leg(
            None, MONO_VIDEO_MODE, 2,
            active_has_packet=True, switch_count=0))
        self.assertFalse(v7_live._should_try_other_mono_leg(
            None, MONO_VIDEO_MODE, 1,
            active_has_packet=False, switch_count=0))

    def test_profile_autodetection_requires_consistent_valid_mono_status(self):
        from tone_code import FOLD_500

        self.assertEqual(v7_live._mono_fold_input_side(
            [[MONO_VIDEO_MODE, MONO_VIDEO_MODE],
             [MONO_VIDEO_MODE, MONO_VIDEO_MODE]], MONO_VIDEO_MODE), 'right')
        self.assertEqual(v7_live._mono_fold_input_side(
            [[MONO_VIDEO_MODE, MONO_VIDEO_MODE], [None, None]],
            MONO_VIDEO_MODE), 'left')
        self.assertIsNone(v7_live._mono_fold_input_side(
            [[MONO_VIDEO_MODE], [None]], MONO_VIDEO_MODE))
        self.assertIsNone(v7_live._mono_fold_input_side(
            [[MONO_VIDEO_MODE, FOLD_500], [None, None]], MONO_VIDEO_MODE))

        # Damaging the profile-bearing body makes the coded word fail closed;
        # startup then keeps its ordinary Fold-500 fallback.
        damaged = self.audio[:v7.PULSE_FRAME, 1:2].copy()
        damaged[v7.PULSE.SYNC_LEN:v7.PULSE.SYNC_LEN+v7.FRAME] = 0
        with np.errstate(divide='ignore', invalid='ignore'):
            self.assertIsNone(v7_live._coded_status_mode(
                damaged, 0, 1.0, v7.RATE))

    def test_default_receiver_dispatches_mono_and_falls_back_to_fold500(self):
        from unittest.mock import Mock

        args = v7_live.parser().parse_args(
            ['receive', '--device', 'null', '--headless'])
        args.no_log = True
        with (patch.object(v7_live, '_detect_mono_fold_side',
                           return_value='right'),
              patch.object(v7_live, '_run_receive', return_value='mono') as run):
            self.assertEqual(v7_live.run_receive(args), 'mono')
        self.assertTrue(args.experimental_mono_fold)
        self.assertEqual(args._detected_mono_video_side, 'right')
        self.assertIsInstance(run.call_args.args[2], MonoFreshFoldWire)
        self.assertEqual(run.call_args.args[2].side, 'right')

        fallback = v7_live.parser().parse_args(
            ['receive', '--device', 'null', '--headless'])
        fallback.no_log = True
        fold = Mock()
        with (patch.object(v7_live, '_detect_mono_fold_side',
                           return_value=None),
              patch.object(v7_live, '_experimental_fold', return_value=fold),
              patch.object(v7_live, '_run_receive', return_value='fold') as run):
            self.assertEqual(v7_live.run_receive(fallback), 'fold')
        self.assertIs(run.call_args.args[1], fold)
        self.assertEqual(len(run.call_args.args), 2)

    def test_startup_probe_reads_synthetic_status_and_falls_back_on_damage(self):
        class InputStream:
            def __init__(self, **kwargs):
                self.callback = kwargs['callback']

            def start(self):
                self.callback(samples, len(samples), None, None)

            def stop(self):
                pass

            def close(self):
                pass

        sounddevice = types.ModuleType('sounddevice')
        sounddevice.query_devices = lambda *_args: {
            'max_input_channels': 2, 'default_samplerate': v7.RATE}
        sounddevice.InputStream = InputStream
        args = types.SimpleNamespace(
            device=3, direction='auto', no_log=True, stop_event=None)

        def detect(audio):
            nonlocal samples
            samples = audio
            with patch.dict(sys.modules, {'sounddevice': sounddevice}), \
                    patch('tools.v7_live.time.monotonic',
                          side_effect=(0.0, 0.0, 0.0, 0.2)):
                return v7_live._detect_mono_fold_side(args, timeout=.1)

        samples = self.audio
        self.assertEqual(detect(self.audio), 'right')
        damaged = self.audio.copy()
        damaged[:, :] = 0
        self.assertIsNone(detect(damaged))

    def test_live_sender_places_input_audio_left_and_video_right(self):
        written = []

        class OutputStream:
            samplerate = float(v7.RATE)

            def __init__(self, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

            def write(self, samples):
                written.append(np.array(samples, dtype=np.float32, copy=True))

        class AudioSource:
            def __init__(self, *_args, **_kwargs):
                pass

            def wait_for_samples(self, *_args):
                return True

            def read(self, count):
                return np.full(count, .1, dtype=np.float32)

            def close(self):
                pass

        import types
        fake_sounddevice = types.ModuleType('sounddevice')
        fake_sounddevice.OutputStream = OutputStream
        with patch.dict(sys.modules, {'sounddevice': fake_sounddevice}), \
                patch.object(v7_live, '_model', return_value=self.model), \
                patch.object(v7_live, '_capture', return_value=lambda: self.image), \
                patch('tools.v7_source_audio.DeviceSourceAudio', AudioSource):
            args = v7_live.parser().parse_args([
                'send', '--device', 'memory', '--source', 'test',
                '--seconds', '.24', '--no-log', '--profile', 'mono-fold-500',
                '--source-audio', 'device', '--source-audio-device', '7'])
            v7_live.run_send(args)

        self.assertGreaterEqual(len(written), 2)
        self.assertEqual(written[0].shape[1], 2)
        self.assertGreater(float(np.max(np.abs(written[0][:, 1]))), 0.0)
        np.testing.assert_array_equal(written[0][:, 0], 0.0)
        np.testing.assert_allclose(written[1][:, 0], .1)
        self.assertGreater(float(np.max(np.abs(written[1][:, 1]))), 0.0)

    def test_embedded_video_audio_uses_the_shared_capture_clock(self):
        written = []
        captures = []

        class OutputStream:
            samplerate = float(v7.RATE)

            def __init__(self, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

            def write(self, samples):
                written.append(np.array(samples, dtype=np.float32, copy=True))

        class SharedSource:
            def __init__(self, source, sample_rate, **kwargs):
                captures.append((source, sample_rate, kwargs))
                self.has_audio = True

                def grab():
                    return MonoVideoWireTests.image

                grab.close = self.close
                grab.paced = True
                self.video_grab = grab

            def wait_for_samples(self, *_args):
                return True

            def read(self, count):
                return np.full(count, .1, dtype=np.float32)

            def close(self):
                pass

        import types
        fake_sounddevice = types.ModuleType('sounddevice')
        fake_sounddevice.OutputStream = OutputStream
        with patch.dict(sys.modules, {'sounddevice': fake_sounddevice}), \
                patch.object(v7_live, '_model', return_value=self.model), \
                patch('tools.v7_source_audio.SharedVideoAudioSource',
                      SharedSource), \
                patch.object(v7_live, '_capture',
                             side_effect=AssertionError('separate capture')):
            args = v7_live.parser().parse_args([
                'send', '--device', 'memory', '--source', 'video',
                '--video-source', 'clip.mkv', '--seconds', '.24', '--no-log',
                '--experimental-mono-fold'])
            v7_live.run_send(args)

        self.assertEqual(captures[0][0], 'clip.mkv')
        self.assertEqual(captures[0][1], v7.RATE)
        self.assertEqual(captures[0][2]['width'], args.capture_width)
        self.assertGreaterEqual(len(written), 2)
        np.testing.assert_array_equal(written[0][:, 0], 0.0)
        np.testing.assert_allclose(written[1][:, 0], .1)
        self.assertGreater(float(np.max(np.abs(written[0][:, 1]))), 0.0)

    def test_selected_output_side_leaves_the_other_leg_silent_and_decodes(self):
        for side, index, silent in (('left', 0, 1), ('right', 1, 0)):
            with self.subTest(side=side):
                sender = MonoFreshFoldWire(self.model, side=side)
                audio = sender.encode(self.model, [self.values]*3)
                self.assertGreater(float(np.max(np.abs(audio[:, index]))), 0.0)
                np.testing.assert_array_equal(audio[:, silent], 0.0)

                live_input = LiveInput(rate=v7.RATE, direction='forward')
                live_input.add(audio[:, index:index+1].copy())
                buffered = live_input.take(1.0)
                self.assertIsNotNone(buffered)
                self.assertGreaterEqual(len(live_input.pulse_hits(buffered)), 1)

                receiver = MonoFreshFoldWire(self.model, side=side)
                mono_model = receiver.model_for(self.model)
                with receiver.receiving():
                    results, info = v7.decode_pulse_stream(
                        mono_model, audio[:, index], sample_rate=v7.RATE,
                        pilot_timing='tone-seeded', frame_boundary='eof',
                        state=v7.PulseState(tail_memory=False))
                self.assertEqual(len(results), 3, info)
                self.assertTrue(all(result.status != 'lost'
                                    for result in results))

    def test_sender_and_receiver_profile_flags_are_explicit(self):
        send = v7_live.parser().parse_args([
            'send', '--device', 'null', '--source', 'test',
            '--experimental-mono-fold'])
        recv = v7_live.parser().parse_args([
            'receive', '--device', 'null', '--experimental-mono-fold'])
        self.assertTrue(send.experimental_mono_fold)
        self.assertTrue(recv.experimental_mono_fold)
        self.assertEqual(send.mono_video_side, 'right')
        self.assertEqual(recv.mono_video_side, 'auto')
        self.assertIsNone(send.source_audio)
        self.assertIsNone(send.source_audio_device)
        self.assertEqual(v7_live._fold_slots(send), 0)
        self.assertEqual(v7_live._fold_slots(recv), 0)
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            v7_live._fold_slots(type('Args', (), {
                'experimental_mono': True,
                'experimental_mono_fold': True,
                'experimental_fold': None,
                'baseline': False,
            })())


if __name__ == '__main__':
    unittest.main()
