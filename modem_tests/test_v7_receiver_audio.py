"""Device-free tests for receiver channel routing and audio handoff."""
import unittest

import numpy as np

from animation_modem import v7
from tools.v7_receiver_audio import (AudioPassthrough,
                                     ReceiverChannelRouter,
                                     ReceiverRuntimeOptions,
                                     resolve_device_index)


class ReceiverChannelRouterTests(unittest.TestCase):
    MONO_500 = 4
    FOLD_500 = 1

    def test_profile_candidate_cannot_disable_playing_audio(self):
        router = ReceiverChannelRouter(self.MONO_500,
                                       stereo_modes=(self.FOLD_500,))
        good = {'confirmed': True, 'channel_modes': (None, self.MONO_500)}
        router.observe_profile_decision(good, now=1.0)
        router.observe_profile_decision(good, now=1.1)
        candidate = {'confirmed': False,
                     'channel_modes': (self.FOLD_500, None)}
        for now in (1.2, 1.3):
            state = router.observe_profile_decision(candidate, now=now)
            self.assertEqual(state['audio_side'], 'left')
            self.assertEqual(state['state'], 'mono-right')
        self.assertEqual(router.last_packet, 1.1)

    def test_single_soundtrack_status_does_not_disable_audio(self):
        router = ReceiverChannelRouter(self.MONO_500)
        good = {'confirmed': True, 'channel_modes': (None, self.MONO_500)}
        router.observe_profile_decision(good, now=1.0)
        router.observe_profile_decision(good, now=1.1)
        stray = {'confirmed': True,
                 'channel_modes': (self.MONO_500, self.MONO_500)}
        state = router.observe_profile_decision(stray, now=1.2)
        self.assertEqual(state['audio_side'], 'left')
        router.observe_profile_decision(good, now=1.3)
        state = router.observe_profile_decision(stray, now=1.4)
        self.assertEqual(state['audio_side'], 'left')
        state = router.observe_profile_decision(stray, now=1.5)
        self.assertIsNone(state['audio_side'])
        self.assertEqual(state['state'], 'dual-mono')

    def test_mono_route_does_not_change_on_packet_loss(self):
        router = ReceiverChannelRouter(self.MONO_500,
                                       initial_video_side='right')
        router.observe(None, self.MONO_500, now=1.0)
        state = router.observe(None, self.MONO_500, now=1.1)
        self.assertEqual(state['state'], 'mono-right')
        self.assertEqual(state['audio_side'], 'left')

        state = router.observe(now=4.0)
        self.assertEqual(state['state'], 'mono-right')
        self.assertEqual(state['video_side'], 'right')
        self.assertEqual(state['audio_side'], 'left')
        self.assertEqual(router.sync_state(4.0, freewheel_seconds=2.0),
                         'sync-lost')

    def test_side_changes_only_after_current_expires_and_other_is_confirmed(self):
        router = ReceiverChannelRouter(self.MONO_500,
                                       initial_video_side='right')
        router.observe(None, self.MONO_500, now=0.0)
        router.observe(None, self.MONO_500, now=.1)

        # Pulses without a validated status do not permit a route change.
        state = router.observe(now=.9)
        self.assertEqual(state['video_side'], 'right')

        # The opposite leg is confirmed only after its own two distinct packet
        # statuses; current-side evidence has meanwhile expired.
        router.observe(self.MONO_500, None, now=.91)
        state = router.observe(self.MONO_500, None, now=.92)
        self.assertEqual(state['state'], 'mono-left')
        self.assertEqual(state['video_side'], 'left')
        self.assertEqual(state['audio_side'], 'right')

    def test_two_active_mono_legs_are_dual_mono_not_audio(self):
        router = ReceiverChannelRouter(self.MONO_500,
                                       initial_video_side='right')
        router.observe(self.MONO_500, self.MONO_500, now=1.0,
                       left_confirmed=True, right_confirmed=True)
        state = router.snapshot(1.0)
        self.assertEqual(state['state'], 'dual-mono')
        self.assertEqual(state['video_side'], 'both')
        self.assertIsNone(state['audio_side'])

    def test_router_recognizes_both_transmitted_mono_profile_statuses(self):
        router = ReceiverChannelRouter(
            (self.MONO_500, 5), stereo_modes=(self.FOLD_500,))

        router.observe(5, None, now=1.0, left_confirmed=True)
        state = router.observe(5, None, now=1.1, left_confirmed=True)

        self.assertEqual(state['state'], 'mono-left')
        self.assertEqual(state['video_side'], 'left')
        self.assertEqual(state['audio_side'], 'right')

    def test_two_active_fold_legs_are_stereo_and_loss_does_not_reclassify(self):
        router = ReceiverChannelRouter(self.MONO_500,
                                       stereo_modes=(self.FOLD_500,))
        router.observe(self.FOLD_500, self.FOLD_500, now=1.0)
        state = router.observe(self.FOLD_500, self.FOLD_500, now=1.1)
        self.assertEqual(state['state'], 'stereo')
        self.assertIsNone(state['audio_side'])

        state = router.observe(self.FOLD_500, now=3.0)
        self.assertEqual(state['state'], 'stereo')
        self.assertEqual(state['video_side'], 'both')
        self.assertIsNone(state['audio_side'])

    def test_freewheel_precedes_configured_sync_loss_warning_threshold(self):
        router = ReceiverChannelRouter(self.MONO_500,
                                       initial_video_side='right')
        router.observe(None, self.MONO_500, now=1.0)
        router.observe(None, self.MONO_500, now=1.1)
        self.assertEqual(router.sync_state(1.5, 2.0), 'playing')
        self.assertEqual(router.sync_state(2.0, 2.0), 'freewheeling')
        self.assertEqual(router.sync_state(3.1, 2.0), 'freewheeling')
        self.assertEqual(router.sync_state(3.101, 2.0), 'sync-lost')


class ReceiverRuntimeOptionsTests(unittest.TestCase):
    def test_live_preferences_are_thread_safe_snapshots(self):
        options = ReceiverRuntimeOptions(audio_output_device=3)
        options.update(audio_muted=True, freewheel_seconds=4.5,
                       show_sync_warning=False, audio_volume=.35)
        self.assertEqual(options.snapshot(), {
            'audio_output_device': 3,
            'audio_output_identity': None,
            'audio_input_identity': None,
            'audio_muted': True,
            'audio_volume': .35,
            'freewheel_seconds': 4.5,
            'show_sync_warning': False,
        })

    def test_passthrough_volume_rejects_values_outside_safe_range(self):
        options = ReceiverRuntimeOptions()
        for value in (-.1, 1.01, float('nan')):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'volume must be'):
                    options.update(audio_volume=value)

    def test_device_identity_resolves_after_portaudio_index_changes(self):
        class Devices:
            entries = [
                {'name': 'Other', 'hostapi': 0,
                 'max_input_channels': 2, 'max_output_channels': 2},
                {'name': 'Selected', 'hostapi': 0,
                 'max_input_channels': 2, 'max_output_channels': 2},
            ]

            @classmethod
            def query_devices(cls, *args):
                if not args:
                    return cls.entries
                return cls.entries[int(args[0])]

            @staticmethod
            def query_hostapis():
                return [{'name': 'Test backend'}]

        selected = {'name': 'Selected', 'hostapi': 'Test backend'}
        self.assertEqual(resolve_device_index(
            Devices, 0, selected, 'output'), 1)
        with self.assertRaisesRegex(ValueError, 'reselect the output device'):
            resolve_device_index(
                Devices, 0, {'name': 'Removed', 'hostapi': 'Test backend'},
                'output')


class _FakeOutputStream:
    def __init__(self, **kwargs):
        self.callback = kwargs['callback']
        self.kwargs = kwargs
        self.samplerate = kwargs['samplerate']
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        pass

    def close(self):
        pass


class _FakeSoundDevice:
    OutputStream = _FakeOutputStream

    @staticmethod
    def query_devices(_device, _kind):
        return {'max_output_channels': 2, 'default_samplerate': 1000}

    @staticmethod
    def check_output_settings(**_kwargs):
        pass


class _Fake44100SoundDevice(_FakeSoundDevice):
    @staticmethod
    def query_devices(_device, _kind):
        return {'max_output_channels': 2, 'default_samplerate': 44100}


class AudioPassthroughTests(unittest.TestCase):

    def test_continuous_capture_audio_does_not_wait_for_decoded_frames(self):
        passthrough = AudioPassthrough(
            1000, sounddevice_module=_FakeSoundDevice)
        passthrough.open(7)
        self.assertEqual(
            passthrough.reader.correction_limit,
            AudioPassthrough.CLOCK_CORRECTION_LIMIT)
        passthrough.set_route('right')
        first = np.array([[.1, .2], [.3, .4], [.5, .6]], dtype=np.float32)
        second = np.array([[.7, .8], [.9, 1.0]], dtype=np.float32)

        self.assertTrue(passthrough.queue_capture(0, first))
        self.assertTrue(passthrough.queue_capture(3, second))

        self.assertEqual(passthrough.buffer.available, 5)
        np.testing.assert_allclose(
            passthrough.buffer.read(5), [.2, .4, .6, .8, 1.0])
        self.assertTrue(passthrough.stream.started)
        passthrough.close()

    def test_negotiated_input_rate_updates_clock_matched_target(self):
        passthrough = AudioPassthrough(
            1000, sounddevice_module=_FakeSoundDevice)
        passthrough.open(7)
        old_reader = passthrough.reader

        passthrough.set_input_rate(1200)

        self.assertIsNot(passthrough.reader, old_reader)
        self.assertEqual(passthrough.reader.nominal_ratio, 1.2)
        self.assertEqual(passthrough.reader.target_samples, 96)
        self.assertEqual(passthrough.target_seconds, .08)
        passthrough.close()

    def test_output_waits_for_jitter_buffer_then_streams_continuously(self):
        passthrough = AudioPassthrough(
            1000, sounddevice_module=_FakeSoundDevice)
        passthrough.open(7)
        passthrough.set_route('right')
        passthrough.queue_capture(
            0, np.full((40, 2), .2, dtype=np.float32))

        early = np.empty((20, 2), dtype=np.float32)
        passthrough.stream.callback(early, len(early), None, None)
        np.testing.assert_array_equal(early, 0.0)
        self.assertEqual(passthrough.buffer.available, 40)

        passthrough.queue_capture(
            40, np.full((40, 2), .2, dtype=np.float32))
        first = np.empty((20, 2), dtype=np.float32)
        second = np.empty((20, 2), dtype=np.float32)
        passthrough.stream.callback(first, len(first), None, None)
        passthrough.stream.callback(second, len(second), None, None)

        self.assertGreater(float(np.min(first)), .19)
        self.assertGreater(float(np.min(second)), .19)
        self.assertFalse(passthrough.reader.underflow)
        passthrough.close()

    def test_audio_stats_distinguish_underflow_from_portaudio_xruns(self):
        passthrough = AudioPassthrough(
            1000, sounddevice_module=_FakeSoundDevice, target_seconds=.004)
        passthrough.open(7)
        passthrough.set_route('right')
        passthrough.note_input_status('input overflow')
        passthrough.queue_capture(
            0, np.full((40, 2), .2, dtype=np.float32))
        output = np.empty((20, 2), dtype=np.float32)
        passthrough.stream.callback(output, len(output), None, None)
        output = np.empty((40, 2), dtype=np.float32)
        passthrough.stream.callback(
            output, len(output), None, 'output underflow')

        stats = passthrough.stats_snapshot()
        self.assertEqual(stats['underflow_events'], 1)
        self.assertEqual(stats['underflow_callbacks'], 1)
        self.assertEqual(stats['input_status'], {'input overflow': 1})
        self.assertEqual(stats['output_status'], {'output underflow': 1})
        self.assertEqual(stats['dropped_samples'], 0)
        self.assertIn('current_clock_correction_ppm', stats)
        self.assertIn('max_clock_correction_ppm', stats)
        passthrough.close()

    def test_frame_audio_uses_the_decoded_input_interval_and_selected_leg(self):
        passthrough = AudioPassthrough(
            1000, sounddevice_module=_FakeSoundDevice)
        passthrough.set_route('right')
        block = np.array([[.0, .1], [.2, .3], [.4, .5], [.6, .7],
                          [.8, .9], [1.0, 1.1]], dtype=np.float32)
        passthrough.record_capture(100, block)

        queued = passthrough.queue_frame(102, 3, 'right')

        self.assertTrue(queued)
        np.testing.assert_allclose(
            passthrough.buffer.read(3), [.5, .7, .9], atol=1e-7)

    def test_frame_audio_is_not_queued_if_its_capture_window_was_dropped(self):
        passthrough = AudioPassthrough(
            1000, sounddevice_module=_FakeSoundDevice)
        passthrough.set_route('left')
        passthrough.record_capture(
            100, np.ones((4, 2), dtype=np.float32))

        queued = passthrough.queue_frame(102, 4, 'left')

        self.assertFalse(queued)
        self.assertEqual(passthrough.buffer.available, 0)

    def test_queue_underflow_fades_in_when_audio_resumes(self):
        passthrough = AudioPassthrough(
            1000, sounddevice_module=_FakeSoundDevice, target_seconds=.02)
        passthrough.open(7)
        passthrough.set_route('right')
        passthrough.record_capture(
            0, np.full((100, 2), .1, dtype=np.float32))
        self.assertTrue(passthrough.queue_frame(0, 60, 'right'))

        steady = np.empty((40, 2), dtype=np.float32)
        passthrough.stream.callback(steady, len(steady), None, None)
        self.assertGreater(float(np.min(steady)), .09)

        interrupted = np.empty((40, 2), dtype=np.float32)
        passthrough.stream.callback(
            interrupted, len(interrupted), None, None)
        self.assertTrue(np.all(interrupted[21:] == 0.0))
        self.assertLess(float(np.max(np.abs(np.diff(interrupted[:, 0])))), .04)
        self.assertTrue(passthrough._fade_in_after_underflow)

        self.assertTrue(passthrough.queue_frame(60, 40, 'right'))
        resumed = np.empty((20, 2), dtype=np.float32)
        passthrough.stream.callback(resumed, len(resumed), None, None)

        self.assertEqual(resumed[0, 0], 0.0)
        self.assertLess(float(np.max(np.abs(np.diff(resumed[:, 0])))), .04)
        self.assertFalse(passthrough._fade_in_after_underflow)
        passthrough.close()

    def test_gap_between_decoded_intervals_is_crossfaded_in_place(self):
        passthrough = AudioPassthrough(
            1000, sounddevice_module=_FakeSoundDevice, target_seconds=.02)
        passthrough.open(7)
        passthrough.set_route('right')
        audio = np.zeros((80, 2), dtype=np.float32)
        audio[:20, 1] = -.1
        audio[20:40, 1] = .4
        audio[40:, 1] = .1
        passthrough.record_capture(0, audio)

        self.assertTrue(passthrough.queue_frame(0, 20, 'right'))
        self.assertTrue(passthrough.queue_frame(40, 20, 'right'))
        self.assertTrue(passthrough.queue_frame(60, 20, 'right'))
        output = np.empty((40, 2), dtype=np.float32)
        passthrough.stream.callback(output, len(output), None, None)

        self.assertFalse(passthrough.reader.underflow)
        self.assertLess(float(np.max(np.abs(np.diff(output[:, 0])))), .04)
        self.assertEqual(passthrough.target_seconds, .02)
        passthrough.close()

    def test_v7_frame_audio_keeps_duration_and_pitch_across_device_rates(self):
        input_rate = 48000
        output_rate = 44100
        frame_samples = v7.PULSE_FRAME
        total_samples = 2*frame_samples
        source_audio = np.column_stack((
            np.zeros(total_samples+32),
            .25*np.sin(2*np.pi*1000*np.arange(total_samples+32)/input_rate)
        )).astype(np.float32)
        passthrough = AudioPassthrough(
            input_rate, sounddevice_module=_Fake44100SoundDevice,
            target_seconds=.02)
        passthrough.open(7)
        passthrough.set_input_rate(input_rate)
        passthrough.set_route('right')
        passthrough.record_capture(5000, source_audio)

        self.assertTrue(passthrough.queue_frame(
            5000, total_samples+32, 'right'))
        output_frames = round(total_samples*output_rate/input_rate)
        output = np.empty((output_frames, 2), dtype=np.float32)
        passthrough.stream.callback(output, output_frames, None, None)

        self.assertEqual(passthrough.reader.nominal_ratio,
                         input_rate/output_rate)
        self.assertAlmostEqual(output_frames/output_rate,
                               total_samples/input_rate, delta=1/output_rate)
        np.testing.assert_array_equal(output[:, 0], output[:, 1])
        spectrum = np.abs(np.fft.rfft(output[:, 0]))
        frequencies = np.fft.rfftfreq(output_frames, 1/output_rate)
        peak = frequencies[np.argmax(spectrum[1:])+1]
        self.assertAlmostEqual(peak, 1000, delta=15)
        self.assertGreater(float(np.sqrt(np.mean(output[:, 0]**2))), .15)
        passthrough.close()

    def test_mono_input_is_not_mistaken_for_the_audio_leg(self):
        passthrough = AudioPassthrough(
            1000, sounddevice_module=_FakeSoundDevice, target_seconds=.004)
        passthrough.set_route('left')

        passthrough.record_capture(
            100, np.array([[.1], [.2], [.3]], dtype=np.float32))

        self.assertEqual(passthrough.buffer.available, 0)
        self.assertFalse(passthrough.queue_frame(100, 2, 'left'))

    def test_selected_side_routes_to_output_and_mute_is_immediate(self):
        passthrough = AudioPassthrough(
            1000, sounddevice_module=_FakeSoundDevice, target_seconds=.004)
        passthrough.open(7)
        self.assertTrue(passthrough.is_open)
        self.assertEqual(passthrough.device, 7)

        passthrough.set_route('right')
        audio = np.array([[.1, .8], [.2, .6], [.3, .4], [.4, .2],
                          [.5, .1], [.6, 0.0]],
                         dtype=np.float32)
        passthrough.record_capture(0, audio)
        self.assertTrue(passthrough.queue_frame(0, 6, 'right'))
        output = np.empty((4, 2), dtype=np.float32)
        passthrough.stream.callback(output, 4, None, None)
        np.testing.assert_allclose(output[:, 0], [.8, .6, .4, .2],
                                   atol=.002)
        np.testing.assert_array_equal(output[:, 0], output[:, 1])

        passthrough.set_muted(True)
        passthrough.stream.callback(output, 4, None, None)
        np.testing.assert_array_equal(output, 0.0)
        passthrough.close()
        self.assertFalse(passthrough.is_open)

    def test_live_volume_change_scales_audio_without_reopening_output(self):
        passthrough = AudioPassthrough(
            1000, sounddevice_module=_FakeSoundDevice, target_seconds=.004)
        passthrough.open(7)
        passthrough.set_route('left')
        passthrough.set_volume(.5)
        audio = np.array([[.4, .8], [.2, .6], [0.0, .4], [-.2, .2],
                          [-.4, 0.0], [-.6, -.2]],
                         dtype=np.float32)
        passthrough.record_capture(0, audio)
        self.assertTrue(passthrough.queue_frame(0, 6, 'left'))
        output = np.empty((4, 2), dtype=np.float32)
        passthrough.stream.callback(output, 4, None, None)
        np.testing.assert_allclose(output[:, 0], [.2, .1, 0.0, -.1],
                                   atol=.002)
        self.assertEqual(passthrough.volume, .5)
        self.assertEqual(passthrough.device, 7)
        passthrough.close()

    def test_output_open_failure_is_reported_without_default_fallback(self):
        class BrokenSoundDevice(_FakeSoundDevice):
            @staticmethod
            def query_devices(_device, _kind):
                raise RuntimeError('device removed')

        passthrough = AudioPassthrough(
            1000, sounddevice_module=BrokenSoundDevice)
        passthrough.open(7)
        self.assertFalse(passthrough.is_open)
        self.assertIn('device removed', passthrough.error)


if __name__ == '__main__':
    unittest.main()
