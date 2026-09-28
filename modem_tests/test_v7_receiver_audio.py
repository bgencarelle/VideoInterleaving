"""Device-free tests for receiver channel routing and audio handoff."""
import unittest

import numpy as np

from tools.v7_receiver_audio import (AudioPassthrough,
                                     ReceiverChannelRouter,
                                     ReceiverRuntimeOptions,
                                     resolve_device_index)


class ReceiverChannelRouterTests(unittest.TestCase):
    MONO_500 = 4
    FOLD_500 = 1

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
        self.assertEqual(router.sync_state(3.1, 2.0), 'sync-lost')


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


class AudioPassthroughTests(unittest.TestCase):

    def test_selected_side_routes_to_output_and_mute_is_immediate(self):
        passthrough = AudioPassthrough(
            1000, sounddevice_module=_FakeSoundDevice)
        passthrough.open(7)
        self.assertTrue(passthrough.is_open)
        self.assertEqual(passthrough.device, 7)

        passthrough.set_route('right')
        passthrough.push(np.array([[.1, .8], [.2, .6], [.3, .4], [.4, .2]],
                                  dtype=np.float32))
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
            1000, sounddevice_module=_FakeSoundDevice)
        passthrough.open(7)
        passthrough.set_route('left')
        passthrough.set_volume(.5)
        passthrough.push(np.array([[.4, .8], [.2, .6], [0.0, .4],
                                   [-.2, .2]], dtype=np.float32))
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
