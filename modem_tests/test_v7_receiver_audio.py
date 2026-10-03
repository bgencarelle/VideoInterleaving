"""Device-free tests for receiver channel routing and audio handoff."""
import time
import unittest

import numpy as np

from animation_modem import v7
from tools.v7_receiver_audio import (AudioPassthrough,
                                     ReceiverChannelRouter,
                                     ReceiverRuntimeOptions,
                                     passthrough_audio_side,
                                     resolve_device_index)


class ReceiverChannelRouterTests(unittest.TestCase):
    MONO_500 = 4
    FOLD_500 = 1

    def test_profile_candidate_keeps_the_route_but_mutes_its_leg(self):
        router = ReceiverChannelRouter(self.MONO_500,
                                       stereo_modes=(self.FOLD_500,))
        good = {'confirmed': True, 'channel_modes': (None, self.MONO_500)}
        router.observe_profile_decision(good, now=1.0)
        router.observe_profile_decision(good, now=1.1)
        candidate = {'confirmed': False,
                     'channel_modes': (self.FOLD_500, None)}
        for now in (1.2, 1.3):
            state = router.observe_profile_decision(candidate, now=now)
            # The assignment is kept; the leg that showed a status is muted.
            self.assertEqual(router.audio_side, 'left')
            self.assertIsNone(state['audio_side'])
            self.assertTrue(state['data_muted'])
            self.assertEqual(state['state'], 'mono-right')
        self.assertEqual(router.last_packet, 1.1)

    def test_single_soundtrack_status_mutes_without_reassigning_the_leg(self):
        router = ReceiverChannelRouter(self.MONO_500)
        good = {'confirmed': True, 'channel_modes': (None, self.MONO_500)}
        router.observe_profile_decision(good, now=1.0)
        router.observe_profile_decision(good, now=1.1)
        stray = {'confirmed': True,
                 'channel_modes': (self.MONO_500, self.MONO_500)}
        state = router.observe_profile_decision(stray, now=1.2)
        self.assertEqual(router.audio_side, 'left')
        self.assertEqual(state['state'], 'mono-right')
        self.assertIsNone(state['audio_side'])
        router.observe_profile_decision(good, now=1.3)
        state = router.observe_profile_decision(stray, now=1.4)
        self.assertEqual(router.audio_side, 'left')
        self.assertIsNone(state['audio_side'])
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

    def test_mono_audio_passthrough_keeps_last_side_after_sync_loss(self):
        router = ReceiverChannelRouter(self.MONO_500)
        router.observe(None, self.MONO_500, now=1.0)
        router.observe(None, self.MONO_500, now=1.1)

        route = router.observe(now=4.0)

        self.assertEqual(router.sync_state(4.0, freewheel_seconds=2.0),
                         'sync-lost')
        self.assertEqual(passthrough_audio_side(route, 2), 'left')
        self.assertIsNone(passthrough_audio_side(route, 1))

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
        self.assertEqual(router.audio_side, 'right')
        # The right leg carried data 0.8 s ago: it is not played yet.
        self.assertIsNone(state['audio_side'])
        self.assertTrue(state['data_muted'])

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


class DataChannelNeverPassesTests(unittest.TestCase):
    """A leg that carries, or recently carried, picture data stays silent."""
    MONO_500 = 4
    FOLD_500 = 1

    def _mono_right(self, **kwargs):
        router = ReceiverChannelRouter(self.MONO_500,
                                       stereo_modes=(self.FOLD_500,), **kwargs)
        router.observe(None, self.MONO_500, now=1.0)
        route = router.observe(None, self.MONO_500, now=1.1)
        self.assertEqual(passthrough_audio_side(route, 2), 'left')
        return router

    def test_hold_is_long(self):
        self.assertGreaterEqual(ReceiverChannelRouter.DATA_HOLD_SECONDS, 10.0)

    def test_stale_side_does_not_play_the_former_data_leg(self):
        # Reported failure: the picture leg stops validating for longer than
        # the 0.75 s loss window while the other leg shows statuses (the
        # picture moved, or crosstalk). The route flipped and the leg that
        # had carried data 0.9 s earlier was played.
        router = self._mono_right()
        router.observe(self.MONO_500, None, now=2.0)
        route = router.observe(self.MONO_500, None, now=2.1)
        self.assertEqual(route['state'], 'mono-left')
        self.assertIsNone(passthrough_audio_side(route, 2))
        self.assertTrue(route['data_muted'])
        self.assertEqual(route['data_channels'], (True, True))

    def test_new_data_leg_is_silenced_at_first_sight(self):
        # The picture appears on the soundtrack leg while the old leg is still
        # inside its loss window: one status, no confirmation.
        router = self._mono_right()
        route = router.observe(self.MONO_500, None, now=1.2)
        self.assertEqual(route['state'], 'mono-right')
        self.assertIsNone(passthrough_audio_side(route, 2))

    def test_unconfirmed_profile_status_silences_its_leg(self):
        router = self._mono_right()
        candidate = {'confirmed': False,
                     'channel_modes': (self.MONO_500, None)}
        route = router.observe_profile_decision(candidate, now=1.2)
        self.assertIsNone(passthrough_audio_side(route, 2))

    def test_swap_passes_old_leg_only_after_the_hold(self):
        router = self._mono_right()
        hold = router.DATA_HOLD_SECONDS
        now = 2.0
        while now < 1.1+hold+1.0:
            route = router.observe(self.MONO_500, None, now=now)
            if now < 1.1+hold:
                self.assertIsNone(passthrough_audio_side(route, 2), now)
            now += 0.25
        self.assertEqual(route['state'], 'mono-left')
        self.assertEqual(passthrough_audio_side(route, 2), 'right')
        self.assertEqual(route['data_channels'], (True, False))
        self.assertFalse(route['data_muted'])

    def test_short_dropout_never_unmutes_a_data_leg(self):
        router = self._mono_right()
        router.observe(self.MONO_500, None, now=1.2)
        for now in (2.0, 5.0, 11.0):
            self.assertIsNone(
                passthrough_audio_side(router.snapshot(now), 2), now)

    def test_long_signal_loss_keeps_soundtrack_and_never_the_data_leg(self):
        router = self._mono_right()
        for now in (2.0, 5.0, 30.0, 600.0):
            route = router.observe(now=now)
            self.assertEqual(passthrough_audio_side(route, 2), 'left', now)
            self.assertEqual(route['video_side'], 'right')
        self.assertEqual(router.sync_state(600.0, 2.0), 'sync-lost')
        # Signal returns on the same leg: the route is unchanged.
        router.observe(None, self.MONO_500, now=601.0)
        route = router.observe(None, self.MONO_500, now=601.1)
        self.assertEqual(route['state'], 'mono-right')
        self.assertEqual(passthrough_audio_side(route, 2), 'left')

    def test_unconfirmed_soundtrack_guess_is_dropped_once_data_is_seen(self):
        # The initial side was a guess (video right, audio left). Data shows
        # on the guessed audio leg and is then lost: nothing was positively
        # the soundtrack, so nothing passes, even after the hold.
        router = ReceiverChannelRouter(self.MONO_500,
                                       initial_video_side='right')
        self.assertEqual(passthrough_audio_side(router.snapshot(0.5), 2),
                         'left')
        router.observe(self.MONO_500, None, now=1.0)
        for now in (1.0, 5.0, 60.0):
            route = router.snapshot(now)
            self.assertIsNone(passthrough_audio_side(route, 2), now)
            self.assertTrue(route['data_muted'])

    def test_stereo_picture_passes_nothing(self):
        router = ReceiverChannelRouter(self.MONO_500,
                                       stereo_modes=(self.FOLD_500,),
                                       initial_video_side='right')
        # One stereo status on one leg marks both legs as data.
        route = router.observe(self.FOLD_500, None, now=1.0)
        self.assertEqual(route['data_channels'], (True, True))
        self.assertIsNone(passthrough_audio_side(route, 2))
        router.observe(self.FOLD_500, self.FOLD_500, now=1.1)
        for now in (1.2, 5.0, 300.0):
            route = router.observe(now=now)
            self.assertIsNone(passthrough_audio_side(route, 2), now)

    def test_no_picture_ever_seen_keeps_previous_behaviour(self):
        for initial, expected in ((None, None), ('right', 'left'),
                                  ('left', 'right'), ('both', None)):
            router = ReceiverChannelRouter(self.MONO_500,
                                           stereo_modes=(self.FOLD_500,),
                                           initial_video_side=initial)
            for now in (0.0, 1.0, 100.0):
                route = router.observe(now=now)
                self.assertEqual(route['audio_side'], expected)
                self.assertEqual(route['audio_side'], router.audio_side)
                self.assertEqual(passthrough_audio_side(route, 2), expected)
                self.assertFalse(route.get('data_muted', False))
            route = router.observe_profile_decision(
                {'confirmed': False, 'channel_modes': (None, None)}, now=101.0)
            self.assertEqual(passthrough_audio_side(route, 2), expected)

    def test_hold_runs_on_audio_time_when_supplied(self):
        router = ReceiverChannelRouter(self.MONO_500)
        router.audio_time = 0.0
        router.observe(None, self.MONO_500, now=1.0)
        router.observe(None, self.MONO_500, now=1.1)
        router.observe(self.MONO_500, None, now=1.2)
        # Wall time passes with the input stopped: the leg is not cleared.
        self.assertIsNone(
            passthrough_audio_side(router.snapshot(1000.0), 2))
        router.audio_time = router.DATA_HOLD_SECONDS-0.01
        self.assertIsNone(passthrough_audio_side(router.snapshot(1000.0), 2))
        router.audio_time = router.DATA_HOLD_SECONDS
        self.assertEqual(
            passthrough_audio_side(router.snapshot(1000.0), 2), 'left')

    def test_data_leg_samples_never_reach_the_output(self):
        from unittest.mock import patch
        with patch.object(AudioPassthrough, 'OUTPUT_MODE', 'callback'):
            passthrough = AudioPassthrough(
                1000, sounddevice_module=_FakeSoundDevice)
            passthrough.open(7)
        self.addCleanup(passthrough.close)
        router = self._mono_right()
        block = np.zeros((100, 2), dtype=np.float32)
        block[:, 0] = 0.25   # soundtrack
        block[:, 1] = 0.75   # picture data
        passthrough.set_route(
            passthrough_audio_side(router.snapshot(1.1), 2), False)
        self.assertTrue(passthrough.queue_capture(0, block))
        np.testing.assert_allclose(passthrough.buffer.read(100), 0.25)
        # The route flips once the old leg expired: nothing may be queued.
        router.observe(self.MONO_500, None, now=2.0)
        route = router.observe(self.MONO_500, None, now=2.1)
        passthrough.set_route(passthrough_audio_side(route, 2), False)
        self.assertFalse(passthrough.queue_capture(100, block))
        self.assertEqual(passthrough.buffer.available, 0)
        # The explicit mute still silences a clean soundtrack route.
        passthrough.set_route('left', True)
        self.assertFalse(passthrough.queue_capture(200, block))
        self.assertEqual(passthrough.buffer.available, 0)


class ReceiverRuntimeOptionsTests(unittest.TestCase):
    def test_live_preferences_are_thread_safe_snapshots(self):
        options = ReceiverRuntimeOptions(audio_output_device=3)
        options.update(audio_muted=True, freewheel_seconds=4.5,
                       show_sync_warning=False, audio_volume=.35,
                       audio_diagnostics=True)
        self.assertEqual(options.snapshot(), {
            'audio_output_device': 3,
            'audio_output_identity': None,
            'audio_input_identity': None,
            'audio_muted': True,
            'audio_volume': .35,
            'freewheel_seconds': 4.5,
            'show_sync_warning': False,
            'audio_diagnostics': True,
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


class _Fake48000SoundDevice(_FakeSoundDevice):
    @staticmethod
    def query_devices(_device, _kind):
        return {'max_output_channels': 2, 'default_samplerate': 48000}


class _FakeBlockingStream:
    """PortAudio blocking stream stand-in: a 2,048-frame buffer drained at the
    device rate by the wall clock, like PortAudio's own playback thread."""

    CAPACITY = 2048

    def __init__(self, **kwargs):
        assert 'callback' not in kwargs
        self.kwargs = kwargs
        self.samplerate = kwargs['samplerate']
        self.latency = kwargs['latency']
        self.active = False
        self.aborted = False
        self.closed = False
        self.written = []
        self.underflows_to_report = 0
        self.write_after_close = False
        self.write_while_full = False
        self._fill = float(self.CAPACITY)  # PortAudio pre-fills with silence
        self._clock = None

    def _drain(self):
        now = time.monotonic()
        if self._clock is not None:
            self._fill = max(0.0, self._fill-(now-self._clock)*self.samplerate)
        self._clock = now

    def start(self):
        self.active = True
        self._clock = time.monotonic()

    @property
    def write_available(self):
        if self.closed:
            raise RuntimeError('stream closed')
        self._drain()
        return int(self.CAPACITY-self._fill)

    def write(self, block):
        if self.closed:
            self.write_after_close = True
            raise RuntimeError('stream closed')
        if self.aborted:
            raise RuntimeError('stream aborted')
        self._drain()
        if self._fill+len(block) > self.CAPACITY+1:
            self.write_while_full = True  # a real write would block here
        self._fill += len(block)
        self.written.append(np.array(block, copy=True))
        if self.underflows_to_report:
            self.underflows_to_report -= 1
            return True
        return False

    def abort(self):
        self.aborted = True
        self.active = False

    def stop(self):
        self.active = False

    def close(self):
        self.closed = True


class _FakeBlockingSoundDevice:
    OutputStream = _FakeBlockingStream
    HOSTAPI = 'ALSA'

    @classmethod
    def query_devices(cls, _device=None, _kind=None):
        return {'max_output_channels': 2, 'default_samplerate': 48000,
                'hostapi': 0}

    @classmethod
    def query_hostapis(cls, _index=None):
        return {'name': cls.HOSTAPI}

    @staticmethod
    def check_output_settings(**_kwargs):
        pass


class _FakeCoreAudioBlockingSoundDevice(_FakeBlockingSoundDevice):
    HOSTAPI = 'Core Audio'


class AudioPassthroughBlockingTests(unittest.TestCase):
    """Default output mode: PortAudio plays; a Python writer refills."""

    def _wait(self, predicate, timeout=2.0):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(.005)
        return predicate()

    def test_blocking_mode_is_the_default_and_opens_without_a_callback(self):
        self.assertEqual(AudioPassthrough.OUTPUT_MODE, 'blocking')
        passthrough = AudioPassthrough(
            48000, sounddevice_module=_FakeBlockingSoundDevice)
        passthrough.open(7)
        try:
            self.assertEqual(passthrough.stream.kwargs['blocksize'], 512)
            self.assertEqual(passthrough.stream.latency, .045)
            self.assertEqual(passthrough.target_seconds, .05)
            self.assertEqual(passthrough.stats_snapshot()['output_mode'],
                             'blocking')
            self.assertTrue(self._wait(lambda: passthrough.stream.written))
        finally:
            passthrough.close()
        core = AudioPassthrough(
            48000, sounddevice_module=_FakeCoreAudioBlockingSoundDevice)
        core.open(7)
        try:
            # CoreAudio's ring is next_pow2(2 x latency x rate): halve it.
            self.assertEqual(core.stream.latency, .0225)
        finally:
            core.close()
        with self.assertRaises(ValueError):
            AudioPassthrough(48000, sounddevice_module=_FakeBlockingSoundDevice,
                             output_mode='bogus')
        with self.assertRaises(ValueError):
            AudioPassthrough(48000, sounddevice_module=_FakeBlockingSoundDevice,
                             output_blocksize=0)

    def test_writer_plays_real_time_capture_audio_continuously(self):
        passthrough = AudioPassthrough(
            48000, sounddevice_module=_FakeBlockingSoundDevice)
        passthrough.open(7)
        stream = passthrough.stream
        try:
            passthrough.set_route('right')
            level = np.full((1024, 2), .25, dtype=np.float32)
            started = time.monotonic()
            for index in range(30):  # 0.64 s of capture at its real rate
                passthrough.queue_capture(index*1024, level)
                delay = started+(index+1)*1024/48000-time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            time.sleep(.1)
        finally:
            passthrough.close()
        audio = np.concatenate(stream.written)[:, 0]
        playing = np.flatnonzero(np.abs(audio-.25) < .01)
        self.assertGreater(len(playing), 20_000)
        run = audio[playing[0]:playing[-1]+1]
        # Once primed, the level plays without gaps or steps until capture
        # stops; the start ramps in from silence.
        self.assertTrue(np.all(np.abs(run-.25) < .01))
        self.assertLess(
            float(np.max(np.abs(np.diff(audio[:playing[0]+1])))), .1)
        self.assertEqual(passthrough.stats_snapshot()['latency_trims'], 0)

    def test_portaudio_underflow_from_write_is_reported(self):
        passthrough = AudioPassthrough(
            48000, sounddevice_module=_FakeBlockingSoundDevice)
        passthrough.open(7)
        try:
            passthrough.stream.underflows_to_report = 2
            self.assertTrue(self._wait(
                lambda: passthrough.stats_snapshot()['output_status'].get(
                    'output underflow') == 2))
        finally:
            passthrough.close()

    def test_close_stops_the_writer_before_closing_the_stream(self):
        passthrough = AudioPassthrough(
            48000, sounddevice_module=_FakeBlockingSoundDevice)
        passthrough.open(7)
        stream, writer = passthrough.stream, passthrough._writer_thread
        self.assertTrue(self._wait(lambda: stream.written))
        passthrough.close()
        self.assertFalse(writer.is_alive())
        self.assertTrue(stream.aborted)
        self.assertTrue(stream.closed)
        self.assertFalse(stream.write_after_close)
        self.assertFalse(stream.write_while_full)
        self.assertIsNone(passthrough.error)
        self.assertFalse(passthrough.is_open)


class AudioPassthroughTests(unittest.TestCase):
    # Most tests run at toy rates (1 kHz, a few samples per block) to check
    # routing, fades and priming; there the 30 ms capture-stall allowance
    # would dwarf the queued audio. These tests keep the real allowance.
    REAL_CAPTURE_STALL_ALLOWANCE = {
        'test_priming_waits_for_target_raised_by_large_callbacks',
        'test_pi_servo_holds_a_large_capture_offset_off_the_limit',
    }

    def setUp(self):
        from unittest.mock import patch
        # These tests drive the output callback by hand; blocking mode (the
        # default) is covered by AudioPassthroughBlockingTests.
        patcher = patch.object(AudioPassthrough, 'OUTPUT_MODE', 'callback')
        patcher.start()
        self.addCleanup(patcher.stop)
        if self._testMethodName not in self.REAL_CAPTURE_STALL_ALLOWANCE:
            patcher = patch.object(
                AudioPassthrough, 'CAPTURE_STALL_SECONDS', 0.0)
            patcher.start()
            self.addCleanup(patcher.stop)

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
        passthrough.DECLICK_SECONDS = 0.0  # start ramp tested separately
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
        for start in range(0, 40, 10):
            passthrough.queue_capture(
                start, np.full((10, 2), .2, dtype=np.float32))
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
        passthrough.DECLICK_SECONDS = 0.0  # start ramp tested separately
        passthrough.set_route('right')
        passthrough.record_capture(
            0, np.full((110, 2), .1, dtype=np.float32))
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

        # 50 samples: the servo target now covers the 40-sample callbacks seen.
        self.assertTrue(passthrough.queue_frame(60, 50, 'right'))
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
        passthrough.DECLICK_SECONDS = 0.0  # start ramp tested separately
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
                                   atol=.005)
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
        passthrough.DECLICK_SECONDS = 0.0  # start ramp tested separately
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

    def test_start_and_route_change_ramp_instead_of_stepping(self):
        passthrough = AudioPassthrough(
            1000, sounddevice_module=_FakeSoundDevice, target_seconds=.02)
        passthrough.open(7)
        passthrough.set_route('right')
        for start in range(0, 60, 10):
            passthrough.queue_capture(
                start, np.full((10, 2), .5, dtype=np.float32))

        first = np.empty((20, 2), dtype=np.float32)
        passthrough.stream.callback(first, len(first), None, None)
        self.assertLess(float(first[0, 0]), .5)
        self.assertLess(float(np.max(np.abs(np.diff(first[:, 0])))), .2)
        self.assertGreater(float(first[-1, 0]), .49)

        passthrough.set_route('left')  # router moved the audio leg
        after = np.empty((20, 2), dtype=np.float32)
        passthrough.stream.callback(after, len(after), None, None)
        self.assertGreater(float(after[0, 0]), 0.0)
        self.assertLess(float(np.max(np.abs(np.diff(after[:, 0])))), .2)
        self.assertEqual(float(after[-1, 0]), 0.0)
        passthrough.close()

    def test_route_cleared_without_reset_fades_to_silence(self):
        passthrough = AudioPassthrough(
            1000, sounddevice_module=_FakeSoundDevice, target_seconds=.02)
        passthrough.open(7)
        passthrough.DECLICK_SECONDS = 0.0
        passthrough.set_route('right')
        for start in range(0, 60, 10):
            passthrough.queue_capture(
                start, np.full((10, 2), .5, dtype=np.float32))
        block = np.empty((20, 2), dtype=np.float32)
        passthrough.stream.callback(block, len(block), None, None)
        passthrough.DECLICK_SECONDS = 0.005
        with passthrough._lock:
            passthrough.route = None  # silence without a buffer reset
        passthrough.stream.callback(block, len(block), None, None)
        self.assertGreater(float(block[0, 0]), 0.0)
        self.assertEqual(float(block[-1, 0]), 0.0)
        passthrough.close()

    def test_bursty_capture_blocks_do_not_modulate_output_pitch(self):
        rate = 48000
        passthrough = AudioPassthrough(
            rate, sounddevice_module=_Fake48000SoundDevice)
        passthrough.open(7)
        passthrough.set_route('right')
        corrections = []
        written = 0
        for step in range(3000):
            while written < (step+1)*256+3584:
                passthrough.queue_capture(
                    written, np.zeros((1024, 2), dtype=np.float32))
                written += 1024
            out = np.empty((256, 2), dtype=np.float32)
            passthrough.stream.callback(out, len(out), None, None)
            if step >= 2600:
                corrections.append(passthrough.reader.current_correction)
        # 1,024-sample arrivals against 256-sample reads swing the raw fill
        # by ~27% of target; unfiltered, that drives the ratio across the full
        # +-0.5% limit at the block rate (measured swing 1.0%).
        # Measured swing with the 0.25 s fill filter: about 0.03%.
        self.assertLess(max(corrections)-min(corrections), 0.0005)
        passthrough.close()

    def test_priming_waits_for_target_raised_by_large_callbacks(self):
        passthrough = AudioPassthrough(
            48000, sounddevice_module=_Fake48000SoundDevice)
        passthrough.open(7)
        passthrough.DECLICK_SECONDS = 0.0
        passthrough.set_route('right')
        block = np.full((1024, 2), .3, dtype=np.float32)
        out = np.empty((4096, 2), dtype=np.float32)
        written = 0
        for _ in range(4):
            passthrough.queue_capture(written, block)
            written += 1024
        # 4,096 queued clears the 80 ms (3,840) target but not one 4,096
        # callback plus one 1,024 capture block plus the 30 ms capture-stall
        # allowance (4,098+1,024+1,440 = 6,562): stay silent.
        passthrough.stream.callback(out, len(out), None, None)
        np.testing.assert_array_equal(out, 0.0)
        self.assertEqual(passthrough.reader.target_samples, 4098+1024+1440)
        for _ in range(2):
            passthrough.queue_capture(written, block)
            written += 1024
        passthrough.stream.callback(out, len(out), None, None)
        np.testing.assert_array_equal(out, 0.0)  # 6,144 < 6,562
        passthrough.queue_capture(written, block)
        written += 1024
        passthrough.stream.callback(out, len(out), None, None)
        self.assertGreater(float(np.min(out)), .29)
        passthrough.close()

    def test_runaway_fifo_is_trimmed_back_to_target_without_a_step(self):
        rate = 48000
        passthrough = AudioPassthrough(
            rate, sounddevice_module=_Fake48000SoundDevice)
        passthrough.open(7)
        passthrough.set_route('right')
        tone = (.5*np.sin(2*np.pi*440*np.arange(rate)/rate)).astype(np.float32)
        written = 0
        out = np.empty((512, 2), dtype=np.float32)
        emitted = []
        for _ in range(4):  # steady playback first
            passthrough.queue_capture(written, np.stack([tone[written:written+1024]]*2, 1))
            written += 1024
        passthrough.stream.callback(out, len(out), None, None)
        emitted.append(out[:, 0].copy())
        # Output starved for 0.4 s: the capture side keeps delivering.
        while written < int(.5*rate):
            passthrough.queue_capture(written, np.stack([tone[written:written+1024]]*2, 1))
            written += 1024
        for _ in range(3):
            passthrough.stream.callback(out, len(out), None, None)
            emitted.append(out[:, 0].copy())
        stats = passthrough.stats_snapshot()
        self.assertEqual(stats['latency_trims'], 1)
        self.assertLess(passthrough.buffered_ms, 100.0)
        joined = np.concatenate(emitted)
        # A 440 Hz sine at 0.5 moves <= 0.03 per sample; a splice step would not.
        self.assertLess(float(np.max(np.abs(np.diff(joined)))), .06)
        passthrough.close()

    def test_tone_with_sustained_clock_drift_holds_fifo_and_pitch(self):
        in_rate, out_rate, drift = 96000, 44100, 0.003
        passthrough = AudioPassthrough(
            in_rate, sounddevice_module=_Fake44100SoundDevice)
        passthrough.open(7)
        passthrough.set_route('right')
        produced = 0
        true_in = in_rate*(1.0+drift)  # capture clock runs 3,000 ppm fast
        tone_hz = 1000.0
        outputs, fills, corrections = [], [], []
        out = np.empty((512, 2), dtype=np.float32)
        callbacks = int(30*out_rate/512)
        for index in range(callbacks):
            t_end = (index+1)*512/out_rate
            while produced < int(t_end*true_in)+1024:
                n = np.arange(produced, produced+1024)
                x = (.5*np.sin(2*np.pi*tone_hz*n/true_in)).astype(np.float32)
                passthrough.queue_capture(produced, np.stack([x, x], 1))
                produced += 1024
            passthrough.stream.callback(out, len(out), None, None)
            outputs.append(out[:, 0].copy())
            if index*512 > 20*out_rate:
                fills.append(passthrough.buffered_ms)
                corrections.append(passthrough.reader.current_correction)
        stats = passthrough.stats_snapshot()
        self.assertEqual(stats['underflow_events'], 0)
        self.assertEqual(stats['latency_trims'], 0)
        self.assertLess(max(fills)-min(fills), 25.0)
        self.assertAlmostEqual(float(np.mean(corrections)), drift, delta=.0005)
        tail = np.concatenate(outputs)[-5*out_rate:]
        crossings = np.flatnonzero((tail[:-1] < 0) & (tail[1:] >= 0))
        periods = np.diff(crossings)/out_rate
        # Output pitch is the true 1 kHz, steady to within 0.1%.
        self.assertAlmostEqual(1.0/float(np.mean(periods)), tone_hz, delta=1.0)
        passthrough.close()

    def test_rate_meter_is_unbiased_for_clustered_callbacks(self):
        from tools.v7_receiver_audio import _RateMeter
        rng = np.random.default_rng(1)
        for cluster in (1, 4):
            meter = _RateMeter()
            period = cluster*1024/96000
            t, readings = 100.0, []
            for index in range(int(30/period)):
                t += period
                for block in range(cluster):
                    meter.add(1024, t+block*1e-4+rng.uniform(0, 5e-4))
                if index*period > 12:
                    readings.append(meter.rate())
            # Counting from each block read 96,305 (+3,176 ppm) for clusters
            # of four; coalesced clusters read the true 96,000.
            self.assertAlmostEqual(float(np.mean(readings)), 96000.0,
                                   delta=96000*50e-6)

    def test_stats_report_measured_rates_and_late_output_callbacks(self):
        from unittest.mock import patch
        passthrough = AudioPassthrough(
            48000, sounddevice_module=_Fake48000SoundDevice)
        passthrough.open(7)
        clock = [100.0]
        out = np.empty((480, 2), dtype=np.float32)
        with patch('tools.v7_receiver_audio.time.monotonic',
                   side_effect=lambda: clock[0]):
            for index in range(300):
                clock[0] += .010 if index != 150 else .110  # one 100 ms stall
                passthrough.stream.callback(out, len(out), None, None)
            stats = passthrough.stats_snapshot()
        self.assertEqual(stats['late_callbacks'], 1)
        self.assertAlmostEqual(stats['max_callback_gap_ms'], 110.0, delta=.5)
        self.assertAlmostEqual(stats['measured_output_rate'], 48000*3.0/3.1,
                               delta=200)
        self.assertAlmostEqual(stats['late_ms_total'], 100.0, delta=.5)
        # The monitor snapshots every 100 ms; the peak must survive until the
        # printed report resets it.
        self.assertAlmostEqual(
            passthrough.stats_snapshot()['max_callback_gap_ms'], 110.0,
            delta=.5)
        passthrough.reset_report_peaks()
        self.assertEqual(passthrough.stats_snapshot()['max_callback_gap_ms'],
                         0.0)
        passthrough.close()

    def test_stall_probe_reports_a_process_wide_gil_stall(self):
        import sys
        import time
        passthrough = AudioPassthrough(
            48000, sounddevice_module=_Fake48000SoundDevice)
        passthrough.open(7)
        time.sleep(.05)
        passthrough.reset_report_peaks()
        interval = sys.getswitchinterval()
        sys.setswitchinterval(1.0)
        try:
            deadline = time.perf_counter()+.08
            while time.perf_counter() < deadline:  # hold the GIL for 80 ms
                pass
        finally:
            sys.setswitchinterval(interval)
        time.sleep(.05)
        stats = passthrough.stats_snapshot()
        passthrough.close()
        self.assertGreater(stats['max_stall_ms'], 40.0)
        self.assertGreaterEqual(stats['stalls_over_10ms'], 1)

    def test_output_buffering_knobs_reach_the_output_stream(self):
        passthrough = AudioPassthrough(
            48000, sounddevice_module=_Fake48000SoundDevice,
            output_latency=.1, output_blocksize=2048)
        passthrough.open(7)
        self.assertEqual(passthrough.stream.kwargs['latency'], .1)
        self.assertEqual(passthrough.stream.kwargs['blocksize'], 2048)
        passthrough.close()
        default = AudioPassthrough(
            48000, sounddevice_module=_Fake48000SoundDevice)
        default.open(7)
        self.assertNotIn('latency', default.stream.kwargs)
        self.assertEqual(default.stream.kwargs['blocksize'],
                         AudioPassthrough.OUTPUT_BLOCKSIZE)
        self.assertEqual(AudioPassthrough.OUTPUT_BLOCKSIZE, 2048)
        default.close()
        host = AudioPassthrough(
            48000, sounddevice_module=_Fake48000SoundDevice,
            output_blocksize=0)
        host.open(7)
        self.assertEqual(host.stream.kwargs['blocksize'], 0)
        host.close()
        with self.assertRaises(ValueError):
            AudioPassthrough(48000, sounddevice_module=_Fake48000SoundDevice,
                             output_blocksize=-1)

    def _drive_field(self, block, seconds=90, true_in=96300.0,
                     burst_blocks=3, burst_every=7, seed=3):
        """A stress case: 96 kHz capture running +3,125 ppm fast (a large but
        in-range clock error) in 1,024-frame blocks, 44.1 kHz output, and
        capture callbacks held back by GIL stalls (up to three blocks, about
        32 ms)."""
        rng = np.random.default_rng(seed)
        passthrough = AudioPassthrough(
            96000, sounddevice_module=_Fake44100SoundDevice)
        passthrough.open(7)
        passthrough.set_route('right')
        produced = 0
        out = np.empty((block, 2), dtype=np.float32)
        corrections = []
        for index in range(int(seconds*44100/block)):
            t = (index+1)*block/44100
            lag = burst_blocks*1024 if rng.random() < 1/burst_every else 0
            while produced+1024 <= int(t*true_in)-lag:
                passthrough.queue_capture(
                    produced, np.zeros((1024, 2), dtype=np.float32))
                produced += 1024
            passthrough.stream.callback(out, block, None, None)
            if t > 30:
                corrections.append(passthrough.reader.current_correction)
        stats = passthrough.stats_snapshot()
        passthrough.close()
        return np.array(corrections), stats

    def test_pi_servo_holds_a_large_capture_offset_off_the_limit(self):
        corrections, stats = self._drive_field(2048)
        # The proportional-only servo tracked the same mean but spent ~60% of
        # the time on the +-5,000 ppm limit (sd ~3,000 ppm): audible warble.
        self.assertAlmostEqual(float(np.mean(corrections)), .003125,
                               delta=.0003)
        self.assertLess(float(np.std(corrections)), .001)
        self.assertEqual(float(np.mean(np.abs(corrections) >= .00499)), 0.0)
        self.assertEqual(stats['underflow_events'], 0)
        self.assertEqual(stats['latency_trims'], 0)
        self.assertEqual(stats['dropped_samples'], 0)

    def test_pi_integral_never_exceeds_the_correction_limit(self):
        from tools.v7_source_audio import ClockMatchedReader, SampleBuffer
        fifo = SampleBuffer(max_samples=200_000)
        fifo.push(np.zeros(150_000, dtype=np.float32))
        reader = ClockMatchedReader(
            fifo, 1000, nominal_ratio=1.0, correction_limit=.005,
            servo_rate=48000)
        for _ in range(300):  # far above target for seconds: windup pressure
            reader.read(256)
            fifo.push(np.zeros(256, dtype=np.float32))
        self.assertLessEqual(reader.servo_ki*reader._integral, .005+1e-12)
        self.assertLessEqual(reader.current_correction, .005)

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
