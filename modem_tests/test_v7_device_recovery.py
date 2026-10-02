"""Stable identity and sample-rate recovery primitives."""
import unittest
import threading
import time
from types import SimpleNamespace
from unittest import mock

import numpy as np

from tools import v7_live
from tools.v7_device_recovery import (
    DEVICE_RATE_FRAME_INTERVAL, DEVICE_RATE_STABLE_FRAMES,
    DeviceRateDebouncer, device_identity, query_device_snapshot,
    resolve_device_index)
from tools.v7_receiver_audio import ReceiverRuntimeOptions


class FakeSoundDevice:
    def __init__(self, devices, hostapis=None):
        self.devices = list(devices)
        self.hostapis = hostapis or [{'name': 'Test API'}]

    def query_devices(self, device=None, kind=None):
        if device is None:
            return list(self.devices)
        if isinstance(device, str):
            matches = [(index, item) for index, item in enumerate(self.devices)
                       if item['name'] == device]
            if len(matches) != 1:
                raise ValueError('device name is missing or ambiguous')
            return matches[0][1]
        return self.devices[int(device)]

    def query_hostapis(self, _index=None):
        return self.hostapis


def _device(name, rate, inputs=2, outputs=2, hostapi=0):
    return {
        'name': name, 'hostapi': hostapi,
        'default_samplerate': rate,
        'max_input_channels': inputs,
        'max_output_channels': outputs,
    }


class DeviceIdentityTests(unittest.TestCase):
    def test_resolves_same_device_after_portaudio_index_changes(self):
        original = FakeSoundDevice([_device('loopback', 48000)])
        identity = device_identity(original, 0)
        changed = FakeSoundDevice([
            _device('unrelated', 44100), _device('loopback', 96000)])

        index = resolve_device_index(changed, 0, identity, 'output')
        snapshot = query_device_snapshot(changed, 0, identity, 'output')

        self.assertEqual(index, 1)
        self.assertEqual(snapshot[0], 1)
        self.assertEqual(snapshot[2], identity)
        self.assertEqual(snapshot[3], 96000)

    def test_does_not_fall_back_to_a_different_device(self):
        sd = FakeSoundDevice([_device('other', 48000)])
        with self.assertRaisesRegex(ValueError, 'loopback is unavailable'):
            resolve_device_index(
                sd, 0, {'name': 'loopback', 'hostapi': 'Test API'}, 'input')

    def test_debounce_requires_five_matching_device_rate_observations(self):
        self.assertEqual(DEVICE_RATE_STABLE_FRAMES, 5)
        self.assertAlmostEqual(DEVICE_RATE_FRAME_INTERVAL, 1/30)
        monitor = DeviceRateDebouncer()
        identity = {'name': 'loopback', 'hostapi': 'Test API'}

        self.assertEqual(
            [monitor.observe(identity, 96000) for _ in range(4)],
            [False]*4)
        self.assertTrue(monitor.observe(identity, 96000))

    def test_rate_or_identity_change_restarts_the_stability_count(self):
        monitor = DeviceRateDebouncer()
        identity = {'name': 'loopback', 'hostapi': 'Test API'}
        for _ in range(4):
            self.assertFalse(monitor.observe(identity, 48000))
        self.assertFalse(monitor.observe(identity, 96000))
        self.assertEqual(monitor.stable_count, 1)
        other = {'name': 'loopback', 'hostapi': 'Other API'}
        self.assertFalse(monitor.observe(other, 96000))
        self.assertEqual(monitor.stable_count, 1)

    def test_invalid_rate_resets_the_candidate(self):
        monitor = DeviceRateDebouncer()
        identity = {'name': 'loopback', 'hostapi': 'Test API'}
        monitor.observe(identity, 48000)
        self.assertFalse(monitor.observe(identity, 0))
        self.assertEqual(monitor.stable_count, 0)


class ReceiverReconnectTests(unittest.TestCase):
    def test_receiver_holds_last_picture_and_reopens_at_new_rate(self):
        stop_event = threading.Event()
        frame_buffer = v7_live.FRAME_BUFFER
        frame_buffer.publish(np.arange(4), ((2, 2),), 0)
        held_generation = frame_buffer.snapshot().generation

        class SoundDevice:
            devices = [_device('receiver input', 48000)]
            streams = []

            @classmethod
            def query_devices(cls, device=None, kind=None):
                if device is None:
                    return list(cls.devices)
                return cls.devices[int(device)]

            @staticmethod
            def query_hostapis():
                return [{'name': 'Test API'}]

            class InputStream:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs
                    self.samplerate = float(kwargs['samplerate'])
                    self.active = True
                    SoundDevice.streams.append(self)

                def start(self):
                    if len(SoundDevice.streams) == 1:
                        def change_rate():
                            time.sleep(.06)
                            SoundDevice.devices[0][
                                'default_samplerate'] = 96000
                        threading.Thread(target=change_rate, daemon=True).start()
                    else:
                        stop_event.set()

                def stop(self):
                    self.active = False

                def close(self):
                    self.active = False

        args = v7_live.parser().parse_args([
            'receive', '--device', '0', '--headless', '--experimental-fold', '0', '--no-log'])
        args.stop_event = stop_event
        model = SimpleNamespace(
            encoding_type=0,
            coder=SimpleNamespace(grids=((2, 2),), shapes=((2, 2),)))

        with mock.patch.dict('sys.modules', {'sounddevice': SoundDevice}), \
                mock.patch('tools.v7_live._model', return_value=model), \
                mock.patch.object(v7_live.P.PULSE, 'warmup_pulse_kernels'), \
                mock.patch.object(v7_live.P, 'warmup_leg_polarity'), \
                mock.patch.object(v7_live.P, 'warmup_equalizer'):
            v7_live.run_receive(args)

        self.assertGreaterEqual(len(SoundDevice.streams), 2)
        self.assertEqual([stream.samplerate for stream in SoundDevice.streams[:2]],
                         [48000.0, 96000.0])
        self.assertEqual(frame_buffer.snapshot().generation, held_generation)

    def test_receiver_freezes_and_reconnects_same_rate_device_return(self):
        stop_event = threading.Event()
        frame_buffer = v7_live.FRAME_BUFFER
        frame_buffer.publish(np.arange(4), ((2, 2),), 0)
        held_generation = frame_buffer.snapshot().generation

        class SoundDevice:
            device = _device('receiver input', 48000)
            devices = [device]
            streams = []

            @classmethod
            def query_devices(cls, device=None, kind=None):
                if device is None:
                    return list(cls.devices)
                return cls.devices[int(device)]

            @staticmethod
            def query_hostapis():
                return [{'name': 'Test API'}]

            class InputStream:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs
                    self.samplerate = float(kwargs['samplerate'])
                    self.active = True
                    SoundDevice.streams.append(self)

                def start(self):
                    if len(SoundDevice.streams) == 1:
                        def disconnect_then_return():
                            time.sleep(.05)
                            SoundDevice.streams[0].active = False
                            SoundDevice.devices = []
                            time.sleep(.12)
                            SoundDevice.devices = [SoundDevice.device]
                        threading.Thread(
                            target=disconnect_then_return,
                            daemon=True).start()
                    else:
                        stop_event.set()

                def stop(self):
                    self.active = False

                def close(self):
                    self.active = False

        args = v7_live.parser().parse_args([
            'receive', '--device', '0', '--headless', '--experimental-fold', '0', '--no-log'])
        args.stop_event = stop_event
        model = SimpleNamespace(
            encoding_type=0,
            coder=SimpleNamespace(grids=((2, 2),), shapes=((2, 2),)))

        with mock.patch.dict('sys.modules', {'sounddevice': SoundDevice}), \
                mock.patch('tools.v7_live._model', return_value=model), \
                mock.patch.object(v7_live.P.PULSE, 'warmup_pulse_kernels'), \
                mock.patch.object(v7_live.P, 'warmup_leg_polarity'), \
                mock.patch.object(v7_live.P, 'warmup_equalizer'):
            v7_live.run_receive(args)

        self.assertGreaterEqual(len(SoundDevice.streams), 2)
        self.assertEqual(frame_buffer.snapshot().generation, held_generation)

    def test_receiver_passthrough_reopens_after_stable_rate_change(self):
        stop_event = threading.Event()

        class SoundDevice:
            devices = [
                _device('receiver input', 48000),
                _device('passthrough output', 48000),
            ]

            @classmethod
            def query_devices(cls, device=None, _kind=None):
                if device is None:
                    return list(cls.devices)
                return cls.devices[int(device)]

            @staticmethod
            def query_hostapis():
                return [{'name': 'Test API'}]

            class InputStream:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs
                    self.samplerate = float(kwargs['samplerate'])
                    self.active = True

                def start(self):
                    pass

                def stop(self):
                    self.active = False

                def close(self):
                    self.active = False

        class Passthrough:
            instances = []

            def __init__(self, input_rate, sounddevice_module=None,
                         status_callback=None, **_kwargs):
                self.sd = sounddevice_module
                self.status_callback = status_callback
                self.error = None
                self.device = None
                self.device_default_rate = None
                self.output_rate = None
                self.route = None
                self.active = False
                self.opens = []
                type(self).instances.append(self)

            @property
            def is_open(self):
                return self.active

            def open(self, device, expected_identity=None):
                index = resolve_device_index(
                    self.sd, device, expected_identity, 'output')
                info = self.sd.query_devices(index, 'output')
                rate = float(info['default_samplerate'])
                self.device = index
                self.device_default_rate = rate
                self.output_rate = rate
                self.active = True
                self.opens.append(rate)
                if len(self.opens) == 2:
                    stop_event.set()

            def close(self):
                self.active = False

            def check_device(self, _identity=None):
                return self.active

            def native_device_sample_rate(self, _identity=None):
                return float(self.sd.query_devices(
                    self.device, 'output')['default_samplerate'])

            def set_input_rate(self, _rate):
                pass

            def set_route(self, route, _muted):
                self.route = route

            def set_volume(self, _volume):
                pass

            def set_muted(self, _muted):
                pass

            def note_input_status(self, _status):
                pass

            def queue_capture(self, _start, _block):
                pass

            def stats_snapshot(self):
                return {
                    'underflow_events': 0, 'input_status': {},
                    'output_status': {}, 'dropped_samples': 0,
                    'buffered_ms': 0.0,
                }

            def reset_report_peaks(self):
                pass

        output_identity = {'name': 'passthrough output',
                           'hostapi': 'Test API'}
        input_identity = {'name': 'receiver input', 'hostapi': 'Test API'}
        options = ReceiverRuntimeOptions(
            audio_output_device=1, audio_output_identity=output_identity,
            audio_input_identity=input_identity)
        args = v7_live.parser().parse_args([
            'receive', '--device', '0', '--audio-output-device', '1',
            '--headless', '--experimental-fold', '0', '--no-log'])
        args.stop_event = stop_event
        args.runtime_options = options
        model = SimpleNamespace(
            encoding_type=0,
            coder=SimpleNamespace(grids=((2, 2),), shapes=((2, 2),)))

        def change_rate():
            time.sleep(.06)
            SoundDevice.devices[1]['default_samplerate'] = 96000

        threading.Thread(target=change_rate, daemon=True).start()
        with mock.patch.dict('sys.modules', {'sounddevice': SoundDevice}), \
                mock.patch('tools.v7_live._model', return_value=model), \
                mock.patch.object(v7_live.P.PULSE, 'warmup_pulse_kernels'), \
                mock.patch.object(v7_live.P, 'warmup_leg_polarity'), \
                mock.patch.object(v7_live.P, 'warmup_equalizer'), \
                mock.patch('tools.v7_receiver_audio.AudioPassthrough',
                           Passthrough):
            v7_live.run_receive(args)

        self.assertTrue(Passthrough.instances)
        self.assertEqual(Passthrough.instances[0].opens[:2], [48000.0, 96000.0])


class AdaptiveProfileTimelineTests(unittest.TestCase):
    def _decoder(self):
        decoder = object.__new__(v7_live._AdaptiveProfileDecoder)
        decoder.supported_modes = frozenset((5,))
        decoder.mono_status_modes = frozenset((5,))
        decoder.preferred_side = 'auto'
        decoder.input_channels = 2
        decoder.active_mode, decoder.active_side = 5, 1
        decoder.candidate = decoder.candidate_last_seen = None
        decoder.candidate_scale = None
        decoder.streak = decoder.generation = 0
        decoder.last_packet = decoder.last_decoded_packet = None
        decoder.state = None
        return decoder

    @staticmethod
    def _packet(position):
        return [{'position': float(position), 'scale': 1.0, 'mode': 5,
                 'side_index': 1}]

    def test_reopened_input_accepts_positions_from_zero_again(self):
        decoder = self._decoder()
        late = 2_000_000
        decision, = decoder.observe_packets(self._packet(late), now=1.0)
        self.assertTrue(decoder.claim(decision))
        # A reopened stream counts capture samples from zero: without a
        # reset every packet reads as already seen and nothing decodes.
        self.assertEqual(decoder.observe_packets(self._packet(5000), now=2.0), ())
        decoder.reset_capture_timeline()
        decision, = decoder.observe_packets(self._packet(5000), now=3.0)
        self.assertTrue(decision['confirmed'])
        self.assertTrue(decoder.claim(decision))
        self.assertEqual((decoder.active_mode, decoder.active_side), (5, 1))


if __name__ == '__main__':
    unittest.main()
