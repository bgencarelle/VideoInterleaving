"""Clean-link reconstruction checks for the sender's encoded-image preview."""
import struct
import socket
import sys
import threading
import time
import unittest
from pathlib import Path
from argparse import Namespace
from unittest import mock

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT/'test_modem_v7', ROOT/'tools'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from animation_modem import v7                                      # noqa: E402
from live_fold import LiveFold                                      # noqa: E402
from mono_video import MonoColourFoldWire, MonoFreshFoldWire          # noqa: E402
from tone_code import FOLD_500, FOLD_1000, coded_pilot_timing          # noqa: E402
from tools import v7_live                                             # noqa: E402
from tools.v7_encoded_preview import (EncodedFrameReconstructor,      # noqa: E402
                                     EncodedPreviewWorker,
                                     decode_preview_image,
                                     preview_datagram)


TARGET = .1521/np.sqrt(1+10**(v7.CLOCK_REL_DB/10))


class EncodedPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with Image.open(v7.REFERENCE_FIXTURE) as image:
            source = v7.image_values(
                v7.prepare_image(image.convert('RGB'), 'box'),
                v7.V7_GRIDS, 'box')
        cls.source = source

    def _frames(self, encode_filter):
        with Image.open(v7.REFERENCE_FIXTURE) as image:
            values = v7.image_values(
                v7.prepare_image(image.convert('RGB'), encode_filter),
                v7.V7_GRIDS, encode_filter)
        # Vary every frame so tail slices from older packets are observable.
        return [np.roll(values, 17*index) for index in range(v7.TAIL_PHASES)]

    def _check_profile(self, profile):
        encode_filter = 'nearest' if profile == 'baseline' else 'box'
        base_model = v7.load_model(TARGET, encode_filter)
        frames = self._frames(encode_filter)
        aspects = [v7.aspect_wire_code((4+index, 3))
                   for index in range(len(frames))]
        history = [(index+1, aspects[index], frame, index)
                   for index, frame in enumerate(frames)]
        fold = None
        mono_wire = None

        if profile == 'baseline':
            model = base_model
            audio = v7.encode_pulse_stream(
                model, frames, start_counter=1, aspect_codes=aspects,
                pilot_tones=True, eof_marker=True)
            results, _info = v7.decode_pulse_stream(
                model, audio, sample_rate=v7.RATE,
                pilot_timing='tone-seeded', frame_boundary='eof')
            expected = v7.values_from(model, results[-1].coeffs)
            reconstructor = EncodedFrameReconstructor(model)
        elif profile.startswith('fold-'):
            slots = int(profile.split('-')[1])
            fold = LiveFold(slots)
            mode = FOLD_500 if slots == 500 else FOLD_1000
            packets = [v7.encode_pulse_frame_coeffs(
                base_model,
                fold.encode_coefficients(base_model, frame),
                index+1, aspect_code=aspects[index], source_index=index,
                eof_marker=True, pulse_profile_code=mode)
                for index, frame in enumerate(frames)]
            audio = v7_live._add_coded_pilots(
                np.concatenate(packets), 1, slots)
            fold.install()
            try:
                with coded_pilot_timing():
                    results, _info = v7.decode_pulse_stream(
                        base_model, audio, sample_rate=v7.RATE,
                        pilot_timing='tone-seeded', frame_boundary='eof')
            finally:
                fold.uninstall()
            expected = fold.values(
                base_model, results[-1], metadata_confirmed=True)
            reconstructor = EncodedFrameReconstructor(base_model, fold=fold)
        else:
            wire_class = (MonoFreshFoldWire if profile == 'mono-fold-500'
                          else MonoColourFoldWire)
            mono_wire = wire_class(base_model, side='both')
            model = mono_wire.model_for(base_model)
            audio = mono_wire.encode(
                base_model, frames, start_counter=1,
                aspect_codes=aspects, eof_marker=True)
            with mono_wire.receiving():
                results, _info = v7.decode_pulse_stream(
                    model, audio, sample_rate=v7.RATE,
                    pilot_timing='tone-seeded', frame_boundary='eof')
            expected = mono_wire.values(model, results[-1])
            reconstructor = EncodedFrameReconstructor(
                base_model, mono_wire=mono_wire)

        self.assertEqual(len(results), len(frames))
        self.assertEqual(results[-1].diag['aspect_code'], aspects[-1])
        actual = reconstructor.reconstruct_values(history)
        self.assertEqual(actual.shape, expected.shape)
        self.assertLess(float(np.mean(np.abs(actual-expected))), .012,
                        msg=f'{profile} clean-link mean reconstruction error')
        self.assertLess(float(np.max(np.abs(actual-expected))), .08,
                        msg=f'{profile} clean-link peak reconstruction error')

    def test_preview_tracks_clean_decoder_for_each_gui_profile_and_tail_phase(self):
        for profile in ('baseline', 'fold-500', 'fold-1000',
                        'mono-fold-500', 'mono-colour-500'):
            with self.subTest(profile=profile):
                self._check_profile(profile)

    def test_preview_datagram_carries_packet_identity_and_handoff_time(self):
        header = struct.Struct('!4sQIQ')
        message = header.pack(b'V7EP', 37, 6, 123456789)+b'jpeg-data'

        self.assertEqual(preview_datagram(message),
                         (37, 6, 123456789, b'jpeg-data'))
        self.assertIsNone(preview_datagram(b'invalid'))

    def test_worker_publishes_latest_reconstruction_over_loopback(self):
        model = v7.load_model(TARGET, 'nearest')
        values = self._frames('nearest')[0]
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(('127.0.0.1', 0))
        receiver.settimeout(.1)
        worker = EncodedPreviewWorker(
            model, receiver.getsockname()[1])
        try:
            worker.start()
            worker.submit(41, 3, values)
            worker.submit(42, 4, values)
            deadline = time.monotonic()+4
            latest = None
            while time.monotonic() < deadline:
                try:
                    packet, _address = receiver.recvfrom(65507)
                except socket.timeout:
                    continue
                parsed = preview_datagram(packet)
                if parsed is not None and parsed[0] == 42:
                    latest = parsed
                    break
            self.assertIsNotNone(latest)
            self.assertEqual(latest[1], 4)
            self.assertGreater(latest[2], 0)
            self.assertEqual(decode_preview_image(latest[3]).size,
                             (v7.V7_GRIDS[0][1], v7.V7_GRIDS[0][0]))
        finally:
            worker.close()
            receiver.close()

    def test_slow_preview_renderer_keeps_only_the_latest_bounded_job(self):
        model = v7.load_model(TARGET, 'nearest')
        values = self._frames('nearest')[0]
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(('127.0.0.1', 0))
        receiver.settimeout(.1)
        entered = threading.Event()
        release = threading.Event()
        original = EncodedFrameReconstructor.reconstruct

        def slow_reconstruct(reconstructor, history):
            entered.set()
            if not release.wait(2):
                raise RuntimeError('slow preview test timed out')
            return original(reconstructor, history)

        worker = EncodedPreviewWorker(model, receiver.getsockname()[1])
        try:
            with mock.patch.object(EncodedFrameReconstructor,
                                   'reconstruct', slow_reconstruct):
                worker.start()
                worker.submit(50, 1, values)
                self.assertTrue(entered.wait(2))
                began = time.monotonic()
                for counter in range(51, 61):
                    worker.submit(counter, counter % 8, values)
                elapsed = time.monotonic()-began
                self.assertLess(elapsed, .25)
                self.assertLessEqual(worker.jobs.qsize(), 1)
                release.set()
                deadline = time.monotonic()+4
                latest = None
                while time.monotonic() < deadline:
                    try:
                        packet, _address = receiver.recvfrom(65507)
                    except socket.timeout:
                        continue
                    parsed = preview_datagram(packet)
                    if parsed is not None and parsed[0] == 60:
                        latest = parsed
                        break
                self.assertIsNotNone(latest)
        finally:
            release.set()
            worker.close()
            receiver.close()

    def test_preview_does_not_change_samples_handed_to_output(self):
        class OutputStream:
            samplerate = 48000
            writes = []

            def __init__(self, **_kwargs):
                pass

            def __enter__(self):
                type(self).writes = []
                return self

            def __exit__(self, *_exc):
                return False

            def write(self, samples):
                type(self).writes.append(np.array(samples, copy=True))

        SoundDevice = type('SoundDevice', (), {'OutputStream': OutputStream})

        class OneFrame:
            def __init__(self):
                self.calls = 0
                self.ended = False

            def __call__(self):
                self.calls += 1
                if self.calls == 1:
                    return np.zeros((96, 80, 3), dtype=np.uint8)
                self.ended = True
                return None

            def close(self):
                pass

        def send(with_preview, port):
            args = Namespace(
                fixture=None, encode_filter='box', rate=None,
                source='test', capture_fps=None, screen_backend='mss',
                speed=1.0, batch_frames=1, seconds=0, mono_sum=False,
                device=0, no_log=True, log=False, brightness=1.0,
                gamma=1.0, baseline=False, profile='fold-500', preview=False,
                encoded_preview_port=port if with_preview else None)
            with mock.patch.dict(sys.modules, {'sounddevice': SoundDevice}), \
                    mock.patch('tools.v7_live._capture',
                               return_value=OneFrame()), \
                    mock.patch('tools.v7_capture.Throttled',
                               side_effect=lambda grab, _hz: grab):
                v7_live.run_send(args)
            return tuple(OutputStream.writes)

        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(('127.0.0.1', 0))
        try:
            without_preview = send(False, receiver.getsockname()[1])
            with_preview = send(True, receiver.getsockname()[1])
        finally:
            receiver.close()
        self.assertTrue(without_preview)
        self.assertEqual(len(with_preview), len(without_preview))
        for actual, expected in zip(with_preview, without_preview):
            np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
