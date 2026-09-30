"""Source/resized stage preview checks for the standalone V7 sender."""
from argparse import Namespace
from io import BytesIO
import socket
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from animation_modem import v7                                      # noqa: E402
from tools import v7_live                                            # noqa: E402
from tools import v7_image_preview                                    # noqa: E402
from tools.v7_image_preview import ImagePreviewWorker                # noqa: E402
from tools.v7_preview_protocol import (pack_preview_datagram,         # noqa: E402
                                       parse_preview_datagram)


TARGET = .1521/np.sqrt(1+10**(v7.CLOCK_REL_DB/10))


class ImagePreviewTests(unittest.TestCase):
    def test_resized_stage_is_the_rgb_input_used_to_create_encoder_values(self):
        model = v7.load_model(TARGET, 'nearest')
        yy, xx = np.mgrid[0:480, 0:640]
        source = np.stack((xx % 256, yy % 256, (xx+yy) % 256), axis=-1)
        source = source.astype(np.uint8)

        values, _aspect, resized = v7_live._values(
            model, source, encode_filter='nearest', brightness=1.0,
            gamma=1.0, return_resized=True)
        expected = v7.prepare_image(Image.fromarray(source), 'nearest')

        self.assertEqual(resized.size, (80, 96))
        np.testing.assert_array_equal(np.asarray(resized), np.asarray(expected))
        self.assertEqual(values.shape, (model.coder.source_count,))

    def test_dct_encode_preview_does_not_reconstruct_encoded_values(self):
        model = v7.load_model(TARGET, 'nearest')
        source = np.zeros((120, 160, 3), dtype=np.uint8)

        values, _aspect, resized = v7_live._values(
            model, source, brightness=1.0, gamma=1.0,
            dct_encode=True, return_resized=True)

        self.assertEqual(resized.size, (160, 120))
        self.assertEqual(values.shape, (model.coder.source_count,))

    def test_preview_datagram_identifies_the_image_stage(self):
        message = pack_preview_datagram(
            37, 6, 123456789, 'resized', b'jpeg-data')

        self.assertEqual(parse_preview_datagram(message),
                         (37, 6, 123456789, 'resized', b'jpeg-data'))
        self.assertIsNone(parse_preview_datagram(b'invalid'))
        with self.assertRaisesRegex(ValueError, 'unknown image preview stage'):
            pack_preview_datagram(1, 0, 1, 'encoded', b'jpeg-data')

    def test_worker_publishes_latest_stage_image_over_loopback(self):
        source = Image.new('RGB', (640, 480), (90, 120, 150))
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(('127.0.0.1', 0))
        receiver.settimeout(.1)
        worker = ImagePreviewWorker(receiver.getsockname()[1])
        try:
            worker.start()
            worker.submit(41, 3, source, source)
            worker.submit(42, 4, source, source)
            deadline = time.monotonic()+4
            newest = {}
            while time.monotonic() < deadline:
                try:
                    packet, _address = receiver.recvfrom(65507)
                except socket.timeout:
                    continue
                parsed = parse_preview_datagram(packet)
                if parsed is not None and parsed[0] == 42:
                    newest[parsed[3]] = parsed
                    if set(newest) == {'source', 'resized'}:
                        break
            self.assertEqual(set(newest), {'source', 'resized'})
            for stage, parsed in newest.items():
                self.assertEqual(parsed[1], 4)
                self.assertGreater(parsed[2], 0)
                self.assertEqual(parsed[3], stage)
                with Image.open(BytesIO(parsed[4])) as image:
                    self.assertLessEqual(image.width, 256)
                    self.assertLessEqual(image.height, 320)
        finally:
            worker.close()
            receiver.close()

    def test_slow_image_renderer_keeps_only_the_latest_bounded_job(self):
        entered = threading.Event()
        release = threading.Event()
        original = v7_image_preview._preview_jpeg

        def slow_encode(frame):
            entered.set()
            if not release.wait(2):
                raise RuntimeError('slow preview test timed out')
            return original(frame)

        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(('127.0.0.1', 0))
        receiver.settimeout(.1)
        worker = ImagePreviewWorker(receiver.getsockname()[1])
        frame = Image.new('RGB', (80, 96), (20, 30, 40))
        try:
            with mock.patch.object(v7_image_preview, '_preview_jpeg',
                                   slow_encode):
                worker.start()
                worker.submit(50, 1, frame, frame)
                self.assertTrue(entered.wait(2))
                began = time.monotonic()
                for counter in range(51, 61):
                    worker.submit(counter, counter % 8, frame, frame)
                self.assertLess(time.monotonic()-began, .25)
                self.assertLessEqual(worker.jobs.qsize(), 1)
                release.set()
                deadline = time.monotonic()+4
                newest = set()
                while time.monotonic() < deadline:
                    try:
                        packet, _address = receiver.recvfrom(65507)
                    except socket.timeout:
                        continue
                    parsed = parse_preview_datagram(packet)
                    if parsed is not None and parsed[0] == 60:
                        newest.add(parsed[3])
                        if newest == {'source', 'resized'}:
                            break
                self.assertEqual(newest, {'source', 'resized'})
        finally:
            release.set()
            worker.close()
            receiver.close()

    def test_image_preview_does_not_change_samples_handed_to_output(self):
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

        sounddevice = type('SoundDevice', (), {'OutputStream': OutputStream})

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
                image_preview_port=port if with_preview else None,
                preview_stage='resized')
            with mock.patch.dict(sys.modules, {'sounddevice': sounddevice}), \
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
