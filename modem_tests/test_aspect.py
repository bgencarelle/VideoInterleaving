"""Aspect is verified header metadata, never extra pixels or frame-counter bits."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import modem_screen
from animation_modem import engines as ENG, core, transport3 as V3
from animation_modem.aspect import (ASPECT_PRESETS, ASPECT_RATIOS, FRAME_MASK,
                                    aspect_code, pack_absolute)
from animation_modem.imaging import (image_values, values_image, display_image,
                                     prepare_image)
from animation_modem.presentation import DeadlinePresentationBuffer
from utilities import modem_v3_check as check


class AspectTests(unittest.TestCase):
    def test_every_preset_round_trips_all_live_engines(self):
        image = Image.new('RGB', (80, 96), (80, 150, 190))
        for name in ENG.engine_names():
            engine = ENG.get_engine(name)
            coder, _ = engine.coder_for(engine.profiles[0], engine.wire)
            values = image_values(image, coder.grids)
            for code in range(8):
                with self.subTest(engine=name, code=code):
                    rx = engine.receiver(coder=coder, coders=ENG.coders_for(engine.wire))
                    packets = [engine.encode(values, coder, counter, 1, 0xffff,
                                             aspect_code=code)
                               for counter in (FRAME_MASK, FRAME_MASK+1, FRAME_MASK+2)]
                    results = rx.feed(np.concatenate(packets)) + rx.flush()
                    self.assertEqual([r.absolute for r in results], [FRAME_MASK, 0, 1])
                    for r in results:
                        self.assertEqual(r.identity, 'verified_header')
                        self.assertEqual(r.count, 0xffff)
                        self.assertEqual(r.extra['aspect_code'], code)
                        self.assertEqual(r.extra['aspect'], ASPECT_RATIOS[code])
                        self.assertEqual(values_image(r.values, r.extra['shapes']).size, (80, 96))

    def test_header_word_contains_aspect_and_native_wire_is_unchanged(self):
        engine = ENG.get_engine('v3')
        coder, _ = engine.coder_for('color-dct')
        values = np.full(coder.source_count, .2)
        for code in range(8):
            with patch.object(V3, 'pack_header', wraps=core.pack_header) as pack:
                engine.encode(values, coder, 123, 1, 1, aspect_code=code)
                word = pack.call_args.args[2]
                self.assertEqual(word >> 29, code)
                self.assertEqual(word & FRAME_MASK, 123)
                raw = core.pack_header(*pack.call_args.args, **pack.call_args.kwargs)
                fields = core._resolve_header(raw, engine.wire)
                self.assertEqual(fields[1], word)  # refits still need the packed word
        a = engine.encode(values, coder, 123, 1, 1)
        b = engine.encode(values, coder, 123, 1, 1, aspect_code=0)
        np.testing.assert_array_equal(a, b)
        for code in (-1, 8):
            with self.assertRaises(ValueError):
                pack_absolute(1, code)

    def test_auto_rotation_override_and_changing_source_dimensions(self):
        for size, code in [((640, 480), 2), ((1920, 1080), 4), ((1080, 1920), 7),
                           ((400, 480), 0), ((2390, 1000), 5)]:
            self.assertEqual(aspect_code(size), code)
        prepare = modem_screen.fitter('hd-dwt')
        for w, h, code in [(640, 480, 2), (1280, 720, 4), (720, 1280, 7)]:
            raw = np.full((h, w, 3), 120, np.uint8)
            im = prepare(raw)
            self.assertEqual(im.size, (80, 96))
            self.assertEqual(im.info['aspect_code'], code)
            self.assertGreater(np.asarray(im).min(), 100)  # no generated bars
        raw = np.full((480, 640, 3), 120, np.uint8)
        self.assertEqual(modem_screen.fitter('hd-dwt', rotate=90)(raw).info['aspect_code'], 6)
        self.assertEqual(modem_screen.fitter('hd-dwt', aspect='native')(raw).info['aspect_code'], 0)
        self.assertEqual(modem_screen.fitter('hd-dwt', aspect='16:9')(raw).info['aspect_code'], 4)

    def test_render_geometry_native_pixels_and_smooth_non_native(self):
        engine = ENG.get_engine('v3')
        coder, _ = engine.coder_for('color-dct')
        image = Image.fromarray(np.random.default_rng(4).integers(20, 220, (96, 80, 3), dtype=np.uint8))
        values = image_values(image, coder.grids)
        for code, ratio in enumerate(ASPECT_RATIOS):
            r = core.Decoded('received', values=values,
                             extra={'shapes': coder.grids, 'aspect': ratio, 'aspect_code': code})
            native = values_image(values, coder.grids)
            saved = display_image(r)
            live = display_image(r, bounds=(480, 576))
            self.assertEqual(saved.height, 96)
            self.assertAlmostEqual(saved.width/saved.height, ratio, delta=1/96)
            self.assertAlmostEqual(live.width/live.height, ratio, delta=1/live.height)
            if code == 0:
                np.testing.assert_array_equal(saved, native)
            else:
                self.assertFalse(np.array_equal(saved, native.resize(saved.size, Image.Resampling.NEAREST)))

    def test_headerless_picture_keeps_last_verified_aspect(self):
        engine = ENG.get_engine('v3')
        coder, _ = engine.coder_for('color-dct')
        values = np.full(coder.source_count, .2)
        rx = engine.receiver(coder=coder)
        packet = engine.encode(values, coder, 1, 1, 1, aspect_code=4)
        first = (rx.feed(packet) + rx.flush())[0]
        damaged = core.Decoded('picture_only', values=first.values, extra={'shapes': coder.grids})
        with patch.object(V3, 'decode_packet', return_value=damaged):
            result = (rx.feed(packet) + rx.flush())[0]
        self.assertEqual(result.extra['aspect_code'], 4)
        self.assertTrue(result.extra['aspect_inferred'])
        rx.reset()
        self.assertEqual(rx._aspect_code, 0)

    def test_camera_capture_signals_actual_aspect_through_wav_and_export(self):
        raw = np.full((480, 640, 3), (80, 150, 190), np.uint8)
        def grab():
            return raw
        grab.sequential = True  # file export can call this fake camera directly
        with tempfile.TemporaryDirectory() as directory:
            wav = Path(directory)/'camera.wav'
            frames = Path(directory)/'frames'
            with patch.object(modem_screen, 'camera_source', return_value=grab), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                modem_screen.main(['--source', 'camera', '--profile', 'hd-dwt',
                                   '--write', str(wav), '--frames', '3', '--numbered'])
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                check.main(['read', '--wav', str(wav), '--save-frames', str(frames)])
            records = [json.loads(line) for line in output.getvalue().splitlines() if line.startswith('{')]
            self.assertEqual([r['frame'] for r in records], [1, 2, 3])
            self.assertTrue(all(r['aspect_code'] == 2 for r in records))
            self.assertEqual(len(list(frames.glob('*.png'))), 3)
            for path in frames.glob('*.png'):
                with Image.open(path) as image:
                    self.assertEqual(image.size, (128, 96))

    def test_write_cli_override_and_masked_presentation_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            wav = Path(directory)/'wide.wav'
            with contextlib.redirect_stdout(io.StringIO()):
                check.main(['write', '--out', str(wav), '--frames', '2', '--aspect', '16:9'])
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                check.main(['read', '--wav', str(wav)])
            records = [json.loads(line) for line in output.getvalue().splitlines() if line.startswith('{')]
            self.assertEqual([r['aspect_code'] for r in records], [4, 4])
        engine = ENG.get_engine('v3')
        coder, _ = engine.coder_for('color-dct')
        rx = engine.receiver(coder=coder)
        packet = engine.encode(np.full(coder.source_count, .2), coder, 7, 1, 1, aspect_code=7)
        result = (rx.feed(packet) + rx.flush())[0]
        presentation = DeadlinePresentationBuffer(48000, engine.wire.packet, len(result.values))
        presentation.put(result, 0, now_ns=0)
        self.assertEqual(presentation._key[1], 7)

    def test_prepare_image_does_not_pad_source(self):
        for preset in ASPECT_PRESETS:
            im = prepare_image(Image.new('RGB', (640, 360), (110, 170, 220)), preset)
            self.assertEqual(im.size, (80, 96))
            self.assertEqual(im.info['aspect_code'], ASPECT_PRESETS.index(preset))
            self.assertGreater(np.asarray(im).min(), 100)
