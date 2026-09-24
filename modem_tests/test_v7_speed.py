from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from animation_modem import v7


FIXTURE = Path(__file__).parent / 'fixtures/v7_reference_face.png'


class V7SpeedTests(unittest.TestCase):
    def test_reference_packet_lengths(self):
        self.assertEqual(v7.speed_length(v7.PULSE_FRAME, v7.RATE, 1), 3920)
        self.assertEqual(v7.speed_length(v7.PULSE_FRAME, v7.RATE, 1.5), 2613)
        self.assertEqual(v7.speed_length(v7.PULSE_FRAME, v7.RATE, 2), 1960)

    def test_speed_transform_keeps_packet_boundaries(self):
        source = np.zeros((v7.PULSE_FRAME*2, 2), np.float32)
        fast = v7.speed_pulse_stream(source, 1.5)
        self.assertEqual(fast.shape, (2613*2, 2))
        self.assertTrue(np.isfinite(fast).all())

    def test_one_point_five_x_pulse_stream_decodes(self):
        model = v7.build_model(
            FIXTURE, .1521/np.sqrt(1+10**(v7.CLOCK_REL_DB/10)),
            encode_filter='nearest')
        image = Image.open(FIXTURE)
        values = v7.image_values(v7.prepare_image(image, 'nearest'),
                                 model.coder.grids, 'nearest')
        wire = v7.encode_pulse_stream(
            model, [values, values, values], source_indices=[0, 1, 2])
        fast = v7.speed_pulse_stream(wire, 1.5)
        results, info = v7.decode_pulse_stream(model, fast)
        self.assertGreaterEqual(len(results), 2, info)
        self.assertEqual([r.diag['source_index'] for r in results[:2]], [0, 1])
        self.assertAlmostEqual(results[0].diag['playback_speed'], 1.5,
                               delta=.02)

    def test_two_x_96khz_stream_preserves_aspect_metadata(self):
        model = v7.build_model(
            FIXTURE, .1521/np.sqrt(1+10**(v7.CLOCK_REL_DB/10)),
            encode_filter='nearest')
        with Image.open(FIXTURE) as image:
            values = v7.image_values(v7.prepare_image(image, 'nearest'),
                                     model.coder.grids, 'nearest')
        aspects = [3, 1, 5, 2]
        wire = v7.encode_pulse_stream(
            model, [values]*len(aspects), aspect_codes=aspects,
            source_indices=list(range(len(aspects))))
        fast = v7.speed_pulse_stream(wire, 2, rate=96000)
        results, info = v7.decode_pulse_stream(
            model, fast, sample_rate=96000)
        self.assertGreaterEqual(len(results), len(aspects)-1, info)
        self.assertEqual([result.diag['aspect_code'] for result in results],
                         aspects[:len(results)])
        self.assertAlmostEqual(results[0].diag['playback_speed'], 2, delta=.02)

    def test_quarter_to_four_x_playback_scale_is_acquired(self):
        model = v7.build_model(
            FIXTURE, .1521/np.sqrt(1+10**(v7.CLOCK_REL_DB/10)),
            encode_filter='nearest')
        with Image.open(FIXTURE) as image:
            values = v7.image_values(v7.prepare_image(image, 'nearest'),
                                     model.coder.grids, 'nearest')
        wire = v7.encode_pulse_stream(
            model, [values]*5, source_indices=list(range(5)))
        cases = ((32000, .25), (48000, 3.0), (48000, 4.0),
                 (96000, 3.0), (96000, 3.4), (96000, 4.0))
        for rate, speed in cases:
            audio = v7.speed_pulse_stream(wire, speed, rate=rate)
            results, info = v7.decode_pulse_stream(
                model, audio, sample_rate=rate)
            with self.subTest(rate=rate, speed=speed):
                self.assertGreaterEqual(len(results), 4, info)
                for result in results[:4]:
                    self.assertAlmostEqual(
                        result.diag['playback_speed'], speed, delta=.03)
                if speed == .25 or (rate == 96000 and speed < 4):
                    self.assertTrue(
                        all(result.status == 'received' for result in results[:4]),
                        info)


if __name__ == '__main__':
    unittest.main()
