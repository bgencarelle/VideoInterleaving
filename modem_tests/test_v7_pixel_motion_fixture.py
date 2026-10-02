"""The generated pixel-motion fixtures and their read-back strip."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import unittest

import numpy as np
from PIL import Image, ImageFilter

from tools.generate_v7_pixel_motion_test import (
    FPS, FRAMES, GRID_COLUMNS, GRID_ROWS, SHAPES, frame_image,
    picture_index, read_picture_index)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT/'modem_tests'/'fixtures'


class PixelMotionFixtureTests(unittest.TestCase):
    def test_frame_image_is_deterministic(self):
        for shape, size in SHAPES.items():
            first = frame_image(shape, 123)
            second = frame_image(shape, 123)
            self.assertEqual(first.size, size)
            self.assertEqual(hashlib.sha256(first.tobytes()).hexdigest(),
                             hashlib.sha256(second.tobytes()).hexdigest())
            self.assertNotEqual(first.tobytes(),
                                frame_image(shape, 124).tobytes())

    def test_picture_index_follows_the_packet_rate(self):
        self.assertEqual(picture_index(0), 0)
        self.assertEqual(picture_index(FRAMES-1),
                         int((FRAMES-1)/FPS*48000/3920))

    def test_strip_reads_back_at_full_size_and_reduced(self):
        for shape in SHAPES:
            for index in (0, 77, 150, FRAMES-1):
                image = frame_image(shape, index)
                expected = picture_index(index)
                self.assertEqual(
                    read_picture_index(np.asarray(image)), expected)
                reduced = image.resize(
                    (GRID_COLUMNS, GRID_ROWS), Image.LANCZOS).filter(
                        ImageFilter.GaussianBlur(1))
                self.assertEqual(
                    read_picture_index(np.asarray(reduced)), expected)

    def test_strip_is_not_read_from_a_picture_without_one(self):
        self.assertIsNone(read_picture_index(
            np.zeros((GRID_ROWS, GRID_COLUMNS, 3), dtype=np.uint8)))
        self.assertIsNone(read_picture_index(
            np.full((GRID_ROWS, GRID_COLUMNS, 3), 255, dtype=np.uint8)))

    def test_fixture_files_have_the_expected_size_and_rate(self):
        if shutil.which('ffprobe') is None:
            self.skipTest('ffprobe is not installed')
        for shape, (width, height) in SHAPES.items():
            path = FIXTURES/f'v7_pixel_motion_{shape}.mp4'
            self.assertTrue(path.is_file(), path)
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries',
                 'stream=codec_type,codec_name,width,height,r_frame_rate,'
                 'nb_frames', '-of', 'json', str(path)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            streams = json.loads(result.stdout)['streams']
            self.assertEqual([s['codec_type'] for s in streams], ['video'])
            video = streams[0]
            self.assertEqual(video['codec_name'], 'h264')
            self.assertEqual((video['width'], video['height']),
                             (width, height))
            self.assertEqual(video['r_frame_rate'], f'{FPS}/1')
            self.assertEqual(int(video['nb_frames']), FRAMES)


if __name__ == '__main__':
    unittest.main()
