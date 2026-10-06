"""End-to-end checks for the experimental color benchmark, all using V7 audio."""
from contextlib import redirect_stdout
import io
import unittest

import numpy as np
from PIL import Image

from animation_modem import v7
from animation_modem.v7_core import adapt_packet_for_output
from tools.v7_color_transforms import ColorTransform, NAMES
from tools.v7_color_wire import ColorWire, fit_statistics, SPLITS
from tools.v7_color_corpus import face_image, face_path, Movie, FIXTURES, ROOT
from tools.v7_color_metrics import bar_metrics, picture_metrics, render


@unittest.skipUnless(face_path(0).is_file(), 'repository face image corpus is unavailable')
class ColorResolutionWireTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = [np.asarray(Image.fromarray(face_image(i)).resize(
            (160, 192), Image.Resampling.BOX)) for i in (0, 200, 400, 599)]
        cls.pixels = np.concatenate([image.reshape(-1, 3)[::16]/255 for image in cls.frames])

    def round_trip(self, wire, frames=None):
        frames = frames or [self.frames[0]]*5
        audio = np.concatenate([adapt_packet_for_output(
            wire.encode_packet(image, i+1, i), 96000) for i, image in enumerate(frames)])
        with redirect_stdout(io.StringIO()):
            results, info, elapsed = wire.decode(audio)
        good = [(r, values) for r, values in results if values is not None]
        self.assertGreaterEqual(len(good), 3)
        self.assertTrue(all(np.isfinite(values).all() for _, values in good))
        self.assertEqual(good[-1][0].diag['source_index'], len(frames)-1)
        self.assertEqual(good[-1][0].diag['wire_profile'], wire.profile)
        return audio, good[-1][1]

    def test_every_color_option_round_trips_through_both_wires(self):
        hooks = v7.decode_frame, v7._equalize_numba, v7._equalize_numpy
        for name in NAMES:
            with self.subTest(transform=name):
                transform = (ColorTransform.fit_pca(name, self.pixels)
                             if name.startswith('pca') else ColorTransform(name))
                x = self.frames[0]/255
                y = transform.inverse(transform.forward(x))
                self.assertTrue(np.isfinite(y).all())
                if name != 'gray':
                    self.assertLess(float(np.mean((y-x)**2)), .0002)
                statistics = fit_statistics(self.frames, transform)
                for profile in ('aspect-fold-500', 'aspect-mono-500'):
                    wire = ColorWire(profile, '3:4', transform, statistics=statistics)
                    audio, values = self.round_trip(wire)
                    decoded = wire.reconstruct(values)
                    self.assertTrue(np.isfinite(decoded).all())
                    self.assertEqual(len(audio), 5*v7.PULSE_FRAME*2)
                    self.assertEqual(hooks, (v7.decode_frame, v7._equalize_numba,
                                             v7._equalize_numpy))

    def test_kernels_and_reallocated_detail_use_fixed_wire_budget(self):
        transform = ColorTransform('ycocg')
        statistics = fit_statistics(self.frames, transform)
        for profile in ('aspect-fold-500', 'aspect-mono-500'):
            for kernel in ('viewer_solve', 'native_source', 'csf_peak', 'band_taper', 'antiring'):
                for split in SPLITS:
                    with self.subTest(profile=profile, kernel=kernel, split=split):
                        wire = ColorWire(profile, '3:4', transform, kernel, split, statistics)
                        audio, values = self.round_trip(wire)
                        self.assertEqual(len(wire.model.mu), 2880)
                        slots = np.count_nonzero(wire.model.rank_tables[0] >= 0)
                        self.assertEqual(slots, 2320 if profile == 'aspect-fold-500' else 1264)
                        for display in (False, True):
                            self.assertTrue(np.isfinite(render(wire, values, (120, 160), display)).all())

    def test_existing_movie_targets_are_scored_after_wire_decode(self):
        movie = Movie(FIXTURES/'v7_pixel_motion_4x3.mp4',
                      ROOT/'tmp/v7-color-resolution/test-cache')
        wire = ColorWire('aspect-fold-500', '4:3', ColorTransform('pillow-ycbcr'))
        frames = [movie.image(movie.frame_index(i*v7.PULSE_FRAME/v7.RATE)) for i in range(5)]
        audio, values = self.round_trip(wire, frames)
        size = (640, 480)
        source = frames[-1]/255
        received = render(wire, values, size)
        metrics = bar_metrics(source, received)
        self.assertIn('vertical_pitch2_contrast', metrics)
        self.assertIn('horizontal_pitch8_resolved', metrics)
        self.assertTrue(np.isfinite(metrics['vertical_pitch4_contrast']))
        # An actual received picture and a wrong-polarity version must not be
        # confused with higher resolution simply because both contain edges.
        inverted = bar_metrics(source, 1-received)
        self.assertFalse(inverted['vertical_pitch8_resolved'])
        comparison = picture_metrics(source, received)
        self.assertTrue(np.isfinite(comparison['ssim']))

    def test_loss_holds_last_picture_without_fabricating_a_frame(self):
        wire = ColorWire('aspect-fold-500', '3:4', ColorTransform('pillow-ycbcr'))
        audio, values = self.round_trip(wire)
        packet = v7.PULSE_FRAME*2
        damaged = audio.copy()
        damaged[2*packet:3*packet] = 0
        with redirect_stdout(io.StringIO()):
            results, _, _ = wire.decode(damaged)
        pictures = {r.diag.get('source_index'): p for r, p in results if p is not None}
        self.assertNotIn(2, pictures)
        self.assertIn(1, pictures)
        self.assertIn(3, pictures)


if __name__ == '__main__':
    unittest.main()
