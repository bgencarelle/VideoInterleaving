"""Dynamic component contracts checked through actual fresh V7 packets."""
from contextlib import redirect_stdout
import io
import unittest

import numpy as np
from PIL import Image
from animation_modem import v7
from animation_modem.v7_core import adapt_packet_for_output
from tools.v7_analog_frame import (
    AnalogFrameWire, ProductionFrameWire, GRID_FAMILIES, REPAIRS, fit, render, SPLITS)
from tools.v7_color_transforms import ColorTransform, NAMES
from tools.v7_color_corpus import face_image, face_path, Movie, FIXTURES, ROOT


@unittest.skipUnless(face_path(0).exists(), 'face corpus unavailable')
class AnalogFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = [np.asarray(Image.fromarray(face_image(i)).resize(
            (160, 192), Image.Resampling.BOX)) for i in (0, 200, 400, 599)]
        cls.pixels = np.concatenate([f.reshape(-1, 3)[::16]/255 for f in cls.frames])

    def packet(self, wire, frame=0, counter=1):
        return adapt_packet_for_output(wire.encode_packet(self.frames[frame], counter, 173+frame), 96000)

    def fresh(self, wire, audio, source_index=173):
        hooks = v7.decode_frame, v7._equalize_numba, v7._equalize_numpy
        with redirect_stdout(io.StringIO()):
            results, _, _ = wire.decode(audio)
        self.assertEqual(hooks, (v7.decode_frame, v7._equalize_numba, v7._equalize_numpy))
        found = [(r, v) for r, v in results if r.diag.get('source_index') == source_index]
        self.assertEqual(len(found), 1)
        result, values = found[0]
        self.assertIsNotNone(values)
        self.assertEqual(result.diag['wire_profile'], wire.profile)
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(len(audio), v7.PULSE_FRAME*2)
        return values

    def test_all_transforms_are_independently_decodable_in_both_profiles(self):
        for name in NAMES:
            transform = (ColorTransform.fit_pca(name, self.pixels) if name.startswith('pca')
                         else ColorTransform(name))
            statistics = fit(self.frames, transform, GRID_FAMILIES['equal'])
            for profile in ('aspect-fold-500', 'aspect-mono-500'):
                with self.subTest(transform=name, profile=profile):
                    wire = AnalogFrameWire(profile, '3:4', transform, statistics)
                    values = self.fresh(wire, self.packet(wire))
                    self.assertEqual(values.size, 3*96*80)
                    self.assertTrue(np.isfinite(render(wire, values, (120,160), True)).all())

    def test_grid_split_repair_contracts_and_arbitrary_first_counter(self):
        transform = ColorTransform('ycocg')
        for grid in GRID_FAMILIES:
            statistics = fit(self.frames, transform, GRID_FAMILIES[grid])
            for profile in ('aspect-fold-500', 'aspect-mono-500'):
                for split in SPLITS:
                    for repair in REPAIRS:
                        with self.subTest(grid=grid, profile=profile, split=split, repair=repair):
                            wire = AnalogFrameWire(profile, '3:4', transform, statistics,
                                                   grid=grid, split=split, repair=repair)
                            values = self.fresh(wire, self.packet(wire, frame=2, counter=29), 175)
                            self.assertEqual(values.size, sum(r*c for r,c in GRID_FAMILIES[grid]))
                            count = np.count_nonzero(wire.model.rank_tables[0] >= 0)
                            self.assertEqual(count, 2320 if profile == 'aspect-fold-500' else 1264)
                            self.assertTrue(np.isfinite(render(wire, values, (120,160), True)).all())

    def test_static_loop_has_no_previous_picture_dependency(self):
        transform = ColorTransform('rgb')
        statistics = fit(self.frames, transform, GRID_FAMILIES['expanded'])
        for profile in ('aspect-fold-500', 'aspect-mono-500'):
            wire = AnalogFrameWire(profile, '3:4', transform, statistics, grid='expanded')
            packets = [self.packet(wire, counter=i+1) for i in range(4)]
            independent = [self.fresh(wire, audio) for audio in packets]
            with redirect_stdout(io.StringIO()):
                received, _, _ = wire.decode(np.concatenate(packets))
            values = [v for r, v in received if r.diag.get('source_index') == 173]
            self.assertEqual(len(values), 4)
            for a, b in zip(independent, values):
                np.testing.assert_allclose(a, b, atol=1e-6, rtol=0)

    def test_production_luma_controls_round_trip(self):
        for profile in ('aspect-fold-500', 'aspect-mono-500'):
            for repair in ('ordinary', 'off', 'linear', 'clip'):
                with self.subTest(profile=profile, repair=repair):
                    wire = ProductionFrameWire(profile, '3:4', repair)
                    self.fresh(wire, self.packet(wire))

    def test_existing_movie_can_be_entered_at_multiple_frames(self):
        movie = Movie(FIXTURES/'v7_pixel_motion_4x3.mp4', ROOT/'tmp/v7-analog-frame/test-cache')
        transform = ColorTransform('ycocg')
        statistics = fit(self.frames, transform, GRID_FAMILIES['expanded'])
        for profile in ('aspect-fold-500', 'aspect-mono-500'):
            wire = AnalogFrameWire(profile, '4:3', transform, statistics, grid='expanded')
            for index in (0, len(movie.pts)//2, len(movie.pts)-1):
                with self.subTest(profile=profile, frame=index):
                    audio = adapt_packet_for_output(wire.encode_packet(movie.image(index), 73, index), 96000)
                    values = self.fresh(wire, audio, index)
                    self.assertTrue(np.isfinite(render(wire, values, (160,120), True)).all())


if __name__ == '__main__':
    unittest.main()
