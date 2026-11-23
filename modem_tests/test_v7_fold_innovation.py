"""Predictive fold candidates keep production geometry and fresh-frame recovery."""
from contextlib import redirect_stdout
import io
import unittest
import numpy as np
from PIL import Image
from animation_modem import v7
from animation_modem.v7_core import adapt_packet_for_output
from tools.v7_analog_frame import ProductionFrameWire
from tools.v7_fold_innovation import FoldInnovationWire, METHODS
from tools.v7_color_corpus import face_image, face_path


@unittest.skipUnless(face_path(0).exists(), 'face corpus unavailable')
class FoldInnovationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training = [np.asarray(Image.fromarray(face_image(i)).resize(
            (160, 192), Image.Resampling.BOX)) for i in (0, 100, 200, 300, 400, 500, 550, 599)]
        cls.source = np.asarray(Image.fromarray(face_image(1100)).resize((160, 192), Image.Resampling.BOX))

    def test_fixed_geometry_and_matched_inverse_through_each_wire(self):
        for profile in ('aspect-fold-500', 'aspect-mono-500'):
            reference = ProductionFrameWire(profile, '3:4')
            for method in METHODS:
                with self.subTest(profile=profile, method=method):
                    wire = FoldInnovationWire(profile, '3:4', method, self.training)
                    np.testing.assert_array_equal(wire.model.coder.positions, reference.model.coder.positions)
                    np.testing.assert_array_equal(wire.codec.hosts, reference.codec.hosts)
                    np.testing.assert_array_equal(wire.codec.guests, reference.codec.guests)
                    for a,b in zip(wire.model.rank_tables, reference.model.rank_tables):
                        np.testing.assert_array_equal(a,b)
                    self.assertEqual(wire.codec.grid.grids, reference.codec.grid.grids)
                    self.assertEqual(wire.codec.D, reference.codec.D)
                    self.assertEqual(wire.model.scale, reference.model.scale)
                    for rank in wire.model.rank_tables:
                        carried = wire.codec.kept[rank[rank >= 0]]
                        self.assertTrue(np.isin(wire.anchors, carried).all())
                    self.assertFalse(np.isin(wire.anchors, wire.codec.kept[wire.codec.hosts]).any())
                    coefficients = wire.codec.grid.forward(ProductionFrameWire.values(wire, self.source))
                    np.testing.assert_allclose(wire.lift(wire.lift(coefficients), inverse=True),
                                               coefficients, atol=1e-10, rtol=0)
                    # Algebra checks alone do not validate acquisition or audio.
                    packet = adapt_packet_for_output(wire.encode_packet(self.source, 91, 1100),96000)
                    self.assertEqual(len(packet),2*v7.PULSE_FRAME)
                    with redirect_stdout(io.StringIO()):
                        packets, _, _ = wire.decode(packet)
                    matched = [(r,v) for r,v in packets if r.diag.get('source_index') == 1100]
                    self.assertEqual(len(matched),1)
                    r,values = matched[0]
                    self.assertEqual(r.diag['wire_profile'],profile)
                    self.assertIsNotNone(values)
                    self.assertTrue(np.isfinite(values).all())
                    self.assertTrue(np.isfinite(wire.reconstruct(values)).all())

    def test_predictive_loop_does_not_depend_on_previous_pictures(self):
        for profile in ('aspect-fold-500', 'aspect-mono-500'):
            wire = FoldInnovationWire(profile,'3:4','ridge-klt', self.training)
            audio = [adapt_packet_for_output(wire.encode_packet(self.source,i+1,1100),96000)
                     for i in range(3)]
            with redirect_stdout(io.StringIO()):
                fresh = [wire.decode(a)[0][0][1] for a in audio]
                continuous = [v for r,v in wire.decode(np.concatenate(audio))[0]]
            self.assertEqual(len(continuous),3)
            for a,b in zip(fresh,continuous):
                np.testing.assert_allclose(a,b,atol=1e-6,rtol=0)


if __name__ == '__main__':
    unittest.main()
