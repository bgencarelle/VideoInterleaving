"""Descending folded-surface capacity probes use the unchanged V7 wire."""
from contextlib import redirect_stdout
import io
import unittest
import numpy as np
from PIL import Image
from animation_modem import v7
from animation_modem.v7_core import adapt_packet_for_output
from tools.v7_analog_frame import ProductionFrameWire
from tools.v7_fold_surface import FoldSurfaceWire,DIMENSIONS,surface_encode,surface_decode
from tools.v7_color_corpus import face_image,face_path,Movie,FIXTURES,ROOT


@unittest.skipUnless(face_path(0).exists(),'face corpus unavailable')
class FoldSurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training=[np.asarray(Image.fromarray(face_image(i)).resize((160,192),Image.Resampling.BOX))
                      for i in (0,200,400,599)]
        cls.source=np.asarray(Image.fromarray(face_image(1100)).resize((160,192),Image.Resampling.BOX))

    def test_descending_dimensions_use_fixed_slots_and_fresh_wire_decoding(self):
        rng=np.random.default_rng(2026)
        for profile in ('aspect-fold-500','aspect-mono-500'):
            reference=ProductionFrameWire(profile,'3:4')
            for d,k in DIMENSIONS:
                for strips in (3,4,5,8):
                    with self.subTest(profile=profile,dimensions=(d,k),strips=strips):
                        source=rng.uniform(.05,.95,(32,d))
                        reconstructed=surface_decode(surface_encode(source,k,strips),d,strips)
                        np.testing.assert_allclose(reconstructed[:,:k],source[:,:k],atol=1e-7,rtol=0)
                        if d>k:
                            self.assertLessEqual(float(np.max(abs(reconstructed[:,k:]-source[:,k:]))),.5/strips+1e-7)
                        wire=FoldSurfaceWire(profile,'3:4',d,k,strips,self.training)
                        physical=wire.codec.grid.forward(ProductionFrameWire.values(wire,self.source))[wire.source_positions]
                        recovered=wire.denormalize(wire.normalize(physical))
                        # Fold-native clipping is the production compander's
                        # existing source loss, not a new unbounded inverse.
                        limit=wire.source_compander.compand[0]*wire.source_sd
                        np.testing.assert_allclose(recovered,np.clip(physical,-limit,limit),atol=1e-9,rtol=0)
                        self.assertEqual(wire.source_diagnostics(self.source)['source_range_overload_fraction'],0.)
                        np.testing.assert_array_equal(wire.codec.hosts,reference.codec.hosts)
                        np.testing.assert_array_equal(wire.codec.guests,reference.codec.guests)
                        np.testing.assert_array_equal(wire.model.coder.positions,reference.model.coder.positions)
                        for a,b in zip(wire.model.rank_tables,reference.model.rank_tables):
                            np.testing.assert_array_equal(a,b)
                        self.assertEqual(wire.model.scale,reference.model.scale)
                        self.assertEqual(len(np.unique(wire.source_positions)),wire.source_positions.size)
                        packet=adapt_packet_for_output(wire.encode_packet(self.source,79,1100),96000)
                        self.assertEqual(len(packet),2*v7.PULSE_FRAME)
                        with redirect_stdout(io.StringIO()):
                            received,_,_=wire.decode(packet)
                        matched=[(r,v) for r,v in received if r.diag.get('source_index')==1100]
                        self.assertEqual(len(matched),1)
                        result,values=matched[0]
                        self.assertIsNotNone(values)
                        self.assertTrue(np.isfinite(values).all())
                        self.assertEqual(result.diag['wire_profile'],profile)
                        self.assertEqual(wire.source_diagnostics(self.source)['extra_source_coordinates'],
                                         (484//k)*(d-k))

    def test_lower_amplitude_is_not_canceled_by_fold_normalization(self):
        for profile in ('aspect-fold-500','aspect-mono-500'):
            reference=FoldSurfaceWire(profile,'3:4',9,1,4,self.training)
            original=reference.codec.grid.forward(reference.values(self.source))
            for headroom in (.75,.5,.25,.125,.0625):
                with self.subTest(profile=profile,headroom=headroom):
                    wire=FoldSurfaceWire(profile,'3:4',9,1,4,self.training,headroom=headroom)
                    np.testing.assert_array_equal(wire.codec.sd_guest,reference.codec.sd_guest)
                    np.testing.assert_array_equal(wire.codec.pattern,reference.codec.pattern)
                    lowered=wire.codec.grid.forward(wire.values(self.source))
                    np.testing.assert_allclose(lowered[wire.carriers],original[wire.carriers]*headroom,atol=1e-10,rtol=0)
                    ordinary=np.setdiff1d(wire.codec.kept,np.union1d(wire.codec.guests,wire.codec.kept[wire.codec.hosts]))
                    np.testing.assert_allclose(lowered[ordinary],original[ordinary],atol=1e-10,rtol=0)
                    self.assertLessEqual(wire.source_diagnostics(self.source)['packed_carrier_abs_peak'],headroom)
                    packet=adapt_packet_for_output(wire.encode_packet(self.source,79,1100),96000)
                    with redirect_stdout(io.StringIO()):
                        received,_,_=wire.decode(packet)
                    matched=[values for r,values in received if r.diag.get('source_index')==1100]
                    self.assertEqual(len(matched),1)
                    self.assertIsNotNone(matched[0])
                    self.assertTrue(np.isfinite(matched[0]).all())

    def test_identity_wrapper_matches_production_audio_and_decoded_fidelity(self):
        movie=Movie(FIXTURES/'v7_pixel_motion_4x3.mp4',ROOT/'tmp/v7-fold-innovation/validation/cache')
        sources=[('face',self.source,'3:4',1100),('movie-first',movie.image(0),'4:3',0),
                 ('movie-last',movie.image(len(movie.pts)-1),'4:3',len(movie.pts)-1)]
        for profile in ('aspect-fold-500','aspect-mono-500'):
            for name,source,layout,index in sources:
                with self.subTest(profile=profile,source=name):
                    reference=ProductionFrameWire(profile,layout)
                    control=FoldSurfaceWire(profile,layout,1,1,4,self.training,normalization='identity')
                    np.testing.assert_array_equal(control.values(source),reference.values(source))
                    a=adapt_packet_for_output(reference.encode_packet(source,79,index),96000)
                    b=adapt_packet_for_output(control.encode_packet(source,79,index),96000)
                    np.testing.assert_array_equal(a,b)
                    with redirect_stdout(io.StringIO()):
                        expected,_,_=reference.decode(a)
                        received,_,_=control.decode(b)
                    self.assertEqual(len(expected),len(received))
                    self.assertTrue(expected)
                    for (r,x),(s,y) in zip(expected,received):
                        self.assertEqual(r.status,s.status)
                        self.assertEqual(r.diag.get('source_index'),s.diag.get('source_index'))
                        self.assertIsNotNone(x)
                        self.assertIsNotNone(y)
                        np.testing.assert_array_equal(x,y)
                        np.testing.assert_array_equal(reference.reconstruct(x),control.reconstruct(y))

    def test_transformed_native_control_preserves_production_wire_fidelity(self):
        movie=Movie(FIXTURES/'v7_pixel_motion_4x3.mp4',ROOT/'tmp/v7-fold-innovation/validation/cache')
        sources=[(self.source,'3:4',1100),(movie.image(0),'4:3',0),
                 (movie.image(len(movie.pts)-1),'4:3',len(movie.pts)-1)]
        for profile in ('aspect-fold-500','aspect-mono-500'):
            for source,layout,index in sources:
                with self.subTest(profile=profile,layout=layout,index=index):
                    baseline=ProductionFrameWire(profile,layout)
                    wire=FoldSurfaceWire(profile,layout,1,1,4,self.training,normalization='fold-native')
                    # This exercises the transformed path, not the no-op branch.
                    before=wire.codec.grid.forward(ProductionFrameWire.values(wire,source))
                    after=wire.codec.grid.forward(wire.values(source))
                    self.assertGreater(float(np.max(abs(before[wire.carriers]-after[wire.carriers]))),1e-5)
                    a=adapt_packet_for_output(baseline.encode_packet(source,79,index),96000)
                    b=adapt_packet_for_output(wire.encode_packet(source,79,index),96000)
                    np.testing.assert_array_equal(a,b)
                    with redirect_stdout(io.StringIO()):
                        expected,_,_=baseline.decode(a)
                        actual,_,_=wire.decode(b)
                    self.assertEqual(len(expected),len(actual))
                    self.assertTrue(expected)
                    for (r,x),(s,y) in zip(expected,actual):
                        self.assertEqual(r.status,s.status)
                        self.assertEqual(r.diag.get('source_index'),s.diag.get('source_index'))
                        self.assertIsNotNone(x)
                        self.assertIsNotNone(y)
                        np.testing.assert_allclose(x,y,atol=1e-9,rtol=0)
                        np.testing.assert_allclose(baseline.reconstruct(x),wire.reconstruct(y),atol=1e-9,rtol=0)


if __name__=='__main__':
    unittest.main()
