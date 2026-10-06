"""Wire-backed geometry controls and independent parameter recovery."""
import unittest
from unittest.mock import patch
import numpy as np
from animation_modem import v7
from animation_modem.v7_core import adapt_packet_for_output
from tools.v7_analog_frame import ProductionFrameWire, render
from tools.v7_analog_frame_bench import first_picture, PROFILES
from tools.v7_fold_dense_probe import target
from tools.v7_geometry_wire import GeometryWire


class GeometryTests(unittest.TestCase):
    def test_transformed_control_and_independent_parameter_recovery(self):
        source=target((640,480),'x',64,.4,.3)
        for profile in PROFILES:
            with self.subTest(profile=profile):
                production=ProductionFrameWire(profile,'4:3')
                control=GeometryWire(profile,'4:3',enabled=False)
                a=adapt_packet_for_output(production.encode_packet(source,1,17),96000)
                b=adapt_packet_for_output(control.encode_packet(source,1,17),96000)
                self.assertLess(np.max(abs(a-b)),1e-9)
                _,x,_=first_picture(production,a,17)
                _,y,_=first_picture(control,b,17)
                self.assertIsNotNone(x)
                self.assertIsNotNone(y)
                self.assertLess(np.max(abs(x-y)),1e-9)
                encoder=GeometryWire(profile,'4:3')
                values=encoder.values(source)
                self.assertTrue(encoder.last_diagnostics['geometry_selected'])
                expected=encoder.parameters(encoder.codec.grid.forward(values))
                packet=adapt_packet_for_output(encoder.encode_packet(source,1,19),96000)
                receiver=GeometryWire(profile,'4:3')
                _,received,_=first_picture(receiver,packet,19)
                self.assertIsNotNone(received)
                recovered=receiver.parameters(receiver.codec.grid.forward(received))
                # Production equalization is lossy even without added noise.
                # Bound parameter error; do not require an exact analog inverse.
                self.assertLess(np.max(abs((expected-recovered)/encoder.parameter_scale)),.003)
                # Rendering from a new receiver has no source-side state.
                shown=render(receiver,received,(640,480),True)
                self.assertTrue(np.isfinite(shown).all())
                self.assertEqual(len(encoder.positions),80)
                self.assertEqual(len(encoder.marker_positions),3)
                for rank in encoder.model.rank_tables:
                    self.assertTrue(np.isin(encoder.indices,rank).all())
                    self.assertTrue(np.isin(encoder.marker_indices,rank).all())
                self.assertFalse(np.intersect1d(encoder.indices,encoder.codec.hosts).size)
                self.assertEqual(len(packet),len(a))
                from tools.v7_geometry_validation import decode_model_gate
                gate=decode_model_gate(profile,'4:3',source,19)
                self.assertTrue(gate['passed'],gate)

    def test_parameter_priors_and_marker_redundancy_do_not_change_the_wire(self):
        for profile in PROFILES:
            production=ProductionFrameWire(profile,'3:4')
            wire=GeometryWire(profile,'3:4')
            np.testing.assert_array_equal(wire.model.gain,production.model.gain)
            np.testing.assert_array_equal(wire.model.phase,production.model.phase)
            self.assertEqual(wire.model.scale,production.model.scale)
            for a,b in zip(wire.model.rank_tables,production.model.rank_tables):
                np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(wire.codec.sd_host,production.codec.sd_host)
            np.testing.assert_array_equal(wire.codec.sd_guest,production.codec.sd_guest)
            expected_prior=np.maximum(production.model.lam[wire.indices],
                wire.scale**2*np.tile([1/3]*8+[.5,.5],wire.ATOMS))
            np.testing.assert_allclose(wire.model.lam[wire.indices],expected_prior,rtol=1e-14)
            coefficients=np.zeros(wire.codec.grid.off[-1])
            atoms=np.zeros((wire.ATOMS,4))
            coefficients=wire.with_parameters(coefficients,atoms)
            # One erased marker cannot change the intended branch.
            for position,mean in zip(wire.marker_positions,wire.marker_means):
                damaged=coefficients.copy();damaged[position]=mean
                self.assertTrue(wire.geometry_selected(damaged))
            groups={int(v7.GROUP_BLOCK[group]) for group,row in enumerate(wire.model.rank_tables[0])
                    if np.isin(row,wire.marker_indices).any()}
            self.assertEqual(len(groups),3)

    def test_continuous_atom_coordinates_and_conventional_fallback(self):
        wire=GeometryWire('aspect-mono-500','4:3')
        atoms=np.zeros((wire.ATOMS,4))
        atoms[0]=[13.25,-4.1,.12,-.07]
        image=wire.correction(atoms,(256,192))
        from tools.v7_geometry_wire import fit_atom
        recovered=fit_atom(image,13.,-4.)
        np.testing.assert_allclose(recovered,atoms[0],atol=.01)
        coefficients=np.zeros(wire.codec.grid.off[-1])
        encoded=wire.with_parameters(coefficients,atoms)
        np.testing.assert_allclose(wire.parameters(encoded),atoms,atol=1e-12)
        # Both quadratures and negative vertical frequencies have an explicit
        # inverse; rendering the same atom does not depend on tile boundaries.
        large=wire.correction(atoms,(512,384))
        self.assertTrue(np.isfinite(large).all())
        source=np.full((240,320,3),128,np.uint8)
        wire.values(source)
        self.assertFalse(wire.last_diagnostics['geometry_selected'])

    def test_decode_gate_rejects_small_parameter_error_that_breaks_the_picture(self):
        from tools.v7_geometry_validation import DiagnosticWire,decode_model_gate
        source=target((640,480),'x',64,.4,.3)
        wire=DiagnosticWire('aspect-mono-500','4:3')
        prepared=wire.force_values(source)
        coefficients=wire.codec.grid.forward(prepared)
        atoms=wire.parameters(coefficients)
        strongest=np.argmax(np.sum(atoms[:,2:]**2,axis=1))
        atoms[strongest,0]+=.1
        damaged=wire.codec.grid.inverse(wire.with_parameters(coefficients,atoms))
        with patch('tools.v7_geometry_validation.first_picture',return_value=(None,damaged,0.)):
            result=decode_model_gate('aspect-mono-500','4:3',source,0)
        self.assertTrue(result['picture_available'])
        self.assertTrue(result['geometry_branch_received'])
        self.assertLess(result['normalized_parameter_max_abs_error'],.003)
        self.assertGreater(result['relative_correction_rms_error'],.1)
        self.assertFalse(result['passed'])

    def test_small_amplitudes_and_one_coarse_coordinate_error(self):
        wire=GeometryWire('aspect-mono-500','3:4')
        atoms=np.zeros((wire.ATOMS,4))
        atoms[0]=[13.499,-4.501,1e-6,-1e-8]
        coefficients=np.zeros(wire.codec.grid.off[-1])
        encoded=wire.with_parameters(coefficients,atoms)
        np.testing.assert_allclose(wire.parameters(encoded),atoms,atol=1e-12)
        # One whole-cycle error in one copy must not move the reconstructed
        # frequency. This specifically guards the observed mono cycle slip.
        damaged=encoded.copy()
        damaged[wire.positions[0]]+=wire.scale[0]/wire.FREQUENCY_LIMIT
        np.testing.assert_allclose(wire.parameters(damaged),atoms,atol=1e-12)
        too_large=atoms.copy();too_large[0,2]=wire.AMPLITUDE_LIMIT*1.1
        with self.assertRaises(ValueError):wire.with_parameters(coefficients,too_large)


if __name__=='__main__':unittest.main()
