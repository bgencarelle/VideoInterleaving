"""Localized reconstruction and real-wire qualification, independent receivers."""
import unittest
from unittest.mock import patch
import numpy as np
from tools.v7_local_detail import LocalDetailWire,local_synthesize
from tools.v7_local_detail_bench import model_gate,transmit,recovery,fixture,band
from tools.v7_analog_frame_bench import PROFILES
from tools.v7_analog_frame import ProductionFrameWire
from tools.v7_local_detail_metrics import fft,attenuate,audio_levels
from tools.v7_local_detail_fit import refine_local,weighted_inner
from tools.v7_local_edge_fit import edge_synthesize,fit_edges
from tools.v7_residual_detail import fixed_modes,dct_factors,project_modes,synthesize_modes,ResidualDetailWire
from animation_modem.v7_core import adapt_packet_for_output


class LocalDetailTests(unittest.TestCase):
    def test_analytic_reconstruction_and_domains(self):
        atom=np.array([[12.25,-7.1,.13,-.04,.17,-.21,.12,.24]])
        width,height=128,160
        y,x=np.mgrid[:height,:width];x=(x+.5)/width-.5;y=(y+.5)/height-.5
        fx,fy,a,b,cx,cy,wx,wy=atom[0]
        expected=np.exp(-.5*(((x-cx)/wx)**2+((y-cy)/wy)**2))*(a*np.cos(2*np.pi*(fx*x+fy*y))+b*np.sin(2*np.pi*(fx*x+fy*y)))
        np.testing.assert_allclose(local_synthesize(atom,width,height),expected,atol=1e-13)
        wire=LocalDetailWire(PROFILES[0],'3:4')
        atoms=np.zeros((wire.ATOMS,8));atoms[:,6:]=.16;atoms[0]=atom[0]
        coefficients=np.zeros(wire.codec.grid.off[-1])
        prepared=wire.with_parameters(coefficients,atoms)
        np.testing.assert_allclose(wire.parameters(prepared),atoms,atol=1e-12)
        for coordinate,value in ((4,.51),(6,0),(2,.51),(0,96.4),(1,-96.4)):
            invalid=atoms.copy();invalid[0,coordinate]=value
            with self.assertRaises(ValueError):wire.with_parameters(coefficients,invalid)
        invalid=atoms.copy();invalid[0,0]=np.nan
        with self.assertRaises(ValueError):wire.with_parameters(coefficients,invalid)

    def test_actual_audio_independent_receiver(self):
        for profile in PROFILES:
            wire=LocalDetailWire(profile,'3:4')
            atoms=np.zeros((wire.ATOMS,8));atoms[:,6:]=.16
            atoms[0]=[45.25,-8.1,.18,-.07,.17,-.13,.12,.24]
            atoms[1]=[-3.2,52.1,.08,.11,-.2,.21,.24,.12]
            empty=np.zeros(wire.codec.grid.off[-1])
            prepared=wire.codec.grid.inverse(wire.with_parameters(empty,atoms))
            source=np.full((128,96,3),128,np.uint8)
            receiver,received,_=transmit(wire,prepared,source,300)
            self.assertIsNot(wire,receiver)
            result=model_gate(wire,atoms,receiver,received,(384,512))
            self.assertTrue(result['passed'],result)
            self.assertEqual(len(wire.positions)+len(wire.marker_positions),259)

    def test_pair_metrics_do_not_credit_false_detail(self):
        rng=np.random.default_rng(44);truth=rng.normal(size=(64,80))
        self.assertTrue(recovery(truth,truth)['passed'])
        for wrong in (np.zeros_like(truth),-truth,rng.normal(size=truth.shape),truth*2):
            self.assertFalse(recovery(truth,wrong)['passed'])
        self.assertFalse(recovery(truth,truth*.49)['passed'])
        self.assertFalse(recovery(truth,truth*.5+rng.normal(size=truth.shape)*.4)['passed'])
        a=fixture(10,'random-texture',48,4,.15)
        b=fixture(11,'random-texture',48,4,.15)
        self.assertFalse(np.array_equal(a,b))
        self.assertGreater(np.sqrt(np.mean(band(np.mean(a/255-b/255,axis=-1),48)**2)),.01)

    def test_fit_preserves_input_and_respects_amplitude_domain(self):
        wire=LocalDetailWire(PROFILES[0],'3:4')
        residual=np.random.default_rng(2).normal(0,.3,(48,40))
        original=residual.copy()
        atoms=wire.fit_residual(residual)
        np.testing.assert_array_equal(residual,original)
        self.assertLessEqual(np.max(abs(atoms[:,2:4])),wire.AMPLITUDE_LIMIT)
        wire.with_parameters(np.zeros(wire.codec.grid.off[-1]),atoms)

    def test_compiled_fft_and_band_against_analytic_signals(self):
        import cmath
        for length in (7,12,16):
            values=np.array([complex(i%3,-i%5) for i in range(length)])
            expected=np.array([sum(values[j]*cmath.exp(-2j*cmath.pi*k*j/length)
                                   for j in range(length)) for k in range(length)])
            np.testing.assert_allclose(fft(values),expected,atol=1e-11)
        y,x=np.mgrid[:48,:40]
        fine=np.cos(2*np.pi*(12*x/40+4*y/48))
        coarse=np.cos(2*np.pi*2*x/40)
        np.testing.assert_allclose(band(fine+coarse,12),fine,atol=1e-12)

    def test_disabled_local_wrapper_preserves_production_audio(self):
        source=fixture(3,'local-texture',32,4,.15,size=(96,128))
        for profile in PROFILES:
            production=ProductionFrameWire(profile,'3:4')
            disabled=LocalDetailWire(profile,'3:4',False)
            a=adapt_packet_for_output(production.encode_packet(source,1,123),96000)
            b=adapt_packet_for_output(disabled.encode_packet(source,1,123),96000)
            np.testing.assert_allclose(a,b,atol=1e-9,rtol=0)

    def test_volume_reduction_preserves_waveform_and_independent_decode(self):
        wave=np.array([[.6,-.3],[.2,.4],[-.8,.1]])
        rms,peak=audio_levels(wave)
        lowered,gain=attenuate(wave,rms*.75,peak*.6)
        self.assertGreater(gain,.59999)
        self.assertLessEqual(gain,.6)
        np.testing.assert_allclose(lowered,wave*gain,rtol=0,atol=1e-14)
        self.assertLessEqual(audio_levels(lowered)[0],rms*.75)
        self.assertLessEqual(audio_levels(lowered)[1],peak*.6)
        for profile in PROFILES:
            wire=LocalDetailWire(profile,'3:4')
            atoms=np.zeros((wire.ATOMS,8));atoms[:,6:]=.16
            atoms[0]=[45.25,-8.1,.18,-.07,.17,-.13,.12,.24]
            prepared=wire.codec.grid.inverse(wire.with_parameters(np.zeros(wire.codec.grid.off[-1]),atoms))
            source=np.full((128,96,3),128,np.uint8)
            _,_,full=transmit(wire,prepared,source,301)
            receiver,received,lower=transmit(wire,prepared,source,301,
                {'rms':full['rms']*.8,'peak':full['peak']*.8})
            self.assertAlmostEqual(lower['volume_gain'],.8,places=5)
            self.assertTrue(model_gate(wire,atoms,receiver,received,(384,512))['passed'])

    def test_continuous_refinement_recovers_off_grid_overlapping_atoms(self):
        truth=np.array([[13.35,-9.1,.16,-.07,.073,-.117,.11,.19],
                        [-5.2,17.4,-.08,.12,-.14,.09,.2,.13]])
        target=local_synthesize(truth,96,112)
        seed=truth.copy();seed[:,0]+=.35;seed[:,1]-=.3
        seed[:,4]+=.025;seed[:,5]-=.02;seed[:,6:]+=.025
        seed[:,2:4]*=.8
        initial_error=target-local_synthesize(seed,96,112)
        refined=refine_local(target,seed,2.,3,12)
        final_error=target-local_synthesize(refined,96,112)
        self.assertLess(weighted_inner(final_error,final_error,2.),
                        weighted_inner(initial_error,initial_error,2.)*.1)
        self.assertLess(np.sqrt(np.mean(final_error**2))/np.sqrt(np.mean(target**2)),.05)
        # Refinement respects the existing independently decodable wire domains.
        self.assertTrue((refined[:,6:]>=.06).all())
        self.assertTrue((refined[:,6:]<=.32).all())
        self.assertLessEqual(np.max(abs(refined[:,2:4])),.5)
        np.testing.assert_array_equal(seed[:,0],truth[:,0]+.35)

    def test_contour_prototype_recovers_missing_edge_without_flat_field_artifacts(self):
        from math import erf,cos,sin
        atom=np.array([[-.143,.037,.2,.12,.004,.025,.25,0.]])
        residual=edge_synthesize(atom,80,96)
        source=np.empty((96,80))
        for y in range(96):
            for x in range(80):
                u=((x+.5)/80-.5+.143)*cos(.2)+((y+.5)/96-.5-.037)*sin(.2)
                source[y,x]=.5+.125*erf(u/.004)
        fitted=fit_edges(residual,source,2)
        error=residual-edge_synthesize(fitted,80,96)
        self.assertLess(weighted_inner(error,error,2.),weighted_inner(residual,residual,2.)*.5)
        empty=fit_edges(np.zeros_like(residual),np.ones_like(source)*.5,2)
        np.testing.assert_array_equal(edge_synthesize(empty,80,96),np.zeros_like(residual))
        self.assertLessEqual(np.max(abs(fitted[:,6])),.5)

    def test_fixed_residual_modes_are_orthonormal_and_roundtrip_independently(self):
        modes=fixed_modes(64,256,192)
        self.assertEqual(len({tuple(v) for v in modes.tolist()}),64)
        vertical,horizontal=dct_factors(modes,256,192)
        probe=np.zeros(64);probe[17]=2.75
        image=synthesize_modes(probe,vertical,horizontal)
        recovered=project_modes(image,vertical,horizontal)
        np.testing.assert_allclose(recovered,probe,atol=2e-12)
        wire=ResidualDetailWire(PROFILES[0],'3:4',64)
        source=np.zeros(wire.codec.grid.off[-1])
        parameters=np.zeros(64);parameters[17]=2.75
        encoded=wire.with_parameters(source,parameters)
        np.testing.assert_allclose(wire.parameters(encoded),parameters,atol=1e-12)
        self.assertEqual(wire.last_diagnostics['saturated_coefficients'],0)

    def test_residual_base_values_are_spatial_not_dct_coefficients(self):
        source=fixture(33,'local-texture',48,4,.15,size=(192,256))
        from tools.v7_color_metrics import render as production_render
        from tools.v7_residual_detail import add_luma
        wire=ResidualDetailWire(PROFILES[0],'3:4',64)
        original=ProductionFrameWire.values(wire,source)
        coefficients=wire.codec.grid.forward(original)
        projected=wire.clean_projection(coefficients)
        spatial_base=wire.base_values(projected)
        image=production_render(wire,spatial_base,(192,256),False)
        zeros=np.zeros(wire.count)
        encoded=wire.with_parameters(projected,zeros)
        receiver_image=wire.render_received(wire.codec.grid.inverse(encoded),(192,256),False)
        np.testing.assert_allclose(receiver_image,image,atol=1e-10)
        restored=wire.codec.grid.forward(spatial_base)
        np.testing.assert_allclose(restored[wire.positions],wire.mean,atol=1e-10)

    def test_experiment_uses_production_direct_dct_source_path_once(self):
        from animation_modem import v7_source_dct
        source=fixture(34,'local-texture',48,4,.15,size=(192,256))
        wire=ResidualDetailWire(PROFILES[0],'3:4',64,enabled=False)
        with patch.object(v7_source_dct,'direct_dct_values',wraps=v7_source_dct.direct_dct_values) as direct:
            with patch.object(v7_source_dct,'source_dct_values',wraps=v7_source_dct.source_dct_values) as native:
                actual=wire.values(source)
        self.assertEqual(direct.call_count,1)
        self.assertLessEqual(native.call_count,1)
        expected=ProductionFrameWire(PROFILES[0],'3:4').values(source)
        np.testing.assert_allclose(actual,expected,atol=1e-9)

    def test_sparse_source_modes_match_known_high_frequency_truth(self):
        from tools.v7_sparse_residual import SparseResidualWire,choose_modes,sparse_synthesize,dct_vector
        from math import cos,pi,sqrt
        values=np.array([.1,.5,-.2,.7,.3,-.1])
        expected=np.array([(sqrt(1/6) if k==0 else sqrt(2/6))*
            sum(float(values[j])*cos(pi*(j+.5)*k/6) for j in range(6)) for k in range(6)])
        np.testing.assert_allclose(dct_vector(values),expected,atol=1e-12)
        truth=np.array([[103.,75.,.04],[33.,112.,-.015]])
        image=sparse_synthesize(truth,192,256)
        selected=choose_modes(image,2)
        np.testing.assert_allclose(selected,truth,atol=1e-11)
        np.testing.assert_allclose(sparse_synthesize(selected,192,256),image,atol=1e-11)
        wire=SparseResidualWire(PROFILES[0],'3:4')
        modes=np.zeros((wire.ATOMS,3));modes[:2]=truth
        encoded=wire.with_parameters(np.zeros(wire.codec.grid.off[-1]),modes)
        np.testing.assert_allclose(wire.parameters(encoded),modes,atol=1e-11)

    def test_fixed_patch_fit_recovers_independent_local_quadratures(self):
        from tools.v7_patch_residual import supports,patch_synthesize,fit_patches,PatchResidualWire
        definitions=supports(4)
        truth=np.array([[31.,-17.,.08,-.025],[23.,12.,-.04,.03],
                        [41.,19.,.055,.015],[17.,-29.,-.035,-.045]])
        image=patch_synthesize(truth,definitions,192,256)
        fitted=fit_patches(image,definitions)
        reconstruction=patch_synthesize(fitted,definitions,192,256)
        relative=np.sqrt(np.mean((image-reconstruction)**2)/np.mean(image**2))
        self.assertLess(relative,.1)
        class FourPatchWire(PatchResidualWire):
            ATOMS=4
        wire=FourPatchWire(PROFILES[0],'3:4')
        encoded=wire.with_parameters(np.zeros(wire.codec.grid.off[-1]),truth)
        np.testing.assert_allclose(wire.parameters(encoded),truth,atol=1e-11)
        # Normalized support/phase definitions preserve point samples when a
        # raster is enlarged by an odd factor (the original centers coincide).
        large=patch_synthesize(truth,definitions,576,768)
        np.testing.assert_allclose(large[1::3,1::3],image,atol=2e-5)
