"""Small independent DP oracle, native encoding and optional-backend bounds."""
import concurrent.futures
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from ascii_color_sharing import ColorSharingEngine
from ascii_encoding import encode_rgb_rows


def reference(rgb, lab, tolerance, edge):
    """Unpruned prototype algorithm, deliberately independent of the C code."""
    out = rgb.copy()
    for y in range(rgb.shape[0]):
        width = rgb.shape[1]
        cost = [1e30]*(width+1)
        loss = [1e30]*(width+1)
        parent = [0]*(width+1)
        representative = [0]*(width+1)
        cost[0] = loss[0] = 0
        for start in range(width):
            total = [0., 0., 0.]
            for end in range(start, width):
                if end > start and sum((float(lab[y,end,c])-float(lab[y,end-1,c]))**2
                                       for c in range(3)) > edge*edge:
                    break
                for c in range(3):
                    total[c] += float(lab[y,end,c])
                count = end-start+1
                def distance(x):
                    return sum((float(lab[y,x,c])-total[c]/count)**2 for c in range(3))
                chosen = min(range(start,end+1), key=distance)
                distances = [sum((float(lab[y,x,c])-float(lab[y,chosen,c]))**2
                                 for c in range(3)) for x in range(start,end+1)]
                if max(distances) > tolerance*tolerance+1e-12:
                    continue
                candidate = cost[start]+10+sum(len(str(int(v))) for v in rgb[y,chosen])
                candidate_loss = loss[start]+sum(distances)
                if candidate < cost[end+1] or (candidate == cost[end+1] and candidate_loss < loss[end+1]):
                    cost[end+1], loss[end+1] = candidate, candidate_loss
                    parent[end+1], representative[end+1] = start, chosen
        end = width
        while end:
            start = parent[end]
            out[y,start:end] = rgb[y,representative[end]]
            end = start
    return out


class ColorSharingTests(unittest.TestCase):
    def test_pruned_native_matches_unpruned_oracle(self):
        rng = np.random.default_rng(1710)
        fixtures = [rng.integers(0,256,(2,15,3),dtype=np.uint8),
                    rng.integers(95,108,(2,15,3),dtype=np.uint8),
                    np.full((2,15,3),128,np.uint8),
                    np.repeat(np.arange(15,dtype=np.uint8)[None,:,None]*8,3,axis=2)]
        for tolerance in (0., 1., 2., 3., 100.):
            engine = ColorSharingEngine(tolerance=tolerance)
            for frame in fixtures:
                with self.subTest(tolerance=tolerance, shape=frame.shape):
                    lifted = engine.lift(frame)
                    lab = engine.perceptual(lifted)
                    np.testing.assert_array_equal(engine.share(frame), reference(lifted,lab,tolerance,6.))
                    self.assertEqual(frame.dtype, np.uint8)

    def test_lift_endpoints_monotonicity_ratios_and_input_ownership(self):
        engine = ColorSharingEngine()
        ramp = np.repeat(np.arange(256,dtype=np.uint8)[None,:,None],3,axis=2)
        old = ramp.copy()
        lifted = engine.lift(ramp)
        self.assertEqual(lifted[0,0,0],0)
        self.assertEqual(lifted[0,255,0],255)
        self.assertEqual(lifted[0,128,0],137)
        self.assertTrue(np.all(np.diff(lifted[0,:,0].astype(int))>=0))
        np.testing.assert_array_equal(old,ramp)
        colors = np.array([[[120,85,65],[200,150,110]]],np.uint8)
        expected = colors.astype(float)*engine.scale[colors.max(axis=2),None]
        self.assertLessEqual(float(np.abs(engine.lift(colors)-expected).max()),.5)

    def test_encoder_matches_python_for_unicode_runs_and_rows(self):
        rng = np.random.default_rng(45)
        rgb = rng.integers(0,256,(8,12,3),dtype=np.uint8)
        rgb[0,:] = (0,9,255)
        rgb[1,:] = (10,99,100)
        chars = np.array(list('MW .é漢█😀'),dtype='U1')[rng.integers(0,8,(8,12))]
        chars[0,0] = ''
        engine = ColorSharingEngine()
        self.assertEqual(engine.encode_colors(rgb,chars),encode_rgb_rows(rgb,chars))
        shared = engine.share(rgb)
        self.assertEqual(engine.encode(rgb,chars),encode_rgb_rows(shared,chars))

    def test_noncontiguous_inputs(self):
        engine = ColorSharingEngine()
        rgb = np.zeros((4,8,3),np.uint8)[:,::2]
        chars = np.full((4,8),'M',dtype='U1')[:,::2]
        self.assertEqual(engine.encode(rgb,chars),encode_rgb_rows(engine.share(rgb.copy()),chars.copy()))

    def test_optional_lightness_protection_and_no_double_lift(self):
        engine = ColorSharingEngine(tolerance=3,lightness_limit=.5)
        ramp = np.rint(np.linspace(0,255,60)).astype(np.uint8)
        rgb = np.repeat(ramp[None,:,None],3,axis=2)
        lifted = engine.lift(rgb)
        shared = engine.share_colors(lifted)
        np.testing.assert_array_equal(shared,engine.share(rgb))
        L = engine.perceptual(lifted)[:,:,0]
        allowed = np.minimum(.5,np.minimum(.02*np.maximum(0,L),.05*np.maximum(0,100-L)))
        actual = np.abs(engine.perceptual(shared)[:,:,0]-L)
        self.assertTrue(np.all(actual<=allowed+1e-10))
        np.testing.assert_array_equal(shared[:,0],lifted[:,0])
        np.testing.assert_array_equal(shared[:,-1],lifted[:,-1])

    def test_420_changes_only_comparison_chroma_with_full_lightness(self):
        engine = ColorSharingEngine(chroma_model='420',edge=100)
        lab = np.array([[[10,1,-1],[11,2,-2]],[[12,3,-3],[13,4,-4]]],dtype=float)
        old = lab.copy()
        model = engine.comparison_model(lab)
        np.testing.assert_array_equal(model[:,:,0],lab[:,:,0])
        np.testing.assert_array_equal(model[:,:,1],np.full((2,2),2.5))
        np.testing.assert_array_equal(model[:,:,2],np.full((2,2),-2.5))
        np.testing.assert_array_equal(lab,old)
        np.testing.assert_array_equal(ColorSharingEngine().comparison_model(lab),lab)

    def test_420_handles_edges_odd_dimensions_and_uses_source_rgb(self):
        engine = ColorSharingEngine(chroma_model='420')
        rgb = np.zeros((3,5,3),np.uint8)
        rgb[:,:2]=(180,75,55);rgb[:,2:]=(45,95,190)
        np.testing.assert_array_equal(engine.share(rgb),engine.lift(rgb))
        rng=np.random.default_rng(420)
        rgb=rng.integers(95,110,(5,9,3),dtype=np.uint8)
        old=rgb.copy();lifted=engine.lift(rgb);shared=engine.share(rgb)
        L=engine.perceptual(lifted)[:,:,0]
        np.testing.assert_array_equal(engine.comparison_model(engine.perceptual(lifted))[:,:,0],L)
        self.assertLessEqual(float(np.abs(engine.perceptual(shared)[:,:,0]-L).max()),1+1e-10)
        for y in range(5):
            source={tuple(v) for v in lifted[y]}
            self.assertTrue(all(tuple(v) in source for v in shared[y]))
        np.testing.assert_array_equal(rgb,old)
        chars=np.full((5,9),'M',dtype='U1')
        self.assertEqual(engine.encode(rgb,chars),encode_rgb_rows(shared,chars))

    def test_invalid_inputs_rejected_before_native_calls(self):
        engine = ColorSharingEngine()
        for frame in (np.zeros((2,2,3)),np.zeros((0,2,3),np.uint8),
                      np.zeros((2,2,4),np.uint8),np.zeros((2,2),np.uint8)):
            with self.assertRaises(ValueError):
                engine.share(frame)
        for kwargs in ({'tolerance':-1},{'edge':float('nan')},{'tint_gamma':0},
                       {'lightness_limit':-1},{'lightness_limit':float('nan')}):
            with self.assertRaises(ValueError):
                ColorSharingEngine(**kwargs)
        with self.assertRaises(ValueError):
            ColorSharingEngine(chroma_model='bad')
        with self.assertRaises(ValueError):
            engine.encode_colors(np.zeros((1,1,3),np.uint8),np.array([['xx']]))
        with self.assertRaises(ValueError):
            engine.encode_colors(np.zeros((1,1,3),np.uint8),np.array([['\ud800']],dtype='U1'))
        invalid = np.array([['M']],dtype='U1')
        invalid.view(np.uint32)[0,0] = 0x110000
        with self.assertRaises(ValueError):
            engine.encode_colors(np.zeros((1,1,3),np.uint8),invalid)

    def test_missing_build_has_explicit_error_not_jit_fallback(self):
        with patch('ascii_color_sharing.library_path') as path:
            path.return_value.is_file.return_value = False
            with self.assertRaisesRegex(RuntimeError,'build_ascii_color_sharing'):
                ColorSharingEngine()

    def test_threads_have_no_shared_frame_state(self):
        engine = ColorSharingEngine()
        frames = [np.full((4,12,3),v,np.uint8) for v in range(0,256,31)]
        expected = [engine.share(frame) for frame in frames]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            actual = list(pool.map(engine.share,frames))
        for a,b in zip(actual,expected):
            np.testing.assert_array_equal(a,b)

    def test_fresh_backend_does_not_import_jit(self):
        code = ('import sys,json,numpy as np; from ascii_color_sharing import ColorSharingEngine; '
                'e=ColorSharingEngine(); e.encode(np.zeros((2,2,3),np.uint8),np.full((2,2),"M",dtype="U1")); '
                'print(json.dumps([m for m in sys.modules if m.startswith(("numba","llvmlite"))]))')
        result = subprocess.check_output([sys.executable,'-c',code],text=True)
        self.assertEqual(json.loads(result),[])
