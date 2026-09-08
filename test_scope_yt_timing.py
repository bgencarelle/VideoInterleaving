"""Registration tests for the X-channel Y-T renderer; no audio hardware needed."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from scope_bake import TraceEmitter, render_luma, render_yt_grid, yt_timeline
from scope_out import (Scope, clip_for_trigger, marker_window,
                       yt_trigger_frame)
from test_scope_web import _extract_fn


class FixedYtTests(unittest.TestCase):
    def setUp(self):
        self.n = 3200
        self.kw = dict(yt_fixed=True, yt_trigger_samples=48, border=.09,
                       grid_rows=32, grid_cols=32, stretch=False, autofit=False)
        self.a = np.tile(np.linspace(.15, .65, 32), (32, 1))

    def render(self, lum, **kwargs):
        return render_luma(lum, self.n, **dict(self.kw, **kwargs))

    def test_unrelated_brightness_cannot_move_or_resize_lower_rows(self):
        original = self.render(self.a)
        edges, border_start, _ = yt_timeline(self.n, 32, trigger_samples=48, border=.09)
        for value in (0., .05, .4, 1.):
            other = self.a.copy()
            other[:16] = value
            frame = self.render(other)
            np.testing.assert_array_equal(frame[edges[16]:border_start],
                                          original[edges[16]:border_start])
            np.testing.assert_array_equal(frame[border_start:], original[border_start:])
            self.assertEqual(frame.shape, (self.n, 2))
        # Demonstrate that this test exercises the previous failure.
        changed = self.a.copy(); changed[:16] = 1.
        old_a = self.render(self.a, yt_fixed=False)
        old_b = self.render(changed, yt_fixed=False)
        target_y = -np.linspace(-.9, .9, 32)[24]
        ia = np.flatnonzero(np.isclose(old_a[:, 1], target_y))
        ib = np.flatnonzero(np.isclose(old_b[:, 1], target_y))
        self.assertNotEqual(int(ia[0]), int(ib[0]))

    def test_margins_and_blank_frames_keep_their_slots(self):
        lum = np.zeros_like(self.a); lum[12:20] = self.a[12:20]
        frame = self.render(lum)
        full = self.render(self.a)
        edges, border_start, _ = yt_timeline(self.n, 32, trigger_samples=48, border=.09)
        np.testing.assert_array_equal(frame[edges[12]:edges[20]], full[edges[12]:edges[20]])
        blank = self.render(np.zeros_like(lum))
        self.assertTrue(np.all(blank[48:border_start, 0] < -.9))
        np.testing.assert_array_equal(blank[border_start:], full[border_start:])

    def test_dwell_detail_survives_inside_a_row(self):
        changed = self.a.copy(); changed[8] = changed[8, ::-1]
        a, b = self.render(self.a), self.render(changed)
        edges, _, _ = yt_timeline(self.n, 32, trigger_samples=48, border=.09)
        self.assertFalse(np.array_equal(a[edges[8]:edges[9], 0], b[edges[8]:edges[9], 0]))
        np.testing.assert_array_equal(a[edges[9]:], b[edges[9]:])

    def test_interlace_rotation_and_odd_row_counts(self):
        for angle in range(4):
            lum = np.rot90(np.ones((31, 23)) * .5, angle)
            for field in range(2):
                out = render_luma(lum, 1600, yt_fixed=True, yt_trigger_samples=24,
                                  grid_rows=lum.shape[0], grid_cols=lum.shape[1],
                                  fields=2, field=field, border=.09)
                self.assertEqual(out.shape, (1600, 2))
                self.assertTrue(np.isfinite(out).all())
                self.assertLessEqual(float(out[:, 0].max()), .900001)

    def test_marker_does_not_overwrite_picture_and_covers_callback_boundaries(self):
        scope = Scope(device="null", samplerate=96000, samples=self.n,
                      yt_mode=True, yt_trigger_us=500.)
        try:
            emitter = TraceEmitter(96000, self.n, grid=(32,32), levels=(0.,1.),
                                   border=.09, sweep="retrace", yt_timing="fixed",
                                   yt_trigger_samples=scope.yt_trigger_samples)
            frame = emitter.emit(self.a)
            scope.show_frame(frame)
            m = scope.yt_trigger_samples
            # The marker has its OWN samples: the picture arrives whole, byte
            # for byte, and the trace is longer by exactly the window.
            self.assertEqual(len(scope._pending), self.n + m)
            np.testing.assert_array_equal(scope._pending[m:], frame)
            scope._frame, scope._pending = scope._pending, None
            period = self.n + m
            chunks=[]
            for count in (17, 29, period - 46, 17, period - 17):
                buf=np.empty((count,2),np.float32)
                scope._callback(buf,count,None,None);chunks.append(buf)
            out=np.vstack(chunks)
            # One rising crossing per trace, wherever in the marker it falls:
            # two whole traces were consumed, so exactly two.
            crossings=np.flatnonzero((out[:-1,0]<.95)&(out[1:,0]>=.95))+1
            self.assertEqual(len(crossings), 2)
            # ...and they are exactly one trace apart, which is the property a
            # scope timebase actually depends on.
            self.assertEqual(int(crossings[1] - crossings[0]), period)
        finally:
            scope.stream.close()

    def test_insufficient_sample_budget_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "two samples per row"):
            render_yt_grid(self.a, 64, trigger_samples=48)

    def test_composite_emit_filter_and_rotation_keep_trigger_unique(self):
        from scope_display import _emit

        class Library:
            def __len__(self):
                return 1

            def thumb(self, _):
                a = np.zeros((32,32,2), np.uint8)
                a[4:28,4:28,0] = 220
                a[...,1] = 255
                return a

        scope=Scope(device="null",samplerate=96000,samples=3200,
                    yt_mode=True,yt_trigger_us=500.)
        try:
            emitter=TraceEmitter(96000,3200,grid=(32,32),levels=(0.,1.),
                                 yt_timing="fixed",yt_trigger_samples=48,
                                 border=.09,sweep="retrace")
            for angle in (0,90,180,270):
                _emit(scope,Library(),None,0,"raster",{},"retrace",2.2,.02,1.,
                      None,.02,emitter=emitter,rotation=angle,lowpass=6000)
                out=scope._pending
                self.assertEqual(out.shape,(3200 + scope.yt_trigger_samples, 2))
                self.assertTrue(np.isfinite(out).all())
                crossings=np.flatnonzero((out[:-1,0]<.95)&(out[1:,0]>=.95))+1
                # Filtering and rotation must not manufacture a second edge,
                # and the one edge belongs to the reserved window at the head.
                self.assertEqual(len(crossings), 1)
                self.assertLess(int(crossings[0]), scope.yt_trigger_samples)
        finally:
            scope.stream.close()

    def test_oversampling_does_not_create_extra_trigger_crossings(self):
        for rate in (48000, 96000):
            for factor in (1, 2, 4):
                n, marker = rate//30, rate//2000
                out = render_yt_grid(self.a, n, trigger_samples=marker, oversample=factor)
                out = yt_trigger_frame(out, marker)
                crossings = np.count_nonzero((out[:-1,0]<.95)&(out[1:,0]>=.95))
                self.assertEqual(crossings, 1)

    @unittest.skipUnless(shutil.which("node"), "Node is needed for browser parity")
    def test_browser_fixed_timing_matches_python(self):
        page = Path(__file__).with_name("templates").joinpath("scope.html").read_text()
        js = _extract_fn(page, "buildYtTrace")
        g = self.a.copy(); g[:4] = 0.; g[7] = 0.; g[7, 12] = .9
        for n, marker in ((3200,48),(1600,24)):
            with self.subTest(n=n), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/"trace.f32"
                source = js + "\nconst fs=require('fs');\n" + (
                    f"const out=buildYtTrace({json.dumps(g.ravel().tolist())},32,32,"
                    f"{n},.02,2.2,32,32,{marker},.09);\n"
                    f"fs.writeFileSync({json.dumps(str(path))},Buffer.from(out.buffer));")
                subprocess.run(["node","-e",source],check=True,capture_output=True)
                actual = np.fromfile(path,np.float32).reshape(-1,2)
                expected = render_yt_grid(g,n,trigger_samples=marker,border=.09)
                np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=0.)

    @unittest.skipUnless(shutil.which("node"), "Node is needed for browser integration")
    def test_browser_render_entry_uses_timing_rate_and_marker_settings(self):
        page=Path(__file__).with_name("templates").joinpath("scope.html").read_text()
        functions="\n".join(_extract_fn(page,name) for name in
                            ("boxGrid","buildTrace","buildYtTrace","addYtTrigger",
                             "markerWindow","prependMarker","renderFromLuma"))
        raw=(self.a*255).astype(np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            fixed=Path(tmp)/"fixed.f32"
            script=functions+f"""
const fs=require('fs');
const raw={json.dumps(raw.ravel().tolist())};
const px=new Uint8ClampedArray(32*32*4);
for(let i=0;i<raw.length;i++){{px[4*i]=raw[i];px[4*i+3]=255;}}
const localRunning=true,actx={{sampleRate:96000}},messages=[];
const worklet={{port:{{postMessage:a=>messages.push(a)}}}};
const lumImg={{naturalWidth:32,naturalHeight:32}},lumCanvas={{width:32,height:32}};
const lumCtx={{drawImage:()=>{{}},getImageData:()=>({{data:px}})}};
let lastHash=0,lastChange=Date.now(),backoff=0,tracesBuilt=0;
const STALL_MS=6000,frameHash=()=>1;
const document={{getElementById:id=>({{value:id==='local-gamma'?'2.2':'0.02'}})}};
const lastData={{scope_yt:true,scope_yt_timing:'fixed',scope_refresh_hz:60,
  scope_yt_grid:[32,32],scope_yt_levels:[0,1],scope_border:.09,scope_yt_trigger_us:500}};
renderFromLuma();
// picture + marker: the window is reserved, not carved out of the picture
if(messages.length!==1||messages[0].length!==(1600+48)*2)throw Error('fixed entry did not emit expected budget: '+messages[0].length);
fs.writeFileSync({json.dumps(str(fixed))},Buffer.from(messages[0].buffer));
lastData.scope_yt_timing='dwell';renderFromLuma();
if(messages.length!==2||messages[1].length!==(3200+48)*2)throw Error('dwell entry changed: '+messages[1].length);
"""
            result = subprocess.run(["node","-e",script],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            actual=np.fromfile(fixed,np.float32).reshape(-1,2)
            expected=np.vstack((marker_window(48),
                                clip_for_trigger(render_yt_grid(
                                    raw/255.,1600,trigger_samples=0,border=.09))))
            np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=0.)


if __name__ == "__main__":
    unittest.main()
