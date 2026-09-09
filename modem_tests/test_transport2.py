"""V2 receive contract: no transmit clock, cadence, future packet or EOF padding."""
import contextlib
import io
import json
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from scipy.signal import resample_poly, butter, sosfilt

from animation_modem import transport2 as v2
from animation_modem import transport as v1
from utilities.modem_v2_check import coder_for, image_values, main


class V2Tests(unittest.TestCase):
    def setUp(self):
        self.layout=v2.PRESETS['tape']
        self.coder=v2.SourceCoder(v1.plane_shapes('color'))
        yy,xx=np.mgrid[:48,:40]
        self.image=Image.fromarray(np.uint8(np.stack([xx*6,yy*5,(xx+yy)*2],axis=-1)))
        self.values=v1.image_values(self.image)

    def packet(self,n=1):
        return v2.encode(self.values,self.layout,self.coder,n,n,20)

    def decode(self,audio,chunk=256,rx=None):
        rx=rx or v2.Receiver(self.layout,self.coder)
        got=[]
        for i in range(0,len(audio),chunk):got+=rx.feed(audio[i:i+chunk])
        return got+rx.flush()

    def test_exact_last_packet_without_guard_or_padding(self):
        audio=self.packet()[:self.layout.packet]
        got=self.decode(audio)
        self.assertEqual([r.absolute for r in got],[1])
        self.assertLess(np.sqrt(np.mean((got[0].values-self.values)**2)),1e-3)

    def test_no_alternates_lost_for_any_callback_size(self):
        audio=np.concatenate([self.packet(n) for n in range(1,7)])
        for chunk in (1,173,256,1024,len(audio)):
            with self.subTest(chunk=chunk):
                got=self.decode(audio,chunk)
                self.assertEqual([r.absolute for r in got],list(range(1,7)))

    def test_each_speed_cold_acquires_one_packet(self):
        for speed in (.5,.67,.8,.95,1.05,1.2,1.5,2.):
            with self.subTest(speed=speed):
                f=Fraction(1/speed).limit_denominator(1000)
                audio=resample_poly(self.packet(),f.numerator,f.denominator,axis=0)
                got=self.decode(audio)
                self.assertEqual([r.absolute for r in got],[1])
                self.assertAlmostEqual(got[0].extra['playback_speed'],speed,delta=.004)

    def test_rate_hints_do_not_require_previous_packets_or_fixed_spacing(self):
        pieces=[]
        for n,speed in enumerate((.8,1.2,.5,1.),1):
            f=Fraction(1/speed).limit_denominator(1000)
            pieces.extend([np.zeros((n*377,2)),resample_poly(self.packet(n),f.numerator,f.denominator,axis=0)])
        self.assertEqual([r.absolute for r in self.decode(np.concatenate(pieces))],[1,2,3,4])

    def test_quarter_speed_optional_range(self):
        rx=v2.Receiver(self.layout,self.coder,min_speed=.25)
        got=self.decode(resample_poly(self.packet(),4,1,axis=0),rx=rx)
        self.assertEqual([r.absolute for r in got],[1])

    def test_half_speed_bandlimit_noise_and_reversed_polarity(self):
        audio=resample_poly(self.packet(),2,1,axis=0)
        audio=-.7*sosfilt(butter(2,8000,fs=v2.RATE,output='sos'),audio,axis=0)
        audio+=np.random.default_rng(88).normal(0,.003,audio.shape)
        got=self.decode(audio)
        self.assertEqual([r.absolute for r in got],[1])
        self.assertTrue(np.isfinite(got[0].values).all())

    def test_wow_flutter_point_three_percent(self):
        audio=self.packet();up=resample_poly(audio,8,1,axis=0)
        t=np.arange(len(audio))/v2.RATE
        for hz in (4,20,60):
            pos=np.arange(len(audio))+.003*v2.RATE/(2*np.pi*hz)*np.sin(2*np.pi*hz*t)
            warped=np.stack([np.interp(pos*8,np.arange(len(up)),up[:,c]) for c in range(2)],axis=1)
            self.assertEqual([r.absolute for r in self.decode(warped)],[1])

    def test_header_erasure_salvages_only_current_picture(self):
        audio=self.packet()
        audio[v2.SYNC_LEN+2*v2.SYMBOL:v2.SYNC_LEN+(2+self.layout.header_symbols)*v2.SYMBOL]=0
        got=self.decode(audio)
        self.assertEqual(len(got),1)
        self.assertIsNone(got[0].absolute)
        self.assertEqual(got[0].identity,'unknown')
        self.assertIsNotNone(got[0].values)

    def test_noise_and_truncated_packets_do_not_invent_images(self):
        rng=np.random.default_rng(4)
        for audio in (np.zeros((18000,2)),rng.normal(0,.02,(18000,2)),
                      np.concatenate([self.packet()[:v2.SYNC_LEN],rng.normal(0,.02,(self.layout.packet,2))]),
                      self.packet()[:-200]):
            self.assertEqual(self.decode(audio),[])

    def test_allocation_is_finite_and_peak_safe(self):
        sigma=np.geomspace(100,.001,self.coder.count)
        coder=v2.SourceCoder(self.coder.shapes,sigma)
        audio=v2.encode(self.values,self.layout,coder,1,1,1)
        self.assertLessEqual(np.max(abs(audio)),.95001)
        quantized=np.rint(audio*32767).astype(np.int16).astype(float)/32768
        result=v2.decode_packet(quantized,self.layout,coder)
        self.assertEqual(result.absolute,1)
        self.assertTrue(np.isfinite(result.values).all())

    def test_lofi_survives_three_khz_model_with_hiss(self):
        layout=v2.PRESETS['lofi'];coder,_=coder_for('color',None,layout)
        audio=v2.encode(image_values(self.image,coder),layout,coder,1,1,1)
        audio=sosfilt(butter(2,3000,fs=v2.RATE,output='sos'),audio,axis=0)
        audio+=np.random.default_rng(8).normal(0,.003,audio.shape)
        rx=v2.Receiver(layout,coder)
        got=rx.feed(audio)+rx.flush()
        self.assertEqual([r.absolute for r in got],[1])
        self.assertTrue(np.isfinite(got[0].values).all())

    def test_wiener_inverse_suppresses_unreliable_coefficients(self):
        sent=self.coder.forward(self.values)
        exact=self.coder.inverse(sent,np.ones(self.coder.count),np.zeros(self.coder.count))
        np.testing.assert_allclose(exact,self.values,atol=1e-12)
        erased=self.coder.inverse(sent,np.zeros(self.coder.count),np.ones(self.coder.count))
        np.testing.assert_array_equal(erased,np.zeros(self.coder.count))

    def test_all_presets_have_fitting_coders_and_roundtrip(self):
        for layout in v2.PRESETS.values():
            with self.subTest(preset=layout.name):
                coder,_=coder_for('color',None,layout)
                self.assertLessEqual(coder.count,layout.capacity)
                self.assertEqual(len(np.unique(layout.pilots)),4)
                audio=v2.encode(image_values(self.image,coder),layout,coder,1,1,1)
                result=v2.decode_packet(audio,layout,coder)
                self.assertEqual(result.absolute,1)
                self.assertEqual(len(result.values),coder.count)

    def test_cli_tape_fast_write_read_streaming(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'test.wav'
            output=io.StringIO()
            with contextlib.redirect_stdout(output),contextlib.redirect_stderr(io.StringIO()):
                main(['write','--preset','tape-fast','--frames','6','--out',str(path)])
                output.seek(0);output.truncate(0)
                main(['read','--preset','tape-fast','--wav',str(path)])
            rows=[json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual([r['frame'] for r in rows],list(range(1,7)))

    def test_nonfinite_input_and_reset(self):
        rx=v2.Receiver(self.layout,self.coder)
        with self.assertRaises(ValueError):rx.feed(np.full((300,2),np.nan))
        rx.feed(self.packet()[:1000]);rx.reset()
        self.assertEqual([r.absolute for r in self.decode(self.packet(2),rx=rx)],[2])


if __name__=='__main__':unittest.main()
