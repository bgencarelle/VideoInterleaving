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

    def test_cli_fixed_write_read_streaming(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'test.wav'
            output=io.StringIO()
            with contextlib.redirect_stdout(output),contextlib.redirect_stderr(io.StringIO()):
                main(['write','--frames','6','--out',str(path)])
                output.seek(0);output.truncate(0)
                main(['read','--wav',str(path)])
            rows=[json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual([r['frame'] for r in rows],list(range(1,7)))

    def test_normal_cli_rejects_alternate_formats(self):
        for command, option in (
            ('write', ['--preset', 'tape-fast']),
            ('live-send', ['--allocation', 'missing.npy']),
            ('live-receive', ['--profile', 'mono']),
            ('write', ['--gain', '.5']),
        ):
            with self.subTest(command=command, option=option):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as exc:
                        main([command, *option])
                self.assertEqual(exc.exception.code, 2)

    def test_progressive_low_band_keeps_all_plane_dc_values(self):
        layout=v2.PRESETS['wide']
        slots=v2.coefficient_slots(layout,tuple(self.coder.shapes))
        bins=np.broadcast_to(layout.data_bins[None,:,None,None],
                             (layout.image_symbols,len(layout.data_bins),2,2)).ravel()[slots]
        self.assertEqual(layout.band[0],375)
        self.assertTrue(np.any(bins==2))
        dc=np.cumsum([0]+[h*w for h,w in self.coder.shapes[:-1]])
        np.testing.assert_array_equal(bins[dc],np.ones(len(dc)))
        sent=self.coder.forward(self.values)
        recovered=self.coder.inverse(sent*(bins<=8))
        for a,b in zip(self.coder._split(recovered),self.coder._split(self.values)):
            self.assertAlmostEqual(float(a.mean()),float(b.mean()),places=10)

    def test_progressive_stream_survives_six_khz_rolloff(self):
        from argparse import Namespace
        from utilities.modem_v2_check import frames_from
        im=frames_from(Namespace(modem_dir=None,frames=1,stride=1))[0][0]
        layout=v2.PRESETS['wide']
        values=image_values(im,self.coder)
        audio=v2.encode(values,layout,self.coder,1,1,1)
        audio=sosfilt(butter(6,6000,fs=v2.RATE,output='sos'),audio,axis=0)
        audio+=np.random.default_rng(10).normal(0,.0003,audio.shape)
        results=self.decode(audio,rx=v2.Receiver(layout,self.coder))
        self.assertEqual(len(results),1)
        self.assertLess(float(np.sqrt(np.mean((results[0].values-values)**2))),.15)

    def test_auto_receiver_preserves_neutral_color_at_slow_speeds(self):
        from animation_modem.audio_common import pcm
        from utilities.modem_v2_check import values_image
        layout=v2.PRESETS['wide']
        values=image_values(Image.new('RGB',(40,48),(128,128,128)),self.coder)
        audio=v2.encode(values,layout,self.coder,1,1,1)
        audio=np.frombuffer(pcm(audio),'<i2').reshape(-1,2)/32768
        for up,down in ((1,1),(5,4),(2,1),(4,1)):
            with self.subTest(speed=down/up):
                signal=resample_poly(audio,up,down) if up!=down else audio.copy()
                original=signal.copy()
                got=self.decode(signal,rx=v2.Receiver(layout,self.coder))
                np.testing.assert_array_equal(signal,original)
                self.assertEqual(len(got),1)
                self.assertEqual(got[0].identity,'verified_header')
                self.assertEqual(got[0].extra['input_path'],'raw')
                self.assertAlmostEqual(got[0].extra['playback_speed'],down/up,places=4)
                np.testing.assert_allclose(np.asarray(values_image(got[0].values,self.coder)),128,atol=1)

    def test_auto_receiver_selects_conditioning_for_hum(self):
        from utilities.modem_v2_check import values_image
        layout=v2.PRESETS['wide']
        values=image_values(Image.new('RGB',(40,48),(128,128,128)),self.coder)
        audio=v2.encode(values,layout,self.coder,1,1,1)
        audio=audio+.3*np.sin(2*np.pi*50*np.arange(len(audio))/v2.RATE)[:,None]
        got=self.decode(audio,rx=v2.Receiver(layout,self.coder))
        self.assertEqual(len(got),1)
        self.assertEqual(got[0].identity,'verified_header')
        self.assertEqual(got[0].extra['input_path'],'conditioned')
        rgb=np.asarray(values_image(got[0].values,self.coder)).mean((0,1))
        np.testing.assert_allclose(rgb,128,atol=2)

    def test_auto_receiver_handles_bass_eq_without_transmitter_changes(self):
        from argparse import Namespace
        from scipy.signal import bilinear,tf2sos
        from utilities.modem_v2_check import frames_from
        layout=v2.PRESETS['wide']
        values=image_values(frames_from(Namespace(modem_dir=None,frames=1,stride=1))[0][0],self.coder)
        clean=np.concatenate([v2.encode(values,layout,self.coder,i,i,2) for i in (1,2)])
        for hz,db in ((375,6),(375,-6),(1000,6),(1000,-6)):
            with self.subTest(hz=hz,db=db):
                # A unity-at-DC/Nyquist bell EQ, Q=1, with its center prewarped.
                w=2*v2.RATE*np.tan(np.pi*hz/v2.RATE)
                b,a=bilinear([1,10**(db/20)*w,w*w],[1,w,w*w],fs=v2.RATE)
                signal=.25*sosfilt(tf2sos(b,a),clean,axis=0)
                got=self.decode(signal,rx=v2.Receiver(layout,self.coder))
                self.assertEqual([q.absolute for q in got],[1,2])
                for q in got:
                    self.assertEqual(q.extra['input_path'],'eq_corrected')
                    self.assertLess(float(np.sqrt(np.mean((q.values-values)**2))),.12)

    def test_nonfinite_input_and_reset(self):
        rx=v2.Receiver(self.layout,self.coder)
        with self.assertRaises(ValueError):rx.feed(np.full((300,2),np.nan))
        rx.feed(self.packet()[:1000]);rx.reset()
        self.assertEqual([r.absolute for r in self.decode(self.packet(2),rx=rx)],[2])


    def test_source_index_is_zero_based_and_survives_the_round_trip(self):
        # The bake is zero-based; the wire is one-based so an erased header
        # decodes as invalid rather than as frame zero. Only the derived
        # source_index should ever reach anything downstream.
        for source in (0, 1, 11, 0xfffe):
            with self.subTest(source=source):
                audio=v2.encode(image_values(self.image,self.coder),self.layout,self.coder,
                                source+1,source+1,0xffff)
                result=v2.decode_packet(audio[:self.layout.packet],self.layout,self.coder)
                self.assertEqual(result.identity,'verified_header')
                self.assertEqual(result.index,source+1)
                self.assertEqual(result.source_index,source)

    def test_folder_pair_rides_the_flags_byte(self):
        from animation_modem.transport2 import FOLDER_LIMIT, pack_folders
        for face, front in ((0,0),(2,1),(15,15),(9,4)):
            with self.subTest(face=face,front=front):
                audio=v2.encode(image_values(self.image,self.coder),self.layout,self.coder,
                                1,1,1,flags=pack_folders(face,front))
                result=v2.decode_packet(audio[:self.layout.packet],self.layout,self.coder)
                self.assertEqual(result.identity,'verified_header')
                self.assertEqual((result.face_folder,result.float_folder),(face,front))
        # Four bits each, so a longer folder list wraps rather than failing.
        self.assertEqual(pack_folders(FOLDER_LIMIT,FOLDER_LIMIT),pack_folders(0,0))
        self.assertEqual(pack_folders(17,33),pack_folders(1,1))
        audio=v2.encode(image_values(self.image,self.coder),self.layout,self.coder,
                        1,1,1,flags=pack_folders(20,18))
        result=v2.decode_packet(audio[:self.layout.packet],self.layout,self.coder)
        self.assertEqual((result.face_folder,result.float_folder),(4,2))

    def test_erased_header_yields_no_index_at_all(self):
        audio=v2.encode(image_values(self.image,self.coder),self.layout,self.coder,1,1,1)
        start=v2.SYNC_LEN+2*v2.SYMBOL
        audio[start:start+self.layout.header_symbols*v2.SYMBOL]=0
        result=v2.decode_packet(audio[:self.layout.packet],self.layout,self.coder)
        self.assertNotEqual(result.identity,'verified_header')
        self.assertIsNone(result.index)
        self.assertIsNone(result.source_index)
        # Without a verified header the flags byte is not evidence of anything.
        self.assertIsNone(result.face_folder)
        self.assertIsNone(result.float_folder)


if __name__=='__main__':unittest.main()
