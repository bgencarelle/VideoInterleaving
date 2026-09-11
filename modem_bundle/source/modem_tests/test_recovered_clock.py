"""Reference-clock sampling works across speed and preserves pilot direction."""
import unittest
from fractions import Fraction
import numpy as np
from scipy.signal import resample_poly
from animation_modem import transport2 as v


class RecoveredClockTests(unittest.TestCase):
    def setUp(self):
        self.layout = v.PRESETS['wide']
        self.coder = v.SourceCoder([(12,12)]*3)
        self.values = np.random.default_rng(21).uniform(-.2,.2,self.coder.count)

    def test_live_path_at_quarter_to_double_speed(self):
        for speed in (.25,.4,.8,1.,1.8,2.):
            with self.subTest(speed=speed):
                ratio=Fraction(1/speed).limit_denominator(1000)
                packet=v.encode(self.values,self.layout,self.coder,1,1,1)
                audio=resample_poly(packet,ratio.numerator,ratio.denominator,axis=0)
                rx=v.Receiver(self.layout,self.coder,recovery=False,fast=True)
                got=rx.feed(audio)+rx.flush()
                self.assertEqual([r.absolute for r in got],[1])
                self.assertTrue(np.isfinite(got[0].values).all())
                self.assertAlmostEqual(got[0].extra['playback_speed'],speed,delta=.005)

    def test_pilot_clock_error_has_correct_sign(self):
        packet=v.encode(self.values,self.layout,self.coder,1,1,1)
        for error in (-.0002,.0002):
            body=v._sample_at(packet,v._body_walk(self.layout)*(1+error))
            result=v.decode_packet(None,self.layout,self.coder,
                                   body=body.reshape(self.layout.symbols,v.N,2))
            self.assertEqual(result.identity,'verified_header')
            self.assertAlmostEqual(result.extra['clock_error'],error,delta=.00003)

    def test_direct_windows_match_full_packet_sampling(self):
        packet=v.encode(self.values,self.layout,self.coder,1,1,1)
        scale=1.0007;offset=.37
        full=v.resample_packet(packet,scale-1,self.layout.packet,offset=offset,fast=True)
        expected=v.decode_packet(full,self.layout,self.coder)
        body=v._sample_at(packet,offset+v._body_walk(self.layout)*scale)
        actual=v.decode_packet(None,self.layout,self.coder,
                               body=body.reshape(self.layout.symbols,v.N,2))
        np.testing.assert_allclose(actual.values,expected.values,atol=1e-6)
        self.assertEqual(actual.identity,expected.identity)
