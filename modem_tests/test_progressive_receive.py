import unittest
from unittest.mock import patch
import numpy as np
from animation_modem import transport2 as v
from animation_modem.progressive import Receiver
from animation_modem.imaging import plane_shapes


class ProgressiveTests(unittest.TestCase):
    def setUp(self):
        self.layout=v.PRESETS['wide']
        self.coder=v.SourceCoder(plane_shapes())
        self.values=np.random.default_rng(14).uniform(-.2,.2,self.coder.count)
        self.audio=v.encode(self.values,self.layout,self.coder,1,1,2)
        self.prefix=v.SYNC_LEN+(3+self.layout.header_symbols)*v.SYMBOL

    def collect(self,audio,rx=None):
        rx=rx or Receiver(self.layout,self.coder)
        out=[]
        for at in range(0,len(audio),256):out.extend(rx.feed(audio[at:at+256]))
        return out+rx.flush()

    def test_partial_before_packet_end_and_no_payload_retry(self):
        rx=Receiver(self.layout,self.coder)
        with patch.object(v,'decode_packet',side_effect=AssertionError('payload retry')):
            early=rx.feed(self.audio[:self.prefix])
            self.assertTrue(early)
            self.assertFalse(early[-1].extra['complete'])
            self.assertEqual(early[-1].absolute,1)
            self.assertGreater(early[-1].extra['received_values'],0)
            rest=rx.feed(self.audio[self.prefix:])+rx.flush()
        final=rest[-1]
        self.assertTrue(final.extra['complete'])
        self.assertEqual(final.extra['fft_symbols'],self.layout.symbols)
        self.assertLess(np.sqrt(np.mean((final.values-self.values)**2)),.001)

    def test_erased_header_still_shows_current_image(self):
        a=self.audio.copy();start=v.SYNC_LEN+2*v.SYMBOL
        a[start:start+self.layout.header_symbols*v.SYMBOL]=0
        got=self.collect(a)
        self.assertTrue(got)
        self.assertTrue(all(r.values is not None and r.absolute is None for r in got))
        self.assertTrue(got[-1].extra['complete'])

    def test_truncated_frame_keeps_preview_and_silence_emits_no_blank(self):
        rx=Receiver(self.layout,self.coder)
        got=rx.feed(self.audio[:self.prefix]);self.assertTrue(got)
        before=got[-1].values.copy()
        rx.reset(preserve_timing=True)
        self.assertEqual(rx.feed(np.zeros((4000,2),np.float32)),[])
        np.testing.assert_array_equal(got[-1].values,before)

    def test_next_frame_never_reuses_old_coefficients(self):
        next_audio=v.encode(-self.values,self.layout,self.coder,2,2,2)
        rx=Receiver(self.layout,self.coder)
        self.collect(self.audio,rx)
        partial=rx.feed(next_audio[:self.prefix])[-1]
        alone=Receiver(self.layout,self.coder).feed(next_audio[:self.prefix])[-1]
        np.testing.assert_allclose(partial.values,alone.values,atol=1e-6)

    def test_noise_and_false_preamble_do_not_generate_image(self):
        rng=np.random.default_rng(17)
        for a in (rng.normal(0,.02,(8000,2)),
                  np.concatenate((self.audio[:v.SYNC_LEN],rng.normal(0,.02,(6000,2))))):
            self.assertEqual(self.collect(a),[])

    def test_half_speed_still_emits_before_completion(self):
        from scipy.signal import resample_poly
        audio=resample_poly(self.audio,2,1,axis=0)
        rx=Receiver(self.layout,self.coder)
        early=rx.feed(audio[:self.prefix*2+16])
        self.assertTrue(early)
        self.assertFalse(early[-1].extra['complete'])
        rest=rx.feed(audio[self.prefix*2+16:])+rx.flush()
        self.assertTrue(rest[-1].extra['complete'])
        self.assertEqual(rest[-1].absolute,1)
