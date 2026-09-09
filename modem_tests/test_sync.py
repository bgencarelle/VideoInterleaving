import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
from scipy.signal import correlate
from PIL import Image
from animation_modem.transport import SYNC, SYNC_ENERGY, sync_correlation, encode, Receiver


class SyncTests(unittest.TestCase):
    def test_near_silence_after_loud_audio(self):
        x=np.zeros(4000)
        x[:1000]=np.random.default_rng(40).normal(0,.1,1000)
        x[1500:1756]=SYNC*1e-8
        x[2500:2756]=SYNC*.5
        # Reproduce the actual numerical failure in the previous implementation.
        cs=np.r_[0,np.cumsum(x*x)]
        energy=cs[256:]-cs[:-256]
        old=np.abs(correlate(x,SYNC,mode='valid',method='fft'))/np.sqrt(np.maximum(energy*SYNC_ENERGY,1e-20))
        self.assertGreater(old[1500],1)
        score=sync_correlation(x)
        self.assertTrue(np.isfinite(score).all())
        self.assertTrue(np.all((score>=0)&(score<=1)))
        self.assertEqual(score[1500],0)
        self.assertAlmostEqual(score[2500],1)

    def test_matches_direct_reference_and_gain_invariance(self):
        x=np.random.default_rng(5).normal(size=800)
        expected=np.array([abs(np.sum(x[i:i+256]*SYNC))/np.sqrt(np.sum(x[i:i+256]**2)*SYNC_ENERGY)
                           for i in range(len(x)-255)])
        for scale in [1,1e-12,-1e6]:
            np.testing.assert_allclose(sync_correlation(x*scale),expected,rtol=1e-12,atol=1e-14)
        self.assertTrue(np.all(sync_correlation(np.zeros(800))==0))

    def test_live_sized_chunks_with_right_channel_preamble_leakage(self):
        im=Image.new('RGB',(40,48),(130,80,60))
        frames=[]
        for i in range(1,21):
            a,_,_=encode(im,i,1,1)
            a[16:272,1]=SYNC*1e-8
            frames.append(a)
        audio=np.concatenate(frames)[137:]
        rx=Receiver();rows=[]
        for i in range(0,len(audio),256):rows.extend(rx.feed(audio[i:i+256]))
        good=[r for r in rows if r.identity=='verified_header']
        self.assertEqual([r.frame for r in good],list(range(2,21)))
        self.assertTrue(all(0<=r.sync_score<=1 for r in good))


if __name__=='__main__': unittest.main()
