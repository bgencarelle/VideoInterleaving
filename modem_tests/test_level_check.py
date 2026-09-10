"""Synthetic level-check tests; no sound device is opened."""
import unittest
from types import SimpleNamespace
import numpy as np
from utilities.modem_level_check import make_probe, analyze, run_capture


class LevelCheckTests(unittest.TestCase):
    def test_gain_delay_routing_noise_and_crosstalk(self):
        rate=48000; delay=1379
        sent,pilot,slots=make_probe(rate)
        received=np.random.default_rng(1).normal(0,1e-6,(len(sent),4))
        received[delay:,3]+=sent[:-delay,0]*.5
        received[delay:,1]+=sent[:-delay,1]*.25
        received[delay:,2]+=sent[:-delay,0]*.05
        rows=analyze(received,rate,pilot,slots,-18.,1.)
        for output,channel,gain in ((1,4,.5),(2,2,.25),(1,3,.05)):
            row=next(r for r in rows if r['output']==output and r['input']==channel)
            self.assertTrue(row['detected'])
            self.assertAlmostEqual(row['gain_db'],20*np.log10(gain),delta=.02)
            self.assertAlmostEqual(row['delay_ms'],delay/rate*1000,delta=.03)
        self.assertFalse(any(r['detected'] for r in rows if r['input']==1))

    def test_silence_and_clipping(self):
        sent,pilot,slots=make_probe(48000)
        silent=analyze(np.zeros_like(sent),48000,pilot,slots,-18.,1.)
        self.assertFalse(any(r['detected'] for r in silent))
        clipped=analyze(np.clip(sent*20,-1,1),48000,pilot,slots,-18.,1.)
        self.assertTrue(any(r['detected'] and r['near_full_scale_percent']>0 for r in clipped))

    def test_callback_copies_arbitrary_blocks_and_finishes(self):
        sent,_,_=make_probe(8000)
        recorded=np.random.default_rng(1).normal(0,.1,sent.shape).astype(np.float32)
        played=[]
        class SD:
            class CallbackStop(Exception):pass
            class CallbackAbort(Exception):pass
            def Stream(self,**kw):
                class Stream:
                    def __enter__(self):
                        at=0
                        while at<len(sent):
                            n=317
                            incoming=np.zeros((n,2),np.float32)
                            count=min(n,len(sent)-at)
                            incoming[:count]=recorded[at:at+count]
                            output=np.empty((n,2),np.float32)
                            try:
                                kw['callback'](incoming,output,n,None,
                                    SimpleNamespace(input_overflow=False,output_underflow=False))
                            except SD.CallbackStop:
                                played.append(output.copy());break
                            played.append(output.copy());at+=n
                        kw['finished_callback']()
                        return self
                    def __exit__(self,*args):pass
                return Stream()
        actual,flags=run_capture(SD(),sent,2,8000,0,1)
        np.testing.assert_array_equal(actual,recorded)
        output=np.concatenate(played)
        np.testing.assert_array_equal(output[:len(sent)],sent)
        self.assertFalse(np.any(output[len(sent):]))
        self.assertFalse(any(flags.values()))


if __name__=='__main__':unittest.main()
