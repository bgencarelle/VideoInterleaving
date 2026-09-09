import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import argparse
import numpy as np
from animation_modem.impairments import Settings, Emulator, add_arguments, settings_from_args


class ImpairmentTests(unittest.TestCase):
    def test_clean_exact_and_no_mutation(self):
        x=np.random.default_rng(4).normal(0,.1,(400,2)).astype(np.float32)
        np.testing.assert_array_equal(Emulator(Settings()).process(x),x)

    def test_state_continuity_and_seed_across_chunk_sizes(self):
        x=np.random.default_rng(4).normal(0,.1,(15000,2)).astype(np.float32)
        original=x.copy()
        s=Settings(lowpass_hz=12000,highpass_hz=300,noise_dbfs=-40,
                   crosstalk=.1,clip=.12,bits=10,dropout_ms=10,dropout_every=.1)
        whole=Emulator(s).process(x)
        em=Emulator(s)
        chunks=np.concatenate([em.process(x[i:i+137]) for i in range(0,len(x),137)])
        self.assertTrue(np.isfinite(whole).all())
        self.assertTrue(np.isfinite(chunks).all())
        np.testing.assert_array_equal(whole,chunks)
        np.testing.assert_array_equal(x,original)
        s.seed+=1
        self.assertFalse(np.array_equal(whole,Emulator(s).process(x)))

    def test_dropout_exact_boundaries(self):
        em=Emulator(Settings(dropout_ms=10,dropout_every=.1))
        out=em.process(np.ones((11000,2),np.float32))
        expected=np.ones_like(out)
        expected[4800:5280]=0
        expected[9600:10080]=0
        np.testing.assert_array_equal(out,expected)

    def test_noise_rms(self):
        out=Emulator(Settings(noise_dbfs=-40)).process(np.zeros((100000,2)))
        self.assertAlmostEqual(float(np.sqrt(np.mean(out**2))),.01,delta=.0001)

    def test_mono_and_clipping(self):
        x=np.tile([.8,-.2],(100,1))
        out=Emulator(Settings(crosstalk=.5,clip=.1)).process(x)
        np.testing.assert_allclose(out,.1)

    def test_overrides_and_invalid_configuration(self):
        p=argparse.ArgumentParser();add_arguments(p)
        s=settings_from_args(p.parse_args(['--emulate','rough','--lowpass-hz','0','--noise-dbfs','-60']))
        self.assertEqual(s.lowpass_hz,0)
        self.assertEqual(s.noise_dbfs,-60)
        self.assertEqual(s.clip,.12)
        for kw in [dict(lowpass_hz=24000),dict(crosstalk=.6),dict(bits=1),
                   dict(dropout_ms=2000),dict(seed=-1),dict(clip=float('nan'))]:
            with self.assertRaises(ValueError): Settings(**kw).validate()


if __name__=='__main__': unittest.main()
