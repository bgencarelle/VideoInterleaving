"""Pilot correction follows time drift beyond the spacing alias of carriers."""
import unittest
import numpy as np
from animation_modem import transport2 as v
from animation_modem.imaging import plane_shapes


class PilotContinuityTests(unittest.TestCase):
    def test_smooth_symbol_drift_preserves_image(self):
        layout=v.PRESETS['wide']
        coder=v.SourceCoder(plane_shapes())
        values=np.random.default_rng(75).uniform(-.2,.2,coder.count)
        packet=v.encode(values,layout,coder,1,1,1)
        # Start within the CP and drift five samples across the frame.
        # Phase increments between adjacent symbols remain small, even when
        # phase differences between the distant pilots exceed pi.
        for step in (-.25, .18):
            with self.subTest(step=step):
                positions=v._body_walk(layout)+np.repeat(np.arange(layout.symbols)*step,v.N)
                body=v._sample_at(packet,positions).reshape(layout.symbols,v.N,2)
                result=v.decode_packet(None,layout,coder,body=body)
                self.assertEqual(result.identity,'verified_header')
                self.assertLess(np.sqrt(np.mean((result.values-values)**2)),.02)
