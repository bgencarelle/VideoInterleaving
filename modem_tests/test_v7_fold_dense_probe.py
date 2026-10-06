"""Known-frequency measurement calibration accompanies a real V7 packet."""
from contextlib import redirect_stdout
import io
import unittest
import numpy as np
from animation_modem.v7_core import adapt_packet_for_output
from tools.v7_analog_frame import ProductionFrameWire,render
from tools.v7_fold_dense_probe import target,measure


class DenseResolutionTests(unittest.TestCase):
    def test_phase_and_off_axis_artifacts_are_not_resolution_gains(self):
        size=(640,480)
        for axis in ('x','y'):
            source=target(size,axis,8,np.pi/2,.05)
            reference=source/255
            self.assertTrue(measure(reference,reference,axis,8)['passed'])
            self.assertFalse(measure(reference,1-reference,axis,8)['passed'])
            unrelated=target(size,'y' if axis=='x' else 'x',8,np.pi/2,.05)/255
            self.assertFalse(measure(reference,unrelated,axis,8)['passed'])
            wire=ProductionFrameWire('aspect-fold-500','4:3')
            packet=adapt_packet_for_output(wire.encode_packet(source,1,0),96000)
            with redirect_stdout(io.StringIO()):
                packets,_,_=wire.decode(packet)
            decoded=[v for r,v in packets if r.diag.get('source_index')==0]
            self.assertEqual(len(decoded),1)
            self.assertIsNotNone(decoded[0])
            received=render(wire,decoded[0],size,True)
            result=measure(reference,received,axis,8)
            self.assertTrue(result['passed'])
            self.assertTrue(np.isfinite(result['contrast']))


if __name__=='__main__':unittest.main()
