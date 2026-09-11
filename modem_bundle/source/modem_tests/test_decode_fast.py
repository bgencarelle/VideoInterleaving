"""Fast receiver preserves payloads through buffer reuse and speed changes."""
import unittest
from fractions import Fraction
import numpy as np
from scipy.signal import resample_poly
from animation_modem import transport2 as v2


class FastDecodeTests(unittest.TestCase):
    def test_long_stream_and_changing_speed(self):
        layout = v2.PRESETS['wide']
        coder = v2.SourceCoder([(12, 12)]*3)
        values = np.random.default_rng(17).uniform(-.25, .25, coder.count)
        packets = []
        for n in range(1, 25):
            speed = (1., .998, 1.002)[(n-1)//8]
            ratio = Fraction(1/speed).limit_denominator(2000)
            packet = v2.encode(values, layout, coder, n, n, 24)
            packets.append(resample_poly(packet, ratio.numerator,
                                         ratio.denominator, axis=0))
        audio = np.concatenate(packets)
        receiver = v2.Receiver(layout, coder, recovery=False, fast=True)
        storage = receiver._storage
        got = []
        for start in range(0, len(audio), 173):
            got.extend(receiver.feed(audio[start:start+173]))
        got.extend(receiver.flush())
        self.assertEqual([r.absolute for r in got], list(range(1, 25)))
        self.assertIs(receiver._storage, storage)
        for result in got:
            self.assertLess(np.sqrt(np.mean((result.values-values)**2)), .03)

    def test_reset_discards_partial_packet_and_reuses_storage(self):
        layout = v2.PRESETS['wide']
        coder = v2.SourceCoder([(12, 12)]*3)
        receiver = v2.Receiver(layout, coder, recovery=False, fast=True)
        values = np.zeros(coder.count)
        packet = v2.encode(values, layout, coder, 1, 1, 1)
        receiver.feed(packet[:1000])
        storage = receiver._storage
        receiver.reset()
        got = receiver.feed(packet)+receiver.flush()
        self.assertEqual([r.absolute for r in got], [1])
        self.assertIs(receiver._storage, storage)
