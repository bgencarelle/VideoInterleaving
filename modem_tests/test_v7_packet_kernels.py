"""The compiled limiter must preserve final wire samples, not just peak bounds."""
import unittest

import numpy as np

from animation_modem.v7_packet_kernels import limit_packet


def numpy_limit(shaped, header, eof, metadata, overlay, meta_start,
                body_start, body_stop, eof_start, body_ratio, metadata_ratio):
    """Frozen pre-kernel float32 staging and two 16-step limiter searches."""
    def assemble(body_gain, metadata_gain):
        packet = shaped*np.float32(body_gain)
        packet += header
        packet[meta_start:meta_start+len(metadata)] += metadata[:, None]*np.float32(metadata_gain)
        packet += eof
        packet += overlay*np.float32(body_gain)
        return packet

    def peak(packet, start, stop):
        return float(np.max(np.abs(packet[start:stop])))

    def below(packet):
        framing = min(peak(packet, 0, body_start), peak(packet, eof_start, len(packet)))
        return peak(packet, body_start, body_stop) <= framing*body_ratio

    body_gain = 1.0
    packet = assemble(body_gain, 1.0)
    if not below(packet):
        low, high = 0.0, body_gain
        for _ in range(16):
            trial_gain = (low+high)*.5
            if below(assemble(trial_gain, 1.0)):
                low = trial_gain
            else:
                high = trial_gain
        body_gain = low
        packet = assemble(body_gain, 1.0)
    metadata_limit = peak(packet, body_start, body_stop)*metadata_ratio
    if peak(packet, meta_start, meta_start+len(metadata)) > metadata_limit:
        low, high = 0.0, 1.0
        for _ in range(16):
            trial_gain = (low+high)*.5
            if peak(assemble(body_gain, trial_gain), meta_start,
                    meta_start+len(metadata)) <= metadata_limit:
                low = trial_gain
            else:
                high = trial_gain
        packet = assemble(body_gain, low)
    return packet


class PacketKernelTests(unittest.TestCase):
    def test_final_samples_match_numpy_with_limiting_and_tone_cancellation(self):
        rng = np.random.default_rng(3272)
        length, body_start, body_stop = 3920, 288, 3744
        for body_level in (0.0, .1, .7, 2.0):
            for metadata_level in (0.0, .1, .8):
                for tone_level in (0.0, .04):
                    with self.subTest(body=body_level, meta=metadata_level, tone=tone_level):
                        shaped = rng.normal(0, body_level, (length, 2)).astype(np.float32)
                        header = np.zeros_like(shaped)
                        header[:body_start] = rng.uniform(-.55, .55, (body_start, 2))
                        eof = np.zeros_like(shaped)
                        eof[-24:] = rng.uniform(-.55, .55, (24, 2))
                        metadata = rng.normal(0, metadata_level, 144)
                        overlay = rng.normal(0, tone_level, shaped.shape).astype(np.float32)
                        arguments = (shaped, header, eof, metadata, overlay, body_stop,
                                     body_start, body_stop, length-24, .84, .5)
                        np.testing.assert_array_equal(limit_packet(*arguments),
                                                      numpy_limit(*arguments))


if __name__ == '__main__':
    unittest.main()
