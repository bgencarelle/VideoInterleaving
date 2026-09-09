import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
from PIL import Image
from scipy.signal import resample_poly, butter, sosfilt
from animation_modem.transport import encode, decode_packet, Receiver, FRAME, image_values, values_image
from animation_modem.audio_common import pair, route


class TransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Image.fromarray(np.random.default_rng(42).integers(0, 256, (96, 80, 3), dtype=np.uint8))
        cls.frames = [encode(cls.source, n, (n-1)%3+1, 3, True)[0] for n in range(1, 9)]
        cls.stream = np.concatenate(cls.frames)

    def receive(self, data, chunk=257):
        rx = Receiver()
        results = []
        for i in range(0, len(data), chunk):
            results.extend(rx.feed(data[i:i+chunk]))
        return results

    def test_pcm_and_arbitrary_chunks(self):
        pcm = np.rint(self.stream*32767).astype(np.int16).astype(float)/32768
        results = self.receive(pcm)
        self.assertEqual([r.frame for r in results], list(range(1, 9)))
        self.assertTrue(all(r.image is not None for r in results))

    def test_each_frame_alone(self):
        for i, audio in enumerate(self.frames, 1):
            results = self.receive(audio)
            self.assertEqual([r.frame for r in results], [i])
            self.assertEqual(results[0].identity, 'verified_header')

    def test_image_and_overlay_roundtrip(self):
        audio, im, _ = encode(self.source, 1234, 2, 3, True)
        result = decode_packet(audio)
        expected = np.asarray(values_image(image_values(im))).astype(float)
        self.assertLess(np.abs(np.asarray(result.image)-expected).max(), 2)
        self.assertEqual((result.frame, result.index, result.count), (1234, 2, 3))
        _, plain, _ = encode(self.source, 1234, 2, 3, False)
        self.assertFalse(np.array_equal(np.asarray(plain), np.asarray(im)))

    def test_gain_polarity_and_crosstalk(self):
        left, right = self.stream[:, 0], self.stream[:, 1]
        audio = np.column_stack((-.6*left+.12*right, .15*left+.4*right))
        self.assertTrue(np.isfinite(audio).all())
        results = self.receive(audio)
        self.assertEqual([r.frame for r in results if r.image], list(range(1, 9)))

    def test_join_midstream(self):
        results = self.receive(self.stream[1234:])
        self.assertEqual([r.frame for r in results if r.image], list(range(2, 9)))

    def test_dropout_is_local(self):
        audio = self.stream.copy()
        audio[2*FRAME:3*FRAME] = 0
        results = self.receive(audio)
        self.assertEqual([r.frame for r in results], list(range(1, 9)))
        self.assertEqual(results[2].status, 'missing')
        self.assertEqual(results[2].identity, 'estimated')
        self.assertTrue(all(r.image is not None for r in results[3:]))

    def test_corrupt_header_does_not_poison_next_frame(self):
        audio = self.stream.copy()
        audio[FRAME+576:FRAME+864] = 0
        results = self.receive(audio)
        self.assertEqual(results[1].status, 'partial_header_unknown')
        self.assertIsNotNone(results[1].image)
        self.assertEqual(results[1].identity, 'estimated')
        self.assertEqual(results[1].frame, 2)
        self.assertEqual([r.frame for r in results[2:] if r.image], list(range(3, 9)))

    def test_bad_first_header_salvages_current_image_without_identity(self):
        audio=self.frames[3].copy()
        expected=decode_packet(audio).image
        audio[576:864]=0
        results=self.receive(audio)
        self.assertEqual(len(results),1)
        self.assertIsNone(results[0].frame)
        self.assertEqual(results[0].identity,'unknown')
        self.assertEqual(results[0].status,'partial_header_unknown')
        np.testing.assert_array_equal(np.asarray(results[0].image),np.asarray(expected))

    def test_preamble_followed_by_noise_is_not_a_recovered_image(self):
        audio=self.frames[0].copy()
        audio[288:]=np.random.default_rng(123).normal(0,.1,audio[288:].shape)
        r=decode_packet(audio)
        self.assertIsNone(r.image)
        self.assertIsNone(r.frame)

    def test_small_clock_mismatch(self):
        audio = resample_poly(self.stream, 10001, 10000, axis=0)
        results = self.receive(audio)
        self.assertEqual([r.frame for r in results if r.image], list(range(1, 9)))

    def test_noise(self):
        rms = np.sqrt(np.mean(self.stream**2))
        audio = self.stream + np.random.default_rng(11).normal(0, rms*10**(-25/20), self.stream.shape)
        results = self.receive(audio)
        self.assertEqual([r.frame for r in results if r.image], list(range(1, 9)))

    def test_filter_delay(self):
        audio = sosfilt(butter(4, 12000, fs=48000, output='sos'), self.stream, axis=0)
        results = self.receive(audio)
        self.assertEqual([r.frame for r in results if r.image], list(range(1, 9)))

    def test_unknown_silence(self):
        results = self.receive(np.zeros((FRAME*3, 2)))
        self.assertEqual(len(results), 3)
        self.assertTrue(all(r.frame is None and r.status == 'unidentified' for r in results))

    def test_channel_routing(self):
        routed = route(self.frames[0], pair('4,2'))
        np.testing.assert_array_equal(routed[:, [3, 1]], self.frames[0])
        self.assertTrue(np.all(routed[:, [0, 2]] == 0))


if __name__ == '__main__':
    unittest.main()
