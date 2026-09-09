import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
from PIL import Image
from animation_modem.transport import PROFILES, FRAME, encode, decode_packet, Receiver, image_values, values_image
from animation_modem.impairments import Emulator, Settings, PRESETS


class ProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=Image.fromarray(np.random.default_rng(42).integers(0,256,(160,120,3),dtype=np.uint8))

    def test_each_profile_has_same_budget_and_independent_roundtrip(self):
        for profile,(size,_) in PROFILES.items():
            with self.subTest(profile=profile):
                audio,im,_=encode(self.source,123,2,10,True,profile)
                self.assertEqual(audio.shape,(FRAME,2))
                self.assertEqual(len(image_values(im,profile)),2880)
                r=decode_packet(audio)
                self.assertEqual((r.frame,r.index,r.count),(123,2,10))
                self.assertEqual(r.profile,profile)
                self.assertEqual(r.profile_identity,'verified_header')
                self.assertEqual(r.image.size,size)
                expected=np.asarray(values_image(image_values(im,profile),profile),dtype=float)
                self.assertLess(np.max(abs(np.asarray(r.image)-expected)),2)
                if profile=='mono':
                    np.testing.assert_array_equal(np.asarray(r.image)[:,:,0],np.asarray(r.image)[:,:,1])

    def test_midstream_profile_switches(self):
        modes=['color','mono','detail','color','detail','mono']
        audio=np.concatenate([encode(self.source,i+1,1,1,True,name)[0] for i,name in enumerate(modes)])
        rx=Receiver();rows=[]
        for i in range(0,len(audio),257):rows.extend(rx.feed(audio[i:i+257]))
        self.assertEqual([r.profile for r in rows],modes)
        self.assertEqual([r.frame for r in rows],list(range(1,7)))

    def test_header_erasure_retains_correct_profile_and_image(self):
        for profile in PROFILES:
            audio,_,_=encode(self.source,777,1,1,True,profile)
            expected=decode_packet(audio).image
            audio[576:864]=0
            r=decode_packet(audio)
            self.assertIsNone(r.frame)
            self.assertEqual(r.profile,profile)
            self.assertEqual(r.profile_identity,'estimated_training')
            self.assertEqual(r.status,'partial_header_unknown')
            np.testing.assert_array_equal(np.asarray(r.image),np.asarray(expected))

    def test_profile_detection_under_mild_impairment(self):
        for profile in PROFILES:
            audio,_,_=encode(self.source,123,1,1,True,profile)
            r=decode_packet(Emulator(Settings(**PRESETS['mild'])).process(audio))
            self.assertEqual(r.profile,profile)
            self.assertEqual(r.frame,123)

    def test_profile_detection_with_polarity_and_crosstalk(self):
        for profile in PROFILES:
            a,_,_=encode(self.source,123,1,1,True,profile)
            l,r=a[:,0],a[:,1]
            mixed=np.column_stack((-.6*l+.12*r,.15*l+.4*r))
            result=decode_packet(mixed)
            self.assertEqual(result.frame,123)
            self.assertEqual(result.profile,profile)


if __name__=='__main__':unittest.main()
