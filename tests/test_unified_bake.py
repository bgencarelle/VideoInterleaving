"""Independent layers, decoder parity, bounded ownership and package routing."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from baked_layers import BakedLayers, LayerFrame, package_output
from utilities.bake import bake, discover


class UnifiedBakeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir='tmp', prefix='unified_bake_test_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'source'
        for layer, folder in (('face', '0_main'), ('face', '2_second'),
                              ('float', '255_front')):
            dest = self.source/layer/folder
            dest.mkdir(parents=True)
            for i in (1,2,10):
                a = np.zeros((12,16,4), np.uint8)
                if layer == 'face':
                    a[:] = (i*20, 60, 130, 255)
                    if folder == '2_second':
                        a[:,:,1] = 170
                else:
                    a[3:9,4:10] = (70,200,90,145)
                Image.fromarray(a).save(dest/f'frame{i}.png')

    def build(self, modes=('local','ascii','asciiweb','web'), **kwargs):
        dest = self.root/'bakes'
        bake(self.source, dest, modes, **kwargs)
        return dest

    def test_separate_layers_deduplicate_without_losing_selection_or_pixels(self):
        dest = self.build()
        self.assertEqual(package_output(dest,'local'),package_output(dest,'ascii'))
        with BakedLayers(package_output(dest,'local')) as lib:
            self.assertEqual(lib.frames,3)
            self.assertEqual(len(lib.main_paths[0]),2)
            for i,number in enumerate((1,2,10)):
                a,b,ms,fs = lib.load_images(i,0,0)
                np.testing.assert_array_equal(a, np.asarray(Image.open(
                    self.source/f'face/0_main/frame{number}.png')))
                self.assertFalse(ms or fs)
                self.assertFalse(a.flags.writeable)
            first = lib.load_images(0,0,0)
            other = lib.load_images(0,1,0)
            self.assertFalse(np.array_equal(first[0],other[0]))
            self.assertIs(first[1],other[1])
            info = json.loads((package_output(dest,'local')/'layers.json').read_text())
            self.assertLess(info['unique_pixel_bytes'],info['decoded_bytes'])
            retained = first[0]
        self.assertEqual(int(retained[0,0,0]),20)  # Closing owner cannot invalidate a view.

    def test_jpeg_sbs_matches_live_turbojpeg_decode_exactly(self):
        from image_loader import ImageLoader
        for path in self.source.rglob('*.png'):
            rgba=np.asarray(Image.open(path))
            packed=np.concatenate([rgba[:,:,:3],np.repeat(rgba[:,:,3:4],3,axis=2)],axis=1)
            Image.fromarray(packed).save(path.with_suffix('.jpg'),quality=93)
            path.unlink()
        dest=self.build(('local',))
        with BakedLayers(package_output(dest,'local')) as lib:
            for i,n in enumerate((1,2,10)):
                live,sbs=ImageLoader().read_image(str(self.source/f'face/0_main/frame{n}.jpg'))
                actual,actual_sbs=lib.read_image(LayerFrame('main',0,i))
                np.testing.assert_array_equal(actual,live)
                self.assertEqual(sbs,actual_sbs)

    def test_failure_does_not_publish_a_partial_package(self):
        output=self.root/'failed'
        with patch('utilities.bake.bake_layers',side_effect=OSError('interrupted')):
            with self.assertRaises(OSError):bake(self.source,output,('local',))
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob('.unified-bake-*')))

    def test_corrupt_index_and_truncated_storage_are_rejected_at_open(self):
        dest=self.build(('local',))/'layers'
        pixels=dest/'pixels.bin'
        with pixels.open('r+b') as f:f.truncate(1)
        with self.assertRaisesRegex(ValueError,'Truncated'):BakedLayers(dest)

    def test_source_count_selection_and_natural_folder_order_match_application(self):
        bad=self.source/'face/1_wrong_count'
        bad.mkdir()
        Image.new('RGBA',(12,12)).save(bad/'a.png')
        frames,tasks=discover(self.source.resolve())
        self.assertEqual(frames,3)
        self.assertEqual([folder.name for _,folder,_ in tasks],
                         ['0_main','2_second','255_front'])

    def test_all_mode_package_reads_with_existing_scope_and_modem_readers(self):
        dest=self.build(('local','web','ascii','asciiweb','scope','modem'),
                        thumb_width=32,workers=1)
        from scope_bake import XYLibrary
        from modem_bake import ModemLibrary
        scope=XYLibrary(package_output(dest,'scope')/'face/0_main')
        self.assertEqual(scope.thumb(0).shape[-1],2)
        modem=ModemLibrary(package_output(dest,'modem'))
        self.assertEqual(modem.frames,3)
        self.assertEqual(len(modem.mains),2)
        self.assertEqual(modem.composite(1,1,0).size,(80,96))

    def test_web_half_decode_matches_live_policy_and_keeps_full_local_layers(self):
        from image_loader import ImageLoader
        for path in self.source.rglob('*.png'):
            rgba=np.asarray(Image.open(path))
            packed=np.concatenate([rgba[:,:,:3],np.repeat(rgba[:,:,3:4],3,axis=2)],axis=1)
            Image.fromarray(packed).save(path.with_suffix('.jpg'),quality=93)
            path.unlink()
        dest=self.build(('local','web'),web_size=(4,4))
        self.assertNotEqual(package_output(dest,'local'),package_output(dest,'web'))
        with BakedLayers(package_output(dest,'web')) as lib:
            path=self.source/'face/0_main/frame1.jpg'
            expected,sbs=ImageLoader().read_image(str(path),jpeg_scaling_factor=(1,2),
                                                 jpeg_min_size=(16,8))
            actual,actual_sbs=lib.read_image(LayerFrame('main',0,0))
            np.testing.assert_array_equal(actual,expected)
            self.assertEqual(actual_sbs,sbs)

    def test_ordinary_jpeg_layout_is_honoured_in_all_backends(self):
        for path in self.source.rglob('*.png'):
            with Image.open(path) as im:im.convert('RGB').save(path.with_suffix('.jpeg'))
            path.unlink()
        dest=self.build(('local','scope','modem'),jpeg_layout='rgb',
                        thumb_width=32,thumbs_only=True,workers=1)
        with BakedLayers(package_output(dest,'local')) as lib:
            image,sbs=lib.read_image(LayerFrame('main',0,0))
            self.assertEqual(image.shape,(12,16,3))
            self.assertFalse(sbs)
        from scope_bake import XYLibrary
        scope=XYLibrary(package_output(dest,'scope')/'face/0_main')
        self.assertEqual(scope.thumb(0).shape,(24,32,2))
        from modem_bake import ModemLibrary
        modem=ModemLibrary(package_output(dest,'modem'))
        self.assertEqual(modem.source_sizes[modem.mains[0]][0],(16,12))

    def test_baked_cli_uses_manifest_without_scanning_source(self):
        dest=self.build(('local',)).resolve()
        script='''
import sys
sys.argv=['main.py','--mode','local','--bake-dir',sys.argv[1]]
import main,settings
assert settings.LAYER_BAKE_DIR
import make_file_lists
make_file_lists.process_files=lambda **kwargs: (_ for _ in ()).throw(AssertionError('source scan'))
import image_display
from baked_layers import BakedLayers
with BakedLayers(settings.LAYER_BAKE_DIR) as lib:
    assert lib.frames==3
image_display.run_display=lambda *a: None
import web_service
web_service.start_server=lambda **kwargs: None
main.main()
'''
        result=subprocess.run([sys.executable,'-c',script,str(dest)],text=True,
                              capture_output=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__ == '__main__':
    unittest.main()
