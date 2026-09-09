import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
from utilities.convert_to_modem import bake_tree
from utilities.bake_assets import write_slab
from modem_bake import ModemLibrary
from modem_display import packet, run_modem
from animation_modem.transport import decode_packet

REPO=Path(__file__).resolve().parent


def source_tree(root, count=3):
    for name,color in [('face/0_rest',(0,0,0,255)),
                       ('face/2_face',(200,40,20,128)),
                       ('face/10_face',(20,200,40,255)),
                       ('float/255_overlay',(10,30,240,128))]:
        folder=root/name;folder.mkdir(parents=True)
        for i in range(count):Image.new('RGBA',(40,48),color).save(folder/f'frame_{i}.png')


class ModemIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.source=self.root/'images';source_tree(self.source)
        self.bake=self.root/'images_modem'
        with contextlib.redirect_stdout(io.StringIO()):bake_tree(self.source,self.bake)
    def tearDown(self):self.temp.cleanup()

    def test_folder_order_alpha_composition_and_packet(self):
        lib=ModemLibrary(self.bake)
        self.assertEqual([p.name for p in lib.mains],['0_rest','2_face','10_face'])
        self.assertEqual(lib.frames,3)
        image=lib.composite(1,1,0)
        expected=Image.alpha_composite(Image.new('RGBA',(40,48),(4,4,4,255)),
                                      Image.new('RGBA',(40,48),(200,40,20,128)))
        expected=Image.alpha_composite(expected,Image.new('RGBA',(40,48),(10,30,240,128))).convert('RGB')
        self.assertLessEqual(abs(np.asarray(image,dtype=float)-np.asarray(expected)).max(),1)
        audio,report,_=packet(lib,500,(1,1,0),True)
        decoded=decode_packet(audio)
        self.assertEqual((decoded.frame,decoded.index,decoded.count),(500,2,3))
        self.assertEqual((report['face_folder'],report['float_folder']),(1,0))
        self.assertEqual(decoded.profile,'color')

    def test_largest_common_count(self):
        for name in ['face/3_short','float/255_short']:
            d=self.source/name;d.mkdir()
            Image.new('RGBA',(40,48)).save(d/'0.png')
        out=self.root/'other_modem'
        with contextlib.redirect_stdout(io.StringIO()):bake_tree(self.source,out)
        lib=ModemLibrary(out)
        self.assertEqual(lib.frames,3)
        self.assertEqual(len(lib.mains),3)
        self.assertEqual(len(lib.floats),1)

    def test_sbs_jpeg_alpha_is_retained(self):
        src=self.root/'sbs'
        for name in ['face/0_base','float/255_mask']:
            d=src/name;d.mkdir(parents=True)
            image=Image.new('RGB',(80,48),(64,64,64))
            image.paste(Image.new('RGB',(40,48),(200,50,20)),(0,0))
            image.save(d/'0.jpg',quality=100,subsampling=0)
        out=self.root/'sbs_modem'
        with contextlib.redirect_stdout(io.StringIO()):bake_tree(src,out)
        lib=ModemLibrary(out)
        pixel=lib.slabs[lib.mains[0]][0,20,20].astype(float)
        self.assertLess(abs(pixel-np.array([200,50,20,64])).max(),3)

    def test_atomic_slab_failure_keeps_old_file(self):
        existing=self.root/'frames.npy'
        np.save(existing,np.array([9]))
        before=existing.read_bytes()
        with self.assertRaises(FileNotFoundError):
            write_slab([self.root/'missing.png'],existing,(40,48))
        self.assertEqual(before,existing.read_bytes())
        self.assertFalse(list(self.root.glob('.slab-*')))

    def test_failed_tree_bake_is_not_published(self):
        (self.source/'face/2_face/broken.png').write_bytes(b'not an image')
        out=self.root/'failed_modem'
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(OSError):bake_tree(self.source,out)
        self.assertFalse(out.exists())
        self.assertFalse(list(self.root.glob('.modem-bake-*')))

    def test_live_uses_shared_clock_and_selector_without_polling_thread(self):
        import index_calculator
        import folder_selector
        emitted=[]
        class Output:
            def __init__(self,*args):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def ready(self):return True
            def reserve(self,*args):return SimpleNamespace(start_time=1,target_time_ns=1_700_000_000_000_000_000)
            def submit(self,audio,slot=None):
                emitted.append(decode_packet(audio));return True
            def finish(self):pass
        args=SimpleNamespace(modem_dir=str(self.bake), modem_pair=None,
            modem_wav=None, modem_channels='1,2', modem_latency='low',
            rotation=None, mirror=None, modem_clock=255, modem_frame_duration=1,
            modem_time_offset_ms=0, modem_prepare_ms=10, modem_receive_margin_ms=15, modem_frames=3, modem_numbered=False,
            modem_log_frames=False, scope_device=None, modem_index_offset_ms=33.3)
        with patch.object(index_calculator,'set_clock_mode'), \
             patch.object(index_calculator,'midi_mode',False), \
             patch.object(index_calculator,'update_index',side_effect=[(0,None),(0,None),(1,None)]) as clock, \
             patch.object(index_calculator,'calculate_free_clock_index',side_effect=[(0,None),(0,None),(1,None)]) as target_clock, \
             patch.object(folder_selector,'update_folder_selection') as select, \
             patch.dict(folder_selector.folder_dictionary,{'Main_and_Float_Folders':(1,0)}), \
             patch('modem_display.PacketOutput',Output), patch('modem_display.time.sleep'), \
             contextlib.redirect_stdout(io.StringIO()):
            run_modem(args)
        self.assertEqual(select.call_count,2)
        self.assertEqual([x.index for x in emitted],[1,1,2])
        self.assertEqual([x.frame for x in emitted],[1,2,3])
        self.assertTrue(all(not call.kwargs for call in clock.call_args_list))
        self.assertTrue(all(call.kwargs['at_time_ns']==1_700_000_000_000_000_000 for call in target_clock.call_args_list))
        self.assertTrue(all(call.kwargs['publish'] is False for call in target_clock.call_args_list))
        # The index offset must reach the clock formula without disturbing the
        # transmitted timestamp: presentation time is unchanged above.
        self.assertTrue(all(call.kwargs['time_offset_ns']==33_300_000 for call in target_clock.call_args_list))
        self.assertTrue(all(x.target_time_ms32 is not None for x in emitted))

    def test_main_wav_path_needs_no_video_or_audio_device_stack(self):
        wav=self.root/'test.wav'
        script="""
import sys,runpy
class Block:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in ('glfw','OpenGL','moderngl','turbojpeg','sounddevice','mido','cv2'):
            raise ImportError('blocked unused dependency '+name)
sys.meta_path.insert(0,Block())
runpy.run_path(sys.argv[1],run_name='__main__')
"""
        # run_path uses the arguments after the main.py path; remove the wrapper path.
        script=script.replace("runpy.run_path(sys.argv[1],run_name='__main__')",
                              "target=sys.argv.pop(1); sys.path.insert(0,str(__import__('pathlib').Path(target).parent)); runpy.run_path(target,run_name='__main__')")
        r=subprocess.run([sys.executable,'-c',script,str(REPO/'main.py'),'--mode','modem',
                           '--modem-dir',str(self.bake),'--modem-pair','1,0',
                           '--modem-wav',str(wav),'--modem-frames','5','-f'],
                          cwd=self.root,text=True,capture_output=True,timeout=30)
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        r=subprocess.run([sys.executable,str(REPO/'modem_receive.py'),'--wav',str(wav),
                           '--headless','--fast'],cwd=self.root,text=True,capture_output=True,timeout=30)
        self.assertEqual(r.returncode,0,r.stderr)
        rows=[json.loads(line) for line in r.stdout.splitlines()]
        self.assertEqual([x['frame'] for x in rows],list(range(1,6)))
        self.assertEqual([x['index'] for x in rows],[1,2,3,1,2])


if __name__=='__main__':unittest.main()
