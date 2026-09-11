"""Read baked modem layers and composite the project's face/float selection."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from animation_modem.imaging import PROFILES


class ModemLibrary:
    def __init__(self, root):
        from make_file_lists import natural_sort_key, check_folder_prefix
        self.root=Path(root).resolve()
        manifest=json.loads((self.root/'modem.json').read_text())
        if manifest.get('format')!='video-interleaving-modem' or manifest.get('version')!=1:
            raise ValueError('Unsupported modem bake format')
        self.profile=manifest['profile']
        if self.profile not in PROFILES or tuple(manifest['size'])!=PROFILES[self.profile][0]:
            raise ValueError('Invalid modem bake profile/dimensions')
        self.size=PROFILES[self.profile][0]
        groups={'main':{},'float':{}}
        self.slabs={}
        seen=set()
        for entry in manifest['folders']:
            layer=entry['layer'];relative=Path(entry['path'])
            folder=(self.root/relative).resolve()
            if (relative.is_absolute() or self.root not in folder.parents or folder in seen
                    or layer not in groups or not check_folder_prefix(str(folder),layer)):
                raise ValueError(f'Invalid/duplicate bake path: {relative}')
            seen.add(folder)
            data=np.load(folder/'frames.npy',mmap_mode='r',allow_pickle=False)
            count=entry['count']
            w,h=self.size
            if (not isinstance(count,int) or count<=0 or data.dtype!=np.uint8
                    or data.shape!=(count,h,w,4) or len(entry['names'])!=count):
                raise ValueError(f'Invalid slab or frame manifest: {relative}')
            groups[layer].setdefault(count,[]).append(folder)
            self.slabs[folder]=data
        common=set(groups['main']) & set(groups['float'])
        if not common:
            raise ValueError('No shared face/float frame count in modem bake')
        self.frames=max(common)  # Same largest-common-count rule as scope/video.
        def key(path):
            return (int(path.name.partition('_')[0]),natural_sort_key(path.name),
                    natural_sort_key(str(path.relative_to(self.root))))
        self.mains=sorted(groups['main'][self.frames],key=key)
        self.floats=sorted(groups['float'][self.frames],key=key)

    def composite(self, index, main_folder, float_folder, background=(4,4,4),
                  rotation=0, mirror=False):
        if not 0<=index<self.frames:
            raise ValueError('Image index is outside the baked sequence')
        if not 0<=main_folder<len(self.mains) or not 0<=float_folder<len(self.floats):
            raise ValueError('Selected face/float folder is outside the manifest')
        main=self.slabs[self.mains[main_folder]][index].astype(np.float32)/255
        front=self.slabs[self.floats[float_folder]][index].astype(np.float32)/255
        # Same straight-alpha over order as renderer.py: background, main, float.
        rgb=np.asarray(background,dtype=np.float32)/255
        rgb=main[:,:,:3]*main[:,:,3:4]+rgb*(1-main[:,:,3:4])
        rgb=front[:,:,:3]*front[:,:,3:4]+rgb*(1-front[:,:,3:4])
        im=Image.fromarray(np.uint8(np.clip(np.rint(rgb*255),0,255)))
        if rotation%360:
            im=im.rotate(rotation%360,expand=True)
        if mirror:
            im=im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return im
