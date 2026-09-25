"""Read baked modem layers and composite the project's face/float selection."""
import json
from pathlib import Path
import numpy as np
from numba import njit
from PIL import Image
from animation_modem.imaging import PROFILES, source_size


def _alpha_over_numpy(main, front, background):
    """Reference for _alpha_over: straight-alpha over, as renderer.py does."""
    main=main.astype(np.float32)/255
    front=front.astype(np.float32)/255
    rgb=np.asarray(background,dtype=np.float32)/255
    rgb=main[:,:,:3]*main[:,:,3:4]+rgb*(1-main[:,:,3:4])
    rgb=front[:,:,:3]*front[:,:,3:4]+rgb*(1-front[:,:,3:4])
    return np.uint8(np.clip(np.rint(rgb*255),0,255))


@njit(cache=True, fastmath=False)
def _alpha_over_kernel(main, front, background, out):
    scale=np.float32(255)
    one=np.float32(1)
    for y in range(main.shape[0]):
        for x in range(main.shape[1]):
            main_alpha=np.float32(main[y,x,3])/scale
            front_alpha=np.float32(front[y,x,3])/scale
            for c in range(3):
                value=(np.float32(main[y,x,c])/scale*main_alpha+
                       background[c]*(one-main_alpha))
                value=(np.float32(front[y,x,c])/scale*front_alpha+
                       value*(one-front_alpha))
                value=np.rint(value*scale)
                out[y,x,c]=np.uint8(min(max(value,np.float32(0)),scale))


def _alpha_over(main, front, background):
    """Background, main, float composited in float32 (bit-identical to
    _alpha_over_numpy, one pass instead of a dozen full-array temporaries)."""
    main,front=np.asarray(main),np.asarray(front)
    if (main.dtype!=np.uint8 or front.dtype!=np.uint8 or main.shape!=front.shape
            or main.ndim!=3 or main.shape[2]!=4):
        return _alpha_over_numpy(main,front,background)
    main,front=np.ascontiguousarray(main),np.ascontiguousarray(front)
    out=np.empty(main.shape[:2]+(3,),np.uint8)
    _alpha_over_kernel(main,front,
                       np.asarray(background,dtype=np.float32)/np.float32(255),out)
    return out


class ModemLibrary:
    def __init__(self, root):
        from make_file_lists import natural_sort_key, check_folder_prefix
        self.root=Path(root).resolve()
        manifest=json.loads((self.root/'modem.json').read_text())
        if manifest.get('format')!='video-interleaving-modem' or manifest.get('version')!=1:
            raise ValueError('Unsupported modem bake format')
        self.profile=manifest['profile']
        # Against the SAMPLING size, not the wire shape. A truncating profile
        # bakes bigger than it transmits -- color-dct is 80x96 pixels behind a
        # 40x48 wire shape -- so checking the wire shape here rejects exactly
        # the bakes that profile exists for.
        # V7 has one fixed source geometry: color-dct's 80x96 sampling grid
        # truncated to 48x40 luma plus 24x20 chroma.  The older lean-dct
        # profile has the same bake dimensions but a different wire shape, so
        # accepting it here would let the sender silently encode the wrong
        # number of coefficients.
        if (self.profile != 'color-dct' or self.profile not in PROFILES
                or tuple(manifest['size']) != source_size(self.profile)):
            raise ValueError('Invalid modem bake profile/dimensions')
        self.size=source_size(self.profile)
        groups={'main':{},'float':{}}
        self.slabs={}
        self.source_sizes={}
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
            source_sizes=entry.get('source_sizes')
            if source_sizes is None:  # pre-aspect-metadata bake compatibility
                source_sizes=[list(self.size) for _ in range(count)]
            if (not isinstance(source_sizes,list) or len(source_sizes)!=count
                    or any(not isinstance(size,list) or len(size)!=2
                           or any(type(value) is not int or value<=0
                                  for value in size)
                           for size in source_sizes)):
                raise ValueError(f'Invalid source dimensions: {relative}')
            groups[layer].setdefault(count,[]).append(folder)
            self.slabs[folder]=data
            self.source_sizes[folder]=[tuple(size) for size in source_sizes]
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
        main=self.slabs[self.mains[main_folder]][index]
        front=self.slabs[self.floats[float_folder]][index]
        source_dimensions=self.source_sizes[self.mains[main_folder]][index]
        # Same straight-alpha over order as renderer.py: background, main, float.
        im=Image.fromarray(_alpha_over(main,front,background))
        if rotation%360:
            im=im.rotate(rotation%360,expand=True)
            if rotation%180==90:
                source_dimensions=source_dimensions[::-1]
        if mirror:
            im=im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        im.info['source_dimensions']=source_dimensions
        return im
