"""Prebake registered RGBA layers using the existing asset slab writer."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utilities.bake_assets import write_slab
from utilities.convert_to_xy import load_rgba, _image_files
from make_file_lists import check_folder_prefix, natural_sort_key
from animation_modem.transport import PROFILES


def bake_tree(source, output, profile='color', jpeg_layout='sbs'):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir():
        raise ValueError(f'Source directory does not exist: {source}')
    if output.exists():
        raise ValueError(f'Output already exists: {output}. Bake to a new directory.')
    if output == source or source in output.parents:
        raise ValueError('Bake output must be outside the source tree')
    if profile not in PROFILES or jpeg_layout not in ('sbs','rgb'):
        raise ValueError('Unsupported profile or JPEG layout')
    tasks=[]
    for kind, name in [('main','face'),('float','float')]:
        tree=source/name
        if not tree.is_dir():
            raise ValueError(f'Missing layer directory: {tree}')
        for folder in sorted([tree,*tree.rglob('*')],key=lambda p:natural_sort_key(str(p))):
            if folder.is_dir() and check_folder_prefix(str(folder),kind):
                files=_image_files(folder)
                if files: tasks.append((kind,folder,files))
    if not all(any(t[0]==kind for t in tasks) for kind in ['main','float']):
        raise ValueError('Need numeric face folders (0–254) and float folders (255_...)')
    size=PROFILES[profile][0]
    output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.modem-bake-',dir=output.parent))
    def loader(path):
        if jpeg_layout=='rgb' and path.suffix.lower() in ('.jpg','.jpeg'):
            with Image.open(path) as im: return im.convert('RGBA')
        rgb,alpha=load_rgba(path)  # Scope baker's RGBA / colour|matte convention.
        if path.suffix.lower() in ('.jpg','.jpeg'):
            alpha=alpha.copy();alpha[alpha<13]=0
        return Image.fromarray(np.dstack([rgb,alpha]))
    entries=[]
    try:
        for kind,folder,files in tasks:
            relative=folder.relative_to(source)
            write_slab(files,stage/relative/'frames.npy',size,loader,Image.Resampling.LANCZOS)
            entries.append({'layer':kind,'path':relative.as_posix(),'count':len(files),
                            'names':[p.name for p in files]})
            print(f'[MODEM BAKE] {relative}: {len(files)} frames, {size[0]}x{size[1]} RGBA')
        manifest={'format':'video-interleaving-modem','version':1,'profile':profile,
                  'size':list(size),'jpeg_layout':jpeg_layout,'folders':entries}
        (stage/'modem.json').write_text(json.dumps(manifest,indent=2)+'\n')
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True)
        raise
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('-i','--input-dir',required=True,type=Path)
    p.add_argument('-o','--output-dir',type=Path)
    p.add_argument('--profile',choices=PROFILES,default='color')
    p.add_argument('--jpeg-layout',choices=['sbs','rgb'],default='sbs',
                   help='Project JPEG convention: colour|matte (default); rgb for ordinary JPEGs')
    a=p.parse_args()
    out=a.output_dir or a.input_dir.with_name(a.input_dir.name+'_modem')
    try:bake_tree(a.input_dir,out,a.profile,a.jpeg_layout)
    except (OSError,ValueError) as exc:p.error(str(exc))


if __name__=='__main__':main()
