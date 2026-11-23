"""Unified offline baker. Every output retains independent face/float layers."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

MODES = ('local', 'web', 'ascii', 'asciiweb', 'scope', 'modem')


def discover(source):
    from make_file_lists import natural_sort_key, check_folder_prefix
    from utilities.convert_to_xy import _image_files
    groups = {'main': {}, 'float': {}}
    for layer, name in (('main', 'face'), ('float', 'float')):
        tree = source / name
        if not tree.is_dir():
            raise ValueError(f'Missing source layer: {tree}')
        folders = [p for p in tree.rglob('*') if p.is_dir()
                   and check_folder_prefix(str(p), layer)]
        def key(p):
            prefix = p.name.partition('_')[0]
            return (int(prefix), natural_sort_key(p.name),
                    natural_sort_key(str(p.relative_to(source))))
        for folder in sorted(folders, key=key):
            files = _image_files(folder)
            if files:
                groups[layer].setdefault(len(files), []).append((layer, folder, files))
    common = groups['main'].keys() & groups['float'].keys()
    if not common:
        raise ValueError('Need numeric face/float folders with a common frame count')
    frames = max(common)
    selected = groups['main'][frames] + groups['float'][frames]
    return frames, selected


def bake_layers(source, output, tasks, frames, jpeg_layout, web_size=None):
    """Stream exact decoder pixels to a deduplicated mmap store; never mix layers."""
    from image_loader import ImageLoader
    from PIL import Image
    loader = ImageLoader()
    output.mkdir(parents=True)
    entries, hashes = [], {}
    original_bytes = raw_bytes = 0
    # Match the existing web worker's half-size decode policy, not a new resize.
    scale = minimum = None
    if web_size is not None and web_half_decode(tasks, jpeg_layout, web_size):
        scale = (1, 2)
        minimum = ((4 if jpeg_layout == 'sbs' else 2)*web_size[0], 2*web_size[1])
    with (output / 'pixels.bin').open('wb') as pixels:
        for layer, folder, files in tasks:
            rows, names = [], []
            for path in files:
                stat = path.stat()
                encoded_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                original_bytes += stat.st_size
                if path.suffix.lower() in ('.png', '.bmp'):
                    with Image.open(path) as im:
                        image, sbs = np.asarray(im.convert('RGBA')), False
                else:
                    image, sbs = loader.read_image(
                        str(path), jpeg_scaling_factor=scale, jpeg_min_size=minimum)
                    if jpeg_layout == 'rgb' and path.suffix.lower() in ('.jpg', '.jpeg'):
                        sbs = False
                image = np.ascontiguousarray(image)
                after = path.stat()
                if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != (
                        after.st_size, after.st_mtime_ns, after.st_ino):
                    raise ValueError(f'Source changed while baking: {path}')
                if sbs and image.shape[1] % 2:
                    raise ValueError(f'Odd-width SBS source: {path}')
                digest = hashlib.sha256(memoryview(image)).digest()
                key = (image.shape, digest)
                raw_bytes += image.nbytes
                offset = hashes.get(key)
                if offset is None:
                    offset = pixels.tell()
                    pixels.write(memoryview(image))
                    hashes[key] = offset
                rows.append([offset, *image.shape, int(sbs)])
                names.append({'name': path.name, 'bytes': stat.st_size,
                              'sha256': encoded_hash})
            relative = folder.relative_to(source)
            dest = output / relative
            dest.mkdir(parents=True)
            np.save(dest / 'index.npy', np.asarray(rows, dtype=np.int64))
            (dest / 'sources.json').write_text(json.dumps(names, indent=2)+'\n')
            entries.append({'layer': layer, 'path': relative.as_posix(),
                            'source': relative.as_posix()})
            print(f'[LAYERS] {relative}: {len(rows)} independent frames', flush=True)
    manifest = dict(format='video-interleaving-layers', version=1,
                    frames=frames, folders=entries, jpeg_layout=jpeg_layout,
                    web_size=list(web_size) if scale else None,
                    source_bytes=original_bytes, decoded_bytes=raw_bytes,
                    unique_pixel_bytes=(output/'pixels.bin').stat().st_size)
    (output / 'layers.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def web_half_decode(tasks, jpeg_layout, size):
    from PIL import Image
    first = tasks[0][2][0]
    with Image.open(first) as image:
        w, h = image.size
    if jpeg_layout == 'sbs' and first.suffix.lower() in ('.jpg', '.jpeg'):
        w //= 2
    return w >= 2*size[0] and h >= 2*size[1]


def estimate(source, modes, jpeg_layout='sbs', web_size=None, thumb_width=128):
    """Header-based planning only; duplicate elimination is measured during bake."""
    from PIL import Image
    if web_size is None:
        import settings
        web_size = settings.HEADLESS_RES
    frames, tasks = discover(Path(source).resolve())
    source_bytes = sum(p.stat().st_size for _,_,files in tasks for p in files)
    display_bytes = web_bytes = scope_bytes = 0
    half_web = 'web' in modes and web_half_decode(tasks, jpeg_layout, web_size)
    for _,_,files in tasks:
        with Image.open(files[0]) as image:
            w,h = image.size
        suffix = files[0].suffix.lower()
        channels = 3 if suffix in ('.jpg','.jpeg') else 4
        display_bytes += len(files)*w*h*channels
        if half_web and suffix in ('.jpg','.jpeg'):
            minimum = ((4 if jpeg_layout=='sbs' else 2)*web_size[0],2*web_size[1])
            if w>=minimum[0] and h>=minimum[1]:
                w,h = (w+1)//2,(h+1)//2
        web_bytes += len(files)*w*h*channels
        with Image.open(files[0]) as image:
            w,h = image.size
        if jpeg_layout=='sbs' and suffix in ('.jpg','.jpeg'):
            w //= 2
        scope_bytes += len(files)*thumb_width*max(1,round(h*thumb_width/w))*2
    estimates = {}
    for mode in modes:
        if mode in ('local','ascii','asciiweb'):
            estimates[mode] = display_bytes
        elif mode=='web':
            estimates[mode] = web_bytes
        elif mode=='scope':
            estimates[mode] = scope_bytes
        elif mode=='modem':
            estimates[mode] = frames*len(tasks)*80*96*4
    print(f'[PLAN] {frames} frames x {len(tasks)} separate folders; source {source_bytes:,} bytes')
    for mode,size in estimates.items():
        print(f'[PLAN] {mode}: {size:,} payload bytes before deduplication/metadata'
              + ('; scope geometry/stipple tables additional' if mode=='scope' else ''))
    print('[PLAN] Estimates assume constant dimensions within each folder; display stores are shared.')
    return dict(source_bytes=source_bytes, payload_estimates=estimates)


def bake(source, output, modes=MODES, *, jpeg_layout='sbs', workers=2,
         thumb_width=128, thumbs_only=False, web_size=None):
    if web_size is None:
        import settings
        web_size = settings.HEADLESS_RES
    source, output = Path(source).resolve(), Path(output).resolve()
    modes = tuple(dict.fromkeys(modes))
    if not modes or any(mode not in MODES for mode in modes):
        raise ValueError('Choose one or more supported modes')
    if not source.is_dir():
        raise ValueError(f'No source directory: {source}')
    if output.exists() or output == source or source in output.parents:
        raise ValueError('Use a new output directory outside the source tree')
    if (jpeg_layout not in ('sbs', 'rgb') or workers < 1 or thumb_width < 16
            or len(web_size) != 2 or min(web_size) <= 0):
        raise ValueError('Invalid layout, workers, thumbnail width or web size')
    frames, tasks = discover(source)
    source_bytes = sum(p.stat().st_size for _,_,files in tasks for p in files)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.unified-bake-', dir=output.parent))
    outputs, measurements = {}, {}
    try:
        half_web = 'web' in modes and web_half_decode(tasks, jpeg_layout, web_size)
        if set(modes) & {'local', 'ascii', 'asciiweb', 'web'}:
            full_modes = set(modes) & {'local', 'ascii', 'asciiweb'}
            if 'web' in modes and not half_web:
                full_modes.add('web')
        else:
            full_modes = set()
        if full_modes:
            measurements['layers'] = bake_layers(source, stage/'layers', tasks,
                                                  frames, jpeg_layout)
            for mode in modes:
                if mode in full_modes:
                    outputs[mode] = 'layers'
        if half_web:
            measurements['web'] = bake_layers(source, stage/'web', tasks, frames,
                                               jpeg_layout, web_size)
            outputs['web'] = 'web'
        if 'scope' in modes:
            from utilities import convert_to_xy as xy
            profile = dict(xy.PROFILES['tiny'], thumb_width=thumb_width,
                           thumbs_only=thumbs_only, jpeg_layout=jpeg_layout)
            jobs = [(folder, stage/'scope'/folder.relative_to(source), profile)
                    for _,folder,_ in tasks]
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for result in pool.map(xy.process_folder, jobs):
                    if result:
                        raise ValueError(result)
            outputs['scope'] = 'scope'
        if 'modem' in modes:
            from utilities.convert_to_modem_dct import bake_tree_dct
            bake_tree_dct(source, stage/'modem', jpeg_layout)
            outputs['modem'] = 'modem'
        sizes = {name: sum(p.stat().st_size for p in (stage/path).rglob('*') if p.is_file())
                 for name,path in outputs.items()}
        manifest = dict(format='video-interleaving-bakes', version=1,
                        source=str(source), frames=frames, outputs=outputs,
                        jpeg_layout=jpeg_layout, independent_layers=True,
                        source_bytes=source_bytes, output_bytes=sizes)
        (stage/'bake.json').write_text(json.dumps(manifest, indent=2)+'\n')
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(f'[BAKE] Selected source images: {source_bytes:,} bytes')
    for mode,size in sizes.items():
        print(f'[BAKE] {mode}: {size:,} bytes ({size/source_bytes:.2f}x source image storage)')
    print('[BAKE] Display modes share a store unless web needs its existing half-decode level.')
    return manifest


def main():
    import settings
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', '--input-dir', required=True, type=Path)
    parser.add_argument('-o', '--output-dir', type=Path)
    parser.add_argument('--modes', nargs='+', choices=(*MODES, 'all'), default=['all'])
    parser.add_argument('--jpeg-layout', choices=('sbs', 'rgb'), default='sbs')
    parser.add_argument('--workers', type=int, default=min(2, os.cpu_count() or 1))
    parser.add_argument('--thumb-width', type=int, default=128)
    parser.add_argument('--thumbs-only', action='store_true', help='Skip scope vector geometry')
    parser.add_argument('--estimate', action='store_true', help='Report storage estimates without writing a bake')
    parser.add_argument('--web-size', default='x'.join(map(str, settings.HEADLESS_RES)), metavar='WxH')
    args = parser.parse_args()
    try:
        size = tuple(map(int, args.web_size.lower().split('x')))
        modes = MODES if 'all' in args.modes else args.modes
        if len(size)!=2 or min(size)<=0 or args.thumb_width<16:
            raise ValueError('Positive WxH and thumbnail width >= 16 required')
        if args.estimate:
            estimate(args.input_dir, modes, args.jpeg_layout, size, args.thumb_width)
            return
        bake(args.input_dir, args.output_dir or args.input_dir.with_name(args.input_dir.name+'_bakes'),
             modes, jpeg_layout=args.jpeg_layout,
             workers=args.workers, thumb_width=args.thumb_width,
             thumbs_only=args.thumbs_only, web_size=size)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
