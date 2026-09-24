"""Prebake registered RGBA layers at 80x96 for 2D-DCT modem transmission."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# Add root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utilities.bake_assets import write_slab
from utilities.convert_to_xy import load_rgba, _image_files
from make_file_lists import check_folder_prefix, natural_sort_key

# 2D-DCT target spatial shape (80x96 Luma canvas)
DCT_SIZE = (80, 96)


def bake_tree_dct(source, output, jpeg_layout='sbs'):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir():
        raise ValueError(f'Source directory does not exist: {source}')
    if output.exists():
        raise ValueError(f'Output already exists: {output}. Bake to a new directory.')
    if output == source or source in output.parents:
        raise ValueError('Bake output must be outside the source tree')
    if jpeg_layout not in ('sbs', 'rgb'):
        raise ValueError('Unsupported JPEG layout')

    tasks = []
    for kind, name in [('main', 'face'), ('float', 'float')]:
        tree = source / name
        if not tree.is_dir():
            raise ValueError(f'Missing layer directory: {tree}')
        for folder in sorted([tree, *tree.rglob('*')], key=lambda p: natural_sort_key(str(p))):
            if folder.is_dir() and check_folder_prefix(str(folder), kind):
                files = _image_files(folder)
                if files:
                    tasks.append((kind, folder, files))

    if not all(any(t[0] == kind for t in tasks) for kind in ['main', 'float']):
        raise ValueError('Need numeric face folders (0–254) and float folders (255_...)')

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.modem-bake-dct-', dir=output.parent))

    def dct_loader(path):
        """Loads and pre-enhances RGBA frames before Lanczos downsampling."""
        if jpeg_layout == 'rgb' and path.suffix.lower() in ('.jpg', '.jpeg'):
            with Image.open(path) as im:
                rgba = im.convert('RGBA')
        else:
            rgb, alpha = load_rgba(path)
            if path.suffix.lower() in ('.jpg', '.jpeg'):
                alpha = alpha.copy()
                alpha[alpha < 13] = 0
            rgba = Image.fromarray(np.dstack([rgb, alpha]))

        # High-pass unsharp mask to boost edge coefficients before DCT truncation
        #rgba = rgba.filter(ImageFilter.UnsharpMask(radius=2, percent=140, threshold=2))
        return rgba

    entries = []
    try:
        for kind, folder, files in tasks:
            relative = folder.relative_to(source)
            # Write 80x96 RGBA slabs with Lanczos resampling
            source_sizes = []

            def tracked_loader(path):
                rgba = dct_loader(path)
                source_sizes.append(list(rgba.size))
                return rgba

            write_slab(files, stage / relative / 'frames.npy', DCT_SIZE,
                       tracked_loader, Image.Resampling.LANCZOS)
            entries.append({
                'layer': kind,
                'path': relative.as_posix(),
                'count': len(files),
                'names': [p.name for p in files],
                # The slab geometry is fixed, but display aspect belongs to
                # the source frames and cannot be recovered from 80x96 later.
                'source_sizes': source_sizes,
            })
            print(f'[DCT BAKE] {relative}: {len(files)} frames, {DCT_SIZE[0]}x{DCT_SIZE[1]} RGBA')

        manifest = {
            'format': 'video-interleaving-modem',
            'version': 1,
            'profile': 'color-dct',
            'size': list(DCT_SIZE),
            'jpeg_layout': jpeg_layout,
            'folders': entries
        }
        (stage / 'modem.json').write_text(json.dumps(manifest, indent=2) + '\n')
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def main():
    p = argparse.ArgumentParser(description="Prebake registered RGBA layers into 80x96 2D-DCT slabs")
    p.add_argument('-i', '--input-dir', required=True, type=Path)
    p.add_argument('-o', '--output-dir', type=Path)
    p.add_argument('--jpeg-layout', choices=['sbs', 'rgb'], default='sbs',
                   help='Project JPEG convention: colour|matte (default); rgb for ordinary JPEGs')
    args = p.parse_args()
    out = args.output_dir or args.input_dir.with_name(args.input_dir.name + '_modem_dct')
    try:
        bake_tree_dct(args.input_dir, out, args.jpeg_layout)
    except (OSError, ValueError) as exc:
        p.error(str(exc))


if __name__ == '__main__':
    main()
