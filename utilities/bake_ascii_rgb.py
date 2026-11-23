"""Compact, compressed independent RGB samples for main.py --mode asciiweb."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from ascii_rgb_layers import FORMAT, RgbLayerSampler, create_bake_context, close_bake_context
from utilities.bake import discover


def pack_sample(image, level=6):
    """Lossless 8-bit levels when every value is one; otherwise half or float."""
    levels = np.round(image * 255)
    if (np.all(np.abs(image * 255 - levels) < 1e-3) and np.all(levels >= 0)
            and np.all(levels <= 255)):
        compact = levels.astype(np.uint8)
    else:
        half = image.astype('<f2')
        compact = half if np.array_equal(half.astype(np.float32), image) else image.astype('<f4')
    scalar = compact.dtype.itemsize
    shuffled = compact.view(np.uint8).reshape(-1, scalar).T.copy().tobytes()
    return zlib.compress(shuffled, level), scalar, zlib.crc32(shuffled), hashlib.sha256(shuffled).digest()


def bake(source, output, *, size=None, font_ratio=.5, mirror=False,
         jpeg_layout='sbs', compression=6, tasks=None, frames=None):
    """Stream one source image at a time; atomically publish a new package."""
    if size is None:
        import settings
        size = settings.ASCII_WEB_MAX_SIZE
    source, output = Path(source).resolve(), Path(output).resolve()
    if (len(size) != 2 or any(type(n) is not int or not 1 <= n <= 160 for n in size)
            or not math.isfinite(font_ratio) or font_ratio <= 0
            or jpeg_layout not in ('sbs', 'rgb')
            or type(compression) is not int or not 0 <= compression <= 9):
        raise ValueError('Invalid grid, font ratio, JPEG layout or compression level')
    if not source.is_dir() or output.exists() or source == output or source in output.parents:
        raise ValueError('Use an existing source and a NEW output outside its tree')
    if tasks is None:
        frames, tasks = discover(source)
    if (type(frames) is not int or frames <= 0 or not tasks
            or {layer for layer, _, _ in tasks} != {'main', 'float'}
            or any(len(files) != frames for _, _, files in tasks)):
        raise ValueError('All selected independent folders need the same frame count')
    from image_loader import ImageLoader
    from PIL import Image
    loader = ImageLoader()

    def read(path):
        if path.suffix.lower() in ('.png', '.bmp'):
            with Image.open(path) as image:
                return np.asarray(image.convert('RGBA')), False
        image, sbs = loader.read_image(str(path))
        if jpeg_layout == 'rgb' and path.suffix.lower() in ('.jpg', '.jpeg'):
            sbs = False
        return image, sbs

    first, sbs = read(tasks[0][2][0])
    source_size = (first.shape[1] // (2 if sbs else 1), first.shape[0])
    del first
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.rgb-web-bake-', dir=output.parent))
    window = sampler = None
    source_bytes = sampled_bytes = unique_bytes = 0
    try:
        window = create_bake_context()
        sampler = RgbLayerSampler(window.ctx, size, source_size, font_ratio, mirror)
        entries, hashes, encoded = [], {}, {}
        with (stage / 'samples.bin').open('wb') as stream:
            for layer, folder, files in tasks:
                relative = folder.relative_to(source)
                dest = stage / relative
                dest.mkdir(parents=True)
                table = np.empty((frames, 4), dtype='<u8')
                # Stream metadata too, rather than accumulate source data/hashes.
                with (dest / 'sources.jsonl').open('w') as metadata:
                    for index, path in enumerate(files):
                        before = path.stat()
                        encoded_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                        encoded_key = (path.suffix.lower(), encoded_hash)
                        row = encoded.get(encoded_key)
                        if row is None:
                            image, sbs = read(path)
                            if sbs and image.shape[1] % 2:
                                raise ValueError(f'Odd-width SBS source: {path}')
                            sampled = sampler.sample(image, sbs)
                            block, scalar, checksum, digest = pack_sample(sampled, compression)
                            key = (scalar, digest)
                            row = hashes.get(key)
                            if row is None:
                                row = (stream.tell(), len(block), scalar, checksum)
                                stream.write(block)
                                hashes[key] = row
                                unique_bytes += sampled.size * scalar
                            encoded[encoded_key] = row
                        after = path.stat()
                        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                                after.st_size, after.st_mtime_ns, after.st_ino):
                            raise ValueError(f'Source changed while baking: {path}')
                        source_bytes += before.st_size
                        sampled_bytes += size[0] * size[1] * 16
                        table[index] = row
                        metadata.write(json.dumps(dict(name=path.name, bytes=before.st_size,
                                                       sha256=encoded_hash)) + '\n')
                np.save(dest / 'index.npy', table, allow_pickle=False)
                entries.append(dict(layer=layer, path=relative.as_posix(), source=relative.as_posix()))
                print(f'[RGB WEB] {relative}: {frames} independent frames', flush=True)
        info = dict(format=FORMAT, version=2, frames=frames, size=list(size),
                    source_size=list(source_size), font_ratio=font_ratio, mirror=bool(mirror),
                    precision=sampler.profile, jpeg_layout=jpeg_layout, folders=entries,
                    compression='zlib', compression_level=compression, layout='byte-shuffled-rgba',
                    source_bytes=source_bytes, sampled_float32_bytes=sampled_bytes,
                    unique_sample_bytes=unique_bytes, unique_blocks=len(hashes),
                    compressed_bytes=(stage / 'samples.bin').stat().st_size)
        (stage / 'rgb_layers.json').write_text(json.dumps(info, indent=2) + '\n')
        total = sum(p.stat().st_size for p in stage.rglob('*') if p.is_file())
        manifest = dict(format='video-interleaving-bakes', version=1, frames=frames,
                        source=str(source), independent_layers=True,
                        outputs={'asciiweb': '.'}, source_bytes=source_bytes,
                        output_bytes={'asciiweb': total})
        # A contained subdirectory matches the unified package path contract.
        payload = stage / 'rgb'
        payload.mkdir()
        for path in list(stage.iterdir()):
            if path != payload:
                path.rename(payload / path.name)
        manifest['outputs']['asciiweb'] = 'rgb'
        (stage / 'bake.json').write_text(json.dumps(manifest, indent=2) + '\n')
        if output.exists():
            raise ValueError('Output appeared while baking; refusing to overwrite it')
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        if sampler is not None:
            sampler.close()
        if window is not None:
            close_bake_context(window)
    print(f'[RGB WEB] {source_bytes:,} source bytes -> {total:,} bake bytes; '
          f'{info["unique_blocks"]:,} unique blocks; lossless compression', flush=True)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', '--input-dir', required=True, type=Path)
    parser.add_argument('-o', '--output-dir', required=True, type=Path)
    import settings
    parser.add_argument('--size', default='x'.join(map(str, settings.ASCII_WEB_MAX_SIZE)), metavar='WxH')
    parser.add_argument('--font-ratio', type=float, default=.5)
    parser.add_argument('--mirror', action='store_true')
    parser.add_argument('--jpeg-layout', choices=('sbs', 'rgb'), default='sbs')
    parser.add_argument('--compression', type=int, choices=range(10), default=6)
    args = parser.parse_args()
    try:
        bake(args.input_dir, args.output_dir, size=tuple(map(int, args.size.lower().split('x'))),
             font_ratio=args.font_ratio, mirror=args.mirror, jpeg_layout=args.jpeg_layout,
             compression=args.compression)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
