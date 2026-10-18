"""Read-only, independently addressable face/float layers. No compositing.

One packed pixel file avoids per-frame opens. Index tables retain dimensions,
SBS interpretation and exact frame identities; duplicate pixels share storage.
"""
from collections import OrderedDict
from dataclasses import dataclass
import json
import mmap
from pathlib import Path
import threading

import numpy as np


def package_output(root, mode):
    """Resolve a unified package without importing any mode's dependencies."""
    root = Path(root).resolve()
    manifest = json.loads((root / 'bake.json').read_text())
    if manifest.get('format') != 'video-interleaving-bakes' or manifest.get('version') != 1:
        raise ValueError('Unsupported unified bake format')
    relative = manifest.get('outputs', {}).get(mode)
    if not isinstance(relative, str):
        raise ValueError(f'Bake has no {mode} output')
    output = (root / relative).resolve()
    if root not in output.parents or not output.is_dir():
        raise ValueError(f'Invalid {mode} bake path')
    return output


@dataclass(frozen=True)
class LayerFrame:
    layer: str
    folder: int
    index: int


class _Paths:
    """Image-list compatible facade, without an N-frames x N-folders list."""
    def __init__(self, library, layer):
        self.library, self.layer = library, layer

    def __len__(self):
        return self.library.frames

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        return [LayerFrame(self.layer, folder, index)
                for folder in range(len(self.library.tables[self.layer]))]


class BakedLayers:
    def __init__(self, root):
        self.root = Path(root).resolve()
        manifest = json.loads((self.root / 'layers.json').read_text())
        if manifest.get('format') != 'video-interleaving-layers' or manifest.get('version') != 1:
            raise ValueError('Unsupported layer bake format')
        self.frames = manifest['frames']
        self.web_size = manifest.get('web_size')
        if type(self.frames) is not int or self.frames <= 0:
            raise ValueError('Invalid baked frame count')
        self.pixel_bytes = (self.root / 'pixels.bin').stat().st_size
        self.tables = {'main': [], 'float': []}
        self.folders = {'main': [], 'float': []}
        seen = set()
        for entry in manifest['folders']:
            layer = entry['layer']
            relative = entry['path']
            folder = (self.root / relative).resolve()
            if layer not in self.tables or self.root not in folder.parents or folder in seen:
                raise ValueError('Invalid/duplicate baked layer path')
            seen.add(folder)
            table = np.load(folder / 'index.npy', mmap_mode='r', allow_pickle=False)
            # offset, height, width, channels, SBS. Validate once, not per tick.
            if (table.dtype != np.int64 or table.shape != (self.frames, 5)
                    or np.any(table[:, :1] < 0) or np.any(table[:, 1:3] <= 0)
                    or np.any((table[:, 3] != 3) & (table[:, 3] != 4))
                    or np.any((table[:, 4] != 0) & (table[:, 4] != 1))
                    or np.any((table[:, 4] == 1) & (table[:, 2] % 2 != 0))):
                raise ValueError(f'Invalid layer index: {relative}')
            # Python integers avoid wrapping a corrupt table's size product.
            if any(int(o)+int(h)*int(w)*int(c) > self.pixel_bytes
                   for o,h,w,c,_ in table):
                raise ValueError(f'Truncated layer pixels: {relative}')
            self.tables[layer].append(table)
            self.folders[layer].append(entry['source'])
        if not all(self.tables.values()):
            raise ValueError('Bake needs both face and float layers')
        self.main_paths, self.float_paths = _Paths(self, 'main'), _Paths(self, 'float')
        self._cache = OrderedDict()
        self._lock = threading.Lock()
        self._pixels = (self.root / 'pixels.bin').open('rb')

    def read_image(self, reference, **_kwargs):
        if not isinstance(reference, LayerFrame):
            raise ValueError('Expected a baked layer reference')
        if (reference.layer not in self.tables or
                not 0 <= reference.folder < len(self.tables[reference.layer]) or
                not 0 <= reference.index < self.frames):
            raise ValueError('Baked index/folder outside the manifest')
        row = self.tables[reference.layer][reference.folder][reference.index]
        offset, height, width, channels, sbs = map(int, row)
        key = (offset, height, width, channels)
        with self._lock:
            image = self._cache.get(key)
            if image is None:
                # Map only this picture, not the entire movie. Released views
                # unmap their pages instead of accumulating a sequence-sized
                # resident set until the OS eventually reclaims it.
                aligned = offset-offset % mmap.ALLOCATIONGRANULARITY
                within = offset-aligned
                mapping = mmap.mmap(self._pixels.fileno(),
                                    within+height*width*channels,
                                    access=mmap.ACCESS_READ, offset=aligned)
                image = np.ndarray((height, width, channels), dtype=np.uint8,
                                   buffer=mapping, offset=within)
                self._cache[key] = image
                while len(self._cache) > 4:
                    self._cache.popitem(last=False)
            else:
                self._cache.move_to_end(key)
        return image, bool(sbs)

    def load_images(self, index, main_folder, float_folder, **_kwargs):
        main, ms = self.read_image(LayerFrame('main', main_folder, index))
        front, fs = self.read_image(LayerFrame('float', float_folder, index))
        return main, front, ms, fs

    def set_paths(self, *_args):
        pass  # Paths are immutable and owned by this library.

    def set_png_paths_len(self, count):
        if count != self.frames:
            raise ValueError('Baked frame count changed')

    def close(self):
        # Outstanding arrays own their mappings and remain valid until released.
        with self._lock:
            self._cache.clear()
            self.tables.clear()
            self._pixels.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
