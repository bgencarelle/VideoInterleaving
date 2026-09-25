"""Runtime image source for the V7 modem.

This is deliberately only a source adapter.  It reuses the application's
folder lists, image loader, and sliding FIFO, while the modem keeps its own
V7 timing and audio output path.
"""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
import threading

import numpy as np
from PIL import Image


class RuntimeImageLibrary:
    """Load normal face/float images ahead of the modem's target index."""

    def __init__(self, *, rebuild=False, capacity=5):
        import settings
        from image_loader import FIFOImageBuffer, ImageLoader
        import make_file_lists

        self.root = Path(settings.IMAGES_DIR).resolve()
        if rebuild or not self._has_lists(settings.GENERATED_LISTS_DIR):
            make_file_lists.process_files()

        main_csv, float_csv = make_file_lists.select_image_list_files()
        self.main_paths, _ = make_file_lists.load_image_paths_from_csv(main_csv)
        self.float_paths, _ = make_file_lists.load_image_paths_from_csv(float_csv)
        repo = Path(__file__).resolve().parent
        self.main_paths = [[self._absolute(path, repo) for path in row]
                           for row in self.main_paths]
        self.float_paths = [[self._absolute(path, repo) for path in row]
                            for row in self.float_paths]
        if not self.main_paths or not self.float_paths:
            raise ValueError('Normal modem source has no face/float image frames')
        if len(self.main_paths) != len(self.float_paths):
            raise ValueError('Normal modem source face/float frame counts differ')
        if not self.main_paths[0] or not self.float_paths[0]:
            raise ValueError('Normal modem source has an empty folder list')

        self.frames = len(self.main_paths)
        self.mains = list(range(len(self.main_paths[0])))
        self.floats = list(range(len(self.float_paths[0])))
        self.loader = ImageLoader()
        self.loader.set_paths(self.main_paths, self.float_paths)

        self.capacity = max(1, int(capacity))
        self._buffers = {}
        self._pending = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=2,
                                        thread_name_prefix='modem-images')
        self.prefetches = 0
        self.fifo_hits = 0
        self.index_misses = 0

    @staticmethod
    def _has_lists(path):
        root = Path(path)
        return root.is_dir() and any(root.glob('*.csv'))

    @staticmethod
    def _absolute(path, repo):
        path = Path(path)
        return str(path if path.is_absolute() else (repo / path).resolve())

    def _buffer(self, folders):
        from image_loader import FIFOImageBuffer
        key = tuple(int(v) for v in folders)
        with self._lock:
            return self._buffers.setdefault(
                key, FIFOImageBuffer(max_size=self.capacity))

    def _path(self, index, folder, layer):
        paths = self.main_paths if layer == 'main' else self.float_paths
        try:
            return paths[index][folder]
        except IndexError as exc:
            raise ValueError(
                f'Normal modem source index/folder is outside the lists: '
                f'{index},{folder}') from exc

    def _read(self, path):
        """Use the normal accelerated loader, with Pillow for PNG sources."""
        try:
            return self.loader.read_image(path)
        except ValueError as exc:
            if Path(path).suffix.lower() != '.png':
                raise
            with Image.open(path) as image:
                rgba = image.convert('RGBA')
                return np.asarray(rgba), False

    def _load(self, index, folders):
        main_folder, float_folder = folders
        main = self._read(self._path(index, main_folder, 'main'))
        front = self._read(self._path(index, float_folder, 'float'))
        return main[0], front[0], main[1], front[1]

    def prefetch(self, index, main_folder, float_folder):
        """Request one source index without making it timing-authoritative."""
        index = int(index) % self.frames
        folders = (int(main_folder), int(float_folder))
        fifo = self._buffer(folders)
        hit = fifo.get(index)
        if hit is not None and hit[0] == index:
            return
        key = (index, *folders)
        with self._lock:
            if key in self._pending:
                return
            future = self._pool.submit(self._load, index, folders)
            self._pending[key] = future
            self.prefetches += 1

        def complete(done):
            try:
                fifo.update(index, done.result())
            except Exception:
                # The foreground lookup will surface the real loading error.
                pass
            finally:
                with self._lock:
                    self._pending.pop(key, None)

        future.add_done_callback(complete)

    def _layers_for(self, index, folders):
        fifo = self._buffer(folders)
        hit = fifo.get(index)
        if hit is not None:
            if hit[0] == index:
                self.fifo_hits += 1
                return hit[1:]
            # FIFOImageBuffer is intentionally tolerant for display playback;
            # modem transmission is not. Never encode a nearby source index.
            self.index_misses += 1

        key = (int(index), *folders)
        with self._lock:
            future = self._pending.get(key)
        if future is not None:
            data = future.result()
        else:
            data = self._load(index, folders)
        return data

    @staticmethod
    def _rgba(image, sbs):
        if isinstance(image, dict):
            raise ValueError('ASCII pre-baked data cannot be a modem source')
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError(f'Unsupported modem source image shape: {array.shape}')
        if sbs:
            width = array.shape[1]
            if width % 2:
                raise ValueError('SBS image width must be even')
            half = width // 2
            rgb = array[:, :half, :3]
            alpha = array[:, half:, 0]
            return Image.fromarray(np.dstack((rgb, alpha)).astype(np.uint8), 'RGBA')
        if array.shape[2] == 4:
            return Image.fromarray(array.astype(np.uint8), 'RGBA')
        return Image.fromarray(array[:, :, :3].astype(np.uint8), 'RGB').convert('RGBA')

    def composite(self, index, main_folder, float_folder,
                  background=(4, 4, 4), rotation=0, mirror=False):
        layers = self._layers_for(int(index) % self.frames,
                                  (int(main_folder), int(float_folder)))
        main, front, main_sbs, front_sbs = layers
        main = self._rgba(main, main_sbs)
        front = self._rgba(front, front_sbs)

        if main.size == front.size:
            size = main.size
        else:
            size = main.size
            front = front.crop((0, 0, min(size[0], front.width),
                                min(size[1], front.height)))
        image = Image.new('RGBA', size, tuple(background) + (255,))
        image.alpha_composite(main)
        image.alpha_composite(front)
        image = image.convert('RGB')
        if rotation % 360:
            image = image.rotate(rotation % 360, expand=True)
        if mirror:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image

    def composite_nearest(self, index, main_folder, float_folder, size,
                          background=(4, 4, 4), rotation=0, mirror=False):
        """``composite(...).resize(size, NEAREST)``, compositing only the
        pixels that nearest-neighbour sampling reads.

        Compositing is per pixel and nearest sampling picks whole source
        pixels, so the result is bit-identical to the full-size path, but a
        1080x1920 source composites 7,680 pixels instead of about two million
        (28 ms -> well under 1 ms). ``info['source_dimensions']`` carries the
        full composite size, which the full path reports as ``image.size``.
        """
        layers = self._layers_for(int(index) % self.frames,
                                  (int(main_folder), int(float_folder)))
        main, front, main_sbs, front_sbs = layers
        main, front = self._layer_array(main), self._layer_array(front)
        height, width = main.shape[0], self._layer_width(main, main_sbs)
        picked, full_size = _nearest_samples(width, height, int(rotation) % 360,
                                             bool(mirror), tuple(size))
        filled = picked < 0
        rows, cols = np.divmod(np.where(filled, 0, picked), width)
        main_pixels = self._sampled_rgba(main, main_sbs, rows, cols)
        front_height, front_width = (front.shape[0],
                                     self._layer_width(front, front_sbs))
        inside = (rows < front_height) & (cols < front_width)
        front_pixels = self._sampled_rgba(
            front, front_sbs, np.where(inside, rows, 0), np.where(inside, cols, 0))
        front_pixels[~inside] = 0                  # outside the cropped float
        image = Image.new('RGBA', tuple(size), tuple(background) + (255,))
        image.alpha_composite(Image.fromarray(main_pixels, 'RGBA'))
        image.alpha_composite(Image.fromarray(front_pixels, 'RGBA'))
        rgb = np.array(image.convert('RGB'))
        rgb[filled] = 0                            # rotate()'s black fill
        image = Image.fromarray(rgb, 'RGB')
        image.info['source_dimensions'] = full_size
        return image

    @staticmethod
    def _layer_array(image):
        if isinstance(image, dict):
            raise ValueError('ASCII pre-baked data cannot be a modem source')
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError(f'Unsupported modem source image shape: {array.shape}')
        return array

    @staticmethod
    def _layer_width(array, sbs):
        if not sbs:
            return array.shape[1]
        if array.shape[1] % 2:
            raise ValueError('SBS image width must be even')
        return array.shape[1] // 2

    @staticmethod
    def _sampled_rgba(array, sbs, rows, cols):
        """The pixels ``_rgba`` would hold at (rows, cols), as uint8 RGBA."""
        out = np.empty(rows.shape + (4,), np.uint8)
        if sbs:
            half = array.shape[1] // 2
            out[..., :3] = array[rows, cols, :3].astype(np.uint8)
            out[..., 3] = array[rows, cols + half, 0].astype(np.uint8)
        elif array.shape[2] == 4:
            out[...] = array[rows, cols].astype(np.uint8)
        else:
            out[..., :3] = array[rows, cols, :3].astype(np.uint8)
            out[..., 3] = 255
        return out

    def close(self):
        self._pool.shutdown(wait=True, cancel_futures=True)


@lru_cache(maxsize=16)
def _nearest_samples(width, height, rotation, mirror, size):
    """Source pixel (flat index, -1 for rotate's fill) behind every pixel of
    ``composite(...).resize(size, NEAREST)``, found by sending an index image
    through the same rotate, mirror and resize, and the full composite size.
    """
    index = np.arange(1, width*height + 1, dtype=np.int32).reshape(height, width)
    image = Image.fromarray(index)                         # mode 'I'
    if rotation:
        image = image.rotate(rotation, expand=True)
    if mirror:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    full_size = image.size
    picked = np.asarray(image.resize(size, Image.Resampling.NEAREST),
                        np.int64) - 1
    picked.setflags(write=False)
    return picked, full_size
