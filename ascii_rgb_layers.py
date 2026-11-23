"""Losslessly compressed, independently sampled RGB web layers.

Only source sampling is baked. Alpha composition, grading and glyph selection
remain live, so arbitrary face/float selections never require pre-mixed frames.
"""
from collections import OrderedDict
import json
import mmap
from pathlib import Path
import threading
import zlib

import numpy as np

from baked_layers import LayerFrame, _Paths


FORMAT = 'video-interleaving-rgb-web-layers'


def baked_web_source_size(root):
    """Probe only the compact bake header before advertising WS dimensions."""
    path = Path(root) / 'rgb_layers.json'
    if not path.is_file():
        return None
    info = json.loads(path.read_text())
    size = info.get('size')
    if (info.get('format') != FORMAT or info.get('version') not in (1, 2)
            or not isinstance(size, list) or len(size) != 2
            or any(type(n) is not int for n in size)
            or not 1 <= size[0] <= 160 or not 1 <= size[1] <= 100):
        raise ValueError('Invalid RGB web bake grid')
    return tuple(size)


class RgbWebLayers:
    rgb_samples = True
    web_size = None

    def __init__(self, root, *, cache_bytes=4 * 1024 * 1024):
        self.root = Path(root).resolve()
        self.manifest = info = json.loads((self.root / 'rgb_layers.json').read_text())
        if info.get('format') != FORMAT or info.get('version') not in (1, 2):
            raise ValueError('Unsupported RGB web bake format')
        self.frames = info['frames']
        self.size = tuple(info['size'])
        if (type(self.frames) is not int or self.frames <= 0 or len(self.size) != 2
                or any(type(n) is not int or not 1 <= n <= 160 for n in self.size)
                or info.get('compression') != 'zlib' or info.get('layout') != 'byte-shuffled-rgba'):
            raise ValueError('Invalid RGB web bake dimensions/codec')
        if type(cache_bytes) is not int or cache_bytes < 0:
            raise ValueError('Invalid RGB cache budget')
        self.cache_budget = cache_bytes
        self.cache_bytes = 0
        self._cache = OrderedDict()
        self._lock = threading.Lock()
        self._closed = False
        self.tables = {'main': [], 'float': []}
        self.folders = {'main': [], 'float': []}
        data = self.root / 'samples.bin'
        size = data.stat().st_size
        seen = set()
        for entry in info['folders']:
            layer = entry['layer']
            folder = (self.root / entry['path']).resolve()
            if layer not in self.tables or self.root not in folder.parents or folder in seen:
                raise ValueError('Invalid/duplicate RGB web layer path')
            seen.add(folder)
            table = np.load(folder / 'index.npy', mmap_mode='r', allow_pickle=False)
            # compressed offset, compressed length, scalar bytes (2/4), CRC32
            if table.dtype != np.dtype('<u8') or table.shape != (self.frames, 4):
                raise ValueError('Invalid RGB web index')
            for offset, length, scalar, checksum in table:
                if (int(length) <= 0 or int(offset) + int(length) > size
                        or scalar not in (1, 2, 4) or checksum > 0xffffffff
                        or int(length) > self.size[0] * self.size[1] * 16 + 1024):
                    raise ValueError('Truncated/invalid RGB web sample block')
            self.tables[layer].append(table)
            self.folders[layer].append(entry['source'])
        if not all(self.tables.values()):
            raise ValueError('RGB web bake needs both face and float layers')
        self.main_paths, self.float_paths = _Paths(self, 'main'), _Paths(self, 'float')
        self._data = data.open('rb')

    def validate_settings(self):
        import settings
        if (tuple(getattr(settings, 'ASCII_WEB_SOURCE_SIZE', ()) or ()) != self.size
                or settings.ASCII_FONT_RATIO != self.manifest['font_ratio']
                or bool(settings.INITIAL_MIRROR) != self.manifest['mirror']):
            raise ValueError('RGB web bake grid/font ratio/mirror differs; rebuild for these settings')

    def read_image(self, reference, **_kwargs):
        if not isinstance(reference, LayerFrame):
            raise ValueError('Expected an RGB baked layer reference')
        with self._lock:
            if self._closed:
                raise ValueError('RGB web bake is closed')
            if (reference.layer not in self.tables or
                    not 0 <= reference.folder < len(self.tables[reference.layer]) or
                    not 0 <= reference.index < self.frames):
                raise ValueError('RGB baked index/folder outside manifest')
            row = self.tables[reference.layer][reference.folder][reference.index]
            key = offset, length, scalar, checksum = tuple(map(int, row))
            image = self._cache.get(key)
            if image is not None:
                self._cache.move_to_end(key)
                return image, False
            expected = self.size[0] * self.size[1] * 4 * scalar
            aligned = offset - offset % mmap.ALLOCATIONGRANULARITY
            within = offset - aligned
            # Map only one compressed block. Never retain a movie-sized mapping
            # or pretend compressed bytes are directly usable pixel arrays.
            with mmap.mmap(self._data.fileno(), within + length,
                           access=mmap.ACCESS_READ, offset=aligned) as mapping:
                decoder = zlib.decompressobj()
                raw = decoder.decompress(mapping[within:within + length], expected + 1)
            if (len(raw) != expected or not decoder.eof or decoder.unused_data
                    or decoder.unconsumed_tail or zlib.crc32(raw) != checksum):
                raise ValueError('Corrupt RGB web sample block')
            # Reverse the lossless byte shuffle, then retain immutable ownership.
            packed = np.frombuffer(raw, np.uint8).reshape(scalar, -1).T.copy().tobytes()
            if scalar == 1:
                # 8-bit levels: exact multiples of 1/255 as float32, nothing else.
                image = (np.frombuffer(packed, np.uint8).reshape(
                    self.size[1], self.size[0], 4).astype(np.float32) / np.float32(255))
                image.flags.writeable = False
            else:
                image = np.frombuffer(packed, dtype=f'<f{scalar}').reshape(
                    self.size[1], self.size[0], 4)
            if not np.isfinite(image).all() or np.any(image < 0) or np.any(image > 1):
                raise ValueError('Invalid RGB web sample values')
            if image.nbytes <= self.cache_budget:
                while self._cache and self.cache_bytes + image.nbytes > self.cache_budget:
                    _, old = self._cache.popitem(last=False)
                    self.cache_bytes -= old.nbytes
                self._cache[key] = image
                self.cache_bytes += image.nbytes
            return image, False

    def load_images(self, index, main_folder, float_folder, **_kwargs):
        main, _ = self.read_image(LayerFrame('main', main_folder, index))
        front, _ = self.read_image(LayerFrame('float', float_folder, index))
        return main, front, False, False

    def set_paths(self, *_args):
        pass

    def set_png_paths_len(self, count):
        if count != self.frames:
            raise ValueError('RGB baked frame count changed')

    def close(self):
        with self._lock:
            if not self._closed:
                self._cache.clear()
                self.cache_bytes = 0
                self.tables.clear()
                self._data.close()
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def upload_sample(context, image, texture=None):
    """Upload a tiny baked float grid without converting/clipping it to uint8."""
    import moderngl
    dtype = 'f2' if image.dtype == np.float16 else 'f4'
    size = (image.shape[1], image.shape[0])
    if texture is not None and (texture.size != size or texture.dtype != dtype):
        texture.release()
        texture = None
    if texture is None:
        texture = context.texture(size, 4, dtype=dtype)
        texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        texture.repeat_x = texture.repeat_y = False
    texture.write(memoryview(image))
    return texture


class RgbLayerSampler:
    """Offline GL sampling with the live renderer's geometry and SBS threshold.

    Retain floating point samples BEFORE composition/8-bit framebuffer rounding.
    Live playback uses the same small GL mix, avoiding CPU/driver rounding drift.
    """
    def __init__(self, context, size, source_size, font_ratio, mirror=False):
        import moderngl
        self.context, self.size = context, tuple(size)
        self.profile = 'gles-mediump' if context.version_code < 330 else 'desktop'
        header = ('#version 300 es\nprecision mediump float;' if self.profile == 'gles-mediump'
                  else '#version 330 core')
        self.program = context.program(vertex_shader=f'''
            {header}
            in vec2 position; in vec2 texcoord; out vec2 v_texcoord;
            void main() {{ gl_Position = vec4(position,0.0,1.0); v_texcoord = texcoord; }}
        ''', fragment_shader=f'''
            {header}
            uniform sampler2D source; uniform bool is_sbs;
            in vec2 v_texcoord; out vec4 fragColor;
            void main() {{
                if (is_sbs) {{
                    vec2 uv_color = vec2(v_texcoord.x * 0.5, v_texcoord.y);
                    vec2 uv_mask = vec2(v_texcoord.x * 0.5 + 0.5, v_texcoord.y);
                    vec3 color = texture(source, uv_color).rgb;
                    float alpha = texture(source, uv_mask).r;
                    if (alpha < 0.05) alpha = 0.0;
                    fragColor = vec4(color, alpha);
                }} else {{ fragColor = texture(source, v_texcoord); }}
            }}
        ''')
        aspect = source_size[0] / source_size[1]
        target = size[0] * font_ratio / size[1]
        x, y = (1.0, target / aspect) if aspect > target else (aspect / target, 1.0)
        left, right = (1., 0.) if mirror else (0., 1.)
        quad = np.asarray([[-x,y,left,1], [x,y,right,1],
                           [x,-y,right,0], [-x,-y,left,0]], np.float32)
        self.buffer = context.buffer(quad.tobytes())
        self.vao = context.vertex_array(self.program,
                                       [(self.buffer, '2f 2f', 'position', 'texcoord')])
        self.target = context.simple_framebuffer(size, components=4, dtype='f4')
        self.texture = None
        self.program['source'].value = 0
        context.disable(moderngl.BLEND)

    def sample(self, image, sbs):
        import moderngl
        image = np.ascontiguousarray(image, dtype=np.uint8)
        shape = (image.shape[1], image.shape[0])
        if (self.texture is None or self.texture.size != shape
                or self.texture.components != image.shape[2]):
            if self.texture is not None:
                self.texture.release()
            self.texture = self.context.texture(shape, image.shape[2])
            self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.texture.write(memoryview(image))
        self.texture.use(0)
        self.program['is_sbs'].value = sbs
        self.target.use()
        self.context.viewport = (0, 0, *self.size)
        self.target.clear(0, 0, 0, 0)
        self.vao.render(moderngl.TRIANGLE_FAN)
        return np.frombuffer(self.target.read(components=4, dtype='f4', alignment=1),
                             np.float32).reshape(self.size[1], self.size[0], 4)

    def close(self):
        if self.texture is not None:
            self.texture.release()
        for item in (self.target, self.vao, self.buffer, self.program):
            item.release()


def create_bake_context():
    """Use the same headless context selection as live ASCII, without app logs."""
    # Use the app's tested context creation rather than a new backend route.
    import settings
    from display_manager import DisplayState, display_init
    previous = settings.ASCII_MODE, settings.SERVER_MODE
    settings.ASCII_MODE, settings.SERVER_MODE = True, False
    try:
        window = display_init(DisplayState(), log_backend=False)
    finally:
        settings.ASCII_MODE, settings.SERVER_MODE = previous
    if window is None or not hasattr(window, 'ctx'):
        if window is not None:
            window.close()
        raise RuntimeError('RGB web baking requires a ModernGL headless context')
    return window


def close_bake_context(window):
    """Release owned GL objects before destroying the offline context."""
    import renderer
    if renderer.ctx is window.ctx:
        for name in ('vao', 'vbo', 'prog'):
            obj = getattr(renderer, name)
            if obj is not None:
                obj.release()
                setattr(renderer, name, None)
        renderer.ctx = None
    for attachment in window.fbo.color_attachments:
        attachment.release()
    window.fbo.release()
    window.ctx.release()
    window.close()
