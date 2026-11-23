"""Fullscreen ASCII in the local window (``--mode asciilocal``).

The picture is the one ASCII mode makes: both layers are sampled on the GPU
into a master character grid, graded, and given characters by
``ascii_converter.grade_cells`` -- the same grading and glyph choice as the
terminal and the browser. Instead of becoming ANSI text it is drawn here as
glyphs in Interleave Schoolbook Mono, from one atlas texture with one
instanced draw, so a weak machine uploads a few kilobytes per picture.

Size control is the browser's (static/ascii_resolution.js): one row count,
mapped to a grid by ``dimensions()`` through the default size, and each shown
cell picks the nearest master cell. The row count is set by the mouse wheel
and keys, or by a random driver built like the folder selector. The local
master is ``ASCII_WEB_MAX_SIZE`` times ``LOCAL_ASCII_MULTIPLIER``, sampled on
this machine, so finer grids cost no other viewer anything.
"""
from collections import deque
import math
import random
import time

import numpy as np

import ascii_converter
import renderer
import settings

ATLAS_FONT_PX = 96        # glyphs are drawn once, large, and mipmapped down
ATLAS_COLUMNS = 8
ATLAS_GUTTER = 8          # empty pixels around each tile: small mip levels must not bleed


def _setting(name, default):
    return getattr(settings, name, default)


# --- Size mapping: a port of AsciiResolution.dimensions --------------------

def _js_round(value):
    """Math.round: halves go up (Python's round() sends them to even)."""
    return int(math.floor(value + 0.5))


def dimensions(rows, default, finest, coarsest=(15, 10)):
    """Grid (cols, rows) for a row count, as the browser computes it.

    Linear in two pieces through ``default``: ``coarsest`` at the fewest rows,
    ``default`` at its own row count, ``finest`` at the most.
    """
    min_cols, min_rows = coarsest
    default_cols, default_rows = default
    max_cols, max_rows = finest
    rows = _js_round(min(max(rows, min_rows), max_rows))
    default_rows = min(max(default_rows, min_rows), max_rows)
    default_cols = min(max(default_cols, min_cols), max_cols)
    if rows == default_rows:
        cols = default_cols
    elif rows < default_rows:
        cols = min_cols + (default_cols - min_cols) * (rows - min_rows) / max(default_rows - min_rows, 1)
    else:
        cols = default_cols + (max_cols - default_cols) * (rows - default_rows) / max(max_rows - default_rows, 1)
    return _js_round(cols), rows


class Zoom:
    """The row count, and what moves it: mouse/keys or the random driver.

    ``target`` is where the driver wants to be; ``rows`` walks there at
    LOCAL_ASCII_WALK_RATE rows a second (0 jumps). Touching the wheel or keys
    takes over from the random driver; ``0`` hands it back.
    """

    def __init__(self, coarsest, default, finest):
        self.coarsest, self.default, self.finest = coarsest, default, finest
        low, high = _setting('LOCAL_ASCII_ROWS', (None, None))
        self.low = int(low) if low else coarsest[1]
        self.high = int(high) if high else finest[1]
        self.low = max(coarsest[1], min(self.low, finest[1]))
        self.high = max(self.low, min(self.high, finest[1]))
        home = _setting('LOCAL_ASCII_HOME_ROWS', None) or default[1]
        self.home = max(self.low, min(int(home), self.high))
        self.rows = float(self.home)
        self.target = self.home
        self.driver = 'random' if _setting('LOCAL_ASCII_ZOOM', 'random') == 'random' else 'manual'
        self.walk_rate = float(_setting('LOCAL_ASCII_WALK_RATE', 30))
        self._last = time.monotonic()
        self._random = {}

    def grid(self):
        return dimensions(self.rows, self.default, self.finest, self.coarsest)

    # Manual control ---------------------------------------------------------
    def nudge(self, steps):
        self.driver = 'manual'
        self.target = max(self.low, min(self.high, int(round(self.target)) + steps))

    def reset(self):
        self.driver = 'random' if _setting('LOCAL_ASCII_ZOOM', 'random') == 'random' else 'manual'
        self.target = self.home
        self._random.clear()

    # Per loop ----------------------------------------------------------------
    def update(self, index):
        """Advance the driver and the walk; True when the shown grid changed."""
        if self.driver == 'random':
            picked = update_zoom_selection(index, self, self._random)
            if picked is not None:
                self.target = picked
        before = self.grid()
        now = time.monotonic()
        elapsed, self._last = now - self._last, now
        if self.walk_rate <= 0:
            self.rows = float(self.target)
        else:
            step = self.walk_rate * elapsed
            delta = self.target - self.rows
            self.rows = float(self.target) if abs(delta) <= step else self.rows + math.copysign(step, delta)
        return self.grid() != before


def update_zoom_selection(index, zoom, state):
    """Pick zoom targets the way update_folder_selection picks folders.

    Same structure, its own state: rest at the home size while the folders
    rest (their rest zone, so both settle together), a first pick when the
    pass leaves it, then periodic picks when the index lands on a modulus
    built from a random multiplier -- offset from the main (3) and float (7)
    triggers so the three do not always fire together -- at least ~1 s apart
    and at most ~20 s, with jitter. Picks come from a pre-shuffled deck of
    the allowed row counts, so every size comes up before any repeats.
    Returns the new target, or None to keep the current one.
    """
    from folder_selector import folder_dictionary, LOCAL_CONTROLFPS
    import index_calculator
    if getattr(index_calculator, 'midi_mode', False):
        return None                                # MIDI clock: home unless steered by hand
    rng = state.setdefault('rng', random.Random(time.time()))
    now = time.time()
    sizes = list(range(zoom.low, zoom.high + 1))
    deck = state.setdefault('deck', deque())
    if len(deck) < len(sizes):
        for _ in range(2 if not deck else 1):
            rng.shuffle(sizes)
            deck.extend(sizes)
    state.setdefault('rand_mult', rng.randint(1, 9))
    state.setdefault('last_time', now)
    state.setdefault('active', False)

    if not folder_dictionary.get('active_cycle', False):
        state['active'] = False                    # folders resting: so is the zoom
        return zoom.home
    if not state['active']:
        state['active'] = True
        state['last_time'] = now
        state['rand_mult'] = rng.randint(1, 4)
        return deck.popleft()

    rand_mult = state['rand_mult']
    jitter = rng.uniform(-rand_mult / LOCAL_CONTROLFPS, rand_mult / LOCAL_CONTROLFPS)
    min_interval = max(1.0, 1.0 + jitter)
    max_interval = max(min_interval, 20.0 + jitter)
    elapsed = now - state['last_time']
    period = LOCAL_CONTROLFPS * (1 + rand_mult // 2) + rand_mult
    if (elapsed >= min_interval and index % period == 11 + rand_mult) or elapsed >= max_interval:
        state['last_time'] = now
        state['rand_mult'] = rng.randint(1, 9)
        return deck.popleft()
    return None


# --- Drawing ------------------------------------------------------------------

def _font():
    from PIL import ImageFont
    path = _setting('LOCAL_ASCII_FONT',
                    'static/fonts/schoolbook-mono/InterleaveSchoolbookMono-Black.woff2')
    try:
        return ImageFont.truetype(path, ATLAS_FONT_PX)
    except Exception as exc:
        print(f"[ASCII-LOCAL] Font {path} unavailable ({exc}); using Pillow's default font")
        return ImageFont.load_default(ATLAS_FONT_PX)


def build_atlas(chars):
    """One white-on-transparent tile per palette entry, in palette order.

    Tiles are the font's natural monospace cell (advance x line height); the
    draw stretches them to the cell, as the browser scales its text. Returns
    the atlas and the tile size; tiles sit ATLAS_GUTTER pixels apart.
    """
    from PIL import Image, ImageDraw
    font = _font()
    ascent, descent = font.getmetrics()
    tile_w = max(1, int(math.ceil(font.getlength('W'))))
    tile_h = ascent + descent
    pitch_w, pitch_h = tile_w + 2 * ATLAS_GUTTER, tile_h + 2 * ATLAS_GUTTER
    rows = int(math.ceil(len(chars) / ATLAS_COLUMNS))
    image = Image.new('L', (pitch_w * ATLAS_COLUMNS, pitch_h * rows), 0)
    draw = ImageDraw.Draw(image)
    for i, ch in enumerate(chars):
        x = (i % ATLAS_COLUMNS) * pitch_w + ATLAS_GUTTER
        y = (i // ATLAS_COLUMNS) * pitch_h + ATLAS_GUTTER
        draw.text((x, y + ascent), ch, fill=255, font=font, anchor='ls')
    return np.asarray(image, np.uint8), (tile_w, tile_h)


VERTEX = """
in vec2 corner;
in float glyph;
in vec3 colour;
uniform vec2 viewport;
uniform vec2 origin;
uniform vec2 cell;
uniform float row_step;
uniform int columns;
uniform vec2 tile_px;       // glyph tile, pixels
uniform vec2 pitch_px;      // tile plus gutter, pixels
uniform float gutter_px;
uniform vec2 atlas_px;
uniform float atlas_columns;
out vec2 uv;
out vec3 tint;
void main() {
    vec2 at = origin + vec2(float(gl_InstanceID % columns) * cell.x,
                            float(gl_InstanceID / columns) * row_step);
    vec2 pixel = at + corner * cell;
    gl_Position = vec4(pixel / viewport * vec2(2.0, -2.0) + vec2(-1.0, 1.0), 0.0, 1.0);
    vec2 t = vec2(mod(glyph, atlas_columns), floor(glyph / atlas_columns));
    uv = (t * pitch_px + gutter_px + corner * tile_px) / atlas_px;
    tint = colour;
}
"""

FRAGMENT = """
uniform sampler2D atlas;
in vec2 uv;
in vec3 tint;
out vec4 result;
void main() {
    result = vec4(tint, texture(atlas, uv).r);
}
"""


class LocalAsciiDisplay:
    """Master-grid capture, size control and the glyph pass for one window."""

    def __init__(self, ctx):
        import moderngl
        self.ctx = ctx
        base = tuple(_setting('ASCII_WEB_MAX_SIZE', (120, 80)))
        mult = max(1, int(_setting('LOCAL_ASCII_MULTIPLIER', 1)))
        whole_scale = _setting('LOCAL_ASCII_MULTIPLY', 'fine') == 'all'
        self.master = (base[0] * mult, base[1] * mult)
        scale = mult if whole_scale else 1
        coarsest = (15 * scale, 10 * scale)
        default = (_setting('ASCII_WIDTH', 60) * scale, _setting('ASCII_HEIGHT', 40) * scale)
        self.zoom = Zoom(coarsest, default, self.master)
        self.line_spacing = float(_setting('LOCAL_ASCII_LINE_SPACING', 1.05))
        self.target = ctx.simple_framebuffer(self.master, components=3)

        atlas, self.tile_px = build_atlas(list(ascii_converter.CHARS))
        h, w = atlas.shape
        self.atlas = ctx.texture((w, h), 1, np.ascontiguousarray(atlas).tobytes())
        self.atlas.build_mipmaps()
        self.atlas.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.atlas.repeat_x = self.atlas.repeat_y = False

        header = "#version 300 es\nprecision mediump float;\n" if ctx.version_code < 330 else "#version 330 core\n"
        self.program = ctx.program(vertex_shader=header + VERTEX, fragment_shader=header + FRAGMENT)
        self.program['tile_px'].value = tuple(float(v) for v in self.tile_px)
        self.program['pitch_px'].value = (self.tile_px[0] + 2.0 * ATLAS_GUTTER,
                                          self.tile_px[1] + 2.0 * ATLAS_GUTTER)
        self.program['gutter_px'].value = float(ATLAS_GUTTER)
        self.program['atlas_px'].value = (float(w), float(h))
        self.program['atlas_columns'].value = float(ATLAS_COLUMNS)
        self.program['atlas'].value = 0
        corners = np.array([0, 0, 1, 0, 0, 1, 1, 1], 'f4')
        self.corners = ctx.buffer(corners.tobytes())
        self.instance_dtype = np.dtype([('glyph', '<f4'), ('colour', 'u1', 3), ('pad', 'u1')])
        self.instances = ctx.buffer(reserve=self.master[0] * self.master[1] * self.instance_dtype.itemsize)
        self.vao = ctx.vertex_array(self.program, [
            (self.corners, '2f', 'corner'),
            (self.instances, '1f 3f1 1x/i', 'glyph', 'colour'),
        ])
        self.glyphs = None
        self.colours = None
        self.dirty = True
        print(f"[ASCII-LOCAL] Master grid {self.master[0]}x{self.master[1]}; "
              f"rows {self.zoom.low}-{self.zoom.high}, home {self.zoom.home}; "
              f"zoom: {self.zoom.driver}")

    # Input -------------------------------------------------------------------
    def attach(self, window):
        import glfw

        def on_scroll(win, dx, dy):
            if dy:
                # Like the browser: scrolling up zooms in (fewer, larger cells).
                self.zoom.nudge(-1 if dy > 0 else 1)

        def on_char(win, codepoint):
            ch = chr(codepoint)
            if ch in '+=':
                self.zoom.nudge(-1)
            elif ch in '-_':
                self.zoom.nudge(1)
            elif ch == '0':
                self.zoom.reset()

        glfw.set_scroll_callback(window, on_scroll)
        glfw.set_char_callback(window, on_char)

    # Per loop ----------------------------------------------------------------
    def update(self, index):
        if self.zoom.update(index):
            self.dirty = True

    def capture(self, main, front, background, main_sbs, front_sbs, framebuffer_size):
        """Sample both layers into the master grid and grade it, as ASCII mode does."""
        self.target.use()
        # The window's projection maps pixels (display_init); the grid quad is
        # in normalised coordinates, as in the headless modes. The image is
        # never drawn to the screen here, so nothing needs the window's back.
        # Set every time: display_init writes its own after a resize or F.
        renderer.update_mvp(np.eye(4, dtype='f4'))
        renderer.set_viewport_size(*self.master, pixel_aspect=_setting('ASCII_FONT_RATIO', .5))
        try:
            renderer.overlay_images_single_pass(main, front, background, main_sbs, front_sbs)
            data = self.target.read(components=3, alignment=1)
        finally:
            # The image quad is never drawn to the screen here, so the
            # renderer keeps the master viewport: restoring the window size
            # would flag the quad dirty and redraw every loop for nothing.
            self.ctx.screen.use()
            self.ctx.viewport = (0, 0, *framebuffer_size)
        grid = np.frombuffer(data, np.uint8).reshape(self.master[1], self.master[0], 3)
        indices, colours, _, _ = ascii_converter.grade_cells(
            grid, width=self.master[0], height=self.master[1], sampled=True)
        self.glyphs = indices.astype(np.float32)
        self.colours = (colours if colours is not None else
                        np.full(grid.shape, 0xe6, np.uint8))
        self.dirty = True

    def draw(self, framebuffer_size, background=(0, 0, 0), framebuffer=None):
        width, height = framebuffer_size
        (framebuffer or self.ctx.screen).use()
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.clear(*(c / 255.0 for c in background))
        self.dirty = False
        if self.glyphs is None or width <= 0 or height <= 0:
            return
        cols, rows = self.zoom.grid()
        master_w, master_h = self.master
        # Nearest master cell per shown cell, as the browser resamples.
        ys = np.minimum(master_h - 1, ((np.arange(rows) + .5) * master_h / rows).astype(int))
        xs = np.minimum(master_w - 1, ((np.arange(cols) + .5) * master_w / cols).astype(int))
        data = np.empty(rows * cols, self.instance_dtype)
        data['glyph'] = self.glyphs[np.ix_(ys, xs)].ravel()
        data['colour'] = self.colours[np.ix_(ys, xs)].reshape(-1, 3)
        self.instances.write(data.tobytes())

        # Fit the picture (the sampled cell proportions) inside the window,
        # then space rows like the browser: line_spacing apart, fitted to
        # the picture height.
        aspect = cols * _setting('ASCII_FONT_RATIO', .5) / rows
        if width / height > aspect:
            area_h, area_w = float(height), height * aspect
        else:
            area_w, area_h = float(width), width / aspect
        cell_h = area_h / (1 + (rows - 1) * self.line_spacing)
        cell_w = area_w / cols
        self.program['viewport'].value = (float(width), float(height))
        self.program['origin'].value = ((width - area_w) / 2, (height - area_h) / 2)
        self.program['cell'].value = (cell_w, cell_h)
        self.program['row_step'].value = cell_h * self.line_spacing
        self.program['columns'].value = cols
        import moderngl
        self.atlas.use(0)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        try:
            self.vao.render(mode=moderngl.TRIANGLE_STRIP, instances=rows * cols)
        finally:
            self.ctx.disable(moderngl.BLEND)

    def close(self):
        for resource in (self.vao, self.instances, self.corners, self.program,
                         self.atlas, self.target):
            try:
                resource.release()
            except Exception:
                pass
