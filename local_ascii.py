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
and keys, or by a random driver built like the folder selector, whose changes
each take a random time and ease in and out. The local master is
``LOCAL_ASCII_GRID`` (``--ascii-grid``), or ``ASCII_WEB_MAX_SIZE`` times
``LOCAL_ASCII_MULTIPLIER``, sampled on this machine, so finer grids cost no
other viewer anything.
"""
from collections import deque
import math
import random
import time

import numpy as np

import ascii_converter
import renderer
import settings

WHEEL_PER_PIXEL = .0015    # static/ascii_viewer.js: resolution * exp(deltaY * .0015)
ATLAS_FONT_PX = 96        # glyphs are drawn once, large, and mipmapped down
ATLAS_COLUMNS = 8
ATLAS_GUTTER = 8          # empty pixels around each tile: small mip levels must not bleed
ATLAS_ALIGN = 64          # tile slots are a multiple of this, so mip levels 1-6 never mix tiles


def _setting(name, default):
    return getattr(settings, name, default)


def decode_scale(source_size):
    """JPEG decode scale for this display: ``(scale, min_size)`` for
    ImageLoader.read_image, or ``(None, None)`` for full size.

    The picture is only ever sampled into the master grid, one sample a
    cell, so a full-size decode and upload is spent on pixels that are
    never read. TurboJPEG decodes at 1/2, 1/4 or 1/8 for a fraction of the
    work (and averages as it shrinks, so the samples are steadier). The
    largest reduction that keeps LOCAL_ASCII_DECODE_ROW_PX decoded pixels
    per finest row is used; files too small for it decode at full size.
    """
    per_row = int(_setting('LOCAL_ASCII_DECODE_ROW_PX', 4))
    if not source_size or per_row <= 0:
        return None, None
    grid = _setting('LOCAL_ASCII_GRID', None)
    if grid:
        rows = int(grid[1])
    else:
        rows = (int(_setting('ASCII_WEB_MAX_SIZE', (120, 80))[1]) *
                max(1, int(_setting('LOCAL_ASCII_MULTIPLIER', 1))))
    need = per_row * rows                  # decoded pixels high, at least
    for divisor in (8, 4, 2):
        if source_size[1] // divisor >= need:
            return (1, divisor), (1, need * divisor)
    return None, None


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

    ``target`` is where the driver wants to be. Mouse and keys act as in the
    browser: the wheel scales the row count, keys move one row, both instant
    (LOCAL_ASCII_WALK_RATE > 0 walks instead). Each random change takes a
    time drawn from LOCAL_ASCII_RANDOM_TRANSITION, eases in and out
    (smootherstep) and moves proportionally, as a wheel zoom does; the next
    random pick is timed from this one, as the folder selector's are. Touching the wheel or keys takes over from the random driver; ``0``
    hands it back.
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
        self.walk_rate = float(_setting('LOCAL_ASCII_WALK_RATE', 0))
        low_s, high_s = _setting('LOCAL_ASCII_RANDOM_TRANSITION', (0.5, 3.0))
        self.transition = (max(0.0, float(low_s)), max(0.0, float(low_s), float(high_s)))
        hold_min, hold_max = _setting('LOCAL_ASCII_HOLD', (2.0, 20.0))
        self.hold = (max(0.0, float(hold_min)), max(0.0, float(hold_min), float(hold_max)))
        self.crossfade = bool(_setting('LOCAL_ASCII_CROSSFADE', False))
        self.cols_for = None               # picture/window shape: rows -> cols in proportion
        self._move = None                  # (start rows, start time, seconds) of a random change
        self._rng = random.Random()
        self._last = time.monotonic()
        self._random = {}

    def dims(self, rows):
        """Grid (cols, rows) for a row count: the browser's mapping, or, when
        the picture fills the window, columns in proportion to the window."""
        if self.cols_for is None:
            return dimensions(rows, self.default, self.finest, self.coarsest)
        rows = _js_round(min(max(rows, self.coarsest[1]), self.finest[1]))
        return self.cols_for(rows), rows

    def grid(self):
        return self.dims(self.rows)

    def layers(self):
        """What to draw for the current row count: ``[(grid, weight), ...]``.

        Between two whole row counts the two neighbouring grids cross-fade
        by how far ``rows`` is between them, so a size change glides instead
        of stepping a row at a time. Off (LOCAL_ASCII_CROSSFADE = False) or
        on a whole count, one grid at full weight.
        """
        low = math.floor(self.rows)
        part = self.rows - low
        if not self.crossfade or part < .01 or part > .99:
            return [(self.grid(), 1.0)]
        return [(self.dims(low), 1.0 - part), (self.dims(low + 1), part)]

    # Manual control ---------------------------------------------------------
    def nudge(self, steps):
        """Keys, as the browser: one whole row from where the size is now."""
        self.driver = 'manual'
        self._move = None
        here = math.ceil(self.rows) if steps < 0 else math.floor(self.rows)
        self.target = max(self.low, min(self.high, here + steps))

    def scale(self, factor):
        """Wheel, as the browser: the row count times a factor, so each notch
        changes the size by the same proportion at every size."""
        self.driver = 'manual'
        self._move = None
        self.target = max(float(self.low), min(float(self.high), self.rows * factor))

    def reset(self):
        self.driver = 'random' if _setting('LOCAL_ASCII_ZOOM', 'random') == 'random' else 'manual'
        self.target = self.home
        self._move = None
        self._random.clear()

    # Per loop ----------------------------------------------------------------
    def update(self, index):
        """Advance the driver and the walk; True when the picture must be redrawn."""
        now = time.monotonic()
        elapsed, self._last = now - self._last, now
        if self.driver == 'random':
            picked = update_zoom_selection(index, self, self._random)
            if picked is not None and picked != self.target:
                self.target = picked
                seconds = self._rng.uniform(*self.transition)
                self._move = (self.rows, now, seconds)
                # The hold counts from this pick, as the folder selector's
                # intervals do; a pick that comes before a change has landed
                # carries on from wherever the size is.
        before = self.grid(), self.rows
        if self._move is not None:
            start, began, seconds = self._move
            t = (now - began) / seconds if seconds > 0 else 1.0
            if t >= 1.0:
                self.rows, self._move = float(self.target), None
            else:
                t = max(0.0, t)
                # smootherstep: no jolt at either end, not just no jump
                e = t * t * t * (t * (6.0 * t - 15.0) + 10.0)
                # Proportional, as the browser's wheel moves: equal times give
                # equal ratios (10 -> 20 rows takes as long as 40 -> 80), so
                # the coarse end does not jump and the fine end does not crawl.
                self.rows = start * (self.target / start) ** e if start > 0 else float(self.target)
        elif self.walk_rate <= 0:
            self.rows = float(self.target)
        else:
            step = self.walk_rate * elapsed
            delta = self.target - self.rows
            self.rows = float(self.target) if abs(delta) <= step else self.rows + math.copysign(step, delta)
        if self.crossfade:
            return self.rows != before[1]
        return self.grid() != before[0]


def update_zoom_selection(index, zoom, state):
    """Pick zoom targets the way update_folder_selection picks folders.

    Same structure, its own state: the folders leaving their rest zone is a
    pick (the first one, and later ones too if the size has been held long
    enough), then periodic picks when the index lands on a modulus
    built from a random multiplier -- offset from the main (3) and float (7)
    triggers so the three do not always fire together -- held at least
    LOCAL_ASCII_HOLD[0] and at most [1] seconds apart, with jitter, counted
    from the previous pick as the folder selector counts. Picks come from a pre-shuffled deck of
    the allowed row counts, so every size comes up before any repeats.
    Returns the new target, or None to keep the current one. The folders'
    rest does not pull the size back home: it comes round once per pass,
    so a size would only ever get one pass before snapping back.
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

    rand_mult = state['rand_mult']
    jitter = rng.uniform(-rand_mult / LOCAL_CONTROLFPS, rand_mult / LOCAL_CONTROLFPS)
    hold_min, hold_max = getattr(zoom, 'hold', (2.0, 20.0))
    min_interval = max(hold_min, hold_min + jitter)
    max_interval = max(min_interval, hold_max + jitter)
    elapsed = now - state['last_time']

    if not folder_dictionary.get('active_cycle', False):
        state['active'] = False                    # folders resting: hold the size
        return None
    if not state['active']:
        state['active'] = True
        if not state.get('started') or elapsed >= min_interval:
            state['started'] = True
            state['last_time'] = now
            state['rand_mult'] = rng.randint(1, 4)
            return deck.popleft()

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

    Tiles are the font's widest advance ('W') by its line height, each
    letter centred across it; the draw stretches them to the cell, as the
    browser scales its text. Returns
    the atlas and the tile size; tiles sit ATLAS_GUTTER pixels apart.
    """
    from PIL import Image, ImageDraw
    font = _font()
    ascent, descent = font.getmetrics()
    tile_w = max(1, int(math.ceil(font.getlength('W'))))
    tile_h = ascent + descent
    # Slots padded to ATLAS_ALIGN: every 2x2 reduction down to 1/64 then stays
    # inside one tile, whatever filter builds the levels.
    pitch_w = -(-(tile_w + 2 * ATLAS_GUTTER) // ATLAS_ALIGN) * ATLAS_ALIGN
    pitch_h = -(-(tile_h + 2 * ATLAS_GUTTER) // ATLAS_ALIGN) * ATLAS_ALIGN
    rows = int(math.ceil(len(chars) / ATLAS_COLUMNS))
    image = Image.new('L', (pitch_w * ATLAS_COLUMNS, pitch_h * rows), 0)
    draw = ImageDraw.Draw(image)
    for i, ch in enumerate(chars):
        # Centred in the cell: a proportional font's narrow letters (i, l, j)
        # would otherwise sit at the left with a gap to the right. A
        # monospaced font's advance is the cell, so nothing moves.
        x = (i % ATLAS_COLUMNS) * pitch_w + ATLAS_GUTTER + (tile_w - font.getlength(ch)) / 2
        y = (i // ATLAS_COLUMNS) * pitch_h + ATLAS_GUTTER
        draw.text((x, y + ascent), ch, fill=255, font=font, anchor='ls')
    return np.asarray(image, np.uint8), (tile_w, tile_h), (pitch_w, pitch_h)


def mip_levels(atlas, strokes):
    """Mip levels 1.. for the atlas, each from the one above it.

    Each 2x2 block becomes ``mean + strokes * (max - mean)``: 0 is the usual
    average, where thin strokes fade to grey as letters shrink; 1 keeps every
    block as bright as its brightest pixel, so small letters keep their ink.
    Only levels below full size change, so it acts only where letters are
    drawn smaller than the atlas -- the smaller, the more.
    """
    levels = []
    level = atlas.astype(np.float32)
    while min(level.shape) > 1:
        h, w = level.shape[0] // 2, level.shape[1] // 2
        blocks = level[:2 * h, :2 * w].reshape(h, 2, w, 2)
        mean, peak = blocks.mean(axis=(1, 3)), blocks.max(axis=(1, 3))
        level = mean + strokes * (peak - mean)
        levels.append(np.clip(level + .5, 0, 255).astype(np.uint8))
    return levels


def glow_strength(cell_h, glow, start, full):
    """Glow under the letters for a cell this many pixels high: none from
    ``start`` up, rising to ``glow`` at ``full`` and below."""
    if glow <= 0 or cell_h >= start:
        return 0.0
    if cell_h <= full or start <= full:
        return float(glow)
    return float(glow) * (start - cell_h) / (start - full)


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
// Rotation (R key / --rotation): the picture is laid out on a canvas of
// size viewport, then turned onto the screen a quarter at a time,
// counter-clockwise as the image itself turns in local mode.
uniform float quarter;
vec2 on_screen(vec2 p) {
    vec2 screen = viewport;
    vec2 s = p;
    if (quarter == 1.0) { s = vec2(p.y, viewport.x - p.x); screen = viewport.yx; }
    else if (quarter == 2.0) { s = viewport - p; }
    else if (quarter == 3.0) { s = vec2(viewport.y - p.y, p.x); screen = viewport.yx; }
    return s / screen * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
}
void main() {
    vec2 at = origin + vec2(float(gl_InstanceID % columns) * cell.x,
                            float(gl_InstanceID / columns) * row_step);
    vec2 pixel = at + corner * cell;
    gl_Position = vec4(on_screen(pixel), 0.0, 1.0);
    vec2 t = vec2(mod(glyph, atlas_columns), floor(glyph / atlas_columns));
    uv = (t * pitch_px + gutter_px + corner * tile_px) / atlas_px;
    tint = colour;
}
"""

GLOW_VERTEX = """
in vec2 corner;
uniform vec2 viewport;
uniform vec2 origin;
uniform vec2 area;
out vec2 at;
// Rotation (R key / --rotation): the picture is laid out on a canvas of
// size viewport, then turned onto the screen a quarter at a time,
// counter-clockwise as the image itself turns in local mode.
uniform float quarter;
vec2 on_screen(vec2 p) {
    vec2 screen = viewport;
    vec2 s = p;
    if (quarter == 1.0) { s = vec2(p.y, viewport.x - p.x); screen = viewport.yx; }
    else if (quarter == 2.0) { s = viewport - p; }
    else if (quarter == 3.0) { s = vec2(viewport.y - p.y, p.x); screen = viewport.yx; }
    return s / screen * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
}
void main() {
    at = corner * area;
    vec2 pixel = origin + at;
    gl_Position = vec4(on_screen(pixel), 0.0, 1.0);
}
"""

# The shown grid's colours, one texel per cell, filtered linearly: between
# the letters each pixel gets a blend of the cells around it. Texel centres
# sit on the cell centres (rows are line_spacing apart, as the letters are).
GLOW_FRAGMENT = """
uniform sampler2D cells;
uniform vec2 area;          // picture size, pixels
uniform float rows;
uniform float cell_h;
uniform float row_step;
uniform float strength;
in vec2 at;
out vec4 result;
void main() {
    vec2 uv = vec2(at.x / area.x,
                   (at.y - 0.5 * cell_h + 0.5 * row_step) / (row_step * rows));
    result = vec4(texture(cells, uv).rgb, strength);
}
"""

FRAGMENT = """
uniform sampler2D atlas;
uniform float fade;
in vec2 uv;
in vec3 tint;
out vec4 result;
void main() {
    result = vec4(tint, texture(atlas, uv).r * fade);
}
"""


class LocalAsciiDisplay:
    """Master-grid capture, size control and the glyph pass for one window."""

    def __init__(self, ctx):
        import moderngl
        self.ctx = ctx
        base = tuple(_setting('ASCII_WEB_MAX_SIZE', (120, 80)))
        grid = _setting('LOCAL_ASCII_GRID', None)
        if grid:
            self.master = (int(grid[0]), int(grid[1]))
        else:
            mult = max(1, int(_setting('LOCAL_ASCII_MULTIPLIER', 1)))
            self.master = (base[0] * mult, base[1] * mult)
        # 'all' scales the coarse and default sizes by as much as the master
        # grew over the web's 120x80; 'fine' leaves them where the web has them.
        whole_scale = _setting('LOCAL_ASCII_MULTIPLY', 'fine') == 'all'
        scale = max(1.0, self.master[1] / base[1]) if whole_scale else 1.0
        coarsest = (_js_round(15 * scale), _js_round(10 * scale))
        default = (_js_round(_setting('ASCII_WIDTH', 60) * scale),
                   _js_round(_setting('ASCII_HEIGHT', 40) * scale))
        self.zoom = Zoom(coarsest, default, self.master)
        self.line_spacing = float(_setting('LOCAL_ASCII_LINE_SPACING', 1.05))
        self.font_ratio = float(_setting('ASCII_FONT_RATIO', .5))
        # The grid's shape (its rows stay as set; columns follow):
        #   'picture' (default): the picture's shape. The whole image fills
        #       the grid, and the grid is fitted as large as the window
        #       allows, so bars appear only where the screen's shape differs.
        #   'window': the window's shape; the image is cropped to cover it.
        #   'grid': the browser's mapping, letterboxed in the window and the
        #       image letterboxed in the grid.
        shape = str(_setting('LOCAL_ASCII_SHAPE', 'picture')).lower()
        self.shape = shape if shape in ('picture', 'window', 'grid') else 'picture'
        self.fill = self.shape != 'grid'
        self.cell_aspect = self.font_ratio
        self.frame_aspect = None
        self.window = None
        self._framebuffer = None
        self.target = ctx.simple_framebuffer(self.master, components=3)

        ascii_converter.refresh_palette()
        # GLYPHS: the palette plus any letters the biography phrase needs.
        atlas, self.tile_px, pitch = build_atlas(list(ascii_converter.GLYPHS))
        h, w = atlas.shape
        self.atlas = ctx.texture((w, h), 1, np.ascontiguousarray(atlas).tobytes())
        self.atlas.build_mipmaps()
        self.strokes = min(1.0, max(0.0, float(_setting('LOCAL_ASCII_STROKES', 0.0))))
        if self.strokes > 0:
            for level, image in enumerate(mip_levels(atlas, self.strokes), 1):
                self.atlas.write(np.ascontiguousarray(image).tobytes(), level=level, alignment=1)
        self.atlas.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.atlas.repeat_x = self.atlas.repeat_y = False

        header = "#version 300 es\nprecision mediump float;\n" if ctx.version_code < 330 else "#version 330 core\n"
        self.program = ctx.program(vertex_shader=header + VERTEX, fragment_shader=header + FRAGMENT)
        self.program['tile_px'].value = tuple(float(v) for v in self.tile_px)
        self.program['pitch_px'].value = (float(pitch[0]), float(pitch[1]))
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
        # Glow: the shown grid's colours as a tiny texture under the letters.
        self.glow = max(0.0, float(_setting('LOCAL_ASCII_GLOW', 0.0)))
        self.glow_from = float(_setting('LOCAL_ASCII_GLOW_FROM', 32))
        self.glow_full = float(_setting('LOCAL_ASCII_GLOW_FULL', 8))
        self.glow_program = self.glow_vao = None
        self.glow_cells = None
        if self.glow > 0:
            self.glow_program = ctx.program(vertex_shader=header + GLOW_VERTEX,
                                            fragment_shader=header + GLOW_FRAGMENT)
            self.glow_program['cells'].value = 1
            self.glow_vao = ctx.vertex_array(self.glow_program, [(self.corners, '2f', 'corner')])
        self.glyphs = None
        self.colours = None
        self.dirty = True
        print(f"[ASCII-LOCAL] Master grid {self.master[0] if not self.fill else 'N'}x"
              f"{self.master[1]}, shaped to the {self.shape}; "
              f"rows {self.zoom.low}-{self.zoom.high}, home {self.zoom.home}; "
              f"zoom: {self.zoom.driver}, random changes "
              f"{self.zoom.transition[0]:g}-{self.zoom.transition[1]:g} s; "
              f"glow {self.glow:g} (cells under {self.glow_from:g}-{self.glow_full:g} px), "
              f"strokes {self.strokes:g}")

    # Input -------------------------------------------------------------------
    def attach(self, window):
        import glfw
        self.window = window

        def on_scroll(win, dx, dy):
            if dy:
                # As the browser: each notch scales the row count by the same
                # proportion (a notch there is 100 px at 0.0015 a pixel,
                # clamped to 200 px); scrolling up zooms in (fewer, larger cells).
                pixels = max(-200.0, min(200.0, -dy * 100.0))
                self.zoom.scale(math.exp(pixels * WHEEL_PER_PIXEL))

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
        # A resize, fullscreen toggle or rotation already redraws (the
        # display code marks the view dirty) and fit() then sees it. Only a
        # picture of a new shape needs flagging here. Asking the window its
        # size every loop cost a window-system round trip each time (1.3 s
        # of a 25 s profiled run).
        if (self.shape == 'picture' and self._framebuffer is not None
                and tuple(renderer._image_size) != self._framebuffer[3:]):
            self.dirty = True

    @staticmethod
    def _canvas(framebuffer_size):
        """Canvas the picture is laid out on, and quarter turns onto the
        screen: the window turned by the rotation (R key / --rotation)."""
        width, height = framebuffer_size
        quarter = int(round(getattr(renderer, '_rotation_angle', 0) or 0)) % 360 // 90
        return (height, width, quarter) if quarter % 2 else (width, height, quarter)

    def _fit_key(self, framebuffer_size):
        key = self._canvas(framebuffer_size)
        return key + tuple(renderer._image_size) if self.shape == 'picture' else key

    def _area(self, width, height, aspect):
        """The largest width/height of this aspect that fits the window."""
        if width / height > aspect:
            return height * aspect, float(height)
        return float(width), width / aspect

    def fit(self, framebuffer_size):
        """Shape the master grid to the picture or the window. True when it
        was rebuilt (the picture must then be captured again)."""
        width, height, _ = self._canvas(framebuffer_size)
        if not self.fill or width <= 0 or height <= 0:
            return False
        key = self._fit_key(framebuffer_size)
        if key == self._framebuffer:
            return False
        self._framebuffer = key
        aspect = width / height
        if self.shape == 'picture':
            image_w, image_h = renderer._image_size
            if image_w > 0 and image_h > 0:
                aspect = image_w / image_h
        self.frame_aspect = aspect
        area_w, area_h = self._area(width, height, aspect)
        rows = self.master[1]
        cols = max(1, _js_round(rows * self.line_spacing * area_w / (self.font_ratio * area_h)))
        # Proportions of one master cell as it lands on the window: width
        # over row pitch. The image is sampled with them, so it is not
        # stretched.
        cell_h = area_h / (1 + (rows - 1) * self.line_spacing)
        self.cell_aspect = (area_w / cols) / (cell_h * self.line_spacing)
        self.zoom.cols_for = lambda r, c=cols, m=rows: max(1, _js_round(r * c / m))
        self.zoom.finest = (cols, rows)
        if (cols, rows) == self.master and self.glyphs is not None:
            return False
        self.master = (cols, rows)
        for resource in (self.vao, self.instances, self.target):
            resource.release()
        self.target = self.ctx.simple_framebuffer(self.master, components=3)
        self.instances = self.ctx.buffer(reserve=cols * rows * self.instance_dtype.itemsize)
        self.vao = self.ctx.vertex_array(self.program, [
            (self.corners, '2f', 'corner'),
            (self.instances, '1f 3f1 1x/i', 'glyph', 'colour'),
        ])
        self.glyphs = self.colours = None
        self.dirty = True
        print(f"[ASCII-LOCAL] Window {width}x{height}: master grid {cols}x{rows}, "
              f"picture {area_w:.0f}x{area_h:.0f}")
        return True

    def capture(self, main, front, background, main_sbs, front_sbs, framebuffer_size):
        """Sample both layers into the master grid and grade it, as ASCII mode does."""
        self.target.use()
        # The window's projection maps pixels (display_init); the grid quad is
        # in normalised coordinates, as in the headless modes. The image is
        # never drawn to the screen here, so nothing needs the window's back.
        # Set every time: display_init writes its own after a resize or F.
        renderer.update_mvp(np.eye(4, dtype='f4'))
        renderer.set_viewport_size(*self.master, pixel_aspect=self.cell_aspect, cover=self.shape == 'window')
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
        width, height, quarter = self._canvas(framebuffer_size)
        master_w, master_h = self.master
        # Fit the picture inside the window, then space rows like the
        # browser: line_spacing apart, fitted to the picture height. The
        # frame is the master grid's shape at every size: each size covers
        # the whole master, and its own rounded cols/rows would move the
        # frame a little on every zoom step. The cells absorb the difference.
        aspect = (self.frame_aspect if self.fill and self.frame_aspect
                  else master_w * self.font_ratio / master_h)
        area_w, area_h = self._area(width, height, aspect)
        origin = ((width - area_w) / 2, (height - area_h) / 2)

        import moderngl
        self.atlas.use(0)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        try:
            layers = self.zoom.layers()
            # Mid-change two neighbouring grids cross-fade (one at weight
            # 1-f, one at f); their letters barely overlap, so the overall
            # ink stays the same through the fade.
            for n, ((cols, rows), weight) in enumerate(layers):
                # Nearest master cell per shown cell, as the browser resamples.
                ys = np.minimum(master_h - 1, ((np.arange(rows) + .5) * master_h / rows).astype(int))
                xs = np.minimum(master_w - 1, ((np.arange(cols) + .5) * master_w / cols).astype(int))
                glyphs = self.glyphs[np.ix_(ys, xs)]
                colours = self.colours[np.ix_(ys, xs)]
                cell_h = area_h / (1 + (rows - 1) * self.line_spacing)
                if n == 0:
                    # Glow once, from the grid that dominates.
                    g_glyphs, g_colours, g_cell_h = glyphs, colours, cell_h
                    if len(layers) > 1 and layers[1][1] > weight:
                        (g_cols, g_rows), _ = layers[1]
                        g_ys = np.minimum(master_h - 1, ((np.arange(g_rows) + .5) * master_h / g_rows).astype(int))
                        g_xs = np.minimum(master_w - 1, ((np.arange(g_cols) + .5) * master_w / g_cols).astype(int))
                        g_glyphs = self.glyphs[np.ix_(g_ys, g_xs)]
                        g_colours = self.colours[np.ix_(g_ys, g_xs)]
                        g_cell_h = area_h / (1 + (g_rows - 1) * self.line_spacing)
                    strength = glow_strength(g_cell_h, self.glow, self.glow_from, self.glow_full)
                    if strength > 0:
                        self._draw_glow(g_glyphs, g_colours, (width, height), (area_w, area_h),
                                        g_cell_h, strength, quarter)
                data = np.empty(rows * cols, self.instance_dtype)
                # --ascii-bio spells on the grid as shown, after resampling,
                # so the phrase reads at every zoom size.
                data['glyph'] = ascii_converter.apply_bio(glyphs.astype(np.int64)).ravel()
                data['colour'] = colours.reshape(-1, 3)
                self.instances.write(data.tobytes())
                self.program['viewport'].value = (float(width), float(height))
                self.program['quarter'].value = float(quarter)
                self.program['origin'].value = origin
                self.program['cell'].value = (area_w / cols, cell_h)
                self.program['row_step'].value = cell_h * self.line_spacing
                self.program['columns'].value = cols
                self.program['fade'].value = float(weight)
                self.atlas.use(0)
                self.vao.render(mode=moderngl.TRIANGLE_STRIP, instances=rows * cols)
        finally:
            self.ctx.disable(moderngl.BLEND)

    def _draw_glow(self, glyphs, colours, viewport, area, cell_h, strength, quarter=0):
        """Under the letters: each cell's colour at its own brightness (what
        its letter's ink stands for), blended across neighbouring cells."""
        import moderngl
        rows, cols = glyphs.shape
        level = 1.0 - glyphs / max(1, len(ascii_converter.CHARS) - 1)
        cells = np.clip(colours * level[..., None] + .5, 0, 255).astype(np.uint8)
        if self.glow_cells is None or self.glow_cells.size != (cols, rows):
            if self.glow_cells is not None:
                self.glow_cells.release()
            self.glow_cells = self.ctx.texture((cols, rows), 3, alignment=1)
            self.glow_cells.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.glow_cells.repeat_x = self.glow_cells.repeat_y = False
        self.glow_cells.write(np.ascontiguousarray(cells).tobytes(), alignment=1)
        self.glow_cells.use(1)
        p = self.glow_program
        width, height = viewport
        area_w, area_h = area
        p['viewport'].value = (float(width), float(height))
        p['quarter'].value = float(quarter)
        p['origin'].value = ((width - area_w) / 2, (height - area_h) / 2)
        p['area'].value = (float(area_w), float(area_h))
        p['rows'].value = float(rows)
        p['cell_h'].value = float(cell_h)
        p['row_step'].value = float(cell_h * self.line_spacing)
        p['strength'].value = float(strength)
        self.glow_vao.render(mode=moderngl.TRIANGLE_STRIP)

    def close(self):
        for resource in (self.vao, self.instances, self.corners, self.program,
                         self.atlas, self.target, self.glow_vao, self.glow_program,
                         self.glow_cells):
            try:
                resource.release()
            except Exception:
                pass
