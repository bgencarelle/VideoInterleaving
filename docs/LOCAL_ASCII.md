# Local ASCII display

`--mode asciilocal` (or `scripts/run_app.sh --asciilocal`) shows
the ASCII picture fullscreen in the local window, as glyphs, instead of the
images themselves. It is local mode in every other respect: the same window,
fullscreen and kiosk setup, frame loading, pacing, folder selection and
status page.

## How the picture is made

1. Both layers are sampled on the GPU into a **master character grid**
   (`--ascii-grid COLSxROWS` / `LOCAL_ASCII_GRID`, or else
   `ASCII_WEB_MAX_SIZE`, 120x80, times `LOCAL_ASCII_MULTIPLIER`), exactly as
   ASCII mode samples its grids.
2. `ascii_converter.grade_cells` grades it and picks each cell's character
   and colour: the same grading and glyph choice as telnet and the browser
   (`ASCII_CONTRAST`, `ASCII_BRIGHTNESS`, `ASCII_GAMMA`, `ASCII_SATURATION`,
   palette, colour settings; the `--ascii-*` flags work here too).
3. The shown grid picks the nearest master cell for each of its cells, as
   the browser zoom does, and draws them in Interleave Schoolbook Mono
   (weight 900, line spacing 1.05) from one atlas texture in one instanced
   draw.

The master is sampled on this machine only, so a multiplier costs no other
viewer anything. Per picture the GPU work is one small grid render and a
draw of at most master-size glyph quads.

## Size control

One **row count** sets the size; zoom changes resolution, not crop.

`LOCAL_ASCII_SHAPE` sets the grid's shape; its rows stay as set and the
columns follow, at every zoom size:

- **`'picture'`** (default): the picture's shape. The whole image fills
  the grid, nothing cropped, and the grid is drawn as large as the window
  allows, so bars appear only where the screen's shape differs from the
  picture's (a 3:4 picture on a 16:10 screen: 126x80, 750x1000 of
  1600x1000).
- **`'window'`**: the window's shape (16:10 at 80 rows is 269x80); the image
  is cropped to cover it, as ASCII mode's cover sampling does.
- **`'grid'`**: the browser's mapping,
  `dimensions()` (a port of `static/ascii_resolution.js`, checked against
  it): 15x10 at the coarse end, `ASCII_WIDTH`x`ASCII_HEIGHT` (60x40) at the
  default, the master at the fine end; the grid is letterboxed in the
  window and the image in the grid.

The grid is reshaped when the window changes (fullscreen toggle, resize,
rotation). Rotation (`R`, `--rotation`) turns the whole ASCII picture,
letters included, as local mode turns the image; mirroring (`M`) mirrors the
sampled picture.

Two drivers set it:

- **Mouse and keys:** as the browser. Each wheel notch scales the row
  count by the same proportion (about 16%); `+`/`-` move one whole row;
  `0` resets. Changes are instant. Touching them takes over from the
  random driver; `0` hands control back.
- **Random** (`LOCAL_ASCII_ZOOM = 'random'`): built like the folder
  selector. It starts at the home size, picks when the folders leave
  their rest zone (and again each time they do, once the size has been
  held long enough; the rest itself does not pull it home), and picks when
  the index lands on its own modulus
  (offset from the main and float triggers), from a shuffled deck of the
  allowed sizes. Picks come at least and at most `LOCAL_ASCII_HOLD`
  (default 2-20 s) apart, counted from each pick as the folder selector
  counts. In a simulated 20 minutes against the real folder selector the
  size changed typically every 4.6 s (2-19 s), the folders every 2.7-3.7 s
  (their count includes rest-zone resets). MIDI clock holds the home size.

Each random change takes its own random time from
`LOCAL_ASCII_RANDOM_TRANSITION` (default 0.5-3 s), eases in and out
(smootherstep), and moves proportionally, as a wheel zoom does: equal times
give equal ratios, so 10 -> 20 rows takes as long as 40 -> 80, and the
coarse end does not jump while the fine end crawls. A pick that comes
before a change has landed carries on from wherever the size is.
`LOCAL_ASCII_WALK_RATE` (default 0, instant, as the browser) can make wheel
and key changes walk instead.

A change steps a whole grid at a time, as a text screen would.
`LOCAL_ASCII_CROSSFADE = True` instead cross-fades the two neighbouring
grids while the size is between two row counts: smoother, but it reads as
graphics rather than text.

The finest grid can be set per run: `--mode asciilocal --ascii-grid 200x120`
(at least 15x10). Zoom then ranges from 10 rows up to its rows; with
`'picture'` or `'window'` its columns come from that shape instead.

## Small letters

Small letters cover little of their cells, so the eye averages them with the
black between them and the picture goes dark. Two arguments counter it, both
off by default:

- `--ascii-glow F` (0-1): under the letters, each cell's colour at its own
  brightness, blended across neighbouring cells (a texture one texel per
  cell, filtered). It starts when cells drop below `LOCAL_ASCII_GLOW_FROM`
  pixels high and is at full strength from `LOCAL_ASCII_GLOW_FULL` down, so
  large letters look as before.
- `--ascii-strokes F` (0-1): the atlas's smaller mip levels keep their
  brightest pixels instead of averaging thin strokes to grey (0 averages as
  before; 1 keeps each 2x2 block as bright as its brightest pixel). It acts
  only where letters are drawn smaller than the atlas, more the smaller they
  get; at the very smallest sizes 1 turns letters into solid blocks.

Example: `--mode asciilocal --ascii-glow 0.6 --ascii-strokes 0.5`.

## Settings (settings.py)

| Setting | Default | Meaning |
|---|---|---|
| `LOCAL_ASCII_SHAPE` | `'picture'` | Grid shape: `'picture'` (whole image, as large as the window allows), `'window'` (fills it, image cropped), `'grid'` (browser's, letterboxed) |
| `LOCAL_ASCII_GRID` | None | Master (finest) grid `(cols, rows)`; `--ascii-grid` sets it. None = 120x80 x multiplier. `'picture'`/`'window'` use only its rows |
| `LOCAL_ASCII_MULTIPLIER` | 1 | Master = 120x80 x this, when no grid is given |
| `LOCAL_ASCII_MULTIPLY` | `'fine'` | `'fine'`: only the finest end grows; `'all'`: coarsest and default sizes grow by as much as the master |
| `LOCAL_ASCII_ROWS` | `(None, None)` | Allowed row range; None = 10 / the master's rows |
| `LOCAL_ASCII_HOME_ROWS` | None | Size at rest and after reset; None = `ASCII_HEIGHT` |
| `LOCAL_ASCII_ZOOM` | `'random'` | `'random'` or `'manual'` |
| `LOCAL_ASCII_WALK_RATE` | 0 | Mouse/keys: rows per second toward a new size; 0 instant, as the browser |
| `LOCAL_ASCII_RANDOM_TRANSITION` | `(0.5, 3.0)` | Random driver: seconds each change takes, drawn per change, eased; `(0, 0)` jumps |
| `LOCAL_ASCII_HOLD` | `(2.0, 20.0)` | Random driver: seconds between changes, at least / at most, counted from each pick |
| `LOCAL_ASCII_CROSSFADE` | False | Cross-fade neighbouring grid sizes mid-change instead of stepping |
| `LOCAL_ASCII_LINE_SPACING` | 1.05 | As the web viewer |
| `LOCAL_ASCII_GLOW` | 0 | `--ascii-glow`: neighbouring cells' colour under small letters |
| `LOCAL_ASCII_GLOW_FROM` | 32 | Cell height (pixels) where the glow starts |
| `LOCAL_ASCII_GLOW_FULL` | 8 | Cell height (pixels) from which it is at full strength |
| `LOCAL_ASCII_STROKES` | 0 | `--ascii-strokes`: keep small letters bright |
| `LOCAL_ASCII_FONT` | `static/fonts/century-schoolbook/CenturySchoolbookBold.ttf` | Path to any TTF/OTF/WOFF2 Pillow can open, from the repo root or absolute |

The range is a look decision: set it by eye on the display it runs on.

## Needs

OpenGL 3.3 or GLES 3 (it falls back to the images on a legacy GL2 context,
with a warning), and Pillow able to read WOFF2 (FreeType built with Brotli:
pip Pillow and Debian's are). Without the font it uses Pillow's default font.
