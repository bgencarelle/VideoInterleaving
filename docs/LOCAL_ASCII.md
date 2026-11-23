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

The browser's: one **row count**, mapped to a grid by `dimensions()` (a port
of `static/ascii_resolution.js`, checked against it): 15x10 at the coarse
end, `ASCII_WIDTH`x`ASCII_HEIGHT` (60x40) at the default, the master at the
fine end. The whole picture stays fitted; zoom changes resolution, not crop.

Two drivers set it:

- **Mouse and keys:** wheel, `+`/`-`, `0` resets. Touching them takes over
  from the random driver; `0` hands control back.
- **Random** (`LOCAL_ASCII_ZOOM = 'random'`): built like the folder
  selector. It rests at the home size while the folders rest, picks when
  they leave rest, then picks when the index lands on its own modulus
  (offset from the main and float triggers), at least ~1 s and at most ~20 s
  apart, from a shuffled deck of the allowed sizes. MIDI clock holds the
  home size.

Mouse and key changes walk at `LOCAL_ASCII_WALK_RATE` rows a second (0
jumps). Each random change takes its own random time from
`LOCAL_ASCII_RANDOM_TRANSITION` (default 2-8 s) and eases in and out; the
next random pick is timed from when it lands, so a slow change is never cut
short.

The finest grid can be set per run: `--mode asciilocal --ascii-grid 200x120`
(at least 15x10). Zoom then ranges from 15x10 up to it.

## Settings (settings.py)

| Setting | Default | Meaning |
|---|---|---|
| `LOCAL_ASCII_GRID` | None | Master (finest) grid `(cols, rows)`; `--ascii-grid` sets it. None = 120x80 x multiplier |
| `LOCAL_ASCII_MULTIPLIER` | 1 | Master = 120x80 x this, when no grid is given |
| `LOCAL_ASCII_MULTIPLY` | `'fine'` | `'fine'`: only the finest end grows; `'all'`: coarsest and default sizes grow by as much as the master |
| `LOCAL_ASCII_ROWS` | `(None, None)` | Allowed row range; None = 10 / the master's rows |
| `LOCAL_ASCII_HOME_ROWS` | None | Size at rest and after reset; None = `ASCII_HEIGHT` |
| `LOCAL_ASCII_ZOOM` | `'random'` | `'random'` or `'manual'` |
| `LOCAL_ASCII_WALK_RATE` | 30 | Mouse/keys: rows per second toward a new size |
| `LOCAL_ASCII_RANDOM_TRANSITION` | `(2.0, 8.0)` | Random driver: seconds each change takes, drawn per change, eased; `(0, 0)` jumps |
| `LOCAL_ASCII_LINE_SPACING` | 1.05 | As the web viewer |
| `LOCAL_ASCII_FONT` | Schoolbook Mono Black | Any font Pillow can open |

The range is a look decision: set it by eye on the display it runs on.

## Needs

OpenGL 3.3 or GLES 3 (it falls back to the images on a legacy GL2 context,
with a warning), and Pillow able to read WOFF2 (FreeType built with Brotli:
pip Pillow and Debian's are). Without the font it uses Pillow's default font.
