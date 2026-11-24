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
viewer anything. Since only the grid is ever read from the picture, JPEGs
are decoded smaller (TurboJPEG's 1/2, 1/4 or 1/8) while each keeps at least
`LOCAL_ASCII_DECODE_ROW_PX` (4) pixels per finest row: less decoding and
far less to upload, for the same picture. Per picture the GPU work is one small grid render and a
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
  random driver; `0` hands control back, and so does leaving them alone for
  `LOCAL_ASCII_MANUAL_TIMEOUT` (30 s; 0 stays manual until `0`), from
  whatever size it is at.
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
black between them and the picture goes dark. Two arguments counter it
(defaults 0.8 and 0.5; 0 turns either off):

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

The defaults are the ASCII service's starting point, equal to
`--mode asciilocal --ascii-grid 160x120 --ascii-glow .8 --ascii-strokes .5`,
so `--mode asciilocal` alone gives that look.

## The biography

The palette is built from the subject's biography (`constantStorage/
bio_constants.py`): the letters of BEN GENCARELLE, the digits of 1978 and
the date's separators, with filler letters for the density range.
`--ascii-bio` makes it readable, in every ASCII mode (default: as it is):

- `palette`: only the biography's characters (`B8GRNEALencra971-` and the
  blank), in the palette's own density order.
- `runs`: the phrase, name then birth date (`BEN GENCARELLE 11-17-1978`,
  then `BIRTH_PLACE` when set),
  runs along the rows through the picture; the faintest cells
  (`ASCII_BIO_RUNS_KEEP`) keep their palette character, so the tone and
  edges survive, and colour carries the rest.
- `reveal`: the palette as usual, but every `ASCII_BIO_REVEAL_EVERY`
  (6-20 s) the phrase surfaces on one line for `ASCII_BIO_REVEAL_FOR`
  (2-5 s).

Here the phrase is applied to the grid as shown, so it reads at every zoom
size. The browser viewer resamples the server's master grid itself, so
there `runs` and `reveal` read only at the master size; other sizes pick
every few letters.

## Receipt printer

What is drawn in the window can be printed, in `--mode local` and
`--mode asciilocal` alike:

- **P** prints the frame to the computer's own default printer: whatever
  CUPS has as its default on Linux and macOS (`lpstat -d`, sent with `lp
  -o fit-to-page`), the default printer on Windows. With no system printer
  it goes to the receipt printer when `--printer` is on; otherwise P is
  ignored.
- `--printer` drives the register's Epson TM-T88V (MICROS-branded,
  Ethernet interface, raw ESC/POS on port 9100): random prints
  `PRINTER_EVERY` apart (2-15 min; `--print-every 2-15` in minutes) and a
  **double tap** on the touchscreen (or a double click). Without the flag a
  stray tap prints nothing.

```bash
python main.py --mode local --printer
python main.py --mode asciilocal --printer --print-every 2-15
python main.py --mode asciilocal --printer text
```

What prints is the picture as it stands on screen. With `PRINTER_PAPER =
'white'` (default) the frame is drawn again off screen on white: the
background is the paper and the picture keeps its tones (a photo prints
positive; ASCII letters print in ink as dark as they are bright on screen,
so the near-black background letters fade away; no glow). `'black'` reads
the window exactly as shown, which for these mostly-black frames is nearly
solid ink. The print is cut to the picture's own rectangle (no letterbox
bars) and turned back by the R rotation, so it stands upright as the
picture does, then fills the paper: the receipt's full 576-dot width
(`PRINTER_ROTATE = 'auto'` instead turns a landscape picture to run along
the roll), or the page, fitted without cropping, on a landscape page when
the picture is wide. Receipts are dithered (Floyd-Steinberg), sent as
raster lines (GS v 0), then fed `PRINTER_FEED_LINES` (10) blank lines and
cut.

Dense pictures are the hard case for the TM-T88V: it wants a 24 V supply
of at least 2 A, and a picture with large dark areas draws the most
current. A printer that resets part way through loses its place in the
picture and prints the rest of the data as text (rolls of garbage). So
pictures print slowly (`PRINTER_SPEED`, 2 of 1-13: fewer dots heated at
once), black burns at most `PRINTER_INK` (85%) of its dots, and the
picture goes in 32-line bands, so after a reset at most one band (2.3 KB)
can turn to garbage before the next band's header starts a fresh picture
command. If a printer still resets on dark pictures, check its supply.

`--printer text` (asciilocal) prints the characters on screen in the
printer's own font B instead: 64 columns, a 9 x 18-dot cell (the screen's
0.5), background left blank, cells too dark to see on screen
(`PRINTER_DARK`) left out, the `--ascii-bio` phrase spelled again after
any shrinking so words stay whole, narrow frames magnified to fill the
width.

Converting and sending run on their own thread, one print at a time; a
printer that is off or unreachable only logs a warning. A print is never
abandoned half sent (the printer would read the next job as the rest of
the picture and print garbage): only connecting has a short time limit. The register
reaches it at `PRINTER_HOST` (172.18.35.171, the address it kept from the
restaurant) through the NetworkManager connection `printer`: `enp1s0`,
172.18.35.180/27, no gateway, so internet stays on Wi-Fi. That connection
returns at every boot.

## Settings (settings.py)

| Setting | Default | Meaning |
|---|---|---|
| `PRINTER` | False | `--printer`: random prints and the double tap on the receipt printer (P prints without it, to the system printer) |
| `PRINTER_MODE` | `'screen'` | `'screen'`: the window's pixels; `'text'`: the ASCII characters (`--printer text`) |
| `PRINTER_HOST` / `PRINTER_PORT` | `172.18.35.171` / 9100 | The receipt printer's address |
| `PRINTER_EVERY` | `(120, 900)` | Seconds between random prints; `--print-every` in minutes |
| `PRINTER_PAPER` | `'white'` | Draw the frame again on white for paper, tones kept; `'black'`: as on screen |
| `PRINTER_DOTS` | 576 | Screen: dots across (80 mm paper) |
| `PRINTER_ROTATE` | None | Receipts: upright, full width; `'auto'` turns a landscape picture to run along the roll |
| `PRINTER_FEED_LINES` | 10 | Receipts: blank lines fed before the cut |
| `PRINTER_SPEED` | 2 | Receipt pictures: print speed 1 (slowest) - 13; None = the printer's own |
| `PRINTER_INK` | 0.85 | Receipt pictures: share of dots burnt where the picture is black |
| `PRINTER_COLUMNS` | 64 | Text: font B columns on 80 mm paper |
| `PRINTER_DARK` | 32 | Text: cells darker than this on screen print blank; 0 prints all |
| `LOCAL_ASCII_DECODE_ROW_PX` | 4 | JPEG decode keeps at least this many pixels per finest row (1/2-1/8 size); 0 = full size |
| `LOCAL_ASCII_SHAPE` | `'picture'` | Grid shape: `'picture'` (whole image, as large as the window allows), `'window'` (fills it, image cropped), `'grid'` (browser's, letterboxed) |
| `LOCAL_ASCII_GRID` | `(160, 120)` | Master (finest) grid `(cols, rows)`; `--ascii-grid` overrides. None = 120x80 x multiplier. `'picture'`/`'window'` use only its rows |
| `LOCAL_ASCII_MULTIPLIER` | 1 | Master = 120x80 x this, when no grid is given |
| `LOCAL_ASCII_MULTIPLY` | `'fine'` | `'fine'`: only the finest end grows; `'all'`: coarsest and default sizes grow by as much as the master |
| `LOCAL_ASCII_ROWS` | `(None, None)` | Allowed row range; None = 10 / the master's rows |
| `LOCAL_ASCII_HOME_ROWS` | None | Size at rest and after reset; None = `ASCII_HEIGHT` |
| `LOCAL_ASCII_MANUAL_TIMEOUT` | 30 | Seconds after the last wheel/key zoom before random zoom resumes; 0 = until `0` |
| `LOCAL_ASCII_ZOOM` | `'random'` | `'random'` or `'manual'` |
| `LOCAL_ASCII_WALK_RATE` | 0 | Mouse/keys: rows per second toward a new size; 0 instant, as the browser |
| `LOCAL_ASCII_RANDOM_TRANSITION` | `(0.5, 3.0)` | Random driver: seconds each change takes, drawn per change, eased; `(0, 0)` jumps |
| `LOCAL_ASCII_HOLD` | `(2.0, 20.0)` | Random driver: seconds between changes, at least / at most, counted from each pick |
| `LOCAL_ASCII_CROSSFADE` | False | Cross-fade neighbouring grid sizes mid-change instead of stepping |
| `LOCAL_ASCII_LINE_SPACING` | 1.05 | As the web viewer |
| `LOCAL_ASCII_GLOW` | 0.8 | `--ascii-glow`: neighbouring cells' colour under small letters |
| `LOCAL_ASCII_GLOW_FROM` | 32 | Cell height (pixels) where the glow starts |
| `LOCAL_ASCII_GLOW_FULL` | 8 | Cell height (pixels) from which it is at full strength |
| `LOCAL_ASCII_STROKES` | 0.5 | `--ascii-strokes`: keep small letters bright |
| `LOCAL_ASCII_FONT` | `static/fonts/century-schoolbook/CenturySchoolbookBold.ttf` | Path to any TTF/OTF/WOFF2 Pillow can open, from the repo root or absolute |

The range is a look decision: set it by eye on the display it runs on.

## Needs

OpenGL 3.3 or GLES 3 (it falls back to the images on a legacy GL2 context,
with a warning), and Pillow able to read WOFF2 (FreeType built with Brotli:
pip Pillow and Debian's are). Without the font it uses Pillow's default font.

## Running it as a service

`scripts/setup_app.sh` writes `vi-asciilocal` next to `vi-local`: a user
service (recommended for a GUI) and a system fallback, both running
`main.py --mode asciilocal --test` with the settings.py defaults (the same
as `--ascii-grid 160x120 --ascii-glow .8 --ascii-strokes .5`), logging to
`vi-asciilocal.log` in the project folder.

    systemctl --user enable --now vi-asciilocal

It conflicts with `vi-local`, since both take the screen: enable one of
them. For receipts, add `--printer` to `ExecStart` in
`~/.config/systemd/user/vi-asciilocal.service`, then
`systemctl --user daemon-reload && systemctl --user restart vi-asciilocal`
(running setup again rewrites the file).
