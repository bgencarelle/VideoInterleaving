# Browser resolution zoom (`/ascii`)

The current drawing path is described in
[`ASCII_CANVAS_RENDERER.md`](ASCII_CANVAS_RENDERER.md): hardware glyph atlas,
native Canvas on software GPUs, and DOM compatibility fallback. The older DOM
optimisation measurements below remain historical, not current resource claims.

## Contract

- Initial view remains the configured `ASCII_WIDTH`/`ASCII_HEIGHT` (currently
  60x40). Reset returns to this size, not to the maximum.
- Zoom **out** increases resolution through small integer steps to 90x60.
  Zoom **in** decreases resolution to 15x10. The whole picture stays fitted to
  the same area: this is not a crop or magnification of a detailed image.
- Wheel/trackpad, pinch, +/- keys and on-screen buttons control resolution.
  `0` or Reset restores the default. Viewport resizing only fits the picture;
  it does not change the chosen resolution.
- Zoom state is local to a tab, survives WebSocket reconnect, and sends NO
  per-viewer resize commands. Reloading restores the configured default.
- Plain TCP ASCII and its temporary resolution sweep are independent and
  unchanged by browser zoom.

Current font: bundled **Interleave Schoolbook Mono**, the user-authorised free
URW C059 derivative. It removes per-glyph spacing corrections; see
[`ASCII_SCHOOLBOOK_MONO.md`](ASCII_SCHOOLBOOK_MONO.md) for source/licence,
rebuilding, visual changes and measured validation. Tone/weight/stroke defaults
remain neutral and the RGB source is unchanged.

The earlier user-requested web font trial used local Century Schoolbook, falling
back to Century Schoolbook L, C059, then the browser's serif font. No proprietary
font was bundled, so availability depended on the viewer's computer, not the
server's fonts. Both initial font measurement and xterm used this same family
list. This was a proportional-font experiment inside fixed terminal cells;
spacing/clipping needs visual judgement in the actual browser. Grid dimensions,
colour grading, source stream and nc output are unchanged by the font trial.

User subsequently clarified a grey GRID background, not merely faint glyph
strokes. Disabled terminal selection overlays (transparent active/inactive
selection theme, clear selection on selection-change/zoom, prevent pointer
gesture defaults) and explicitly kept terminal surfaces black. This prevents
zoom gestures/dragging from tinting the read-only picture with a selection
overlay; it is a candidate fix, not a visually confirmed diagnosis. Foreground
colours/exposure/contrast/font weight were not changed. A 45-frame raw tint
check contained full 255/255/255 white and 5214 near-white instructions, but
raw values do not diagnose browser paint. Tests include pointer-default
prevention and selection clearing; user-browser confirmation remains needed.

The user's screenshot then established the actual background defect: bundled
xterm uses Tango ANSI palette black `#2e3436`, while the browser encoder emitted
SGR 40 (palette black). Setting only theme.background/CSS to black did not
change that cell background. Fixed by using SGR 49 (default theme background)
in browser output AND explicitly setting theme.black to `#000000`. Both scripts
have cache-busting query values for this fix. Tests assert the output uses 49,
not 40, and both theme background/black are zero. The earlier font-weight and
selection-overlay explanations were not the cause demonstrated by this image.
No source foreground RGB, contrast/exposure, font weight or nc encoder changed.

## Local picture controls (defaults not yet selected)

User requested numeric adjustment controls and a boolean for hiding them. The
Picture controls panel now provides paired sliders/number fields for foreground
brightness (0..3), contrast (0..3), gamma (.25..3), saturation (0..3), and font
weight (100..900, steps of 100). Initial tone values are all 1, font weight 400:
the existing picture is unchanged until a control is adjusted. These are
CLIENT DISPLAY adjustments on the already graded shared master, not backend
grading settings and not a new server render. Glyph choices remain unchanged.

Tone order: luminance-based saturation, channel contrast about 127.5, brightness
gain, then exponent 1/gamma. Gamma greater than 1 lifts midtones. Clamping and
8-bit rounding occur at output. Every redraw uses the CURRENT original master
colours, never a previously adjusted picture, so changes cannot accumulate.
Neutral tone controls bypass transformation exactly; active colour work/cache
is bounded to unique selected colours in a single picture. Background is not
processed by the foreground transform.

Background has a colour picker, hex readout and a 0..100 lightness slider with
numeric input. With the initial achromatic colour the slider runs black to
white. After picking a hue it changes HSL lightness while retaining that hue
and saturation, including across the black/white endpoints. The terminal theme
and CSS surface background both track the selected colour; SGR 49 continues
to use that theme background, not the Tango ANSI-black slot. Letterboxing
outside the terminal remains black.

`ASCII_VIEWER_CONFIG.hideControls` is an actual boolean (initially false).
The Hide controls checkbox and H key hide/show BOTH picture and zoom panels;
one small Show controls button remains for touch/keyboard accessibility.
Keyboard H ignores editable input fields. Hiding controls does not pause the
animation or change picture settings. Reset picture restores the initial
appearance only; resolution has its own existing Reset/0 control.

Copy settings copies JSON containing resolution, appearance values and the
hideControls boolean, and shows a selectable JSON text box if clipboard access
fails. The visible preview stays current; changing settings clears the Copied
status. Settings survive socket reconnect in the same tab but are NOT silently
saved to browser storage or written as application defaults. Reload restores
the configured starting values. When the user asks to adopt values later, these
display settings belong in the viewer's config; do not apply them a second
time to backend grading.

All controls are local: no additional WebSocket requests, server resampling,
grading changes, source-frame alterations or nc changes. Node tests cover
neutral identity, black/white endpoints, transform math, non-mutating source
colours, per-frame cache bounds, paired numeric fields, background hue retained
across black, theme/CSS synchronization, font weight, export/reset, boolean
startup hiding, checkbox/H/restore button and unchanged server defaults/traffic.
Public page and all three scripts verified; evidence:
`tmp/ascii_web_zoom/appearance_controls_verified.json`. Real-browser visual
inspection is still the user's check, rather than proven by the DOM stubs.

User's current dial-in candidate is 60x40, brightness 1.23, contrast/gamma 1,
saturation 1.04, font weight 900, background #000000, hideControls false. These
are NOT adopted defaults: user asked whether weight could go higher, not to
save the candidate. A numerical weight beyond 900 need not select a heavier
Schoolbook face. Added Extra stroke (`fontStroke`, 0..1 px, .01 steps), default
zero, using currentColor CSS text stroke on this bundle's DOM-rendered glyph
spans. This increases ink coverage without altering tint, background, glyph
identity, cell spacing or server rendering. It remains a visual experiment;
stroke can crowd neighbouring letters at high values. Suggested first trial
is .15–.30 px with weight 900. Reset/export/hide controls include the new field;
tests verify its neutral colour transform, CSS value, numeric field and reset.

User then requested line spacing with compensating scale. Added `lineSpacing`
slider/number, .50..2.00, default 1.00. This changes row pitch, not source grid
dimensions or server data. This bundle uses the DOM renderer; its row origins
are translated relative to ORIGINAL cell geometry, avoiding xterm's lineHeight
minimum of 1 and avoiding feedback across repeated fits. For n rows of original
height h, visible height is h/n * (1+(n-1)*lineSpacing), retaining the full first
and last row boxes. The fixed picture box is refitted to that visible height,
so tighter lines fill the same viewing area rather than shrinking the picture.
At very tight settings glyphs can overlap. Row translations are reapplied after
resizing/font changes/writes; Reset clears them. Future non-DOM renderer addons
would need an equivalent implementation (fallback currently uses native pitch).
Tests cover min/max/default row geometry, last-row containment, stable repeated
fit, numeric changes, fixed fitted height, export and reset. No image colour,
source resolution, server default or network request changes. Public scripts
and control verified; user-browser appearance is still the visual authority.

## Browser CPU/RAM optimisation

User explicitly requested reducing browser CPU and memory (separate from the
previous 30.4%/376-MiB SERVER observation). Captured 120 consecutive actual
90x60 web master pictures and saved the previous browser implementation before
editing. Playwright and Chromium headless shell were installed ONLY under
gitignored `tmp/browser_tools` and `tmp/pw-browsers`, not application dependencies
or OS packages. No services, routing, source grades, defaults or nc settings
changed for these tests.

The default page now uses `AsciiGridRenderer`, not xterm's terminal buffer/parser.
It retains/reuses DOM text runs and changes only different text, colour or
spacing. Proportional-font run grouping, glyph advance measurement, row clipping,
cell geometry and DPR rounding match the bundled xterm DOM implementation.
Thus Schoolbook, extra stroke, background and line spacing keep the same
appearance. Font measurement is batched/cached (bounded to 512 glyphs), source
sampling indices are cached for the current grid, and rows have bounded layout
containment. Geometry refits happen only on geometry/control/viewport/DPR
changes, not on every incoming picture. The page no longer loads xterm JS,
terminal-buffer machinery, text-input helpers or unused terminal stylesheets.
Other pages can still use the original static assets; none were deleted.

ANSI decoding is a single-pass bounded scanner rather than regex token arrays,
split/map arrays and per-instruction RGB slices. Two decode buffers are reused;
malformed data is decoded into scratch, never into the currently displayed
master. Source colours/glyphs are unchanged. The browser no longer converts
the selected grid back to ANSI for a second parser. Gzip retains at most one
in-flight and one pending frame; its input view avoids an extra ArrayBuffer copy.
Hidden tabs close the socket, discard pending work and keep only current
display state. On visibility return they reconnect to the CURRENT stream with
local controls intact, not queued history.

### Measured results (isolated software-rendered Chromium)

Same 120-frame moving replay at 15-FPS source cadence, same 1000x800 viewport,
DPR 1, 1.8-second warmup and 12-second observation per case. CPU is SUMMED
browser/renderer/GPU/utility process CPU on a one-core convention; PSS sums
proportionally allocated resident pages, avoiding RSS shared-page double-count.
These are headless sandbox results, NOT user-device CPU/RAM or server CPU/RAM.

| Browser case | CPU before -> after | PSS MiB before -> after | Post-GC JS heap MiB before -> after |
|---|---:|---:|---:|
| Default 60x40 | 117.2% -> 98.7% | 404.6 -> 245.9 | 2.63 -> 1.57 |
| Tuned 60x40 | 120.0% -> 102.5% | 429.7 -> 263.9 | 2.68 -> 1.59 |
| Maximum 90x60 | 133.2% -> 133.3% | 631.3 -> 396.3 | 3.06 -> 1.76 |

At 60x40 this is about 15–16% less whole-browser CPU and 39% less PSS, with
about 40% less retained JS heap. Summed RSS was 586 -> 429 MiB in the default
case, but is not the PSS metric above. The 90x60 case remains CPU-SATURATED in
this software-rendered sandbox; CPU did NOT decrease there. Completed renders
in the measurement harness increased from 104 to 129 (about 24%) while memory
fell about 37%; do not report maximum mode as having solved its CPU limit or
achieving 15 FPS in this environment. Decode CPU was about .7–.9 ms/call versus
5–6 ms/call before. No bitrate, FPS setting, colour precision, source resolution
or font-quality reduction was used to achieve the resource improvement.

### Validation and reproducibility

- Old/new parsers exactly match all 120 actual moving 90x60 pictures.
- 24 before/after screenshot pairs have ZERO changed pixels: eight conditions
  for each of three actual pictures at DPR 1, 2 and 1.5 respectively. Conditions
  cover default/tuned, 90x60, 15x10, line spacing .5/2, coloured background and
  weight 900 plus extra stroke. These establish tested Chromium parity, not a
  claim about untested fonts/browsers or every possible image.
- Node renderer tests assert stable span reuse after warmup, no mutations for
  an unchanged picture, bounded row/run state, colour transforms, resize, DPR
  updates and disposal. Controller tests include buffer reuse, scratch parsing,
  hidden-tab close/reconnect and the existing controls/latest-only checks.
- Benchmarks: `tmp/ascii_browser_perf/{before,after_final}/report.json` and
  `tmp/ascii_browser_bench.py`. Golden images/state/diffs:
  `tmp/ascii_browser_perf/parity*/`. Actual replay inputs and original assets
  are preserved under `tmp/ascii_browser_perf/`; never commit generated results.
- Temporary browser flags disable Chromium's local-network access checks ONLY
  for the isolated localhost replay harness (intercepted HTML otherwise has an
  unknown address space). No application security settings were changed.

Run the persistent renderer test with:

```sh
node tests/ascii_grid_renderer.test.js
.venv/bin/python -m unittest tests.test_ascii_resolution_zoom -v
```

## Shared work, not per-viewer server rendering

`main.py --mode asciiweb` enables `ASCII_WEB_SOURCE_SIZE=(90,60)` without
changing the configured initial view dimensions. Live SBS uses one directly
sampled, font-aware GL grid, independently selected/composited live layers and
one grading/glyph conversion per publication. Pre-baked ASCII sources use the
existing live layer merge followed by one shared 90x60 grid selection. There
are no pre-mixed layer bakes or per-client image-render targets.

Every viewer receives the same master picture. `static/ascii_resolution.js`
decodes its RGB/indexed ANSI cells. `static/ascii_grid_renderer.js` selects
centre-nearest glyph/colour pairs at the chosen grid and updates persistent
coloured text runs directly. There is no blur, averaging,
RGB reconstruction or new glyph-brightness grading in the browser. This is
NOT equivalent to rendering each selected size directly from the original
source: a low-resolution view is a subset of the shared 90x60 glyph lattice.
Default grid DIMENSIONS are preserved; exact cell parity with the old direct
default-sized render is not promised. A bake cannot recover detail absent
from its original glyph grid merely by selecting a larger browser size.

The browser holds two reusable decode buffers (current and scratch) and one
latest paint state, rather than queuing animation history. Compressed
input likewise has one decompression in flight and one replaceable pending
frame. Zooming redraws the cached current master without a network round trip.

## WebSocket compatibility and compression

Connections receive the existing clear sequence followed by a JSON source
dimension message: `{"type":"resize","cols":90,"rows":60}`. This describes
the MASTER, not a request to change the tab's selected resolution.

Plain messages remain full ANSI frames. Browsers with native
`DecompressionStream` advertise `{"type":"encoding","encoding":"gzip"}`
once per connection. Their binary frames contain byte `g` followed by a gzip
compressed UTF-8 ANSI frame. `ascii_web_server.py` compresses once per
publication and reuses the SAME bytes for every gzip viewer. Older browsers
continue receiving plain ANSI. No new Python/browser dependency is required.

Smaller browser grids reduce browser drawing work but **do not reduce network
bytes per viewer**: all resolutions receive the shared maximum stream. Gzip
reduces that stream losslessly, independently of selected resolution. In one
30-second normal-default live-SBS run, simultaneous clients received 9.33 Mbps
plain versus 2.79 Mbps gzip, approximately 14.9 FPS. There were 445 identical
decoded frame hashes across the two clients. These are WebSocket payload
counts, excluding network framing/headers; they do NOT use the temporary
heavy-colour nc profile and do NOT establish the separate 0.5-Mbps nc target.

## Public asset routing

The page loads its scripts, styles and Unscii font under `/ascii/static/` so
it does not rely on the site's generic `/static/` upstream. The HTTP handler
supports both this namespaced path and the existing proxy's concatenated
upstream spelling `/asciistatic/`, with the SAME static-directory traversal
checks. The font URL in terminal.css is relative. No nginx configuration,
VideoInterleaving checkout or reference deployment is modified.

## Validation and limitations

Focused Python/Node tests exercise all 51 row counts, RGB/indexed decoding,
exact source-cell selection, full-size identity, defaults/endpoints, wheel,
pinch, buttons/keys/reset, reconnect, latest-only xterm writes and gzip
decompression, shared compression identity/count, ignored viewer resize
requests, one master GL target, and asset path/traversal handling.

Initial live checks: two simultaneous WebSocket clients (text and gzip), actual
captured ANSI through all 51 browser resolutions, and external page/assets/WSS
checks at testing3.gencarelle.fun. Evidence is in `tmp/ascii_web_zoom/` and
`tmp/ascii_resolution_web_ready.log`. The later browser-resource task added
isolated Chromium replay/pixel tests (see below); those supersede the initial
DOM-stub-only limitation, but are still not measurements of the user's browser.

Run from the repository root:

```sh
.venv/bin/python -m unittest tests.test_ascii_resolution_zoom tests.test_ascii_web_broadcast -v
node tests/ascii_resolution.test.js
```
