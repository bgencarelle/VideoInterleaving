# Interleave Schoolbook Mono (`/ascii`)

This font remains in use. The later Canvas/atlas renderer and fresh browser
resource comparison are documented in
[`ASCII_CANVAS_RENDERER.md`](ASCII_CANVAS_RENDERER.md); this file's performance
table records the preceding DOM-only font change.

User-authorised replacement for the local proportional Schoolbook font trial.
The web viewer now bundles a renamed monospace derivative of **URW C059**, the
free Century Schoolbook revival. It does not bundle proprietary Century
Schoolbook, and does not depend on fonts installed on a visitor's computer.

## Font changes

- Three real faces: regular (400), bold (700) and black (900). Black is the bold
  outline with a round 25-unit-per-side offset (`BLACK_EMBOLDEN`), built with
  skia-pathops. It gives weight 900 real ink without the per-glyph stroke, which
  roughly halved fallback-path browser CPU in a same-session check (stroke 1 on
  bold: ~28.5% of one core; black, no stroke: ~21.9%; neutral: ~21.4%). All
  faces share the **981/1000 em advance**. Black's W overhangs the cell by 18
  units per side; the grid clips per row, not per glyph. Other control weights use
  the browser's normal weight selection; this is not a variable font.
- Ink is horizontally centred in each cell. The original W advance accommodates
  the entire ASCII alphabet in both faces, so **no glyph needs compression,
  stretching or outline scaling**. Vertical coordinates/baselines are unchanged.
- Kerning/substitution tables are removed; CSS also disables kerning/ligatures.
  The result cannot change cell placement for pairs such as `fi` or `AV`.
- Original CFF outlines are redrawn as unhinted Type 2 charstrings with translated
  coordinates. Geometric outlines are exact; original per-glyph hints are not
  retained. Pixel appearance is intentionally different because glyph centring
  and font rasterisation changed. This is not a screenshot-identical font swap.
- Both the OpenType hmtx and **CFF charstring widths** are fixed. The inherited
  CFF nominal width is reset, rather than accidentally adding it to the new
  advance. Browser sanitizers can use CFF widths independently of hmtx.
- Coverage is U+0020–U+007E, including the full application ASCII palette.
  Unsupported characters use the browser's monospace fallback; this is not a
  claim of equal cell widths for arbitrary Unicode or emoji.

## Browser changes

`static/ascii_viewer.js` uses `Interleave Schoolbook Mono, monospace` and loads
both faces before initial fitting. The production renderer has a declared
monospaced fast path: one initial cell measurement, one constant row/container
spacing, **no glyph-width probes/cache**, no per-span letter-spacing corrections,
and run splitting **only on adjusted RGB colour**. The previous proportional
path remains available for private diagnostics; it is not used by this font.

This preserves the shared 90x60 source, centre-nearest zoom selection, RGB
colours, glyph identities, clock/visibility lifecycle and all local controls.
No backend grading, resolution, or frame rate was changed. Browser picture
defaults are now the user-approved look: weight 900 (Black face), extra stroke 0,
line spacing 1.05, neutral tones, black background, 60x40 initial view and visible
controls.
Plain `nc`, its sweep and all other application modes are untouched.

Existing test workers need no restart: Python already reads the viewer template
and static assets on request. Reload `/ascii` to get the new font/controller;
existing open tabs retain their old loaded code until reloaded. No nginx,
service, routing or original `gencarelle.fun` deployment changes are involved.

## Free source, licence and rebuilding

Assets: `static/fonts/schoolbook-mono/`. Upstream C059 sources are pinned to
Artifex URW base35 commit `3c0ba3b5687632dfc66526544a4e811fe0ec0cd9` with
SHA-256 checks enforced by `utilities/build_schoolbook_mono.py`.

Fonts and modification tool are distributed under AGPL-3.0; upstream's
PostScript/PDF font exception is included unchanged. Attribution/copyright
remain in the font. `source/LICENSE`, `source/COPYING`, original OTFs and build
notes are shipped, as is a reproducible **corresponding-source ZIP** containing
the original inputs and modification script. The controls panel links to the
licence and source download. These are font-specific licensing notices, not a
claim to relicense the surrounding application.

Build-only dependencies are pinned in `source/requirements.txt` (fontTools
4.67.0, Brotli 1.2.0 and Zopfli 0.4.3). The application never imports
fontTools or compiles fonts at startup. Rebuild instructions are in
`static/fonts/schoolbook-mono/source/README.md`. Generated WOFF2 files are small,
normal shipped web assets, not a video bake. Their one-time download does not
change the RGB stream's per-frame network workload.

## Validation

Font tests check all 95 printable ASCII advances in **both hmtx and CFF**,
regular/bold agreement, font names/fixed-pitch flags, outline bounds/centring,
absence of kerning/substitution and exact translated outline commands against
upstream. Rebuilding produces identical WOFF2, manifest and source ZIP bytes.
HTTP unit checks cover both `/ascii/static/` and the existing nginx-compatible
`/asciistatic/` spelling for fonts and source/licence files.

Actual Chromium replay verifies equal measured widths across all 95 characters
at every control weight 100–900, on DPR 1 and 2. Eighteen screenshot/control
cases cover 60x40, 90x60 and 15x10, at weights 400/700/900. **123 actual replayed
frames** retain exact selected glyphs and RGB colours in the DOM. Checks also
assert consistent row extent, no per-span spacing, copy/reset/hide/restore
controls and encoding-only WebSocket messages (no client resize requests).

The final focused regression passed **69 Python/Node-backed tests**. Public
Chromium verified animation, both WOFF2 downloads (200), default weight 400 /
stroke 0, hide/restore and no page errors at
`https://testing3.gencarelle.fun/ascii`. The public source ZIP and licence
downloads matched local bytes exactly. The existing web/WS worker and terminal
sweep retained their original PIDs/listeners; no worker restart was needed.

Visual inspection of the 60x40 regular screenshot confirms recognisable,
full-picture Schoolbook ASCII with the newly centred fixed cells. These checks
are not a substitute for the user's judgement of the changed typography.

Artifacts: `tmp/schoolbook_mono/{before,check,perf}/`,
`tmp/schoolbook_mono_check.py`, `tmp/schoolbook_mono_bench.py`, Python/Node test
logs and font metrics. The private browser benchmark compares the saved
pre-change frontend, this frontend and the original baseline in fresh Chromium
processes, two reverse-order 15-second observations each, same preserved inputs
and viewport as earlier baseline work. Original baseline uses its indexed bake;
before/mono use the identical actual RGB/gzip replay. Only before/mono is an
equal-source font/presentation comparison; original versus current remains a
different-presentation workload comparison, not equal-quality evidence.

### Resource result

Same-cohort averages, 100%=one core, memory=sum of browser-process PSS:

| Viewer | CPU | PSS | Updates/s | Styled runs |
|---|---:|---:|---:|---:|
| Original default | 45.8% | 202.0 MiB | 15.00 | 846 |
| Saved proportional Schoolbook default | 106.8% | 260.9 MiB | 15.09 | 2226 |
| New monospace Schoolbook default | 95.2% | 241.8 MiB | 15.06 | 2212 |

The font/presentation change saves **about 11% CPU and 7% PSS** against the
same-source proportional Schoolbook viewer in these software-rendered Chromium
runs. Style recalculation fell from 167.8 to 112.3 ms/s; layout stayed roughly
182–184 ms/s. Run count barely fell: varying RGB colours still split most cells.
The synthetic fixed-tint test now produces one run per row, but removing real
colour variation is NOT an accepted production optimisation.

**This is not baseline parity:** actual current still uses about **2.08x** the
original-default browser CPU and ~20% more PSS in this cohort. Both default
presentations differ in source detail/font/RGB representation. Do not subtract
these observations from a different cohort's earlier 43.6/95.5% measurements.
This task removes per-glyph spacing work; it does not solve all browser painting
cost or reduce network traffic. User typography approval remains separate.
