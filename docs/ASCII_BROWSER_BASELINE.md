# Original-browser baseline comparison

This records the **pre-monospace Schoolbook** default comparison. The subsequent
user-authorised C059 monospace derivative and fresh before/after/original
measurements are documented in [`ASCII_SCHOOLBOOK_MONO.md`](ASCII_SCHOOLBOOK_MONO.md).
Keep cohorts separate; do not silently substitute their numbers into this table.
The subsequent Canvas/atlas work has its own fresh original/DOM/new-renderer
cohort in [`ASCII_CANVAS_RENDERER.md`](ASCII_CANVAS_RENDERER.md).

This compares the original `/ascii` viewer from the read-only checkout
`/home/linuser/VideoInterleaving` with the current viewer. It is NOT the earlier
"first zoom implementation versus optimised zoom implementation" comparison.
No original services, configuration, source files or deployment are changed.

## Why two kinds of comparison are necessary

The original viewer uses Unscii, receives a configured 60x40 baked ASCII grid,
and uses xterm. The current production-default viewer uses Schoolbook, receives
a shared 90x60 RGB master, shows 60x40 by default, and supports local picture
controls and zoom. Comparing those as-is measures the whole feature/appearance
tradeoff; it does NOT isolate renderer efficiency or establish equal quality.

The harness therefore measures:

1. **original60**: unchanged original template/static assets, 60x40 indexed
   frames produced by its actual baked-source worker, normal Unscii font.
2. **matched_current60**: current renderer, EXACT SAME 60x40 frame strings,
   same Unscii font file and same physical picture area. The shared source is
   intentionally 60x40 here, not the production 90x60 master. This is an
   explicitly modified diagnostic served only by the isolated test harness.
3. **original90**: original viewer configured 90x60, fed actual 90x60 RGB
   master pictures. This is a larger workload supported by the original
   viewer, NOT a claim about the deployed baseline's default output.
4. **matched_current90**: current renderer with the same 90x60 input, Unscii
   and physical picture area as original90.
5. **actual_current60**: unmodified current browser code/defaults: Schoolbook,
   60x40 view, shared 90x60 master and negotiated gzip, controls visible.
   This uses the preserved actual live RGB capture cohort, not the old bakes.

The renderer-only pairs use uncompressed text in both viewers. That isolates
renderer cost rather than charging gzip/inflate to just one participant.
The actual-current case includes gzip/inflate. Source samples are moving in
every case; none are a static still. No frame-rate/quality cut is used.

## Source evidence and matching

Original checkout commit: `2df1a6723ae99405422f8eaa367dd343926e26d1`.
Relevant working-tree template, assets, settings and worker match that HEAD.
`tmp/ascii_og_prepare.py` saves the original files and SHA-256 fingerprints
under `tmp/ascii_og_browser/original/` (generated artifacts, never commit).

The baked input is 120 pictures from original first face folder
`images_ascii/face/00_C_BG_faceSource_960`, indices 0,2,...,238, no float overlay.
The original `image_display.py::load_and_render_frame` is extracted by AST and
executed as-is with the actual bake arrays and original 16->235 ANSI mapping.
No main.py import, GL, original server, image re-grading or replacement encoder
is involved. Inputs have 90x60 bake cells; the original worker's exact
cover/linspace selection produces 60x40 wire frames with 2400 indexed escapes
per frame, mean 28,496.6 bytes. This matches the expected baked-path structure,
not the duplicated-column legacy live-SBS path previously benchmarked by mistake.

The 90x60 RGB input is the same preserved 120-picture live capture used for the
previous browser optimisation study. Original90 and matched_current90 receive
identical UTF-8 ANSI strings. Actual_current60 receives the corresponding
existing gzip packets; source cells/colours are not reconstructed from bakes.

Font and picture area also matter: the original single-axis CSS fit and the
new fixed-extent fit do not necessarily yield the same physical width even at
the same rows/columns. For MATCHED cases only, the test intercepts the current
JS font constant to use the original Unscii, and the fitted-aspect constant to
match the original's measured picture bounds. It hides the current controls to
avoid obscuring the matched picture. These overrides never reach public pages.
Original code is not patched. Initial matched screenshots at both 60x40 and
90x60 are pixel-identical (648x800 pixels each) in the chosen Chromium setup.

## Resource measurement

The isolated replay uses real Chromium at 1000x800, DPR 1, 15-FPS source cadence,
2.5-second moving warmup, 15-second observations and two passes. Every case
starts a fresh browser process. The second pass reverses case order. Local
WebSocket per-message deflate is disabled: the real original server doesn't
negotiate it, and the current case already has application-level gzip.

CPU sums browser, renderer, GPU and utility process CPU time, 100% = one core.
PSS sums proportionally allocated resident memory; RSS is also recorded but
double-counts shared pages. Post-GC JS heap is recorded separately. Completed
renderer update counts are recorded so a saturated viewer cannot look cheaper
merely by displaying fewer frames. Startup/warmup/forced-GC work is outside the
observation interval. These are sandbox software-rendered Chromium measurements,
NOT the user's browser or server utilisation.

The recorded original60/matched60 inputs form a strict renderer comparison.
Original60 versus actual_current60 is a deliberately UNMATCHED full-default
comparison: different source detail, glyph/tint appearance, font, controls and
wire encoding. Do not attribute its whole difference to any single feature.
The 90x60 pair is another matched renderer diagnostic, not original deployment
resource usage. High-resolution CPU saturation must remain explicit.

## Results: two balanced 15-second observations per case

| Case | CPU, one-core convention | Browser PSS | Post-GC JS heap | Completed updates/s |
|---|---:|---:|---:|---:|
| Original default 60x40 | 43.6% | 204.6 MiB | 3.09 MiB | 15.0 |
| Current renderer, matched 60x40 diagnostic | 39.5% | 207.0 MiB | 2.09 MiB | 15.0 |
| Original viewer, matched 90x60 RGB diagnostic | 132.3% | 632.4 MiB | 3.11 MiB | 10.2 |
| Current renderer, matched 90x60 RGB diagnostic | 132.0% | 370.4 MiB | 2.39 MiB | 11.6 |
| Current site defaults, 60x40 view / 90x60 master | 95.5% | 255.5 MiB | 1.88 MiB | 15.1 |

**The actual current-default website is NOT cheaper than the original-default
viewer in this test.** It uses about 2.19x the CPU and 24.9% more whole-browser
PSS. The input/font/features differ, so this measures their combined deployed-
default tradeoff, not equal-detail rendering efficiency or a causal font-only
effect. These current defaults are neutral picture controls and Schoolbook
weight 400, NOT the user's privately dialled tab values (not adopted as defaults).

The strict 60x40 renderer comparison shows about 9.3% less CPU, about 1.2% more
PSS (essentially unchanged within this short test), and about 32% less retained
JS heap. Thus the earlier 39%-PSS improvement versus the FIRST ZOOM VIEWER must
NOT be advertised as a RAM saving over the original viewer. Original/current
CPU per pass was 40.9/38.3% and 46.2/40.7%; two short passes are not a universal
platform/resource guarantee.

Both 90x60 RGB diagnostics remain CPU-saturated; whole-browser CPU did not
materially decrease. The new renderer completed more updates and used less
PSS. Original90 PSS varied 775.7/489.1 MiB versus 374.5/366.3 MiB for current90;
the mean ~41% difference is a sample result, not a stable memory cap or a
guaranteed improvement for all streams. Neither viewer achieved the 15-FPS
source cadence in this maximum-size software-rendered test.

The finished runs' initial 60x40 and 90x60 matched screenshot pairs both have
ZERO changed pixels, not just nominally equal grid dimensions. Relevant
original source and asset SHA-256 hashes still match the read-only snapshot
after all tests. No original/current production code, default, public service
or routing was modified as part of this comparison.

## Artifacts

- `tmp/ascii_og_browser/preparation.json`: original commit, source hashes,
  selected bake folder/indices and byte counts.
- `tmp/ascii_og_browser/browser_report.json`: balanced per-run observations.
- `tmp/ascii_og_browser/summary.json`: averages over the balanced passes.
- `tmp/ascii_og_browser/*_frame0.png`: initial identical-frame screenshots.
- `tmp/ascii_og_browser/matched_pixel_check.json`: geometry/pixel validation.
- `tmp/ascii_og_browser/measurement.log`: live benchmark output.
- `tmp/ascii_og_browser/smoke_*`: ONE-SECOND harness smoke tests only. Never use
  these as the final resource comparison.
- Script: `tmp/ascii_og_browser_bench.py`; Playwright/browser tools remain under
  gitignored tmp/, not application dependencies or services.
