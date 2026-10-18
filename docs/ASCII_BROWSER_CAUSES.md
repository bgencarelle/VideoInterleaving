# Browser-cost diagnosis: current-viewer ablations

This follows the original-baseline comparison in `ASCII_BROWSER_BASELINE.md`.
The question is WHY the actual current-default viewer costs more, not whether
the renderer-only diagnostic improved. These tests never change the public
page, server settings, chosen font or image appearance.

## Method

Use the SAME preserved 120-picture current 90x60 RGB/gzip replay, 60x40 view,
1000x800 viewport, neutral tone settings and default Schoolbook weight 400.
Every run uses a fresh browser, 2.5-second moving warmup and 12-second sample;
two passes reverse the case order. The frame source stays at 15 FPS. Same
whole-browser CPU/PSS measurement method as the baseline comparison.

Only one factor changes per diagnostic:

| Case | Single change from the current default |
|---|---|
| `actual_default60` | None: reference for THIS diagnostic cohort |
| `actual_unscii60` | Replace Schoolbook with the original Unscii font; keep master, colour data, picture area and controls |
| `actual_source6060` | Send the exact already-selected 60x40 cells instead of the full 90x60 master; same visible glyphs/tints/font and gzip |
| `actual_hidden_controls60` | Use the actual Hide controls toggle |
| `actual_fixed_tint60` | Set the actual contrast control to zero, producing one fixed RGB tint while preserving glyphs/font/source |

The last case deliberately destroys colour variation. It is a diagnostic for
colour-run fragmentation/style churn/paint cost, NOT an acceptable replacement,
a compression proposal, or a recommendation to remove the accepted colours.
The font case similarly changes appearance and is not silently deployed.

The source-size case is generated once using the current centre-nearest
selector. All 120 reduced pictures are checked to contain exactly the same
displayed glyph and RGB arrays as the normal master's 60x40 selection. This
isolates source decode/inflate/storage overhead from rendering complexity.
It does NOT provide 90x60 zoom detail and is not adopted as a production design.

Initial static screenshots hide the controls ONLY during screenshot capture,
then restore each case's intended control visibility BEFORE moving warmup and
resource measurement. This avoids translucent-control compositing noise in
picture comparisons without removing the panel from its CPU measurement case.

Do not subtract these CPU figures from a different cohort's 43.6% original
viewer observation. These are one-factor comparisons against their OWN current
default run. Do not sum all differences as an additive budget; factors interact,
and two short repeats do not establish a universal platform guarantee.

## Results

| Current-viewer diagnostic | CPU (100% = one core) | PSS MiB | Visible text runs |
|---|---:|---:|---:|
| Default reference | 106.1% | 262.9 | 2261 |
| Unscii only | 96.6% | 248.4 | 2254 |
| Exact selected 60x40 source only | 101.6% | 255.8 | 2260 |
| Controls hidden only | 102.4% | 259.9 | 2257 |
| Fixed tint (contrast 0), diagnostic ONLY | 61.2% | 220.1 | 1037 |

This supports colour-run fragmentation and colour/style/paint churn as the
largest tested extra cost, not a claim that RGB arithmetic itself is expensive.
The fixed-tint case retains source, glyphs, font and timing, yet collapses many
text runs and eliminates most varying foreground-style updates. It reduces
CPU about 42% on average; it is NOT a visually acceptable fix. Colour variation
and run count interact, so the test does not split their individual costs.

Switching only to Unscii saves about 9% CPU in both passes, while run count
stays essentially the same. Thus the font has a measurable but much smaller
effect than the fixed-tint diagnostic. Sending only selected 60x40 cells saves
about 4% and produces a pixel-identical initial image; all 120 source reductions
also exactly match glyph/RGB arrays. Master decoding is not the main cause.

Controls-hidden averaged 3.5% lower CPU, but the effect reverses between passes
(default/hidden CPU 98.1/105.8%, then 114.1/99.1%). Do NOT claim that hiding the
controls is a confirmed CPU improvement. It is small/noisy relative to the
other costs and does not explain the baseline gap. Default CPU itself varies
98.1–114.1%, so these short runs do not justify precise additive percentages.
Fixed tint is consistently much cheaper (62.3/60.1%). Font/source-size savings
are directionally consistent in both passes.

No font, tint, resolution, source stream, control default or renderer code was
changed in production by these diagnostic runs; production frontend SHA-256
fingerprints remain identical. The next performance target is coloured-glyph
rendering/batching and style/paint work, not silently quantizing the accepted
RGB colours or attributing the whole cost to Schoolbook or gzip.

Artifacts: `tmp/ascii_browser_causes/{report,summary,comparison,source60_exact,source60_pixel_check}.json`,
frame-zero screenshots, reduced source fixtures and `measurement.log`.
`smoke_*` files contain one-second harness checks only and are NOT reportable
resource results. Script: `tmp/ascii_browser_causes.py` using the private replay
helpers in `tmp/ascii_og_browser_bench.py`.
