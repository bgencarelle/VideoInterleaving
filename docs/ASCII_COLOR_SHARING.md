# Low-memory terminal RGB colour sharing

This optional engine preserves the accepted colour-sharing prototype's output,
without loading Numba/llvmlite. It changes RGB tints only; image selection,
independent-layer compositing, glyph selection, frame pacing and client handling
stay outside the engine. ASCII-web and normal application defaults are unchanged.

## Build and test

From the repository root, using a C99 compiler (`cc`, or set `CC`):

```bash
.venv/bin/python utilities/build_ascii_color_sharing.py
.venv/bin/python -m unittest tests.test_ascii_color_sharing -v
```

The builder creates a source/platform-keyed shared library under
`_cache/ascii_color_sharing/`, using an atomic replacement. There are no Python
extension ABI dependencies, external numerical libraries or worker threads.
Rebuild on each target machine and after changing the C source. Linux was tested
here; other platforms need their own build/parity/performance validation. Do not
enable fast-math or FMA contraction: floating-point operation order and DP tie
breaking are part of the output contract. Import never launches a compiler or
falls back to a JIT. A missing build gives an explicit setup error.

## Use

```python
from ascii_color_sharing import ColorSharingEngine

engine = ColorSharingEngine(tolerance=1.0, edge=6.0, tint_gamma=0.9)
text = engine.encode(rgb, chars)
```

`rgb` is a uint8 `(height, width, 3)` array of graded foreground tints. `chars`
is a matching native NumPy `U1` array already chosen by the original converter.
The result is the original UTF-8 row format, with truecolour SGR run encoding and
CRLF between rows, without framing or a final SGR reset. Callers retain those
responsibilities. Public methods also expose `lift`, `share`, and `encode_colors`
for isolated tests. Inputs are not mutated; the engine is reentrant. Arrays and
native workspaces are call-local, not retained frame history.

Experimental `lightness_limit=0.5` (default `None`, disabled) constrains run
representatives' OKLab lightness separately from total colour distance. The
allowed change tightens further near black/white, and representatives are
chosen inside the whole run's feasible lightness interval. `share_colors(rgb)`
accepts already lifted tints without applying the midtone curve again. These
options support offline chroma experiments; they are not enabled in the accepted
gentle stream. Rebuild the helper after source/API changes as described above.

Experimental `chroma_model='420'` (default `'full'`) coarsens only the internal
comparison features: each smooth 2x2 block shares its two OKLab chroma values,
while every cell retains its original L. Strong-boundary blocks keep their
original features, and run edge checks still use the original full-resolution
features. Representatives remain actual lifted RGB tints from the same row;
there is no inverse transform, gamut mapping, image resampling or new encoding.
The comparison array is expanded for the existing DP, so this is fewer distinct
chroma samples, not a claim of reduced feature-buffer memory. It is off in the
accepted baseline. Any tolerance bound in this mode applies to the comparison
model, not original full-chroma colour distance; original RGB error must be
evaluated separately. L error remains bounded by the unchanged model tolerance.

The accepted live experiment still uses `tmp/ascii_rgb_runs_live.py` on port
2323, at 60x40/15 FPS, direct `images_sbs`, contrast 1.0, glyph brightness 1.5,
tint exposure 1.05, tint gamma .9, gentle sharing (OKLab bound 1, edge stop 6).
The launcher now uses this engine rather than importing experimental JIT code.
It does not install these settings as defaults in `main.py` or `settings.py`.

For temporary live comparisons the launcher accepts
`ASCII_SHARING=gentle|medium|heavy` (error bounds 1/2/3). Omitting it preserves
the accepted gentle baseline. This is a launcher-only override, not a client
control or an application-wide setting.

## What changed

- C implements tint lifting, the original dynamic program and UTF-8 encoding.
- The original NumPy linear-RGB/OKLab transform is retained with identical
  operation order, avoiding representative/tie changes from numerical drift.
- A conservative triangle-inequality bound stops extending impossible colour
  runs. Any two cells in a valid radius-r run must be at most 2r apart; once that
  fails, no longer run can become valid. The original tolerance epsilon is kept.
- The native encoder emits the exact same colour escapes and glyphs as the
  Python encoder, including Unicode, empty U1 cells and per-row colour resets.

This is not palette reduction, bit truncation, dithering, geometric resizing or
pre-mixing. The small C helper is unrelated to the transport/V7 kernels.

## Validation and measurements

`tests/test_ascii_color_sharing.py` checks an independent unpruned Python oracle,
Unicode encoding, bounds, noncontiguous inputs, source ownership, tint endpoints,
thread reentrancy, explicit missing-build errors and a fresh process free of
Numba/llvmlite imports. The focused existing ASCII/scheduling suite also passes.

The workspace parity harness `tmp/ascii_native_parity.py` compares the frozen
accepted JIT implementation against the rewrite on 18 real 60x40 pictures,
three real 90x60 pictures and 90 synthetic pictures, across sharing tolerances
0/1/2/3 and edge thresholds 0/6: 888 frame cases, 812,616 cell cases. RGB arrays
and complete terminal text match exactly. The reference JIT code is validation
only, never imported by the new engine or live launcher.

Matched converter benchmarks use 18 source grids and 720 warmed conversions in
separate fresh processes, with identical accepted grading and midtone lift:

| Measurement | RGB without sharing | Old sharing | Native sharing |
|---|---:|---:|---:|
| CPU time/frame | 1.006 ms | 1.978 ms | 0.858 ms |
| Median wall time/frame | .922 ms | 1.755 ms | .809 ms |
| Process RSS | 139.5 MiB | 270.0 MiB | 143.9 MiB |
| Mean payload | 42,385 bytes | 31,476 bytes | 31,476 bytes |
| Numba imported | no | yes | no |

These are isolated converter costs, not matched whole-application resource
measurements. The native encoder's reduced Python work offsets the sharing
work on these samples; this is not a guarantee for every source/grid/platform.
Evidence is under `tmp/ascii_native_rewrite/` (`parity.json`, converter JSONs and
the saved source grids). Live before/after content varies and must not be used
to claim byte savings or exact application-RAM reductions without matching it.

The native live launcher was also checked through the external test endpoint:
176 distinct frames in 12 seconds, 14.68 FPS, while a deliberately non-reading
client timed out and disconnected. Whole-application observations were
318.6–344.2 MiB RSS and 17.35% of one CPU core, not content/uptime-matched to the
old application's observation. Maximum receiver gap remained about 204 ms;
this rewrite does not claim to fix all delivery jitter or terminal paint costs.
The fresh actual application log is `tmp/ascii_rgb_runs_live_native.log`.
