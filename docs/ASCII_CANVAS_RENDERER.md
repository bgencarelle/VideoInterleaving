# `/ascii`: batched WebGL / native Canvas renderer

The task is to avoid the remaining per-frame styled-text layout/paint cost
without lowering RGB quality, grid resolution, cadence or dropping Schoolbook.
Server production, bakes, gzip/wire format, terminal output and V7 are untouched.

## Renderer selection

`ASCII_VIEWER_CONFIG.renderer: 'atlas'` selects `AsciiAtlasRenderer` in
`static/ascii_atlas_renderer.js`. It chooses:

1. **Hardware WebGL2** (`data-renderer="atlas"`): one reusable glyph-alpha
   texture and a single instanced draw for the entire coloured grid.
2. **Native Canvas 2D** (`data-renderer="canvas"`): when WebGL2 is unavailable,
   rejects a major performance caveat, or reports a software renderer such as
   SwiftShader/llvmpipe/WARP. Native text commands avoid DOM layout and expensive
   GPU emulation. This branch does **not** claim one draw call or a reused atlas;
   it uses the browser's native font machinery each frame.
3. **Existing DOM grid** (`data-renderer="grid"`): final compatibility fallback
   if Canvas/open/shader/allocation fails, a WebGL context is lost, resource
   budgets are exceeded or a glyph needs the existing Unicode/font fallback.
   A diagnostic `data-renderer-failure` records the reason. Current master,
   resolution and picture controls are retained; no '?' or black-frame fallback
   is substituted. Context restoration does not repeatedly retry a failed GPU.

`allowSoftwareGL: true` is a private diagnostic escape hatch, **not a production
default**. The software WebGL screening used more CPU than DOM despite removing
layout/style work, so merely checking whether WebGL exists was insufficient.
The sandbox's actual WebGL implementation is ANGLE/SwiftShader, not physical GPU
validation. Hardware rendering behaviour is exercised through software WebGL
for correctness; performance on the user's physical GPU is not established.

The DOM renderer can still be selected explicitly with `renderer: 'grid'`.
There is no server-side viewer-specific recomputation or new resize protocol.

## What each frame does

- The existing latest-only gzip/ANSI receiver and two master buffers are unchanged.
- Centre-nearest source indices are cached for the selected grid. Font glyph IDs
  and RGB bytes populate two reusable arrays (7 bytes/cell; 37,800 at 90x60).
- Local tone controls use the same existing transform afresh on original colours,
  with a per-frame unique-colour cache. Source buffers are not changed.
- Hardware updates two small buffers and submits one instanced triangle-strip
  call. Positions/UVs come from instance ID and geometry uniforms, not HTML cells.
  No per-frame texture uploads, glyph rasterisation, DOM cell/style changes,
  synchronous GL readback, framebuffer snapshots or JavaScript pixel arrays.
- Software/native Canvas clears to the chosen background, draws the exact
  selected glyphs in their original/adjusted RGB, and clips each row as before.
  Stroke and fill retain the glyph's own colour; overlapping rows retain order.

No black palette substitution, indexed-colour approximation, temporal tint
reuse, font change, cell deletion or quality/bandwidth reduction is introduced.
The one-time renderer script/font downloads do not reduce per-frame wire bytes.

## Geometry and text

Both branches use the same bundled Interleave Schoolbook Mono faces. Initial
font/cell geometry matches the saved DOM viewer. A hidden baseline probe runs
only when fitting/font settings change, never on ordinary moving frames.
Canvas backing pixels and atlas rasterisation follow the **actual fitted picture
size and DPR**, including low-resolution zoom: a 15x10 view is not an enlarged
low-resolution bitmap of 12px glyphs. The entire picture retains its fitted
extent when viewport, zoom or row spacing changes.

The hardware atlas covers the 95 printable ASCII characters with transparent
padding for stroke. Text is rasterised white to alpha, then tinted by the exact
RGB in the shader with premultiplied-alpha blending over the chosen background.
The atlas is reused until viewport/DPR, grid, weight, stroke or line spacing
changes. Tone, background and moving pictures never rebuild it.

Canvas/native and WebGL text are **not generally screenshot-identical to DOM**.
Compositor/font baseline rounding, grayscale versus DOM text antialiasing and
atlas alpha sampling can affect edge pixels and subpixel placement. We did not
add brightness/gamma compensation to conceal that difference. Existing neutral
defaults remain unchanged; user judgement of the rendering remains separate.

## Ownership, budgets and scheduling

- Atlas: at most 8 Mi pixels (32 MiB RGBA), also bounded by MAX_TEXTURE_SIZE.
- Canvas backing: at most 16 Mi pixels, bounded by GPU/renderbuffer dimensions
  where applicable. These are allocation guards, **not whole-browser RAM caps**.
- Resize replaces only the current buffers/texture; no resolution-history cache.
  Dispose deletes GL resources, removes handlers/elements and releases a live
  context. There is no animation timer inside either renderer.
- Browser paint still follows source revisions/requestAnimationFrame, not an
  independent playback clock. Resize redraws the held current master immediately;
  it does not clear the display and wait for another network frame.
- Hidden tabs keep the existing disconnect/reconnect behaviour. Failover redraws
  the latest retained picture, never a queue of historical pictures.

## Validation

Focused regression: **70 Python/Node-backed tests passed**. New tests exercise
one instanced draw, exact selected glyph IDs/RGB bytes, transformation caching,
stable atlas/resource counts on moving frames, geometry/font changes, software
GPU detection/native Canvas, allocation limits, shader errors, loss/disposal and
controller recovery with local settings and the latest picture preserved.

Actual Chromium checks use the preserved live master/gzip frames and the saved
pre-change monospace DOM viewer, not stub-only evidence:

- **369 actual frame/grid comparisons** verify every selected glyph and every
  neutral/adjusted RGB cell across DOM, native Canvas and forced software WebGL.
- **54 screenshots** cover DPR 1/2; 60x40, 90x60 and 15x10; weights 400/700/900;
  stroke 0/.3/.5/1; packed/loose spacing; background and tone controls.
- Real pixel checks find exact opaque RGB inside glyphs and exact nonblack
  background for empty frames, in both Canvas and WebGL.
- Copy/hide/restore/reset, viewport refitting, stable font/atlas build counts on
  motion, no-client-resize messages, unavailable-Canvas fallback and actual
  `WEBGL_lose_context` recovery all pass.
- Screenshots were visually inspected and pixel differences are retained rather
  than described as zero. Some native DPR-2 default cases matched exactly; other
  cases differ in edge rasterisation/placement. All source cell identities,
  colours, weights, stroke/spacing settings and picture extents remain intact.

Artifacts: `tmp/ascii_atlas/{before,check,perf}/`,
`tmp/ascii_atlas_check.py`, `tmp/ascii_atlas_bench.py`, regression log and the
private replay helper. `smoke_*` are one-second checks, not final resource data.
`phase/` contains discarded baseline-phase diagnostics; no hardcoded device-
specific phase corrections were adopted.

The balanced resource benchmark uses fresh Chromium processes, the same 1000x800
viewport and actual RGB/gzip replay for DOM versus new renderer, 2.5-second moving
warmup, two reversed-order 15-second observations. Original baseline is also
replayed read-only with its original indexed source/Unscii; it is an as-is
presentation comparison, **not equal-input/font detail**. Browser CPU sums all
browser/renderer/GPU/utility processes (100%=one core), memory is summed PSS, and
completed updates/s are reported so frame dropping cannot masquerade as savings.

### Results: fresh balanced 15-second observations

| Viewer | CPU (100%=one core) | Browser PSS | Updates/s |
|---|---:|---:|---:|
| Original default 60x40 | 42.6% | 211.2 MiB | 15.00 |
| Saved mono DOM 60x40 | 90.8% | 241.5 MiB | 15.06 |
| New automatic 60x40 (**native Canvas here**) | 26.6% | 200.8 MiB | 15.00 |
| Forced software WebGL 60x40, diagnostic only | 260.4% | 241.6 MiB | 14.56 |
| Saved mono DOM 90x60 | 137.3% | 383.1 MiB | 11.80 |
| New automatic 90x60 (**native Canvas here**) | 37.9% | 203.4 MiB | 15.00 |

At the default view, same-input CPU falls **about 71%** and PSS **about 17%**
against the saved DOM renderer. Style/layout work measured **0 ms/s** during the
moving observation versus 104/175 ms/s for DOM. Controls were visible during
resource measurement; hiding them was only for initial unobscured screenshots.
No atlas/font rebuilds occurred during ordinary motion.

Against the original-default workload, CPU is **about 38% lower** in this
cohort, despite different source/font/RGB representation. That is an as-is
resource result, not proof of identical appearance to Unscii/indexed bakes.
Original PSS varied 223.0/199.4 MiB, versus new 201.9/199.7 MiB: describe memory
as **roughly baseline range**, not a dependable saving over original.

At maximum resolution, CPU falls about 72% and PSS about 47% versus DOM while
updates rise from 11.8 to the 15-FPS source cadence. These are sandbox sample
results, not guarantees on every device. Counts are renderer updates/submissions,
not physical-monitor presentation measurements.

Forced software WebGL regressed dramatically (262.9/257.9% CPU), which is why
production rejects major performance caveats/reported software GPUs. No genuine
hardware-GPU speed claim is made. The native path was consistent at 26.8/26.4%
CPU (default) and 38.2/37.5% (maximum). All averages and per-case metadata live
in `tmp/ascii_atlas/perf/{report,summary}.json`.

The automatic renderer is enabled in the public test template. Fresh Chromium
at `https://testing3.gencarelle.fun/ascii` selected **native Canvas** here,
showed animation with **zero text spans**, and passed control/copy/reset/zoom/
viewport checks with only the gzip encoding request sent upstream. Public page
errors were empty and the new module returned HTTP 200. Existing HTTP/WS worker
PID 41408 and terminal-sweep PID 41407/listeners remained running; no restart,
server-source switch, nginx/service/routing change or original deployment edit
was made. Evidence: `tmp/ascii_atlas/public_check.json` and public screenshots.
