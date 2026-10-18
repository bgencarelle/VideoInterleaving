# ASCII display and bake evaluation — 2026-10-16

## Conclusion

The ASCII path is partly optimised, but is not yet an efficient, mode-specific
bake/display pipeline. Full-resolution layer uploads are the main avoidable
rendering work. There are also geometry and delivery correctness problems.
This pass evaluates the existing pipeline and temporary candidates; production
code was not changed by this evaluation. Face and float stay independent.

## Existing path

`image_display.py` runs the existing exact-clock/deadline preparation loop, loads
two independent layers, uploads changed textures, composites into a small GL
framebuffer, reads RGB back, calls `ascii_converter.to_ascii`, and publishes to
`exchange_ascii`. `ascii_server.py` delivers Telnet bytes; `ascii_web_server.py`
delivers WebSocket text to xterm in `templates/ascii_viewer.html`.

Already useful optimisations:

- Producer uses `ASCII_FPS` (currently 15) without an extra server-side rate cap.
- Current output FBO is only 60x40; colour grading is performed after converter
  resizing/cropping, rather than across full source pixels.
- Unchanged immutable layers avoid texture uploads; held pictures avoid redraws.
- Normal text conversion emits colour escapes only when the colour changes.
- Clock-index selection remains exact; there is no catch-up picture backlog in
  the producer. Profile observations were exact 91/91 (images), 92/92 (raw bake).

## 1. Mode-sized independent layers: largest bake/render opportunity

The current unified ASCII output shares full-resolution decoder pixels with
local mode. That saves JPEG decoding, but leaves large texture uploads and an
inflated disk representation. It is not the final ASCII bake design.

Actual display-loop profile, 96-position/four-folder images_sbs sample, pair
`(0,0)`, defaults 60x40 at 15 FPS, Mesa software GL, two-second warmup and
six-second measurement, no consumers:

| Stage | Images median | Raw bake median |
|---|---:|---:|
| Changed texture upload | 3.40 ms | 3.66 ms |
| GL draw submission | 0.090 ms | 0.083 ms |
| RGB readback / completion | 0.270 ms | 0.279 ms |
| Complete ASCII conversion | 2.16 ms | 1.93 ms |
| Of conversion: Python ANSI row building | 1.35 ms | 1.31 ms |

Stage timings are wall times, overlap preparation work, and include driver/GIL
scheduling; they are not additive CPU attribution. Process CPU/RSS were 19.0% /
303.9 MiB from images and 15.6% / 273.0 MiB from the full-resolution raw bake.
CPU 100% means one core. These are representative warmed observations.

### Independent-layer sampling prototype

Sample each layer through the current GL geometry/filter/SBS-matte rules into
its own terminal-grid RGBA plane. Keep sampled colour and alpha separate from
all other layers; mix selected planes live. Compress each layer/frame as an
independently indexed chunk. No final image or text is baked.

For 28 real layer frames (seven positions, two face/two float folders):

- Original JPEG payload: **2,450,667 bytes**.
- Full decoded source payload: **116,121,600 bytes**.
- Sampled 60x40 float32 RGBA: **1,075,200 bytes**.
- Those independent planes with standard-library zlib level 1:
  **143,075 bytes**, about **17.1x smaller than JPEGs** on this subset.
- Warm decompression median: **0.074 ms per layer**.
- Tiny live GL mixing plus RGB readback median: **0.165 ms**.

These are payload/microbenchmark measurements, excluding index/provenance data;
they are not an end-to-end production CPU/RAM result or a full-tree storage
measurement. Compression was evaluated as individual chunks in memory; a
production reader would map/index compressed chunks and decompress only the
requested layers into a bounded working set.

Tested 28 real picture pairings plus 12 adversarial RGBA transparency/filtering
cases. Float32 planes reproduced **40/40 original RGB framebuffers and ASCII
strings byte-for-byte**. Uint8 planes also matched all 40 on this software GL
backend, but that does not establish equivalent filter precision across other
GPUs. Float16 failed RGB equality in 21 cases and text equality in 12: it is
not a safe assumed substitute. The initial safe prototype retains float32 and
lossless compression. Any production format needs a defined sampling contract,
backend checks, size/aspect metadata and a strategy for incompatible sizes.

## 2. Geometry is applied twice, losing horizontal detail

`display_manager.py` chooses a 60x40 FBO. `renderer.compute_transformed_quad`
letterboxes the 720x960 portrait using ordinary square-pixel viewport aspect,
leaving only 30 image columns inside that FBO. `to_ascii` then applies the
0.5 font-aspect correction, doubles width and crops the centre.

A horizontal source-ramp check confirmed **30 distinct samples in 60 final
columns**. Sampling once with font-aware geometry produces **60 distinct
samples**. Thus rendering at terminal dimensions alone has not eliminated
redundant scaling or preserved the available horizontal detail.

Correction should define fit/cover and character-cell aspect once, sample each
layer onto that same grid, composite live, then grade/encode the grid without a
second geometric transform. This deliberately improves sampling and changes
existing pixels; it should not be silently bundled into a claimed byte-identical
performance optimisation. The small-layer parity measurements above reproduce
the current geometry, not this proposed geometry correction.

## 3. Text building is unnecessarily Python-heavy

`ascii_converter._build_colored_rows` loops over every cell, constructing Python
string fragments. There are also two identical function definitions; only the
second runs, so the duplication is maintenance debt, not doubled runtime work.

A temporary Numba UTF-8/ANSI encoder candidate retained exact colour-run resets
and Unicode characters, including multi-byte glyphs. Twelve ASCII/Unicode,
colour-run and dimension cases matched the existing strings exactly. On a warm
60x40 random-colour microbenchmark, median row-string generation fell from
**0.807 ms to 0.024 ms** (~34x). This excludes grading, GL, networking and cold
compilation; it is not a whole-application speedup. Compile/warm before playback.

The legacy chars/colours `.npy`/`.npz` worker in `load_and_render_frame` uses a
separate per-character escape builder, missing the normal converter's colour-run
encoding. It should share an encoder if that legacy path is retained. A new
independent RGBA-grid bake must still composite before grading/text conversion.

## 4. WebSocket pacing adds latency and bursts

The third-party WebSocket server defaults to `selectInterval=0.1` and is not
woken when the separate broadcast thread queues a frame. This is coarse relative
to a 66.7 ms producer interval.

Actual main.py/client runs on loopback, about four seconds each:

| Delivery | Frames | Receive gap p50 / p95 | Gaps below 5 ms |
|---|---:|---:|---:|
| Telnet | 62 | 66.65 / 67.60 ms | 0 |
| WebSocket | 60 | 100.13 / 100.29 ms | 19 |

WebSocket payload hashes matched all 60 producer publications. Publication to
client-receipt delay was **41.6 ms median / 78.8 ms p95**. These are server/socket
measurements, not browser paint times. A wakeable event loop with a single latest
pending picture is preferable to merely lowering a polling interval and trading
more idle CPU for less delay.

## 5. Current WebSocket queue clearing can corrupt framing

`broadcast_loop` calls `client.sendq.clear()` when three items accumulate. But
the library puts the unsent **remainder** of a partially sent packet at the head
of that same queue. Clearing it can splice a fresh frame into the unfinished
payload of the previous frame. It also mutates a queue consumed on another
thread.

Reproduction using the actual `_sendBuffer` and `broadcast_loop` code: send 17
bytes of a 2,408-byte packet, simulate EAGAIN, retain its remainder with two more
packets queued, then publish again. The broadcaster discarded the **2,391-byte
remainder**, corrupting the stream. The existing mocked queue-bound test does not
cover this partial-send case.

Correct latest-only delivery must complete an in-flight protocol packet, or
close that connection; replace only whole, not-yet-started picture messages.
Keep queue ownership on the I/O thread and preserve control messages.

## 6. Browser and subscription handling can still become stale

- Browser `ws.onmessage` writes every received frame directly into xterm, without
  a latest-pending slot or completion callback. A slow terminal renderer can
  replay a backlog even if server-side queuing is bounded. Use at most one
  outstanding xterm write plus one replaceable latest picture.
- `FrameExchange.get_frame()` waits for a new notification even when it already
  holds a picture. A client joining after a held/MIDI picture was published can
  remain blank until the next publication. Reproduction: publish first, then
  subscribe with a timeout; the current exchange returns None. Add a versioned
  latest-frame read so a new subscriber gets the current picture immediately and
  a slow subscriber obtains the newest version without waiting an extra tick.
- Per-client UTF-8/home-prefix construction is smaller work than texture uploads
  or row building. It can be shared once per publication, but is not the first
  bottleneck to optimise.
- Browser font fitting adds letter spacing after estimating cell aspect, and
  resize-control JSON is not handled by the direct `term.write` message handler.
  Those paths merit visual verification before changing presentation geometry.

The desktop browser tool returned **browser.disconnected**. Native rendering,
network tooling and the application were available; browser CPU/RAM, physical
cell geometry and network-to-screen latency were not measured.

## Recommended implementation order

1. Make delivery correct and latest-only: partial WebSocket packets, current
   picture on join, wakeable outbound work and bounded xterm writes.
2. Fix the double geometry pass with an explicit, tested font-aware grid contract.
3. Bake/compress independent layer samples for that grid; retain live mixing,
   grading, palettes and exact producer clock selection.
4. Share a measured compiled UTF-8/ANSI encoder across normal/legacy text paths.
5. Repeat source/bake CPU/RSS/storage tests with real clients and browser visual
   inspection when a browser is connected. Do not report microbenchmark savings
   as delivered application savings.

Artifacts: `tmp/ascii_review_profile_{images,baked}.json`,
`tmp/ascii_review_layers.json`, `tmp/ascii_review_rows.json`,
`tmp/ascii_review_network.json`, `tmp/ascii_review_delivery.json`; reproducible
temporary runners are `tmp/ascii_evaluate_*.py` and `tmp/run_ascii_evaluation.py`.

## Telnet client sizing implementation — 2026-10-16

`main.py --mode ascii --port 2323` now enables a shared source-aspect RGB
composite (bounded to 320x480 before even-size rounding). Layers are still
selected and composited live. `ascii_terminal.py` converts the current composite
to client dimensions, sharing encoded output by size with a 20-variant cache
cleared on each publication. This is runtime sizing, not the proposed bake.

The Telnet server negotiates NAWS, including fragmented messages and escaped
IAC bytes. Non-reporting clients default to 60x24. Reported dimensions are
clamped to 160x60; zero dimensions are ignored. Applied resizes are limited to
two per second. Subnegotiation storage is bounded to 64 bytes, incoming traffic
to 8192 bytes per one-second connection window, and socket sends to a two-second
timeout. The existing concurrent-client semaphore remains in effect.

Output uses an alternate screen, hidden cursor, disabled autowrap and explicit
row positioning without newlines; disconnect cleanup restores screen/cursor and
autowrap where the peer remains connected. New connections receive the current
RGB picture without waiting for another publication. Each sender fetches the
latest picture after a send; no application-level frame history is queued.

Initial user inspection identified noise/blotchiness after increasing the RGB
render target. Client-sized conversion now uses area downsampling (linear when
enlarging), rather than point sampling. A checker-texture regression test
verifies suppression of false light/dark glyph patterns. This changes sampling
and requires user visual comparison; it is not a byte-identical default-picture
claim. The existing palette and grading values are retained.

User inspection also identified zoom/cropping from the converter's legacy cover
policy. Client-sized output now uses contain/fit geometry with centred black
padding, accounting for character-cell aspect. A source-edge regression check
verifies both sides of the picture survive conversion into a wide terminal.

Real socket checks exercised 80x24, 120x40, 20x8, a 300x200 request capped to
160x60, fragmented NAWS, malformed negotiation recovery and the 60x24 fallback.
They verified complete frames of the expected cell dimensions without CR/LF.
Evidence: `tmp/check_ascii_2323.py`, `tmp/ascii_2323_socket_checks.json`.
WebSocket/browser changes and broader public-service deployment hardening are
still outstanding; the proposed independent-layer bake has not been built.

### User-feedback correction: preserve the original artwork

The higher-resolution RGB and area-filtered client conversion above regressed
colour and perceived detail in user inspection. That approach is superseded for
the running Telnet service: its GL target is restored to the original configured
60x40, and its canonical text is produced by the original no-size-override
`to_ascii` path. `TerminalPictures` parses this internally generated text into
glyph/foreground-colour cells once per picture, then fits/resamples those cells
to each output size. It does not re-average RGB or re-quantise colours per size.
This preserves the existing character-grid proportions and canonical detail;
larger terminals enlarge existing cells rather than gaining source detail.

A regression check verifies every glyph and active foreground colour matches
the original converter at 60x40. Explicit black terminal background prevents a
client's theme background from altering the picture. Window negotiation now logs
reported/output sizes, or the absence of NAWS, for diagnosing the user's resize
report. Real-client resizing remains unconfirmed until their negotiation is
observed; synthetic socket resizing passes. The 22 focused checks and live
multi-size/malformed-negotiation checks pass after this correction.

Following user feedback, terminal output now picks one of four fixed image grids:
30x20, 45x30, 60x40, 90x60, selecting the largest that fits reported dimensions.
Smaller-than-minimum windows retain a fit fallback to avoid wrapping. Raw TCP
clients can select grids with keys 1 through 4 (Enter if line-buffered); a later
NAWS report restores automatic selection. The external test connection logged
no NAWS report, explaining the unchanged image on window resize. Manual choices
are explicit and cannot be checked against an unreported physical window size.
The 24 focused checks pass; live raw-socket checks verified all four manual
choices with complete correctly sized frames. User-reported colour differences
remain unresolved pending a screenshot/client details; canonical 60x40 colour
and glyph equality alone does not verify terminal presentation.

### Resolution correction: real sampling grids

The user correctly identified that enlarging canonical cells does not add
resolution. That interim workaround is superseded. The running service now
retains a source-aspect RGB composite bounded to 360x480 (360x480 for the current
720x960 source). Each of the four grids is generated directly from that RGB
picture before grading/character quantisation, then cached for clients sharing
that scale. Encoded cells are no longer enlarged or decoded/re-encoded. Terminal
padding and explicit cursor placement wrap those generated rows directly.

A fine-bar regression check verifies that the 90x60 output contains independently
resolved structure rather than a nearest-enlarged 60x40 grid. A separate check
verifies terminal framing preserves every generated glyph/foreground colour.
The 37 focused converter/terminal/lazy-import/scheduling checks pass, as do the
live multi-size socket checks. Original no-argument converter behaviour remains
available; client sampling is intentionally different and visual colour parity
is still not established. The previous canonical-grid equality evidence applies
only to the superseded interim implementation.

### Matched comparison against main (`9591126d5`)

At the user's request, loaded the actual `main` versions of `image_loader.py`,
`renderer.py`, and `ascii_converter.py` into isolated modules. On seven source
positions and four face/float pairings, 56/56 layer decodes, 28/28 RGB composites
at matching 60x40 geometry, and 28/28 original-size ASCII conversions matched
exactly. The client-sized high-RGB/area-filtered path differed in all 28 ASCII
outputs. Grading constants and the blend shader are unchanged from main.

A separate audit drove the actual current `run_display` loop with deliberately
changing folder selections, inspecting GL texture bytes before each draw and
comparing them to independently decoded main-branch source files at the displayed
index/folder identity. All 106 textures in 53 draws matched (29 distinct indices,
all four pairings). All 52 publications were exact requested index/folder
identities. With the deliberately abrupt folder changes, many ticks held the
previous picture: 52/177 observed ticks were exact. That is a limited identity
audit, not a real-time throughput claim or proof for all sources/backends.

The new pipeline changes both the resolution and operation order: main samples
each colour/matte layer in the small GL target and blends those samples; client
sizing blends a larger target and area-reduces the completed RGB before grading.
These operations are not generally equivalent around transparency/detail. The
converter also changes main's cover/nearest geometry to contain/area. Matching
grading settings therefore cannot establish matching colour or texture. No
wrong-layer pairing was observed in these checks; visual regression remains.

Artifacts: `tmp/ascii_compare_main.{py,json}`,
`tmp/ascii_live_identity_audit.{py,json,log}` and
`tmp/ascii_main_vs_new_{0,1}_{0,1}.png`. The latter are matched 60x40 ANSI glyphs
rendered with the bundled font and xterm palette; they demonstrate sampling/
texture differences without requiring a browser. This comparison did not change
the running service's image pipeline.

### Corrected delivery: four direct layer renders

The screenshots supplied by the user show a lower-resolution, muted-colour new
output versus the taller default. The high-RGB/area-filtered path is superseded.
`ascii_render.AsciiGridCapture` now renders the selected face and float textures
independently into four small FBOs on the existing context-owning display thread,
using the original shader, source-aspect geometry and sampling order. FBO sizes
follow main's even-pixel rounding (45 columns use a 46-pixel RGB target). Every
grid is converted with main's cover/nearest geometry and existing grading, with
only explicit width/height arguments added. No image arithmetic or grading is
performed on the network thread beyond that original converter call; shared
per-grid text caches avoid duplicate conversions for clients at the same scale.

The comparison against the actual main modules now includes all four grids:
**112/112 RGB renders and 112/112 ASCII strings match byte-for-byte** across seven
pictures and four independent layer pairings on software GL. This is evidence
for the corrected renderer/converter, not a claim that arbitrary terminals share
the same palette. The 36 focused converter/terminal/lazy-import/scheduling checks
pass. Actual main.py socket checks also receive complete correctly sized frames
for manual selections 1–4. Port 2323 runs the four-target path.

The screenshot comparison and source-grid selection are distinct issues: scale
2 intentionally has fewer cells than the default scale 3. For a same-resolution
visual comparison to default, select 3 (60x40). Prior sampling/filtering and
cell-enlargement descriptions above are historical rejected iterations.

### Direct images_sbs sampling and unblurred colour (current implementation)

User inspection still found muted colour and apparent repeated/resized detail.
Two remaining causes were identified: each small FBO still entered the legacy
converter's resize/crop (duplicating horizontal samples), and the default 7x7
colour-only Gaussian blur softened adjacent tints. `ASCII_COLOR_BLUR` now defaults
to 0; the optional blur remains available through settings. Grading controls are
retained.

The current implementation samples the two independently selected full JPEG
textures onto each exact-sized grid, using character-cell pixel aspect in the
GL fit geometry. `to_ascii(..., sampled=True)` validates the RGB grid dimensions
and bypasses all resize/crop work, then runs grading/glyph/ANSI generation.
Odd-width grids are exact (45 RGB columns for 45 characters). The ordinary
converter entry point remains unchanged for non-grid callers. Four scale
selection, bounds, manual controls and padding remain in place. Tiny windows
below the smallest supported grid retain the bounded compatibility fallback.

`tmp/ascii_source_grid_check.py` reads full JPEGs from the actual `images_sbs`
paths in the application lists, not the unified sample package or any bake.
Across 125 pairs (five indices, five face and five float folders), all 500
direct renders match an independent main-branch shader reference with equivalent
font-aware geometry, and all 500 grading outputs match the original converter
fed an already sampled grid with blur disabled. Patching resize to raise verifies
that none of these conversions resizes the grid again. A source ramp resolves
30, 45, 60 and 90 distinct columns at the respective grids, demonstrating real
sampling resolution rather than enlarged text. Results:
`tmp/ascii_source_grid_check.{json,log}`. The 38 focused tests pass, including
source tint preservation and sampled-grid dimension/resize checks. These checks
validate the new sampling contract; earlier byte-equality claims against main's
legacy geometry/7x7 blur do not describe this corrected appearance.

The local application was launched with an explicit absolute `--dir` pointing
to `/home/linuser/modemTest/images_sbs`, on the requested test port 2323. Startup
logs confirm direct source-layer grids, blur 0 and no post-grid resize. No bake
argument is used.

The user supplied `gencarelle.fun:2323` as a **read-only baseline reference**,
not a deployment destination. A read-only TCP capture found its fixed 60x40,
home-plus-text/per-character-SGR stream differs from the local implementation's
protocol. A local/public frame-match assertion therefore failed; no deployment
claim follows from the local checks. The reference server and its routing,
deployment and nginx configuration must not be changed. A previous nginx read
attempt was denied and made no changes. User visual confirmation against the
reference remains outstanding.

### Alpha and dull-white investigation

`tmp/ascii_alpha_white_check.py` exercises RGBA and SBS layers on the real GL
path at all four scales: opaque white, half-transparent white over black, black
over white, white over white, two differently coloured translucent layers and
transparent white. All 48 cases matched independently calculated straight-alpha
RGB-space results with **zero channel error**. Layer order is main over
background, then float over that result. The SBS writer
`utilities/reencode_for_jps_sbs.py` stores RGB and alpha separately without
premultiplying; the shader's straight-alpha formula matches that representation.
These synthetic checks establish blend arithmetic, not every source asset's
matte quality or a linear-light compositing model.

A separate defect was confirmed after grading: the old `RGB*5//255` colour
mapping floored channels into a nominally uniform cube, but actual xterm cube
levels are 0,95,135,175,215,255. Thus neutral (254,254,254) mapped to grey 215,
and warm near-white highlights mapped to pinkish (255,215,215), even at shipped
brightness for warm highlights. Mapping now uses a setup-time nearest-level LUT
with uint16 arithmetic, preserving the existing palette's black special case.
It changes ANSI colour selection only, not alpha, grading, gamma or glyphs.
Near-white and warm-white cases now select palette white 231. Exact cube-level,
highlight and grading regressions pass in the 40 focused checks. GL alpha checks
still match all 48 cases after the colour fix. Before/after evidence:
`tmp/ascii_alpha_white_check{,_after}.{json,log}`.

Earlier equality evidence against main's floor-based ANSI encoding is historical;
the corrected nearest-palette colours intentionally differ from that encoder.

### Skin-tone colour correction (current defaults)

User feedback identified unnatural skin contrast after the white fix. A 101-step
skin ramp retained 80 graded RGB values with the old 1.4 brightness shared by
tint and glyphs, then only eight palette colours; the lightest 27 steps clipped
red to 255. This produced flat skin highlights and large palette shade changes.

Colour rendering now defaults to `ASCII_TRUECOLOR=True` with
`ASCII_COLOR_BRIGHTNESS=1.0`. Character selection retains the original contrast,
brightness 1.4 and grayscale gamma. Truecolour tints share the saturation and
contrast stages but use their own exposure before RGB encoding. White stays
white; skin highlights retain distinct values rather than clipping together.
`ASCII_TRUECOLOR=False` retains the nearest-256-colour fallback. This is a
server-side rendering setting, not a new client command or negotiation feature.
Input, scale selection, alpha blending and source selection were not changed
by this colour task. The reference service is untouched.

`ascii_encoding.py` uses precomputed channel string fragments and colour runs,
avoiding per-cell decimal formatting. A compiled encoder candidate was faster
but increased isolated process RSS by about 126 MiB, so it was discarded. The
shipped table-driven path requires no JIT import: isolated palette/truecolour
RSS was 140.2/140.6 MiB (not a whole-application memory measurement).

`tmp/ascii_colour_check.py` validates 108 real-source picture/scale cases: glyphs
are identical between colour modes, and the skin ramp retains 101 truecolour
tints. Matched 60x40 conversion medians were 0.934 ms palette and 0.877 ms
truecolour. Payloads increased from mean 11,333 to 40,811 bytes/frame, about 3.6x;
this is an explicit bandwidth regression for improved colour. The 45 focused
checks pass, including Unicode/run encoding, tint/glyph exposure independence,
contrast/gamma controls and white preservation. The 48 GL alpha checks still
pass with zero error. Evidence: `tmp/ascii_colour_check.{json,log}`,
`tmp/ascii_colour_encoding_cost.json`, `tmp/ascii_truecolor_alpha_check.{json,log}`
and `tmp/ascii_skin_colour_preview.png`. Full RGB requires a terminal that
interprets 24-bit ANSI foreground colours; client visual confirmation remains
part of this test.

### Lossless bandwidth evaluation (no production changes)

User asked whether the roughly 10.8 Mbps maximum-grid payload could be reduced
without reducing visual quality. `tmp/ascii_delta_evaluate.py` captures the
current truecolour stream and encodes exact character/RGB cell changes. An
independent terminal interpreter verifies every candidate transition reconstructs
the full target exactly. In four-second samples, 57 transitions at 60x40 saved
2.44% and 52 at 90x60 saved 3.06%; 96.2%/92.7% of cells changed. The temporary
Python candidate added median 1.82/4.02 ms encoding time. Delta repaint is not
adopted on this evidence. Results: `tmp/ascii_delta_evaluate.json`.

Independent zlib level-1 compression of 30 captured 90x60 frames reduced mean
payload from 85,700 to 28,565 bytes/frame (66.7% saving), with decompression
verified byte-for-byte. At 15 FPS this sample projects 10.28 to 3.43 Mbps before
transport overhead. Results: `tmp/ascii_lossless_compression_evaluate.json`.
Compression preserves every colour/glyph/frame but requires client-side
decompression; plain nc cannot display compressed bytes. No compressed endpoint,
client protocol, routing or production encoder change was made by this
evaluation. The reference service was not accessed or changed.

### Perceived frame-rate comparison

After the user reported motion below the configured 15 FPS, receiver-side
completion timing was measured, rather than quoting the setting. An eight-second
check observed 14.93 FPS locally at 60x40, 14.84 at 90x60 and 14.80 through the
test endpoint `testing3.gencarelle.fun:2323`; all completed frame hashes were
distinct. 119 received test pictures matched local pictures exactly. The local
application monitor sampled 15.0 loop FPS and zero index delta throughout, with
occasional ~185 ms receive gaps. Evidence: `tmp/ascii_cadence_check.{py,json}`
and `tmp/ascii_cadence_monitor.txt`. Socket receipt is not terminal paint timing.

At the user's explicit request, a simultaneous 12-second, 60x40 comparison read
from the reference `gencarelle.fun:2323` without sending it any commands, and
selected the existing 60x40 preset only on the test endpoint:

| Metric | Reference | Test |
|---|---:|---:|
| Completed distinct frames | 164 | 178 |
| Delivery FPS | 13.67 | 14.82 |
| Median / p95 receive gap | 68.1 / 98.8 ms | 66.7 / 86.6 ms |
| Maximum gap | 121.6 ms | 182.7 ms |
| Gaps over 100 ms | 5 | 1 |
| Mean bytes/frame | 28,571 | 43,241 |
| Mean colour escapes/frame | 2,400 | 2,321 |
| Payload rate | 3.12 Mbps | 5.13 Mbps |

Evidence: `tmp/ascii_reference_cadence.{py,json}`. These network paths originate
on the development machine, not the user's machine; no terminal paint work is
performed by the receivers. The test delivers more completed pictures in this
sample, with similar typical pacing but a worse isolated stall and roughly 51%
larger frame payloads. Truecolour parsing/painting and the user's network remain
unmeasured causes of lower perceived cadence. No production or reference server
configuration was changed during this investigation.

### Top-to-bottom redraw correction

The user confirmed top-to-bottom tearing. Terminal payloads now bracket each
complete picture with synchronized-update mode 2026 begin/end markers. Supporting
terminals defer painting until the end marker; unsupported terminals may still
paint progressively. Disconnect cleanup also ends synchronized mode. No terminal
capability negotiation is added.

The direct-grid producer now bypasses the redundant legacy main-target draw,
readback and text conversion. It samples only requested independent source grids;
requests expire after two seconds, with a 60x40 idle default. A newly requested
grid waits for its own source render rather than substituting another grid.
Character/RGB encoding and the 15 FPS target are preserved. Sender polling is
still present; this change does not claim to eliminate every source of jitter.

47 focused tests pass, including complete sync batches, demand expiry and missing
grid handling. The real-source colour harness still passes 108 glyph comparisons.
After restarting only the local images_sbs ASCII service on port 2323, short live
checks verified complete RGB sync batches at every preset: 60x40 15.14 FPS
(p95/max gap 69.6/85.8 ms), 90x60 14.78 FPS (72.4/109.0 ms), 30x20 15.00 FPS,
45x30 14.99 FPS. Evidence: `tmp/ascii_redraw_live_check.json` and
`tmp/ascii_colour_check_redraw.log`. These are short, unmatched receiver samples,
not terminal paint measurements or proof all stalls are fixed. Client visual
confirmation is pending; the reference service was not changed.

### Perceptual ANSI comparison experiment

At the user's request, `tmp/ascii_perceptual_compare.py` generates five offline
colour treatments for three face pictures (indices 511, 1332, 1776, unmodified
images_sbs sources, transparent float): current RGB, nearest RGB palette mapping,
nearest OKLab mapping, minimum CIEDE2000 mapping, and ordered two-colour mixing
fitted in linear light with an OKLab residual/endpoint-contrast penalty. All
candidates preserve current grading and the exact original glyphs. The fixed
palette is xterm 16..255; no client palette redefinition is used.

CIEDE2000 mean warm-tone error decreased from 8.50 to 7.03, 8.34 to 6.93 and
7.49 to 6.52 (about 13–17% improvement over RGB nearest); p95 errors fell from
13.0–13.5 to 9.34–9.39. These delta-E values are not percentages. Warm-tone
selection is a colour heuristic, not facial segmentation. CIEDE2000 payloads
are 14.5–15.8 KB versus 43.8–44.4 KB for RGB, about 64–67% smaller. The ordered
candidate improves a spatially smoothed linear-light/OKLab metric but worsens
per-cell delta-E and introduces texture; it requires visual evaluation.

This is an offline experiment, not a realtime kernel or accepted quality
result. Combined candidate generation takes about 0.25 seconds per sample.
Glyph-coverage-aware matching and motion stability have not been evaluated.
Optional scikit-image supplies CIELAB/CIEDE2000 only in the temporary harness;
application dependencies and colour defaults are unchanged. Validation includes
OKLab black/white anchors, a published CIEDE2000 reference pair, exact palette
colour preservation, byte-level glyph preservation and valid palette IDs.

Because chat image attachments were not visible, the user-authorized terminal
comparison loop temporarily replaces the local playback on port 2323. Connect
with `nc testing3.gencarelle.fun 2323`: five seconds per option, five options per
face, three faces, then repeat. Requires 60 columns and 41 rows (one label row
plus all 40 picture rows); no commands or client input are parsed. Payloads are
precomputed, framed with synchronized updates, viewer capacity is 20 and sends
have a two-second timeout. The external test connection was verified to deliver
all five labelled 60x41 frames. Evidence: `tmp/ascii_perceptual/report.json`,
`manifest.json`, `live_check.json` and `tmp/ascii_perceptual_generate.log`.
The normal images_sbs playback command is preserved in the earlier handoff;
reference routing/service configuration was not changed.

### Established quantizer follow-up

User rejected the custom candidates and requested an established conversion
process. Installed libimagequant was found via ctypes (Pillow also reports
libimagequant support); no packages were installed. After a declined external
header read, the API signatures were checked against the upstream public 2.17.0
header. `tmp/ascii_libimagequant_compare.py` calls the installed library directly,
quality maximum 100, speed 1, 240 fixed opaque colours and maximum palette size
240. Every returned entry is checked against actual xterm 16..255 colours.
Library remapping is compared at dithering levels 0, .35 and 1; dithering is
entirely libimagequant's implementation. CIEDE2000 remains as a comparison.
All three source samples passed the existing glyph/palette validation. Evidence
is under `tmp/ascii_libimagequant/` and `tmp/ascii_libimagequant_generate.log`.

The local :2323 experiment now alternates RGB before each candidate at the
user-requested two-second cadence: A RGB, B LIQ no dither, A RGB, C LIQ 35%,
A RGB, D LIQ 100%, A RGB, E CIEDE2000. Three expressions repeat; 60x41 terminal
required. The external test endpoint was verified to deliver the first two
comparison pairs with full 60x41 payloads. These are fixed-palette character-grid
tests, not adaptive-palette or full-source-resolution experiments, and not proof
of acceptable visual quality. No normal application colour defaults changed.

### Full-source independent-layer quantization comparison

User clarified that dithering must be tested before compositing, at full source
resolution. `tmp/ascii_source_quantize_compare.py` now applies libimagequant to
each layer's 720x960 colour half independently, never its SBS alpha half. Every
source alpha half is checked byte-for-byte unchanged. Only then are textures
composited and sampled using the normal renderer and current grading. All
variants use the original RGB glyphs to isolate colour changes. They emit RGB
without a second quantizer: blending, GL sampling and grading create non-palette
colours, so this experiment does not demonstrate final ANSI bandwidth savings.

The local output-only loop on :2323 now shows A original RGB before each of
B SOURCE no dither, C SOURCE dither 35%, D SOURCE dither 100%, E POST dither
100%, holding each for two seconds. There are three source pairs: index 511
with blank float, 1332 with digital float, 1776 with anime float. The final
post-composite option is explicitly labelled for comparison. Precomputation,
alpha preservation, glyph preservation and external delivery of the first two
comparison pairs passed. Evidence: `tmp/ascii_source_quantize/report.json`,
`live_check.json`, and `tmp/ascii_source_quantize_generate.log`. This replaces
the previous temporary loop; normal playback code/defaults are unchanged.

The user requested an actual end-to-end ANSI test after clarifying that the
above source-first experiment still transmitted RGB. The current version of
`tmp/ascii_source_quantize_compare.py` now performs a final CIEDE2000 nearest
palette remap after normal compositing/sampling/grading, without a second dither.
Only A is RGB; B–D independently process full-resolution source colour halves
then transmit indexed ANSI. E remains a post-composite library-dither ANSI
comparison. Alpha and original glyph preservation checks still pass. The
external endpoint was checked through all eight A/B/A/C/A/D/A/E pictures:
all B–E contain indexed colours 16..255 and no truecolour escapes; every packet
has 60x41 cells, and measured intervals were 2.00 seconds. Updated evidence is
`tmp/ascii_source_quantize/{report,live_check}.json`. This supersedes the RGB-only
output limitation of the preceding experiment; visual acceptance is pending.

### Stage-by-stage diagnosis after perceptual mismatch feedback

User reported that none of the end-to-end ANSI candidates matched perceptually
and approved isolating the stages. `tmp/ascii_stage_diagnose.py` retains all
three source pairs and original glyphs, saving original RGB tints, source-first
processed RGB tints before the final remap, and the final indexed palette RGB.
The warm-tone mask is selected once from the original with widened arithmetic;
all stage errors are computed on that same mask. Cell-pooled mean CIEDE2000:

| Source treatment | Original vs pre-final RGB | Original vs final ANSI | Pre-final RGB vs ANSI |
|---|---:|---:|---:|
| No dither | 7.17 | 8.15 | 2.24 |
| Dither .35 | 4.63 | 7.70 | 4.71 |
| Dither 1.0 | 3.73 | 7.50 | 6.00 |

These are delta-E units, not percentages, and the distances are not additive.
The final remap substantially erodes the benefit seen before palette mapping;
source processing also has nonzero error. The report additionally includes
per-cell percentiles and a glyph-coverage-weighted average of swatch delta-E.
That weighted average is NOT a terminal appearance model or a capture of actual
terminal rendering. Rasterized three-stage views were inspected; real terminal
capture and terminal calibration remain unmeasured.

The local :2323 loop now repeats original RGB -> SOURCE RGB -> original RGB ->
FINAL INDEXED ANSI for each B/C/D source treatment, then the next picture, two
seconds per image. This diagnostic intentionally includes clearly labelled RGB
intermediates; only FINAL INDEXED ANSI uses palette output. An external receiver
verified the first four 60x41 frames, colour escape formats and 2.00 s intervals.
Evidence: `tmp/ascii_stage_diagnose/{report,live_check}.json`, per-case NPZ stage
arrays and rasterized `stages_*.png`; harness log
`tmp/ascii_stage_diagnose_generate.log`. No production conversion defaults were
changed and no quantizer was accepted on this evidence.

### Chafa DIN99d benchmark

User requested an established optimized perceptual transform after online
research. Chafa's official homepage/manual document DIN99d colour picking and
240-colour output (excluding terminal-dependent entries 0..15):
https://hpjansson.org/chafa/man/ . Installed Chafa 1.14.5 was used directly, with
one thread, work level 9, dithering and automatic preprocessing disabled,
foreground-only output on black and the project unscii-16 font. No installation
or production dependency/default change was needed.

`tmp/ascii_chafa_compare.py` compares A original RGB, B Chafa RGB mapping with
original glyphs, C Chafa DIN99d mapping with original glyphs, and D Chafa's full
DIN99d renderer with symbol selection constrained to the current ASCII palette.
For B/C, each original tint fills a constant 8x16 footprint and Chafa renders
only solid symbols; its foreground palette IDs are extracted and applied to
the unchanged original glyphs. D instead receives the normal full-resolution GL
composite with shared tint grading and selects its own glyphs/density. It is not
a colour-only comparison and does not retain the original glyph brightness rule.
All B–D output fixed indexed colours 16..255, not truecolour. These benchmarks
run after compositing and are not a new source-first bake method.

Across the same three pairs, warm-tone mean CIEDE2000 for B vs C is 8.52 vs
7.21, 8.55 vs 6.88, and 7.95 vs 6.72 (15–19% improvement against Chafa RGB, not
against every previous method). C payloads are 14.6–15.2 KB; D 12.4–13.5 KB.
CLI subprocess times including image loading are 119–161 ms for C and 147–163
ms for D. These are offline experiment costs, not a validated 15 FPS backend.
No visual-quality acceptance or realtime integration is claimed.

The local output-only :2323 loop now alternates A/B/A/C/A/D at two seconds per
image across three pairs. External test-endpoint receipt verified all six
60x41 packets, palette-only candidates and 2.00 s intervals. Evidence:
`tmp/ascii_chafa/{report,live_check}.json`, input PNGs, ANSI samples and
`tmp/ascii_chafa_generate.log`. Actual terminal screenshots remain unavailable;
user evaluation is pending. The reference endpoint/configuration is untouched.

### Terminal palette/transport assumption check

After user questioned the implementation, Chafa was tested on all 240 known
palette entries: RGB mode changed one gray (138 to 128), DIN99d mode one gray
(218 to 215); each occurred ten times in the repeated chart. Thus its mapping
is not perfectly palette-idempotent. On face 511, all 2400 raw indexed SGR IDs
matched the independently extracted/re-encoded IDs; this found no parser error.

`tmp/ascii_ansi_transport_check.py` isolates a still-unverified assumption:
whether explicit RGB and equivalent indexed SGR look the same in the user's
terminal. A is original RGB, B is DIN99d-mapped colours sent as RGB, C is the
exact same mapped colours sent as indexed ANSI. All 2400 glyph/RGB tuples in
B/C are independently decoded and compared exactly for each of three pictures.
The local :2323 loop now displays A/B/C at two seconds each; external receipt
of the first three labelled 60x41 frames and their output formats passed.
Evidence: `tmp/ascii_ansi_transport_check/report.json`. A B/C visible difference
would indicate a palette/terminal colour-handling mismatch; identical B/C with
both differing from A points toward quantization rather than transport. No
such user result is assumed yet, and no production defaults were changed.

### Source-first DIN99d with actual area averaging

User clarified transform order and that averaging full-resolution dither is
intentional, then requested implementation. Investigation confirmed the old GL
grid readback uses ordinary LINEAR filtering (no mipmaps/area integration), so
it sampled a few source texels rather than averaging each character footprint.
The corrected offline `tmp/ascii_full_pipeline.py` quantizes each independent
720x960 source colour plane in floating-point DIN99d, with serpentine
Floyd–Steinberg error diffusion in linear RGB. DIN99d equations follow the
standard transform documented by Chafa's public `chafa-color.c`; this is a new
compiled experimental kernel, not a claim of invoking Chafa's CLI dithering.
Diffusion does not cross sharp alpha transitions. Source alpha remains exact.
No source layers are premixed or baked together.

Only afterward, normal full-resolution GL compositing is performed, followed
by explicit linear-light INTER_AREA reduction to 60x40 and current grading.
The matched RGB reference uses the same area reduction; it deliberately differs
from the older sparse-grid sampling baseline. A fine black/white checkerboard
test yields sRGB 188 (linear-light half intensity), not encoded-RGB 128. All
240 fixed palette colours survive the source kernel unchanged. Mean image
CIEDE2000 between source-dithered and unquantized area RGB is 0.426, 0.426 and
0.372 for the three pairs; these are whole-image means, not skin-only figures.

Final output candidates compare nearest DIN99d palette mapping with an
experimental discrete spatial palette search. The latter minimizes two
low-pass linear-opponent residuals using average project-font glyph coverage,
plus local colour-excursion penalties, over twelve nearby palette entries.
It is not S-CIELAB, a validated terminal appearance model, or an implementation
of the cited edge-aware research paper. The optimizer preserves glyphs and is
asserted to decrease its own objective; that does not establish visual quality
or motion stability. Offline sample generation is about one second per pair
after compilation, not a realtime application backend. No production defaults
or dependencies were changed.

The local :2323 loop now shows A unquantized RGB AREA reference before each
B source DIN dither RGB AREA, C source DIN -> nearest ANSI, D source DIN ->
spatial ANSI, at two seconds per image across three source pairs. Only C/D are
indexed ANSI; B isolates the pre-final result. External receipt verified the
six A/B/A/C/A/D 60x41 frames, actual escape formats and 2.00-second intervals.
Evidence: `tmp/ascii_full_pipeline/{report,live_check}.json`, per-pair NPZs,
`tmp/ascii_full_pipeline_generate.log`. Visual acceptance remains pending.

User rejected the image conversions but confirmed the RGB baseline was fine,
then requested a side-by-side visual colour-space/palette guide. The local
:2323 experiment now serves `tmp/ascii_colour_space_guide.py`: a static 63x41
chart with truecolour sRGB ramps on the left and nearest-CIEDE2000 fixed ANSI
mapping on the right. Includes neutral, warm/cool/dark skin, near-white, shadow,
saturated/muted hue ramps and all 240 actual palette entries. No grading,
compositing, image resampling or dithering is involved. The bottom eight rows
show the exact same intended palette RGB colours via truecolour/indexed SGR;
those mappings are asserted exact. External delivery of the complete chart and
both escape formats passed. Evidence: `tmp/ascii_colour_guide/{guide.ansi,mapping.json}`.
This is a truecolour-vs-fixed-palette guide, not a claim that DIN99d is another
display gamut. Production rendering defaults remain untouched.

### High-detail glyphs with simplified full-RGB runs

User clarified plain nc is the target and ASCII-web is to stay unchanged at
60x40. The goal is to approximate high-resolution RGB appearance with a cheaper
stream, rather than preserve all RGB cells exactly or force a fixed ANSI
palette. User approved a high-detail-glyph / reduced-colour-complexity test.

`tmp/ascii_rgb_runs_compare.py` directly samples independent images_sbs layers
at the original font-aware 90x60 grid, with current truecolour grading and no
area-sampling or palette changes. Every variant retains exactly the original
90x60 glyphs. A compiled per-row dynamic program chooses short shared-colour
runs, minimizing estimated RGB escape bytes. Representatives are original
source tint colours nearest each run's mean OKLab; no synthetic palette is
imposed. Per-cell OKLab distances (100x scale) are bounded at 1, 2 or 3 for
gentle/medium/strong. Runs cannot cross an adjacent jump above 6 in those units.
This is horizontal, colour-edge-constrained simplification, not a full 2D
segmentation or a semantic face/feature model.

Across the same three pairs, baseline payloads are 91.2–98.6 KB. Gentle gives
60.4–63.1 KB (32–39% saving), medium 38.9–43.4 KB (52–61%), strong 28.9–33.7 KB
(63–71%). Whole-image mean CIEDE2000 is .34–.37, .77–.79 and 1.11–1.14;
these do not establish acceptability. Median simplification times are 2.7–4.9 ms
on the test host after compilation. The encoder byte cost is additional to this
kernel timing. JIT/RSS, concurrent-viewer costs and moving-video stability are
not yet benchmarked for production; this does not change application defaults.

Validation checks a hard-edge fixture, maximum per-cell error bounds, all
5400 unchanged glyphs and independently decoded RGB cells per variant. The
output-only local :2323 loop now alternates A/B/A/C/A/D, two seconds per image
across three pictures. It requires 90 columns and 61 rows including the label.
External receipt verified all six 90x61 packets, truecolour-only escapes and
2.00-second intervals. Evidence: `tmp/ascii_rgb_runs/{report,live_check}.json`,
NPZs and `tmp/ascii_rgb_runs_generate.log`. A rasterized contact sheet was
inspected, not an actual terminal screenshot. User visual acceptance is pending.
ASCII-web and the reference endpoint/configuration remain untouched.

User then requested only the second still picture at a faster cadence; the
saved 1332 variants were served at .8 seconds (A/B/A/C/A/D), with no regeneration.
Afterward user requested a run at medium. `tmp/ascii_rgb_runs_live.py` is an
opt-in launcher around the actual `main.py --mode ascii --port 2323 --dir
/home/linuser/modemTest/images_sbs` pipeline. It wraps truecolour encoding with
the medium run kernel (OKLab bound 2, edge stop 6), fixes output at 90x60, and
interprets no client controls. Original clock/preparation and per-grid rendering
remain in charge; glyph selection happens before the tint simplification. The
compiled kernel is warmed before opening the port. This experimental launcher
imports Numba/JIT infrastructure and its added memory cost is not yet measured;
it is not installed as a production default or applied to ASCII-web.

External eight-second receiver check observed 118 distinct full 90x60 RGB sync
batches, 14.66 FPS, mean 37,852 bytes/frame (4.44 Mbps observed), median/p95 gaps
66.6/86.7 ms and a 212 ms maximum. This unmatched sample is not a general quality,
bandwidth or zero-stall guarantee. Evidence: `tmp/ascii_rgb_runs_live.log` and
`tmp/ascii_rgb_runs_live_check.json`. The live medium run replaces the temporary
still-image comparison on :2323; no changes to reference routing/service.

User found live 90x60 medium visually solid, then requested a terminal-sized
60x40 medium run; ASCII-web was explicitly left unchanged. The opt-in launcher
now fixes terminal output to 60x40 (same medium error/edge bounds and 15 FPS
clock target). Earlier 90x60 evidence is preserved in its original log. A
12-second external receiver check observed 177 distinct 60x40 frames at 14.69
FPS, mean 18,165 bytes/frame, about 2.13 Mbps, median/p95 gaps 66.6/68.6 ms and
203 ms maximum. Five matched original/medium encoding samples averaged
31,440/17,376 bytes, 44.7% saving at this resolution; sample coverage is small.
Evidence: `tmp/ascii_rgb_runs_live_60x40.{log}` and
`tmp/ascii_rgb_runs_live_60x40_check.json`. Normal production defaults remain
unchanged; active test service is moving 60x40 medium, with no client controls.

User subsequently requested slightly higher brightness/contrast and the lower
intensity compression option. The same opt-in launcher now runs 60x40 gentle
RGB sharing (OKLab bound 1, edge stop 6), contrast 1.1, glyph brightness 1.5,
and separate tint brightness 1.05. These are launcher/CLI overrides, not normal
application defaults or ASCII-web changes. The previous medium log is preserved.
External four-second receipt verified 61 distinct 60x40 truecolour frames at
14.96 FPS (mean 24,425 bytes/frame in this unmatched sample). Evidence:
`tmp/ascii_rgb_runs_live_60x40_gentle.log` and corresponding `_check.json`.
The active :2323 experiment is now gentle with this brighter grading; no
matched bandwidth or clipping-quality claims are inferred from the short check.

User observed noise and black clipping. A converter grayscale-ramp check found
the raised contrast/tint settings map source 0..12 to black and reach white at
233; output is monotonic, with no integer wraparound. Previous neutral grading
mapped only 0 to black. User approved reverting contrast alone to 1.0. Active
launcher now retains gentle sharing, glyph brightness 1.5 and tint brightness
1.05 at 60x40, with neutral contrast. External receipt of a complete 60x40 RGB
frame passed after restart. Current log:
`tmp/ascii_rgb_runs_live_60x40_gentle_neutral.log`. Prior logs are preserved;
production defaults and ASCII-web remain unchanged.

User approved neutral contrast visually but found midtones dark. The opt-in
launcher adds a separate tint-value gamma .9 immediately before gentle colour
sharing; original glyph selection/gamma remain unchanged. A one-time value
scale LUT and compiled kernel scale all three RGB channels together, preserving
ratios within rounding. Checks pin black/white endpoints, monotonic grayscale,
midgray 128 -> 137, and <=.5 channel rounding error on two warm colours. Existing
contrast 1.0, glyph brightness 1.5 and tint exposure 1.05 remain unchanged.
After restart, a complete external 60x40 RGB frame was verified. Current log:
`tmp/ascii_rgb_runs_live_60x40_gentle_midlift.log`. This tint curve does not undo
any highlight clipping from the retained exposure setting. Visual acceptance
and matched new payload measurements remain pending; no application defaults
or ASCII-web changes were made.

### Accepted appearance and pre-integration measurements

User accepted the 60x40 gentle/midtone-lift live appearance and approved
measurement before production integration. Accepted experiment: gentle RGB runs
(maximum per-cell deltaEOK 1, hard adjacent-edge stop 6), contrast 1.0, glyph
brightness 1.5, tint exposure 1.05, separate tint gamma .9, 15 FPS target, direct
images_sbs with original independent layers and clock. ASCII-web remains
unchanged. The temporary live service remains running with these settings.

`tmp/ascii_accepted_benchmark.py` captures 18 actual independently composited
60x40 grids (six indices, three folder pairings), then benchmarks each converter
in a separate fresh process for 720 warmed conversions. Both paths use exactly
the same accepted grading and midtone curve; RGB-only uses the equivalent
vectorized tint LUT without importing JIT infrastructure, and gentle uses the
experimental compiled kernels. All output glyphs match. These are converter
measurements, not matched whole-application CPU/RAM benchmarks:

| Metric | RGB-only accepted grading | Gentle accepted grading |
|---|---:|---:|
| Mean encoded frame | 42,385 bytes | 31,476 bytes |
| Payload at 15 FPS | 5.09 Mbps | 3.78 Mbps |
| Median conversion wall time | .964 ms | 1.774 ms |
| Mean CPU per frame | 1.035 ms | 2.050 ms |
| Fresh converter RSS | 139.4 MiB | 269.6 MiB |

Matched static-source payload saving is 25.7%, with about 1.01 ms additional
CPU/frame and a serious 130.1 MiB JIT-related memory regression. Thirty
same-frame samples from accepted live logs give 38,052 -> 26,309 bytes (30.9%
saving); content differs from the earlier medium/90x60 measurements, so those
savings should not be quoted for this accepted configuration. Live aggregate
measurements remain content-dependent and exclude network overhead.

`tmp/ascii_accepted_live_measure.py` opens one small-receive-buffer local socket
that consumes nothing for 12 seconds while observing a second fast reader via
the public test endpoint. The server logged a send timeout and the stalled
connection's disconnect; draining 25,477 buffered bytes reached EOF. The other
reader received 176 distinct full frames at 14.67 FPS, median/p95 gaps
66.6/68.2 ms, max gap 204 ms. During this observation the actual Python process
used 19.65% of one CPU core and RSS ranged 467.6–495.3 MiB. No matched reference
whole-application resource run was made. The first resource probe accidentally
selected the xvfb wrapper; this was corrected to Python and the reported run is
the corrected measurement. This short test is not a comprehensive load test or
proof of terminal paint latency or bounded kernel/terminal buffering in all
network conditions.

Evidence: `tmp/ascii_accepted_measure/{summary,live,converter_rgb,converter_gentle}.json`,
`source_grids.npz`, `source_identities.json`, `live_log_excerpt.txt` and harnesses
above. No production integration has occurred. Next engineering issue is
retaining the accepted appearance and byte savings without the experimental
JIT-stack memory overhead, then validating moving-content stability and
matched application resources before adopting defaults.

### Native colour-sharing rewrite (2026-10-17)

User requested fixing the JIT prototype's RAM/CPU overhead. The optional engine
is now `ascii_color_sharing.py` plus `native/ascii_color_sharing.c`, explicitly
built using `utilities/build_ascii_color_sharing.py`. It implements tint lift,
the same perceptual DP and UTF-8 run encoding without Numba/llvmlite or runtime
compilation. A conservative triangle-inequality bound stops examining longer
segments once no representative could satisfy the original radius constraint.
The NumPy perceptual transform retains the prototype's operation order and
rounding; this is not a new approximation, quantizer or visual tuning change.
The C build disables fast-math/FMA contraction to preserve tie-breaking.

Validation compares the frozen accepted JIT reference on 18 real 60x40 pictures,
three real 90x60 pictures and 90 random/smooth/flat fixtures across tolerances
0/1/2/3 and edge thresholds 0/6. All 888 frame cases (812,616 cell cases) match
RGB arrays and full terminal text exactly. Eight new unit tests cover an
independent unpruned Python oracle, Unicode/empty glyph encoding, validation,
ownership, noncontiguous arrays, threading, explicit missing-build errors and
no JIT imports in a fresh process. The focused suite passes 55 tests. Address
and undefined-behaviour sanitizer/output-guard checks pass on 12 grid sizes.

Fresh matched converter runs (18 source grids, 720 conversions per engine):

| Metric | RGB-only accepted grading | JIT gentle | Native gentle |
|---|---:|---:|---:|
| CPU per frame | 1.006 ms | 1.978 ms | .858 ms |
| Median conversion wall time | .922 ms | 1.755 ms | .809 ms |
| Process RSS | 139.5 MiB | 270.0 MiB | 143.9 MiB |
| Mean encoded payload | 42,385 bytes | 31,476 bytes | 31,476 bytes |

The rewrite removes 126.1 MiB and about 57% of converter CPU versus the JIT
prototype on this host, preserving every payload byte. The native encoder
offsets the colour-sharing arithmetic cost versus the Python RGB-only encoder
in these samples. These are converter figures, not a matched application-RAM
or general cross-platform performance claim.

The live test launcher now uses the new engine with exactly the accepted
60x40/gentle/neutral-contrast/midtone-lift settings, direct images_sbs and the
same frame clock. Before startup :2323 was free; no unrelated process was
terminated. A 12-second external run received 176 distinct frames at 14.68 FPS
(median/p95 gaps 66.1/82.6 ms; maximum 204 ms), mean 20,197 bytes/frame for that
content. Whole-application CPU was 17.35% of one core and RSS 318.6–344.2 MiB;
content/uptime are unmatched to the earlier application probe, so those figures
do not establish exact whole-app savings. A concurrent non-reading local
client timed out/disconnected; the healthy reader continued and draining the
stalled socket reached EOF. The active test service remains on :2323 with the
native engine. ASCII-web, ordinary defaults and reference endpoint are unchanged.

Setup/API/limitations are in `docs/ASCII_COLOR_SHARING.md`. Evidence:
`tmp/ascii_native_rewrite/{summary,parity,live,converter_rgb,converter_gentle,converter_native}.json`,
source grids/identities, `tmp/ascii_rgb_runs_live_native.log` and
`tmp/ascii_native_sanitize.c`. Legacy JIT code remains only in validation
harnesses and is never imported by the new live launcher. Target-Mac validation,
normal CLI/default integration and broader moving-content visual acceptance
remain separate follow-up work.

User subsequently requested trying medium with the rewritten engine. The live
launcher now uses tolerance 2 (medium), retaining edge stop 6, 60x40/15 FPS,
contrast 1.0, glyph brightness 1.5, tint exposure 1.05 and tint gamma .9. External
receipt verified 15 complete 60x40 frames after restart. Current log is
`tmp/ascii_rgb_runs_live_native_medium.log`; earlier gentle evidence remains
unchanged. This is an experimental runtime override, not a new application or
ASCII-web default, and visual acceptance of medium is pending.

### Actual main-branch baseline versus current native medium

User asked for savings against main rather than the RGB prototype. The
`tmp/ascii_vs_main.py` harness loads actual `main` renderer/converter/settings
from commit `9591126d53d4f07a32edb41ba8a0d908a61143d5`. Eighteen matched source
identities are rendered through main's original 60x40 geometry for its converter,
and through the current independent font-aware grid for the current path.
Each converter is measured in a separate fresh process over 720 warmed frames.
Payloads include the respective server framing: main home+rows versus current
row-positioned synchronized updates. Main uses its actual defaults: indexed
colours, blur 7, brightness 1.4. Current uses full RGB, no blur, the accepted
brightness/midtone lift and medium tolerance 2. Output quality is intentionally
different; source identity and nominal character-grid size are matched.

| Metric | Actual main | Current native medium |
|---|---:|---:|
| Mean frame payload | 9,650 bytes | 24,492 bytes |
| Payload at 15 FPS | 1.16 Mbps | 2.94 Mbps |
| Converter CPU/frame | 3.937 ms | 1.007 ms |
| Converter median wall time/frame | 1.030 ms | .977 ms |
| Converter process RSS | 139.5 MiB | 144.2 MiB |

Thus converter CPU is 74.4% lower, converter RAM is 4.6 MiB higher, and payload
is 2.54x main, not a bandwidth saving against main. Process CPU accounts for
all worker-thread time (including main's OpenCV blur); wall latency improves
much less. These are not matched whole-application CPU/RAM measurements and
cannot establish total application resource savings. Bandwidth is normalized
to 15 FPS rather than hiding differences in delivered cadence. No external
reference endpoint was used as a proxy for main's implementation.

For context, the same current RGB frames/settings without colour sharing use
43,443 bytes/frame including framing (5.21 Mbps at 15 FPS). Medium sharing
reduces this to 24,492 bytes (43.6% saving); that is the relevant same-quality-
family optimization, not a claim that full RGB is cheaper than indexed main.
Evidence: `tmp/ascii_vs_main_20261017/{main,current,unshared,summary}.json`,
matched source identities and saved main/current grids. The public live medium
test was not restarted or retuned during this measurement.

### Current visual contract: gentle RGB baseline

User explicitly established accepted gentle RGB appearance—not mainline's
appearance or medium sharing—as the visual baseline. Mainline remains a
bandwidth/resource comparison only. Future output experiments must preserve
the accepted target or be shown explicitly against it, retaining exact frame
selection. At the user's go-ahead, the native live launcher was switched back
from medium to gentle (tolerance 1, edge stop 6), keeping 60x40, target 15 FPS,
contrast 1.0, glyph brightness 1.5, tint exposure 1.05 and tint gamma .9.
External receipt of a complete 60x40 RGB frame passed. Active baseline log:
`tmp/ascii_rgb_runs_live_native_gentle_baseline.log`. Mainline colour restoration
is not an accepted replacement; no default or ASCII-web changes were made.

### Minimal mainline colour-threshold experiment

User approved testing a small mainline colour correction against the accepted
gentle RGB reference. `tmp/ascii_main_colour_ab.py` loads the actual
origin/main converter/settings at 9591126d5. Candidate C changes only
`r*5//255`, `g*5//255`, `b*5//255` to a nearest-channel lookup against the actual
cube levels 0,95,135,175,215,255. Main's blur 7, brightness 1.4, saturation .9,
glyph gamma .9, legacy source sampling, character selection and palette escape
table are unchanged. Even the legacy index-16 -> index-235 black alias is
preserved to isolate the threshold change. The old thresholds are not uniformly
darkening: they can round either up or down relative to the actual cube values.

All 18 matched source pairs retain identical B/C glyphs; near-white 254 now
chooses cube level 255 instead of 215. Mean original/corrected payloads are
9,650/10,371 bytes (home+original rows), about 7.5% more for this candidate.
No overall visual improvement is claimed before feedback.

The local :2323 live baseline process was temporarily replaced by a static
comparison loop of the previously preferred second-expression picture (1332,
face source with blank float). A is accepted gentle RGB, B actual main unchanged,
C threshold-only correction. Sequence A/A/B/C/B/C holds each packet .8 seconds,
so A is visible for 1.6 seconds and B/C alternate directly. It needs 60x41 cells
including a label. External delivery checked all six frames, formats, exact B/C
glyph equality and .80-second intervals. The baseline launcher/settings remain
unchanged and available for restoration; this does not change normal defaults
or ASCII-web. Evidence: `tmp/ascii_main_colour_ab/{report,live_check}.json`, saved
ANSI frames and `tmp/ascii_main_colour_ab_server.log`.

User rejected all panels in this comparison. Audit confirmed B equals the clean
origin/main converter for the saved RGB input and primary-colour channel order
is correct (indices 196,46,21), but this does not validate the reconstructed
render/input path or prove A matches the accepted live application's reference.
Those references were generated from cached manually rendered grids, not
captured from the accepted live pipeline. Do not treat this experiment as a
visually valid baseline or continue tuning against it. The static comparison
was stopped and the unchanged accepted native gentle live launcher restored
on :2323. External receipt verified 15 complete moving 60x40 truecolour frames.
Current log: `tmp/ascii_rgb_runs_live_native_restored.log`. All grading/sharing
settings remain at the accepted baseline. Future AB work must first validate
its reference against actual live output; mainline candidate acceptance remains
unresolved. Audit details: `tmp/ascii_main_colour_ab/audit.json`.

### Actual deployed reference, not reconstructed mainline

User explicitly directed using `gencarelle.fun:2323` as the visual reference.
A four-second read-only capture sent zero commands and received 54 distinct
complete 60x40 frames, 13.30 FPS, mean 28,545 bytes/frame. The endpoint sends
indexed colour instructions for all 2400 cells per frame (129,600 indexed
escapes total), with no truecolour escapes in this capture. Three actual raw
frames and hashes are saved under `tmp/ascii_gencarelle_reference/` with
`capture.json`. No deployment/configuration or endpoint state was changed.

This differs materially from the reconstructed origin/main converter benchmark,
which emits same-colour runs and averaged 9,650 bytes on its matched fixtures.
Those main-branch numbers remain code-benchmark results, not measurements of
the deployed visual reference. Do not treat the source-code reconstruction as
equivalent to this service or infer its live settings from local defaults.
Captured content and timing are unmatched to local frames, so the new endpoint
sample also does not establish a matched saving. The accepted local gentle
RGB visual baseline remains unchanged and running; the actual remote stream
is the read-only reference for investigating the mainline appearance.

User approved trying heavy at 60x40 before shrinking the grid or adding a
stronger custom setting. The launcher now accepts an experiment-only
`ASCII_SHARING=gentle|medium|heavy` environment override, defaulting to the
accepted gentle baseline. Active :2323 run uses `ASCII_SHARING=heavy`
(tolerance 3), retaining edge stop 6, contrast 1.0, glyph brightness 1.5,
tint exposure 1.05 and tint gamma .9. No palette or reference reconstruction
is involved. External four-second receipt verified 61 distinct full 60x40 RGB
frames, 15.07 FPS, mean 16,559 bytes/frame for this unmatched content sample.
Gentle default and ASCII-web remain unchanged. Current evidence:
`tmp/ascii_rgb_runs_live_native_heavy.log` and `_check.json`; visual acceptance
of heavy remains pending. Restore gentle by launching without the override.

User rejected heavy for crushed shadows and apparent white-to-black changes.
The heavy run was stopped after saving three actual wire frames, and the
accepted gentle launch restored unchanged. A diagnostic-only
`ASCII_SHARING_CAPTURE` option saves at most six actual live encoder input/
glyph/text samples (counts 1,31,61,91,121,151); it is disabled by default and
does not reconstruct sources. Gentle output reproduced those samples exactly.
Offline heavy processing of the same six inputs respected its deltaEOK<=3
bound, created no new black cells, and turned none of nine near-white cells
dark. This small sample does NOT reproduce or explain the user's reported
white/black artefact; it is not proof that terminal painting or other content
is correct. Flattening of shadow gradations remains an unacceptable visual
tradeoff for this heavy trial. External receipt of 164 complete live 60x40
frames passed after restoring gentle. Evidence is under
`tmp/ascii_heavy_shadow_audit/` (`heavy_wire_*.ansi`, `live_input_*.npz`,
`audit.json`); current log is `tmp/ascii_rgb_runs_live_gentle_shadow_restore.log`.
Gentle remains the visual baseline; no production or ASCII-web default changed.

### 4:2:0-like chroma/protected-lightness experiment

User proposed retaining full brightness detail while reducing chroma, and
approved a test. `tmp/ascii_chroma_experiment.py` uses six actual captured live
encoder inputs (whose gentle text is asserted byte-identical to the capture)
plus 18 matched source grids with accepted grading. It averages OKLab chroma
within 2x2 cells only in smooth regions, leaves L unchanged before RGB rounding,
avoids large lightness/colour edges and the darkest/brightest cells, and reduces
chroma toward neutral when needed to remain in sRGB rather than channel-clipping.
Black, white, primary flat patches and a hard black/white edge remain unchanged.
This is a 4:2:0-like visual experiment, not a YUV codec or new wire format.

The native helper/API gained optional `lightness_limit` (default disabled) and
`share_colors` for already lifted tints. Protected representatives must satisfy
every cell's lightness interval, tightening toward both endpoints; among those,
the same perceptual representative rule is used. Default gentle/native output
still matches the frozen accepted prototype in all 888 parity cases. The focused
suite passes 56 tests and updated address/undefined sanitizer checks pass on
12 grids, with protection both off and on. New helper builds are explicitly
source-keyed as before; no JIT dependency was added.

Across 24 cases, mean encoded body sizes are gentle 28,799 bytes, chroma-only
38,835, chroma plus .5 lightness protection 33,442, chroma plus 1.0 protection
31,257, and protected sharing without chroma subsampling 31,237. Every glyph is
unchanged. Protected variants show lower mean lightness error than gentle,
with no new black cells or bright-to-dark conversions in the test; this does
not explain the prior reported terminal artefact or establish visual acceptance.
The initial unconstrained-representative version was slightly worse in payload;
selecting inside the feasible interval did not reverse the result. The candidate
is not a bandwidth win versus gentle: separate brightness changes still require
full RGB instructions in plain nc. Nothing was deployed or retuned on :2323;
the accepted live gentle baseline remains running. Evidence:
`tmp/ascii_chroma_experiment/report.json` and per-case NPZs. Further improvements
must be justified against the accepted baseline rather than assuming codec-style
chroma savings transfer to terminal RGB text.

CPU follow-up: `tmp/ascii_chroma_cpu.py` measures each variant in a fresh process
on the same six captured live encoder inputs, 240 warmed conversions each.
Gentle colour-stage CPU averages 2.069 ms/frame (median wall 1.913 ms), versus
17.790 ms/frame (median wall 17.050 ms) for chroma subsampling plus protected
sharing. At 15 FPS that projects about 3.1% versus 26.7% of one core for this
stage, not the full application. This six-frame cohort differs from the earlier
18-frame converter benchmark; do not mix their absolute times. An initial
short in-process timing was superseded by these fresh-process measurements.
The chroma prototype uses Python/NumPy block processing and repeated gamut
checks, so its current cost is not an inherent limit of chroma subsampling.
Nevertheless it loses both bandwidth and CPU against accepted gentle and was
not deployed. Evidence: `tmp/ascii_chroma_experiment/cpu_{gentle,chroma}.json`.

### Corrected interpretation: subsampled comparison model, not an RGB pipeline

User clarified that 4:2:0 was intended to simplify the existing colour-sharing
model, not add a separate processing pipeline, and approved that limited test.
The native engine now has optional `chroma_model='420'`, default `'full'`.
It averages only OKLab comparison a/b within smooth 2x2 blocks, keeps every
original comparison L bit-for-bit, and keeps original features in blocks with
a strong colour/lightness boundary. Run boundary checks use the original full
features, not the coarsened model. Source RGB representatives, accepted tint
lift, glyphs and run encoder are unchanged. No RGB reconstruction, gamut
mapping, new palette or transport is used. Expanded model storage and the
initial full transform remain; this first version does not claim sparse-memory
or transform-work reductions.

`tmp/ascii_comparison_model420.py` evaluates 24 same inputs (six actual live
captures with exact accepted text reproduction, plus 18 matched source grids).
At gentle tolerance 1 in both cases, mean body sizes are 28,799 bytes full-model
versus 28,176 coarsened-model: 2.16% saving (about 623 bytes/frame). Both keep
max L error below 1, create no new black cells or bright-to-dark conversions,
and preserve all glyphs. As expected for a chroma approximation, max original
full-colour deltaEOK rises from 1.00 to 4.90, while mean error rises from .141
to .167. The model tolerance is not a bound on original full-chroma colour
distance; this requires visual validation rather than calling it equivalent.

Fresh-process 720-conversion measurements on those same 24 inputs give colour
stage CPU .969 ms/frame full-model versus 1.058 ms/frame coarsened-model;
median wall .734/.804 ms. This is about .09 ms extra, not the earlier separate
pipeline's 17.79 ms, but it is not a CPU saving. At 15 FPS the stage projects
1.45/1.59% of one core, excluding rendering/loading. The focused suite passes
58 tests; default full-model native output still passes all 888 exact parity
cases, and updated address/undefined sanitizer checks pass 12 grids. The new
mode is not enabled in the live service: accepted gentle remains unchanged
and listening on :2323. Evidence: `tmp/ascii_model420/{report,cpu_gentle,cpu_model420}.json`
and per-case NPZs. Actual visual acceptance and motion stability remain pending.

User approved visual A/B after clarification that no visual improvement had
yet been demonstrated. `tmp/ascii_model420_ab.py` uses actual accepted live
encoder capture `live_input_31.npz`; A's text is asserted byte-identical to the
captured accepted output. B changes only comparison chroma to the new '420'
model. Original RGB representatives, glyphs, grading, geometry and midtone curve
stay the same, with no source reconstruction. This picture uses 21,451/20,675
body bytes (3.62% saving), not a replacement for the 24-case aggregate result.
The local :2323 process temporarily serves A/B alternating at .8 seconds,
requiring 60x41 cells with the label. External receipt verified A/B/A/B, matching
glyphs and .80-second intervals. Gentle launcher/defaults remain unchanged and
available for restoration. Evidence: `tmp/ascii_model420_ab/{report,live_check}.json`
and `tmp/ascii_model420_ab_server.log`. Visual acceptance is pending.

User reported the gentle A/B looked identical and requested the same model
comparison at heavy. `ASCII_AB_SHARING=heavy` now generates/serves A original
full-chroma heavy versus B heavy with comparison-model-only '420', both tolerance
3 and unchanged source/glyphs/grading. The same actual live input is used, and
its captured gentle text is still verified exactly before either heavy variant
is generated. A heavy is therefore not mislabelled as the captured gentle output.
For this one picture body sizes are 14,969 vs 14,897 bytes (.48% saving), not
an aggregate claim. The :2323 loop alternates A/B at .8 seconds; external
receipt checked A/B/A/B, equal 60x40 glyphs plus label and .80-second intervals.
Gentle remains the accepted saved baseline. This model-only change does not
enable additional lightness protection; heavy's shadow issue is not presumed
fixed. Evidence: `tmp/ascii_model420_ab_heavy/{report,live_check}.json` and
`tmp/ascii_model420_ab_heavy_server.log`.

User requested more varied images (not face-plus-black) and a pure-RGB reference.
`tmp/ascii_capture_colourful.py` obtains three fresh inputs from actual main.py
render/converter execution with controlled still indices and active layer pairs:
(511, face 11/float 8), (888, face 9/float 11), (1776, face 15/float 6). In capture
workers only, folder/clock selection is fixed and network server startup is
replaced by an internal pictures consumer; no viewer/monitor ports are opened
and public comparison :2323 is untouched while capturing. Both layers are still
selected/composited independently through the normal application renderer.
Saved encoder RGB/glyph inputs and accepted-gentle text are checked exactly.

`tmp/ascii_heavy_colourful_ab.py` now serves A pure RGB (no sharing, accepted
grading and tint lift), B original heavy, C heavy with comparison-only '420'.
For each picture it runs A/B/A/C twice, .8 seconds per packet, then the next
picture; all three repeat. Every variant retains identical original glyphs and
has a 60x41 labelled display. A is explicitly not the gentle-shared baseline:
it disables sharing only, as requested. Mean image/grade/source reconstruction
is not involved. Individual body sizes pure/heavy/model420 are 35,568/12,762/12,749,
37,279/15,210/14,976 and 40,715/16,353/16,296 bytes. These are per-picture figures,
not general compression or quality guarantees. External receipt verified the
first complete A/B/A/C/A/B/A/C cycle, equal glyphs, .80-second intervals and
transition to picture 2. Evidence: `tmp/ascii_colourful_live_sources/` worker
captures/logs, `tmp/ascii_heavy_colourful_ab/{report,live_check}.json` and
`tmp/ascii_heavy_colourful_ab_server.log`. Accepted gentle settings/defaults
and the actual gencarelle reference service remain unchanged.

User then selected picture 3 only, followed by requesting ten random pictures
at 1.5 seconds each, with only original RGB vs new heavy. Active experiment is
now `tmp/ascii_rgb_heavy_random10.py`: ten unique randomly selected image indices
with random face folders and nonblank active float folders, captured through
the actual application worker pipeline (no network ports opened by workers).
The seed and choices are recorded so the set remains fixed for repeatable AB.
A is original full RGB without sharing; B is heavy tolerance 3 with comparison
model '420', with identical accepted grading/glyphs and no lightness protection
added. Each A/B pair uses the same actual captured input and lasts three seconds
total (1.5 seconds per variant); ten pairs loop in thirty seconds.

All twenty saved 60x41 labelled packets were checked for dimensions and equal
glyphs per pair. External receipt verified A/B for pictures 1 and 2 at 1.50-second
intervals. Mean body sizes in this random set are 40,090 bytes RGB and 17,165
bytes new-heavy; these are sample sizes, not proof of visual acceptance or a
matched comparison against gencarelle. Gentle baseline settings remain intact.
Evidence: `tmp/ascii_rgb_heavy_random10/{selection,report,live_check}.json`,
actual captures/worker logs, saved ANSI frames and
`tmp/ascii_rgb_heavy_random10_server.log`. The active :2323 process serves this
static comparison loop; ASCII-web and the remote reference remain untouched.

User requested looping only the first face folder, explicitly as A/B.
`tmp/ascii_first_face_loop.py` now launches actual moving application playback
with face/float selection fixed to (0,0) (first face folder, blank float),
launch-relative free clock and forward wrapping rather than ping-pong. These
are wrapper-only test overrides, not production clock/default changes.
`ASCII_AB_PERIOD=1.5` and `ASCII_CHROMA_MODEL=420` enable original-RGB/no-sharing
versus new-heavy sharing within the normal encoder path. A/B labels are tied
to the encoded picture version; all original 40 image rows are shifted down one
to accommodate the label. Clock selection keeps advancing during each 1.5-second
mode interval; pictures are not replayed or held at switches. Thus this is a
motion/stability AB, not identical-index side-by-side rendering.

External receipt over five seconds verified 77 complete distinct 60x40 picture
frames plus label, both A/B modes and transitions at 1.50/1.53/1.47 seconds
(quantized to publication times). Source is images_sbs, all accepted grading
is unchanged, and gentle baseline remains the default without wrapper/env
overrides. Active :2323 now serves this moving first-face-folder test. Evidence:
`tmp/ascii_first_face_ab_live.log` and `_check.json`. ASCII-web and the actual
gencarelle reference service remain untouched.

First-folder moving AB CPU follow-up: `tmp/ascii_first_face_cpu.py` captures
six actual first-folder/blank-float inputs at indices 0/444/888/1332/1776/2220
using the no-port application workers, without changing the public test. Each
path is measured in a separate fresh process over 480 warmed encodes. Original
RGB uses the actual AB path (native tint lift, Python RGB row encoder); new
heavy uses native sharing with tolerance 3/comparison model '420'. Original RGB
averages 1.008 ms CPU/frame, heavy 1.264 ms: about .256 ms extra, projecting
1.51%/1.90% of one core at 15 FPS for this encoding stage only. Median wall
times are .746/1.128 ms. This cohort/stage differs from earlier converter
benchmarks and must not be mixed with them.

A separate twelve-second observation of the running first-folder AB received
181 frames. Whole-application CPU during stable A/B intervals was 19.10%/17.76%
of one core, RSS 346.7 MiB. These alternating live intervals show different
poses and include rendering/loading/delivery, so the lower observed B CPU is
not a matched causal saving; the controlled encoding test above is the overhead
comparison. Evidence: `tmp/ascii_first_face_cpu/{rgb,heavy,live}.json`, actual
capture NPZs and worker logs. The moving public AB was left unchanged.

User asked how new-heavy CPU compares with OG code. A fresh actual-main
comparison uses the same 18 matched source identities and the respective
main/current saved GL grids as before, 720 warmed conversions per fresh process.
Actual main commit 9591126d5 averages 4.699 ms process CPU/frame, median wall
1.098 ms, 139.2 MiB converter RSS. Current heavy tolerance 3/'420' model averages
1.679 ms CPU/frame, median wall 1.691 ms, 144.3 MiB RSS. This is 64.3% less
total converter CPU, not lower wall latency: main's OpenCV blur can consume
worker-thread CPU in parallel. CPU-time normalization at 15 FPS gives about
7.05% vs 2.52% of one core for conversion, not total live application utilization.
The full converter scope includes grading and text/framing, unlike the earlier
encoding-only first-folder test; cohorts and scopes must not be mixed.
Payload means remain 9,650 vs 19,685 bytes, reflecting different output quality.
This compares local source code, not CPU on the remote gencarelle deployment;
no matched OG whole-application resource run was made. Evidence:
`tmp/ascii_heavy420_vs_main/{main,heavy420}.json`, saved grids/identities.

### Actual 30-second full-application main versus new-heavy verification

User explicitly requested running main-branch baseline, not just sampling the
current AB or comparing converter microbenchmarks. `tmp/ascii_30s_main_vs_heavy.py`
archives origin/main 9591126d5 into an isolated repo-local test directory and
runs its actual application, then current application/new heavy, in separate
processes. Both use images_sbs first face/blank float, 60x40, target 15 FPS,
30 IPS forward free clock, same software GL environment, and a fast loopback
reader. Test-only folder/clock wrappers set a common source-relative timeline;
both warm through source seconds 0..10, then measure seconds 10..40 without
holding or resetting the clock. No colour/render/delivery implementation is
patched in main. Main retains actual indexed/blur-7/brightness-1.4 defaults;
new heavy uses accepted RGB grading, native sharing tolerance 3 and '420'
comparison model, no AB caption. These are different output quality paths.

The initial harness attempted a held index-zero warmup, which main could miss
without retrying; it produced no measurement and was discarded. Corrected runs
use continuously advancing clocks for warmup and measurement. Viewer/monitor
bind to private 127.0.0.1:2335/2336 via server_config's explicit test-port
override. The remote gencarelle deployment is untouched.

| Actual 30-second observation | origin/main | New heavy |
|---|---:|---:|
| Measured duration | 30.0003 s | 30.0003 s |
| Received payload | 3,674,345 bytes | 8,382,097 bytes |
| Actual payload rate | .980 Mbps | 2.235 Mbps |
| Completed frames / delivery FPS | 399 / 13.32 | 450 / 15.00 |
| Mean complete frame | 9,209 bytes | 18,627 bytes |
| Rate normalized to 15 FPS | 1.105 Mbps | 2.235 Mbps |
| Whole-app CPU, one-core convention | 51.30% | 16.90% |
| Mean RSS | 380.1 MiB | 318.8 MiB |
| RSS range | 376.4–382.9 MiB | 317.1–321.7 MiB |
| Median / p95 receive gap | 69.2 / 90.9 ms | 66.4 / 82.4 ms |
| Maximum gap | 158.2 ms | 87.8 ms |

This workload shows ~67% less whole-app CPU and ~16% lower mean RSS, while new
heavy sends 2.28x the actual 30-second bytes and 2.02x the bytes per frame.
The actual-byte ratio also reflects heavy delivering more frames. It is one
pair of runs on one first-folder workload, not a universal system/platform
claim. Requested source timelines are matched; delivered frame sets differ
because original/current scheduling and cadence differ. These loopback payload
counts exclude TCP/IP headers and do not measure terminal paint time or CPU on
the deployed gencarelle server.

Evidence: `tmp/ascii_30s_main_vs_heavy/{main,heavy,report}.json`, process logs and
test-only archive. Both private workers were stopped after measurement. The
public first-folder moving original-RGB/new-heavy AB was restored on :2323 and
external receipt of a full 60x41 labelled frame verified. Current public log:
`tmp/ascii_first_face_ab_after_30s_check.log`. Accepted gentle defaults remain
unchanged. `ASCII_TEST_PORT` and optional AB-period overrides are launcher-only
test controls, not normal CLI/default or ASCII-web changes.

### Payload attribution and effective-resolution caveat

User questioned whether the large bandwidth difference could really be only
RGB. `tmp/ascii_payload_attribution.py` separates the same 18 actual-path
packets into glyphs, visible colour instructions and framing/state controls.
Both paths emit 2400 glyph bytes. Actual main averages 665.5 indexed colour
instructions at 10.77 bytes (7165 bytes total); new heavy averages 927.1 RGB
instructions at 17.41 bytes (16,145 total). Main framing is 85 bytes; current
framing/state overhead is 1140 bytes, including 40 indexed reset instructions
that colour no visible cell at the exact 60x40 size. Thus the gap combines
longer escapes, ~39% more visible colour changes, and ~1055 extra framing bytes,
not RGB escape length alone. Blur/quantization make main's colours more
correlated; current preserves finer variations.

An important comparison limitation was also verified: in the tested OG path,
every adjacent horizontal glyph pair is identical. Original main's legacy
cover/nearest conversion doubles columns after the small 60x40 GL readback,
leaving only 30 independent horizontal glyph positions in nominal 60x40 output.
All 18 stored main fixtures show this. To ensure it was not a reconstruction
artifact, a fresh actual main application was run privately and three raw
received frames were checked: all have 100% identical horizontal pairs. Current
fixtures have about 43% pair equality from natural repetition, not a forced
duplication. The three actual gencarelle reference captures have 50.1%, 44.7%,
41.9%, so that deployed stream is not exhibiting the same forced duplication.
No remote configuration was inspected or changed to explain the difference.

The 30-second main-code numbers remain valid for the code/path actually run,
but matching source timeline and nominal 60x40 did NOT match effective glyph
detail. Do not present them as equal-visual-information compression efficiency
or as the deployed gencarelle baseline. The rendering, blur, grading and colour
precision differences all need to remain explicit.

Diagnostic ablations hold current graded/lifted picture and glyphs fixed:
unshared RGB body 42,381 bytes, heavy RGB 18,623, indexed nearest-cube with no
blur 11,941, indexed plus blur 7 9923. Indexed ablations are lossy diagnostics,
not accepted replacements. Evidence: `tmp/ascii_payload_attribution/{report,summary,actual_main_pairs}.json`,
actual main raw frames and capture log. The public moving AB is unchanged.

### 0.5-Mbps live-SBS target: exact-update and bounded-error trials

User explicitly requested 0.5 Mbps with little/no visible change. Interpret
this as 500,000 bit/s of plain-terminal payload at the existing 60x40/15-FPS
contract, not a new frame-rate or resolution reduction. Budget is 4167 bytes
per update. No baking, premixing, fixed-palette replacement or ASCII-web change.

Added experimental `ascii_delta.TerminalDelta`, enabled in the TCP handler only
by `settings.ASCII_DELTA_UPDATES`; the temporary live launcher exposes
`ASCII_DELTA_UPDATES=1`. Default is OFF. Each connection retains only its own
last screen; skipped producer pictures do not become queued updates. Resize
clears encoder state, reconnect starts fresh, send failure closes connection.
Unknown/incomplete ANSI input falls back to full output and forgets state.
The encoder removes redundant state controls, carries foreground between rows,
does not send invisible space tints, skips unchanged visible cells and chooses
between repainting short unchanged gaps and cursor positioning. It never
changes a visible glyph/tint. Input is restricted to the application's complete
single-column-glyph, black-background terminal pictures, not arbitrary ANSI.

`tmp/ascii_delta_measure.py` captured an actual first-face/blank-float live SBS
heavy/'420' process over private loopback, source-relative warmup 10 seconds,
30.06-second measurement, 452 pictures at 15.04 FPS. Both encoders processed
the identical received pictures. Every reconstructed delta screen was compared
against its full input using an independent terminal emulator, all exact.

* Full stream: 18,624.6 mean bytes/update, 2.240 Mbps observed.
* Lossless delta: 17,617.6 mean bytes/update, 2.119 Mbps observed (5.4% saving).
* Experimental Python delta CPU: 6.47 ms/update, additional to application
  work. Worker: 17.90% one core, mean RSS 318.7 MiB. This was shadow encoding
  in the measurement process, not a server-CPU delta benchmark. Do not deploy
  this prototype as a demonstrated CPU improvement.

`tmp/ascii_temporal_trials.py` replayed those exact consecutive inputs, holding
each displayed tint only if within a bound of the CURRENT input tint, never
relative to an accumulating approximation. Glyphs/cadence unchanged. At extra
0.5/1/2/3 DEOK, 15-FPS-normalized rates were 2.074/2.117/2.376/2.509 Mbps.
More retention disrupted spatial colour runs and INCREASED bytes. Rejected;
no live integration. These bounds are relative to already-heavy output, not
the accepted gentle/unshared original.

`tmp/ascii_spatial_trials.py` tested additional native full-chroma run sharing
on the same heavy input, preserving glyphs and original edge stop 6. At extra
3/4/6 DEOK, normalized rates were 2.049/1.796/1.455 Mbps. At extra 6, p95 colour
error was 4.90 DEOK; increasing tolerance up to 24 barely improved rate because
edge constraints remain. These are error/bandwidth diagnostics, NOT visually
accepted profiles. No assertion that the added errors are invisible. Even
these stronger trials did not reach 0.5 Mbps. Do not relax edge constraints or
declare the target achieved on this evidence.

Evidence: `tmp/ascii_delta_measure/report.json`, its 452 raw ANSI inputs and
worker log; `tmp/ascii_temporal_trials/report.json`,
`tmp/ascii_spatial_trials/report.json`. Six new delta unit tests cover random
moving screens (including skipped frames), repaint/cursor choices, unchanged
frames, invisible space tints, resize, independent clients and fallback. The
public original-RGB/heavy moving AB is left on its previous settings; new
delta/colour experiments are not silently enabled. Target remains unmet.

### Accepted-heavy reference and run-coherent temporal follow-up

User judged the CURRENTLY DISPLAYED heavy/'420' profile good, then authorized
continued bandwidth work. This accepts that heavy appearance, NOT the stronger
offline sharing trials above, and does not change shipped gentle defaults.

Added opt-in `ascii_temporal.TemporalRunPlanner` and a reentrant native
`ascii_temporal_plan` kernel to the existing explicitly built helper. A row DAG
chooses between retaining a prior cell and drawing a single-colour interval.
Candidates are actual current/previous row RGB tints, not a fixed palette.
Each candidate must meet the full-chroma error bound relative to the CURRENT
accepted-heavy input; the previous approximation is never the error reference.
Draws cannot cross original input edge jumps above 6. Cost estimates charge
cursor placement, RGB instruction length and UTF-8 glyph bytes. Output glyphs
and frame clock are unchanged. `TemporalTerminalEncoder` compares ACTUAL final
delta costs against exact-reference updates from that same displayed screen,
falling back to reference colours if those cost less. It retains only one
per-client screen, resets on clear/resize and is not used by ASCII-web.

Matched replay of all 452 consecutive actual live SBS captures:

| Extra full-chroma error bound vs heavy | Mean bytes | Rate at 15 FPS |
|---|---:|---:|
| 0 (exact delta) | 17,618 | 2.114 Mbps |
| 0.5 DEOK | 17,055 | 2.047 Mbps |
| 1 DEOK | 16,327 | 1.959 Mbps |
| 2 DEOK | 15,180 | 1.822 Mbps |
| 3 DEOK | 13,239 | 1.589 Mbps |
| 4 DEOK | 11,648 | 1.398 Mbps |
| 6 DEOK | 10,207 | 1.225 Mbps |

For extra bound 1, mean/p95/max added error was .212/.881/1.000 DEOK, measured
against heavy output (not original unshared RGB). Every glyph and every delta
screen reconstruction was verified. Planner including conversion/feature
calculation averaged 1.90 ms/frame in offline replay. Full offline harness
averaged 17.15 ms/frame including BOTH encodes, colour extraction, independent
emulation and error measurement; this is NOT live encoder/application CPU.
Even the stronger bounds remain far from 0.5 Mbps. Do not claim achievement or
visual invisibility from numerical bounds alone.

`tmp/ascii_heavy_temporal_ab.py` now exposes only bound-1 as a temporary public
moving AB: A = accepted heavy reference, B = optimized heavy/temporal runs,
switching every 3 seconds while the first-face/blank-float clock advances.
This REPLACES the earlier unshared-RGB/heavy comparison on test :2323.
It adds a 60-column label row, preserving the 60x40 picture. Cold joins and A
phases send full reference pictures; B uses per-connection state. Headers are
never colour-approximated. Normal application configuration remains unchanged.

A private actual-network 30.07-second check received 452 updates, 15.03 FPS,
and reconstructed all 2460 screen cells throughout both phases. A/B mean sizes
18,872/16,127 bytes include labels but show DIFFERENT moving poses; use the
matched replay above for savings claims, not this live ratio. Only verified
test workers were stopped when switching :2323; VideoInterleaving checkout,
services and deployed reference were untouched.

Validation: 59 focused ASCII tests pass, including six new temporal tests for
current-reference/no-drift bounds, zero-error exactness, real row-tint
candidates, adapter screen reconstruction, fallback/phase/resize/reset and API
validation. Native ASan/UBSan and output guards pass 20 dimension combinations
(including 160x60) at four tolerances. Existing native sharing tests still pass
after rebuilding the helper; default colour-sharing routines were not altered.
Evidence: `tmp/ascii_temporal_run_trials/report.json`, saved reference/chosen
samples, `tmp/ascii_heavy_temporal_private_report.json`,
`tmp/ascii_heavy_temporal_public.log`, `tmp/ascii_temporal_sanitize.c`.

### Exact adaptive RGB aliases: offline rejection

After the user authorized further work, tested terminal OSC-4 palette slots as
dynamic aliases for EXACT accepted-heavy RGB values. This is not quantization:
unaliased colours still use full RGB, and every visible glyph and RGB value
must match the current heavy reference exactly. In principle a palette update
can recolour many existing indexed cells without repainting their glyphs.

`tmp/ascii_alias_trials.py` replays all 452 captured consecutive live heavy
pictures, trying 16/32/64/128/240 programmable slots and assignment-overlap
weights 0/4/16 (15 configurations, 6780 reconstructed screens). An independent
emulator retains indexed cells, processes batched OSC updates, then resolves
their current RGB values. Every screen in every configuration matches its
current input exactly. Alias assignment uses actual current RGB representatives
and matching of old slot locations; there is no fixed palette or premixing.

Best result was 17,291.5 bytes/frame, 2.075 Mbps at 15 FPS (64 slots, overlap
weight zero; larger slot counts gave the same result). Compared with 2.114
Mbps exact delta this is only 1.85% better, still worse than the previously
measured bounded-error run-coherent candidate at 1.959 Mbps. Slot churn and
OSC programming overhead offset retained-cell savings. Diagnostic total CPU
including matching and independent screen emulation was 8.2–15.4 ms/frame,
not a live application benchmark. No claim of global alias-optimizer optimality.

Rejected for this task: too little bandwidth benefit and extra capability
requirements. OSC-4 parsing, live recolouring, colour precision and palette
restoration have NOT been verified in the user's terminal; modifying palette
slots can affect other terminal content. This script NEVER opens a terminal
or socket. No alias implementation was added to normal application code or
enabled in the public test. No palette reset/query traffic was sent to users.
The public accepted-heavy versus extra-bound-1 temporal AB remains unchanged.
0.5 Mbps remains unmet; neither lower FPS nor stronger colour loss was silently
substituted. Evidence: `tmp/ascii_alias_trials/report.json` and trial script.

### User-requested live size cycle

User asked to loop through proposed smaller sizes. Public :2323 now runs
`tmp/ascii_size_loop.py`: 60x40 -> 45x30 -> 40x27 -> 30x20, five seconds each,
repeating with a continuously advancing first-face/blank-float source clock.
It uses the SAME accepted heavy/'420' colours at every size, without the extra
temporal colour approximation, to isolate the visual size tradeoff. GL requests
and samples each source grid directly; there is no bake or enlargement of a
smaller picture. The 40x27 scale is added only in this temporary launcher.
Output is centred in a cleared black 60x40 picture area plus one label row;
clear occurs inside the synchronized paint batch so joins/switches cannot
leave stale borders. Application defaults and ASCII-web are unchanged.

Replaced only the verified previous temporary AB worker. External receipt
verified all four labelled geometries and exact row widths over 22 seconds.
Observed mean bytes/update were 19074/11117/9480/6438, respectively, equivalent
to 2.289/1.334/1.138/.773 Mbps at 15 FPS. These intervals have different moving
poses and are not matched content. The earlier .49-Mbps area estimate for
30x20 was NOT verified: smaller grids did not retain proportional run-sharing
efficiency, and this size-only loop does not use the extra temporal encoder
from that estimate's 60x40 starting point. Do not claim 0.5 Mbps from resizing
on this evidence. This loop is for user visual judgement, not an approved
smaller default. Evidence: `tmp/ascii_size_loop_verified.json`,
`tmp/ascii_size_loop.log`. Replaces the previous two-phase temporal AB test.

User then suggested approximately 50x32. Narrowed the temporary live size loop
to 60x40 versus 50x32, five seconds each, with the same accepted heavy colours,
direct GL sampling and centred labelled 60x41 canvas. No other settings changed.
External verification received 180 frames across both phases and validated all
row counts/widths. This short, unmatched-pose sample averaged 19,134/14,002
bytes per update (2.296/1.680 Mbps normalized to 15 FPS); it is not a matched
bandwidth comparison or visual acceptance of 50x32. Current log/evidence:
`tmp/ascii_size_50x32_loop.log`, `tmp/ascii_size_50x32_verified.json`.

User requested a recognisability side quest: decrease one pixel per ten frames
and cycle. Public test :2323 now runs `tmp/ascii_resolution_sweep.py`, starting
50x32, decreasing width one column per TEN distinct published pictures, with
height round(width*32/50), minimum one, then wrapping 1x1 -> 50x32. Height is
proportional rather than decreasing both dimensions independently (avoids
progressive aspect distortion). Dimensions are labelled; colours remain the
accepted heavy/'420' profile. All 50 grids are direct SBS samples, launcher-only
scales; producer demand includes ONLY the current size, not every recently
visited size. Animation advances continuously; there is no seek/reset/bake.
Replaced only the verified prior size-loop worker; normal defaults and ASCII-web
remain unchanged. External verification reconstructed 511 received frames,
checked every width/height and label, all 50 sizes, exactly 10 frames in every
complete size interval, and the smallest-to-largest wrap. Evidence:
`tmp/ascii_resolution_sweep_verified.json`, `tmp/ascii_resolution_sweep.log`.

User then requested decrementing one ROW instead. Updated the temporary sweep
to height 32..1, width round(height*50/32): 50x32 -> 48x31 -> ... -> 2x1, then
repeat. Still exactly ten distinct published pictures per size; same heavy
colours and proportional geometry. Replaced only the verified column-sweep
worker. External verification checked 331 frames, all 32 heights, exact row
widths, ten frames per complete interval and 2x1 -> 50x32 wrap. Current evidence:
`tmp/ascii_row_sweep_verified.json`, `tmp/ascii_row_sweep.log`.

User identified 14x9 as their recognisability minimum and requested a higher
starting resolution. Current temporary sweep now starts 90x60, decrements one
row per ten published pictures with width round(height*90/60), and stops at
14x9 (height 9) before wrapping back to 90x60. Output canvas is 90x61 including
the label; viewers need enough terminal space to see the full maximum picture.
Same accepted heavy colours/direct SBS sampling; no new application defaults.
External verification checked 531 received frames, all 52 sizes, the 14x9
floor, exactly ten pictures per complete interval and the wrap. User's minimum
is a recognisability judgement, not approval to ship a 14x9 default or a claim
that its quality matches the accepted heavy 60x40 reference. Current evidence:
`tmp/ascii_row_sweep_90_to_14_verified.json`, `tmp/ascii_row_sweep_90_to_14.log`.

User requested measured rates for EVERY sweep size. Read the actual public
stream for 104.17 seconds, accumulating three complete size intervals for each
of the 52 resolutions. Measured FPS per size ranged 14.79–15.18. Counts include
the 90-column label, synchronized framing, clear, row positioning, SGR and
glyph bytes; exclude TCP/IP headers and initial/final connection controls.
Intervals run from a size's first received complete picture to the next size's
first complete picture. Initial partial interval is discarded. A first
attempt incorrectly required exactly ten received pictures per interval and
timed out without producing a final report; a shared advancing sweep may
deliver fewer pictures to a particular client. The corrected measurement
counts actual received bytes/frames and elapsed time over every complete
observed size interval; it does not assume ten received frames or 15 FPS.

Selected actual Mbps (15-FPS-normalized Mbps in parentheses): 90x60 4.065
(4.024); 60x40 2.380 (2.381); 48x32 1.621 (1.615); 30x20 .763 (.759);
26x17 .593 (.595); 24x16 .531 (.525); 22x15 .483 (.489); 14x9 .233 (.236).
Thus 22x15 is the largest sampled size below 0.5 Mbps in THIS workload, while
24x16 is slightly above it. This does not establish a hard cap for other
pictures. Each size contains about two seconds of combined observations over
three moving intervals, not 30 seconds continuously at each size. Different
resolutions see different poses. No 50x32 sample exists in this 90:60 sweep:
its height-32 entry is 48x32, not the user's previous 50x32 candidate.

All 52 actual and normalized rates, frame/byte counts and interval timings are
preserved in `tmp/ascii_sweep_rates/{report.json,rates.csv,rates.md}`. Measurement
script: `tmp/ascii_measure_sweep_rates.py`. Public sweep/settings unchanged.

### Explicit new task: browser resolution zoom

User now explicitly authorized changing `/ascii` (ASCII-web), previously kept
separate/unchanged during plain-nc experiments. Implemented browser-local
resolution zoom, configured default view (currently 60x40), zoom-out maximum
90x60 and zoom-in minimum 15x10, with a fixed fitted picture extent. Wheel,
pinch, +/- controls and Reset/0 operate locally; no per-viewer server resize
requests. ASCII-web generates one 90x60 master per source publication, not one
render per client/resolution. Client selects glyph/colour pairs from that
shared lattice; it does NOT reproduce original-source direct sampling at
every target grid, and dimension-default preservation is not exact old-cell
parity. Normal grading is unchanged; no temporary heavy/temporal profile was
silently made the ASCII-web default.

Added optional negotiated gzip, encoded once per publication and shared by
all capable viewers, plain-text fallback otherwise. Browser queues are latest
only for both xterm writes and decompression. Zoom changes browser drawing
work, NOT bytes per viewer: every tab still receives the same master stream.
One actual 30-second two-client run observed 9.33 Mbps plain versus 2.79 Mbps
gzip at ~14.9 FPS, with 445 identical decoded frame hashes. This is normal
ASCII-web grading, NOT the heavy-nc workload or evidence for its .5-Mbps goal.

The initial external check found site's generic /static/ upstream returning
502. Namespaced all viewer assets under /ascii/static/, with handler aliases
for direct serving and the existing proxy's /asciistatic/ concatenation. Font
URL is relative. External page, all six assets (including font), source-size
metadata and gzip WSS were then verified at testing3.gencarelle.fun. No nginx,
VideoInterleaving checkout, reference endpoint or physical devices changed.
The verified temporary asciiweb process serves HTTP :1980 and WS :2424 as
assigned by server_config; the existing nc row sweep on :2323 was left running.

73 focused Python/Node tests pass: all 51 browser row levels, source-cell
selection, RGB/indexed parsing, wheel/pinch/keys/reset, reconnect, one in-flight
and replaceable latest write/decompression, shared compression count/identity,
metadata, no per-viewer resizing, one master GL target and traversal-safe
asset aliases. An actual server picture was also decoded/resampled through
all 51 client grids. Real browser paint/visual inspection remains unverified;
no browser executable/driver was installed. See `docs/ASCII_WEB_ZOOM.md` and
`tmp/ascii_web_zoom/` for contract, limitations and protocol/network evidence.
Current temporary web log: `tmp/ascii_resolution_web_ready.log`.

### Browser CPU/RAM follow-up (not server resources)

User explicitly requested browser CPU/RAM optimisation after the separate
server CPU measurement. Captured 120 consecutive actual master frames and a
snapshot of the current frontend. Installed isolated Playwright/Chromium under
gitignored tmp/ for real browser testing, without changing application Python
requirements or OS packages. Replaced the visible page's xterm parse/buffer/
row-rebuild cycle with a read-only persistent text-run grid, preserving xterm
font metrics and proportional glyph spacing. Added single-pass ANSI parsing
with two reusable frame buffers; removed browser ANSI re-encoding and xterm
bundle/styles from the page; refit only on layout changes. Hidden tabs stop
work/close their socket and reconnect to current content when visible.

Matched moving replay in software-rendered Chromium (12 seconds/case): default
60x40 whole-browser CPU 117.2% -> 98.7% of one core, summed process PSS
404.6 -> 245.9 MiB, post-GC JS heap 2.63 -> 1.57 MiB. Tuned 60x40 CPU
120.0% -> 102.5%, PSS 429.7 -> 263.9 MiB. Thus ~15% CPU and ~39% PSS reduction
for tested 60x40 cases. These are BROWSER, not server metrics or user-device
measurements. Maximum 90x60 remains CPU-bound at ~133%; memory fell ~37% and
completed render count increased 104 -> 129. Do NOT claim reduced maximum-mode
CPU or a universal solved resource limit. No FPS, resolution or quality cuts.

24 screenshot pairs (three actual pictures, DPR 1/2/1.5, eight control/size
conditions each) had ZERO changed pixels. Old/new parsers match all 120 moving
inputs. 74 focused Python/Node tests pass, including renderer node reuse,
unchanged-picture no-op, geometry/DPR, buffer reuse, hidden-tab reconnect and
all picture controls. Actual external Chromium also loaded the live public
page without xterm, received animation, exercised brightness/stroke/spacing/
background, hide/restore and Reset, with no page errors. Both application
listeners and nc sweep remain running; services/reference endpoint untouched.
Details: `docs/ASCII_WEB_ZOOM.md`, evidence `tmp/ascii_browser_perf/`, including
`before/report.json`, `after_final/report.json`, `parity*/report.json` and
`public_verified.json`. Local tuning values remain unadopted as defaults.

### Correct original-browser baseline comparison

User asked to compare against the ORIGINAL viewer, not the initial zoom
implementation. Read-only `~/VideoInterleaving` at 2df1a6723 provides original
template, Unscii font, xterm and worker; relevant files match HEAD and remain
unchanged. Original service uses images_ascii. AST-extracted original baked
worker produces 120 actual moving 60x40 indexed frames from 90x60 first-face
bakes, no float overlay: 2400 colour instructions and mean 28,496.6 bytes.

Two balanced 15-second Chromium observations/case, fresh browser per run,
reverse order second pass, same source cadence/viewport. Strict matched pairs
use identical frame strings, font, grid and physical picture area (including
test-only Unscii/aspect overrides in CURRENT served copies, never production).
Both finished matched screenshot pairs are pixel-identical. Separate current-
defaults case includes actual 90x60 gzip master, 60x40 view and Schoolbook.

Original default 60x40: 43.6% one-core browser CPU, 204.6 MiB PSS, 15.0 updates/s.
Current matched 60x40: 39.5%, 207.0 MiB, 15.0 updates/s: ~9% less CPU, essentially
NO whole-browser RAM saving versus original. Retained JS heap 3.09 -> 2.09 MiB.
Actual current web DEFAULTS: 95.5%, 255.5 MiB, ~15 updates/s. Thus actual current
default feature/appearance bundle costs ~2.19x CPU and ~25% more PSS than the
original default viewer in this sandbox. Different inputs/font/features: do
not call that a renderer-only comparison or attribute it to just the font.
These are default settings, NOT the user's dialled private-tab values.

Matched 90x60 RGB diagnostic: original/current CPU both ~132% (saturated), PSS
means 632/370 MiB, updates ~10.2/11.6 per second. Original PSS varies strongly
between passes; no steady-state memory guarantee or lower-CPU claim. The earlier
15% CPU/39% PSS savings against the INITIAL ZOOM VIEWER are still that separate
comparison and must never be substituted for the original-browser baseline.

Full methodology/results/caveats: `docs/ASCII_BROWSER_BASELINE.md`. Evidence:
`tmp/ascii_og_browser/{preparation,browser_report,summary,comparison,matched_pixel_check}.json`,
screenshots and original asset snapshot. Scripts `tmp/ascii_og_prepare.py`,
`tmp/ascii_og_browser_bench.py`. No production settings/code/services/routing
changed in this comparison; public browser and nc sweep remained available.

### Current-browser cause isolation

User asked what causes the remaining browser heaviness. Isolated one-factor
diagnostics use the SAME 120 actual current-master pictures, 60x40 view,
viewport, source cadence and neutral default appearance, two reverse-order
12-second observations per fresh browser. Production frontend untouched.

Own-cohort default: 106.1% one-core browser CPU, 262.9 MiB PSS, ~2261 text runs.
Only Unscii: 96.6%, ~2254 runs (~9% lower CPU). Only preselected 60x40 source:
101.6%, ~2260 runs (~4% lower); visible glyph/RGB values match all 120 originals
and initial screenshot is pixel-identical. Only controls hidden: 102.4% mean,
but pass direction is inconsistent, so no reliable control-hiding benefit.
Fixed tint via actual contrast=0 control, keeping glyphs/font/source: 61.2%,
220.1 MiB PSS and ~1037 runs (~42% lower CPU). This DESTROYS colour variation
and is a diagnostic, not an accepted quality reduction or deployment change.

Evidence supports fragmented, changing colour/style runs and their text-layout/
paint costs as the largest tested factor. Font is a smaller contributor; master
decode/source size and controls do not explain most of the baseline gap. Do not
sum ablation differences or compare their 106% reference directly to the other
cohort's 95% figure. Default CPU ranges 98–114% across passes; flat tint is
consistently cheaper at 62/60%. Next target: cheaper coloured-glyph drawing and
batching while preserving RGB, not silently lowering quality. Details:
`docs/ASCII_BROWSER_CAUSES.md`; `tmp/ascii_browser_causes/` contains reports,
exact source checks, screenshots and production-file hashes.
