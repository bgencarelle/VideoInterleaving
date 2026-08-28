# Scope Mode

Draws the interleaved composition on an oscilloscope in XY mode, using the
sound card as a two-channel DAC. Left channel drives X, right drives Y, and
the beam traces whatever path those two voltages walk.

It runs off the same clock and the same folder selector as every other output
mode, so it stays in sync with MIDI/MTC without any new sync code. It never
decodes a JPEG, opens a GL context, or touches the image loader.

---

## 1. The two-stage pipeline

**Bake (offline, once).** Images become a compact geometry library. All the
expensive work — vectorizing, simplifying, path ordering — happens here, where
you can afford to spend a second per frame.

**Playback (runtime).** Read the index from the clock, look up two baked
frames, composite them, and push samples to the audio device. Vector playback
mostly slices arrays; raster and stochastic modes additionally construct their
XY route from the baked thumbnail. On the reference Python path, a 1600-sample
stochastic trace benchmarks around 13 ms on the development machine, inside a
33 ms/30 Hz trace period; the runtime prints its own measured cost at startup.

The bake remains one-time and resolution-independent; changing renderer,
sample rate, or trace rate does not require rebaking.

---

## 2. Files

| File | Role |
|---|---|
| `scope_out.py` | Audio engine. Owns the device, sample-rate negotiation, and the buffer the DAC reads. Converts geometry into samples. |
| `scope_bake.py` | Shared library. Format definition, geometry helpers, and the two composite functions. Imported by *both* the baker and the runtime; imports neither. |
| `utilities/convert_to_xy.py` | The baker. Offline only. Images → library. |
| `scope_display.py` | The runtime. Clock → index → composite → audio. |
| `test_scope_pair.py` | Inspection tool. Preview one pair on screen and on the scope simultaneously. |

### Repo changes

Scope is a **standalone mode**, like ascii / asciiweb / local / web: selected
once, and it owns the run. It does not attach to another mode, and no existing
mode's code path is altered.

| Repo file | Change |
|---|---|
| `main.py` | `--mode scope` and its `--scope-*` options; dispatches to `scope_display` |
| `settings.py` | `XY_DIR` discovery and the `SCOPE_*` defaults |
| `server_config.py` | `MODE_SCOPE` (monitor port only, no servers) |
| `requirements.txt` / `system-requirements.txt` | `sounddevice`, `libportaudio2` |
| `.gitignore` | baked libraries (`*_xy/`, `*.npy`) |

`image_display.py`, `renderer.py`, `image_loader.py`, `folder_selector.py`,
`index_calculator.py` and `make_file_lists.py` are **untouched**.

### Running it

```
python main.py --mode scope --dir images --xy-dir images_xy --scope-raster
```

For the no-Z Osci-style renderer:

```
python main.py --mode scope --xy-dir images_xy --scope-mode stochastic
```

or directly, using the settings.py defaults:

```
python scope_display.py --dir images --xy-dir images_xy
```

Configuration flows CLI → `settings.SCOPE_*` → `scope_display`, the same way
ASCII mode flows through `settings.ASCII_MODE`. An installation can therefore
be configured once in `settings.py` and launched with a bare `--mode scope`.

Scope options passed to another mode are reported as ignored rather than
silently dropped.

### Why scope cannot run alongside the video

`folder_selector.update_folder_selection()` is stateful **and** stochastic —
its RNG is seeded from `time.time()`. Two independent callers advance two
independent sequences and pick *different folders*, so two processes would show
different content within seconds of the first switch, not merely drift.

Scope mode is therefore the sole caller in its process, exactly as
`image_display` is in the others.

---

## 3. What a bake produces

The baker mirrors your source tree: `images/face/02_foo/` becomes
`images_xy/face/02_foo/`. That mirroring is load-bearing — the runtime finds a
folder's library by its path relative to the images root, so folder 3 on the
scope is folder 3 on screen by construction.

Each library directory contains:

| File | Contents |
|---|---|
| `verts.npy` | Every vertex, `(V,2) int16`, normalized to ±1. One flat array. |
| `poly_starts.npy` | `(P+1,) int32`. Polyline *p* occupies `verts[poly_starts[p] : poly_starts[p+1]]`. |
| `frame_starts.npy` | `(F+1,) int32`. Frame *i* owns polylines `frame_starts[i]` to `frame_starts[i+1]`. |
| `flags.npy` | `(P,) uint8`. One per polyline — see below. |
| `thumbs.npy` | `(F, h, 96, 2) uint8`. A 96px-wide thumbnail per frame: `[luminance, alpha]`. |
| `names.json` | Source filenames, for provenance. |

This is a CSR layout — flat arrays plus offset tables, no per-frame Python
objects. It memory-maps, so a large library costs disk rather than RAM.

**int16, not float32**, because you're feeding a 16-bit DAC into a scope with
maybe 9 bits of usable resolution. Float would be pure waste.

**Geometry, not samples.** The library stores *shapes*, not the audio buffer.
That's why the same bake works at 44.1kHz on your Mac and 48kHz on a Pi, and
why changing frame rate doesn't require rebaking.

### The flags

| Flag | Meaning | Drawn? | Occludes? |
|---|---|---|---|
| 0 | Interior stroke (a luminance band boundary) | yes | no |
| 1 | Silhouette loop (from the alpha matte) | yes | yes |
| 2 | Matte-only loop | **no** | yes |

Flag 2 exists for full-bleed layers. When a matte's outline hugs the canvas
edge, it's just the picture frame — drawing it wastes vertices on a rectangle.
But it can't simply be deleted, because inverse mattes (an opaque background
with a face-shaped hole) need that outer boundary for the even-odd test to
come out right. So it participates in occlusion and is never drawn.

---

## 4. Three rendering modes

They are genuinely different pictures, and they use different parts of the
library.

### Vector — line art

Uses `verts` / `poly_starts` / `frame_starts` / `flags`.

The beam traces outlines continuously. Sharp, bright, abstract — the
"oscilloscope music" look. Brightness is even because samples are distributed
by arc length.

How the bake builds it:

1. **Silhouette** — threshold the alpha channel, trace the region boundaries.
   These become flag 1 (or 2 if they hug the canvas).
2. **Interior** — posterize luminance into `bands` levels and trace each
   level's regions. These become flag 0. *Not Canny:* edge detection on a
   photograph yields dozens of disconnected fragments, and no vertex budget
   can render those as anything but straight chords.
3. **Allocate** — rank contours by area (silhouettes weighted up by
   `sil_boost`), then give each contour its own vertex count between `min_v`
   and `max_v`. Contours that don't fit the budget are dropped entirely.
   A single global tolerance would starve small contours into 2-point chords,
   and small contours — eyes, mouth — are what make a face recognizable.
   Dropping something cleanly beats drawing everything badly.
4. **Order** — greedy nearest-neighbour tour. Closed loops get rotated to
   start at the point nearest the beam, which shortens travel more than a
   full 2-opt pass would.
5. **Subdivide** — split long segments so no segment exceeds `max_seg`. This
   bounds the error of the runtime occlusion test.

At runtime, `merge()` composites: the float's matte loops cull main-layer
segments whose midpoints fall inside them, then the float is drawn on top.
Fragments shorter than `min_feature` are discarded as clutter.

Then `rasterize()` walks the path, spending samples in proportion to segment
length so brightness is even, and moving ~8× faster across jumps between
shapes (`JUMP_GAIN`) so travel lines stay dim.

### Raster — scanlines

Uses `thumbs.npy` **only**.

The beam sweeps a serpentine path over a grid and **lingers on bright cells**.
Brightness comes from dwell time, which is how you get a grayscale image out
of two channels with no Z/intensity input. This is the two-channel equivalent
of a video-to-scope adapter.

Layers composite in the thumbnail domain with real alpha —
`float*float_alpha + main*main_alpha*(1-float_alpha)`, exactly the formula in
`renderer.py`'s fragment shader. No occlusion geometry needed, and inverse
mattes work with no special casing.

The grid **auto-sizes to the sample budget**, because raster needs roughly one
sample per cell. 882 samples ≈ 33×25; 1470 ≈ 37×50.

### Stochastic — Osci-style bitmap walk, no Z axis

Uses `thumbs.npy` **only**. It has no horizontal rows and does not use a third
brightness channel. Luminance becomes probability: bright pixels are selected
more often, nearby unvisited pixels are visited first, and the phosphor turns
that visit density into visible tone.

The renderer constructs a continuous route in four stages:

1. Convert composited luminance to a visit-probability map using `trim` and
   `gamma`.
2. Add a modest gradient term so silhouettes, eyes, and mouth edges survive.
3. Walk to the nearest unvisited accepted pixel, periodically reseeding into a
   different bright region so the route covers the whole subject.
4. Resample local target-to-target moves with equal beam time, make reseed jumps
   deliberately faster/dimmer, and begin the next trace at the previous trace's
   endpoint.

This is based on Osci-render's bitmap strategy rather than its SVG renderer.
There is still no invisible travel without Z: the nearest-neighbour walk and
endpoint chaining make travel short, while `--scope-walk-reseed-ms` controls
the unavoidable longer relocations.

At 48 kHz, start with `--scope-fields 1 --scope-fps 15` (3200 samples) when the
tube can tolerate 15 Hz, or 30 Hz (1600 samples) when flicker matters more.

### Scanlines and the bake ceiling

The runtime area-averages the baked thumbnail down to whatever grid the sample
budget supports — a proper summed-area resample, not line-skipping, which would
alias badly on scanline content. So the thumbnail is a resolution-independent
store like the vectors, and grid size is a runtime decision.

But its size is a **hard ceiling on scanline count**, and exceeding it is
reported rather than silently clamped.

| `--thumb-width` | max scanlines | per folder (2220 frames) | 32 folders |
|---|---|---|---|
| 48 | 64 | 14 MB | 0.44 GB |
| 64 | 85 | 24 MB | 0.77 GB |
| **96** (default) | **128** | 55 MB | 1.75 GB |
| 128 | 171 | 97 MB | 3.11 GB |
| 256 | 341 | 388 MB | 12.4 GB |

There is a natural ceiling worth knowing: `cells ≈ samples / density`, so no
realistic budget resolves past about 138×184 (6400 samples at density 0.25).
Baking wider than ~128 stores data nothing can read.

Three ways to add scanlines, trading against different things:

| Setting | Grid at 1600 samples | Cost |
|---|---|---|
| default | 34×45 | — |
| `--scope-rows 60` | 26×60 | columns |
| `--scope-density 0.5` | 48×64 | brightness (half a sample per cell) |
| `--scope-samples 3200` | 48×64 | index rate (every other frame shown) |

`--scope-rows` is a pure trade — rows × cols is fixed by the budget, so at 100
rows you are down to 16 columns. `--scope-density` grows both dimensions but
thins the trace. `--scope-samples` grows both at full density, paid for in
skipped indices.

### Spending the budget

Two controls change how the fixed budget is spent:

- **`--density`** — samples per cell. 1.0 (default) is the finest grid. Above
  1 is a *trade*: fewer, longer, brighter scanlines carrying less detail. Raise
  it only if thin traces read too dim on your tube.
- **`--trim`** (default 0.02) — cells below this brightness are dropped from
  the sweep, so black margins cost nothing and their samples go to the subject.
  It also controls stray lines: with no Z channel the beam is always on, so
  wherever it crosses a dark area it leaves a faint trail. Raising trim to
  0.08–0.16 cuts that noticeably. `--no-trim` disables it entirely.

### Retrace

The sweep is a **closed loop** by default: it runs down the rows and back up
over the same path, so there is no jump from the last row to the first. Without
this the beam has to fly back across the picture every frame, and with no Z
channel to blank it you see a bright diagonal. The return pass redraws the same
cells, so brightness is unchanged — it costs nothing. `--retrace` restores the
old behaviour if you want the CRT-style flyback as an effect.

Faint diagonals that remain are the beam crossing dark regions between bright
ones. They are inherent without blanking, and they are genuinely dim on
hardware — brightness is dwell time, and the beam is moving fast there.

Neither changes how long a trace takes. Trace duration is `1/fps`, always,
regardless of image complexity. What they change is how thinly the budget is
spread over the picture.

Three controls decide how that fixed budget is spent:

**`--density`** — samples per cell. At 1.0 you get the most cells, each with
about one sample, which reads as dots. At 2–3 you get fewer, larger cells drawn
as solid bright lines. Lower resolution, better-formed image. This is the
knob for "make it draw properly rather than finely".

**`--fields`** — interlacing. Field *f* draws rows *f, f+fields, f+2·fields…*
With half the rows per pass, each row gets twice the samples, so **horizontal
resolution doubles at no extra cost**, and phosphor persistence merges
successive fields back into a whole image. The cost is temporal: consecutive
fields come from consecutive indices, so fast motion combs — exactly like
interlaced video. At 48kHz/30ips, 2 fields takes ~34 columns to ~49.

**`--even-rows`** — constant beam time per row instead of brightness-weighted.
Without it, bright content in the upper frame eats the global budget and the
lower rows starve — the beam "runs out of time" toward the bottom and renders
it as sparse dots. With it, every row gets equal time. The cost is that dark
rows also consume their share, appearing as faint scanlines, since without a Z
channel there's no way to blank them.

#### Interlacing — the resolution lever

At 48kHz and 30fps you get 1600 samples for ~1530 cells: about one sample per
cell. Resolution is grid-limited, and the grid is sample-limited. Skipping
dark cells does not help — measured on real content, black cells already
consume only **3.3%** of the budget, so scanning content spans alone would buy
just 1.12× per axis.

Since samples per second is fixed by the hardware, the only remaining lever is
**time**. With `fields = N`, one trace draws only rows where
`row % N == field`. The full grid can then hold N times as many cells for the
same per-trace cost, so **both axes gain √N**, and the phosphor integrates
successive fields into one picture. Analog TV did exactly this, for exactly
this reason.

| fields | rows in full grid | full-image refresh at 30fps |
|---|---|---|
| 1 | 54 | 30 Hz |
| 2 | 82 | 15 Hz |
| 3 | 102 | 10 Hz |
| 4 | 120 | 7.5 Hz |

Default is 4. The cost is that a full image takes N traces, so fast motion can
comb, and a low full-refresh rate can visibly crawl on a scope with short
persistence. Lower it to 2 if either bothers you.

Because interlacing needs a fresh field on *every* trace, the runtime pushes
one whenever a trace completes — not only when the index changes. Otherwise a
held frame would show one field forever and stay permanently combed.

---

## 4b. Mix mode

`--mix [HZ]` (default 120) alternates raster and vector on successive traces.
Above flicker fusion the phosphor sums them, so you see one image: raster
supplies tone and mass, vector supplies the outline and features.

The switch rate **is** the trace rate, so each pass gets `rate / HZ` samples —
400 at 48kHz/120Hz, 800 at 60Hz. Lower HZ gives each mode more detail but
starts to read as two alternating pictures rather than one combined one.
60–120 Hz is the useful range.

Both modes need their data, so this requires a **full bake**, not
`--thumbs-only`.

The beam has to jump between the end of the raster sweep and the start of the
vector path, so expect a faint connector. It is dim (fast traversal) and gets
less noticeable as HZ rises.

---

## 5. `--thumbs-only`

A bake flag. It writes `thumbs.npy` and skips vectorizing entirely.

- **Much faster** — seconds instead of minutes, since vectorization is all the
  work.
- **Raster and stochastic modes work.**
- **Vector mode does not** — there's no geometry, so it falls back to the idle
  circle.

Use it when you've settled on raster/stochastic and are iterating on content.
Do a full bake when you want live cycling through all three modes.

---

## 6. The sample budget

Everything downstream depends on one number: **the path length per trace**,
N. The DAC consumes samples at a fixed rate and the beam position *is* those
samples, so

```
trace duration = N / sample_rate
```

That is the only coupling between detail and refresh, and it is unavoidable.
Note what is *not* a parameter here: frame rate. It falls out as
`sample_rate / N`. `--fps` is a convenience that sets `N = rate/fps`; `--samples`
sets N directly and is the honest knob. Frames may even vary in length between
traces — the audio callback handles it.

```
samples per trace = sample_rate / fps        # the same equation, rearranged
```

44.1kHz ÷ 30fps = 1470. That's the entire resolution budget for one image —
every line, every scanline cell.

A related invariant that's easy to get wrong:

```
beam time per index = sample_rate / ips
```

(Interlacing does not break this. It spends the same beam time, but spreads one
image over several traces so the grid can be larger.)

This is **fixed**. Lowering `fps` below `ips` doesn't buy more samples per
index; it buys *fewer indices*, because each trace takes longer and the ones
that arrive meanwhile are dropped. At `--fps 1` you show one index per second
and skip 97% of the animation.

So `fps = ips` is the sweet spot: every index is shown, each gets the maximum
possible samples in one continuous trace. That's the runtime default. A piece
running at 15 or 24 ips therefore gets *more* detail per frame automatically
(3200 or 2000 samples at 48kHz), with no settings to change.

Setting `fps` to a multiple of `ips` draws each index more than once — steadier
on a digital scope, but the samples divide accordingly. Total light per index
is identical either way; you're only choosing how it's split.

### Real-time streaming (`--realtime`)

By default the audio buffer holds one complete trace and only swaps at a
boundary, so a new index waits up to a full trace (33 ms at 30 fps).

`--realtime` replaces that with a continuous generator: samples are produced
row by row, and the current index is re-read at every row boundary. A new index
lands within about a millisecond, mid-trace, and the beam simply carries on
from where it is with the new content.

The splice point is a **row**, not a sample, and that is deliberate. Sample #k
of two different indices sit at different screen positions, because dwell
follows brightness — splicing there would teleport the beam. Row 30 is at the
same y in every frame, so resuming there is continuous.

Raster only: vector paths have no positional correspondence between frames, so
they still switch at pass boundaries, as does the raster/vector mode itself,
which always finishes the pass it is in.

Costs roughly 0.5 ms of work per 256-sample block inside the audio callback
(about 9% of the budget at 48 kHz on a laptop). Watch for underruns on slower
machines.

### Frame-boundary swapping

The audio callback only swaps in a new image **when the current trace
finishes**. Replacing the buffer mid-trace would make the beam jump from its
position in one image to the same offset in a different one — a bright tear on
every index change.

If a new index arrives while one is already queued, the older one is
**dropped**, not queued. The scope shows the index that's current *now*; it
never falls behind replaying stale ones. Drop, never lag.

---

## 7. Command reference

### Folder naming and discovery

The baker applies the same gate as `make_file_lists`: main folders must start
`0_`–`254_`, float folders `255_`. Anything else (`999_backup`, non-numeric
prefixes) is skipped, because the list builder ignores it and baking it is
wasted time and disk. `--all-folders` overrides this.

`settings.XY_DIR` is discovered from `IMAGES_DIR` by the `<source>_xy`
convention, so a tree called `150_91` resolves to `150_91_xy` with no
`--xy-dir` needed.

### Audio device

Scope mode takes the output device. If the installation also plays sound, the
two cannot share a stereo device — plan on a second interface, or a
multichannel one carrying both.

### Baking

```
python utilities/convert_to_xy.py -i images -o images_xy [options]
```

| Flag | Effect |
|---|---|
| `-i` / `-o` | Source tree / output tree (default `{input}_xy`) |
| `--profile tiny\|std` | `tiny` = 150 vertices/frame, for small scopes. `std` = 400. |
| `--budget N` | Override main-layer vertex budget |
| `--bands N` | Luminance levels for interior detail. 2 = starker, 4 = more tonal. |
| `--min-verts N` | Floor on vertices per contour. Raise for fewer/cleaner shapes. |
| `--thumbs-only` | Raster/stochastic luminance data only; skip vectorizing. |

Floats always bake silhouette-only — they're mattes, with no interior worth
tracing.

### Playback

```
python main.py --mode scope --xy-dir images_xy [options]
```

| Flag | Effect |
|---|---|
| `--scope-mode vector\|raster\|stochastic` | Select the renderer. |
| `--scope-raster` | Compatibility alias for `--scope-mode raster`. |
| `--scope-stochastic` | Alias for `--scope-mode stochastic`. |
| `--scope-walk-radius PX` | Stochastic nearest-neighbour search radius (default 10). |
| `--scope-walk-stride PX` | Stochastic source-pixel step (default 1). |
| `--scope-walk-reseed-ms MS` | Time between weighted region changes (default 5 ms). |
| `--scope-walk-edge F` | Edge-importance gain (default 0.35). |
| `--scope-fps N` | Scope redraw rate. Defaults to `IPS`. Sets `N = rate/fps`. |
| `--scope-samples N` | Path length per trace directly. Overrides FPS. |
| `--scope-realtime` | Stream continuously; index changes land within a row (raster only). |
| `--scope-mix [HZ]` | Alternate raster/vector each trace (default 120 Hz). |
| `--scope-mix-duty F` | Fraction of mixed passes spent on raster (default 0.5). |
| `--scope-sweep MODE` | `alternate` (default), `palindrome`, or `retrace`. |
| `--scope-rows N` | Raster scanline count (default: auto from budget). |
| `--scope-gamma F` | Raster/stochastic luminance exponent (default 2.2). |
| `--scope-density F` | Raster samples per cell (1.0 = finest). |
| `--scope-trim F` | Ignore luminance below this level (default 0.02). |
| `--scope-fields N` | Raster interlacing (1 = progressive). |
| `--ask` / `--scope-ask` | Choose the audio output interactively. |
| `--device X` | Audio output by index or name fragment, e.g. `--device Scarlett` |

While scope mode owns a terminal, press `v` to cycle
`VECTOR → RASTER → STOCHASTIC → VECTOR`. Press `p` to print the current mode
and live settings as reusable command-line flags. Cycling is disabled in
`--scope-realtime` and `--scope-mix`, whose audio/scheduling paths are fixed at
startup; restart with `--scope-mode` to leave either one.

### Preview

```
python test_scope_pair.py --xy-dir images_xy [--bg N --fg N --index N] [options]
```

Adds `--play`, `--ips`, `--loop`, `--exposure`, `--size`, `--ask` and
`--device` on top of the playback flags.

### Choosing the audio output

By default the system default output is used. `--ask` lists every
stereo-capable output with its host API and native sample rate, and prompts —
but only when more than one exists, and only when there's a terminal to prompt
on, so it can live in a kiosk launch script without ever hanging. Blank input
or an out-of-range number falls back to the default.

The listed sample rate matters: it sets `samples per trace = rate / fps`, so a
96kHz interface gives more than twice the detail of a 44.1kHz built-in output.

Once you know which device you want, `--device` takes an index or a
case-insensitive name fragment (`--device Scarlett`) — better than an index for
scripts, since indices reshuffle when hardware is plugged or unplugged.

Keys: `space` play/pause · `←/→` step · `m` toggle occlusion cull · `r`
vector↔raster · `+/-` intensity · `[/]` gamma · `,/.` speed · `l`
pingpong↔loop · `q` quit.

`--exposure` stands in for the scope's intensity knob. Raster detail hides
inside a saturated image without it — try 0.3–0.4.

---

## 8. Things that look like bugs but aren't

- **The composite gets brighter where the float occludes the main.** Fixed
  sample budget over a shorter path. A real vector display does the same.
- **A fully transparent frame draws a circle.** Deliberate. Zero volts on both
  channels is a stationary dot at screen center, which burns phosphor on an
  analog CRT. The idle circle spreads that energy.
- **Folder switches land slightly late.** The selector is stochastic, so it
  can't be evaluated ahead of the audio latency the way the index can.
- **Folder 0 looks wrong.** `folder_selector` deliberately skips main folder 0
  — it's the background plate the artwork never shows. Test with folder 1+.
- **Inverse mattes look inverted.** `255_*_BG_Matte` folders are opaque
  background with a face-shaped hole. Float-on-top therefore masks the
  *surroundings* and reveals the face. That's correct compositing.
