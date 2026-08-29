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
mostly slices arrays; raster constructs a pass from the baked thumbnail;
stochastic updates a continuous walk's probability field. On the supplied
portrait, 1,600 stochastic output samples benchmark around 11 ms on the
development machine, inside a 33 ms/30 Hz buffer period; the runtime prints its
own measured cost at startup.

The bake remains one-time and resolution-independent; changing renderer,
sample rate, or trace rate does not require rebaking. Bakes made before the raw
stochastic luminance channel remain compatible, but one rebake after this
change removes raster-specific horizontal preconditioning from stochastic.

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

At startup, scope also applies the normal display's frame-count rule: it keeps
the largest frame count shared by both face and float folders, excludes baked
folders of other lengths, and preserves numeric-prefix order within each
layer. This keeps the two counts and every selector index identical across
scope, local, and web modes even when an old or partial bake is still present.

Each library directory contains:

| File | Contents |
|---|---|
| `verts.npy` | Every vertex, `(V,2) int16`, normalized to ±1. One flat array. |
| `poly_starts.npy` | `(P+1,) int32`. Polyline *p* occupies `verts[poly_starts[p] : poly_starts[p+1]]`. |
| `frame_starts.npy` | `(F+1,) int32`. Frame *i* owns polylines `frame_starts[i]` to `frame_starts[i+1]`. |
| `flags.npy` | `(P,) uint8`. One per polyline — see below. |
| `thumbs.npy` | `(F, h, 256, 3) uint8` by default. `[raster luminance, alpha, raw luminance]`. Legacy sizes and two-channel files still load. |
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

## 4. Four rendering modes

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

New bakes give stochastic the raw luminance channel. Raster keeps its separately
preconditioned channel because horizontal sharpening compensates a horizontal
scan; applying that same directional correction to a directionless walk was an
unintentional source of false edges. This is extra stored source data, not a Z
or intensity output channel.

The renderer is a continuous sample source, not a per-trace route:

1. Composite the current thumbnail into a probability field. Pixels at or below
   0.2 are excluded; every candidate gets a fresh `luminance ** gamma` test.
2. At a fixed target clock (48 kHz by default), spiral outward to the nearest
   accepted unvisited pixel using radius 10 and a source-scale-aware stride.
   The default resolves to 2 on a 256-pixel bake, equivalent to Osci's stride 4
   on an approximately 480-pixel source.
3. Clear the short visited history every ten targets and randomly relocate every
   5 ms, matching the current Osci-render implementation.
4. Continue the same state across audio buffers and image-index changes. A trace
   boundary does not rebuild, cap, or resample a waypoint list.

The target clock, image clock, and DAC clock are separate. `IPS` changes the
probability field. `SCOPE_WALK_HZ` advances the walk. The device sample rate
samples that path: at 96 kHz the default 48 kHz walk gets two DAC samples per
target interval instead of drawing twice as many targets per second. Raising
the DAC rate therefore improves representation of the same waveform; it is not
an image-resolution knob.

If the optional software low-pass is enabled, stochastic uses a causal filter
whose state crosses audio buffers. The circular per-trace filter remains for
repeating completed paths; applying it to a continuous walk would falsely join
the end of each buffer to its beginning.

This is based on Osci-render's bitmap strategy rather than its SVG renderer.
There is still no invisible travel without Z: nearest-neighbour motion keeps
most connectors local, while `--scope-walk-reseed-ms` controls the unavoidable
long relocations. Lowering `--scope-fps` merely makes larger audio buffers and
slower content handoff; it no longer reduces the number of target decisions per
second. Start at the normal 30 Hz and let phosphor persistence integrate the
continuous visit density.

### Fusion — corresponding-position multiplexing

Fusion generates equal-length position arrays from every requested renderer,
then selects between their corresponding entries round-robin. `vr` produces
`V[0], R[1], V[2], R[3]...`; `vrs` produces
`V[0], R[1], S[2], V[3]...`. It performs no arithmetic averaging of XY
coordinates and does not convert the components into a stochastic probability
field. Unlike mix, switching happens within the array rather than once per
whole trace.

```
--scope-mode fusion --scope-fusion vrs   # vector + raster + stochastic
--scope-mode fusion --scope-fusion vr    # vector + raster
--scope-mode fusion --scope-fusion sv    # stochastic + vector
--scope-mode fusion --scope-fusion sr    # stochastic + raster
```

`v`, `r`, and `s` are component names, not output channels, and fusion still
has no Z. Press `f` while fusion is active to cycle the four presets.

### Scanlines and the bake ceiling

The runtime area-averages the baked thumbnail down to whatever grid the sample
budget supports — a proper summed-area resample, not line-skipping, which would
alias badly on scanline content. So the thumbnail is a resolution-independent
store like the vectors, and grid size is a runtime decision.

For raster, its size is a **hard ceiling on scanline count**, and exceeding it
is reported rather than silently clamped. Stochastic also uses the stored
spatial field directly, so a bake can remain useful above raster's grid ceiling.
On the supplied portrait, raw contrast was already ample (5th–95th percentile
span 0.81); enlarging the field, not stretching tone, was the useful change.
Width 96 retained 63% of source edge energy, while width 256 retained 90%.

| `--thumb-width` | max scanlines | per folder (2220 frames) | 32 folders |
|---|---|---|---|
| 48 | 64 | 21 MB | 0.66 GB |
| 64 | 85 | 36 MB | 1.16 GB |
| 96 | 128 | 83 MB | 2.63 GB |
| 128 | 171 | 146 MB | 4.67 GB |
| **256** (default) | **341** | **582 MB** | **18.6 GB** |
| 480 | 640 | 2.05 GB | 65.5 GB |

The baker writes each folder's thumbnail array through a temporary on-disk
memmap and atomically installs it when complete. Parallel workers therefore do
not each hold a 582 MB array in RAM, and an interrupted folder does not replace
its previous valid `thumbs.npy` with a partial file. Rebake the complete tree
when changing width; runtime rejects mixed main/float dimensions.

There is a natural ceiling worth knowing: `cells ≈ samples / density`, so no
realistic budget resolves past about 138×184 (6400 samples at density 0.25).
That is raster's ceiling. The default 256px raw channel still benefits
stochastic because its target search runs in stored source-pixel space.

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

### Sweep continuity and interlace

The default `alternate` sweep draws one direction per trace and starts the next
trace at the endpoint the DAC actually received. It then reverses direction,
so fresh traces chain without a flyback and without spending half the samples
redrawing the same image. `palindrome` draws down and back inside one trace;
it is safe when the same trace must repeat, but each direction gets half the
budget. `retrace` deliberately restores the visible CRT-style flyback.

Faint diagonals inside an image are travel between bright regions. They are
inherent without Z blanking and are dimmer than drawn detail because the beam
crosses them quickly.

Trace duration is always `samples / sample_rate`, independent of image
complexity. Two controls decide how that fixed raster budget is spent:

**`--density`** — samples per cell. At 1.0 you get the most cells, each with
about one sample, which reads as dots. At 2–3 you get fewer, larger cells drawn
as solid bright lines. Lower resolution, better-formed image. This is the
knob for "make it draw properly rather than finely".

**`--fields`** — interlacing. Field *f* draws rows `f, f+fields,
f+2·fields…`. Runtime raises the trace rate to `fields × IPS`, so each trace is
shorter but a complete picture still receives `sample_rate / IPS` samples.
The result is the same grid and same 30 Hz complete-picture rate as progressive,
with the screen touched at 60/90/120 Hz for 2/3/4 fields. It buys steadier
phosphor coverage relative to simply raising FPS, not extra resolution over a
30 Hz progressive picture.

| fields | trace rate at 30 IPS | complete-picture rate | grid budget |
|---|---:|---:|---:|
| 1 (default) | 30 Hz | 30 Hz | 1×1600 samples at 48 kHz |
| 2 | 60 Hz | 30 Hz | 2×800 samples |
| 3 | 90 Hz | 30 Hz | 3×533 samples |
| 4 | 120 Hz | 30 Hz | 4×400 samples |

Because interlacing needs a fresh field on *every* trace, the runtime pushes
one whenever a trace completes — not only when the index changes. Otherwise a
held frame would show one field forever and stay permanently combed.

---

## 4b. Mix mode

`--scope-mix [HZ]` (default 120) follows a triangular whole-trace route:
`VECTOR -> RASTER -> STOCHASTIC -> RASTER -> ...`. Above flicker fusion the
phosphor sums all three, so you see one image: raster supplies tone and mass,
vector supplies the outline, and stochastic supplies directionless bitmap
detail. Whole traces are switched rather than sample-interpolated, which would
draw false connectors between unrelated XY positions.

The switch rate **is** the trace rate, so each pass gets `rate / HZ` samples —
400 at 48kHz/120Hz, 800 at 60Hz. Lower HZ gives each mode more detail but
starts to read as three separate component pictures rather than one combined
one.
60–120 Hz is the useful range.

All three modes need their data, so this requires a **full bake**, not
`--thumbs-only`.

`--scope-mix-duty` remains the raster fraction. The non-raster fraction is split
equally between vector and stochastic, so duty 0.5 at 120 Hz yields 30 vector,
60 raster and 30 stochastic passes per second. The beam handoff is chained for
raster and stochastic; vector can still contribute a faint fast connector.

Mix owns the trace clock, so `--scope-samples` is rejected rather than silently
overriding the requested switch rate. Automatic raster interlace is used only
when total and raster traces per source image are whole numbers. The web preview
joins a complete component/field exposure—four traces at the default—so it
shows `V+R+S+R` rather than alternating incomplete halves.

---

## 5. `--thumbs-only`

A bake flag. It writes `thumbs.npy` and skips vectorizing entirely.

- **Much faster** — seconds instead of minutes, since vectorization is all the
  work.
- **Raster, stochastic, and `sr` fusion work.**
- **Vector mode does not** — there's no geometry, so it falls back to the idle
  circle.
- **Fusion presets containing `v` do not** — they require contour geometry.

Use it when you've settled on raster/stochastic and are iterating on content.
Do a full bake when you want live cycling through all three modes.

---

## 6. The sample budget (vector and raster)

Completed vector and raster passes depend on one number: **the path length per
trace**, N. The DAC consumes samples at a fixed rate and the beam position *is*
those samples, so

```
trace duration = N / sample_rate
```

That is the vector/raster coupling between detail and refresh.
Note what is *not* a parameter here: frame rate. It falls out as
`sample_rate / N`. `--fps` is a convenience that sets `N = rate/fps`; `--samples`
sets N directly and is the honest knob. Frames may even vary in length between
traces — the audio callback handles it.

```
samples per trace = sample_rate / fps        # the same equation, rearranged
```

44.1kHz ÷ 30fps = 1470. For a completed vector or raster image, that is the
budget for every line or scanline cell.

Stochastic mode is the exception. Its walk does not restart at a trace boundary
and has no per-image waypoint count. The audio buffer may still be 1470 samples,
but the independent walk continues into the next buffer and the current image
only changes its acceptance probabilities. DAC bandwidth and analogue slew can
soften that path; the nominal DAC rate is not its picture timer.

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
| `--scope-mode vector\|raster\|stochastic\|fusion` | Select the renderer. |
| `--scope-fusion vrs\|vr\|sv\|sr` | Choose round-robin position-array sources for fusion. |
| `--scope-raster` | Compatibility alias for `--scope-mode raster`. |
| `--scope-stochastic` | Alias for `--scope-mode stochastic`. |
| `--scope-walk-radius PX` | Stochastic nearest-neighbour search radius (default 10). |
| `--scope-walk-stride PX` | Stochastic source-pixel step. Default 0 auto-scales: 2 at the default width 256; 1 at 96 and 4 at 480. |
| `--scope-walk-reseed-ms MS` | Time between random region changes (default 5 ms). |
| `--scope-stochastic-gamma F` | Stochastic-only exponent (default 2); `--scope-walk-gamma` remains an alias. |
| `--scope-walk-edge F` | Optional non-Osci edge probability (default 0). |
| `--scope-walk-hz HZ` | Target decisions per second (default 48000), independent of faster DAC rates. |
| `--scope-fps N` | Scope redraw rate. Defaults to `IPS`. Sets `N = rate/fps`. |
| `--scope-samples N` | Path length per trace directly. Overrides FPS; incompatible with mix. |
| `--scope-realtime` | Stream continuously; index changes land within a row (raster only). |
| `--scope-mix [HZ]` | Triangular vector/raster/stochastic/raster whole-trace mix (default 120 Hz). |
| `--scope-mix-duty F` | Raster fraction; remainder splits equally between vector/stochastic (default 0.5). |
| `--scope-sweep MODE` | `alternate` (default), `palindrome`, or `retrace`. |
| `--scope-rows N` | Raster scanline count (default: auto from budget). |
| `--scope-gamma F` | Active renderer's exponent: raster default 2.2, stochastic default 2; sets both in mix. |
| `--scope-density F` | Raster samples per cell (1.0 = finest). |
| `--scope-trim F` | Ignore luminance below this level (default 0.02). |
| `--scope-fields N` | Raster interlacing (1 = progressive). |
| `--ask` / `--scope-ask` | Choose the audio output interactively. |
| `--device X` | Audio output by index or name fragment, e.g. `--device Scarlett` |

While scope mode owns a terminal, press `v` to cycle
`VECTOR → RASTER → STOCHASTIC → FUSION → VECTOR`; in fusion, `f` cycles
`VRS → VR → SV → SR`. Press `p` to print the current mode
and live settings as reusable command-line flags. Cycling is disabled in
`--scope-realtime` and `--scope-mix`, whose audio/scheduling paths are fixed at
startup; restart with `--scope-mode` to leave either one.

### Preview

```
python test_scope_pair.py --xy-dir images_xy [--bg N --fg N --index N] [options]
```

Pass `--fusion vrs|vr|sv|sr` to preview a fusion preset; press `f` to cycle
the presets in the preview window.

Adds `--play`, `--ips`, `--loop`, `--exposure`, `--size`, `--ask` and
`--device` on top of the playback flags.

### Choosing the audio output

By default the system default output is used. `--ask` lists every
stereo-capable output with its host API and native sample rate, and prompts —
but only when more than one exists, and only when there's a terminal to prompt
on, so it can live in a kiosk launch script without ever hanging. Blank input
or an out-of-range number falls back to the default.

For completed raster/vector passes, the listed rate sets `samples per trace =
rate / fps`, so 96 kHz gives more pass samples than a 44.1 kHz built-in output.
For stochastic, a higher device rate samples the same independent 48 kHz walk
more densely; it does not advance the image or target clocks faster.

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
