# Scope mode — every argument, explained

Read from the code, not from either handoff. Source: `main.py` lines 84–150
(argparse), 302–357 (plumbing into settings), `settings.py` 86–114 (defaults),
`scope_display.py` 210–250 (what actually gets read).

---

## First: the three numbers everything else hangs off

Almost every flag below is really an indirect way of moving one of these. If
you hold these in your head the rest stops being arbitrary.

```
sample_rate     from the audio device. You do not set it. 96000 typical.
samples/trace   = sample_rate / trace_rate. Vector/raster pass budget; stochastic buffer size.
IPS             = 30. settings.IPS. How fast the artwork advances. Unrelated to the beam.
```

For vector and raster, the DAC eats a fixed number of samples per completed
pass, so more pass detail costs time. Stochastic is different: it is a
continuous target stream and a "trace" is only an audio buffer. Lowering its
trace rate does not create more target decisions per second.

Two rates that are easy to conflate and are not the same thing:

- **trace rate** — how often the beam repaints the screen. This is what you see
  as flicker.
- **picture rate** — how often the image content changes. This is `IPS`.

Before interlace those were forced equal. That was the whole refresh problem.

---

## Group A — the sample budget

### `--scope-fps N` (default: `IPS`, so 30)

Trace rate. Named "fps" but it is not a frame rate in the video sense; it is
how many times per second the beam completes a pass.

It computes `samples = sample_rate / N`. In raster, raising fps to 60 halves
the pass budget and shrinks the grid by √2 in each axis. In stochastic it only
halves the buffer duration; the walk state and target clock continue unchanged.

**With `--scope-fields` set, do not pass this at all.** The code computes
`fields * IPS` for you. If you pass a value that isn't a multiple it overrides
you and prints why.

### `--scope-samples N` (default: unset)

The same setting, stated honestly. Sets the path length per trace directly;
trace rate falls out as `sample_rate / N`.

Overrides `--scope-fps` completely. Use it when you want a specific budget and
don't care what rate that implies. **It also suppresses the automatic
`fps = fields * IPS` calculation** — if you pass both `--scope-samples` and
`--scope-fields`, the code leaves your sample count alone and prints what the
correct value would have been, but does not change it. That is deliberate: you
asked for a specific budget explicitly.

It cannot be combined with `--scope-mix`. Mix rate is itself the trace clock;
letting an explicit sample count override it made the displayed duty/pass rates
false. The CLI now rejects the combination, and a stale settings.py combination
prints a warning and follows `SCOPE_MIX`.

### `--scope-fields N` (default: 1 — new)

Interlace. `N=2` draws every other row per trace and alternates which set,
so a complete picture takes 2 traces.

This is the only flag here that gives you something for nothing, because it
changes *when* rows are drawn rather than how many samples they get. A picture
still costs `sample_rate / IPS` samples in total and the grid is sized from that
total — so the grid does not change — but the beam covers the full screen
height N times as often.

Measured: 69×92 grid and 1.13 samples/drawn cell at `fields` = 1, 2, 3 and 4.
Identical. Only the repaint rate moves.

Cost: interline twitter on fine horizontal detail (one scanline updates at
30 Hz while its neighbours update on the other field), and a growing seam
between traces — 0.062 on a ±1 screen at N=2, 0.202 at N=4. **Start at 2.**

Silently ignored in vector, realtime and mix mode, each with a printed reason.

---

## Group B — how the sample budget is divided into a grid

These four interact, and the interaction is where the confusion lives.

The grid is computed roughly like this:

```
cells = samples * fields / density        # how many cells the budget affords
rows, cols = split(cells, by aspect)      # or forced by --scope-rows
if autofit: grow both by 1/sqrt(fraction surviving trim), capped at 2.5x
clamp to the baked thumbnail size
```

### `--scope-density F` (default 1.0)

Target samples per grid cell. **This is the resolution/stability dial and its
direction is counterintuitive:** lower density means *more* cells, i.e. a finer
grid, because `cells = samples / density`.

- `1.0` — one sample per cell. The finest grid that is stable.
- Below `~0.3` — the grid outruns the sample count so far that which cells get
  visited changes from trace to trace. Reads as moving black flecks, not as
  detail. Hard-clamped at 0.25 with a printed warning.
- Above `1.0` — fewer, longer, brighter scanlines with less detail. A genuine
  trade, not a downgrade. Worth it only if thin traces read too dim on your
  tube.

Your handoff has the measured stability curve: at a 1 px content shift, the
fraction of lit cells that move is 19% at density 1.0, 23% at 0.7, 36% at 0.4,
52% at 0.25. That is why 0.4 looked better on a still and was wrong in motion.

### `--scope-trim F` (default 0.02, you run 0.10)

Cells dimmer than this are dropped from the sweep entirely.

Two separate effects, and the second is the one people miss:

1. **It removes stray lines.** Without it the beam sweeps across black margins,
   and "dim" on a scope is still a visible line. Raising trim kills those.
2. **It reallocates budget.** A dropped cell costs no samples, so its share goes
   to the subject. On a portrait over black this is most of the frame, which is
   why trim is a resolution setting as much as a cleanliness setting.

Range that does anything: 0.08–0.16. Your 0.10 is sensible.

### `--scope-no-autofit` (default: autofit ON)

Autofit is the correction that makes trim's second effect actually happen.

Without it, the grid is sized for the whole rectangle, then trim throws half the
cells away — so half your samples were budgeted for cells that are never swept,
and the picture comes out coarser than the budget allows. Autofit does one cheap
probe pass, measures what fraction survives trim, and grows the grid by
`1/√fraction` (capped at 2.5×) to compensate.

On a dark background this is roughly a 2× resolution win. **Leave it on.** The
flag exists to turn it off for A/B comparison.

One thing that is not obvious: **once `calibrate()` has run, autofit is
bypassed.** Calibration produces a fixed `grid_rows`/`grid_cols`, and
`render_luma` sets `autofit = False` whenever those are supplied. So the `a` key
at runtime does nothing in a calibrated run. The autofit correction is still
applied — `calibrate` does its own equivalent grow step — it just isn't this
code path doing it.

### `--scope-rows N` (default: auto)

Force the scanline count instead of deriving it from the budget and aspect.

Pure trade against columns: the budget is fixed, so rows you add come out of
horizontal resolution. Hard-capped by the baked thumbnail height — if you ask
for more than the bake has, it clamps and tells you what `--thumb-width` to
rebake with.

Mostly a diagnostic. The auto split is usually right.

---

## Group C — tone

### `--scope-gamma F` (raster default 2.2; stochastic default 2)

Contrast of the dwell curve. Applied as `weight ** gamma` on the per-cell
brightness before the beam path is walked, so it maps image brightness onto
beam dwell time.

Useful band is 1.8–2.2. Below that the image goes flat because dwell differences
shrink; above it, midtones collapse toward the floor and you lose the tonality
that raster mode exists to provide.

This is the one Group B/C flag that does **not** change the grid, so it is
cheap to sweep with the `[` and `]` keys while watching.

In stochastic mode this controls the walk's fresh per-candidate luminance
probability. Its default is 2, equivalent to Osci-render's Image Threshold 0.1.
Osci's UI default of 0.5 maps to exponent 6, which discards too many portrait
midtones for this material.

In mix mode `--scope-gamma` sets both raster and stochastic. Add
`--scope-stochastic-gamma F` when the two need different values; it overrides
only stochastic. The older spelling `--scope-walk-gamma` remains accepted as
a compatibility alias.

Fusion follows the same two-gamma rule: `--scope-gamma` initializes both
luminance sources, and `--scope-stochastic-gamma` can separate the raw
stochastic contribution from raster's grid-compensated contribution.

---

## Group D — the output path

### `--scope-oversample N` (default 1)

Generate `N ×` samples, bandlimit circularly, decimate back down.

The problem it solves: point-sampling a path is not anti-aliased, so geometry
finer than the sample spacing folds back as noise rather than averaging into the
signal. Oversample-and-decimate turns that detail into correct low-frequency
content. It is what makes a grid finer than one cell per sample usable rather
than just noisy.

It cannot exceed Nyquist. The gain is accuracy below it, not bandwidth. Your own
measurement: ~5% change on real content. `4` is plenty; `1` is fine.

### `--scope-lowpass HZ` (default off)

Low-pass the XY output at this corner, to emulate a softer DAC or a physical RC
filter — a Pi headphone jack, say. Try 800–8000.

This is a *simulation* tool: it lets you see on good hardware what the install
hardware will do, before you get there. Audition corners with `scope_lowpass.py`
first, or cycle them live with `l` (off → 12k → 6k → 3k → 1.5k).

Completed repeating vector/raster traces use a circular zero-phase filter.
Stochastic does not repeat at buffer boundaries, so it uses a causal stateful
filter whose state carries across buffers; treating each stochastic buffer as
a loop creates a false seam and is explicitly avoided.

### `--scope-sweep MODE` (default `alternate`)

How consecutive traces connect. This is more consequential than it looks.

- **`alternate`** — each trace sweeps one direction; the next resumes from where
  it ended and draws the *next* index. No flyback, full sample density. This is
  the right answer and the default.
- **`palindrome`** — sweeps down then back up the same path in one trace. Also
  avoids a flyback, but the return pass redraws the same image, so every cell is
  visited twice at half density. It is the safe fallback for when a fresh frame
  is *not* guaranteed every trace.
- **`retrace`** — closes the loop with an explicit dim flyback segment. Shows the
  CRT diagonal deliberately. Aesthetic choice.

**This changed with the recent work.** `alternate` emits frames with `close=False`
— last and first sample on opposite sides of the picture, no samples budgeted for
the jump. That is correct only if a fresh chained frame arrives every single
trace. It previously did not, and the repeated frame looped across that jump as
a full-brightness diagonal. Raster now renders per-trace and gates on
`scope.ready()`, so this cannot happen and `palindrome`'s reason to exist has
largely gone in raster mode. Vector still emits per index, so the old warning
still applies there.

---

## Group E — which engine runs

### `--scope-mode vector|raster|stochastic|stipple|fusion` (default `SCOPE_RENDER_MODE`)

Selects the rendering engine explicitly:

- `vector` traces the baked contour geometry at constant arc length.
- `raster` draws dwell-modulated horizontal scanlines.
- `stochastic` performs a luminance-weighted nearest-neighbour XY walk. It has
  no scanline spacing and needs no Z/brightness channel.
- `stipple` builds stable luminance-weighted image positions, orders them by
  unrestricted Euclidean proximity, and resamples that completed route. It
  keeps the stochastic maze mode available under its existing name.
- `fusion` generates corresponding position arrays for the selected renderers
  and selects their entries round-robin by array index.

### `--scope-fusion vrs|vr|sv|sr` (default `vrs`)

Selects fusion's round-robin component set: all three, vector+raster,
stochastic+vector, or stochastic+raster. `vr` produces
`V[0], R[1], V[2], R[3]...`; no XY coordinates are averaged. Press `f` in
fusion mode to cycle the presets.

### `--scope-raster` / `--scope-stochastic` / `--scope-stipple`

Compatibility shortcuts for `--scope-mode raster` and
`--scope-mode stochastic` or `--scope-mode stipple`. They are mutually
exclusive with each other and with an explicit `--scope-mode`.

### Stochastic controls

| flag | default | effect |
|---|---:|---|
| `--scope-walk-radius PX` | 10 | Radius searched for a nearby unvisited accepted pixel. |
| `--scope-walk-stride PX` | 0 (auto) | Source-scale-aware spacing: 1 at the compact default width 128; explicit higher values are coarser. |
| `--scope-walk-reseed-ms MS` | 5.0 | Interval between random relocations. |
| `--scope-gamma F` | 2.0 | Fresh `luminance ** gamma` acceptance test for every candidate in stochastic mode. |
| `--scope-stochastic-gamma F` | 2.0 | Stochastic-only gamma override, especially for mix. `--scope-walk-gamma` is an alias. |
| `--scope-walk-edge F` | 0.0 | Optional gradient probability. Zero matches Osci-render. |
| `--scope-walk-hz HZ` | 48000 | Target decisions per second, independent of image rate and faster DAC rates. |

Stochastic always excludes luminance at or below 0.2, as Osci-render does; a
larger `--scope-trim` raises that floor. Density, rows, fields, autofit, and
sweep are raster-only. `--scope-border` also applies to stochastic and stipple:
it replaces the requested tail share after the complete continuous path has
been generated, then rejoins that path's original endpoint so the walk or
route stays continuous across buffers. Fusion applies one border after its
component-position multiplexer, rather than interleaving pieces of several
component-local rectangles. The common `--scope-lowpass` remains available to
audition a bandwidth-limited XY chain.

There is no stochastic waypoint budget. At the default 48 kHz target clock a
48 kHz DAC chooses one target per sample; a 96 kHz DAC samples each target
interval twice. The current image may change at 30 IPS while the walk carries
on continuously through that change.

### `--scope-stipple-points N` (default 768)

Number of deterministic luminance-weighted positions used to construct a
stipple route. This is an image-route parameter, not a sample-rate clock.
Repeated positions provide dwell; Euclidean nearest-neighbor ordering keeps
unavoidable no-Z connectors local. Stipple shares `--scope-gamma`,
`--scope-trim`, and optional `--scope-walk-edge`, but ignores stochastic's
radius, stride, reseed interval, and walk rate. It shares the fixed-extent
`--scope-border`; the border returns to the unbordered route endpoint so it
does not restart the next frame's nearest-neighbour tour.

Current bakes store 1024 source-detail candidates analyzed at 256px rather than
a complete 256px luminance plane. The runtime reweights those candidates, so
live gamma remains functional and the compact 128px raster field does not limit
stipple coordinate placement.

### `--scope-precondition F` (compact-bake default 0.45)

Horizontal-only raster compensation applied after luminance has been reduced
to the final sweep grid. It does not add a stored channel. Legacy bakes default
to zero because their raster channel may already contain bake-time sharpening;
an explicit value overrides that compatibility choice.

### `--scope-min-feature F` (default 0.02)

Vector and fusion presets containing `v`: shortest stroke the occlusion cull
keeps. Irrelevant in raster, stochastic, and `sr` fusion.

### `--scope-realtime` (default off)

Stream continuously from a worker thread so index changes land within a row
rather than at a trace boundary, cutting latency to ~1 ms.

Raster only; refuses in vector, and mix refuses it. **Not for slow hardware** —
if the generator can't keep up you get audio underruns, which sound and look
worse than the latency you were fixing. It also uses `SweepSource`, which is the
second implementation of the raster algorithm your handoff flags as having
drifted from `raster_frame` once already.

Also incompatible with `--scope-fields`, since interlace needs whole traces.

### `--scope-mix [HZ]` (default off; bare flag gives 120)

Run a triangular whole-trace sequence at this rate:
`VECTOR -> RASTER -> STOCHASTIC -> RASTER -> ...`. Above flicker fusion the
phosphor sums all three: raster supplies tone, vector supplies outlines, and
stochastic adds directionless bitmap detail. Whole traces are switched rather
than interpolated; interpolating unrelated XY coordinates would draw false
connector shapes between them.

Note it **overrides your trace rate** — `fps = mix_hz` — because the switch rate
*is* the trace rate. So `--scope-mix 120` silently puts you at 120 traces/sec and
a correspondingly small budget. It is incompatible with realtime and with an
explicit `--scope-samples`. The raster part can still use `--scope-fields`;
automatic field sizing is based on the raster share and is enabled only when
both total and raster traces per source image are whole numbers. Otherwise it
stays progressive so an interlaced picture never straddles two source images.

The hardware sees one trace at a time, but the web preview joins enough traces
to contain every raster field plus at least one vector and one stochastic pass.
At the default duty this is the complete four-trace `V,R,S,R` exposure, rather
than alternating incomplete `V+R` and `S+R` previews.

### `--scope-mix-duty F` (default 0.5)

Fraction of mixed passes spent on raster. The remainder is split equally
between vector and stochastic. At 0.5 the exact repeating sequence is
`V, R, S, R`; at 120 Hz that is 30 vector, 60 raster, and 30 stochastic
passes per second. Raise to 0.6–0.8 if tone needs more beam time. Inert without
`--scope-mix`.

---

## Group F — plumbing

### `--xy-dir DIR` (default `settings.XY_DIR`)

The baked libraries. This is the real input to scope mode.

### `--dir DIR`

The *image* source folder. **Inert in scope mode.** Scope reads the manifest from
the bake and never opens an image. It is a general flag that applies to the other
four modes; it is listed in the scope docs only because it confused someone once.

### `--scope-list-from-images`

Rebuild the folder manifest by scanning the image tree instead of reading it from
the bake. Only needed for a bake made before manifests existed. Slow — it is the
71k-file rescan the CSR layout was designed to avoid.

### `--device X` / `--scope-device X`

Audio output, by index or name fragment: `--device Scarlett`.

**Prefer the name.** PortAudio indices reshuffle whenever hardware is plugged or
unplugged, so an index that works today is a wrong-device bug at the install.
When you do use an index, it indexes the *output-only* list — the same numbering
`--ask` prints, not the full device list.

### `--ask` / `--scope-ask`

Choose interactively. Only prompts when more than one output exists. Resolved
early in `main.py`, before file lists are built, because prompting from deep
inside `run_scope` meant the question appeared after a long silence and read as
a hang.

---

## Live keys (while running)

From `scope_controls.py`, verified against `SPECS` and `HELP`:

| key | does | recalibrates? |
|---|---|---|
| `-` / `=` | trim down / up (0.0–0.60, steps) | yes |
| `,` / `.` | density finer / coarser (0.20–4.0, multiplies) | yes |
| `[` / `]` | active renderer gamma down / up (0.4–10.0) | no |
| `l` | lowpass cycle: off → 12k → 6k → 3k → 1.5k | no |
| `v` | vector → raster → stochastic → vector | yes |
| `w` | sweep: alternate → palindrome → retrace | no |
| `a` | autofit on / off | yes |
| `p` | **print current settings as a command line** | — |
| `h` | help | — |
| `q` | quit | — |

`p` is the one to know. Tune by ear on the tube, hit `p`, paste the result into
your launch script. That is the intended workflow and it is not obvious from any
doc.

Mode cycling is unavailable in realtime and mix modes because those select a
different audio/scheduling path at startup.

Not live-adjustable, because they need the audio stream reopened: `--scope-fps`,
`--scope-samples`, and `--scope-fields`.

---

## Gotchas, ranked by how much time they'll cost you

1. **Density is inverted from intuition.** Lower = finer: `cells = samples /
   density`. Below about 0.3, cells start receiving less than one sample and
   the picture can break into moving flecks.

2. **Mix owns the trace clock.** `--scope-mix 120` means 120 complete passes
   per second. `--scope-fps` is superseded and `--scope-samples` is rejected.

3. **Interlace needs integer registration.** In mix, both total traces/index
   and raster traces/index must be whole numbers. The default 120 Hz, duty 0.5,
   and 30 IPS gives four total and two raster traces, so it is exact.

4. **Partially rebaking is not registration.** Main and float thumbnails must
   have the same dimensions and frame count. Runtime rejects mismatches instead
   of silently dropping one layer; rebake the complete tree after changing
   `--thumb-width`.

5. **There is still no invisible travel without Z.** Raster and stochastic
   hand off at the beam's actual endpoint. A vector pass has its own first
   vertex, so entering vector can still make one fast, faint connector.

6. **Use a device name for unattended installs.** `SCOPE_DEVICE = "Scarlett"`
   works in settings.py; `--device Scarlett` overrides it. Numeric PortAudio
   ordering can change when hardware is reconnected.
