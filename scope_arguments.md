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
samples/trace   = sample_rate / trace_rate.  THE budget. Everything competes for it.
IPS             = 30. settings.IPS. How fast the artwork advances. Unrelated to the beam.
```

The one thing to internalise: **the DAC eats samples at a fixed rate, so the
only way to draw more detail is to spend more time, and the only way to spend
more time is to refresh less often.** Every "quality" flag here is a rule for
dividing a fixed sample budget. Nothing creates samples.

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

It does not set anything directly. It computes `samples = sample_rate / N`, and
*that* is the real quantity. Raising fps to 60 does not make the picture
smoother, it halves your sample budget and therefore shrinks the grid by √2 in
each axis. Before the interlace work, that was the only knob you had for
flicker and it cost you resolution every time.

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

### `--scope-gamma F` (default 2.2)

Contrast of the dwell curve. Applied as `weight ** gamma` on the per-cell
brightness before the beam path is walked, so it maps image brightness onto
beam dwell time.

Useful band is 1.8–2.2. Below that the image goes flat because dwell differences
shrink; above it, midtones collapse toward the floor and you lose the tonality
that raster mode exists to provide.

This is the one Group B/C flag that does **not** change the grid, so it is
cheap to sweep with the `[` and `]` keys while watching.

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

### `--scope-mode vector|raster|stochastic` (default `SCOPE_RENDER_MODE`)

Selects the rendering engine explicitly:

- `vector` traces the baked contour geometry at constant arc length.
- `raster` draws dwell-modulated horizontal scanlines.
- `stochastic` performs a luminance-weighted nearest-neighbour XY walk. It has
  no scanline spacing and needs no Z/brightness channel.

### `--scope-raster` / `--scope-stochastic`

Compatibility shortcuts for `--scope-mode raster` and
`--scope-mode stochastic`. They are mutually exclusive with each other and
with an explicit `--scope-mode`.

### Stochastic controls

| flag | default | effect |
|---|---:|---|
| `--scope-walk-radius PX` | 10 | Radius searched for a nearby unvisited accepted pixel. |
| `--scope-walk-stride PX` | 1 | Spacing of source-pixel candidates. Higher is faster/coarser. |
| `--scope-walk-reseed-ms MS` | 5.0 | Interval between weighted jumps to another bright region. |
| `--scope-walk-edge F` | 0.35 | Extra visit probability assigned to luminance gradients. |

`--scope-trim` and `--scope-gamma` also apply to stochastic mode. Density,
rows, fields, autofit, border, and sweep are raster-only. The common
`--scope-lowpass` remains available to smooth the resulting XY signal.

### `--scope-min-feature F` (default 0.02)

Vector only. Shortest stroke the occlusion cull keeps. Irrelevant in raster.

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

Alternate raster and vector traces at this rate. Above flicker fusion the
phosphor sums them: raster supplies tone, vector supplies outline.

Note it **overrides your trace rate** — `fps = mix_hz` — because the switch rate
*is* the trace rate. So `--scope-mix 120` silently puts you at 120 traces/sec and
a correspondingly small budget. Incompatible with realtime and with fields.

### `--scope-mix-duty F` (default 0.5)

Fraction of mixed passes spent on raster. Raise to 0.6–0.8 if the vector outline
overpowers the tone. Inert without `--scope-mix`.

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
| `[` / `]` | gamma down / up (0.4–5.0) | no |
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

1. **`--scope-dc-comp` does not exist.** It is in the run command in both
   handoffs. `grep -rn "dc_comp\|precompensate" *.py` finds nothing on the
   branch. The documented command fails with an argparse error. The
   `precompensate_hpf` function exists in your local `scope_out.py` but nothing
   calls it.

2. **You cannot pin the audio device from `settings.py`.** `scope_display.py:227`
   reads `settings.SCOPE_DEVICE_SPEC`, but `main.py:357` sets that to `None`
   unconditionally, and `settings.SCOPE_DEVICE` is only consulted when
   `SCOPE_DEVICE_RESOLVED` is true, which only `--device`/`--ask` sets. So the
   device must come from the CLI every time. For an unattended install that
   means the launch script is the only place it can live.

3. **Density is inverted from intuition.** Lower = finer. `cells = samples /
   density`.

4. **`--scope-mix` hijacks your trace rate.** `fps = mix_hz`, so `--scope-mix 120`
   is also `--scope-fps 120` and a quarter of your usual budget.

5. **Eight `SCOPE_*` settings are undeclared in `settings.py`** and exist only if
   the CLI creates them: `SCOPE_RASTER`, `SCOPE_SAMPLES`, `SCOPE_REALTIME`,
   `SCOPE_LIST_FROM_IMAGES`, `SCOPE_DEVICE`, `SCOPE_DEVICE_SPEC`, `SCOPE_ASK`,
   `SCOPE_BLOCKSIZE`. Everything reads them via `getattr(..., default)` so
   nothing breaks — but you cannot set them in `settings.py` and expect them to
   be picked up unless you add the line yourself.

6. **`SCOPE_BLOCKSIZE` is documented but not read.** `Scope.__init__` takes
   `blocksize=512` as a Python default; no code looks for the setting. Your own
   measurement says blocksize isn't a meaningful CPU lever anyway (1.2 µs
   callback overhead).

7. **The `a` autofit key is inert in a calibrated run**, because calibration
   supplies a fixed grid and that disables the autofit path.

8. **Several flags use truthiness, not `is not None`.** `--scope-fps 0`,
   `--scope-rows 0`, `--scope-oversample 0` are silently ignored rather than
   erroring. `--scope-trim`, `--scope-gamma`, `--scope-density`,
   `--scope-min-feature` and `--scope-mix-duty` correctly use `is not None`, so
   `--scope-trim 0` does work.

---

## Correction to something I told you earlier

Two posts ago I said the supplement was wrong to claim `--scope-ask` and
`--scope-device` exist. **I was wrong and the supplement was right.** `main.py`
line 136 is `parser.add_argument("--device", "--scope-device",
dest="scope_device", ...)` and line 141 is the same pattern for `--ask` /
`--scope-ask`. Both spellings work. I had grepped for the flag strings in a way
that only surfaced the first alias and then asserted the negative from it.

Everything else in `SCOPE_CORRECTIONS.md` still holds — I re-checked
`--scope-dc-comp`, `scope_profile.py` and `scope_screen.py` against the branch
while writing this.
