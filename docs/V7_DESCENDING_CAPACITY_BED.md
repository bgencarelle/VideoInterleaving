# Descending analog dimension-packing test bed

## Logical reset: source geometry before literal dimension ratios

### Solved fidelity gate: fold-native source coordinates

The experimental nonlinear control is repaired. The old design applied a new
source compander and then passed its output through the fold's existing
compander, followed by an unrelated, poorly conditioned inverse. The replacement
uses the production fold's own compressed source coordinates:

```
u = 0.5 + 0.5 * production_compress(c / source_sd)
```

The surface operates on `u`. Its outputs are carried directly in the existing
fold residual domain, without a second guest companding pass. At reception,
surface unpacking precedes the original production guest expansion. Keep the
production fold scale, power model, step, signature size and ordinary ranks.
For 1→1 preserve the production signature itself as well.

This transformed control executes real source forward and inverse transforms;
it is not the no-op branch. On the checked face and first/last movie frames,
both profiles emit bit-identical audio to production and recover decoded values
within approximately 3e-15. A dedicated fidelity regression checks both emitted
audio and decoded/display values, in addition to the separate no-op control.

`fold-native` is the new default. The benchmark validates a transformed 1→1
control against production on every selected source/profile before scoring any
packing configuration. It writes `CONTROL-VALIDATION.json` and stops the sweep
if audio/decoded errors exceed 1e-9 or a picture is unavailable. Previous
arctangent/linear sweep conclusions remain invalid; repairing the control does
not retroactively validate them or prove a dimensional-resolution gain.

Use odd strip counts as additional candidates: three/five-strip midpoints
include the zero-detail compressed coordinate exactly. Four/eight-strip
midpoints do not. This is source-quantization geometry, not noise tuning.

```bash
.venv/bin/python tools/v7_fold_innovation_bench.py \
  --packing --normalization fold-native --strips 3 5 \
  --out tmp/v7-fold-innovation/fold-native \
  --training 16 --faces 2 --movie-entries 3 --short-side 240 --loops 3
```

The prospective valid sweep remains a clean-wire source-capacity experiment;
the 1.5×/3× spatial-resolution objective is still to be measured.

**Historical validation status:** the arctangent 1→1 control failed useful fidelity through
the clean wire. The dimensional and headroom sweeps are invalid as evidence for
a packing-capacity boundary or a resolution winner. The previous finite-value /
acquisition checks were functional tests, not an adequate fidelity gate. Suspend
capacity verdicts and further sweeps until the transformed control is validated.

A new genuine pass-through control (`normalization='identity'`) preserves source
values, fold scales/signature, emitted audio and decoded pictures. A wire-backed
regression asserts exact production equivalence; it passed in stereo and mono.
This validates the no-op wrapper, not the failed nonlinear 1→1 transform.

Stage tracing on the first 4:3 movie frame found six of 484 stereo mapped values
and two of 484 mono values outside the inverse's [-1,1] domain after clean fold
recovery. The inverse then produced very large physical-guest error. This
localizes an observed failure to the experimental recovery/inverse interface;
it is not a validated general analog-capacity limit. Raw trace evidence is in
`tmp/v7-fold-innovation/validation/TRACE.json`.

The user requested reevaluation of the underlying problem rather than literal
implementation of illustrative ratios and amplitude suggestions. The active
question is: what source-aware, perceptually accurate representation of a
higher-detail frame can fit the existing analog carrier capacity?

The implemented nested-strip mapper is a crude candidate, not a representative
test of the full Shannon–Kotel'nikov family. Grouping arbitrary coefficients,
quantizing extra coordinates into strips, and then selecting 9→1 does not design
a surface around the actual source distribution or perceptual distortion.
Changing output amplitude without considering source-domain scale, inverse
conditioning, and the fold's existing companding is similarly incomplete.

The previous arctangent 1→1 control did not preserve useful clean-wire fidelity.
Consequently these sweeps characterize tested implementations but cannot
establish a general packing boundary or dismiss analog dimension reduction.
Functional acquisition/finite-value tests are not evidence of capacity or
perceptual preservation. Suspend further broad ratio/amplitude sweeps until
the control and the source representation are suitable.

Separate three design questions explicitly:

1. Source reduction: identify real spatial structure in the higher-detail frame
   and represent it with fewer perceptually relevant degrees of freedom. Retain
   actual within-frame residuals where the structural model fails. Potential
   direction: geometry-adaptive edge/curve atoms, with subpixel position,
   orientation, contrast and curvature as analog parameters plus residuals.
   Compare against dense texture/phase-changing patterns; no guessed detail is
   credited as source recovery.
2. Channel mapping: map that compact representation into the existing carrier
   values using source-aware geometry and companding. Account for every adaptive
   parameter in the existing picture payload; no free side channel or indices.
   Scaling is a source-range / surface-spacing / carrier-headroom design, not
   merely attenuation followed by an otherwise unchanged inverse.
3. Verification: first validate a perceptually stable identity-dimension mapping
   through the actual clean wire and trace errors before/after each inverse.
   Then measure genuine 1.5×/2×/3× source-correlated detail using a calibrated
   frequency/phase grid and requested natural/movie examples. Defer added-noise
   optimization, preserving frame independence and existing cadence throughout.

A dimension ratio is an internal design choice. The success criterion is
faithful finer structure per fixed packet. No universal 1.5–3× result is claimed;
the next experiments must establish it rather than infer it from a ratio.

## Held-out and dense confirmation

The user requested completion of held-out confirmation and denser resolution
measurements. Read only completed computed outcomes/charts, not streamed logs
or individual trial records. `tools/v7_fold_confirmation.py` produces the
compact `OUTCOME.json/md` for corpus runs.

Freeze the validated `fold-native` run's training hashes and mapping settings.
The corpus runner's `--frozen-from` checks training identity and existing-layout
mapping/model digests. Previously unseen held-out aspect layouts use the same
fixed rule and the same training faces, never held-out fitting pixels.

Held-out test: 12 faces from the test partition and five entries in each of its
three movies, at 480-pixel short-side scoring. Compare 7→3, 5→2, 3→2 and 1→1 at
three strips with production controls. Every scored packet starts a fresh
receiver. Existing validated static-loop evidence is retained; this run does
not repeat those loops. There are 486 scored corpus packets plus fidelity gates.

```bash
.venv/bin/python tools/v7_fold_innovation_bench.py \
  --packing --normalization fold-native \
  --dimensions 7to3 5to2 3to2 1to1 --strips 3 \
  --frozen-from tmp/v7-fold-innovation/fold-native --partition test \
  --out tmp/v7-fold-innovation/heldout \
  --training 16 --faces 12 --movie-entries 5 --short-side 480 --loops 0
```

`tools/v7_fold_dense_probe.py` tests known-frequency grayscale sinusoids across
13 frequencies (8–40 cycles/picture), four phases, two source contrasts (0.05,
0.3), both spatial axes and three layouts (3:4, 4:3, 16:9). Compare production,
7→3 and the lower-distortion 3→2 setting. Source statistics/settings remain frozen.
Each target also passes the transformed 1→1 fidelity gate before being scored.
There are 3,744 scored packets plus 1,248 control packets; raw and final-display
measurements are reported independently. No added channel-noise tuning occurs.

The dense metric measures source-relative contrast, phase and full-raster
off-pattern error, preventing an average profile from hiding orthogonal artifacts.
A phase passes at contrast >=0.2, phase error <=pi/8 and off-pattern residual
<=0.25 of source RMS variation. A reliable cutoff requires at least 3/4 phases
to pass at every tested frequency from the starting point to that cutoff.
Isolated high-frequency passes do not define a reliable resolution limit.
Missing/unestablished limits are not reported as infinite improvement.

```bash
.venv/bin/python tools/v7_fold_dense_probe.py \
  --out tmp/v7-fold-innovation/dense-confirmation \
  --frozen-from tmp/v7-fold-innovation/fold-native
```

Completed outputs: `DENSE-OUTCOME.json`, profile charts and `REPORT.md`. The
calibration smoke run completed 96 scored packets with all fidelity gates passed;
metric regressions verify phase inversion and orthogonal patterns cannot pass,
alongside an actual clean V7 decode. The native dimensional run's one-frame
pitch-6→pitch-4 mono observation is a 1.5× directional lead, not a claim that
the acceptance target is met. Confirmation must establish gains consistently
in both axes and show held-out visual distortion. Concurrent runs do not certify
the realtime processing floor.

### Completed confirmation

Both runs finished and were reviewed by GPT-6 Luna (low) using only final
computed outputs. Held-out: 486/486 pictures, zero errors, fidelity/control
validation passed, and no observed 1.5×/2×/3× directional-frame gains. Dense:
3,744 scored packets / 7,488 source-display views, no unavailable views, and
fidelity/control validation passed. Reliable cutoffs did not consistently
improve across profile, aspect, axis and source contrast; some regressed and
some remained unestablished. Null limits are not infinite gains.

The requested general 1.5–3× spatial-resolution improvement is not established
by these settings. The earlier one-frame directional lead did not provide
held-out confirmation. This is now a valid negative result for the tested
fold-native mappings, not a general impossibility result for analog source
coding. Do not proceed to added-noise tuning as a substitute for the missing
clean-resolution gain. The next candidate needs different source-aware
representation or surface geometry, using the validated test bed.

Combined report and charts: `tmp/v7-fold-innovation/CONFIRMATION.md`.

## Historical descending sweeps

The user selected a top-down experiment: start at improbable dimensional
compression and work down, instead of spending effort on marginal improvements
before finding the clean-wire capacity boundary. This implements the folded-
surface branch of `V7_FIXED_FOLD_INNOVATIONS.md`.

## Implemented sweep

Run the following order, with four- and eight-strip surface mappings at each:

| Source coordinates → transmitted coordinates | Local coordinate ratio |
|---|---:|
| 9 → 1 | 9× |
| 7 → 2 | 3.5× |
| 5 → 2 | 2.5× |
| 7 → 3 | 2.33× |
| 3 → 2 | 1.5× |
| 1 → 1 | identity-dimension normalization control |

Always include the unchanged production default. The 1→1 mapping includes its
own centering/range normalization, so it is not substituted for the default.
User-provided dimension pairs are sorted from largest ratio down, and every
requested pair is attempted; do not stop after an aggressive case fails.

`tools/v7_fold_surface.py` recursively maps extra coordinates into alternating
strips of the transmitted coordinates. Its inverse peels off layers in reverse
order. Extra coordinates are deliberately quantized at strip midpoints; this
is a lossy analog amplitude mapping, not an entropy-coded image bitstream.
Nested layers exponentially increase the required amplitude precision. The
9→1/eight-strip case has eight layers: that is intentionally an aggressive
finite-precision stress test, not a predicted successful configuration.

## Frozen carrier and actual additional data

Freeze production aspect grids, 1920/480/480 coefficient pools, ordinary rank
tables, fold hosts/guests, outer fold step/companding, signature count, model
scale, cadence, headers, metadata, pilots and EOF. Source grouping and matched
inverse interpretation change inside usable fold guest data. Shared mapping
statistics are trained on the training face partition only. No extra per-frame
index/mode/normalization message is supplied.

There are 484 usable guest coordinates; the 16 signature coordinates remain
reserved. Group `k` old guests as carriers for `d` source coordinates, keeping
the old physical guests as the first `k` source coordinates. Remaining source
coordinates are actual previously uncarried luma DCT positions, ordered by
aspect-correct physical frequency. Exclude all ordinary positions carried in
any rank table, all hosts, and all original guests from that extra list. Preserve
leftover carriers if 484 is not divisible by `k`. Reject a requested dimension
if the fixed source grid cannot supply enough distinct coordinates; record the
configuration error explicitly rather than silently reducing its ratio.

This first implementation analyzes source detail within the existing full
96×80 brightness lattice. It cannot claim source frequencies above that
lattice's Nyquist limit. The fixed carrier layout need not prohibit a later
higher-resolution source-analysis lattice, but that is not implemented here.
Packing only the guest part means whole-picture capacity grows less than the
local `d/k` ratio; do not convert that ratio directly into a spatial-resolution
claim. The user's 1.5–3× linear-detail gate still requires decoded evidence.

## What is measured

Every scored picture traverses a real, clean V7 packet with a fresh receiver;
there is no noisy-channel tuning in this stage. Test held-out-from-training face
images and movie entry at separated timestamps. Loop static images separately
to verify frame independence. Keep partial pictures and count unavailable ones.

Save:

* Faithful target pass/contrast/phase results at each available movie pitch.
* Natural source-correlated detail, brightness/color distortion and similarity.
* Actual emitted levels, packet samples and warm processing deadlines.
* Extra source-coordinate count and nesting depth.
* Source-range overload fraction, with clipping made explicit.
* Source-only coefficient projection error as a diagnostic attached to a real
  wire trial, never as an alternative scored reconstruction.
* Source/default/candidate images and frozen model/source/code hashes.

`PACKING-<profile>.png` charts every tested ratio against actual movie-target
passing fractions, with all missing/error trials retained in the denominator.
`packing-summary.json` retains counts and source overload. Existing gain/loss
and resolution charts show perceptual tradeoffs. Sparse fixture pitches cannot
establish a general 1.5× limit; follow promising configurations with dense
frequency/phase probes and held-out images before acceptance.

## Commands

```bash
.venv/bin/python -m unittest modem_tests.test_v7_fold_surface -v
.venv/bin/python tools/v7_fold_innovation_bench.py \
  --packing --out tmp/v7-fold-innovation/descending-arctan \
  --training 16 --faces 2 --movie-entries 3 --short-side 240 --loops 3
```

Custom extension, still sorted from aggressive to modest:

```bash
.venv/bin/python tools/v7_fold_innovation_bench.py \
  --packing --dimensions 12to1 9to1 7to2 5to2 3to2 1to1 \
  --strips 2 4 8 --out tmp/v7-fold-innovation/descending-custom
```

Use a new output directory after code/configuration changes. Dimensions outside
the current source-grid capacity are expected recorded rejections, not evidence
against the mathematical family. Baseline tests verify all default dimension /
strip combinations through stereo and mono audio while pinning geometry/ranks.

## First run and normalization correction

The first `descending` run completed 256/256 pictures with zero trial errors.
All 64 static-loop checks delivered every member, with identical independently
and continuously decoded values. It did not establish the 1.5× linear-resolution
gate. Aggressive mappings generally degraded source-correlated natural detail,
with some isolated mono bar improvements.

Its linear, face-fitted three-sigma source ranges clipped roughly 40–53% of
movie coordinates, including the identity-dimension control. Validation faces
had much lower overload (roughly 0.4–2.7%). These measurements characterize that
normalizer/mapping combination, not a reliable general packing-capacity boundary.
Do not reject the dimension-reduction family from this confounded run.

The corrected default is an invertible arctangent normalization:

```
u = 0.5 + atan((c - mean)/range) / pi
c = mean + range * tan(pi*(u - 0.5))
```

Finite source values need no hard clipping to training ranges. The inverse
uses explicit finite endpoints and a physical coefficient bound for decoded
outliers. Surface quantization remains intentionally lossy, and inverse
conditioning at extreme amplitudes is not assumed harmless. Report both actual
source overload and the fraction outside the old linear training range. Tests
check normalized source round trips as well as actual wire acquisition for every
dimension/density/profile combination.

`--normalization linear` reproduces the old source mapping; `arctan` is the
default. The corrected clean sweep uses `descending-arctan`. No hiss/channel-
noise tuning is introduced by this source-range correction.

### Corrected sweep evidence

`descending-arctan` completed 256/256 pictures with no errors and zero source
overload. All 64 static-loop trials delivered all members identically through
continuous and fresh-state decoding. No mapping established the 1.5× gate.
Stereo surface cases passed no scored bars; mono had isolated pitch-4/pitch-8
passes but no pitch-2/pitch-3 recovery. Natural detail also deteriorated.

The identity-dimension arctangent control is itself unsatisfactory through the
wire: source-only round-trip coefficient MSE is approximately 3e-32 stereo /
9e-32 mono, while its four-strip physical-guest reconstruction margin lost
22.6 dB stereo / 11.7 dB mono relative to production. This demonstrates that
removing source clipping did not establish a clean, stable carrier mapping.
Endpoint inverse conditioning and finite clean fold/decoder precision are
relevant hypotheses, not a proven isolated cause. Do not infer a general
dimension-packing capacity limit from a failing 1→1 control.

Before another full sweep, characterize the actual clean carrier transfer and
validate a stable identity-dimension control. Then evaluate alternative surface
geometry/scaling/placement and denser frequency/phase probes. This is a clean-
wire structural diagnostic, consistent with deferring added-noise tuning.
Reports and examples are in `tmp/v7-fold-innovation/descending-arctan/`.

## Packed-amplitude headroom sweep

The user requested lowering amplitude when the aggressive mapping approaches
endpoints, accepting the signal/noise tradeoff. Test this directly before
discarding the mapping. `--headroom` scales the centered packed amplitudes by
`h` before the existing fold; the receiver divides by `h` before surface
unpacking. Full-amplitude fold normalization, outer step, companding, ordinary
coefficients, model scale and signature pattern remain fixed across levels.
Recalibrating guest scale at each level would cancel the intended attenuation.

Sweep `h = 1, 0.75, 0.5, 0.25, 0.125, 0.0625`. These are picture guest-amplitude
levels, not reductions of the entire audio packet or changes to header/pilot
levels. Retain real emitted RMS/peaks and clean decoded fidelity. Inverse gain
is `1/h`; its dB value is recorded alongside packed amplitude peak and endpoint
occupancy. This intervention adds carrier headroom; it does not mathematically
remove the arctangent inverse's conditioning. Whether it improves net recovery
is measured through the actual wire.

First isolate the 9→1 stress case and the 1→1 control at four strips:

```bash
.venv/bin/python tools/v7_fold_innovation_bench.py \
  --packing --dimensions 9to1 1to1 --strips 4 \
  --headroom 1 .75 .5 .25 .125 .0625 \
  --out tmp/v7-fold-innovation/headroom \
  --training 16 --faces 2 --movie-entries 3 --short-side 240 --loops 3
```

Source-coordinate means/ranges and surface geometry are unchanged between
amplitude levels. The record identifies headroom separately; the fold table's
signature is held fixed to avoid confounding the sweep with signature-noise
changes. Both ends are explicitly preconfigured with the same headroom.
Focused wire tests assert that physical carriers really shrink by `h`, ordinary
coefficients do not change, and guest normalization/signature remain identical.
All default mapping checks and ten lowered-amplitude/profile wire cases passed.

### Headroom results

The sweep completed 256/256 pictures with zero errors and zero source overload.
All 64 static-loop checks delivered every member identically. Packed peaks
decreased from near 1 to at most 0.0625 as intended. Neither stereo nor mono
9→1 passed any of the 60 scored movie bar cases at any amplitude; the 1.5× gate
remains unproven. At 6.25%, the physical-guest signal/reconstruction-error margin
fell a further 11.38 dB stereo / 13.27 dB mono relative to the full-amplitude
9→1 case. The identity-dimension control also deteriorated as amplitude fell.

This result concerns packed-output attenuation and matched inverse expansion,
with all fold scales held fixed. It does not test changing source-coordinate
amplitude/range before normalization or redesigning the folded surface. Do not
reject those distinct interventions on this evidence. Reports, measured
headroom chart and comparisons are under `tmp/v7-fold-innovation/headroom/`.
Characterize actual clean transfer/inverse errors before another broad sweep;
the identity-control problem has not been solved by output attenuation.
