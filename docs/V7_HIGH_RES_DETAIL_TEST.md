# Test design: recoverable high-resolution source detail

## Mandatory implementation constraint

**NumPy and SciPy are forbidden for numerical computation. All new numerical
algorithms must run in Numba-compiled code**, including fitting, spectral search,
synthesis, calibration, residual analysis and scoring. Python is for orchestration,
input/output and reporting. No NumPy/SciPy fallback may be used to pass a benchmark.
Array storage and array intrinsics compiled by Numba are permitted; interpreted
NumPy/SciPy numerical kernels are not. The existing production V7 transport is
the fixed baseline, not a numerical-library rewrite target for this experiment.

## Source-first rule and volume control

Conversion invariant: the production direct-DCT source returns **spatial coder-grid
values**. `grid.forward()` converts those values to coefficients for editing;
`grid.inverse()` converts edited coefficients back once. `base_values()` already
returns spatial values and must be rendered directly, without another inverse.
The new fixed-residual benchmark's `source-screen` and `source-screen-v2` runs
violated this invariant and are explicitly invalidated. Regression tests now
verify the zero-residual base and that the direct-DCT source is called once.
The corrected benchmark rerun is `tmp/v7-residual-detail/source-screen-corrected/`.
Its losses are ordinary source-model tradeoffs, not the catastrophic losses in
the invalid double-inverse run. Do not reuse the invalid images or measurements
as evidence about source representation or analog capacity.

### Fixed-support residual fitting checks

`tools/v7_sparse_residual.py` carries source-selected DCT coordinates and
normalized amplitudes; its independent analytic truth regression checks the
Numba DCT-II against an explicit cosine sum and reconstructs two high-frequency
modes. Amplitudes are DCT coefficients divided by `sqrt(height*width)`, paired
with a cosine basis whose pixel mean square is one. This convention is
dimension-independent; it is not an orthonormal-coefficient wire convention.

`tools/v7_patch_residual.py` uses fixed overlapping supports with a local phase
origin, two frequencies and two quadrature amplitudes per support. Its first
four-support independently varied truth regression failed with approximately
44.9% relative RMS error. The fitter independently seeded neighboring supports
with the same dominant wave and retained coarse FFT frequencies. Sequential
residual seeding and bounded continuous frequency refinement repair that
fixture to below the provisional 10% source-recovery threshold. This is a
fitting regression, **not** a general source-quality or audio-resolution result.

`tools/v7_sparse_source_bench.py` now projects the adaptive control through the
same clean support model as its forced candidate. It records the chosen format
and hashes both residual implementations; picture labels distinguish the patch
and sparse formats. Existing V7 is rendered by the production renderer so its
ordinary coefficients cannot be mistaken for experimental parameters. These
are encoder-side projections only, and every image is labeled pre-audio.

Completed refined patch source screen:
`tmp/v7-patch-detail/source-screen-refined-v2/`, four held-out face pictures for
each profile and each 4/16-support budget. Both stereo budgets have four
overall-MSE wins, but mean improvements are only about 0.63% and 1.26%,
respectively. Neither demonstrates a general fine-band improvement; the
16-support candidate worsens the 24-cycle band. Mono has zero overall-MSE
wins for either budget. **These results do not establish the requested 1.5x
resolved-detail gain and do not qualify the model for an audio capacity claim.**
The earlier patch screen predates continuous refinement and projected adaptive
controls; retain it as historical source-only evidence.
The v2 rerun preserves sequential amplitude seeds and prevents clipped amplitude
updates from increasing the per-support fitting objective. Its aggregate scores
agree with the first refined screen to numerical precision.

**A faithfully transmitted bad source approximation is not a capacity result.**
Before sending a fitted experimental picture, compare its intended reconstruction
with the original and the existing V7 encoder-side reference. For paired probes,
the intended decoded difference must already meet the fine-detail recovery
criterion, and both constituent fits must improve source fidelity. Reject a
failing pair before experimental audio encoding. Keep its original, existing V7
and failed pre-audio reconstruction visible, with the audio columns explicitly
labeled as source-rejected. Report all rejected families and pairs; screening
cannot convert selective successes into a general information-recovery claim.

For passing source models, turn down experimental waveform volume before decoding
with one uniform gain no greater than one. Cap both RMS and peak at the actual
existing-V7 packet's measured levels. Record original levels, attenuation and
received levels, then repeat decode qualification on the reduced-volume audio.
This changes volume, not the picture parameters or packet timing. Do not adjust
the recovered image's gain or alignment to improve its score.

The runner still has a separate known-atom transport preflight. That validates
parameter recovery without presenting source-fit failure as a capacity outcome.

This rule was added after the original pilot overemphasized successful delivery
of unsuccessful source fits. The source representation currently uses eight
windowed waves, initially a discrete center/width search and RGB-MSE selection. Those
choices can favor broad approximation errors over useful natural fine detail.
Source rejection is evidence against that model on the rejected input, not a
transport limitation. Improving the fitter/representation must precede further
audio capacity claims on those inputs.

Completed corrected screen: `tmp/v7-local-detail/source-first-volume-verified/`.
Of 40 source pictures, **32 were rejected before experimental audio encoding**.
Only the eight pictures in the four passing local-texture pairs entered the
experimental pipe. All ten transport checks (eight accepted pictures plus two
known-atom preflights) passed after volume reduction; worst floor-normalized
detail RMS error was 0.00257. Actual mean RMS ratios were 0.999999 for both
profiles; peaks were also below the reference. The x/y localized-texture pairs
still passed, while rejected families remained explicitly rejected. Seven
focused local-detail tests passed, including independent decoding after volume
reduction. This fixes test sequencing and the volume control, not the source
representation's failures or its approximately 176 ms fitting/encoding cost.

## Question and scope

Can a localized, overlapping directional-atom representation carry more
faithfully recoverable source detail than existing V7, through the same
aspect/fold-500 audio layout, while preserving useful base-picture quality?

This protocol defines the full confirmation experiment. The initial localized
codec and decode-first pilot runner are implemented in `tools/v7_local_detail.py`
and `tools/v7_local_detail_bench.py`. The pilot does not yet implement the full
boundary calibration/held-out confirmation schedule below and cannot establish
a 1.5x resolution claim. The eight-global-wave codec is a historical control.

## Source-fitting work after volume control

`tools/v7_local_detail_fit.py` now refines continuous positions, widths and
frequencies with bounded variable projection. Amplitudes are refitted at each
geometry step. Each atom is optimized against the other atoms' residual, and
accepted moves must decrease a shared pixel-plus-gradient objective. The
gradient term gives missing fine detail more weight than whole-image RGB MSE
alone. This is a fitting surrogate, not a perceptual quality or resolution score.
All numeric work is Numba-compiled, including the small pivoted linear solve;
no NumPy/SciPy solver, FFT fallback or BLAS solve is used.

The source-only runner is `tools/v7_local_detail_source_bench.py`. Its existing-V7
column is explicitly an encoder-side fold projection, not received audio. It
compares the old fit, refined fit and optional contour prototype against the
same original and displaced-coefficient base. Full pictures and fixed eye/hair
crops include rejected candidates. It generates no audio.

The initial four-face/eight-picture refinement run at
`tmp/v7-local-detail/source-refinement/` improved the old fit but produced **zero
overall-MSE wins over existing V7**. The 48-cycle fine-band errors improved, but
the simultaneous base/detail quality floor did not pass. No new audio capacity
tests followed that failure.

The next source-only contour prototype (`tools/v7_local_edge_fit.py`) represents
the difference between a sharp and blurred edge, smoothly localized along a
tangent. Parameters are center x/y, normal angle, length, sharp/blurred widths,
contrast and curvature. The initial fitter uses straight contours (zero
curvature); it is not a finished arbitrary-contour codec. It fits only missing
edge detail, not independent tile brightness. Its wire mapping is **not installed**.

Source budget experiments reserve the same conservative 32-coordinate footprint
per physical atom plus three markers, and explicitly remove those conventional
coefficients from every candidate base. They compare 2 / 4 / 8 atoms, reserving
67 / 131 / 259 slots. This tests whether parameter-displacement cost is erasing
the intended benefit. Equal source-only footprints are not a completed edge
transport inverse or evidence that the edge parameters pass the audio gate.

```bash
.venv/bin/python tools/v7_local_detail_source_bench.py \
  --out tmp/v7-local-detail/source-example --faces 4 --atoms 4 --edges
```

Nine focused tests pass, including continuous recovery of overlapping off-grid
atoms, missing-edge recovery and absence of flat-field artifacts. Completed
aggregate outputs must decide source usefulness; analytic test success alone
does not qualify natural-picture improvements.

### Completed source-only budget screen

`tmp/v7-local-detail/source-budget-screen/` completed all 24 source cases:
four test faces, both profiles and all three reserved-slot budgets. No audio was
generated. GPT-6 Luna (low) reviewed the completed aggregate and pictures.

- Continuous refinement improved the old fit, but the 8-atom version produced
  no overall-MSE wins over existing V7. Increasing the parameter budget displaced
  more useful conventional information than the atoms could restore.
- Two refined atoms / 67 slots gave one stereo picture an overall-MSE win.
  Mean stereo MSE was effectively unchanged (+0.18%), with improved 48-band
  error but worse 24-band error. This is not a clean general detail gain.
- The two-edge mono prototype improved all three measured fine bands in three
  of four cases, but produced no overall-MSE wins. Mean overall error increased
  about 17.5%. Visible contour/background artifacts prevent treating its band
  improvements as a useful-picture win.
- Four- and eight-atom edge models also failed the overall source-quality floor.
  The edge prototype remains unmapped; its fit cost was about 0.46 seconds for
  two atoms and scales upward. These are prototype costs, not a transport limit.

The source-quality floor remains unmet. These candidates are rejected before
new audio capacity scoring. The next design must retain more of the conventional
base and describe useful detail with materially less overhead; more fitting
iterations alone do not solve the observed source-representation tradeoff.
Nine local-detail tests and sixteen geometry/integration/lazy-import checks
passed after this work (25 total). `git diff --check` passed.

## Implemented pilot

Eight overlapping Gaussian-windowed directional atoms reconstruct directly on
the requested output raster. Physical parameters are x/y frequencies, cosine
and sine amplitudes, center x/y and width x/y. Each atom uses 32 existing luma
coordinates: a three-copy coarse coordinate and one fine analog offset for
each of the eight physical parameters. Amplitudes retain signed-square-root
companding before coarse/fine mapping. Centers use a 1/128-image coordinate scale;
widths use 1/256-image units. Fine offsets preserve continuous values. All repeated
coordinates occupy independent blocks. Three separate-block branch markers bring
the total to **259 ordinary slots**.
Conventional coefficients outside those positions remain conventional. Fallback
retains the 256 parameter-slot coefficients and pays the three-marker cost.

The fitter searches the most energetic overlapping supports and frequency
candidates on each frame's residual, then jointly refits quadrature amplitudes
with box-constrained coordinate descent. All of that runs in Numba, including
an explicit radix-two spectral search and separable atom synthesis. The analysis
filter uses a native mixed-radix FFT, not `numpy.fft` or `scipy.fft`.
It uses no per-tile brightness offsets or temporal state. The prototype searches
a discrete center/width set; the wire accepts continuous centers and widths.
Its search quality and timing are experimental, not guaranteed realtime. Immutable
dictionary windows may be cached; source fits and decoded images may not.
The localized parameter range is three times the historical global-wave range to
improve conditioning. Emitted RMS/peak must be measured and matched-power
confirmation is still required before crediting a capacity gain.

Run from the repository root (new output directory required):

```bash
.venv/bin/python tools/v7_local_detail_bench.py \
  --out tmp/v7-local-detail/example --pairs 1 --faces 2
```

For a larger absolute-frequency screen:

```bash
.venv/bin/python tools/v7_local_detail_bench.py \
  --out tmp/v7-local-detail/screen --pairs 4 --faces 12 \
  --families edges local-texture random-texture \
  --frequencies 24 36 48 64 80 --loads 4 8 16 --directions x y mixed
```

The runner first qualifies known overlapping atoms through volume-controlled
audio. It screens source reconstructions before transmitting fitted models, then
checks each accepted fitted model before enabling eligible pair scores. Sources are
generated independently of the Gaussian dictionary using compact raised-cosine
supports or random fine-band texture. All output images include both forced
diagnostics and actual adaptive output. Before-audio images are encoder-side
fold projections, not decoded audio. Unavailable outputs are labeled; the tool
does not synthesize black frames. Exit status 2 means decode qualification failed.

`SUMMARY.json`, `MODEL-DECODE-VALIDATION.json`, `paired-recovery.json`, source/code
hashes, audio RMS/peak, timing measurements, comparison PNGs and `REPORT.md` are
saved. Decode tests report absolute error, unfloored relative error and whether
the existing 0.01 RMS floor assisted a pass. Only the harness uses the 10% gate;
the renderer tolerates analog detail loss without applying this rejection rule.

The random-texture family fills the selected frequency band, so its independent
content varies with bandwidth and raster size, not the contour count. The runner
does not repeat identical random textures under different `--loads` labels.

### Build evidence

Twenty-seven focused local-detail, geometry, fold, integration and lazy-import
tests passed. Tests include independently constructed audio receivers, analytic
atom evaluation, explicit FFT checks against direct analytic sums, filtering of
known spatial frequencies, rejection of absent/wrong/inverted detail, bounded
source fitting and unchanged production audio when the experiment is disabled.

Completed `tmp/v7-local-detail/coarse-fine-numba/`: 14/14 model-decode gates
passed for both profiles, two fine-detail families and two test faces per profile.
The independent-detail pilot pairs failed the recovery criterion in every path;
no resolution gain is claimed. Mean full encode times were approximately 181 ms
per profile, exceeding the 81.67 ms interval. Earlier runs are retained as
failed conditioning evidence, not capacity results. Expanded final pilot with
edges and x/y/mixed directions: `tmp/v7-local-detail/final-pilot/`.

That expanded run completed with **42/42 decode checks passing**. Worst actual
relative detail-reconstruction error was 0.00296 (0.296%), below 10% without
requiring the weak-detail floor. All 40 source pictures were available. There
was one independent pair in each of nine family/direction conditions per profile.
Localized texture passed in x and y, with approximately 0.90 / 0.88 correlation
and 0.44 / 0.48 relative source-detail error; existing V7 passed neither. Mixed
localized texture, edges and random texture did not pass. These are pilot leads,
not held-out reliability or 1.5x resolution results.

Mean forced emission RMS was 12.3% above existing stereo and 16.1% above existing
mono; matched-power confirmation is absent. Full encoding averaged 180 / 190 ms
and every timed frame missed the 81.67 ms interval. Those unresolved power and
cadence floors prevent overall acceptance even for the favorable texture cases.

Luna's final code audit found a frequency-domain guard gap; explicit +/-96
encoder checks and bounded receiver coordinates were added. Residual formation,
source scaling and source-model error scoring were also given explicit Numba
kernels. Twenty-seven focused tests passed again. Confirmation of those last
changes is `tmp/v7-local-detail/verified-build/`.

The verified build completed successfully: **42/42 checks**, the same x/y
localized-texture pilot passes, and 27 passing focused tests. Worst normalized
parameter error was 0.0000889; worst floor-normalized detail RMS error 0.00295.
Mean encode times were 203.1 ms stereo / 185.5 ms mono; all 40 timed frames missed
the interval. Mean emission RMS ratios were 1.1240 / 1.1625. The build and harness
are validated for this initial 3:4 clean-wire screen, while broader layout
qualification, matched-power confirmation, natural-source gains, calibration and
the full held-out schedule remain required before any overall acceptance claim.

The quantity of interest is recovered independently variable picture structure,
not output dimensions, transmitted parameter count, image sharpening, or a
Shannon-capacity estimate. A structured-pattern win cannot establish an arbitrary
texture-capacity win. Never label a local coefficient ratio a whole-picture
resolution ratio.

## Fixed resources and reconstruction

- Existing grids, carrier positions, component ratios, fold hosts/guests,
  signatures, metadata, EOF, sample rate and packet duration remain fixed.
- Test stereo M/S and actual mono separately; do not treat one M/S leg as mono.
- Each scored output comes from one packet and a new receiver. No decoder gets
  source images, source seeds, fitted encoder state, test labels, or temporal
  picture information. Single-image looping cannot average measurements.
- Charge every adaptive atom position, direction, scale, amplitude, phase,
  selector and protective copy to existing picture-data slots. Record which
  conventional coefficients are displaced and how their absence affects the
  base. No unannounced source-dependent fields or compression bitstream.
- Prototype baseline: a 192x160 luma reconstruction canvas. It is not the
  scored reference raster. For a frequency target above its Nyquist limit,
  enlarge the internal canvas before testing (for example to 288x240), and
  explicitly record the computational cost. Do not silently cap a 3x probe.
- Evaluate at a fixed source-aspect raster with 768-pixel short side, identical
  for both decoders. Choose analysis and output sampling with margin above the
  highest tested frequency. Increasing raster size earns no resolution credit.
- Reconstruct the base by the existing agreed interpolation path. Evaluate
  received detail atoms directly on the high-resolution canvas. Smooth atom
  support is required; independent tile means/hard tile cuts are excluded.
- Freeze the atom definitions, carrier allocation, conditioning, priors and
  selection rule before confirmation. Source fitting may vary per frame;
  receiver definitions and capacity may not adapt using held-out truth.

## Stage A: decode qualification, before capacity scoring

1. Analytic atom check: independent reference evaluation versus reconstruction,
   including noninteger position/frequency, rotated atoms, mixed signs, small
   amplitudes, intersecting supports, and atoms crossing former tile boundaries.
2. Parameter-coordinate roundtrip: error <=1e-9, including every selector and
   coarse/fine boundary. Reject out-of-domain fits rather than silently clipping.
3. Through actual V7 audio, force the new representation even when adaptation
   would choose conventional encoding. Reconstruct with a separately created
   receiver. Require a picture, the intended branch, normalized parameter error
   <=0.003, and detail reconstruction RMS error <=0.1 times intended-detail RMS,
   using the existing 0.01 RMS floor for weak corrections. Use explicit fixed
   normalizing ranges for new atom parameters; retain these error thresholds.
4. Retain production and transformed 1-to-1 fidelity controls. Audio and decoded
   differences must be <=1e-9 where equivalence is intended. This requirement
   does not falsely demand exact equivalence after changing decoder priors.
5. Qualify both profiles and every selected layout (3:4, 4:3, 16:9). Include
   different counters and isolated frame entry. Show model-before/audio-after
   images even for rejected models. Qualification has no capacity-gain score.

Stop on a mapping or decode failure. Save diagnostic images and gate errors.
Do not loosen thresholds after observing results. For weak details, also report
absolute error and the reference floor so a floor-assisted pass is transparent.

## Stage B: bracket the existing V7 detail boundary

Measure existing V7 at both contrasts (0.05 and 0.3), both spatial axes and all
three layouts/profiles using known-frequency probes. Use eight phases, not one
fortunate alignment. Include diagonal and off-center structures separately.

Legacy resolution criteria are retained: contrast transfer >=0.2, phase error
<=pi/8, off-pattern RMS residual <=0.25 of source RMS. At least 75% of phases
must pass at every sampled frequency from the low-frequency starting point to
the reliable cutoff. Show the contrast/error curves alongside the pass flag.

Refine a boundary to a <=2% frequency bracket. Let L0 be the highest continuously
passing frequency and U0 the next failing frequency. Use conservative gain
lower bounds L_candidate / U0, not just ratios of two sparse passing samples.
A ceiling-limited candidate cutoff is right-censored; an unresolved baseline
cannot produce an infinite gain. Pure sinusoids are calibration and a favorable
family result, not the decisive information-recovery test.

## Stage C: paired, independently variable fine detail

Construct sources I_A=B+D_A and I_B=B+D_B. B is the same coarse picture and color;
D_A and D_B are independently chosen luma-only details with zero DC and energy
above the calibrated base band. Generate them from native high-resolution truth,
not an enlarged low-resolution picture. Constrain both sources to valid gamut
without clipping. Measure coarse-band leakage after source rasterization instead
of assuming ideal orthogonality survives the encoder's nonlinear processing.

Each source is independently encoded and decoded. Pairing occurs only in the
analysis: the receiver cannot combine packets. Compare decoded difference
(output_A-output_B) with true difference (D_A-D_B). Shared base content cannot
receive credit for predicting the independent detail.

Use four distinct families:

1. Smooth short contours and edge bundles, with independently chosen positions,
   orientations, widths and strengths. Generate using a different construction
   from the candidate atom dictionary to avoid a dictionary-matched advantage.
2. Localized mixed-frequency texture patches, with independently chosen phase,
   direction and amplitude. Include multiple overlapping structures.
3. Random edge/patch arrangements across the frame, including support boundaries,
   corners and crossings; no regular tile layout is provided to the receiver.
4. Band-limited random texture in the added band. This tests generic independent
   detail. Failure here narrows a structured-detail claim rather than proving
   all analog source compression impossible.

For each family, test horizontal, vertical and mixed/diagonal variation, both
contrasts, and independent structure loads of 4, 8, 16 and 32. Probe added bands
near 1.0, 1.5, 2.0 and 3.0 times the conservative baseline boundary, subject to
the sampling margin. Do not equate structure count with linear resolution.

Paired recovery measures, computed in the target fine band:

- Correlation rho = dot(truth, decoded)/(norm(truth)*norm(decoded)).
- Signed transfer g = dot(truth, decoded)/dot(truth, truth).
- Unexplained error e = norm(decoded-g*truth)/norm(truth).
- Absolute normalized reconstruction error norm(decoded-truth)/norm(truth).
- Recovery of individual known features and off-band/false-detail energy.

Provisional recovery criterion: rho >=0.8, 0.5 <=g<=1.5, e<=0.5. Freeze it before
screening and publish every component, not only an aggregate pass. Verify its
calibration with unchanged truth (pass), absent detail, wrong random detail,
phase inversion and artificial sharpened/aliased images (fail). These numeric
screening thresholds are not claims of perceptual losslessness.

Start with four independent source pairs per condition in screening. Confirm
only frozen promising configurations with twelve NEW pairs per condition;
require at least 11/12 passing at each claimed load, band, direction and contrast.
Report pair-level counts and uncertainty. These are modest test-set reliability
criteria, not population guarantees. Report structured and random-texture
results independently. A broad claim needs both axes and all required families;
a directional or model-friendly-only gain must be labeled narrowly.

## Stage D: real-source confirmation and quality floors

Use 12 test-partition faces from the first face folder, selected before results,
and five fixed entry points in each existing test movie. Movie fixtures are
additional stress sources, not automatically natural-image evidence. Optionally
add real-source same-base pairs by independently replacing only fine bands,
without using their originals at the receiver.

Show Original / Existing V7 / Experimental BEFORE audio / Experimental AFTER
audio for every entry, plus 1:1 eye/hair/edge crops. Show both final adaptive
output and any forced rejected model in separately labeled diagnostics.
Measure source-relative fine-band correlation and reconstruction error, coarse
structure, color, ringing and delivery. Natural SSIM alone is not a resolution
or information-capacity measure. No failed decode is replaced by a black image.

Provisional quality-loss screening limits: mean SSIM decrease <=0.01, mean
delta-E76 increase <=1, and edge-overshoot increase <=0.05 of edge-step contrast.
Also show per-image losses and feature displacement; means cannot hide a badly
damaged picture. These thresholds require explicit agreement before a final
acceptance claim; they operationalize a screen, not perceptual-lossless proof.

## Budget, cadence and reporting

Use matched sample count, receiver acquisition, frame headers and emission
limits. Report actual RMS, peak and occupied slot counts. A capacity lead that
uses a materially different power budget needs matched-power confirmation before
it counts. Include a matched-power control and verify model decode again at that
level. Uniform waveform attenuation is an explicit volume control applied before
decoding. Do not correct image gain after decoding or use louder emission for a gain.

Benchmark warmed Numba source fitting, encode, decode and final high-resolution
render separately; also report cold setup. No cached fits or decoded pictures
may bypass measured work. Isolate timing from tests and other benchmarks.
Report distributions and misses against the actual packet interval. A clean
detail lead without preserved cadence is not overall acceptance. Check static
loops and isolated movie entry without temporal fusion.

Before confirmation, save code/model hashes, source hashes, seeds and schedules
in manifests. Seeds are generator-side evidence, never receiver inputs. Read
only completed computed summaries/charts, not streamed trial logs. Keep separate
results for source-model fidelity, transport fidelity, effective detail gain,
natural quality and processing cost; do not let success in one substitute for
failure in another. No added-noise optimization until a clean-wire lead is
established. Clean synthetic waveform results do not validate tape.

Required artifacts: decode gates; frequency-boundary brackets; independent-detail
passing curves versus frequency AND structure load; paired gain/loss tables;
all comparison images and crops; RMS/peak tables; isolated timing distributions;
and a concise report stating exactly which families/axes pass or fail.
