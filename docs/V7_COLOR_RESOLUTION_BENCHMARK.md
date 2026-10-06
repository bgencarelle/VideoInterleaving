# V7 color-transform and resolution benchmark design

The next experiment is governed by `V7_ANALOG_FRAME_INDEPENDENT.md`. This
document describes the earlier constrained implementation and its artifacts;
its fixed-grid scores must not eliminate color families from the new tests.

## Objective and status

The runner is implemented in `tools/v7_color_resolution_bench.py`. The design
below defines its experiments; an implementation check is not evidence of a
resolution improvement.

Find a source-coding configuration that recovers more genuine spatial detail
than the current GUI defaults through the same V7 audio wire. Every experiment,
including tuning and clean references, must encode audio packets and run the
receiver. Do not rank color round trips, ideal coefficient reconstructions, or
image-only kernel outputs.

Sharpness means narrower faithful edges. Increased resolution means correctly
recovered finer source structures, rather than larger output images, stronger
halos, amplified noise, or invented/aliased patterns.

## Baselines and fixed conditions

Run two independent competitions:

* `aspect-fold-500` stereo with the GUI's `viewer_solve` kernel.
* `aspect-mono-500` mono with the GUI's `viewer_solve` kernel.

Record the resolved GUI defaults and receiver settings in each run manifest,
including kernel parameters, direct-DCT settings, aspect/tail choices, speed,
target level, output/capture rates, display reconstruction and upscaler. Resolve
defaults from code at run time rather than copying possibly stale numbers.
Use factory defaults, not a user's saved preferences, for the primary baseline.

For each competition preserve packet duration, slot count, channel count,
framing, pilot/metadata overhead, playback speed and emission/level constraints.
Use the production EOF and edge-counted acquisition path. Do not substitute
known packet boundaries or transmitted coefficients for received results.
Run audio in memory; a physical audio device is unnecessary.

Use matching experimental sender/receiver models for each representation.
Rebuild coefficient statistics, transmit gains and fold tables as needed.
Any candidate-specific configuration is frozen and shared before the run;
per-frame transform/allocation information must fit existing metadata capacity,
or its overhead must be charged explicitly. Do not silently grant extra slots.

Log emitted body RMS/peak, header level and body ceiling activations. Follow the
same production level policy for all candidates; differences in real emitted
power must be visible, not hidden by candidate-specific normalization.

## Repository corpus

### Face images

Use only the first face folder:

`images_sbs/face/00_C_BG_faceSource_960/benFaceSource0000.jpg` through
`benFaceSource2220.jpg` (2,221 images in this checkout).

These are 1440x960 SBS assets. Interpret them using the application's
`modem_image_source.py` semantics: the left 720x960 half is RGB and the red
channel of the right half is alpha. Composite on the existing `(4, 4, 4)`
background. Preserve native content resolution until the candidate's source
projection stage. No float overlay is added in this experiment.

Use contiguous partitions to reduce leakage between neighbouring frames:

| Purpose | Image indices |
|---|---|
| Model fitting / PCA training | 0..599 |
| Guard interval, unused | 600..699 |
| Allocation/kernel validation | 700..1299 |
| Guard interval, unused | 1300..1399 |
| Held-out evaluation | 1400..2220 |

Freeze these partitions before tuning. For validation static tests choose 24
indices rounded from an evenly spaced inclusive sequence over 700..1299.
For held-out static tests do the same over 1400..2220. Publish exact indices.

Static exposure: repeat each selected image for 24 packets with a fresh receiver
state. Score packets 0..6 as acquisition/settling and 7..23 as steady state;
report both, including failures. Keep counters continuous within an exposure.
Provide enough trailing real packets to flush receiver lookahead, but do not
score the flush packets.

Motion exposure: play indices 700..1299 for validation and 1400..2220 for final
evaluation at one new source image per packet. Also run at half that source
rate (each image held for two packets). Use full contiguous streams, without
resetting tail memory between images. Report the initial seven packets
separately. Do not tune on the held-out sequence.

Freeze source-coordinate ROIs covering eyes, hair, mouth and background detail
using validation images before final scoring. Use identical ROIs across
candidates; exclude fully transparent background from natural-detail scores.

### Existing movies

| Asset in `modem_tests/fixtures/` | Geometry | Duration | Role |
|---|---|---|---|
| `v7_pixel_motion_4x3.mp4` | 640x480 | 10 s | Validation, landscape calibration |
| `v7_pixel_motion_5x6.mp4` | 400x480 | 10 s | Validation, alternate aspect |
| `v7_pixel_motion_16x9.mp4` | 854x480 | 10 s | Held-out wide resolution/motion |
| `v7_pixel_motion_3x4.mp4` | 480x640 | 10 s | Held-out portrait resolution/motion |
| `v7_robot_count_sync_test.mp4` | 640x480 | 30 s | Held-out moving graphics/text and frame identity |

All five are 30 fps in this checkout. Decode the existing movie files once into
a deterministic RGB frame/PTS cache. These decoded frames are the source truth;
do not compare against pristine generator frames that omit H.264 artifacts.
Source caching is preparation, not a scored experiment.

At each packet's source sampling timestamp select the latest movie frame whose
PTS is at or before that time, using the actual configured packet cadence.
Use identical timestamps for all candidates. Preserve aspect using production
layout/letterboxing rules. Score only the active picture area. The 5:6 fixture
must use the production-supported layout mapping, not an invented layout.

Use target boxes from `tools/generate_v7_pixel_motion_test.py`: fine bars,
checkerboards, edges, diagonal, text, ramps and identity strip. Map their
coordinates through the actual aspect transform; do not assume its 80x96
coordinate system is the transmitted coefficient grid for every profile.

Freeze all tuning before running held-out movies. They share target designs,
so their main independent value is unseen aspect/geometry, not unseen content.

## Candidate catalog

Include the following, with a published forward/inverse definition, input color
encoding, scaling and quantization convention for each:

* Production Pillow YCbCr baseline; explicit BT.601 and BT.709 YCbCr variants.
* RGB.
* YCoCg and integer reversible YCoCg-R.
* Orthogonal opponent RGB (normalized brightness, red-green, yellow-blue axes).
* JPEG 2000 integer reversible color transform (RCT).
* PCA/KLT trained solely on the training partition, without automatic whitening.
* CIE XYZ, CIE Lab and OKLab with explicit white point/transfer functions.
* ICtCp with explicit SDR-to-working-domain mapping and inverse.
* HSV and HSL with documented hue representation. Plain hue and circular
  encodings are distinct variants; any extra component consumes payload budget.
* Grayscale as a separate color-sacrificing upper-reference, not a color winner.

Test gamma-coded and linear-light variants of applicable linear transforms as
explicit variants. PCA uses bounded, deterministic pixel sampling across the
training images, with fixed component ordering and eigenvector signs. Publish
its matrix and mean. A color PCA transform is not a spatial DCT replacement.

## Kernel and allocation experiments

For every candidate evaluate:

1. Current plane geometry/coefficient allocation, with a compatible adaptation
   of `viewer_solve` and refitted statistics/fold tables.
2. Tuned allocation plus tuned kernel under the same wire capacity.

Retune YCbCr too: every alternative must beat both today's default and a YCbCr
candidate given the same optimization budget. Otherwise a gain may just be
better allocation/kernel tuning.

Kernel shortlist: `viewer_solve`, `native_source`, `csf_peak`, `band_taper`,
`antiring`. Adapt brightness-dependent operations explicitly for each transform;
do not label PCA component zero as perceptual luminance without justification.
Keep parameters within declared kernel ranges. Freeze a finite parameter list
and an equal number of wire-backed validation evaluations per candidate.

For tuned allocation enumerate discrete plane/coefficient splits and fold
supports, preserving all reserved slots. Explicitly include reallocating some
color capacity to higher spatial frequencies. Publish retained coordinates,
fold guests/hosts and power gains. Refit the model for each allocation.

Use successive screening: all transforms in clean and `hiss-40` validation;
then the best six color candidates get kernel/allocation tuning; then the best
three get the full held-out channel matrix. Report every screened result and
reason for elimination. Grayscale is reported separately. All stages use audio.

## Channel matrix

Use existing `tools/v7_torture_matrix.py` impairment machinery with explicit
settings frozen into the manifest. Primary final cases:

* `clean-96k`.
* `hiss-45`, `hiss-40`, `hiss-35`.
* `lowpass-12k`, `lowpass-10k`.
* `wow-flutter`, `fast-flutter`.
* `soft-saturation`, `dropouts`.
* Stereo: `azimuth-12us`, `crosstalk-10pct`, `right-minus-4db`.

Use five fixed seeds for stochastic cases, shared between all candidates;
apply impairments continuously across each audio stream, not independently to
each packet. Run every case applicable to each profile. Synthetic results are
controlled comparisons, not tape validation.

## Resolution scoring on received pictures

Maintain a presentation timeline. Compare the displayed image, including
last-good holds and partial decodes, against the intended source at that time.
Also report decoded-only quality separately; never exclude failures silently.
Use internal source timestamps/metadata for alignment, not target-strip OCR.

Primary target measurements:

* Horizontal/vertical bar contrast and phase at each existing pitch (8, 6, 4,
  3, 2 source-grid samples/cycle). A resolved group must preserve frequency and
  polarity, have at least 20% source-relative fundamental contrast and at most
  25% normalized fit residual against the actual decoded source template.
  These are proposed benchmark thresholds, frozen before candidate tuning.
* Report finest passing pitch and contrast curves in cycles per picture width
  and height; also retain raw fits so threshold choices are auditable.
* Edge 10%-90% width, overshoot/undershoot and adjacent-region ripple. Do not
  claim slanted-edge MTF from targets that do not meet its sampling assumptions.
* Checker/diagonal pattern fidelity, wrong-frequency energy and motion-phase
  stability: reject apparent detail caused by aliasing or ringing.
* Natural-image multiscale band error and source/output band correlation in
  frozen ROIs. High-frequency output energy alone is not a sharpness score.
* Small-text template fidelity by text size; optional OCR is secondary and
  uses one frozen recognizer/configuration for every candidate.

Score at two locations, both after wire decode: the reconstruction before
viewer enhancement and the actual GUI-equivalent display reconstruction.
Use identical viewer settings and a common aspect-preserving evaluation raster
per asset. Perform any measurement resampling with one frozen method. Report
active-image dimensions so output resizing cannot masquerade as extra detail.

Secondary constraints: perceptual color error, clipping, global SSIM/PSNR,
temporal error relative to source motion, packets validated, pictures shown,
hold durations, post-dropout recovery, warmed encode/decode p50/p95 and deadline
misses. Capture partially decoded frames as-is and hold on undecodable frames,
following receiver policy.

## Acceptance and artifacts

Report stereo and mono separately. Use paired per-clip/per-image comparisons;
bootstrap whole clips/image blocks rather than treating adjacent frames as
independent samples. Show medians, worst cases and paired uncertainty.

A proposed resolution winner must improve at least one resolved target pitch
or faithfully recovered high-band contrast, agree with natural-detail evidence,
and avoid systematic aliasing, ringing or motion instability. Report trades in
color and runtime explicitly. Do not reduce all outcomes to one weighted score.

Apply the transport spec's stability requirement: packet/picture stability
within 5% of the baseline or better in every situation, with no clean-link
picture skipped between the first two and last. Compare this to both default
and equally tuned YCbCr baselines. Only held-out results support a final claim.

Runner: `tools/v7_color_resolution_bench.py`, supported by
`tools/v7_color_transforms.py`, `tools/v7_color_wire.py`,
`tools/v7_color_corpus.py`, and `tools/v7_color_metrics.py`. It reuses production
kernel, packet, receiver and impairment paths. The runner has `screen`, `tune`,
and `evaluate` stages with frozen manifests.

Write artifacts under `tmp/v7-color-resolution/<run-id>/`: asset hashes and
split/source-PTS manifests, configuration/model identities, seeds, per-frame
CSV/JSON, aggregate tables, resolution curves, representative audio WAVs,
source/baseline/candidate crops and comparison movies. Include failures and
worst cases, not just attractive examples. Record code revision and worktree
diff identity for reproducibility in a modified checkout.

## Running the implemented suite

Run from the repository root using the repository Python. FFmpeg/ffprobe are
used to decode the existing movies and optionally write comparison movies.

```bash
.venv/bin/python tools/v7_color_resolution_bench.py screen \
  --out tmp/v7-color-resolution/screen

.venv/bin/python tools/v7_color_resolution_bench.py tune \
  --from-run tmp/v7-color-resolution/screen \
  --out tmp/v7-color-resolution/tune

.venv/bin/python tools/v7_color_resolution_bench.py evaluate \
  --from-run tmp/v7-color-resolution/tune \
  --out tmp/v7-color-resolution/evaluate --videos --audio
```

Screen covers all 22 transform variants plus the refitted YCbCr control on both
profiles. Tuning keeps up to six fitted candidate families per profile and
always includes refitted YCbCr and grayscale. It evaluates five kernels and
four coefficient splits (1920/480/480, 2160/360/360, 2400/240/240,
2848/16/16), plus two viewer-solve regularization variants. Evaluation keeps
up to three families plus the controls, using frozen training statistics and
transform matrices. Models for held-out aspect layouts are constructed from
those frozen statistics, without fitting held-out pixels.

Completed stages include `GAIN-LOSS.png` plus `gain-loss.csv/json`. The chart
shows median paired changes from the current GUI-default YCbCr wire: resolution
in recovered cycles per picture, fine-bar and hair-detail contrast, pictures
shown, edge halo, and color error. Green means improvement and red means
regression throughout; lower halo/color error is favorable. The CSV/JSON retain matched counts and
full configuration names when a cell is blank or a chart row is abbreviated.

By default 64 deterministic images sampled across indices 0..599 fit the
statistics/PCA; use `--training-images 600` for the complete training partition.
Both choices use training pixels only. The fitted statistics use the declared
unwindowed source analysis; kernels do not independently refit on test frames.

For a quick implementation check:

```bash
.venv/bin/python tools/v7_color_resolution_bench.py screen \
  --transforms pillow-ycbcr ycocg pca \
  --assets stills movies --stills 1 --training-images 4 \
  --max-packets 8 --short-side 120 --cases clean-96k \
  --out tmp/v7-color-resolution/quick-check --fail-fast
```

Truncated exposures, reduced still counts/raster sizes, incomplete asset
coverage, or a restricted final channel/seed matrix mark a run as limited.
That flag propagates to later stages. A tuning stage must use the same dataset
options as its screening stage. Quality measurements from a limited run must
not be presented as final resolution evidence.

Long runs checkpoint completed trials. To resume, repeat the original command
and options, replacing `--out PATH` with `--resume PATH`. Completed trials are
skipped. The implementation/dependency hashes and experimental settings must
match; an altered implementation requires a fresh run. Stage inputs likewise
must be complete and use matching code. Failed configurations are recorded
explicitly and prevent a run being used as a completed stage input.

### Explicit implementation conventions

* The receiver is configured in advance for the frozen experimental profile.
  It still acquires packet timing and checks status/metadata/EOF from audio.
  This suite measures wire reception and recovery, not automatic profile-switch
  streaks. No packet boundaries are supplied to the decoder.
* The production baseline follows the live sender's YCbCr preparation, luma
  repair, kernel masks, pinned model and fold. Alternative configurations use
  native-pixel color conversion, the production block/decimation/projection
  geometry, fitted statistics, and a bounded inverse-RGB-gradient brightness
  correction. No PCA component is assumed to be perceptual luminance. Plain
  HSV/HSL hue is wrapped on inverse; circular hue encodings are not included.
* Models keep 2,880 candidate coefficient positions. Mono still transmits only
  its 1,264 fresh slots; stereo keeps its 2,320 slots and fixed-tail behavior.
  The 500-host fold and 16 signature reservations are retained.
* Native and display results both come from wire decode. The display path
  uses production recommended DCT/edge/guided-color settings and a CPU model
  of the GUI's Mitchell shader. Grain and dither are off for measurement.
  Alternatives bridge received RGB into a common brightness/color display
  representation before those enhancements; this is a proposed experimental
  receiver display path, not a claim that the existing GUI understands every
  transform. GPU-rendered display validation remains a separate future check.
* Spatial face ROIs are fixed normalized boxes (not anatomical tracking).
  Color error is currently CIE Delta E 76, explicitly labeled, not Delta E 00.
* Resolution curves include fit phase/residual checks. Their PNGs and CSVs are
  saved for the movie targets. Initial unavailable frames carry no fabricated
  pixel score; availability/delivery counts accompany all quality aggregates.
  Held frames are scored against the current intended source. Missing metadata
  identities are logged rather than aligned using receiver iteration order.
* Runtime records warmed per-packet encoding and average stream decoding cost;
  per-packet decoder p95 and automatic winner acceptance are not implemented.
  `paired.json` compares with default and equally tuned YCbCr; `uncertainty.json`
  resamples whole exposures. Final selection requires inspecting these results,
  resolution curves and visual artifacts together.

End-to-end regression checks, each involving encoded V7 audio:

```bash
.venv/bin/python -m unittest modem_tests.test_v7_color_resolution_bench -v
```

### Implementation validation, 2026-10-05

The limited workflow runs under `tmp/v7-color-resolution/verified-*` completed
without configuration errors:

* Screening: 10 configurations, one static exposure, 10 wire trials.
* Tuning: 38 configurations, one static exposure, 38 wire trials.
* Held-out evaluation: eight configurations, six exposures (a held-out still,
  two consecutive-image rates, and all three held-out movies), 144 wire trials
  across clean, hiss and dropout cases. It produced 144 comparison movies and
  emitted/impaired WAV pairs. Resolution curves are written when a target
  picture is available to measure.

These checks used four training images, eight packets per exposure and a
120-pixel short-side measurement raster. They verify the workflow and artifacts,
not resolution improvements or full channel tolerance. The screening/tuning
movies were separately exercised in `workflow-screen` during development.
Completed-run resume was also checked without re-encoding completed trials.

The integration/lazy-import suite passed (11 tests). Full modem discovery ran
879 tests with two failures: untagged HD video matrix detection and the frozen
mono packet digest. Both failures reproduce in standalone invocations that do
not import the new benchmark modules.
