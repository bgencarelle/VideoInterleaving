# V7 frame-independent analog detail experiment

## Current direction: fixed aspect-fold-500 layout

The user clarified that the target is perceptually or close to perceptually
lossless analog recovery with a large signal/noise margin. The active candidate
design and implemented tests are in `V7_FIXED_FOLD_INNOVATIONS.md`.

The user has redirected the experiment toward lossless source-data compression
on the existing aspect-fold-500 grid, retaining its existing component ratios.
The grid/allocation sweep is stopped; its checkpoints remain exploratory evidence.
The variable-grid proposal below is historical, not the active experiment.

Freeze production component grids, aspect-dependent coefficient support,
component ratios, 500-host fold structure, wire slots, frame cadence, levels,
headers, metadata, pilots and EOF. Each frame remains independently decodable.
Focus subsequent work on redundancy removal before the existing fold and an
explicit matched inverse after reception, using the same face/movie corpus.

The earlier prohibition on digital image compression remains in force until
the user clarifies whether this new request changes it. Reversible analog
decorrelation or intra-frame prediction can reduce residual energy; it does not
by itself create spare coefficient slots. Any claimed additional packing must
demonstrate capacity and noise tolerance through real V7 encode/decode at the
existing levels. Do not label omitted residuals or source quantization lossless.

Clarify the losslessness boundary before selecting the new coder: preserving
the selected production picture data exactly before channel impairment is
different from preserving original RGB pixels, which the current sampling and
coefficient omission do not do. A lossless digital coder would require an
explicitly allowed finite-precision source representation and a payload mapping
experiment. No candidate may depend on previous pictures or loop accumulation.

## Historical variable-grid specification

This specification superseded the
fixed-grid candidate-screening/shortlisting strategy in
`V7_COLOR_RESOLUTION_BENCHMARK.md`. Earlier measurements remain constrained
baseline results, not evidence against a color representation.

## Charter

Improve genuinely recovered spatial detail through the existing analog-valued
V7 wire. Every frame can be the first frame received; continuously looping one
image is a normal case. Current frame rate and useful picture quality are the
minimum acceptable performance. Use no digital image codec, entropy bitstream,
inter-frame prediction, rotating picture tail, or repeated-frame averaging as
a prerequisite for picture recovery.

Keep existing headers, metadata layout, pulse acquisition, pilots, EOF,
packet duration, audio bandwidth and channel count. The experiment changes
picture data and its agreed interpretation. Sender and receiver load the same
fixed experimental configuration before transmission; there is no new on-wire
configuration message. A layout is selected through the existing aspect field.
The receiver still obtains timing/status/metadata from audio.

All 22 implemented color variants remain eligible for matched allocation tests.
Fixed-allocation scores must not eliminate a representation. Grayscale is an
explicit color-sacrificing reference, not a color-preserving winner.

## What a grid means and why it changes

A component grid is its spatial sampling/analysis lattice: `(rows, columns)`.
Its DCT has the same dimensions. A separate support list determines which
frequency coefficients are actually sent. Grid size is not the slot count and
is not a claim about resolved detail. Coefficient `(u,v)` still means `u/2`
vertical and `v/2` horizontal cycles across the picture, independent of how
many samples evaluate that basis. Aspect affects the physical frequency order.

YCbCr's current 96x80 brightness / 48x40 color grids encode a brightness/color
assumption. RGB, PCA and other components need not obey it. Initial finite grid
families therefore are:

* `legacy`: 96x80, 48x40, 48x40 (production geometry control).
* `equal`: three 96x80 grids (avoid pre-discarding other components' detail).
* `expanded`: three 144x120 grids (provide a larger analysis/reconstruction
  lattice and room to test new supports; no extra audio slots).

Use native source pixels, with source conversion before averaging for nonlinear
transforms. Preserve aspect through existing picture-layout behavior. Fit
statistics separately per transform/grid family using training pixels only.
Decoder placement, inverse-transform amplitude and interpolation must follow
the same grids/support as the sender. Never reduce candidate color back to the
old 48x40 display planes merely for compatibility.

## Matched source coder

Each configuration freezes color conversion and inverse, plane grids, support,
power gains, fold mapping/scales, kernel and physical-luminance repair. It
retains the existing stereo/mono slot budgets and 500-host analog fold in the
initial experiment. Fixed stereo tail slots carry current-frame data only.

Test the four existing total-2880 candidate splits: 1920/480/480,
2160/360/360, 2400/240/240 and 2848/16/16. These are coefficient-pool counts;
stereo transmits 2320 ordinary slots and mono 1264, with folded guests as usual.
Report actual carried coordinates, including reserved signature positions.

Kernels receive actual component-grid sizes, actual ordinary support bounds and
actual wire masks including usable fold guests. Test production geometry and
matched geometry explicitly. Spatial kernel application to all components is
an explicit option, rather than automatically assuming PCA component zero is
luminance. Fit display assumptions to the matched reconstruction path.

Luminance/color recovery gets its own factor:

* `off`: raw component coding, useful to identify repair-induced damage.
* `physical`: correction using inverse RGB and source linear-light luminance,
  projected into the carried component supports.
* `physical-clip`: physical repair plus bounded correction for final RGB
  clipping; preserve DC/mean exposure and remain within the analog level budget.

The production YCbCr reference must exercise the available ordinary, linear-
light-target and clip-aware luma options too. An optimized YCbCr control gets
the same tuning budget as alternatives. Existing options are not proof that
color loss is fixed: measure their effect through the wire.

## Wire-backed tests

Use the first SBS face folder and existing five movie fixtures, with the
training/validation/held-out partitions in the earlier document. Use a small,
fixed validation sample first, including multiple movie timestamps/aspects.
Every transform gets the same declared grid/allocation opportunities before
any family is shortlisted. A configuration's poor result does not eliminate
untested configurations in that family.

For each scored source:

1. Encode a normal current-frame packet and genuine lookahead/flush packet if
   the production acquisition path needs it; transmit no earlier picture.
2. Apply the channel continuously to that short audio stream.
3. Create a fresh receiver, acquire through pulses/EOF, and score the first
   picture packet by received identity. Flush pictures are not scored or used
   as previous-picture information.
4. Separately loop the same image to verify no drift, tails or startup-only
   quality advantage. Decode every member independently as well as in a
   continuous fresh-picture stream; disable temporal picture fusion.

Start with clean and `hiss-40`, shared deterministic seeds. Random-entry movie
samples include fine-pattern phase changes, natural frames and moving text.
Retain first-frame failures and partial pictures as results. Never fabricate a
black picture. Expand successful matched configurations to the full original
channel matrix and contiguous movies after the small tests establish benefit.

## Acceptance and readable results

Compare at equal cadence and level with both actual GUI default and equally
tuned YCbCr. Warm kernels/Numba before timing. Encoding and decoding must keep
up with the current packet cadence; offline processing time is not a reason to
lower the transmitted frame rate. Report per-packet deadline misses and display
delivery, not just mean CPU time.

Current useful brightness/detail/temporal quality and frame cadence are floors.
A resolution improvement may trade a small amount of color fidelity: retain
color-error curves and examples, test luma-repair settings before judging that
trade, and show the tradeoff frontier. No arbitrary color-error percentage is
treated as acceptable without a user-set tolerance. Large color loss is not
hidden in a brightness score.

Primary evidence: passing fraction at every target pitch and phase, contrast
curves, source-correlated natural high-band detail, faithful edge width,
ringing/aliasing, first-packet availability and cadence. Measure both decoded
pre-enhancement RGB and matched final display. Do not equate a larger output
raster or extra high-frequency noise with increased resolution.

Produce compact gains/losses tables and charts by stereo/mono and clean/noisy
case, with measurement counts and unavailable values. Show detail, delivery,
ringing, color error and processing time. Avoid hiding conditional gains with
one median over all cases. Save source/default/candidate crops and model hashes
under `tmp/v7-analog-frame/`. Generated artifacts stay gitignored.

## Implementation sequence

1. Add dynamic component grids and matching codecs/reconstruction in benchmark
   modules; reuse production V7 packet and receiver functions.
2. Verify exact grid/placement/amplitude contracts through real audio for all
   variants and both profiles. Each regression test includes wire transmission.
3. Add fresh-receiver/random-entry and static-loop trials, including luma factors.
4. Run the small equal-opportunity matrix, inspect failures and gains/losses.
5. Expand promising configurations without changing wire cadence or framing.

Keep shared configurations and results in files so follow-up work can reference
one specification and compact tables rather than repeat exploratory reasoning.

## Implementation and commands

`tools/v7_analog_frame.py` implements dynamic component geometry, matched
packing/unpacking, ordinary/folded support masks, and reconstruction. Fold hosts
and guests can come from any component; RGB and XYZ therefore no longer fail
merely because component zero cannot supply 500 suitable mono hosts. Component
DCs are always included. `tools/v7_analog_kernels/matched_viewer.py` provides a
grid-aware spatial gain; production `viewer_solve` remains the reference.
Its expanded-grid broadcast failure is a kernel-contract limitation, not a
failed color representation. Physical repair evaluates the carried support and
its omitted-position model means, targeting averaged linear-light luminance.

`tools/v7_analog_frame_bench.py` scores single packets, each with a fresh
receiver. It samples movie entries across their complete duration rather than
only the initial frames. It saves matched models, source hashes, actual carried
coordinates, trial errors, timing/deadline flags, emission levels, per-frame
resolution contrast/phase/pass results, paired gain/loss tables and charts.
With `--loops`, static images also undergo continuous-loop versus independent-
packet checks. No picture averaging is enabled. Single-packet acquisition works
in the tested clean paths, so the runner does not add a flush picture.

Initial compact validation (not a selection run):

```bash
.venv/bin/python -m unittest modem_tests.test_v7_analog_frame -v
.venv/bin/python tools/v7_analog_frame_bench.py \
  --out tmp/v7-analog-frame/pilot \
  --candidates ycocg rgb --grids equal expanded --splits current detail \
  --training 4 --faces 1 --movie-entries 2 --loops 3 --short-side 120
```

Equal-opportunity grid/allocation validation for every implemented transform:

```bash
.venv/bin/python tools/v7_analog_frame_bench.py \
  --out tmp/v7-analog-frame/all-family-allocation \
  --training 16 --faces 2 --movie-entries 3 --short-side 240
```

This defaults to all 22 transforms, all three grid families, all four splits,
both wire profiles, clean and hiss-40. Production ordinary/off/linear/clip luma
controls are included automatically. All configurations remain eligible after
this allocation pass; add `--repairs off physical physical-clip` for the matched
repair pass. PCA is fitted only on the training face partition. Run held-out
`--partition test --faces 12 --short-side 480` after freezing candidate settings;
the current runner supports explicit settings, not automatic winner selection.

Output directories enforce manifest/code identity on resume. Use a new directory
after code or configuration changes. `--save-audio` retains emitted/impaired WAVs.
The chart's columns use different physical units: positive means improvement
relative to the actual default, not a summed overall score. Measurement counts
and failures remain explicit in CSV/JSON. Timing from overlapping processes is
implementation evidence only; acceptance timing must run without a competing
benchmark/test process.

### Verification check-in

The focused wire suite passed five methods: all 22 transforms in both profiles,
all three grid families/four splits/three repair options in both profiles,
fresh entry at beginning/middle/end of an existing resolution movie, four-member
single-image loops compared with independent packet decoding, and production
ordinary/off/linear/clip luminance controls. The integration/lazy-import suite
also passed all 11 tests. These establish packet/geometry behavior, not better
resolution or tape performance.

The initial `tmp/v7-analog-frame/pilot` has incomplete control trials caused by
a model-output-directory bug; retain it only as debugging evidence. The fixed
runner creates that directory before any controls and uses Pillow for charts,
avoiding an optional Matplotlib dependency. Use `verified-pilot` for replacement
results. Earlier fixed-grid screening was interrupted intentionally; its saved
gain/loss chart covers only completed checkpoints.
