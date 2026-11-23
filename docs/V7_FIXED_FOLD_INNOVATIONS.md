# Fixed-layout analog fold innovation experiment

## Current gate: 1.5–3× genuinely resolved detail before noise tuning

The implemented top-down folded-surface test bed is documented in
`V7_DESCENDING_CAPACITY_BED.md`. It starts at 9→1 and works down to 1→1 on
clean actual V7 packets, carrying additional source coordinates in fixed slots.

The user's first milestone is at least a 1.5× resolution increase, with 3× as
the stretch target. Interpret this provisionally as linear spatial resolution
in both axes, not output raster size or total pixel count: 1.5× per axis is
2.25× resolved picture elements; 3× per axis is 9×. Keep these definitions
explicit in reports so an area gain cannot be presented as a linear gain.

Prioritize clean, actual V7 encode/decode and structural source-data capacity.
Defer hiss/noise sweeps, robustness tuning, and predictor-noise optimization
until a candidate establishes this clean-wire resolution gain. Retain the
real receiver's acquisition, equalization, fold quantization and reconstruction;
an ideal coefficient inverse is not a scored substitute for the wire.

Freeze the existing carrier grids, aspect layout, component allocation ratios,
500-host positions, signature count, packet cadence, audio levels and framing.
Permit new matched mathematical picture-data packing inside those fixed slots.
Source analysis may examine finer original-image structure; that does not
change the carrier grid. Higher source frequencies must be carried explicitly
or recovered by a demonstrated source-constrained representation. Do not call
decoder sharpening, generic predicted texture or a larger raster extra capacity.

The first pilot below only recoded the same 484 usable guest coefficients. It
was a signal-margin experiment, not a demonstrated 1.5–3× capacity design, and
does not satisfy this milestone. Further same-support residual/SNR tuning is
deferred. Subsequent candidates must target additional independent fine-detail
degrees of freedom or a quantitatively validated near-perceptual reduction of
the source representation.

Candidate directions for the clean-capacity stage:

The user's concrete suggestion is folding three source components into a
two-dimensional vector. Prioritize Shannon–Kotel'nikov dimension-reducing
analog mappings, particularly folded/serpentine surfaces, before another
same-dimensional linear transform. Reference: Floor and Ramstad,
https://arxiv.org/abs/0904.1538 (analog point-to-point mappings, including
dimension reduction). This is a mathematical family to adapt and test, not a
claim of an already implemented capacity gain.

A first explicit 3-to-2 construction normalizes three detail coordinates
`a,b,c` to `[0,1)`, chooses `N` strips, and forms

```
k = floor(N*c)
t = a if k is even else 1-a
u = (k+t)/N
v = b
```

The matched inverse extracts strip `k` from `u`, reverses its alternating
orientation to recover `a`, recovers `b=v`, and sets `c=(k+0.5)/N`. Endpoints
need an explicit shared convention. This transmits two analog amplitudes,
with strip location implicit in one amplitude, not a new metadata/index
stream. The deliberate clean source error in `c` is bounded by `1/(2N)`;
finite-wire precision controls the usable `N`. Ordinary within-strip errors
in the recovered `a` are amplified by `N`; boundary errors require separate
measurement. Companding and an orthogonal perceptual coordinate rotation can
precede the mapping, with the inverse applied after reception.

Apply this to same-frame fine-detail coordinates, rather than merely calling
the RGB color triple a 3D vector. Pack its two outputs into fixed existing
picture slots and account for every source mean/scale/parameter. All scored
outputs still traverse the actual wire. Exact, continuous, stable preservation
of arbitrary 3D values in 2D is not the premise: bounded perceptually negligible
projection error is.

3-to-2 gives 1.5× source-coordinate capacity, not automatically 1.5× linear
resolution. If capacity scaled directly with frequency area, the linear gain
would be sqrt(1.5), about 1.22×. Test this primitive first, then 5-to-2 or 7-to-3
surface constructions targeting at least 2.25× coordinate capacity, before
claiming the user's 1.5×-per-axis gate. A 3×-per-axis target corresponds to 9×
frequency-area capacity unless source structure reduces the required degrees
of freedom. Every factor must be demonstrated on decoded detail, not assumed
from the mapping's dimension ratio.

* Nested lattice / successive analog folds: pack multiple fine-detail residual
  coordinates into each current guest amplitude, using explicit matched inverse
  and controlled source-domain quantization. Preserve existing outer hosts and
  signaling; test finite clean-wire precision, not infinite-precision algebra.
* Multiscale lifting plus joint residual packing: exploit deterministic
  intra-frame coarse/fine structure, carrying actual fine-band corrections in
  the same slots. Prediction alone must not be credited as carried detail.
* Structured vector source coding: represent locally correlated detail in
  fewer continuous coordinates, with fixed matched reconstruction and explicit
  distortion accounting. Any adaptive parameters must occupy existing picture
  data capacity; there is no free mode/index side channel or digital bitstream.

A clean resolution claim requires source-correlated contrast, correct phase
and polarity, bounded aliasing/ringing, and passing fractions across multiple
phases/frames. Establish the default's reliable frequency limit per axis, then
test at 1.5×, 2× and 3× that limit with common thresholds. Report directional and
content-specific gains separately; an isolated passing bar is not a general
1.5× gain. Use the requested face/movie corpus for visual/detail evidence and
supplement its sparse bar pitches with denser known-frequency probes as needed.

Cadence, levels and frame independence remain fixed. Report source distortion,
color loss and delivery explicitly during this stage, but do not make noise
optimization the prerequisite for exploring clean-wire resolution capacity.
The numerical target is a test gate, not a promise that every source can be
packed at that factor through this wire.

## Initial same-support residual experiment

This implementation follows the fixed-grid
direction in `V7_ANALOG_FRAME_INDEPENDENT.md`. "Lossless" here means perceptually
or close to perceptually lossless through an inherently lossy analog channel.
No digital image codec or entropy bitstream is introduced.

## Hypothesis

Keep the established production aspect geometry, 1920/480/480 component pools,
coefficient positions, ranks, hosts/guests, signature count, fold step, companding,
packet cadence, model scale, headers, metadata, pulses, pilots and EOF. Test both
production stereo and mono aspect-fold-500 paths. The experimental changes are
the picture-data coordinates in the existing usable guest slots and their
training-fitted normalization. No additional coefficients are carried yet.

Suppose `x` is a vector of reliable ordinary current-frame coefficients and `y`
is the vector of 484 usable fold guests. Transmit innovations

```
z = (x - mean_x) / sd_x
r = y - mean_y - B z
t = r Q
```

The existing fold transmits scaled/companded `t`. After ordinary acquisition
and decoding, reconstruct with `y_hat = mean_y + B z_hat + t_hat Q.T`.
`Q` is orthogonal, and this coordinate change is exactly reversible before
fold quantization/clipping/channel damage. Mean vectors, prediction weights,
rotations and scales are frozen from training faces and shared out of band.

The gain hypothesis is energy concentration, not magically reduced slot count.
If residual scale falls, the fixed fold amplitude carries it louder relative to
noise; unscaling converts that margin into smaller source-domain errors. The
cost is predictor-reference noise and possible residual overload on unfamiliar
content. Approximate error covariance is

```
Cov(error_y) = B Cov(error_z) B.T + Q Cov(error_t) Q.T
```

when the two error sources are independent. Actual channel errors need not be
independent: measure final distortion rather than assuming the formula proves
a benefit. A large enough recovered detail signal/error margin is the practical
target. Smaller training residuals alone are not evidence of improvement.

## Candidates grounded in established mathematics

### Control: centered residual scaling (`residual-scale`)

Set `B=0`, `Q=I`; center guests on their training mean and fit residual RMS.
This isolates gains from normalization and a frozen mean from gains due to
prediction. Compare it alongside the unchanged production default and existing
off/linear/clip luminance controls.

### 1. Local spectral lifting (`local-lift`)

For each guest, ridge-fit a predictor from four nearby spatial-frequency anchors.
This is a triangular lifting transform: its inverse adds the prediction using
ordinary anchors from this same packet. It exploits local spectral covariance
without a recursive guest-to-guest noise chain. DC and chroma allocations are
unchanged. There is no per-frame mode map or prediction side channel.

### 2. Noise-penalized conditional innovations (`noise-ridge`)

Predict each guest from up to 32 ordinary low-frequency luma anchors, excluding
all folded hosts. Fit in standardized coordinates with the declared objective

```
min_B  E ||y - mean_y - B z||^2 + 0.15 ||B||_F^2
```

and bound each predictor row norm by its original guest scale. Ridge shrinkage
and the norm bound limit reference-noise amplification. The initial penalty is
a design regularizer, not a measured tape covariance. Future tuning may use
actual training-only wire noise measurements, with equal tuning opportunities
for controls. Prefer this over unregularized training-fit residual minimization.

### 3. Conditional block KLT (`ridge-klt`)

Apply the same bounded predictor, then diagonalize training residual covariance
in fixed blocks of eight guest coordinates. Orthogonal rotations concentrate
correlated residual energy into modes with individually fitted scales. Small
blocks constrain computation and prevent one rejected guest from spreading
error across the entire picture. Noise/fallback sensitivity remains a testable
cost. No signature coordinates participate in prediction or rotation.

These are new candidate compositions for this modem; no claim is made that
ridge, lifting, or KLT themselves are new mathematical inventions.

## Wire-backed evaluation and acceptance

`tools/v7_fold_innovation.py` wraps the production source and fold. Its receiver
inverts innovations before the production display path sees the values.
The predictor digest joins the existing pre-agreed signature table identity;
signature/header sizes remain fixed. The wire itself carries the current-frame
residual, rather than a decoder-only guessed detail enhancement.

`tools/v7_fold_innovation_bench.py` reuses fresh-single-packet trials and report
generation. Score the first face folder and existing resolution movies on clean
and hiss-40 channels. Keep source/hash partitions and shared channel seeds.
Save original/default/candidate pictures, bar pass/contrast/phase evidence,
color/brightness/detail distortion, level/peak/RMS and warm processing timing.

Additional evidence is `guest_signal_to_reconstruction_error_db`, computed on
the original physical guest coefficients after matched inverse. This includes
source coding, fold loss, shrinkage and channel distortion: it is not pure audio
SNR. Every scored margin is computed from a fresh wire decode. Missing pictures
remain delivery failures, not black substitutes. Static-loop checks require
continuous and independent decoding to agree without temporal fusion.

No configuration is accepted merely for reducing residual RMS. It must improve
real decoded detail while preserving cadence and useful perceptual quality;
explicitly show any color tradeoff. Hold out movie/face samples before selection.
Check fixed scale and actual emitted RMS/peaks to catch louder-wire advantages.

```bash
.venv/bin/python -m unittest modem_tests.test_v7_fold_innovation -v
.venv/bin/python tools/v7_fold_innovation_bench.py \
  --out tmp/v7-fold-innovation/pilot --training 16 \
  --faces 2 --movie-entries 2 --short-side 240 --loops 3
```

This first pass tests whether residual-energy coding beats the existing wire
before pursuing more ambitious multi-detail packing into the same guest slots.
Training-only energy reductions are diagnostic, never scored resolution gains.

## Pilot evidence

The command above completed 192 scored single-packet trials with zero errors.
Every configuration delivered 6/6 pictures in clean stereo, clean mono and
stereo hiss-40; every configuration delivered 3/6 in mono hiss-40. All 64 static
loop trials delivered all members with identical independently/continuously
decoded values. Focused wire and integration/lazy-import checks passed 13 tests.

Local prediction reduced training residual energy by 22–25%; broad ridge and
ridge/KLT reduced it by 64–67%. Despite that, every innovation candidate had a
lower mean physical-guest reconstruction margin than the current default.
Broad ridge lost 1.87 dB in clean stereo / 2.66 dB in noisy mono; KLT lost
4.35 dB in clean stereo. Some bar-pattern passes improved, but no candidate
resolved pitch-2 or established an overall perceptual-quality/resolution win.

Reports, margin charts, comparisons and per-frame evidence are under
`tmp/v7-fold-innovation/pilot/`. These findings reject the tested configurations
as replacements, not the mathematical families. The next controlled experiment
should distinguish training mismatch, reference-noise propagation, decoder
fallback, and scaling effects using actual wire measurements before adding
more predictors or expanding the corpus. Timing overlapped verification tests
and is not realtime acceptance evidence.
