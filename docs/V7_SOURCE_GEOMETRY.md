# Current-frame source geometry on the fixed V7 wire

Prospective localized-atom information-recovery test protocol:
[`V7_HIGH_RES_DETAIL_TEST.md`](V7_HIGH_RES_DETAIL_TEST.md).

## Current rewrite: coherent residuals (v2)

The current `tools/v7_geometry_wire.py` replaces the initial tiled model below.
The historical v1 source is retained in gitignored
`tmp/v7-geometry/geometry_wire_v1.py`; its completed measurements are unchanged.

The rewrite reuses `ProductionFrameWire.values()` and therefore the current
V7 direct, Numba-backed source encoder, kernel and masks. It has no changes to
carrier grids, fold-500, signatures, metadata, EOF or cadence. Eight global
Fourier atoms replace at most 80 ordinary luma coefficients, with three additional
ordinary coefficients carrying redundant conventional/geometry selection. Each
atom has ten carried coordinates: three copies each of coarse x/y cycle counts,
fine x/y subcycle offsets, and companded cosine/sine amplitudes. The decoded atom still
has four parameters (x/y frequency and both amplitudes). Frequencies are cycles
across the whole image, in coordinates
centered at the image midpoint. This avoids phase wrapping and normalization of
small direction vectors. The decoder reconstructs continuous functions across
the entire image; there are no independently fitted brightness patches or tile
joins. All 83 parameters/selection slots are accounted for.

`fit_atom` and `synthesize` use cached Numba compilation. Separable sine/cosine
factors avoid trigonometric evaluation for every raster sample. An FFT proposes
frequencies; continuous least-squares fitting refines them. FFT seeding still
uses NumPy, and cold compilation must be distinguished from warmed frame cost.
No fitted source results are cached; every frame is independently analyzed.

The encoder predicts the clean conventional support with model means at omitted
ordinary coefficients, production host rounding, guest compander saturation,
and reserved signature slots excluded. This deterministic encoder-side model
is not a substitute for measured audio decoding. Geometry is selected only
when its predicted RGB reconstruction MSE beats conventional representation by
at least 10%, including the cost of the 80 displaced coefficients. Otherwise,
the 80 ordinary coefficients are retained. The three selection coefficients
are replaced by their model means in the displayed fallback image, so fallback
has an explicit three-coefficient cost rather than being falsely described as
free. Markers and coarse-frequency copies use separate OFDM blocks and median
decisions. Only coefficients carried at every counter phase are eligible.

Carriers minimize displaced source variance; precision is defined separately
by an equal transmitted gain-domain range (`scale = reference / model.gain`).
The receiver's MMSE variance priors match the parameter distribution instead
of retaining face-coefficient priors. Rank tables, modulation gains, phase,
emission scale and original fold host/guest statistics are preserved.
Signed square-root amplitude companding allocates more precision to weak
corrections, with an exact signed-square inverse and explicit domain checks.
Emitted RMS and peak are measured, not inferred from unchanged modulation gain.

Conditioning tests include noninteger frequencies, negative vertical frequency,
both quadratures, exact parameter normalization/inverse, independent receiver
recovery and conventional fallback. The transformed no-op control still has to
agree with production audio and decoded values within 1e-9. Eighteen focused
geometry/dense/fold/integration/lazy-import tests passed after the rewrite.

```bash
.venv/bin/python tools/v7_geometry_bench.py \
  --out tmp/v7-geometry/coherent-real-images --partition test \
  --faces 12 --faces-only --skip-dense

.venv/bin/python tools/v7_geometry_bench.py \
  --out tmp/v7-geometry/coherent-dense --dense-only
```

The dense runner now preserves `phase_index` separately from measured phase
error and saves all four phase comparison images for 20 and 64 cycles. Image
labels are Original / Existing V7 / Experimental source-aware V7. Improvements
and failures must both be shown. `ENCODE-OUTCOME.json` includes warmed measured
encoding times, selection counts and deadline misses. Only completed computed
outputs are inspected, with GPT-6 Luna (low) providing read-only supervision.

Known limits remain: sparse global waves do not represent arbitrary edges or
texture; frequency errors can accumulate spatially; selection minimizes an
encoder-model MSE, not perceptual quality; redundancy is not tape validation.
Clean-wire tests must establish whether this
rewrite improves natural images, preserves the sinusoid lead, and meets the
processing floor. Do not equate successful fallback with extra resolved detail.

### Completed coherent real-image check

The 12 test-partition faces completed through both profiles: 48/48 pictures,
zero errors and loop failures, fidelity gate passed. Luna's completed-output
review found **zero geometry selections in 24 experimental frames**. Thus the
near-production picture quality is successful conventional fallback, not an
extra-detail result. Mean SSIM changes were -0.000030 stereo and -0.000063 mono;
mean color delta E increases were 0.000588 and 0.001605 respectively.

Warmed encoding means: 58.69 ms stereo / 63.91 ms mono, versus production's
17.77 / 15.57 ms. Each experimental profile had one deadline miss out of 12
against an 81.67 ms packet interval. The rewrite is much faster than the old
tiled fitter but does not yet establish the required processing floor. The
predictor is not a model of actual equalizer noise or display sharpening, and
its 10% RGB-MSE selection criterion cannot certify perceptual gain. The
geometry-selected transport branch has separate wire-backed test coverage;
this face run exercises only fallback. No noisy-channel robustness is claimed.

These completed coherent-real-image figures used the earlier direct-frequency
33-slot version, before the decode-first correction below. They are retained
as historical fallback evidence, not performance numbers for the current version.

Images, including fallback cases: `tmp/v7-geometry/coherent-real-images/comparisons/`.
Dense frequency/phase confirmation is still separate evidence.

### Mandatory decode-first procedure

The user requires validating the actual transformed model decode before capacity
testing. A stronger gate found that a direct-frequency stereo wave had roughly
11% RMS correction error, although normalized parameter errors passed the older
bound. That exposed conditioning of long-span frequency parameters rather than
a tile problem. The current coarse/fine frequency mapping was introduced to fix
that: coarse cycle counts are recovered by nearest-integer rounding, with fine
analog subcycle offsets carried separately. No entropy bitstream or extra frame
is introduced. Both coordinates consume explicit ordinary picture slots.

`tools/v7_geometry_validation.py::decode_model_gate` forces geometry even when
the adaptive encoder would fall back. Before scoring a source, the benchmark
requires an actual received picture, the intended geometry marker, normalized
parameter error <=0.003, and correction RMS error <=10% of intended correction
RMS (with 0.01 reference floor for weak corrections). Checking correction error
is essential: small parameter errors can still make a large spatial error.
Failures stop capacity scoring and save diagnostic before/after images. The
no-op production fidelity gate is retained as a separate requirement.

`tools/v7_geometry_validation.py` generates explicitly diagnostic columns:
Original / Existing V7 / Forced model BEFORE audio / Forced model AFTER V7
audio. The pre-audio column uses the deterministic noiseless fold model. The
forced branch is shown even if rejected by the adaptive encoder; those pictures
diagnose source representation and transport separately and do not assert extra
resolved detail. Parameter-map and known fractional-frequency fit checks run
first. A diagnostic run on four real test faces is stored in
`tmp/v7-geometry/decode-first/`.

Nineteen focused tests passed, including a regression where a small frequency
error passes the parameter bound but must fail the correction reconstruction
gate. The previous completed `coherent-dense` run predates this mandatory gate
and is marked accordingly in its `VALIDATION-STATUS.json`; it does not validate
the current codec or provide new general capacity conclusions.

### Luna audit repairs and completed decode confirmation

The first four-face diagnostic failed three mono model-decode gates despite
picture availability. It also lost approximately 0.08 mean SSIM before audio:
selecting the strongest picture coefficients for parameters was an expensive
allocation mistake. Low-source-variance carriers reduced pre-audio SSIM loss
to roughly 0.003, but fixed gain-domain scaling alone did not pass decoding.
Parameter-specific MMSE priors alone also failed all eight correction gates.

The final repairs were: appropriate receiver priors, signed square-root
amplitude coordinates, triple coarse-cycle coordinates, and independent-block
marker redundancy. `tmp/v7-geometry/luna-fixes-conditioned/` completed and
GPT-6 Luna (low) independently verified **8/8 forced-model decode gates passed**
at unchanged tolerances. Worst normalized parameter error: 0.000873; worst
relative correction RMS error: 0.0346 (limits 0.003 and 0.1). Exact parameter
inversion and production-control gates passed too.

Mean forced emitted RMS versus existing V7 was 0.99768 stereo and 1.00577 mono;
peak maxima were 0.7373 and 0.7444. Warmed forced encode means were 58.91 and
66.39 ms, with no deadline misses in these eight cases. These are small-sample
measurements, not general realtime certification.

The source model is still rejected adaptively on all four faces. Mean SSIM
changes after audio remain -0.00481 stereo / -0.00888 mono. Thus this confirms
that the intended model can be decoded, not that it adds natural-image detail.
Reports include all forced before/after pictures, including rejected models.
Expanded 12-face confirmation: `tmp/v7-geometry/luna-fixes-confirmation/`.

The expanded run completed: **24/24 forced model-decode gates passed** across
12 test faces and both profiles; mapping and production controls passed.
Maximum normalized parameter errors remained 0.000581 stereo / 0.000873 mono.
Mean forced emitted RMS ratios were 0.99673 stereo / 0.99927 mono. Source-model
SSIM deltas before audio were -0.00396 / -0.00730 and after audio -0.00473 /
-0.00967. All adaptive choices still rejected geometry. Twenty-one focused
tests passed, including tiny-amplitude inversion, an erroneous coarse-coordinate
copy, and explicit rejection of an out-of-domain amplitude.

The expanded run overlapped the focused test suite, so its recorded encoder
deadline misses are not isolated performance evidence. A separate warmed
72-packet measurement writes `ISOLATED-ENCODE-SUMMARY.json` in the confirmation
directory. This distinction concerns timing only, not the completed decoding
gates or source/model/audio image comparisons.

The isolated 72-packet run measured mean forced encoding at 64.96 ms stereo /
66.32 ms mono, with one deadline miss in each 36-packet group. Profiling found
native-resolution RGB resizing and unnecessary frequency iterations as costs.
The fitter now stops on convergence and passes native float32 directly to the
F-mode resizer rather than allocating float64 then recasting every plane. The
five focused geometry methods passed after those changes. Final 12-face
confirmation of that exact version: `tmp/v7-geometry/luna-fixes-final/`.

The final version completed and Luna confirmed **24/24 model gates passed**,
with both mapping and production controls passing. Worst normalized parameter
error was 0.000873 at the unchanged 0.003 bound. Final forced-source SSIM deltas
were -0.00472 stereo / -0.00967 mono, with zero adaptive geometry selections.
Mean emitted RMS ratios were 0.99655 / 0.99927. These are successful decoder
repairs, not a natural-resolution gain.

Final diagnostic mean encode times were 81.42 / 83.13 ms, with one / two misses
against the 81.67 ms interval across 12 frames per profile. This leaves the
realtime floor unresolved. The earlier isolated reference is specifically
`ISOLATED-ENCODE-SUMMARY.json` (64.96 / 66.32 ms), not the concurrent diagnostic
summary's 439 ms mono number. Timing variability must not be represented as a
proven algorithmic improvement or guaranteed realtime performance.

## Historical v1 implementation and evidence

## Purpose

Replace recursive packing of unrelated DCT coordinates with a source-aware
representation. This is a candidate implementation; a working decoder is not
evidence that the 1.5–3× spatial-resolution target has been met.

`tools/v7_geometry_wire.py` uses the production aspect/fold-500 grids, ranks,
component allocations, signatures, metadata, EOF and packet duration. It
replaces 448 carried ordinary luma coefficients outside the fold hosts with
64 tiles × seven real-valued parameters. Fold guests and the other picture
coefficients continue through the production pipeline. Parameters are
centered/scaled using the existing model statistics; no extra packet or
out-of-band per-picture parameters are available to the receiver.

Each tile carries:

1. Normal x.
2. Normal y.
3. Edge position or periodic phase.
4. Correction contrast.
5. Local brightness correction.
6. Edge width or periodic frequency.
7. Edge/periodic model selector.

An edge correction is a sharp step minus a Gaussian-softened step. A periodic
correction is a sinusoid with fitted direction, frequency, phase and contrast.
The encoder chooses the atom with lower local residual error. Both alternatives
use the same seven slots. The selector itself is carried as an analog value;
there is no entropy stream. Geometry comes from the current source picture,
not a training picture or preceding frame.

The base image uses the remaining carried support. The renderer replaces the
parameter coefficient positions with model means before rendering the base,
then evaluates the received geometry at output resolution. It never sees the
source image or source-side fit state. Existing coefficients retained outside
the parameter slots carry the remaining conventional picture information;
there is no separately coded residual for arbitrary unrepresented structures.

## Checks and acceptance

`modem_tests/test_v7_geometry_wire.py` checks both stereo M/S and mono profiles:

- A transformed coefficient roundtrip with geometry disabled agrees with
  production audio and decoded values within 1e-9.
- A new, separately constructed receiver recovers the analog parameters.
- All 448 slots are outside fold hosts, and packet length agrees with production.
- Rendering needs no source-side state.

Clean wire still has production reconstruction/equalization error. Parameter
recovery is bounded rather than asserted exact. A matching no-op control alone
does not establish parameter conditioning or picture quality.

`tools/v7_geometry_bench.py` measures faces from the first face folder and
existing movie entry frames through actual fresh V7 audio encoding/decoding.
It writes gain/loss charts and source/production/geometry comparison images.
Static faces additionally exercise continuous and fresh-state single-image
loops. Dense sinusoids test 11 frequencies from 8 to 64 cycles per picture,
four phases, two contrasts, both axes, three layouts and both wire profiles.
The shared measurement rejects incorrect phase and off-pattern artifacts;
cutoffs require a continuous passing band, not isolated high-frequency wins.

The encoder does not cache fitted geometry. Encoding measurements include a
new fit even after kernel warmup. Useful natural picture quality and cadence
remain acceptance floors; pattern-specific gains cannot compensate for losing
them. Synthetic clean-wire evidence is not tape validation. No additional
noise tuning is justified until the clean-resolution milestone is established.

```bash
.venv/bin/python -m unittest modem_tests.test_v7_geometry_wire \
  modem_tests.test_v7_fold_dense_probe modem_tests.test_v7_fold_surface \
  tests.test_modem_integration tests.test_lazy_imports -v

.venv/bin/python tools/v7_geometry_bench.py \
  --out tmp/v7-geometry/pilot-v2 --faces 2 --movie-entries 3
```

Initial wire/control checks: 17 tests passed. The first benchmark launch stopped
before scoring because its model-output directory had not been created. That
output setup was fixed; the replacement run uses a separate artifact directory.

## Known design limits to measure

- One atom per tile cannot represent multiple crossing edges or arbitrary hair.
- Nonoverlapping tiles can produce seams when local models disagree.
- Edge width, frequency and phase are sensitive to analog parameter errors.
- The initial source analysis uses a 256×256 raster; this limits fitting detail.
- Fitting against carried support approximates the eventual decoded base; fold
  quantization and equalizer distortion can change that base.
- Replacing conventional coefficients spends real picture capacity. This is
  not a free sharpening stage, and lost natural detail must appear in reports.
- Per-tile selector transitions can change atom families after channel damage.

Completion evidence belongs below after the computed reports are available.

## Completed pilot-v2 evidence

GPT-6 Luna (low) reviewed only completed aggregate outputs and the design.
Corpus: 32/32 pictures, zero errors, eight loop trials, fidelity gate passed,
maximum transformed-control error zero. Dense: 2,112 scored packets / 4,224
views, no unavailable views, fidelity gate passed.

The candidate does not meet acceptance. Dense cutoffs were inconsistent,
frequently unestablished or below production. One mono 4:3 x-axis case at
contrast 0.3 reached 64 versus production's 20 cycles/picture, but stereo
3:4 x-axis at the same contrast reached only 8 versus production's 20.
Those results do not establish general both-axis 1.5× resolution.

Paired mean display SSIM declined by approximately 0.041 on both profiles;
color delta E worsened by approximately 0.79–0.83 and detail correlation
declined. Mean encoding cost increased by approximately 334–363 ms. These
aggregate deltas are evidence against adoption, not a resolved-detail gain.
Reports and comparisons: `tmp/v7-geometry/pilot-v2/REPORT.md` and its `dense/`
subdirectory.

The review identified plausible weaknesses: one dominant periodic atom cannot
represent multi-frequency texture; bounded parameters can saturate; tiles lack
boundary continuity; replacing conventional detail costs too much natural
picture fidelity. A no-op fidelity gate does not validate the source model.
No specific encode/decode parameter formula mismatch has been established by
these aggregate results; parameter roundtrip and source-model approximation
must be distinguished before attributing the loss to an inverse error.

The implementation remains experimental. The requested resolution/quality/
cadence combination is unfinished; this pilot must not be promoted as its fix.

Direct aggregate verification corrected the supervisor's winning-case label:
the gain is mono **4:3, x axis, contrast 0.3**, not mono 3:4 y. It occurs in
both raw and display measurements: 20 to 64 sampled cycles/picture (3.2×),
with 64 the highest frequency tested. There is one winning panel out of 24
per view. This is real conditional probe evidence despite the general failure.
The new geometry fitter is Python/NumPy/SciPy, not Numba-compiled; its measured
overhead cannot establish the optimized algorithm's cadence limit. Transport
continues to use the existing Numba path. Natural-quality losses remain valid
evidence independently of that implementation-performance limitation.
