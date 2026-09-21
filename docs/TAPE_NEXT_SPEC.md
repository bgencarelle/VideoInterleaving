# Tape-Next: a 15 fps, foundation-first analog picture wire

**Status: proposed design, not an implemented or validated wire.**
Written 2026-09-20 against the existing
[transport specification](transport_timing_and_encoding.md), its
[state machines](transport_state_machines.md), and the recorded V6 experiments
in [MODEM_TODO.md](../MODEM_TODO.md). Backward compatibility has no value in
this proposal. The name is provisional, not a registered V7 profile.

## 1. Recommendation

Build a narrower, frame-independent analog transform wire with:

1. A **common-mode color foundation**: the same essential picture signal on
   both tape tracks. Either usable track carries a complete coarse picture.
2. Separate, expendable stereo enhancement streams, confined to a higher band.
3. **15 new pictures per second**, including foundation and current-frame
   enhancement. Do not count held or interpolated frames as successful updates.
4. Pulse-counted acquisition followed by continuous time-base correction,
   short OFDM symbols, and channel-aware analog reconstruction.
5. Small error-corrected metadata which is not needed to interpret image slots.
6. A narrower whole-waveform emission target, including synchronization and
   transitions, rather than a payload-only frequency limit.

The central change is **spend bandwidth on a picture that survives, not on
retaining 2,880 coefficients at an unacceptable frame rate**. Keep the 80x96
canvas; accept and quantify reduced effective spatial detail.

This is the recommended first prototype, not a claim that OFDM is ultimately
the best waveform for tape. Section 14 describes alternatives worth trying
once this prototype provides an honest bandwidth/energy/fps control.

## 2. Requirements and meaning of “works”

### Hard requirements

- Output geometry: 80 pixels wide by 96 high; source/display aspect is separate.
- Nominal cadence: exactly 15 fps at the reference clock. The user's 12–15 fps
  requirement is interpreted as a 15 fps design target and a 12 fresh-fps
  acceptance floor within a measured operating envelope.
- Frame-local decoding: no previous decoded picture is needed for the next one.
- Cheap, deterministic classical decoding; no learned hallucination or model
  distribution dependency.
- Normal acquisition remains LTC-style edge/pulse measurement. No FFT template
  search. Retain the `measure_pulses` architecture, adjusting its constants and
  thresholds to the new filtered waveform if measurements require it.
- Show a usable damaged picture. Hold the last good picture on total loss.
  Before the first picture, report “acquiring”; do not synthesize a black frame.
- No requirement to recover independent enhancement streams after losing a track.

### Proposed medium envelope

Start with stereo audio recording paths with a usable core approximately
0.7–5.5 kHz and a desirable extension to about 8 kHz. These are **design
assumptions**, not a specification for all low-quality cassette decks. A mono
path is an intentional coarse-picture fallback. Telephone-band audio, severe
hard clipping, and arbitrary dropouts cannot be promised full service.

At 15 transmitted fps, 12 fresh decoded fps means at least 80% picture recovery
over a stated measurement interval. A complete one-second audio dropout cannot
produce twelve genuine new pictures. Report operating envelope, interval,
longest freeze, and recovery latency rather than hiding this limit in an average.
Off-speed tape also changes physical picture cadence unless playback is retimed;
all nominal fps arithmetic below is at the reference speed.

Priority order: timing and recognizable structure; believable coarse color;
fresh-picture cadence; detail. Geometry alone is not fidelity.

## 3. What the existing revisions teach us

Numbers below are from sections 2 and 5–10 of the existing specification.

| Existing path | Useful property | Obstacle for this goal | Decision |
|---|---|---|---|
| V1 | Independent analog-valued pictures at 15 fps | Raw pixels spend bandwidth poorly; wide band; correlation acquisition | Keep independence, discard raw-pixel allocation |
| V2 tape-fast | 14.35 fps, DCT, 1,260-value capacity, carriers 1.125–10.125 kHz | No explicit foundation diversity; historical timing | Keep as a serious fidelity/rate control |
| V3 `wire` | 15 fps and existing pulse receiver | Carriers extend to 20.25 kHz | Keep bounded receiver architecture, narrow the wire |
| V5 / `wire-hd` | 13.76 fps; graded transform recovery | Same wide band; 800 copies are expensive | Compare transform at equal budget, not by version name |
| `wire-tape` | 16.48 fps; 800 originals plus 800 copies | Still 12.75 kHz carriers / 14 kHz ceiling; copies consume half the values | Strong current control; reduce band and protect a smaller base |
| V6 | Explicit luma/chroma foundation and reliability fusion | 8.96 fps already fails the rate requirement | Keep priority concept, discard coefficient count and packet length |
| V6-repeat | Stronger isolated-track survival | 6.04 fps; reported regressions in several other synthetic cases | Reject full-picture repetition |
| Tape-ordered V6 | Essential coefficients moved out of treble; time spreading | Still 8.96 fps; copies can share the same physical damage | Keep importance-aware placement, redesign the transport budget |

Important deductions:

- Copy separation helps independent errors, not a shared tape dropout or shared
  treble loss. The tape-placement diagnostic already found cases where copies
  added little beyond putting the originals in a healthier band.
- A rank-deficient stereo channel cannot reveal two arbitrary independent
  streams. A regularized inverse cannot manufacture that missing information.
- A once-per-frame speed estimate and a once-per-frame magnitude calibration
  are different from tracking time warp and gain changes *within* the frame.
- A carrier ceiling is not an emission mask. Filtering, preamble edges, frame
  boundaries, and cyclic-prefix length must be designed together.
- The existing image-vector RMSE is against the coder's clean source projection.
  It does not say how much source detail the projection discarded.

## 4. Information budget before algorithm choice

A real bandpass channel of width B has roughly 2B real signal dimensions per
second. This is a degrees-of-freedom bound, **not** a reliable-bit-rate promise.
For two independent 7 kHz tracks at 15 fps, the optimistic budget is about
`2 tracks * 2 * 7000 / 15 = 1867` real dimensions per frame before timing,
pilots, shaping, protection, and channel conditioning.

V6's 3,600 analog values cannot simply be squeezed into that budget by changing
their permutation. Quantizing, predicting, or dropping coefficients changes
the problem; none provides free robust dimensions. If the channel is mono,
the independent-dimension budget is approximately halved.

The proposal carries **576 unique coefficients**, of which 240 form the
protected foundation. This is deliberately less than `wire-tape`'s 800.
Whether the resulting softness is acceptable is an early go/no-go question.

### Measured source-only sanity check

One calculation was performed for this proposal: noiseless DCT truncation of
the checked-in `modem_tests/fixtures/v6_face_1110.png`, through the existing
preparation path. Its prepared-plane hash matches
`d0270070c30be57d9c319223c9595ff9705e7732bc98e4205f289506d9462edc`.
No audio, acquisition, equalizer, gains, confidence gates or tape simulation
participate in this calculation.

| Retained support | Values | Y RMSE | Cb RMSE | Cr RMSE | Y PSNR |
|---|---:|---:|---:|---:|---:|
| TN15 foundation | 240 | 0.16837 | 0.03577 | 0.05408 | 21.50 dB |
| TN15 full | 576 | 0.12347 | 0.02899 | 0.04407 | 24.19 dB |
| Current `wire-tape` DCT rectangles | 800 | 0.10714 | 0.02592 | 0.04061 | 25.42 dB |
| Current V6 DCT rectangles | 2,880 | 0.06315 | 0.01030 | 0.01485 | 30.01 dB |

RMSE uses prepared planes in [-1,1]; PSNR uses peak range 2, hence
`10*log10(4/MSE_Y)`. Each projection is `idctn(mask*dctn(plane))`, with
orthonormal type-II transforms and the rectangles specified below or in the
existing spec. These are source-support comparisons, not full codec results.

On this one image, TN15 gives up about 1.23 dB luma PSNR against the current
tape support, and 5.82 dB against V6 support, before any medium damage. That
is a real cost. A claim of better practical video must earn it back through
bandwidth, cadence and recovery; the table does not establish that it will.

## 5. Proposed physical geometry: TN15

This section fixes prototype arithmetic. Filter and tracking qualification in
sections 9 and 15 is required before these constants become a released wire.

| Parameter | Proposed value |
|---|---:|
| Reference sample rate | 48,000 Hz |
| Frame stride | 3,200 samples = 66.6667 ms = 15 fps |
| Pulse acquisition region | 288 samples |
| Useful OFDM length | 128 samples |
| Cyclic prefix | 32 samples |
| Symbol stride | 160 samples = 3.3333 ms |
| Body | 18 symbols: 2 training + 16 payload epochs |
| End guard | 32 samples |
| Check | `288 + 18*160 + 32 = 3200` |
| Active positive-frequency bins | 2 through 21, inclusive |
| Pilot bins | 3, 7, 14, 21 |
| Lowest / highest active carrier | 750 / 7,875 Hz |
| Core data bins | 2, 4, 5, 6, 8, 9, 10, 11, 12, 13 |
| Enhancement data bins | 15, 16, 17, 18, 19, 20 |

Bins 0, 1, and 22–64 are zero in the unshaped body. Use Hermitian completion
and the ordinary real 128-point inverse transform. Acquisition still uses
edges, and demodulation still uses fixed-size 128-point real FFTs.

The longer CP is a deliberate 11.1% symbol-duration cost relative to the
current 144-sample symbols. It provides more room for a physically shaped
waveform and channel dispersion. It does not correct wow/flutter by itself,
and 32 samples is not a promise that every tape/channel impulse response fits.

### Exhaustive slot budget

Count a complex cell as two real analog dimensions; header QPSK carries two
bits per complex cell, not two arbitrary analog coefficients plus two bits.

| Region | Available per frame | Assigned |
|---|---:|---|
| Core, identical on both tracks | `16*10*2 = 320` unique real positions | 240 foundation + 76 header-bit dimensions + 4 repeated anchors |
| Enhancement, independent tracks | `16*6*2*2 = 384` real positions | 336 detail + 48 known refresh-training positions |

Counting both physical tracks gives 1,024 real data-bin positions: 640 common
positions and 384 enhancement positions. There is no unbudgeted FEC or pilot
overhead. Two initial training symbols and all four pilot bins are outside
that data-bin count.

No 12 fps alternative wire is specified here. First demonstrate the 15 fps
candidate; an eventual 12 fps choice must publish new complete arithmetic,
rather than slowing the audio and accidentally changing the occupied band.

## 6. Source coding: spend detail where it matters

### Preparation

For the first prototype, use the existing `prepare_image()` and
`image_values(..., coder.grids)` path: RGB 80x96 Lanczos preparation,
full-range Pillow YCbCr, BOX chroma sampling, and values in [-1,1]. Source grids
are `(96,80), (48,40), (48,40)` in row/column order. Preserve the existing
prepared-plane vector as a control and record the Pillow version.

An interoperable release must freeze exact preparation rounding/resampling
semantics and fixtures. The approximate BT.601 equations alone are insufficient
to reproduce Pillow bytes. Changing the picture preparation and transport in
the same first comparison would obscure the source of any improvement.

Apply an orthonormal two-dimensional DCT-II independently to each plane.
Retain these top-left frequency rectangles, dimensions given as rows x columns:

| Tier | Y | Cb | Cr | Unique values |
|---|---:|---:|---:|---:|
| Foundation | 16x12 = 192 | 6x4 = 24 | 6x4 = 24 | 240 |
| Total retained | 24x20 = 480 | 8x6 = 48 | 8x6 = 48 | 576 |
| Enhancement only | 288 | 24 | 24 | 336 |

Missing frequencies reconstruct as zero DCT coefficients. The result is a
current-frame low-pass reconstruction, not an upscaled claim of native 80x96
detail. Foundation-only decoding has substantially less detail again.

Start with DCT because it is already understood, orthonormal, inexpensive, and
requires no coefficient-support metadata. Compare CDF 9/7 only with matched
airtime, bandwidth, energy, and transmitted counts; inverse-basis energy must
be included when comparing distortion for a nonorthogonal transform.

### Gains and coefficient order

Use fixed, published gains in the first experiment; do not need a decoded
per-frame variance table to interpret an analog coefficient. Within each tier,
sort by `(u/H)^2 + (v/W)^2`, then plane order Y, Cb, Cr, then row, then column.
Here H,W are the full sampling-grid dimensions, not the retained rectangles.

An initial reproducible gain candidate is:

```text
r_i = sqrt((u/H)^2 + (v/W)^2)
v_i = (1 + 12*r_i)^(-2)       # design prior, not a measured image variance
w_i = 1 for Y, 2 for Cb/Cr   # proposed preference for preserving color
g_i = (w_i / v_i)^(1/4)
```

Normalize gains separately within the foundation and enhancement so
`mean(g_i^2 * v_i) = 1`; then multiply foundation gains by sqrt(2).
Training/pilot complex amplitudes start at unit magnitude; header QPSK has
unit magnitude. All are scaled together by the waveform headroom rule.
These are initial constants for testing, not measured optimal color weights.

Why a fourth root? For independent additive noise n_i, inverse-gain error
contributes `w_i*n_i/g_i^2`, while expected transmit energy is `g_i^2*v_i`.
Minimizing the first sum for a fixed second sum gives
`g_i proportional to (w_i*n_i/v_i)^(1/4)`. This derivation assumes a linear
channel, no clipping, and known statistics. It does not establish optimality
under tape saturation or justify trusting the frequency prior as measured RMS.

DC coefficients can dominate real pictures despite the simple prior. The
prototype must report DC-driven crest loss; an independently protected
bounded compander for anchors is an experiment, not silently assumed here.

## 7. Placement and stereo behavior

### Common core

Transmit **identical complex core data, pilots, header, and core training on
both tracks**, with identical phase rotations. Do not negate or independently
scramble this region between tracks.

Each receive input sees an effective scalar channel for the core: if the
physical stereo channel is H, its core response is the corresponding row sum
of H. Decode each usable input independently and combine estimates by noise
precision. No 2x2 inverse is needed to recover the core.

This handles isolated track loss and ordinary same-polarity mono downmix far
better structurally than sending independent essentials on both tracks.
It does **not** defeat destructive phase cancellation, opposite-polarity
downmix, shared dropout, or a null in both effective core responses. Select
inputs before combining; never blindly sum captured tracks.

Number the 160 core complex cells by payload epoch first, then ascending
core carrier. Logical cell i is placed in physical cell `(37*i) mod 160`:

- i=0..37: the 38 QPSK header cells described in section 8.
- i=38..157: the 240 sorted foundation coefficients, consecutive pairs as I,Q.
- i=158..159: four analog anchors, in order Y DC, Cb DC, Cr DC, Y `(1,0)`.

Use the same coefficient gains for repeated anchors as for their originals.
This permutation is a bijection because 37 and 160 are coprime. It spreads
the core through time/frequency without a runtime assignment optimization.
Publish the resulting anchor separations in vectors; it does not guarantee
an arbitrary burst can be corrected. Most foundation values are simultaneous
cross-track copies, **not time-separated copies**.

### Enhancement

Payload epochs 7 and 8 carry known high-band refresh training rather than
detail. All other epochs carry six data carriers on two tracks, with two real
components each: `14*6*2*2 = 336` coefficient slots.

Enumerate eligible enhancement slots by epoch, ascending carrier, track, I/Q.
Coefficient i occupies slot `(101*i) mod 336` (also a bijection). This scatters
the enhancement instead of placing all chroma in one short interval.

Use the existing two-row orthogonal-training idea for the enhancement:
track patterns `[+1,+1]` then `[+1,-1]` in both the opening training pair and
the refresh pair. Estimate the per-carrier 2x2 channel, regularize it, and
discard unreliable dimensions. Do not pretend independent streams are
recoverable from one mixed observation when the channel loses rank.

An intact isolated track with negligible crosstalk may yield some enhancement;
the contractual fallback is the common foundation. Stereo is a detail bonus,
not a prerequisite for color and structure.

### Deterministic phases

Avoid a library-version-dependent RNG for the new wire. For body symbol s
(including training), bin k, and transmit track c, use phase
`(pi/2) * ((s*s + 3*k + s*k + 2*c) mod 4)` in the enhancement, and the same
formula with c=0 in the core. Known pilots follow the corresponding region's
phase rule. These arbitrary deterministic rotations require crest-factor
testing; they are not asserted to be a PAPR optimum.

## 8. Minimal, protected metadata

The physical wire, transform, gain table, and slot map are fixed. A bad header
must not make otherwise reliable pixels uninterpretable.

Proposed 16-bit metadata word, big-endian, MSB first:

| Bits | Content |
|---|---|
| 15..13 | Fixed format tag `101` |
| 12..10 | Aspect code, existing eight aspect ratios |
| 9..0 | Frame sequence modulo 1024 |

Append CRC-16/CCITT-FALSE over the two bytes: polynomial 0x1021, init 0xFFFF,
no reflection, xorout 0; check(`123456789`)=0x29B1. Append six zero tail bits.
Encode the resulting 38 bits with a constraint-length-7 rate-1/2 convolutional
code, generators octal (171,133). For an unambiguous convention, initialize a
7-bit register to zero, update `r=((r<<1)|bit)&127`, emit parity(r&0o171) then
parity(r&0o133). Tail bits force the six-bit memory back to zero.

Map consecutive coded bits to QPSK as
`((1-2*b0) + j*(1-2*b1))/sqrt(2)`: 76 coded bits, 38 complex cells.
Use soft-decision Viterbi with known zero initial/final states, followed by
CRC and tag checks. CRC detects errors; the convolutional code supplies FEC.
This replaces a long uncoded header plus combinatorial bit-flip search.

The same header is already present on both common-core tracks. No extra
replica is hidden in the budget. Require a header-code vector before release.

On header failure: decode the fixed image map, label identity unknown, and
retain the last verified aspect or use the native 5:6 aspect before first lock.
Do not label a guessed counter as verified. No folder, count, timestamp,
transform switch, dynamic gains, or coefficient-support map is carried.

## 9. Whole-waveform shaping and levels

### Proposed emission acceptance mask

- Useful carrier envelope: 750–7,875 Hz.
- Whole-stream target band: 500–9,000 Hz, including sync and transitions.
- Target integrated energy outside that band: at least 40 dB below in-band
  energy, measured on a continuous multi-frame signal.
- Also publish PSD and worst local out-of-band levels; integrated energy alone
  can hide a narrow spur. Check silence boundaries and isolated first/last frames.

Finite packets are not mathematically brick-wall bandlimited. The mask is a
test target, not an assertion that every sample sequence already meets it.

Use a continuous, causal band-shaping filter with carried state; account for
its delay in packet timestamps. Do not independently `filtfilt` each packet
and assume the result is the continuous physical channel. Avoid hard reset of
filter state at every frame. Any window overlap must use cyclic extensions
and retain a demonstrably valid FFT window; arbitrary Hann windows destroy
the assumed orthogonality.

**Release blocker:** select and publish filter taps, transition/window rules,
effective impulse-response energy outside the CP, and filtered-pulse detection
vectors. If mask, clean decode, and 32-sample CP cannot all be met, revise
the geometry and repeat the full 3,200-sample budget. Do not call this solved
by specifying a cutoff frequency alone.

Preserve the current 16-bit biphase preamble as the first acquisition candidate
inside the 288-sample region. Filter it with the same continuous waveform path.
Its useful transitions must survive core-band loss. Verify crossing bias and
false locks rather than assuming current thresholds transfer unchanged.

### Headroom

Scale an entire body's training, pilots, header, and analog payload by the
same positive scalar when enforcing headroom. Do not normalize individual
symbols or independently scale their pilots after normalization. The receiver
can then observe frame gain through its training and pilots.

Start with body peak <=0.8 before final filtering, check actual output peaks,
and lower the common scale if needed. Do not clip by default. The preamble
amplitude can remain separately fixed as in the current wire. Measure RMS,
crest factor, inter-frame gain variation, and clipping at matched tape record
levels. Equal WAV peak is not equal effective channel SNR.

## 10. Timing and equalization: repair the channel before the image

1. Edge-count the preamble on each available receive track; estimate start and
   coarse speed without spectral template matching.
2. Resample onto the reference clock with a bounded, smooth time-warp estimate.
3. Estimate effective scalar core responses and the enhancement 2x2 responses
   from the opening training pair.
4. At every payload epoch, measure common-core pilot phase at bins 3,7,14.
   Fit phase against frequency with reliability weights; slope estimates
   residual time offset. Reject cycle slips and implausible derivatives.
5. Feed the filtered offset/rate estimate back into the resampler, or use a
   bounded second pass over the buffered frame. A phase-only correction after
   the FFT cannot undo arbitrary intra-symbol time warp and ICI.
6. Track pilot magnitude as well as phase. Fit only a low-order smooth change
   relative to training; do not infer a fresh arbitrary channel at every bin
   from three pilots. Core notches and inconsistent pilots reduce confidence.
7. Use bin 21 and the midpoint training pair to diagnose/update enhancement.
   One high-band pilot cannot identify a new arbitrary 2x2 matrix each symbol.
8. Revisit the edge prediction at the next preamble. On disagreement, reacquire
   promptly instead of interpreting incorrect symbol boundaries as image noise.

Timing matters even here: a 10 microsecond residual offset causes about
`2*pi*7500*10e-6 = 0.47 radians` of phase error at 7.5 kHz. A constant offset
is correctable; rapid variation within a symbol creates interference. Narrowing
the band helps sensitivity but does not eliminate the need for tracking.

Per-input skew is not necessarily shared tape-speed error. Separate a common
transport clock estimate from per-input delay/phase correction. When inputs
disagree, choose the reliable one for timing rather than averaging blindly.

## 11. Reconstruction and loss semantics

For scalar observations `y_j = a_j*c + noise_j`, use the prior variance v and
noise variance n_j to form the linear-MMSE estimate:

```text
c_hat = (sum(a_j*y_j/n_j)) / (1/v + sum(a_j*a_j/n_j))
confidence = v*sum(a_j*a_j/n_j) / (1 + v*sum(a_j*a_j/n_j))
```

Here observations are real demapped coefficient components after known phase
handling; a_j includes coefficient gain and residual effective channel gain.
Complex channel solving precedes this formula. If observations have correlated
noise, use their covariance or conservatively cap precision; duplicate tracks
do not automatically provide two independent noise samples.

- Fuse common observations and repeated anchors by reliability.
- Treat clipped/transient-contaminated or badly conditioned observations as
  erasures rather than amplifying their inverse-channel noise.
- Apply soft shrinkage to uncertain AC coefficients. Avoid blindly copying
  V6's confidence floors: different gains, priors, and noise estimates change
  the numerical meaning of confidence.
- Prioritize mean luma and mean chroma validity. A weak DC observation must not
  silently become a confident large brightness or hue change.
- Reconstruct whatever current-frame foundation/detail is usable. Missing
  enhancement contributes zero, not a temporal prediction dependency.
- Do not add stale chroma or temporal smoothing to the baseline. Optional
  concealment must be separately labelled and must not increase fresh-fps counts.

Calibration of confidence and a minimum usable-foundation rule remain prototype
work. Freeze those thresholds using negative/no-signal controls and held-out
images, not the single face fixture. Before they are frozen this is a design
specification, not an interoperable decoder acceptance specification.

## 12. State machines

### Encoder

```text
SOURCE_AT_15HZ -> PREPARE -> DCT -> FOUNDATION / DETAIL
 -> FIXED_GAINS -> HEADER_CRC_FEC -> DETERMINISTIC_PLACEMENT
 -> COMMON_CORE + STEREO_DETAIL + TRAINING/PILOTS
 -> IFFT_128 / CP_32 -> COMMON_BODY_HEADROOM -> PULSE_FRAME
 -> CONTINUOUS_SHAPING -> DEVICE_RATE_RESAMPLE -> EMIT_3200_REFERENCE_SAMPLES
```

Carry shaping/resampling state across frames. An overrun drops/schedules source
work explicitly; it must not produce an ever-growing playback queue.

### Receiver

```text
RESET -> EDGE_SEARCH -> CANDIDATE_START_AND_SPEED
 -> BUFFER_FRAME -> TIMEBASE_CORRECT -> TRAIN_CHANNELS
 -> TRACK_EPOCH_PILOTS -> OPTIONAL_BOUNDED_TIMEBASE_REFIT
 -> CORE_FUSION + DETAIL_MMSE + HEADER_VITERBI/CRC
 -> INVERSE_DCT
    -> usable + verified metadata: RECEIVED / DEGRADED
    -> usable + failed metadata: PICTURE_ONLY
    -> no usable picture: HOLD_LAST / ACQUIRING_IF_NONE
 -> NEXT_EDGE_AGREES: COAST
 -> NEXT_EDGE_MISSING: SHORT_BOUNDED_PREDICTION / EDGE_SEARCH
 -> DISCONTINUITY: RESET
```

No multi-layout candidate scan, transform guessing, or dependency on a previous
header to know the physical map. Bound buffering to two frames plus known
filter/resampler lookahead. The implementation must declare its coast horizon
and avoid publishing noise after timing confidence expires.

## 13. Why not the tempting shortcuts?

| Shortcut | Why it is not the first recommendation |
|---|---|
| Keep V6 and speed playback up | Meets fps by moving the spectrum upward, defeating the tape-band objective |
| More identical copies | Buys redundancy by sacrificing rate/detail; shared failures still erase both |
| Strong FEC over the entire image | Requires quantization and a new rate/distortion budget; can reintroduce an image-level cliff |
| H.264/AV1 with long prediction chains | Excellent compression on suitable channels, but error propagation, buffering and decoder complexity conflict with this baseline |
| Send only motion residuals | A lost reference contaminates later pictures; saves airtime only by accepting another recovery problem |
| Learned joint source/channel codec | Training-domain dependence, model provenance and compute exceed the cheap deterministic objective |
| Random projections / compressed sensing | Natural images are compressible, not guaranteed sparse in a known support; iterative recovery and color stability are not free |
| DFT-spread OFDM as an automatic PAPR cure | Lower peaks for conventional constellations do not prove lower peaks for these heavy-tailed analog coefficients; spreading also distributes notch damage |
| Replace the picture with FM pixels | Simple and amplitude-tolerant, but FM deviation consumes bandwidth and its threshold remains; count values/fps first |

These are engineering tradeoffs, not declarations that digital or single-carrier
techniques cannot work. A matched-budget result may overturn the recommendation.

## 14. Weird ideas worth bounded experiments

### A. Replace selected repetition with small analog parity groups

Within the core tier, send K coefficients through a fixed real `(K+R)xK`
well-conditioned linear transform. Disperse rows in time and frequency and
solve a small reliability-weighted regularized system at the receiver.

Potential benefit: a small burst damages many coefficients slightly instead of
one critical coefficient catastrophically. Costs: R additional dimensions,
noise enhancement, denser error propagation, and possible peak growth. A square
Hadamard rotation alone adds **no erasure redundancy**. Benchmark tiny groups
(e.g. K=8), and debit every parity row from enhancement or foundation detail.
Keep anchors out of broad mixing until conditioning is demonstrated.

### B. Absolute rolling enhancement, never predictive residuals

Keep a complete current-frame foundation at 15 fps. Spend enhancement slots on
alternating bands of **absolute** higher-frequency coefficients. In low motion,
older detail can improve apparent sharpness. At a cut or motion, drop stale
bands and show the fresh foundation.

This explicitly trades detail freshness for static resolution; it must not be
advertised as 15 fps full-detail video. The current fixed map carries only
current-frame detail. A rolling-detail variant needs a revised age/phase rule,
cut detection, quality measurements, and bounded stale-detail lifetime.

### C. Common anchor FM + linear analog detail

Try constant-envelope narrowband FM/continuous-phase transmission for a tiny
anchor set, such as mean Y/Cb/Cr and very coarse luma, while leaving most of the
picture linear analog. Amplitude compression could then damage texture before
destroying mean color. Anchor FM is still sensitive to timebase error, and
frequency deviation/guard bands must be paid for explicitly. This is a separate
waveform candidate, not something that fits unused slots for free.

### D. Single-carrier pulse-shaped analog QAM

Compare a pulse-shaped single-carrier analog stream with short training bursts
and a time-domain equalizer, retaining edge acquisition and the same source
budget. A subband split can still keep core separate from enhancement.

Potential advantage: less multitone beating and more controllable peaks.
Risks: heavy-tailed analog symbols still have peaks; equalizer noise enhancement
and burst/ISI propagation may outweigh the gain. Use time-domain FIR processing
for this experiment rather than expanding normal acquisition into FFT searches.
Reject it unless matched-band, matched-record-level tests show a real benefit.

### E. Source statistics without fragile side information

Fit a fixed gain table on varied image material, then test held-out content.
Alternatively send a very small repeated scale-class word, but account for its
FEC and behavior on failure. Do not let adaptive gain metadata become a new
single point of failure for every picture coefficient.

## 15. Validation and release gates

### First: falsify the source-budget assumption cheaply

Before building a new modem, compute noiseless DCT projections for foundation
and full retained sets. Compare with V2 tape-fast, `wire-tape`, and current V6
on the same prepared images. Include the checked-in face, saturated color,
gradients, fine text, diagonal lines, motion and cuts. If 576 values cannot
deliver acceptable pictures, revise the allocation now. No transport can
restore detail intentionally discarded at the source.

### Then: deterministic implementation conformance

Publish exact source revision and prepared-plane, coefficient/gain, slot-map,
header/FEC, float waveform, PCM WAV, and decoded-output references. Include:

- An exact budget assertion and exhaustive slot bijection/no-collision check.
- Foundation and enhancement basis impulses, especially all three DC terms.
- Core-only, left-only, right-only, and same-polarity mono-downmix tests.
- Header corruption with preserved analog picture recovery.
- Arbitrary input chunk boundaries, start offsets, and input discontinuities.
- Complete streaming acquisition; direct `decode_packet` is insufficient.
- Two independent vector generations and another numerical environment.
- Filter-tail, output-spectrum, peak and CP-contamination measurements.

### Limited diagnostic impairments

Use a small set that isolates mechanisms: additive noise; controlled band loss;
gain step; mild saturation; mono mixing/track loss; one timed burst; smooth
time warp with a clean→damaged→clean transition. Test both shared and independent
track faults. Specify noise power, waveform level, event timing and actual
overlap with the tested clip. These diagnose defects, not a realistic cassette.

### Real-media acceptance

Use the user's actual recordings to select the eventual default. Compare
matched source clips at matched record levels, documenting deck/media, noise
reduction, channel wiring and capture rate. Report:

1. Verified identity rate and usable-picture rate separately.
2. Fresh decoded fps, played fps, held-frame fraction, and longest freeze.
3. Luma and chroma error against prepared source, plus added channel error
   against each candidate's own clean projection; do not mix those references.
4. Foundation-only fidelity, full-detail fidelity, hue/mean-color stability,
   motion/cut artifacts, and incorrect-picture publications on noise.
5. Reacquisition time after damage and after each track returns.
6. Encode/decode wall time, memory and queue latency on named hardware.

Provisional engineering gates: 15 fps clean stream; >=12 fresh fps over a
declared steady 10-second impaired interval inside the accepted envelope;
relock within two complete clean frames after damage ends; p99 decode below
33 ms on the target CPU; bounded two-frame buffering. These are targets to
measure, not existing test results. Record startup/filter latency separately.
Agree source-quality and color thresholds from the projection comparison
before accepting a high frame-recovery percentage as success.

### Release blockers, explicitly unresolved

- Source projection quality at this substantially reduced coefficient budget.
- Final whole-waveform filter/window taps and CP/mask compatibility.
- Filtered-edge thresholds, timing-loop gains, speed range and cycle-slip rules.
- Calibrated priors, confidence gates and minimum usable-foundation criterion.
- Crest factor and usable recording level on real media.
- Reference vectors for the newly proposed geometry, phase rule and header code.

Until those are closed, two implementations can agree on the allocation yet
produce different sample streams or acceptance decisions. Do not describe this
proposal as a finished interoperable standard or as proven better than V6.

## 16. Reading and provenance

Repository evidence takes precedence over recollection of version names:

- [Existing wire spec](transport_timing_and_encoding.md), sections 2–4:
  sample geometry, pulse timing, pilots and headers; sections 6–10: source
  transforms and redundancy; section 11: current decoder; section 12: vectors.
- [V6 experiment record](../MODEM_TODO.md), “Full-repeat control evaluation”
  and “Tape placement and shared-lift diagnostic”: the actual limited evidence
  behind reducing high-frequency dependence and indiscriminate repetition.
- [State machines](transport_state_machines.md): streaming behavior to retain.

External background consulted / identified during this review:

- Szymon Jakubczak and Dina Katabi, **A Cross-Layer Design for Scalable Mobile
  Video**, MobiCom 2011 (SoftCast); and Jakubczak, Rahul and Katabi,
  **One-Size-Fits-All Wireless Video**, HotNets 2009. Bibliography verified via
  [the author's publication data](https://people.csail.mit.edu/dina/data/publications.json).
  Relevant research direction: analog/linear source-channel mapping and graceful
  video degradation. Full papers were not retrieved in this pass; the gain
  optimization above is derived here, not quoted as a paper-specific result.
- [Single-carrier FDMA overview and references](https://en.wikipedia.org/wiki/Single-carrier_FDMA),
  including Myung, Lim and Goodman, *Single carrier FDMA for uplink wireless
  transmission*, 2006, DOI `10.1109/MVT.2006.307304`: motivation for investigating
  crest-factor reduction, not evidence for this analog tape payload.
- Julius O. Smith, [Constant-Overlap-Add Cases](https://www.dsprelated.com/freebooks/sasp/Constant_Overlap_Add_COLA_Cases.html):
  why arbitrary windowing and assumptions about alias cancellation are unsafe
  after spectral modification. This is background, not a TN15 filter design.

The common-core arrangement, exact TN15 budget, allocation and experiment
priorities are this proposal's engineering synthesis. Their value must be
established by the stated source, streaming and real-media gates.
