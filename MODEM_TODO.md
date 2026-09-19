# Modem TODO — next-session handoff

## Ground rules

- The modem is experimental and always in testing; it is not shipping.
- **Backward compatibility is not required.** Wire format, CLI, recordings,
  defaults, and internals may change when that improves the current design.
- Update sender and receiver together. Do not add compatibility code for old
  revisions or expect old recordings to decode.
- Work only in the modem subsystem unless the task explicitly says otherwise.
- Preserve the 80x96 wire image and never introduce a black-frame fallback.
- Timing acquisition remains pulse-counted through `measure_pulses`, not FFT
  correlation.

## Current priority

### 0. Investigate severe color banding after audio compression

Real recording tests show that audio compression can leave the decoded picture
recognizable but introduce severe color banding. Treat this as a new priority
before adding frame-rate or image-size features.

- [ ] Preserve a clean reference recording and an audio-compressed copy of the
  same transmission, using identical source frames and modem settings.
- [ ] Identify whether the banding is introduced by:
  - chroma coefficients being lost or over-weighted;
  - repeated coefficient copies combining inconsistently;
  - per-channel recovery selecting mismatched color information;
  - audio-codec clipping, companding, or level normalization;
  - inverse-wavelet ringing or coefficient quantization.
- [ ] Compare clean/compressed results separately for luminance and chroma:
  record decoded PSNR/error, color-plane histograms, gradients, and the number
  of distinct output colors.
- [x] Run raw Y/Cb/Cr survival measurements on the MP3-320 ceiling sweep.
  The damage is not chroma-only: both luma and chroma degrade substantially.
- [ ] Test whether protecting low-frequency chroma coefficients improves color
  continuity without reducing clean recovery or damaged-channel recovery.
- [ ] Test coefficient weighting/quantization that sacrifices high-frequency
  detail before average color. The picture must remain decodable with useful
  color when compression is severe.
- [ ] Test whether codec-specific preprocessing (level headroom, DC/low-band
  protection, or reduced chroma amplitude) prevents the artifact.
- [ ] Add a repeatable compressed-audio fixture and regression test once the
  failure mechanism is understood.
- [ ] Accept a fix only if it reduces banding on the compressed recording while
  preserving clean-image fidelity, color, and the existing recovery behavior.

Do not assume this is a display-scaling problem: decoded display defaults to
nearest-neighbour specifically to preserve received values. First inspect the
received coefficient values and the audio path before changing presentation.

### 1. Run final software verification

- [ ] Run the modem suite:

  ```bash
  .venv/bin/python -m unittest discover -s modem_tests -v
  ```

  The last full run before the recent capture/scheduling changes ran 268 tests.
  Its only residual issues were the existing `test_rgb_fidelity` missing-bake
  import error and the two known failures in `test_rolloff_engine…` and
  `test_unspread_layout…`. Reassess rather than blindly accepting those if the
  result changes.

- [ ] Run the focused integration/lazy-import checks:

  ```bash
  .venv/bin/python -m unittest \
    tests.test_modem_integration tests.test_lazy_imports -v
  ```

- [ ] Confirm `git diff --check` is clean.
- [ ] Preserve `scratch/`; it contains test artifacts and is intentionally
  untracked.

### 2. Validate through real analog audio/cassette

This is the next product-level test. Do this before adding robustness features.

- [ ] Record and replay using the same deck and tape.
- [ ] Confirm sustained auto-detection and acceptable image/color recovery.
- [ ] Test clean → damaged/dropout → clean recovery.
- [ ] Note deck, tape type, levels, sample rate, duration, verified-frame rate,
  and visible failure mode.
- [ ] Save a short representative clean recording and a recovery/damage
  recording as local regression fixtures. Do not commit large captures without
  explicit approval.

Known-good live commands:

```bash
python3 modem_screen.py --profile hd-dwt --device "BlackHole 2ch" \
  --source screen --capture-width 320 --encode-filter box \
  --prepare-ms 80 --log-frames

python3 utilities/modem_v3_check.py live-receive \
  --device "BlackHole 2ch" --channels 1,2 --on-loss damaged \
  --scaling raw --verbose --summary-seconds 2
```

For sender health, use **played**, not queued, as the meaningful count. Expected
live behavior is approximately 13.76 played fps, zero underflows, and deadline
misses at or near zero.

### 3. Run real-image codec comparison last

- [ ] After all other tests, provide a valid `images_modem/modem.json` bake and
  run:

  ```bash
  tools/compare_codecs.py --modem-dir images_modem
  ```

- [ ] Confirm the current codec improves every impaired channel; a small loss
  on the clean channel is currently expected. This remains blocked until the
  bake exists.

## Confirmed working — do not reopen without contradictory evidence

### Live capture and playback

- [x] Camera capture uses the device's default capture dimensions and FFmpeg
  scales to `--capture-width` before RGB frames enter Python.
- [x] Screen capture uses FFmpeg, auto-detects the macOS screen device, and
  scales before frames enter Python.
- [x] Scheduled audio waits are sample-counted after placement; DAC timestamp
  jitter no longer discards nearly every queued packet.
- [x] `hd-dwt` fills the receiver window.
- [x] Playback stays near 13.76 fps with deadline misses near zero.
- [x] The receiver continuously detects and displays the stream without a
  `--codec` argument.

### Damaged-channel recovery

- [x] Clean → damaged → clean recovery was confirmed live and offline.
- [x] A damaged channel holds its existing `InputLevel` gain.
- [x] Re-admission requires approximately three consecutive good packets.
- [x] Re-admission performs one-step gain recalibration from the preamble.
- [x] Verbose diagnostics expose pulses, timing channel, gains, peaks, limiter
  counts, decode attempts/results, resets, and no-decode heartbeats.

Do not redesign recovery unless a new recording reproduces a failure. If one
does, save it and reproduce it offline before changing receiver state.

### Aspect and image preparation

- [x] Every source is stretched to the native 80x96 wire image without bars.
- [x] `--aspect auto` chooses the nearest preset from actual source dimensions
  after rotation; explicit presets override it and `native` means 5:6.
- [x] Aspect code occupies bits 29–31 of `absolute`; public frame counters use
  the lower 29 bits.
- [x] All eight presets round-trip through sender, receiver, live display,
  saved frames, and WAV decoding.
- [x] Decoded display/export defaults to nearest-neighbour to preserve decoded
  pixels; `--scaling smooth` explicitly enables Lanczos.
- [x] Encoder preparation supports `box`, `nearest`, `lanczos`, and `bicubic`;
  Lanczos remains the general default. Box is the current recommended screen
  test setting.

Aspect codes:

| Code | Display aspect |
|---:|:---|
| 0 | native 5:6 |
| 1 | 1:1 |
| 2 | 4:3 |
| 3 | 3:2 |
| 4 | 16:9 |
| 5 | 2.39:1 |
| 6 | 3:4 |
| 7 | 9:16 |

Packing remains:

```text
encoded_absolute = (absolute & 0x1fffffff) | (aspect_code << 29)
aspect_code = encoded_absolute >> 29
absolute = encoded_absolute & 0x1fffffff
```

## Deferred until analog validation is complete

Do not start these merely because they are unchecked. Use analog results to
decide whether they are necessary and in what order.

### Pitch-shift correction

- [ ] Estimate pitch shift from the `measure_pulses` preamble edge ratio versus
  nominal cadence.
- [ ] Correct symbols either by resampling by `1/r` or sampling the spectrum at
  `r*k`; benchmark both before choosing.

### Pitch-to-hue control

- [ ] Derive `intentional_pitch = edge_ratio / cadence_ratio` so tape-speed
  variation is cancelled.
- [ ] Apply a 0.3–0.5 semitone deadband and smooth over 3–5 frames.
- [ ] Map one octave to 360 degrees and rotate Cb/Cr at display time.
- [ ] Apply hue only when both channels agree on the intentional shift.

### Constant frequency-offset correction

- [ ] Estimate a constant offset independently per channel from training
  symbols and correct it before symbol decoding.

### Independent channel timing — later

- [ ] Detect materially large `skew_samples`.
- [ ] Maintain separate channel timing only when needed.
- [ ] Verify improvement against real cassette-head azimuth error.

### Optional 96x80 landscape profile — later

Aspect signalling already restores landscape display geometry. A 96x80 wire
profile is **not** needed to solve aspect ratio; it could only reduce anamorphic
squeeze and coefficient loss for wide images.

- [ ] Consider it only if analog testing shows a meaningful landscape-quality
  problem.
- [ ] If implemented, use a separate profile/header code rather than an aspect
  preset.

## New investigation: lossy audio compression and color banding

Testing shows that FLAC is clean, while lossy audio can leave severe color
banding. The practical target is deliberately limited to three media classes:

1. MiniDisc-style ATRAC audio;
2. MP3;
3. M4A/AAC.

This is not a perfection target. A successful result is a recognizable picture
with stable average color and graceful loss of fine detail. Clean-wire fidelity
is not required after lossy compression, and occasional damaged detail is
acceptable. Persistent false color, severe posterization, or total loss of
recovery is not acceptable.

### First priority: decode-only mitigation (current wire unchanged)

Try these before changing the wire format. The goal is to preserve useful color
while allowing fine detail to degrade.

#### Implementation note — 2026-09-19

Confidence-gated chroma stabilization was enabled by default in the live-picture
and offline-WAV display paths. On the 60-frame MP3-320 fixture it added about
**0.089 seconds total**, or **1.5 ms/frame / 5.2%**, including PNG output.

**Tape reevaluation, 2026-09-19:** real tape performance was reported as nearly
entirely degraded after this change. Tape is the primary target, and its pilot
error/coverage must not trigger MP3-oriented temporal blending and blur.
Stabilization has therefore been removed from live reception and raw offline
decoding is again the default. `tools/decode_wav.py --stabilize-chroma` is now
an explicit lossy-media preview experiment; `--raw` remains accepted for
compatibility. Do not restore stabilization as a default without real tape
evidence.

If the runtime cost is too high, first revert the integration rather than the
underlying decoder:

1. Remove the `stabilize_chroma(...)` calls from `animation_modem/live_picture.py`.
2. Make `tools/decode_wav.py --raw` the default again, or remove its default
   stabilization call.
3. Keep `animation_modem/imaging.py` and its tests temporarily; the helper can
   remain available for an opt-in mode while performance is evaluated.

If the feature is abandoned entirely, also remove `stabilize_chroma` from
`animation_modem/imaging.py` and remove `modem_tests/test_chroma_recovery.py`.
The separate `max_carrier_hz` decoder option in `core.py`, `transport3.py`, and
`tools/decode_wav.py` is opt-in and is not required for this rollback.

- [ ] Use existing per-slot reliability and noise variance to measure chroma
  confidence separately from luma confidence.
- [x] Initial confidence-gated Cb/Cr shrinkage: suppress unreliable chroma
  detail before it creates false color bands.
- [x] Hold/blend toward the previous frame's chroma, or use a short temporal
  average, when
  current chroma low-frequency coefficients are unreliable. Keep current luma.
- [x] Initial confidence-gated spatial chroma smoothing, limited to
  low-confidence regions; do not blur reliable chroma or luma.
- [ ] Evaluate per-plane and per-coefficient candidate selection rather than
  selecting one whole-frame stereo leg for every plane.
- [ ] Calibrate the Wiener/noise model for codec-shaped errors, which may be
  structured rather than Gaussian.
- [ ] Test a small chroma-only dither or constrained color reconstruction as a
  last-resort visual mitigation. Measure whether it hides banding without
  adding unacceptable noise.
- [ ] Compare every mitigation against the unmodified decoder using decoded
  PNGs, chroma gradients, distinct-color counts, and color-plane error.

Decode-only constraints:

- [ ] Do not alter luma unless evidence shows luma is the source of the color
  error.
- [ ] Do not turn damaged chroma into black or clear a picture.
- [ ] Preserve the current damaged-frame and hold-last-good policies.
- [ ] Keep the clean-wire result byte-identical where confidence is high.

### Later: codec-friendly wire/profile experiments

Only pursue these if decode-only mitigation is insufficient:

- [ ] Test a 14–18 kHz carrier ceiling and retain the best measured point.
- [ ] Protect low-frequency Cb/Cr coefficients with more repeats and robust
  lower-band carrier placement.
- [ ] Interleave important coefficients across time, frequency, and stereo legs.
- [ ] Use short independently recoverable source blocks with local checksums;
  never use one unbounded variable-length payload.
- [ ] Test codec-friendly modulation with fewer stronger carriers and smoother
  symbol transitions.
- [ ] Compare ATRAC/MiniDisc-style, MP3, and AAC/M4A paths at practical recording
  settings. Opus is not a target medium for this project.
- [ ] Use joint stereo where available and avoid normalization/clipping where
  possible.
- [ ] If source compression is added, spend saved capacity on FEC and chroma
  protection before fine detail.

Current evidence: FLAC is perfect; 320k joint-stereo MP3 is usable but
degraded; lower carrier ceilings improve the compressed path. External pilot
tones did not show a clear benefit and became harmful at high levels.

### Codec-model constraint

MP3/AAC/ATRAC are perceptual **audio** codecs. They optimize for what a human
listener is unlikely to hear, not for preservation of the phase, amplitude, and
cross-carrier relationships that carry image information. A modem waveform can
therefore remain loud and apparently intact while its visual symbols are
irreversibly damaged. This is a wire-format problem, not something that can be
solved generally by decoder-side smoothing or by choosing the visually best
decoded frame.

- [ ] Stop treating decode-only chroma stabilization as the lossy-media
  solution; retain it only as graceful degradation.
- [ ] Define a codec-aware lossy profile with a heavily protected, independently
  recoverable visual base layer before adding detail.
- [ ] Give broad luma structure and average Cb/Cr color explicit protection;
  spend remaining capacity on detail only after the base layer survives.
- [ ] Use separate luma-only and chroma-only synthetic fixtures to measure which
  wire components survive each target codec, rather than inferring survival
  from RGB or from auditory transfer measurements.
- [ ] Require objective Y/Cb/Cr metrics and recognizability after the actual
  codec round trip; do not use visual inspection alone to select a profile.

### Diagnostic sweep requirements

The codec diagnostic must cover the complete audio baseband, not only the modem
carrier range. Sweep from **0 Hz through Nyquist (24 kHz at 48 kHz sample
rate)**, including near-DC, the modem band, and the upper edge. Exact DC is a
special case because audio paths may remove it; test both DC and a small set of
near-DC tones.

- [ ] Run a deterministic linear/log sweep from 0 to 24 kHz with known phase.
- [ ] Run stepped tones at every modem carrier plus extra points around band
  edges and suspected codec transitions.
- [ ] Run white-noise and pink-noise controls across the full baseband.
- [ ] Run impulse and abrupt-transition tests to expose codec pre-echo,
  post-echo, ringing, and frame-boundary smearing.
- [ ] Measure magnitude, phase, coherence, and time-domain ringing before and
  after MP3/AAC conversion.
- [ ] Repeat the sweep using visual static, spatial-frequency sweeps, and
  coefficient-domain random modem frames so image damage can be mapped back to
  audio frequency and wavelet plane.

### Lossy acceptance criteria

- [ ] Recognizable image survives all three target media classes.
- [ ] Average color remains stable; no persistent severe color banding.
- [ ] Fine detail may soften or disappear before color and broad shapes fail.
- [ ] The decoder continues to recover after ordinary codec damage and short
  dropouts.
- [ ] Record practical settings and failure characteristics for each medium;
  do not optimize only for a synthetic maximum-bitrate case.

### Real tape signal-chain variables

Do not model a tape machine as only a low-pass filter plus white noise.
Recording normally uses ultrasonic **AC bias** (while electronics or capture
hardware may also introduce DC offset), and consumer decks may apply
Dolby/dbx-style noise reduction. Noise reduction is level- and
frequency-dependent companding, so it can change carrier amplitudes over time
and damage OFDM relationships despite an apparently adequate passband.

- [ ] Record whether tape tests use no NR, Dolby B/C/S, dbx, or another system,
  and whether playback matches the recording setting.
- [ ] Capture silence and a zero-centered test signal to measure DC offset,
  bias leakage, mains hum, and stationary noise before testing pictures.
- [ ] Compare NR off against each available NR mode using the same recording,
  level, and source waveform.
- [ ] Measure level sweeps as well as frequency sweeps to expose companding,
  pumping, attack/release behavior, and modulation of nearby carriers.
- [ ] Check for ultrasonic bias leakage or intermodulation in the digitized
  recording; do not assume a 48 kHz capture represents energy above Nyquist.
- [ ] Keep real tape captures as the authority. Synthetic DC offset and simple
  companding tests may reproduce a failure but are not substitutes for the
  actual deck and tape formulation.

### Revised tape direction: robust high-resolution encode/decode

Real tape currently shows substantially more noise and color banding than the
phaser/distortion emulator. That emulator is not a sufficient acceptance test:
its damage is comparatively smooth and deterministic, while tape combines
colored noise, dropouts, azimuth/skew, wow/flutter, saturation, crosstalk, bias
leakage, and possibly time-varying noise-reduction gain.

The active direction remains the baked **80x96 higher-resolution image**. Do
not abandon it for a permanently coarse wire. Improve encoding, acquisition,
reliability estimation, and reconstruction so tape noise removes trustworthy
detail progressively rather than filling the high-resolution result with false
detail and false color. Stereo redundancy remains an experiment, but resolution
is a fixed design objective rather than the first thing traded away.

Implemented experiment: `modem_screen.py --profile hd-dwt --wire tape` uses a
real lower-band encoder. It reallocates the complete 3680-slot HD-DWT payload
to carriers at 375-12750 Hz using 30 image symbols, reconstructs the same 80x96
image at 8.72 fps, and mandatorily limits the whole emitted waveform (including
the preamble) to 14 kHz. This is categorically different from `--emit-ceiling`
on the wide wire, which discards already-allocated upper carriers. Validate the
new wire on real tape before making it the default.

The 8.72 fps result is a capacity baseline, **not an acceptable final frame
rate**. Its arithmetic is: 34 carriers minus 4 pilots = 30 data carriers;
stereo complex symbols carry 120 real coefficient slots per image symbol; four
dense header symbols reclaim 160 slots; `(3680-160)/120` therefore requires 30
image symbols. With training, header, sync, and guard this is 5504 samples, or
8.72 fps at 48 kHz. Do not blindly tune that number: improve source allocation.

Promising next tape coder: retain all 1920 luma wavelet coefficients but retain
only the 120-coefficient LL base from each chroma plane. That is 2160 original
coefficients, still reconstructing the 80x96 luma image with stable coarse
color. A 19-image-symbol tape frame has 2440 slots, leaving 280 strategically
placed repeats and producing 3920 samples / **12.24 fps**. This directly spends
capacity according to the stated priority (luma first) instead of transmitting
480 chroma coefficients per plane plus 800 generic repeats. Benchmark this
profile against the 8.72 fps full-payload baseline and real tape before choosing
the final wire. Also evaluate whether a stronger header code can safely use
both stereo header lanes; existing `header_split` alone is known to lose
diversity and is not the answer merely because it reaches 9.2 fps.

Implemented lower-resolution alternative: `--profile tape-80x60 --wire tape`
uses the same 375-12750 Hz carriers and 14 kHz whole-waveform ceiling, but a
1904-sample frame runs at **25.21 fps**. Its 760-slot budget carries 360
luma-prioritized DCT values (20x15 luma and 6x5 Cb/Cr), each sent twice on
frequency-diverse opposite-channel slots (720 total), reconstructed on an
80x60 / 40x30 sampling grid. This is genuinely lower
resolution and is separate from the high-resolution tape direction; compare
both on real tape rather than replacing the 80x96 goal silently.

- [ ] Re-run the same real tape capture with V3/color-DCT and V5/HD-DWT, using
  identical levels, frames, deck settings, and objective Y/Cb/Cr measurements.
- [ ] Treat V3 as a serious tape candidate. Its naturally coarse 40x48 luma and
  20x24 chroma representation, DCT energy compaction, and lack of a multilevel
  inverse-wavelet reconstruction may fail more gracefully under coefficient
  noise than V5.
- [ ] Test stereo-redundant protection for the most important high-resolution
  coefficients instead of using stereo only as a 2x-capacity MIMO path.
- [ ] Do not place duplicate coefficients on the same frequencies on both
  channels. Diversify copies across channel, carrier, symbol/time, and tape
  direction so one dropout or narrow noisy band does not erase both.
- [ ] Decode each channel independently first, then fuse per coefficient using
  pilot error, equalizer variance, clipping, and consistency between copies.
  Do not average an obviously bad copy into a good one.
- [ ] Protect luma DC/low DCT terms and average Cb/Cr first. Erase uncertain
  detail coefficients to zero so the result becomes softer rather than noisy
  or falsely colored.
- [ ] Preserve the 80x96 reconstruction while making uncertain detail decay
  toward a stable estimate rather than random noise or false color.

### Optional calibration leader and stronger preamble

A long optional leader may train the receiver for a particular deck, tape,
noise-reduction setting, level, and capture interface. A single constant tone
only measures one frequency and cannot characterize the stereo matrix, colored
noise, azimuth, companding, or carrier-dependent phase. Test a deterministic
multi-frequency, multi-level stereo calibration sequence instead.

- [ ] Design an optional leader of up to about 10 seconds containing silence,
  known pulse timing, stepped levels, per-carrier probes, both stereo polarities,
  and repeated known OFDM symbols.
- [ ] Estimate DC offset, noise floor/covariance, hum and narrow interferers,
  per-carrier complex gain, stereo crosstalk, channel skew/azimuth, clipping,
  and level-dependent companding from the leader.
- [ ] Use the learned channel only as a prior. Continue tracking pilots and
  pulse timing per packet because wow/flutter, dropouts, and NR gain vary after
  the leader.
- [ ] Keep every frame independently acquirable with the existing pulse-counted
  timing path. Starting playback mid-tape must still work without the leader;
  do not replace edge-counted acquisition with FFT correlation.
- [ ] Evaluate a stronger per-frame preamble with more robust in-band edges or
  repeated timing evidence, measuring the bandwidth/frame-rate cost against
  real tape acquisition failures.
- [ ] Store and report calibration confidence. Fall back safely to the ordinary
  decoder when the leader is absent, stale, or inconsistent with current audio.

### Future idea: holographic-style block spreading

Keep this as an idea, not the current direction. Distribute each important
high-resolution coefficient across channel, carrier, and a short bounded time
window using a deterministic orthogonal transform such as Walsh-Hadamard plus
interleaving and redundancy. Local wire damage would then become diffuse weak
image error rather than loss of one region or color component. Use short,
independently recoverable blocks and erase unreliable observations before the
inverse transform; one unbounded transform would spread a severe error across
the whole picture and make recovery latency unacceptable.
