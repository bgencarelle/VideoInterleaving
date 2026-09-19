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
