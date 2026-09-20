# Modem TODO — V6 analog tape transport

## Authoritative direction — 2026-09-20

Return to **analog-valued image coefficients**, keep the strongest timing and
recovery infrastructure, and validate **one tape-band wire against historical
V2**. We have useful components, but previous experiments changed too many
variables at once and some conclusions were wrong.

This roadmap supersedes the previous priorities in this file. V6 is the next
implementation target, not an already implemented or validated transport.

**First milestone:** match historical V2's clean picture and damaged-color
behavior using pulse-counted timing on a verified tape-band waveform. Only then
pursue more resolution or frame rate.

## Ground rules

- Work only in the modem subsystem unless explicitly instructed otherwise.
- The modem remains experimental. Backward compatibility is not required;
  update sender and receiver together when the wire changes.
- Keep the baked 80x96 image/reconstruction target, but report actual transmitted
  detail and effective resolution honestly. An 80x96 output array alone does not
  prove 80x96 picture detail. Historical controls retain their native geometry.
- Use `transport3.measure_pulses` and pulse-counted streaming acquisition, not
  FFT-correlated timing. V2's historical acquisition is a benchmark, not the
  acquisition implementation to import into V6.
- Show recoverable damaged frames as-is. Hold the previous image when
  reconstruction fails. Never introduce a black-frame fallback.
- Raw reconstruction/display is the baseline. Temporal smoothing and chroma
  stabilization remain opt-in and must not conceal decoder failures.
- Use `.venv/bin/python`, including `-m pip` when installing dependencies.
- Use an explicit device for audio. Live loopback commands use `BlackHole 2ch`.
- Keep measurement scripts, PNGs, WAVs and captures in untracked `scratch/`.
  Stage source/documentation changes; never stage or commit `scratch/`.

## What we actually learned

- **V2 is a useful baseline.** Its clean face reconstruction is good, and it
  recovered pictures under the tested bandwidth loss and speed changes.
- **V1/V2 are not wavelet codecs.** V1 sends image-plane values; V2 applies DCT
  and power allocation. The claim that we were restoring a "V1/V2 wavelet path"
  was incorrect. Wavelets came later.
- **The digital detour imposed an unnecessary bottleneck.** The 192-byte frame
  budget drove severe compression and frame-loss thresholds. VP8/JPEG 2000
  solved a different problem from the intended analog wavelet transport. They
  are not V6's image path, payload budget, or design baseline.
- **Filtering was not the whole explanation.** Removing Lanczos/bilinear did
  not rescue JPEG 2000. Wavelet analysis is not simply another blur: severe
  information loss came from coefficient removal and quantization. Audit each
  resize and preparation step rather than assuming nearest-neighbor fixes the
  codec or treating all transform filtering as an error.
- **Timing and color recovery matter independently of sharpness.** Contact
  sheets expose color shifts, ghosting with a missing track, and acquisition
  dependence on the left track.
- **Measurements need repair before rankings can be trusted.** Picture
  recovery was mislabeled as verified recovery; V1/V2 PSNR used different
  references; some `LOST` tiles could mean missing identity rather than missing
  pixels. A 40 ms dropout starting after two seconds never affected the roughly
  one-second V1 clip. These are material benchmark errors, not passing evidence.
- The historical checkout actually tested was **`99d60665`**, despite later
  naming `2b3ea477`. Record actual revisions and source provenance in every run.
- The cropped source was 720x960 (3:4); historical color decodes were 40x48
  (5:6) with aspect-preserving preparation. Earlier comparison stretching was
  wrong. Distinguish source aspect, native grid and display aspect explicitly.

## 1. Freeze experiments and establish reproducible controls

- [ ] Record the actual historical V2 checkout `99d60665`, its configuration,
  source preparation, native geometry, wire layout and decoder settings.
- [ ] Pin the current analog `color-wavelet` control and current tape-wire
  control to exact revisions/configurations, including any uncommitted changes.
- [ ] Keep these three controls available throughout V6 development.
- [ ] Stop using the digital experiments to guide the V6 decision. Preserve
  their provenance as experiments; do not treat them as the intended transport.
- [ ] Save a reproducible clean round trip before changing any transport code.

## 2. Build one trustworthy comparison

- [ ] Use the same actual cropped left half of the RGB SBS face source for each
  candidate. Record the file names, source dimensions and crop coordinates.
- [ ] Make image preparation explicit: resizing, plane sampling, grading,
  padding/stretching and reconstruction. Avoid hidden or repeated resizing.
- [ ] Preserve correct source/display aspect and use raw reconstruction.
- [ ] Show **one fixed source frame per visual case**, with nearest-neighbor
  enlargement. Do not choose the first surviving or best-looking frame.
- [ ] Measure acquisition attempts/results and CRC-verified identity separately.
- [ ] Count recovered pictures, including unidentified pictures, separately from
  verified identity. Do not label unidentified-but-recovered pictures `LOST`.
- [ ] Match quality measurements by trustworthy frame identity or independently
  known test alignment; never zip surviving frames against sequential references.
- [ ] Measure fidelity against the source and additional channel damage against
  each codec's own clean decode as separate results, with consistent geometry.
- [ ] Report luma/chroma quality, actual waveform bandwidth, frame rate, decode
  cost and recovery latency. For live playback count played, not queued, frames.
- [ ] Ensure every impairment demonstrably occurs within the recording. Log
  dropout times/durations and include clean → damaged → clean recovery.
- [ ] Record signal levels and noise definitions so different transmit amplitudes
  are not mistaken for source-coder robustness.
- [ ] Use synthetic ideal tests for fidelity/cost and limited synthetic damage
  diagnostics to expose failures, not to claim cassette realism or tape rankings.
- [ ] Run appropriate modem tests and `git diff --check`; document unavailable
  fixtures and actual failures rather than copying old pass counts. Integration
  test paths from other branches may not exist on the standalone checkout.
- [ ] Supply verified copy-pasteable live sender/receiver commands alongside each
  candidate's offline test instructions, using the device's actual sample rate.

## 3. Combine proven parts into V6

- [x] Use the current pulse-counted acquisition and streaming infrastructure.
  Preserve bounded buffering, scheduling, diagnostics and prompt reacquisition.
- [x] Carry **analog-valued coefficients**, with V2's channel-aware reconstruction
  as a reference. Do not import V2 wholesale or turn its values into a tiny
  digital image payload.
- [x] Compare DCT versus wavelets on **the same slot budget, bandwidth and
  timing**. Do not attribute differences between different wires to the transform.
- [x] Choose one tape-compatible candidate band. V6 starts from the existing 375–12,750 Hz
  carrier envelope and approximately 14 kHz whole-waveform ceiling; verify the
  entire waveform, including synchronization, training and symbol transitions.
- [ ] Measure the historical V2 control's actual emission too: its declared
  carrier band alone does not prove that its whole waveform fits tape.
- [ ] Change one major variable at a time and retain comparable baseline results.

## 4. Protect a recognizable color foundation

- [x] Prioritize coarse luma **and coarse chroma**, then add detail.
- [x] Distribute important information across time, frequency and both tracks.
- [x] Make synchronization recoverable from either track; exercise both
  left-track and right-track loss and their return.
- [x] Evaluate per-coefficient reliability/channel-aware recovery so unreliable
  observations do not contaminate good observations or generate false color.
- [ ] Agree the foundation redundancy budget before fixing the allocation.
  Complete recovery after losing either track requires redundancy or reduced
  information. Do not accidentally duplicate everything or promise complete
  recovery from independent nonredundant lanes.
- [ ] Measure the clean-detail, frame-rate and recovery costs of any redundancy.
  Preserve useful stereo capacity while protecting only what evidence justifies.

### Full-repeat control evaluation — 2026-09-20

The full-repeat control is implemented as `wire-v6-repeat` with
`v6-repeat-dct` and `v6-repeat-wavelet`. It sends all 2,880 analog coefficients
twice, across opposite tracks and separated carriers. It preserves the
375–12,750 Hz carrier band and 14 kHz whole-waveform ceiling, but reduces the
frame rate from 8.96 fps to approximately 6.04 fps at 48 kHz.

The 12-frame synthetic matrix showed clear gains under either-track loss and
some severe Type-I/Type-II and MP3 wavelet cases, but it was not universally
better: DCT MP3, wow/flutter, hiss, and several wavelet low-pass cases regressed.
The full-repeat wire is therefore a comparison/control profile, not the V6
default. Keep its results in `scratch/` and require matched real-tape captures
before choosing between foundation-only and full-repeat protection.

### Tape placement and shared-lift diagnostic — 2026-09-20

Slot audit (`v6.slot_report`): foundation V6 put the 100 most important values
in the dense-header spare carriers (9–12.4 kHz) and packed the 720-value
foundation into symbols 2–10 and its copies into symbols 28–34 — high in
frequency *and* concentrated in time.

`v6.tape_coder()` (bench-only, not wired into the CLI) keeps the same wire,
source coder and 720-copy budget but changes placement: foundation homes on
bins 2–10 (≤3.75 kHz, bin 1 demoted for hum), copies on bins 11–18 (≤6.75 kHz)
on the opposite track ~half a packet later, all tiers time-interleaved; the
header spares take the least important detail. `TAPE_COPY_SPREAD` is 7 bins
(2.6 kHz) rather than 8 to fit the low zone. `tape_coder(t, copies=False)` and
`nocopy_coder(t)` are no-copy controls.

`tools/bench_v6_tape_lift.py` (temporary) adds a shared Wallace spacing-loss
lift (54.6·d/λ dB at 1⅞ ips, seeded events on a fixed absolute timeline,
small per-track differences, optional skew/gain wander, −45 dBFS treble hiss).
Stand-in source: scipy `face` (the face source is not in this checkout).
Mean picture RMSE vs each variant's own clean decode, 6 s:

| case | V6 DCT | tape DCT | repeat DCT | V6 wav | tape wav | repeat wav |
|---|---|---|---|---|---|---|
| hiss-45 | .037 | .017 | .033 | .026 | .016 | .027 |
| lift-mild | .047 | .024 | .040 | .042 | .027 | .035 |
| lift-severe | .055 | .032 | .057 | .054 | .038 | .067 |
| lift-worn | .082 | .045 | .082 | .057 | .039 | .068 |
| lift-skew | .101 | .064 | .087 | .153 | .101 | .104 |
| mute-left | .068 | .071 | .011 | .070 | .066 | .006 |

Clean fidelity vs source is unchanged (≈.075 DCT, ≈.080 wavelet). Under
shared lifts the copies barely matter (no-copy tape ≈ tape); they matter for
track loss. Full-repeat remains best only for track loss. Synthetic only —
the hiss shape favours low carriers by construction; real tape decides.

Next: per-symbol magnitude refit from pilots (separately); then real-tape A/B
of foundation V6 vs tape placement before changing the default.

## 5. Let real tape select the final allocation

- [ ] Record the same short sequence using historical V2 and the best current
  analog candidate, with matched source, levels, deck and capture settings.
- [ ] Preserve stereo playback captures and their sample rates. Record deck,
  tape formulation, record/playback levels and noise-reduction settings.
- [ ] Measure actual roll-off, channel imbalance, timing drift and dropout
  bursts; distinguish acquisition, identity and reconstruction failures.
- [ ] Investigate color banding against those captures, including clipping,
  crosstalk and time-varying noise-reduction gain where present. Do not assume
  tape is just a low-pass filter plus white noise.
- [ ] Adjust allocation and recovery to the measured failures, then retest the
  clean controls and clean → damaged → clean transitions.
- [ ] Choose one default profile and remove preset sprawl after validation.

Real tape is authoritative. MP3/AAC/ATRAC and synthetic worn-deck experiments
may remain secondary diagnostics, but they do not set V6's priorities or
justify default smoothing. Optional calibration leaders, new correction modes,
spreading schemes, larger images and higher frame rates wait for evidence that
they address a measured problem after the first milestone.

## Definition of stable / acceptance gate

A stable tape transport produces a recognizable picture with believable color,
degrades detail before structure, reacquires promptly after damage, and runs
within its decode deadline. It shows recoverable damaged frames and holds the
previous image when reconstruction fails.

- [ ] Match historical V2's clean picture and damaged-color behavior.
- [ ] Demonstrate pulse-counted timing on a verified tape-band waveform.
- [ ] Demonstrate prompt recovery after damage and either-track interruptions;
  report partial-recovery limits honestly under the agreed redundancy budget.
- [ ] Meet the decode/playback deadline with measured margin.
- [ ] Confirm the result on real tape before declaring a stable default.

**Only after this gate: pursue more resolution or frame rate.**
