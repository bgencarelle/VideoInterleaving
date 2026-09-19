# Modem TODO (standalone-modem)

## Verify current implementation

* [ ] Run `modem_tests`.

  * Expected: 258 tests run.
  * Only the 3 known failures remain:

    * `test_rgb_fidelity`
    * `test_rolloff_engine…`
    * `test_unspread_layout…`

* [ ] Run:

  ```bash
  tools/compare_codecs.py --modem-dir images_modem
  ```

  * Confirm the current codec improves results on every channel except `clean`.

* [ ] Test live transmission over BlackHole.

  * `hd-dwt` fills the window.
  * Zero deadline misses.
  * Receiver auto-detects the stream without requiring `--codec`.

* [ ] Test a real cassette round-trip using the same deck and tape.

  * Confirm decode reliability and image quality.

## Open bug: channel does not recover after damage

Under heavy compression or distortion on the left channel, one channel can remain undecoded even after the signal becomes clean again.

This has not yet been reproduced from an offline recording.

* [ ] Capture approximately 20 seconds from BlackHole:

  1. clean signal
  2. introduce damage
  3. return to clean signal

* [ ] Save the capture as:

  ```text
  stuck.wav
  ```

* [ ] Decode `stuck.wav` using the current modem-check CLI.

* [ ] Reproduce the stuck-channel state offline.

* [ ] Identify which receiver state fails to recover after clean packets return.

## Next: low-cost robustness improvements

### Pitch-shift correction

* [ ] Estimate pitch shift from the preamble.

  * Measure the edge ratio using `measure_pulses`.
  * Compare it with the nominal cadence.

* [ ] Correct each symbol using the measured ratio `r`.

  * Option A: resample by `1/r`.
  * Option B: sample the spectrum at `r * k`.

### Pitch-to-hue control

Use intentional pitch changes as a display control while cancelling tape-speed variation.

* [ ] Compute:

  ```text
  intentional_pitch = edge_ratio / cadence_ratio
  ```

* [ ] Apply a deadband of approximately `0.3–0.5` semitone.

* [ ] Smooth the estimate over `3–5` frames.

* [ ] Map one octave to `360°` of hue rotation.

* [ ] Rotate `Cb/Cr` at display time.

* [ ] Only apply hue rotation when both channels show the corresponding pitch shift.

### Per-channel recovery

* [ ] When one channel becomes damaged, hold that channel's existing `InputLevel` gain.

* [ ] Require approximately 3 consecutive good packets before re-admitting the channel.

* [ ] On re-admission, perform a one-step gain recalibration from that channel's preamble.

### Constant frequency-offset correction

* [ ] Estimate a constant frequency offset independently for each channel using the training symbols.

* [ ] Apply the correction before symbol decoding.

### Independent channel timing

* [ ] Detect when `skew_samples` is large enough to indicate channel timing misalignment.

* [ ] Maintain separate timing estimates for the left and right channels when needed.

* [ ] Verify this improves tolerance to cassette-head azimuth error.

## Wire-format notes

* The wire format changed in patch 4.
* Sender and receiver must be updated together.
* Recordings created before that wire-format change are not expected to decode with the current implementation.
* Mono transmission is possible, but equivalent quality is approximately 7 fps.

# Anamorphic aspect support

## Goal

Transmit every image using the native `80x96` wire geometry without letterboxing, while preserving the source aspect ratio for display.

The sender stretches the source to exactly `80x96`.

The receiver decodes the normal `80x96` image, reads the transmitted aspect preset, then stretches the image to the intended display aspect.

Benefits:

* no bandwidth spent on black letterbox bars;
* all transmitted pixels carry image content;
* codec internals remain unchanged;
* the wire image remains `80x96`.

For example, a `16:9` source is stretched to `80x96` before transmission and restored to `16:9` only after decode.

## Aspect signalling

Store the aspect preset in the **top 3 bits** of the 32-bit `absolute` field.

Bit layout:

```text
31        29 28                         0
+-----------+----------------------------+
| aspect    |        frame counter       |
| 3 bits    |          29 bits           |
+-----------+----------------------------+
```

Bits `29–31` carry the 3-bit aspect preset.

Bits `0–28` remain the frame counter.

Encode with:

```text
encoded_absolute =
    (absolute & 0x1fffffff) |
    (aspect_code << 29)
```

Decode with:

```text
aspect_code = encoded_absolute >> 29
absolute = encoded_absolute & 0x1fffffff
```

A 29-bit frame counter is vastly larger than required for a tape-length recording, so sacrificing the top 3 bits has no practical effect on usable recording duration.

Do not use `count` for aspect signalling because:

* live transmission uses `0xffff`;
* that value would appear as preset code `7`;
* repurposing its bits would impose unnecessary recording-length limits;
* other code already depends on `count`.

## Aspect presets

| Code | Display aspect |
| ---- | -------------- |
| 0    | Native `5:6`   |
| 1    | `1:1`          |
| 2    | `4:3`          |
| 3    | `3:2`          |
| 4    | `16:9`         |
| 5    | `2.39:1`       |
| 6    | `3:4`          |
| 7    | `9:16`         |

Preset `0` represents the existing native output.

## Implementation tasks

### 1. Audit uses of `absolute`

Before changing the header:

* [ ] Find every downstream use of `absolute`.

* [ ] Verify that nothing expects bits `29–31` to contain frame-counter data.

* [ ] Check:

  * frame ordering;
  * `LivePicture`;
  * presentation logic;
  * `--save-frames` filenames;
  * wraparound handling;
  * comparisons and indexing based on frame number.

* [ ] After packet decode, ensure all frame-number logic uses the masked 29-bit value.

### 2. Sender

Update `modem_screen` and the current modem write/live-send path.

* [ ] Add:

  ```text
  --aspect PRESET
  ```

* [ ] Map `PRESET` to a 3-bit aspect code.

* [ ] Resize the source directly to `80x96`.

  * Do not letterbox.
  * Do not add black bars.

* [ ] Put the aspect code into bits `29–31` of `absolute`:

  ```text
  encoded_absolute =
      (absolute & 0x1fffffff) |
      (aspect_code << 29)
  ```

* [ ] Leave codec processing unchanged after the `80x96` image is produced.

### 3. Receiver

Update `decode_packet` / `Receiver`.

* [ ] Extract the aspect code from bits `29–31`:

  ```text
  aspect_code = absolute >> 29
  ```

* [ ] Mask `absolute` back to the 29-bit frame counter:

  ```text
  absolute &= 0x1fffffff
  ```

* [ ] Convert `aspect_code` to the corresponding display aspect.

* [ ] Store it in:

  ```text
  extra["aspect"]
  ```

* [ ] Ensure all downstream frame-number logic sees only the masked value.

### 4. Display and saved output

Update:

* live display;
* `--save-frames`;
* `tools/decode_wav`.

For each decoded frame:

* [ ] Decode normally at `80x96`.

* [ ] Read `extra["aspect"]`.

* [ ] Stretch the decoded image to the requested display aspect.

* [ ] Use smooth interpolation.

* [ ] Do not use `NEAREST`.

* [ ] Keep preset `0` identical to the current native output.

### 5. Tests

* [ ] Round-trip all 8 aspect presets.

* [ ] Verify the transmitted image remains exactly `80x96`.

* [ ] Verify bits `29–31` contain the correct aspect code.

* [ ] Verify bits `0–28` preserve the frame counter.

* [ ] Verify the receiver masks `absolute` before frame ordering or indexing.

* [ ] Verify preset `0` produces exactly the existing native geometry.

* [ ] Verify non-native presets contain no sender-generated letterbox bars.

* [ ] Verify live display and saved frames restore the same aspect ratio.

## Compatibility

Sender and receiver should be updated together.

A receiver that does not understand the aspect bits may still decode the image payload, but it can interpret `absolute` as a very large frame number and will ignore the intended display aspect.

## Later: landscape transmission profile

A separate `96x80` landscape transmission grid could reduce anamorphic squeezing for wide sources.

* [ ] Keep this as a separate transmission profile.

* [ ] Use an available profile/header code rather than overloading the aspect preset.

* [ ] Implement only after the `80x96` anamorphic path is complete and tested.
