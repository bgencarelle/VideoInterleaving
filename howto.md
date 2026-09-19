# Modem common-use how-to

For step-by-step clean, 96 kHz, live-loopback, real-tape, and objective scoring
procedures for the new wires, see [`docs/TAPE_WIRE_TESTING.md`](docs/TAPE_WIRE_TESTING.md).

Every modem change should be tested with a live sender/receiver command pair;
the relevant copy-paste commands are included in that guide.

Run commands from this directory. Use the repository virtual environment:

```bash
.venv/bin/python ...
```

Temporary WAVs, decoded PNGs, reports, and comparison images belong in
`scratch/`. It is intentionally untracked.

## Check the installation

```bash
.venv/bin/python utilities/check_modem_setup.py
```

List audio devices:

```bash
.venv/bin/python utilities/modem_v3_check.py live-receive --list-devices
```

For real audio testing, select the intended device explicitly. Do not use
built-in speakers for modem tests.

## Send a live source

Screen:

```bash
.venv/bin/python modem_screen.py \
  --source screen \
  --device "BlackHole 2ch"
```

Camera:

```bash
.venv/bin/python modem_screen.py \
  --source camera \
  --camera 0 \
  --device "BlackHole 2ch"
```

Video file:

```bash
.venv/bin/python modem_screen.py \
  --source video \
  --file input.mp4 \
  --device "BlackHole 2ch" \
  --no-loop
```

Useful sender options:

```text
--profile color-dct|color-wavelet|hd-dwt|tape-80x60
--wire wide|tape
--aspect auto|native|16:9|4:3|...
--encode-filter box|nearest|lanczos|bicubic
--rotate 0|90|180|270
--mirror
--region LEFT,TOP,WIDTH,HEIGHT
--frames N
--seconds N
--write scratch/output.wav
--emit-ceiling HZ
--log-frames
--quiet
```

### Wire/profile choices

| Profile and wire | Picture | Rate at 48 kHz | Information band | Purpose |
|---|---:|---:|---:|---|
| `hd-dwt`, `wide` | 80x96 | 13.76 fps | 375–20,250 Hz | Wide-band reference |
| `hd-dwt`, `tape` | 80x96 | 16.48 fps | 375–12,750 Hz | Redundant luma-first tape coder |
| `tape-80x60`, `tape` | 80x60 | 25.21 fps | 375–12,750 Hz | Fast redundant tape profile |

Both tape wires bound the **whole emitted waveform**, including preamble
harmonics, at 14 kHz. Both send every retained DCT coefficient twice, on
opposite stereo channels and frequency-diverse carriers.

For the high-resolution tape transmission, use `--wire tape`. It reconstructs
80x96 from 800 luma-heavy DCT coefficients, duplicates every coefficient,
stays below 12.75 kHz, and bounds total emission at 14 kHz:

```bash
.venv/bin/python modem_screen.py --source test --profile hd-dwt \
  --wire tape --frames 30 --write scratch/tape-wire.wav
```

For the 25 fps lower-resolution tape profile:

```bash
.venv/bin/python modem_screen.py --source test --profile tape-80x60 \
  --wire tape --frames 50 --write scratch/tape-80x60.wav
```

This reconstructs 80x60 at 25.21 fps using luma-heavy DCT allocation. It is a
real 375-12750 Hz wire with total emission bounded at 14 kHz. Every coefficient
has a diversified opposite-channel copy.

At a 96 kHz output-device rate, the sender resamples automatically: carrier
frequencies and picture rate remain the values above. The device receives twice
as many samples per frame; it does not play the modem twice as fast.

`--write` creates a WAV instead of opening an audio device. This is the best
way to make a repeatable test recording:

```bash
.venv/bin/python modem_screen.py \
  --source test \
  --profile hd-dwt \
  --frames 60 \
  --write scratch/clean.wav
```

The modem is low resolution by design. Use silhouettes, faces, broad shapes,
and color fields; small text will not survive the 80x96 source grid.

## Receive live audio

Run this in a second terminal:

```bash
.venv/bin/python utilities/modem_v3_check.py live-receive \
  --device "BlackHole 2ch"
```

The receiver automatically identifies wide HD, redundant 80x96 tape, and the
80x60 tape wire. It does not need `--codec`, `--profile`, or `--wire` arguments.

Useful receiver options:

```text
--save-frames scratch/live-frames
--on-loss hold|damaged
--buffer-frames 1|2
--headless
--silent
--scaling raw|smooth
--width 480 --height 576
-v|--verbose
```

`raw` scaling preserves decoded pixels with nearest-neighbour. `smooth` is
only a display choice. `hold` keeps the last good image; `damaged` shows a
recoverable damaged image as-is.

## Write and read a fixed WAV

Write a modem WAV from an existing bake:

```bash
.venv/bin/python utilities/modem_v3_check.py write \
  --modem-dir images_modem \
  --frames 60 \
  --out scratch/clean.wav
```

Common write options:

```text
--codec v3|v4|v5
--profile color-dct|color-wavelet|hd-dwt
--stride N
--aspect auto|native|16:9|...
--encode-filter box|nearest|lanczos|bicubic
--numbered
```

Decode a WAV to PNGs:

```bash
.venv/bin/python tools/decode_wav.py scratch/clean.wav \
  --out scratch/decoded \
  --frames 60 \
  --scale 4
```

Decoder options:

```text
--start N                 Start at frame N.
--frames N                Decode N frames; 0 means all.
--scale N                 PNG display scale; native output follows the profile
                          (80x96 or 80x60).
--max-carrier-hz HZ      Ignore carriers above HZ during decoding.
--stabilize-chroma        Opt into experimental chroma stabilization.
--raw                     Compatibility option; raw decoding is now default.
-v|--verbose              Print acquisition and recovery diagnostics.
```

Raw decoding is the default and should be used for tape and measurements.
`--stabilize-chroma` is an experimental lossy-media preview option, not a tape
recovery feature.

The offline decoder also identifies all three wire/profile combinations
automatically:

```bash
.venv/bin/python tools/decode_wav.py scratch/tape-80x60.wav \
  --out scratch/tape-80x60-decoded --frames 0 --raw
```

## Test an impaired WAV without changing the file

The `read` command can emulate common channel damage before decoding:

```bash
.venv/bin/python utilities/modem_v3_check.py read \
  --wav scratch/clean.wav \
  --save-frames scratch/rough-frames \
  --emulate rough \
  --verbose
```

Individual impairment arguments are:

```text
--lowpass-hz HZ       Low-pass cutoff; 0 disables it.
--highpass-hz HZ      High-pass cutoff; 0 disables it.
--noise-dbfs DB       White-noise level, relative to peak 1.0.
--gain-db DB          Gain on both channels.
--right-gain-db DB    Additional right-channel gain.
--crosstalk FRACTION  Stereo mixing, from 0 to 0.5.
--clip PEAK           Hard clipping threshold, e.g. 0.12.
--bits N              Quantization depth, 2 to 24; 0 disables it.
--dropout-ms MS       Periodic silence length.
--dropout-every SEC   Period between dropouts.
--seed N              Repeatable impairment noise seed.
```

## Lossy-codec diagnostics

Measure MP3-320 transfer behavior from DC through Nyquist:

```bash
.venv/bin/python tools/test_mp3_channel.py \
  --out scratch/mp3-channel \
  --seconds 8
```

Measure damage on randomized real modem frames:

```bash
.venv/bin/python tools/test_mp3_modem_damage.py \
  --out scratch/mp3-modem-damage \
  --frames 60
```

Decode a lossy WAV while testing carrier ceilings:

```bash
.venv/bin/python tools/decode_wav.py input.wav \
  --out scratch/raw-18000 \
  --frames 60 \
  --raw \
  --max-carrier-hz 18000
```

Measure luma and chroma independently against reference PNGs:

```bash
.venv/bin/python tools/measure_plane_survival.py \
  scratch/reference \
  scratch/raw-18000
```

This reports separate Y, Cb, and Cr PSNR, SSIM, MAE, and normalized RMSE.

## Run the 96 kHz synthetic tape matrix

Full-payload 80x96 tape wire:

```bash
.venv/bin/python tools/test_tape_matrix.py \
  --profile hd-dwt \
  --frames 30 \
  --out scratch/tape-matrix-hd
```

Redundant 80x60/25 fps tape wire:

```bash
.venv/bin/python tools/test_tape_matrix.py \
  --profile tape-80x60 \
  --frames 50 \
  --out scratch/tape-matrix-80x60
```

Run selected cases with repeated `--only`:

```bash
.venv/bin/python tools/test_tape_matrix.py \
  --profile tape-80x60 \
  --only type-ii --only worn-deck
```

The matrix assumes 96 kHz playback and covers band limits, colored hiss,
wow/flutter, azimuth, crosstalk, gain mismatch, DC/hum, bias leakage,
saturation, dropouts, approximate NR pumping, and combined Type I/II/worn-deck
cases. Results are written to `summary.csv` with Y/Cb/Cr PSNR and SSIM. These
are deterministic regression tests, not substitutes for a real tape capture.

## Synthetic modem bench

This is an ideal in-memory test, not a tape or codec simulation:

```bash
.venv/bin/python tools/bench_modem.py \
  --modem-dir images_modem \
  --frames 24
```

Useful probes:

```bash
# Test loss of the upper audio band.
.venv/bin/python tools/bench_modem.py --band-limit-hz 15000

# Test deterministic added noise.
.venv/bin/python tools/bench_modem.py --noise-dbfs -40

# Test both together.
.venv/bin/python tools/bench_modem.py \
  --band-limit-hz 15000 \
  --noise-dbfs -40 \
  --frames 24
```

These synthetic tests are useful for regression and decode cost. They do not
predict MP3/AAC/ATRAC behavior, because those codecs use an auditory
perceptual model.

## Basic troubleshooting

- If no frames decode, use `--verbose` and verify the WAV is stereo 16-bit.
- If live audio is silent or routed incorrectly, use `--list-devices` and set
  `--device` explicitly on both ends.
- If the image is too soft, check `--scaling raw` before changing the wire.
- If MP3/AAC colors are unstable, compare `--raw` with stabilized output and
  measure Y/Cb/Cr rather than judging RGB by eye.
- Keep generated files under `scratch/`; do not commit them.
