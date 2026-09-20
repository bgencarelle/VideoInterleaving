# Testing the tape wires and profiles

Run every command from the repository root with `.venv/bin/python`. Put all
generated WAVs, PNGs, and logs under `scratch/`; do not commit them.

**Every wire test must include a live command pair.** Start the receiver string
in one terminal, then one sender string in another. Offline WAV success alone
does not validate live scheduling, device-rate adaptation, or acquisition.

## V6 analog comparison candidates

V6 is additive and experimental; it has not passed the real-tape acceptance
gate. The **baseline** DCT and CDF 9/7 variants use the same 5,360-sample wire
frame, 8.96 fps rate, 375–12,750 Hz carriers, 14 kHz whole-waveform ceiling,
2,880 analog coefficient originals, and 720 protected foundation copies. The
copied foundation is 20x24 luma plus 10x12 Cb and Cr. Baseline copies are on
another OFDM symbol, the opposite tape track, and at least eight carriers away.

The newer **tape-ordered placement** is a bench-only coder on the same `WIRE_V6`;
it is not a separate receiver candidate or live profile. It places foundation
homes on low carriers, interleaves them in time, puts copies on the opposite
track at least seven bins away, and avoids bin 1 for foundation homes. Run its
shared synthetic spacing-loss comparison with:

```bash
.venv/bin/python tools/bench_v6_tape_lift.py --frames 4
```

Its results are diagnostic only: the script explicitly does not model any
particular deck, and real tape captures remain authoritative.

Run the fixed-frame synthetic comparison (diagnostic only):

```bash
.venv/bin/python tools/bench_v6.py --frames 4
.venv/bin/python tools/bench_v6.py \
  --source path/to/rgb-sbs-frame.jpg --crop-left-half --frames 4
```

Each JSON line reports pulse acquisition, verified identity, recovered pictures,
source fidelity, damage relative to the transform's own clean decode, bandwidth,
frame rate, decode cost, and frame latency separately.

Offline DCT round trip:

```bash
.venv/bin/python utilities/modem_v3_check.py write \
  --codec v6 --profile v6-dct --frames 24 --out scratch/v6-dct.wav
.venv/bin/python utilities/modem_v3_check.py read \
  --codec v6 --wav scratch/v6-dct.wav
```

Replace `v6-dct` with `v6-wavelet` for the transform-controlled alternative.
For live loopback, start the receiver first:

```bash
.venv/bin/python utilities/modem_v3_check.py live-receive \
  --codec v6 --device "BlackHole 2ch"
.venv/bin/python utilities/modem_v3_check.py live-send \
  --codec v6 --profile v6-dct --device "BlackHole 2ch"
```

These commands keep raw display/reconstruction defaults. Do not enable chroma
stabilization when collecting baseline results.

## Full-repeat control

The full-repeat diagnostic sends all 2,880 analog coefficients twice instead of
copying only the 720-value foundation. It keeps the same 375–12,750 Hz band and
14 kHz ceiling, but runs at about 6.04 fps at 48 kHz. This is the older
“stronger copy wins” diversity behavior: copies use opposite tracks and
separated carriers, then combine by measured reliability.

Run its synthetic comparison with:

```bash
.venv/bin/python tools/bench_v6_repeat.py
```

It is also available through the normal sender/receiver commands:

```bash
.venv/bin/python utilities/modem_v3_check.py live-receive \
  --codec v6-repeat --device "BlackHole 2ch"
.venv/bin/python modem_screen.py --source camera \
  --wire v6-repeat --profile v6-repeat-dct \
  --device "BlackHole 2ch" --capture-fps 15
```

## What to compare

| Label | Sender arguments | Picture | Rate | Information band |
|---|---|---:|---:|---:|
| Wide reference | `--profile hd-dwt --wire wide` | 80x96 | 13.76 fps | 375–20,250 Hz |
| Redundant tape HD | `--profile hd-dwt --wire tape` | 80x96 | 16.48 fps | 375–12,750 Hz |
| Fast tape | `--profile tape-80x60 --wire tape` | 80x60 | 25.21 fps | 375–12,750 Hz |

Both tape wires limit the complete emitted signal, including preamble
harmonics, to 14 kHz. The 80x96 profile sends 800 luma-heavy DCT coefficients
twice; the fast profile sends 360. Copies use opposite stereo channels and
separated carriers.

The receiver detects these combinations automatically. Do not pass a codec,
profile, wire, or carrier cutoff to the receiver.

## Encoder and display filtering

The normal sender default is `--encode-filter lanczos`. This is spatial
anti-aliasing during source preparation, not temporal smoothing, but the
conversion from the prepared image to the profile's coefficient grid also
currently uses Lanczos. Therefore `--encode-filter nearest` disables only the
first resize; it does not make the complete encoder nearest-neighbour.

For controlled filter comparisons, repeat the same test with:

```bash
.venv/bin/python modem_screen.py --source test \
  --profile tape-80x60 --wire tape --encode-filter nearest \
  --frames 50 --write scratch/wire-test/tape-80x60-nearest.wav
```

The decoder does not apply temporal or chroma stabilization unless
`--stabilize-chroma` is explicitly supplied. Offline decoding is raw by
default; live display also defaults to `--scaling raw` (nearest-neighbour).
Do not use `--stabilize-chroma` or `--scaling smooth` for baseline wire
measurements.

## 1. Clean file round-trip

Always do this before involving an audio device or tape deck.

### Generate test WAVs

```bash
mkdir -p scratch/wire-test

.venv/bin/python modem_screen.py --source test \
  --profile hd-dwt --wire wide --seconds 10 \
  --write scratch/wire-test/wide-48.wav

.venv/bin/python modem_screen.py --source test \
  --profile hd-dwt --wire tape --seconds 10 \
  --write scratch/wire-test/tape-hd-48.wav

.venv/bin/python modem_screen.py --source test \
  --profile tape-80x60 --wire tape --seconds 10 \
  --write scratch/wire-test/tape-80x60-48.wav
```

### Decode them

```bash
for name in wide tape-hd tape-80x60; do
  .venv/bin/python tools/decode_wav.py \
    "scratch/wire-test/$name-48.wav" \
    --out "scratch/wire-test/$name-clean-frames" \
    --frames 0 --raw --verbose \
    > "scratch/wire-test/$name-clean.log"
done
```

Every packet should report `verified_header`. The final summary identifies the
detected wire. Expected names are `wire-hd`, `wire-tape`, and `wire-tape-25`.

## 2. Verify 96 kHz playback files

The live sender adapts automatically when its output device is already set to
96 kHz. It preserves the frequencies and picture rate; it emits twice as many
samples per frame. These files reproduce that timing for an offline test:

```bash
for name in wide tape-hd tape-80x60; do
  ffmpeg -y -hide_banner -loglevel error \
    -i "scratch/wire-test/$name-48.wav" \
    -ar 96000 -ac 2 -c:a pcm_s16le \
    "scratch/wire-test/$name-96.wav"

  .venv/bin/python tools/decode_wav.py \
    "scratch/wire-test/$name-96.wav" \
    --out "scratch/wire-test/$name-96-frames" \
    --frames 0 --raw --verbose \
    > "scratch/wire-test/$name-96.log"
done
```

Again, every clean packet should verify. Do not speed up the 48 kHz samples by
merely relabeling the WAV header; perform a real resample as above.

## 3. Live direct-loopback test

Select explicit devices. On macOS, use BlackHole rather than built-in speakers.
Set the device rate to 96 kHz before starting; the modem never changes it.

Start the receiver first:

```bash
.venv/bin/python utilities/modem_v3_check.py live-receive \
  --device "BlackHole 2ch" \
  --save-frames scratch/wire-test/live-received \
  --on-loss damaged --scaling raw --verbose
```

Then run one sender at a time:

```bash
# Wide reference
.venv/bin/python modem_screen.py --source test \
  --profile hd-dwt --wire wide --device "BlackHole 2ch"

# Redundant 80x96 tape wire
.venv/bin/python modem_screen.py --source test \
  --profile hd-dwt --wire tape --device "BlackHole 2ch"

# Fast redundant tape wire
.venv/bin/python modem_screen.py --source test \
  --profile tape-80x60 --wire tape --device "BlackHole 2ch"
```

The receiver should relock and print the new wire/profile after changing
senders. Stop one sender before starting another.

### Copy-paste live command pairs

Receiver for all current wires:

```bash
.venv/bin/python utilities/modem_v3_check.py live-receive --device "BlackHole 2ch" --on-loss damaged --scaling raw --verbose
```

80x96 wide sender:

```bash
.venv/bin/python modem_screen.py --source test --profile hd-dwt --wire wide --device "BlackHole 2ch" --log-frames
```

80x96 tape-band sender:

```bash
.venv/bin/python modem_screen.py --source test --profile hd-dwt --wire tape --device "BlackHole 2ch" --log-frames
```

80x60 redundant 25 fps sender:

```bash
.venv/bin/python modem_screen.py --source test --profile tape-80x60 --wire tape --device "BlackHole 2ch" --log-frames
```

## 4. Real tape test

Use the three 96 kHz WAVs from section 2. Record them to the same tape using the
same deck, level, direction, and noise-reduction setting. Put a spoken or silent
separator between cases, but do not normalize the files independently.

Record these details:

- tape formulation and condition;
- deck and speed;
- input and playback levels;
- no NR, Dolby B/C/S, dbx, or another mode;
- whether record and playback NR settings match;
- capture interface and sample rate;
- channel order, clipping, and any AGC or enhancement setting.

Capture playback as stereo, 16-bit PCM WAV at 96 kHz. Disable AGC, noise
suppression, echo cancellation, normalization, and mono mixing. Preserve both
channels even if one sounds worse.

Name the separated captures clearly:

```text
scratch/wire-test/wide-tape.wav
scratch/wire-test/tape-hd-tape.wav
scratch/wire-test/tape-80x60-tape.wav
```

Decode without experimental display processing or carrier rejection:

```bash
for name in wide tape-hd tape-80x60; do
  .venv/bin/python tools/decode_wav.py \
    "scratch/wire-test/$name-tape.wav" \
    --out "scratch/wire-test/$name-tape-frames" \
    --frames 0 --raw --verbose \
    > "scratch/wire-test/$name-tape.log"
done
```

Do **not** use `--max-carrier-hz` for this comparison. Both tape encoders have
already allocated all information below their declared ceiling.

## 5. Score luma and chroma separately

Use each profile's own clean decoded frames as its reference:

```bash
for name in wide tape-hd tape-80x60; do
  .venv/bin/python tools/measure_plane_survival.py \
    "scratch/wire-test/$name-96-frames" \
    "scratch/wire-test/$name-tape-frames" \
    | tee "scratch/wire-test/$name-plane-score.txt"
done
```

Compare:

- verified packets versus attempted packets;
- Y PSNR and SSIM first;
- Cb/Cr stability and average color;
- whether damage appears as softness, snow, bands, or false color;
- recovery after dropouts and relocking time.

Do not compare the 80x60 output directly against an 80x96 PNG. Each tape
capture must be scored against the clean output from the same profile.

## 6. Run the synthetic 96 kHz tape matrix

Redundant 80x96 tape profile:

```bash
.venv/bin/python tools/test_tape_matrix.py \
  --profile hd-dwt --frames 60 \
  --out scratch/tape-matrix-hd
```

Fast redundant tape profile:

```bash
.venv/bin/python tools/test_tape_matrix.py \
  --profile tape-80x60 --frames 100 \
  --out scratch/tape-matrix-80x60
```

Quick selected cases:

```bash
.venv/bin/python tools/test_tape_matrix.py \
  --profile tape-80x60 --frames 30 \
  --only type-ii --only worn-deck \
  --out scratch/tape-matrix-quick
```

Review `summary.csv`, `comparison.png`, individual logs, WAVs, and decoded
frames. Synthetic results are regression evidence only; use the real tape
scores to choose the wire.

## 7. Interpret common failures

- **No packets:** acquisition/preamble or level failure; inspect verbose pulse
  diagnostics and clipping before judging the image coder.
- **Verified but noisy luma:** payload reliability is overestimating tape
  coefficients; compare the two stereo legs and inspect Y metrics.
- **Stable luma, color bands:** coarse chroma or chroma-copy fusion is failing;
  inspect Cb and Cr separately.
- **Direct loopback passes, tape fails:** expected evidence of a tape-path
  problem, not permission to tune against the loopback.
- **One channel fails:** retain the capture. The 80x60 profile should recover
  every coefficient from its opposite-channel copy.
- **96 kHz fails but 48 kHz passes:** verify the file was resampled rather than
  header-relabelled and confirm the capture really remained stereo PCM.

Keep all real-tape WAVs and their settings together. A result without deck,
level, NR, and capture-rate metadata is difficult to reproduce.
