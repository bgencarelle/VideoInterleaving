# V7 Transport Specification

This document describes the V7 transport as implemented in
`animation_modem/v7.py`, its live input adapter, the application sender, and
`tools/v7_live.py`. It is a current-state specification, not a claim that every
wire option has been validated on real tape. Section 11's first sender-resize
ablation is implemented as an opt-in experiment; its later stages and Section
12 remain planned work.

## 1. Current status and scope

V7 is the pulse-framed modem transport used by modem mode. The live wire uses
the edge-counted pulse acquisition retained in `animation_modem/transport3.py`;
it does not use the continuous 72-bit clock-track proposal described in older
sections of this document. The live packet carries an OFDM image body, a
CRC-protected metadata symbol, low-bin timing tones, and an end-of-packet (EOF)
marker. The live sender defaults always include EOF; the option to omit it is
retained for legacy-wire tests and comparisons.

The transport package is independent of application settings, renderers, and
audio devices. `modem_v7_display.py` adapts it to the application image library
and audio output. `tools/v7_live.py` provides a standalone capture sender and
receiver. Synthetic decode tests are available; real tape and deck validation
remains pending.

### Validation priority

After the relevant synthetic/unit tests pass, prioritize real-time audio
playback as the next validation step. This takes priority over additional
synthetic-only benchmarks or analysis while live playback remains available and
unverified. Synthetic and in-memory loopbacks validate software paths but do
not substitute for real-time playback. Repeat this order after subsequent
changes; if playback is blocked, record the concrete blocker and leave live
validation explicitly outstanding.

## 2. Packet format and acquisition

At the 48 kHz reference geometry, one pulse packet is 3,920 samples:

| Region | Reference samples | Description |
|---|---:|---|
| Pulse header | 288 | Edge-counted biphase-mark preamble |
| OFDM body | 3,456 | 24 symbols × 144 samples |
| Metadata symbol | 144 | One OFDM symbol carrying 40 protected bits |
| Guard | 32 | Packet endpoint region; final 24 samples carry the EOF marker, and packet-wide timing tones continue through it |
| **Total** | **3,920** | **12.245 packets/s at 1×** |

The pulse word contains 16 known bits with 8-sample half-bits and nominal
amplitude 0.55. `transport3.measure_pulses()` locates the forward word;
`measure_pulses_both()` shares that Schmitt-edge pass while matching the
forward and reversed gap patterns. Both interpolate zero crossings and
estimate packet scale. V7 timing acquisition is pulse-counted, not
FFT-correlated. Captures are measured in their native sample coordinates; the
accepted scale is normalized by the capture rate relative to 48 kHz.

The receiver supports two packet-completion modes:

- **`baseline`** commits a packet after a following pulse header is found at
  the expected packet spacing and scale. A finite stream's final packet has
  no following header and is therefore not committed.
- **`eof`** validates a 24-sample alternating-polarity marker in the existing
  32-sample guard (four runs of 4, 8, 6, and 6 samples at ±0.55). It uses the
  measured header-to-marker packet scale and can commit a final packet without
  a following header. Missing, truncated, or damaged markers do not commit
  that packet.

EOF changes neither packet length nor frame rate. The standalone live sender
and receiver and the application sender enable EOF by default. Their
`--no-eof-marker` and `--no-modem-eof-marker` options are retained for legacy
wire tests and comparisons. The low-level `encode_pulse_frame()` /
`encode_pulse_stream()` and `decode_pulse_stream()` APIs still default to the
older framing (no EOF marker, next-header boundaries) for unit tests. Bench
tools that model the default live wire use `tools/v7_wire_profile.py` (section
9).

### 2.1 Reverse playback

`tools/v7_live.py receive` detects forward and reversed pulse words by default;
`--direction forward|reverse|auto` can force a direction. A reversed candidate
is retained until its full packet interval has arrived, boundedly extracted,
and normalized into forward time. The receiver then re-acquires the forward
preamble and uses the ordinary V7 decoder. Reverse packets require a validated
EOF marker and independently valid metadata; provisional loop-field metadata
is held without updating picture-tail memory. Packets without EOF are not yet
supported in reverse.

Playback direction is separate from source-animation direction. The UI and
receiver diagnostics report both; the displayed playback direction is
confirmed after two distinct metadata-validated arrivals. A direction change
resets order-dependent tail history while retaining the learned loop lock. A
missing or damaged reverse packet leaves the last good picture in place.

Turn-arounds on a packet boundary, as in a sampler's ping-pong loop or a tape
reversing exactly there, get two rules. At a reverse-to-forward turn, the last
reversed preamble and the first forward preamble are exactly `SYNC_LEN` apart.
The matcher discards opposite words only when they are closer than one
preamble length (`transport3.OPPOSITE_WORD_SPACING`), so both packets are
acquired. Both preambles usually arrive in the same wake, and the forward one
does not yet complete a forward packet, so the receiver decodes the reversed
packet in that wake (`v7_live_input.select_packet_hit`). At a
forward-to-reverse turn, the last forward packet is not decoded: the receiver
next wakes on the reversed packet's trailing preamble, one packet later, and
shows that newer packet instead. `modem_tests/test_v7_reverse.py` pins both on
the default wire.

The paired synthetic CPU check is reproducible with
`.venv/bin/python test_modem_v7/reverse_cpu.py`. On a warmed 4,048-sample
interior-packet window (500 paired batches of five packets, 48 kHz,
tone-seeded timing), automatic direction detection added 1.75% over
forward-only detection and decoding; reverse decode added 1.61% over the
auto-detected forward path. These local CPU-time measurements include
acquisition and one packet decode, but exclude audio capture, rendering, and
real-medium effects.

## 3. OFDM body and stereo mapping

| Parameter | Current value |
|---|---:|
| Reference sample rate | 48,000 Hz |
| FFT size / cyclic prefix | 128 / 16 samples |
| Symbol duration | 144 samples (3 ms at 48 kHz) |
| Symbols per body | 24 |
| Data and pilot carriers | FFT bins 4–34, or 1.5–12.75 kHz |
| Emission shaping | 63-tap Kaiser FIR, nominal 14 kHz edge |

The body uses a 128-point real FFT/IFFT per symbol. Bins 4 and 34 are
continual pilots. Scattered pilots use bins 5, 9, 13, 17, 21, 25, 29, and 33,
each on one of three symbol phases. There are 79 data blocks after pilot
reservation.

Each data cell carries mid/side components:

```text
L = (M + S) / sqrt(2)
R = (M - S) / sqrt(2)
```

The 13 blocks on bins 5–9 are M-only. The remaining 66 blocks carry M and S,
each with I and Q components. Thus mono summing retains the M content and loses
S content. A one-leg polarity retry is made by the decoder when ordinary
acquisition finds no packet; the two-channel equalizer handles the resulting
polarity correction. Mono input is represented internally as a shared M
observation rather than an invented S channel.

## 4. Image coding, ranks, and model tables

The prepared source canvas is 80×96 pixels. V7 samples Y on a 96×80 grid and
Cb/Cr on 48×40 grids. The transmitted DCT corners are 48×40 for Y and 24×20
for each chroma plane, for 2,880 coefficients total. The image values use
Pillow YCbCr conversion; the source coder performs the DCT once and reconstructs
omitted coefficients as zero/model mean.

The canonical profiles are `nearest`, `box`, `lanczos`, and `bicubic`. Their
variance, mean, rank, gain, level, and phase data are stored in
`animation_modem/v7_model_tables.npz`; the expected SHA-256 is
`2392943a287fdf1cd2ac773c2fc065179006bfab4646726e72e023f3921ffb1c`. The
canonical variance tables were derived using seeded crops/flips of
`modem_tests/fixtures/v7_reference_face.png`; they are not measured over the
baked image library. Canonical sender and receiver load the same checked-in
tables. A custom model can be derived from an image, but both ends must use
matching model data.

Coefficients are ranked by the frozen variance table and placed in three
tiers:

| Rank range | Count | Per-packet carriage |
|---|---:|---|
| Head | 0–207 | 208, in M-only blocks |
| Body | 208–2,223 | 2,016, in stereo-capable blocks |
| Tail | 2,224–2,879 | 96 of 656, rotated over seven slices |

Each packet therefore carries 2,320 coefficient slots; all 2,880 ranks are
refreshed over seven tail slices. The counter modulo seven selects the slice.
Receiver tail memory retains received tail values until their slice is
refreshed, then falls back to the model mean; `--no-tail-memory` disables it.
The eight values of each component group are spread across symbols with a
normalized 8-point Hadamard transform. Stereo slots are ordered with all M
slots before S slots, so mono playback drops a low-priority suffix of the rank
sequence instead of holes throughout the picture.

## 5. Metadata and loop fields

The metadata symbol uses 11 known pilots on bins 4, 7, 10, …, 34 and 20 QPSK
data cells for a five-byte word. Its payload and check field are:

| Bits | Meaning |
|---|---|
| Byte 0, bits 7–5 | Aspect code |
| Byte 0, bits 4–3 | Source encoding: nearest, box, lanczos, or bicubic |
| Byte 0, bits 2–0 | Tail slice, 0–6; 7 is invalid |
| Bytes 1–2, bit 15 | Playback direction (0 up, 1 down) |
| Bytes 1–2, bits 14–0 | One-based source index; zero is invalid |
| Bytes 3–4 | CRC-16/CCITT-FALSE of bytes 0–2 XOR a rotating mask |

CRC uses polynomial `0x1021`, initial value `0xFFFF`, and no final XOR. Source
indices are zero-based in application APIs and encoded one-based, with a
maximum source index of 32,766. Metadata pilots are self-referenced; the
receiver estimates the metadata symbol's response before selecting a
non-default source-encoding model. An invalid metadata word does not update
the selected source, aspect, or tail slice.

Slices 0–4 carry an ordinary CRC. When loop information is supplied, slice 5
XORs the CRC with loop phase `p`, and slice 6 XORs it with loop length `N`;
the top bit of the length field marks a one-way loop. The receiver learns
these masks and then checks the corresponding slices against them. A correct
receiver clock can use `ticks = unix_ns × 30 // 1e9` and the recovered loop
fields to compute the loop index and picture lag. MIDI-clock operation has no
wall-clock phase.

Known loop-lock limitation: the implementation counts matching observations
for a tail slice but does not require distinct packet counters. A tone-assisted
metadata retry can submit the same packet twice to the lock. A one-packet probe
using tail slice 5 learned `p` from those two reads; distinct-packet
confirmation is therefore not guaranteed by the current code.

## 6. Timing-reference tones and playback speed

The live sender's optional reference tones occupy FFT bins 1 and 3, leave bin 2
clear, are M-only, and are −20 dB relative to body RMS. They are separate from
the OFDM pilots. The live sender enables them by default. Receiver timing
options are `baseline`, `tone-seeded`, `tone-joint`, and `tone-replaced`; the
standalone live receiver defaults to `tone-seeded`. Pilot timing can retry a
failed metadata decode using the measured tone phase. Tone-assisted channel
equalization (`m-reference`) and pulse-warp timing are opt-in; their defaults
are off/baseline.

### Coded pilot prototype

The standalone V7 live tools now transmit an experimental 12-chip status with
their default M=500 fold. Bin 1 stays steady; bin 3 alternates 12 known BPSK
chips and 12 status chips. Three balanced status codewords identify fold-off,
M=500, and M=1000 at minimum Hamming distance six. Chip transitions are shaped
inside the cyclic prefix. The live receiver decodes the chip signs and removes
them before its tone-assisted timing fit. The normal edge-counted pulse path
remains primary acquisition, and the fold signature still measures per-slot
noise. This prototype remains outside `animation_modem/`; `--baseline` on both
standalone endpoints selects the previous steady-tone, unfolded profile.

An initial no-tone ablation only tested compatibility, not coded metadata:
with the optional tones disabled, the synthetic live loopback still displayed
71 pictures in 6 seconds at both M = 500 and M = 1,000, with picture scores
within 0.2 points of the tone-on runs. In the offline 17-case stress run, most
conditions scored 20 frames; the 0.3% jitter case scored 19. This suggests the
current receiver can fall back when the tones are absent; it does not establish
that an on/off code can be decoded over tape or real audio hardware.

Packets may be pitch-shifted in the range 0.01×–4× (pulse scales 100×–0.25×).
The receiver measures the resulting pulse scale and resamples the packet onto
the reference grid. Slower playback reduces both packet rate and carrier
frequencies; faster playback increases both. With a 14 kHz nominal emission
edge, the full-band Nyquist guide is
`output_sample_rate / 28,000` (about 1.71× at 48 kHz and 3.43× at 96 kHz).
This is not a transmit limit; above it, resampling filters or aliasing remove
high-frequency detail. The application sender uses the output device's rate;
the standalone sender follows its DAC's native rate unless `--rate` is given.
Standalone receive capture is capped at 96 kHz.

## 7. Decoder behavior and live input

The decoder locates pulse headers, chooses the configured packet endpoint,
resamples the body, decodes metadata, then demodulates the OFDM symbols. It
fits the two-channel pilot response and per-symbol fade/noise, performs a
per-cell 2×2 MMSE equalization and grouped 8-value LMMSE reconstruction, and
applies coefficient confidence gates. The normal path uses fixed-size
128-point per-symbol real FFTs. Clock-cancellation templates are retained for
the separate clock-word stream; the current pulse-framed live path does not
transmit or cancel that old clock track.

The live input adapter:

- corrects a strongly inverted right leg and detects new headers
  incrementally;
- applies a bounded autoleveler before header search (gain 0.5×–32×, gradual
  rise and immediate reduction), because pulse and EOF thresholds are
  absolute;
- keeps bounded audio history and resets partial acquisition across input
  callback gaps;
- schedules decode work when a new header arrives and normally decodes only
  the newest complete packet.

The EOF-aware stream decoder can decode a single packet when given its marker.
The live input adapter, however, still waits for a new header before calling
the decoder. A finite live input ending immediately after an EOF marker can
therefore leave its last packet unprocessed. The live-input test explicitly
expects the last packet to be omitted when no following header arrives.

The live receiver distinguishes status from displayability. A frame that
passes the strict live acceptance gates (pulse confidence at least 0.45,
metadata accepted, head confidence at least 0.85, head coverage at least 0.75,
and maximum pilot-noise estimate at most 0.08) is accepted. Before loop masks
are learned, `metadata accepted` can include the implementation's provisional
metadata path. A weaker but displayable reconstruction may still be shown and
labeled degraded by the standalone UI. The display path holds its previous
picture when a result is neither accepted nor displayable; it has no
black-frame fallback.

## 8. Runtime and defaults

The standalone live sender (`tools/v7_live.py`) defaults to the M=500 luma
fold with coded pilots (section 10.7): box encoding profile, brightness 1.0,
gamma 1.0, 1× playback, and the EOF marker. `--baseline` selects the previous
nearest / brightness 1.05 / steady-pilot profile on both ends. The standalone
receiver defaults to the matching coded M=500 profile, EOF boundaries,
tone-seeded pilot timing, baseline pulse timing, tone equalization off, one
decode batch, one frame of history, and tail memory enabled. Both require an
explicit audio device.

The application sender (`main.py --mode modem`) does not fold yet. It sends
the nearest encoding profile with steady pilot tones and the EOF marker at 1×.
Moving it to the fold-500 coded default is planned (section 10.8).

The low-level encoder's pilot-tone and EOF-marker switches default off, and
the low-level decoder defaults to next-header boundaries and baseline pilot
timing. These defaults serve unit tests of the older framing; live adapters set
the live defaults explicitly, and bench tools use
`tools/v7_wire_profile.py`.

Numba is imported unconditionally by the V7 transport and pulse-acquisition
modules. The default equalizer and selected acquisition kernels use Numba JIT;
the standalone live receiver calls `warmup_equalizer()` before opening its
audio stream so first-use compilation does not stall capture. The
`--force-float32` path is an experimental arithmetic path, not an optional
Numba installation mode.

## 9. Verification evidence and limits

### Test inventory and routine checks

Run the modem regression suite from the repository root with:

```text
.venv/bin/python -m unittest discover -s modem_tests -v
```

The `modem_tests/` suite covers V7 transport, sender/receiver behavior, timing,
quality, impairments, tables, and shared modem support:

```text
modem_tests/test_clock.py
modem_tests/test_impairments.py
modem_tests/test_modem_clock.py
modem_tests/test_v7_aliasing.py
modem_tests/test_v7_capture_video.py
modem_tests/test_v7_decode_speed.py
modem_tests/test_v7_display.py
modem_tests/test_v7_encode_speed.py
modem_tests/test_v7_eof.py
modem_tests/test_v7_experimental_fold.py
modem_tests/test_v7_gl_viewer.py
modem_tests/test_v7_image_quality.py
modem_tests/test_v7_live_input.py
modem_tests/test_v7_loop.py
modem_tests/test_v7_metadata.py
modem_tests/test_v7_mono.py
modem_tests/test_v7_mono_torture.py
modem_tests/test_v7_perceptual_resize.py
modem_tests/test_v7_pilot_tones.py
modem_tests/test_v7_pulse_warp.py
modem_tests/test_v7_receiver_gui.py
modem_tests/test_v7_reverse.py
modem_tests/test_v7_speed.py
modem_tests/test_v7_tables.py
modem_tests/test_v7_tone_equalization.py
modem_tests/test_v7_torture_matrix.py
modem_tests/test_v7_wire_profile.py
```

The standalone coded-pilot prototype has a separate suite:

```text
.venv/bin/python -m unittest discover -s test_modem_v7 -p 'test_*.py' -v
```

Its current regression module is `test_modem_v7/test_tone_code.py`. On
2026-09-27, the modem suite passed **217 tests** and the prototype suite passed
**13 tests**. Rerun the relevant suite and report fresh results after changes;
these counts are a dated snapshot, not permanent expectations. Both suites are
synthetic/unit evidence, not a real device or tape test.

For CPU measurements, run `test_modem_v7/cpu_smoke.py`. Always report the new
feature beside its matched baseline, with absolute time and delta/percentage
from the same machine and workload; do not present a feature-only timing.
Measurements are informational, not pass/fail thresholds. See
`test_modem_v7/HOWTO.md` for invocation and scope.

For slow reverse playback, run
`.venv/bin/python test_modem_v7/slow_reverse_torture.py`. It stores one fixed
eight-packet reference/mirror series in `tmp/v7_slow_reverse_series.npz`, then
tests that identical packet payload series from 0.25× through 0.01× through the
chunked `LiveInput` acquisition and reverse decoder. At 48 kHz, three warmed
runs on 2026-09-27 received all eight frames at every speed, with validated EOF
and source order 7→0. The series SHA-256 was
`044b010d38f604aae0e30caffe4ca0b1630295234450ae0c2f3241ef435ff41b`.

| Playback speed | Pulse acquisition CPU / 8-packet series | Reverse decode CPU / packet | Live-input CPU / one core |
| ---: | ---: | ---: | ---: |
| 0.25× | 1.89 ms | 0.98 ms | 0.82% |
| 0.10× | 3.15 ms | 1.05 ms | 0.91% |
| 0.05× | 5.22 ms | 1.12 ms | 1.40% |
| 0.025× | 11.67 ms | 1.12 ms | 2.36% |
| 0.01× | 25.34 ms | 1.38 ms | 5.63% |

The same fixed packets were also scored for image quality against their matching
reference/mirror source. The source was Lanczos-resized to 405×540; decoded
80×96 images were bicubic-resized to that same display size. SSIMULACRA2 is
higher-is-better. Pixel RMSE and PSNR use the displayed RGB images.

| Playback speed | Mean SSIMULACRA2 | RGB RMSE / 255 | RGB PSNR |
| ---: | ---: | ---: | ---: |
| 0.25× | −35.575 | 11.949 | 26.58 dB |
| 0.20× | −35.572 | 11.948 | 26.58 dB |
| 0.15× | −35.588 | 11.949 | 26.58 dB |
| 0.10× | −35.573 | 11.949 | 26.58 dB |
| 0.075× | −35.572 | 11.949 | 26.58 dB |
| 0.05× | −35.572 | 11.949 | 26.58 dB |
| 0.025× | −35.570 | 11.948 | 26.58 dB |
| 0.0125× | −35.571 | 11.948 | 26.58 dB |
| 0.01× | −35.571 | 11.948 | 26.58 dB |

Quality is effectively invariant with playback speed in this clean synthetic
case: normalized source-value RMSE stays at 0.068049, and the 0.01× display
differs from the 0.25× display by only 0.18/255 RGB RMSE across corresponding
frames. The roughly −35.6 SSIMULACRA2 and 26.58 dB PSNR describe this fixture's
baseline encode/reconstruction quality, not tape quality. This comparison
isolates whether speed changes fidelity; it does not establish that the image
encoding is optimal.

A fast-playback follow-up tested the same reverse series at 1×, 1.5× and 2×
with 96 kHz capture, comparing each result to a 1×/96 kHz baseline. All eight
frames passed EOF and metadata validation at each speed. Mean SSIMULACRA2 was
−35.574 at 1×, −35.619 at 1.5×, and −35.596 at 2×; normalized source-value
RMSE was 0.068050, 0.068053, and 0.068056 respectively. This indicates no
material quality change at 2× when the capture sample rate preserves the
expanded signal bandwidth. At 48 kHz, 1.5× also decoded all eight frames with
mean SSIMULACRA2 −35.645. At 2×/48 kHz only 1/8 frame had independently valid
metadata; the remaining frames were rejected and the live receiver holds its
last good picture. This matches the transport's Nyquist guidance: 2× playback
at 48 kHz pushes the fast wire's full band beyond the capture bandwidth; 96 kHz
has sufficient Nyquist headroom for this synthetic case.

An additional 2×/48 kHz clean-reverse probe found all eight pulse headers and
EOF markers but only one independently valid metadata frame. Before playback
speed conversion, the same source wire was optionally filtered with a
zero-phase fourth-order Butterworth low-pass at 14, 12, 11, 10, or 8 kHz. The
14–10 kHz settings still produced only one valid frame; 8 kHz produced none.
This simple prefilter did not recover the packets. The speed converter already
uses a polyphase anti-alias filter when compressing the waveform to the 48 kHz
capture clock; that prevents out-of-band energy from folding back, but it also
removes high wire carriers that the fast playback needs. At 96 kHz, 2× puts the
nominal 14 kHz emission edge near 28 kHz, below the 48 kHz Nyquist limit, so the
full-matrix failures there are due to the specified fixed-bandwidth impairments
(notably Type I/II and low-pass cases), not ordinary sample-rate aliasing. A
more aggressive low-pass is therefore not a general 2× fix; use sufficient
capture bandwidth or deliberately design a lower-bandwidth wire profile.

The folding diagnostic is reproducible with
`.venv/bin/python test_modem_v7/alias_fold.py`. It synthesizes a 28 kHz
component and 56/84 kHz harmonics at 192 kHz, then downsamples to 48 kHz. Naïve
decimation folds these to 20/8/12 kHz at their original amplitudes; polyphase
anti-alias resampling suppresses those aliases by about 74–76 dB. On the V7
reverse packet probe, 2×/96 kHz receives 8/8 frames in order; 2×/48 kHz with
polyphase filtering receives 1/8; deliberately unfiltered decimation receives
0/8. All three retain 8/8 pulse hits and EOF validations. This distinguishes
folded harmonic energy from high-carrier removal: allowing aliasing does not
recover the V7 metadata/picture data.

That script also tests candidate sender-wide fourth-order zero-phase LPFs at
13, 15, and 18 kHz before speed conversion. They all leave clean 2×/48 kHz
recovery at 1/8 and clean 2×/96 kHz at 8/8. A separate clean sweep over 1.7×–2.2×
found 15 kHz made a small improvement at 1.9×/48 kHz (SSIMULACRA2 −42.048 vs
−42.366) but still decoded only 8/8 headers with no robustness margin; at 2×
and above, all three cutoffs left recovery unchanged. The 18 kHz filter dropped
1.9×/48 kHz to 5/8 valid frames. At 1×, the 13 kHz filter reduced SSIMULACRA2
by about 0.06–0.09 points. There is no consistent fast-speed recovery gain to
justify a sender-wide cutoff; it trims useful wire energy at ordinary speed.

The clean reverse series was swept more finely from 1.7× through 2.2×. At
48 kHz, all eight frames remained independently valid through 1.8×; quality
then fell at 1.9×, only one frame was valid at 2.0× and 2.1×, and none at
2.2×. At 96 kHz, all eight frames decoded in order at every tested speed with
stable quality:

| Speed | 48 kHz valid frames | 48 kHz mean SSIMULACRA2 | 96 kHz valid frames | 96 kHz mean SSIMULACRA2 |
| ---: | ---: | ---: | ---: | ---: |
| 1.7× | 8/8 | −35.677 | 8/8 | −35.590 |
| 1.8× | 8/8 | −35.696 | 8/8 | −35.625 |
| 1.9× | 8/8 | −42.366 | 8/8 | −35.643 |
| 2.0× | 1/8 | −46.159 | 8/8 | −35.596 |
| 2.1× | 1/8 | −49.669 | 8/8 | −35.645 |
| 2.2× | 0/8 | n/a | 8/8 | −35.594 |

Every point still acquired 8/8 reverse headers and validated 8/8 EOFs at both
rates; failures at 48 kHz were metadata/picture recovery, not scale estimation.

The 96 kHz full impairment matrix was also sampled at those six speeds. In
forward playback, cases with all 11 metadata-valid results / received totals
were: 1.7× 23/25 and 256/275; 1.8× 23/25 and 253/275; 1.9× 22/25 and 249/275;
2.0× 21/25 and 242/275; 2.1× 21/25 and 239/275; 2.2× 20/25 and 233/275.
The reverse EOF matrix met its 10-picture cold-start acceptance in 21/25,
22/25, 20/25, 21/25, 19/25, and 19/25 cases respectively, with 228, 229, 220,
221, 208, and 204 received frames out of 300. Failures accumulate mainly in
low-pass and Type I/II combinations as speed rises. Clean frames remain intact
at 96 kHz, confirming that these impairment-matrix failures are not caused by
sample-rate aliasing at that capture rate. Results are saved under
`tmp/v7-torture-speed-1_7x/` through `tmp/v7-torture-speed-2_2x/` and
`tmp/v7-torture-reverse-1_7x/` through `tmp/v7-torture-reverse-2_2x/`.

Live-input CPU includes 1,024-sample block ingestion, rolling-buffer work,
incremental pulse acquisition, and packet decode, normalized by the series'
media duration. It is a single-machine synthetic measurement, not an audio-device
or tape benchmark. The same warmed benchmark's 30-second unknown-speed silence
baseline used 2.22% of one core while retaining the full 882,000-sample startup
cap. At 0.01×, isolated packet demodulation is only about 1.4 ms; the higher
continuous live percentage is chiefly input-buffer maintenance and pulse
acquisition, not slow OFDM work.

As a separate audio-stack check, the last packet (source index 7) from this
same saved series was reversed and captured through a 48 kHz PulseAudio null
sink at 0.1× and 0.01×. Both acquisitions and EOF validations passed; warmed
decode ran 15/15 times at each speed. Median CPU was 1.03 ms (p95 1.10 ms) at
0.1× and 1.38 ms (p95 1.45 ms) at 0.01×. The one-second leading/trailing
silence absorbs duplex startup latency; it is needed so capture startup does
not clip the reverse packet's EOF marker. This validates the PulseAudio path,
not physical audio hardware.

For repository test-selection cautions, follow `AGENTS.md`; in particular, do
not blanket-discover `tests/`, which includes interactive and audio-device scope
tests.

**Concurrent-work note (2026-09-26):** another thread is optimizing calls to
`main.py`. Treat that work as separate from V7 transport changes; inspect its
existing diff before touching `main.py` and do not overwrite or fold those
changes into a modem commit without coordination.

The shared image fixture is `modem_tests/fixtures/v7_reference_face.png`.
The synthetic impairment matrix runner is `tools/v7_torture_matrix.py`.

The synthetic impairment matrix was run with:

```text
.venv/bin/python tools/v7_torture_matrix.py --out tmp/v7-spec-rerun
```

It encodes 12 identical reference-fixture packets, resamples the wire from
48 kHz to 96 kHz, and applies 25 deterministic synthetic cases (seed 2026),
including clean, low-pass, hiss, wow/flutter, crosstalk, track imbalance,
dropouts, NR pumping, saturation, mono sum, and one-leg-only paths.

**Default wire.** The matrix tests the wire the senders emit: the M=500 fold
with coded pilots and the EOF marker, decoded with the coded-pilot timing
hook, tone-seeded timing and EOF boundaries (`tools/v7_wire_profile.py`). The
EOF marker commits every packet, so forward acceptance requires 12 results,
12 metadata-valid and 12 displayable. On 2026-09-27 all 25 cases passed. The
lowest `received` counts were `mains-buzz` (9 of 12), `hiss-35` and `dropouts`
(11 of 12); every other case received all 12.

**Historical baseline.** `--profile baseline` reruns the older nearest /
no-tone wire with next-header boundaries (EOF only for `--direction
reverse`). Forward, it returns 11 frames, because the final packet has no
next-header witness, and its acceptance requires 11 results, 11
metadata-valid and 11 displayable. The saved baseline run failed only
`lowpass-4k` (11 frames, 10 received, 1 lost, 10 metadata-valid, 11
displayable): a synthetic threshold failure, not a claim that the image was
wholly undecodable. `--pilot-ab` compares steady tones and always uses the
baseline wire. The speed and slow-reverse tables below were measured on this
baseline wire; rerun them with `--profile baseline` to reproduce them.

`tools/v7_mono_torture.py` uses the same default wire. The allocation and grid
benches (`tools/v7_luma_budget.py`, `tools/v7_grid_comparison.py`) derive
their own models, which the pinned fold tables do not cover, so they use the
live framing (EOF marker, coded pilots with a fold-off status, tone-seeded
timing, EOF boundaries) with the fold off. Every one of these tools accepts
`--profile baseline`.

The saved matrix output is `tmp/v7-spec-rerun/results.json`.

The same 25-case matrix was repeated at 1×, 1.5× and 2× playback at its
96 kHz sample rate, using the same seed and 12 packets per case. The runner now
accepts `--speed`; for example:

```text
.venv/bin/python tools/v7_torture_matrix.py --profile baseline --speed 2 \
  --out tmp/v7-torture-speed-2x
```

| Speed | Cases with 11/11 metadata-valid | Received / 275 | Clean mean value RMSE | Clean image SSIM |
| ---: | ---: | ---: | ---: | ---: |
| 1× | 24/25 | 270/275 | 0.067602 | 0.982180 |
| 1.5× | 23/25 | 257/275 | 0.067616 | 0.982214 |
| 2× | 21/25 | 242/275 | 0.067602 | 0.982182 |

At 1×, `lowpass-4k` is the existing known metadata failure. At 1.5×,
`lowpass-4k` and `type-i` fail full metadata recovery (`type-i` receives 4/11).
At 2×, `lowpass-10k` receives 8/11, `type-i` 0/11, `type-ii` 4/11, and
`lowpass-4k` 0/11. Their missing metadata causes those packets to be rejected;
the receiver retains its last good picture. For cases with all 11 metadata
frames, paired mean image-SSIM change versus 1× was +0.00063 at 1.5× and
+0.00026 at 2×, so the primary fast-speed regression is robustness on the
combined filtering/impairment cases rather than a general clean-image quality
shift. The baseline `dropouts` and `mains-buzz` cases also have partial received
counts despite metadata-valid frames, as shown in the per-case results.

These are the matrix's **forward-playback** cases at 96 kHz. Reverse-direction
validation remains the separate EOF-marked receiver path described above; these
matrix counts should not be read as reverse-playback torture results. Outputs
are in `tmp/v7-torture-speed-1x/`, `tmp/v7-torture-speed-1_5x/`, and
`tmp/v7-torture-speed-2x/`.

The matrix now also supports `--direction reverse`; that mode encodes EOF,
reverses the samples, runs bidirectional pulse acquisition restricted to the
reverse word, and decodes each hit through `decode_reverse_packet`. A 96 kHz
slow-reverse run used 12 packets per case and the same 25 cases and seed at
0.25×, 0.1×, 0.05×, 0.025×, and 0.01×. Each case is accepted when all 12 pulse
hits and EOFs validate, at least 11 metadata words validate, and at least 10
frames are received/displayable. The two-picture allowance is the cold-start
hold for an invalid and a provisional rotating-CRC slice.

| Reverse speed | Cases meeting reverse acceptance | Received / 300 | Clean RMSE | Clean image SSIM |
| ---: | ---: | ---: | ---: | ---: |
| 0.25× | 21/25 | 221/300 | 0.067618 | 0.981333 |
| 0.10× | 17/25 | 189/300 | 0.067618 | 0.981333 |
| 0.05× | 18/25 | 186/300 | 0.067618 | 0.981333 |
| 0.025× | 17/25 | 176/300 | 0.067618 | 0.981333 |
| 0.01× | 18/25 | 186/300 | 0.067618 | 0.981333 |

Every clean reverse case found all 12 hits, validated all 12 EOF markers, and
received 10 pictures; clean attempted-frame fidelity was stable through 0.01×.
The matrix's aggregate `image_quality` currently includes coefficients from
results marked lost, so it is not yet a score of only the pictures displayed by
the live receiver. The per-frame received/lost counts are authoritative for
availability; add a received-only/hold-last-good score before using this metric
to compare visible-picture quality. Additional acceptance failures were:

| Speed | Cases failing beyond the two-frame cold-start allowance |
| ---: | --- |
| 0.25× | `dropouts`, `type-i`, `mains-buzz`, `highpass-300` |
| 0.10× | `wow-flutter`, `bias-leak-30k`, `dropouts`, `type-i`, `type-ii`, `fast-flutter`, `mains-buzz`, `highpass-300` |
| 0.05× | `wow-flutter`, `dropouts`, `type-i`, `type-ii`, `fast-flutter`, `mains-buzz`, `highpass-300` |
| 0.025× | `wow-flutter`, `dc-hum`, `dropouts`, `type-i`, `type-ii`, `fast-flutter`, `mains-buzz`, `highpass-300` |
| 0.01× | `wow-flutter`, `dropouts`, `type-i`, `type-ii`, `fast-flutter`, `mains-buzz`, `highpass-300` |

Some failures are acquisition/EOF losses (`type-ii`, `highpass-300`); others
retain timing and EOF but lose metadata or valid pictures under severe
impairment. No uniform monotonic trend appeared across speeds; this synthetic
matrix changes how fixed-frequency filtering and interference overlap the
slowed wire. Results are in `tmp/v7-torture-reverse-0_25x/` through
`tmp/v7-torture-reverse-0_01x/`.

A separate synthetic 3.4 kHz low-pass probe returned 11 frames, with zero
metadata-valid/received frames and all 11 tagged lost; the decoder's
displayable diagnostic was true for those results. This probe and the 4 kHz
matrix result do not establish telephone-band support. No telephone-band
reliability guarantee is made by this specification.

Additional synthetic codec round trips used a 16-packet pulse stream carrying
the same reference image in every packet, decoded with tone-seeded timing and
EOF framing. MP3 used FFmpeg's `libmp3lame`; the ATRAC encoder was built locally
from `atracdenc`, with FFmpeg used to decode ATRAC3 and ATRAC3plus. ATRAC audio
was encoded at 44.1 kHz and resampled back to the 48 kHz wire rate before V7
decode. The results were:

| Impairment | Received | Lost | Metadata-valid | Displayable |
| --- | ---: | ---: | ---: | ---: |
| MP3 320 kbit/s CBR | 16/16 | 0 | 16/16 | 16/16 |
| MP3 LAME V0 | 15/16 | 1 | 15/16 | 16/16 |
| Dolby-B-like synthetic model: no decode, matched decode, and ±3 dB mistrack | 16/16 in each case | 0 in each case | 16/16 in each case | 16/16 in each case |
| ATRAC1 SP (292 kbit/s) | 0/16 | 16 | 14/16 | 16/16 |
| ATRAC3 LP2 | 0/16 | 16 | 3/16 | 15/16 |
| ATRAC3plus | 16/16 | 0 | 16/16 | 16/16 |
| ATRAC3 LP4 (64 kbit/s) | 0/15 | 15 | 0/15 | 14/15 |
| Dolby-B-like model followed by ATRAC1 SP | 0/16 | 16 | 14/16 | 16/16 |

These are small, single-fixture synthetic probes, not a broad codec matrix.
The Dolby-B-like cases use a simplified sliding-treble-gain model, not a Dolby
encoder/decoder or hardware. “Displayable” is a decoder diagnostic and does
not mean the packet passed metadata validation or was received. The ATRAC
encoder's own CTest suite passed all 23 tests; that validates the local codec
build, not V7 robustness on real ATRAC recordings.

These codec probes were one-off diagnostics, not checked-in regression tests.
Their generated inputs, encoded files, and local ATRAC build were kept under
`tmp/v7-codec-tests/`, `tmp/atracdenc-src/`, and `tmp/atrac-build/`.
The ATRAC encoder CTests were run from the repository root with
`ctest --test-dir tmp/atrac-build --output-on-failure`.

Real-media validation remains pending: actual tape/deck captures and later
analysis notes are required before synthetic results can be treated as
evidence of real-media performance.

## 10. Fold proposal

**Status: experimental live profile, not integrated into the production
transport.** `tools/v7_live.py` defaults to coded M=500; `--baseline` on both
ends restores the prior profile, and `--experimental-fold M` selects another
pinned size (see 10.7). The
measurements below are synthetic. They come from `test_modem_v7/`, run on 11 frames of an 810×1080
portrait face video: frames 1, 3, 5, … fit the statistics and frames 2, 4,
6, … are scored. Each reconstruction is scored with SSIMULACRA2 at 405×540
against the source at 405×540 (higher is better; about 90 is visually
lossless). No real tape or deck has been tested.

### 10.1 Idea

V7 sends 2,880 coefficients as analog values. On a clean path each slot
arrives with far more precision than the picture needs, while detail beyond
the 48×40 Y corner is not sent at all. Linear rearrangements cannot move that
spare precision into resolution: for a linear analog code, sending the
highest-variance coefficients is already mean-square optimal. A nonlinear 2:1 mapping
can. It trades SNR for resolution the way FM trades bandwidth for SNR, and it
has the same kind of threshold.

The **M weakest Y slots of the body tier** each carry two Y coefficients.
With `h` the slot's own coefficient (the host) and `u` the most important Y
coefficient that V7 does not send today (the guest), both normalised to unit
variance:

```text
s = D·round(h / D) + β·clip(u, −2.5, 2.5)        β = 0.8·D / 5
```

- **Host:** sent coarsely, as a multiple of the step D.
- **Guest:** rides inside the step as a small analog residual.
- **Scaling:** `s` is scaled by `1/sqrt(1 + D²/12 + β²)` and replaces the
  host coefficient at the host's own variance. Slot power, slot layout,
  Hadamard spreading and the equaliser are therefore unchanged: a normal V7
  encode of the modified coefficients is a folded packet.
- **Host slots:** Y only, ranks 208–2,223 (the body tier), taken from the
  weakest end. Never the head, never the rotating tail, never Cb/Cr.
- **Guests:** Y coefficients of the 96×80 grid outside the 48×40 corner,
  in fitted-variance order.
- **Step D:** chosen for a 30 dB design SNR from training frames.
  - Offline harness (`fold_modem.py`): D is re-chosen from each run's
    training frames. On the reference run that gave 0.97 for both M; the
    10.2 results use it.
  - Live prototype: the frozen tables are the authority, fitted on the
    reference fixture. D = 0.970 in `fold_table_500.json` and 0.8247 in
    `fold_table_1000.json`; the 10.7 results use these.

The receiver:
1. takes the equaliser's per-coefficient estimate and confidence for the host
   slots;
2. removes the MMSE shrink by dividing by the confidence;
3. rounds to the step, giving the host;
4. reads the remainder as the guest;
5. rebuilds Y on the full 96×80 grid.

**Fallback:** a host whose equaliser confidence is below 0.9 is read as a
plain (noisy) host, and its guest is dropped.

### 10.2 Measured results

Real V7 modem, 1×, box encode filter, current slot layout, still pictures,
mean over every decoded steady packet:

| Condition | No fold | Fold 500 (+fallback) | Fold 1,000 (+fallback) |
|---|---:|---:|---:|
| Clean | −16.9 | −7.0 | **−4.1** |
| Low-pass 12 kHz | −17.0 | −7.0 | **−4.3** |
| Low-pass 10 kHz | −17.0 | −7.1 | **−4.3** |
| Dropouts (12 ms every 0.7 s) | −17.3 | −7.6 | **−4.7** |
| Wow/flutter (0.45 % at 0.55 Hz, 0.12 % at 7.3 Hz) | −17.7 | **−10.3** | −10.4 |
| Random speed jitter 0.1 % RMS, 20–300 Hz | −19.5 | **−16.3** | −18.3 |
| Fast flutter (adds 0.1 % at 25 Hz, 0.05 % at 60 Hz) | −19.4 | −19.8 | −21.5 |
| Fast flutter + 12 kHz low-pass + dropouts | −24.0 | −24.5 | −26.8 |
| Random speed jitter 0.3 % RMS | −33.0 | −43.1 (**−34.3**) | −49.8 (−43.4) |

- **Low-pass and dropouts** do not affect folding. The Hadamard spreading and
  the equaliser absorb them before the folded symbols are read.
- **Fast flutter and jitter** hurt. Timing smear moves symbols across a step.
  The fallback has a measurable effect only in the 0.3 % jitter case, where
  it brings fold 500 back to within 1.3 points of no folding.
- **Contact sheets:** where folding scores even, it looks different: fine
  grain instead of blur.

On the simulated channel (per-slot noise, below the 30 dB design point), Y
folding costs about 3.5 points (M = 500) to 6 points (M = 1,000) at 20 dB,
and 5–10 points at 15 dB.

Folding into Cb/Cr slots raised the mean colour error (CIEDE2000) from about
3.4 to 7–11 and is excluded. With Y-only hosts, colour error is unchanged.

### 10.3 Recommendation

**M = 500 with the fallback.** It is now the default profile in the standalone
V7 live prototype, with `--baseline` to restore the prior profile. The earlier
measurements show about +10 on clean, low-pass and dropouts, +7 with slow wow,
and −0.4 to −1.3 at worst (fast flutter, combined impairments, 0.3 % jitter).
M = 1,000 gains more on clean paths but loses 2–3 points under flutter; it
remains an explicit experiment rather than the default.

This matches the recovery priority in `AGENTS.md`: colour is unchanged, and
on bad paths only detail degrades.

### 10.4 Proposed implementation

1. **Frozen fold table.** Freeze M, D, the host list, the guest positions and
   the guest variances into the model tables, under the same hash check as
   the canonical tables. Sender and receiver must not depend on local frames
   for these statistics.
2. **Sender.** Apply the fold to the coefficient vector inside
   `encode_frame_coeffs()`, before gains and slot placement. It applies to
   `modem_v7_display.py` and `tools/v7_live.py send`.
3. **Receiver.** Unfold inside `decode_frame()` from the equaliser's `xhat`
   and `conf` for the host slots, apply the fallback, then reconstruct Y on
   the 96×80 grid. Hosts are in the body tier, so tail memory is unaffected.
4. **Signalling.** The 2-bit source-encoding field is fully used by the four
   encode filters. Options:
    - reassign one filter code (for example `bicubic`) to "box + fold";
    - extend the metadata word;
    - for the standalone live prototype, set the same fold profile on both
      ends; it defaults to coded M=500 and `--baseline` restores the prior mode.

    The live prototype uses an in-band signature and coded fold-size status
    (10.7), so normal packets can pass through and a signature-missing packet
    can be routed to its pinned table.
5. **Tests.** `modem_tests/test_v7_experimental_fold.py` covers the
   prototype:
   - noiseless round trip (hosts within half a step, guests exact);
   - signature detection;
   - pass-through of normal packets and of packets folded with another
     table;
   - the confidence fallback, the noise gate and guest weighting;
   - fail-closed table loading;
   - receiver hooks restored after a failed run.

   A production implementation would add a regression that an unfolded wire
   decodes identically.

### 10.5 Related measurements that need no folding

Same harness, same frames:

- **Encode filter.** `box` beats today's default `nearest` by 4.8–7 points
  on every tested case through the real modem (clean −23.9 → −16.9; Type II
  −27.5 → −20.9; dropouts −24.6 → −17.8), with no wire change. The receiver
  already selects the model from the metadata.
- **Y-first coefficient selection.** Choosing the 2,880 coefficients by
  variance across all three planes keeps about 2,620 Y and 260 Cb/Cr
  coefficients, instead of 1,920 and 960.
  - With box sampling, it scores −16.8 → −5.8 at 50 dB and −18.6 → −8.1 at
    25 dB (simulated channel).
  - Mean colour error (CIEDE2000) rises from 2.6 to 3.3.
  - It is a wire change, and it trades against the colour priority, so it
    needs a decision before adoption.
- **Prefiltering.** A smooth taper before the coefficient cutoff reduces
  ringing. At best it only matches today's box sampling, and on the Y-first
  selection it scores 3–7 points worse. Reconstructing on a finer grid
  without more coefficients does not help.

### 10.6 Reproducing

```text
.venv/bin/python -m pip install ssimulacra2 scikit-image
export NUMBA_CACHE_DIR=tmp/numba_cache
.venv/bin/python test_modem_v7/fold_modem.py      FRAME... [--quick]
.venv/bin/python test_modem_v7/compare_filters.py FRAME... [--quick]
.venv/bin/python test_modem_v7/layout_oracle.py   FRAME... [--quick]
```

Full instructions and the reference numbers are in `test_modem_v7/HOWTO.md`.
Results are written to `tmp/test_modem_v7/`.

### 10.7 Live prototype

`tools/v7_live.py send|receive` defaults to coded M=500. `--baseline` on both
ends restores the previous unfolded profile; `--experimental-fold M` selects
M=500 or M=1000 explicitly (M=0 is also accepted as the legacy baseline
spelling). Each folded profile loads the frozen
`test_modem_v7/fold_table_<M>.json` built from the reference fixture. The
sender folds each captured source frame exactly once in coefficient space
before pulse encoding and overlays the coded status. The
receiver despreads status chips before tone timing, retains each packet's
equaliser output, and unfolds before display. Its prototype hooks are restored
in `finally`; the fold-specific integration remains a standalone prototype.

Table loading fails closed:
- **Pinned files.** Each table file must match a SHA-256 pinned in
  `live_fold.py` and be in canonical form. A rebuilt table must be pinned
  deliberately.
- **One model per table.** A table records the digest of the model it was
  built for: the canonical `box` profile under the current model-table
  hash. The sender refuses to start with any other encode filter or
  fixture. The receiver shows packets from any other model without
  unfolding.

The prototype retains a signature. The 16 weakest host slots carry a ±3D
pattern instead of data. The pattern is drawn from the table's identity, and
it serves three purposes:
- **Detection.** Ordinary signature-based decoding unfolds packets that carry
  the selected table's pattern (score ≥ 0.5). A valid coded mode can authorize
  the matching pinned table when its signature is hidden; a normal packet
  without matching coded status is shown unchanged.
- **Noise measurement.** The pattern's residual measures the packet's symbol
  noise on exactly the folded slots. Timing smear from fast flutter and
  jitter shows up there, not in the equaliser confidence.
- **Weighting.** Guests are weighted by β²/(β² + noise²), and above 0.3 steps
  of noise the packet is not unfolded.

The coded status identifies fold size, not a revision hash; signature-missing
authorization assumes one pinned receiver table per fold size.

Measured on the live receive path (`live_fold.py selftest`: a moving
sequence, pilot tones, EOF, LiveInput blocks, live receiver arguments; the
same 11 frames). The table shows the change against no folding for M = 500
and M = 1,000:

| Condition | No fold | Change, M = 500 | Change, M = 1,000 |
|---|---:|---:|---:|
| Clean | −18.3 | +9.6 | +14.5 |
| Low-pass 10 kHz | −18.3 | +9.6 | +14.2 |
| Dropouts | −23.9 | +8.4 | +12.7 |
| Wow/flutter | −19.0 | +7.7 | +8.3 |
| Random jitter 0.1 % | −20.6 | +4.6 | +4.2 |
| Fast flutter | −20.8 | +2.2 | +0.6 |
| Fast flutter + 12 kHz low-pass + dropouts | −26.2 | +1.8 | +0.4 |
| Random jitter 0.3 % | −38.5 | −0.8 | −2.3 |

With the noise-weighted guests, folding no longer loses under fast flutter.
Without the weighting, M = 500 lost 2.4 points there.

Additive noise is the weak case, with the same change against no folding:

| Condition | No fold | Change, M = 500 | Change, M = 1,000 |
|---|---:|---:|---:|
| Hiss −45 dBFS | −19.7 | +5.0 | +4.6 |
| Type II tape model | −21.3 | +1.8 | 0.0 |
| Hiss −40 dBFS | −22.9 | −0.1 | −2.5 |
| Type I tape model | −29.5 | −1.6 | −4.1 |
| Hiss −35 dBFS | −31.9 | −1.8 | −4.1 |

Hosts are sent coarsely whatever the channel. The receiver can drop guests
on a noisy packet, but it cannot restore host precision. M = 500 therefore
costs up to about 2 points on noisy tape. Real tape remains untested.

Loopback through the real `tools/v7_live.py` sender and receiver
(`test_modem_v7/live_loopback.py`, one frame, clean): −18.7 unfolded,
−7.6 at M = 500, −2.5 at M = 1,000. Mismatches:
- A folded sender with a normal receiver scores −22.1.
- A folding receiver with a normal sender scores −18.7, unchanged.
- An M = 1,000 sender with an M = 500 receiver scores −25.2: the packets are
  shown unfolded, not unfolded with the wrong table.
- `--experimental-fold` with `--encode-filter nearest` refuses to start.

### 10.8 Planned: fold and coded pilot in the application sender

The standalone live tools send the fold-500 coded profile, but the application
sender (`main.py --mode modem`, `modem_v7_display.py`) still sends the
unfolded nearest profile. This plan moves the prototype into the transport
package so the application can send the same wire. Every step is gated by
paired CPU checks (percentage change in CPU time and wall time against a
matched baseline on each tested machine) and by picture quality on the
reference fixture.

1. **Move the runtime into `animation_modem/`.** This covers the table-driven
   fold codec (`FoldCodec.from_table`, encode and unfold), the pinned tables
   with their SHA-256 pins, and the coded-pilot overlay and status kernel
   (`add_tone_code`, `decode_tone_spectrum`, `coded_pilot_timing`). Table
   building and fitting stay in `test_modem_v7/`, which then imports the
   runtime from the package. The moved code has no application imports.
2. **Replace the monkeypatch hooks with decoder options.** The prototype wraps
   `v7.decode_frame`, the equalisers and `v7.pilot_tone_timing`. In the
   package, `decode_frame` returns the host-slot equaliser output when asked,
   and tone timing accepts recovered chip signs directly.
3. **Switch the application sender.** Make the fold-500 coded profile the
   default in `modem_v7_display.py`, with a baseline switch that restores
   today's profile. The box filter needs full compositing instead of the
   nearest-only fast path (`composite_nearest`), so measure the prepare-time
   change against `--modem-prepare-ms` first. Coordinate any `main.py` option
   change with the separate `main.py` work (section 9).
4. **Tests.** Carry `modem_tests/test_v7_experimental_fold.py` and the
   coded-pilot tests over to the package, and add an application-sender round
   trip that decodes fold-500 packets with the standard receiver.
5. **Docs.** Update section 8, `docs/MODEM_MODE.md` and
   `test_modem_v7/HOWTO.md`.

## 11. Perceptual sender preprocessing

Status: the first resize ablation is implemented as an opt-in experiment in
11.2. It remains disabled by default, and no mode is claimed to be generally
better or ready to replace the existing path. The later sender stages and
Section 12 remain planned experiments. Continue in the milestone order below;
do not enable an unmeasured stage by default. None of this work requires a new
wire format.

### 11.1 Goal, scope, and fixed contracts

Improve the decoded picture by preparing source content for the frequencies V7
actually retains. Favor recognizable small features, acceptable color, and
reduced ringing over nominal sharpness. All processing is classical and
deterministic. No learned models, random jitter, grain injection, region-based
packet allocation, or receiver changes belong in this work.

- Live integration: `tools/v7_live.py::_values`, before the current brightness
  and gamma adjustments. Keep their ordering and settings identical in paired
  comparisons. Share preprocessing through a self-contained modem module; do
  not import application settings into `animation_modem`.
- Bake integration: `utilities/convert_to_modem_dct.py`, where full-resolution
  layers are still available. Existing Lanczos-reduced layers cannot regain
  lost source detail through runtime preprocessing.
- Coefficient integration: inside the fold encoder's existing full-DCT path,
  before host/guest construction, not by transforming pixels back and forth.
- Preserve packet length, metadata, encoding codes, model tables, fold pins,
  synchronization, receiver behavior, and the 80×96 prepared canvas.
- Experiments declare canonical `box` encoding and use its pinned fold table.
  Reject a perceptual-preprocess/other-model combination with a clear error.
  The disabled path must retain existing outputs.

The grids are rows × columns: Y 96×80, Cb/Cr 48×40. Retained corners are Y
48×40 and Cb/Cr 24×20. Chroma is half the retained luma dimensions per axis,
one quarter its coefficient count per plane; it is quarter-sized per axis
relative to the luma sampling grid. Fold-500 adds selected outside-corner
guests, with signature slots reducing the number that carry picture data.
Read guest and signature counts from the loaded table, not a hard-coded claim
about an effective rectangular bandwidth.

The head/body/tail allocation is variance-ranked, not spatial-frequency sorted.
The 656 tail coefficients refresh in seven packets (about 0.57 s at 1×).
Any reference to a body or tail band must use actual `model.order` membership;
a radial low-pass does not isolate the tail.

### 11.2 First experiment: disentangle light handling and detail weights

Add a sender-only `--perceptual-resize` selection with these stable names:

| Value | Light domain | Resampling |
|---|---|---|
| `off` | Existing path | Existing selected encoding filter; default |
| `linear-box` | Linear RGB | Area averaging |
| `gamma-detail` | Gamma-encoded RGB | Detail-weighted area averaging |
| `linear-detail` | Linear RGB | Detail-weighted area averaging |

Use `off --encode-filter box` as the baseline. Use the standard piecewise sRGB
transfer function, not a power-2.2 approximation. Return linear-domain results
to gamma-encoded RGB before the existing YCbCr conversion. Linear averaging
preserves average light energy; it does not guarantee visible catchlights.
Detail weighting may improve visibility but is not energy preserving.

Implement a documented DPID-inspired kernel, without claiming reference-DPID
equivalence unless that algorithm is actually reproduced:

1. For each output pixel, compute exact source-pixel area-overlap weights and
   the area-weighted RGB mean in the selected light domain.
2. For contributing pixels, compute a scalar RGB distance from that mean:
   the square root of the mean squared difference across the three channels.
3. Multiply area weights by `1 + strength * min(distance / scale, 2)`, where
   `scale = max(weighted_RMS_distance, 1/255)` for normalized RGB.
4. Renormalize and average RGB with the same scalar weights for all channels.
   Strength zero is area averaging in that domain. Initial strength candidates
   are 0, 0.25, 0.5, and 1; do not automatically change strength per frame.

Use a separate `--perceptual-detail-strength` in [0, 1], default 0.25 for the
detail candidates. Freeze the selected value for an entire comparison. Keep
intermediate math floating-point and quantize only at the existing prepared
RGB interface for this first experiment. Do not simultaneously redesign the
YCbCr conversion, brightness adjustment, or capture resize.

**First-step implementation:** `animation_modem/perceptual_resize.py` provides
the exact-overlap area kernel and the three non-default modes. Linear modes use
the standard piecewise sRGB transfer; detail modes apply the RGB-distance
weighting above with shared channel weights. The Numba kernel caches per-axis
footprints. `tools/v7_live.py send` exposes `--perceptual-resize` and
`--perceptual-detail-strength`; preprocessing requires the canonical box model
and a matching pinned M=500 or M=1,000 fold table. `off` remains byte-identical
to the established selected-filter path and remains the default. The bake
converter is unchanged because bake integration is gated on a resize candidate
winning.

Precompute footprint geometry for repeated source sizes. Hot pixel loops use
Numba, with compilation excluded from steady-state measurements but reported
separately. Verify constants, bounds, identity-size input, unusual dimensions,
and deterministic output. Live captures around 160 pixels wide undergo roughly
2× reduction here, not the tenfold reduction possible in full-resolution bakes.

### 11.3 Statistics instrumentation before additional filtering

Collect these measurements with the initial resize comparison, without
refitting or changing the declared model:

- Per-plane and per-tier coefficient means and variances relative to canonical
  `model.mu` and `model.lam`; summarize ratios and the largest departures.
- Actual tail-slot energy, host normalized magnitudes, and guest normalized
  magnitudes before clipping, using the pinned `sd_guest` values.
- Guest fractions above 2σ and 2.5σ, plus maxima/percentiles. Exclude signature
  slots from picture-guest statistics and report their count separately.
- Unfolded-packet and unfolded-slot rates with explicit denominators.
- Decoded fidelity and send cost, using the same source/packet order as box.

Use held-out scoring frames; never fit and score on the same frames. The
canonical model is not refitted even on the training split in this experiment.
Decide whether the existing box tables are adequate from these measurements.
All four 2-bit profile codes are occupied. A new model or replacement profile
requires a separate compatibility decision and new pinned fold tables; do not
silently replace `lanczos`/`bicubic` or relax digest validation.

### 11.4 Later sender stages, independently switchable

**Band shaping.** Apply a fixed gain array to the full DCT before folding.
Define the array from DCT coordinates and frozen tier membership. Keep DC at
gain 1; start with luma-only gains. Test a small acutance lift independently
from a short cutoff taper. Do not taper recovered guests simply because they
are outside the rectangular corner. Record the exact gain array or generation
parameters in every result. No promise of halo elimination is made.

**Guest soft knee.** `FoldCodec.encode_coefficients(values)` currently accepts
spatial values and computes the DCT in `_split`; it is not an API accepting a
full coefficient vector. Refactor that internal boundary only as needed to
avoid a second transform. Apply the knee to normalized picture guests before
the existing safety clip and before constructing folded symbols. Leave host
quantization, signature symbols, fold power normalization and tables intact.
Initial knee: for magnitude `a <= 2`, output `a`; otherwise output
`2 + 0.5*tanh((a-2)/0.5)`, restoring the sign. This is continuous with matching
slope at 2 and bounded by 2.5. It is nonlinear compression, not a fixed gain.
The receiver intentionally displays the compressed value without expansion.
Measure error against original guests as well as clip counts: eliminating
clips by construction is not evidence of improved fidelity.

**Luma-guided chroma.** Experimental and luma-preserving. Define filtering in
linear RGB where light averaging is intended, then convert to the existing
gamma-domain YCbCr convention. Do not substitute linear-RGB-derived chroma
into that convention. Guide chroma reduction with coherent luma edges, bound
the guidance, and retain chroma-only boundaries. Test saturation compensation
separately, initially off. Evaluate after the 24×20 chroma cut and normal
receiver reconstruction, not just on prepared pixels.

**Change-adaptive softening, live only.** A frame difference is a change
detector, not a displacement estimate. Use timestamp-based attack/release
smoothing, bounded local strength, and explicit resets on source/shape changes
and cuts. Preserve static regions. Report actual tail-slot energy before/after
filtering; do not equate a spatial blur with removing tail ranks. This stage
must earn its place on motion sequences without worsening static flicker or
slow drifts. Final thresholds and time constants are experimental parameters
to record, not values the implementation agent should auto-tune invisibly.

### 11.5 Bake integration

Add an opt-in modem bake option only after a resize candidate wins. Filter
linear-light RGB premultiplied by alpha; filter alpha consistently and
unpremultiply only where alpha is nonzero. Transparent RGB must not leak into
visible edges. Preserve the existing bake format and dimensions and record
preprocessing settings in a backward-compatible manifest field.

Layers are composited after baking. Nonlinear detail weighting per layer is
not equivalent to filtering the final composite: evaluate representative
composites over several backgrounds, including fine translucent edges. Heavier
offline codec-in-the-loop optimization is deferred, not required for the first
bake implementation. Application-runtime coefficient shaping and knees are
gated on explicit application fold-profile support; do not assume that support
from the standalone prototype or a nonexistent section reference.

### 11.6 Evaluation and delivery gates

Existing evidence in `test_modem_v7/HOWTO.md` favors box over nearest by about
7 SSIMULACRA2 points and Lanczos by about 2 on its face sequence. A taper and
finer DCT reconstruction did not improve that reference run. These are starting
observations, not predictions for the new stages.

- Score decoded output at a fixed documented display size against the same
  Lanczos-resized source reference. Fix crop, aspect, brightness, gamma, and
  display reconstruction across sender comparisons.
- Use the face sequence plus fine bright/dark features, saturated boundaries,
  text, static frames, slow drifts, and cuts. Report per-case scores, not only
  an average. Report ΔE2000 with an explicit sRGB-to-Lab convention and fixed
  skin/lip regions, plus temporal error in source-defined static regions.
- Run the existing default-wire torture matrix unchanged; every required case
  must retain 12/12 packets. Also report unfold rates and picture fidelity;
  packet counts alone are insufficient. Synthetic impairments are regression
  checks, not claims about real tape.
- Named on-screen comparisons are sufficient: the user chooses the preferred
  result. No randomized or blind trial is required. Promote a candidate only
  after user preference plus improved metrics, or neutral metrics with a clear
  user preference. Report regressions rather than hiding them in averages.
- Pair warmed CPU time and wall time against box on each tested machine.
  The complete enabled live preprocessing path must add at most 25% to the
  send path in both measures. The historical roughly 1.4 ms/frame is context,
  not a portable threshold. Bake cost has no realtime limit.

The reproducible first-step report is written by
`test_modem_v7/perceptual_resize.py`; its JSON includes per-plane/tier
coefficient deviations, actual transmitted tail-slot energy, host and
pre-clipping guest magnitudes, guest threshold fractions, clean coefficient
unfold denominators, decoded fidelity, and paired warmed sender cost. A smoke
run on nine held-out frames from `images_sbs/face/00_C_BG_faceSource_960`
uses the SBS color panel, a fixed Lanczos capture proxy of 160×213, the frozen
box model, and the pinned M=500 table. In that run, `off --encode-filter box`
scored −19.38 mean SSIMULACRA2, 27.16 dB luma PSNR, and ΔE2000 3.039. The best
resize scores were around −17.94 (about +1.44 SSIMULACRA2), with luma PSNR
within 0.03 dB and ΔE2000 around 3.04. On Linux x86_64 with an AMD EPYC-Milan
CPU (4 logical CPUs), Python 3.13.5 and Numba 0.67.0, paired warmed sender cost
rose from 0.510 ms/frame to at least 0.795 ms/frame (+46%) and up to 2.183
ms/frame (+312%); process CPU deltas were similar, also exceeding the 25%
budget. The coefficient-perfect clean fold check unfolded 9/9 packets and
4,356/4,356 picture slots for every setting; it is not a tape or waveform
decode test. With this limited face-only sample and failed CPU budget, no
candidate is promoted and canonical box-table adequacy remains undecided.
Results and Numba caches stay under repo-local `tmp/`.

A profile of that run found four costs, and the resize was then made faster
without changing any output byte:

- **Linear-light quantization.** A binary search whose branches the CPU
  cannot predict cost about 1 ms/frame. A bin lookup with a one-step
  correction now finds the same code.
- **Gamma input conversion.** It divided every sample by 255. Both domains
  now convert through a 256-entry table.
- **The detail kernel.** It made three strided passes over every footprint.
  Each output row's contributors are now gathered once, and the three passes
  run as vectorized loops across the row. Shorter footprints are padded with
  zero weights, which add an exact +0.0.
- **The Pillow round trip.** The sender now hands captured RGB uint8 arrays
  straight to the resize.

The kernels are also cached on disk now. `modem_tests/test_v7_perceptual_resize.py`
pins every output byte, and the area and detail kernels' floats, against a
verbatim copy of the first implementation.

The table below is paired on a different machine from the run above: a 2-vCPU
Intel Xeon (Emerald Rapids), Python 3.11.15, Numba 0.67.0 and NumPy 2.4.4. It
is measured from a read-only 160×213 capture array with fold-500 coefficient
encoding included, over 400 rounds, as medians against `off` in the same
rounds. CPU time was within 1% of wall time.

| Setting | Before | After |
|---|---|---|
| `linear-box` | +221% | +29% |
| `gamma-detail`, strength 0 (box weights) | +48% | +11% |
| `gamma-detail`, strength 0.25 / 1.0 | +144% / +144% | +69% / +70% |
| `linear-detail`, strength 0.25 / 1.0 | +307% / +320% | +83% / +88% |

The gamma box-weight control is inside the 25% budget, and `linear-box` is
just over it. Both detail domains still exceed it: at strength 0.25 the
detail weighting still adds about 0.3 ms/frame over box weights in the same
domain on this machine. Compiling the resize
kernels at sender start fell from about 1.3 s in every process to about
0.3 s once they are cached. `test_modem_v7/perceptual_resize.py` now times the
same read-only array form the live sender receives. Its quality results
cannot change, since every output byte is the same.

`tools/v7_torture_matrix.py` now defaults to the wire the senders emit (fold
500, coded pilots, EOF marker; Section 9), where the Section 11.6 12/12 request
applies: every forward packet is committed by its EOF marker. The historical
baseline remains available as `--profile baseline`: 11 decoded results from 12
next-header packets, 11/11 metadata-valid and displayable in 24/25 cases, with
`lowpass-4k` at 10/11. The `--speed` and `--direction reverse` modes work with
either profile. Forward 1×/96 kHz conversion remains pinned byte-for-byte to
the original `resample_poly` path by `modem_tests/test_v7_torture_matrix.py`.

Delivery order: (1) resize ablations plus statistics; (2) decide table adequacy;
(3) band shaping; (4) knee; (5) guided chroma; (6) change adaptation; (7) bake.
Each stage has independent on/off comparison and must not invalidate earlier
gates. Keep experiment artifacts and Numba caches under repo-local `tmp/`.

## 12. Receiver display GUI and reconstruction modes

The receiver-shell implementation is available as
`.venv/bin/python tools/v7_live.py gui`. It opens in configuration without
opening an audio stream, enumerates input devices for explicit selection,
exposes every `receive` CLI option, starts/stops the existing receiver path,
and switches between Setup and Live. The Live view is image-first with optional
diagnostics; `P` or the Image only button hides all application UI, and `F`
toggles fullscreen. Fullscreen hides the toolbar, status strip, and diagnostics
on entry. The HUD reappears at the top edge or on key input, stays up while a
control is open, and hides again after two idle seconds. It reads
decoded values and live diagnostics directly from the receiver mailbox/status
callback. The display menu provides eleven presentation choices: nearest,
bilinear, sharp-bilinear, Mitchell bicubic, Spline36, Robidoux, Robidoux Sharp,
cubic B-spline, Kaiser-windowed sinc, Hann-windowed sinc, and EWA Jinc. Nearest
remains the initial display default and preserves the legacy 8-bit RGB
conversion and nearest-sampled shader path. All other modes reconstruct float
Y/Cb/Cr planes; selecting a mode does not alter transport decoding or decoded
exports. Float planes, the shader, and any filter intermediates are allocated
only when a non-nearest mode is selected.
Standalone image-sequence preview is available as
`.venv/bin/python tools/v7_viewer.py IMAGE...`.

Status: implementation brief, September 27, 2026. The receiver and preview
currently implement the eleven display choices above, mouse-accessible menus,
and an explicitly saved display preference in the standalone viewer. The
additional filters are experimental presentation choices; Nearest remains the
reference/default, and no new mode is claimed to have won. The DCT and custom
V7 Perceptual paths below remain proposals, not available display modes.

### 12.1 Ownership and immutable boundaries

The live receiver supplies decoded float spatial values and plane shapes to
`tools/v7_gl_viewer.py` through its latest-frame mailbox. Treat those arrays as
read-only. Display processing must not mutate decoder state, tail memory,
metadata, diagnostics measurements, or `--save-dir` decoded exports.

Retain GLFW/ModernGL, the OpenGL 3.3 core requirement, lazy graphics imports,
the dark visual style, the large picture, and the four existing diagnostic
cards. Keep the GL context and resource lifecycle on the viewer thread.
Viewer-only helpers may be factored into `tools/v7_display_*.py`; do not grow a
second decoder, import GUI dependencies into headless receive, or modify other
application modes. Additional DCT transforms described here are an explicit
viewer-only experiment, not changes to pulse acquisition or transport decode.

Preserve metadata aspect correction. The 80×96 canvas is a sampling geometry;
square display pixels would distort most sources. Use framebuffer pixels for
sampling and logical-window coordinates converted for hit testing on HiDPI.

### 12.2 Exact nearest and the float path

`nearest` must retain the current `values_image` → 8-bit RGB texture → nearest
shader path, with no dither or new color correction. Pillow's rounding,
clipping, chroma resize and integer color conversion are part of this reference.
Float-plane nearest sampling is not pixel-identical and must not replace it.

For new modes, split values into owned contiguous float32 Y/Cb/Cr planes,
mapping normalized values to the existing YCbCr convention without premature
clipping. Upload single-channel float textures; handle a one-plane frame as
grayscale. Define chroma center alignment explicitly and use clamp-to-edge
sampling. Each plane spans the same normalized frame rectangle and uses the
same `uv`, so reduced chroma grids are centered on that rectangle. Values use
`v = code/127.5 - 1`; recover full-range luma as
`Y = (vY + 1)/2` and 8-bit BT.601 chroma centered at code 128 as
`Cb/Cr = vCb/Cr / 2 - 0.5/255`. Convert with `R = Y + 1.402 Cr`,
`G = Y - 0.344136 Cb - 0.714136 Cr`, and `B = Y + 1.772 Cb`, then clamp
RGB to `[0, 1]`. This matches the existing Pillow YCbCr code convention and
uses full-range levels, not limited-range video levels. Shader reference tests
compare neutral gray and saturated RGB colors against the Pillow conversion.

Keep float intermediates through filtering. Define one output transfer policy
to avoid accidental double sRGB encoding: gamma-encoded RGB written to a
non-sRGB-converting framebuffer is the initial policy. Optional deterministic,
screen-pixel-anchored, zero-mean dither is added immediately before 8-bit output
quantization with peak amplitude at most half an output code step. A fixed
blue-noise tile may be shipped with documented provenance; an alternative
deterministic pattern must be named accurately, not called blue noise without
justification. Float precision and dither reduce contouring; they do not
guarantee the absence of bands. Dither is always disabled for legacy nearest.

### 12.3 Implemented filters and orthogonal effects

The current display menu uses these stable mode names and fixed kernel variants:

| Name | Reconstruction contract |
|---|---|
| `nearest` | Exact legacy reference and initial default |
| `bilinear` | Hardware-linear sampling of float planes |
| `sharp-bilinear` | Preserve block interiors, antialias boundaries using separate horizontal/vertical output footprints |
| `bicubic` | Mitchell–Netravali cubic with B=C=1/3 |
| `spline36` | Interpolating Spline36 cubic, support radius 3 |
| `robidoux` | Cubic B/C kernel with B=12/(19+9√2), C=113/(58+216√2) |
| `robidoux-sharp` | Cubic B/C kernel with B=6/(13+7√2), C=7/(2+12√2) |
| `cubic-bspline` | Cubic B-spline, B=1 and C=0; smooth and non-interpolating |
| `kaiser-sinc` | Separable radius-3 sinc with Kaiser window β=8.6 |
| `hann-sinc` | Separable radius-3 sinc with Hann window |
| `ewa-jinc` | Radial Jinc×Jinc reconstruction, support radius 3.2383154842 |

Spline36, Robidoux, Robidoux Sharp, cubic B-spline, Kaiser sinc, and Hann sinc
are resampled into bounded 4× float intermediates, then linearly sampled for
the final display scaling. EWA Jinc uses a radial kernel in the display shader;
its LUT is generated on demand. The filter choices are opt-in and affect only
presentation. Keep source planes unclipped until the final RGB conversion.

Sharp bilinear uses `p = uv*plane_size - 0.5`, `f = fract(p)`, and the separate
per-axis output footprint `s = max(output_size/plane_size, 1)`. It remaps the
fraction as `f' = clamp((f - 0.5)*s + 0.5, 0, 1)` and samples bilinearly at
`(floor(p) + f' + 0.5)/plane_size`. Below 1× magnification `s=1`, so it reduces
to ordinary bilinear. This is a coordinate remap, not exact area integration;
it does not promise equal integer block widths or eliminate motion shimmer.
Direct source-plane shader sampling uses clamp-to-edge taps and normalizes each
footprint at image boundaries. Bilinear sampling uses one hardware texture read
per plane; direct cubic and EWA shader kernels use multiple taps when selected.

Dering and luma-guided chroma are optional processing stages, not competing
upscaler names. Dither is an output option. Crossfade and tape/CRT styling are
deferred optional effects and must not delay the core GUI/modes. Crossfade, if
implemented, shows the first frame immediately, blends for at most one measured
frame interval, bypasses cuts, and reports the blend duration. It is not motion
interpolation. No temporal effect is enabled by default.

Cache CPU-generated intermediates per frame generation and selected filter.
Bounded 4× intermediate dimensions keep work independent of desktop resolution;
the final scaling pass still costs screen-resolution-dependent GPU work.

### 12.4 DCT reconstruction: precise implementation boundary

The mailbox contains spatial samples, not coefficients. For each plane:

1. Compute a 2D DCT-II with orthonormal normalization.
2. Copy all coefficients into the top-left of a zero-filled 4H×4W array.
3. Multiply coefficients by `sqrt((4H*4W)/(H*W))`, i.e. 4 for this scaling.
4. Apply the matching orthonormal inverse transform to obtain the larger plane.

This samples the same cosine expansion on the finer cell-centered grid.
Account for the corresponding sample centers in GPU texture coordinates.
Do not compare every fourth enlarged pixel directly to an original pixel:
their centers do not coincide for this even enlargement factor. Verify against
direct cosine evaluation at selected coordinates instead.

Do not clip planes before this operation, discard outside-corner guests, or
apply a new rectangular cutoff. Test constants, single basis components,
amplitude, orientation, grayscale, and boundaries. Reuse buffers and perform
transforms once per new frame, not on toolbar redraws. Lazily load the existing
CPU transform dependency. This is a cosine-consistent presentation of decoded
samples, including their noise; it is not recovery of the lost source image.

### 12.5 Custom mode: V7 Perceptual

Goal: smooth reconstruction with readable small facial features, restrained
halos, and less color bleeding. Initial settings are light halo restraint,
light chroma guidance, definition off, and output dither on. Initial numerical
strengths are halo 0.25, chroma guidance 0.25, and definition 0. These are trial
settings, not an instruction to make this mode the user's default.

Implement the following separately measurable stages on float intermediates:

**A. DCT base.** Use 12.4 exactly. Initially apply no extra coefficient gains.
The sender already may shape coefficients; do not silently sharpen twice.

**B. Soft halo restraint.** At each enlarged luma pixel, derive local bounds
from a 3×3 neighborhood on the original decoded grid. Use a bilinearly sampled
local range/edge field to avoid per-cell parameter jumps. Allow a small
range-relative overshoot and soft-compress only excursions beyond it; blend
the correction by `halo_strength` in [0, 1]. An initial allowed margin is
`0.05 * local_range + 1/255` in normalized luma. Above upper bound `b`, a
candidate bounded mapping is `b + margin*tanh((value-b)/margin)`, with its
symmetric counterpart below the lower bound. Leave in-range values unchanged.
Reduce strength around isolated source extrema using a documented bounded
local-extremum mask; test catchlights and pupils explicitly. This heuristic
restrains reconstruction overshoot, not all ringing already in decoded samples.

**C. Optional definition.** Add a bounded local-contrast correction at one to
two source-pixel scales, not a finest-band boost. Initial experiment: subtract
an edge-aware local luma mean, multiply by `definition` in [0, 0.25], and cap
the correction to 0.02 normalized luma. Suppress it in near-flat regions, near
output limits, and where halo restraint is active. Default is zero. Record
kernel and thresholds, and do not adapt global strength per frame.

**D. Guided chroma.** Start from smooth DCT-reconstructed chroma. Use coherent
luma boundaries to downweight chroma contributions across the boundary, with
bounded guidance and a blend back to the unguided result. Use a fixed-size
joint-bilateral neighborhood at the intermediate grid; publish spatial/range
weights and their sample-center mapping. Guide with lightly smoothed luma so
fine grain does not print into color. Preserve chroma-only boundaries; test
isoluminant color edges. `chroma_guidance` is in [0, 1], initially 0.25.
Luma is unchanged by this stage; automatic saturation gain is initially off.

**E. Output.** Convert and dither according to 12.2. Preserve all input arrays.

No temporal history is required for this mode. A later change-aware option may
reduce added definition in changing regions, with timestamp-smoothed control,
but may not claim to identify stale tail coefficients from pictures alone.
True coefficient-age-aware processing needs a separately specified read-only
diagnostic interface. Never invent that metadata in this implementation.

### 12.6 GUI, preferences, and interaction

The receiver GUI Live toolbar opens a mouse-selectable menu showing all eleven
display modes at once and retains Nearest as its initial mode. The menu also
supports Up/Down, Enter, and Esc. The standalone viewer has the compact GL
toolbar, versioned saved default, diagnostics toggle, and fullscreen controls.
U/Shift+U cycles display modes in the viewer; digits 1–9 select the first nine
entries. The preview caller accepts `--display-mode`.

The controls below are remaining interaction proposals, not shipped behavior.
When extending the GUI, keep the existing dark style, reserve toolbar space in
windowed mode, and use explicit hit regions rather than introducing a second
windowing toolkit.

Toolbar controls:

- **View:** the eleven implemented filters above. Unsupported future modes are
  disabled with a short reason. Only working modes participate in keyboard
  cycling.
- **Variant:** shown for bicubic only if a second variant is implemented; the
  current mode is Mitchell–Netravali.
- **Effects:** popover for implemented independent effects and output dither.
- **Mode controls:** labeled contextual sliders, not a universal strength.
  V7 Perceptual exposes Halo restraint, Definition, and Chroma guidance.
- **Compare:** enable a draggable, labeled wipe of the same full-size picture;
  reference dropdown offers Nearest, Bicubic, and DCT when available. Both sides
  share a frame generation, viewport, and aspect. No side randomization or
  hidden labels: the user judges what they prefer.
- **Save as default**, **Reset mode**, **Screenshot**, **Info**, **Fullscreen**.

Preserve F, I and Esc semantics. U/Shift+U cycling is implemented; extend direct
digit selection if desired, and reserve A for the wipe and S for screenshots.
Dropdowns support pointer input,
keyboard focus, arrows and Enter; Esc closes an open control before leaving
fullscreen. Sliders expose their numeric values. Do not overload [ and ] with
an ambiguous global strength; they may adjust the focused slider.

Fullscreen toolbar auto-hides after inactivity, reappears at the top edge or
on control-key use, and stays visible while a control is open. Schedule redraws
for label expiry and toolbar hiding. Mode changes show a brief label; title
and picture-info card show active mode and measured GPU time. Existing decoder
diagnostics measurements remain unchanged. Ordinary rendering is event-driven;
diagnostic updates already can trigger redraws independently of picture arrival.

Store versioned JSON preferences at
`$XDG_CONFIG_HOME/modemTest/v7_display.json` (otherwise
`~/.config/modemTest/v7_display.json`). Validate enums, ranges and version;
ignore invalid settings with a visible notice. Write atomically only when
Save as default is pressed. Merely experimenting must not persist changes.
Precedence: explicit CLI fields > valid saved fields > factory settings.
Factory reconstruction is legacy nearest. An unsupported saved mode falls
back visibly to nearest without overwriting the saved preference.

If display controls are added to headless receive, add `--display-mode NAME` and explicit contextual overrides such as
`--display-variant`, `--display-halo`, `--display-definition`,
`--display-chroma-guidance`, and `--display-dither on|off` to receive and preview.
Use parser defaults of unset for preference-overridable fields. Do not add the
draft's ambiguous `--display-strength`. Headless receive must not initialize
GUI/preferences merely because these parser options exist.

### 12.7 Preview caller and screenshots

`tools/v7_viewer.py` is implemented as a standalone caller of the same viewer,
with no audio capture or decoder thread. It accepts image paths for a
still/sequence and an explicit playback FPS. Pause, frame stepping, and the
decoded-frame fixture format remain proposed extensions.
Image previews use the documented box preparation and fixed V7 spatial grids;
they demonstrate presentation, not decoded-wire fidelity.

Also accept a documented `.npz` decoded-frame fixture containing `values`
(frame × value float array), `shapes` (plane × 2 integer array), `aspect`
(one wire aspect code per frame), and optional `timestamps` (seconds). Validate
sizes, finite values and codes; load without pickle. This lets experiments
feed exact decoded float pictures without changing normal decoded exports.
The live and preview callers share settings parsing and the `run` interface;
do not copy the viewer implementation into the launcher.

Screenshots are explicit display exports, separate from `--save-dir`. Default
to `tmp/v7_viewer/screenshots/`, allow an output-directory override, and avoid
overwriting existing files. Save the visible picture/wipe at framebuffer
resolution without toolbar or diagnostics, with a JSON sidecar recording mode,
parameters, reference, source generation, aspect, viewport, and output policy.
Perform framebuffer readback only on request, outside performance samples.

### 12.8 Failure handling and resource lifetime

Compile/validate candidate shader programs independently while retaining the
legacy program. Allocate new textures/framebuffers and upload a complete frame
before replacing the last usable resources. A failure must not release the
last good picture or publish a mixed-generation Y/Cb/Cr set. On unsupported
mode/resource failure, report it, disable that candidate, and use legacy
nearest or retain the last successfully rendered picture. No black fallback.
Before the first picture, retain the existing acquiring presentation; do not
fabricate a decoded frame. Release superseded resources on the GL thread.

### 12.9 Verification, budgets, and delivery order

Use actual offscreen GL rendering for shader/reference checks where supported,
and report skipped GL checks rather than pretending CPU tests exercised them.
No audio device is needed for these checks.

Required checks:

- Legacy nearest picture pixels match the previous renderer at identical
  viewport sizes, including fractional magnification and portrait aspects.
- Input arrays and normal decoded saved-frame bytes remain unchanged after
  exercising each mode and switching modes during a sequence.
- DCT normalization/coordinates, color conversion, grayscale and chroma-edge
  alignment meet analytic/reference checks described above.
- Mode/upload failure retains a usable picture; unsupported modes cannot be
  selected. Verify settings precedence, explicit persistence, HiDPI hit tests,
  resize, fullscreen, screenshot orientation and comparison synchronization.
- Compare clean, hiss and fast-flutter decoded fixtures at actual screen sizes;
  create labeled contact sheets under `tmp/`. Compute SSIMULACRA2 against the
  same aspect-correct source reference without GUI overlays. Freeze dither and
  color policy for kernel comparisons; separately ablate precision and dither.
- The user chooses their favorite named mode and saves it as default. A blind
  trial, a metric winner, and a claim of zero shimmer/banding are not shipping
  requirements. Report fidelity tradeoffs openly; optional effects need an
  observable benefit, not just an implemented control.

Measure warmed UI-thread CPU with `--profile-ui`, paired with nearest, and GPU
time using asynchronously retrieved timer queries. Do not wait synchronously
for a query result on each redraw. Report median and high-percentile times,
machine/GPU/driver, framebuffer size, frame/redraw rate, and settings. Target:
at most two additional percentage points of one CPU core and GPU p95 below
2 ms per redraw at the tested size. Report absolute cost and relative changes;
do not generalize results to other GPUs. Measure comparison mode and optional
vsync-rate crossfade separately. Include CPU forward/inverse DCT and uploads;
exclude compilation, screenshots, and setup from steady-state figures while
reporting their latency separately. If a budget fails, keep the mode explicitly
experimental or optimize it; never silently substitute a different algorithm.

Delivery order:

1. GUI shell, exact nearest, preferences, live wiring and preview caller.
2. Float path, bilinear, bicubic and sharp bilinear; contextual controls.
3. Wipe comparison, screenshots and asynchronous GPU timing.
4. DCT and separable Lanczos, validated independently and measured.
5. V7 Perceptual: DCT base, then halo restraint, guided chroma, optional
   definition, with on/off comparisons for each component.
6. Optional temporal/stylized effects only after user interest is confirmed.

Finally compare box/perceptual sender × ordinary/perceptual display as four
separate configurations. Hold sender fixed while choosing a display kernel,
and display fixed while judging sender improvements. No implementation agent
should automatically select or persist a new default based on metric scores.

## 13. Mono enhancement

**Status: experimental standalone mono fold-off and all-fresh mono 500-class
video prototypes implemented, September 27, 2026; not production-integrated.**
The rotating fold-off prototype remains in `test_modem_v7/mono_wire.py`; the
video prototype is in `test_modem_v7/mono_video.py`. Both are opt-in through
`tools/v7_live.py`. Their tests live in `modem_tests/test_v7_mono_wire.py` and
`modem_tests/test_v7_mono_video.py`. The moving-scene synthetic comparison is
in `tools/v7_mono_video_bench.py`. The live default and application integration
remain unchanged; real tape validation remains outstanding.

### 13.1 Goal and scope

An identical-leg mono V7 wire would remove single-leg M/S interference and
refresh the existing 2,880-coefficient corner over seven packets within today's
packet geometry. It targets reliable mono-sum, left-only, right-only, and
two-leg reception. These paths carry the same information, but their noise and
decoded fidelity need not be identical.

The target is the 48×40 Y plus two 24×20 chroma corners, not the entire picture
carried by today's live stereo fold-500 profile. Stereo's outside-corner luma
guests are not included. Matching stereo on stills or approaching it within a
specified dB gap is an evaluation question, not a promised result.

Today's stereo body has 2,320 coefficient slots: 1,264 M and 1,056 S. With
the existing modulator, L=(M+S)/√2 and R=(M−S)/√2, so averaging the legs
cancels S and leaves only M. The existing M-first rank map does not make those
1,264 slots cover the whole image: across its seven tail phases, the M union
contains ranks 0–1,231 plus 32 of ranks 1,232–1,295, and no ranks 1,296 onward
(pinned by `modem_tests/test_v7_mono.py`). Thus 1,616 of the 2,880 corner
coefficients are never recovered by a mono sum. A single leg avoids exact S
cancellation but is not the intended two-leg M/S observation used by the
existing equalizer. This explains why the current stereo wire can look very
poor after downmix even though the full stereo decode is good. The reported
9–12 dB live gap motivates the experiment; it is not a universal
channel-independent penalty.

### 13.2 Wire and receiver design

- Retain the 48 kHz reference geometry: 3,920 samples per packet, 24 OFDM
  symbols, 128-sample FFT, 16-sample prefix, and 12.245 packets/s at 1×.
- Retain body carriers at bins 4–34 (1.5–12.75 kHz). This describes the OFDM
  band, not the entire signal: timing tones remain on bins 1 and 3, and the
  nominal emission edge remains 14 kHz.
- Send data and continual/scattered pilots on M only, with S zero, so L = R.
  There are 79 data blocks × two I/Q groups × eight coefficients = 1,264 slots.
- Keep the 208-coefficient head on its current 26 M groups for the first
  experiment. Head relocation away from bins 5–6 is a separate measured
  ablation, not part of the initial allocation change.
- Preserve metadata content, EOF framing, and edge-counted pulse acquisition.
  Compatibility signalling must be settled explicitly as described below;
  a distinct-sync alternative would change the sync word, not packet length.
- Normalize against the stereo reference at matched leg RMS. Measure pilot
  power, peaks, crest factor, and limiter behavior; equal RMS alone does not
  establish equal headroom or identical noise-reduction behavior.
- Estimate one complex channel gain per cell per leg. Combine observations
  using their complex responses and noise estimates, or use the surviving
  leg. Raw SNR-weighted leg addition is insufficient under azimuth/phase error.
  Use a single-stream equalizer or a correctly reduced S-prior-zero path.

Two independently noisy copies can yield a 3 dB combining advantage over one
copy. The gain relative to today's stereo M slots must be measured with the
actual gains, rank allocation, pilots, normalization, and noise correlation.
No uniform +3 dB per-slot or single-leg quality guarantee is assumed.

### 13.3 Allocation and fold overhead

Let F be fresh physical slots, T rotating physical slots, and G nominal folded
guest positions. Before signature overhead:

```text
F + T = 1264
F + G + 7T >= 2880
G <= F - 208          # fresh hosts only; protected head; one guest per host
```

Tail slots are whole Hadamard groups of eight. The initial allocation is:

| Candidate | Fresh slots F | Nominal guests G | Tail slots T | Nominal seven-phase capacity |
|---|---:|---:|---:|---:|
| Mono fold-off: first prototype | 992 | 0 | 272 | 2,896 |
| Mono 500-class: deferred | 1,072 | 500 | 192 | 2,916 |
| Mono high-fold: before overhead only | 1,152 | 944 | 112 | 2,880 |

Fold-off carries 992 fresh coefficients and rotates the remaining 1,888 using
1,904 available tail positions: 16 positions are padding. A complete refresh
cycle is seven packets, about 0.57 s at 1×, provided those packets are decoded.

The original 1,160-fresh/1,000-guest proposal is invalid: protecting the head
leaves only 952 fresh hosts. The 1,152/944/112 alternative is the maximum
guest count under these pre-overhead rules, but is not a complete live design.

The current live fold uses 16 host positions as a fixed signature. Each
replaces both a picture host and its guest. Thus a nominal fold-500 table
delivers 484 guests and sacrifices 16 hosts; nominal fold-944 would deliver
928 guests and sacrifice 16 hosts. The seven-phase picture budget loses 32
coefficients, not just 16, unless those omitted coefficients are carried
elsewhere. The high-fold row therefore provides only 2,848 under that design
and does not cover the full corner.

The 500-class row has enough aggregate slack after subtracting 32 (2,884),
but needs explicit rank coverage proving the signature-displaced hosts and
guests are carried elsewhere. These are capacity budgets, not pinned tables.
Any high-fold allocation must be recalculated with the final signature scheme.

The rotating allocation above is the still-image prototype, not the video
allocation. The all-fresh video profile below uses 1,264 fixed physical M slots
on every packet and does not reuse `MONO_OFF` or its rotating rank map.

The signature measures noise on folded slots as well as table identity; coded
status alone does not replace this function. New mono tables must explicitly
budget that measurement, host/guest locations, padding, per-plane means and
scales, and complete seven-phase coverage. Guests may be luma or chroma within
the existing corner; today's outside-corner luma tables cannot simply be reused.

### 13.4 Signalling and compatibility

The current 12-chip status words repeat `0011`, `0101`, and `0110` three times.
Repeating their complements `1100`, `1010`, and `1001` supplies three candidate
mono words. All six words are balanced and retain minimum pairwise distance
six. Test chip alignment, polarity handling, despreading, noise/erasures, and
unknown-mode behavior before assigning permanent mono mode identities.

The current pulse metadata has no spare profile bit. Its aspect, encoding,
tail slice, direction, source index, and CRC/loop-mask fields are allocated.
The older clock word's profile field is not available here. A second witness
requires a separately specified wire change.

**Mono requires an updated receiver for the initial prototype.** Existing
receivers can display received or displayable packets without recognized coded
status; rejecting an unfolding table does not reject a different rank layout.
An unknown status or changed CRC mask must not be advertised as making old
receivers hold. They may display incorrectly assigned coefficients.

New receivers must validate layout identity before body interpretation and
picture/tail-state updates. An unknown or invalid layout holds the last good
picture. A known layout may show a damaged but decodable picture; an
undecodable packet holds the last good picture. There is no black fallback.

If rejection by old receivers becomes a requirement, investigate a distinct
pulse sync word with the same length and edge-counted acquisition. This needs
forward/reverse template separation and false-acquisition tests against old
receivers; it is not yet a proven rejection guarantee. Choose that alternative
explicitly before freezing the wire.

### 13.5 Tail state and quality hypotheses

Reset tail history on layout changes while retaining the last displayed good
picture. Current history is indexed by coefficient, so switching layouts does
not inherently permute stored values, but freshness and update schedules differ.
Folded guests need explicit history/update handling as well as host ranks.
Keep the seven phases and loop-field slices 5 and 6, and verify reverse playback,
direction changes, dropped packets, and profile changes with the new layout.

Current tail ages advance on `TailStore.update()` calls and expiry substitutes
the model mean; it does not gradually fade detail or independently track elapsed
packet time. Specify and test aging across losses before relying on a 0.57 s
staleness bound. More rotation increases the importance of cut and motion tests.

A missing coefficient and a stale coefficient have different errors. For
unrelated zero-mean values with variance lambda, substitution of the mean has
expected error lambda, while reuse of the previous value has expected error
2 lambda. Real motion depends on temporal correlation. Counting stale values as
missing does not produce a worst-case moving-picture estimate.

The original numerical quality table, fold SNR thresholds, and 1.56–1.71×
capacity ratio are not adopted as forecasts. They assumed a rank^-1.5 model,
uniform relative slot error, a guaranteed mono power gain, and a variable fold
step. The live fold instead uses pinned steps, confidence fallback, measured
signature noise, and guest weighting. A Shannon slot-capacity comparison under
assumed +3 dB is conditional, not a measured image-quality or resolution ratio.

Use the frozen V7 variance/gain arrays and representative content to measure
host loss, guest recovery, chroma fidelity, and temporal error. Stereo retains
more simultaneous physical slots; the useful still/motion trade must be measured.

### 13.6 Evaluation and delivery order

1. **Freeze the experimental contract.** Specify layout identity, receiver
   compatibility, exact rank coverage, profile-aware tail reset, per-leg channel
   combining, and level normalization. Start with mono fold-off and the existing
   protected head. Keep it opt-in.
2. **Validate fold-off.** Compare with today's standalone live stereo reference
   (box, fold-500, coded pilots, EOF) and that reference's mono-sum/left-only/
   right-only reception. Also use an unfolded stereo control to isolate the
   effect of excluding outside-corner guests.
3. **Measure visible output.** Use SSIMULACRA2 against both the prepared source
   and the matching stereo decode, with fixed aspect, display reconstruction,
   brightness, gamma, and score size. Include stills, slow pans, hard cuts, fine
   detail, and saturated/chroma boundaries. Score actual displayed sequences,
   including held pictures, and report packet availability separately. Do not
   use PSNR against a higher-resolution source to claim the mono gap is closed.
4. **Check transport.** Exercise stereo, mono sum, left-only, and right-only
   inputs through the default-wire synthetic torture matrix: hiss, Type I/II
   models, wow/flutter, fast flutter, azimuth, crosstalk, mains buzz, NR pumping,
   and dropouts. At the standard forward reference point, target 12/12 packets
   with valid metadata and report received/displayable/held counts separately.
   Test 44.1 kHz resampling and 0.5×–2× speed with capture bandwidth stated:
   use 96 kHz for full-band 2×; 44.1/48 kHz fast cases are bandwidth-limited
   tests, not guaranteed full-band recovery. At 48 kHz the nominal 14 kHz
   emission-edge limit is about 1.71×. Verify reverse and loop-field behavior.
5. **Measure cost and play live.** Pair warmed encode/decode CPU and wall time
   with stereo on the same workload, including leg combining and any unfolding.
   Report absolute time and percentage change. After relevant synthetic checks,
   prioritize real-time playback per Section 1. Synthetic tape models remain
   regression checks; real tape/deck validation belongs to actual captures.
6. **Evaluate folded tiers later.** Start with a fully budgeted 500-class table;
   reconsider high folding only after solving host and signature allocation.
   A tier must demonstrate a useful moving-content benefit over mono fold-off
   and disclose still/chroma regressions before promotion. The sender cannot
   measure the future medium's SNR: explicit tier selection is the initial
   model, with no automatic fold-500 default based on theoretical thresholds.

Before pinning folded tables, decide whether a future layered stereo wire
should share this M layout and put enhancements on S. That is a separate design
decision, not a dependency of the fold-off experiment. The first deliverable is
evidence for or against mono fold-off, not a promise to ship all tiers.

### 13.7 Implemented prototype and initial results

The standalone sender/receiver opt into the profile with `--experimental-mono`:

```text
.venv/bin/python tools/v7_live.py send --source test \
  --device 'BlackHole 2ch' --experimental-mono
.venv/bin/python tools/v7_live.py receive \
  --device 'BlackHole 2ch' --experimental-mono
```

The sender emits the mono layout, identical L/R legs, M-only data pilots, the
`MONO_OFF` coded status, and EOF. The receiver requires tone-seeded timing and
EOF; it validates the coded profile before equalization and holds/rejects any
unknown or non-mono layout. The prototype uses the nearest model and current
sender brightness default unless explicitly overridden. It does not claim that
old receivers safely reject mono packets. The receiver only enables the mono
profile for the duration of this receive run; it does not dynamically switch
between mono and stereo layouts.

The six status words are defined in `test_modem_v7/tone_code.py`: the three
existing fold words and their balanced complements. `FOLD_OFF` and `MONO_OFF`
are independently recognized; only `MONO_OFF` has the original rotating mono
layout. `MONO_500` identifies the distinct all-fresh 500-class video profile.
`MONO_1000` remains reserved by live profiles and is used only by the offline
fold-off comparison control. The existing `MONO_OFF` wire remains 992 fresh
plus 272 rotating slots, with 16 padding positions on the final phase. Clean
synthetic decode confirmed that all seven phases cover the 2,880 corner
coefficients and that the channel fit/equalizer run with S prior zero.

Reproduce the targeted tests and the 25-case synthetic channel matrix with:

```text
.venv/bin/python -m unittest modem_tests.test_v7_mono_wire \
  modem_tests.test_v7_pilot_tones modem_tests.test_v7_wire_profile \
  modem_tests.test_v7_experimental_fold -v
.venv/bin/python tools/v7_mono_enhancement.py \
  --out tmp/v7-mono-enhancement
```

The tests verify six balanced status words at minimum distance six; complete
rank coverage; identical legs; forward decode from stereo, mono sum, and either
leg; rejection of a known stereo fold-off status before image equalization;
reverse EOF/status handling; sender CLI integration; and hook restoration.
All 25 synthetic cases at 96 kHz returned 12 results with 12 valid metadata
words, 12 displayable pictures, and 12 validated EOF markers. The soft
saturation case had 9 `received` results and 3 degraded/lost-but-displayable
results; the other cases had 12 received. This is a synthetic tape-model check,
not real tape evidence.

A clean static comparison used 12 repeated packets from `v7_reference_face.png`,
the same box-prepared source, 405×540 output, and SSIMULACRA2 against that
prepared source. Averaged over counters 8–12, after the seven tail phases had
refreshed the full corner, the scores were:

| Playback | SSIMULACRA2 vs prepared source |
|---|---:|
| Current live stereo fold-500, both legs | +6.49 |
| Current live stereo fold-500, mono sum | −44.38 |
| Current live stereo fold-500, left leg only | −64.22 |
| Current live stereo fold-500, right leg only | −64.21 |
| New mono fold-off, both legs / mono sum / either leg | −9.85 |

For context, stereo fold-off without the outside-corner guests scored −9.86
steady-state on the same fixture. The new mono profile therefore improves this
fixture's current live-wire mono sum by about 34.5 SSIMULACRA2 points and either
single leg by about 54.4 points, while remaining about 16.3 points behind the
full current stereo fold-500 decode. These are SSIMULACRA2 score differences,
not dB, and the stereo fold-500 wire includes outside-corner guests that mono
does not carry.

The first mono packet scored −49.06, worse than the current stereo mono sum
(−44.39); mono cold-start detail remains poor until the larger rotating set has
arrived. On this repeated still, the seventh packet is the first steady one.
These are single-fixture results, not predictions for other content or
channels.

The quality difference has two separate causes. First, the old stereo layout
uses the M/S transform above, so mono summing removes S; its fixed M rank
allocation also omits ranks 1,296–2,879 entirely and half of ranks 1,232–1,295.
Second, the mono fold-off prototype deliberately reassigns its 1,264 per-packet
slots to 992 fresh ranks and 272 rotating ranks, so the union refreshes the
whole 2,880-coefficient corner after seven packets. Its identical legs survive
sum or single-leg playback without M/S cancellation. It is consequently a
large improvement over current stereo downmix after warm-up, but remains below
full stereo because it does not send S independently or the live fold-500
outside-corner guests. The seven-packet still result is a refresh-cycle
measurement; moving content can make rotating detail stale or delayed.

Reverse playback validated EOF and the mono status. On an eight-packet cold
reverse sequence, two loop-field slices were held under the existing
independent-metadata startup rule; six pictures were received. This is the
existing reverse cold-start behavior, not a mono status failure.

A paired warmed process-CPU probe used 16 repeated box-profile packets per
batch, four timed batches per profile, and per-packet medians after a warm-up
batch. On a 4-logical-CPU AMD EPYC-Milan VM (Python 3.13.5, NumPy 2.2.4), it
measured:

| Profile | Encode CPU ms/packet | Decode CPU ms/packet |
|---|---:|---:|
| Mono fold-off | 1.322 | 1.140 |
| Stereo fold-off, coded status | 1.293 | 1.046 |
| Current live stereo fold-500 | 1.288 | 1.014 |

Against the current live profile, this is about +2.6% encode CPU and +12.4%
decode CPU. It is a local process-CPU measurement, not a wall-time or real-time
audio result; paired wall-time and additional-machine checks remain outstanding.

Real-time playback remains outstanding. The available audio device list in the
development environment contained PulseAudio `pulse`/`default`, not
`BlackHole 2ch`, so no real audio stream was opened. Synthetic loopback does not
substitute for the Section 1 playback validation.

### 13.8 All-fresh mono video with a 500-class fold

The video-only profile is separately opt-in with `--experimental-mono-fold`;
it does not alter `--experimental-mono` or the `MONO_OFF` rotating wire:

```text
.venv/bin/python tools/v7_live.py send --source test \
  --device 'BlackHole 2ch' --experimental-mono-fold --mono-video-side left
.venv/bin/python tools/v7_live.py receive \
  --device 'BlackHole 2ch' --experimental-mono-fold --mono-video-side left
```

It requires the canonical box model, coded pilot timing, EOF framing, and no
pre-encode perceptual resizer. Every packet transmits the same 1,264 M-only
physical slots: the protected 208-coefficient head followed by 1,056 fresh body
slots. There is no rotating tail and the live receiver disables tail memory.
Five hundred hosts within the fresh body carry the fold; the final 16 hosts are
the fold-table signature and replace their host and guest picture values. The
remaining 484 folded guests are the next box-model corner ranks. This gives
1,732 current-frame picture coefficients at most (1,248 host-picture values
plus 484 guests), with the wire budget often described approximately as 1,750
coefficient positions. The fold step and complete table identity are pinned in
`test_modem_v7/mono_video.py`.

The sender's `--mono-video-side left|right` selects the output leg for the
complete modem signal, including timing/status tones; the other output leg is
zero and remains available for a separate audio track. The receiver uses the
matching option to run pulse acquisition and decoding on only that input leg,
ignoring audio on the other. Both default to `left`; the sender GUI exposes the
choice and describes which side remains clear. The two-leg form is retained in
the offline comparison harness, not used by this side-selected live profile.

`MONO_500` is the distinct coded status for this rank map. The updated receiver
checks the status before image equalization and rejects both the legacy
`MONO_OFF` rotating layout and stereo fold status, holding the previous image.
This status gate is not a promise that pre-existing receivers reject the new
profile; they do not implement this check. The benchmark-only all-fresh
fold-off control uses the reserved `MONO_1000` code as a separate identity and
is not exposed by the live CLI or GUI.

The runner `tools/v7_mono_video_bench.py` compares the new profile against the
current stereo fold-500 decode, that stereo wire downmixed to mono, the
rotating mono fold-off profile with tail memory on/off, and an all-fresh mono
fold-off control. The mono video profiles use the selected left leg; the other
leg is silent. It scores packets 8–23 against the matching generated source
frame and reports decoded, current-frame-displayable, held-picture, EOF, encode
and decode costs separately. The encode/decode times are local wall time per
packet (not process CPU), measured after decoder warm-up. Pans, six-packet cuts,
and a moving graphic blob are deterministic synthetic scenes; the previous review script
`tmp/mono/video.py` is unavailable in this checkout, so these are scene-class
substitutes, not identical stimuli. The selected Type-II and fast-flutter cases
are limited synthetic regressions, not cassette emulation. No real-tape result
is inferred from them.

The first complete run contains four 24-packet scenes, with 16 scored frames per
scene and profile/channel pair. Mean SSIMULACRA2 across the four scenes, plus
mean per-packet wall time across the run, is:

| Profile | Clean | Type II | Fast flutter | Encode ms/packet | Decode ms/packet |
|---|---:|---:|---:|---:|---:|
| Stereo fold-500, both legs | −4.70 | −16.67 | −13.03 | 1.425 | 1.064 |
| Stereo fold-500, mono-summed | −34.52 | −36.99 | −35.21 | 1.425 | 1.032 |
| Rotating mono fold-off, tail memory on | −54.65 | −54.98 | −54.88 | 1.511 | 1.260 |
| Rotating mono fold-off, tail memory off | −39.85 | −40.44 | −40.55 | 1.511 | 1.231 |
| All-fresh mono fold-off, left leg | −34.50 | −35.80 | −35.44 | 1.346 | 1.159 |
| All-fresh mono 500-class fold, left leg | **−29.23** | **−35.02** | **−34.14** | 1.496 | 1.180 |

All 72 profile/scene/channel runs received and displayed 24/24 packets, held no
previous pictures, and validated 24/24 EOF markers. Across the four scenes, the
fold improved on its all-fresh fold-off control by 5.27 points clean, 0.78 in
Type II, and 1.31 under fast flutter. The generated cut scene remains difficult
in absolute score; the fold advantage there is about 3.1 clean points and below
1 point in the two stresses. These are comparative synthetic results, not
performance claims for cassette media or the missing original stimuli. Results
are saved at `tmp/v7-mono-video-left/results.json`.

Reproduce the profile checks and complete comparison with:

```text
.venv/bin/python -m unittest modem_tests.test_v7_mono_video \
  modem_tests.test_v7_mono_wire modem_tests.test_v7_send_gui \
  modem_tests.test_v7_receiver_gui -v
.venv/bin/python tools/v7_mono_video_bench.py \
  --out tmp/v7-mono-video-left
```
