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

### Standalone audio-device recovery

The standalone sender and receiver bind to explicit PortAudio device
identities. A stopped or unavailable sender output pauses packet transmission;
a stopped receiver input pauses decode while its last good picture remains
displayed. Recovery never silently selects an unrelated default device. Device
identity and reported sample rate must be stable for five 30-fps intervals
(about 167 ms) before reopening. The reopened stream's negotiated sample rate
is authoritative: the sender rebuilds rate-dependent packet conversion, and
the receiver clears samples and pulse/decode state across the discontinuity
before reacquiring. The receiver keeps the prior image until a new valid frame
is decoded. A lost optional sender source-audio input is replaced with silence
while video continues; passthrough-output recovery is independent of receiver
video capture and decode. These runtime recovery paths are software behavior,
not real-tape validation.

### Validation priority

After the relevant synthetic/unit tests pass, prioritize real-time audio
playback as the next validation step. This takes priority over additional
synthetic-only benchmarks or analysis while live playback remains available and
unverified. Synthetic and in-memory loopbacks validate software paths but do
not substitute for real-time playback. Repeat this order after subsequent
changes; if playback is blocked, record the concrete blocker and leave live
validation explicitly outstanding.

#### Current development-host resources (checked 2026-09-29)

`Xvfb` and `xvfb-run` are installed. The receiver setup GUI was smoke-tested
under `xvfb-run -a` and remained open without startup errors for an 8-second
check; this verifies virtual-display startup, not a complete interaction or
render-quality test. Use Xvfb for GUI/capture smoke tests when no desktop is
attached, for example:

```text
timeout 8s xvfb-run -a .venv/bin/python tools/v7_live.py gui
```

PortAudio currently reports ALSA devices `pulse` and `default`, each with up to
32 input and 32 output channels. Concurrent test processes may occupy the
default low-numbered channels. Reserve a disjoint single channel or stereo pair
for each live stream and specify that port/channel route in the stream setup;
record the actual device, channel mapping, and sample rate with the run. Device
names alone do not establish that a route is a loopback, so verify the chosen
route before sending any test signal and never assume `pulse` is safe. The
current `tools/v7_live.py` CLI opens mono or stereo streams on logical channels
0 or 0–1; it has no channel-offset option. To isolate a live stream on a
different pair, request enough stream channels to include the selected
zero-based indices, then route only those columns in the input/output buffers
(and zero-fill unused output columns); a single-port test uses one selected
column. The V7 CLI needs explicit stream-channel routing support before it can
select such non-default ports. Verify that mapping without physical playback
before a live run. This host inventory does not itself count as a real-time
playback validation.

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

The canonical profiles are `nearest`, `box`, `lanczos`, and `bicubic`; only
`nearest` and `box` are named on the wire (the metadata's former second
encode-filter bit is the screen bit, section 5), so live senders encode with
those two. Their
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
| Byte 0, bit 4 | Screen bit: 0 the aspect code is the picture's own; 1 it is the sender's fixed aspect layout, the picture pillar- or letterboxed inside it |
| Byte 0, bit 3 | Source model: 0 nearest, 1 box |
| Byte 0, bits 2–0 | Tail slice, 0–6; 7 is invalid |
| Bytes 1–2, bit 15 | Playback direction (0 up, 1 down) |
| Bytes 1–2, bits 14–0 | One-based source index; zero is invalid |
| Bytes 3–4 | CRC-16/CCITT-FALSE of bytes 0–2 XOR a rotating mask |

The aspect code (V7_ASPECT_RATIOS) is what the receiver letterboxes or
pillarboxes the picture to, and with `--aspect-layout auto` it also names the
packet's aspect layout. A sender with a fixed layout boxes each picture into
that layout's ratio and sends the layout's code with the screen bit set, so a
receiver on `auto` decodes and displays it without matching settings. Packets
with the screen bit clear keep the earlier byte layout bit for bit. Bit 4 was
the high bit of the encode filter: an older receiver reads a screen-bit packet
as lanczos or bicubic. The live sender's source index wraps at 32,767
(about 44.6 minutes of packets) instead of stopping.

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

### Splices: digital time-stretch and pitch-shift

Resampled speed changes (tape, varispeed, a clock offset) scale time and
pitch together and are followed by the pulse scale. Digital time-stretchers
of the WSOLA family (ffmpeg `atempo`, most players' speed controls) and the
pitch shifters built on them do not: they keep the local waveform scale and
change the duration by cutting or repeating chunks of a few hundred samples.
The header edges then measure the pitch, while a packet that contains a
splice is shorter or longer than that scale predicts by the chunk.

The EOF decoder treats this as information:

- **Endpoint.** If the EOF mark is not where the header predicts, the next
  header is searched from 40% to 205% of the predicted packet length (the
  splices of a ×0.5..×2 pitch shift) and the mark is looked for just before
  it (their spacing is fixed); an endpoint shorter than 36% of a packet is
  never accepted. If a splice took the mark itself, the
  next header is the witness: the mark's end is the next packet's origin.
  When the body does not fit an EOF found at the predicted place (cyclic-
  prefix correlation below 0.95 per symbol), or the metadata fails there,
  the next-header endpoint is tried too and kept if it fits better.
- **Per-symbol offsets.** The body is read at the header (pitch) scale. Each
  OFDM symbol's cyclic-prefix correlation is evaluated for every offset
  between 0 (header-anchored) and the measured jump (EOF-anchored) in 2-sample
  steps; a dynamic program picks the monotone staircase with the fewest
  steps (2.5 correlation units per step; offsets a whole number of symbols
  from either anchor are penalised as aliases). Any number of splices per
  packet are allowed. The map is used only when it beats the packet-wide
  affine map, and is refused when the path never reaches the endpoint's
  offset (within 25% of a packet): that endpoint belongs to a later packet.
- **Pilot identity.** The cyclic prefix says where a symbol starts, not
  which symbol it is; with many splices another symbol's intact copy is
  often nearby. Symbols on the path at either anchor (correlation ≥ 0.95,
  at least 3) give a flat 2×2 channel from their pilots. Each chosen symbol's
  pilots, after its own cell rotation, must match that channel (gain, phase
  and a ±2-sample ramp free; residual 0 = match, a neighbour scores
  0.4–1.0). If an intact chosen symbol fails (residual > 0.5), every
  correlation peak (≥ 0.75) is checked, the residual is subtracted from the
  correlation and the staircase is found again; symbols still failing are
  erased.
- **Erasures.** Symbols the splice falls in (correlation below 0.9) are
  erased: zero weight in the channel and timing fits, their fitted pilots in
  the fade/noise refit, both legs erased in the equalizer. Head gates scale
  with the fraction of symbols present. Their coefficients are missing for
  that frame (shown as decoded, per the partial-frame rule).
- **Metadata.** If the EOF-anchored metadata symbol fails its CRC, the
  header-anchored one is tried. In a recording (not the live latest-only
  path), a spliced packet whose metadata is still lost takes its
  neighbour's encode and aspect with the slice and index advanced; it must
  pass every quality gate and is marked `metadata_predicted`.
- **Resync.** A recording decode no longer stops at a packet without any
  endpoint: it is held as lost and decoding continues at the next header.

Cost: about 0.1 ms per clean packet for the cyclic-prefix check, about
2.4 ms per packet under heavy splicing. Phase-vocoder processing (Rubber
Band) rebuilds every phase and is not supported.

`tools/v7_timing_bench.py` measures packets decoded and mean luma PSNR of the
decoded pictures against a clean decode, for a recording (`stream`) and the
live latest-only path (`live`), 12 packets each; `--ffmpeg` adds ffmpeg
stretch/pitch cases. Before → after this change (stream / live packets; the
earlier recording decode also stopped at the first packet without an
endpoint):

| Condition | Before | After |
|---|---|---|
| clean, speed 1.02/0.97, flutter 0.1%, noise 24 dB | 12 / 12 | 12 / 12 |
| flutter 0.3% | 4 / 4 | 4 / 4 |
| tempo +3% (splices) | 8 / 8 | 10 / 9 |
| tempo +10% (splices) | 0 / 3 | 7 / 4 |
| tempo −10% (repeats) | 3 / 5 | 7 / 6 |
| pitch +6% / −6% (tempo kept) | 7 / 7, 1 / 5 | 9 / 9, 7 / 6 |
| ffmpeg atempo 1.05 / 1.10 / 0.90 | 4 / 8, 4 / 4, 3 / 4 | 12 / 9, 10 / 6, 7 / 7 |
| ffmpeg pitch +6% / −6% | 9 / 9, 4 / 6 | 11 / 12, 11 / 10 |

Decoded spliced packets score 35–55 dB against the clean decode on
average. Remaining losses: a splice through a header (the packet is not
found), several splices in one packet, metadata lost to a splice on the live
path, and flutter at 0.3% RMS, which is a pilot-noise gate result (noise
0.09–0.15 against the 0.08 live limit) rather than a timing failure.

#### Large pitch shifts (×0.5 to ×2, tempo kept)

Varispeed (time and pitch together) decodes 12/12 from ×0.5 to ×2 at a
96 kHz capture. A pitch shift with the tempo kept is a resample followed by
a time-stretch, and the stretcher overlap-adds windows every 10–20 ms: most
symbols land in a crossfade of two source positions. An oracle over the
bench's WSOLA (Hann overlap at half-window hops) counts the body symbols
that survive anywhere in the output as one clean copy (≥ 90% from one
source position):

| Shift | 1024 window | 2048 window | 4096 window |
|---|---|---|---|
| ×2.0 | 38% | 59% | 68% |
| ×1.5 | 20% | 41% | 51% |
| ×1.25 | 7% | 32% | 42% |
| ×0.8 | 0% | 11% | 24% |
| ×0.67 | 0% | 7% | 19% |
| ×0.5 | 0% | 1% | 11% |

Pitch-down discards material and crossfades what is left, so a decoder
alone cannot reach ×0.5. Erasing every symbol that is not a clean copy (an
oracle) decodes worse than keeping the mixtures, because too few remain.
Measured at 96 kHz, 24 packets (received / displayed partial frames, mean
luma PSNR against the clean decode):

| Shift | WSOLA 2048 | ffmpeg asetrate+atempo |
|---|---|---|
| ×2.0 | 0 / 11 at 15 dB | 5 at 36 dB / 28 at 19 dB |
| ×1.5 | 6 at 21 dB / 16 at 21 dB | 9 at 29 dB / 13 at 20 dB |
| ×1.25 | 9 at 25 dB / 13 at 16 dB | 18 at 34 dB / 5 at 21 dB |
| ×0.8 | 1 at 15 dB / 19 at 13 dB | 8 at 18 dB / 14 at 14 dB |
| ×0.67, ×0.5 | 0–4 at 10–12 dB | 0–1 at 10–12 dB |

Cost: clean packets unchanged (about 1.04 ms); spliced packets about 2.1 ms
(pitch +6%) and 2.9 ms (tempo +10%), 5.8 ms under a ×1.25 WSOLA shift. The
cyclic-prefix scores are computed from running sums, which paid for the
pilot check.

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

The application sender (`main.py --mode modem`) defaults to stereo Fold 500:
the pinned Box model, coded pilots, and EOF marker at 1×. `--modem-baseline`
selects the previous fold-off profile with nearest encoding and steady pilots
(section 10.8).

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

### 8.1 Sender capture-to-audio latency profile (2026-09-28)

The production `run_send()` producer, Fold 500 encoder, bounded batch queue, and
write loop were profiled with a generated 320×240, 15 fps video file read via
the normal FFmpeg capture pipe. The output was a software sink that paced each
write to its sample duration; no audio device was opened. The run emitted
122/122 stereo packets, each 3,920 samples at 48 kHz (81.67 ms), using Box,
Fold 500, 1× playback, and one frame per batch. Capture timestamps mark each
RGB frame's arrival in Python; the frame-to-values interval also includes
waiting for the producer's next capture slot.

| Interval or stage | Median (ms) | p95 (ms) |
|---|---:|---:|
| FFmpeg frame-read call, including its 15 fps wait | 66.6 | 66.9 |
| Frame arrival to start of image-value preparation | 32.4 | 62.5 |
| Image preparation and DCT values | 0.688 | 0.954 |
| Fold coefficient transform | 0.403 | 0.563 |
| OFDM/audio packet encode | 1.317 | 1.836 |
| Coded-status overlay | 0.180 | 0.255 |
| Speed/pulse resampling | 0.018 | 0.035 |
| Frame arrival to completed packet in producer queue | 35.4 | 65.6 |
| Packet queue residence before `stream.write()` handoff | 348.3 | 356.3 |
| Frame arrival to `stream.write()` handoff | 383.4 | 415.4 |
| Frame arrival to simulated full-packet consumption | 465.2 | 497.1 |

The initial profile showed queue residence dominating latency: the sender's
0.4 s startup cushion left roughly 0.35 s of queued audio. A startup-cushion
A/B on the same FFmpeg-paced synthetic source measured:

| Startup cushion | Capture to `stream.write()` (median / p95, ms) | Queue residence (median, ms) |
|---:|---:|---:|
| 0.400 s (~5 packets) | 383.4 / 415.4 | 348.3 |
| 0.245 s (3 packets) | 222.0 / 252.3 | 186.1 |
| 0.163 s (2 packets) | 146.2 / 176.5 | 110.0 |
| 0.082 s (1 packet) | 59.8 / 90.1 | 24.7 |

The sender used a two-packet cushion at the time of this profile. The current
live sender starts after one emitted packet (81.67 ms at 1×) and caps the
producer queue at one batch, prioritizing freshness across capture modes. At
other playback speeds the cushion scales with the emitted packet duration. In
the synthetic 15-fps profile, the one-packet setting reduced median
capture-to-handoff from
146.2 ms to 59.8 ms and p95 from 176.5 ms to 90.1 ms. This reduces the margin
against a brief capture/encode stall in favor of lower live delay. Frame
preparation and packet construction take only a few milliseconds; capture
cadence contributes additional frame age before encoding. The final interval
adds one packet's 81.67 ms sample duration before
the receiver can decode the complete frame. These are simulated output-clock
measurements, not measured DAC latency; actual camera timing, audio-driver
behavior, and GUI rendering are not included. The reusable profiler, generated
video, and summaries are in the ignored `tmp/v7-zero2-profile/` directory.

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
The live stereo one-leg dropout suite is `tools/v7_stereo_dropout_torture_suite.py`.

Run the live suite on PulseAudio/PipeWire-Pulse with Xvfb available:

```text
.venv/bin/python tools/v7_stereo_dropout_torture_suite.py --receiver-profile fold-500
```

It runs a clean control followed by left-only, right-only, and seeded random
1–20 ms one-leg dropouts through a PortAudio relay into the live Fold-500 GUI
receiver, after a two-second receiver-acquisition warmup. A pass requires
consecutive displayed source indices, valid packet metadata, no genuinely
held frames (displayable degraded frames are allowed), correct per-leg erasure
detection, no audio stream errors, and no loss of packet headers on the intact
leg. The live sender, relay captures, schedules, receiver logs, screenshots,
and aggregate report are saved under `tmp/`.

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

### 10.8 Application sender stereo Fold 500 integration

The application sender (`main.py --mode modem`) now defaults to the pinned
stereo Fold 500 profile: Box model, folded luma coefficients, coded pilot
status, and EOF marker. `--modem-baseline` selects the previous fold-off wire
and its nearest-resize default. Folded packets are decoded with the existing
standalone live receiver configured for the matching Fold 500 profile.

The sender-side Fold 500 codec, its pinned table, and coded-pilot overlay live
in `animation_modem/`; table fitting and receiver-side prototype hooks remain
in `test_modem_v7/`. `animation_modem.v7.encode_pulse_frame_coeffs` lets both
sender paths frame transformed coefficients without an extra DCT pass. Tests
check coefficient and coded-pilot parity with the standalone sender, decode
application WAV output with the standalone receiver profile, and verify the
fold-off baseline switch. Repeat paired CPU and wall-time measurements on the
target host before treating local encode timings as a performance claim.

The remaining package work is receiver-side: move unfold/equalizer capture and
coded-status timing out of prototype monkeypatch hooks into explicit transport
decoder APIs, while retaining the fail-closed profile checks and verifying the
same wire on loopback.

### 10.9 Planned: normal-speed receiver CPU and memory profile

Establish the production-default stereo Fold 500 receive cost and assess
feasibility on a Raspberry Pi Zero 2 W with 512 MB RAM. Until that hardware is
available, use a deliberately conservative **15× CPU slowdown** for planning;
this is an estimate, not a target-device result. Restrict the assessment to
normal-speed playback.

Profile at least 2,000 changing synthetic packets. Generate the packet stream in
a separate process and store it under a named, gitignored `tmp/` run directory;
the receiver process must stream fixed-size blocks from the file rather than
load or memory-map the whole capture. Use the live 1,024-sample block cadence,
`LiveInput` rolling buffer and acquisition, persistent `PulseState`, canonical
Box model, stereo Fold 500, coded-pilot timing, EOF boundaries, default
precision, pulse timing and tone equalization. Open no audio device and do not
include sender work in receiver measurements. Pin the Git revision, table and
fixture hashes, Python/dependency versions, CPU, commands and all settings in a
run manifest.

After warmup, separately measure process CPU and wall time for input buffering,
leveling and pulse acquisition; packet decode (status, metadata, demodulation
and equalization); Fold 500 unfold/reconstruction; and latest-frame publication.
Also report complete per-frame processing latency. Collect at least 2,000
successful frame samples and report median, p95, p99 and maximum by stage,
successful frames, coded-status and EOF validation counts, and CPU seconds per
second of nominal input. Derive frame cadence from `PULSE_FRAME / RATE`, not
from the image-body duration. The playback interval and estimated target CPU
utilization must be stated explicitly; apply 15× only to measured CPU work, not
audio duration, sleep, or RAM.

Record RSS, PSS, private memory and peak RSS after imports, model loading, JIT
warmup, and at regular points through the sustained run. Report cold JIT compile
time/peak separately when feasible using a dedicated run-local
`NUMBA_CACHE_DIR`; do not delete or overwrite existing Numba caches. Avoid
counting mapped input pages as receiver memory. Measure graphics-context,
windowed display and setup-GUI overhead separately from headless decode; module
imports alone are not a rendering measurement. Document that Pi estimates use
cross-CPU proxies, and account for the fact that memory does not scale with the
CPU slowdown. Keep all outputs and reproducible scripts in `tmp/`.

Report the baseline before changing code. Rank optimization opportunities by
measured CPU and memory cost, then validate any later changes against identical
packet/profile correctness and picture reconstruction. Synthetic timing does
not establish audio callback reliability on the Pi.

#### Initial host-only baseline (2026-09-28; exploratory)

The first 2,000-packet profile used a 4-vCPU AMD EPYC-Milan 2.0 GHz KVM host
(Python 3.13.5, NumPy 2.2.4, SciPy 1.18.1, Numba 0.67.0), not a Pi. Two
cached-JIT runs supplied 4,000 sustained frame samples. A third run used a new,
empty Numba cache for the cold-start and memory comparison. All three decoded,
reconstructed, and validated 2,000/2,000 pictures, Fold 500 status words, and
EOF markers. The stream contained 16 deterministic variants of the reference
fixture and was generated in a separate process. No audio device or graphics
context was opened.

The reference packet is 3,920 samples at 48 kHz, so its interval is **81.67 ms**
(12.245 packets/s). The table reports pooled warmed process-CPU time per frame;
the last column applies the assumed 15× CPU factor to p95 only.

| Stage | Host median (ms) | Host p95 (ms) | Host p99 (ms) | Estimated Pi p95 (ms) |
|---|---:|---:|---:|---:|
| Input buffering, leveling, acquisition | 0.487 | 0.610 | 0.760 | 9.1 |
| Packet-hit selection | 0.041 | 0.065 | 0.081 | 1.0 |
| Status, metadata, demodulation, equalization | 1.062 | 1.365 | 1.642 | 20.5 |
| Fold 500 unfold/reconstruction | 0.298 | 0.413 | 0.532 | 6.2 |
| Latest-frame publication | 0.006 | 0.009 | 0.011 | 0.1 |
| **Complete receiver processing** | **1.918** | **2.380** | **2.908** | **35.7** |

Pooled p95 and p99 process-CPU time project to about **35.7 ms and 43.6 ms**,
respectively; warmed wall-time p95/p99 were 2.39/2.92 ms. The two warmed runs
used 2.40–2.41% of one host core in instrumented receiver work, or an estimated
36.0–36.2% of one Pi core under the 15× assumption. One of 4,000 warmed samples
reached 6.90 ms on the host (103.5 ms under 15×), above the 81.67 ms packet
interval; this is a single observed outlier, so the profile supports
normal-speed feasibility as a hypothesis, not a deadline guarantee. Timings
exclude actual audio callbacks, pixel rendering, and window-system work.

| Receiver memory stage | Warm-cache RSS | Warm-cache PSS | Cold-cache RSS | Cold-cache PSS |
|---|---:|---:|---:|---:|
| Imports | 172.3 MiB | 164.6 MiB | 172.3 MiB | 164.7 MiB |
| Model and fold table | 173.7 MiB | 166.0 MiB | 173.8 MiB | 166.1 MiB |
| After receiver JIT warmup | 229.3 MiB | 221.5 MiB | 355.0 MiB | 347.3 MiB |
| Peak RSS during run | **234.3 MiB** | — | **357.8 MiB** | — |

With cached JIT, RSS rose from 231.8–232.0 MiB at 500 frames to 232.7–232.8 MiB
at 2,000; PSS rose from 224.0–224.3 to 224.9–225.2 MiB. That is a small
settling increase, not evidence of unbounded growth over this run. Cold
compilation took 16.0 seconds on this host and had a much larger memory peak.
These are process measurements; they do not include OS, desktop,
GPU/framebuffer, or other application memory. The warmed headless footprint
looks plausible against 512 MB, while the cold peak leaves little room for the
Pi's OS; neither case is validated on ARM.

In the pre-change cold-cache run, the first successful frame spent 110 ms of
host CPU in input/acquisition, versus a warmed p99 of 0.76 ms. An isolated
first call to `v7.leg_polarity()` took 132 ms with a fresh Numba cache; the
production startup warmups left its `_leg_correlation_sums` Numba signature
uncompiled.

#### Polarity-kernel cache follow-up (2026-09-28)

The production receiver now calls `v7.warmup_leg_polarity()` before opening the
audio stream. The `cache=True` Numba kernel lives alone in
`animation_modem/v7_input_kernels.py`, so unrelated edits to the large `v7.py`
module do not invalidate it. Numba writes the compiled signature to its disk
cache on first use and loads it on compatible later processes; changing the
kernel module source or the Python/Numba cache environment causes a rebuild.
Use `NUMBA_CACHE_DIR` to choose a cache location, or Numba's default package
`__pycache__` location.

A 2,000-packet run with a fresh cache validated all frames, Fold 500 status
words, and EOF markers. Cold compilation (all receiver warmups) completed
before capture in 16.42 s on this host; the first live-frame acquisition then
took 0.55 ms of CPU, and the maximum across the run was 0.94 ms. A repeat
process loaded the polarity `.nbi`/`.nbc` cache entries (confirmed with
`NUMBA_DEBUG_CACHE=1`), completed startup warmup in 0.22 s, and its first
acquisition took 0.59 ms. Thus the one-time JIT cost is before audio capture,
and unchanged source reuses compiled code on subsequent runs. Peak receiver RSS
was 359.4 MiB for that cold-cache run and 234.5 MiB for its cached repeat. Cold
compilation time on ARM will differ and should not be estimated by the 15× CPU
multiplier.

#### Decoder substage profile (2026-09-28; exploratory)

A cached-JIT 2,000-packet run added per-call CPU/wall timers around the production
Fold 500 decoder components. All 2,000 pictures, coded statuses, and EOF markers
validated. The table gives host process-CPU median/p95 and the p95 projection
using the planning 15× factor. The rows are nested/inclusive as marked; do not
sum them. Wall-time medians tracked process CPU within a few microseconds on this
host.

| Operation | Host median (ms) | Host p95 (ms) | Estimated Pi p95 (ms) | Scope |
|---|---:|---:|---:|---|
| Metadata decode | 0.140 | 0.206 | 3.09 | Includes metadata sampling, FFT, and CRC parse |
| Metadata RFFT | 0.028 | 0.042 | 0.63 | Nested in metadata decode |
| Metadata parse/CRC | 0.015 | 0.021 | 0.31 | Nested in metadata decode |
| Body sample interpolation | 0.083 | 0.128 | 1.92 | Fractional-delay sample walk before frame demodulation |
| Body RFFT demodulation | 0.053 | 0.089 | 1.34 | 24 fixed-size symbol transforms |
| Coded status decode | 0.019 | 0.027 | 0.41 | Nested in tone timing/channel fit |
| Pilot-tone timing, excluding status | 0.076 | 0.114 | 1.71 | Exclusive estimate |
| Channel fit, excluding tone timing | 0.136 | 0.189 | 2.83 | Exclusive estimate |
| Fade/noise estimation | 0.059 | 0.087 | 1.30 | Per-frame channel/noise work |
| Equalizer Numba kernel | 0.197 | 0.290 | 4.35 | Nested in equalizer dispatch |
| Equalizer dispatch | 0.220 | 0.328 | 4.92 | Includes setup and the Numba kernel |
| Remaining body-frame work | 0.148 | 0.210 | 3.15 | Approximate exclusive residual, including confidence/gating/result assembly |
| **Body-frame decode total** | **0.728** | **0.966** | **14.49** | Includes RFFT, channel, fade/noise, equalizer, and residual |

An otherwise identical run without per-function timers measured the complete
decode stage at **1.089 ms median / 1.353 ms p95 / 1.627 ms p99**. The detailed
instrumentation raised that to 1.183/1.514 ms at median/p95, about 0.10/0.16 ms
of measurement overhead, so use the component profile for ranking rather than
as an uninstrumented total or deadline budget. The status decoder and body RFFT
are small individually; the larger components are the combined channel fit,
equalizer, metadata path, and body-frame confidence/result work. Within a
20%-faster equalizer kernel scenario, the saving would be only about 0.04 ms on
the host per packet (roughly 0.6 ms per packet under 15×), so measure a change
before taking on decoder complexity.

##### Follow-up: metadata remainder, Numba kernels, and channel-fit evaluations

A second 2,000-packet run wrapped the metadata stages, individual compiled
kernels, and both channel residual evaluators. It again validated 2,000/2,000
frames, coded Fold 500 statuses, and EOF markers. Every frame used tone-seeded
timing and the absolute timing-quality gate; no comparison refit ran. Source-line
tracing was disabled in the reported run: sampling it every tenth metadata call
inflated those calls, so its coarse group measurements are not suitable as
absolute timings. The table reports host process-CPU milliseconds; the
projection is 15× host p95. Rows explicitly marked inclusive overlap their
nested rows.

| Operation | Host median (ms) | Host p95 (ms) | Host p99 (ms) | Estimated Pi p95 (ms) |
|---|---:|---:|---:|---:|
| Metadata decode total | 0.135 | 0.186 | 0.230 | 2.8 |
| └ sample interpolation | 0.008 | 0.013 | 0.019 | 0.2 |
| └ metadata RFFT | 0.027 | 0.038 | 0.055 | 0.6 |
| └ CRC/word parse | 0.015 | 0.021 | 0.028 | 0.3 |
| └ remainder excluding the three above | 0.084 | 0.121 | 0.152 | 1.8 |
| Coded-status decode, inclusive | 0.022 | 0.032 | 0.046 | 0.5 |
| └ coded-status Numba kernel | 0.014 | 0.020 | 0.028 | 0.3 |
| Tone timing, inclusive of coded status | 0.096 | 0.138 | 0.173 | 2.1 |
| Channel fit, excluding tone timing | 0.152 | 0.198 | 0.240 | 3.0 |
| └ channel-fit Numba kernel | 0.081 | 0.107 | 0.122 | 1.6 |
| └ pilot-residual evaluation | 0.019 | 0.026 | 0.033 | 0.4 |
| └ timing-residual evaluation | 0.015 | 0.022 | 0.029 | 0.3 |
| Channel fit, inclusive of tone timing | 0.251 | 0.320 | 0.387 | 4.8 |
| Fade/noise Numba kernel | 0.056 | 0.080 | 0.090 | 1.2 |
| Equalizer Numba kernel | 0.197 | 0.239 | 0.300 | 3.6 |
| Body-frame decode total | 0.746 | 0.879 | 1.099 | 13.2 |

The equalizer kernel is the largest individual measured operation. The channel
path is the next larger combined cost: about 0.152 ms median excluding tone
timing, including 0.081 ms in the compiled fit and 0.034 ms across the two
residual-evaluation calls; tone timing adds about 0.096 ms inclusive of coded
status. Metadata decode takes about 0.135 ms median. Sampling, its RFFT, and CRC
account for about 0.051 ms, leaving about 0.084 ms in interpolation, response
handling, and symbol demodulation. These values rank optimization candidates;
they do not show that metadata or channel fitting alone is the end-to-end
bottleneck. No decoder optimization was made in this follow-up. Actual audio
callback and GUI-rendering costs and Pi-specific performance remain unmeasured.

##### Metadata demodulation optimization (2026-09-28; host A/B)

The first optimization keeps the wire, pilot thresholds, response threshold,
hard decisions, and CRC parser unchanged. The production default path now uses
the cached Numba helper in `animation_modem/v7_metadata_kernels.py` to apply
metadata phase correction, sum the two channels, interpolate pilot response,
and pack the 40 hard-decision bits. The helper is warmed by
`warmup_equalizer()` before live capture. Explicit `force_float32=True` decoding
retains its previous NumPy path.

The paired control replaces only `decode_metadata()` with the previous NumPy
implementation in the same profiler process. Each side processed two runs of
2,000 changing packets with the normal production Fold 500 profile and no
component timers or line tracing. The table pools 4,000 host process-CPU
samples per side; estimated target savings multiply the host CPU difference by
15. The full receiver stage covers acquisition through frame publication.

| Stage | Prior NumPy median (ms) | Optimized median (ms) | Prior p95 (ms) | Optimized p95 (ms) | 15× p95 saving (ms) |
|---|---:|---:|---:|---:|---:|
| Decode status/metadata/body/equalization | 1.095 | 1.010 | 1.492 | 1.274 | 3.3 |
| Complete receiver processing | 1.992 | 1.883 | 2.567 | 2.290 | 4.2 |

A component-timed 2,000-packet A/B measured metadata decode at 0.138 ms median
with the prior path and 0.071 ms with the helper, a 0.067 ms (about 49%)
reduction. Timer instrumentation raises the surrounding decode totals, so use
the untimed A/B above for end-to-end comparisons. Across both paired runs, all
2,000 frames decoded, reconstructed, and validated their coded Fold 500 status
and EOF markers. Every timed A/B CSV row matched on source index, receiver
status, coded status mode, and EOF result. A separate integrity run compared
aspect, encoding, tail slice, direction, status, coded status, EOF result, and
SHA-256 hashes of all 2,000 unfolded frame-value arrays; every field and hash
matched exactly. The focused NumPy-oracle tests also check clean, perturbed,
silent, float32/float64 input, explicit float32, fractional offset, mistimed
metadata, and loop metadata across all seven tail slices.

The isolated helper's first compile plus call took 0.99 s host CPU; loading its
cached signature and calling it took 0.14 s. This helper lives in its own module
so later edits to `v7.py` do not invalidate its cache. Changing `v7.py` for this
integration did invalidate that module's existing Numba kernels, so the first
full receiver warmup after the edit took 11.5 s on this host, versus 0.22–0.24 s
for cached baseline runs; this is a module-wide cold-cache effect, not the
helper's isolated compile cost. Warmed candidate runs took 0.23–0.25 s and peaked
at about 235 MiB RSS. Warmup completes before capture. Cold compilation and
memory on ARM remain to be measured. A later run with a dedicated empty cache
and the final metadata/equalizer code took 16.89 s to warm all receiver kernels
and peaked at 372.0 MiB RSS; its cached repeat warmed in 0.224 s and peaked at
235.3 MiB. The earlier fresh-cache comparison was 16.42 s / 359.4 MiB, so the
current cold peak is about 12.6 MiB higher on this host. Cold cache/compiler
memory on the Pi needs particular validation; do not project these RAM values
with the CPU slowdown factor.

The reusable control profiler, summaries, CSVs, and logs are in the ignored
`tmp/v7-zero2-profile/` directory. This measured gain makes the metadata change
a reasonable first optimization to retain.

##### Equalizer substage profile and group-solve cleanup (2026-09-28)

The equalizer was separated into per-cell stereo MMSE estimation and per-group
LMMSE solve/confidence helpers. Production still calls them through one fused
Numba dispatcher. For diagnosis only, the temporary profiler calls the two
compiled helpers separately from Python; those stage timings include the extra
Python/Numba dispatch and should be used to rank substage costs, not to predict
the production total.

Across two 2,000-packet diagnostic runs, per-cell MMSE took about **0.072 ms
median**, while the group solve took about **0.123 ms median**. This identifies
the group solve as the larger kernel substage. Hoisting repeated channel
conjugates/products in the per-cell loop was tested and rejected: its measured
per-cell median remained about 0.072 ms.

The group solve previously copied the 8×9 right-hand side into a temporary
matrix before immediately forward-solving it into another matrix. The solve
now reads each model-weighted or frame-observation RHS directly as it performs
the same forward-substitution operations, removing that intermediate. The
diagnostic group-stage median moved from about 0.123 to 0.118 ms across the
small run set, with per-cell timing unchanged. Untimed fused-production runs
showed decode medians around 0.993 ms before and 0.997 ms after, within run
variation; therefore this small local cleanup is retained, but is not credited
with a demonstrated end-to-end CPU reduction. Across all 2,000 packets, the
unfolded frame-value hashes and metadata/outcome fields matched the pre-cleanup
run exactly. The existing equalizer-versus-NumPy oracle passes across all seven
tail phases.

The group solve remains the most promising equalizer substage for further
measurement, but its roughly 0.12 ms host median bounds the opportunity. Any
more invasive solve changes should show a worthwhile uninstrumented whole-
receiver gain and preserve confidence/recovery decisions before being retained.

The JSON summary, per-frame CSV, and log are in
`tmp/v7-zero2-profile/receiver-decode-components-final.json`,
`tmp/v7-zero2-profile/per-frame-decode-components-final.csv`, and
`tmp/v7-zero2-profile/receiver-log-decode-components-final.jsonl`. No audio
device or graphics context was opened.

##### Channel residual pairing and comparison-refit profile (2026-09-28)

The pilot-misfit and weighted timing-residual checks use the same predicted
pilot values and observations. Previously each metric call repeated the shared
Numba residual kernel. `channel_joint()` now obtains both metrics from one
kernel pass; the float32/reference path likewise shares its pilot prediction
and indexing work. The pilot and timing thresholds, relative comparison, and
baseline fallback decisions are unchanged.

A focused 2,000-case tone-seeded profile forced the comparison fit on every
case. It kept the tone candidate for 1,917 cases and rejected it through the
pilot-residual gate on 44 and timing-residual gate on 39. The uninstrumented
channel-joint median/p95 was 0.258/0.300 ms on the host, including the candidate
and baseline fits. A 10,000-call residual microbenchmark measured about 12 us
saved per paired metric evaluation. Focused tests retain the successful
relative-gate path and independently exercise both rejection reasons; paired
float32 and float64 outputs match the NumPy references.

For an end-to-end A/B, two warmed 2,000-packet receiver runs used the same
changing Fold 500 capture. The separate-pass control emulated the former work
by evaluating the residual kernel twice. Across 4,000 samples per side, the
paired path reduced host process-CPU median from 0.997 to 0.971 ms for decode
and from 1.865 to 1.829 ms for complete per-frame receiver processing. The
corresponding p95 values were 1.202 to 1.185 ms for decode and 2.216 to 2.165
ms end to end. All 4,000 reconstructions had identical SHA-256 hashes and
matching source/status/profile/EOF fields; each run decoded and validated
2,000/2,000 packets. These short A/Bs show a small whole-receiver gain, not a
large Pi CPU reduction; apply the 15× projection only as a host estimate until
direct Pi measurement is available. Reusable profiles and captures are in the
gitignored `tmp/v7-zero2-profile/` directory.

##### Pulse-scan input preparation (2026-09-28)

`LiveInput._count_headers()` previously multiplied the stereo scan window by
the leveler gain, then `pulse_frame_hits()` allocated a second mono array. The
new cached `_mono_gain()` input kernel fuses those operations and passes the
mono window directly to the existing edge-counted pulse detector. It is warmed
with the other live-input kernels before capture. Float32 and float64 outputs
match the prior NumPy expression bit-for-bit.

On the same 2,000-packet capture, pooled across two uninstrumented runs per
side, input buffering/leveling/acquisition median CPU moved from 0.486 to
0.479 ms per frame; p95 was 0.600 versus 0.602 ms. All 2,000 reconstructed
frame hashes and metadata/status/EOF fields matched. The median reduction is
small and the p95 is unchanged, so this is recorded as a modest copy-reduction
gain, not a material receiver-tail improvement. The full receiver A/B is
subject to run-to-run variation in its other stages.

The shell was a TTY without `DISPLAY` or `WAYLAND_DISPLAY`, so real viewer/GL
memory and rendering cost were not measured. The reusable profiler, manifests,
per-frame CSVs, logs, packet capture, and `lscpu` output are in
`tmp/v7-zero2-profile/`; they are gitignored.

Reproduce the stream generation and warmed receiver run with:

```bash
NUMBA_CACHE_DIR=tmp/v7-zero2-profile/sender-numba-cache \
  .venv/bin/python -B tmp/v7_zero2_receiver_profile.py generate \
  --packets 2001 --output tmp/v7-zero2-profile/fold500-2001.f32 \
  --manifest tmp/v7-zero2-profile/generator-manifest.json
NUMBA_CACHE_DIR=tmp/v7-zero2-profile/receiver-numba-kernel-isolated-cold \
  .venv/bin/python -B tmp/v7_zero2_receiver_profile.py receive \
  --packets 2000 --wire-packets 2001 \
  --input tmp/v7-zero2-profile/fold500-2001.f32 \
  --output tmp/v7-zero2-profile/receiver-results-kernel-isolated-cached.json \
  --csv-output tmp/v7-zero2-profile/per-frame-kernel-isolated-cached.csv \
  --memory-interval 500
```

For the cold-cache run, use a new empty cache directory rather than removing an
existing cache:

```bash
mkdir -p tmp/v7-zero2-profile/receiver-numba-cold-repro
NUMBA_CACHE_DIR=tmp/v7-zero2-profile/receiver-numba-cold-repro \
  .venv/bin/python -B tmp/v7_zero2_receiver_profile.py receive \
  --packets 2000 --wire-packets 2001 \
  --input tmp/v7-zero2-profile/fold500-2001.f32 \
  --output tmp/v7-zero2-profile/receiver-results-cold-repro.json \
  --csv-output tmp/v7-zero2-profile/per-frame-cold-repro.csv \
  --memory-interval 500
```

## 11. Perceptual sender preprocessing

Status: the first resize ablation is implemented as an opt-in experiment in
11.2. It remains disabled by default, and no mode is claimed to be generally
better or ready to replace the existing path. The later sender stages and
Section 12 remain planned experiments. Continue in the milestone order below;
do not enable an unmeasured stage by default. None of this work requires a new
wire format. Section 11.7 records the implemented opt-in source-domain DCT
experiment and benchmark; it does not supersede the resize experiment or
change the sender default.

### 11.1 Goal, scope, and fixed contracts

Improve the decoded picture by preparing source content for the frequencies V7
actually retains. Favor recognizable small features, acceptable color, and
reduced ringing over nominal sharpness. All processing is classical and
deterministic. No learned models, random jitter, grain injection, region-based
packet allocation, or receiver changes belong in this work.

- Existing resize-track live integration: `tools/v7_live.py::_values`, before
  the current brightness and gamma adjustments. Keep their ordering and
  settings identical in paired comparisons. Share preprocessing through a
  self-contained modem module; do not import application settings into
  `animation_modem`.
- Bake integration: `utilities/convert_to_modem_dct.py`, where full-resolution
  layers are still available. Existing Lanczos-reduced layers cannot regain
  lost source detail through runtime preprocessing.
- Existing coefficient-shaping integration: inside the fold encoder's existing
  full-DCT path, before host/guest construction, not by transforming pixels
  back and forth. The separate native-source analysis implemented in 11.7 uses
  an explicit DCT/IDCT bridge to preserve the current coder-grid interface.
- Preserve packet length, metadata, encoding codes, model tables, fold pins,
  synchronization, receiver behavior, and the 80×96 prepared canvas for the
  existing resize track. Section 11.7 implements a separate opt-in path that
  bypasses that prepared canvas while preserving the coder-grid value-vector
  contract.
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

### 11.7 Native-source DCT analysis and perceptual coefficient reduction

**Current direction:** Section 11.8 is the corrective execution plan for this
work. It prioritizes optimizing the existing full-resolution preparation call.
The experiments and measurements below are historical evidence, not completion
of that plan or approval to substitute block averaging for full-resolution work.

**Status: implemented as an opt-in live-sender path and reproducible benchmark.**
This is a separate experiment, not a replacement for the current resize path or
the 11.2 resize ablation. The default remains the resize-first sender and wire
path. For integer-ratio BOX geometries, that path now uses the faster Pillow
`Image.reduce` operation described in Section 11.8; its channel rounding can
differ by one code value from generic `resize(BOX)`. Other filters and
non-integral geometries keep the generic resize behavior. `--dct-encode`
selects the native-source path and is mutually exclusive with
`--perceptual-resize`. All DCT-enhancement controls are off by default. The
receiver, wire, rank maps, fold tables, status codes, packet duration, and
current transmitted coefficient budget remain unchanged. Use the canonical
`box` model and the existing matching fold table for each tested profile; do
not refit or silently substitute model/fold statistics for this experiment.
Enhancement options without `--dct-encode` are an argument error, as is a frame
marked as already prepared at 80×96.

#### Objective

Test whether mapping the source directly into the DCT representation preserves
more useful picture information than first resizing RGB to the prepared canvas
and then transforming that raster. The initial comparison uses the existing
wire budget. The design should also keep source analysis separate from budget
selection so future, explicitly supported profiles can select a different
coefficient budget without requiring a different RGB resize path.

The native source-DCT path must not resize RGB to 80×96 before analysis or pass
an 8-bit YCbCr image between stages. It may analyze a capture at its configured
source dimensions, but the capture path must provide the unprepared frame rather
than its 80×96 prepared result. Images smaller than the required coder grid are
rejected; this path does not upscale a small source. The original source
dimensions continue to determine the aspect code. The separate `area-box`
candidate is explicitly a spatial-resampling path and does average source pixel
areas before its target-grid DCT.

#### Source-to-coder-grid pipeline

For an input RGB frame of shape `H×W×3`, apply the existing brightness and gamma
controls in float64:

```text
x = clip((rgb / 255) * brightness, 0, 1)
x = x ** (1 / gamma)
```

The paired baseline and DCT candidates must use identical control values. Use
neutral brightness and gamma for the primary source-analysis comparison;
non-neutral tone tests follow separately because this path intentionally applies
tone before DCT analysis. Convert the resulting gamma-encoded RGB planes to
floating-point full-range YCbCr using the same BT.601/Pillow convention as the
receiver:

```text
Y  =  0.299000*R + 0.587000*G + 0.114000*B
Cb = clip(128/255 - 0.168736*R - 0.331264*G + 0.500000*B, 0, 1)
Cr = clip(128/255 + 0.500000*R - 0.418688*G - 0.081312*B, 0, 1)
```

Use the receiver-compatible neutral chroma center `128/255`, not `0.5`; report
the fraction of chroma samples clipped to the full-range endpoints. No 8-bit
YCbCr intermediate is created.

For each Y, Cb, and Cr plane, use the corresponding rows×columns grid from
`model.coder.grids`. Let the full-resolution plane be `P` of size `H×W`, and
let its coder grid be `r×c`:

```text
F = dctn(P, norm="ortho")
C = F[:r, :c] * sqrt((r*c) / (H*W))
grid_plane = idctn(C, norm="ortho")
values = 2*grid_plane - 1
```

This selects the source plane's low-frequency DCT representation directly and
evaluates it on the coder grid. It does not construct an intermediate resized
RGB image. The selected planes are concatenated in the existing Y, Cb, Cr
order and passed through the existing spatial-value/coder interface. The
existing coder then applies the current DCT/rank/fold processing, including
selected fold guests. This interface round trip must recover the selected grid
coefficients within numerical tolerance when no clipping or enhancement is
applied.

The current coder grids and sent corners are:

| Plane | Coder grid, rows×columns | Ordinary sent corner, rows×columns |
|---|---:|---:|
| Y | 96×80 | 48×40 |
| Cb | 48×40 | 24×20 |
| Cr | 48×40 | 24×20 |

All dimensions come from the loaded model. They are not constants in the new
encoder. The ordinary corners describe the current model representation;
fold-500 may also carry selected coefficients outside those corners. Its rank
and guest membership are not equivalent to a simple frequency band.

The existing coder-value convention is bounded to `[-1, 1]`. The compatibility
candidate clips `values` to that range before invoking the coder and reports
the fraction and magnitude of samples clipped. DCT truncation can overshoot
near edges, so clipping is a nonlinear image operation and can alter
coefficients, including guest statistics. Include an unclipped float diagnostic
when the existing model/fold path accepts it; do not describe clipped output as
an exact retained-spectrum round trip. Final RGB display clipping remains at
the receiver output as today.

The DCT uses a finite-frame cosine basis and can show boundary ringing. Report
that behavior rather than claiming that source-domain DCT eliminates aliasing
or all resampling artifacts. The direct-DCT candidate is the reference for
testing reduction strategies, not an assumed visual winner.

#### Invariants

- With `--dct-encode` disabled, output remains byte-identical to the established
  sender. With it enabled and enhancements off, the output is the plain direct
  source-DCT candidate.
- Plane dimensions and the concatenated value count come from
  `model.coder.grids`; the current vector contains 11,520 values.
- A source plane already at its coder-grid dimensions is reproduced within
  numerical tolerance by the DCT/IDCT mapping.
- With enhancements off, flat planes remain constant within numerical
  tolerance. Chroma gain leaves neutral grey unchanged; non-neutral flat-color
  changes are expected.
- Results are finite. The bounded compatibility path returns values in
  `[-1, 1]` and reports its clipping; the unclipped diagnostic is not subject to
  that range invariant.
- Taper does not directly modify coefficients outside its declared sent
  luma region or any chroma coefficients. Any later clipping effects are
  measured separately.

#### Coefficient reduction and perceptual bands

The first `weighted-tent`, `weighted-cosine`, and `weighted-gaussian`
implementation used the wrong frequency coordinate. It mapped source mode
`u` to `u*r/H` (and `v` to `v*c/W`) before averaging signed coefficients.
DCT-II mode indices count cosine modes over the complete image frame; changing
raster dimensions does not rescale those indices. For example, the old
720-to-96 mapping sent source mode 80 to target bin 11 even though its
same-frame target mode is 80, outside the target grid. This mixed unrelated
cosines and folded out-of-band modes into low-frequency output, causing the
nearly constant brown collapse in the archived results.

The corrected implementation retains matching whole-frame mode indices and
applies a gain to each retained coefficient independently. It never averages
signed coefficients, and modes outside the target support are omitted rather
than reassigned to lower bins. For normalized target-axis mode `q=k/N`, the
separable windows are `1-q` (tent), `0.5+0.5*cos(pi*q)` (cosine), and
`exp(-0.5*(q/0.5)^2)` (Gaussian); all preserve DC at unit gain. These are
coefficient-domain low-pass experiments, not spatial resampling operators, and
remain opt-in. The old collapsed rows are invalid as candidate-quality results
and must be replaced by measurements using the corrected implementation.

For a same-frame DCT projection, retain matching low-frequency mode indices
and apply the orthonormal amplitude factor `sqrt(grid area/source area)`, as in
`direct-retention`. A true spatial resampling candidate must instead apply its
declared pixel-domain filter/resampling operator and then transform that
result, or use the exact DCT-domain operator `D_target * R * D_source.T`; a
normalized mean of neighboring signed DCT coefficients is not equivalent.
Exact resampling-kernel probes are recorded in the evaluation section.

The repaired `area-box` candidate uses exact pixel-footprint overlaps to build
separable spatial averaging operators `R_y` and `R_x`, then computes
`dctn(R_y @ plane @ R_x.T, norm='ortho')` on each target coder grid. It does not
average or renumber source-frequency bins. The direct output remains a regular
model-grid value vector and is sent through the unchanged production Fold-500
path. Select it explicitly with `--dct-encode --dct-aggregation area-box`; the
native direct-retention path remains the default DCT transform option. This
correct spatial reducer is not a Fold-native transform and remains an explicit
benchmark/CLI experiment because its full-resolution color-plane preparation
misses the sender latency budget.

The corrected weighted modes remain opt-in in the live sender through
`--dct-encode --dct-aggregation ...`; the benchmark includes them in its default
candidate catalog. The default DCT path remains direct-retention.

#### Fold-native direct projection

`fold-native-projection` bypasses the model-grid value vector entirely. It
builds the union of the actual Fold-500 kept and guest positions, projects the
native RGB frame onto the enclosing source-DCT rectangles (60×50 for luma and
24×20 for each chroma plane), and applies the normal source-to-coder-grid
amplitude factor. The projection is float32 and uses the linear BT.601/Pillow
coefficients around neutral chroma 128/255; it omits only the tiny chroma
endpoint clamp (at most 1/(2×255) per saturated source sample).

The resulting sparse full-grid vector is passed to
`Fold500.encode_dct_coefficients`, which performs the pinned host quantization,
guest clipping, and reserved signature-slot replacement. The folded result then
uses the same `encode_folded_coefficients_packet` pulse, coded-pilot, metadata,
and EOF path as the application sender. No receiver, model, table, or wire
changes are made. This candidate is an explicit still-image benchmark option
(`--variant fold-native-projection`) and requires
`--profile stereo-fold-500`.

Fixed perceptual frequency-band weighting is a separate operation: multiply
each retained coefficient by its declared gain in normalized whole-frame DCT
coordinates. Keep DC at unit gain, use smooth transitions, and declare separate
luma/chroma behavior. For at least one candidate, map a documented spatial
contrast-sensitivity curve to cycles per degree using the fixed display size
and viewing distance; cap and archive the resulting gains. Do not adapt these
gains frame-by-frame in the first experiment; temporal gain changes could make
fine detail flicker. Perceptual weighting changes the reconstructed picture;
it does not add wire capacity.

Also test fixed perceptual frequency-band weighting. Define the curves in
normalized horizontal and vertical DCT coordinates, keep DC at unit gain, use
smooth transitions, and declare separate luma/chroma behavior. For at least
one candidate, map a documented spatial contrast-sensitivity curve to cycles
per degree using the fixed display size and viewing distance; cap and archive
the resulting gains. Include a conservative mid-band luma emphasis and at least
one luma/chroma-weighted profile. Do not adapt these gains frame-by-frame in
the first experiment; temporal gain changes could make fine detail flicker.
Perceptual weighting changes the reconstructed picture; it does not add wire
capacity.

#### Optional pre-enhancement controls

The controls below are implemented opt-in candidates. Their default values are
no-ops; each is valid only with `--dct-encode`:

| Option | Values | Default |
|---|---|---:|
| `--dct-sharpen` | `off`, `taper`, `usm` | `off` |
| `--dct-sharpen-strength` | 0.0–1.0 | 0.25 |
| `--dct-clarity` | 0.0–1.0 | 0.0 |
| `--dct-chroma-gain` | 1.0–1.3 | 1.0 |
| `--dct-aggregation` | `off`, `area-box`, `weighted-tent`, `weighted-cosine`, `weighted-gaussian` | `off` |
| `--dct-band-profile` | `off`, `mid-luma`, `perceptual-color` | `off` |

`off` remains the plain direct-DCT path. Taper and USM are mutually exclusive;
clarity and chroma gain may combine with either. Run a one-factor comparison
before testing combinations.

**Taper sharpen** operates on the source-DCT luma coefficients inside the
ordinary sent luma corner only. With row and column indices `u,v` and sent
corner `R×C` (rows×columns), define:

```text
f = sqrt((u/R)^2 + (v/C)^2)
b(f) = (27/4) * f^2 * (1-f), for f < 1; otherwise 0
C[u,v] *= 1 + strength*b(f)
```

This boost is zero at DC and at the elliptical boundary, and peaks at one for
`f=2/3`. Coefficients outside the sent corner are not directly modified by
taper. If a later pixel-domain clipping step is enabled, it can still change
the resulting coefficient statistics; report that separately.

**USM sharpen** and **clarity** operate on the full-resolution luma plane
before its DCT. Define their Gaussian radii in source pixels from the source
dimensions and model grid, separately for rows and columns. USM uses
`Y += strength*(Y - G_sigma(Y))` with `sigma=0.8` coder-grid pixels; clarity
uses the same form with `sigma=6` coder-grid pixels. Both have broad frequency
responses: clarity is not confined to low/mid frequencies, and either can
increase energy in fold-guest coefficients. Use a boundary rule compatible
with the DCT frame assumptions and measure guest changes; do not claim these
operations stay inside a frequency band.

**Chroma gain** applies `Cb/Cr = neutral + gain*(Cb/Cr - neutral)` using the
receiver-compatible neutral value `128/255`, followed by the declared source
chroma bounds. Neutral grey is unchanged; non-neutral flat colors are
intentionally changed. Report chroma clipping and decoded color error.

#### Candidate batch and evaluation

The original full-matrix run used 19 candidates, including six weighted-average
variants with the invalid source-to-target DCT index mapping. The corrected
default catalog retains all 19 variants so their repaired behavior is measured.
`area-box-resample` and `fold-native-projection` are specialized, explicit-only
probes; the former is not Fold-native and the latter is stereo Fold-500 only.
Run the candidates through the actual modem, using identical full-size
source frames, model/fold tables, wire budget, channel settings, and receiver
display reconstruction:

| Archived candidates | Comparison |
|---|---|
| 1 | Current resize-first sender baseline |
| 1 | Direct source-DCT retention, plain compatibility path |
| 1 | Direct source-DCT unclipped diagnostic, if accepted by the current model |
| 6 | Tent, cosine, and Gaussian coefficient windows, with clipped and unclipped pairs; each applies gains to matching DCT modes |
| 2 | Taper sharpen at strengths 0.25 and 0.5 |
| 2 | USM sharpen at strengths 0.25 and 0.5 |
| 2 | Clarity at strengths 0.15 and 0.3 |
| 2 | Chroma gain at 1.1 and 1.2 |
| 2 | Predeclared perceptual luma/chroma band-weight profiles |

The separate `area-box-resample` and `fold-native-projection` probes are
explicit-only and are not part of the 19-candidate default count.

The direct-retention candidate is the principal test of source-domain analysis.
The clipped/unclipped diagnostic separates coefficient behavior from the
existing coder-value bound. Enhancement candidates are judged independently;
do not combine all gains before the one-factor results are understood.

Use native-resolution 720×960 left-eye images from the SBS face set, without
resizing the eye image, plus high-resolution chart/text and float-precision smooth-gradient
scenes. Include fine edges, saturated boundaries, texture, luminance/chroma
gradients, and moving detail. First compare all variants on the clean channel
with stereo fold-500, mono-fold-500, and mono-colour-500. Then run the fixed
Type II synthetic channel on selected finalists in those same profiles. Type II
is a regression input, not tape emulation.
Render through the receiver's DCT reconstruction at a fixed 1080×900 display
size using the same aspect-preserving viewport and display settings for every
variant. Compare against the original source shown in that same viewport.
The `perceptual-color` curve uses 96-dpi pixels, a 600-mm viewing distance, and
the displayed aspect-preserving viewport to map DCT modes to cycles per degree;
archive those assumptions with the gain curves.

For each scene/profile/channel, deliver numbered side-by-side PNGs and a
contact sheet. Report SSIMULACRA2 as a guard, not the sole judge; inspect the
images for detail, ringing, chroma errors, gradient banding, and clipping. Also
report source-DCT/grid clipping fractions, fold-guest clip fractions and
exceedance magnitudes, decode success, waveform peak/RMS, and encoder time.
Keep three results distinct: (1) the direct source-DCT reconstruction mapped to
the coder grid and enlarged through the receiver-equivalent DCT display path;
(2) an ideal reconstruction using the actual model/fold coefficient selection
without waveform/channel effects; and (3) the modem-decoded result. These
separate source-analysis loss, coefficient-budget/fold-selection loss, and
waveform/channel effects.

Keep all experiment code opt-in and the default byte-identical. Synthetic
clean/Type II measurements guide candidate selection; real tape validation
remains the user's measurement. Archive commands, variant definitions,
comparison images, and results under repo-local `tmp/`.

Run the complete clean-wire matrix (two native SBS images plus the built-in
float-precision gradient and chart/text scenes) with:

```bash
.venv/bin/python tools/v7_source_dct_bench.py \
  --out tmp/v7-source-dct-bench --channel clean-96k
```

The standard stereo Fold-500 benchmark sends candidate value vectors through
the same `modem_v7_display.encode_values_packet` function as the application
sender. `fold-native-projection` instead enters the same production path through
`encode_folded_coefficients_packet` after applying the same pinned
`animation_modem.v7_fold.Fold500`. Both paths use the production coded-pilot
overlay and loop/aspect/source-index metadata. Source-preparation timing for all
variants is the median of seven warmed runs; the projection's PNG-only inverse
transform is excluded. Output adaptation
uses `animation_modem.v7_core.adapt_packet_for_output`, the helper called by
`PacketOutput`; at 1× and a 96 kHz device rate this selects per-packet
`band_limited` adaptation. This bypasses only image compositing/preparation for
the candidate branch under test. Experimental mono profiles continue through
their matching `MonoFreshFoldWire` implementations.

Run Type II for selected finalists with repeated `--variant` options and
`--channel type-ii`; for example:

```bash
.venv/bin/python tools/v7_source_dct_bench.py \
  --out tmp/v7-source-dct-type-ii \
  --variant resize-first --variant direct-retention \
  --variant direct-unclipped \
  --channel type-ii
```

The still-scene matrix sends 12 packets per trial. Run a packet-synchronous
moving-detail sequence (12 changing native 720×960 frames) for the baseline,
direct-retention candidate, and unclipped diagnostic with:

```bash
.venv/bin/python tools/v7_source_dct_bench.py \
  --out tmp/v7-source-dct-motion-clean --motion-only \
  --variant resize-first --variant direct-retention \
  --variant direct-unclipped --profile stereo-fold-500 \
  --profile mono-fold-500 --profile mono-colour-500 \
  --channel clean-96k
```

The motion report pairs decoded frames by packet counter and reports temporal
change error in a fixed source region outside the moving target.

#### Evaluation record

**Production-path correction (2026-09-29):** the earlier V7 source-DCT receive
matrices below used the standalone `tools/v7_live` fold/tone wrapper and
`speed_pulse_stream` at 96 kHz. The application sender instead uses
`modem_v7_display.encode_values_packet` with `animation_modem.v7_fold.Fold500`
and `add_fold500_coded_pilot`; `PacketOutput` adapts each 1× packet with
`band_limited` at the device rate. Those differences change the emitted
96-kHz waveform, so the earlier received-score tables and decoded images are
historical diagnostics, not production-path rankings. Focused clean and Type II
production-parity checks in `tmp/v7-source-dct-production-parity-focused/` and
`tmp/v7-source-dct-production-parity-type-ii/` cover the two native face
images, resize-first/direct-retention/direct-unclipped, and the application
stereo Fold-500 sender. All 144/144 packets displayed and passed EOF validation.
On clean, direct-retention's received SSIMULACRA2 change against resize-first
was +0.32 and +0.01 on the two images; direct-unclipped was +0.35 and +0.14.
On Type II, changes were −0.37 and +0.01 for direct-retention, −0.54 and +0.13
for direct-unclipped. These remain small/inconsistent gains, so no candidate is
promoted. This focused check does not rerun the full 228-row matrix. The
source-grid analyses remain valid, and the pinned standalone/application fold
coefficient identity is covered by `modem_tests.test_v7_application_fold`.

**Fold-native direct projection (2026-09-29):** a partial separable source-DCT
projection now covers Fold-500's actual kept/host/guest/signature support and
enters the unchanged production packet path directly. The clean 12-packet run in
`tmp/v7-fold-native-projection-clean-final/` displayed and EOF-validated all
24/24 candidate packets. On the two 720×960 face images, the received
SSIMULACRA2 changes versus resize-first were +0.35 and +0.14; side-by-side
inspection showed no clear visible improvement. Median full image-to-packet
time was 7.41 ms versus 2.96 ms for resize-first (paired median ratio 2.50), so
the candidate is still well outside the few-percent latency target and is not
promoted. The exact Fold coefficient and packet path is covered by
`modem_tests.test_v7_source_dct`.

The earlier 12-packet finalist run is archived in
`tmp/v7-source-dct-finalists-12pkt/`. It covers resize-first, direct retention,
and the unclipped diagnostic on both native face images and both synthetic
scenes, through clean and Type II channels in all three profiles. All 72 rows
displayed and EOF-validated 12/12 packets (864/864 total). Averaged over the two
native face images, direct retention's received SSIMULACRA2 change versus
resize-first was +0.16 / −0.16 / +0.03 for clean stereo / mono-fold / mono-colour
and −0.14 / −0.06 / −0.34 for Type II. The face results therefore show no
consistent source-DCT gain. Direct retention did improve the synthetic gradient
and chart scores slightly in several profiles; it did not change the profile
ordering or establish a general image-quality win. Unclipped direct retention
was effectively tied with clipped direct retention.

The earlier full clean run is in `tmp/v7-source-dct-clean-final/`; it covers all
19 candidates but predates the 12-packet finalist gate and weighted-index
correction below. Its three signed
coefficient-averaging candidates collapsed fine detail and the smooth gradient
(about −97 mean received SSIMULACRA2 across all four scenes and profiles,
versus about −18 for resize-first). They are rejected as finalists. The
archived 12-packet full clean run is `tmp/v7-source-dct-clean-12pkt/`; it
contains the original weighted-index bug and is not a measurement of the
corrected weighted variants. It contains 228
paired rows (four scenes × three profiles × 19 candidates), all on the clean
96 kHz channel. Every row displayed and EOF-validated all 12 packets: 2,736 /
2,736 packets total. This is packet-level success, not a claim that every
candidate produced a usable image. The six weighted signed-frequency
aggregators collapsed to near-constant images in source preparation, before
modem encoding; their rows are classified separately below. The quality table
reports image-preserving candidates' mean received SSIMULACRA2 change against
resize-first over 12 paired scene/profile comparisons; positive is better.
`Wins` counts strictly positive paired changes.

| Candidate | Mean ΔSSIMULACRA2 | Wins |
|---|---:|---:|
| direct-unclipped | +0.39 | 10/12 |
| direct-retention | +0.35 | 10/12 |
| usm-025 | +0.07 | 6/12 |
| clarity-015 | +0.05 | 6/12 |
| band-mid-luma | −0.05 | 4/12 |
| taper-025 | −0.30 | 4/12 |
| usm-050 | −0.51 | 4/12 |
| clarity-030 | −1.25 | 3/12 |
| chroma-110 | −1.96 | 3/12 |
| taper-050 | −2.18 | 4/12 |
| chroma-120 | −6.21 | 2/12 |
| band-perceptual-color | −9.36 | 1/12 |

The small positive direct-retention aggregates are driven by the synthetic
chart (+0.36 mean) and gradient (+1.02), not by consistent native-face gains:
the two face images averaged +0.08 and −0.06, respectively. Direct-unclipped
was similarly close to baseline (+0.12 and +0.04 on those faces). The
mid-luma profile gained +1.05 on the gradient but averaged −0.64 on each face;
the perceptual-color profile lost 32.95 on the gradient and about 2 points per
face. Direct retention is a near-tie rather than a reliable general improvement.

The six weighted signed-frequency aggregators are **source-transform
failures**, not merely low-scoring modem decodes. Their mean source-grid
SSIMULACRA2 was −96.9 and mean received score was −97.1; visual outputs were
near-constant brown fields because signed averaging cancelled useful AC energy.
All 72 scene/profile rows still passed packet framing (864/864 packets
displayable and EOF-validated), which confirms that this failure occurred in
the candidate's image transform before transmission, not in transport decode.
Their −78.5 mean deltas versus resize-first are retained only as failure
diagnostics and are excluded from the image-preserving candidate ranking.

**Weighted DCT correction and recheck (2026-09-29):** the broken signed-mode
averaging was replaced with same-index coefficient gains. The tent, cosine,
and Gaussian modes now multiply each retained DCT coefficient by separable
frequency windows; no coefficient is mixed with a neighboring mode. The full
21-variant production-path comparison on
`images_sbs/face/00_C_BG_faceSource_960/benFaceSource0000.jpg` used stereo
Fold-500 and the clean 96 kHz channel. All 252/252 packets were displayable and
all EOF markers validated. Received SSIMULACRA2 on the full 1080×900 comparison
canvas was −44.385 for resize-first. The corrected weighted variants scored:

| Candidate | Canvas ΔSSIMULACRA2 | Viewport ΔSSIMULACRA2 | Prep + encode/packet | Relative to resize-first |
|---|---:|---:|---:|---:|
| weighted-tent | −8.84 | −4.80 | 34.02 ms | 11.25× |
| weighted-cosine | −4.80 | −2.79 | 36.70 ms | 12.14× |
| weighted-gaussian | −2.66 | −1.60 | 43.54 ms | 14.40× |

Clipped/unclipped pairs produced identical received scores. The transform no
longer produces the near-constant brown collapse, but the tapers soften detail
and none improves on resize-first. The decoded planes were already reconstructed
to the benchmark viewport (675×900) and centered on the canvas; the initial
score included the shared gray side bars. A second score pass cropped both
reference and decoded images to that exact viewport. Its baseline was −57.767;
the table reports viewport-only deltas. This is a single source/profile
recheck, not the earlier 12-case aggregate. The 21-row report, viewport-only
scores, and comparison sheets are in
`tmp/v7-first-face-weighted-spectral-recheck/`; no candidate is promoted.

Per-trial source-preparation wall times in the finalist run were about 1.8 ms
for resize-first and 38–51 ms for direct source DCT at 720×960. A warmed paired
measurement on the native 0000 face frame (three warmups, ten interleaved
rounds) measured resize-first at 2.19 ms median and direct retention at 47.65
ms median: +2,073% wall time. CPU medians were 2.20 ms and 47.64 ms
respectively, also far beyond the Section 11.6 25% budget. Full measurements
and environment details are in
`tmp/v7-source-dct-finalists-12pkt/timing-pair.json`. No candidate is promoted
and the default remains unchanged. The unclipped diagnostic does not offset
this cost or provide a consistent native-face quality improvement.

#### Fold-aware grid-DCT encode timing

The opt-in `modem_v7_display.encode_image_dct_packet` path transforms the
application's already sampled YCbCr planes directly to the full-grid DCT vector
consumed by Fold 500. It skips the intermediate flattened spatial-value vector
and enters the same `Fold500.encode_dct_coefficients` and production packet
path. It still includes the application's BOX resize to 80×96; it is not the
native-resolution `source_dct_values` projection above, whose preparation cost
remains far outside the encode budget. Since the new path produces a
byte-identical packet, this is an encoder implementation/timing result, not an
image-quality candidate.

Run the paired timing gate with:

```bash
.venv/bin/python tools/modem_v7_dct_encode_bench.py
```

On the native 720×960 left-eye crop, 120 alternating pairs with ten warmups
measured resize-first at 2.833 ms median and direct grid-DCT at 2.822 ms; the
paired median was 0.9882× (−1.18%, −0.034 ms). Packets were byte-identical.
The timer includes source preparation, Fold 500, coded-pilot insertion, and
packet synthesis; it excludes compositing, output-rate adaptation, and audio
device output. The full measurements are in
`tmp/v7-folded-grid-dct-timing.json`; byte-level parity is covered by
`modem_tests.test_v7_wire_profile`.

The unchanged default-wire gate is archived in
`tmp/v7-source-dct-wire-gate/`: all 25 cases produced 12/12 displayable,
EOF-validated outputs (300/300). `hiss-35`, `dropouts`, and `mains-buzz` still
reported some lost decode statuses, while the output path remained displayable
for every packet. This confirms the wire regression gate; it does not make the
synthetic impairments a tape model.

The earlier packet-synchronous motion results are archived in
`tmp/v7-source-dct-motion-final/`. All 18 profile/variant/channel trials
displayed and EOF-validated 12/12 frames (216/216). On this synthetic moving
checker/bar scene, direct retention improved mean received SSIMULACRA2 by
5.1–7.4 points over resize-first, but the decoded high-frequency checker detail
remained visibly soft. Static-region temporal-delta MAE increased slightly
(0.04–0.17 Y levels) in the same comparisons; direct-unclipped was similar or
slightly worse. The sequence therefore shows a motion-detail score gain without
a temporal-stability gain, and does not override the native-face results or the
preprocessing-cost gate.

| Profile | Clean ΔSSIMULACRA2 | Clean Δtemporal MAE (Y) | Type II ΔSSIMULACRA2 | Type II Δtemporal MAE (Y) |
|---|---:|---:|---:|---:|
| stereo-fold-500 | +7.42 | +0.04 | +7.12 | +0.11 |
| mono-fold-500 | +5.82 | +0.09 | +5.09 | +0.17 |
| mono-colour-500 | +5.91 | +0.06 | +5.66 | +0.10 |

#### Follow-on dither and transform probes

The follow-on dither probe is archived in `tmp/v7-dither-wire-probe/`. It
used stochastic rounding at the fold host quantizers, with white, pink, or
brown-correlated threshold fields; the same unchanged receiver decoded all
42 twelve-packet sequences (504/504 packets displayable and EOF-validated).
Across two scenes, three profiles, and both direct-DCT variants, the mean
per-frame SSIMULACRA2 change versus ordinary nearest rounding was −1.25 for
white, −1.15 for pink, and −0.96 for brown. Averaging the 12 decoded RGB frames
offline changed those means to +0.68, +0.73, and +0.77, respectively. That
average is not what the receiver displays; the single-frame dither result is
worse, so colored dither is not a promotion candidate.

The direct-DCT, dither, corrected chroma-transfer, and viewport-fit probes used
the existing frozen M=500 tables only: `stereo-fold-500`, `mono-fold-500`, and
`mono-colour-500`. They did not retune the fold or test M=1000. The replacement
production-parity clean matrix generates packets at the nominal 48 kHz
reference rate, then applies the production per-packet 1× `band_limited`
adapter for 96 kHz output and receiver decode. No receiver-side 48 kHz or
44.1 kHz comparison was run.

Other ideal-fold screens tested stronger luma shelves, coefficient
companding, edge-weighted coefficient fitting, exact signed DCT resampling
kernels, chroma-detail projection into luma, and perceptual choices of fold
host quantizer levels. None showed a repeatable large gain; aggressive shelves
and edge-weighted quantization noticeably regressed the face fixtures, while
the better exact-resampling and chroma-transfer cases were near ties. These
screens and decoded examples are under `tmp/v7-dct-resample-probe/`,
`tmp/v7-edgefit-probe/`, `tmp/v7-chroma-luma-transfer-probe/`, and
`tmp/v7-lattice-*.png`. The probes reinforce the current limit:
with the same 2,880 transmitted coefficients and unchanged DCT reconstruction,
transform-only changes can redistribute error but cannot restore omitted
spatial detail. A genuinely large fidelity jump likely needs a receiver-side
image prior or a larger wire budget.

A corrected chroma-to-luma probe then targeted only the actual 500 fold guest
slots, transferring the least-squares RGB projection of otherwise-unsent
high-frequency Cb/Cr coefficients into those luma modes. The earlier screen
had mistakenly selected chroma modes beyond the entire 48×40 chroma grid and
missed most coefficients lost at the smaller 24×20 transmitted corner. With
the corrected mask, the strength-1.5 ideal-fold screen gained 0.05–0.12
points on the native-face comparisons, but lost on the gradient and chart; its
12-pair mean was −0.03. The actual stereo modem probe is archived in
`tmp/v7-chroma-guest-wire-probe/`: all 144 packets (three scenes × two
variants × two channels × 12 packets) displayed and EOF-validated. Against
direct retention, mean-frame SSIMULACRA2 changed by −0.03 on clean and +0.04
on Type II when averaged across the face, gradient, and chart. The face gained
about +0.10/+0.13 on clean/Type II, while synthetic scenes were flat or slightly
worse. Visual comparison showed no compelling detail recovery, so this remains
rejected.

One additional diagnostic fitted each transmitted DCT mode to the actual
aspect-correct receiver viewport instead of analyzing the source frame's DCT.
After the ordinary FoldCodec quantizer, it averaged +0.05 SSIMULACRA2 over the
12 scene/profile pairs (8 wins), again a near-tie. Bypassing host quantization
and directly placing those fitted values into the receiver's 2,880 model modes
plus fixed guest modes gave a +1.93 mean score reference, but that version is
not encodable through the existing fold quantizer. The outputs and metrics are
in `tmp/v7-display-fit-candidate/` and
`tmp/v7-receiver-subspace-upper-bound/`. The gap between these two diagnostic
results points to host quantization and its fixed codepoints as an important
remaining loss; the practical encoder-only fit recovers almost none of that
idealized gain.

#### Future wire budgets

Separating source DCT analysis from coefficient selection provides a basis for
future supported budgets. This proposal tests only the existing budget and
current model/fold tables. A different budget requires a defined coefficient
map, matching sender/receiver profile, and compatible model/fold tables; it
must not be inferred dynamically from available bandwidth or introduced as an
unannounced wire change.

#### Block-integrated projection timing probe (2026-09-29)

`source_fold_block_dct_coefficients` is an explicit-only approximation probe
for reducing the Fold-native projection cost. A compiled RGB block-average pass
feeds separable projections using the original full-resolution DCT basis
integrated over each block. This keeps retained modes at native-source
frequencies; it is not a same-index DCT of a resized image. It is exact for
block-constant input and loses within-block detail on general images. It sends
the resulting coefficient vector through the unchanged Fold-500 and production
packet/receiver path. The normal sender and wire are unchanged.

The production comparison in
`tmp/v7-under3-block-multi-cached/results.json` covers four 720×960 inputs from the
first `images_sbs/face` folder, resize-first and 8× block projection, stereo
Fold-500, clean 96 kHz output, and 12 packets per case. All 96/96 packets were
displayable and EOF-validated. Received viewport SSIMULACRA2 improved by
+0.17, +0.27, +0.21, and +0.21 (mean +0.21; four wins). Side-by-side viewport
inspection showed no clear visible improvement, so this remains a timing
probe, not a promoted quality transform.

A 200-pair randomized-order profile compares the normal resize/value path, the
byte-identical direct grid-DCT path in the pending sender changes, and the
block-8 candidate. Reproduce all three with:

```bash
.venv/bin/python tools/v7_fold_block_encode_profile.py \
  --source images_sbs/face/00_C_BG_faceSource_960/benFaceSource0000.jpg \
  --source images_sbs/face/00_C_BG_faceSource_960/benFaceSource0016.jpg \
  --source images_sbs/face/00_C_BG_faceSource_960/benFaceSource0037.jpg \
  --source images_sbs/face/00_C_BG_faceSource_960/benFaceSource0044.jpg \
  --out tmp/v7-under3-paired-three-way-profile.json --rounds 200
```

The cached-plan block-8 candidate measured 2.28–2.34 ms/frame median across the
four images; the direct grid-DCT path measured 2.87–3.03 ms and resize-first
measured 2.88–3.01 ms. Paired candidate/direct-grid median ratios were
0.75–0.78. The stated sub-2 ms baseline was not reproduced by this in-process
profile, even for the pending direct grid-DCT sender path. Block-8 P90 was
3.02–3.67 ms, so its median meets the target but its tail latency does not do so
reliably. The cached projector setup is reported separately (about 0.17–2.06 ms
once per source geometry); per-frame timings exclude that setup and source/model
loading. The 7-run quality-benchmark timings are less stable and remain in the
results JSON; use the 200-pair profile for latency comparison. The native
full-resolution projection remains about 5–6 ms preparation before packet
encoding. No transform is promoted.

A 12× block-factor probe in `tmp/v7-under3-block12-paired-profile.json` lowered
the complete-path median to 2.10–2.19 ms/frame and P90 to 2.27–2.82 ms across
the same four images. Its production decode comparison in
`tmp/v7-under3-block12-multi/results.json` lost 0.55–1.10 viewport
SSIMULACRA2 points (mean −0.79; zero wins). All 96/96 packets still displayed
and passed EOF validation, but the quality loss rules this factor out.

### 11.8 Corrective plan: full-resolution preparation optimization

#### Agreed objective and correction of scope

The requested engineering task is to optimize full-resolution pixel preparation
first, then its forward DCTs, while retaining acceptable numerical and decoded
image output. The first concrete task is to split tone/color/clipping costs,
record dtype and allocation behavior, and implement neutral-tone fast paths
plus single-allocation normalization. That work and the partial-DCT experiment
are measured in Steps 1–4 below; neither produced an accepted sender
optimization. The subsequent integer-factor BOX reducer in this section does
improve the resize-first sender: the warmed 100-frame result is about 0.9 ms
for preparation and 2.5 ms for preparation plus packet, versus the pre-reducer
1.7 ms and 3.3 ms baseline. The full-resolution DCT results remain slower
algorithmic comparisons, not the source of the sender improvement.

The block-integrated Fold-500 projection in 11.7 was an assistant-proposed
approximation in pursuit of the earlier image-quality/under-3-ms encode goal.
“4×–12×” describes square source-pixel averaging blocks, not decode settings,
speed multipliers, or added wire capacity. It is a different algorithm that
discards within-block detail. Its measurements do not demonstrate optimization
of the existing full-resolution preparation function. Keep it as an opt-in
research result; do not use its timings to claim completion of this plan.

The long-term goal remains better received images with low complete-encode
latency. The immediate milestones below are preparation targets, not promises
that the full-resolution path will meet the earlier 3-ms complete-encode goal.
Preserve the packet format, receiver, model/fold tables, coefficient budget,
tone ordering, aspect metadata, and normal sender defaults. Support source
geometry through the existing full-resolution input contract; do not introduce
block-divisibility restrictions or an RGB prescale as an implicit optimization.

#### Step 0: repair the commit boundary

Commit `03848213` was described as separate pending V7 integration work, but it
also includes experimental entry points in `tools/v7_live.py` and
`modem_v7_display.py` that import the still-untracked
`animation_modem/v7_source_dct.py`. The separation is therefore incomplete:
those experimental paths can fail on a fresh checkout.

**Progress (2026-09-29):** a repair is committed on and pushed to branch
`fix/v7-source-dct-checkout-boundary` as `96398f35`. It removes the live
`--dct-encode` entry point, native-source capture-preserve-size plumbing, and
`encode_folded_source_packet` dependency from that branch. The independent
direct grid-DCT packet path and its parity benchmark remain because they do not
depend on the untracked source-DCT module. The user's separate receiver-audio
worktree edits were not included. The repair branch has not been merged;
`modem-v7-integration` and its origin still point at `03848213`.

The repair branch was checked without the untracked source-DCT files: 51
targeted sender, capture, audio-source, wire-profile, Fold, and lazy-import tests
passed; `tools/v7_live.py send --help` and compilation succeeded; and the
committed Python tree has no source-DCT import, `--dct-encode`, or
`preserve_size` references. No live audio stream was opened.

The dependency inspection, repair commit, branch push, clean-tree tests, CLI
help, compilation, and dependency scan are complete. The target integration
branch still needs review and merge; keep that review boundary explicit while
full-resolution preparation profiling proceeds on the repair branch.

Before packaging any later opt-in feature:

1. Inspect the committed dependency graph and identify every experimental
   entry point, supporting refactor, test, and documentation dependency.
2. Merge the reviewed repair branch without rewriting the already-pushed
   commit. Keep the removed implementation locally with the experiment.
3. Re-verify the resulting target branch independently of untracked files. Check
   CLI startup, normal sender behavior, lazy imports, and relevant packet-parity
   tests. Do not validate solely against the mixed development worktree.
4. Package any later opt-in feature with all of its dependencies and tests in
   the same coherent change. Report the exact commit contents before any push.

#### Step 1: establish the reference and measurement harness

**Baseline-scope correction (2026-09-29):** the 46.670-ms Step 1 reference is
the original *experimental full-resolution source-DCT function*. It is not the
existing sender baseline. The actual sender resizes first, then computes the
small coder-grid values. A fresh 200-pair production-path profile is saved at
`tmp/v7-source-dct-production-baseline-20260929/current-numba-baseline-profile.json`.
For `benFaceSource0000.jpg`, resize-first preparation measured 1.585/1.822 ms
median/P90 and preparation plus the production packet measured 2.911/3.871 ms.
The four-source profile in `tmp/v7-under3-paired-three-way-profile.json` puts
resize-first total medians at 2.88–3.01 ms. The full-resolution partial-DCT
numbers below were compared with the full-resolution reference, not paired
against this sender baseline; they do not establish a sender speedup. The
candidate did use the fused Numba color kernel, but its separable DCT uses
NumPy/BLAS matrix products; this does not change the baseline mismatch. I did
not carry the Numba block-average kernel into the full-resolution algorithm,
because it is an approximation and the spec keeps block averaging separate.
The existing Numba block-8 projector is measured as its own path: the fresh
profile reports 1.077/1.767 ms preparation and 2.272/3.437 ms
preparation-plus-packet on this image. Keep that block-averaging result
separate from full-resolution DCT conclusions.

**Reference baseline captured (2026-09-29; 200 timed samples per path, 10
warm-ups):** the file hash is
`f1b55bbe2fba878763b4fb526a65fa81853177719a7484ef9fad8641d6163cf0`; the
decoded left-eye RGB array is 960×720×3 `uint8`, C-contiguous, read-only. The
reference output and full report are in
`tmp/v7-source-prep-baseline-20260929/{baseline-output.npz,profile.json}`.
Output values are float64, length 11,520, hash
`121da3288dd5c6714239c443adf5e42ff8c24dc79787488dd220f6788421bd0f`.

On the 4-vCPU AMD EPYC-Milan host (Python 3.13.5, NumPy 2.2.4, SciPy 1.18.1,
Pillow 11.1.0), with no line tracing or competing benchmark, the experimental
full-resolution reference preparation measured 46.670 ms median / 48.660 ms
P90. With that experimental preparation, the production Fold-500
prepare-plus-packet path measured 49.458 / 53.437 ms. Stage medians
(P90) were: normalization/input checks 2.461 (2.738) ms; brightness/tone clip
6.581 (7.143); recomputed brightness diagnostic scan 2.043 (2.220); neutral
gamma branch below timer resolution; YCbCr arithmetic 9.334 (10.585); chroma
clip 0.632 (0.720); chroma diagnostic scan 0.510 (0.578); forward DCTs Y/Cb/Cr
4.411/4.398/4.424 ms; inverse DCTs Y/Cb/Cr 0.058/0.026/0.026 ms. The current
tone and chroma clipping fractions are both zero; the final coder-grid clipping
fraction is 0.00104167. A separate `tracemalloc` call observed a 66,483,492-byte
peak; it is not a latency measurement and may not include all native FFT memory.
The report includes the complete 200-sample raw timing arrays, setup costs,
array layouts, expression/allocation inventory, software, CPU, and thread
environment. This run establishes the full-resolution algorithmic reference
and saved numerical oracle; the existing sender baseline is recorded in the
scope correction above.

Use the same decoded 720×960 left-eye frame from the first `images_sbs/face`
folder, initially `00_C_BG_faceSource_960/benFaceSource0000.jpg`, with identical
settings for reference and candidate. Retain the original full-resolution
`source_dct_values` implementation as an evaluation reference and save its
output values and diagnostics under `tmp/` before modifying the algorithm.
Record source identity, dimensions, dtype, settings, software versions, CPU,
and numerical-library thread configuration.

Measure both individual preparation stages and the complete uninstrumented
call. Warm up all paths; exclude loading, plan creation, and compilation from
steady-state timing and report their costs separately. Alternate reference and
candidate order across at least 200 paired runs. Record median and P90, sample
counts, and raw timings. Run timing jobs serially without simultaneous quality
benchmarks or test suites competing for CPU. Final performance numbers must
come from runs without line tracing.

First split the previously reported 25.13-ms tone/color/clipping stage into:
brightness/tone adjustment, gamma, YCbCr arithmetic, clipping, and diagnostic
scans. Record input/intermediate dtypes, shapes, strides, ownership, and the
full-image temporary arrays each expression creates. Measure allocations
separately from latency when instrumentation affects execution. Timings locate
cost; they do not establish whether precision, allocation, layout, bandwidth,
or arithmetic is the cause.

#### Step 2: normalization and exact neutral-tone fast paths

**Initial Step 2 result (2026-09-29):** on branch
`perf/v7-source-dct-preparation`, the reference implementation is retained as
`source_dct_values_reference`; `source_dct_values` now uses a single owned
float64 normalization destination where conversion is needed and skips the
neutral brightness/clip pass only when normalization proves the input is in
[0, 1]. Neutral gamma remains an identity. Non-neutral tone and accepted float
inputs retain the prior clipping/range behavior. This is still float64.

The comparison artifact
`tmp/v7-source-prep-comparison-20260929/comparison.json` contains 200
alternating reference/candidate pairs (10 warm-ups), raw timings, isolated
stage samples, and separate allocation traces. These timings are for the
experimental full-resolution path, not the existing resize-first sender. Paired
median preparation was 47.795 ms reference / 39.572 ms candidate (−17.2%);
P90 was 51.764 / 43.990 ms (−15.0%). The experimental-preparation-plus-
production-Fold-500-packet median was 49.269 / 40.974 ms (−16.8%); P90 was
53.275 / 45.553 ms (−14.5%). The
candidate's traced peak was 49,894,406 bytes versus 66,483,588 bytes, about one
16.6-MB RGB float64 frame lower. These traced numbers are allocation evidence,
not timings.

For the saved 720×960 frame, values remained bit-identical (max/RMS error 0,
zero changed values); tone, Cb, Cr, and coder-grid clipping decisions had zero
changed entries. Three production packets decoded and passed all three EOF
markers for each path; rendered receiver images were byte-identical (900×675,
zero changed pixels). The source-path tests and all 422 `modem_tests` passed.
This exact normalization/neutral-tone step is complete at float64. The
arithmetic/color work and its final paired result are recorded in Step 3 below;
no reduced-precision transform has been accepted.

Implement and evaluate these first, at the original precision:

- Validate shape and dtype inexpensively. For uint8 input, use its guaranteed
  finite 0–255 range instead of repeated content scans. Preserve necessary
  finite/range checks and the existing accepted float-input behavior.
- Convert to the working dtype once and normalize into that owned destination
  buffer using destination operations, avoiding successive full-image arrays.
- Skip neutral brightness multiplication when the input-range contract makes
  that exact. Skip neutral gamma computation. Avoid redundant copies and
  clipping only where bounds are guaranteed; float inputs need particular care.
- Preserve brightness, clipping, gamma, color conversion, and chroma clipping
  order. Reuse computed expressions for diagnostics without changing their
  meaning or modifying caller-owned input.

Deliver a before/after stage and whole-function timing report, output errors,
and clipping-decision comparisons. Only then benchmark float32 against float64
through the entire preparation call, including transforms and final packing.
Reduced precision is a separately reported tradeoff, not an assumed equivalent
implementation.

#### Step 3: reduce pixel-pass and temporary-array costs

**Step 3 result (2026-09-29):** two original-precision color implementations
were compared. The buffered NumPy arithmetic preserved exact outputs and
lowered peak memory, but its isolated YCbCr/chroma stage was not faster, so the
experimental candidate uses a fused, parallel Numba color loop instead. It
writes the three contiguous float64 planes directly and counts chroma clip
decisions while clipping. At the measured neutral brightness/gamma settings, normalization's
owned RGB buffer is reused directly, so no extra tone buffer is created; the
fused color kernel avoids full-image arithmetic temporaries. For non-neutral
controls, the existing tone/gamma operations remain ahead of this kernel.

The final 200-pair run is saved in
`tmp/v7-source-prep-final-comparison-20260929/comparison.json`. Reference versus
fused candidate median/P90 preparation was 43.293/47.201 ms versus
19.904/22.161 ms. The experimental-preparation-plus-production-packet path
was 44.795/48.803 ms
versus 21.368/23.712 ms, a 52.3% median reduction; encode-only time stayed
about 1.47 ms. The fused candidate's isolated color+chroma median was 0.594 ms;
the reference color arithmetic and chroma clip/diagnostic stages were 8.868 ms
and 1.183 ms. A separate alternating kernel microbenchmark measured buffered
NumPy at 11.100/13.384 ms median/P90 and Numba at 1.457/2.206 ms. Isolated
stage timings are diagnostic; accept on the paired complete-call measurements.

Values, diagnostics, and packet samples were bit-identical. Tone/chroma/grid
clip masks had zero changed decisions, and all three decoded packets passed EOF
validation. Receiver-rendered RGB remained byte-identical with zero changed
pixels. Traced peak memory fell from 66,483,588 to 38,834,644 bytes. This is
allocation evidence, separate from latency.

Numba used four threads. On an empty cache, eager compilation of the color
kernel took 1,339 ms; its first execution after compilation took 4.526 ms and
the subsequent direct-kernel median/P90 was 2.876/3.466 ms. The full candidate's
first preparation call took about 1.49 s when it compiled lazily. The paired
steady-state run excludes this one-time compile/first-call cost. Keep it
explicit in any application startup or real-time decision. The compiled path is
still float64 and is exact against the original expression path on the saved
frame, random RGB input, and saturated primary colors.

Within the experimental full-resolution pipeline, preparation is substantially
faster than its original reference; this does not beat the existing
resize-first sender baseline. The next section evaluates forward-DCT
configuration. Precision remains unchanged.

After the first measured change, optimize tone and color array operations using
owned reusable buffers and destination scaling/clipping. Produce planes in a
layout suitable for the DCT and include that layout's creation cost. Avoid
recomputing brightness/color expressions just to collect diagnostics.

If array operations remain dominant, compare a compiled fused loop combining
normalization, brightness, clipping, gamma, color conversion, and chroma
clipping against the optimized array implementation. Report compilation and
first-call cost separately. Keep the original reference available throughout
evaluation. Exercise neutral and non-neutral brightness/gamma, saturated
colors, values at and around clipping boundaries, and accepted input dtypes.

#### Step 4: forward DCT configuration

**Step 4 full-transform selection (2026-09-29):** the screening run is saved
under `tmp/v7-source-dct-transform-screen-20260929/transform-screen.json` and
covered 23 paired 40-sample configurations, including separate/batched float64
and float32 transforms, workers 1/2/4, and overwrite on/off. SciPy returned the
requested float32/float64 output dtype. Float32 reduced coefficients had max
absolute error `5.96e-7`, RMS `1.36e-7`, and no changed clipping decisions, but
the fastest measured complete-call median was the exact float64 separate,
four-worker, overwrite-enabled setting. A dedicated alternating 200-pair
comparison is in `tmp/v7-source-dct-dct-final-20260929/`:
float64 four-worker overwrite reduced median/P90 preparation from
18.939/21.258 ms to 9.716/11.292 ms. Values, coder-grid clipping decisions,
production packet audio, EOF validation (3/3), decoded grid values, and
900×675 receiver RGB were identical. This remains the best full-transform
reference configuration for the experimental full-resolution pipeline. The
partial-DCT method below is also experimental; it is not the production sender.

**Initial retained-mode separable projection result (NumPy/BLAS, 2026-09-29):**
the first `source_dct_values` experiment evaluated the orthonormal DCT-II
retained corner directly with cached float64 cosine bases and two matrix
products per plane. It did not resize or block-average the source. The
full-resolution reference remains selectable via
`source_dct_values_configured(..., dct_partial=False, dct_workers=4,
dct_overwrite=True)`. The final alternating
200-pair report is `tmp/v7-source-dct-partial-screen-20260929/partial-profile.json`.
Against that best full-DCT configuration, partial preparation measured
7.955/15.067 ms median/P90 versus 19.461/31.329 ms; preparation plus the
production Fold-500 packet measured 9.630/18.130 ms versus 21.151/33.424 ms.
Paired candidate-minus-full P90 deltas were −6.558 ms for preparation and
−4.895 ms for prepare-plus-packet. Packet encoding alone was slightly slower
(1.607/1.891 ms versus 1.462/1.738 ms), but the total path improved. Timing
tails varied on this four-CPU host; paired distributions and raw samples are
preserved in the report.

These are within-experiment comparisons, not a performance win against the
sender. The actual resize-first sender prepared the same frame in 1.585/1.822
ms and completed preparation plus packet in 2.911/3.871 ms in the separate
200-pair profile above. Thus the partial-DCT run's medians were about 5.0× the
sender's preparation and 3.3× its complete path. Since those reports were not
alternated in one paired run, treat the ratios as a scope correction, not a
precision claim; the candidate fails the existing sender's latency gate either
way. The new three-path attempt and the follow-up two-path alternating run are
not used as canonical results because the sender-path timings were unstable:
the three-path run measured 4.700/5.417 ms and the two-path run measured
4.504/5.868 ms, versus 2.911/3.871 ms in the isolated production-baseline
profile. Their raw reports are
`tmp/v7-source-dct-production-baseline-20260929/production-baseline-profile.json`
and `tmp/v7-source-dct-production-baseline-20260929/resize-first-vs-partial-paired.json`.
No full-resolution sender speedup is claimed.

The predeclared float64 tolerance passed: max absolute coder-value error was
`1.14e-14`, RMS `2.18e-15`; on the first image both methods clipped the same
12 grid values, with zero changed mask entries. Basis-plan setup was 4.34 ms
and 1.80 MB, excluded from steady-state samples. Full-call traced peaks were
similar (33.58 MB full
DCT, 33.86 MB partial); only retained coefficient storage fell substantially,
from 16.59 MB to 92 KB. The isolated per-plane transform median/P90 was
4.251/9.066 ms full versus 0.802/1.982 ms partial; the complete paired call is
the acceptance timing.

The four-image report is
`tmp/v7-source-dct-quality-four-20260929/quality-four-images.json`. Each path
used production Fold-500 packet synthesis and the production 96 kHz output
adaptation before receiver decoding. Across four images, both paths produced
12/12 displayable packets and validated 12/12 EOF markers. Packet audio and
decoded grid hashes matched; all four 900×675 viewport comparisons had zero
changed RGB samples/pixels. Across the four frames, maximum absolute/RMS coder
error was `1.14e-14`/`2.22e-15`; there were zero changed grid-clipping
decisions.
`source_dct_values` selects the partial float64 method in the experimental
module; it has not been integrated into the live sender. No tape-channel test
or live playback-latency claim is made.

**Direct Numba projection follow-up (2026-09-29):** in the isolated
`perf/v7-source-dct-preparation` worktree, both retained-mode matrix products
were moved into parallel Numba loops with cached mode-first float64 bases. The
kernel uses `fastmath=True`; the four-image numerical and receiver checks below
passed. This is a genuine Numba projection, not a Numba wrapper around BLAS.

The warmed, alternating 200-pair comparison against the existing resize-first
sender is
`tmp/v7-source-dct-numba-pure-prod-20260929/resize-first-vs-numba-pure-final.json`.
It used four Numba and four OpenBLAS threads with a 50 ms out-of-timer settle
interval between paths. The resize-first path measured 2.245/2.524 ms
preparation and 3.952/4.817 ms preparation-plus-packet; the Numba
full-resolution path measured 13.497/15.720 ms and 15.255/17.308 ms. Paired
candidate-minus-sender deltas were 11.246/13.374 ms for preparation and
11.091/13.250 ms for the complete path. An independent same-thread-setting
sender profile, `tmp/v7-source-dct-numba-pure-prod-20260929/canonical-sender-four-threads.json`,
measured 1.608/2.058 ms preparation and 2.957/4.106 ms complete, confirming that
the Numba candidate does not approach the sender latency gate. The paired report
retains raw samples and order-stratified results.

**100-frame warmup follow-up:** 10 separate frames warmed each path, followed
by 100 distinct measured frames (`benFaceSource0010.jpg`–`benFaceSource0109.jpg`).
The paired raw report is `tmp/v7-source-dct-numba-100frame-warmup-20260930.json`;
the resize-only report is
`tmp/v7-source-dct-numba-100frame-resize-only-warmup-20260930.json`. In the
paired run, resize-first measured 2.007/2.466 ms preparation and 3.729/4.399 ms
complete; Numba partial measured 13.197/15.977 ms and 15.028/17.551 ms. The
Numba first warmup call took 3.290 s including compilation; its tenth warmup
call was 14.350 ms. In the isolated resize-only run, the 100-frame warmed
median/P90 was 1.677/1.906 ms preparation and 3.306/3.689 ms complete. Its
measured first-ten and last-ten medians were stable at 1.681/1.658 ms for
preparation and 3.300/3.269 ms end-to-end. The paired run still showed an order
effect when the resize-first path followed Numba, so the resize-only pass is the
cleaner view of resize warmup behavior.

**Exact current standalone live-build comparison:** because `tools/v7_live.py`
has its own `_values` and LiveFold-500 packet path, it was measured separately
from the `modem_v7_display` profile above. The 100-frame paired report is
`tmp/v7-source-dct-current-live-build-100frame-20260930.json`; it invokes the
current `_values`, LiveFold-500 coefficient/pilot encoder, and 48 kHz speed
adapter, without opening an audio device. With 10 warmup frames, current-build
resize-first measured 2.132/2.620 ms preparation and 3.926/5.025 ms complete;
the Numba candidate measured 14.544/17.887 ms and 16.287/19.724 ms. Paired
candidate-minus-current deltas were 12.544/15.749 ms for preparation and
12.226/16.168 ms complete. The paired sender-only report,
`tmp/v7-source-dct-current-live-build-100frame-resize-only-20260930.json`,
measured the pre-integer-reducer path at 1.706/2.079 ms preparation and
3.315/4.165 ms complete; first-ten and last-ten medians were stable. The Numba
first warmup call took 3.334 s including JIT compilation; the tenth took
16.384 ms. The interleaved profile has an order effect after Numba, so use the
sender-only profile for the clean pre-optimization baseline.

#### Integer-factor BOX preparation fast path (2026-09-30)

The profile suite and artifacts are under
`tmp/v7-source-prep-deep-profile-20260930/`. It covers the current live
`tools/v7_live._values` path, `modem_v7_display._source_values`, the source-DCT
reference and optimized transform variants used by the preparation/transform/
partial/fused comparison scripts, native Fold projection, block-8 projection,
and the exact live packet path. It also stores cProfile data for the complete
`tools/v7_source_dct_bench.py` and `tools/v7_fold_block_encode_profile.py`
entry points. Whole-bench profiling showed why source preparation must be
isolated: the source-DCT quality bench spent 30.8 s cumulatively in 24
SSIMULACRA2 scores, including 12.4 s encoding 65 PNGs, far more than in
preparation.

The current resize-first sender profile attributed about 1.3 ms/frame to
Pillow's generic full-resolution BOX resize on a 720×960 RGB frame. For BOX
input dimensions divisible by the prepared 80×96 grid,
`animation_modem.v7.prepare_image` now uses `Image.reduce` with those exact
integer factors; other dimensions and filters retain the generic resize path.
For the 720×960 face frames this uses 9×10 source-pixel regions. The changed
rounding is bounded by one RGB code value; this is an optimization of the
existing resize-first pipeline, separate from the native-resolution DCT and
the Fold block-projector approximation.

This gain comes from exact integer-factor reduction, **not** from changing
aspect-ratio handling. `prepare_image` already maps every input onto the fixed
80×96 canvas. The fast path is selected exactly when
`width % 80 == 0 and height % 96 == 0`; it works for varying source ratios
(including 640×480, 720×960, and 800×480) and falls back to generic BOX resize
when either axis is not divisible. The transmitted aspect metadata is not
involved in this dispatch.

A reproducible size suite is generated and benchmarked by
`tools/v7_prepare_image_size_bench.py`. It writes 15 deterministic textured
RGB images of different sizes/aspects to
`tmp/v7-prepare-image-size-suite-20260930/images/` and stores every warmed raw
timing sample in `results.json`. Each case has 40 paired runs with randomized
operation order; timings cover RGB conversion plus the 80×96 BOX preparation,
exclude image loading and `image_values`, and use no audio device. The table
reports median/P90 milliseconds for the old generic resize and current
`prepare_image`:

| Input | Path | Generic resize | Current prepare | Median change | Max RGB delta |
|---|---|---:|---:|---:|---:|
| 320×240 | fallback | 0.225/0.262 | 0.228/0.254 | −1.4% | 0 |
| 400×480 | reduce (5×5) | 0.470/0.500 | 0.165/0.182 | +65.0% | 1 |
| 640×480 | reduce (8×5) | 0.593/0.626 | 0.305/0.340 | +48.6% | 1 |
| 800×480 | reduce (10×5) | 0.752/0.772 | 0.345/0.367 | +54.1% | 1 |
| 720×960 | reduce (9×10) | 1.365/1.432 | 0.555/0.599 | +59.3% | 1 |
| 1280×768 | reduce (16×8) | 1.728/2.014 | 0.716/0.984 | +58.5% | 1 |
| 1440×960 | reduce (18×10) | 2.399/2.599 | 0.963/1.178 | +59.9% | 1 |
| 1920×1152 | reduce (24×12) | 3.869/4.244 | 1.683/2.088 | +56.5% | 1 |
| 1280×720 | fallback | 1.655/1.717 | 1.665/1.727 | −0.6% | 0 |
| 1600×900 | fallback | 2.461/2.536 | 2.445/2.503 | +0.6% | 0 |
| 1920×1080 | fallback | 3.482/3.557 | 3.489/3.586 | −0.2% | 0 |
| 2560×1440 | reduce (32×15) | 6.368/6.692 | 2.904/3.422 | +54.4% | 1 |
| 3840×2160 | fallback | 14.599/14.892 | 14.683/15.209 | −0.6% | 0 |
| 721×961 | fallback | 1.377/1.447 | 1.380/1.532 | −0.2% | 0 |
| 1080×1920 | fallback | 3.705/3.902 | 3.732/3.963 | −0.7% | 0 |

For all eight eligible images, coder-value error was at most 0.00380 RMS
(maximum one RGB code value); the seven fallback outputs were byte-identical
to generic resize. Fallback timing differences are within measurement noise
(−0.7% to +0.6%), as expected from retaining the same resize operation. The
suite verifies common HD/UHD inputs too: they do not necessarily qualify just
because width divides by 80 (for example, 1920×1080 and 3840×2160 fail the
height-divisibility check). Rerun with
`.venv/bin/python tools/v7_prepare_image_size_bench.py`; the exact dimensions,
eligibility decisions, accuracy metrics, and raw samples are in the JSON.

#### Old/new smoke and synthetic-torture comparison

The repository's standard modem torture fixture is 180×240. It does not enter
the integer-reduce path, and generic BOX resize versus current preparation is
byte-identical at that size. The size suite and `test_v7_prepare_image.py` now
cover both that fallback behavior and eligible dimensions. To compare the
optimized path under channel stress, the full 25-case default-wire synthetic
torture matrix was also run old/new on a real 720×960 face crop (12 repeated
packets per case, fixed seed). Received/lost counts matched in 24 of 25 cases;
across all 25, both paths decoded and displayed 12/12 frames and validated
12/12 EOF markers. `lowpass-4k` had 10/12 metadata-valid packets in both runs
and was the same absolute matrix failure for both.

Under `soft-saturation`, the old/new `received` status was 10/12 versus 9/12 for
that particular crop, although both still displayed all 12 frames. A follow-up
on four unique face crops gave received counts 10→9, 5→6, 11→11, and 12→12;
each old/new run displayed 12/12. Thus the one-code rounding change can move a
borderline packet's received/lost classification in a severe synthetic
channel, but this sample did not show a systematic direction or a display
drop. The clean paired decode check across the four unique crops remains
12/12 with no changed clipping decisions. Raw matrix results are in
`tmp/v7-source-prep-deep-profile-20260930/torture-old-vs-reduce/`, including
`comparison.json` and `soft-saturation-multi-image/comparison.json`.

In a warmed 100-distinct-frame sender-only run (10 warmups, 50 ms settling,
the current LiveFold-500 encoder, coded pilots and 48 kHz packet adapter, no
audio device), preparation measured **0.904/1.125 ms median/P90** and
preparation-plus-packet **2.535/2.952 ms**. Against the sender-only profile just
above, that is −0.802 ms median preparation (−47%) and −0.780 ms median total
(−24%). The raw report is
`tmp/v7-source-prep-deep-profile-20260930/current-live-resize-reduce-100frame.json`.
The four-distinct-image quality check is
`box-reduce-quality-four-unique.json`. Across the four 3:4 face frames, coder
values changed by max/RMS `0.00784`/`0.00206–0.00212`, with no changed clipping
decisions. Every resize and reduced path decoded 12/12 packets and validated
12/12 EOF markers. Received Y-PSNR and Y-MAE were marginally better for all
four samples. Decoded RGB differs from the old resize path by MAE
`0.213–0.254` and at most three code values; packet audio is therefore not
byte-identical, while the measured picture quality did not regress.

`profile.json` includes warmed cProfile summaries, allocation probes, and
steady-state call timings; each `.pstats` file can be opened with `pstats`. On
the updated live path, `Image.reduce` is about 0.48 ms under cProfile and
`image_values` about 0.27 ms. In the final warmed isolated profile, both the
Numba partial-DCT and optimized full-DCT preparation calls were about 11 ms,
versus 0.74 ms for the current live preparation path. Traced whole-call peaks
were about 0.19 MB for the live path, 33.6 MB for optimized full DCT, and
33.9 MB for Numba partial DCT. The reducer improves actual default sender
preparation; the experimental full-resolution DCT algorithms remain slower.

#### Aspect-aware DCT canvas geometry follow-up (2026-09-30)

The three-bit V7 aspect field selects one of eight fixed display ratios; it is
not an arbitrary exact-ratio value. The viewer uses that selection to choose
the display viewport. For the tested 720×960 sources, the code is 3:4 exactly.
This means preparation may map the source non-uniformly onto the fixed 80×96
sampling grid and let the viewer's 3:4 viewport restore the intended outer
frame geometry. The metadata does not undo arbitrary crops, zooms, or shifts;
those changes remain in the decoded pixels and must be scored as such.

To test whether a coder-grid-aligned DCT raster makes the math cheaper, the
source was Lanczos-mapped to `(80k)×(96k)` before source DCT. These canvases
match the 5:6 luma-grid aspect and make every input dimension the same integer
multiple of its coder-grid dimension. All candidates used the same aspect code,
default Fold-500 packet path, and decoded viewport. Across four distinct
source frames and 40 uninstrumented
preparation samples per variant, `k=6` (480×576) reduced DCT preparation from
**37.57/55.45 ms median/P90** for native 720×960 DCT to **22.58/24.15 ms**,
while mean received Y-PSNR remained
27.5772 dB versus 27.5773 dB for native DCT. Decoded RGB differed from native
DCT by mean 0.086 and at most two code values. All candidates decoded and
validated 12/12 packets. This is a meaningful geometry result for the DCT
experiment, but it is still far slower than current resize-first preparation
at 0.71/0.80 ms.

The aligned-grid sweep also tested smaller and larger multiples. `k=1` and
`k=2` measured 7.25/7.49 ms and 9.00/9.51 ms, respectively; `k=10` creates an
800×960 canvas, slightly upscaling the 720-pixel source width to match the
sampling-grid aspect, and measured 53.77/64.20 ms. Its decoded result was almost
identical to native DCT, with mean RGB error 0.008 and maximum one code value,
but it was slower. The raw report is
`tmp/v7-source-prep-deep-profile-20260930/aspect-aligned-grid-dct-sweep-k1-6-and10.json`.

The aligned rasters were also tested with the exact Fold-support matrix
projection, which avoids reconstructing spatial planes before Fold-500. With
four BLAS threads, 40 warmed preparation samples per variant gave median prep
times of **0.689 ms** for resize-first, **4.933 ms** for native Fold
projection, **6.947 ms** for aligned `k=1`, **8.040 ms** for `k=2`, and
**13.625 ms** for `k=6`. The native projection's mean received Y-PSNR was
27.5787 dB; aligned results stayed within 0.005 dB, and all decoded 12/12
packets. A one-BLAS-thread run did not improve those prep medians. This route
skips the DCT/image/DCT detour, but remains slower than the optimized sender
path; raw samples for four and one BLAS threads are in
`aspect-fold-projection-sweep-openblas4.json` and
`aspect-fold-projection-sweep-openblas1.json`.

A separate zoom-and-crop sweep did not support using overscan zoom to improve
the truncated DCT. Relative to native DCT, 1%, 2%, and 4% centered zooms reduced
mean received Y-PSNR by about 1.07, 2.82, and 6.34 dB; the tested ±2/±4-pixel
crop offsets did not recover those losses. Pure ±1/±2-pixel translations also
did not improve on the unshifted native-DCT result. Those results are in
`aspect-geometry-zoom-crop-shift-sweep.json` in the same artifact directory.
The useful next geometry constraint is therefore matching the DCT analysis
canvas to the fixed coder grid and the decoded viewport—not assuming metadata
will reverse zoom, crop, or translation.

#### Direct-DCT outcome and decision (2026-09-30)

The direct-DCT experiment in `animation_modem/v7_source_dct.py` analyzes source
pixels before the ordinary 80×96 resize, projects the retained spectrum onto
the existing coder-grid value-vector contract, and exercises the unchanged
Fold-500 packet/decode path. I compared native 720×960 analysis with rasters
mapped to integer multiples of the 80×96 coder grid, and separately tested an
exact Fold-support projection that bypasses the grid reconstruction round trip.

Matching the analysis raster to the coder-grid aspect cut the full-DCT median
from 37.57 ms to 22.58 ms at 480×576 (40% faster than native DCT; P90 improved
from 55.45 to 24.15 ms), with effectively unchanged decoded quality. The exact
Fold-support projection measured 4.93 ms on the native raster and 6.95 ms at
the smallest aligned raster. Both remain substantially slower than the current
resize-first preparation path at about 0.7 ms. Small zoom-and-crop experiments
reduced decoded Y-PSNR rather than improving it. The conclusion is to retain
direct DCT as an explicit, measured experiment—not to use it as the sender
preparation optimization or change the default path. The size-gated Pillow BOX
reducer is the measured sender improvement; its dispatch is independent of
aspect metadata.

A shared three-plane Numba kernel was also evaluated in a randomized 100-frame
comparison (`tmp/v7-source-dct-numba-fused3-100frame-20260930.json`). Despite a
small isolated-kernel improvement, its complete preparation was 1.899/4.448 ms
slower median/P90 than the per-plane Numba calls, and preparation-plus-packet
was 2.003/4.275 ms slower. It was removed from the candidate implementation;
the per-plane Numba result remains the measured version.

A 100-sample thread-count sweep of the per-plane Numba transforms
(`tmp/v7-source-dct-numba-thread-sweep-20260930.json`) found four threads best:
the three-plane transform median/P90 was 5.225/7.023 ms at four threads,
compared with 2.092/2.505 ms for the NumPy/BLAS projection. One, two, and three
Numba threads were slower (16.498/17.364, 7.865/8.588, and 5.664/6.578 ms).
The Numba outputs were bit-identical across these thread counts. This confirms
that thread tuning does not close the projection-performance gap.

The same candidate was slower than the best full-DCT path as well (see
`tmp/v7-source-dct-numba-pure-quality-20260929/full-dct-paired-profile.json/partial-profile.json`):
10.477/12.820 ms preparation versus 9.588/12.377 ms, and 12.101/14.664 ms
preparation-plus-packet versus 11.139/14.394 ms. The isolated per-plane
transform was 4.794/11.947 ms for the Numba loops, compared with 1.118/2.278 ms
for the NumPy/BLAS projection and 5.035/10.534 ms for the full SciPy DCT.
Cached basis creation took 4.287 ms and 1.80 MB; the first Numba call including
compilation took 1.971 s in the full-DCT profile (3.363 s in the production-pair
process). The float64 result is 11,520 values / 92,160 bytes. Traced whole-call
peaks were 33.58 MB for full DCT and 33.86 MB for Numba partial DCT.

The four-image report is
`tmp/v7-source-dct-numba-pure-quality-20260929/quality-four-images.json`.
Maximum/RMS coder-value error was `9.55e-15`/`2.04e-15`; clipping decisions
did not change. All 12/12 packets were displayable with validated EOF markers,
packet audio matched, and all four decoded 900×675 viewports had zero changed
pixels. The direct Numba kernel therefore preserves the checked image result,
but it is not a preparation or sender-performance optimization. It remains an
isolated experiment and is not integrated into the live sender.

The source-DCT prototype has 18 focused regression tests; the integer-factor
BOX reducer adds two more preparation checks.
`python -m unittest discover -s modem_tests -v` passed all 468 tests, including
the integer-factor reducer regression checks; the 12-test modem-integration/
lazy-import suite passed, and `tools/v7_live.py send
--help` completed successfully. These checks ran after the steady-state timing
jobs.

Once pixel preparation has been measured and optimized, compare:

1. Float64 and float32, checking actual transform output dtype and downstream
   numerical error rather than assuming single precision is retained.
2. Contiguous planes and the current layout, charging any layout conversion to
   preparation time.
3. Three separate transforms and a batched transform over spatial axes only;
   never transform across the color-plane axis.
4. A small explicit worker set, such as 1, 2, and 4. Choose by complete-call
   latency and P90, with the host/thread environment recorded.
5. Overwrite-capable transforms only for buffers whose contents can safely be
   consumed.

Verify orthonormal scaling and retained-mode coordinates against the reference.
The retained-mode variants above complete the current partial/separable
projection experiment. Any further projection change must include all
preparation/layout costs and compare against both the best full DCT and actual
resize-first sender. Fewer output coefficients are not evidence of a faster
implementation. Reuse prior projection code only where its numerical contract
matches this path; do not substitute block averaging.

#### Step 5: combined numerical, image, and encode acceptance

For every retained change, require an improvement in the complete preparation
call, not only its isolated stage. Report median and P90 and investigate tail
regressions. Exact fast paths should preserve reference results; for changes
in precision or arithmetic order, report maximum absolute error, RMS error,
and changed clipping decisions/counts. Declare and justify numerical tolerances
before accepting a candidate rather than choosing them to fit observed errors.

Run promising candidates through production Fold-500, packet synthesis, output
adaptation, and receiver decoding on the same four first-folder face images.
Inspect received viewport comparisons and report image metrics separately from
displayable-packet and EOF-validation counts. A numerical near-tie or successful
packet decode alone is not proof of acceptable picture quality. Recheck full
preparation-plus-packet encoding directly; do not add an assumed fixed encoder
cost to preparation measurements. Distinguish this timing from output-device
and real-time playback latency. Synthetic channels are not tape validation.

Use this reporting table, including median/P90 for new measurements. The
resize-first column is the existing sender; full-DCT and partial-DCT columns
are experimental full-resolution paths. The block-8 column is a separate
approximation and must not be treated as a full-resolution result.

| Measurement | Existing resize-first sender | Full-resolution DCT oracle | Initial partial-DCT (NumPy/BLAS) | Numba block-8 approximation |
|---|---:|---:|---:|---:|
| Source preparation | 1.585/1.822 ms | 19.461/31.329 ms | 7.955/15.067 ms | 1.077/1.767 ms |
| Preparation plus packet | 2.911/3.871 ms | 21.151/33.424 ms | 9.630/18.130 ms | 2.272/3.437 ms |
| Isolated forward DCT, per plane | Not applicable | 4.251/9.066 ms | 0.802/1.982 ms | Not measured |
| Numerical comparison | Different resize-first sampling basis | Float64 full-resolution oracle | Four-image max/RMS `1.14e-14`/`2.22e-15`; zero changed clipping decisions versus oracle | Approximate; see Section 11.7 |
| Received viewport comparison | Quality baseline | Full-DCT receiver render | Four images: zero changed pixels versus oracle | See Section 11.7 |

The later direct-Numba comparison uses its own warmed 200-pair profile and the
same-process sender baseline; its full-DCT comparison uses the paired
full-resolution profile. These measurements are kept separate from the
first-iteration table above:

| Measurement | Resize-first sender | Full-resolution partial-DCT, direct Numba |
|---|---:|---:|
| Source preparation, median/P90 | 2.245/2.524 ms | 13.497/15.720 ms |
| Preparation plus packet, median/P90 | 3.952/4.817 ms | 15.255/17.308 ms |
| Numerical/clipping result | Different resize-first sampling basis | Four-image max/RMS `9.55e-15`/`2.04e-15`; zero changed clipping decisions versus full-DCT oracle |
| Receiver result | Quality baseline | 12/12 displayable and EOF-validated; zero changed viewport pixels versus full-DCT oracle |

The separate paired full-resolution comparison was:

| Measurement | Full-resolution DCT | Full-resolution partial-DCT, direct Numba |
|---|---:|---:|
| Source preparation, median/P90 | 9.588/12.377 ms | 10.477/12.820 ms |
| Preparation plus packet, median/P90 | 11.139/14.394 ms | 12.101/14.664 ms |
| Isolated forward transform per plane, median/P90 | 5.035/10.534 ms | 4.794/11.947 ms |

The first, NumPy/BLAS production-baseline and full-resolution results are
separate runs, so their cross-column ratios are a scope correction, not a paired
acceptance result. The direct-Numba sender comparison above is paired; its
full-DCT comparison is a separate paired full-resolution profile. Stage medians
need not sum to whole-function medians. Selection, scaling,
inverse DCTs, and packing were together about 0.58 ms in the full-resolution
experiment; that optimization opportunity does not alter the sender-baseline
comparison.

#### Milestones and communication contract

- **First deliverable:** baseline stage split, dtype/allocation findings,
  single-allocation normalization, and exact neutral-tone fast paths with
  whole-call and numerical comparisons.
- **Below 35 ms preparation:** engineering target for redundant-pass removal
  and buffer/neutral-path work, subject to measured evidence.
- **Below 25 ms preparation:** engineering target for fused pixel preparation
  and validated DCT precision/layout changes, not a prediction.
- **Reassessment:** the initial NumPy/BLAS retained-mode projection passed the
  declared numerical and four-image receiver checks and improved timing against
  the full-resolution oracle, but remained slower than the resize-first sender.
  The direct Numba port also passed quality checks, but was slower than both the
  NumPy/BLAS projection and full-DCT path. Neither is an accepted sender
  optimization. Any future full-res candidate must be timed directly against
  that sender baseline; block averaging remains a separate approximation
  requiring an explicit scope decision.

Each progress report must state which requested step was completed, what code
changed, the input/settings tested, reference versus candidate median/P90,
numerical and image findings, and what remains unmeasured. Distinguish a
benchmark-only prototype from a live-sender feature and an experiment from an
accepted optimization. Preserve results under `tmp/` and record durable
conclusions here. Do not describe work as unrelated or independently shippable
until its committed dependencies have been checked.

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
callback. The receiver auto-selects mono-fresh or mono-colour decoding from the
in-band coded profile status; those manual profile overrides are CLI-only. The
display menu provides eleven presentation choices: nearest,
bilinear, sharp-bilinear, Mitchell bicubic, Spline36, Robidoux, Robidoux Sharp,
cubic B-spline, Kaiser-windowed sinc, Hann-windowed sinc, and EWA Jinc. Nearest
remains the initial display default and preserves the legacy 8-bit RGB
conversion and nearest-sampled shader path. All other modes reconstruct float
Y/Cb/Cr planes; selecting a mode does not alter transport decoding or decoded
exports. Float planes, the shader, and any filter intermediates are allocated
when a non-nearest display mode or DCT reconstruction is selected. Advanced
receiver settings add DCT reconstruction choices Off, 2×, 4×, 8×, and Viewport
size. Off is the default; fixed scales reconstruct a larger cosine grid before
the display scaler, while Viewport size reconstructs directly at the picture
viewport's framebuffer dimensions and refreshes when that viewport changes.
Standalone image-sequence preview is available as
`.venv/bin/python tools/v7_viewer.py IMAGE...`.

Status: implementation brief, September 28, 2026. The receiver and preview
implement the eleven display choices above; the receiver also exposes the
display-only DCT reconstruction choices in Advanced settings. Nearest and DCT
Off remain the reference/default. The additional filters and DCT reconstruction
are experimental presentation choices; no new mode is claimed to have won. The
custom V7 Perceptual path below remains a proposal, not an available display
mode.

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
the final display scaling when DCT reconstruction is Off. With DCT
reconstruction active, the selected kernel runs in the display shader so its
intermediate does not grow beyond the selected DCT grid. EWA Jinc uses a radial
kernel in the display shader; its LUT is generated on demand. The filter choices
are opt-in and affect only presentation. Keep source planes unclipped until the
final RGB conversion.

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

Avoid rebuilding DCT intermediates on toolbar redraws. Fixed 2×/4×/8× dimensions
bound the CPU work independently of desktop resolution; Viewport size is
intentionally viewport-resolution-dependent. The final scaling pass still costs
screen-resolution-dependent GPU work.

### 12.4 DCT reconstruction: precise implementation boundary

The mailbox contains spatial samples, not coefficients. For each plane, compute
a 2D DCT-II with orthonormal normalization, copy the shared low-frequency corner
into a zero-filled array of the target dimensions, multiply coefficients by
`sqrt((target_height*target_width)/(source_height*source_width))`, and apply the
matching inverse transform. The supported targets are Off (no transform), fixed
2×/4×/8× grids, or the picture viewport's framebuffer dimensions. Viewport mode
uses the same horizontal and vertical scale ratios for the reduced chroma
planes, and rebuilds when the picture viewport changes.

This samples the same cosine expansion on the target cell-centered grid.
Account for the corresponding sample centers in GPU texture coordinates. Do
not compare every Nth enlarged pixel directly to an original pixel: their
centers do not generally coincide. Verify against direct cosine evaluation at
selected coordinates instead.

Do not clip planes before this operation or apply a new rectangular cutoff.
Test constants, single basis components, amplitude, orientation, grayscale,
and boundaries. Keep it display-only and perform transforms on picture updates
or when the target viewport changes, not on toolbar redraws. Lazily load the
existing CPU transform dependency. This is a cosine-consistent presentation of
decoded samples, including their noise; it is not recovery of the lost source
image.

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

The 12-chip tone status words repeat `0011`, `0101`, and `0110` three times.
Their complements `1100`, `1010`, and `1001` supply the three mono words. The
six identities are also encoded in the biphase pulse preamble. Their equal-weight
words preserve the canonical 256-sample pulse span and edge count, with minimum
pairwise Hamming distance six; the historical preamble remains the Fold-500
word. The compiled Schmitt detector identifies the profile from short/long edge
intervals in the same pulse acquisition pass. Live profile probing therefore
does not run the tone-chip demodulator. Tone status remains on the wire and the
frame decoder checks it against the pulse identity while using it for pilot
despreading and tone timing.

The current pulse metadata has no spare profile bit. Its aspect, encoding,
tail slice, direction, source index, and CRC/loop-mask fields are allocated. The
profile identity uses the existing sync preamble rather than a metadata bit.

**Mono requires an updated receiver for the initial prototype.** Existing
receivers can display received or displayable packets without recognized coded
status; rejecting an unfolding table does not reject a different rank layout.
An unknown status or changed CRC mask must not be advertised as making old
receivers hold. They may display incorrectly assigned coefficients.

New receivers must validate layout identity before body interpretation and
picture/tail-state updates. An unknown or invalid layout holds the last good
picture. A known layout may show a damaged but decodable picture; an
undecodable packet holds the last good picture. There is no black fallback.

Mono profiles use a distinct same-length pulse word, so older receivers that
only recognize the historical Fold-500 preamble will not acquire those mono
packets. Compatibility with other pre-existing pulse matchers is not a rejection
guarantee; test cross-version receivers before freezing compatibility claims.

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
`MONO_1000` identifies the colour-weighted all-fresh profile in §13.9. The
offline all-fresh fold-off comparison that temporarily used `MONO_1000` has
been retired. The existing `MONO_OFF` wire remains 992 fresh plus 272 rotating
slots, with 16 padding positions on the final phase. Clean synthetic decode
confirmed that all seven phases cover the 2,880 corner coefficients and that
the channel fit/equalizer run with S prior zero.

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

Real-time playback remains outstanding. At the time of this historical check,
the available device list contained PulseAudio `pulse`/`default`, not
`BlackHole 2ch`, so no real audio stream was opened. The current host inventory
and Xvfb smoke check are recorded in Section 1; they do not retroactively
validate playback. Synthetic loopback does not substitute for that validation.

### 13.8 All-fresh mono video with a 500-class fold

The video-only profile is selected as `--profile mono-fold-500`. The sender GUI
defaults to stereo `fold-500`; selecting this mono profile does not alter the
legacy `--experimental-mono` or the `MONO_OFF` rotating wire:

```text
.venv/bin/python tools/v7_live.py send --source test \
  --device 'BlackHole 2ch' --profile mono-fold-500 --mono-video-side right
.venv/bin/python tools/v7_live.py receive \
  --device 'BlackHole 2ch'
```

The default receiver starts with the ordinary stereo Fold-500 decoder and
classifies the profile-coded pulse preamble independently on each input leg.
Three distinct, matching `MONO_500` words switch the decoder to this profile on
the third packet; the first two candidate packets are held. Right is preferred
if both legs validate. Three matching stereo Fold-500 words switch back in the
same way. An ambiguous or unsupported pulse word is an invalid observation and
never inherits the preceding packet's mode. A profile transition resets
profile-specific tail state but keeps pulse acquisition. Input gaps and
scale-aware sync inactivity clear a pending three-packet candidate. The frame
decoder also checks the tone-coded status against the pulse identity before
displaying the packet. The GUI leaves legacy profile and decoder-tuning switches
out of its setup fields; those specialized CLI paths remain available for
recovery and experiments.

The per-leg profile probe now uses the existing compiled Schmitt edge finder to
match the short/long interval word; it does not separately decode the 12 tone
chips. A warmed 2,000-packet synthetic mixed-profile run (48 kHz stereo,
1,024-sample blocks, Linux x86-64) reduced total receiver CPU from 29.19 s
(35.75% of one core over 81.67 s of audio) to 6.43 s (7.87%), a 78% reduction.
The profile-probe scan fell from 24.93 s to 2.38 s; its per-block p95 dropped
from 16.08 ms to 0.63 ms. All three transitions occurred, 1,999 valid statuses
were observed, and all 1,993 submitted displayable frames passed EOF validation.
This is a warmed synthetic host profile, not a capture-card or tape result.

It requires the canonical box model, coded pilot timing, and EOF framing.
Optional pre-encode perceptual resizing changes picture samples before encoding
but does not change the wire layout. Every packet transmits the same 1,264
M-only physical slots: the protected 208-coefficient head followed by 1,056 fresh body
slots. There is no rotating tail and the live receiver disables tail memory.
Five hundred hosts within the fresh body carry the fold; the final 16 hosts are
the fold-table signature and replace their host and guest picture values. The
remaining 484 folded guests are the next box-model corner ranks. This gives
1,732 current-frame picture coefficients at most (1,248 host-picture values
plus 484 guests), with the wire budget often described approximately as 1,750
coefficient positions. The fold step and complete table identity are pinned in
`test_modem_v7/mono_video.py`.

The sender's `--mono-video-side left|right` selects the output leg for the
complete modem signal, including timing/status tones; the other output leg
carries source audio. The default mapping is **left audio, right video**. The
receiver defaults to `--mono-video-side auto`: it tries the right input leg
first and probes the other leg for a consistent pulse train and `MONO_500` pulse
word. The frame decoder confirms that identity against the coded tone status
before displaying the packet. If right and left both carry valid mono video,
right remains selected. Explicit `left` or `right` disables
automatic side selection. Once selected, pulse acquisition and decoding use
only that video leg, ignoring audio on the other. The sender GUI exposes the
side choice and soundtrack/input-device/off audio-source selector. Source audio
defaults to the video's embedded soundtrack; when no track is present the
channel is zero-filled. An explicitly selected input device can provide
microphone, line, or loopback audio. Independent gain and additional delay
controls affect only that channel. The output stream remains stereo and the
ordinary receiver does not decode or delay the audio leg.

The sender GUI exposes an input-device audio source for every capture source
when this mono-video profile is selected. Capture FPS is a dropdown populated
from the selected camera driver's reported modes (or video frame rate / display
refresh choices when available). Brightness and gamma can be edited from Setup
while sending; updates apply to subsequent captured frames without restarting
the audio stream.

For example, use the embedded soundtrack (the default), an explicitly selected
input device, or silence:

```text
--source-audio source
--source-audio device --source-audio-device 'BlackHole 2ch'
--source-audio off
```

Audio is inserted after packet speed conversion, so its pitch and sample rate
remain natural while the modem packet duration changes with `--speed`. Startup
waits for up to one emitted packet of source samples to preserve the audio
beginning; any remaining underflow is zero-filled. The presentation delay uses
the actual emitted packet sample count, plus the optional
`--source-audio-delay-ms`. This delay therefore scales with speed and output
rate. The first captured video frame is held for the first packet during that
startup wait, preserving the visual event paired with the initial audio
samples. A bounded fractional reader tracks independent capture/output clocks
with at most 0.1% rate correction; extreme overflow still drops oldest samples
to cap latency, while underflow is zero-filled. Logged runs report maximum
clock correction and dropped samples. The synthetic 500-ppm drift test models
over 80 seconds and confirms the FIFO stays bounded without sample drops.

On POSIX hosts, video and embedded audio share one FFmpeg demux/process and
source clock; the audio output preserves late audio-stream start timestamps as
leading silence. A real FFmpeg smoke check confirmed a soundtrack beginning at
200 ms stays silent until that point, and a video with no audio yields silence. The
burst-offset unit test sends a known burst beside real mono-encoded packets and
locates it from the pulse-counted header, including at 2× packet speed. A live
flash/click test is still needed to calibrate any residual source-start offset.

`MONO_500` is the distinct coded status for this rank map. The default adaptive
receiver checks the status before image equalization, holds packets from a
candidate layout, and changes its decoder only after three matching packet
statuses. The standalone `MonoFreshFoldWire` gate also rejects a different
status when that specialized decoder is selected explicitly. This is not a
promise that pre-existing receivers reject the new profile; they do not
implement this check. `MONO_1000` identifies the colour-weighted profile in
§13.9. A previous offline fold-off control used that status during the
historical comparison below; the control is no longer in the wire
implementation or current benchmark.

The runner `tools/v7_mono_video_bench.py` compares the current stereo fold-500
decode, that stereo wire downmixed to mono, the rotating mono fold-off profile
with tail memory on/off, and both all-fresh mono 500-class folded profiles. The
removed all-fresh fold-off control remains only as a historical row in the
original comparison below. The mono video profiles use the selected video leg;
the other leg is silent in this video-only benchmark. It scores packets 8–23
against the matching generated source frame and reports decoded,
current-frame-displayable, held-picture, EOF, encode and decode costs
separately. The encode/decode times are local wall time per packet (not process
CPU), measured after decoder warm-up. Pans, six-packet cuts, a moving graphic
blob, and a colour chart are deterministic synthetic scenes; the previous
review script `tmp/mono/video.py` is unavailable in this checkout, so these are
scene-class substitutes, not identical stimuli. The selected Type-II and
fast-flutter cases are limited synthetic regressions, not cassette emulation.
No real-tape result is inferred from them.

The original comparison, before the colour profile was added and the fold-off
control retired, contained four 24-packet scenes, with 16 scored frames per
scene and profile/channel pair. Its mean SSIMULACRA2 across the four scenes,
plus mean per-packet wall time, was:

| Profile | Clean | Type II | Fast flutter | Encode ms/packet | Decode ms/packet |
|---|---:|---:|---:|---:|---:|
| Stereo fold-500, both legs | −4.70 | −16.67 | −13.03 | 1.425 | 1.064 |
| Stereo fold-500, mono-summed | −34.52 | −36.99 | −35.21 | 1.425 | 1.032 |
| Rotating mono fold-off, tail memory on | −54.65 | −54.98 | −54.88 | 1.511 | 1.260 |
| Rotating mono fold-off, tail memory off | −39.85 | −40.44 | −40.55 | 1.511 | 1.231 |
| All-fresh mono fold-off, left leg | −34.50 | −35.80 | −35.44 | 1.346 | 1.159 |
| All-fresh mono 500-class fold, left leg | **−29.23** | **−35.02** | **−34.14** | 1.496 | 1.180 |

All 72 rows in that historical run received and displayed 24/24 packets, held no
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
  modem_tests.test_v7_send_gui modem_tests.test_v7_receiver_audio -v
.venv/bin/python tools/v7_mono_video_bench.py \
  --out tmp/v7-mono-colour
```

### 13.9 Mono colour-weighted fold (MONO_1000)

`--profile mono-colour-500` is an opt-in sender profile; the existing
`mono-fold-500` default is unchanged. It uses the same 3,920-sample packet,
`MONO_PILOT_VALUES`, M-only 1,264 fresh slots per packet, 208 protected head
coefficients in 26 groups, remaining fresh slots in 132 groups, `scale*sqrt(2)`,
tail memory off, required EOF marker, 500 fold slots, 16 signature slots,
`U_CLIP = 2.5`, and `--mono-video-side` leg handling as `mono-fold-500`.

The profile-specific changes are:

| Item | `mono-fold-500` (`MONO_500`) | `mono-colour-500` (`MONO_1000`) |
|---|---|---|
| Coded status | `MONO_500` (4) | `MONO_1000` (5) |
| Wire profile | `mono-fresh-500` | `mono-colour-500` |
| Coefficient order | `model.order` | `colour_order(model)` below |
| Fold hosts | ranks 764–1,263 of `model.order`, any plane | 500 weakest luma coefficients among the 1,264 fresh slots |
| Fold guests | ranks 1,264–1,763 of `model.order`, any plane | first 500 luma coefficients after the fresh 1,264 |
| Fold step D | 1.1409647181002305 | 0.6 |

The coefficient order is constructed deterministically:

1. Preserve `model.order[:208]` as the protected head without reordering it.
2. Copy `model.lam` to a floating-point array and multiply entries whose
   `model.plane` is 1 or 2 by `CHROMA_RANK_WEIGHT = 4.0`.
3. Take `model.order[208:]` and sort it by weighted lambda, descending, using a
   stable sort.
4. Concatenate the protected head and sorted remainder to produce the 2,880
   coefficient order.
5. The first 1,264 entries are fresh. Hosts are the final 500 luma entries in
   that fresh prefix; guests are the first 500 luma entries after it.

For the pinned box model, the resulting allocations are:

| Set | Luma | Cb + Cr | Total |
|---|---:|---:|---:|
| Fresh | 1,054 | 210 | 1,264 |
| Hosts | 500 | 0 | 500 |
| Guests | 500 | 0 | 500 |

No host is in the protected head, and no guest is in the fresh set. As in the
base codec, the final 16 hosts carry the fold-table signature. The complete
colour fold-table SHA-256 is
`76e6f7b3592fcc2d657197d86ab189a320ca16b1f3bf73ed0fc3a5bed24989f6`.
The byte-identity baseline for three `mono-fold-500` packets on this machine is
`ba74e9ff3426cf8fa58f7a8fa5562adc9c7b3b1cabc389eba8213a6f1e659036`.

The default receiver uses three distinct matching status packets before
switching its active layout. `MONO_500` selects `MonoFreshFoldWire`;
`MONO_1000` selects `MonoColourFoldWire`. Packets in a candidate layout are
held until the third status confirms the switch, when that confirming packet
is decoded with the new rank map. The same rule applies when switching back to
stereo Fold-500 or between the mono layouts; no session-level profile latch is
used. Invalid or ambiguous status breaks the candidate streak and holds the
last picture. The selected input leg is tracked per packet, with right
preferred if both legs validate.

The synthetic comparison uses four motion-scene classes (slow pan, fast pan,
cut every six packets, and moving blob) and a saturated colour-chart scene. The
four-scene values are means of the 16 scored packets per scene. The chart values
are its 16-packet means, shown for each synthetic channel:

| Synthetic channel | Fresh four-scene mean | Colour four-scene mean | Δ | Fresh colour-chart mean | Colour colour-chart mean |
|---|---:|---:|---:|---:|---:|
| Clean 96 kHz | −29.233 | −24.410 | +4.823 | −58.645 | −53.750 |
| Type II | −35.020 | −36.649 | −1.630 | −62.053 | −58.918 |
| Fast flutter | −34.136 | −33.330 | +0.807 | −61.962 | −58.279 |

The 2026-09-28 rerun meets the clean four-scene +4.0 target: `mono-colour-500`
scored 4.823 points above `mono-fresh-500-fold`. It scored 1.630 points below
fresh-fold under Type II and 0.807 points above it under fast flutter. The
colour-chart score improved on all three channels, by 4.895, 3.135, and 3.683
points respectively. No fold-step or rank-weight tuning was performed. Mean
decode cost across all 15 scene/channel rows was 1.151 ms/input packet for
`mono-fresh-500-fold` and 1.113 ms/input packet for `mono-colour-500` (−3.32%);
mean encode cost was 1.360 and 1.372 ms/packet respectively.

The `stereo-fold500-both` four-scene control means were −4.697, −16.672, and
−13.032 for clean, Type II, and fast flutter. The `stereo-fold500-mono-sum`
means were −34.518, −36.992, and −35.214; both controls match the §13.8 table
within 0.01 points.

All 90 rows decoded 24 results, received and displayed all 24 current frames,
and validated 24/24 EOF markers. Across the five scenes and three channels,
both `mono-fresh-500-fold` and `mono-colour-500` had 360/360 current frames
displayable and no held-picture frames.

These scores are from synthetic channels, not tape. Type-II and fast-flutter
impairments are limited regression inputs and are not cassette emulation. No
real-tape performance claim is made. The complete row data is saved at
`tmp/v7-mono-colour/results.json`.

### Evaluation protocol: Mono 500, stereo Fold 500, and source filters

This protocol defines the next controlled comparison; it does not claim that
the existing benchmark runners implement every requirement below. Earlier
eight-image filter sweeps and generated-motion scores are exploratory. Do not
combine their absolute scores: source preparation, reference geometry, temporal
history, and ideal versus waveform reconstruction differ. No default changes
follow from those measurements alone.

#### 1. Freeze and validate the measurement path

Save a run manifest containing the Git revision and relevant local diff,
model/fold-table hashes, source-file hashes, software/metric versions, hardware,
commands, random seeds, and all sender/receiver settings. Store scripts,
manifests, per-frame results, contact sheets, and clips under a named repo-local
`tmp/` run directory. Temporary paths alone are not durable provenance: retain
the manifest and exact reproduction instructions with any published conclusion.

Before scoring candidates, establish these correctness gates:

- Load SBS sources through the production source loader. Verify that the colour
  image and alpha side are interpreted correctly, rather than treating the
  entire SBS file as an ordinary RGB photograph. Pin alpha/background handling,
  colour conversion, orientation, aspect handling, and image dimensions.
- Define one reference-rendering function and one decoded-picture rendering
  function. Record their dimensions and interpolation methods. Use the same
  unfiltered reference for every filter/profile pair. An additional comparison
  against prepared 80×96 source pixels may isolate transport loss, but must be
  labelled separately from end-to-end source fidelity.
- Check metric identity on identical reference pixels and deterministic repeat
  runs. Check that displayed and scored pixels are identical. Do not rescale or
  tune references separately for a candidate to improve its score.
- Match decoded pictures to explicit source-frame identities and presentation
  times. Verify that association using deliberately distinct consecutive
  frames, including a dropped packet and a cut. Do not silently substitute a
  decoder-result ordinal when source identity is missing.
- For ideal reconstruction, simulate the actual profile's transmitted rank
  map, fold/signature slots, receiver initialization, and history policy. Mono
  500 must receive only its fresh slots and recoverable folded guests; passing
  a full encoder coefficient vector directly into an unfold routine is not
  sufficient evidence of a profile-constrained ideal decode. Perturbing omitted
  coefficients at the receiver input must not change the reconstructed image.
- Compare ideal and clean-waveform reconstructions on the same static source.
  Record pixel/coefficient error, fold acceptance, profile status, and EOF
  validation. Explain discrepancies before using the ideal path for rankings;
  do not assume waveform and ideal scores must be identical.

An unexplained mapping, reference, or rank-mask failure invalidates the quality
comparison. Preserve its diagnostic output and fix the measurement path before
collecting a larger sweep.

#### 2. Fixed sources and paired experiment matrix

Freeze a held-out corpus before tuning: at least four independent still sources
in each of six categories—faces/natural photographs, fine texture/repeated
patterns, text/UI/line art, smooth gradients, saturated colour boundaries, and
dark detail/bright highlights. Adjacent frames from one clip do not count as
independent sources. Document selection rules and keep tuning sources separate.
Include both ordinary production material and labelled diagnostic patterns.

Freeze at least two sequences for each temporal class: static hold, slow pan,
fast motion, hard cuts, and gradual transition. Record frame cadence, duration,
cut times, and any synthetic motion-generation parameters. Sequences must span
multiple complete stereo tail cycles and include both startup and settled
operation. Use real motion clips where available; generated pans alone do not
cover temporal quality.

The primary matrix is `mono-fold-500` versus `fold-500` with both stereo legs,
each using the canonical frozen Box model and these sender settings:

| Pre-encode resize | Detail strengths |
|---|---|
| `off` with `--encode-filter box` | Not applicable; baseline |
| `linear-box` | Not applicable |
| `gamma-detail` | 0, 0.25, 0.5, 1 |
| `linear-detail` | 0.25, 0.5, 1 |

Check separately that `linear-detail` strength 0 equals `linear-box`. Feed
identical prepared capture frames to both profiles for each setting. Pin
brightness, gamma, capture scaling, frame cadence, sample rates, playback speed,
receiver history, equalization, and display reconstruction; list their actual
values. Use each profile's intended history policy rather than disabling stereo
history to make its layout resemble Mono. Reset receiver state between runs.

Mono-summed stereo and all-fresh mono fold-off are optional diagnostic controls,
not substitutes for the full-stereo baseline. Capture-scaler choices (direct
80×96, aspect-preserving 80-pixel width, 160-pixel width) belong in a separately
labelled paired experiment; do not change capture preparation within a filter
comparison.

#### 3. Three separate quality experiments

1. **Profile-constrained ideal reconstruction:** measure the layout/filter
   ceiling without waveform or channel errors, using the validated rank mask
   and state simulation. Label whether stereo has complete static history or
   a time-varying coefficient history. An ideal ceiling is not a live result.
2. **Clean waveform, static holds:** encode and decode each still repeatedly
   from reset. Report startup separately; score settled output only after at
   least one complete stereo tail refresh and valid profile acquisition.
   Verify all expected packets, statuses, fold decisions, and EOF markers.
3. **Clean waveform, motion:** use the frozen sequences at their specified
   cadence. Score against the current intended frame, not a previous frame
   chosen for a better match. Report startup, steady motion, and each post-cut
   frame through at least one full stereo tail cycle. Review stale detail,
   shimmer, colour instability, and recovery after cuts as well as frame scores.

Use a common primary clean capture rate of 96 kHz at 1× playback; label any
48 kHz compatibility run separately. Keep impairment checks to a small,
deterministic regression set, reported independently from clean fidelity.
Synthetic Type-II/flutter labels are not evidence of real tape performance.
Real tape assessment is a separate user measurement with its own acquisition
and alignment record.

For every expected presentation interval, record received/damaged/missing
status, current-frame displayability, profile/EOF validity, and whether the
display held its previous picture. Score held pictures against the current
intended source and report hold counts, durations, and longest consecutive
hold. Before the first displayable picture, report no-picture duration and
metric coverage rather than inventing a black frame or dropping the interval
silently. Distinguish conditional decoded-frame quality from displayed-sequence
quality so loss cannot improve a headline score by removing difficult frames.

#### 4. Quality reporting and blind review

SSIMULACRA2 is the primary structure metric: **higher is better**, including
positive scores; −40 is better than −60, and +20 is better than either. It is
not a percentage. Report supporting luminance PSNR (higher is better) and
ΔE2000 colour error (lower is better), with metric implementations and settings.

Every quality table must identify corpus, reference geometry, reconstruction
path, temporal window, and scored/expected frame counts. Use these columns:

| Setting | Stereo score | Mono score | Stereo filter gain | Mono filter gain | Stereo advantage |
|---|---|---|---|---|---|
| Each matched setting | Absolute | Absolute | Versus stereo Box | Versus Mono Box | Stereo minus Mono |

Positive filter gain means improvement within that profile. Positive stereo
advantage means stereo scored higher. For example, stereo −40 and Mono −55
give a **15-point stereo advantage**, not a 15% improvement. Do not present a
subtracted score as another image-quality score.

Retain per-source paired differences and report category means, medians, worst
cases, and an equal-category overall mean. Report sequence-level summaries
separately from stills. Use a fixed-seed paired bootstrap over independent
sources/sequences for 95% confidence intervals; adjacent video frames are not
independent samples. Small or overlapping differences need visual evidence,
not a claim of a decisive ranking from a single mean.

Produce randomly labelled, blinded comparisons at native 80×96 and intended
display size using the same display scaler. Include the reference and baseline,
record the label key separately, and retain reviewer preferences and reasons.
Review motion as clips at the actual cadence, not only contact sheets. Inspect
text readability, gradients, colour boundaries, clipping, texture, stale detail,
and temporal stability. Summarize category regressions alongside aggregate wins.

#### 5. Paired CPU, latency, and resource measurements

Measure quality and performance in separate runs: metric calculation, saved
images, and diagnostic scans must not enter codec timings. Warm compilation,
tables, and caches before steady-state measurement; report cold startup
separately. Use at least 30 paired timed batches, alternate baseline/candidate
order with a recorded schedule, and report batch sizes and warm-up counts.
Benchmark on the intended host as well as recording the development host.

Report process-CPU milliseconds per frame and wall-time median/p95 for source
preparation/filtering, encode, decode, and decode plus picture reconstruction.
Distinguish batch throughput from single-frame service time. Report peak memory
and CPU consumption at the actual frame rate, with 100% explicitly meaning one
fully occupied logical CPU.

For end-to-end capture tests, include FFmpeg child-process CPU, capture scaling,
queueing, and presentation. Measure source-to-display frame age using traceable
timestamps or a visible timed stimulus; record clock/alignment uncertainty.
Report latency median/p95, queue depth, dropped/repeated frames, missed frame
deadlines, and audio callback under/overruns over a fixed run duration. A Python
preprocessing saving alone is not an end-to-end CPU saving. Keep the sender GUI
event-driven: benchmark instrumentation must not add capture preview, per-frame
UI updates, or per-frame logging to the normal sender path.

Pure synthetic runs open no audio device. If live I/O is needed, explicitly
select and verify a virtual/loopback device; never assume a named default is a
loopback or send test signals to physical speakers.

#### 6. Decision and reproducibility gates

Before the final held-out run, record numeric target-host budgets for added
CPU, p95 frame age, memory, missed deadlines, and acceptable category quality
regressions. The budget values are an explicit project decision, not numbers to
choose after seeing which filter wins. Without agreed budgets, publish findings
as exploratory rather than a default recommendation.

A default candidate must pass the measurement correctness gates, show
repeatable paired quality gains across the corpus, pass blind still/motion
review without material text/colour/temporal regressions, fit those resource
budgets, and preserve clean decode/profile/EOF reliability. Publish failures
and exclusions with their reasons. Repeat only where changed code, failures,
or unresolved variability justify it. Archive the manifest, per-source results,
comparison assets, and exact commands with the decision. Until these gates are
met, retain `off --encode-filter box` as the low-cost default.

### Proposed extension: receiver audio WAV capture

This is a proposal for a later receiver feature, not a current wire or GUI
capability. The V7 receiver currently saves decoded pictures as image files but
does not record incoming audio to WAV. The legacy `main.py --modem-wav` option
saves generated modem waveforms and is a different workflow.

For confirmed `MONO_500` reception, an optional recorder could write the
non-video input leg to a user-selected WAV file. It should record the captured
samples at the input device's actual sample rate, independently of passthrough
mute and output-device selection. It must not infer an audio leg from pulse
activity or from packet loss: recording starts only when the coded profile and
channel route identify the non-video leg. Unknown/stereo profiles remain
unrouted. Recording I/O belongs on a bounded worker queue, not the PortAudio
callback; queue overflow and file/device failures must be visible, with no
silent fallback. The file format, behavior across route loss/reacquisition, and
GUI file-picker controls remain open implementation details.

Before exposing this extension, benchmark receive/decode CPU and callback
stability with recording off and on, and verify duration/sample continuity on a
synthetic input. The implementation and overhead results are deferred until
that check.
