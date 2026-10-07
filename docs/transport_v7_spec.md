# V7 Transport Specification

> **This spec is dirty. Do not rely on it.**
> Live testing has found places where the code and this document disagree,
> and places where both are wrong. Until each section has been checked
> against a live sender-to-receiver run, treat every statement here as
> unverified. The working list of known faults is `docs/V7_FIXES.md`.

This document describes the V7 audio modem wire as the code implements it
today. The code is the reference:

| Part | File |
|---|---|
| Packet, OFDM body, metadata, decoder | `animation_modem/v7.py` |
| Pulse preamble and edge-counted acquisition | `animation_modem/transport3.py` |
| Source coder, resampling, emission filter | `animation_modem/v7_core.py` |
| Live input buffer, leveler, header scan | `animation_modem/v7_live_input.py` |
| Standalone sender, receiver and GUI entry | `tools/v7_live.py` |
| Application sender (`main.py --mode modem`) | `modem_v7_display.py`, `animation_modem/v7_fold.py`, `animation_modem/v7_coded_pilot.py` |
| Wire profiles and receiver-side fold | `test_modem_v7/` (`live_fold.py`, `folding.py`, `aspect_fold.py`, `aspect_mono.py`, `mono_video.py`, `mono_wire.py`, `slice_wire.py`, `tone_code.py`) |

Earlier revisions of this file held proposals, measurement diaries and plans.
They were removed; git history keeps them. Sections 15 and 16 list known
limits and open questions; section 17 is the roadmap.

## 1. Scope and status

V7 sends one small picture per audio packet. A packet has an edge-counted
pulse preamble, an OFDM picture body, one CRC-protected metadata symbol, two
low timing tones and an end-of-packet (EOF) marker. Timing acquisition counts
preamble edges; it does not use FFT correlation.

`animation_modem/` does not import application settings, renderers or audio
devices. Two senders exist: the application sender (`main.py --mode modem`,
baked image library) and the standalone live sender (`tools/v7_live.py send`,
camera, screen, video or test source). There is one receiver:
`tools/v7_live.py receive` (and its GUI, `tools/v7_live.py gui`).

Every result behind this design comes from synthetic channels and in-memory
loopback. Real tape and deck playback is not validated.

When a packet is damaged the receiver shows what it decoded; when a packet
cannot be decoded it keeps the last good picture. There is no black-frame
fallback.

## 2. Packet format and acquisition

All sample counts are at the 48 kHz reference geometry. One packet is 3,920
samples (81.67 ms, 12.245 packets/s at 1×):

| Region | Samples | Content |
|---|---:|---|
| Preamble | 288 | 16 idle, 256-sample biphase-mark word, 16 idle |
| Body | 3,456 | 24 OFDM symbols × 144 samples |
| Metadata | 144 | One OFDM symbol, 40 bits |
| Guard | 32 | 8 samples, then the 24-sample EOF marker |
| **Total** | **3,920** | |

The preamble, metadata symbol, timing tones and EOF marker are written
identically to both channels.

### 2.1 Preamble word and profile ID

The preamble is a 16-bit biphase-mark word with 8-sample half-bits at
amplitude 0.55. There are six words (`transport3.PROFILE_PREAMBLE_BITS`), five
in use and one reserved. All
have the same first edge, last edge and edge count, so timing is the same for
every word; the pattern of short and long intervals carries a profile ID
(section 14). The words have minimum Hamming distance six. ID 1 is the
original V7 word.

`transport3.measure_pulses()` finds the forward word and returns position,
scale and confidence. `measure_pulses_both()` matches the forward and the
time-reversed interval patterns on one Schmitt-edge pass (hysteresis
0.12 × 0.55) and discards a candidate that matches both. The `_profile`
variants also return the profile ID; an ambiguous word still gives a timing
hit with no ID. Zero crossings are interpolated and one scale is fitted over
the word. Hits with confidence below 0.45 are skipped.

Scale is the packet's length relative to the reference. The accepted range is
0.25 to 100 (playback 4× down to 0.01×), multiplied by `capture_rate / 48000`
when the capture runs at another rate.

### 2.2 Packet endpoint

The decoder has two endpoint modes (`frame_boundary`):

- `eof`: the packet ends at its EOF marker, four runs of 4, 8, 6 and 6 samples
  at +0.55, −0.55, +0.55, −0.55. The header-to-marker distance gives the packet
  scale. A packet can be committed without a following header. A missing or
  damaged marker does not commit the packet, except through the splice rules
  in section 6.3.
- `baseline`: the packet is committed when the next header is found at the
  expected spacing and scale. The last packet of a finite stream is not
  committed.

Both senders always emit the marker, and the standalone receiver defaults to
`eof`. The low-level functions `encode_pulse_frame()`, `encode_pulse_stream()`
and `decode_pulse_stream()` default to no marker, no tones and `baseline`.
Bench tools that should model the sent wire use `tools/v7_wire_profile.py`.

Neither sender has an option to omit the marker: the application sender
rejects `--no-modem-eof-marker`. The standalone receiver has
`--frame-boundary baseline|eof`.

### 2.3 Reverse playback

`tools/v7_live.py receive --direction auto|forward|reverse` (default `auto`)
selects which preamble orientations are searched. A reversed preamble arrives
at the end of its packet. The receiver waits until the whole predicted packet
is in its buffer, takes that region plus a 64-sample margin on each side,
reverses it and decodes it with the ordinary forward decoder
(`v7.decode_reverse_packet`). A reversed packet is shown only with a validated
EOF marker and metadata that passes on its own; provisional metadata (section
5) is not enough.

Playback direction is separate from the source loop's direction (section 5).
The displayed playback direction changes after two distinct validated
arrivals in the new direction (`v7_live_input.DirectionStreak`). A direction
change resets the tail memory and the last verified metadata and keeps the
learned loop constants.

Turn-arounds on a packet boundary:

- Reverse to forward: the last reversed preamble and the first forward one
  are exactly 288 samples apart. Opposite-direction words are treated as one
  ambiguous read only when closer than one word length (256 samples), so both
  are kept. Both usually arrive in the same wake, and the receiver decodes
  the reversed packet (`v7_live_input.select_packet_hit`).
- Forward to reverse: the last forward packet is not decoded. The receiver
  next wakes on the reversed packet's trailing preamble and shows that packet.

### 2.4 Emitted levels

The header and the end marker are pulses and serve as the packet's level
reference, so they are the loudest part of every packet and the picture body
stays below them:

- The band-limited header's peak is 3 dB below full scale, the same in
  every packet (`HEADER_PEAK_DB`). The end marker's pulses are at the
  header's pulse level (`emitted_pulse_level`). The shaped header peak is
  therefore about 1.5 dB above the EOF pulse level.
- For EOF-marked packets, the image body is scaled after the body-derived
  timing tones are included. Its final peak is kept at least 1.5 dB below the
  lower final peak of the header and EOF regions. The metadata symbol is
  scaled separately and kept at least 0.5 dB below the final body peak. Quiet
   components are not raised to meet these ceilings by default. Experimental
   nested senders opt into thread-local `body_auto_level()`, which can raise
   a quiet body by up to 12 dB before the final EOF-aware level fit. The final
   framing/body/metadata margins still apply to those packets.
- The timing tones are added at a level that follows the body. They are
  included in the final level measurements, since they can add constructively
  to the body, header, metadata or EOF. With them the header's peak is about
  2.5 dB below full scale. A header nearer full scale costs lossy codecs
  frames (MP3 at 192 kbit/s and below).
- The low-level no-EOF encoder path retains the historical body ceiling
  relative to the header peak (`BODY_BELOW_HEADER_DB`).

The preamble's edge detector uses a Schmitt band of ±0.2 of the nominal
header amplitude (`EDGE_HYSTERESIS`).

## 3. OFDM body and stereo mapping

| Parameter | Value |
|---|---:|
| Reference sample rate | 48,000 Hz |
| FFT size / cyclic prefix | 128 / 16 samples |
| Symbol | 144 samples (3 ms) |
| Symbols per body | 24 |
| Bin spacing | 375 Hz |
| Data and pilot bins | 4 to 34 (1.5 to 12.75 kHz) |
| Timing tone bins | 1 and 3 (375 and 1,125 Hz) |

Pilots:

- Continual pilots on bins 4 and 34 in every symbol.
- Scattered pilots on bins 5, 9, 13, 17, 21, 25, 29 and 33. Each is present
  on every third symbol, at phase `((bin − 5) / 4) mod 3`.
- Pilot amplitude is 1.5·√2. The mid (M) pilot is real; the side (S) pilot is
  imaginary with alternating sign.

A data block is one (bin, phase) pair on bins 5 to 33: the eight symbols
`phase + 3k`. Removing the scattered pilots' own phases leaves 79 blocks. The
13 blocks on bins 5 to 9 carry M only. The other 66 carry M and S. Each
stream of a block carries an I group and a Q group of eight real values, so
there are 26 + 264 = 290 groups and 2,320 value slots per packet: 1,264 on M
and 1,056 on S. The eight values of a group are spread over the block's eight
symbols with a normalized 8-point Hadamard transform.

Each cell is mapped to the channels as

```text
L = (M + S) / sqrt(2)
R = (M - S) / sqrt(2)
```

and multiplied by a fixed per-cell phase table from the model file.

The preamble and body are filtered with a zero-phase 63-tap Kaiser (β = 8.6)
low-pass at 14 kHz (`v7_core.bound_emission`), and the packet peak is
restored afterwards. The metadata symbol, the timing tones and the EOF marker
are added after that filter.

For an output device above 48 kHz the sender resamples each packet so the
carrier frequencies and the packet duration are unchanged
(`v7_core.band_limited`). Device rates at or below 48 kHz are used as they
are, which scales frequencies and duration with the rate. At a playback speed
other than 1× the packet is resampled to `1/speed` of its duration
(`v7_core.speed_resample`), which scales every frequency by `speed`.

## 4. Image coding, ranks and model tables

The source is prepared as an 80 × 96 (width × height) RGB picture and
converted with Pillow's YCbCr conversion. Values are `code/127.5 − 1`. Y is
sampled on 96 rows × 80 columns; Cb and Cr on 48 × 40 (box-reduced from the
luma grid). A DCT-II (orthonormal) is taken per plane. The base wire sends
the low-frequency corner of each plane: 48 × 40 for Y and 24 × 20 for each
chroma plane, 2,880 coefficients. Coefficients that are not sent are
reconstructed as zero, or as the model mean for sent positions that were not
received.

A model holds, per coefficient, a mean, a variance, a rank, a transmit gain
and the level scale. The canonical models are frozen in
`animation_modem/v7_model_tables.npz`, SHA-256
`2392943a287fdf1cd2ac773c2fc065179006bfab4646726e72e023f3921ffb1c`, checked at
load. They were derived once from `modem_tests/fixtures/v7_reference_face.png`
(`tools/v7_freeze_tables.py`), not from any image library. The file holds
four models named by resize filter: `nearest`, `box`, `lanczos`, `bicubic`.
The wire can name only `nearest` and `box` (section 5); `metadata_word()`
rejects the other two. A custom model can be derived from another image, but
then both ends must hold the same one.

The transmitted level comes from the model: `unit_rms` is the RMS of a packet
encoded from coefficients drawn from the model's own Gaussian statistics, and
the packet is scaled to a fixed target from that number. There is no
per-picture level control on the stereo wire (section 15).

Coefficients are ordered by model variance and placed in three tiers:

| Tier | Ranks | Slots per packet | Carried on |
|---|---:|---:|---|
| Head | 0 to 207 | 208 | The 26 M-only groups |
| Body | 208 to 2,223 | 2,016 | The stereo blocks' groups, all M groups first, then all S groups |
| Tail | 2,224 to 2,879 | 96 | The last 12 groups (all on S) |

The tail has 656 coefficients and is sent 96 per packet over seven slices.
Slice `counter mod 7` is sent; the seventh slice holds the last 80 and 16
empty slots. The receiver's tail memory (`TailStore`) keeps a received tail
value for at most six further packets and then falls back to the model mean.
`receive --no-tail-memory` disables it.

Within a tier, ranks are interleaved across up to eight groups at a time
(`v7.windows`), so one lost block removes every eighth rank of a 64-rank span
instead of eight neighbours.

## 5. Metadata and loop fields

The metadata symbol uses the same FFT geometry as a body symbol, M only. Bins
4, 7, 10, …, 34 are 11 known pilots. The other 20 bins carry 40 bits as QPSK:
a five-byte word.

| Bits | Meaning |
|---|---|
| Byte 0, bits 7–5 | Aspect code (table below) |
| Byte 0, bit 4 | Screen bit. 0: the aspect code is the picture's own. 1: it is the sender's fixed aspect layout and the picture is pillar- or letterboxed inside it |
| Byte 0, bit 3 | Model: 0 `nearest`, 1 `box` |
| Byte 0, bits 2–0 | Tail slice, 0 to 6; 7 is invalid |
| Bytes 1–2, bit 15 | Source loop direction: 0 counting up, 1 counting down |
| Bytes 1–2, bits 14–0 | Source index plus one; 0 is invalid |
| Bytes 3–4 | CRC-16 of bytes 0–2, XOR a mask on slices 5 and 6 |

Aspect codes 0 to 7 are 1:1, 4:3, 3:2, 16:9, 1:1, 3:4, 2:3, 9:16. The sender
picks the nearest family and sets the portrait bit when the picture is
taller than wide (`v7.aspect_wire_code`). The receiver letterboxes or
pillarboxes the 80 × 96 sampling grid to that ratio.

The CRC is CRC-16/CCITT-FALSE: polynomial `0x1021`, initial value `0xFFFF`, no
final XOR. Source indices are zero-based in the APIs and at most 32,766. The
standalone sender's index is the packet count modulo 32,767 (it wraps after
about 44.6 minutes at 1×).

The receiver decodes the metadata symbol three ways, from the channel sum and
from each channel alone, and chooses among the candidates
(`PulseState.accept_candidates`): a CRC match first, then the index closest to
the previous verified packet's. Metadata that fails does not change the
model, aspect or tail slice in use.

### Loop fields

Slices 0 to 4 carry a plain CRC. When the sender supplies loop information,
the CRC on slice 5 is XORed with the loop phase `p` and on slice 6 with the
loop length `N` (15 bits; the top bit of that field marks a one-way loop
instead of ping-pong). `p = 0xFFFF` means the index does not follow a clock.

`LoopLock` learns each value after it arrives twice in a row on its slice and
then checks those slices against it. Until a value is learned, a slice 5 or 6
packet is accepted provisionally when its model, aspect and screen bit equal
the last verified packet's and its index is within 64 of it.

With a correct clock, `ticks = unix_ns × 30 // 10^9` and the learned `N` and
`p` give the index the loop shows now and how late the displayed picture is
(`loop_index`, `loop_lag_ticks`).

## 6. Timing tones, coded status and playback speed

### 6.1 Timing tones

Two tones on FFT bins 1 and 3 run through the whole packet, equal on both
channels. Bin 2 is empty. Each tone's RMS is 20 dB below the body's RMS. Their
phase origin advances by one packet length per packet counter, so the tones
are continuous across packets. They are separate from the OFDM pilots.

The decoder's `pilot_timing` modes are `baseline` (tones ignored),
`tone-seeded`, `tone-joint` and `tone-replaced`. The standalone receiver
defaults to `tone-seeded`. When metadata fails and a tone mode is active, the
decoder retries the metadata symbol once at the offset the tones measure.
Tone-assisted gain equalization (`--tone-equalization m-reference`) and
within-packet pulse-warp timing (`--pulse-timing pulse-warp`) are off by
default.

### 6.2 Coded status

Every profile keys the bin-3 tone with a 12-chip status
(`test_modem_v7/tone_code.py`; the application sender's copy for Fold 500 is
`animation_modem/v7_coded_pilot.py`):

- Bin 1 stays steady.
- Across the 24 body symbols, bin 3 is multiplied by ±1 per symbol. Even
  symbols carry the fixed code `+ − − − − − + + − + + +`. Odd symbols carry
  the 12 status chips (bit 0 is +1, bit 1 is −1).
- A chip change is an 8-sample raised-cosine ramp inside the cyclic prefix.
  Outside the body bin 3 is steady.
- The six status words are a four-chip pattern repeated three times: `0011`,
  `0101`, `0110`, `1100`, `1010`, `1001` for codes 0 to 5 (code 2 is
  reserved, section 14.1). Their pairwise
  distance is six. A word is accepted when `2 × errors + erasures < 6`.

The receiver decodes the chips, removes them and then uses the tones for
timing. The status code equals the preamble's profile ID. The frame decoder
requires the tone status to match the profile it was dispatched for; otherwise
the packet is held (section 14.1).

### 6.3 Playback speed and splices

The receiver measures each packet's scale from its preamble and resamples the
body onto the reference grid, so tape speed, varispeed and clock offsets are
followed per packet. Both senders accept a speed of 0.25 to 4. An output at
`rate` Hz keeps the full band up to speed `rate / 28000`; faster speeds lose
the top of the band. Receive capture uses the input device's default rate,
capped at 96 kHz.

Digital time-stretchers and pitch shifters of the WSOLA family keep the local
waveform scale and cut or repeat chunks. The preamble then measures pitch
while the packet is shorter or longer than that predicts. In `eof` mode the
decoder handles this as follows.

- **Endpoint.** If the EOF marker is not where the header predicts, the next
  header is searched from 40% to 205% of the predicted packet length and the
  marker is looked for just before it. If the marker itself is gone, the next
  header is the endpoint. An endpoint shorter than 36% of a packet is not
  accepted. When the body fits a marker found at the predicted place badly
  (mean cyclic-prefix correlation below 0.95), or the metadata fails there,
  the next-header endpoint is tried and kept if it fits better.
- **Per-symbol offsets.** The body is read at the header scale. Each symbol's
  cyclic-prefix correlation is evaluated at offsets between 0 and the measured
  jump in 2-sample steps, and a dynamic program picks a monotone staircase
  with a cost of 2.5 correlation units per step. The staircase is not used
  for jumps under 24 samples or when it stops more than 25% of a packet short
  of the endpoint.
- **Pilot identity.** Intact symbols at either end of the path give a flat
  channel from their pilots, and each chosen symbol's pilots must fit it
  (residual at most 0.5); otherwise other correlation peaks are tried.
- **Erasures.** A symbol whose correlation is below 0.85, or whose pilots
  still do not fit, is erased: it has no weight in the channel and timing fits
  and both channels are erased in the equalizer. The head gates (section 7)
  scale with the share of symbols present.
- **Metadata.** If the EOF-anchored metadata fails, the header-anchored one is
  tried. If a spliced packet's metadata is still lost and a neighbouring
  packet's metadata is known (the previous packet of a recording, or the
  previous packet in the live window, at most three packets back), the packet
  takes that model and aspect with the slice and index advanced. It is marked
  `metadata_predicted` and provisional.
- **Resync.** In a recording decode, a packet with no endpoint is held as
  lost and decoding continues at the next header. On the live path, a packet
  whose header was cut can be recovered from the previous packet's EOF
  marker.

Phase-vocoder processing rebuilds every phase and is not supported.

## 7. Decoder behaviour and live input

For each packet the decoder finds the header, finds the endpoint, resamples
the body and metadata, decodes the metadata, then demodulates the 24 symbols
with 128-point real FFTs. It fits a two-channel pilot response and a
per-symbol fade and noise estimate, equalizes each cell with a 2 × 2 MMSE
solve, solves each group of eight by LMMSE against the model's variances and
returns an estimate and a confidence (0 to 1) per coefficient.

If no packet is found in a two-channel capture, the decoder retries once with
the right channel inverted (one channel wired with reversed polarity cancels
the M-only preamble in the sum). A one-channel input is decoded as an M
observation.

Gates (`v7.py`):

| Gate | Condition |
|---|---|
| Decoded | Head confidence ≥ 0.70 and head coverage ≥ 0.50 |
| Displayable when not decoded | Head confidence ≥ 0.60 and coverage ≥ 0.50; pilot noise ≤ 2.0 |
| Accepted (live) | Metadata accepted, head confidence ≥ 0.85, coverage ≥ 0.75, pilot noise ≤ 2.0 |

Head confidence is the mean confidence of the 208 head coefficients. Coverage
is the share of them with confidence at least 0.15. Pilot noise is the larger
of the two channels' mean pilot residuals, with locally erased symbols
excluded. It is relative: the residual's power over the power a unit cell
is received at on the pilot bins, so it does not move with the input level;
2.0 is noise as strong as the picture signal the model expects. A symbol is
locally erased when its residual passes 0.08 on that scale and stands well
above its channel's median or the other channel. A packet that is not accepted
but is displayable is shown and labelled degraded. A packet that is neither
leaves the previous picture on screen.

The live input (`v7_live_input.LiveInput`):

- inverts the right channel when the two channels correlate below −0.3 over
  the newest packet length, with hysteresis;
- levels the input before the header search: once per packet it moves a gain
  towards putting the 99.5th percentile of the newest packet at 0.55, limited
  to 0.5× to 32×, rising by at most 1.5× per packet and falling at once. The
  loudest samples of a packet are its header's pulses (section 2.4), so the
  gain is steady whatever the picture. The decoder uses the same gain;
- scans only audio it has not scanned and keeps one packet of history plus
  one packet and a quarter-packet guard;
- hands audio to the decoder only when a new header has arrived. In `eof`
  mode the decoder then takes the newest packet whose marker validates, which
  is normally the packet before the new header.

`receive --temporal-fusion held` (off by default; GUI: Held-picture
averaging) averages successive accepted packets of the fold profiles, stereo
and mono,
before the fold is undone (`animation_modem/v7_temporal.py`). It is a
per-coefficient Kalman filter on the equalizer output: an observation is the
unshrunk estimate with the noise variance its confidence implies; the
packet-to-packet change beyond that noise, measured in eight groups along the
model's rank, is added to the state variance before each update. A held
picture is therefore averaged (at most 24 packets) and a moving one is shown
as decoded. The claimed noise is calibrated on the second difference of three
packets and only ever lowered. A packet that is not accepted, an input gap, or
a change of model, layout or direction starts the average again. Coefficients
a packet does not carry stay with the tail memory.

Because a wake needs a new header, a finite input that ends right after an
EOF marker leaves its last forward packet undecoded. An input gap clears the
buffer, the tail memory and the last verified metadata; the picture on screen
stays.

## 8. Runtime and defaults

Standalone sender (`tools/v7_live.py send`): profile `fold-500`, `box` model,
brightness 1.0, gamma 1.0, speed 1×, coded status, EOF marker. `--profile`
selects another profile (section 14). The hidden `--experimental-fold 0`
sends the unfolded wire: `nearest` model, brightness 1.05, steady tones. The
sender starts output after one packet is buffered and queues at most one
encoded batch. It uses the output device's native rate unless `--rate` is
given.

A local video file can be paused, restarted and seeked from the sender GUI
(commands on the same control pipe as live brightness and gamma). Pause does
not stop transmission: the sender goes on emitting packets of the held picture
at the normal rate, so the receiver keeps its lock, and the file's soundtrack
is silent until play. The index in the packets is the packet count, so it
keeps counting through pause, seek and restart. The GUI saves the position of
each file and Start resumes there (`--video-start SECONDS`); a changed or
missing file, or a position past the end, starts from the beginning, and the
file still repeats from the beginning when it ends. The position is counted
from the frames read: the reader's start plus frames delivered divided by the
file's frame rate (the readers pass every source frame through once), reset
on seek, restart and repeat; wall time is used only for a file whose frame
rate cannot be probed. The GUI has one preview at a time: off, the picture in
its window, or the same picture in a pop-out window. The pop-out is a small
separate process the GUI forwards the sender's preview pictures to, so it
follows pause and seek; sending does not depend on it.

Standalone receiver (`tools/v7_live.py receive`): automatic profile dispatch
(section 14.1), `eof` endpoints, `tone-seeded` timing, forward and reverse
detection, one packet per decode, one packet of history, tail memory on.
Automatic dispatch requires `eof` and a tone timing mode. The hidden
`--experimental-fold M` (0 or 500), `--experimental-mono-fold` and
`--experimental-mono-colour` fix one profile instead.

Both need an explicit `--device`.

Application sender (`main.py --mode modem`): stereo Fold 500 with the `box`
model, coded status and EOF marker, speed 1×. It has no other profile:
`--no-modem-pilot-tones` and `--no-modem-eof-marker` are rejected.
`--modem-encode-filter` offers `nearest` and `box`, the two models the wire
can name; Fold 500 accepts only `box`.

Numba is a hard dependency of the transport. The receiver compiles the pulse,
polarity, equalizer and status kernels before it opens its audio stream.
`--force-float32` selects an alternative arithmetic path, not a Numba-free
one.

### Audio device recovery

Sender and receiver remember the selected device's name and host API. When
the stream stops or its rate changes, they poll every 1/30 s and reopen only
after the same device and rate have been seen five times in a row. A missing
or ambiguous device is never replaced by another one. The sender rebuilds its
rate-dependent packet conversion at the new rate. The receiver starts
acquisition again and keeps its picture until a new packet decodes.

## 9. Verification

Run from the repository root:

```bash
.venv/bin/python -m unittest discover -s modem_tests -v
.venv/bin/python -m unittest tests.test_modem_integration tests.test_lazy_imports -v
.venv/bin/python -m unittest modem_tests.test_v7_live_input -v   # one focused suite
```

Do not blanket-discover `tests/`; see `AGENTS.md`. Never send test output to
physical speakers.

`modem_tests/fixtures/v7_pixel_motion_{4x3,16x9,3x4,5x6}.mp4`, built by
`tools/generate_v7_pixel_motion_test.py`, are a reference picture for the sender at each picture shape: a zone plate, line pairs, checkerboards, moving edges, colour patches, ramps, stepped text and a binary picture-index strip, all pitched in reduced samples, to show and measure lost resolution, aliasing, flicker, ringing, colour error and skipped pictures.

### Required tests

A change to the wire, a wire profile, the sender's levels or the receiver
must keep these passing, and a new profile must be added to each:

| Requirement | Test |
|---|---|
| Reverse torture: every live profile, played backwards through the channel models, shows the same picture as forwards, in descending order | `modem_tests/test_v7_reverse_torture.py` |
| Speed: every live profile at 0.5×, 1.5× and 2×, forwards and backwards, shows the same picture as at 1× | `modem_tests/test_v7_reverse_torture.py` |
| Speed acquisition from 0.25× to 4× on the base wire | `modem_tests/test_v7_speed.py` |
| Reverse acquisition, turn-arounds, slow reverse, tape rocking | `modem_tests/test_v7_reverse.py` |
| Level independence: the same recording at a quarter and at four times the level decodes to the same statuses | `modem_tests/test_v7_level_independence.py` |
| Emitted levels: header/EOF/body/metadata hierarchy after tone mixing, with no full-scale overflow | `modem_tests/test_v7_levels.py` |

The profiles under reverse and speed test are Fold 500, Aspect Fold 500,
aspect-mono-500 and one channel of stereo-slices; the channel models are
clean, hiss, wow and flutter, the Type I model and dropouts.

### Comparing designs

A comparison of two designs runs every situation of
`tools/v7_torture_matrix.py` plus one channel alone, channel dropouts and MP3,
reports every situation, and treats a situation that gets worse as a
regression. Report luma error, colour error (CIEDE2000), flicker on stills
and motion error on a moving source, as change against the shipped profile.
Noisy situations are run with several seeds and a difference counts only when
it clears their spread. SSIMULACRA2 is for gross ranking only.

Hiss situations are named by noise level in dB below full scale, so `hiss-35`
is louder noise than `hiss-45`.

A difference counts only when the measurement can resolve it. One run on a
few pictures cannot resolve a change of a few percent: score per picture and
report each change with a range from resampling the pictures (and the noise
seeds); a change whose range includes zero is "no measurable difference".

### Sender and receiver stay in step

The sender and the receiver are one version. A change to what the sender
encodes (levels, tables, coefficient selection, fold, signalling) comes with
the matching receiver change, or with a test showing the unchanged receiver
decodes it, in the same patch. Prior recordings need not stay decodable.

### Stability rule for a new combination

A new combination of settings, tables or profile is accepted only if its
stability is within 5% of the current one, or better, in every situation.
Stability is counted, not scored: the packets received as valid and the
pictures shown, out of those sent, in each situation, and no picture skipped
between the first two and the last on a clean link.

All evidence for this wire is synthetic: unit tests, in-memory loopback and
simulated impairments (`tools/v7_torture_matrix.py`,
`tools/v7_timing_bench.py`, `test_modem_v7/`). None of it is tape validation.
Real tape and deck playback is unvalidated. The `type-i` and `type-ii`
situations are not faithful tape models: their saturation is the same at all
frequencies and is driven far harder than a recording level anyone would use.

## 10. The fold

The fold puts two luma coefficients in one slot. It is the default on both
senders (Fold 500). The transport is unchanged: a folded packet is a normal
V7 encode of modified coefficients. The codec is `test_modem_v7/folding.py`;
the application sender's encoder is `animation_modem/v7_fold.py`.

### 10.1 Mapping

For fold size M, the *hosts* are the M lowest-ranked luma coefficients of the
body tier. The *guests* are M luma coefficients the base wire does not send.
With host mean `mu`, host and guest standard deviations `sd_h` and `sd_g` and
step `D`:

```text
h = (host - mu) / sd_h
u = guest / sd_g

linear guests:     s = D*round(h/D) + beta * clip(u, -2.5, 2.5)     beta = 0.8*D/5
companded guests:  s = D*round(h/D) + A * c(u)                      A = 0.4*D
                   c(u) = sign(u) * ln(1 + mu_c*min(|u|, U)/U) / ln(1 + mu_c)

sent coefficient = mu + sd_h * s / sqrt(P)
P = 1 + D^2/12 + beta^2             (linear)
P = 1 + D^2/12 + 0.12 * A^2         (companded)
```

The host is quantized to a multiple of the step and the guest rides inside
the step. Dividing by √P keeps the slot's power at the host's model variance.

### 10.2 Tables

Both ends must hold the same hosts, guests, scales and step. They are frozen
in a JSON table whose SHA-256 is pinned in the code; a table with another
hash, or one built for another model, is refused.

| Table | Guests | D | Other |
|---|---|---:|---|
| `fold_table_500.json` (Fold 500; identical copy `animation_modem/v7_fold_table_500.json`) | companded, U = 12, μ = 4 | 1.0 | guest noise limit 0.15 |

It was fitted on the reference fixture for the canonical `box` model, with
host confidence limit 0.9, slot noise limit 0.3 and 16 signature slots. The
other profiles build their fold from frozen layout tables with constants in
their modules (section 14).

### 10.3 Signature slots

The last 16 hosts carry no picture. They carry `±3·D` in a fixed sign pattern
drawn from a generator seeded with the table's identity hash. Their guests are
not sent. A fold of M therefore carries M − 16 picture guests and gives up 16
host values.

The receiver computes each host's symbol from the equalizer output divided by
its confidence, and uses the signature for two things:

- **Identity.** The score is the mean of (signature symbols × pattern) / 3D:
  about 1 on a packet folded with this table, about 0 otherwise. At 0.5 or
  more the table is taken as present. `signature_error` is the RMS distance
  from the pattern in units of 3D.
- **Noise.** The RMS distance of the signature symbols from the pattern is the
  symbol noise on the folded slots. Above 0.3·D every host is read as an
  unquantized value and all guests are dropped. With companded guests, above
  the table's guest noise limit the guests are dropped and the hosts are still
  read as steps. Otherwise each guest is scaled by its MMSE weight at that
  noise and expanded.

Independently, a host whose equalizer confidence is below 0.9 is read as an
unquantized value and its guest is dropped. Dropped guests are zero.

A packet whose coded status selected the table is unfolded even when the
signature score is below 0.5. A packet with neither is shown without
unfolding.

The receiver obtains the equalizer output by wrapping `v7._equalize_numba`,
`v7._equalize_numpy` and `v7.decode_frame` while a profile is installed.

## 11. Sender picture preparation

### 11.1 Goal and contracts

Prepare source pictures for the frequencies the wire keeps: recognizable
small features, acceptable colour and less ringing, in preference to nominal
sharpness. The contracts for any preparation stage:

- Classical and deterministic. No learned models, random jitter or grain.
- Sender only. No change to packet length, metadata, model tables, fold
  tables, synchronization or the receiver.
- The canonical `box` model and its pinned fold table. Other combinations are
  rejected.
- With every option off the sender's output is unchanged.
- The live hook is `tools/v7_live.py::_values`, before brightness and gamma.
- Shared code lives in `animation_modem/` and imports no application module.

### 11.2 Built (all off by default, standalone sender only)

- `--perceptual-resize linear-box | gamma-detail | linear-detail` with
  `--perceptual-detail-strength` (0 to 1, default 0.25): area or
  detail-weighted area reduction to the prepared picture, in linear or
  gamma-encoded RGB (`animation_modem/perceptual_resize.py`). Not available
  on the unfolded wire.
- `--dct-encode`: computes the wire's coefficients from the unprepared source
  frame instead of from the 80 × 96 resize (`animation_modem/v7_source_dct.py`).
  Not available on the unfolded wire. Mutually exclusive with
  `--perceptual-resize`. A frame smaller than the coder grid falls back to a
  box resize.
- With `--dct-encode`: `--luma-adjust` (re-fits luma to the chroma the
  receiver will have), `--luma-adjust-linear`, `--pixel-encode` with
  `--pixel-grid` and `--pixel-detail` (section 14.4), and hidden sharpen,
  clarity, chroma-gain, aggregation and band-profile options.
- `--clip-aware-encode` (folded profiles): re-fits sent luma so ringing falls
  into the receiver's black and white clip.
- With `--dct-encode`: `--dct-kernel NAME` (with `--dct-kernel-param
  NAME=VALUE`, `--dct-kernel-dir DIR`), a pluggable DCT kernel read from
  `dct_kernels/` at launch (section 11.4).
- The default resize path uses Pillow's integer `reduce` for a box resize when
  the source is a whole multiple of 80 × 96.

### 11.3 Plans (not built)

The plans the owner kept. Plans struck on review: deciding whether the
`box` tables are adequate (answered: they are not; see roadmap item 1),
guest soft knee, luma-guided chroma, change-adaptive softening, bake
integration.

1. Promotion rule: no stage becomes a default without a measured comparison
   against `box`, a viewer preference, and at most 25% added send-path time.
2. Other coefficient budgets: profiles that carry a different selection of
   coefficients, each with its own profile and tables. The aspect layouts,
   pixel grids and stereo-slices (section 14) are built instances and are no
   longer part of this plan. Still open under it:
   - band shaping, a fixed luma gain array applied to the full DCT before
     folding, with DC at gain 1. For mean-square error the gains are already
     optimal: a coefficient of variance λ sent as an analogue value over a
     noisy channel under a power limit wants gain proportional to λ^(−1/4),
     which is what the model tables use, and the coefficients too weak to
     be worth their power are the ones left out (reverse water-filling).
     Band shaping is therefore only a question of perceptual weights: with a
     weight w per coefficient the gain becomes proportional to (w/λ)^(1/4).
     Candidate weights are a contrast-sensitivity curve or a JPEG-style
     table; which looks best is an A/B question, not a formula.
3. Block-averaged Fold 500 projection, kept as an opt-in research
   approximation (`v7_source_dct.py::FoldBlockDCTProjector`). The 8×8 box
   average gave no visible benefit; other kinds of averaging and other
   kernels are to be tried.

### 11.4 DCT downscale kernels

The direct encode brings the source down in three stages: a block mean and a
fixed 2:1 decimation (aliasing 54 dB or more below the picture), then a
truncation of the DCT to the coder grid. The truncation is a brick wall at the
edge of the sent band, which is why a hard edge overshoots by about 9% and
flat areas next to it show a faint mesh. A kernel decides what is handed to
that last step; it changes nothing on the wire, in the fold tables or in the
receiver.

One kernel is one Python file in `dct_kernels/` (more folders: `--dct-kernel-dir`,
`$V7_KERNEL_DIR`). Files are read when the sender or the GUI starts, and again
on `R` in the GUI (which also tells a running sender). A file that fails to
import or to run on a test picture is reported and skipped; a kernel that
fails on a live frame is bypassed for that frame. The contract is the docstring
of `animation_modem/v7_kernels.py`; `dct_kernels/README.md` has the short form.

A kernel supplies any of:

- a linear window over the DCT, written as a 1-D resampling `kernel(x)` in
  sent-pixel units (the host derives its frequency response), a
  `response(nu)`, or a full 2-D `gain(ctx)`. Its gain at DC is forced to 1.
  With luma adjustment on, the same window is applied to the luminance the
  adjustment aims at, in the coded (gamma) domain, so the window and the
  adjustment agree; filtering that goal in linear light instead made a gentle
  window harsh on dark edges (Lanczos overshoot 15% against 7%);
- a non-linear `post(grid, ctx)`, run on the final values after luma
  adjustment, with `ctx.project` (keep only the coefficients the wire carries)
  and `ctx.reduce` (local min/max of the source);
- a `prefilter()` that chooses the pre-shrink factor (1.75 to 8 times the
  luma grid; 4 is the shipped encoder) of the block-average stage before the
  transform. It can instead return `{'full_source': True}` to project the
  original-resolution frame directly onto the coder-grid DCT coefficients,
  skipping both the block-average and 2:1 decimation stages. The fixed coder
  grid, transmitted band and wire format are unchanged. The ordinary 2:1
  decimation filter was measured within 0.1 dB across tested designs on
  natural pictures; block size matters more: on dense texture factor 8 is 8 dB
  closer to the full-resolution transform in the top half of the band than 4,
  while on smooth pictures nothing changes. Native-source color conversion,
  partial DCT projection and coder-grid inverse transforms run in Numba rather
  than materializing a full-frame NumPy DCT.

`tools/v7_kernel_bench.py --display bilinear|bicubic|nearest` judges a
kernel through the viewer's upscaler instead of an ideal enlargement.

`luma_mix` and `chroma_mix` are added to every kernel (0 off, 1 as written).
In the GUI the kernel, its parameters, DCT sharpen strength, clarity and
chroma gain are live: Left/Right steps a value (Shift: five times) or
switches kernel, and the next frame uses it.

`tools/v7_kernel_bench.py` ranks kernels on synthetic pictures with no audio
in the loop (edge overshoot and width, ripple, retained amplitude of fine
gratings, time per frame; `--guests` includes the fold's guest coefficients,
`--sheet` and `--image` write a contact sheet). Measured on the reference
encode and the shipped kernels with luma adjustment on and guests present:
the reference overshoots 6.4% with a 0.88-pixel edge; Anti-ringing refit
2.0% and 1.00 pixel with the same retained detail (a window cannot do this:
every linear window that removes the overshoot also lowers the amplitude of
fine detail); the Lanczos, Mitchell and Gaussian windows trade detail for
less ringing. The refit costs about 4 ms per 1080p frame, windows under 1 ms
(a kernel that takes more than 25 ms per frame is reported in the GUI).
Scored with SSIMULACRA2 against the full-detail picture on four natural
pictures (ideal channel, guests present), the unwindowed reference and the
refit rank together (mean -14.4 and -15.4) and every window lower (-23 to
-41): that metric rewards retained detail and does not see halos, so it does
not choose between them. Which looks best is a question for the eye.
This ranks kernels; it is not tape validation, and by section 11.3's rule none
is the default until it has a viewer preference.

## 12. Receiver display

The receiver publishes the newest decoded picture (float values, plane shapes,
aspect code, and the pixel grid when the packet names one) through
`tools/v7_display.LatestFrame`, a one-slot mailbox with no queue. Display code
reads it and does not change decoder state or `--save-dir` exports.

`tools/v7_gl_viewer.py` is the GLFW/ModernGL (OpenGL 3.3) viewer used by
`receive`. `tools/v7_live.py gui` opens `tools/v7_receiver_gui.py`, which
starts in a setup page without opening audio, lists input devices, exposes the
receive options, and switches between Setup and Live. `tools/v7_viewer.py
IMAGE...` shows still images through the same viewer with no audio or decoder
(`--fps`, `--loop`, `--fullscreen`, `--display-mode`, `--show-info`).

Display upscalers (`DISPLAY_MODES`): `nearest` (8-bit RGB picture from Pillow,
nearest-sampled), `bilinear`, `sharp-bilinear`, `bicubic` (Mitchell–Netravali,
B = C = 1/3), `spline36`, `robidoux`, `robidoux-sharp`, `cubic-bspline`,
`kaiser-sinc` and `hann-sinc` (separable, radius 3, Kaiser β = 8.6) and
`ewa-jinc` (radial, radius 3.238). All except `nearest` work on float Y/Cb/Cr
planes and convert to RGB in the shader with the full-range BT.601 matrix
that matches Pillow's convention.

Further display stages (receiver GUI settings):

- **DCT reconstruction** (`off`, `2x`, `4x`, `8x`, `16x`, `viewport`):
  re-evaluates each plane's cosine expansion on a larger grid. The plane's
  DCT is zero-padded to the target size, scaled by the square root of the
  size ratio and inverted. `viewport` uses the picture viewport's framebuffer
  size.
- **Pixel display**: shows the sent grid as hard pixels (the pixel grid the
  packet names, else 40 × 48 luma) and locks the other display settings.
- **Edge reconstruction** (`on`, `high`, `off`) with strength 100, 75, 50 or
  25%: rebuilds luma as a flatter, sharper picture that still matches the
  received coefficients (`animation_modem/v7_dct_display.py`).
- **Display grain** (`off`, `flat`, `detail`): `flat` is fine per-pixel noise
  in flat areas only. `detail` is band-limited noise on a lattice tied to the
  coder grid (1.5 cells per luma grid sample), so its energy stays just above
  the band the wire carries at any window size; it is strongest where the
  picture has detail and absent in black. It adds no information.
- **Output dither** (on, off): adds ±1 code of triangular noise per channel
  before the 8-bit framebuffer rounds, so smooth gradients do not band on a
  large screen. On by default; Pixel display is never dithered.
- **Colour detail** (`off`, `guided`): rebuilds each chroma plane on the luma
  grid. Within a small neighbourhood the slope of chroma against luma is
  measured in the band both planes were sent in; that slope times the luma
  detail outside the band fills the chroma coefficients the wire did not
  carry. Received chroma coefficients are unchanged
  (`v7_dct_display.guided_chroma_plane`). The receiver GUI defaults to
  `guided`; this is inferred display detail, not additional received data.

Defaults: the receiver GUI starts with `bicubic`, DCT reconstruction `4x`,
edge reconstruction `on` at 75%, luma-guided colour detail and grain off. The
plain viewer starts with the saved mode or `nearest`. The saved mode is
`$XDG_CONFIG_HOME/modemTest/v7_display.json` (else `~/.config/...`),
`{"version": 1, "mode": NAME}`.

Viewer keys: `F` fullscreen, `I` diagnostics, `U` / `Shift+U` and `1` to `9`
choose the upscaler, `Esc` closes a menu, leaves fullscreen, then closes. In
the GUI, `P` toggles image-only view.

## 13. Mono wires

### 13.1 Why a mono sum of the stereo wire fails

On the stereo wire the mono sum `(L + R)/2` cancels S and leaves M. M carries
1,264 of the 2,320 slots: the head, and 1,056 body slots. The 96 tail slots
are all on S. Through the rank interleave, the M slots hold ranks 0 to 1,231
and 32 of ranks 1,232 to 1,295. The remaining 1,616 of the 2,880 coefficients
never reach a mono listener, in any packet. All 500 hosts of Fold 500 have
ranks of 1,546 or more, so they are on S and a mono sum loses every guest
too.

One channel alone, L or R, is `(M ± S)/√2`: every stereo slot holds two
unrelated coefficients added together. It is not a mono signal and the
receiver must keep treating it as one channel of an M/S pair.

### 13.2 Mono wire design

A mono wire keeps the packet format of section 2 and changes the body:

- Data and pilots are on M only. The S pilots are zero, and the receiver
  swaps in M-only pilot tables while it decodes (`mono_wire.mono_channel_profile`).
- All 158 M groups are used: 1,264 slots. The head stays on its 26 groups.
- The amplitude is scaled by √2.
- The same 1,264 coefficients are sent in every packet. There is no rotating
  tail and the receiver uses no tail memory, so every picture is built from
  one packet.
- The sender writes the packet, including tones and EOF marker, to one output
  channel (`--mono-video-side left|right`, default `right`) and silences it on
  the other. The other channel carries source audio
  (`--source-audio source|device|off`), delayed by one packet plus
  `--source-audio-delay-ms`.

### 13.3 Signalling and reception

A mono profile is named by its preamble word and the equal coded status
(codes 3, 4, 5). The metadata has no profile bit.

The default receiver scans each input channel separately. A mono profile is
taken from the channel that shows its word. When both channels show it, the
right channel is used, unless `--mono-video-side left|right` fixes the
channel. The receiver decodes that one channel as a one-channel input. Its
audio passthrough (`--audio-output-device`) plays the other input channel.

A receiver that does not know a profile must not interpret its body. This
receiver holds the last picture for an unknown or unconfirmed status.

## 14. Wire profiles

### 14.1 Profile IDs and dispatch

| ID | Preamble and status name | Profile |
|---:|---|---|
| 0 | `FOLD_OFF` | `aspect-fold-500`; with the metadata model bit `nearest`, a pixel grid |
| 1 | `FOLD_500` | `fold-500` |
| 2 | `FOLD_1000` | Reserved, unused (was `fold-1000`, removed) |
| 3 | `MONO_OFF` | `aspect-mono-500`; with the metadata model bit `nearest`, `stereo-slices` |
| 4 | `MONO_500` | `mono-fold-500` |
| 5 | `MONO_1000` | `mono-colour-500` |

The standalone sender selects one with `--profile`. All of them require the
canonical fixture, coded status and the EOF marker. IDs are not renumbered or
reused. The unfolded wire (`--experimental-fold 0`, and
`tools/v7_wire_profile.py` `baseline` for benches) uses preamble ID 1 with
steady tones and no status.

The default receiver (`_AdaptiveProfileDecoder`) works as follows.

1. It reads the preamble ID on each input channel for every complete packet.
   The two channels must not disagree.
2. It starts in `fold-500`. A different ID becomes active after three
   consecutive packets that show it; those packets are held. An unreadable ID
   clears the count. A switch resets the tail memory and the last verified
   metadata.
3. It dispatches IDs 0, 1, 3, 4 and 5. The reserved ID 2 and the unfolded
   wire are not dispatched.
4. Before the body is interpreted, the decoded tone status must equal the
   dispatched ID and the active profile. Otherwise the packet is held.

### 14.2 Fold 500

The base wire of sections 3 and 4: the 48 × 40 luma and 24 × 20 chroma corners,
head, body and a tail rotating over seven packets, on M and S. The fold of
section 10 is applied with the pinned table: 500 luma hosts, companded guests
from outside the luma corner, 16 signature slots. Model `box`.

### 14.3 Aspect Fold 500

Same slots as Fold 500 (208 head, 2,016 body, 96 tail, 500 folded). What
changes is which coefficients they carry. For a layout W:H, coefficient
(u, v) of a plane (u vertical, v horizontal frequency index) is ordered by
the key `u²·W² + v²·H²`, ties by u then v. The first 1,920 luma and 480 per
chroma plane are kept; the next 500 luma are the guests. Statistics are the
canonical `box` model's variance curve re-read in picture frequency. The
tables are frozen in `test_modem_v7/aspect_tables.npz` (hash pinned). The fold
is companded with D = 1.0, U = 12, μ = 4 and guest noise limit 0.1.

Layouts: `1:1`, `4:3`, `3:2`, `16:9`, `3:4`, `2:3`, `9:16`.
`--aspect-layout auto` (default on both ends) uses the layout of each packet's
aspect code. A sender with a fixed layout boxes each picture into that ratio
and sends the layout's code with the screen bit set, so a receiver on `auto`
follows it. When a packet's metadata fails, the receiver tries the last
confirmed layout and shows the packet only if that layout's fold signature
scores at least 0.5.

Tail modes (`--aspect-tail`) say what the 96 tail slots carry:

| Mode | Tail slots |
|---|---|
| `fixed` (default) | The 96 strongest of the 656 tail coefficients in every packet; the other 560 are not sent; no tail memory |
| `chroma` | The 656 lowest-ranked coefficients (all chroma), 96 per packet over seven packets |
| `split` | 48 further luma coefficients in every packet and 48 rotating chroma |
| `luma` | 96 further luma coefficients in every packet; the 96 weakest chroma are not sent |

The tail mode is not signalled. Sender and receiver must be set alike.

### 14.4 Pixel grids

A pixel grid is Aspect Fold 500 with a coefficient set chosen so that a small
picture arrives pixel for pixel: all coefficients of a luma rectangle, and
per chroma plane the 480 lowest of a rectangle half that size per axis. The
sender's `--pixel-encode` area-averages the frame to the rectangle.
`--pixel-grid` picks the size (rows × columns of luma):

| Layout | `robust` (default) | `large` |
|---|---|---|
| 1:1 | 42 × 42 | 48 × 48 |
| 4:3 | 38 × 50 | 42 × 56 |
| 3:2 | 36 × 52 | 40 × 60 |
| 16:9 | 32 × 58 | 36 × 64 |

Portrait layouts are the transposes.

- `robust`: the whole rectangle fits the 1,904 ordinary luma slots. Nothing
  rides as a picture guest.
- `large`: the rectangle exceeds the ordinary slots; its 400 to 496 highest
  coefficients (rectangle size minus 1,904) ride as fold guests.
- **Filler slots.** Luma positions outside the rectangle fill the remaining
  luma slots. The 16 lowest-ranked luma slots are filler in both grids and
  carry the fold signature, so every rectangle coefficient is carried. The
  receiver zeroes everything outside the rectangle before display.
- **Tail.** `fixed` or `chroma` only. The receiver uses its aspect tail mode
  when it is one of those, else `fixed`.
- **Signalling.** Status 0 with the metadata model bit set to `nearest`, which
  the ordinary layouts never send. The tables are in
  `test_modem_v7/pixel_tables.npz` (hash pinned).
- **Telling the grids apart.** The two grids of a layout use the same
  signature slots with sign patterns that are exactly uncorrelated. The
  receiver decodes with the last confirmed grid (`robust` first) and accepts
  it when the signature score is at least 0.5 and the signature error at most
  0.7. If not, it decodes the packet again with the other grid: on the first
  unconfirmed packet and then on every fourth. A packet is held only while no
  grid has ever been confirmed.

On `fold-500`, `--pixel-encode` uses the base wire's 48 × 40 luma corner (a
40 × 48 picture) and involves no pixel-grid table.

### 14.5 mono-fold-500

Mono wire (section 13.2), status 4. The 1,264 slots carry corner ranks 0 to
1,263 of the `box` model. Ranks 764 to 1,263 are the hosts and ranks 1,264 to
1,763 the guests (luma or chroma, whatever those ranks are). Linear guests,
D ≈ 1.141. The last 16 hosts are signature slots, leaving 484 guests. The
fold's identity hash is pinned in `mono_video.py`.

### 14.6 mono-colour-500

Mono wire, status 5. The head is unchanged; all later ranks are re-ordered
with chroma variance multiplied by 4. The 1,264 slots carry the first 1,264
of that order. Hosts are the 500 lowest-ranked luma coefficients among them,
guests the next 500 luma coefficients of the order. Companded guests, D = 1.0,
U = 12, μ = 4, guest noise limit 0.15, 16 signature slots. Identity hash
pinned in `mono_video.py`.

### 14.7 aspect-mono-500

The wire of mono-colour-500 over an aspect layout's 2,880 coefficients
(`aspect_tables.npz`, the `chroma` tables' plane split) instead of the fixed
corners. Status 3 with the metadata model bit `box`. `--aspect-layout` and
the screen bit work as in 14.3. There is no tail setting.

### 14.8 stereo-slices (experimental)

Each output channel is an independent mono wire (section 13.2) with its own
preamble, pilots, metadata, tones and EOF marker. Both are built in one
packet: with left and right cell values `l` and `r`, the encoder sets
`M = (l + r)/2` and `S = (l − r)/2`.

Per plane, the coefficients nearest DC in the layout's picture frequency
(the key of 14.3) are cut into a *base* and, beyond it, as many *detail*
coefficients: 1,020 luma, 110 per chroma plane. Each base coefficient is
paired with one detail coefficient, strongest base with weakest detail. A
channel's 1,264 coefficients are:

| Count | Content |
|---:|---|
| 1,020 | Luma: left = base + 0.35 × detail, right = base − 0.35 × detail |
| 16 | Filler carrying a fixed ±3 pattern, the same on both channels; not used by the receiver as built |
| 8 | Marker: `+ − + + − + − −` at 3 standard deviations, positive on the left channel and negative on the right |
| 110 + 110 | Cb and Cr base only, identical on both channels |

Slots are ranked with chroma variance multiplied by 4. No fold is used (the
optional fold in the module is off) and there is no pixel grid. Tables are in
`test_modem_v7/slice_tables.npz` (hash pinned).

Signalling: status 3 (the aspect mono status) with the metadata model bit set
to `nearest`. The layout follows the aspect code; with failed metadata the
last layout is kept.

Receiver:

- The marker score per input channel says what that channel holds: at least
  0.4 is left, at most −0.4 is right, between is a sum. The score is averaged
  per input channel (a quarter of each new clean packet), so swapped cables
  are followed and one bad packet does not relabel a channel.
- Left and right together: base from the mean, detail from half the
  difference divided by 0.35. This gives 2,040 luma coefficients.
- A sum: the detail cancels and the base remains.
- One channel: the slot is split between base and detail by their model
  variances.
- When both channels' averaged markers agree within 0.04 and imply a
  crosstalk share between 0.02 and 0.3, that share is removed before joining.
- The second channel is joined only when it decoded the same source index. A
  channel is *damaged* when its head confidence is below 0.9 or symbols were
  erased. A damaged channel is not joined with a clean one; two damaged or two
  clean channels are joined.

## 15. Known limits

- **Real tape is unvalidated.** All measurements are synthetic.
- **The equalizer's confidence is pessimistic.** Against the coefficients
  actually sent, the squared error is 0.2 (clean) to 0.7 (white noise at
  -40 to -30 dBFS) of what the confidence implies, so noisy slots are shrunk
  slightly more than they need to be.
- **Reverse playback joins only one stereo-slices channel.** The second
  channel is decoded only for forward packets.
- **A partly damaged slices channel is not mixed per slot.** It is dropped
  whole in favour of the clean channel.
- **The body is sent quieter when the picture is loud.** A packet whose
  body would come within 1.5 dB of the header is scaled down (section 2.4),
  which costs it signal-to-noise on a noisy link.
- **The receiver does not yet check the body against the header**, nor the
  end marker's level against the header's.
- **Reverse playback through a 4 kHz low-pass** shows far fewer pictures than
  forward playback of the same recording.
- **Sender level calibration comes from a statistical model.** `unit_rms` is
  measured on coefficients drawn from the model; real pictures exceed it.
- **The unfolded wire is not dispatched automatically.** It needs
  `receive --experimental-fold 0`.
- **The aspect tail mode is not signalled** (section 14.3).
- **The loop lock can learn from one packet.** `LoopLock` counts matching
  observations, not distinct packets, and the decoder may submit one packet's
  metadata more than once (splice and tone retries).
- **The live input needs a following header.** The last forward packet of a
  finite input is not decoded.

## 16. Open questions

Questions about the code's intent, with the owner's answers. Answered
questions whose work is done were removed from this list.

1. The receiver-side fold and every profile except the application sender's
   Fold 500 live in `test_modem_v7/` and are installed by replacing functions
   of `animation_modem.v7` at run time. **Decided, pending:** move the
   receiver-side fold and the wire profiles into `animation_modem/` and
   replace the run-time function replacement with arguments. Not done yet
   (section 17, item 3).

Decided and done:

- The hidden `send --experimental-mono` option (the rotating mono wire, which
  sent status 3 with the model bit `nearest`, the signalling of
  `stereo-slices`) was removed together with its receiver option.
- The user-facing no-fold options were retired: `send --baseline`,
  `receive --baseline` and `main.py --modem-baseline`. The transport can still
  encode and decode a packet without a fold.
- Fold 1000 was removed for now. Its status and preamble ID 2 stay reserved.
- The continuous clock-track code (`encode_stream`, `decode_stream`,
  `receive --refine`) was deleted from `v7.py`. `clock_word` is kept;
  `decode_frame` uses it for its cancellation template.
- `main.py --modem-encode-filter` offers only `nearest` and `box`.

## 17. Roadmap

1. Refit the coefficient model on real pictures. Real pictures carry about
   3.5 to 4 times the energy the model's tables assume, which makes bodies
   louder than designed and makes the receiver smooth noisy slots too hard.
   Prior recordings need not stay decodable.
2. Receiver level checks using the header and end pulses: check the body is
   in range against the header; compare the end pulse (or the next header)
   with the header; re-level through the packet when they differ. Also test a
   very light limiter applied to the body only, against the current
   whole-body scaling.
3. Move the receiver-side fold and wire profiles into `animation_modem/` with
   arguments instead of run-time function replacement (section 16).
4. Grid and picture-ratio signalling as two separate metadata fields, using
   the bits a non-rotating tail frees; arbitrary ratios if the bits allow.
5. Profile comparison on mixed picture shapes (16:9, 4:3, 3:4, 5:6) with the
   non-rotating tail as the default everywhere. Low priority.
6. A faster header (about 0.4 more pictures a second), only after playback
   from 0.25x to 1x and from 2x to 4x is confirmed to lose nothing.
7. Matched sum/difference pairing on the stereo profile against
   stereo-slices, to decide whether stereo-slices stays. Delayed.
8. Sender conveniences: resume a movie where it stopped and playback
   controls on the preview window are built (section 8); the position is
   counted from the frames read. Open: a variable-frame-rate file is placed
   by its average rate, and a file with no stored frame count repeats on its
   probed duration.
9. Housekeeping: prune tests that only pin internals; port the stereo-slices
   sender and join to numba; find the cold-start test flakiness
   (`test_embedded_video_audio_uses_the_shared_capture_clock` and first-run
   failures in a fresh checkout).
