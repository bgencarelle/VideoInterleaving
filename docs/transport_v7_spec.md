# V7 Transport Specification

This document describes the V7 transport as implemented in
`animation_modem/v7.py`, its live input adapter, the application sender, and
`tools/v7_live.py`. It is a current-state specification, not a claim that every
wire option has been validated on real tape.

## 1. Current status and scope

V7 is the pulse-framed modem transport used by modem mode. The live wire uses
the edge-counted pulse acquisition retained in `animation_modem/transport3.py`;
it does not use the continuous 72-bit clock-track proposal described in older
sections of this document. The live packet carries an OFDM image body, a
CRC-protected metadata symbol, optional low-bin timing tones, and an optional
end-of-packet (EOF) marker.

The transport package is independent of application settings, renderers, and
audio devices. `modem_v7_display.py` adapts it to the application image library
and audio output. `tools/v7_live.py` provides a standalone capture sender and
receiver. Synthetic decode tests are available; real tape and deck validation
remains pending.

## 2. Packet format and acquisition

At the 48 kHz reference geometry, one pulse packet is 3,920 samples:

| Region | Reference samples | Description |
|---|---:|---|
| Pulse header | 288 | Edge-counted biphase-mark preamble |
| OFDM body | 3,456 | 24 symbols × 144 samples |
| Metadata symbol | 144 | One OFDM symbol carrying 40 protected bits |
| Guard | 32 | Packet endpoint region; final 24 samples carry EOF when enabled, and packet-wide timing tones continue through it |
| **Total** | **3,920** | **12.245 packets/s at 1×** |

The pulse word contains 16 known bits with 8-sample half-bits and nominal
amplitude 0.55. `transport3.measure_pulses()` locates its Schmitt-triggered
edges, interpolates zero crossings, matches the short/long edge-interval
pattern, and estimates packet scale. V7 timing acquisition is pulse-counted,
not FFT-correlated. Captures are measured in their native sample coordinates;
the accepted scale is normalized by the capture rate relative to 48 kHz.

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
and receiver and the application sender enable EOF by default. The low-level
`encode_pulse_frame()` / `encode_pulse_stream()` and
`decode_pulse_stream()` APIs default to legacy framing unless their EOF options
are specified.

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

Packets may be pitch-shifted in the range 0.25×–4×. The receiver measures the
resulting pulse scale and resamples the packet onto the reference grid. Faster
playback increases both packet rate and carrier frequencies. With a 14 kHz
nominal emission edge, the full-band Nyquist guide is
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

The application sender and standalone live sender default to the nearest
encoding profile, 1× playback, pilot tones enabled, and EOF markers enabled.
The standalone sender additionally defaults to brightness 1.05 and gamma 1.0.
The standalone receiver defaults to EOF boundaries, tone-seeded pilot timing,
baseline pulse timing, tone equalization off, one decode batch, one frame of
history, and tail memory enabled. Both standalone sender and receiver require
an explicit audio device.

The low-level encoder's pilot-tone and EOF-marker switches default off, and
the low-level decoder defaults to legacy next-header boundaries and baseline
pilot timing. This distinction is intentional in the current implementation:
the application and standalone live adapters set live defaults explicitly,
while unit tests and synthetic baseline runs can select the older wire.

Numba is imported unconditionally by the V7 transport and pulse-acquisition
modules. The default equalizer and selected acquisition kernels use Numba JIT;
the standalone live receiver calls `warmup_equalizer()` before opening its
audio stream so first-use compilation does not stall capture. The
`--force-float32` path is an experimental arithmetic path, not an optional
Numba installation mode.

## 9. Verification evidence and limits

The V7 test suite was run with:

```text
.venv/bin/python -m unittest discover -s modem_tests -p 'test_v7_*.py' -v
```

Result: **117 tests passed**. This covers the V7 unit tests, including frozen
table integrity, packet metadata, pulse/EOF framing, live-input behavior,
loop handling, and decode paths. It is synthetic/unit evidence, not a real
device or tape test.

The matching test modules, all relative to the repository root, are:

```text
modem_tests/test_v7_capture_video.py
modem_tests/test_v7_decode_speed.py
modem_tests/test_v7_encode_speed.py
modem_tests/test_v7_eof.py
modem_tests/test_v7_gl_viewer.py
modem_tests/test_v7_image_quality.py
modem_tests/test_v7_live_input.py
modem_tests/test_v7_loop.py
modem_tests/test_v7_metadata.py
modem_tests/test_v7_mono.py
modem_tests/test_v7_mono_torture.py
modem_tests/test_v7_pilot_tones.py
modem_tests/test_v7_pulse_warp.py
modem_tests/test_v7_speed.py
modem_tests/test_v7_tables.py
modem_tests/test_v7_tone_equalization.py
```

The shared image fixture is `modem_tests/fixtures/v7_reference_face.png`.
The synthetic impairment matrix runner is `tools/v7_torture_matrix.py`.

The synthetic impairment matrix was run with:

```text
.venv/bin/python tools/v7_torture_matrix.py --out tmp/v7-spec-rerun
```

It encodes 12 identical reference-fixture packets, resamples the wire from
48 kHz to 96 kHz, and applies 25 deterministic synthetic cases (seed 2026),
including clean, low-pass, hiss, wow/flutter, crosstalk, track imbalance,
dropouts, NR pumping, saturation, mono sum, and one-leg-only paths. The
baseline decoder is expected to return 11 frames because the final packet has
no next-header witness. Its acceptance check requires 11 results, 11
metadata-valid results, and 11 displayable results; it does not require every
result status to be `received`.

The run failed only the `lowpass-4k` acceptance row: it returned 11 frames,
10 received, 1 lost, 10 metadata-valid, and 11 displayable. The other 24 cases
met the matrix's stated acceptance check. This is a synthetic threshold
failure, not a claim that the image was wholly undecodable: it records that
one of the 11 frames did not pass metadata validation under that specific
4 kHz low-pass impairment.

The saved matrix output is `tmp/v7-spec-rerun/results.json`.

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

**Status: proposal, not implemented on the wire.** A live prototype exists:
`tools/v7_live.py --experimental-fold M` on both ends (see 10.7). The
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

**M = 500 with the fallback, as an opt-in mode.** Measured: +10 on clean,
low-pass and dropouts, +7 with slow wow, and −0.4 to −1.3 at worst (fast
flutter, combined impairments, 0.3 % jitter). M = 1,000 gains more on clean
paths but loses 2–3 points under flutter.

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
   - for a first live experiment only, set the same `--experimental-fold`
     flag on both ends.

   The live prototype instead marks folded packets with an in-band signature
   (10.7), so a folding receiver also shows normal packets correctly.
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

`tools/v7_live.py send|receive --experimental-fold M` loads a frozen table,
`test_modem_v7/fold_table_<M>.json`, built from the reference fixture. It
holds the hosts, guests, guest scales and the step. The sender folds each
frame's values before the normal encode. The receiver wraps
`v7.decode_frame` and the equaliser entry points while it runs; they are
restored in a `finally`. It keeps each packet's equaliser output and unfolds
before display. `animation_modem` is unchanged.

Table loading fails closed:
- **Pinned files.** Each table file must match a SHA-256 pinned in
  `live_fold.py` and be in canonical form. A rebuilt table must be pinned
  deliberately.
- **One model per table.** A table records the digest of the model it was
  built for: the canonical `box` profile under the current model-table
  hash. The sender refuses to start with any other encode filter or
  fixture. The receiver shows packets from any other model without
  unfolding.

The prototype adds a signature. The 16 weakest host slots carry a ±3D
pattern instead of data. The pattern is drawn from the table's identity, and
it serves three purposes:
- **Detection.** The receiver unfolds only packets that carry its own
  table's pattern (score ≥ 0.5). Normal packets, and packets folded with a
  different table, are shown as they are.
- **Noise measurement.** The pattern's residual measures the packet's symbol
  noise on exactly the folded slots. Timing smear from fast flutter and
  jitter shows up there, not in the equaliser confidence.
- **Weighting.** Guests are weighted by β²/(β² + noise²), and above 0.3 steps
  of noise the packet is not unfolded.

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

## 10. Fold proposal

**Status: proposal, not implemented on the wire.** The measurements below are
synthetic. They come from `test_modem_v7/`, run on 11 frames of an 810×1080
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
- **Step D:** chosen for a 30 dB design SNR (D ≈ 0.97 for both M = 500 and
  M = 1,000).

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

**M = 500 with the fallback, as an opt-in mode.** Measured: +10 on clean,
low-pass and dropouts, +7 with slow wow, and −0.4 to −1.3 at worst (fast
flutter, combined impairments, 0.3 % jitter). M = 1,000 gains more on clean
paths but loses 2–3 points under flutter.

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
   - for a first live experiment only, set the same `--experimental-fold`
     flag on both ends.

   A sender/receiver mismatch affects only the folded slots.
5. **Tests.** Add fold/unfold round-trip tests (noiseless unfold within the
   step quantisation; the fallback path) to `modem_tests/`, plus a regression
   that an unfolded wire decodes identically.

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
