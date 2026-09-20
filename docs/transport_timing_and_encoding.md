# Modem transport, timing, and encoding specification

This is a reimplementation specification for the modem wires that exist in
the repository history. It is deliberately source-referenced: a statement
without a source reference is either a derivation from cited constants or is
labelled **UNVERIFIED**.

## Branch provenance

The current modem implementation is in `standalone-modem`, currently at
`37b93447` (`origin/standalone-modem`). The early V1/V2 modem history is on
the `experiment` branch (`8224a36f`, `origin/experiment`), and the V1/V2
source commits cited below are reachable from that history. The project also
has separate `main` (`0e30eb25`, `origin/main`) and `scope` (`01506d18`,
`origin/scope`) branches; they are application/scope histories, not alternate
implementations of the wires specified here.
`[C:git branch -a -vv, repository state captured 2026-09-20]`

There is no literal local or remote ref named `modem` in the checkout used to
write this document. Do not silently pull `main` or `scope` changes into a
wire definition; their source files may have the same filenames but are not
this transport's provenance. **UNVERIFIED:** the contents of any external
branch named `modem` not present in this clone.

## 0. Provenance and notation

### Source-reference convention

`[C:path:lines]` means the file at the current `standalone-modem` checkout,
commit `37b93447c2eee6a8647f0bcb0ca2972dd0268375`. Historical references use
`[H364:path:lines]` for the V1 introduction commit
`364c29286a279680444cb7a1274db8de5101a79a`, `[H337:path:lines]` for the V2
introduction commit `33734bcf4308a8f9256eff4695905b745a18d356`, and
`[H99:path:lines]` for the later pinned V2 integration snapshot
`99d60665688ef6af197be37ebcded57bc36aaf25` (`99d60665`). Commit ranges in
the version table identify where a wire or source coder was introduced; the
line citations describe the implementation being specified.

The line numbers are intentionally part of this document's contract. If a
future edit moves a cited implementation, update this document and rerun the
vector script.

### Version provenance

| Version / wire | Introduction or control commit | Active source of truth |
|---|---|---|
| V1 | introduced `364c2928`; pinned final historical snapshot `99d60665` | `[H364:animation_modem/transport.py:12-136]`, `[H99:animation_modem/transport.py:14-155]` |
| V2 | introduced `33734bcf`; pinned final historical snapshot `99d60665` | `[H337:animation_modem/transport2.py:25-310]`, `[H99:animation_modem/transport2.py:25-342]` |
| V3 `wire` | `7e664932` (V3 magic/profile cleanup); current geometry at HEAD | `[C:animation_modem/transport3.py:66-68]` |
| `wire-hd` | `34ae7739` | `[C:animation_modem/transport3.py:70-76]` |
| V5 `hd-dwt` | `34ae7739`; tape reallocations `d3a9cbad` | `[C:animation_modem/engines.py:129-170]`, `[C:animation_modem/wavelet.py:705-833]` |
| `wire-tape` | `d3a9cbad` | `[C:animation_modem/transport3.py:78-84]`, `[C:animation_modem/engines.py:34-49]` |
| `wire-tape-25` | `d3a9cbad` | `[C:animation_modem/transport3.py:86-92]` |
| V6 | `4a672b07` | `[C:animation_modem/v6.py:1-224]` |
| V6 full-repeat | `914756a2`; evaluation note `37b93447` | `[C:animation_modem/transport3.py:102-108]`, `[C:MODEM_TODO.md:139-152]` |
| V6 tape placement | `4a672b07` | `[C:animation_modem/v6.py:150-186]` |

The V3-family current source is a single transport: V3, V5, and both tape
wires reuse the same OFDM encoder/decoder and differ primarily in layout and
coder selection. `[C:animation_modem/engines.py:1-12]`

The companion state-machine attachment is
[`transport_state_machines.md`](transport_state_machines.md). It expands the
shared encoder/receiver control flow into named V1, V2, V3, V5, tape, and V6
instances rather than treating “shared transport” as an undocumented
abstraction.

## 1. Names, magic, profiles, transforms, and receiver map

### Name map

| User-facing name | Layout name | Profile(s) and header code | Header magic | Transform / source coder |
|---|---|---|---|---|
| V1 color | historical fixed format | `color`, not a current profile code | `SI01` | raw YCbCr image samples |
| V1 non-color | historical fixed format | `detail` or `mono`; both use `SI02` in the legacy header | `SI02` | raw image samples |
| V2 wide | `wide` | static endpoint profile, normally `color` | `V3` because it is progressive | DCT |
| V2 tape | `tape` | static endpoint profile | `V2` | DCT |
| V2 tape-fast | `tape-fast` | static endpoint profile | `V2` | DCT |
| V2 narrow | `narrow` | static endpoint profile | `V2` | DCT |
| V2 lofi | `lofi` | static endpoint profile | `V2` | DCT |
| V3 wire | `wire` | `color-dct` code 0; `color-wavelet` code 1 | `V4` in the current dense layout | DCT or one-level Haar wavelet |
| wire-hd | `wire-hd` | `hd-dwt` code 2 | `V4` | two-level CDF 9/7 DWT with 800 copies |
| V5 | `wire-hd` | `hd-dwt` code 2 | `V4` | same active coder as `wire-hd` |
| wire-tape | `wire-tape` | `hd-dwt` code 2 | `V4` | 800-value DCT plus 800 stereo copies; the profile name is legacy/misleading here |
| wire-tape-25 | `wire-tape-25` | `tape-80x60` code 3 | `V4` | 360-value DCT plus 360 stereo copies |
| V6 | `wire-v6` | `v6-dct` code 0; `v6-wavelet` code 1 | `V6` | DCT or two-level CDF 9/7, 720 foundation copies |
| V6-repeat | `wire-v6-repeat` | `v6-repeat-dct` code 0; `v6-repeat-wavelet` code 1 | `VR` | DCT or two-level CDF 9/7, every coefficient copied |

The current four-code legacy registry is exactly
`color-dct`, `color-wavelet`, `hd-dwt`, `tape-80x60`; V6 names are experimental
and reuse codes 0/1 only inside their distinct wire magic and geometry.
`[C:animation_modem/core.py:732-766]`, `[C:animation_modem/imaging.py:27-57]`,
`[C:animation_modem/core.py:203-224]`, `[C:animation_modem/engines.py:227-233]`

### Which receiver decodes which wire?

| Receiver | Wires decoded |
|---|---|
| Historical V1 `transport.Receiver` | V1 fixed `FRAME=3200` packets only; no V2/V3 auto-detection. `[H99:animation_modem/transport.py:352-469]` |
| Historical V2 `transport2.Receiver` | One caller-selected V2 preset; it does not identify arbitrary presets from the wire. `[H99:animation_modem/transport2.py:612-796]` |
| Current `transport3.Receiver` without `candidates` | Only its configured layout/coder; header profile selection works only when `coders` is supplied. `[C:animation_modem/transport3.py:851-871]` |
| Current utility receiver | `wire-tape-25`, `wire-tape`, `wire`, `wire-hd`, `wire-v6`, and `wire-v6-repeat`, in packet-length order after candidate construction. `[C:utilities/modem_v3_check.py:96-128]`, `[C:animation_modem/transport3.py:597-600]` |
| `transport3_v5.ReceiverV5` | Parked experimental mono/LDPC path for a supplied layout; it is not the active V5 receiver. `[C:animation_modem/transport3_v5.py:1-18]`, `[C:animation_modem/engines.py:129-167]` |
| `tools/decode_wav.py` | Legacy offline tool currently lists only `wire-hd`, `wire-tape`, and `wire-tape-25`; it does not list V6. `[C:tools/decode_wav.py:74-86]` |

The current utility's receiver tries candidate layouts, verifies the layout
magic/top-bin/profile/header CRC, locks to the first verifying candidate, and
restarts candidate detection after four unverified packets. `[C:animation_modem/transport3.py:851-871]`, `[C:animation_modem/transport3.py:911-919]`

## 2. Shared current V3-family sample geometry

### Sample clock and packet arithmetic

The reference rate is 48,000 Hz. Hardware is not requested to run at that
rate; it is the geometry/reference clock and the WAV rate used by the offline
writer. `N=128` is the useful FFT length, `CP=16`, `SYMBOL=144`, `SYNC_LEN=288`,
and `GUARD=32`. A layout with `S` total OFDM symbols has:

```text
packet_samples = SYNC_LEN + S * SYMBOL
frame_samples   = packet_samples + GUARD
fps(rate)       = rate / frame_samples
carrier_hz      = bin * rate / N
```

These formulas and constants are `[C:animation_modem/core.py:37-62]`,
`[C:animation_modem/core.py:163-195]`.

| Layout | top bin / carrier band at 48 kHz | image symbols | header symbols | total symbols | packet / frame samples | fps |
|---|---:|---:|---:|---:|---:|---:|
| `wire` | 54 / 375–20,250 Hz | 14 | 4 | 20 | 3,168 / 3,200 | 15.000000 |
| `wire-hd` | 54 / 375–20,250 Hz | 16 | 4 | 22 | 3,456 / 3,488 | 13.761468 |
| `wire-tape` | 34 / 375–12,750 Hz | 12 | 4 | 18 | 2,880 / 2,912 | 16.483516 |
| `wire-tape-25` | 34 / 375–12,750 Hz | 5 | 4 | 11 | 1,872 / 1,904 | 25.210084 |
| `wire-v6` | 34 / 375–12,750 Hz | 29 | 4 | 35 | 5,328 / 5,360 | 8.955224 |
| `wire-v6-repeat` | 34 / 375–12,750 Hz | 47 | 4 | 53 | 7,920 / 7,952 | 6.036217 |

The layout definitions are `[C:animation_modem/transport3.py:66-108]`; the
header-symbol and packet derivations are `[C:animation_modem/core.py:157-240]`.
The frame-rate values are generated again by `tools/spec_vectors.py`.

### Transmit-rate adaptation and emission ceilings

Receiving is rate-independent: the preamble supplies the received sample
scale, and a faster input clock oversamples the same waveform. Transmitting is
different. For a device rate above 48 kHz, `emit_ratio()` selects a bounded
rational approximation and `band_limited()` resamples the packet with a
polyphase anti-image filter, retaining the reference carrier band and restoring
the encoded peak. Rates at or below 48 kHz are not resampled by this path.
`[C:animation_modem/core.py:554-637]`

The current tape layouts set `emission_ceiling=14000`. `bound_emission()` then
applies a 63-tap Kaiser-window FIR through `filtfilt`, renormalizing the peak;
the 63-tap choice is constrained by the 16-sample cyclic prefix. `[C:animation_modem/core.py:640-683]`

`wire` and `wire-hd` have no declared whole-waveform emission ceiling. Their
information carriers end at 20.25 kHz, but their square-edged preamble is not
automatically limited. `[C:animation_modem/transport3.py:66-76]`

## 3. Current V3-family preamble, pilots, and training

### Preamble and pulse measurement

The active V3-family preamble is a 16-bit biphase-mark word:

```text
PREAMBLE_BITS = (0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0)
HALF = 8 samples; PREAMBLE_AMPLITUDE = 0.55
```

Each bit starts with a transition; bit 1 adds a middle transition. The
preamble is written identically to both output channels after the OFDM body is
normalized. `[C:animation_modem/transport3.py:27-45]`, `[C:animation_modem/transport3.py:133-137]`, `[C:animation_modem/transport3.py:541-550]`

The edge detector uses a Schmitt-style threshold of
`EDGE_HYSTERESIS * PREAMBLE_AMPLITUDE = 0.12 * 0.55 = 0.066`. Pulse
measurement linearly interpolates sign crossings, checks the expected gap
pattern within `max(1.2, 0.45 * expected_gap)`, accepts scale within
`.98*min_scale .. 1.02*max_scale`, and requires confidence at least 0.45.
`[C:animation_modem/transport3.py:31-33]`, `[C:animation_modem/transport3.py:140-151]`, `[C:animation_modem/transport3.py:205-242]`

`Receiver(pulse_only=True)` is the active acquisition default. The normal
states are cold edge acquisition, coast using the predicted next packet,
reacquisition after a miss, and correlation fallback when pulse search fails.
The correlation and waveform-fit paths are retained fallbacks, not the normal
timing path. `[C:animation_modem/transport3.py:583-706]`, `[C:animation_modem/transport3.py:708-785]`

### Training, pilots, and phase seed

`phases(layout, seed=52001)` creates one deterministic complex phase per
symbol, carrier, and channel using NumPy's default RNG with the layout-derived
offset `seed + top_bin*97 + image_symbols`. `[C:animation_modem/core.py:267-270]`

For orthogonal-training layouts, training rows 0 and 1 carry both channel
streams with the phase-locked `[1, 1]` and `[1, -1]` patterns. The four payload
pilots are known value 1 on every body symbol. `[C:animation_modem/transport3.py:495-503]`, `[C:animation_modem/transport3.py:524-534]`

The pilot bins are `[3, 5, 21, 45]` for a 54-bin layout and are proportionally
folded into `[3, 5, 16, 34]` for the tape/V6 34-bin layouts. Their 48 kHz
frequencies are respectively 1,125, 1,875, 7,875, 16,875 Hz for the wide
layout and 1,125, 1,875, 6,000, 12,750 Hz for the tape layout. `[C:animation_modem/core.py:107-127]`

## 4. Current header and carrier mapping

### Header bytes

The active V3-family header is big-endian `>2sBBIHHI`, followed by a
big-endian `zlib.crc32` of those 16 payload bytes:

| Bytes | Field | Meaning |
|---:|---|---|
| 0–1 | `magic` | layout generation: `V4` for dense current layouts, `V6` or `VR` for V6 |
| 2 | `flags` | face folder in high nibble and float folder in low nibble, each modulo 16 |
| 3 | `packed_top_profile` | bits 0–5 `top_bin`; bits 6–7 profile code |
| 4–7 | `absolute` | unsigned 32-bit frame word; current aspect packing uses bits 29–31 and leaves bits 0–28 as the frame counter |
| 8–9 | `index` | one-based source/frame index, 1..`count` |
| 10–11 | `count` | unsigned 16-bit sequence count |
| 12–15 | `stamp_ms` | unsigned 32-bit timestamp modulo 2^32 |
| 16–19 | CRC | CRC-32 of bytes 0–15, big-endian |

The struct, CRC, folder packing, profile bits, and range checks are
`[C:animation_modem/core.py:716-806]`; the aspect word is
`[C:animation_modem/aspect.py:6-37]`; public decode masks the aspect bits at
`[C:animation_modem/core.py:1538-1551]`.

The wire CRC is CRC-32/ISO-HDLC with width 32, polynomial `0x04C11DB7`
(reflected implementation polynomial `0xEDB88320`), initial value
`0xFFFFFFFF`, reflected input and output, and final XOR `0xFFFFFFFF`. The
implementation invokes `zlib.crc32(payload)` with the API seed omitted/zero
and serializes the resulting uint32 big-endian. The numeric parameters are the
standard zlib CRC-32 contract; the repository confirms the API call and byte
ordering but does not reimplement the polynomial. `[C:animation_modem/core.py:768-806]`

The eight current aspect codes are `0=5:6`, `1=1:1`, `2=4:3`, `3=3:2`,
`4=16:9`, `5=2.39:1`, `6=3:4`, and `7=9:16`. `[C:animation_modem/aspect.py:6-26]`

For historical headers, V1's ordinary payload offsets are bytes 0–3 magic,
4–7 absolute, 8–11 index, 12–15 count, 16 width, 17 height, 18 FPS, and
19 numbered, followed by CRC bytes 20–23. Its `ST` variant keeps bytes 0–3 as
`ST/profile/numbered`, bytes 4–7 absolute, 8–11 index, 12–15 count, and
16–19 timestamp before the same four-byte CRC. `[H364:animation_modem/transport.py:101-119]`
V2 uses bytes 0–1 magic, byte 2 flags, byte 3 top bin, bytes 4–7 absolute,
8–9 index, 10–11 count, 12–15 timestamp, and CRC bytes 16–19; V2 does not
assign profile or aspect bits to any header byte. `[H99:animation_modem/transport2.py:286-294]`

Header QPSK uses 80 slots and gain 1.6. The same header is placed on both
channels unless `header_split` is explicitly selected. Header carriers are the
lowest 20 data carriers; dense layouts reuse the remaining data carriers in
header symbols for image values. `[C:animation_modem/core.py:54-57]`, `[C:animation_modem/core.py:134-155]`, `[C:animation_modem/core.py:716-717]`, `[C:animation_modem/transport3.py:505-523]`

### Image slot order

`_slot_order()` emits dense-header spare slots first, followed by image-symbol
slots. With `spread_carriers=True`, it sorts carrier positions middle-out
within the symbol/channel/IQ ordering; without spreading, carrier order is
outermost. `[C:animation_modem/core.py:475-512]`

Ordinary DCT and Haar coders rank coefficients by normalized spatial frequency
and assign lower ranks to earlier slots. `[C:animation_modem/core.py:436-457]`
Coder-specific V5/tape/V6 placement can override this through `coder.slots()`.
`[C:animation_modem/core.py:460-473]`

### Header tolerance and drift correction

If a hard header CRC fails, the decoder considers only the 14 least-reliable
QPSK bits and tries combinations up to the configured tolerance (default 2),
accepting only a corrected word whose complete CRC and structural checks pass.
`[C:animation_modem/core.py:1069-1099]`

If measured symbol drift exceeds 0.25 samples across the packet, the decoder
uses the already verified header and body pilots as known phase references,
refits the global drift rate, re-solves the channel, and retains the original
result if the refit cannot re-verify the header. `[C:animation_modem/core.py:1102-1207]`

## 5. Historical V1

V1 was first added in commit `364c2928` on the historical `experiment`
history; the introductory file already contains the fixed geometry, three
profiles, raw image preparation, header, and direct OFDM encoder specified
below. `[H364:animation_modem/transport.py:12-136]` The later pinned snapshot
`99d60665` adds the fixed off-speed template bank, packet-spacing rate history,
and sinc packet correction while preserving the same V1 frame geometry and
image/header format. `[H99:animation_modem/transport.py:32-48]`,
`[H99:animation_modem/transport.py:352-469]`

V1 is the frame-local analog OFDM link in the pinned historical checkout.
The constants are `RATE=48000`, nominal `FPS=15`, `FRAME=3200`, `N=128`,
`CP=16`, `SYMBOL=144`, and image dimensions 40x48. It has 19 body symbols:
two training, two header, and 15 image symbols; `PACKET=288+19*144=3024`,
leaving a 176-sample zero tail inside the 3,200-sample frame. `[H99:animation_modem/transport.py:14-24]`

The preamble is a 256-sample random waveform synthesized from FFT bins 6:109,
with RNG seed 41015 and peak 0.65. Acquisition searches the seven fixed scale
templates `(-.024,-.016,-.008,0,.008,.016,.024)`. `[H99:animation_modem/transport.py:25-48]`

V1 has pilots `[6,19,35,50]`, all carriers 3..54 excluding pilots, and a
19x52x2 random phase table. The default phase table uses seed 41015; detail
and mono use seeds 41016 and 41017. `[H99:animation_modem/transport.py:17-29]`, `[H99:animation_modem/transport.py:50-59]`

V1 profiles are color `(40,48) luma + (20,24) Cb/Cr`, detail `(48,56) luma +
(8,12) Cb/Cr`, and mono `(48,60)` with no chroma. Preparation aspect-pads to
the profile dimensions with Lanczos; source conversion is Pillow YCbCr, luma
is full-size, and chroma is BOX-resized. Values are mapped from uint8 to
`[-1,1]`. `[H99:animation_modem/transport.py:50-63]`, `[H99:animation_modem/transport.py:85-101]`

There is no transform, coefficient rank, power allocation, or redundancy in
V1: image-plane values are placed directly as analog QPSK I/Q components with
image gain 0.7. `[H99:animation_modem/transport.py:141-151]`

The normal V1 header is 24 bytes: `>4sIIIBBBB` (magic, absolute, index,
count, width, height, FPS, numbered), followed by CRC-32. Color uses `SI01`;
the other legacy profiles use `SI02`. A target-time variant uses `ST` plus
profile/numbered bytes and a 32-bit millisecond timestamp. `[H99:animation_modem/transport.py:120-140]`

The V1 decoder identifies a profile by training-phase coherence, equalizes a
2x2 channel, fits phase slope from pilots per body symbol, decodes the header
with CRC, and reconstructs values. Headerless but coherent data may be emitted
as an estimate; unreliable pixels are spatially concealed from the same image.
`[H99:animation_modem/transport.py:199-290]`, `[H99:animation_modem/transport.py:180-196]`

V1 acquisition is correlation-based, tracks frame spacing after three plausible
locks, and uses windowed-sinc packet resampling for speed error. The default
identity threshold is 0.45. `[H99:animation_modem/transport.py:352-469]`

The early V1 acquisition was simpler: one fixed-rate sync correlation and
packet-spacing inference were used before the later scale bank/rate-history
changes. `[H364:animation_modem/transport.py:262-365]`

## 6. Historical V2 at `99d60665`

V2 was first added in commit `33734bcf`, also on the historical `experiment`
history. The introduction used the four presets `wide`, `tape`, `tape-fast`,
and `narrow`, had no `progressive` layout flag, and always packed header magic
`V2`. `[H337:animation_modem/transport2.py:50-147]`,
`[H337:animation_modem/transport2.py:262-310]` The later `99d60665` snapshot
added the progressive wide layout/magic `V3`, the `lofi` preset, profile-aware
slot geometry, reliability-weighted reconstruction, and the current frame-local
decoder path. `[H99:animation_modem/transport2.py:50-147]`,
`[H99:animation_modem/transport2.py:167-229]`,
`[H99:animation_modem/transport2.py:388-450]`

V2 keeps `RATE=48000`, `N=128`, `CP=16`, `SYMBOL=144`, `SYNC_LEN=288`, and
`GUARD=32`. Its `Layout` derives packet and frame lengths from
`2 + header_symbols + image_symbols`. `[H99:animation_modem/transport2.py:25-35]`, `[H99:animation_modem/transport2.py:50-124]`

The historical preset values are:

| Preset | carriers / band | image symbols | packet / frame | fps | capacity |
|---|---:|---:|---:|---:|---:|
| `wide` | 1–54 / 375–20,250 Hz | 15 | 3,312 / 3,344 | 14.354067 | 3,000 |
| `tape` | 3–27 / 1,125–10,125 Hz | 35 | 6,192 / 6,224 | 7.712082 | 2,940 |
| `tape-fast` | 3–27 / 1,125–10,125 Hz | 15 | 3,312 / 3,344 | 14.354067 | 1,260 |
| `narrow` | 3–21 / 1,125–7,875 Hz | 44 | 7,776 / 7,808 | 6.147541 | 2,640 |
| `lofi` | 3–10 / 1,125–3,750 Hz | 24 | 6,912 / 6,944 | 6.912442 | 384 |

These are directly derived from `PRESETS`, `Layout.capacity`, `Layout.packet`,
`Layout.frame`, `Layout.fps`, and `Layout.band`. `[H99:animation_modem/transport2.py:63-124]`, `[H99:animation_modem/transport2.py:127-138]`

V2's sync waveform is a 256-sample random FFT waveform using seed 41015 and
peak 0.65. It searches 29 logarithmically spaced scales from 0.5 through 2.0.
`[H99:animation_modem/transport2.py:33-47]`

The phase table uses seed `52001 + top_bin*97 + image_symbols`. Progressive
layouts use pilots `[3,5,21,45]`; non-progressive layouts choose four points
inside their carrier range. Header carriers are the lowest 20 data carriers.
`[H99:animation_modem/transport2.py:63-85]`, `[H99:animation_modem/transport2.py:141-144]`

The V2 header is `>2sBBIHHI` plus CRC-32: magic, flags, top_bin, absolute,
index, count, and timestamp. It has no profile bits; profile and allocation
are static endpoint configuration. Progressive layouts write magic `V3`,
non-progressive layouts magic `V2`. Header gain is 1.6 and image gain is 0.7.
`[H99:animation_modem/transport2.py:282-342]`

V2 converts images through Pillow YCbCr, pads the source to the luma wire
shape with Lanczos, and BOX-resizes each plane to its configured shape.
`[H99:animation_modem/imaging.py:15-29]`, `[H99:animation_modem/imaging.py:63-72]`

Its `SourceCoder` applies an orthonormal 2-D DCT once per plane. Allocation is
`1/(1+12*sqrt(fy^2+fx^2))`; transmit gain is normalized square-root allocation
and decode is reliability-weighted Wiener reconstruction. `[H99:animation_modem/transport2.py:147-204]`

Progressive V2 slot mapping ranks all planes by normalized spatial frequency and
places low ranks in carrier-major order, so coarse coefficients use lower
carriers. Non-progressive layouts use sequential slots. V2 has no copies or
FEC. `[H99:animation_modem/transport2.py:207-229]`

The decoder uses the coarse correlation bank, sinc resampling, two training
symbols for a 2x2 equalizer, per-symbol pilot phase fitting, channel-combined
header with single-channel fallbacks, CRC identity, and Wiener reconstruction.
It reports a picture-only result when the header is lost but coherence and
coverage remain usable. `[H99:animation_modem/transport2.py:388-450]`, `[H99:animation_modem/transport2.py:612-796]`

## 7. Current V3 wire and `wire-hd`

### Source preparation and color

Current source preparation first converts to RGB, resizes the source to the
80x96 sampling canvas with the selected filter (default Lanczos), and records
an aspect code. Coefficient preparation resizes that image to the coder's
sampling grid with Lanczos, converts to Pillow YCbCr, BOX-resizes each plane,
and maps uint8 values to `[-1,1]`. Reconstruction maps back to uint8 and
upsamples Cb/Cr with bilinear interpolation. `[C:animation_modem/imaging.py:144-149]`, `[C:animation_modem/imaging.py:173-198]`

The exact RGB↔YCbCr matrix, chroma offsets, and range convention are supplied
by Pillow's `Image.convert('YCbCr')`; this repository does not define those
coefficients. They are therefore **UNVERIFIED at the wire-spec level** and a
reimplementation must pin or reproduce the Pillow conversion used by the
vector environment. `[C:animation_modem/imaging.py:173-181]`

For `color-dct` and `color-wavelet`, the source grid is 80x96 luma plus 40x48
Cb/Cr; the transmitted shape is 40x48 luma plus 24x20 Cb/Cr, 2,880 values.
`[C:animation_modem/imaging.py:23-33]`, `[C:animation_modem/imaging.py:44-57]`

`color-dct` uses `SourceCoder`: one orthonormal 2-D DCT per plane, frequency
allocation from the normalized spatial-frequency table, and normalized
square-root gains. `[C:animation_modem/core.py:293-360]`, `[C:animation_modem/core.py:370-436]`

The exact DCT allocation is
`sigma(i)=1/(1+12*sqrt(fy(i)^2+fx(i)^2))`; the transmitted gain is
`sqrt(sigma/mean(sigma))`, renormalized to unit mean square. `[C:animation_modem/core.py:277-360]`

`color-wavelet` uses a one-level masked 2-D Haar transform. This is not the
V5 CDF 9/7 transform. `[C:animation_modem/wavelet.py:483-498]`

The Haar masks are fixed source arrays, separately for luma and chroma; a
reimplementation must copy those arrays rather than infer a new top-N policy.
`[C:animation_modem/wavelet.py:21-334]`, `[C:animation_modem/wavelet.py:336-498]`

### Current V3 wire

The current `wire` layout is progressive, orthogonal-training, spread-carrier,
dense-header, top bin 54, 14 image symbols, and 3,280 capacity values. Its
magic is `V4` because dense layouts need a distinct magic even though the
transport implementation is the V3-family transport. `[C:animation_modem/transport3.py:49-68]`, `[C:animation_modem/core.py:203-224]`

The active profile code is in the packed top-bin byte: 0 for `color-dct`, 1
for `color-wavelet`. `[C:animation_modem/core.py:732-766]`

### `wire-hd`

`wire-hd` keeps the same 375–20,250 Hz band and dense-header physical format,
but expands to 16 image symbols and 3,680 capacity values. `[C:animation_modem/transport3.py:70-76]`

Its active `hd-dwt` coder uses a two-level CDF 9/7 lifting transform on the
80x96 / 40x48 grids. It keeps the half-resolution pyramid: 1,920 luma plus
480 Cb plus 480 Cr originals, then sends 800 copies, filling all 3,680 slots.
`[C:animation_modem/wavelet.py:705-767]`

The retained coefficient order is the packed CDF pyramid order: the coarsest
LL and detail bands are kept to a band boundary. Copies start with all LL
coefficients, then strongest detail values by measured RMS. `[C:animation_modem/wavelet.py:708-762]`

The CDF gains use the fixed `HD_BAND_RMS` table; the reliability inverse uses
per-slot equalizer weight/noise and a luma-first/chroma-strict recovery gate.
`[C:animation_modem/wavelet.py:686-700]`, `[C:animation_modem/wavelet.py:768-833]`

The fixed RMS rows are, in `(LL,LH,HL,HH)` order: Y
`(1.80,.457,.457,.303)`, Cb `(0.871,.093,.093,.058)`, and Cr
`(0.891,.135,.135,.085)`. `[C:animation_modem/wavelet.py:694-698]`

The CDF 9/7 lifting constants are alpha `-1.586134342059924`, beta
`-0.052980118572961`, gamma `0.882911075530934`, delta `0.443506852043971`,
and scale K `1.230174104914001`; the boundary extension mirrors the last/first
sample rather than wrapping the opposite image edge. `[C:animation_modem/wavelet.py:502-538]`, `[C:animation_modem/wavelet.py:540-591]`

## 8. V5 active wire

V5 is not a separate packet geometry in the active path. `V5Engine` selects
`WIRE_HD`, declares profile code 2, and uses the ordinary V3 `encode()` and
`Receiver`. `[C:animation_modem/engines.py:129-170]`

Therefore active V5 has the `wire-hd` sample timing, V3 16-bit header, V3
16-bit biphase preamble, orthogonal training, four pilots, V4 magic, and
pulse-counted V3 receiver. The source coder is the CDF 9/7 `hd_dwt_coder`
described above. `[C:animation_modem/engines.py:40-45]`, `[C:animation_modem/transport3.py:133-242]`

There is also a separate `PREAMBLE_V5`, `HEADER_FORMAT_V5`, and
`transport3_v5.ReceiverV5` implementation: it uses a 12-bit biphase preamble,
18-byte V5 header, mono OFDM, and LDPC coefficient decoding. That path is
**PARKED / NOT ACTIVE V5** in the engine registry and must not be mistaken for
the wire-hd V5 format. `[C:animation_modem/core.py:59-62]`, `[C:animation_modem/core.py:765-786]`, `[C:animation_modem/transport3.py:110-131]`, `[C:animation_modem/transport3_v5.py:1-18]`, `[C:animation_modem/engines.py:129-167]`

## 9. Tape wires and placement

### `wire-tape`

`wire-tape` moves the complete active payload below 12.75 kHz, declares a
14 kHz whole-waveform ceiling, and uses 12 image symbols. Its `hd-dwt` name is
selected specially by `engines.coder_for()`: the actual coder is
`tape_80x96_coder()`, a DCT `StereoRepeatCoder`, not the CDF 9/7 `hd_dwt_coder`.
It has 800 originals and 800 copies, 1,600 transmitted values, 80x96/48x40
source grids, and 16.483516 fps. `[C:animation_modem/transport3.py:78-84]`, `[C:animation_modem/engines.py:34-49]`, `[C:animation_modem/wavelet.py:885-968]`

The DCT tape coder's originals are 28x24 luma and 8x8 Cb/Cr, giving 800
coefficients; the 80x96 output remains a reconstructed grid, not 80x96
transmitted detail. `[C:animation_modem/wavelet.py:964-968]`, `[C:animation_modem/imaging.py:27-33]`

### `wire-tape-25`

`wire-tape-25` has five image symbols, 760 capacity, and 25.210084 fps. Its
`tape-80x60` coder transmits 360 originals and 360 copies from 80x60 / 40x30
sampling grids, using 15x20 luma and 5x6 Cb/Cr wire shapes. `[C:animation_modem/transport3.py:86-92]`, `[C:animation_modem/wavelet.py:959-962]`, `[C:animation_modem/imaging.py:31-33]`

### Stereo-repeat placement algorithm

`StereoRepeatCoder.slots()` partitions available slots by channel, alternates
original homes across the two channels, then solves each copy assignment with
a global linear assignment. The cost penalizes same-channel and too-close
frequency placements and prefers `TAPE_CENTRE_BIN=10`; `MIN_COPY_SPREAD=8` for
the DCT stereo-repeat coder. `[C:animation_modem/wavelet.py:885-956]`, `[C:animation_modem/wavelet.py:699-702]`

The current tests require opposite-channel and frequency-diverse copies for
both tape layouts, and clean decoding at 48 and 96 kHz. `[C:modem_tests/test_tape_wire.py:21-120]`

## 10. V6 and V6-repeat

### V6 coder geometry

V6 uses source grids `(96,80), (48,40), (48,40)` and wire shapes
`(48,40), (24,20), (24,20)`: 11,520 source values and 2,880 analog originals.
The DCT foundation consists of 24x20 luma and 12x10 Cb/Cr low-frequency
corners, 720 values total. `[C:animation_modem/v6.py:19-41]`

The V6 DCT coder applies one DCT through `SourceCoder`; the wavelet coder uses
two-level CDF 9/7 with the deepest LL bands as the foundation. Both transforms
are wrapped by `ProtectedAnalogCoder`. `[C:animation_modem/v6.py:189-201]`

Forward coding removes base gains, appends selected foundation values, then
applies a combined gain table. Decode either averages clean copies or performs
reliability/noise-weighted soft fusion, then applies confidence floors:
protected luma/chroma floors are 0.05/0.15 and unprotected luma/chroma floors
are 0.45/0.60. `[C:animation_modem/v6.py:101-148]`

Full-repeat sets `copy_of=np.arange(ORIGINAL_VALUES)`, so every coefficient is
marked protected. Consequently V6-repeat uses the protected floors for every
luma/chroma coefficient: 0.05 for luma and 0.15 for both chroma planes; the
0.45/0.60 unprotected floors apply only to foundation-only V6 coefficients
outside its 720-value foundation. `[C:animation_modem/v6.py:115-119]`,
`[C:animation_modem/v6.py:204-216]`

For reliability mode, each observation uses `a=gain*reliability`,
`precision=a^2/noise`, posterior coefficient
`variance*sum(a*y/noise)/(1+variance*sum(a^2/noise))`, and confidence
`variance*sum(a^2/noise)/(1+variance*sum(a^2/noise))`. The gate is
`clip((confidence-floor)/(.85-floor), 0, 1)`. `[C:animation_modem/v6.py:131-148]`

### V6 tape placement

For both V6 transforms, originals are assigned in coefficient-rank order to
the first 2,880 slots from `_slot_order()`. Each foundation copy is assigned to
the opposite channel, at least eight carrier bins away, and preferably a
different OFDM symbol; a SciPy linear-assignment solve chooses the mapping.
`[C:animation_modem/v6.py:150-186]`

The V6 wire is top bin 34, 29 image symbols, 3,640 capacity, 3,600 used
values, and magic `V6`. `[C:animation_modem/transport3.py:94-100]`

The V6 full-repeat control keeps the same band and ceiling but expands to 47
image symbols and 5,800 capacity. It copies all 2,880 originals, uses 5,760
values, and identifies itself with magic `VR`. `[C:animation_modem/transport3.py:102-108]`, `[C:animation_modem/v6.py:204-216]`

### V6 recovery behavior

`ProtectedAnalogCoder.inverse()` combines the original/copy observations using
equalizer reliability and noise variance, then applies transform-specific
inverse reconstruction. A failed or low-confidence coefficient is softened
instead of being promoted to false detail or color. `[C:animation_modem/v6.py:125-148]`

The V6 tests require copies to use the opposite channel, separated frequency,
and different symbol; they also test either-track loss and the 14 kHz ceiling.
`[C:modem_tests/test_v6.py:29-83]`, `[C:modem_tests/test_v6_repeat.py:25-63]`

These are diversity controls, not tape acceptance evidence. The project note
records that full-repeat improved some isolated-track and severe synthetic
cases but regressed DCT MP3, wow/flutter, hiss, and several wavelet low-pass
cases; real tape remains required. `[C:MODEM_TODO.md:139-152]`

## 11. Current decoder implementation

### Equalizer and payload extraction

The active decoder FFTs each 128-sample body symbol, estimates a per-carrier
2x2 channel from the two orthogonal training rows, regularizes with energy in
unused bins, and computes equalized symbols, reliability weights, and noise
variance. `[C:animation_modem/core.py:883-963]`

Pilot phases are unwrapped and fit as a weighted frequency slope/intercept per
channel and body symbol. The pilot error is used for result quality and for
single-input retry. `[C:animation_modem/core.py:1342-1392]`

The payload is gathered in the layout/coder slot order, divided by image gain,
and passed to the selected source coder. If a header verifies and `coders` is
provided, the profile code selects the matching coder; otherwise the caller's
fallback coder is used. `[C:animation_modem/core.py:1464-1521]`

### Acquisition and retries

After pulse acquisition, the receiver predicts the next packet from the
measured frame length and coasts when the local pulse word agrees. It decodes
the joint stereo signal first; if the header/pilot result is not sufficiently
good, it retries each input alone and keeps the best verified/recoverable
candidate. `[C:animation_modem/transport3.py:819-849]`, `[C:animation_modem/core.py:1218-1280]`

The single-input retry threshold is pilot error 0.20; erased equalizer weights
below 0.05 are assigned effectively infinite noise. `[C:animation_modem/core.py:1210-1215]`

If no header verifies but coverage/coherence remain usable, the decoder may
return `picture_only`; the receiver retains the last detected layout and aspect
for damaged-header presentation. `[C:animation_modem/core.py:1453-1554]`, `[C:animation_modem/transport3.py:920-929]`

## 12. Deterministic specification vectors

Generate vectors with:

```bash
.venv/bin/python tools/spec_vectors.py
```

The script uses seed `20260920`; current wires receive successive seeds
20260921 onward. Every vector is a one-packet 48 kHz stereo PCM16 WAV and is
decoded immediately with the matching implementation. For V1/V2, the script
uses `git show` to extract the pinned historical source files, so it does not
silently substitute the current transport. `[C:tools/spec_vectors.py:1-237]`

The recorded environment for this document was Python **3.13.5**, NumPy
**2.2.4**, and SciPy **1.18.1**. These are an execution pin, not a repository
dependency pin: `requirements-modem.txt` currently specifies minimum NumPy and
SciPy versions only. `[C:requirements-modem.txt:1-6]`, `[C:tools/spec_vectors.py:1-25]`

| Vector | File | SHA-256 | identity | measured clean RMSE | expected tolerance |
|---|---|---|---|---:|---:|
| V1 | `v1.wav` | `5e5528c6134d509f9b16b4edf82c25c742dece2bbd6bfbe1dadd840cdd44ae77` | verified_header | 0.100042 | 0.11 |
| V2 | `v2-99d60665.wav` | `b582b6e61b8e6ed4bd33e5481e59bf3ad76355a871c4022e439bc5b8e9351903` | verified_header | 2.28e-8 | 0.001 |
| V3 `wire` | `v3.wav` | `c01f1376e1b547dba378027afcfa8535ad753cb508283e38533697d430936c6b` | verified_header | 3.12e-8 | 0.01 |
| `wire-hd` | `wire-hd.wav` | `7e478cbed49183712599f923309b14a0be950a6be8033c6065cb5315a74551cd` | verified_header | 5.77e-8 | 0.01 |
| V5 | `v5.wav` | `c8beca14ad771cd87c5884a86f9c61fac8ed2774926e84ae8544dd25c4905646` | verified_header | 5.51e-8 | 0.01 |
| `wire-tape` | `wire-tape.wav` | `34a0ed6ea420687cc1557908d1b14bca650390c3f0eae9d5c99e96b321838d01` | verified_header | 0.001406 | 0.07 |
| `wire-tape-25` | `wire-tape-25.wav` | `b78f3c5ff1a0306158452edf2b71ba26a8475db78cd669179687bcc6f6efebb9` | verified_header | 0.001059 | 0.08 |
| V6 | `v6.wav` | `821cdf8e509838e4d9fe5ad6e4d2d71c6019d4b0f820536bee97997cc787a9aa` | verified_header | 0.004078 | 0.55 |
| V6-repeat | `v6-repeat.wav` | `f8424ed6b95a337b5039cfe01821aa09fec07216706afa658d613a33ba37a1e0` | verified_header | 0.002618 | 0.60 |
| V6 tape placement | `v6-tape-placement.wav` | `828d74764bf20b9dee5ec552d5ed3ef46130af49c05043c785016a5253e9bde7` | verified_header | 0.003557 | 0.55 |

The tolerance values are clean synthetic acceptance thresholds: the V1 vector
uses a normalized RGB-pixel threshold of 0.11 `[C:tools/spec_vectors.py:149-173]`;
historical V2's clean packet
test uses 1e-3; V3/HD uses 0.01; tape tests use 0.07/0.08; and V6 random-
coefficient tests use 0.55/0.60. `[H99:modem_tests/test_transport2.py:36-40]`, `[C:modem_tests/test_transport3.py:95-106]`, `[C:modem_tests/test_tape_wire.py:38-100]`, `[C:modem_tests/test_v6.py:42-52]`, `[C:modem_tests/test_v6_repeat.py:47-63]`

## 13. One V6 DCT packet, end to end

1. **Prepare source.** Convert the source to RGB, resize to 80x96, convert to
   YCbCr, and sample 96x80 luma plus 48x40 Cb/Cr values. `[C:animation_modem/imaging.py:144-149]`, `[C:animation_modem/imaging.py:173-181]`
2. **Transform.** `SourceCoder` applies one orthonormal DCT to each sampling
   plane and retains the 48x40 / 24x20 low-frequency corner. `[C:animation_modem/core.py:293-380]`, `[C:animation_modem/v6.py:189-192]`
3. **Protect.** Select the 720 indices in the 24x20 / 12x10 foundation,
   append their analog values, and apply combined source/copy gains. `[C:animation_modem/v6.py:34-41]`, `[C:animation_modem/v6.py:101-123]`
4. **Place.** Rank originals by normalized spatial frequency, put them into
   `wire-v6` slots, and solve the opposite-channel, separated-frequency,
   different-symbol copy assignment. `[C:animation_modem/v6.py:44-49]`, `[C:animation_modem/v6.py:150-186]`
5. **Build header.** Pack magic `V6`, flags, top bin 34/profile code,
   aspect-packed absolute, one-based index/count, timestamp, then CRC-32.
   For the deterministic example `magic=V6`, `flags=0`, `top_bin=34`,
   `profile=0`, `absolute=1`, `index=1`, `count=1`, and `stamp_ms=0`, the
   16-byte payload is
   `56360022000000010001000100000000`, the CRC is `c660570f`, and the complete
   20-byte header is
   `56360022000000010001000100000000c660570f`.
   The corresponding V6-wavelet header changes byte 3 to `0x62` and has CRC
   `7bedc6a4`. V6-repeat uses magic `VR`: profile 0 gives CRC `0e22778f`,
   and profile 1 gives CRC `b3afe624`.
   `[C:animation_modem/core.py:794-806]`, `[C:animation_modem/core.py:215-221]`
6. **Build OFDM.** Write orthogonal training rows, four unit pilots, header
   QPSK, and image I/Q values with gain 0.7 into 35 symbols. `[C:animation_modem/transport3.py:495-534]`
7. **IFFT and framing.** Apply deterministic per-symbol/channel phase, IFFT
   128 samples, prepend the 16-sample CP to each symbol, prepend the 288-sample
   sync region, append 32 guard samples, peak-normalize the OFDM body to 0.95,
   and apply the 14 kHz emission bound. `[C:animation_modem/transport3.py:536-550]`, `[C:animation_modem/core.py:643-683]`
8. **Acquire.** The receiver edge-counts the biphase preamble on both tracks,
   chooses the pulse word closest to the expected scale, and predicts the next
   5,360-sample frame. `[C:animation_modem/transport3.py:708-760]`, `[C:animation_modem/transport3.py:963-969]`
9. **Demodulate.** Remove CP, FFT each body symbol, solve the per-carrier 2x2
   equalizer, track pilot phase, decode/tolerate the header CRC, and gather
   payload slots. `[C:animation_modem/core.py:883-981]`, `[C:animation_modem/core.py:1076-1099]`, `[C:animation_modem/core.py:1464-1521]`
10. **Reconstruct.** Fuse original/copy observations with reliability and
    noise variance, apply confidence floors, inverse-DCT the retained corner,
    and return a verified 80x96 reconstructed value vector. `[C:animation_modem/v6.py:125-148]`

## 14. Conformance and implementation rules

### Normative language and platform policy

An interoperable implementation **MUST** reproduce the cited sample geometry,
header serialization, phase generation, slot mapping, and decoder acceptance
rules. It **MAY** use a different programming language or FFT library only if
the resulting vectors and decode outcomes remain within the stated tolerances.
“UNVERIFIED” means that this checkout does not establish the behavior in code;
it is not permission to invent a compatible behavior. `[C:tools/spec_vectors.py:1-25]`

Operating system is intentionally **non-normative**. The project has used
Linux, Windows, and macOS. Pip version and resolver history are also
non-normative and were not pinned; only the recorded Python/NumPy/SciPy
execution versions are a reproducibility reference for the supplied vectors.
`[C:requirements-modem.txt:1-6]`

### Exact sample and array conventions

All current wire arrays are shaped `(samples, 2)` with channel 0 then channel
1. Each complex OFDM carrier stores two real values: element 0 is the real/I
component and element 1 is the imaginary/Q component. The encoder writes the
four-value cell ordering `(channel0-I, channel0-Q, channel1-I, channel1-Q)`;
the decoder reshapes and gathers the same order. `[C:animation_modem/transport3.py:524-534]`, `[C:animation_modem/core.py:1476-1521]`

The body starts at `SYNC_LEN`; each symbol is exactly `CP` samples followed by
`N` useful samples. The decoder intentionally samples from `CP-4` through
`CP-4+N`, not from an arbitrary symbol boundary. `[C:animation_modem/core.py:1229-1235]`

### Numeric, clipping, and PCM rules

The normative current float path uses finite NumPy floating-point arrays. The
encoder rejects non-finite source values and normalizes the OFDM body to the
requested headroom before optional emission filtering. `[C:animation_modem/transport3.py:482-550]`

PCM output is signed little-endian 16-bit: clip to `[-1,1]`, multiply by
32767, round with NumPy's `rint`, cast to `<i2`, and write interleaved stereo
bytes. `[C:animation_modem/audio_common.py:29-30]`

The historical V1/V2 writers use the same PCM16 little-endian convention in
their audio helper, but their older float normalization paths differ and must
be taken from their cited commits. `[H99:animation_modem/audio_common.py:29-38]`

### Channel and level semantics

The current preamble is transmitted identically on both channels. Input
levelling is per-channel because channel amplitude is not payload information;
relative phase and channel skew remain meaningful. `[C:animation_modem/audio_common.py:33-58]`
If joint equalization fails, the current decoder retries each receive input
alone; this is a recovery path, not a second wire format. `[C:animation_modem/core.py:1218-1280]`

### Current candidate order and compatibility

When the utility receiver is given the complete candidate set, the receiver
sorts candidates by packet length. The exact current order is:

```text
wire-tape-25 (1872), wire-tape (2880), wire (3168),
wire-hd (3456), wire-v6 (5328), wire-v6-repeat (7920)
```

The utility constructs these candidates and the receiver performs the sort.
`[C:utilities/modem_v3_check.py:96-128]`, `[C:animation_modem/transport3.py:597-600]`

V1 and V2 are not backward-compatible with current V3-family receivers. V2
requires a caller-selected preset/profile/allocation, while current V3-family
candidate detection requires the layout magic, top-bin, profile code, and CRC
to agree. `[H99:animation_modem/transport2.py:612-633]`, `[C:animation_modem/core.py:1050-1066]`

### Status and loss semantics

`verified_header` means the complete header passed CRC and structural checks;
`picture_only` means payload usability passed without verified identity;
`lost` means no usable picture was materialized. `received` and `degraded` are
quality statuses attached to a decoded result. `[C:animation_modem/core.py:1523-1554]`

The receiver preserves the last detected layout/aspect across a damaged header,
and the application decides whether to show a recoverable damaged picture or
hold the previous one. Black-frame fallback is not a modem-wire recovery
primitive. `[C:animation_modem/transport3.py:920-929]`, `[C:MODEM_TODO.md:27-31]`

### Physical-path and performance boundaries

Synthetic vectors establish clean decoding, not tape acceptance. A physical
test must record the sample rate, level, channel pair, deck/tape settings,
noise-reduction state, and impairment timing. `[C:MODEM_TODO.md:94-106]`, `[C:MODEM_TODO.md:154-187]`

The current live decoder keeps bounded capture storage, performs fixed-size
128-point FFT demodulation, and records acquisition/decode CPU time in the
diagnostics. The specification does not claim a universal CPU deadline for
machines not covered by the recorded environment; that margin remains an
acceptance measurement. `[C:animation_modem/transport3.py:621-645]`, `[C:animation_modem/transport3.py:931-947]`

### Conformance test set

A reimplementation should run, at minimum, the seeded clean vectors, arbitrary
input chunk sizes, exact packet EOF without guard, 48/96 kHz tape vectors,
off-speed playback, damaged-header tolerance, single-input retries, profile
mismatch rejection, emission-ceiling checks, and V6 copy-placement checks.
Existing tests cover these cases in `[C:modem_tests/test_transport3.py:80-150]`,
`[C:modem_tests/test_sample_rate_independence.py:50-270]`,
`[C:modem_tests/test_tape_wire.py:38-120]`,
`[C:modem_tests/test_v6.py:42-105]`, and
`[C:modem_tests/test_v6_repeat.py:25-63]`.

The vector manifest records source seed, source/pinned commit, output hash,
identity, measured clean RMSE, and tolerance. `[C:tools/spec_vectors.py:149-237]`

## 15. Number audit and discrepancies

The following checks were performed against the cited code and generated
vectors:

| Check | Result |
|---|---|
| Packet/frame/fps arithmetic for all current layouts | Matches `Layout` properties and vector manifest. |
| Current V3 dense magic | The implementation emits `V4`, not literal `V3`; this document records `wire` as the V3-family wire and `V4` as its current magic. `[C:animation_modem/core.py:203-224]` |
| V5 preamble/header claims | The active V5 path uses the standard V3 preamble/header; the separate V5 preamble/header/LDPC implementation is parked. `[C:animation_modem/engines.py:129-167]` |
| `wire-tape` transform name | The profile is named `hd-dwt` at the CLI but layout-specific construction selects DCT `StereoRepeatCoder`; this document records the implementation, not the misleading profile label. `[C:animation_modem/engines.py:34-49]` |
| Tape/V6 whole-waveform ceiling | Only tape/V6 layouts set `emission_ceiling=14000`; wide current layouts have no whole-waveform ceiling. `[C:animation_modem/transport3.py:66-108]` |
| V6 placement | V6 copies are opposite-channel, frequency-separated, and symbol-separated by the assignment cost; the vector script records these invariants rather than claiming independent tape-track failure. `[C:animation_modem/v6.py:169-182]` |
| Historical V1/V2 vectors | Regenerated from the exact pinned historical source files using `git show`; they are not decoded by current transport code. `[C:tools/spec_vectors.py:133-204]` |
| CRC algorithm | The call, seed behavior as exposed by the code, and big-endian output are confirmed; the underlying zlib polynomial/reflection details are library-defined and therefore explicitly UNVERIFIED from repository code. `[C:animation_modem/core.py:768-806]` |
| Color matrix | Pillow YCbCr conversion is confirmed; numeric matrix/range coefficients are not defined in this repository and remain UNVERIFIED at wire-spec level. `[C:animation_modem/imaging.py:173-181]` |
| Python/NumPy/SciPy pin | Observed execution was Python 3.13.5 / NumPy 2.2.4 / SciPy 1.18.1, but requirements only specify minimum NumPy/SciPy versions. This is an environment discrepancy requiring packaging work if bit-for-bit environment pinning is required. `[C:requirements-modem.txt:1-6]` |
| OS and pip policy | OS and pip versions are intentionally non-normative; Linux, Windows, and macOS are supported by project history, and pip resolver history was not pinned. `[C:requirements-modem.txt:1-6]` |
| Generic `tools/decode_wav.py` receiver coverage | It does not include V6 candidates even though the utility receiver does; this is a tooling discrepancy, not a wire-format discrepancy. `[C:tools/decode_wav.py:74-86]`, `[C:utilities/modem_v3_check.py:96-128]` |
| State-machine attachment | Per-version encode/decode and V6 placement state machines are attached in `docs/transport_state_machines.md`; they are descriptive expansions of the cited implementation, not a second executable state machine. `[C:docs/transport_state_machines.md:1-263]` |

No claim in this document promotes synthetic vectors or the V6 controls to
real-tape acceptance. Real tape captures, measured levels, track behavior,
dropout shape, and matched decoder provenance remain required by the project
acceptance gate. `[C:MODEM_TODO.md:154-187]`
