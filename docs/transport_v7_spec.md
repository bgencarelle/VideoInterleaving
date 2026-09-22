# V7 transport specification — DRAFT

**Status:** proposal with a bench-only prototype (`tools/v7_proto.py`,
exercised by `tools/v7_bench.py`) and an experimental live path
(`tools/v7_live.py`); nothing in `animation_modem/` implements it.
§19 is the current live-wire amendment; the earlier continuous-clock sections
remain as a design comparison.
already corrected. It is written against `docs/transport_timing_and_encoding.md`
at `c6ef8192` (cited as `[SPEC §n]`) and uses the same conventions:
48 kHz reference geometry, `(samples, 2)` arrays, channel 0 then 1, I then Q.
Backward compatibility is explicitly **not** a goal; no V1–V6 receiver can
decode V7 and V7 does not try to decode them.

Evidence tags: **[MEASURED]** means a number I computed for this draft
(script in Appendix A); **[BENCH]** means `tools/bench_v6_tape_lift.py`
results recorded in `MODEM_TODO.md`; **[PROTO]** means measured with the
V7 prototype (§17); **[DESIGN]** means an untested choice that §13 says how to
test; **[LIT]** means an external source (§16).

---

## 1. Requirements

| # | Requirement | Source |
|---|---|---|
| R1 | Frame rate **12–15 fps** at every supported medium. | user |
| R2 | Survives low-quality audio media: ferric cassette, worn decks, MP3/AAC at low bitrates, mono-summing playback, telephone-band paths. Real tape is authoritative. | user, `MODEM_TODO.md` |
| R3 | Degrade detail before structure, and structure before colour identity. | `MODEM_TODO.md` acceptance gate |
| R4 | Never black; show damaged frames, hold the last good frame on failure. | `AGENTS.md` |
| R5 | Timing is edge/pulse-counted (LTC-style), not FFT correlation. | `AGENTS.md` |
| R6 | Cheap, deterministic decode: fixed small FFTs, small fixed-size linear solves. | `AGENTS.md` |
| R7 | Keep the baked 80×96 image target and V6 source grids. | `MODEM_TODO.md` |

---

## 2. What the existing versions got right and wrong

Every row is a mechanism that appears in V1–V6 `[SPEC §2–§11]`, with a verdict.

| Mechanism | Where used | Verdict for V7 | Why |
|---|---|---|---|
| OFDM, N=128, CP=16, 375 Hz bins | V1–V6 | **Keep** | Cheap, proven; 3 ms symbols tolerate flutter-induced ICI. |
| Analog (uncoded) coefficient I/Q | V1–V6 | **Keep** | Graceful degradation; the digital detour failed `[MODEM_TODO]`. |
| Per-frame biphase pulse preamble (288 samples) | V3–V6 | **Replace** with a continuous clock track (§5) | Preamble + training + header cost 1,184 of V6's 5,360 samples (22%) and give timing only once per frame. |
| Two training symbols per frame | V2–V6 | **Replace** with scattered + continual pilots (§6.3) | Channel is estimated once, then drifts across the packet; the longer the packet, the worse (the full-repeat wow/flutter regression). |
| 20-byte QPSK header in 4 header symbols | V3–V6 | **Replace** with a 72-bit word inside the clock track | Identity rides the most robust band (0.5–1 kHz) and survives the loss of all OFDM carriers. |
| Per-packet peak normalisation to 0.95 | V3–V6 `[SPEC §13 step 7]` | **Replace** with a fixed RMS level | Recording level moves frame to frame with content, which pumps tape NR and makes the noise floor content-dependent. |
| 2×2 MMSE equaliser with orthogonal `[1,1]`/`[1,−1]` training | V3–V6 | **Keep and promote** to M/S precoding (§6.5) | The training rows are already the mid/side basis. Using M/S for data makes mono playback and parametric-stereo codecs keep the important stream. |
| Allocation `sigma = 1/(1+12r)`, gain `sqrt(sigma)` | V2–V6 `[SPEC §7]` | **Replace** (§7.3) | [MEASURED] 3.6–8.6 dB worse than the analog optimum at 10–30 dB channel SNR, and worse than flat gains. It puts essentially all power (≥99.5%) on 720 coefficients. The model also fits real images poorly: measured σ ≈ model^2.1, r = 0.71. |
| Middle-out slot order, header spares first | V3–V6 | **Drop** | Put the 100 most important values at 9–12.4 kHz `[BENCH, MODEM_TODO]`. |
| Low-carrier-first, time-interleaved placement | tape-ordered V6 | **Keep** as the backbone of §8 | [BENCH] roughly halved picture error under shared lifts and hiss. |
| Full cross-track copies of the foundation | V6, V6-repeat | **Replace** with mono cells (§8.2) | Copies cost 720 slots. [BENCH] they do nothing under shared lifts; they only help track loss. Mono cells give the head tier the same cross-track protection at a third of the cost. |
| Confidence-gated LMMSE reconstruction | V6 | **Keep**, generalised to groups (§9.6) | Correct tool for analog data with known reliabilities. |
| Hold last good frame, no black | all | **Keep** | R4. |
| 14 kHz whole-waveform ceiling, gentle FIR | tape, V6 | **Keep** | Verified tape-safe; the new signal is bounded the same way. |
| CDF 9/7 wavelet option | V5, V6 | **Drop from the wire** (DCT only) | One transform halves test surface. DCT with SoftCast allocation is the studied analog-video optimum [LIT: SoftCast]. |
| LDPC / digital payload | parked V5 | **Drop** | Cliff effect and bottleneck `[MODEM_TODO]`. |

---

## 3. Evidence gathered for this draft

All [MEASURED] numbers come from Appendix A. They are channel models, not
full modem runs; §13 lists the modem-level confirmations still required.

### 3.1 Gain law (the largest single win)

Coefficient statistics: 120 random crops (scale 0.45–1.0) and flips of the
supplied test image `modem_tests/fixtures/v7_reference_face.png` (SHA-256
`f44aa3c3…257b`), run through the V6 DCT source path. All [MEASURED] image
statistics in this draft use that fixture only.
Metric: coefficient-domain reconstruction SNR under AWGN at equal mean
transmit power, LMMSE decoding.

| Channel SNR | Current law | SoftCast law, V2 model table | SoftCast law, measured table | Flat |
|---:|---:|---:|---:|---:|
| 10 dB | 11.3 | 13.7 | 14.9 | 12.4 |
| 15 dB | 12.3 | 15.4 | 17.1 | 13.7 |
| 20 dB | 13.3 | 17.2 | 19.4 | 15.1 |
| 25 dB | 14.4 | 19.1 | 21.7 | 16.6 |
| 30 dB | 15.6 | 21.1 | 24.2 | 18.2 |

The current code comment says "power proportional to standard deviation
is the analog optimum" (`core.py`, `SourceCoder.__init__`), but applies
`sqrt(sigma)` as an **amplitude** gain. Transmit power therefore goes as σ³.
SoftCast's optimum is amplitude gain λ^(−1/4) = σ^(−1/2), i.e. transmit
power ∝ σ [LIT: SoftCast].

### 3.2 Coefficient variance model [MEASURED]

Per-plane fit of mean-square DCT coefficient versus normalised index radius
`r = hypot(ky/Gy, kx/Gx)` (G = sampling-grid size):

| Plane | DC mean-square | AC fit λ(r) | residual (nepers) |
|---|---:|---|---:|
| Y  | 78.2 | 1.01e3 / (1 + 50.6 r)^3.37 | 0.39 |
| Cb | 12.6 | 2.53 / (1 + 17.7 r)^3.80 | 0.49 |
| Cr | 48.5 | 2.90 / (1 + 10.6 r)^4.30 | 0.48 |

Energy by rank (all planes pooled, sorted by λ): ranks 3–720 hold 64.6%
of energy across a 31.4 dB range. Ranks 720–1440 hold 1.3% (5.0 dB range);
ranks 1440–2224 hold 0.46%; the tail, ranks ≥2224, holds 4×10⁻⁴. One source
image only: the normative table must be baked from the real library (§7.2).

### 3.3 Time spreading [MEASURED]

LMMSE error for 8 coefficients in 8 slots, plain versus 8-point Hadamard
mixing, **averaged over all erasure positions**:

| Variance range in group | 1 slot erased: plain → Hadamard | 2 erased | random partial fades |
|---:|---|---|---|
| 0 dB (equal) | identical | identical | identical |
| 10 dB | 0.42 → 0.25 | 0.83 → 0.54 | 0.185 → 0.180 |
| 20 dB | 0.27 → 0.047 | 0.52 → 0.13 | 0.091 → 0.071 |
| 30 dB | 0.21 → 0.014 | 0.40 → 0.041 | 0.056 → 0.030 |

Spreading helps only when a group mixes **different** variances: the small
coefficients act as analog parity for the large one. It is useless for
equal-variance groups. For a *known, monotone* fade across the group it is
slightly harmful. Consequences for §8: spread along **time** (bursts are
position-random), never along **frequency** (lifts and low-pass are
monotone and known); build groups from rank windows so each group spans
the local variance range.

### 3.4 Peak-to-average ratio [MEASURED]

Gaussian analog loading of bins 4–34: per-frame PAPR is 11.2 dB median and
12.9 dB at the 99th percentile. Iterated clip-and-filter confined to the
used bins:

| Clip level | p99 PAPR | In-band MER | Level gain |
|---:|---:|---:|---:|
| 6 dB | 7.0 dB | 16.8 dB | +5.9 dB |
| 7 dB | 7.8 dB | 20.1 dB | +5.1 dB |
| 8 dB | 8.7 dB | 24.0 dB | +4.2 dB |

The clipping noise is a fixed floor, so this is a tape-only, transmit-side
option (§6.7). It is worthless for lossy codecs, where absolute level does
not set the noise.

### 3.5 Medium facts used [LIT]

- **Ferric cassette:** about 30 Hz–16 kHz at −20 dB but only about 12 kHz at
  Dolby level. High-frequency MOL is about −8 dB at 10 kHz. SNR 54–60 dBA.
- **Tape dropouts:** Wallace spacing loss, 54.6·d/λ dB. [BENCH] reproduced
  it exactly. Re-run on the fixture (6 s, DCT, mean picture RMSE):
  tape-ordered V6 versus V6 is 0.032 vs 0.056 (hiss −45), 0.049 vs 0.080
  (severe lift), 0.067 vs 0.105 (worn contact), and 0.052 vs 0.061 on a clean
  channel against the source.
- **MP3 (LAME VBR):** low-pass 16.5–19.9 kHz depending on preset.
- **HE-AAC v2 (16–48 kbit/s):** downmixes stereo to mono plus 2–3 kbit/s of
  parametric side information. All independent L/R information is lost; only
  the sum survives.
- **LTC:** biphase-mark, 80 bits per frame, 960–2400 Hz, self-clocking.
  Designed for audio tape.
- **Pilot tone (Nagra):** the historical precedent for recovering tape speed
  from a recorded reference.
- **DVB-T scattered pilots** and **SoftCast:** the templates for §6.3 and §7.

---

## 4. Architecture

```text
            ┌──────────────────── one 72 ms frame (3,456 samples) ─────────────────────┐
  clock     │ biphase-mark, 72 bits @ 1000 bit/s, 500–1000 Hz, identical on L and R    │
  track     │ [sync 16][counter 20][profile 4][aspect 3][folders 8][rsv 4][CRC-16][pol] │
            ├──────────────────────────────────────────────────────────────────────────┤
  OFDM      │ 24 symbols × 144 samples, continuous, no gaps, no preamble, no training  │
  body      │ carriers 4–34 (1.5–12.75 kHz), scattered + continual pilots              │
            │ data cells carry M (mono-safe) and S (side) analog streams               │
            └──────────────────────────────────────────────────────────────────────────┘
  transmit = OFDM body + clock track   (sample-aligned; frame f starts at 3456·f)
```

The receiver pipeline:

1. Separate out the clock band.
2. Count edges, run a PLL and build a speed-corrected time map.
3. Read the identity word.
4. Optionally cancel the clock waveform from the OFDM band.
5. Sample each OFDM symbol at its time-mapped position and FFT it.
6. Track the channel and fades from the pilots.
7. Run 2×2 M/S equalisation.
8. Run group LMMSE reconstruction.
9. Fill the tail store.
10. Display.

---

## 5. Clock and identity track

### 5.1 Waveform

| Parameter | Value |
|---|---|
| Code | Biphase-mark (FM): a transition at every bit start, plus one mid-bit for a `1` |
| Bit period | 48 samples (1000 bit/s); half-bit 24 samples |
| Spectrum | fundamentals 500 Hz (`0`) and 1000 Hz (`1`) |
| Shaping | Unit-amplitude square wave filtered by a linear-phase low-pass: passband ≤ 1100 Hz, stopband ≥ 1400 Hz, ≥ 50 dB attenuation. Zero crossings, and hence pulse counts, are preserved. |
| Level | RMS = OFDM-body RMS − 3 dB [PROTO: −10 dB left only ~13 dB over OFDM sidelobe leakage; −6 dB failed under dbx-style companding] |
| Channels | Identical on both channels, i.e. it lives entirely in M |
| Continuity | Continuous across frames; frame f's bit 0 begins at sample 3456·f |

### 5.2 Word (72 bits per frame, MSB-first per field)

| Bits | Field | Notes |
|---|---|---|
| 0–15 | sync `0011 1111 1111 1101` | the LTC sync pattern [LIT] |
| 16–35 | frame counter, 20 bits | wraps after about 21 h at 13.89 fps; replaces `absolute`/`stamp_ms` |
| 36–39 | profile, 4 bits | selects variance table, grids, tail phases |
| 40–42 | aspect code, 3 bits | same codes as `[SPEC §4]` |
| 43–50 | folders, 8 bits | face high nibble, float low nibble |
| 51–54 | reserved, zero | |
| 55–70 | CRC-16/CCITT-FALSE over bits 16–54 | poly 0x1021, init 0xFFFF, no reflection, xorout 0; check value 0x29B1 [MEASURED]. The 39 payload bits are left-padded with one zero bit to 5 bytes. |
| 71 | polarity bit | chosen so the word has an even number of transitions, so every frame starts at the same level |

### 5.3 Receiver clock path (normative behaviour)

1. Band-select 300–1300 Hz on M = (L+R)/2. If M fails the CRC, retry L and R
   alone.
2. Detect edges with a Schmitt trigger. Hysteresis is 0.4 × the running
   clock envelope, with an absolute floor of 0.1 × its 90th percentile so
   silence produces no edges.
3. Interpolate crossings linearly. This generalises `measure_pulses`
   `[SPEC §3]` from a 16-bit word to a continuous stream.
4. Run a bit-clock loop that predicts each boundary, nudges its phase by
   0.25 × the error of the nearest edge within ±⅓ bit (period by 0.02 ×),
   and coasts when no edge is found. Three misses in eight boundaries mean
   it locked to mid-bits: shift half a bit. Snapping to every edge is
   wrong: the band-limited clock shifts edges by pattern (±2.5 samples RMS
   [PROTO]), and tape phase error shifts them further.
5. Decide each bit by the classic biphase rule: `1` if an edge falls in the
   middle half of the bit (¼–¾), else `0`. It tolerates ±¼-bit
   pattern-dependent edge shifts; polarity or tone-energy decisions failed
   under a 120 Hz high-pass [PROTO].
6. **Separate timing from identity.** Frame starts are sync matches with
   ≤ 3 bit errors confirmed by another near-sync 72 bits away. A CRC-valid
   word gives a verified counter; other frames inherit counters by position
   (`picture_only`). A frame feeds the time map only if its start is within
   8 samples of its neighbours' prediction, and only up to its first
   internal bit slip, so dropouts cannot shift timing by whole bits.
7. Output the **time map** τ(n) from those bit boundaries (Savitzky-Golay
   smoothed), then refine it by least-squares alignment of the regenerated
   clock (decoded words through the same band-pass) over 20 ms windows.
   Keep a refined point only if correlation ≥ 0.6 and |correction| ≤ 3
   samples. The residual (~0.4 samples RMS [PROTO]) is removed by §9.3.

---

## 6. OFDM body

### 6.1 Geometry

| Quantity | Value |
|---|---|
| Reference rate | 48,000 Hz (receive is rate-independent; transmit resamples as in `[SPEC §2]`) |
| N / CP / symbol | 128 / 16 / 144 samples (2.667 ms useful, 3.0 ms total) |
| Symbols per frame F | 24 → **3,456 samples, 72.0 ms, 13.889 fps** |
| Symbol s of frame f starts at | 3456·f + 144·s; FFT window at +12 (CP−4), as in `[SPEC §2]` |
| Carriers | bins 4–34 = 1,500–12,750 Hz (31 carriers) |
| Clock guard | bins 1–3 carry nothing |
| Emission ceiling | 14 kHz, with the existing gentle FIR `[SPEC §2]` |

Profile note: the block structure (§6.3) fixes F = 3 × 8. A 15+ fps profile
needs a different pilot period, and §13 E7 tests whether it is worth the
capacity.

### 6.2 Phase table

Per-cell random phase is kept from `phases()` `[SPEC §3]` with a new seed
offset (`seed=70001 + profile`). It decorrelates frames spectrally and
keeps PAPR Gaussian. Pilot cells use the same table.

### 6.3 Pilots

- **Continual pilots:** bins 4 and 34, every symbol.
- **Scattered pilots:** bins b ∈ {5, 9, 13, …, 33}, on symbols with
  `s mod 3 = φ(b)`, where `φ(b) = ((b−5)/4) mod 3`. Resolved:
  φ = {5:0, 9:1, 13:2, 17:0, 21:1, 25:2, 29:0, 33:1}.
- **Overhead:** 112 of 744 cells, 15.1% [MEASURED]. Every scattered carrier
  is revisited every 3 symbols (9 ms). Pilots are 4 bins (1.5 kHz) apart in
  frequency, which suits the smooth response of tape and codecs.
- **Pilot value:** amplitude 1.5 × data RMS (+3.5 dB), in the M/S basis:
  M = +1 and S = j·(−1)^v, where v counts visits of that carrier in the
  frame. Consecutive visits are orthogonal (`[1, j]`, `[1, −j]`), and every
  visit puts equal pilot power on both output tracks. [PROTO: the earlier
  S = (−1)^v put all of each visit's pilot power on one track, leaving the
  other track with no pilot on alternate symbols, which broke per-symbol
  fade fitting.]

### 6.4 Blocks

A **block** (b, φ) is the 8 cells at carrier b on symbols s ≡ φ (mod 3),
namely s = φ, φ+3, …, φ+21. Pilots occupy whole blocks, so a data block is
always 8 full cells. There are **79 data blocks** (carriers 5–33 × 3 phases,
minus the 8 scattered-pilot blocks) [MEASURED].

### 6.5 M/S precoding

Each data cell carries complex m and s:

```text
x_L = (m + s)/√2        x_R = (m − s)/√2
```

- **Mono summing or parametric-stereo codecs:** L+R = √2·m, so M survives
  and S is lost. The mono-aware slot order (§8.3) makes the lost S half the
  less important ranks.
- **One track polarity-inverted:** the 2×2 equaliser absorbs it, but the M-only
  preamble, clock and metadata cancel in the L+R acquisition sum. The pulse
  receiver retries once with the right leg inverted when nothing is found and
  reports `polarity_inverted`. A *mono sum* of an inverted leg keeps only S and
  is not recoverable.
- **One track lost:** the survivor carries (m ± s)/√2. Where s = 0
  (mono blocks, §8.2) m survives at −3 dB.
- **Clean stereo:** the 2×2 equaliser inverts both, so there is no cost.

### 6.6 Level

- **OFDM body:** fixed RMS target. Tape profile: −18 dBFS, or −14 dBFS with
  clip-and-filter. Digital and codec media: −16 dBFS.
- **No per-frame normalisation.** A transparent safety limiter at −1 dBFS
  catches statistical peaks; its action is logged.

### 6.7 Optional transmit headroom

Clip-and-filter confined to bins 4–34, with 4 iterations at 8 dB
(§3.4: +4.2 dB level, 24 dB MER floor).

- Tape use only; off by default.
- It does not change the wire, and the receiver is unaware of it.

---

## 7. Source coding

### 7.1 Grids

Unchanged from V6: sampling grids Y 96×80, Cb/Cr 48×40. Kept DCT corners
Y 48×40, Cb/Cr 24×20. That is 2,880 coefficients `[SPEC §10]`. Image
preparation and colour conversion are exactly `[SPEC §7]` (JFIF full-range
BT.601 via Pillow).

### 7.2 Variance table (normative per profile)

- **The table:** `λ[0..2879]` is the mean-square of each kept coefficient over
  every frame of the baked library. Store it as float32 with a SHA-256 in the
  spec, as `HD_BAND_RMS` is today.
- **Both ends:** TX and RX must use the identical table. The profile code
  selects it.
- **Bring-up fallback:** the §3.2 fit, with DC entries at the measured values.

### 7.3 Rank and gain

- **Rank:** coefficients sorted by λ descending, stable by index.
- **Gain** [LIT: SoftCast]: `g_i = λ_i^(−1/4) · sqrt(P / Σ_sent sqrt(λ_j))`,
  where P is the per-frame analog power budget and the sum runs over the
  coefficients sent in that frame.
- **Mean removal:** the DC entries carry the plane mean minus the library
  mean; both ends know the library mean.

### 7.4 Tiers

| Tier | Ranks | Carriage |
|---|---|---|
| Head | 0–207 | mono blocks (§8.2) |
| Body | 208–2223 | stereo blocks, every frame |
| Tail | 2224–2879 | 96 per frame on a 7-phase rotation (phase = counter mod 7); full refresh every 7 frames (0.50 s) |

---

## 8. Placement

### 8.1 Health order

Sort data blocks by carrier ascending, then φ ascending. This is the
tape-ordered principle `[BENCH]`: importance falls as frequency rises, so
every low-pass (tape HF, MP3 low-pass, phone band) removes the least
important data first.

### 8.2 Mono blocks

- **Which blocks:** blocks on carriers ≤ 9 (≤ 3.375 kHz), 13 blocks
  [MEASURED], carry **M only**. S is zero there.
- **Capacity:** 16 values per block (8 cells × I/Q), 208 values in all, which
  hold the head tier.
- **Survival:** this is the only tier that must survive telephone band, mono
  playback, parametric stereo and loss of either track. All four are
  satisfied by the frequency (≤ 3.4 kHz), the stream (M) and zero S [DESIGN].

### 8.3 Stereo blocks

- **Capacity:** the remaining 66 blocks carry 32 values each: groups M-I,
  M-Q, S-I and S-Q of 8. Total 2,112.
- **Slot order (mono-aware):** an S slot on carrier b is ordered as if it sat
  on carrier b + 6 (`S_ORDER_OFFSET`, 2.25 kHz); ties go M before S, then φ,
  then I before Q. Mono playback loses every S slot, so this pushes the lost
  half towards lower-importance ranks, while a low-pass still removes the
  highest carriers last. Interleaving M and S per block (the earlier draft)
  put S-carried ranks immediately after the head.
- **The last 12 slots** (S of carriers 31–33, 96 values) form the tail window.

### 8.4 Groups and spreading

- **Group:** the 8 values of one component (for example M-I) of one block,
  one per symbol. Its members lie 9 ms apart and span 63 ms.
- **Rank windows:** fill groups from windows of 8·k consecutive ranks, where
  k is the number of groups in the window (8, or fewer for the last partial
  window). Group j of a window takes ranks `{W + j + k·t : t = 0..7}`. Each
  group therefore spans its window's local variance range (§3.3). Windows
  follow health order.
- **Spreading:** each group is multiplied by the normalised 8-point Hadamard
  matrix before it is written to its 8 cells. Member t goes to symbol φ + 3t.
- **Effect:** a burst of up to 9 ms removes at most one member of any group.
  In the head, where variance falls fastest, that costs about 5–15× less
  error than unspread placement [MEASURED §3.3].

### 8.5 Mapping pseudocode

```text
ranks   = argsort(-λ, stable)
blocks  = data blocks sorted by (carrier, φ)
mono    = [b for b in blocks if b.carrier <= 9]            # 13
stereo  = [b for b in blocks if b.carrier > 9]             # 66
head    = ranks[0:208]
body    = ranks[208:2224]
tail    = ranks[2224:2880][96*p : 96*(p+1)]  where p = counter mod 7
slots   = [(b,'M','I'),(b,'M','Q') for b in mono]
        + sorted([(b,c,q) for b in stereo for c in (M,S) for q in (I,Q)],
                 key = (b.carrier + (6 if c == S else 0), c, b.φ, q))
fill groups from windows over head ++ body ++ tail, in slot order
x_group = Hadamard8 · (g ⊙ coefficient values of the group)
```

---

## 9. Receiver

### 9.1 Timing

Sample symbol s of frame f at received positions τ(3456f + 144s + 12 + n),
n = 0..127, using the existing windowed-sinc sampler (`_sample_at`). Speed
and wow/flutter are removed **before** the FFT. This replaces per-packet
scale fitting and `_refine_drift` `[SPEC §4]`.

### 9.2 Clock cancellation (optional, recommended)

Regenerate the clock waveform from the decoded bits, time-warp it with τ,
estimate its per-channel gain by least squares over the frame, and subtract
it. This removes the clock's spectral leakage into bin 4 and above
[DESIGN; expected leakage before cancellation is about −25 dB relative to
bin-4 data].

### 9.3 Channel

- **Joint timing and channel fit**, per receive track: a static 1×2
  response per pilot carrier times a common per-symbol timing track δ(s)
  (piecewise linear, knots every 4 symbols):
  `y(s,b) = e^{j2πbδ(s)/N} · (h_M(b)·p_M + h_S(b)·p_S)`, solved by
  alternating least squares (responses given δ, then a Gauss–Newton phase
  step for δ). Remove the known CP−4 window phase before fitting. Then
  interpolate responses linearly between pilot carriers. [PROTO: pairwise
  pilot estimates were corrupted by residual clock timing; the joint fit
  raised clean-channel MER from 8–20 dB to 20–33 dB.]
- **Frames are self-contained:** estimation uses only the frame's own pilots
  plus the continual pilots of the adjacent ±1 symbol.
- **Common phase:** per symbol, fit common phase and a linear phase residual
  from all pilots in that symbol.

### 9.4 Fade model

For each symbol s and channel c, fit
`log|H_c(b,s)| = h_c(b) + u_c(s) − v_c(s)·f_b`:

| Term | Meaning | Estimated from |
|---|---|---|
| h_c(b) | static response | all frame pilots |
| u_c(s) | broadband gain wobble (level wander, NR pumping) | per-symbol pilots |
| v_c(s) ≥ 0 | spacing-loss slope (Wallace) | per-symbol pilots |

Each symbol has 4–5 pilots for 2 unknowns. The fitted attenuation sets each
cell's reliability.

### 9.5 Noise

Per symbol and channel, estimate noise from pilot residuals against the
fitted channel, smoothed over ±1 symbol and floored at the frame median.
Additive noise does not vanish when the signal does: without the floor, a
dropout's silent symbols looked noiseless and were trusted [PROTO]. Never use
decision-directed estimation on analog data.

### 9.6 Reconstruction

1. Run the per-cell 2×2 MMSE, `W = P·Hᴴ·(H·P·Hᴴ + N)⁻¹` with P the diagonal
   M/S prior power, or MRC where S is known zero. The prior sits on the left
   and the inverse on the right; the two orders agree only when a cell's M
   and S priors are equal. This gives per-slot (reliability, noise) exactly
   as the V6 decoder does.
2. **Per group:** the observations are `y = diag(a) · Hadamard8 · (g ⊙ x) + n`
   with prior `x ~ N(0, λ)`. Solve the 8×8 LMMSE (316 solves per frame).
3. **Confidence gate:** use V6's `clip((c−floor)/(.85−floor), 0, 1)`
   `[SPEC §10]`, with floors 0.05/0.15 (Y/C) for the head tier and
   0.45/0.60 elsewhere.
4. **Tail store:**
   - Each tail coefficient keeps its last value and age.
   - It is used while age < 7 frames.
   - It is cleared to zero when the head-tier change between frames exceeds a
     scene-cut threshold [DESIGN].
   - This is transmission reassembly, not temporal smoothing. Coefficient age
     is reported in diagnostics.

### 9.7 Status

| Status | Condition |
|---|---|
| `verified` | clock CRC valid and every tier reconstructed |
| `degraded` | clock CRC valid, some tiers gated |
| `picture_only` | flywheel timing without CRC, pilots coherent |
| `lost` | no usable head tier; hold the previous frame (R4) |

---

## 10. Budget comparison

| Wire | Frame samples | fps | Unique coefficients per frame | Unique coefficients/s | Share of samples spent on sync, training and header |
|---|---:|---:|---:|---:|---:|
| V6 | 5,360 | 8.96 | 2,880 | 25.8k | 22% |
| wire-tape | 2,912 | 16.48 | 800 | 13.2k | 41% |
| **V7** | **3,456** | **13.89** | **2,320** (all 2,880 within 0.5 s) | **32.2k** | **0%** (15.1% pilot cells, 3 carriers of clock guard) |

---

## 11. Expected behaviour per medium [DESIGN — each row is a test in §13]

| Medium | What survives |
|---|---|
| Clean stereo | everything; tail refreshes every 0.5 s |
| Ferric cassette, good deck | all tiers; top carriers noisier, detail gated first |
| Isolated lifts or dropouts | clock (low band); head and body with per-symbol fade-aware gating; bursts ≤ 9 ms are diagnostic cases |
| One track dead | clock, head tier at −3 dB, body degraded (m ± s mixed) |
| Mono playback / HE-AAC v2 PS | clock, head, M slots (the more important ranks, §8.3); S detail and the tail lost |
| MP3/AAC ≥ 96 kbit/s | everything up to the codec low-pass; codec noise handled as per-symbol noise |
| Telephone band (300–3,400 Hz) | clock, pilots on bins 4, 5 and 9, head tier (a thumbnail) |

---

## 12. Conformance additions (beyond `[SPEC §12]`)

1. **Image vector:** the face fixture through §7–§8. Record the
   prepared-plane, λ-table, WAV and decoded hashes; tolerance 2–3× the
   measured clean RMSE.
2. **Stream vector:** 20 frames, with cold start at a random sample offset.
   Expect every frame verified from the 3rd onward.
3. **Impairment vectors:** one each of spacing-loss lift, one-track mute,
   mono sum and a 3.4 kHz low-pass. Record the expected per-tier survival.
4. **Clock-word vectors:** 16 words with known bits and a CRC check.

---

## 13. Validation plan: order and kill criteria

Change one thing at a time, against the V6 baseline and tape-ordered V6, and
finish on real tape.

| Step | Change | Keep it if | Otherwise |
|---|---|---|---|
| E1 | SoftCast gains + baked λ table, on the **existing V6 wire** | clean picture unchanged, and hiss/lift RMSE improves ≥ 2 dB in `bench_v6_tape_lift` | re-examine the table; keep old gains |
| E2 | Clock track replaces preamble and header (OFDM still has training) | pulse lock at 0.5–2× speed and under `wow-flutter` ≥ V6; identity survives `lift-severe` | keep the preamble, add the clock only for speed |
| E3 | Clock-derived resampling before FFT | wow/flutter RMSE ≤ V6 at 72 ms frames | fall back to per-frame drift refit |
| E4 | Scattered pilots replace training; per-symbol fade model | lift RMSE improves; clean within 0.5 dB | keep training, add per-symbol magnitude refit |
| E5 | M/S precoding + mono blocks | mono-sum and one-track cases keep a recognisable head; clean unchanged | plain L/R |
| E6 | Hadamard time groups | burst/dropout RMSE improves; lift and low-pass not worse | plain interleave |
| E7 | Tail rotation; F = 24 vs alternatives | fps ≥ 12 with the tail refreshed ≤ 0.5 s | fewer coefficients, no rotation |
| E8 | Real tape: matched captures of V6, tape-ordered V6 and V7 | V7 wins or ties on every acceptance-gate item | revert the failing mechanism |

### 13.1 Synthetic V7 tape torture matrix

The tracked runner `tools/v7_torture_matrix.py` exercises the current V7 pulse
wire against a deterministic, synthetic 96 kHz tape-path matrix.  It is a
regression test for the encoder/decoder and impairment handling, not a model of
any particular tape deck; real tape captures remain authoritative.

Run it from the repository root with:

```text
.venv/bin/python tools/v7_torture_matrix.py
```

The runner generates 12 identical packets from the canonical reference face,
encodes at the 48 kHz wire rate, resamples to 96 kHz, and feeds the damaged
stream through the pulse-counted V7 decoder.  The seed defaults to `2026` and
can be changed with `--seed`; `--frames`, `--only`, and `--out` support shorter
diagnostic runs.  Results are written to the ignored `scratch/` directory.

The matrix deliberately excludes the former composite `worn-deck` case.  Its
19 current cases cover clean playback, four low-pass ceilings, three hiss
levels, wow/flutter, azimuth delay, crosstalk, track-level imbalance, DC plus
hum, bias leakage, soft saturation, dropouts, unmatched NR pumping, and Type I
and Type II combined paths.  Each row also reports rendered-image Y/Cb/Cr
PSNR, global SSIM, MAE, and normalized RMSE through
`tools/measure_plane_survival.py`; the vector RMSE alone is not the image
quality gate.  A passing run requires every generated case to recover the
expected packet count after the final-header boundary, validate metadata, and
produce displayable frames.  A lost picture may still be reported inside that
recovered packet count; the displayability/hold-last-frame rule is the relevant
acceptance condition.

### 13.2 Mono slot order and MMSE correction

Reference face, nearest filter, pulse wire, mean picture RMSE. "Before" is
b5560c1f-era code (interleaved M/S slots, prior applied on the wrong side of the
2×2 inverse); "after" adds the §8.3 slot order and the §9.6 correction.

| Case | Before | After |
|---|---|---|
| Clean stereo | 0.068 | 0.068 |
| Stereo played as mono, (L+R)/2 | 0.113 | 0.091 |
| One leg only (other silent) | 0.113 | 0.109 |
| One leg polarity-inverted | no frames | 0.068 |
| Torture: crosstalk 10 % | 0.075 | 0.068 |
| Torture: right track −4 dB | 0.084 | 0.068 |
| Torture: all other cases | — | equal or up to 0.0015 better; soft saturation +0.0005 |

The mono result equals its structural limit (every S slot lost, M perfect):
the decoder loses nothing beyond what the layout drops.

---

## 14. Rejected alternatives

| Alternative | Why rejected |
|---|---|
| SSTV-style FM line scan (Robot/PD modes) | Robust, but seconds per image; fails R1. |
| Compressed digital video + FEC | Cliff effect; the project already measured the bottleneck. |
| SoftCast 3D-DCT GoP (16 frames) | About 1.1 s latency, and one burst damages the whole GoP. Possibly revisit as a 2-frame temporal pair. |
| Hadamard across frequency | Harmful for monotone known fades (§3.3). |
| Full cross-track copies | Cost 720 slots for track-loss-only benefit [BENCH]. |
| N = 256 symbols | Halves CP overhead but doubles flutter ICI. Revisit only after E3 proves resampling. |
| Digital base layer (hybrid digital-analog) | Identity is already digital in the clock word. A digital picture layer reintroduces the cliff. |

---

## 15. Open questions

1. Dolby B-like mismatch and the repo's NR pumping are harmless in the
   prototype, and dbx-like companding needed the −3 dB clock (§17). Real
   Dolby B/C and dbx hardware are still untested.
2. Scene-cut threshold for the tail store: derive it from library statistics.
3. Does a real low-bitrate AAC encoder use intensity stereo that collapses
   stereo blocks above some frequency? If so, extend the mono-block range to
   that crossover.
4. The λ table from one test image is a stand-in; the bake may move the tier
   boundaries.
5. 44.1 kHz and odd device rates: the transmit resampler is unchanged from
   `[SPEC §2]`; confirm the clock filter's group delay is compensated in τ.

---

## 16. Sources

- SoftCast: Jakubczak & Katabi, *A Cross-Layer Design for Scalable Mobile Video* — https://groups.csail.mit.edu/netmit/wordpress/wp-content/themes/netmit/papers/softcast.pdf
- Linear timecode — https://en.wikipedia.org/wiki/Linear_timecode
- Parametric stereo (HE-AAC v2) — https://en.wikipedia.org/wiki/Parametric_stereo
- EBU, HE-AAC v2 overview — https://tech.ebu.ch/docs/techreview/trev_305-moser.pdf
- Compact cassette tape types and formulations — https://en.wikipedia.org/wiki/Compact_Cassette_tape_types_and_formulations
- LAME low-pass by preset (Hydrogenaudio) — https://wiki.hydrogenaudio.org/index.php?title=LAME
- DVB-T framing and pilots (ETSI EN 300 744) — https://dvb.org/wp-content/uploads/2019/12/a012_dvb-t_june_2015.pdf
- Pilottone (Nagra speed reference) — https://en.wikipedia.org/wiki/Pilottone
- Tone reservation / clip-and-filter PAPR — https://www.researchgate.net/publication/327712045_Tone_Reservation_Based_Gaussian_Clipping_and_Filtering_for_OFDM_PAPR_Mitigation
- Wallace, *The reproduction of magnetically recorded signals* (Bell System Technical Journal, 1951) — https://ieeexplore.ieee.org/document/6772680

---

## 17. Prototype findings

`tools/v7_proto.py` implements §5–§9 as a Python simulation (about 100 ms
per frame; not optimised). `tools/v7_bench.py` encodes the face fixture at
V6's measured stream level (RMS 0.152), damages it, and decodes 84 frames
(6 s at 13.89 fps). ATRAC uses the open-source atracdenc encoder at 44.1 kHz;
the Dolby B- and dbx-like models in `tools/v7_media.py` are rough
approximations, not the real circuits. The λ table is fitted on the same
face being tested, so these numbers are somewhat flattering.

Picture RMSE against the source, mean over all frames; 0.052 is the
truncation floor (discarding coefficients, no channel):

| Case | V7 | identity verified |
|---|---:|---:|
| clean | 0.053 | 83/84 |
| hiss −45 dBFS | 0.053 | 83/84 |
| severe shared lift | 0.057 | 83/84 |
| worn contact | 0.055 | 83/84 |
| lift + track skew | 0.060 | 83/84 |
| one track muted | 0.133 | 83/84 |
| 10 kHz low-pass | 0.053 | 83/84 |
| wow/flutter | 0.057 | 83/84 |
| 12 ms dropouts | 0.066 | 73/84 |
| Type I tape | 0.065 | 83/84 |
| NR pumping (repo) | 0.055 | 83/84 |
| Dolby B-like, no decode / ±3 dB mistrack | 0.053 | 83/84 |
| dbx-like, no decode | 0.092 | 79/84 |
| MP3 320k CBR / LAME V0 | 0.054 | 83/84 |
| ATRAC1 SP 292k | 0.062 | 83/84 |
| ATRAC3plus (atracdenc, 352.8k) | 0.054 | 83/84 |
| mono sum | 0.106 | 83/84 |
| Type I tape → MP3 320k | 0.069 | 83/84 |
| Dolby B-like → ATRAC1 SP | 0.062 | 83/84 |
| MP3 320k → mono sum | 0.107 | 83/84 |

Frame 1 is never verified: the clock loop needs one frame to lock.

Draft changes forced by the prototype (already applied above): clock level
−3 dB (§5.1); clock decoder, flywheel framing and gated LS refinement
(§5.3); quadrature S pilots (§6.3); joint timing/channel fit (§9.3); noise
floor at the frame median (§9.5). Tried and rejected: windowed-OFDM symbol
tapers and guard carriers (≤ 5 dB less clock-band leakage), and high-passing
the OFDM body (clears the clock band but adds −25 dB inter-symbol distortion
on every carrier).

Open results: one-track loss is worse than V6 (0.133 vs 0.081; V6's full
foundation copies protect more than V7's mono head). Real tape and hardware
MiniDisc captures are still required; no composite synthetic “worn deck” case
is treated as an acceptance result.

## Appendix A — evidence scripts

Run from the repository root with `.venv/bin/python`. No external
images or downloads are needed.

- **A.1 Gain law and variance fit** (§3.1–3.2): 120 crops/flips of
  `modem_tests/fixtures/v7_reference_face.png` only, through
  `prepare_image` → `image_values` → V6 `SourceCoder` DCT. Per-coefficient
  λ = mean square. Compare `g² ∈ {sigma_model, 1/sigma_model, λ^(−1/2), 1}`,
  each normalised to equal mean transmit power; MSE_i = λ_i·N/(g_i²·λ_i + N).
- **A.2 Hadamard versus plain** (§3.3): total LMMSE trace for
  `y = A·x + n`, A ∈ {I₈, H₈/√8}, λ log-spaced over 0/10/20/30 dB, averaged over
  every 1- and 2-slot erasure position and over 400 random partial-fade draws.
- **A.3 PAPR** (§3.4): 4,800 random symbols on bins 4–34, grouped into
  24-symbol frames, 4 iterations of clip-to-k·RMS then zeroing of unused bins.

A.1–A.3 were one-off scratch scripts and are not checked in; the prototype
results in §17 are reproducible with `tools/v7_bench.py`.

## 19. Current live-wire amendment: pulse V7

The current live sender/receiver does **not** use the continuous 72-bit clock
track described in the original proposal. That design remains a research
comparison. The live wire returns to the V3–V6 LTC-style pulse acquisition path
because it provides lower latency, speed recovery, and prompt reacquisition.

### 19.1 Frame geometry and cadence

At the 48 kHz reference rate, a current live frame is:

```text
288 samples pulse preamble
3456 samples V7 OFDM body: 24 × 144
144 samples CRC metadata OFDM symbol
32 samples guard
--------------------------------
3920 samples = 12.245 fps
```

The body remains the 24-symbol, 128-point/16-CP V7 body. The metadata symbol
keeps the rate above the 12 fps requirement. This 3,920-sample format is the
only supported live V7 pulse format.

The pulse preamble is measured with `transport3.measure_pulses()`, not FFT
correlation. The receiver accepts a bounded playback scale of approximately
0.25×–2×, resamples the body to the reference grid, and uses consecutive pulse
positions to estimate frame-to-frame timing drift. The body walk uses the
measured frame duration, so smooth wow/flutter is corrected across the payload.

### 19.2 CRC-protected live metadata

The live metadata word is three payload bytes followed by CRC-16/CCITT-FALSE.
The source index is zero-based in the application API and one-based on the
wire; wire zero is therefore invalid rather than an accidental frame zero:

```text
payload byte 0 bit 7..5: aspect code
payload byte 0 bit 4..3: source encoding: 00=nearest, 01=box, 10=lanczos, 11=bicubic
payload byte 0 bit 2..1: revision/extension; bit 1 advertises mono-sum
payload byte 0 bit 0:    fixed live marker
payload bytes 1..2:      one-based source-frame index, big-endian
CRC:              polynomial 0x1021, init 0xFFFF, xorout 0
```

The 40 bits are mapped to 20 QPSK cells. Eleven pilots are spread across the
metadata band and the remaining twenty bins carry data. Both tracks carry the
mono-safe M signal. The metadata symbol estimates its own complex response
from those pilots. CRC failure holds the previous metadata. The receiver
decodes this self-referenced symbol with a bootstrap model before selecting the
source encoding model, so sender and receiver no longer need a manually
matched `--encode-filter`. Unknown revision values or wire index zero are
rejected and retain the prior metadata. The live UI additionally requires
three consecutive reliable requests before changing aspect.

### 19.3 Live source preparation

The live V7 path defaults to nearest-neighbor sampling at both source stages:
source to the 80×96 preparation canvas and prepared image to the V7 coder
grids. This intentionally produces a clean pixelated image rather than Lanczos
ringing. Existing V3–V6 callers retain Lanczos by default; `image_values()` now
accepts the same explicit `encode_filter` choices as `prepare_image()`.
V7 live sending also applies a small source brightness multiplier of 1.05 by
default, with explicit `--brightness` and `--gamma` controls for camera
matching. These are source-only presentation adjustments; they do not alter
the wire or decoder.

Nearest live conversion increases source projection error on the checked-in
face relative to Lanczos, but does not increase audio RMS, peak, or carrier
bandwidth. This is an intentional visual trade, not a claim of free fidelity.

### 19.4 Accelerated playback profiles

The canonical V7 packet remains 3,920 samples at the 48 kHz reference clock.
An accelerated sender time-compresses each complete pulse packet before output;
it does not use pitch-preserving time stretch:

```text
speed 1.0x: 3920 samples, 12.245 fps, body top 12.75 kHz
speed 1.5x: 2613 samples at 48 kHz, 18.367 fps, body top 19.125 kHz
speed 2.0x: 1960 samples at 48 kHz, 24.490 fps, body top 25.5 kHz
```

The receiver measures the resulting pulse scale and resamples the body back to
the reference grid. `1.0x` is the tape-compatible baseline. `1.5x` is a
digital/wideband candidate; `2.0x` requires a path whose capture Nyquist and
analog bandwidth retain the expanded carriers, normally a 96 kHz path. At a
fixed 48 kHz output, downsampling before 2.0x playback necessarily loses some
highest carriers, so acquisition success alone does not prove full-band
fidelity.

Standalone sending exposes this as `tools/v7_live.py send --speed`; the
application sender exposes `--modem-speed` (or its `--speed` alias). Packets
are sped independently so
the preamble, metadata, and terminal guard remain local to each packet. No
speed field is needed on the wire: pulse timing identifies the effective speed.

### 19.5 Timing, level, and recovery

Live receive uses a slow-rise autoleveler based on the newest frame window,
bounded to `0.5×..32×`, with prompt gain reduction. Near-silence is gated
before pulse acquisition. Input callback drops clear the partial window and
hold the last good image rather than stitching samples across a gap.

Pulse confidence, foundation confidence/coverage, and timing delta are
reported. A frame with an untrustworthy foundation is `lost` and holds the
previous coefficient vector. A degraded usable frame remains distinguishable
from a held frame.

### 19.5 Capture, CPU, and live diagnostics

`tools/v7_live.py` is an explicit-device experimental tool. Camera capture
probes the lowest supported FPS and smallest resolution, and reuses the
V3–V6 `Throttled` newest-frame capture path. Screen capture defaults to `mss`;
FFmpeg is explicit for screen and remains the camera device/mode backend.

The current prototype precomputes rank/placement tables, vectorizes group
mixing and IFFT, batches receiver solves, caches clock templates, and reports
stage timing. The fixed UI shows peak/RMS levels, incoming/decoded FPS,
verified/lost counts, pulse confidence, timing delta, gain, and foundation
quality.

The live receiver decodes only the newest complete pulse frame in its bounded
window. The V7 package remains outside the production engine registry until
real-media validation and long-run CPU/reacquisition testing are complete.

## 18. Bench-only live camera and screen path

`tools/v7_live.py` provides an explicit-device experimental sender and receiver:

```bash
.venv/bin/python tools/v7_live.py send --source camera \
  --device 'BlackHole 2ch'
.venv/bin/python tools/v7_live.py send --source screen \
  --device 'BlackHole 2ch'
.venv/bin/python tools/v7_live.py receive --device 'BlackHole 2ch'
```

The sender reuses `modem_screen.py`'s camera and screen capture sources,
selects the smallest camera mode supporting the requested capture rate, and
prepares frames with the V7 source grids. Live V7 uses the V3-style
pulse-counted frame preamble and runs at approximately 12.245 fps. Live V7
defaults to nearest-neighbor
sampling at both the 80x96 preparation step and the coder-grid sampling step;
`--encode-filter` can select another explicit filter. Screen capture defaults to
the existing `mss` path; `--screen-backend ffmpeg` selects the FFmpeg pipe
explicitly. Camera capture continues to use FFmpeg because it probes the
smallest supported device mode and performs the RGB pipe conversion. The live
pulse frame is 3,920 samples (288 preamble + 3,456 body + 144 metadata symbol
+ 32 guard), or 12.245 fps. The receiver accumulates the explicit 48 kHz input,
runs the V7 prototype
decoder, displays the newest usable reconstruction, and can save frames with
`--save-dir`. A `--headless` receiver is available for loopback diagnostics.
The window is resizable; keyboard `F` toggles fullscreen, `I` toggles the
diagnostic panel, and `Escape` exits fullscreen. Diagnostics are shown by
default and can be hidden with `--no-diagnostics`; `--no-log` suppresses routine
console status output for standalone embedded use. Sender and receiver print one
hardware/status line at startup; routine per-frame output is disabled by default
and can be enabled with `--log`.
The sender's optional `--mono-sum` emits one audio channel containing the
shared M signal and advertises that choice in metadata; the receiver accepts
mono input devices automatically. The sender uses an energy-preserving sum
with a safety limiter rather than a simple average, avoiding an unnecessary
3 dB mono-level loss.
The receiver's optional `--mono-compatible` presentation mode preserves luma
while confidence-gating temporal/spatial chroma stabilization for mono or
one-leg playback. It does not alter the wire or encoder.

This is deliberately not a live production integration. The current prototype
uses the offline V7 shaping path for each batch, so batch boundaries can create
filter transients; the clock counter continues between batches to expose that
failure rather than silently resetting. It also requires a real 48 kHz device
and does not resample to arbitrary native device rates. Real-device/tape use
must wait for a stateful clock/filter implementation and a matched live timing
test.

### Live metadata amendment

The live pulse wire no longer uses guard length as aspect metadata. The extra
144-sample metadata symbol carries the packed aspect, encoding, and revision
fields described above, followed by CRC-16/CCITT-FALSE. It is repeated through
known even-bin pilots and odd-bin data carriers and decoded only when the CRC
passes. An invalid metadata word holds the previous metadata. The UI additionally
requires three consecutive reliable requests before changing its displayed
aspect. This amendment is intentionally separate from the continuous-clock
architecture described in earlier sections so the older draft remains available
for comparison.

## 20. Stability baseline and future experiments

The current live pulse wire is the protected baseline. Future experiments must
be explicit, reversible, and compared against this baseline; they must not
silently replace a working decoder path.

Baseline invariants:

- 3,920-sample pulse frame at 48 kHz, approximately 12.245 fps;
- V3–V6-style `measure_pulses()` acquisition;
- next-header pulse as the current-frame timing/completion witness;
- CRC-protected metadata and held/debounced aspect changes;
- nearest-neighbor live source sampling;
- slow-rise autoleveling and silence gating;
- input-gap recovery and hold-last-valid-frame behavior;
- degraded display only above the explicit foundation/noise quality baseline;
- vectorized encoder, batched solves, bounded live history, and diagnostics.

### 20.1 Decode CPU baseline and completed optimization

The decoder equalizer is now vectorized across all data blocks and both
quadratures. The previous implementation entered Python once per block/channel
and dispatched many tiny matrix operations. The new implementation preserves
the same MMSE equations and quality gates while assembling the block priors and
solves in batches.

Measured on the same fixture and pulse wire:

```text
                         old EQ       vectorized EQ
clean CPU/frame           47.7 ms          26.4 ms
Type-I CPU/frame          55.6 ms          31.6 ms
```

This is approximately a 43--45% reduction in decoder CPU time. Direct A/B
results matched at clean, hiss, low-pass, wow/flutter, and dropout conditions.
Type-I remained received/displayable; RMSE changed from 0.0698 to 0.0710.
The change is decoder-only and does not alter the wire, encoder, redundancy, or
quality thresholds. Numba was evaluated as an optional tool but is not required
by the runtime; the NumPy implementation is faster to deploy and has no JIT
startup cost.

### 20.2 Completed low-risk decode work

The following decoder improvements are now implemented:

- make pulse acquisition incremental instead of rescanning and concatenating a
  large rolling audio window on every live tick;
- move decoding to a worker so Tk meters and presentation cannot be blocked by
  equalization;
- render `values_image()` only when a new frame arrives, not on every UI tick;
- lazily build alternate source models after metadata identifies the encoding,
  while retaining the nearest bootstrap model;
- precompute remaining noise/index lookup arrays and avoid per-frame temporary
  dictionaries.
- keep live rolling audio in `float32` and avoid concatenating the retained
  history when no new capture block has arrived;
- publish only the newest usable decoded frame to a bounded depth-one handoff,
  so a slow downstream consumer cannot create backlog latency.

The worker keeps the decoder and UI presentation independent; a damaged decode
does not terminate the display loop. These are CPU/latency optimizations only.
Do not remove wire redundancy or change the 12.245-fps frame geometry without a
separate resilience comparison.

### 20.3 Stereo diversity and the current combining limit

The live decoder already uses both received channels jointly. Pilot-derived
channel matrices feed a per-cell 2×2 MMSE equalizer, followed by the grouped
LMMSE reconstruction. This is the correct linear combiner for the current M/S
wire; simply averaging the stereo pair would discard S information rather than
produce free additional gain.

The mono-sum mode is therefore a compatibility path for one-channel hardware,
not a higher-quality stereo decoder. Any further stereo-pair dB improvement
would require better channel/noise weighting validated against the tape matrix;
no safe extra gain was found in this final pass.

### 20.4 Recommended timing experiment: pilot-derived nonlinear warp

The pulse endpoints correct affine frame scale. Flutter can still warp the
middle of a frame. The safest next experiment is a gated second pass:

1. decode with the baseline affine pulse map;
2. estimate residual per-symbol timing from known OFDM pilot phase;
3. fit a robust monotone low-order time map;
4. resample the body with that map;
5. rerun the fixed-size FFT and source reconstruction;
6. retain the second result only if pilot residual, timing curvature, foundation
   quality, and metadata CRC improve together.

If the fit is underdetermined, non-monotone, too large, or worse than the
baseline, retain the baseline result. Timing variation inside one useful FFT
symbol remains an ICI limit and cannot be recovered perfectly afterward.

Required A/B results are clean RMSE, pilot residual, timing residual, CPU time,
latency, received/lost counts, and the existing wow/flutter matrix.

### 20.5 Lower-risk experiments

- Use weighted circular pilot-phase fits with carrier outlier rejection.
- Compare a denser pilot schedule against the lost analog payload capacity.
- Add per-plane quality reporting so chroma loss is not confused with timing
  failure.
- Add timing-loss duration and reacquisition latency to the live diagnostics.

### 20.6 Deferred changes

Do not yet add a second timing track, full-picture repetition, stronger FEC,
temporal prediction, learned source coding, or a new modulation family. Each
would change multiple variables and make the current stable emulation harder to
audit. Every future change requires a reversible flag, clean vectors, matched
impairment results, and a real-media acceptance plan.

## Appendix A: EOF acquisition proposal

This appendix records a positive acquisition proposal for later implementation;
it does not change the V7 baseline wire.

The proposal is to retain the existing body decoder and timing calculations,
and add an explicit loud end-of-frame pulse in the existing guard budget. The
receiver would use classical packet-clock logic:

```text
header lock → count known packet geometry → validate EOF pulse → commit frame
```

The header remains the start/timing reference and the EOF pulse becomes the
observed packet-end reference. This has useful properties:

- no packet-length or FPS increase is required;
- acquisition no longer needs to search payload crossings;
- the body OFDM decoder remains unchanged;
- packet commit no longer depends on the next frame's header;
- header/EOF distance supplies an independent timing and completeness check;
- the approach is deterministic, inexpensive, and appropriate for noisy tape;
- loss recovery can be an explicit timeout/reacquisition transition.

An EOF-only prototype must change its encoder and decoder together. It must
validate clean generated WAVs, playback-scale PPM, timing residuals, CPU, and
the existing tape matrix against the baseline before any wire change is made
default.

### A.1 Bench implementation findings

The paired bench implementation replaces the proposal's broad candidate
search with a small packet state machine. The wire remains the 3,920-sample
format above; the final 32 guard samples carry the loud EOF marker:

```text
SEARCH_HEADER → PACKET_CLOCK → VALIDATE_EOF → COMMIT
       ▲                              │
       └──────── timeout/junk ───────┘
```

Header acquisition now:

- qualifies polarity with a Schmitt threshold;
- timestamps transitions at the interpolated zero crossing, using samples that
  actually bracket zero;
- checks the known polarity sequence and each expected short/long gap in order;
- fits the packet origin and playback scale from the accepted edge run.

EOF acquisition now:

- opens only near the predicted packet end;
- validates the known EOF polarity transitions sequentially;
- measures the marker pulse spacing and the header-to-EOF packet duration;
- rejects the packet before metadata, body sampling, FFT, or equalization when
  the header or EOF state fails.

After a valid EOF, the packet is committed immediately. The next header is
used only to start acquisition of the next packet. Silence or arbitrary audio
between EOF and that header is not treated as a packet boundary and cannot
re-anchor the packet already committed. The OFDM body retains the existing
affine sample walk; EOF acquisition does not add a second body timing map.

The paired encoder/decoder torture run covered 19 generated WAV impairment
cases and decoded 8/8 packets in every case. Acquisition CPU was lower in all
19 cases, by approximately 1.4--7.1 ms per eight-packet pass. Representative
packet-scale results were:

```text
                         baseline PPM     EOF PPM
clean                         0              -1
wow/flutter               -3575           -3574
Type-I                    -2781           -2723
Type-II                   -1748           -1693
```

These are bench results only. The existing V7 pulse wire and its
`measure_pulses()` receiver remain the protected live baseline until the EOF
state machine is migrated and revalidated on real media.
