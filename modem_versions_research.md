# Modem versions V1–V7 — a deep research trace

**Scope.** Every modem transport generation in this repository, from the first
`transport.py` on 2026-09-09 to the current V7 tree on 2026-10-02, including
versions and receiver families that exist only in history or on archived
branches. For each version: how it worked mechanically, what functionality it
added, what it replaced, and why the next version replaced it.

**Method.** Everything below is read out of the repository: `git log`,
`git show <commit>:<path>`, the two normative reimplementation specs written on
the `standalone-modem` lineage (`docs/transport_timing_and_encoding.md`,
`docs/transport_state_machines.md`, both at `af4830d1`), the current
`docs/transport_v7_spec.md`, and the current source tree. Numbers are quoted
from code or from the commit message that measured them, and the measuring
commit is named. Where a figure is a dated snapshot rather than a current
measurement, it is marked as such.

**Source-reference shorthand.** `[Hxxx:path]` = historical file at commit
`xxx…`; `[C:path]` = current tree. This is the convention the standalone specs
use, kept here so claims can be re-checked with `git show`.

---

## 1. Executive summary

Seven modem generations exist, built in a compressed two-and-a-half week
(2026-09-09 → 2026-10-02) and then narrowed down to one. The through-line is
consistent and unusually disciplined:

> **Analog-valued image coefficients → one transform at the source coder →
> one OFDM modulator → edge-counted (LTC-style) timing acquisition → degrade
> detail before structure, structure before colour → never black.**

Every version is one of two kinds of change:

- **Transport changes** (V1→V2→V3): wire geometry, source coding, acquisition
  algorithm, identity signalling. These break compatibility with each other by
  construction.
- **Profile/coder changes** (V4→V5→V6): the physical wire stays byte-identical
  while the transform and the protection scheme change underneath. These are
  *compatible at the transport level* and differ only in header profile code.

| Ver | Date | Transport file | What it fundamentally changed |
|---|---|---|---|
| **V1** | 2026-09-09 | `animation_modem/transport.py` | First working modem. Raw image-plane values, correlation acquisition, 40×48 picture. |
| **V2** | 2026-09-09/10 | `animation_modem/transport2.py` | DCT source coding + power allocation; layout presets; progressive/interleaved coefficient ordering; automatic EQ recovery. |
| *(receiver family)* | 2026-09-12 | `waveform_receiver.py`, `progressive.py`, `reference_receiver.py` | Side experiment: three alternative receiver architectures on top of the V2 wire. Never selected. |
| **V3** | 2026-09-13 | `animation_modem/transport3.py` | Replaced correlation acquisition with edge/pulse counting. Body kept byte-identical to V2. 12–60× less CPU. |
| **V4** | 2026-09-18 | `animation_modem/wavelet.py`, `engines.py` | Engine registry. One-level Haar DWT as an alternative source coder on the V3 wire. |
| **V5** | 2026-09-18/19 | `animation_modem/wavelet.py` (CDF 9/7), `fec.py` | Two-level CDF 9/7 DWT + stereo-repetition protection. Plus a parked LDPC/mono path. |
| **V6** | 2026-09-20 | `animation_modem/v6.py` | Returned to analog coefficients on a tape-band wire; 720-value *foundation* duplicated onto opposite channel, carrier and symbol. |
| **V7** | 2026-09-20 → | `animation_modem/v7.py`, `v7_core.py`, `transport3.py` (pulse only) | New wire from spec: OFDM body + CRC-protected metadata symbol + timing tones + EOF marker; SoftCast gains, M/S precoding, Hadamard groups, tail-slice rotation; folds, mono wires, stereo slices. |

**What is true in the tree today.** V1–V6 are *gone*. `53cc384b` (2026-09-21,
"Remove legacy modem transports and keep V7 core") deleted `transport.py`,
`transport2.py`, `core.py`, `decoder.py`, `wavelet.py`, `fec.py`, `v6.py`,
`transport3_v5.py`, `progressive.py`, `waveform_receiver.py`,
`reference_receiver.py`, `modem_screen.py` and `utilities/modem_v3_check.py`.
What survives from earlier generations is deliberately tiny: the 16-bit
biphase-mark pulse word in `animation_modem/transport3.py` (V3's acquisition
geometry, reused verbatim by V7), the allocation/reconstruction algebra in
`v7_core.py` (V2's `default_allocation` + channel-aware Wiener inverse), and the
`imaging.py` profile registry.

---

## 2. Branch and lineage map

### 2.1 Branches present in this clone

```
modem-v7-integration      f3cf11c9  2026-10-02  <- HEAD, all V7 work
feature/v7-encoded-preview c0199d12  2026-09-30
feature/v7-live-preview-recovery fde81404 2026-09-30
fix/v7-source-dct-checkout-boundary 96398f35 2026-09-29
perf/v7-source-dct-preparation      96398f35 2026-09-29
v7-image-calibration       299f38f8  2026-09-30
main                       9591126d  2026-09-24
experiment                 8224a36f  2026-09-17  <- V1/V2 live here
scope                      01506d18  2026-09-08  <- pre-modem baseline
ascii_test                 5bd7cedb  2026-01-13
cursor                     b1aaed46  2026-01-06
forOldStable               2674f25c  2025-11-16
```

### 2.2 Lineage

```
scope (01506d18, 09-08)  pre-modem application/scope tree
   │
   └─ experiment (8224a36f, 09-17)
        │   V1 364c2928 (09-09)   transport.py
        │   V2 33734bcf (09-09)   transport2.py
        │   V2 99d60665 (09-10)   pinned V2 snapshot
        │   70d3892d (09-12)      waveform / progressive / reference receivers
        │   V3 b55ef9b4 (09-13)   transport3.py  <- acquisition replaced
        │   97570900 (09-13)      transport.py/transport2.py deleted, core.py born
        │   7e664932 (09-16)      profile codes + magic b'V3'
        │   2491da33 (09-17)      ported to standalone-modem
        │
        └─ standalone-modem  (archived; remote ref no longer in clone,
                               reachable from V7's ancestry)
              af4830d1 (09-20)   docs/MODEM_INDEX.md + the two normative specs
              V4 467cbaf2 (09-18)  wavelet + engine registry
              V5 34ae7739 (09-18)  CDF 9/7 + LDPC
              d3a9cbad   (09-19)   wire-hd / wire-tape / wire-tape-25
              V6 4a672b07 (09-20)   v6.py analog tape transport
              0fdc309d   (09-20)   V7 spec + bench-only prototype ("v7?")
                  │
                  └─ modem-v7-integration  (HEAD)
                       80f8390c (09-21)  V7 promoted into animation_modem/
                       c8a98fb4 (09-21)  application sender + source metadata
                       53cc384b (09-21)  legacy V1-V6 deleted
                       ... 183 commits, 09-21 → 10-02 ...
                       f3cf11c9 (10-02)  stereo slices

   main: 63ba05c4 (09-16) merged modem back into main (scope came with it);
         main has been app/nginx work since, tip 9591126d (09-24).
```

### 2.3 Where each version's authoritative description lives

| Version | Live source of truth | Historical reference spec |
|---|---|---|
| V1 | — (deleted) | `[H99:animation_modem/transport.py]`, spec §5 |
| V2 | — (deleted) | `[H99:animation_modem/transport2.py]`, spec §6, `MODEM_V2_RECOVERY.md` |
| V3 family (V3/V5/tape/V6) | — (deleted) | `af4830d1:docs/transport_timing_and_encoding.md` §2–§4, §7–§11 |
| V4 | — (deleted) | `af4830d1:docs/transport_timing_and_encoding.md` §7 (`color-wavelet`) |
| V5 | — (deleted) | spec §8 (active) + §8 note (parked LDPC path) |
| V6 | — (deleted) | spec §10 |
| V7 | `animation_modem/v7.py` + `docs/transport_v7_spec.md` (current-state) | `0fdc309d:docs/transport_v7_spec.md` (draft) |

The two specs are unusually valuable and are quoted heavily below. They were
written on `standalone-modem` at `af4830d1` and were the *only* place V1–V6 are
still described in detail after `53cc384b` deleted their code.

---

## 3. Invariant threads (what never changed)

These survived every generation and are still asserted by `AGENTS.md`:

1. **Analog coefficients, not digital bytes.** From V2 onwards the payload is
   continuously-valued transform coefficients. The one excursion — a digital
   VP8/JPEG-2000 detour around 2026-09-19 — was explicitly recorded as a
   mistake: *"The digital detour imposed an unnecessary bottleneck. The
   192-byte frame budget drove severe compression and frame-loss thresholds.
   VP8/JPEG 2000 solved a different problem from the intended analog wavelet
   transport."* (`MODEM_TODO.md` @ `4a672b07`)
2. **Exactly one transform, applied at the source coder.** `v7_core.SourceCoder`
   documents the failure mode in its own docstring: a double transform smears
   energy from 0.1% of slots back across 100% of them, collapsing 39 dB clean
   PSNR to 8 dB at −45 dBFS noise versus 33 dB when transformed once.
3. **Timing acquisition is edge/pulse-counted, never FFT-correlated.** V2's
   correlation template bank is the explicit villain of the V3 commit; V7 keeps
   `transport3.measure_pulses` and its Numba kernel.
4. **Degrade detail → structure → colour.** Encoded in V6's
   `recovery_floor` (protected luma 0.05 / chroma 0.15; unprotected luma 0.45 /
   chroma 0.60) and again in V7's head/body/tail rank tiers.
5. **No black-frame fallback.** A damaged-but-decodable frame is shown as-is; an
   undecodable packet holds the last good picture. Never black.
6. **Every packet is self-describing.** Rate reference inside the packet, CRC
   identity, and a bounded "hold the layout, retry four times, then
   re-detect" contract — from V2's `Receiver` docstring through V7's live
   receiver unchanged in spirit.
7. **The transport package never imports application code.** `animation_modem/`
   must stay independent of settings, renderers and audio devices; app code
   (`modem_v7_display.py`, `tools/v7_live.py`) adapts it.

---

## 4. V1 — the first working modem

**Commit.** `364c2928`, 2026-09-09, "tweaking modem branch to reduce latency",
on the `experiment` lineage. Introduced
`animation_modem/transport.py` plus `decoder.py`, `playback.py`,
`modem_display.py`, `modem_bake.py`, `utilities/convert_to_modem.py`, and a
`main.py --mode modem` path.
**Pinned snapshot.** `99d60665` (2026-09-10) — the state the standalone spec
treats as the canonical historical V1.
**Deleted.** `97570900` (2026-09-13, "deleting old shit").

### 4.1 How it worked

A fixed-geometry frame-local analog OFDM link.

```
SYNC(288) | symbol0 training | symbol1 training | symbol2 header | symbol3 header
          | symbols 4..18 image (15 symbols)
          | 176-sample zero tail
FRAME = 3200 samples @ 48 kHz -> 15.000 fps
```

| Parameter | Value |
|---|---|
| `RATE, FPS, FRAME` | 48000, 15, 3200 |
| `N, CP, SYMBOL` | 128, 16, 144 |
| Image canvas | 40 × 48 |
| Body symbols | 19 = 2 training + 2 header + 15 image |
| `PACKET` | `288 + 19*144` = 3024 (176-sample tail) |
| Pilots | `[6, 19, 35, 50]` |
| Data carriers | FFT bins 3…54 minus pilots (52) |
| Header | 24 bytes incl. CRC32, stereo-repeated |

**Source coding: none.** V1 transmitted image-plane samples directly. Spec
§5 is blunt: *"There is no transform, coefficient rank, power allocation, or
redundancy in V1."* The DCT buys nothing here; the wire is a straight
per-sample modulator.

**Three profiles, all exactly 2,880 values** — the count that every later
version also lands on:

| Profile | Luma | Chroma | Magic | Phase seed |
|---|---|---|---|---|
| `color` | 40 × 48 = 1920 | Cb/Cr 20 × 24 = 480 + 480 | `SI01` | 41015 |
| `detail` | 48 × 56 = 2688 | Cb/Cr 8 × 12 = 96 + 96 | `SI02` | 41016 |
| `mono` | 48 × 60 = 2880 | — | `SI02` | 41017 |

Preparation aspect-pads with Lanczos, converts with Pillow YCbCr, BOX-resizes
chroma, maps uint8 → `[-1, 1]`.

**Acquisition: correlation.** The preamble is a 256-sample random waveform
synthesised from FFT bins `6:109`, seed 41015, peak 0.65. Acquisition
correlates against **seven fixed off-speed templates**
`(-.024, -.016, -.008, 0, .008, .016, .024)` — i.e. it could only detect a
handful of discrete speed errors, then relied on a packet-spacing rate tracker
and 8-tap windowed-sinc resampling to refine.

**Concealment.** `conceal_values` does spatial concealment: a Gaussian filter
(σ = 1.5) fills unreliable regions, where "reliable" is equaliser weight
≥ 0.55.

**Status contract.** `Result` carried `received`, `degraded`,
`partial_header_unknown`, `corrupt_header`. Header candidates were tried in
order — stereo mean, channel 0, channel 1 — with CRC + structural checks, then
salvage thresholds.

### 4.2 What V1 added

Everything: the OFDM modulator (`N=128`, `CP=16`), the SVD/Wiener per-carrier
equaliser, the pilot/phase-table design, the stereo M=1 framing, CRC-protected
self-description, concealment on damage, and the whole sender/receiver/bake/
display chain.

### 4.3 What V1 could not do

- **No compression.** 2,880 values per frame is a hard ceiling, so the picture
  is 40×48 and there is no headroom to trade.
- **No power allocation.** Equal carriers for a picture whose energy is
  overwhelmingly low-frequency wastes almost the whole band budget.
- **Acquisition was fragile and expensive.** Fixed-template correlation is the
  thing V2 kept and V3 deleted.
- **Header had no room.** 4 bytes of the 24-byte header were freed for a
  shared-time millisecond field mod 2³², which is a symptom, not a design.

---

## 5. V2 — DCT source coding, layouts, and automatic EQ recovery

**Commits.** `33734bcf` (2026-09-09) introduced
`animation_modem/transport2.py`. `99d60665` / `2b3ea477` (2026-09-10,
"integrating modem 2 and it's working well. now to port it to main") is the
pinned snapshot the spec treats as historical V2. `MODEM_V2_RECOVERY.md` and
`utilities/modem_v2_check.py` came with it.
**Deleted.** `97570900` (2026-09-13).

### 5.1 How it worked

Same OFDM physics, but the payload became transform coefficients with a
frequency-dependent power allocation, and the geometry became a *parameterised
layout* instead of a constant.

| Parameter | Value |
|---|---|
| `RATE, N, CP, SYMBOL` | 48000, 128, 16, 144 |
| `LOW_BIN, SYNC_LEN, GUARD` | 3, 288, 32 |
| `HEADER_BYTES` | 16 payload + 4 CRC32 = 20 bytes = 160 bits = 80 QPSK |
| `HEADER_SLOTS` | 80, **identical on both channels** (diversity, not split) |
| `SYNC_SCALES` | 29 log-spaced values, 0.5 → 2.0 |

**The `Layout` abstraction** — `Layout(top_bin, image_symbols, name,
progressive)` with cached derived properties (`carriers`, `pilots`,
`data_bins`, `header_bins`, `header_symbols`, `symbols`, `packet`, `frame`,
`fps`, `band`, `capacity`, `max_speed`). This object is the ancestor of
V7's `V7_GRIDS`/`V7_SHAPES` constants and of every later wire table.

**The five presets:**

| Preset | top_bin | carrier band @48k | image symbols | progressive |
|---|---:|---|---:|---|
| `wide` | 54 | 375 – 20,250 Hz | 15 | yes |
| `tape` | 27 | 375 – 10,125 Hz | 35 | no |
| `tape-fast` | 27 | 375 – 10,125 Hz | 15 | no |
| `narrow` | 21 | 375 – 7,875 Hz | 44 | no |
| `lofi` | 10 | 375 – 3,750 Hz | 24 | no |

**Source coding: orthonormal 2-D DCT per plane plus power allocation.**
`default_allocation` is a separable low-frequency emphasis

```
sigma(fy, fx) = 1 / (1 + 12*hypot(fy, fx))
gain          = sqrt(sigma / mean(sigma))          # power ∝ stddev is the analog optimum
```

The gain table is renormalised to unit mean square, and reconstruction is
**channel-aware Wiener**: it never divides by weak gains, weighting by
`gain²·variance / (gain²·variance + noise)` and erasing coefficients whose
equaliser weight collapses.

**Progressive slot ordering.** `coefficient_slots` sorts *all* planes
together by normalised spatial frequency, places the three DC terms first, then
ascending carriers — "coarse image first in audio frequency". Non-progressive
layouts fall back to plain `arange`. This is the mechanism V6's "coarse first"
and V7's rank tiers both inherit.

**Header magic encodes progression.** `magic = b'V3' if layout.progressive else
b'V2'`, so the receiver refuses a non-progressive layout carrying a progressive
header.

**Tier ladder.** `Decoded.tier` is `best / better / good / poor / none` from
coefficient coverage (≥ .95 / ≥ .7 / ≥ .35 / else). A single vestigial counter
named `tiers` survives in today's `animation_modem/timing.py`.

### 5.2 What V2 added

- **DCT + power allocation.** The single most consequential change in the
  whole history. It is what makes analog transport viable at all: it spends
  carrier power where the picture has energy.
- **Parameterised layouts and five presets**, including narrowband tape
  variants — bandwidth as a first-class design variable.
- **Progressive ordering** so partial reception still yields a coherent
  low-frequency image.
- **Automatic EQ recovery** (`MODEM_V2_RECOVERY.md`). The 16-sample CP is only
  0.333 ms, so a long bass EQ impulse response contaminates adjacent symbols
  and the two training symbols. On a bad header or pilot error > 0.15 the
  receiver additionally fits a **128-tap stereo FIR** from the known preamble
  and training prefix, tries regularised frequency-domain inversion of that same
  packet, and keeps it *only when* the existing recovery-quality comparison
  improves. Result is reported as `input_path=eq_corrected`.
  Measured on consecutive frames through Q=1 bell EQ of ±6 dB at 375 and
  1000 Hz: image-value RMS error 0.26–1.15 → **0.04–0.08**.
- **A single automatic receiver for live and WAV.** Acquisition searches
  0.25×–2×; full-band timing first, low-band fitting as roll-off fallback. No
  input-filter or speed-range switches.
- **A receive-status contract** beyond pass/fail: `received` / `degraded` /
  `picture_only` / `lost`, plus coverage, pilot error, coherence and tier.

### 5.3 What V2 could not do

**Its acquisition was the bottleneck, and it knew it.** `_acquire_window`
correlates each unlocked buffer against a **48-scale × 128-tap template bank**
through an `optimize=False` einsum. The V3 commit message measures it on one
2.1 GHz core:

| Condition | v2 `Receiver` | v2 progressive | v3 |
|---|---:|---:|---:|
| locked, contiguous | 5.0% | 7.6% | **4.7%** |
| cold start | 55.8% | 57.4% | **4.8%** |
| dropout every 4 packets | 171.7% | 172.3% | **7.1%** |
| tape 8% fast | 6.2% | 9.4% | **4.8%** |
| tape 25% slow | 5.5% | 7.7% | **3.5%** |
| pure hiss, no lock | 273% | 294% | **3.7%** |

(percent CPU relative to real time; **all 30 packets verified in every case**).
The verdict in the commit subject line is exact: *"Locked decode was never the
problem."*

---

## 6. The receiver-family interlude (2026-09-12)

**Commit.** `70d3892d`, "making a modem only path for decoding later."
Added `MODEM_WAVEFORM_RECEIVER.md`, `animation_modem/waveform_receiver.py`
(+205 lines) and `modem_tests/test_waveform_receiver.py`; the same lineage also
carried `animation_modem/progressive.py`, `reference_receiver.py` and
`presentation.py`.

Three alternative receiver architectures on the *unchanged* V2 wire:

| File | Idea |
|---|---|
| `waveform_receiver.py` | Direct waveform-domain receiver; documented in `MODEM_WAVEFORM_RECEIVER.md` (47 lines). |
| `progressive.py` | "Live, one-pass symbol reception with bounded-rate partial image snapshots. Reuse transport2 acquisition/wire format. No trial image decoding or EQ replay. Only new symbols are transformed. A new packet owns a fresh coefficient buffer." Transforms each arriving symbol once, accumulates into a coefficient buffer, and emits bounded-rate partial pictures — a *streaming* decoder rather than a per-packet trial decoder. |
| `reference_receiver.py` | A deliberately naive reference implementation for cross-checking. |

**Outcome.** None of the three was ever selected. `97570900` deleted
`progressive.py` (216 lines) and `decoder.py` the same day V3 landed. Their
ideas survive in spirit: "no trial image decoding" is exactly what V7's live
input adapter does (it decodes only the newest complete packet), and the
per-symbol-once discipline is V7's per-cell MMSE pipeline.

**Why it matters for the version story.** It is the project's one recorded
attempt to fix the *architecture* (receiver loop) instead of the *algorithm*
(acquisition), before V3 chose the algorithm.

---

## 7. V3 — edge-counted acquisition (the pivotal change)

**Commits.** `b55ef9b4` (2026-09-13) "modem v3: countable LTC-style preamble,
edge-gated acquisition"; `96cad679`; `7e664932` (2026-09-16) profile codes +
magic `b'V3'`; `97570900` same day — deleted `transport.py`/`transport2.py`
and created `animation_modem/core.py` (570 lines) as the shared
geometry/coder/decoder module.

### 7.1 The idea

V2's preamble was a random-phase noise burst. Its edge intervals have no
structure, so finding it means correlating against a bank of resampled
templates. V3 replaced it with an **LTC-style biphase-mark word**, so edge
intervals take exactly two values (short = 8 samples, long = 16) and playback
speed falls out of *measured / nominal* the way a timer capture gives it to an
LTC reader.

```
HALF              = 8        # samples per half-bit: '1' -> 3 kHz, '0' -> 1.5 kHz
PREAMBLE_BITS     = (0,0,0,0,0,0,1,0,0,1,0,1,1,1,0,0)
PREAMBLE_AMPLITUDE= 0.55     # down from v2's 0.65
EDGE_HYSTERESIS   = 0.12     # Schmitt threshold, relative to preamble amplitude
```

Three design decisions in that block are worth reading as decisions:

1. **The bit word was chosen by exhaustive search** over balanced 16-bit words
   for the lowest autocorrelation sidelobe: **0.312**, against 0.531 for an
   arbitrary word. V2's noise burst reaches 0.080 — so the correlation
   *fallback* is more sidelobe-prone here. "The edge gate and the header CRC are
   what disambiguate."
2. **Amplitude dropped to 0.55 deliberately.** The preamble held the packet peak
   in 73 of 100 measured frames under V2, which meant any level chosen to keep
   it from clipping buried the picture **7.7 dB** further down than necessary.
3. **Hysteresis band, not a bare sign test.** `edge_intervals` Schmitt-triggers
   at ±0.12·amplitude "because a hysteresis band … is what keeps hiss from
   manufacturing edges near the zero crossings."

### 7.2 How speed is measured

```
NOMINAL_GAPS = diff(NOMINAL_EDGES)        # the two-value nominal pattern
MIN_RUN      = len(NOMINAL_GAPS) - 4     # tolerate a few corrupted edges
_runs(gaps, tolerance=0.28)               # maximal spans of consistently short-or-long intervals
```

Two subtleties are documented and both matter:

- **Run, not buffer statistics.** *"The preamble is followed immediately by
  OFDM, whose zero crossings are dense and irregular. Taking statistics over a
  whole buffer lets those crossings drag the estimate — measured directly, two
  body edges moved the apparent span from 224 to 266 samples."*
- **Unit from the minimum, not the first gaps.** *"A word can open with a run of
  long ones — this one starts with five. Estimating from the first few gaps
  therefore locks onto 2·unit and rejects the real word at the first short
  interval."*

### 7.3 Three-tier acquisition

```
_acquire():
  1. COAST   — transmitter emits contiguous frames; verify the predicted
               preamble at the predicted offset (cheap). If not yet arrived,
               return None *without* inflating search_after, which would
               silently disable coasting.
  2. EDGE CAPTURE — the common, cheap path. measure_speed() per channel, then
               _fit_preamble() with reach=0, 3 Newton iterations.
  3. CORRELATION — decimated fallback for signal too weak to count edges on.
               Fired zero times in every clean-signal benchmark.
```

The scale sweep was removed entirely: *"Edge capture lands within ~0.3%, well
inside the Newton loop's basin, so no scale sweep is needed — that sweep was
7 scales × 2 channels of correlation and was the bulk of acquisition cost."*

**Result: speed exact to 0.3% across 0.5–2.0× at 26 µs per buffer, against
5543 µs for the bank.** ~210× cheaper.

### 7.4 Two smaller but important V3 fixes

- **The preamble now sits on both channels.** V2 left channel 1 silent through
  it.
- **`encode()` normalises up as well as down.** V2 only ever attenuated, leaving
  a payload already ~21 dB below full scale a further 2.8 dB down. This alone is
  worth **3–4 dB PSNR** and, at noise 0.03, takes header verification from
  **12/30 to 30/30**.

### 7.5 The compatibility contract V3 established

> *"The body is byte-identical to v2, so transport2.decode_packet demodulates a
> v3 packet unmodified. The preambles differ, so the two versions cannot
> acquire each other's audio; two tests assert this so it fails loudly rather
> than half-decoding."*

This is the project's first explicit version-incompatibility test, and the
pattern persists: V6 and V7 both have distinct magics (`V6`, `VR`, and the V7
pulse-word profiles) precisely so a stale receiver fails loudly.

Test suite: **122 → 150**.

### 7.6 The single consolidated wire

After V3, the five V2 presets were deleted ("the old preset table is gone").
From `7e664932` onward there is exactly one layout, `WIRE`, with a long comment
explaining every choice:

```
WIRE = Layout(top_bin=54, image_symbols=14, name='wire',
              progressive=True, orthogonal_training=True,
              spread_carriers=True, dense_header=True, header_width=20)
# 3,168 packet / 3,200 frame samples, 15.000 fps, 3,280 capacity values
```

- **`top_bin=54`** (375–20,250 Hz) "chosen by the EQ+noise sweep".
- **spread carriers** so "the image planes ride the middle of the band and EQ
  tints nothing".
- **LOW *and narrow* header** (`header_width=20`, four header symbols) "measured:
  widening to 27 drops treble+noise identity from ~30/32 to ~19/32". A full-band
  header (the old `header_width=50`) is faster but loses identity under a 15 kHz
  lowpass or 2× playback; a split header was measured strictly worse still.
- **`dense_header`** — the header symbols' idle carriers still carry image,
  "which pays for dropping to 14 image symbols while holding the full
  2880-coefficient color-dct picture".

---

## 8. V4 — modular engines and a Haar wavelet coder

**Commit.** `467cbaf2`, 2026-09-18, "v4 modular engine: wavelet transform +
engine registry".

### 8.1 What changed

Exactly one thing conceptually: **the source coder became pluggable, and a
wavelet became one option.**

| Change | Detail |
|---|---|
| `animation_modem/engines.py` | New `Engine` registry: `v3` (DCT) and `v4` (wavelet), with `get_engine`, `engine_names`, `coder_for`, `coders_for`. |
| `animation_modem/wavelet.py` | `WaveletCoder`: orthonormal 2-D **Haar** DWT, one level. |
| `PROFILE_CODES` | Extended to four entries: `color-dct`, `lean-dct`, `color-wavelet`, `lean-wavelet`. |
| CLI | `--codec v3|v4` wired through `modem_v3_check.py`; `modem_screen.py` dispatches on profile. |

The wire is untouched: "Same wire, same slot count, same power-allocation
machinery, same header — only the transform inside the coder changes." Haar is
chosen because it is **orthonormal** (Parseval holds, so V2's allocation and
Wiener de-bias still apply unchanged) and "trivial to invert exactly with
periodic boundaries". Coefficients are kept by a **measured fixed mask**
(`MASK_LL_Y` and friends, large literal boolean arrays that a reimplementation
must copy rather than re-infer).

### 8.2 Measured A/B (from the commit message)

| | clean | at −20 dB noise |
|---|---:|---:|
| v3 (DCT) | **27.4 dB** | **0 / 8 frames acquired** |
| v4 (Haar) | 24.2 dB | **8 / 8 frames acquired** |

All 246 `modem_tests` pass (1 expected failure).

### 8.3 The finding

**The wavelet trades ~3 dB of clean quality for total immunity at −20 dB
noise.** That is the first appearance of the project's central trade-off —
clean fidelity versus impairment tolerance — and it is what motivated V5: a
better transform that might not have to pay the full clean-quality penalty.

---

## 9. V5 — CDF 9/7, redundancy copies, and a parked LDPC path

**Commits.** `34ae7739` (2026-09-18) "v5: CDF 9/7 DWT profile (hd-dwt), stereo
wire compat"; `61c40e95` (09-19) invertible CDF 9/7; `fb50cbe4` (09-19) fit to
the wire budget; `2b6bb9da` (09-19) truncated DWT at the baked 80×96;
`4c91b70c` (09-19) noise-resistant v5.

### 9.1 The active V5

**V5 is not a new packet format.** `V5Engine` selects `WIRE_HD`, declares
profile code 2 (`hd-dwt`), and calls the ordinary V3 `encode()` and
`Receiver`. The V7-spec audit is explicit: *"V5 is not a separate packet
geometry in the active path."*

```
WIRE_HD = Layout(top_bin=54, image_symbols=16, name='wire-hd',
                 progressive=True, orthogonal_training=True,
                 spread_carriers=True, dense_header=True, header_width=20)
# 3,456 / 3,488 samples, 13.761468 fps, 3,680 capacity
```

- **Two-level CDF 9/7 lifting DWT** on the 80×96 luma / 40×48 chroma grids.
  JPEG 2000 lifting constants α −1.586134342059924, β −0.052980118572961,
  γ 0.882911075530934, δ 0.443506852043971, K 1.230174104914001; boundary
  extension mirrors rather than wraps.
- **Half-resolution pyramid retained**: 1,920 luma + 480 Cb + 480 Cr
  originals, **then 800 copies**, filling all 3,680 slots. This is V5's
  redundancy idea: the wire is fully spent, so protect the important values by
  sending them twice.
- Fixed band-RMS gain table, `(LL, LH, HL, HH)`: Y `(1.80, .457, .457, .303)`,
  Cb `(0.871, .093, .093, .058)`, Cr `(0.891, .135, .135, .085)`.
- Reliability inverse with a **luma-first / chroma-strict recovery gate** — the
  first explicit colour-priority floor, and the direct ancestor of V6's
  `recovery_floor` and V7's tiers.

### 9.2 Three real bugs the follow-ups fixed

`61c40e95` is worth reading in full because it documents a class of error:

> *"The v5 encode was broken: the CDF 9/7 lifting lifted each polyphase against
> itself (non-invertible, 1D round-trip RMSE ~3.9) and the oversampled 96×112
> grid plus redundant pyramid could not carry an invertible transform (any
> truncated pyramid is undecodable by construction)."*

Fixed by cross-polyphase lifting per JPEG 2000 (round trips now ~1e−16) and a
**non-redundant** pyramid pack: deepest level keeps LL/LH/HL/HH, shallower
levels keep LH/HL/HH only, and `cdf97_inverse_2d` reconstructs each level's LL
from the coarser level.

`fb50cbe4`: the "2000 mono capacity" was a stale hand estimate; refitting to
the *real* 3,680-slot budget moved the picture from 40×32 to 56×40.
`2b6bb9da`: dropped `grids == shapes` and decoded at the baked 80×96 with a
truncated DWT instead.

### 9.3 The parked LDPC path

A completely separate implementation shipped alongside and was **never wired
up**: `animation_modem/transport3_v5.py::ReceiverV5` plus `fec.py`.

| Property | Active V5 | Parked `ReceiverV5` |
|---|---|---|
| Preamble | V3 16-bit biphase | `PREAMBLE_BITS_V5` 12-bit |
| Channels | stereo | **mono** |
| Header | V3 16-byte | `HEADER_BYTES_V5=14`, `HEADER_SLOTS_V5=72`, `'>2sIHHI'`, magic `b'V5'` |
| Protection | stereo-repeat copies | **LDPC** |
| Selection | `V5Engine` | *never selected by `coder_for()`* |

`fec.py`: systematic LDPC, `LDPC_N=11520`, `LDPC_K=10000`, `M=1520`,
`DV=3`, `DC=6`, rate ~13/15 (87%), sparse `H` built by circulant blocks,
decoded by **belief propagation** (`ldpc_decode_belief_prop`), plus
`pack_coefficients_to_llrs` / `unpack_llrs_to_coefficients`.

The spec's audit is blunt about the confusion this caused: *"That path is
**PARKED / NOT ACTIVE V5** in the engine registry and must not be mistaken for
the wire-hd V5 format."* LDPC is an **error-correcting-code** approach; it was
tried and dropped in favour of analog repetition, because a *hard* code needs a
reliable erasure model and an analog slot's reliability is continuous.

### 9.4 What V5 added

- A **better transform** than Haar (CDF 9/7 beats it on clean quality while
  keeping the noise advantage).
- **Redundancy as full-duplex of importance**: with every slot used, protect the
  top coefficients by duplication.
- **Colour-priority recovery gating**, which became a permanent project value.
- The `d3a9cbad` (09-19) **tape reallocation**: `wire-tape` and
  `wire-tape-25` move the payload below 12.75 kHz and declare a 14 kHz
  whole-waveform ceiling:

| Layout | band | image sym | packet/frame | fps | capacity | coder |
|---|---|---:|---|---:|---:|---|
| `wire-hd` | 375–20,250 Hz | 16 | 3,456/3,488 | 13.761 | 3,680 | CDF 9/7 + 800 copies |
| `wire-tape` | 375–12,750 Hz | 12 | 2,880/2,912 | 16.484 | 1,600 | 800 DCT + 800 stereo copies |
| `wire-tape-25` | 375–12,750 Hz | 5 | 1,872/1,904 | 25.210 | 760 | 360 DCT + 360 copies |

The 14 kHz ceiling is the first explicit statement that **the tape, not the
modem, is the bandwidth limit** — and it survives into V6 and V7.

---

## 10. V6 — back to analog on a tape-band wire

**Commit.** `4a672b07`, 2026-09-20, "Build V6 analog tape transport", plus
`914756a2` (full-repeat control), `3b250ae3`/`46df8a25` (tape-ordered
placement), `f318880d`, `37b93447`, `960fddf5`, `bfd03edd`.

`MODEM_TODO.md` at `4a672b07` is the project's own honest post-mortem and is
worth quoting:

> "Return to **analog-valued image coefficients**, keep the strongest timing and
> recovery infrastructure, and validate **one tape-band wire against historical
> V2**. We have useful components, but previous experiments changed too many
> variables at once and some conclusions were wrong."
>
> "V1/V2 are not wavelet codecs. V1 sends image-plane values; V2 applies DCT and
> power allocation. The claim that we were restoring a 'V1/V2 wavelet path' was
> incorrect. Wavelets came later."
>
> "The digital detour imposed an unnecessary bottleneck. The 192-byte frame
> budget drove severe compression and frame-loss thresholds. VP8/JPEG 2000
> solved a different problem from the intended analog wavelet transport."

### 10.1 How V6 worked

```
WIRE_V6 = Layout(top_bin=34, image_symbols=29, name='wire-v6',
                 progressive=True, orthogonal_training=True,
                 spread_carriers=True, dense_header=True, header_width=20,
                 emission_ceiling=14000)
# 5,328 packet / 5,360 frame samples, 8.955224 fps, 3,640 capacity, magic 'V6'
```

**V6 changed source coding and protection only. Acquisition and modulation were
V3's, unchanged.** That is the whole point: it is a controlled A/B of *coding*
against a *fixed* wire.

**Geometry.**

| | Values |
|---|---|
| `V6_GRIDS` | `(96,80), (48,40), (48,40)` = 11,520 source samples |
| `V6_SHAPES` | `(48,40), (24,20), (24,20)` = 2,880 analog originals |
| `FOUNDATION_SHAPES` | `(24,20), (12,10), (12,10)` = **720** foundation values |
| Foundation content | 24×20 luma + 12×10 Cb/Cr low-frequency corners |
| Transmitted | 2,880 originals + 720 copies = 3,600 of 3,640 slots ("Forty spare values are deliberately left unused rather than changing one transform's budget") |

**`ProtectedAnalogCoder`** is V6's core idea. It wraps either transform (DCT via
`SourceCoder`, or two-level CDF 9/7 with the deepest LL bands as foundation)
and:

1. removes base gains, appends the 720 foundation values, applies a combined
   gain table (`gains = sqrt(full/mean(full))`, renormalised);
2. on decode, either averages clean copies or performs **reliability/noise
   weighted soft fusion**: `a = gain·rel`, `precision = a²/noise`, posterior
   coefficient `σ²·Σ(a·y/noise) / (1 + σ²·Σ(a²/noise))`;
3. applies **confidence floors** — protected luma 0.05 / chroma 0.15;
   unprotected luma 0.45 / chroma 0.60. Gate
   `clip((confidence − floor)/(0.85 − floor), 0, 1)`.

Those floors *are* the project invariant "degrade detail before structure,
structure before colour", written as code.

**Placement (baseline).** Originals go in coefficient-rank order into the first
2,880 slots from `_slot_order()`. Each foundation copy is assigned to the
**opposite channel**, **at least 8 carrier bins away**, and preferably a
**different OFDM symbol** — solved as a SciPy linear-assignment problem.

**Placement (tape-ordered, bench-only).** Same wire, same values, different
placement: carrier bin 1 (375 Hz) is demoted with health 28 rather than
forbidden, minimum copy spread drops to 7 bins, slots are ordered low-to-high
carrier health with a coprime time walk, foundation homes stay **at or below
4 kHz**, copies stay **at or below 7 kHz**, and every copy is cross-track, ≥7
bins away and ≥ half an image-symbol block away.

**V6-full-repeat control** (`wire-v6-repeat`, `914756a2`): `top_bin=34`,
47 image symbols, 7,952-sample frames, 6.036 fps, 5,800 capacity, magic `VR`,
`copy_of = arange(2880)` so *every* coefficient is protected.

### 10.2 What V6 added

- **Foundation duplication as a first-class, tested mechanism**, with explicit
  invariants (opposite channel, separated frequency, different symbol) enforced
  in `modem_tests/test_v6.py`, `test_v6_repeat.py`, `test_v6_tape.py`.
- **Soft fusion with confidence floors** — the prototype for every later
  confidence gate.
- **A 14 kHz whole-waveform ceiling** and a genuinely tape-band carrier band
  (375–12,750 Hz).
- **Honest measurement discipline**, codified in `MODEM_TODO.md`: never zip
  surviving frames against sequential references; count *played* frames, not
  queued; report luma/chroma separately; record noise definitions so different
  transmit amplitudes aren't mistaken for source-coder robustness; **synthetic
  channels are not tape validation**.
- **A recorded negative result.** Full-repeat "improved some isolated-track and
  severe synthetic cases but regressed DCT MP3, wow/flutter, hiss, and several
  wavelet low-pass cases; real tape remains required."

### 10.3 What V6 could not do

It was slow (8.96 fps) and it fixed nothing that V7's new wire did not fix
better. It is best read as **the controlled experiment that produced V7's
requirements**, not as a shipping generation. `MODEM_TODO.md` also shows the
synthesis target: "Keep the baked 80x96 image/reconstruction target, but report
actual transmitted detail and effective resolution honestly. An 80x96 output
array alone does not prove 80x96 picture detail."

---

## 11. V7 — the current transport

**Start.** `0fdc309d`, 2026-09-20, commit subject literally **"v7?"** — a
675-line draft spec plus a bench-only prototype (`tools/v7_proto.py`,
`tools/v7_bench.py`, `tools/v7_media.py` with tape / MP3 320k / V0 / ATRAC1 SP /
ATRAC3plus damage models). This is a genuinely new design, not a V6 increment.
**Promotion.** `80f8390c` (09-21) into `animation_modem/`; `c8a98fb4` (09-21)
application sender and source metadata; `53cc384b` (09-21) legacy deletion.
**Today.** `modem-v7-integration` at `f3cf11c9` (2026-10-02), 183 commits after
`53cc384b`.

### 11.1 What the draft spec proposed, and what actually shipped

The draft (`0fdc309d:docs/transport_v7_spec.md`) proposed a **continuous
72-bit clock/identity track** (§5) as the timing reference. It was not built.
Current §1 states plainly:

> "The live wire uses the edge-counted pulse acquisition retained in
> `animation_modem/transport3.py`; it does not use the continuous 72-bit clock
> track proposal described in older sections of this document."

V7 chose V3's pulse framing. Everything else in the draft shipped. The module
docstring still advertises the clock track ("clock/identity track (§5) … and
the receiver chain of §9 (edge-counted clock, …)") — a reminder of how much of
the draft was implementation.

### 11.2 Packet format (current)

| Region | Samples | Description |
|---|---:|---|
| Pulse header | 288 | Edge-counted biphase-mark preamble |
| OFDM body | 3,456 | 24 symbols × 144 |
| Metadata symbol | 144 | One OFDM symbol, 40 protected bits |
| Guard | 32 | Packet endpoint; final 24 carry the EOF marker; tones continue |
| **Total** | **3,920** | **12.245 packets/s at 1×** |

Constants: `RATE, N, CP, SYM, F = 48000, 128, 16, 144, 24`;
`FRAME = F*SYM`; `PULSE_FRAME = PULSE.SYNC_LEN + FRAME + META_SYMBOL + 32`.
Bins 4–34 = 1.5–12.75 kHz; 63-tap Kaiser FIR with a nominal 14 kHz emission
edge.

**Pulse header — inherited from V3, extended by V7.** `SYNC_LEN=288`, `HALF=8`,
`PREAMBLE_BITS=(0,0,0,0,0,0,1,0,0,1,0,1,1,1,0,0)`, `PREAMBLE_AMPLITUDE=0.55`,
`EDGE_HYSTERESIS=0.12`, Numba `_pulse_word_kernel`, `measure_pulses()` and
`measure_pulses_both()` (forward + reversed gap patterns in one Schmitt pass).

**V7's addition: the profile lives in the pulse word.** Six constant-weight
16-bit words, every one with the *same first/last edge and the same edge count*
so pulse timing geometry is unchanged — the interval pattern carries the
identity, and pairwise Hamming distance is ≥ 6:

```python
PROFILE_PREAMBLE_BITS = {
  0: Fold-off / aspect-fold-500
  1: Fold-500                       # the historical V7 word
  2: Fold-1000
  3: Mono-off / aspect-mono-500
  4: Mono-500
  5: Mono-1000
}
```

Because the metadata word has no spare profile bit, the *wire profile identity
is carried by the preamble itself*. This is the V3/V6 "fail loudly, don't
half-decode" principle applied again, and it is why `AGENTS.md` can insist
Fold-500 stereo M/S must be preserved: change the word and legacy tape stops
acquiring.

### 11.3 OFDM body and stereo M/S

- 128-point real FFT/IFFT per symbol; bins **4 and 34 are continual pilots**;
  scattered pilots on **5, 9, 13, 17, 21, 25, 29, 33**, one of three symbol
  phases each; **79 data blocks** after pilot reservation.
- Each cell carries mid/side: `L = (M + S)/√2`, `R = (M − S)/√2`.
- **The 13 blocks on bins 5–9 are M-only**; the other 66 carry M and S, each
  with I and Q. So a mono sum retains M and loses S.
- The receiver retries once with the **right leg inverted** when ordinary
  acquisition finds nothing, and represents mono input as a *shared M
  observation* rather than inventing an S channel.

This is where `AGENTS.md`'s "a single raw M/S leg is not ordinary mono" comes
from, and §13.1 quantifies the cost: today's stereo body has 1,264 M and
1,056 S slots, but the M-first rank map means **1,616 of the 2,880 corner
coefficients are never recovered by a mono sum** (pinned by
`modem_tests/test_v7_mono.py`).

### 11.4 Image coding and the three tiers

Source canvas 80×96. V7 samples Y on a 96×80 grid, Cb/Cr on 48×40. The
**transmitted** DCT corners are 48×40 Y + 24×20 each chroma = **2,880
coefficients**, `dctn`/`idctn` with `norm='ortho'`, from `v7_core.SourceCoder`.

Coefficients are ranked by a **frozen variance table**
(`animation_modem/v7_model_tables.npz`, SHA-256 pinned to
`2392943a287fdf1cd2ac773c2fc065179006bfab4646726e72e023f3921ffb1c`) and split:

| Tier | Ranks | Count | Carriage |
|---|---|---:|---|
| Head | 0–207 | 208 | M-only blocks |
| Body | 208–2,223 | 2,016 | stereo-capable blocks |
| Tail | 2,224–2,879 | 96 of 656 | **rotated over seven slices**, counter mod 7 |

**2,320 slots per packet; all 2,880 ranks refreshed over seven packets.** The
receiver's `TailStore` retains received tail values until their slice refreshes,
then falls back to the model mean (`--no-tail-memory` disables). This is V7's
answer to the latency/coverage tension — partial refresh over time instead of a
bigger packet. Each component group of eight is spread across symbols with a
normalised 8-point **Hadamard** transform, and **all M slots precede S slots**,
so mono playback drops a low-priority suffix of the rank sequence rather than
holes throughout the picture.

The canonical encoder profiles are `nearest`, `box`, `lanczos`, `bicubic`;
**only `nearest` and `box` are named on the wire**, so live senders use those.

### 11.5 Metadata symbol

11 known pilots on bins 4, 7, 10, … 34; 20 QPSK data cells for a **five-byte**
word:

| Bits | Meaning |
|---|---|
| b0 7–5 | Aspect code (`V7_ASPECT_RATIOS`) |
| b0 4 | **Screen bit** — aspect code is the sender's fixed layout and the picture is boxed inside it |
| b0 3 | Source model: 0 nearest, 1 box |
| b0 2–0 | Tail slice 0–6 (7 invalid) |
| b1–2 15 | Playback direction (0 up, 1 down) |
| b1–2 14–0 | One-based source index (0 invalid, max 32,766; wraps at 32,767 ≈ 44.6 min) |
| b3–4 | CRC-16/CCITT-FALSE of bytes 0–2 XOR a rotating mask |

Slices 0–4 carry an ordinary CRC; **slice 5 XORs it with loop phase `p` and
slice 6 with loop length `N`** (top bit of length = one-way). A correct clock
computes `ticks = unix_ns × 30 // 1e9` and derives loop index and picture lag.

An invalid metadata word **does not** update the selected source, aspect or
tail slice. Known limitation, documented: the loop-lock "counts matching
observations for a tail slice but does not require distinct packet counters",
so distinct-packet confirmation is not guaranteed.

### 11.6 Timing tones, speed range, and splices

Optional reference tones on **bins 1 and 3, bin 2 clear**, M-only, **−20 dB**
relative to body RMS, continuous through the packet. Separate from the OFDM
pilots. Receiver timing options: `baseline`, `tone-seeded` (standalone live
default), `tone-joint`, `tone-replaced`; `m-reference` tone equalisation and
pulse-warp timing are opt-in and **off by default**.

**Speed range 0.01×–4×** (pulse scales 100×–0.25×). Slower playback reduces
packet rate and carrier frequencies together; faster increases both. Nyquist
guide `output_sample_rate / 28,000` (1.71× at 48 kHz, 3.43× at 96 kHz). **Digital
time-stretch and pitch-shift splices** are handled separately, because a
resampled speed change scales time and pitch together while a splice is a
splice.

### 11.7 EOF framing — V6's lesson applied

- **`baseline`** commits a packet after the *next* pulse header arrives at the
  expected spacing. A finite stream's final packet is therefore never
  committed.
- **`eof`** validates a 24-sample alternating-polarity marker inside the
  existing 32-sample guard — four runs of 4, 8, 6, 6 samples at ±0.55, level
  detection threshold 0.08, search fraction 0.012 — and can commit the final
  packet without a following header. Missing, truncated or damaged markers do
  not commit.

EOF changes **neither packet length nor frame rate**. Live senders and receivers
enable it by default; `--no-eof-marker` / `--no-modem-eof-marker` are retained
for legacy-wire comparisons; the low-level `encode_pulse_frame()` /
`decode_pulse_stream()` APIs still default to the older framing for unit tests.

Known limitation, honestly documented: the live input adapter still waits for a
new header before decoding, "so a finite live input ending immediately after an
EOF marker can leave its last packet unprocessed. The live-input test
explicitly expects the last packet to be omitted when no following header
arrives."

### 11.8 Decoder behaviour and acceptance gates

Per-cell 2×2 MMSE equalisation, grouped 8-value LMMSE reconstruction,
coefficient confidence gates, per-symbol fade/noise model, all on fixed-size
128-point real FFTs. `f3139741` (09-22) records the fix and its numbers:

> "Per-cell 2×2 MMSE computed `S⁻¹(P Hᴴ)` instead of `P Hᴴ S⁻¹`. That is only
> right when a cell's M/S priors are equal; fixing it brings the crosstalk and
> track-imbalance torture cases back to clean quality."
> "Order stereo slots with S on carrier b as if on b+6, so mono playback loses
> the less important ranks (mono RMSE 0.113 → 0.091, equal to the structural
> limit; clean stereo unchanged)."

**Live acceptance gates** (strict): pulse confidence ≥ 0.45, metadata accepted,
head confidence ≥ 0.85, head coverage ≥ 0.75, max pilot-noise estimate ≤ 0.08.
A weaker but displayable reconstruction is still shown and labelled degraded.
The display path **holds its previous picture when a result is neither accepted
nor displayable; it has no black-frame fallback.**

### 11.9 Folds — trading SNR for resolution

V7's most distinctive mechanism, and the explicit continuation of V4's
trade-off (§8.2) with numbers.

The observation: on a clean path each slot arrives with far more precision than
the picture needs, while detail beyond the 48×40 Y corner is not sent at all.
For a *linear* analog code, sending the highest-variance coefficients is already
mean-square optimal — so no rearrangement helps. A **nonlinear** mapping can.

```
s = D·round(h/D) + β·clip(u, −2.5, 2.5)          β = 0.8·D/5
s = s / sqrt(1 + D²/12 + β²)
```

The **M weakest Y slots of the body tier** each carry two Y coefficients: a
**host** sent coarsely as a multiple of step `D`, and a **guest** — the most
important unsent Y coefficient — riding inside the step as an analog residual.
Slot power, slot layout, Hadamard spreading and the equaliser are all
unchanged, so a normal V7 encode of modified coefficients *is* a folded packet.

The shipped M=500 table additionally **compands** the guest:
`A·sign(u)·ln(1 + μ·min(|u|,U)/U)/ln(1+μ)` with `U=12, μ=4, A=0.4D`, and the
receiver drops the guests when measured signature-slot noise passes 0.15.
Host slots are **Y only**, ranks 208–2,223 — never the head, never the rotating
tail, never Cb/Cr. Folding into Cb/Cr was measured and **excluded** (colour
error CIEDE2000 3.4 → 7–11).

Measured (SSIMULACRA2, higher better, still pictures, 1×, box filter):

| Condition | No fold | Fold 500 | Fold 1,000 |
|---|---:|---:|---:|
| Clean | −16.9 | −7.0 | **−4.1** |
| Low-pass 12 kHz | −17.0 | −7.0 | **−4.3** |
| Low-pass 10 kHz | −17.0 | −7.1 | **−4.3** |
| Dropouts (12 ms / 0.7 s) | −17.3 | −7.6 | **−4.7** |
| Wow/flutter 0.45% @0.55 Hz + 0.12% @7.3 Hz | −17.7 | **−10.3** | −10.4 |
| Speed jitter 0.1% RMS 20–300 Hz | −19.5 | **−16.3** | −18.3 |
| Fast flutter (+0.1% @25 Hz, 0.05% @60 Hz) | −19.4 | −19.8 | −21.5 |
| Fast flutter + 12 kHz LP + dropouts | −24.0 | −24.5 | −26.8 |
| Random speed jitter 0.3% RMS | −33.0 | −43.1 (**−34.3** w/ fallback) | −49.8 (−43.4) |

**Recommendation on record: M = 500 with the fallback.** Low-pass and dropouts
don't affect folding at all (the Hadamard spreading and equaliser absorb them
before the folded symbols are read); flutter and jitter hurt, because timing
smear moves symbols across a step — which is precisely an FM-style threshold
effect, and the fallback's only measurable job is the 0.3% jitter case.
"Colour is unchanged, and on bad paths only detail degrades."

### 11.10 Mono enhancement and stereo slices

Mono wires (§13, 2026-09-27) remove exact S cancellation by using an
identical-leg mono V7 wire with 1,264 plain slots and no tail rotation; a
one-packet probe learned `p` from two reads of the *same* packet, so
distinct-packet confirmation is documented as unguaranteed. Mono requires an
updated receiver; "an unknown status or changed CRC mask must not be advertised
as making old receivers hold. They may display incorrectly assigned
coefficients."

**Stereo slices** (`f3cf11c9`, 2026-10-02, HEAD) is the newest design, and it
was rewritten in place after `97d3ea07`. The first version split the picture
**spatially** — each channel carried one checkerboard half of a luma and colour
rectangle, and a single channel filled the missing samples from their
neighbours. The current version splits the picture **in the frequency domain
and recombines it linearly**:

> "Per plane the picture's coefficients nearest DC in picture frequency (the
> aspect layouts' ordering) are cut in two: a *base* (1,020 luma, 110 per
> colour plane) and, beyond it, as many *detail* coefficients. Every base
> coefficient is paired with one detail coefficient, strongest base with
> weakest detail, and one slot carries both:
>
>     left  slot = base + DETAIL * detail
>     right slot = base - DETAIL * detail
>
> * both channels: the sum gives the base, the difference the detail: the whole
>                  picture (2,040 luma coefficients);
> * a mono sum:    the detail cancels: the base, a complete lower-detail
>                  picture, exactly;
> * one channel:   the base with the detail riding on it. DETAIL below 1 keeps
>                  that disturbance small, and the pairing keeps it away from
>                  the coefficients that matter most."

`DETAIL = 0.35`; colour is the base alone, identical on both channels, so the
colour shown does not depend on how many channels arrive (`SLICE_CHROMA_DETAIL=1`
splits colour too). Eight marker slots carry a fixed pattern sent positive on
the left and negative on the right, so a mono sum reads zero; the receiver
averages each marker over packets, so swapped cables and single bad packets do
not change what a channel is taken for. The markers also **measure crosstalk**,
which is then removed (`r = (1−e)/(1+e)`). The receiver estimates base from the
two channels' mean and detail from half their difference, each against the
model's per-coefficient variance, and **refuses to join a damaged channel with
a clean one**. `mono-slices` was **removed** in the rewrite; the profile now
requires two output channels and has no pixel grid.

An optional fold (`SLICE_GUESTS`, off as built) would give a lone channel some
detail by folding the strongest detail coefficients into the weakest base slots,
companded identically on both channels; it is off because "it costs the
two-channel picture more than it gives one channel."

Measured against Aspect Fold 500 with the fixed tail, as change in picture
error (`100 − SSIMULACRA2`, negative is better):

| Condition | Δ | Condition | Δ |
|---|---:|---|---:|
| clean | +0.7 % | type II tape | +0.6 % |
| left channel only | −29 % | soft saturation | −2.9 % |
| right channel only | −31 % | wow and flutter | +0.3 % |
| mono sum | −1.9 % | 12 kHz low-pass | +1.1 % |
| hiss −40 dB | +0.1 % | MP3 320 | −2.5 % |
| hiss −35 dB | +0.6 % | MP3 192 | −24 % (31/32 frames vs 27) |
| type I tape | −0.3 % | | |

One channel alone is **2.3 % worse** than `aspect-mono-500` on that channel,
which folds 500 more coefficients; folding this wire too closes that gap (250
guests +1.1 %, 500 guests −0.4 %) but costs the two-channel picture more
(clean +2.7 % and +4.3 %), so it is built without. Tables are
`v7-stereo-slices-3`, SHA-256
`991c133d81816ee77f5b6591210fd589c05b290ddf0566a78f1f704e726fe041`.

### 11.11 Runtime posture

Numba is imported unconditionally by `v7.py` and the pulse modules; the live
receiver calls `warmup_equalizer()` before opening the stream so JIT
compilation cannot stall capture. Defaults:

| Endpoint | Default |
|---|---|
| Standalone sender `tools/v7_live.py` | coded M=500 fold, box, brightness 1.0, gamma 1.0, 1×, EOF, coded pilots |
| Standalone receiver | matching coded M=500, EOF boundaries, tone-seeded pilot timing, baseline pulse timing, tone equalisation off, decode batch 1, history 1, tail memory on |
| Application `main.py --mode modem` | stereo **Fold 500**: pinned Box model, coded pilots, EOF, 1× (`--modem-baseline` restores fold-off + nearest + steady pilots) |
| GUI profiles | Aspect Fold 500 (default), Fold 500, and mono aspect-colour-500 / colour-500 |

**Audio-device recovery** is explicitly documented as *software behaviour, not
real-tape validation*: explicit PortAudio device identities; a stopped sender
pauses transmission, a stopped receiver pauses decode while holding its last
good picture; "Recovery never silently selects an unrelated default device";
identity and sample rate must be stable for five 30-fps intervals (~167 ms)
before reopening; the reopened stream's negotiated rate is authoritative.

---

## 12. Comparison tables

### 12.1 Wire geometry

| | V1 | V2 | V3 `wire` | `wire-hd` (V5) | `wire-tape` | `wire-v6` | **V7** |
|---|---|---|---|---|---|---|---|
| Frame samples | 3,200 | per layout | 3,200 | 3,488 | 2,912 | 5,360 | **3,920** |
| fps @48k | 15.000 | per layout | 15.000 | 13.761 | 16.484 | 8.955 | **12.245** |
| Carriers | 3–54 | 3–top_bin | 4/5–54 | 4/5–54 | 4/5–34 | 4/5–34 | **4–34** |
| Band @48k | 375–20,250 Hz | preset-dependent | 375–20,250 Hz | 375–20,250 Hz | 375–12,750 Hz | 375–12,750 Hz | **1,500–12,750 Hz** |
| Emission ceiling | — | — | none | none | 14 kHz | 14 kHz | **14 kHz (Kaiser FIR)** |
| Preamble | 256-sample random, 7 templates | 29-scale sync bank | **16-bit biphase, countable** | 16-bit biphase | 16-bit biphase | 16-bit biphase | **16-bit biphase × 6 profiles** |
| Channels | stereo | stereo | stereo | stereo | stereo | stereo | **stereo M/S (+ mono wires)** |
| Capacity | 2,880 | per layout | 3,280 | 3,680 | 1,600 | 3,640 | **2,320/packet (2,880 over 7)** |
| Magic | `SI01`/`SI02` | `V2`/`V3` | `V3`→`V4` | `V4` | `V4` | `V6`/`VR` | **pulse word + metadata CRC-16** |

### 12.2 Source coding and protection

| | Transform | Values carried | Redundancy | Recovery gate |
|---|---|---|---|---|
| V1 | **none** (raw image planes) | 2,880 | none | concealment (Gaussian σ=1.5, w≥.55) |
| V2 | orthonormal DCT | 2,880 | none | coverage tiers; channel-aware Wiener |
| V3 | DCT | 2,880 (3,280 slots) | none | `received`/`degraded`/`picture_only`/`lost` |
| V4 | 1-level Haar | 2,880 | none | same, wavelet masks |
| V5 | 2-level CDF 9/7 | 1,920+480+480 originals | **800 copies** | luma-first / chroma-strict |
| V6 | DCT **or** CDF 9/7 | 2,880 originals | **720 foundation copies**, opposite channel / ≥8 bins / different symbol | floors 0.05/0.15 protected, 0.45/0.60 unprotected |
| V7 | orthonormal DCT | 2,880 ranks over 7 packets | **fold** (M=500 default: host + companded guest), tail rotation, stereo copies | head 0.85 conf / 0.75 coverage; per-cell 2×2 MMSE + LMMSE |

### 12.3 Acquisition

| | Method | Measured CPU (1 core) | Speed accuracy |
|---|---|---|---|
| V1 | 7 fixed off-speed correlation templates | not recorded | spacing tracker + 8-tap sinc |
| V2 | 48-scale × 128-tap template bank, `optimize=False` einsum | **273%** (hiss) / **172%** (dropout / 4) / **56%** (cold) / **5%** (locked) | template-selected, then Newton |
| V2 receiver family | one-pass symbol streaming (`progressive.py`) | never selected | as V2 |
| V3+ | **Schmitt edge count → run analysis → Newton fit**, correlation only as decimated fallback | **3.5–7.1%** across all conditions | **exact to 0.3% across 0.5–2.0×** |
| V7 | V3 pulse count, **Numba `_pulse_word_kernel`**, profile-word matching, tone-seeded pilot timing | `warmup_equalizer()` before capture | 0.01×–4×, pulse scales 100×–0.25× |

### 12.4 Functionality progression — what each version *added*

| Version | Newly added | Not yet present |
|---|---|---|
| V1 | OFDM modulator + equaliser, stereo framing, CRC self-description, concealment, the whole send/receive/bake chain | compression, power allocation, speed robustness |
| V2 | **DCT + power allocation**, parameterised `Layout`, 5 presets, progressive slot ordering, **automatic 128-tap EQ recovery**, single automatic receiver, status/tier contract | cheap acquisition |
| *receiver family* | streaming one-pass decoder, naive reference decoder, waveform decoder | a selected architecture |
| V3 | **edge/pulse-counted acquisition**, 3-tier coast/edge/correlation, bidirectional level normalisation, preamble on both channels, **one consolidated wire**, 122→150 tests, cross-version incompatibility tests | a better transform, any redundancy |
| V4 | **engine registry**, Haar DWT coder, 4 profile codes, `--codec` flag, first clean-vs-noise trade-off data | a good wavelet |
| V5 | **CDF 9/7** (fixed to be invertible), **redundancy copies**, colour-priority recovery gate, **LDPC/FEC** (parked), **tape wires** with a 14 kHz ceiling, 25 fps variant | analog *on a tape-band wire*, cross-channel diversity |
| V6 | **foundation duplication** with enforced diversity invariants, soft fusion with confidence floors, tape-ordered placement, full-repeat control, the measurement-discipline rewrite | a better acquisition story, resolution/speed parity |
| V7 | new wire (metadata symbol, EOF marker, timing tones), **M/S precoding**, **SoftCast gains**, **Hadamard time groups**, **tail-slice rotation**, **folds**, coded pilot status, loop fields + reverse playback, 0.01×–4×, **mono wires**, **stereo slices**, perceptual/perceiver pipeline, GL viewer, device recovery | real-tape validation |

### 12.5 Why each version replaced the last

| Replaced | Because |
|---|---|
| V1 → V2 | No compression and no power allocation — 2,880 raw samples is a hard ceiling and equal carriers for a low-frequency-dominated picture wastes the band. |
| V2 → V3 | Correlation acquisition cost 56–294% of a core and was unusable on hiss or after dropouts, "though locked decode was never the problem". |
| V3 → V4/V5 | 2,880 fixed coefficients is the ceiling; V4 proved the transform is the axis that moves the clean/noise trade-off (27.4 dB/0 frames vs 24.2 dB/8 frames). |
| V4 → V5 | Haar was orthonormal and robust but gave up too much clean quality; CDF 9/7 recovers it. V5 also discovered the wire has no spare slots, hence copies. |
| V5 → V6 | V5 was on a 20.25 kHz wire — not a tape wire — and changed several variables at once, so its conclusions were not trustworthy. V6 re-ran DCT vs wavelet on **the same slot budget, bandwidth and timing**. |
| V6 → V7 | 8.96 fps, and V6 was a controlled experiment rather than a design. V7 keeps V6's requirement list (analog coefficients, colour-last degradation, no black frames, pulse-counted timing, 14 kHz) and rebuilds the wire around them. |

---

## 13. What survives in the tree

| Surviving artefact | Origin | Why it survived |
|---|---|---|
| `animation_modem/transport3.py` | **V3** | V7's pulse acquisition. Docstring: "retains only that acquisition path and its pulse geometry; legacy V1-V6 packet encoders, OFDM decoders, and receiver state machines are intentionally gone." |
| `v7_core.SourceCoder` | **V2** algebra, V7 geometry | `default_allocation` `1/(1+12·hypot)`, gains `sqrt(σ/mean σ)`, channel-aware Wiener, plus V7's grids/shapes truncation and the double-transform warning. |
| `v7_core` `REFERENCE_RATE/SYNC_LEN/GUARD` | V3/V6 constants | 48000 / 288 / 32. |
| `animation_modem/imaging.py` `PROFILES` | **V2** | `color-dct`, `lean-dct` (plus the wavelet names' descendants). |
| `animation_modem/timing.py` | **V2/V3** | `expand_timestamp`, `TimingStats`, `PresentationBuffer`, `ProgressSummary` — plus a vestigial `tiers` counter, the only surviving trace of the named tier ladder. |
| `animation_modem/impairments.py` | V2→V6 | `clean` / `mild` / `rough` / `hostile` presets, still used by V7 torture tests. |
| `animation_modem/audio_common.py`, `playback.py` | V2→V6 | PCM/device plumbing and scheduled playback, carried forward for V7. |
| `animation_modem/perceptual_resize.py`, `imaging.py` | V7 / V2 | Source preparation; `imaging.PROFILES` still carries `color-dct` / `lean-dct`. |
| `v7_fold.py`, `v7_coded_pilot.py`, `v7_fold_table_500.json` | **V7** | Fold tables and coded pilot status. |
| `modem_v7_display.py`, `tools/v7_live.py` | **V7** | Application adapter and standalone sender/receiver. |
| `docs/transport_v7_spec.md` | **V7** | Current-state spec (its §11/§12 hold *planned* work, not shipped behaviour). |

**Gone from `animation_modem/`:** `transport.py`, `transport2.py`, `core.py`,
`decoder.py`, `wavelet.py`, `fec.py`, `v6.py`, `transport3_v5.py`,
`progressive.py`, `waveform_receiver.py`, `reference_receiver.py`,
`presentation.py`, `engines.py`, `live_picture.py`, `preview_overlay.py`,
`audio_buffer.py`, `capture.py`, `channel_scan.py`, `aspect.py`.

**Gone elsewhere:** `modem_screen.py`, `utilities/modem_v3_check.py`,
`utilities/modem_v2_check.py`, `utilities/convert_to_modem.py`,
`MODEM_V2.md`, `MODEM_V2_RECOVERY.md`, `MODEM_WAVEFORM_RECEIVER.md`,
`docs/MODEM_INDEX.md`, `docs/TAPE_WIRE_TESTING.md`,
`docs/transport_timing_and_encoding.md`, `docs/transport_state_machines.md`.

**Docs inconsistency worth knowing:** `docs/UPGRADE_NOTES.md` still cites
`utilities/modem_v3_check.py` and similar paths that no longer exist, and
`docs/MODEM_MODE.md` still leads with the V3-era `wide-v3` + `color-dct`
defaults while the application sender actually defaults to stereo Fold 500.
`docs/MODEM_INDEX.md` and `docs/transport_timing_and_encoding.md` /
`docs/transport_state_machines.md` were **deleted along with the code they
documented** — `af4830d1` is the only place V1–V6 are still described in full,
so `git show af4830d1:docs/transport_timing_and_encoding.md` is mandatory
reading for anyone who wants the V1–V6 detail back.

---

## 14. Verification notes

Commands used for this trace, all from the repository root, all read-only:

```bash
git log --all --format='%h %ad %s' --date=short --reverse
git branch -a -vv
git show 99d60665:animation_modem/transport2.py
git show b55ef9b4:animation_modem/transport3.py
git show af4830d1:docs/transport_timing_and_encoding.md
git show af4830d1:docs/transport_state_machines.md
git show 0fdc309d:docs/transport_v7_spec.md
git log -1 --format=%B <commit>          # full commit message, including measurements
```

Caveats on the numbers quoted here:

- The V2/V3 CPU table, the V4 A/B figures and the V7 fold table come from
  **commit messages and the current spec**, not from re-running anything. They
  were measured on specific hardware and workloads; treat them as the project's
  recorded results, not as reproducible on any machine.
- `modem_tests` passed **217 tests and `test_modem_v7` 13 tests on 2026-09-27**
  per `docs/transport_v7_spec.md` §9 — a dated snapshot. The suite has grown
  since (`modem_tests/` now holds 51 files including `test_v7_slices.py` and
  `test_v7_send_gui.py`), so that count is stale by construction.
- **No real tape or deck has been validated at any version.** This is stated in
  `AGENTS.md`, `MODEM_TODO.md`, `docs/transport_v7_spec.md` §1, and the
  standalone spec's closing paragraph. Every "tape" number in this document is a
  synthetic or bench approximation.
- Audio must not be sent to physical speakers during any of this: run audio
  tests only on an explicitly verified virtual/loopback route.

---

## 15. Appendix — encode/decode code walkthrough

Everything above describes *what* each version does. This section shows *how*,
by quoting the actual encode and decode bodies. All versions share one
skeleton; the differences are concentrated in five places:

```
ENCODE   pixels ─► [A] transform ─► [B] allocation/gain ─► [C] slot map
              ─► [D] OFDM grid (+ training/header/pilots) ─► [E] preamble/frame
DECODE   frame  ─► [E'] acquire ─► [D'] FFT + channel estimate + pilot fit
              ─► [C'] slot gather ─► [B'] Wiener/de-bias ─► [A'] inverse transform
```

`[A]`/`[A']` and `[B]`/`[B']` live in the source coder. `[C]`–`[E]` live in the
transport. Read the per-version differences as edits to those five stages.

### 15.1 V1 — `transport.py`

**Encode.** There is no `[A]`/`[B]`: raw image values go straight onto carriers.

```python
# [C] 2,880 real values -> 15 symbols x 48 carriers x 2 channels (I/Q)
v = image_values(im, profile).reshape(15, 48, 2, 2)
# [D] header = 20 bytes + CRC32 = 192 bits, QPSK, repeated on BOTH channels
raw = struct.pack('>4sIIIBBBB', magic, absolute, index, count,
                  width, height, FPS, numbered)
raw += struct.pack('>I', zlib.crc32(raw))
bits = np.unpackbits(np.frombuffer(raw, np.uint8)).reshape(2, 48, 2)
qpsk = ((bits[..., 0]*2-1) + 1j*(bits[..., 1]*2-1)) / np.sqrt(2)
carriers[0, :, 0] = 1                          # training symbol 0, channel 0
carriers[1, :, 1] = 1                          # training symbol 1, channel 1
carriers[2:4, DLOC, :] = qpsk[..., None]       # header symbols
carriers[4:, DLOC, :] = (v[..., 0] + 1j*v[..., 1]) * .7   # image I/Q pairs
carriers[2:, PLOC, :] = 1                      # pilots
spectrum[:, ALL, :] = carriers * PROFILE_PHASES[profile]  # random per-carrier phase
wave = np.fft.irfft(spectrum, n=N, axis=1)
wave = np.concatenate([wave[:, -CP:], wave], axis=1).reshape(-1, 2)   # CP
out = np.zeros((FRAME, 2), np.float32)
out[16:272, 0] = SYNC            # [E] preamble on CHANNEL 0 ONLY
out[SYNC_LEN:PACKET] = wave      # zero tail out to FRAME=3200
```

Two facts are visible here and both were fixed later: the preamble is on
channel 0 only, and there is **no level normalisation at all**.

**Decode.**

```python
body = samples[SYNC_LEN:PACKET].reshape(19, SYMBOL, 2)[:, CP-4:CP-4+N]
spectrum = np.fft.rfft(body, n=N, axis=1)
received = spectrum[:, ALL, :]
# [D'] profile inference: de-rotate by each profile's training phases and
# pick the smoothest channel (adjacent-symbol coherence)
for name, phases in PROFILE_PHASES.items():
    matrix = np.stack([received[0]/phases[0,:,0,None],
                       received[1]/phases[1,:,1,None]], axis=-1)
    score = abs(sum(matrix[:-1].conj()*matrix[1:])) / sqrt(...)
coherence, profile, h = max(fits)
# 2x2 regularised equaliser, closed form: (H^H H + 4n I)^-1 H^H
gram = hH @ h; gram[:,0,0] += 4*noise; gram[:,1,1] += 4*noise
inv = (adj/det) @ hH
equal = einsum('kij,skj->ski', inv, received) / phases
# pilot phase line fit per channel, then de-rotate
equal[2:,:,channel] *= exp(-1j*(slope[:,None]*ALL + offset[:,None]))
# [C'] header candidates: stereo mean, ch0, ch1; CRC32 verifies
candidates = [equal[2:4,DLOC]*weights[DLOC].sum(-1)/weights[DLOC].sum(-1),
              equal[2:4,DLOC,0], equal[2:4,DLOC,1]]
v = equal[4:, DLOC] / .7
values = np.stack([v.real, v.imag], axis=-1).ravel()
im = values_image(conceal_values(values, pixel_weights, profile), profile)
```

Concealment is `[A']`'s stand-in: `gaussian_filter` of reliable samples
(`w >= .55`) fills the rest, per plane.

### 15.2 V2/V3 — the source coder appears

**`SourceCoder.forward`/`inverse` — stages `[A][B]` and `[B'][A']`.**

```python
def forward(self, values):                       # V2
    coeffs = np.concatenate([dctn(p, norm='ortho').ravel()
                             for p in self._split(values)])
    return coeffs*self.gains                    # [B] power allocation
```

```python
def inverse(self, sent, reliability=None, noise_variance=None):   # V2
    if reliability is None:
        coeffs = sent/self.gains
    else:                                        # channel-aware Wiener de-bias
        gain = self.gains*reliability
        denominator = gain**2*self.variance + noise_variance
        coeffs = divide(sent*gain*self.variance, denominator, where=denom>1e-20)
    return np.concatenate([idctn(p, norm='ortho').ravel()
                           for p in self._split(coeffs)])
```

with, in `__init__`:

```python
gains = np.sqrt(sigma/np.mean(sigma))            # power proportional to stddev
self.gains = gains/np.sqrt(np.mean(gains**2))    # unit mean square
self.variance = sigma**2
```

**V3 encode is V2's body plus three edits** (`transport3.py`):

```python
raw = pack_header(..., magic=MAGIC if layout.progressive else b'V2')
...
out[16:16+len(PREAMBLE), 0] = PREAMBLE   # V3: preamble on BOTH channels
out[16:16+len(PREAMBLE), 1] = PREAMBLE   # (V2 wrote channel 0 only)
out[SYNC_LEN:layout.packet] = wave
peak = float(np.max(np.abs(out)))
if peak > 0:
    out *= headroom/peak                 # V3: normalise UP as well as down
```

The `sent[coefficient_slots(layout, coder.shapes)] = coder.forward(values)`
line is what implements `[C]`: coefficients are scattered into the flat slot
order by rank/frequency, not left in raster order.

**Decode — `core.decode_packet`.** This is the long-lived decoder shared by
V3/V4/V5/V6. The stage order:

```python
# [D'] FFT, 2x2 channel estimate, weights, noise variance, coherence
spectrum = rfft(body, n=N, axis=1)
equal, weights, variance, coherence, skew = _equalise_spectrum(spectrum, layout)
# [D'] pilot phase-slope fit per channel (clamped so nulls can't run away)
equal[2:,:,channel] *= np.exp(-1j*(slope[:,None]*carriers + offset[:,None]))
# [D'] header: CRC verify, then a tolerance-corrected retry
fields = _resolve_header(raw, layout)
if fields is None and header_tolerance:
    fields = _correct_header(raw, flat, header_tolerance, layout)
# [D'] drift refit / clipping discount only when the packet is actually dirty
if timing_drift > DRIFT_REFIT: equal, weights, variance = _refine_drift(...)
# [C'] flatten slots, [B'] gather weights+noise, select coder from profile code
picture = coders.get(fields[5], coder) if fields and coders else coder
values = picture.inverse(sent[slots], per[slots], per_noise[slots])
```

The profile code in the header (`fields[5]`) *selects the inverse transform*.
That single line is what makes V4 and V5 coder-swaps rather than new wires:
`coder` is the only thing that changes.

### 15.3 V4 — Haar replaces the DCT

Only `_forward_transform`/`_inverse_transform` change; `SourceCoder.forward`
and `.inverse` (and therefore `[B]`/`[B']`) are inherited untouched.

```python
class WaveletCoder(SourceCoder):
    def _forward_transform(self, plane, rows, cols):
        ll, lh, hl, hh = haar_level(plane)
        mask_ll, mask_lh, mask_hl, mask_hh = _get_masks(*plane.shape)
        return _apply_mask_and_pack(ll, lh, hl, hh, mask_ll, mask_lh,
                                    mask_hl, mask_hh, rows, cols)
    def _inverse_transform(self, corner, rows, cols, grid):
        mask = _get_masks(grid[0], grid[1])
        return _unpack_and_apply_mask(corner, rows, cols, grid, *mask)
```

The masks are measured fixed arrays (top-N by average magnitude), not a rule.
The gain/allocation machinery is shared verbatim, which is exactly why the
Haar variant is a fair A/B.

### 15.4 V5 — CDF 9/7 and the truncation-aware inverse

```python
class Cdf97Coder(SourceCoder):
    def _forward_transform(self, plane, rows, cols):
        subbands = cdf97_forward_2d(plane, self.levels)
        order = _cdf97_pack_order(_cdf97_subband_dims(*plane.shape, self.levels))
        out = np.concatenate([subbands[l][b].ravel() for l, b, _, _ in order])
        return out[:rows*cols] if len(out) >= rows*cols else pad(out, rows*cols)
    def _inverse_transform(self, corner, rows, cols, grid):
        # zero-fill every dropped tail, then the true multi-level inverse
        subbands = {}
        for level, band, sr, sc in _cdf97_pack_order(_cdf97_subband_dims(*grid, self.levels)):
            full = np.zeros((sr, sc)); take = min(sr*sc, len(flat)-idx)
            full.ravel()[:take] = flat[idx:idx+take]; idx += take
            subbands[(level, band)] = full
        return cdf97_inverse_2d(subbands, self.levels, out_shape=grid)
```

Truncation is mid-subband: the inverse zero-fills the dropped tails
(JPEG 2000 semantics), which is why the source grid can be 96×112 while the
wire corner is 56×44. `[B]`/`[B']` are still `SourceCoder`'s.

### 15.5 V6 — protection wraps the coder

V6 does not change `[A]` or `[B]` either. It wraps a base coder and adds
copies, so `forward`/`inverse` are about *fusion*, not transform:

```python
def forward(self, values):
    raw = self.base.forward(values)/self.base.gains      # originals, no gain
    return np.concatenate([raw, raw[self.copy_of]])*self.gains   # + 720 copies

def inverse(self, sent, reliability=None, noise_variance=None):
    if reliability is None:                              # clean: average copies
        raw = y/self.gains
        kept = raw[:self.n_orig].copy()
        kept[self.copy_of] = (kept[self.copy_of] + raw[self.n_orig:])/2
    else:                                                # soft fusion by precision
        a = self.gains*rel
        top = a*y/noise; bottom = a*a/noise
        np.add.at(top, self.copy_of, top[self.n_orig:])
        np.add.at(bottom, self.copy_of, bottom[self.n_orig:])
        kept = self.variance*top/(1 + self.variance*bottom)
    # then base._inverse_transform(...)
```

The `recovery_floor` array set in `__init__` (protected luma/chroma
0.05/0.15, unprotected 0.45/0.60) is applied as a confidence gate on `kept`.
The copies' *placement* is separate: `_slot_order()` for baseline, a
`linear_sum_assignment` solve for the opposite-channel/separation constraint,
and the tape-ordered path for the low-carrier/interleaved candidate.

### 15.6 V7 — the pipeline rebuilt around `[C][D][E]`

**Encode.** `[A]` is `v7_core.SourceCoder` (DCT + allocation, unchanged in
spirit from V2). `[B]` adds rank gating and a mean removal. `[C]` is the rank
table with tail rotation. `[D]` adds Hadamard spreading and M/S.

```python
def encode_frame_coeffs(model, coeffs, counter, ..., right_coeffs=None):
    idx = model.rank_tables[counter % TAIL_PHASES]     # [C] which tail slice
    ranks = np.maximum(idx, 0)
    def data_cells(values):
        cells = np.zeros((F, 65, 2), complex)          # symbol, bin, M/S
        vals = model.gain[ranks]*(values - model.mu)[ranks]   # [B]
        vals[idx < 0] = 0                              # slots not sent this packet
        tx = _hadamard8(vals)                          # spread 8 groups over symbols
        flat = cells.reshape(-1)
        flat.real[CELLS_I] = tx[GROUP_IS_I].ravel()
        flat.imag[CELLS_Q] = tx[~GROUP_IS_I].ravel()
        return cells
    X = data_cells(coeffs)
    X += SCATTERED_PILOTS
    # [D] M/S -> L/R
    XL = (X[...,0] + X[...,1])/np.sqrt(2)*model.phase
    XR = (X[...,0] - X[...,1])/np.sqrt(2)*model.phase
    waves = np.fft.irfft(np.stack((XL, XR), -1), n=N, axis=1)*model.scale
    return np.concatenate((waves[:, -CP:, :], waves), axis=1).reshape(-1, 2)
```

`encode_pulse_frame_coeffs` then builds the full packet — body, pulse word,
metadata symbol, shaper, tones, EOF:

```python
out[PULSE.SYNC_LEN:PULSE.SYNC_LEN+FRAME] = body
preamble = PULSE.profile_preamble(pulse_profile_code)      # [E] profile identity
out[16:16+len(preamble), :] = preamble[:, None]
# metadata symbol: 11 pilots + 20 QPSK cells = 5 bytes (aspect/model/tail/index/CRC)
vals = metadata_symbols(aspect_code, model.encoding_type, counter % TAIL_PHASES,
                        source_index, loop, direction)
meta_wave = np.fft.irfft(meta[:,0]/np.sqrt(2)*model.phase[-1], n=N)
shaped = bound_emission(out, EMISSION_EDGE_HZ, RATE)       # 14 kHz Kaiser FIR
shaped[meta_start:meta_start+META_SYMBOL, :] += meta_pcm[:, None]  # after shaper
if pilot_tones: shaped = _add_pilot_tones(shaped, counter, ...)
if eof_marker:  shaped[...] += marker                       # 4/8/6/6 runs at ±0.55
```

**Decode.** `decode_frame` is `[E'][D'][C'][B'][A']`:

```python
# [E'] resample onto the reference grid using the pulse-derived time map
pos = n + np.interp(n, nom, d); seg = _sample_at(x, pos, taps=4)
if cancel: seg[:, ch] -= (seg[:,ch] @ clk)/(clk @ clk) * clk   # clock cancel
windows = seg.reshape(F, SYM, 2)[:, WIN:WIN+N, :]
# [D'] one rfft, then the joint two-channel pilot fit + per-symbol fade/noise
Z = np.fft.rfft(windows, axis=1) * _receive_rotation(model)
H, _ = channel_joint(Z, force_float32=..., return_timing_diag=True)
H, noise, _ = fade_and_noise(fade_input, H, ...)
# [C'][B'] per-cell 2x2 MMSE, then an 8-value group LMMSE with the rank priors
xhat, conf, got = _equalize_numba(model, Z, H, noise, counter)
gate = np.clip((conf-floor)/(.85-floor), 0, 1)
current = mu + xhat*gate
coeffs = prev_tail.copy(); coeffs[got] = current[got]          # tail memory
# acceptance gate; else hold previous picture, never black
if head_confidence < HEAD_MIN_CONFIDENCE or head_coverage < HEAD_MIN_COVERAGE:
    return Result(counter, 'lost', display_coeffs if displayable else prev_tail, ...)
```

The group solve is `A Λ Aᵀ + diag(σ²)` per 8-value group, solved by Cholesky:

```python
S[i,k] = system[group, i, k]; S[i,i] += sig[i]     # priors + this frame's noise
# ... Cholesky, then all 8 estimates as RHS ...
confidence = 1.0 - (lam_g[j] - weight)/lam_g[j]
```

`[A']` is again `SourceCoder.inverse`, reached through the profile dispatcher,
with the tail values substituted from `TailStore` where this packet did not
carry them.

### 15.7 Stage-by-stage comparison

| Stage | V1 | V2 | V3 | V4 | V5 | V6 | V7 |
|---|---|---|---|---|---|---|---|
| `[A]` forward | none | `dctn(ortho)` | `dctn` | Haar DWT + masks | CDF 9/7, multi-level, truncated | base (DCT or CDF) | `dctn` + grids truncation |
| `[B]` gain | none (`*0.7`) | `sqrt(σ/mean σ)` | same | same | same | base gains removed, copies appended, re-gained | `gain[rank]*(x−mu)`, Hadamard |
| `[C]` slot map | raster 15×48×2 | rank/frequency `coefficient_slots` | same | same | same | `_slot_order` + assignment solve | rank tiers + 7-slice tail rotation |
| `[D]` training | 2 symbols | 2 symbols | 2 symbols | same | same | same | 2 continual + scattered pilots, 3 phases |
| `[D]` identity | 192-bit QPSK CRC32, stereo | 160-bit QPSK + CRC32, stereo | same, magic `V3`/`V2` | same | same | magic `V6`/`VR` | 5-byte metadata symbol + CRC-16/CCITT |
| `[D]` stereo | raw I/Q | raw I/Q | raw I/Q | raw I/Q | raw I/Q + copies | M/S implicit via copies | explicit M/S `(M±S)/√2` |
| `[E]` preamble | random, ch0 only | random sync bank | 16-bit biphase, both ch | same | same | same | same + 6 profile words |
| `[E]` normalise | **none** | down only | **up and down** | same | same | same | shaper + −1 dBFS limiter |
| `[E']` acquire | 7-template correlation | 48-scale bank | edge count → Newton | same | same | same | edge count (Numba) + tones |
| `[D']` channel | 2×2 + pilot line | 2×2 + pilot line | 2×2 + pilot line + drift refit | same | same | same | joint 2-ch + fade + 2×2 MMSE |
| `[B']` inverse | `/0.7` | Wiener de-bias | same | same | same | soft fusion + floors | per-cell MMSE + group LMMSE + gate |
| `[A']` inverse | concealment | `idctn` | `idctn` | iHaar | CDF 9/7 inverse | base inverse | `idctn` + tail memory |
| Damage policy | conceal / corrupt | tiers | status contract | same | luma-first gate | floors | hold last frame, never black |

The one-line summary of the whole table: **`[A]`/`[B]` changed five times
(V1→V2→V4→V5→V6), `[E']` changed once and mattered most (V2→V3), and V7
rebuilt `[C]`–`[E]` around a rank schedule rather than a fixed raster.**

---

## 16. Mechanisms from V1–V6 that V7 did **not** carry over

This section is the inverse of §12.4: not what each version added, but what it
had that V7 lacks, and whether the omission looks deliberate or accidental.
Verified by grepping the current tree — each "absent" claim below was checked
against `animation_modem/` and `tools/`, not inferred.

### 16.1 Absent and probably worth revisiting

**(1) V2/V3 automatic long-FIR EQ recovery.** On a bad header or pilot error
> 0.15, V2 fitted a **128-tap stereo FIR** from the known preamble and training
prefix, did a regularised frequency-domain inversion of the *whole packet*, and
kept it only if recovery quality improved (`input_path=eq_corrected`). Measured
through ±6 dB bell EQ at 375/1000 Hz: image RMS error 0.26–1.15 → 0.04–0.08.
V7 has per-carrier gains from two training symbols, scattered pilots,
`fade_and_noise`, and optional `m-reference` tone equalisation — but no
whole-packet static-FIR inversion. A per-carrier gain cannot undo a channel
whose impulse response exceeds the 16-sample CP (that is ISI/ICI); a static
filter inversion can. Tape bass boost and head-bump EQ are exactly that case.

**(2) V3's other two acquisition tiers.** V3 was explicitly three-tier: coast →
edge capture → decimated correlation. V7 kept only the edge tier
(`measure_pulses` / `measure_pulses_both`, forward + reversed). The **coast**
tier verified the predicted next preamble at one-frame spacing instead of
re-searching, which is both a CPU win and continuity across brief dropouts; the
**correlation fallback** existed for signal too weak to count edges on. Neither
is in V7. (The draft's own clock-track flywheel — `find_words`, `time_map`,
`refine_time_map` — is written and unused, and would be a natural home for a
coast/PLL.)

**(3) V6's enforced diversity placement for the critical tier.** V6 placed its
720-value foundation by `linear_sum_assignment` with hard constraints: opposite
channel, ≥8 carrier bins away, preferably a different OFDM symbol; the
tape-ordered variant tightened this to homes ≤4 kHz, copies ≤7 kHz, ≥7 bins
apart, ≥half an image-symbol block in time. V7's rank→slot map is static: the
**head (208 ranks) sits on the M-only bins 5–9 in every packet**, the body in
every packet, and only the tail rotates (7 slices). M/S gives channel
redundancy for the head, but there is **no second-frequency copy**. A fixed
notch in 1.5–3.4 kHz would erase the head on every packet permanently.

**(4) V2/V3 input conditioning highpass.** V2's automatic receiver fell back to
a zero-phase high-pass when search failed; `core._conditioning_filter` was a
Butterworth-4 highpass. V7 has only the bounded autoleveler
(`v7_live_input._update_level`, gain 0.5–32×). Tape rumble and DC below the
1.5 kHz lowest carrier can trip the Schmitt edge detector, and nothing removes
them before header search.

**(5) A robustness wire variant.** V5 shipped `wire-tape-25` (1,904 samples,
25.2 fps, 760 capacity) and V6 shipped `wire-v6-repeat` (7,952 samples,
6.04 fps, every coefficient copied). V7 has **one packet geometry** — 3,920
samples, 12.245 fps — and its profiles (fold/mono/slices/aspect) vary content,
not packet length. On dropout-heavy tape, loss is per-packet: halving packet
length halves the picture lost per event and speeds recovery; a full-repeat mode
trades frame rate for maximum robustness. Both ends of that spectrum are gone.

**(6) V2/V3's "keep the correction only if it improves" guard.** V2's recovery
path kept the EQ correction *only* when `_recovery_quality` improved. V7's
`m-reference` tone equalisation, drift refit and pulse-warp are applied without
an explicit accept/reject comparison against the uncorrected result.

**(7) V5's parked LDPC, for the small payload.** `fec.py` (rate-13/15 LDPC,
belief propagation) was the wrong tool for analog image coefficients — but it
was never tried on the 5-byte metadata / loop fields, where V7 currently
*rejects* an invalid word (or retries with tone assistance) rather than
correcting it. A short block code is the efficient version of that retry.

**(8) V4's engine registry.** `engines.py` is why DCT vs Haar vs CDF 9/7 could
be compared on one fixed wire. V7's model tables are pinned and it has one
coder, so a new transform means re-freezing tables. If V7 ever wants a
perceptual or wavelet source coder, the registry pattern is the missing piece.

### 16.2 Deliberately dropped — do not resurrect without a reason

| Mechanism | Why V7 dropped it |
|---|---|
| V1/V2 `conceal_values` spatial concealment | `AGENTS.md`: show the damaged frame as-is, hold the last good frame on failure, no black fallback. V7's `TailStore` + hold is the intended policy. |
| V1 profile inference from training coherence | V7 carries profile identity in the pulse word and metadata; coherence inference would re-introduce ambiguity. |
| V2's five bandwidth presets | Replaced by one wire plus a 0.01×–4× speed range; "the old preset table is gone." |
| V2/V3 correlation template bank | Deliberately replaced — 273% of a core on hiss. |
| V1's random-phase noise-burst preamble | Replaced by the countable biphase word. |
| V3's split header | Measured strictly worse than the low-and-narrow header. |

### 16.3 Already carried over (so it is not a gap)

- **Per-plane confidence floors** — `v7._gate_floor` is head luma .05 / chroma
  .15 and non-head luma .45 / chroma .60: byte-for-byte V6's `recovery_floor`.
- **V2's allocation algebra** — `default_allocation`, unit-mean-square gains,
  channel-aware Wiener de-bias and the DC-rescue rule all live in
  `v7_core.SourceCoder`.
- **V3's acquisition geometry** — biphase word, Schmitt edges, preamble on both
  channels, up-and-down normalisation.
- **Identity discipline** — CRC-protected self-description, loud cross-version
  incompatibility, hold-the-layout retry.
- **Presentation** — bounded buffers, device/rate recovery, autolevel,
  presentation scheduling (`timing.py`).