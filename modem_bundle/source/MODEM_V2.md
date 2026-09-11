# Modem v2 — theory, state, and handover

**Recovery update:** see [MODEM_V2_RECOVERY.md](MODEM_V2_RECOVERY.md) for the
current receive state machine, tests and commands. The measurements and known
bugs below describe the earlier prototype; allocation inversion, acquisition,
EOF handling, WAV alternate loss and small-preset sizing have since changed.

This covers the analog image link on the `modem` branch: why it is built the
way it is, what was measured, what landed, what is proposed but unfinished,
and what to do next. Read `MODEM_MODE.md` first for how v1 is operated.

Every number here came from a measurement in-repo. Where an idea was tried and
failed, it is recorded as failed — the point of this document is that nobody
re-derives a dead end.

---

## 1. What the link actually is

An **analog** image link over stereo audio. 52 OFDM subcarriers at 375 Hz
spacing across 1125–20250 Hz, 19 symbols per packet, 15 packets per second.

The critical property is that it is analog, not digital. Each subcarrier
carries a real pixel value in its I and Q components, scaled to ±1. There is
no quantisation, no entropy coding, no error-correcting code on the image. A
noisy channel produces a noisy picture rather than a failed decode. Only the
24-byte header is digital (QPSK + CRC32), and that asymmetry is the source of
most of the link's odd behaviour: **the header dies before the picture does,
every time.**

The value budget, for v1's `color` profile:

```
Y 40x48 = 1920  +  Cb 20x24 = 480  +  Cr 20x24 = 480  =  2880 values
2880 values per 6400 real audio samples  =  45% of the theoretical ceiling
```

That 45% decomposes as: bandwidth 0.80 × symbols 0.79 × cyclic prefix 0.89 ×
duty 0.945 × pilots 0.92. There is no compression anywhere in v1 — 2880 values
carry exactly 2880 image samples, raw.

---

## 2. What limits picture quality

This is the single most useful thing to understand, and it splits in two.

**On a clean digital path the modem is transparent.** Measured end-to-end
through BlackHole: 37.98 dB PSNR against a source-coding ceiling of 37.94 dB.
The channel contributes −0.04 dB. Nothing about the modem is limiting the
picture; 4:2:0 chroma and 8-bit quantisation at 40×48 are. More quality there
means more *values*, and the layout work below is worth about +15%.

**On an analog path the channel costs 9–16 dB.** That is where every
optimisation below applies, and none of them help on a clean cable.

Use `utilities/modem_link_budget.py` against a real bake to see which regime
you are in. It separates the source-coding ceiling from the channel's
contribution and reports the transmit-power split.

---

## 3. The three levers, measured

### 3.1 Band — narrow it for tape

Carriers above the tape's roll-off are not merely useless, they are harmful.
The MMSE equaliser boosts an attenuated bin back up and boosts the hiss in it
along with the signal. Above 12 kHz that is **42% of transmit power doing net
harm on a cassette**.

Measured with tape-like hiss at −45 dBFS:

| roll-off | header | PSNR | bin coverage |
|---|---|---|---|
| none | 100% | 26.61 | 1.000 |
| 14 kHz | **0%** | 23.38 | 0.854 |
| 10 kHz | 0% | 23.04 | 0.656 |

Narrowing costs values but buys two things: power concentrated where it
survives, and **playback-speed headroom**, since the top carrier sets the
aliasing limit at `24000 / f_top`. A 20.25 kHz ceiling aliases above 1.19×; a
10 kHz ceiling survives 2.37×.

**Narrow by choosing bins, not by scaling every length.** Both reach a 10 kHz
ceiling, but bin selection keeps the symbol at 3 ms where scaling by k=2 makes
it 6 ms — and a longer symbol means more of the tape's flutter happens *inside*
one symbol, where the per-symbol pilot fit cannot track it. v2's presets do it
by bin selection for this reason.

### 3.2 Power allocation — the graded-quality lever

For analog transmission the optimum allocates power proportional to a
component's **standard deviation**, not its variance:

```
minimise  Σ σ_i² σ_n² / (P_i + σ_n²)   subject to   Σ P_i = P
gives     P_i ∝ σ_i
```

Take the 2D DCT of each plane, send low-frequency coefficients loud and fine
detail quiet. Measured against flat raster transmission:

| channel | today | with allocation |
|---|---|---|
| −3 dB @ 20 kHz, sd 0.02 | 36.33 | **41.55** |
| −3 dB @ 14 kHz, sd 0.05 | 28.37 | **38.38** |
| −3 dB @ 8 kHz, sd 0.05 | 21.59 | **34.52** |

**+5 dB on a good channel, +13 dB on a bad one — and the gain grows as the
channel worsens.** That growth *is* the graded quality: as the link degrades,
the quiet high-frequency coefficients sink under the noise first, so the
picture loses detail progressively instead of falling apart. Good/better/best
in one waveform, no negotiation, no handshake — which matters because the link
is one-way.

It costs no header bits. The image library is baked and immutable, so the
allocation table is computed once from the bake and shipped as a constant,
exactly like `PROFILE_PHASES`. `modem_v2_check.py allocate` builds it.

**Two things that do NOT work, both measured:**

- *Reordering alone is worthless* — +0.5 dB, and that was clipping artefacts.
  The DCT is orthonormal, so total noise energy does not depend on which
  coefficient occupies which slot, and white noise stays white through the
  inverse transform, so it does not even look different. Position is not a
  lever; power is.
- *Pairing loud coefficients with robust carriers is actively worse* — 2 to
  5 dB worse than plain allocation. Error per value goes as `(σ_n/g_i)²/u_i`,
  so putting low power `u` on low gain `g` multiplies both penalties and
  concentrates all the damage in the same places. Allocate, do not map.

### 3.3 Level — free SNR, but only on a noise-limited path

`SYNC` is normalised to 0.65 peak while the image body only reaches 0.31. The
body can rise 6.4 dB before the waveform peak changes at all. Raising it ×1.8
buys **+3.8 dB PSNR** on a noise-limited channel with the total peak unchanged.

But it *costs* 4 dB on a clipping-limited path. There is currently no output
level control at all — `0.7` is hardcoded in `transport.py` — and the optimum
swings ±4 dB depending on which regime you are in. A `--modem-gain` flag is the
cheapest real improvement available.

PAPR reduction by searching `PROFILE_PHASES` seeds was tried: only ~1 dB
(12.98 → 12.02 over 200 seeds). Not worth a wire-format note.

---

## 4. Analog playback: speed, flutter, rumble

### Rumble is solved, and the fix is free

Every data bin lives above 1125 Hz. Rumble and mains hum sit far below and
never reach the demodulator — the FFT already rejects them. They kill the link
because `sync_correlation` normalises by buffer peak and window energy, so
out-of-band energy inflates both and drags the score under threshold. The
*detector* was the victim, not the demodulator.

A 4th-order 600 Hz–22 kHz band-pass ahead of `Receiver.feed` costs 0.44 dB on
a clean source and makes rumble a non-event at **every level tested including
full scale**. Landed, on by default, `--no-input-filter` to disable.

### Speed error

v1 tolerance before any work: **±0.2% for the header, ±1% for a picture.** A
well-aligned cassette deck is ±0.5%; a tired one is worse. So v1 works on a
good deck and fails on a sloppy one.

The mechanism: `decode_packet` slices a fixed FFT window at `[CP-4 : CP-4+N]`,
which is 4 samples of slack inside a 16-sample prefix. Speed error walks each
symbol off that window, and once the walk exceeds the prefix no amount of pilot
phase correction helps.

Two approaches were tried:

- **Inter-packet spacing** (landed as `a56aaba6`): packets leave exactly
  `FRAME` samples apart, so the gap between sync locks measures speed
  directly. The estimate is excellent — within 0.01% — and gets ±4%. But it
  needs three consecutive packets and gates on a ±4% plausibility window,
  which quietly breaks the "every packet independently decodable" principle
  the rest of the design rests on. **Wrong shape; superseded by v2.**
- **Cyclic-prefix autocorrelation** (tried, failed, do not repeat): the
  textbook self-clocking primitive. The prefix is only `CP/(N+CP)` = 11% of
  the signal, so the statistic peaks near 0.11 against a noise-like OFDM body
  and never separates from the floor. Measured estimates were wrong by tens of
  percent at every speed.
- **A bank of scaled preambles** (v2): correlate against 29 templates spanning
  0.5×–2.0×, which finds the rate and the packet position together. Dumber,
  and it works.

Note the asymmetry: **slow is recoverable, fast is not.** At 0.5× everything
lands at half frequency and fits. At 2× the top carrier folds past Nyquist and
is gone. Above roughly 1.19× on the wide band you lose the top bins
permanently unless you capture at 96 kHz.

### Flutter is already fine

0.3% at 4–60 Hz decodes cleanly, which covers cassette spec (0.1–0.3% WRMS).
The per-symbol pilot phase fit re-aligns each symbol independently, so slow
wow is absorbed. This is why keeping symbols short matters.

### Resampling

If you correct speed, **do not use linear interpolation.** The carriers reach
20.25 kHz against a 24 kHz Nyquist, where `np.interp` loses about half the
signal (relative error 0.49). Lanczos-8 gives 0.016. Linear interpolation was
enough on its own to keep the header down even after the rate was known
exactly.

---

## 5. Delivery: newest wins, do not schedule

v1 schedules presentation against a shared wall clock. The transmitter picks
the image index for a projected `target_time` and the receiver holds the frame
until then. That is correct for a live two-machine install with chrony, and it
is the only mode whose correctness depends on **two clocks agreeing** — scope
and ascii call `update_index(png_paths_len, PINGPONG)` and read one clock at
render time, so they can be late but never wrong.

For an analog source it is meaningless. A timestamp recorded onto tape says
nothing about the current wall clock. The right model is the LTC/MTC one this
was based on: decode when you can, emit only complete CRC-validated words,
newest wins, never wait. v2's `live-receive` does exactly that.

Useful diagnostic, already instrumented: `decode_error_ms` is
`now_ns - target_time_ns` at the receiver. Run `modem_receive.py --no-quiet`
and take its median. A large stable value is a pipeline offset (input latency,
margin misconfiguration); drift is a clock problem. In this session it went
from −100.8 ms to +3.6 ms once configured correctly.

`--modem-index-offset-ms` shifts *which image the clock returns* without
touching when it is shown. The timing flags (`--modem-receive-margin-ms`,
`--modem-time-offset-ms`) both land in `target_time_ns`, which moves the index
and the presentation instant together — so they change total delay and can
never correct an index that disagrees with the other modes.

---

## 6. State of the code

### Landed and working

| commit | what |
|---|---|
| `cb84f3f2` | Receive path 1.5× faster, byte-identical output. Bounded sync search, O(N) energy, no SVDs. |
| `560e2030` | Input band-pass, input `blocksize=256`, `--quiet` back to off. |
| `8d7f3e74` | `--modem-index-offset-ms`. |
| `c3eee2b9` | `utilities/modem_link_budget.py`. |

### Landed but wrong shape

`a56aaba6` — inter-packet speed tracking. Works to ±4%, but makes a packet's
decode depend on the two before it. Superseded by v2's approach; consider
reverting when v2 lands.

### Proposed, incomplete

`animation_modem/transport2.py` plus `utilities/modem_v2_check.py`. Not wired
into `main.py`; the tool is the only entry point. What it demonstrates:

- Parametric layout. On a 10 kHz roll-off the `tape` preset scores **25.76 dB
  against `wide`'s 19.13 dB** — the band argument, validated.
- Header on the lowest 20 carriers, repeated on both channels at 1.6×
  amplitude. Survives the tape channel that erased v1's.
- Speed working at 0, ±5%, ±10% and −33%, where v1 died above 1.5%.

---

## 7. The one change that decides everything

**Allocation currently loses about 3 dB instead of winning +5 to +13 dB.**

The measured gain came from a standalone experiment where the receiver knew
each carrier's gain and formed an MMSE estimate of each coefficient.
`transport2.py` does not do that:

- `decode_packet` computes `weights` — per-carrier reliability, exactly what an
  MMSE estimate needs — and **discards it** for the image path, using it only
  for a coverage percentage.
- `SourceCoder.inverse` **divides** by the allocation gain, so a coefficient
  given little power has its channel noise amplified by `1/gain`.

The fix is to carry `weights` into the coefficient estimate and replace the
division with a Wiener filter:

```
ĉ_i  =  sent_i · g_i / (g_i² + σ_n²/σ_i²)      instead of      ĉ_i = sent_i / g_i
```

Until this lands, v2 trails v1 on exactly the channels it was built for, and
nothing else in v2 is worth adopting.

---

## 8. Known bugs

- v2 needs lookahead past a packet's end, so it cannot decode a stream's last
  packet without trailing padding. The tool pads around it.
- v2's WAV `read` path recovered 6 of 12 packets — dropping alternates. **Not**
  the decoder: driving the live send callback and decoding its output gave 37
  of 37.02 expected, and the fake-device receive test gave 17 of 18. The bug is
  in the WAV read path specifically.
- v2 acquisition is patchy rather than monotonic across speed: ±5%, ±10% and
  −33% work, ±20% does not. Not diagnosed.
- v2 acquisition costs 22–90 ms/frame against v1's 1.34 ms. The 29-template
  bank dominates and needs a decimated coarse stage before the fine search to
  be viable on a Pi.

---

## 9. Open decisions

1. **Band preset for the installation.** `tape` keeps 40×48 at 7.71 fps;
   `tape-fast` keeps 14.35 fps at ~26×32. Watch the ping-pong at 7.7 fps
   before deciding — if it holds up, `tape` keeps the resolution.
2. **Whether to add `--modem-gain`.** Cheap, and the optimum swings ±4 dB
   between a noise-limited and a clipping-limited path.
3. **Whether to revert `a56aaba6`** once v2's speed handling is trusted.

---

## 10. Running it

See README section 6. In short:

```bash
python utilities/modem_v2_check.py allocate --modem-dir images_modem \
    --out modem_allocation.npy

python utilities/modem_v2_check.py live-receive --device "BlackHole 2ch" --preset tape
python utilities/modem_v2_check.py live-send --modem-dir images_modem \
    --device "BlackHole 2ch" --preset tape --allocation modem_allocation.npy
```

`bench` compares v1 and v2 across simulated channels; `write`/`read` are the
tape loop.
