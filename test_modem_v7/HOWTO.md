# test_modem_v7: V7 picture-quality experiments

Experiments on getting more picture through the V7 wire at 1x speed. Nothing
here changes the modem. Each script either wraps the unchanged encoder and
decoder or simulates the channel. They are measurements to decide what to
build, not features.

| Script | Channel | Question |
|---|---|---|
| `compare_filters.py` | real modem | Does the `box` or `lanczos` encode filter beat today's `nearest`? |
| `layout_oracle.py` | simulated (per-slot noise) | Ringing and prefilters; luma-first coefficient choice; where to fold |
| `fold_modem.py` | real modem | Does luma folding survive warble, jitter, low-pass and dropouts? |
| `live_fold.py` | live receive path | Fold tables (`build`) and an offline live-path check (`selftest`) |
| `live_loopback.py` | real live tools | `tools/v7_live.py` send and receive back to back, no sound card |
| `tone_code.py` | simulated packet wire | Can a coded pilot carry status and a timing track under synthetic impairments? |
| `compare_tone_fold.py` | simulated packet wire | How do coded pilots compare with current V7 and folded-500/1000? |

`common.py` holds shared helpers: loading, scoring, DCT grids and variance
fitting. `folding.py` holds the fold codec and the equaliser hook used by
`fold_modem.py`.

## Setup

Use the repo's venv plus two scoring packages, and run from the repo root:

```bash
.venv/bin/python -m pip install ssimulacra2 scikit-image
export NUMBA_CACHE_DIR=tmp/numba_cache     # repo-local, like every other artifact
```

## Frames

Every script takes source frames as arguments, and **their order matters**:
- **Frames 1, 3, 5, …** are used only to fit statistics.
- **Frames 2, 4, 6, …** are the only ones scored.

So no frame is scored against statistics fitted on itself. Use real
full-size frames, not 80x96 thumbnails. The results below came from 11 frames
of an 810x1080 portrait face video. Scores are SSIMULACRA2 (higher is better;
about 90 is visually lossless). Every reconstruction is resized to 405x540
and compared with the source at 405x540, which is how a receiver shows it.

```bash
FRAMES="frames/f01.webp frames/f02.webp ... frames/f11.webp"
```

## 1. Smoke test (about 3 minutes)

```bash
.venv/bin/python test_modem_v7/compare_filters.py $FRAMES --quick   # a few seconds
.venv/bin/python test_modem_v7/fold_modem.py      $FRAMES --quick   # ~30 s
.venv/bin/python test_modem_v7/layout_oracle.py   $FRAMES --quick   # ~3 min
```

Each prints one JSON line per result. What to look for:
- `compare_filters`: `box` scores several points above `nearest`.
- `fold_modem`: `fold 500` and `fold 1000` score above `base` on `clean`
  and on `dropouts`.

## 2. Full runs

```bash
.venv/bin/python test_modem_v7/compare_filters.py $FRAMES    # ~5 min
.venv/bin/python test_modem_v7/fold_modem.py      $FRAMES    # ~15 min
.venv/bin/python test_modem_v7/layout_oracle.py   $FRAMES    # ~20 min
```

Results (`results.json`) and contact sheets (source next to reconstructions)
go to `tmp/test_modem_v7/<script>/`, which git ignores. Useful options:
- `fold_modem.py --cases dropouts 'jitter 0.3%' --folds 500 --design-db 25`
- `layout_oracle.py --experiments ringing luma --snr 50 30 25 20`
- `--out DIR` on any script.

## Expected results (the reference run)

### compare_filters.py: encode filter, real modem

| Case | nearest (today) | box | lanczos |
|---|---|---|---|
| clean | -23.9 | **-16.9** | -18.8 |
| hiss -40 | -27.8 | **-21.6** | -23.0 |
| type II | -27.5 | **-20.9** | -22.5 |
| type I | -33.4 | **-28.6** | -29.9 |
| wow/flutter | -25.4 | **-18.5** | -20.5 |
| fast flutter | -25.8 | **-19.0** | -20.8 |
| low-pass 10k | -24.0 | **-17.0** | -18.9 |
| dropouts | -24.6 | **-17.8** | -19.6 |

`--modem-encode-filter box` needs no wire change. Receivers pick the model
from the packet's `encoding_type`.

### layout_oracle.py: simulated channel

- **ringing:** a smooth taper before the cutoff trades sharpness for less
  ringing and never scores better. On its best setting it only matches box
  80x96, and on the luma-first layout it loses 3-7 points. Reconstructing on
  a finer 192x160 grid alone does not help either: -18.8 against -16.8 for box.
- **luma:** choosing coefficients by variance across planes keeps about
  2,620 luma and 260 colour coefficients instead of 1,920 and 960. That
  moves -16.8 to about -6/-8 at every SNR. The colour cost is dE 2.6 -> 3.3.
  Weighting luma harder does not help further.
- **fold:** folding hosts in luma keep dE at 3.4. Hosts in colour give
  dE 7-11 and should never be used. With luma first and no folding, the
  scores are -7.7 / -8.6 / -10.4 at 50 / 30 / 25 dB. Luma fold 1000 gives
  +6.2 / -3.0 / -11.5, and fold 500 gives +0.6 / -4.6 / -10.1.
- **bad:** below its design SNR, luma folding costs about 3.5 (fold 500) to 6
  (fold 1000) points at 20 dB, and 5-10 points at 15 dB.

### fold_modem.py: luma folding, real modem, today's layout, box filter

Fold steps are chosen for 30 dB. `+fb` means low-confidence slots fall back
to plain values.

| Case | No fold | Fold 500 (+fb) | Fold 1000 (+fb) |
|---|---|---|---|
| clean | -16.9 | -7.0 | **-4.1** |
| low-pass 12k | -17.0 | -7.0 | **-4.3** |
| low-pass 10k | -17.0 | -7.1 | **-4.3** |
| dropouts | -17.3 | -7.6 | **-4.7** |
| slow wow/flutter | -17.7 | **-10.3** | -10.4 |
| random jitter 0.1% | -19.5 | **-16.3** | -18.3 |
| fast flutter 25/60 Hz | -19.4 | -19.8 | -21.5 |
| warble + low-pass 12k + dropouts | -24.0 | -24.5 | -26.8 |
| heavy jitter 0.3% | -33.0 | -43.1 (**-34.3**) | -49.8 (-43.4) |

Reading:
- **Low-pass and dropouts** do not hurt folding. The Hadamard spreading and
  the equaliser absorb dropouts before they reach the folded symbols.
- **Fast flutter and jitter** do hurt: timing smear pushes symbols over a
  step.
- **Fold 500 with fallback** is the robust choice: +10 on clean, low-pass
  and dropouts, +7 with slow wow, and -0.5 to -1.3 at worst.
- **Look at the contact sheets, not only the numbers.** Where folding scores
  even (fast flutter), it looks different: a fine grain replaces blur.

## Live prototype: send/receive with folding

`tools/v7_live.py` (`vi.modem-send` / `vi.modem-receive`) takes
`--experimental-fold M`, with `M` = 500 (recommended) or 1000. Both ends load
the same frozen table, `test_modem_v7/fold_table_<M>.json`. It is built from
the repository's reference fixture, so neither machine's own frames are
involved.

The prototype fails closed:
- **Pinned tables.** The file must match the SHA-256 pinned in
  `live_fold.py`. Both ends print the hash at startup.
- **Box only.** A table only folds the canonical `box` profile. The sender
  refuses `--encode-filter nearest` and custom fixtures. The receiver shows
  packets from other models unfolded.
- **Own signature only.** The receiver unfolds only packets that carry its
  own table's signature. Packets folded with another table are shown
  unfolded, never mis-unfolded.
- **Hooks restored.** The receiver's hooks on `v7` are restored when it
  exits, including on errors.

```bash
# receiver first (any machine), then the sender
./vi.modem-receive --device "BlackHole 2ch" --experimental-fold 500
./vi.modem-send    --device "BlackHole 2ch" --source video --video-source clip.mp4 \
                   --encode-filter box --experimental-fold 500
```

Replace BlackHole with your real input and output (deck, interface) for tape.

No packet flag exists yet. Instead the 16 weakest folded slots carry a fixed
signature, and it does two jobs:
- **Packet detection.** A receiver with the flag unfolds only packets that
  carry the signature, so it shows normal packets unchanged. It is safe to
  leave on.
- **Noise measurement.** The signature measures each packet's noise on
  exactly the folded slots. The receiver uses it to weight the extra detail,
  and it drops the detail when the packet is too noisy (above 0.3 steps).

### Try it without a sound card

`live_loopback.py` runs the real `tools/v7_live.py` sender and headless
receiver in one process. A stand-in audio device joins them in real time.
It saves every picture shown and scores it against the frame:

```bash
.venv/bin/python test_modem_v7/live_loopback.py --frame FRAME --fold 0
.venv/bin/python test_modem_v7/live_loopback.py --frame FRAME --fold 500
.venv/bin/python test_modem_v7/live_loopback.py --frame FRAME --fold 500 --receiver-fold 0   # mismatch
```

Reference run: one 810x1080 frame, 6 s, about 70 pictures, SSIMULACRA2.

| Sender fold | Receiver fold | Score |
|---:|---:|---:|
| 0 | 0 | -18.7 |
| 500 | 500 | **-7.6** |
| 1000 | 1000 | **-2.5** |
| 500 | 0 | -22.1 (folded, shown unfolded) |
| 0 | 500 | -18.7 (normal packets pass through) |
| 1000 | 500 | -25.2 (other table: shown unfolded, not mis-unfolded) |

`live_fold.py selftest FRAME...` runs the same receive path offline:
- the live sender's value path;
- a moving sequence, with a different frame every packet, pilot tones and
  EOF;
- LiveInput in 1,024-sample blocks, decoding with the live receiver's
  arguments.

It then damages the audio. Reference run (the 11 frames):

| Case | No fold | Fold 500 | Fold 1000 |
|---|---:|---:|---:|
| clean | -18.3 | -8.7 | **-3.8** |
| low-pass 10k | -18.3 | -8.7 | **-4.1** |
| dropouts | -23.9 | -15.5 | **-11.2** |
| wow/flutter | -19.0 | -11.3 | **-10.7** |
| random jitter 0.1% | -20.6 | **-16.0** | -16.4 |
| fast flutter 25/60 Hz | -20.8 | **-18.6** | -20.2 |
| warble + low-pass + dropouts | -26.2 | **-24.4** | -25.8 |
| heavy jitter 0.3% | -38.5 | -39.3 | -40.8 |

Most of the gain under flutter comes from the noise-measured weighting of
the extra detail. Without it, fast flutter scored -23.2 with fold 500.

Hiss is folding's weak spot (`--cases hiss-45 hiss-40 hiss-35 type-ii type-i`,
or any other torture-matrix case name):

| Case | No fold | Fold 500 | Fold 1000 |
|---|---:|---:|---:|
| hiss -45 dBFS | -19.7 | **-14.7** | -15.1 |
| type II | -21.3 | **-19.5** | -21.3 |
| hiss -40 dBFS | -22.9 | -23.0 | -25.4 |
| type I | -29.5 | -31.1 | -33.6 |
| hiss -35 dBFS | -31.9 | -33.7 | -36.0 |

The folded host is sent coarsely. The receiver can drop the extra detail on a
noisy packet, but it cannot restore the host's precision, and the sender
cannot know the tape's noise in advance.

Rebuilding with `live_fold.py build --folds 500 1000` prints new pins. Paste
them into `TABLE_SHA256` in `live_fold.py`; until you do, the prototype
refuses the new tables. The step D belongs to each table: 0.970 for M=500 and
0.8247 for M=1000. These can differ from what `fold_modem.py` picks, because
that script refits D on its own training frames.

Regression tests for the prototype are in the modem suite:

```bash
.venv/bin/python -m unittest modem_tests.test_v7_experimental_fold -v
```

## Experimental coded pilot (tone_code.py)

This separate experiment leaves `animation_modem/` unchanged. On the 24 image
symbols, bin 1 remains a steady reference and bin 3 carries 12 known balanced
BPSK chips on even symbols and 12 status chips on odd symbols. The 12-bit
status is fold mode (2), pinned table ID (4), CRC-4 (4), and spare (2). The
receiver acquires packet starts with the normal edge-counted pulse detector,
uses bin 1 to propose smoothed symbol-window timing corrections, and selects
a correction only if the known bin-3 code correlates better. The status
codebook/CRC can resolve up to two low-confidence chips (or one nearby weak
hard decision), but it never guesses a missing-tone chip.

Run the coded channel against steady-tone and no-tone controls:

```bash
.venv/bin/python test_modem_v7/tone_code.py selftest --packets 12
.venv/bin/python -m unittest discover -s test_modem_v7 -p 'test_*.py' -v
```

In the 12-packet synthetic run, all coded status words decoded in clean,
10 kHz low-pass, hiss (-45 to -35 dBFS), wow/flutter, fast flutter, mains buzz,
and random jitter (0.1% and 0.3%). The dropout case decoded 11/12. Neither
steady-tone nor no-tone controls produced an accepted status in these runs;
their raw code-correlation scores can nevertheless spike under some
impairments, so acceptance also depends on tone presence and a valid status
CRC. These are short synthetic checks, not tape validation.

Compare directly against the pinned folded-500 and folded-1000 baselines:

```bash
.venv/bin/python test_modem_v7/compare_tone_fold.py --packets 12
```

For each fold and impairment, it sends the same folded frame as no-tone,
steady-tone, and coded-tone packets. The original 16-slot fold signature and
noise measurement are active in every run. Each row contains four image paths:
`blind` (no fold table), `status_selected` (use that packet's decoded mode/table),
`latched_status` (retain the last valid status through a rejected packet), and
`oracle` (the sender's table supplied in advance). The coded-status decoder
also probes every tone variant; accepted statuses on steady/no-tone controls
count as false accepts. Results go to
`tmp/test_modem_v7/tone_fold_compare/results.json` (or `--out DIR`).

Earlier fixed-table picture comparisons measured coded-tone impact on the
folded image, but did not let the decoded status choose the fold table. The
metadata-assisted receiver comparison below addresses that question directly.

Compare against the current live V7 defaults as well as a no-fold box control:

```bash
.venv/bin/python test_modem_v7/compare_tone_fold.py \
  --include-current-v7 --include-box-v7 --pilot-timing tone-seeded \
  --tones steady coded --packets 12 \
  --out tmp/test_modem_v7/tone_v7_status
```

The current baseline uses `nearest`, brightness 1.05, steady tones, and
`tone-seeded` receive timing. Folded sends use the pinned `box` profile; the
no-fold box control separates the filter change from the folding gain. The
same tone-seeded receiver is used for every row. Picture demodulation still
does not consume the prototype's timing-offset estimate; this comparison tests
the metadata-driven fold-table choice.

On this reference-fixture run, coded fold-500 improved SSIMULACRA2 over the
current V7 baseline by 1.17–14.85 points across the eight cases; fold-1000
improved it by 0.14–18.96 points. Against the matched no-fold box control, the
ranges were -0.41–10.92 (M=500) and -1.44–15.03 (M=1000); the losses occur at
0.3% jitter. Coded-tone picture scores stayed within -0.52 to +0.29 points of
steady-tone folds. The fold-signature noise measurement changed by at most
0.004 steps for M=500 and 0.0048 for M=1000.

The current-V7 baseline, un-folded box control, and folded steady-tone controls
each had **0/96 valid coded-status accepts** when probed by the coded-status
decoder (0 false accepts). Their `correct` count is not applicable because
those packets do not transmit the coded status. Coded fold-500 decoded 94/96
statuses correctly (2 rejected, 0 wrong accepts); fold-1000 decoded 93/96
(3 rejected, 0 wrong accepts). Those are metadata-channel results, not image
recovery results.

### Does the status help recover the folded image?

Run the receiver-choice comparison (the current-V7 picture baseline above is
available separately):

```bash
.venv/bin/python test_modem_v7/compare_tone_fold.py \
  --pilot-timing tone-seeded --tones steady coded --packets 12 \
  --out tmp/test_modem_v7/tone_v7_metadata_image
```

`blind` decodes a folded packet as ordinary V7 data. `status_selected` uses a
valid status from that packet to choose M=500 or M=1000; if that status is
rejected, it shows the packet without unfolding. `latched_status` carries
forward the last valid mode/table. `oracle` supplies the sender's table in
advance and is the upper-bound check. All paths use the same decoded V7
coefficients, and the fold's embedded signature still has to match before
unfolding is applied.

On the reference fixture across eight synthetic cases, the 5 scored frames per
case recovered the oracle image score exactly with latched metadata for both
folds. Compared with blind decoding, latched status selection improved
SSIMULACRA2 by an average of **8.93 points for M=500** (range +0.24 to +13.69)
and **11.27 points for M=1000** (range +0.27 to +20.26). Immediate per-packet
selection applied the fold to 38/40 scored frames for M=500 and 37/40 for
M=1000; latching recovered all 40/40 for both. The exceptions match rejected
status packets. Steady-tone controls supplied no usable status and applied no
fold, as expected. Detailed per-case metadata and image scores are in
`tmp/test_modem_v7/tone_v7_metadata_image/results.json`.

## How folding works (folding.py)

The weakest M luma body slots each carry two luma coefficients:

    s = D*round(h/D) + beta*clip(u, -2.5, 2.5)

- `h` is the slot's own coefficient (the host), sent coarsely in steps of D.
- `u` is the next luma coefficient V7 does not send (the guest, from the
  96x80 grid outside the 48x40 corner), carried as a small analog residual.
- Both are normalised to unit variance.

The symbol replaces the host at the host's own variance, so a normal V7
encode of the modified coefficients is a folded packet. Power and the
equaliser are unchanged.

The receiver:
1. reads the equaliser's per-coefficient estimate and confidence, using a
   hook on `v7._equalize_numba`;
2. removes the MMSE shrink;
3. splits the symbol back into host and guest.

The offline harness (`fold_modem.py`) uses the plain codec: no signature,
confidence fallback only. The live prototype adds the signature, which
detects folded packets, measures their noise and weights the guests. A real
implementation would put the unfolding step inside `decode_frame`, not in a
wrapper.
