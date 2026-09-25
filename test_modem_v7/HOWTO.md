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

This is an experimental harness. A real implementation would need a packet
flag (the 2-bit `encoding_type` is full) and an unfolding step inside
`decode_frame`, not a monkeypatch.
