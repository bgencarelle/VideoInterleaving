# V7 DCT-kernel defaults: comparison and tuning

**Date:** 2026-10-04

## Test method

`tools/v7_kernel_defaults_sweep.py` was run on every one of the 13 shipped
kernels at both the ideal reconstruction and bilinear-upscale display modes.
For each kernel it scored the current defaults, then swept each declared
parameter independently at its low, midpoint, and high values while holding
the remaining parameters at their defaults. This produced 110 parameter probes
per display mode. Four natural images bundled with scikit-image—`astronaut`,
`chelsea`, `coffee`, and `rocket`—determined the SSIMULACRA2 ranking. Six frames
(0, 12, 36, 72, 144, 216) from
`modem_tests/fixtures/v7_pixel_motion_16x9.mp4` were a separate stress hold-out,
not part of the tuning score.

All tests used the sender's direct-DCT path, GUI-default luma adjustment,
sent-rectangle coefficient masks, and an audio-free ideal receiver. The
benchmark also measured edge overshoot, rise width, ripple, retained sine-wave
contrast, and warmed encode time. The screen is repeatable with:

```bash
.venv/bin/python tools/v7_kernel_defaults_sweep.py \
  --display ideal \
  --image tmp/v7-kernel-tuning/frames/pixel_motion_000.png \
  --image tmp/v7-kernel-tuning/frames/pixel_motion_012.png \
  --image tmp/v7-kernel-tuning/frames/pixel_motion_036.png \
  --image tmp/v7-kernel-tuning/frames/pixel_motion_072.png \
  --image tmp/v7-kernel-tuning/frames/pixel_motion_144.png \
  --image tmp/v7-kernel-tuning/frames/pixel_motion_216.png \
  --csv tmp/kernel-ideal.csv
.venv/bin/python tools/v7_kernel_defaults_sweep.py \
  --display bilinear \
  --image tmp/v7-kernel-tuning/frames/pixel_motion_000.png \
  --image tmp/v7-kernel-tuning/frames/pixel_motion_012.png \
  --image tmp/v7-kernel-tuning/frames/pixel_motion_036.png \
  --image tmp/v7-kernel-tuning/frames/pixel_motion_072.png \
  --image tmp/v7-kernel-tuning/frames/pixel_motion_144.png \
  --image tmp/v7-kernel-tuning/frames/pixel_motion_216.png \
  --csv tmp/kernel-bilinear.csv
```

Repeat `--image FRAME.png` for each hold-out frame. Use `--defaults-only` to
score the table without parameter probes. Results and extracted frames were
kept under the ignored repository-local `tmp/v7-kernel-tuning/` directory.

## Mean natural-image score

Higher SSIMULACRA2 is better. The reference is the shipped encode with no
optional kernel. The first pair of score columns is before tuning; the latter
pair uses the updated defaults.

| Kernel | Ideal before | Ideal tuned | Δ | Bilinear before | Bilinear tuned | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reference | -26.71 | -26.71 | 0.00 | -45.28 | -45.28 | 0.00 |
| Anti-ringing refit | -27.53 | -27.53 | 0.00 | -45.96 | -45.96 | 0.00 |
| Band taper | -38.42 | -33.06 | +5.36 | -53.31 | -49.58 | +3.73 |
| CSF Diamond | -44.29 | -27.18 | +17.11 | -50.24 | -45.27 | +4.97 |
| Mid-band emphasis | -34.24 | -26.24 | +8.01 | -44.96 | -43.75 | +1.21 |
| Gaussian | -43.77 | -43.77 | 0.00 | -55.69 | -55.69 | 0.00 |
| Lanczos | -29.09 | -27.37 | +1.72 | -47.06 | -46.01 | +1.05 |
| Magic Kernel Sharp | -28.80 | -28.80 | 0.00 | -44.30 | -44.30 | 0.00 |
| Mitchell-Netravali | -39.32 | -34.42 | +4.91 | -53.65 | -50.64 | +3.01 |
| Shock-filter edges | -27.05 | -26.66 | +0.39 | -42.65 | -43.53 | -0.88 |
| Slepian (DPSS) | -47.46 | -30.93 | +16.53 | -56.91 | -49.61 | +7.30 |
| TV cartoon refit | -37.74 | -32.95 | +4.79 | -52.78 | -50.30 | +2.48 |
| Upscaler pre-compensation | -31.25 | -31.25 | 0.00 | -39.20 | -39.20 | 0.00 |
| Viewer-model solve | -39.36 | -31.60 | +7.76 | -35.47 | -36.61 | -1.14 |

## Best kernel choice by benchmark mode

The GUI labels `Reference` as the shipped default (no kernel), and marks the
highest-natural-score kernel separately for each tested display mode. Those
labels are recommendations, not a claim that one kernel wins for every
viewer: `ideal` is the ideal-reconstruction test, while `bilinear` is a fixed
bilinear-resampling simulation. The bilinear test does not detect the receiver
window size or a runtime resolution threshold. The six-frame hold-out is
reported separately and does not select the winner.

| Profile | Best ideal-test kernel | Best bilinear-test kernel |
| --- | --- | --- |
| General V7 | Mid-band emphasis | Viewer-model solve |
| Aspect Mono | Mid-band emphasis | Viewer-model solve |
| Aspect Stereo | Viewer-model solve | Viewer-model solve |

The sender GUI shows the selected kernel's profile-specific parameter defaults
on its parameter rows. Choosing a marked winner is still explicit; the default
kernel remains `Reference` unless changed.

The CSF Diamond default moved from the worst ideal-mode and second-worst
bilinear-mode natural-image mean to within 0.47 points of reference in ideal
mode and 0.01 points in bilinear mode. On the stress hold-out it scored `-61.39`
against reference `-61.37` in ideal mode, and `-64.22` against `-64.22` in
bilinear mode. Disabling chroma shaping gained 0.30 ideal-mode and 0.06
bilinear-mode points; chroma shaping was retained by default. Its tuned ideal
edge overshoot/ripple were `7.99%` / `1.06%`,
versus reference `7.53%` / `1.05%`; in bilinear mode they were `4.44%` / `0.47%`
versus `3.98%` / `0.40%`.

## Default changes

- `csf_diamond`: `amount=.05`, `oblique=.95`, `p_norm=1.9`, `rolloff=.55`,
  and `HOST_DEFAULTS['luma_mix']=.25`. Chroma shaping remains on. On the four
  natural images, reducing luma mix from `1.0` to `.25` was the dominant gain;
  larger diamond contrast/roll-off settings reduced quality and increased
  edge ringing.
- `band_taper`: `power=3.0`, balancing the improved detail retention of higher
  powers against the additional edge overshoot.
- `csf_peak`: `amount=.05`, `rolloff=.55` for a restrained mid-band lift without
  cutting the top of the transmitted band.
- `lanczos`: `width=.9`, a modest quality improvement over `1.0` without using
  the most aggressive width tested.
- `mitchell`: `C=.8` to recover detail while keeping `B=.33`.
- `shock_edge`: `rounds=2`, reducing the default work from three projection
  rounds while retaining most of the measured edge steepening.
- `slepian`: `nw=.5`, `floor=.15`, improving retained detail while maintaining
  a smoother band edge than the higher-floor alternatives.
- `tv_cartoon`: `keep=.5`, retaining more source texture while preserving the
  stylized refit.
- `viewer_solve`: `lam=.5`, a more regularized solve with lower measured edge
  ripple.

Anti-ringing, Gaussian, Magic Kernel Sharp, and Upscaler pre-compensation kept
their existing defaults: their highest-scoring probes either approached an
identity/no-op setting, weakened the intended edge/ringing tradeoff, or did not
hold up for the matched bilinear-viewer setting. The shock-edge bilinear
natural-image mean decreased by 0.10 points after reducing rounds; this small
quality trade buys lower encode cost and the ideal-mode result improved by
0.39.

## Updated benchmark snapshot

Warmed local encode-time results, in milliseconds per 960x800 source frame,
were collected with `tools/v7_kernel_bench.py --score` in ideal-display mode.
Encoding cost does not depend on the viewer's display scaler:

| Kernel | ms/frame |
| --- | ---: |
| Reference | 5.21 |
| Anti-ringing refit | 9.45 |
| Band taper | 6.06 |
| CSF Diamond | 6.19 |
| Mid-band emphasis | 6.12 |
| Gaussian | 6.03 |
| Lanczos | 6.03 |
| Magic Kernel Sharp | 6.14 |
| Mitchell-Netravali | 5.94 |
| Shock-filter edges | 12.53 |
| Slepian (DPSS) | 6.35 |
| TV cartoon refit | 11.97 |
| Upscaler pre-compensation | 10.05 |
| Viewer-model solve | 16.56 |

These timings are local, include encoding and luma adjustment, and are not
hardware-independent guarantees. Bilinear reconstruction's chroma-plane
coefficient offset was corrected in `tools/v7_kernel_bench.py`; the bilinear
quality scores above were rerun with the corrected path. The corrected full
parameter sweep is `tmp/v7-kernel-tuning/sweep-bilinear-corrected.csv`. The
separate CSF performance note contains the earlier exact/deferred Fold
comparison; its CSF parameters predate this retuning.

## Aspect Mono profile

Aspect Mono and Aspect Stereo have separate profile defaults, measured with
their own layout-specific coefficient masks. The sender GUI and live controls
resolve the selected profile's values unless a saved or command-line override
is present. Other profiles keep the shared tuned defaults.

The Aspect Mono benchmark builds the actual `AspectMonoWire` masks for each
image's aspect code, including the luma fold guests, and scores the reconstructed
image at that aspect. The natural-image objective and six-frame V7 hold-out are
the same as above. The final defaults-only runs are:

```bash
.venv/bin/python tools/v7_kernel_defaults_sweep.py \
  --defaults-only --profile aspect-mono-500 --display ideal \
  --image tmp/v7-kernel-tuning/frames/pixel_motion_000.png \
  --image tmp/v7-kernel-tuning/frames/pixel_motion_012.png \
  --image tmp/v7-kernel-tuning/frames/pixel_motion_036.png \
  --image tmp/v7-kernel-tuning/frames/pixel_motion_072.png \
  --image tmp/v7-kernel-tuning/frames/pixel_motion_144.png \
  --image tmp/v7-kernel-tuning/frames/pixel_motion_216.png \
  --csv tmp/v7-kernel-tuning/aspect-mono-ideal-final.csv
.venv/bin/python tools/v7_kernel_defaults_sweep.py \
  --defaults-only --profile aspect-mono-500 --display bilinear \
  --image tmp/v7-kernel-tuning/frames/pixel_motion_000.png \
  --image tmp/v7-kernel-tuning/frames/pixel_motion_012.png \
  --image tmp/v7-kernel-tuning/frames/pixel_motion_036.png \
  --image tmp/v7-kernel-tuning/frames/pixel_motion_072.png \
  --image tmp/v7-kernel-tuning/frames/pixel_motion_144.png \
  --image tmp/v7-kernel-tuning/frames/pixel_motion_216.png \
  --csv tmp/v7-kernel-tuning/aspect-mono-bilinear-final.csv
```

Higher SSIMULACRA2 is better. Parentheses show the delta from the Aspect Mono
reference (no kernel); natural pictures and the six-frame hold-out are kept as
separate columns.

The best natural-image score is Mid-band emphasis in ideal mode and
Viewer-model solve in bilinear mode. The GUI labels both choices for this
profile; the hold-out scores remain separate from this selection.

| Kernel | Ideal natural | Ideal hold-out | Bilinear natural | Bilinear hold-out |
| --- | ---: | ---: | ---: | ---: |
| Reference | -54.93 | -86.65 | -62.54 | -145.98 |
| Anti-ringing refit | -54.93 (-0.00) | -87.10 (-0.45) | -62.54 (-0.00) | -147.51 (-1.53) |
| Band taper | -54.85 (+0.08) | -85.57 (+1.08) | -62.58 (-0.04) | -144.98 (+1.00) |
| CSF Diamond | -54.98 (-0.05) | -86.95 (-0.30) | -62.53 (+0.01) | -146.65 (-0.67) |
| Mid-band emphasis | -54.56 (+0.37) | -85.53 (+1.12) | -62.05 (+0.49) | -140.19 (+5.79) |
| Gaussian | -54.95 (-0.02) | -86.68 (-0.03) | -62.57 (-0.03) | -146.42 (-0.44) |
| Lanczos | -54.92 (+0.01) | -86.58 (+0.07) | -62.53 (+0.01) | -145.95 (+0.02) |
| Magic Kernel Sharp | -55.26 (-0.33) | -87.83 (-1.18) | -62.24 (+0.31) | -144.21 (+1.77) |
| Mitchell-Netravali | -55.15 (-0.22) | -87.81 (-1.16) | -62.64 (-0.09) | -149.19 (-3.21) |
| Shock-filter edges | -54.89 (+0.04) | -86.27 (+0.37) | -62.44 (+0.11) | -147.29 (-1.31) |
| Slepian (DPSS) | -55.01 (-0.08) | -86.93 (-0.28) | -62.64 (-0.10) | -147.69 (-1.71) |
| TV cartoon refit | -55.04 (-0.11) | -86.67 (-0.02) | -62.63 (-0.09) | -146.29 (-0.31) |
| Upscaler pre-compensation | -54.96 (-0.03) | -85.98 (+0.67) | -62.22 (+0.32) | -141.97 (+4.01) |
| Viewer-model solve | -54.69 (+0.24) | -85.63 (+1.02) | -61.90 (+0.64) | -135.28 (+10.70) |

The Aspect Mono settings reduce luma mix to `0.25` for most kernels; the
exceptions are CSF Diamond (`0.1`) and Mid-band emphasis (`2.0`, with chroma
shaping off). Additional selected values are: Band taper `edge=0.8` with
`use_guests=1`; Gaussian `sigma=0.1`; Lanczos `width=0.5`; Magic Kernel Sharp
`sharp=1.5`; Mitchell `B=0`; Shock edges `strength=0.9`; Slepian `floor=0.5`;
TV `keep=0.75`; and no post-solve halo-cleanup rounds for Upscaler pre-comp and
Viewer-model solve. These are Aspect-Mono-specific; the shared defaults above
are unchanged.

## Aspect Stereo profile

The `aspect-fold-500` benchmark builds the actual Aspect Fold masks for every
aspect code, using the sender GUI's default fixed tail. It scores each image at
its aspect ratio and includes the luma fold guests in the sent mask. The final
defaults-only runs use the same four natural-image objective and six-frame
hold-out as the Mono sweep:

```bash
.venv/bin/python tools/v7_kernel_defaults_sweep.py \
  --defaults-only --profile aspect-fold-500 --display ideal \
  --image tmp/v7-kernel-tuning/frames/pixel_motion_000.png \
  --image tmp/v7-kernel-tuning/frames/pixel_motion_012.png \
  --image tmp/v7-kernel-tuning/frames/pixel_motion_036.png \
  --image tmp/v7-kernel-tuning/frames/pixel_motion_072.png \
  --image tmp/v7-kernel-tuning/frames/pixel_motion_144.png \
  --image tmp/v7-kernel-tuning/frames/pixel_motion_216.png \
  --csv tmp/v7-kernel-tuning/aspect-stereo-ideal-final.csv
.venv/bin/python tools/v7_kernel_defaults_sweep.py \
  --defaults-only --profile aspect-fold-500 --display bilinear \
  --image tmp/v7-kernel-tuning/frames/pixel_motion_000.png \
  --image tmp/v7-kernel-tuning/frames/pixel_motion_012.png \
  --image tmp/v7-kernel-tuning/frames/pixel_motion_036.png \
  --image tmp/v7-kernel-tuning/frames/pixel_motion_072.png \
  --image tmp/v7-kernel-tuning/frames/pixel_motion_144.png \
  --image tmp/v7-kernel-tuning/frames/pixel_motion_216.png \
  --csv tmp/v7-kernel-tuning/aspect-stereo-bilinear-final.csv
```

Higher SSIMULACRA2 is better. Parentheses show the change from the Aspect
Stereo reference; natural-image and hold-out scores remain separate:

Viewer-model solve has the highest natural-image mean in both tested modes for
Aspect Stereo. The GUI labels it as the best ideal and bilinear benchmark
choice.

| Kernel | Ideal natural | Ideal hold-out | Bilinear natural | Bilinear hold-out |
| --- | ---: | ---: | ---: | ---: |
| Reference | -41.60 | -66.36 | -61.44 | -108.56 |
| Anti-ringing refit | -41.60 (-0.00) | -66.38 (-0.02) | -61.44 (-0.00) | -108.72 (-0.16) |
| Band taper | -41.48 (+0.12) | -66.24 (+0.13) | -61.43 (+0.01) | -107.87 (+0.69) |
| CSF Diamond | -41.83 (-0.23) | -66.43 (-0.06) | -61.44 (-0.00) | -108.95 (-0.39) |
| Mid-band emphasis | -41.13 (+0.47) | -66.13 (+0.24) | -60.64 (+0.80) | -104.43 (+4.13) |
| Gaussian | -41.57 (+0.03) | -66.36 (+0.01) | -61.44 (-0.00) | -108.61 (-0.05) |
| Lanczos | -41.61 (-0.01) | -66.27 (+0.10) | -61.41 (+0.03) | -108.55 (+0.01) |
| Magic Kernel Sharp | -41.85 (-0.25) | -66.59 (-0.23) | -61.44 (+0.00) | -110.05 (-1.48) |
| Mitchell-Netravali | -41.49 (+0.11) | -66.38 (-0.02) | -61.48 (-0.04) | -109.32 (-0.76) |
| Shock-filter edges | -41.08 (+0.52) | -66.57 (-0.20) | -60.99 (+0.45) | -115.78 (-7.21) |
| Slepian (DPSS) | -41.45 (+0.15) | -66.32 (+0.05) | -61.47 (-0.03) | -108.73 (-0.17) |
| TV cartoon refit | -41.47 (+0.13) | -66.34 (+0.02) | -61.46 (-0.02) | -108.49 (+0.07) |
| Upscaler pre-compensation | -41.71 (-0.11) | -66.19 (+0.17) | -60.98 (+0.46) | -105.43 (+3.13) |
| Viewer-model solve | -40.89 (+0.71) | -66.23 (+0.14) | -60.48 (+0.96) | -101.07 (+7.49) |

The selected Aspect Stereo settings use low luma mixes for kernels that
otherwise over-shape the much larger stereo allocation: `0.05` for Anti-ringing,
Band taper, CSF Diamond, Gaussian, Magic Kernel Sharp, Mitchell, Slepian and TV;
`0.5` for Lanczos; and `0.2` for Upscaler pre-compensation and Viewer-model
solve. Band taper uses `edge=0.9` and the fold guests; CSF Diamond's luma mix
is `0.05`; Mid-band emphasis uses `luma_mix=2.0` with chroma shaping off. TV
uses `keep=0.75`; the two viewer-matched kernels disable halo cleanup with
`tame=0`. Shock-filter edges retains the shared defaults, which scored best on
the natural-image objective. The stereo-specific values do not alter the
Aspect Mono or general V7 defaults.
