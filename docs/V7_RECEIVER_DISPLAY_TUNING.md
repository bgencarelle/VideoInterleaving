# V7 receiver display defaults

**Date:** 2026-10-04

## Scope and method

This tunes the standalone V7 receiver GUI's display stages; these settings do
not change packet decoding or the wire. The current Aspect Fold 500 sender
default (Viewer-model solve) and the Native-source DCT path were evaluated
through Aspect Fold 500's aspect-specific coefficient masks. Four natural
images (`astronaut`, `chelsea`, `coffee`, `rocket`) set the ranking. Six frames
(0, 12, 36, 72, 144, 216) from
`modem_tests/fixtures/v7_pixel_motion_16x9.mp4` are reported separately as a
hold-out.

The benchmark ideal-decodes the coefficients selected by each mask, runs the
same receiver plane reconstruction stages as
`tools.v7_gl_viewer.dct_reconstruct_planes`, and scores a 900-pixel-high output
with SSIMULACRA2 (higher is better). It compared DCT reconstruction (`off`,
`2x`, `4x`, `8x`, `16x`, `viewport`), edge reconstruction and strength, guided
colour detail, and the available display upscalers. These are image-quality
measurements on ideal coefficient data, not audio-channel or tape results.

## Result and recommendation

The receiver GUI now defaults **Colour detail** to **Luma-guided**. At the
current display defaults—4× reconstruction, Bicubic, and edge reconstruction
on at 75%—guided colour detail improved the four-image mean for both encoders:

| Encode path | Colour detail off | Luma-guided | Change |
| --- | ---: | ---: | ---: |
| Aspect Fold 500 default, Viewer-model solve | -63.800 | -63.727 | +0.073 |
| Native-source DCT | -63.843 | -63.780 | +0.063 |

The six-frame motion hold-out goes the other way at those settings: its mean
changes from -142.407 to -142.911 for Viewer-model solve, and from -140.241 to
-140.449 for Native-source DCT. The per-frame effect is mixed; for example,
the first motion frame improves, while several later frames lose detail score.
Guided colour detail estimates chroma detail from luma, so it can improve
natural-image colour edges without recovering additional transmitted data.
The GUI keeps **Off · colour as sent** available as a one-click alternative.

The remaining defaults stay at **4× DCT reconstruction**, **Bicubic**, and
**edge reconstruction on at 75%**:

- Exact `viewport` reconstruction improved the natural-image mean by only 0.002
  points over 4× with colour detail off, while the six-frame hold-out slightly
  preferred 4×. A warmed CPU check at a 900×750 viewport measured about 2.7 ms
  for 4× versus 6.7 ms for `viewport` per frame.
- 8×, 16× and `viewport` were within 0.01 points of 4× on the natural-image
  mean at the strongest configurations. For the tested screen size, 4× was
  also substantially cheaper than exact `viewport` reconstruction.
- Bicubic was within 0.003 points of the best display filter on natural images
  for both encoders, and within 0.003 on the Viewer-model solve hold-out.
  Hann-sinc ranked highest on natural images; Robidoux ranked highest on that
  hold-out. This difference is negligible, so Bicubic remains the balanced
  default.
- Raising edge strength to 100% reduced natural-image scores slightly (about
  0.02 points versus 75%) but substantially improved the motion hold-out. The
  existing 75% setting remains the natural-image balance and avoids making
  natural textures look overly flat. Turning edge reconstruction off scored
  substantially worse on the hold-out.

Display grain remains off: it intentionally adds noise rather than detail.
Output dither remains on to reduce visible banding in smooth gradients. Pixel
display remains an explicit hard-pixel view, not the default for direct-DCT
images.

## Receiver configuration

`tools/v7_gl_viewer.py` defines the receiver GUI's recommended display values;
`tools/v7_receiver_gui.py` uses them to initialize its setup and live controls.
The defaults apply to both tested encode paths because the encoder choice is
not signalled as a separate decoder display mode in the packet.
