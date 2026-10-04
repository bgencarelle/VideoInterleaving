# CSF Diamond performance note

**Date:** 2026-10-04
**Fixture:** first frame of `modem_tests/fixtures/v7_pixel_motion_16x9.mp4`
(`854x480`; V7 grids `96x80`, `48x40`, `48x40`).

## Measurements

After warming Numba and kernel caches, 120 paired per-frame samples measured
the sender's `_values()` preparation plus Fold 500 coefficient encoding:

| Mode, luma adjustment off | Median per frame |
| --- | ---: |
| Reference | 2.910 ms |
| CSF Diamond, exact filtering | 3.290 ms |
| CSF Diamond, deferred Fold gain | 2.970 ms |

The deferred path's paired median overhead over reference was `0.062 ms`, versus
`0.383 ms` for exact CSF filtering. With luma adjustment on, the exact path
measured `5.207 ms` versus `4.516 ms` for reference; gain deferral is disabled
for that mode.

These are local measurements, not a hardware-independent guarantee. The
deferred timing includes source conversion and Fold coefficient generation,
but not pulse-frame rendering or audio output.

## Output behavior

The direct path applies the gain before its value clamp. Fold gain deferral
reuses Fold's existing DCT and applies the gain after that clamp, so clipped
samples can differ. On unclipped inputs the paths matched within `3e-13` in the
regression test. On reconstructed RGB from three frames in each of five
repository clips, deferred versus exact CSF filtering scored SSIM `0.9925` to
`0.9975` (normalized RMS error `0.0097` to `0.0149`).

Deferral is limited to pure-gain kernels in the Fold sender when luma
adjustment and clip-aware encoding are off. Kernels with post-filters or
pre-filters and other sender paths keep exact filtering. CSF Diamond continues
to shape chroma by default; `color_planes=0` opts out.

## Implementation and validation

- Cached gain matrices are reused; a Numba gain-builder was slower than the
  existing NumPy calculation and was not used.
- Exact windowing uses SciPy's grid DCT/IDCT. The luma-target transfer
  functions use cached Numba loops to avoid temporary arrays.
- Fold applies deferred per-plane gains during its existing coefficient
  transform. The wire format and frame cadence are unchanged.
- After integrating the remote GUI/pre-shrink changes, 59 focused kernel/Fold
  tests, 126 sender-GUI tests, and integration/lazy-import tests passed. The
  ASCII tests also passed during the original validation.
- Full modem discovery during the original validation had one isolated failure in
  `test_untagged_hd_files_are_read_with_the_hd_colour_matrix`: the test received
  `None` instead of `bt709`. The same failure reproduced when run alone.
