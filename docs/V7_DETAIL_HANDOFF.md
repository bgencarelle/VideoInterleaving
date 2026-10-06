# V7 higher-resolution converter handoff

The higher-detail goal is **not solved**. The original theory folded
higher-dimensional source-detail vectors into fewer analog amplitudes using
Shannon–Kotel’nikov-style serpentine mappings. More strips reduced source
projection error but amplified channel error and boundary failures. Recursive
packing compounded sensitivity. Three-to-two offers 1.5× coordinate capacity,
not 1.5× resolution per axis; strip confirmations showed no general gain.

Work shifted to source-aware waves, contours and DCT residuals. The final
fixed-support converter carries frequencies and quadrature amplitudes in
displaced conventional luma slots. Useful synthesized detail has not consistently
compensated for lost base coefficients.

Fixed a benchmark double inverse-DCT: `base_values()` is already spatial.
Earlier residual `source-screen` and `source-screen-v2` results are invalid.
Regressions confirm one production direct-DCT source call. Synthetic fitting
passes, but natural-image gains remain small and inconsistent.

`tools/v7_one_image_roundtrip.py` encodes a face through stereo aspect-fold-500
audio and an independent decoder. Its 576×768 output is blurry with wave
artifacts. Local outputs: `tmp/v7-one-image-roundtrip/`; refined source summary:
`tmp/v7-patch-detail/source-screen-refined-v2/`. Always show images inline.

Thirty focused/integration tests passed, plus the final fitting regression.
Preserve wire resources, independent frames, matched volume, source-first
qualification and Numba computation. No 1.5× gain or tape validation is established.
