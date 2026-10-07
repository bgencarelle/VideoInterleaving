# V7 higher-resolution converter handoff

## Resolved: see `V7_SK_VERDICT.md`

Now in the live tools as two experimental sender options, `aspect-mono-nested`
and `stereo-nested` (sender GUI profile list; the receiver reads them without a
setting). One table per layout, soft per-packet decoding, speed and reverse
supported. Start at the top section of `V7_SK_VERDICT.md`.
Open issues (grain, banding, colour, the filter still to build): `V7_NESTED_FOLD_ISSUES.md`.

The folding question is answered. An adaptive nested fold carries 58% more
effective luma detail on the clean real wire (1.26× linear), 52% at hiss −60,
shrinking to 8% at hiss −40. The Shannon ceiling at measured V7 precision is 1.74× clean, so
3× is impossible and 1.5× is out of reach of any mapping built here. The strip
trials below measured a defective mapping, not the idea. Code:
`tools/v7_sk_fold.py`, `v7_sk_adaptive.py`, `v7_sk_study.py`, `v7_sk_wire.py`; tests:
`modem_tests/test_v7_sk_fold.py`. Not yet run on faces, mono or tape.

## Earlier state (superseded where it conflicts)

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
