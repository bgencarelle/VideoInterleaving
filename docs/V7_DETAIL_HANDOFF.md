# V7 higher-resolution converter handoff

The requested higher-detail converter is **not solved**. The experimental
fixed-support residual converter (`tools/v7_patch_residual.py`) successfully
encodes one face image into the existing stereo aspect-fold-500 audio packet
and decodes with a separate receiver. The 576×768 output remains blurry and
has wave artifacts; larger raster dimensions are not additional resolved detail.

Fixed a benchmark double inverse-DCT: `base_values()` already returns spatial
values. Earlier residual `source-screen` and `source-screen-v2` results are
invalid. Regression tests verify the production direct-DCT path is called once.
Patch fitting now uses sequential residual seeds and continuous frequency
refinement; independent synthetic quadratures pass the provisional 10% error
threshold. Natural-picture improvement remains small and inconsistent.

Local artifacts (gitignored): `tmp/v7-one-image-roundtrip/` contains the original,
audio-decoded comparison, experimental image, WAV and summary. Reproduce with
`tmp/v7_one_image_roundtrip.py` while available locally. Refined source results:
`tmp/v7-patch-detail/source-screen-refined-v2/`. Always show images inline.

Verification: 30 focused/integration tests passed, followed by the patch-fitting
regression after its final adjustment. Preserve wire resources, independent
frames, source-first qualification, matched volume and Numba numerical kernels.
No 1.5× resolution claim or tape validation is established.
