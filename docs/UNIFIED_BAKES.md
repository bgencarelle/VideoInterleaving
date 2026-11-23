# Unified, independent-layer bakes

For the newer compact **RGB ASCII-web** path, use
[`ASCII_RGB_BAKES.md`](ASCII_RGB_BAKES.md) and `utilities/bake_ascii_rgb.py`.
It compresses independently sampled layers instead of storing the full decoder
pixels described below. Other modes and this generic baker remain unchanged.

`utilities/bake.py` builds a package for any or all `main.py` modes. **Face and
float are always separate.** Folder selection and compositing happen at playback
time. No pre-mixed pairs, final JPEGs, ASCII strings or audio packets are baked.

## Build

Run from the repository root:

```bash
# Storage planning only; writes no assets.
.venv/bin/python utilities/bake.py -i images_sbs --modes all --estimate

# All modes, including vector-capable scope libraries.
.venv/bin/python utilities/bake.py -i images_sbs -o images_bakes --modes all

# Or choose the outputs you need.
.venv/bin/python utilities/bake.py -i images_sbs -o display_bakes \
    --modes local web ascii asciiweb
```

Use a new destination outside the source tree. The complete package is staged
alongside its destination and installed only after all requested outputs succeed.
Existing packages are not overwritten. Rebuild to a new directory after changing
source pictures: a bake is an explicit snapshot, not an automatically fresh cache.

Inputs use numeric `face/0_...` through `face/254_...` and `float/255_...` folders.
Natural ordering and the application's largest-common-frame-count rule select
the playable folders. Display manifests retain filenames and source checksums.

Options:

- `--jpeg-layout sbs` (default): project colour/matte JPEGs; `rgb` explicitly
  selects ordinary opaque JPEGs for all backends. PNG/WebP alpha remains.
- `--workers 2`: bound parallel scope folder jobs. Other writers stream frames.
- `--thumb-width 128`: scope thumbnails, matching the current baker default.
- `--thumbs-only`: omit scope vector contours. Raster/stochastic/stipple remain;
  vector-containing scope mix/fusion settings require a normal full bake.
- `--web-size WxH`: match `settings.HEADLESS_RES`; defaults to that setting.
  Only the existing eligible half-size JPEG decode level is prepared. There is
  no new arbitrary downscale. Playback rejects a changed web target when that
  reduced decode level is in use, instead of silently reducing quality.

## Play

```bash
.venv/bin/python main.py --mode local    --bake-dir images_bakes
.venv/bin/python main.py --mode web      --bake-dir images_bakes
.venv/bin/python main.py --mode ascii    --bake-dir images_bakes
.venv/bin/python main.py --mode asciiweb --bake-dir images_bakes
.venv/bin/python main.py --mode scope    --bake-dir images_bakes --device null
.venv/bin/python main.py --mode modem    --bake-dir images_bakes \
    --modem-clock 255 --modem-pair 0,0 --modem-frames 30 --modem-wav output.wav
```

`--bake-dir` selects its own source: omit `--dir`, `--xy-dir`, `--modem-dir` and
runtime-image source flags. Original direct-image and standalone XY/modem CLI
paths remain available. Modem uses the existing application V7 controls; the
separate standalone V7 sender/receiver is unaffected.

## Runtime work

| Mode | Independent baked input | Work retained live |
|---|---|---|
| Local | Lossless original decoder pixels and SBS interpretation | Exact index/folders, orientation, GPU uploads and compositing |
| Web | Same pixels, or existing eligible half-decode level | Composition and JPEG encoding |
| ASCII / ASCII-web | Full-resolution decoder pixels | Composition **before** grading/conversion, live ASCII settings |
| Scope | Existing per-folder XY/thumbnail/stipple format | Layer mixing, trajectories, field latching and output |
| Modem | Existing per-folder 80x96 RGBA slabs | Layer mixing, encoding, timestamp reservations and audio output |

Display modes share one store when their decode levels match. No folder-pair
cross product is generated. Runtime changes to face/float pairings remain valid.
SBS data is kept intact, preserving the renderer's original matte sampling.

Display playback reads the manifest without a source scan or a frames-by-folders
Python path list. Offset/shape tables are mmap-backed. Only requested pixel ranges
are mapped; released views unmap their pages instead of accumulating movie-sized
RSS. A four-view identity cache reuses duplicate assets. The existing bounded
deadline preparer and exact-index selection remain in control.

## CPU, RAM and storage tradeoffs

Decoded display pixels eliminate decompression and per-frame file opens, but are
**much larger on disk than JPEGs**. This is a CPU-versus-storage tradeoff, not a
universally smaller representation. Identical decoded assets share storage while
remaining independently selectable layers. Completed builds print actual bytes
and source-size ratios. Count shared stores only once when summing package size.

The selected workspace `images_sbs` source has 2,221 frames in 29 folders and
about **6.03 GiB** of JPEGs. Its full decoder pixels are estimated at **248.8 GiB
before deduplication**, assuming constant dimensions within each folder. This is
a header estimate, not a full-tree bake measurement. Cold storage traffic and
page stalls can be worse even when warm decode CPU improves.

Scope and modem reuse their specialised representations without changing scope
defaults or wire design. Scope operational evidence belongs to the latest
appendix of `docs/SCOPE_UPGRADE.md`.

`utilities/bake_assets.py` remains a legacy generic slab writer and supplies
`write_slab` to the modem baker. Its standalone `frames.npy` is **not** the new
display format; use the unified command and `--bake-dir` for supported playback.

## Measured comparison — 2026-10-16

Representative sample: 96 source positions spread across the 2,221-frame
`images_sbs` sequence, two face and two float folders; 29.90 MiB of original
JPEGs. The original files were hardlinked into a temporary test tree. Playback
selected pair `(0,0)` in both source and baked cases. CPU 100% means one core;
RAM is process median RSS. These are warmed software measurements, not universal
bounds or hardware/browser timing evidence.

| Mode | Images CPU → baked CPU | Images RAM → baked RAM | Bake storage / source |
|---|---:|---:|---:|
| Local | 38.2% → 30.0% | 297.5 → 289.0 MiB | **38.23×** |
| Web | 14.8% → 11.7% | 260.1 → 254.4 MiB | **9.56×** |
| ASCII | 18.8% → 15.1% | 295.2 → 272.8 MiB | **38.23×**, shared with local |
| ASCII-web | 18.5% → 15.8% | 294.5 → 272.8 MiB | **38.23×**, shared with local |
| Scope raster | 60.7% → 9.5% | 376.8 → 353.7 MiB | 0.62×, thumbs-only sample |
| Modem | 28.8% → 10.5% | 289.4 → 269.5 MiB | 0.38× |

**All display bakes used more storage. No final measured mode used more warmed
CPU or RSS on this workload.** Full local/ASCII pixels took 1,143.08 MiB; web's
eligible half-decode level took 285.82 MiB; scope 18.67 MiB; modem 11.28 MiB.
The shared local/ASCII store is counted once. Full vector-capable scope bakes
have additional geometry; the table's scope storage is not a full-geometry claim.

Display numbers average two trials, each with a two-second warmup and six-second
measurement, using software GL without network clients. The web bake had to match
the actual `HEADLESS_RES=(320,400)` decode policy: an initial full-resolution
candidate regressed to 18.3% CPU / 272.3 MiB versus direct images' 14.9% / 261.1
MiB. The corrected mode-specific decode level above passed exact pixel/render
parity. This finding also corrected the Python baker API's target default to read
settings, just as the CLI already did.

Scope used actual `main.py`, raster at unchanged defaults and NullStream, with
six-second HTTP-observed phases. With a 512-pixel trace viewer, CPU/RSS were
80.7% / 378.2 MiB → 27.2% / 358.5 MiB. Luminance viewer: 63.2% / 378.3 MiB →
9.7% / 353.9 MiB. Scope source snapshots include the existing deadline preparer;
there is no trace-scheduler replacement. Modem used actual `main.py` over the
verified virtual `pulse -> loop -> loop.monitor` route, 100 packets, skipping
the first twelve for measurement. Both sources submitted 88/88 measured exact
indices. This is producer evidence, not receiver-display validation.

Direct-source and unified display loops selected exact identities on every
sampled update in these resource trials. Separate checks compare decoder pixels,
GL framebuffers, JPEG bytes and final ASCII output while trying every face/float
combination in the sample. The package/native application modem WAV entry points
also exported identical bytes from the same independently baked slabs.

Evidence: `tmp/unified_measure_summary.json`,
`tmp/image_resources_*_unified_{images,baked}_{1,2}.json`,
`tmp/unified_scope_{images,baked}.json`, `tmp/unified_modem_resources.json`,
`tmp/unified_pixel_parity.json`, and `tmp/unified_modem_wav_parity.json`.

Final focused validation passed **153 tests**, with one utility-script skip and
12 subtests. Pixel checks compared **384 decoded layer frames** and **24
source/baked GL framebuffer, ASCII-text and JPEG-byte results**, including all
four live face/float combinations at selected positions. Actual package-backed
`main.py` runs recorded 121 local swaps, 60 web publications, 61 ASCII
publications, 60 ASCII-web publications and 81 sampled scope adoptions, without
harness errors. The mapping-cleanup tests also verify that outstanding pixel
views stay valid after their library closes.

The required broader `modem_tests` discovery was also run: **1,072 tests in
196 seconds; 118 errors and two failures**. The errors are experimental
`test_v7_analog_frame` cases whose `matched_viewer` kernel lacks profile defaults
for `aspect-fold-500` / `aspect-mono-500`. The two failures are existing slice
tests (error-message precedence and a reconstruction threshold). Representative
analog and both slice failures reproduced with every module changed by the
baker task blocked from import, confirming that these failures do not exercise
the new bake paths. They were not modified as part of this delivery. Logs:
`tmp/unified_modem_regression.log` and
`tmp/unified_unrelated_failure_reproduction.{log,json}`. The overall repository
suite is therefore not reported as green.
