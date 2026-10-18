# Compressed RGB ASCII-web bakes

Run from the repository root:

```bash
.venv/bin/python utilities/bake_ascii_rgb.py \
    -i images_sbs -o tmp/images_sbs_ascii_rgb

.venv/bin/python main.py --mode asciiweb --bake-dir tmp/images_sbs_ascii_rgb
```

Do not run the playback command alongside a viewer using the same ports. The
existing public test workers were **not restarted or switched** by this work.
The validation package in `tmp/ascii_rgb_validation/bake` is a **120-position,
six-folder sample**, not a full replacement for the 2,221-frame source tree.
The full-tree command above has not been run as part of this delivery.

## What is baked

This is the `/ascii` **RGB ASCII-web** pipeline, not the JPEG `--mode web`
pipeline. The default is the existing 90x60 shared master, font ratio .5.
The browser still receives identical RGB ANSI/gzip pictures; its font, local
zoom, controls, network workload and rendering work are unchanged.

The baker decodes and samples each face/float image **separately** using the
live renderer's bilinear texture sampling, initial source aspect/letterboxing,
SBS colour/matte interpretation, and `<.05` SBS alpha cutoff. It stores RGBA
floating point samples before blending or final 8-bit rounding. Nothing is
quantized to a palette, no extra blur is applied, and no face/float pair is
pre-mixed. Each source frame and folder remains independently addressable.

Playback retains the existing exact-clock deadline scheduler. It reads only
the selected independent grids and uploads small nearest-sampled float textures.
The same GL alpha blend produces the final master; grading, glyph selection and
RGB encoding then run on that small grid. Grading must follow composition:
baking final glyphs separately and pasting non-space cells would change partially
transparent layers. This implementation removes large source decoding/uploads
and source-resolution sampling, **not all GL or all live conversion work**.
Live picture grading and browser adjustments remain single-stage as before.

Options: `--jpeg-layout sbs|rgb`, `--size WxH`, `--font-ratio N`, `--mirror`,
`--compression 0..9` (default 6). Normal playback currently requires the existing
90x60 master and its configured font ratio/mirror. A changed sampling geometry
requires a new bake. SBS is the default for JPEGs; ordinary JPEGs need `rgb`.
PNG/WebP retain their real alpha. No physical devices or audio are opened.

## Storage and memory

- A unified `bake.json` routes **only asciiweb** to `rgb/`; old unified packages
  and the raw-pixel baker are unchanged.
- `rgb_layers.json` records dimensions, precision profile, independent folders,
  source size, codec and measured byte counts. Per-folder `sources.jsonl` retains
  ordered original filenames, source byte lengths and SHA-256 hashes.
- One `samples.bin` contains individually zlib-compressed frames. Byte shuffling
  improves compression without changing values. Float16 is used **only if every
  sampled float32 value round-trips exactly**; otherwise the block retains float32.
- Repeated encoded images avoid duplicate decoding/sampling. Identical sampled
  grids share blocks across folders without sharing their selection identity.
- Each folder's `index.npy` is memory-mapped: offset, length, scalar size, CRC32.
  Index bounds and paths are checked on open; decoding checks checksum, stream
  completion, exact decompressed size and finite/ranged sample values.
- Only the requested compressed block is mapped and immediately unmapped after
  bounded decompression. Compressed bytes cannot be mapped as decoded pixels.
- Immutable decompressed grids use a **4 MiB byte-limited LRU**, not a sequence
  cache. The deadline preparer/current textures can also hold their bounded
  in-flight grids. This is **not a 4 MiB whole-process RAM limit**. Index pages
  scale with frame/folder count; full images never enter the playback cache.
- Builds stream source images, stage beside the destination and publish only
  on success. Existing outputs are refused; source files are never overwritten.
  No generated package belongs in git.

GL sampling is a renderer-defined operation. Playback rejects a different
GLES-mediump/desktop precision profile instead of silently changing rounding.
Exact parity below was measured on the workspace's desktop-GL/llvmpipe backend; it
does not prove equality to every other GPU driver's live sampling.

## Validation (2026-10-18)

Focused regression: **65 tests passed**, including the new bake/storage tests,
existing independent-layer packages, ASCII conversion, web broadcast/zoom,
deadline preparation and lazy imports. Tests cover corruption, invalid paths,
truncation, cache limits/disabled caching, close ownership, interrupted builds,
overwrite refusal, source-free package routing and independent layer selection.
Synthetic RGBA/SBS checks include nonblack backgrounds, transparency cutoff and
letterboxing; final rendered RGB and ASCII match exactly.

Actual source sample: 120 positions (0,2,...,238), three face folders (normal,
sharp, another style) and three floats (transparent and two nonempty layers).
**36 cross-paired/nonsequential composite checks matched RGB framebuffer bytes,
final RGB ANSI text and gzip bytes exactly.** Same images, colours, glyphs and
browser payload; this is not a lower-quality bandwidth experiment.

Sample storage: **76,429,023 source bytes -> 22,037,590 package payload bytes**
(about **71% smaller**; tiny top-level manifest additional), 601 unique sample
blocks for 720 layer frames. Do not extrapolate this ratio as a full-tree result.

Two reversed-order, fresh-process real-display-loop observations used the same
120-position/six-folder source lists, two-second moving warmup and six-second
measurement, no listeners/viewers. Both paths used the actual scheduler; raw
source lists were supplied by the private fixture instead of rescanning disk.
CPU convention: 100%=one core; RAM is warmed median process RSS.

| Path | CPU, two runs | RSS, two runs | Whole-run publications/s |
|---|---|---|---|
| Live source | 19.5%, 21.8% | 292.6, 292.6 MiB | 14.14, 15.01 |
| Compressed RGB bake | 7.5%, 8.3% | 256.7, 256.9 MiB | 14.39, 14.39 |

On this cohort the bake reduced average CPU about **62%**, RSS about **12%**.
Rates vary with missed deadlines; no clock drift/queued replay was introduced.
Only compare these matched fresh-process runs with one another, not with the
separate public worker or original-browser observations.

Repeated-context tests exposed an existing diagnostic-history bug: the GL
logger merged disk history back into session state and doubled it at subsequent
context creations. Two regression attempts were OOM-killed before this was
isolated. Offline baking now disables app-history logging; the logger keeps
session state separate and skips parsing histories over 1 MiB, preserving such
files untouched. A regression test covers both safeguards. Existing home-directory
history was not deleted or rewritten by the fix. All numbers in this table were
rerun after the fix; earlier inflated-memory observations are retained only as
`before_log_guard_*` artifacts. Benchmarks were sequential, not concurrent.

A separate matched unpaced 120-picture server-pipeline benchmark, reversed
order twice, included source reads, upload, blend/readback, grading/glyph/ANSI
encoding and gzip. Process CPU fell from **16.74 to 6.15 ms/frame** (~63%).
The major removed costs were decoder loading (~5.70 -> 1.15 ms) and upload
(~6.06 -> .04 ms); tiny blend/readback, conversion and gzip remain. This is not
a network benchmark or an original-browser resource comparison.

Artifacts/scripts: `tmp/ascii_rgb_validation/`, `tmp/ascii_rgb_benchmark.py`,
`tmp/ascii_rgb_loop_check.py`, `tmp/ascii_rgb_regression.log`.

**This reduces server production work, not the browser's measured 119% CPU
regression.** Browser font/style/layout/paint optimisation remains separate.
