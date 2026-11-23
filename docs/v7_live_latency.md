# V7 live EOF-readiness latency check

Date: 2026-09-29. This records a short software-paced comparison of live-input
readiness; it is not an audio-device or GUI render-to-photon measurement.

## Method

Encode 60 EOF-marked V7 packets, then feed the same signal in 1,024-sample
blocks at the 48 kHz reference rate. Run once with `LiveInput` using legacy
header readiness and once with EOF-marker readiness. Both runs use the EOF
decoder, tone-seeded timing, decode the same image path, convert decoded values
with `values_from`, and publish to a `LatestFrame` mailbox. Timing is measured
from each packet's expected EOF sample to mailbox publication. The final
partial 1,024-sample block is omitted, leaving the last packet incomplete in
both runs. Discard packet indices 0–9 from the latency summary.

The stream lasts about 4.9 seconds per mode (9.8 seconds total). There are 49
steady-state observations per mode. Input scheduling lateness remained below
3 ms p95 in both runs.

## Results

| Readiness | EOF-to-mailbox median | EOF-to-mailbox p95 |
|---|---:|---:|
| Following header | 23.66 ms | 32.56 ms |
| Validated EOF marker | 13.57 ms | 22.47 ms |
| **Reduction** | **10.09 ms** | **10.09 ms** |

The reduction is a per-frame latency offset; it does not accumulate into a
shorter total playback duration as the stream runs. This test excludes actual
audio callback/device behavior and the GL viewer's poll, render, VSync, and
display scanout. The raw measurements are retained locally under
`tmp/v7-eof-10s-publication-20260929.json`; `tmp/` artifacts are not committed.
