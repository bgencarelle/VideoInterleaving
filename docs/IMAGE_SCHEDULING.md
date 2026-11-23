# Image scheduling and frame accuracy

Frame accuracy means selecting the source index and folder pair appropriate
to presentation time. It is separate from image fidelity and output frame rate.

**Current policy (2026-10-15):** the deadline-fed preparation implementation at
the end of this document supersedes both the broad lookahead/nearest-ready
design and the insufficient immediate-current/next handoff. Presentation uses
exact clock identities; preparation lead includes measured queue/load cost.

## Local, web, ASCII and ASCII-web

These modes share `image_display.run_display`. The display requests the current
clock index and gives exact buffered images priority, without the old rolling
index correction steering selection toward previously late pictures.

Loads are keyed by index **and folder pair**. Completed frames from an old
folder selection cannot satisfy a new selection. The bounded source FIFO searches
all candidates regardless of completion order; an old first entry cannot hide
a later exact hit. One-way loops use modular distance, while ping-pong uses
ordinary index distance. Nearby-frame fallback retains the configured tolerance.

For the free clock, preparation requests sample the next three presentation
ticks using the shared clock function, so prediction follows IPS, loop wrap and
ping-pong turns. MIDI input remains authoritative: no future MIDI event is
invented. Its current index and adjacent direction-based requests use the same
folder-aware loading path. In-flight requests are deduplicated and bounded.

A held clock is checked again while its exact image is missing, allowing a late
load to replace the fallback without waiting for another clock change. Web
capture identity includes both requested and displayed index, allowing such a
correction to be published during a held clock.

## Application modem

The live modem predicts presentation at packet completion plus receiver
allowance. `PicturesAhead.wait_exact` primes a requested job explicitly;
nearest-ready playback fallback cannot accidentally satisfy startup priming.
The prepared-job cache is bounded to 16 entries. Upcoming free-clock requests
also cover adjacent source ticks, so a small reservation shift across an index
boundary does not turn an almost-correct prediction into a missing job.

After opening the audio device, startup reserves a slot with at least 500 ms
preparation lead. Its actual presentation timestamp and device packet interval
drive exact startup priming. The first packet uses that same reservation;
subsequent packets return to the normal adaptive preparation lead. If startup
work exceeds the reservation, normal deadline rejection/recalculation applies.

During playback an unfinished exact job may consume up to 10 ms of spare
reservation lead, retaining 10 ms for packet construction/submission. Otherwise
nearest-ready fallback continues, with the actual source index in metadata.
The per-packet log now includes `exact_picture` alongside `wanted_index` and
`source_index`.

Offline WAV export already prepares each exact sample-time-selected picture
synchronously. It shares the corrected source FIFO but has no real-time
fallback or audio-slot priming. MIDI retains event-driven index selection.

## Other schedulers

Scope uses `ScopeFrameScheduler`, not the display FIFO: it replaces a single
desired request, waits for audio consumption, and pins multi-field pictures
through `FieldGroupLatch`. Standalone V7 uses its capture mailbox/generation
and source preparation cache rather than the application image scheduler.
Those paths retain their existing scheduling semantics; the display-clock
prediction above must not be applied to their independent timelines.

## Validation

`tests.test_image_scheduling_accuracy` covers completion order, folder identity,
loop-wrap selection and exact startup priming in the presence of ready neighbors.
Focused clock, import, ASCII and scope-scheduler tests accompany it.

An initial 96-packet `images_sbs` live run on the existing PortAudio `pulse` ->
PulseAudio `loop` software route improved from **83/96 exact requested indices**
to **96/96**, with maximum absolute index error falling from 19 to zero. These
are separate live runs on a wall-clock source, not identical-content trials or
physical/tape validation. Logs: `tmp/v7_main_images_live.log` and
`tmp/v7_main_images_scheduling_after.log`.

Five-second actual-renderer runs under Xvfb/software GL, using `images_sbs`
and the free clock, compared the unchanged HEAD scheduler with this version:

| Mode | Exact requested/displayed updates before | After | Maximum absolute index error before/after |
|---|---:|---:|---:|
| Local | 300/301 | 300/301 | 1 / 1 |
| Web | 0/76 | 75/77 | 22 / 2 |

These measure application render-loop selections, not physical display scanout
or browser arrival time. Source clock phases differ between runs. Harness and
per-update records: `tmp/image_schedule_run.py` and
`tmp/image_schedule_{local,web}_{before,after}.json`.

A further isolated modem run was also 96/96 exact
(`tmp/v7_main_images_scheduling_isolated.log`). Running a test suite and offline
export concurrently with another modem run reduced it to 65/96 exact, with a
maximum absolute error of 53 (`tmp/v7_main_images_scheduling_final.log`). That
overloaded run motivated adjacent-tick coverage; exact presentation remains
dependent on preparation completing before audio deadlines, not a guarantee
under arbitrary host load.
The final adjacent-tick implementation also achieved 96/96 exact in its
isolated run (`tmp/v7_main_images_scheduling_neighbors.log`).

Focused final validation passed 59 tests; log:
`tmp/image_scheduling_final_tests.log`. The 48-packet offline PCM payload remains
byte-identical to the pre-change export.

## CPU/RAM follow-up and additional mode checks

The accuracy changes have a measurable resource cost. Two trials per version
ran sequentially with `images_sbs`, fixed folder pair `(0, 0)`, Xvfb/software
GL, two seconds of renderer warmup and six seconds of measurement. Baselines
load HEAD's `image_display.py` and `image_loader.py`, keeping other current
code/dependencies constant. The table averages the two trials. CPU is process
CPU time divided by wall time: **100% means one logical core**, not the whole
machine. RAM is median process RSS, including GL/library/allocator memory.

| Mode | CPU before | CPU after | RSS before | RSS after |
|---|---:|---:|---:|---:|
| Local | 66.2% | 70.5% | 384 MiB | 426 MiB |
| Web | 16.8% | 23.4% | 281 MiB | 308 MiB |
| ASCII | 26.0% | 35.8% | 333 MiB | 447 MiB |
| ASCII-web | 26.6% | 37.6% | 338 MiB | 443 MiB |

Local loaded about 30 source pairs/s in both versions. Web/ASCII/ASCII-web
loaded about 15/s before and 30/s after: alongside output-time predictions,
the current implementation also requests the adjacent source index. At the
15-fps output rate that extra index is often not displayed. This is a concrete
next opportunity to reduce speculative work without sacrificing predicted
presentation indices.

The raw-layer FIFO capacity remains 15, but the corrected full-candidate search
retains more entries than the old front-pruning lookup did. Bounded capacity
therefore does not mean unchanged RSS. Extra active decodes and allocator
retention also contribute; these short runs establish a higher working set,
not a long-duration memory-leak result.

Across the warmed trials, web/ASCII/ASCII-web selection improved from one
source-index step behind to exact on every sampled update. Local was exact
in both. Renderer resource measurements exclude network clients and client-side
terminal/browser costs. Harness/artifacts: `tmp/image_schedule_resources.py`,
`tmp/image_resources_{mode}_{before,after}_{1,2}.{json,log}`.

Actual `main.py --mode ascii` Telnet and `--mode asciiweb` WebSocket clients
each received 62 frames over approximately 4.1 seconds. The WebSocket check
also verifies the HTTP upgrade and accept key. Inspection found an unrelated
undefined `now` in `ascii_web_server.broadcast_loop`, after its first broadcast;
the unused assignment is removed. `tests.test_ascii_web_broadcast` exercises
multiple frames and the existing bounded slow-client backlog policy. Delivery
artifacts: `tmp/ascii_delivery_check.{py,json}` and `tmp/{ascii,asciiweb}_delivery.log`.

The application modem was measured separately on the refreshed existing
PortAudio `pulse` -> PulseAudio `loop` software route (44.1-kHz sink/monitor).
Two sequential 120-packet live sessions compared HEAD's scheduler/loader with
the current version, skipping the first 12 packets:

| Modem scheduler | CPU | Median RSS | Exact requested indices |
|---|---:|---:|---:|
| Before | 170.3% | 318.0 MiB | 108/108 |
| After | 259.9% | 444.8 MiB | 108/108 |

This is a substantial cost: about 0.90 extra logical cores and 127 MiB in this
sample, with no accuracy improvement needed during these particular warmed
windows. The extra adjacent-tick preparation and larger prepared cache favor
accuracy over work avoided. CPU includes background preparation workers and
their numerical libraries. Only one run per modem version was measured; wall
clock phases/content and scheduling noise differ, so this is a workload sample,
not a universal multiplier. Harness/artifacts:
`tmp/modem_schedule_resources.{py,json}` and its before/after logs.

### Scope web

Scope's separate scheduler and source cache were not changed by this work.
The current raster runtime-image path was checked with actual HTTP clients and
`--device null --no-scope-gui`; this is **NullStream/software-renderer evidence**,
not PortAudio, physical output, or browser execution.

| Server phase | CPU | Median RSS |
|---|---:|---:|
| No viewer | 113.8% | 413.7 MiB |
| One 512-pixel trace MJPEG viewer | 127.1% | 415.1 MiB |
| One luminance-stream viewer | 116.6% | 415.3 MiB |
| After viewers disconnected | 117.4% | 415.5 MiB |

The trace client received about 83 JPEGs in six seconds; the luminance client
about 182. The no-viewer phase was measured with periodic `/data` requests.
These are sequential live phases with changing source content, so small CPU
differences include workload/scheduling noise. Scope's one-second monitor
snapshots reported absolute index differences of 0–2 steps (up to about 67 ms
at 30 IPS), with zero reported DAC dropouts or pre-adoption presentation drops
at the sampled phase endpoints. These are not per-frame browser-arrival
measurements and should not be compared directly with renderer update counts.

The page, still JPEG, trace MJPEG and luminance MJPEG endpoints responded;
the saved 512-pixel still was visually inspected. Desktop browser tooling
reported `browser.disconnected`, so browser JS CPU/RAM and end-to-end browser
presentation timing remain unmeasured. Artifacts:
`tmp/scope_web_resources.{py,json,jpg}`, `tmp/scope_web_resources.log`.

The existing Node/Python web-renderer comparison passed its extent, aspect,
trigger and lit-pixel checks (`tmp/scope_web_resource_parity.log`). Focused
ASCII broadcasting, display selection, conversion/scaling and scope scheduling/
preview regression checks passed 26 tests (`tmp/ascii_scope_resource_checks.log`).

## Low-working-set follow-up — exact index is authoritative

The user clarified that the image selected by the clock at a presentation
moment, sampled at the configured FPS, is vital. A nearby available image must
not be treated as synchronized. The earlier speculative approach is superseded:

- Shared display modes keep at most two FIFO entries and use one decode worker.
  Each tick requests the current exact index and the index for the next actual
  presentation deadline. No intervening adjacent source indices are loaded.
  Obsolete decoded entries are released; startup inspection no longer pins a
  complete source pair for the session.
- Exact-only FIFO selection replaces nearest-picture display fallback. A late
  picture leaves the last visible picture in place, increments the existing
  miss counters, and is not published as a new synchronized web/ASCII frame.
  MIDI remains event-driven; future MIDI events are not guessed.
- Local GL drawing/swapping occurs when an image or view changes, rather than
  recompositing the same unchanged image on every polling tick. Input polling,
  view dirtiness and the configured loop-FPS pacing continue.
- JPEG loading retains at most two content/scale-keyed decoded assets. Identical
  file bytes under different source filenames reuse immutable decoded pixels.
  Different content or decode scale does not reuse them. This is particularly
  useful for repeated transparent layers and does not conflate source indices.
- Application modem preparation uses one worker and two prepared slots, exact
  current/next packet timestamp requests, and no separate live raw-layer
  prefetch. The live handoff waits for its exact job; an expired reservation is
  rejected/recomputed, never filled with a nearby picture. Existing audio
  deadline/underflow/starvation diagnostics remain authoritative.
- The application modem now uses the same reversible BLAS worker budget as
  standalone V7. Its full-resolution compositor fuses the two opaque-background
  alpha-over operations in a Numba kernel, preserving Pillow's integer rounding
  without allocating full intermediate RGBA layers. Normal and read-only source
  signatures are warmed before opening audio. Raw layer/folder buffering is
  limited to one active folder buffer and one entry.
- Runtime scope thumbnails reuse at most two identical JPEG-content assets and
  avoid an unnecessary full-size RGB-to-RGB Pillow copy. Its scheduler, defaults,
  trace geometry, output cadence and field-latching semantics are unchanged.

Full-resolution source pixels, layer interpretation and compositing output are
preserved. No lower-resolution decode, lower output FPS or reduced image quality
is introduced to achieve these results.

### Measured resources versus original baseline

Display measurements reuse the preceding harness/settings: `images_sbs`, fixed
pair `(0, 0)`, two warm seconds, six measurement seconds, two trials per mode,
Xvfb/software GL. Baselines are the previously recorded HEAD renderer/loader
runs. CPU 100% is one logical core; RAM is median process RSS.

| Mode | Original CPU | Current CPU | Original RSS | Current RSS |
|---|---:|---:|---:|---:|
| Local | 66.2% | 46.0% | 384.3 MiB | 287.4 MiB |
| Web | 16.8% | 14.9% | 280.7 MiB | 266.4 MiB |
| ASCII | 26.0% | 23.9% | 333.2 MiB | 286.2 MiB |
| ASCII-web | 26.6% | 23.6% | 337.8 MiB | 286.1 MiB |
| Application modem | 170.3% | 25.9% | 318.0 MiB | 297.5 MiB |

Modem uses the earlier 120-packet live `pulse`/`loop` baseline and one new run,
with the first 12 packets excluded from CPU/RSS measurement. All 120 packets
completed with zero deadline misses, underflows or producer starvations; all
108 measured packets carried exactly their requested index. Numerical-library
worker budgeting contributes heavily to the CPU reduction. This route was
refreshed as stereo 44.1 kHz, not physical/tape output.

All warmed display observations selected the exact requested index:
local 724/724, web 182/182, ASCII 183/183 and ASCII-web 184/184. Real main.py
Telnet/WebSocket clients received 61/60 frames respectively over about four
seconds after the final changes. These observations do not measure physical
scanout or browser network-to-screen latency.

Scope was compared separately with HEAD/current `scope_image_source.py` in the
same current application, fixed pair `(0, 0)`, launch-relative clock, runtime
images, raster output and NullStream:

| Scope phase | Original CPU | Current CPU | Original RSS | Current RSS |
|---|---:|---:|---:|---:|
| No viewer | 105.3% | 60.9% | 411.7 MiB | 398.4 MiB |
| One 512-pixel trace viewer | 122.8% | 76.4% | 413.2 MiB | 400.8 MiB |
| One luminance-stream viewer | 106.5% | 60.0% | 413.3 MiB | 401.0 MiB |
| After viewers | 108.9% | 61.0% | 414.2 MiB | 401.1 MiB |

Both scope versions reported zero sampled DAC dropouts/presentation drops.
Its sampled absolute clock/adopted-image differences remained 0–2 source steps;
this optimization does not remove the existing trace/output/preview latency.
Browser tooling remains disconnected; these are server/HTTP and NullStream
measurements. Baked scope sources do not use the optimized runtime decoder.

These are representative workload measurements across the six main.py modes,
not universal CPU/RSS bounds for arbitrary source sizes or settings. Repeated
identical assets benefit most from lossless content reuse.

### Correctness and compiled-path evidence

- **102 focused regression tests passed**, including exact FIFO selection,
  release of obsolete entries, content/scale invalidation, read-only ownership,
  scope source handling, imports, ASCII and modem/DCT/application integration.
- Fused compositing matched Pillow exactly on multiple random RGB/RGBA/SBS
  pictures, cropped overlays, rotations and mirroring. An isolated benchmark
  on three real pictures reduced compositor CPU from 29.1–30.9 ms to
  7.2–7.9 ms per picture, with identical pixels.
- Default and adaptive-stereo-nested application exports each emitted
  **bit-identical PCM**. Their live receiver loops decoded all 48 pictures
  identically with smoothing 0 and 2 (192 picture comparisons in total).

Artifacts: `tmp/image_resources_{mode}_bounded_{1,2}.{json,log}`,
`tmp/modem_bounded_resources.json`, `tmp/scope_bounded_{before,bounded}.json`,
`tmp/bounded_compositor_bench.{py,json}`, `tmp/bounded_modem_parity.{py,json}`,
`tmp/bounded_main_regression.log`, and `tmp/ascii_bounded_delivery_runner.log`.

## Injected slow-storage / single-core contention check — 2026-10-15

User-directed stress check of actual `main.main()` in all six modes, using
`images_sbs`, fixed folders `(0, 0)`, launch-relative free clock, normal FPS
and full-resolution layers. Each subprocess had three seconds of baseline,
six seconds of injected load, and seven seconds of recovery. Conditions:

- **Slow storage:** add 40 ms to each image-read entry, plus 250 ms every
  tenth read (290 ms total for that read). Both layers incur the delay.
  Scope injects at its thumbnail decode entry, before content-cache lookup.
  These are controlled latency injections, not actual filesystem throttling.
- **Embedded-like:** pin the application to one allowed logical CPU. During
  the fault window, run three busy competitor processes on that same CPU and
  add 8 ms per image-read/decode entry. The single-core affinity remains during
  recovery; the competitors and read delays stop. This simulates constrained
  CPU scheduling, not any particular embedded processor's instruction speed.

Runs were sequential. Local uses Xvfb/software GL; web/ASCII observe server
publication, without network clients. Scope uses raster/runtime images and
NullStream, observing output adoption. Modem uses the refreshed existing
`pulse -> loop` virtual route, observing accepted/rejected timestamped packets.
No physical output, browser scanout or receiver network queue is measured.

### Results

| Mode | Slow-storage recovery | Single-core/contention recovery | Main observed risk |
|---|---:|---:|---|
| Local | 94 ms | 11 ms | Approximately 6.08 s without a new picture under slow reads; fewer presentations under CPU contention |
| Web | 156 ms | 14 ms | Approximately 6.09 s without publication under slow reads |
| ASCII | 132 ms | 16 ms | Approximately 6.07 s without publication under slow reads |
| ASCII-web | 128 ms | 14 ms | Approximately 6.07 s without publication under slow reads |
| Scope | 126 ms | 183 ms | Approximately 6.02/6.07 s without new adoption under storage/CPU faults |
| Modem | 28 ms | 36 ms | Missed packet slots rather than late-index transmission |

Recovery is the first observed presentation after removing the fault within
baseline p95 index error plus one source step. For modem it is the first accepted
packet with the exact reserved target index, measured at producer submission,
not receiver display. Scope adoption was sampled every ~50 ms, so its recovery
timing has that sampling granularity. Tiny recovery times do not describe the
duration of the preceding freeze.

**No persistent post-fault drift was observed.** Final three-second p95 visible
errors were one source step for local, one/two for web, two for both ASCII modes,
and two for scope. They returned to their ordinary FPS/adoption-age envelope,
without replaying obsolete source indices. During the freezes, held pictures
became approximately 168–170 steps behind at p95 (about 5.6 seconds at 30 IPS).
This is a real synchronization loss while the fault persists, even though it
does not become cumulative scheduling drift afterward.

All shared-display publications/swaps used the index requested by that
iteration: zero mislabeled/substituted pictures. However, selection and actual
completion are different moments. Under contention the largest fresh-picture
error against the clock at the swap/publication hook was five source steps
for local, four for web, and one for each ASCII mode. Thus **exact selection
alone does not guarantee an exact index at render completion** when the render
thread is descheduled. These measurements do not include downstream client
buffering or physical scanout. Scope's sampled new-adoption error during the
fault was at most two steps, but its held picture aged during the long gap.

Modem accepted 99 packets in the slow-read run and 120 in the contention run;
every accepted packet's picture matched its reserved presentation index.
Diagnostics reported six/two deadline misses respectively, zero output
underflows and zero producer starvations. Reserved silence/missed output is
still a delivery interruption despite those zero underflow counters. Scope
reported zero sampled DAC dropouts but hundreds of producer deadline misses;
zero DAC dropouts does not establish fresh image delivery.

The risk found is deadline starvation/frozen pictures and transient render-age
error, rather than evidence of an accumulating independent playback clock.
With permanently slower-than-deadline storage, strict current/next selection
can keep missing indefinitely; this test establishes recovery once the fault
ends, not acceptable continuous delivery on every low-end machine. Existing
web/client queues, browser presentation and a live modem receiver were not
stress-tested by this producer-side harness.

Harness/results: `tmp/scheduling_stress_worker.py`,
`tmp/analyze_scheduling_stress.py`, `tmp/stress_summary.json`, and
`tmp/stress_{local,web,ascii,asciiweb,scope,modem}_{disk,embedded}.{json,log}`.
An initial scope instrumentation wrapper failed the constructor-signature
compatibility check before output; adding `functools.wraps` fixed the harness,
and both scope cases were rerun successfully. Application code was unchanged.

## Trace and actual `main` comparison — 2026-10-15

Follow-up requested by the user: trace the failure and compare against the
actual `main` branch, not just HEAD versions of selected modules. A complete
`git archive main` snapshot at **9591126d5** ran in
`tmp/main_stress_checkout`, with the existing image inputs symlinked in.
No existing checkout was switched or modified. Current means the uncommitted
working tree. `tmp/trace_revision_provenance.json` records commit IDs and source
hashes. Fault timing and conditions match the preceding check; one sequential
run per revision/mode/condition, with no network consumers.

### Display comparison

Slow-read results during the six-second fault window:

| Mode | Main fresh picture changes | Current changes | Main longest fresh-picture gap | Current gap | Main/current p95 visible error, source steps |
|---|---:|---:|---:|---:|---:|
| Local | 118 | 1 | 119 ms | 6.09 s | 4 / 170 |
| Web | 76 | 1 | 160 ms | 6.09 s | 6 / 169 |
| ASCII | 76 | 1 | 153 ms | 6.07 s | 6 / 169 |
| ASCII-web | 72 | 1 | 149 ms | 6.07 s | 5 / 169 |

The gap column spans the entire run, counting changes of actual picture index,
not repeated swaps/publications of a held picture. Main keeps updating with
nearby/late pictures; current mostly holds one picture. **This is a sustained
slow-storage continuity regression in the recent exact/two-slot preparation
path**, not evidence that main was continuously exact. Nearly every main
fault-window presentation selected an index different from that iteration's
clock request. Its tolerant lookup and rolling correction accept that lag.
Neither revision showed persistent accumulating drift after the fault ended.

Under one-CPU contention, p95 visible errors main/current were local 5/10,
web 8/4, ASCII 7/3, ASCII-web 7/3 source steps. Fresh changes were local
119/32, web 61/80, ASCII 64/89 and ASCII-web 66/89. Current improves the sampled
headless-mode lag but worsens local continuity. Main/current final recovery
returned to ordinary presentation-age envelopes; first recovery observations
were 13–85 ms on main and 9–147 ms on current across these runs. This is a
short synthetic comparison, not a hardware performance guarantee.

### Traced mechanism

The instrumentation records read duration, job start/completion index, ideal
clock index at completion, FIFO queries, cancellation, swap/publication and
independent visible-picture samples.

- Main has up to eight workers, adjacent-index preparation and tolerant FIFO
  selection with a rolling correction. It completed 176–177 picture jobs
  during the storage fault. Current has one worker, two pending jobs and exact
  current/next selection; it completed 44, with median preparation 83–85 ms.
  The two sequential 40 ms layer reads already exceed both a 33 ms local tick
  and a 67 ms headless presentation interval, before decode/render cost.
- Current completed pictures were typically **2–3 source indices old**, with
  p95 8–10 indices old during long-read spikes. Those images fail exact lookup
  or are removed by the current/next-only retain set. A small cache does not
  create useful lookahead if all jobs target deadlines inside their own load
  duration.
- There is also a concrete priority inversion in
  `image_display.py::request_picture`: after requesting the current index,
  requesting the next tick can cancel that queued current job. Storage-fault
  traces recorded **134** such successful cancellations in local and **44**
  in each headless mode. Example: at t=3.150 s the current clock index is 146,
  job 145 is running, and requesting future index 147 cancels queued job 146.
  Job 145 completes at t=3.186 s when the clock is already 148. No clock drift
  is required to explain the starvation.
- Under local CPU contention, current jobs often complete only 0–1 source
  steps behind, yet presentations still miss: being ready between polling
  ticks does not ensure a match at the next tick, and software GL/thread
  scheduling adds presentation age. Read throughput alone is not sufficient.

These findings support correcting deadline selection and request priority
inside the existing pipeline. They do **not** justify replacing the scheduler,
restoring nearest-image presentation, or claiming that all recent resource
savings are free of continuity tradeoffs. No production fix was made during
this diagnostic pass.

### Scope/modem comparability

Main has no runtime scope thumbnail adapter or application V7 modem. Its scope
is baked-source-only and modem uses the older baked V3 pipeline. Therefore a
main runtime-image result for those modes would be fabricated. Source inspection
also confirms main's modem already measures encode duration, adjusts
`prepare_ms` to 1.5 times that duration, and rejects expired reservations:
adaptive preparation is already present in that pipeline.

Trace harnesses and results: `tmp/run_main_trace_comparison.py`,
`tmp/scheduling_stress_worker.py`, `tmp/analyze_main_trace.py`,
`tmp/trace_comparison_summary.json` and `tmp/trace_{main,current}_*.{json,log}`.

### Scope and modem trace results

Current runtime scope completed 80/105 decoded pairs during the slow-storage /
CPU-contention fault windows. Median pair work took 97.8/99.2 ms; completed
pairs were typically four source indices behind, with p95 age 12/7 steps.
Example: a pair for index 210 completed at t=4.095 s when the clock was 214;
the next job started already one index late after 82.6 ms of queue wait.
Only **2/178** and **2/171** exact-pair lookups succeeded. This explains why
output held despite the scheduler thread continuing to execute callbacks.

`ScopeFrameScheduler._record_success` counts a render-callback return, not
necessarily an emitted trace: `emit_scheduled_request` returns immediately on
`ready_pair(...) is None`. The traced 178/171 callbacks therefore must not be
reported as 178/171 fresh pictures. Likewise its deadline-miss count includes
any positive scheduling lateness, not just lost output. Adoption and pair-ready
counters are the appropriate freshness evidence. First observed recovery was
226/174 ms in these traced repeats, with final p95 adopted-image age two steps.

For a genuinely common scope path, both full revisions also ran the **same
2221-frame `images_xy` bake**, raster/NullStream, one-CPU affinity and three
competitors during seconds 3–9. Main/current completed **178/177** producer
trace calls in that six-second interval, with p95 completion age **one source
step on both**. P95 producer call duration was 4.34/0.78 ms. Neither had the
runtime-image freeze. These are producer-completion observations, not a
cross-version DAC-adoption measurement; main lacks current adoption identity
telemetry. No synthetic JPEG delay applies to this already-baked path.

Current modem preparation median duration rose from about 19 ms normally to
99.6/88.5 ms under storage/CPU faults. Its target timestamps allow preparation
before presentation, so unlike the display path many completed pictures were
still ahead of their target. It accepted 14/34 packets during the fault windows,
all matching their reserved exact source index. End-of-run output diagnostics
reported **7/2 deadline misses**, zero output underflows and zero producer
starvations. This includes callback deadline rejection as well as producer
submission rejection; accepted submission is not proof of receiver display.
The first post-fault exact accepted packets arrived after 329/26 ms in these
repeats. Timing varies between runs; the earlier 28/36 ms figures were not a
worst-case bound. A V3-main versus V7-current receiver/display comparison was
not performed, and runtime-image modem has no counterpart on main.

Additional artifacts: `tmp/run_auxiliary_traces.py`,
`tmp/scope_baked_trace_worker.py`, `tmp/baked_scope_trace_{main,current}.json`,
`tmp/baked_scope_trace_summary.json` and current scope/modem trace JSON/logs.
All 22 subprocess cases completed successfully with no recorded harness errors.

## Deadline-fed preparation implementation — 2026-10-15

User approved the proposed correction: feed the existing scheduler with the
known future source indices far enough ahead to cover preparation time, while
preserving miss/hold rather than backlog/drift behavior.

### Implementation

- `image_preparation.py::DeadlinePreparer` is a bounded data-preparation helper,
  not a playback clock. It maps future presentation ticks through the caller's
  existing clock function, deduplicates `(index, folder pair)` identities, and
  retains useful jobs/results until their assigned deadlines.
- Two decode workers overlap storage waits. Lead uses a rolling p90 of measured
  submission-to-completion time (including queue wait), a 25% margin, and measured
  display work. Expired running work **also trains lead**; otherwise a slowdown
  could expire every job before it contributed a timing sample.
- Outstanding work is limited to the worker count, separately from completed
  future-picture storage. A larger ready-picture allowance must not create a
  larger decode queue that inflates measured preparation time. Busy workers
  cause a new future reservation to be skipped, not queued behind stale work.
- Only expired or invalidated queued work is cancelled. Running obsolete jobs
  continue to count against capacity until finished. There is no current-job
  cancellation to make room for the following tick. Source failures have bounded
  retry/backoff and existing error reporting; they never produce black fallback.
- Local permits up to eight preparation identities and headless display modes
  four. These are upper limits, not eagerly filled sequence caches: the moving
  preparation window normally retains only a few identities. Capacity pressure
  skips new reservations rather than extending an unbounded queue or shifting
  presentation time. Ordinary full-resolution layers and output FPS remain.
- `image_display.py` consumes only the exact current index/folder pair. A startup
  seed preserves held-clock/prebaked-ASCII publication. Real folder changes
  invalidate old preparation rather than guessing the stateful folder selector.
  MIDI prepares its observed held identity, without predicting future events.
- `renderer.TexturePairStage` uses two reusable GL texture-pair slots, active
  plus upcoming, on the owning GL thread. Uploading the upcoming pair cannot
  overwrite active textures. Asset identity avoids unchanged-layer uploads.
  Ordinary local output remains GPU composited, without extra CPU composites.
- Local polling retains FPS pacing and skips missed timer ticks. If a blocking
  swap has already paced the iteration, the timer does not add a second refresh
  wait; the next iteration samples the current clock, not a backlog. Tiny timer
  overruns retain a one-millisecond tolerance. Unchanged pictures/views do not
  redraw, but exposure requests one redraw without reinitializing the window or
  reuploading textures. Miss logs are limited to one per second; counters retain
  misses. The normal headless ticker behavior is retained.
- Web/ASCII also avoid redrawing a held FBO when no new output is available.
  The dedicated local-frame bridge retains its repeated-publication path for
  held-picture/source metadata. Headless capture identity includes folders.
- Runtime scope uses the same preparation helper only for thumbnail pairs, with
  a maximum of twelve small pairs. Its existing consumption-driven trace worker,
  field latches and output adoption rules are retained. Atomic pair decode and
  request/start/ready timestamps remain available; source errors remain visible.
  One exact last-used pair is pinned separately until the worker requests a new
  pair, so remaining interlaced fields can finish after UI-clock preparation has
  moved on. This is exact identity retention, not nearby-index fallback. Baked
  scope sources bypass this adapter. Legacy source activity fields now reflect
  deadline work, alongside detailed `deadline_preparation` diagnostics.
- Modem already reserves future packet timestamps and adapts encoding lead.
  Its production path needed no additional change in this implementation pass;
  exact timestamp rejection was retained and checked in the overload matrix.

The helper bounds work by identity count, not a universal byte limit. Decoded
memory depends on source dimensions; GPU staging also retains its two source
pairs. Measured process RSS, rather than the queue bound alone, is the resource
acceptance criterion. Future folder changes, exceptional disk stalls and OS
descheduling can still miss presentations; there is no claim of physical
nanosecond scanout precision.

### Validation and measured results

**116 focused tests passed**, including deadline retention, no useful-job
cancellation, expired-work timing feedback, separate worker/ready capacity,
held-MIDI startup seeding, failure retry, exact scope pairs, decode timestamps,
interlaced pair retention, texture ownership, blocking-swap pacing and the
existing offscreen local-frame bridge smoke test.

Actual GL staging matched the direct renderer byte-for-byte on **eight
framebuffers**, with varying RGBA and SBS layers. The visible slot stayed
unchanged while the following picture was uploaded. Actual main.py Telnet and
WebSocket clients received **60/61 frames in about four seconds**.

The final overload matrix used the preceding six-second fault profiles, with
three baseline seconds and seven recovery seconds, same images/folders and no
network consumers. All twelve updated display/scope cases completed without
harness errors; the two added cases exercised two-field storage overload and
four-field CPU contention. Source hashes were held constant throughout this
matrix and its eight warmed resource trials. A subsequent local-only exposure /
blocking-swap correction was checked with the complete focused suite, actual
window checks and another warmed local run; the nonblocking/headless pacing path
retains the measured behavior.

| Mode | Slow-read updates: broken handoff -> fixed | Longest gap: broken -> fixed | CPU-fault updates: fixed |
|---|---:|---:|---:|
| Local | 1 -> **71** | 6.09 s -> **0.367 s** | 97 |
| Web | 1 -> **48** | 6.09 s -> **0.400 s** | 89 |
| ASCII | 1 -> **48** | 6.07 s -> **0.401 s** | 87 |
| ASCII-web | 1 -> **48** | 6.07 s -> **0.399 s** | 90 |
| Scope runtime images | 3 -> **60** sampled adoptions | 6.02 s -> **0.451 s** | 53 sampled adoptions |

Display updates are swap/publication observations; scope adoption is sampled
about every 50 ms and can miss intermediate adoptions. Longest gaps span each
complete run. Every shared-display update used the exact requested index/folder
identity. Under CPU contention, fresh local completion age reached two source
steps; headless completion age reached one. Scope retains its trace/adoption
latency: its fault-window fresh sampled-adoption age reached two/three steps.
Held-picture p95 error under slow reads fell to five steps for local, seven for
headless modes, and eight for scope, rather than approximately 169 steps.

All cases recovered without playing through a backlog. First observed
post-fault recovery was 7–129 ms across the default updated modes in these runs;
scope's observation has ~50 ms granularity. Two-field/four-field fault runs
continued adopting pictures and recovered after 125/68 ms, with longest gaps
351/302 ms. These are observations, not worst-case bounds or hardware timing
guarantees. Main's approximate-index policy still delivered more storage-fault
updates in several modes; this implementation retains the user's exact/miss
policy instead of restoring that approximation.

Normal warmed resource trials remain below the original pre-scheduling
baseline. CPU 100% means one logical core; RSS is process median:

| Mode | Original CPU | Fixed CPU | Original RSS | Fixed RSS |
|---|---:|---:|---:|---:|
| Local | 66.2% | 40.9% | 384.3 MiB | 312.0 MiB |
| Web | 16.8% | 15.3% | 280.7 MiB | 271.7 MiB |
| ASCII | 26.0% | 19.2% | 333.2 MiB | 310.0 MiB |
| ASCII-web | 26.6% | 18.7% | 337.8 MiB | 309.9 MiB |

These are two-trial averages using the existing two-second warm/six-second
measurement harness, fixed pair `(0, 0)`, software GL and no clients. Exact
clock/selected-index observations were local **723/723**, web **183/183**, ASCII
**183/183** and ASCII-web **183/183**. After the local exposure/VSync correction,
the extra local run measured **39.7% CPU / 310.2 MiB**, exact **362/362**.

The actual local idle/exposure check made one initial draw and one requested
refresh draw, **two total texture uploads**, and **one window initialization**:
exposure caused no texture upload or window rebuild. With an emulated 16.67 ms
blocking swap and a 60 IPS/60 FPS source, local sustained **54.3 swaps/sec**
instead of being timer-limited to roughly 30. This is a software pacing test,
not a physical scanout/latency measurement.

Scope received a separate matched HEAD-before/current-after comparison of its
runtime source and engine, with fixed pair `(0, 0)`, raster and NullStream:

| Scope phase | Before CPU / RSS | Fixed CPU / RSS |
|---|---:|---:|
| No viewer | 107.8% / 411.8 MiB | 63.3% / 398.3 MiB |
| 512-pixel trace viewer | 121.8% / 413.1 MiB | 77.7% / 399.7 MiB |
| Luminance viewer | 106.4% / 413.3 MiB | 63.8% / 399.8 MiB |
| After viewers | 107.2% / 414.1 MiB | 62.6% / 400.9 MiB |

Before/after HTTP clients each received 83 trace JPEGs and 180 luminance JPEGs
in their six-second phases. Both versions' sampled index/adoption errors stayed
within 0–2 source steps, with zero sampled DAC/pre-adoption drops. This includes
the earlier lossless thumbnail optimizations; it does not attribute all gains
to the new preparation helper. Main lacks runtime-image scope, so this baseline
uses HEAD's original runtime source/engine, not a fabricated main equivalent.

Modem's unchanged production path was also exercised in the overload matrix.
All **94/113 accepted submissions** in the storage/CPU cases matched their
reserved exact indices. Output diagnostics reported **7/3 deadline misses**,
zero producer starvations, and **one virtual-route output underflow in the CPU
case** (zero in the storage case). Producer acceptance is not proof of receiver
display or uninterrupted audio. Its recovered exact submission and timestamp
rejection behavior were preserved; end-to-end receiver stress was not added.

Artifacts: `tmp/deadline_verified_stress_summary.json`,
`tmp/deadline_verified_{mode}_{condition}.{json,log}` and interlaced cases,
`tmp/image_resources_{mode}_deadline_verified_{1,2}.json`,
`tmp/deadline_scope_resources_{before,after}.json`,
`tmp/deadline_final_regression.log`, `tmp/local_idle_refresh_check.json`,
`tmp/local_blocking_swap_check.json`, `tmp/texture_stage_gl_parity.json`,
`tmp/deadline_verified_ascii_delivery.json`, and `tmp/deadline_final_sources.json`.
