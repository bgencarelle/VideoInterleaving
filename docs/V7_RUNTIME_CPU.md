# Standalone V7 runtime CPU budget

## Change

The standalone sender and receiver run their sessions with one BLAS worker by
default. `tools/v7_runtime.py::blas_budget()` limits already-loaded NumPy and
SciPy BLAS pools through `threadpoolctl`. SciPy's BLAS module is loaded before
pool enumeration, including the library used by Numba `np.dot` kernels.

The budget covers sender producer threads, receiver decode threads, offline
WAV operations and reconnects, and restores the prior library limits on exit,
including exceptions. The GUI sender uses the same CLI entry; the receiver GUI
uses the same receive entry. Numba's parallel-kernel thread count stays intact.

The reason is worker overhead on small, frequent matrix operations. Four BLAS
workers were consuming several CPU-seconds per second of paced audio, despite
the actual decode work needing much less. A one-worker budget reduces that
overhead while retaining the existing compiled arithmetic and wire format.

`V7_BLAS_THREADS=N` overrides the worker budget. `V7_BLAS_THREADS=0` retains the
libraries' own settings, useful for before/after comparisons. Limits are
process-wide for supported BLAS libraries while a session is running; they are
not a portable promise about every numerical backend. Mac performance has not
been measured here.

## Matched measurements (2026-10-15)

Linux x86-64 sandbox, AMD EPYC-Milan, Python 3.13, four default BLAS workers.
Two OpenBLAS libraries were present: system NumPy and SciPy's bundled library.

The local harness `tmp/v7_cpu_profile.py` exercised the production encoder and
the production live receive loop, with a fake input device delivering 1,024
samples every 1,024/48,000 seconds. This is a software timing/CPU measurement,
not PortAudio, a DAC or tape validation. The receiver published every picture
into a collecting display mailbox. Receiver measurements include Python
profiling overhead and session initialization; the sender was warmed before
timing its 48-picture encode loop. CPU time covers all process threads.

### Adaptive stereo nested, 48 moving 200×240 pictures

| Workload | Before CPU seconds | After CPU seconds | CPU reduction |
|---|---:|---:|---:|
| Sender encode | 2.460 | 0.555 | 77.4% |
| Paced receiver, smoothing off | 13.265 | 1.221 | 90.8% |
| Paced receiver, two smoothing passes | 12.850 | 0.925 | 92.8% |

Sender wall time decreased from 0.620 to 0.555 seconds. Paced receiver wall
time remained about 4.6–4.9 seconds, as expected for a fixed-rate input.
All 48 pictures were published in each receiver run.

The entire emitted waveform was bit-identical. Across all 48 decoded pictures,
the largest absolute before/after coefficient difference was 1.45e-15 with
smoothing off and 2.67e-15 with two passes, consistent with BLAS rounding order.

Artifacts: `tmp/v7_cpu_matched_before/`, `tmp/v7_cpu_matched_after/` contain
metrics, waveforms, all decoded values, and sender/receiver cProfile data.
Run the harness in separate processes with `V7_BLAS_THREADS=0` and then the
default environment; avoid competing CPU benchmarks during measurement.

### Default aspect Fold-500, 48 moving 960×720 pictures

The larger-source comparison also controls the saved startup-profile hint so
both receivers begin in the same state, without reading/writing GUI preferences.

| Workload | Before CPU seconds | After CPU seconds | CPU reduction |
|---|---:|---:|---:|
| Sender encode | 0.850 | 0.418 | 50.8% |
| Paced receiver, first session | 1.094 | 0.967 | 11.6% |
| Paced receiver, warmed session | 0.675 | 0.645 | 4.4% |

Sender wall time was 0.426 → 0.418 seconds. All 48 pictures were published
in both receive sessions. Emitted waveforms and decoded values were
bit-identical. The warmed default receiver was already inexpensive; its small
timing difference is not evidence of the large nested-profile benefit applying
to every profile.

Artifacts: `tmp/v7_cpu_default_controlled_before/` and
`tmp/v7_cpu_default_controlled_after/`.

## Regression and device-route checks

- 359 focused tests passed: runtime worker limits/restoration, file workflows,
  sender transport, both GUIs, source audio/capture, device recovery, adaptive
  profile dispatch, live input, application modem integration and lazy imports.
- `utilities/check_modem_setup.py` passed and now checks `threadpoolctl`.
- An adaptive-stereo-nested live sender completed a finite local-video run
  through the verified `pulse` → PulseAudio `loop` virtual route at 48 kHz,
  recording `tmp/v7_cpu_live_after.wav`. That application-output recording
  decoded as four verified pictures with zero lost. This recording is before
  device routing; it does not measure downstream monitor audio.
- `git diff --check` passed. Full modem discovery's known analog-frame and
  stereo-slices failures are recorded in `V7_PLAYBACK_FILE_IO.md`; the focused
  performance regression run above does not claim to clear those failures.

## Compiled-path audit follow-up

NumPy operations execute native code too. The useful distinction in this audit
is whether a packet crosses Python/NumPy repeatedly and allocates temporary
arrays, or whether its arithmetic is fused into a compiled kernel.

| Path | Finding/action |
|---|---|
| `v7.encode_pulse_frame_coeffs` final limiter | Moved staged packet assembly, peak scans and both 16-round searches into `v7_packet_kernels.py`, using cached Numba kernels with the GIL released. Trial peak searches no longer allocate complete trial packets; metadata trials read only metadata samples. |
| `v7.windows` rank placement | Tables depend only on a group count. Cache read-only tables instead of reconstructing them during packet decoding. |
| `nested_fold.choose_level`, `clean`, encode/decode | Python wrappers already dispatch to compiled `tools/v7_sk_fold.py` kernels. Their names in cProfile do not imply Python arithmetic. |
| `v7_source_dct._direct_planes` and `direct_dct_values` | Much of their profile time is compiled preparation called through Python wrappers. Inspect callees before treating wrapper time as unfused NumPy. |
| Float-frame gamma and clip-aware luma | Keep the measured-fast NumPy power and SciPy FFT-based DCT paths, as recorded in `AGENTS.md`. |
| Audio callback meters, input leveler, CP recovery scoring | NumPy expressions remain here. They are candidates for a separately measured fused-kernel pass, with acquisition/recovery behavior preserved. |

The limiter preserves the original float32 rounding after each mix stage,
float64 metadata multiplication and both gain-search iteration counts. It
does not use fast-math or change framing, tones or gain policy.

The isolated real packet-encoder comparison uses the previous implementation
from HEAD as its oracle, with one BLAS worker on both paths. Median CPU time
over five warmed runs of 64 packets:

| Packet case | Before | After | Reduction |
|---|---:|---:|---:|
| No pilot overlay | 0.1016 s | 0.0830 s | 18.3% |
| Pilot overlay enabled | 0.1042 s | 0.0853 s | 18.1% |

Actual emitted packets matched bit-for-bit across eight different coefficient
amplitudes in each case. Local harness/evidence:
`tmp/v7_limiter_bench.py`, `tmp/v7_limiter_fused_bench.log`.

Full warmed-pipeline comparisons, relative to the preceding one-worker BLAS
baseline (not the original four-worker baseline):

| Workload, 48 moving pictures | Previous CPU | Compiled/cache CPU |
|---|---:|---:|
| Nested sender, 200×240 | 0.555 s | 0.508 s |
| Default sender, 960×720 | 0.418 s | 0.381 s |
| Nested warmed receiver, two smoothing passes | 0.925 s | 0.870 s |
| Default warmed receiver | 0.645 s | 0.596 s |

All emitted audio and all 48 live-decoded value arrays were bit-identical to
the one-worker baseline, with smoothing both disabled and set to two passes.
The receiver profile no longer recorded rank-window table construction during
the measured decode. Receiver timing differences are small enough that the
eliminated table work is stronger evidence than a precise percentage claim.

Artifacts: `tmp/v7_cpu_compiled_nested_warm/` and
`tmp/v7_cpu_compiled_default_after/`. These are warmed-cache comparisons. A
cold-cache exploratory run after editing the shared decoder module had to
recompile existing decoder kernels while input arrived and fell behind
(`tmp/v7_cpu_compiled_nested_after/`); it is excluded from steady-state CPU
figures. First-use JIT compilation remains separate from these runtime gains.

Regression checks after this follow-up: **383 tests passed**, including the
frozen NumPy limiter oracle (24 combinations of body/metadata/tone amplitude),
packet levels, level independence, metadata, geometry/wire profiles, the live
and GUI checks above, and application integration/lazy imports. Log:
`tmp/v7_compiled_regression.log`.

## Capture reuse and callback allocation follow-up

### Sender

The live producer keeps a one-picture `PreparedFrameCache`. `Throttled.snapshot()`
returns a picture and its publication generation together under its lock.
`FilePlayback` also supplies a picture generation which advances only for a
new decoded picture, so repeated paused pictures retain their identity. Shared
soundtrack pictures use the retained decoded-picture object as their identity.

Reuse requires the same picture object, capture token, tone settings, DCT
options, selected kernel/parameters, rotation/mirror and preview setting. A
kernel reload is distinct. Sources without a stable capture token bypass reuse.
There is no pixel hashing and no assumption that a newly captured screen is
unchanged just because it resembles its predecessor.

Only source preparation and its deferred gains are reused. Packet generation,
metadata/counters, index stamps and soundtrack reads continue on every packet.
The cache retains one picture, not an accumulating history. Its lifetime is one
send session, so reconnecting starts a fresh cache.

Matched warmed CPU measurements for 48 packets from 12 different pictures,
each capture used for four packets (three runs, median; one BLAS worker):

| Sender profile | Reprepare each packet | Reuse captured picture | Reduction |
|---|---:|---:|---:|
| Default aspect Fold-500 | 0.3858 s | 0.1610 s | 58.3% |
| Adaptive stereo nested | 0.4482 s | 0.1820 s | 59.4% |

Emitted waveforms were bit-identical. These gains apply to held/slowly updated
captures; a new capture every packet still needs new preparation.

### Receiver callback and alignment

`v7_input_kernels.select_and_measure()` selects physical channels, creates the
decoder-owned sample block and calculates float64 peak/RMS/activity in one
compiled pass. It avoids the previous float64 block conversion, multiple
absolute-value/square temporaries and second decoder copy. The selected block
is also handed to soundtrack capture before queuing for decoding; soundtrack
capture synchronously copies its selected audio leg. No queued stream buffer is
reused while the decoder owns it.

The callback signature is warmed before opening input. NaN/silence activity
policy and sample values are preserved. RMS can differ at floating-point
reduction roundoff; tests compare it to the float64 reference within 2e-15.

The channel aligner avoids duplicate float32 conversions and, after its normal
history update/remeasurement, bypasses delay-line concatenation/output copies
when the applied offset is zero. Signed offsets, float32 quantization for
float64 callers, and transitions back to zero retain the previous behavior.

Matched isolated medians (five runs):

| Work | Before CPU | After CPU | Reduction |
|---|---:|---:|---:|
| 2,000 callbacks, 1,024 samples, selecting 2 of 8 channels | 0.0587 s | 0.0125 s | 78.8% |
| Align 48 packets, zero skew | 0.00677 s | 0.00363 s | 46.4% |
| Align 48 packets, 128-sample skew | 0.00482 s | 0.00441 s | 8.5% |

The full paced adaptive-nested receiver used 0.787 CPU seconds in its warmed
two-pass-smoothing run, versus 0.870 seconds before these changes. All 48
decoded pictures were bit-identical with smoothing disabled and enabled; the
waveform fed to the live loop was also identical. This is a software-fed live
receive-loop comparison, not physical/tape evidence.

Harness/results: `tmp/v7_live_work_bench.py`, `tmp/v7_live_work_metrics.json`,
`tmp/v7_cpu_reuse_callback_after/`. The sender pause integration test observes
packet preparation requests rather than counting expensive source conversions,
and checks that paused packets continue while conversion calls are reduced.

Regression validation after this pass: **438 tests passed**, including capture
identity/paused-seek invalidation, live-setting/kernel cache invalidation,
deferred-gain replay, callback buffer ownership, mono/reordered multichannel
metering, nonfinite/activity policy, signed alignment and zero-offset
transitions, sender transport, receiver soundtrack routing, both GUIs and the
previous packet/integration suites. Log:
`tmp/v7_reuse_callback_regression.log`.
