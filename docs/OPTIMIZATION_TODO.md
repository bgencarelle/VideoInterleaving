# Optimization TODO

Goal: tighten startup, frame delivery, and browser/display latency without changing
wire compatibility, source-clock semantics, or scope DAC timing.

## Rules

- Measure before and after each change.
- Keep local, ASCII, ASCII-web, and scope-web optimizations mode-scoped where possible.
- Do not replace the wall-clock source index globally; it preserves cross-mode alignment.
- Do not perform rendering, JPEG encoding, or blocking work in the scope audio callback.
- Preserve latest-frame behavior: stale work may be dropped, but output must not fall back to black or silence.

## Phase 0 — Baseline instrumentation

- [ ] Record startup-to-first-frame and startup-to-first-client-frame latency.
- [ ] Record frame deadline lateness: p50, p95, p99, and missed deadlines.
- [ ] Record image-loader queue depth, FIFO age, dropped work, and displayed index lag.
- [ ] Record scope callback underflows, trace-boundary handoff latency, and preview render time.
- [ ] Record browser luma rebuild count, worklet queue depth, and duplicate frames.
- [ ] Add small synthetic fixtures so timing comparisons do not depend on the full image tree.

## Phase 1 — Shared/local playback

- [ ] Preserve valid generated-list and folder caches; rebuild only for `--rebuild` or a changed source manifest.
- [ ] Remove the duplicate first-image decode during local startup.
- [ ] Bound image-loader work to one or two latest-needed tasks; cancel or ignore obsolete futures.
- [ ] Replace render-time-plus-sleep pacing with monotonic absolute frame deadlines.
- [ ] Choose one primary local pacing mechanism: VSYNC or explicit deadlines, not both.
- [ ] Decimate or move `psutil`, disk, process, and entropy monitor work off the render thread.
- [ ] Split transform-only updates (`R`/`M`) from full display/window reinitialization.
- [ ] Cache GLES/backend detection across display reconfigurations.
- [ ] Rate-limit FIFO-miss logging and avoid synchronous log I/O on the render path.
- [ ] Remove redundant window/event checks after measuring input-latency impact.

## Phase 2 — ASCII and ASCII-web

- [ ] Publish frames at effective `ASCII_FPS` instead of producing source-rate frames that clients discard.
- [ ] Make raw-image ASCII conversion use `ASCII_FPS`, not the unrelated server capture rate.
- [ ] Benchmark a CPU-only raw-image ASCII path; avoid GL render/readback when no visual GL output is needed.
- [ ] Replace per-character ANSI color escapes with run-length color encoding.
- [ ] Cache stable resize/index arrays, crop geometry, and reusable ANSI fragments.
- [ ] Bound ASCII FIFO depth and pending worker tasks for lower display age.
- [ ] Add decoded-image LRU caching only after measuring memory and ping-pong behavior.
- [ ] Remove unused `shared_state.exchange` writes after checking for external consumers.
- [ ] Add Telnet send timeouts and latest-frame replacement for slow clients.
- [ ] Avoid repeated bytes/text conversion in the WebSocket broadcaster.
- [ ] Coalesce browser terminal updates when the client is behind.

## Phase 3 — Scope web interface

- [ ] Separate `want_luma()` from `want_tap()` so browser luma does not enable trace capture.
- [ ] Cache encoded luma and trace-preview JPEGs by sequence, size, settings, and quality.
- [ ] Skip browser trace rebuilds when luma and all rendering parameters are unchanged.
- [ ] Use transferable buffers or a latest-only queue for main-thread to AudioWorklet updates.
- [ ] Cache repeated interlace luminance composition while keeping field-specific trace generation.
- [ ] Align browser rebuild cadence with the server's actual `IPS`.
- [ ] Add an explicit vector-mode luma publication path, or disable/explain browser-local audio there.
- [ ] Consider a bounded producer wakeup instead of polling scope readiness every 8.3 ms.
- [ ] Keep preview generation separate from the DAC callback and actual scope output.
- [ ] Lower or adapt the default preview resolution after measuring multi-client CPU cost.

## Phase 4 — Validation and cleanup

- [ ] Run clean-wire modem round trips after shared timing changes.
- [ ] Verify local rotation, fullscreen transitions, letterboxing, and Retina framebuffer sizing.
- [ ] Verify ASCII byte output remains identical at neutral grading settings.
- [ ] Verify Telnet/WebSocket clients receive the newest frame without unbounded queues.
- [ ] Verify scope output is unchanged with zero web clients, one client, and multiple clients.
- [ ] Verify scope callback has no new allocations, locks, or blocking operations.
- [ ] Document before/after measurements and keep only changes that improve p95 latency or jitter.
