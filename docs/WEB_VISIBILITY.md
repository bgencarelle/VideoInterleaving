# Web viewer visibility and freshness

## Policy

The clock remains authoritative. A returning viewer requests the current
stream; it must not replay a hidden-tab backlog. A reconnect can still take a
network round trip, during which an existing picture may remain visible.

## Fixes (2026-10-19)

- ASCII-web gzip work is scoped to its WebSocket connection. An unfinished
  decode from a hidden/closed connection cannot block a resumed connection,
  overwrite its picture, clear its pending data, or close its socket.
  Only the current connection negotiates encoding. Local zoom/appearance and
  the latest-only renderer remain unchanged.
- Web MJPEG closes the image stream while hidden and opens a fresh stream on
  return. Old status-poll responses cannot restart a newer stream. Image error
  handling is attached from JavaScript rather than invoking an inline function
  before the script has loaded.
- Both scope pages cancel pending preview retries on hide/resume. Preview and
  luminance retries are coalesced, and hidden pages do not initiate retries.
  A running browser-audio session refreshes its luminance stream on return,
  even if the old image element still had a source. Background trace rebuilding
  is suppressed; this does not start/stop an AudioContext or change an output
  destination. Scope render arithmetic, defaults and server scheduling are
  unchanged.
- Plain ASCII TCP encoding/protocol is unchanged.

## Evidence and limits

The ASCII controller regression test first reproduced a frozen old gzip decode
blocking a new connection. It now verifies that the new picture renders before
the old decode is released, and that the old result cannot overwrite it.
`tests/web_visibility.test.js` exercises MJPEG hide/resume, stale status
responses, and preview/luminance retry cancellation on both scope pages.

Actual `main.py` web and ASCII-web image-backed processes and a baked scope
process were exercised in Chromium. Each of `/`, `/ascii`, the scope dashboard
and `/scope2` completed three scripted visibility changes plus real CDP page
freeze/resume cycles, with fresh stream requests/data and no page errors.
Before/after screenshots were inspected. Scope used NullStream for this
HTTP-preview lifecycle check; browser audio was not started. This is not a
PortAudio, hardware scope, or browser-local audio playback validation.

Artifacts: `tmp/live_web_visibility_20261019_v2/summary.json`, application logs
and before/after screenshots; runner `tmp/check_live_web_visibility.cjs`.
Supporting checks passed 49 Python tests and six Node test entries across
visibility, ASCII rendering, protocol, page routing, lazy imports and scope
output-mode/timing checks. Safari and actual macOS tab switching have not been
verified here; slow-network reconnect latency and end-to-end frame age are not
bounded by these checks.

## Shared-clock presentation check — goal not yet met (2026-10-19)

An additional controlled run sampled actual native-window pixels, browser
canvas/MJPEG pixels, and plain ASCII TCP receipt against the application's
shared birth-time clock at 30 IPS. A 128-image Gray-coded JPEG/matte fixture
made the frame identity visible without trusting server publication counters.
All processes used the same clock epoch and independent face/float pair `(0,0)`;
the harness pinned that pair because the one-folder fixture triggered the
stochastic selector's empty-deque case. This does not validate stochastic folder
changes. All modes ran concurrently on the software-rendered Linux test host.

Median / 95th-percentile measured frame age, milliseconds:

- Local native-window pixels: 39 / 56.
- Plain ASCII TCP receipt: 42 / 51 (not terminal paint timing).
- Web MJPEG browser pixels: 115 / 141.
- ASCII-web canvas pixels: 99 / 152.
- Scope raster HTTP-preview pixels: 178 / 224 (NullStream, not physical output).

These steady-state statistics exclude the first 300 ms after tab return and
positions within eight source ticks of ping-pong turns, where the same image
appears on both sides and pixels alone cannot disambiguate signed age.
Separate resume samples show ASCII-web retaining its pre-hide picture for
roughly 150 ms before new frames arrived, then jumping directly to live data.
There was no replay of the two-second hidden interval. That held picture was
still about two seconds old; reconnect success is not presentation sync.

The browsers trail local on this host: the primary cross-mode synchronization
goal is **not established** by the visibility fixes. These are measured results,
not approved offset budgets or compensation defaults. Scope frame-code reading
was corrected to find the fixed pilot bars in actual trace pixels rather than
assuming the trace's geometry. Raw first-attempt data is retained separately.

Artifacts: `tmp/cross_mode_clock_20261019/{summary.json,browser_samples.json,local_samples.json,tcp_samples.json}`,
screenshots and application logs. Helpers: `tmp/build_clock_identity_source.py`,
`tmp/cross_mode_clock_worker.py`, `tmp/check_cross_mode_clock.cjs`.
Modem, physical scope, real terminal paint, Safari/macOS, and arbitrary network
delay are not covered. No synchronization compensation was changed by this test.

## Delivery repair and scope-page promotion (2026-10-19)

The owner's restored FIFO loading, visible-loop pacing and Wayland fullscreen
fixes at `86989a6b7` are the baseline. This repair does not change
`image_display.py`, `image_loader.py`, `image_preparation.py`,
`display_manager.py`, the scope producer/output pipeline, or modem code.
Nearest-available fallback and smooth playback are retained; exact-index
admission is not reinstated.

- ASCII-web producers now replace a single pending-picture mailbox. Only the
  socket thread admits messages to the WebSocket library's queue. Setup,
  resize and other controls are preserved; partially written frames are never
  removed. Legacy clients need not negotiate flow control.
- The browser opts into one acknowledged picture in flight. Each picture has
  a numbered control message; only its matching acknowledgement grants credit.
  Duplicate/stale ACKs and repeated negotiation cannot release another picture.
  A paused connection can still receive its already-in-flight picture; the
  next admitted picture is the newest waiting one, not a saved sequence.
  Hidden tabs disconnect and return on a fresh connection. Old asynchronous
  decodes cannot acknowledge or overwrite that connection.
- Scope preview and luminance now use the existing length-framed multipart
  writer, matching `LiveImageStream`. Previously local reader edits required
  lengths that those endpoints did not emit. Integration tests feed bytes from
  both actual HTTP handlers into the actual JavaScript parser, rather than
  mocking the reader at the lifecycle-test boundary.
- The former scope2 layout is now `templates/scope.html`, including browser
  audio/output controls. `/scope`, `/scope2`, `/scope.html` and `/scope2.html`
  serve the same template; scope mode's root serves it too. There is no second
  independently maintained scope2 template. Hidden scope sessions abort the
  luminance feed while retaining the AudioContext and its last usable trace.
- A small socket send buffer limits bytes, not an exact number of pictures.
  Application mailboxes and decode queues do not store playback history; this
  is not a claim of zero in-flight/network latency or exact cross-mode sync.

Focused regression checks: 63 Python tests and 11 Node test entries passed,
including real TCP/WebSocket handshake, preserved setup, partial 512-KiB writes,
legacy delivery, numbered ACKs and latest-only replacement during a pause.

Actual-process browser check: `main.py --mode web`, `asciiweb`, and baked scope
with `--device null`, software Chromium under Xvfb. Web, ASCII-web, scope root
and `/scope2` each completed a six-second throttling phase (64 KiB/s HTTP,
120 ms configured latency), return to normal bandwidth and three visibility /
CDP freeze-resume cycles, including a six-second hidden interval. Sampled
displayed-image hashes continued changing after recovery; screenshots were
inspected and all four pages reported no JavaScript errors. These are
advancement/recovery checks, not measurements of exact frame age or scanout.
CDP HTTP throttling is not evidence that WebSocket traffic was throttled.

The actual ASCII browser separately withheld numbered ACKs for six seconds:
at most one further picture arrived before release. The real-socket regression
then checks that publishing 90 replacements yields only the latest pending
picture after the matching ACK. The first browser attempt incorrectly assumed
CDP's `frozen` state alone stopped WebSocket ACK processing; this Chromium build
continued processing them. That first failed test assertion is retained, not
reported as evidence of a server backlog. The corrected explicit-ACK-withholding
test passed.

Browser-local scope audio was started on the root page and retained through
the recovery cycles (472 traces built, AudioContext rate 96 kHz), then stopped,
started in X-only mode and stopped again. Browser output was muted; PulseAudio
was verified beforehand as the existing `auto_null` / `auto_null.monitor`
96-kHz null route, without creating devices or changing defaults. This checks
the real AudioContext/worklet/luminance integration, not physical audio,
waveform quality, tape, or hardware scope output. Safari/macOS and hardware
WebGL remain unverified.

Artifacts: `tmp/web_scope_repair_20261019_v2/summary.json`, process logs and
before/after screenshots; `tmp/check_web_scope_repair.cjs`,
`tmp/web_scope_repair_browser_v2.log`, `tmp/web_scope_repair_final_python.log`.
The first invalid freeze-assumption attempt is retained separately in
`tmp/web_scope_repair_20261019/` and `tmp/web_scope_repair_browser.log`.
