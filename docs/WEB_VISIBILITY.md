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
