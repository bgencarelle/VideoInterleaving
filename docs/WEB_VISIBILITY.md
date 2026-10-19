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
