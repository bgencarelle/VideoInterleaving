# GUI Mods: V7 Sender Preview and Device Recovery

## Purpose

Track the standalone V7 sender/receiver preview, sender-GUI usability, saved
preferences, and live audio-device recovery work. This is separate from the baked
`main.py --mode modem` pipeline.

Feature branch: `feature/v7-live-preview-recovery`
Base branch: `modem-v7-integration`
Worktree: `~/modemTest/tmp/v7-live-preview-recovery`

## How to use this checklist

- Check an item only after its implementation is complete and the listed
  verification has passed.
- Add a short note under **Progress log** for each completed step, including the
  tests or smoke check used.
- Keep the branch isolated from the source-DCT changes in the original
  worktree.
- Commit the completed feature on this branch after final verification. Do not
  include generated artifacts or unrelated worktree changes.

## Work items

### 1. Branch and worktree

- [x] Create `feature/v7-live-preview-recovery` from the committed
  `modem-v7-integration` base in a separate worktree.
- [x] Move this coding session to the new worktree and verify the branch is
  clean before feature edits.
- [x] Rebase onto the latest V7 integration base before push.

### 2. Optional native video preview

- [x] Add `--preview` to `tools/v7_live.py send`; default off and only valid for
  `--source video`.
- [x] When enabled, open the selected file/URL in an ordinary desktop video
  player, separately from the sender's capture/encode path.
- [x] Prefer a controllable player such as `ffplay` when available; keep launch
  non-blocking and clean up a player process owned by the sender.
- [x] Request infinite looping for finite file/VOD sources when the sender loops
  them; do not request looping for live sources. If only the system-associated
  player is available, report that repeat behavior is controlled by that app.
- [x] Keep the preview video-only when the player supports muting, so it does not
  add a second soundtrack to the sender's audio output.
- [x] Document that the separate player has its own playback timeline and is not
  frame-synchronized with the sender.

### 3. Sender GUI source controls

- [x] Add a video-only preview toggle to `tools/v7_send_gui.py` and forward it to
  the CLI command.
- [x] Add a **Change source** action on the Live page. It must gracefully stop
  the sender, return to Setup with the source selector focused, and require an
  explicit Start for the next source.
- [x] Preserve source-specific selections while switching, including video
  path/URL and live mode, camera, screen/display, region, and capture options.

### 4. Restore last sender configuration

- [x] Save the last sender-GUI settings in a versioned JSON file in the user's
  config directory.
- [x] Restore the last source and setup values on launch; do not auto-start the
  sender.
- [x] Persist audio-device identities by stable name/host API where available,
  rather than relying only on transient device indexes.
- [x] Handle invalid preferences and unavailable devices with safe defaults and
  a visible status message; write preferences atomically.

### 5. Audio-device loss and sample-rate recovery

Cover the sender's primary output and optional source-audio input, plus the
receiver's primary input and optional passthrough output.

- [x] Detect a stopped/failed stream and device removal/reindexing using the
  selected device's stable identity; do not silently fall back to an unrelated
  default device.
- [x] On primary sender-output loss, pause packet output, discard stale queued
  audio, keep the process alive, and clearly announce the paused state.
- [x] On primary receiver-input loss, stop decoding new samples, keep displaying
  the last good picture, and visibly announce the lost device.
- [x] If an optional sender source-audio input disappears, keep video sending
  with silence on that audio leg and announce the audio-route loss. If receiver
  passthrough output disappears, keep video capture/decoding running and report
  that passthrough is unavailable.
- [x] Require the same device identity and reported sample rate for five
  consecutive 30-fps intervals (about 167 ms) before reopening after a device or
  sample-rate change.
- [x] Reopen using the stream's negotiated sample rate. On the sender, rebuild
  rate-dependent packet conversion and resume with fresh audio, not stale PCM.
- [x] On the receiver, clear rate-dependent capture/decode state and queued
  samples across the discontinuity; update passthrough resampling if its input
  rate changed. Keep the last good picture until a new valid frame arrives.
- [x] Expose recovery/loss status in CLI output and the relevant sender/receiver
  GUI status areas.

### 6. Tests and documentation

- [x] Add mocked-device tests for removal, return, index changes, sample-rate
  changes, five-interval stability, and reconnect behavior without stale audio.
- [x] Test that receiver output holds the last good picture until a fresh valid
  frame is decoded after recovery.
- [x] Test preview CLI validation/player arguments, looping policy, and process
  cleanup.
- [x] Test sender-GUI preview command construction, source switching, and
  preference restore/fallback behavior.
- [x] Update sender help and the relevant standalone V7 documentation in
  `docs/MODEM_MODE.md` and `docs/transport_v7_spec.md`.
- [x] Run focused sender, receiver-audio, input/decode, and GUI tests.
- [x] Run `.venv/bin/python -m unittest discover -s modem_tests -v` from the
  worktree root.
- [x] Smoke-test the sender and receiver GUIs under Xvfb and exercise the
  preview loop path with a short generated clip; non-looping policy is pinned by
  CLI tests.
- [x] Exercise device recovery with mocked devices; use only an explicitly
  verified virtual/loopback route for any live audio test.

### 7. Final review and commit

- [x] Review `git diff` and `git status`; confirm only this feature's files are
  included and no generated artifacts are present.
- [x] Confirm all preceding checkboxes represent completed and verified work.
- [x] Commit the finished feature on `feature/v7-live-preview-recovery` with a
  descriptive commit message.
- [x] After review, fast-forward `modem-v7-integration` to the feature commit;
  preserve and reapply the original worktree's uncommitted
  `docs/MODEM_MODE.md` edit.

## Progress log

- [x] Branch/worktree created from `4623e654`, then rebased onto the latest
  `modem-v7-integration` commit `909fb78b`; worktree is
  `~/modemTest/tmp/v7-live-preview-recovery`.
- [x] Preview implementation and verification; CLI tests passed and muted
  `ffplay` loop smoke passed under Xvfb.
- [x] Sender-GUI workflow/preferences implementation and verification; source
  switching, device identity restore, and no-auto-start are pinned by tests.
- [x] Device-recovery implementation and verification; mocked output/input
  loss and 48→96 kHz reconnect tests passed.
- [x] Full modem test run (489 tests); focused tests and GUI/player smoke checks
  passed.
- [x] Application integration and lazy-import checks (12 tests) passed after
  rebasing onto the V7 integration base.
- [x] PortAudio end-to-end loopback under Xvfb: PulseAudio `loop` null sink to
  `loop.monitor`; sender exited cleanly and receiver decoded 43 baseline frames
  through source index 45 at 0.5×.
- [x] Integrated `--preview` video send on the same PortAudio null-sink route:
  67 packets sent, 65 decode reports, and no `ffplay` process remained after
  sender exit.
- [x] Repeated the PortAudio preview send after rebasing to the latest V7 commit:
  68 packets sent and 66 decode reports.
- [x] Final review and feature commit (`feat: add V7 sender preview and device recovery`).
- [x] Merged locally by fast-forward to feature commit `fde81404`; reapplied the
  original `docs/MODEM_MODE.md` edit, which remains unstaged in the worktree.
