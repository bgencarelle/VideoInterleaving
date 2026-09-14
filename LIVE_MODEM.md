# Fixed live modem

The live transport is `lean-v3` with `color-lean`: stereo, 48 kHz,
2768 samples per frame, approximately 17.34 fps. There is no mode word and
no new wire overhead. Playback speed comes from preamble pulse timing.

Run the sender and receiver in separate terminals:

```sh
python modem_screen.py --source camera --device 0 --capture-fps 30
python utilities/modem_v3_check.py live-receive --device 0
```

Use `--source screen` for screen capture. Select the actual input/output
devices for your audio path; BlackHole is only one possible test path.
The receiver prints its input device name and channels at startup and keeps
that information visible in the window.

## Freshness and loss

- `--buffer-frames 2` is the default; `--buffer-frames 1` reduces the queue.
  Capacity follows the measured frame duration. Oldest audio is discarded on
  overflow and the pulse reader reacquires after the gap. This bounds queued
  work, not operating-system stalls or physical audio-device latency.
- `--on-loss hold` keeps the last verified, non-degraded picture (default).
- `--on-loss black` blanks damaged or overdue frames.
- `--on-loss damaged` shows recoverable damaged pictures, otherwise black.
  Missing-frame timeouts follow measured playback speed with a small margin.
- Logging and image saving use a separate worker with only one pending item.
  Slow outputs may skip records/images rather than stall the decoder.
- Q, Escape, or window close quits the receiver; terminal Ctrl-C quits either
  program. Headless reception supports Ctrl-C as well.

The older `--buffer-ms` override is retained for compatibility. Prefer frame
counts. Live reception no longer accepts preset/profile arguments. Offline
experiments can still use explicit formats; those are not live compatibility
guarantees.

## Image and audio trade-offs

Grayscale images use the same fixed colour geometry with neutral chroma;
the receiver displays them without a format flag. This does not reclaim the
chroma bandwidth. The image format remains 40x48 luma and 10x12 chroma planes.

The nominal carrier band is 375–20250 Hz and requires stereo. A successful
virtual loopback test does not validate restricted-bandwidth, noisy, clipped,
channel-mixed, or voice-processed physical paths. This patch retains the
working baseline; it does not claim universal real-world audio compatibility.
