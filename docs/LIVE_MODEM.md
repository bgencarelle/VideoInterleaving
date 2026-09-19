# Self-describing live modem

Stereo, and nothing else fixed. There is no mode word and no new wire
overhead; playback speed comes from preamble pulse timing.

Three things used to have to be named identically at both ends. None of them
does now:

| | how the receiver gets it |
|---|---|
| sample rate | read off the device it opened, and the cadence off the preamble |
| profile | declared in two spare bits of the header's top_bin byte |
| preset | identified by decoding against each candidate until the CRC verifies |

The current user-facing choices are self-describing and detected by the
receiver automatically: wide 80x96 (`--wire wide`), redundant tape 80x96
(`--wire tape --profile hd-dwt`), and redundant tape 80x60
(`--wire tape --profile tape-80x60`). At 48 kHz they run at approximately
13.76, 16.48, and 25.21 fps respectively. The tape wires use 375-12750 Hz
carriers and limit the complete waveform to 14 kHz. The receiver does not need
the sender's wire or profile arguments.

The v2 layouts are gone outright, so there is no transport generation left to
disagree about. Nothing is shared state: the receiver reads the sample rate
off its device, the profile off the header, and the wire is fixed.

## No sample rate is requested

The frame is a sample count, not a duration, and the decoder gets its timing
by counting preamble edges the way an LTC reader counts its own clock. So
neither program asks a device to run at any particular rate: streams open at
whatever the device is already set to, and the rate reported back is used for
buffer sizing, the displayed frame duration, and the reported speeds and
frequencies. Nothing else. `read` takes the rate from the WAV header the same
way, so a 44.1 kHz recording of a 48 kHz transmission decodes as what it is.

48 kHz remains the reference clock the geometry was designed against; each
wire's frame rate is listed above, and `write` stamps 48 kHz into a new file. It is a
label, not a request. A receiver at 44.1 kHz sees the same transmission as a
0.919 timing scale and reports a true 1.00x playback speed; measured picture
error across 44.1, 48, 88.2, 96, 176.4 and 192 kHz inputs stays under 0.004
RMS, with 44.1 kHz the worst of them because the top carrier is nearest its
Nyquist there.

## The sender is band-limited

Receiving is indifferent to the rate for free — a faster capture clock only
oversamples. **Transmitting is not symmetric**: the band a sender emits is its
own clock times the carrier geometry, so playing the sample array unchanged out
of a 96 kHz device would put the picture at 750–40500 Hz, past a low-end DAC's
reconstruction filter and past any sane receiver's Nyquist.

So a transmitter resamples to its device instead, holding the emitted carriers
at 375–20250 Hz while preserving whichever wire's frame rate is selected.
Nothing is asked of the hardware; the signal adapts to it. Energy above 24 kHz
measures 57 dB down, and the kept band is bit-identical in level to the
reference — resampling overshoots intersample peaks by about 2.3 dB, so the
packet is renormalised to the level `encode` set rather than clipping.

Below the reference rate nothing happens: the band scales down with the clock
(345–18605 Hz at 44.1 kHz), which is further inside the DAC's passband, not
outside it. So the resampling cost is zero at 44.1 and 48 kHz, and 3–6% of a
core at 88.2–192 kHz.

The ratio is deliberately coarse — 88.2 kHz uses 11/6, not the exact 147/80,
because that transition band needs ~15000 taps and 5 ms a frame. 11/6 costs
1129 taps and lands 0.23% off, and that residual is not corrected: the receiver
measures the scale that actually arrives, so it is a real 1.0023x transmission
rather than an error. Clean multiples (96, 192 kHz) stay exact.

The upshot is that every sender/receiver rate pair works, not just matched
ones. A 192 kHz sender reaches a 44.1 kHz receiver; both senders print the band
they are emitting whenever their device is not at the reference rate.

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
- `--on-loss hold` keeps the last good picture.
- `--on-loss damaged` shows a recoverable damaged picture as-is; if no new
  picture can be decoded, the last good picture is held. The modem video path
  never emits or displays black as a fallback.
- Logging and image saving use a separate worker with only one pending item.
  Slow outputs may skip records/images rather than stall the decoder.
- Q, Escape, or window close quits the receiver; terminal Ctrl-C quits either
  program. Headless reception supports Ctrl-C as well.

The older `--buffer-ms` override is retained for compatibility. Prefer frame
counts.

Live reception takes no preset or profile argument at all, and `read`'s are
fallbacks used only when no header verifies. Identifying the preset costs one
decode attempt per candidate -- about 1.4 ms each, so roughly 10 ms against a
58 ms frame -- and is paid once per lock, not per frame, because the layout is
adopted as soon as one verifies. Acquisition can do this because the preamble
is the one part of the packet that does not depend on the layout.

## Image and audio trade-offs

The tape 80x96 coder puts 24x28 luma and 8x8 Cb/Cr DCT coefficients on the
wire, duplicates all 800 retained values across opposite stereo legs and
separated carriers, and reconstructs at 80x96. The tape 80x60 coder puts 20x15
luma and 6x5 Cb/Cr coefficients on the wire, duplicates all 360 retained
values, and reconstructs at 80x60. The receiver reconstructs the geometry named
by the header.

The carrier band is 375–20250 Hz and requires stereo; it is held there on any
sending device above the reference rate and scales down below it, as above.
A successful
virtual loopback test does not validate restricted-bandwidth, noisy, clipped,
channel-mixed, or voice-processed physical paths. This patch retains the
working baseline; it does not claim universal real-world audio compatibility.
