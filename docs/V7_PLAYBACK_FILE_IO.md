# Standalone V7 playback and WAV workflows

These controls use the current standalone sender and receiver
(`tools/v7_live.py`, `vi.modem-send`, `vi.modem-receive`, and their GUIs).

## Playback and finite runs

- `--source-passes N`: stop after N local-video passes; zero loops.
- `--no-loop`: alias for one source pass, then stop the sender.
- A resumed or sought partial pass counts as a pass when it reaches EOF.
  Restart resets the pass count. Pause, seek and restart retain their existing
  behavior; picture and soundtrack readers reopen together.
- `--frames N`: exact packet limit, including a partial final encoding batch.
  Packets accepted by the output stream count toward the limit across reconnects.
- `--seconds S`: duration limit; zero means unlimited. Live timing starts after
  encoder warmup. Offline timing uses emitted packet duration.
- With several limits, the first reached ends the run. A normal stop drains
  queued output and finalizes the WAV header.

## Audio channels and source geometry

`send --channels 3,5` routes logical left/right to physical outputs 3 and 5.
`receive --channels 5,3` reads physical inputs 5 and 3 as logical left/right.
Pairs are 1-based, ordered, positive and distinct. The receiver validates the
pair against the input device or WAV before acquisition. Omitted receiver
channels retain automatic first-pair/mono-input behavior.

Selection precedes level detection, profile probing, channel alignment and
decoding. Mono picture-side and soundtrack-side choices are relative to the
logical pair. Stereo M/S remains stereo M/S. WAV output stores the logical
modem channels, without unused physical channels.

`send --rate HZ` requests an output rate. The sender GUI provides the same
setting as a text field; blank follows the host's device clock. Host APIs may
resample a requested rate. Actual stream rate controls the emitted samples.

`--rotation 0|90|180|270` (`--rotate` is an alias) rotates the source clockwise.
`--mirror` mirrors the rotated source horizontally. Geometry metadata follows
the transform. Source and encoder previews use the transformed picture.

The sender GUI separately exposes **Luma target · linear light**
(`--luma-adjust-linear`) and **Pre-shrink · linear light**. These are independent
stages; the former is enabled only with Direct DCT luma adjustment.

## WAV export and live recording

`send --write out.wav` exports a local video or synthetic test source without
opening an audio device. A test source requires a packet or duration limit.
A video without explicit limits exports one pass. An explicitly limited video
export repeats as needed unless `--source-passes` supplies an earlier stop.
Default export rate is 48 kHz; `--rate` overrides it.

Offline video is sampled at modem cadence by FFmpeg, without `-re` or a
wall-clock latest-frame mailbox. The synthetic source uses an injected sample
clock. Embedded sound is read on the same emitted-sample timeline for mono
profiles, retaining the sender's one-packet soundtrack delay and optional trim.
The selected packet/pass limit ends the file; it does not append extra packets
to flush that intentional soundtrack delay. Live audio-device input is not an
offline export source.

`send --record capture.wav` records the logical samples submitted while
sending live camera, screen, video, local-app or test sources. It records the
application waveform, not device-loopback audio or downstream dropouts. The
recording appends across same-rate reconnects. A rate/channel change requires a
new recording rather than mixing clocks in one WAV.

Both operations write streaming 32-bit float WAV through `soundfile`, preserving
the application samples without PCM16 clipping or requantization. Existing
output filenames are replaced on a new run. `--write` and `--record` are
mutually exclusive. Stop/cancel closes the writer and finalizes its header.

## Decode a recorded WAV

```bash
./vi.modem-receive --read recording.wav --headless --save-dir decoded
./vi.modem-receive --read multichannel.wav --channels 3,4 --headless
```

The receiver GUI has **Decode WAV file · blank = live input**, input-pair and
decoded-image-folder controls. WAV decode runs in its background receiver
thread; no input audio device is required and no soundtrack output is opened.
Clear the file field to return to live reception.

PCM and float files are streamed at their recorded sample rate. A lookahead pass
uses the existing multi-packet profile confirmation to establish a startup hint,
then rewinds for normal decoding, including the acquisition pictures. Every
packet still validates its status/profile/signature. An unconfirmed short or
damaged recording can yield no pictures. Profile changes inside a recording
follow the normal receiver confirmation policy.

Decoded PNGs use the existing receiver image-export path. Damaged displayable
pictures remain partial; undecodable packets retain the last good display.
When logging is enabled, completion reports decoded, verified and lost counts.

## Implementation and checks

- `tools/v7_capture.py`: file transport and mouse-follow compatibility.
- `tools/v7_source_audio.py`: matched finite soundtrack readers.
- `tools/v7_live.py`: device sessions, limits, channel selection and file feed.
- `tools/v7_file.py`: sample-clock export, source transforms and file lookahead.
- `tools/v7_wav_io.py`: streaming WAV I/O.
- Sender/receiver GUI modules: controls, preferences and command construction.

Focused device-free checks:

```bash
.venv/bin/python -m unittest modem_tests.test_v7_file_controls \
  modem_tests.test_v7_send_transport modem_tests.test_v7_send_gui \
  modem_tests.test_v7_receiver_gui modem_tests.test_v7_source_audio -v
```

These cover fake-device routing/recording, finite passes, batch-boundary limits,
deterministic exports at 48/96 kHz, WAV decode without device enumeration,
embedded soundtrack export, GUI arguments, and independent source transforms.

### Validation evidence (2026-10-15)

- Focused playback/file/GUI/capture/source-audio checks passed: 311 tests.
- The application integration, lazy-import and ASCII suites passed: 26 tests.
- Full package-aware modem discovery ran 1,061 tests. It reported 118
  analog-frame subtest errors from missing `matched_viewer` profile defaults,
  plus two stereo-slices failures (validation-message expectation and a
  reconstruction-error threshold). Both failure signatures reproduce from
  untouched HEAD in a separate baseline tree with the same local face corpus.
- A two-second local video with embedded sound played once and exited normally
  through PortAudio `pulse` into the existing PulseAudio `loop` null sink,
  recording a logical-channel float WAV at 48 kHz. Routing was refreshed first:
  defaults were `loop` / `loop.monitor`, both exposing a stereo 48-kHz route.
- A three-packet offline WAV decoded as three verified pictures, zero lost,
  and exported the recovered PNGs through `--save-dir`.
- Sender and receiver setup canvases were rendered and visually inspected.
  Logs, recordings and canvas images are local artifacts under `tmp/`.
