# Modem mode

The `modem` branch adds an output mode that reuses VideoInterleaving's baked
face and float layers, composites one complete RGBA image at a time, and sends
it through the frame-independent stereo audio modem. The existing scope,
local, web, and ASCII modes are left on their existing paths.

## Tape direction and defaults

The target medium is analog tape: noisy, warbly, band-limited. The long-term
goal is one tape-compatible bandwidth carrying the baked 80x96 resolution with
cheap, deterministic (classical-computing) decode. Timing sync is
pulse-counted, LTC-style (`measure_pulses`, `pulse_only`), not FFT-correlated.
Recovery priority on bad tape: the image must still DECODE with acceptable
color; losing detail/softness is fine. The V4 goal is a wire whose whole band
fits inside the tape bandwidth -- the picture rides fully within the medium
instead of relying on graceful degradation when the band exceeds it.
The `V3_PRESETS` table is interim — the most tape-compatible (preset, profile)
pair becomes the default, then the sprawl gets refactored away. The chosen
default is `wide-v3` + `color-dct`: the full 2880-slot picture at 375-20250 Hz,
14.35 fps. It was picked over the 15% faster `hires-v3` because on the bench's
`--noise-dbfs` sweep it verified 24/24 at -35 dBFS where hires-v3 lost 5, and
over the `lean-*` presets because those shrink the picture. Preset/profile
choices are validated by SYNTHETIC IDEAL round trips in `tools/bench_modem.py`
(decode cost + fidelity on a clean wire, with cheap `--band-limit-hz` and
`--noise-dbfs` recovery-margin probes); tape emulation is kept to a minimum
because it cannot be made realistic — real tape measurements are done on
hardware, outside the simulations. See AGENTS.md "Long-term direction".

## Install modem dependencies

Normal `scripts/setup_app.sh` setup now includes modem dependencies through
`requirements.txt` -> `requirements-modem.txt`, and verifies them in `.venv`.
An existing environment is updated when either requirements file changes or a
modem import is missing. PortAudio is a native system dependency; Python
dependencies are listed in `requirements-modem.txt`.

For an existing checkout, rerun your usual `scripts/setup_app.sh` command. To verify
without changing the environment or touching audio devices:

```bash
.venv/bin/python utilities/check_modem_setup.py
```

For standalone baking/receiving tools, install the native PortAudio dependency
first, then use a virtual environment:

```bash
python -m pip install -r requirements-modem.txt
python utilities/check_modem_setup.py
```

Debian/Ubuntu: `libportaudio2`. Fedora, Arch, Homebrew, and MacPorts provide
PortAudio as `portaudio`.
Full `main.py --mode modem` transmission still uses the normal application
requirements, which are installed by setup.

FFmpeg with the `rubberband` filter is optional and only needed to generate the
pitch-probe experiment files (`utilities/modem_pitch_probe.py make`). It is not
required for baking, live transmission, or reception. Setup does not change
working audio device/channel selections or clock synchronization.

## Prepare a modem bake

The modem bake is a compact, memory-mapped RGBA slab for each numeric face and
float folder. It mirrors the source tree and records the source order in
`modem.json`.

```bash
python utilities/convert_to_modem_dct.py \
  --input-dir images \
  --output-dir images_modem
```

The 2D-DCT baker always downsamples to an 80x96 RGBA canvas and writes
`profile: color-dct` in `modem.json`; the decoder identifies the profile from
the transmitted frame. Use `--jpeg-layout
rgb` when JPEGs are ordinary RGB images. The default `sbs` follows the project
convention used by `utilities/convert_to_xy.py`: the left half is colour and
the right half is a matte. PNG/WebP alpha is read directly.

Each folder entry also records every source frame's dimensions. The slab stays
80x96 for encoding, while the original dimensions supply the aspect metadata
sent to the receiver (including quarter-turn rotation). Older bake manifests
without these dimensions remain readable, but they can only report the baked
80x96 aspect; rebake the source tree to recover its original aspect ratio.

The bake refuses to overwrite an existing output directory. Bake to a new
directory after source changes. It writes temporary slabs beside the final
tree and publishes the tree only after each file and the manifest complete.
The existing `utilities/bake_assets.py` writer is shared by both bakers.

## Run a fixed-pair WAV test

Use a fixed zero-based face,float folder pair to test output without an audio
device or live folder selection:

```bash
python main.py --mode modem \
  --modem-dir images_modem \
  --modem-pair 1,0 \
  --modem-wav modem_test.wav \
  --modem-frames 100
python utilities/modem_v3_check.py read --wav modem_test.wav
```

The WAV export samples the application clock at `settings.IPS` while packets
are written, so it can skip source indices when the wire packet rate is lower
than IPS and follows the configured ping-pong direction. With
`--modem-frames 0`, it writes one forward source pass at that IPS rate. Use
`--modem-cycles N` instead for complete clock cycles; with ping-pong enabled a
cycle includes the forward and return trips, while a one-way loop cycles once
through the source sequence. The two length options are mutually exclusive.
The WAV path does not invoke the project's stochastic folder selector.

## Standalone V7 live sender and receiver (experimental)

For direct camera, screen, or video-file transmission, use the standalone
`vi.modem-send` and `vi.modem-receive` wrappers. This path is separate from
`main.py --mode modem`. The application path defaults to the baked image slabs,
but `--modem-source images` loads ordinary source images and encodes them at
runtime. Both paths use V7 pulse framing; the standalone live tools default to
the pinned M=500 fold with coded pilot status, while that fold is not the
application mode's default. From the repository root, start the standalone
receiver first. The sender selects the matching box profile and the receiver
despreads the chips before tone-assisted timing.

```bash
# Terminal 1
./vi.modem-receive --device "BlackHole 2ch"

# Terminal 2; interactively asks which capture source to use
./vi.modem-send --device "BlackHole 2ch"
```

For non-interactive runs, pass a source explicitly. No additional fold option is
needed for the default M=500 coded profile:

```bash
./vi.modem-send --device "BlackHole 2ch" --source screen
./vi.modem-send --device "BlackHole 2ch" --source video \
  --video-source clip.mp4
```

To configure the standalone sender with a small event-driven GUI instead, run
`./vi.modem-send-gui`. It launches the same `vi.modem-send` path in a separate
process and stops it with a graceful interrupt. It plays at the rate the OS has
the output device set to (shown on the live page); change that rate in the OS
audio settings. The CLI keeps `--rate` for scripted use. Its **Change source** action
stops the current send and opens the source picker; select the next source and
press **Start**. Source-specific settings are retained while switching. The GUI
restores the last configuration on launch but does not auto-start the sender.
For video, use **Browse**, type a file path or URL, or drop a file onto the
window. **Open source in player** opens a separate desktop player; when `ffplay`
is available, finite file/VOD preview requests infinite looping and mutes its
audio. Live sources are not looped. The player has its own playback clock and
is not frame-synchronized with the sender. A system-associated player fallback
may control repeat and audio behavior itself. Camera and screen/display options
are discovered when their pickers open.

The GUI offers four wire profiles: **Aspect Fold 500** (stereo, the default),
**Fold 500** (stereo), and the mono-video **aspect colour Fold 500** and
**colour Fold 500**. The receiver follows the profile carried by each packet.
**Direct DCT encode** is on by default (it encodes from the full-size frame;
frames smaller than the coder grid fall back to the Box resize). Saved
settings from earlier GUI versions keep everything except the profile, Direct
DCT encode and the encoder filter, which start at the new defaults once. Under **Advanced**:

- **Clip-aware encode** (`--clip-aware-encode`, opt-in) re-fits the sent luma
  coefficients so ringing around bright and dark edges falls into the
  receiver's black/white clip, where it is invisible (about 2 ms per frame at
  the sender; no receiver change).
- **Pre-encode downscaler** applies to the resize path only (Direct DCT encode
  off).

The CLI keeps its previous defaults; the other profiles and `--baseline`
remain CLI options.

During a live send or receive, a lost primary audio device pauses transmission
or decoding and reports the selected device; the receiver retains its last good
picture. Recovery targets the same device identity, without switching silently
to the system default. A device/rate pair must remain stable for five 30-fps
intervals (about 167 ms) before reconnecting. The reopened stream's negotiated
sample rate is used to rebuild rate-dependent state. Loss of an optional sender
source-audio input leaves video sending active with silence on that audio leg;
loss of receiver passthrough output does not stop video decoding.

For file/stream video and FFmpeg-based screen capture, **Capture width** is an
intermediate FFmpeg downscale that limits raw-frame pipe bandwidth and CPU. The
sender then prepares the fixed 80×96 V7 image, so those paths currently rescale
twice. It is not the wire resolution. The camera's normal path can prepare
80×96 directly; the opt-in perceptual resizer runs on the intermediate image
(160 pixels wide by default), not the source's full resolution. The GUI help
describes which sources use this setting.

The standalone V7 receiver GUI can be launched with
`.venv/bin/python -m tools.v7_receiver_gui`. Its **Display grain** choice
(off by default) adds fine noise, about 2.5/255, only where the decoded picture
is flat. That breaks up the regular ringing ripple of the band-limited picture
without touching edges or texture. The flat-area mask is computed on the
decoded 96×80 luma grid and the grain in the shader, so it adds no meaningful
decode time; the pattern changes with each new picture. It selects an explicit input and
can optionally pass the non-video channel to an explicitly selected output
device. Passthrough starts at 1.0 volume (VLC's default 100%, unity gain);
adjust its live, persisted 0–1 volume control or mute it as needed. The CLI
exposes the same setting as `--audio-volume`. The GUI also has a live
sync-warning control. The Info view reports the detected profile, video/audio
channel assignment, output state, and sync state; the decoded-image destination
is chosen with **Browse**. Device preferences are matched by device name and
host API when PortAudio indices change. Passthrough plays through a PortAudio
blocking stream: PortAudio's own thread plays from a short output buffer (about
46 ms) and a Python writer refills it, so a decoder or GUI stall spends
buffered audio instead of dropping output blocks. A clock servo matches the
capture and output clocks within 0.5%. For diagnosis,
`V7_AUDIO_OUTPUT_MODE=callback` restores the Python output callback, and
`V7_AUDIO_OUTPUT_LATENCY` / `V7_AUDIO_OUTPUT_BLOCKSIZE` override PortAudio's
buffering; `audio_runtime` reports show measured rates, stalls, trims and
PortAudio underflows. There is no receiver WAV recorder yet;
the proposed recording extension and its overhead gate are documented in
`transport_v7_spec.md`.

The device argument is required on both sides. Route the sender's output to the
receiver's input; on a same-machine loopback, select the loopback device for
both. The sender runs until Ctrl-C (or `--seconds N`); stop the receiver with
Ctrl-C. Use `--baseline` on **both** commands to restore the previous live
profile (nearest resize, brightness 1.05, and steady pilot tones). To select
another experimental table, pass `--experimental-fold 1000` on both commands;
the sender uses box/brightness 1.0 by default. Folded profiles require the
default fixture and the canonical box model. See `test_modem_v7/HOWTO.md` for
fold tables, coded-pilot behavior, loopback instructions, and regression tests.

### Aspect Fold 500 (experimental)

`--profile aspect-fold-500` keeps the V7 wire unchanged (2,320 slots, the same
2,880 coefficients, 500 folded guests, metadata and CRC) but chooses which DCT
coefficients to send from the picture's shape instead of the fixed 5:6
48×40 corner. For a 16:9 frame, luma reaches 66 horizontal and 37 vertical
frequencies instead of 40 and 48. It is signalled with profile code 0
(the former fold-off word, which live senders no longer emit), so a receiver
that follows packet profiles picks it up automatically. It needs pilots and
EOF markers.

- `--aspect-layout` (default `auto`): `auto` follows the source aspect carried
  in each packet's metadata; `1:1`, `4:3`, `3:2`, `16:9`, `3:4`, `2:3` and
  `9:16` fix a layout (a projected-frame shape). A sender with a fixed layout
  pillar- or letterboxes each picture into it and signals the layout in the
  metadata (aspect code plus the screen bit), so the receiver's `auto`
  follows it; forcing a layout on the receiver is only for tests. With
  `auto`, the receiver holds the last good picture until it has valid
  metadata.

One setting is not signalled and must match on both ends:

- `--aspect-tail` (default `chroma`): what the 96 tail slots carry. `chroma`
  is V7's rotating chroma tail (the lowest-ranked 656 coefficients are all
  chroma); `split` sends 48 extra luma frequencies in every packet plus 48
  rotating chroma; `luma` sends 96 extra luma frequencies in every packet and
  drops the 96 weakest chroma ones.

The sender GUI shows both under **Advanced** when the profile is selected; the
receiver GUI shows them under advanced options. Tables are frozen in
`test_modem_v7/aspect_tables.npz` (hash-pinned; rebuild with
`python test_modem_v7/aspect_fold.py build`, then update `TABLES_SHA256`).

`--profile aspect-mono-500` is the mono-video counterpart: mono colour Fold 500
(the same 1,264 M-only slots in every packet on one output leg, chroma ranked
×4, a luma-only 500-slot fold, the other leg free for source audio) over the
same aspect layouts. It is signalled with profile code 3 (the rotating mono
fold-off word, which only the hidden `--experimental-mono` sender emits; the
default receiver now reads code 3 as aspect-mono-500). `--aspect-layout`
applies and must match; there is no tail setting because mono packets carry
no tail.

List available device names with `.venv/bin/python -m sounddevice`. On separate
machines, use the actual output-device name for sending and input-device name
for receiving; they do not need to match.

## Live output from the application

The remaining instructions here are for the application path through
`main.py --mode modem`, not the standalone `vi.modem-*` V7 capture tools above.

This path sends the **stereo Fold 500** V7 profile by default, using the pinned
Box model, coded pilot status, and the EOF marker. To send the previous fold-off
wire, use `--modem-baseline`; that mode defaults to nearest resize and steady
pilot tones. Fold 500 requires the Box model, coded pilot tones, and the EOF
marker, so other `--modem-encode-filter` values and `--no-modem-pilot-tones` or
`--no-modem-eof-marker` are rejected unless baseline mode is selected. The
standalone tools and application sender now use the same stereo Fold 500
profile.

List PortAudio devices with:

```bash
.venv/bin/python -m sounddevice
```

Start the matching Fold 500 receiver on the input side, then the application
sender on the output side. For a same-machine loopback, select the verified
loopback device on both:

```bash
# Terminal 1
./vi.modem-receive --device "BlackHole 2ch"

# Terminal 2
.venv/bin/python main.py --mode modem --modem-dir images_modem \
  --device "BlackHole 2ch" --modem-channels 1,2
```

`--modem-channels` is a one-based pair of physical output channels. The
receiver's `--channels` has the same one-based convention. Use device IDs when
name matching is ambiguous. Only one transmitter should use a channel pair.

The modem mode polls the existing VideoInterleaving clock and calls the
existing stateful face/float folder selector when the image index changes. It
does not run a second selector or scan the source images at runtime. The
largest frame count common to the face and float baked layers is used, matching
the scope manifest rule. `--modem-pair MAIN,FLOAT` bypasses selection for
repeatable inspection. `--modem-frames N` limits a live run; zero means run
until Ctrl-C.

### Shared-time presentation

Like ASCII and scope, the modem samples the existing clock at its output cadence;
it does not change `IPS`, the shared epoch, ping-pong behavior, or Chrony. For the
free clock, it evaluates that same index function at a future presentation time:

1. Reserve the next available audio send time, allowing encoding to finish first.
2. Map PortAudio stream time to shared wall time and add the 63 ms packet duration
   plus a receiver allowance (15 ms by default).
3. Evaluate the image index at that exact target timestamp and encode it into the
   packet alongside the frame/index/count metadata.
4. Start playback at the reserved DAC time. If preparation or the DAC misses the
   deadline, discard the unsent packet and target fresh work; never send a stale
   timestamp deliberately. A completed packet's guard still makes the cadence 15 fps.
5. The receiver holds early frames until their shared timestamp and displays late
   frames as soon as possible, skipping older due frames if needed.

The audio-to-wall-clock mapping is refreshed for every packet, so device drift
and Chrony slew do not accumulate into a separate animation clock. This uses
[sounddevice's documented stream and DAC timestamps](https://python-sounddevice.readthedocs.io/en/latest/api/streams.html).
Scheduling remains at most one pending packet. The separate modem selector
thread and multi-frame FIFO remain removed. The existing folder selector still
runs on current shared time once per source-index change; only image-index lookup
is projected forward. Random folder choices and future MIDI events are not
predicted or synchronized by this transport.

**Update both transmitter and receiver. No image rebake is needed.**
Remove the old `--modem-time-offset-ms 120`: scheduled waiting and transmission
are now accounted for automatically.

```bash
# Receiver machine (GUI; headless reports decode timing only):
python utilities/modem_v3_check.py live-receive --device "BlackHole 2ch" --channels 1,2

# Transmitter machine:
python main.py --mode modem --modem-dir images_modem \
  --device "BlackHole 2ch" --modem-channels 1,2
```

`--modem-latency` defaults to `low`; callbacks contain 256 samples. The existing
`--modem-time-offset-ms` hook now adds EXTRA receiver allowance to the scheduled
target; it is no longer a manually guessed total transmission offset. Prefer
`--modem-receive-margin-ms` to set the allowance directly, e.g. 30 on a slower
receiver. Total allowance must be 0..2000 ms. Increasing it gives the receiver
more time to finish before the chosen shared timestamp; it does not change the
index formula. `--modem-prepare-ms` defaults to 25 and grows when measured image
composition/encoding needs more time. Neither setting guarantees a hardware deadline.

MIDI clocks retain immediate, untimed packets since future input is unknown.
For a live MIDI-clock run, the sender opens the first available MIDI input port;
there must be at least one input. `CLIENT_MODE` is not supported by the V7
sender. Deterministic WAV export requires the free clock because it cannot be
driven by future live MIDI input. The new receiver still accepts old SI01/SI02
headers and displays those frames immediately.

### Timing reports

`--modem-log-frames` adds the shared `target_time_ns` to transmitter records.
Shutdown reports `send_deadline_misses` separately from audio underflows.
The receiver prints:

- `decode_error_ms`: decode completion minus target time. Negative means the
  image arrived early enough to wait; positive means it is already late.
- `display_submit_error_ms`: GUI submission minus target time. The accompanying
  rolling median/p95 makes variation visible. This excludes compositor/monitor
  scanout and is not a measurement of photons appearing on the screen.
- `presentation_frames_skipped`: frames dropped because a newer frame was already
  due or the bounded presentation queue overflowed.

If decode error is regularly positive, increase receiver allowance by the observed
lateness plus a little headroom. This is an explicit adjustment, not a new clock
synchronizer or automatic feedback channel. GUI work, terminal logging, optional
PNG saving and physical display refresh can still add delay. Receiver `--quiet`
disables per-frame JSON when timing diagnostics are no longer needed.

The ST timestamp header is still 24 bytes with CRC32 and repetition on both audio
channels. Width/height/FPS are implied by the version/profile, freeing four bytes
for shared-time milliseconds modulo 2**32. The receiver resolves the nearest epoch
(wrap period about 49.7 days; live time difference must be under 24.85 days).
All three image profiles and the 63 ms packet length are unchanged. Header salvage
never invents a trusted timestamp. Replayed WAV timestamps are not compared to
current shared time and are not used to delay replay.

## Receiver

The decoder lives in `utilities/modem_v3_check.py` under the `read` (WAV file)
and `live-receive` (audio device) subcommands. The audio-path emulation presets
are applied after channel selection and before demodulation:

```bash
python utilities/modem_v3_check.py read --wav wifi_vst_out.wav --emulate mild
python utilities/modem_v3_check.py read --wav wifi_vst_out.wav --emulate rough
python utilities/modem_v3_check.py read --wav wifi_vst_out.wav --lowpass-hz 12000 --noise-dbfs -45
python utilities/modem_v3_check.py live-receive --device BlackHole --emulate mild
```

The `read` subcommand is what Audacity/VST round-trips use: record a VST-processed
transmission, then decode it with the emulation flags as a cross-check of what the
medium is doing on its own.

`animation_modem/` contains the transport package. It remains independent of
the project's image renderer and can be used by another application.

## Dependencies and tests

Install the modem subset with:

```bash
python -m pip install -r requirements-modem.txt
```

Run the existing project tests and the modem tests:

```bash
python -m unittest discover -s modem_tests -v
python -m unittest test_modem_integration test_lazy_imports -v
```

The integration tests use temporary synthetic face/float trees. They verify
folder ordering, alpha compositing, SBS JPEG loading, largest-common frame
selection, atomic bake failure handling, selector call discipline, and a
headless `main.py --mode modem --modem-wav` round trip with GL/audio imports
blocked.

The modem is a separate stereo output path. It does not preserve program audio,
does not survive mono summing, and does not alter the XY scope representation.
