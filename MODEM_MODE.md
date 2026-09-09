# Modem mode

The `modem` branch adds an output mode that reuses VideoInterleaving's baked
face and float layers, composites one complete RGBA image at a time, and sends
it through the frame-independent stereo audio modem. The existing scope,
local, web, and ASCII modes are left on their existing paths.

## Prepare a modem bake

The modem bake is a compact, memory-mapped RGBA slab for each numeric face and
float folder. It mirrors the source tree and records the source order in
`modem.json`.

```bash
python utilities/convert_to_modem.py \
  --input-dir images \
  --output-dir images_modem \
  --profile color
```

`--profile` accepts `color`, `detail`, or `mono`; the decoder identifies the
profile from the transmitted frame. The default is `color`. Use `--jpeg-layout
rgb` when JPEGs are ordinary RGB images. The default `sbs` follows the project
convention used by `utilities/convert_to_xy.py`: the left half is colour and
the right half is a matte. PNG/WebP alpha is read directly.

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
  --modem-frames 100 \
  -f
python modem_receive.py --wav modem_test.wav
```

The `-f`/`--modem-numbered` flag burns the absolute modem frame and source
index into the pixels. The WAV export loops through source indices; it does not
invoke the project's stochastic folder selector.

## Run live output

List PortAudio devices using the receiver wrapper:

```bash
python modem_receive.py --list-devices
```

Start the receiver on the input side, then the modem mode on the output side.
BlackHole or a stereo cable can connect the two:

```bash
python modem_receive.py --device BlackHole --channels 1,2 --emulate clean
python main.py --mode modem --modem-dir images_modem \
  --device BlackHole --modem-channels 1,2 -f
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
python modem_receive.py --device "BlackHole 2ch" --channels 1,2

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
index formula. `--modem-prepare-ms` defaults to 10 and grows when measured image
composition/encoding needs more time. Neither setting guarantees a hardware deadline.

MIDI clocks retain immediate, untimed packets since future input is unknown.
The deterministic WAV inspection export also remains untimed. The new receiver
still accepts old SI01/SI02 headers and displays those frames immediately.

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

`modem_receive.py` is the existing modem decoder packaged as a repository
entry point. It has the same WAV/live, `--headless`, `--fast`, frame-saving, and
audio-emulation options as the standalone modem project. The emulation presets
are applied after channel selection and before demodulation:

```bash
python modem_receive.py --device BlackHole --emulate mild
python modem_receive.py --device BlackHole --emulate rough
python modem_receive.py --device BlackHole --lowpass-hz 12000 --noise-dbfs -45
```

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
