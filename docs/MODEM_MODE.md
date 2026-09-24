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
modem import is missing. System packages include SciPy and Pillow's Tk bridge
on Debian, Tk bindings on every supported platform, and PortAudio.

For an existing checkout, rerun your usual `scripts/setup_app.sh` command. To verify
without changing the environment or touching audio devices:

```bash
.venv/bin/python utilities/check_modem_setup.py
```

For standalone baking/receiving tools, install the OS Tk/PortAudio dependencies
first, then use a virtual environment:

```bash
python -m pip install -r requirements-modem.txt
python utilities/check_modem_setup.py
```

Debian/Ubuntu packages: `python3-scipy python3-tk python3-pil.imagetk libportaudio2`.
Fedora uses `python3-tkinter portaudio`; Arch uses `tk portaudio`. Homebrew uses
`python-tk portaudio`; Tk must match the Python version used for the venv.
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

## Run live output

List PortAudio devices using the receiver wrapper:

```bash
python utilities/modem_v3_check.py live-receive --list-devices
```

Start the receiver on the input side, then the modem mode on the output side.
BlackHole or a stereo cable can connect the two:

```bash
python utilities/modem_v3_check.py live-receive --device BlackHole --channels 1,2 --emulate clean
python main.py --mode modem --modem-dir images_modem \
  --device BlackHole --modem-channels 1,2
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
index formula. `--modem-prepare-ms` defaults to 10 and grows when measured image
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
