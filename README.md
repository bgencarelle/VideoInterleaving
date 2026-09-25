# VideoInterleaving

VideoInterleaving renders ordered image sequences as a local window, browser
stream, terminal video, oscilloscope signal, or audio modem. Each mode runs as
its own application process.

## Modes at a glance

| Mode | Output | Default address or device |
| --- | --- | --- |
| `local` | Full-screen desktop window | Monitor: `http://localhost:8888/` |
| `web` | Browser MJPEG viewer | Viewer: `http://localhost:8080/`; monitor: `http://localhost:1978/` |
| `ascii` | ANSI color video over Telnet | Telnet: port `2323`; monitor: port `2324` |
| `asciiweb` | Browser ASCII viewer over WebSocket | Viewer: `http://localhost:1980/ascii`; WebSocket: port `2424` |
| `scope` | Stereo XY waveform | Audio output; monitor: port `8890` |
| `modem` | V7 pulse-framed audio transport | Selected audio output; no web server |

`local` is the default when `--mode` is omitted. The old `SERVER_MODE` and
`ASCII_MODE` settings do not select the application mode; use `--mode` instead.
Run `main.py --help` for the complete list of accepted options.

## Quick start

### Install

On Debian/Ubuntu/Raspberry Pi OS, the setup script installs system and Python
dependencies, creates `.venv`, checks the modem bindings, and configures
systemd units when systemd is available:

```bash
git clone https://github.com/bgencarelle/VideoInterleaving.git
cd VideoInterleaving
sudo ./scripts/setup_app.sh
```

For a manual Python environment, use Python 3.11 or newer. The virtual
environment uses `--system-site-packages` to reuse compatible OS-provided Python
packages, while `sounddevice` needs the native PortAudio library. On
Debian-family systems, install the listed native dependencies first:

```bash
sudo apt update
sudo apt install $(grep -v '^#' system-requirements.txt | tr '\n' ' ')
```

Then create the environment with `--system-site-packages`:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python utilities/check_modem_setup.py
```

`requirements.txt` includes the modem and scope Python requirements. Native
libraries (including PortAudio, OpenGL, and TurboJPEG) are listed in
`system-requirements.txt` for Debian-family systems. On macOS, run
`./scripts/setup_app.sh` to install the Homebrew dependencies. If setting up
manually, install the native libraries required by your mode and use the
package-manager Python when reusing its Python packages.

### Point the app at an image library

Pass the root of your image library with `--dir`. The app looks for `face/` and
`float/` trees under that root; each contains image folders and ordered frames
used by the composition:

```text
images/
├── face/
│   ├── 0_rest/
│   └── 1_subject/
└── float/
    └── 0_overlay/
```

Without `--dir`, the defaults prefer `images_sbs` when it exists, then `images`.

PNG, WebP, JPEG, and several NumPy image formats are accepted. For faster
JPEG-based playback, an SBS JPEG stores color in its left half and its matte in
its right half. SBS is optional; source images can be used directly. The
optional converter at `utilities/reencode_for_jps_sbs.py` scans WebP files and
needs OpenCV (`cv2`) and `tqdm`; `tqdm` is not installed by the default setup.

### Start a mode

Run from the repository root. Use `.venv/bin/python` so the application uses
the environment installed above:

```bash
# Desktop window (also the default mode)
.venv/bin/python main.py --mode local --dir images

# Browser video
.venv/bin/python main.py --mode web --dir images

# Telnet ASCII video
.venv/bin/python main.py --mode ascii --dir images

# Browser ASCII video
.venv/bin/python main.py --mode asciiweb --dir images
```

Connect to ASCII Telnet with `nc localhost 2323` or `telnet localhost 2323`.
In web mode, open `http://localhost:8080/`; the status monitor is on port
`1978`. Video, Telnet, and WebSocket listeners are network-facing by default.
Monitor endpoints are loopback-only by default; `--test` exposes them on all
interfaces, and local mode enables that setting automatically.

## Common options

```text
--mode {web,ascii,asciiweb,local,scope,modem}
--dir PATH                 Image source root (overrides settings.py)
--port PORT                Port override for ASCII modes
--rotation {0,90,180,270}  Rotate local or scope output
--mirror / --no-mirror     Flip local or scope output horizontally
--help                     Show all options, including scope and modem flags
```

`--port` changes the Telnet port in `ascii` mode. In `asciiweb` mode it sets
the WebSocket base port, which listens on `PORT + 1`. Web, local, and scope
ports are fixed by the mode configuration.

ASCII tone grading is available in both `ascii` and `asciiweb` modes. The
stages apply in this order: contrast, brightness, then gamma:

```bash
.venv/bin/python main.py --mode ascii --dir images \
  --ascii-contrast 1.4 --ascii-brightness 1.2 --ascii-gamma 0.9
```

Contrast `1.0` and brightness `1.0` are neutral; gamma below `1.0` lifts
shadows. See `main.py --help` for validation ranges and the full scope/modem
option lists.

## Scope output

Scope mode needs an XY bake generated from the image library:

```bash
.venv/bin/python utilities/convert_to_xy.py -i images -o images_xy
.venv/bin/python main.py --mode scope --dir images --xy-dir images_xy \
  --scope-mode raster --device "Your audio output"
```

Stereo output sends X on the left channel and Y on the right. For a
single-input Y-T scope, connect the X output; the X trigger marker is enabled
by default. Use `--device null` for the virtual/browser-rendered path without
opening an audio device. See [`docs/SCOPE_MODE.md`](docs/SCOPE_MODE.md) for
renderer, connection, and sample-budget details.

## V7 modem output

Bake image layers once, then select the V7 modem mode. The output device must
provide the two channels named by `--modem-channels` (a one-based pair):

```bash
.venv/bin/python utilities/convert_to_modem_dct.py \
  --input-dir images --output-dir images_modem

.venv/bin/python main.py --mode modem --modem-dir images_modem \
  --device "Your audio output" --modem-channels 1,2 --modem-pair 0,0
```

`--modem-pair` selects fixed, zero-based face and float folder indices. Omit it
for the live folder selector; that selector needs a rest face folder plus at
least one other face folder. For a finite WAV export, add `--modem-wav PATH` and
`--modem-frames N`; WAV export requires the free-running clock and does not open
an audio device. The transport is described in
[`docs/transport_v7_spec.md`](docs/transport_v7_spec.md). Synthetic tests do not
establish real tape/deck or telephone-band reliability.

### Run the standalone V7 live sender and receiver

For a direct camera, screen, or video-file link, use the standalone V7 tools
instead of `main.py --mode modem` (which sends the baked image library above).
Open two terminals from the repository root and start the receiver first:

```bash
# Terminal 1: audio input and display
./vi.modem-receive --device "BlackHole 2ch"

# Terminal 2: audio output; prompts for a capture source in an interactive shell
./vi.modem-send --device "BlackHole 2ch"
```

For a non-interactive send, specify the source. `screen` is a simple live
capture; for a clip, use `--source video --video-source PATH`:

```bash
./vi.modem-send --device "BlackHole 2ch" --source screen --encode-filter box
# or
./vi.modem-send --device "BlackHole 2ch" --source video \
  --video-source clip.mp4 --encode-filter box
```

`--device` is required on both commands. Route the sender's output into the
receiver's input; device names can differ on separate machines. List PortAudio
devices with `.venv/bin/python -m sounddevice`. Press Ctrl-C in each terminal to
stop it. `box` is the recommended encode filter. For the experimental fold,
start both ends with the same setting:

```bash
./vi.modem-receive --device "BlackHole 2ch" --experimental-fold 500
./vi.modem-send --device "BlackHole 2ch" --source screen \
  --encode-filter box --experimental-fold 500
```

Use `1000` instead on both ends to select the larger fold. See
[`test_modem_v7/HOWTO.md`](test_modem_v7/HOWTO.md) for prototype details.

## Runtime and development notes

- The normal image clock is free-running. MTC/MIDI/LTC synchronization is not
  active for ordinary local, web, ASCII/ASCII-WebSocket, or scope playback.
- `scripts/run_app.sh` re-anchors to the repository root and restarts the app
  after it exits. With no arguments it starts local mode.
- On Linux, `setup_app.sh` creates systemd units for local, web, ASCII, and
  ASCII-WebSocket modes. Scope and modem are launched with `main.py`.
- Runtime logs are written under `logs/`.
- Run the V7 synthetic/unit tests with:

  ```bash
  .venv/bin/python -m unittest discover -s modem_tests -p 'test_v7_*.py' -v
  ```

## Repository map

| Path | Purpose |
| --- | --- |
| `main.py` | CLI configuration and mode dispatch |
| `animation_modem/` | Standalone modem transport package |
| `modem_v7_display.py` | Application V7 modem sender |
| `scope_display.py`, `scope_out.py` | Scope mode and audio output |
| `utilities/` | Bakers and setup diagnostics |
| `modem_tests/`, `tests/` | Modem and application test suites |
| `docs/` | Mode guides and transport specifications |
| `scripts/` | Setup, launcher, and kiosk scripts |
