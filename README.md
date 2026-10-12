# VideoInterleaving

VideoInterleaving plays a folder of pictures as moving images. It can show
them in a window, in a web browser, as text in a terminal, on an oscilloscope,
or send them as sound that another computer turns back into pictures (the
modem).

## Modes at a glance

| Mode | What you get | Where to look |
| --- | --- | --- |
| `local` | A full-screen window | On the screen; status page at `http://localhost:8888/` |
| `web` | Video in a web browser | `http://localhost:8080/` |
| `ascii` | Video made of text, over Telnet | `nc localhost 2323` |
| `asciiweb` | Video made of text, in a web browser | `http://localhost:1980/ascii` |
| `scope` | Pictures drawn on an oscilloscope | Your audio output, wired to the scope |
| `modem` | Pictures sent as sound | Your audio output, wired to a receiver |

If you leave out `--mode`, you get `local`.

## Install

On Debian, Ubuntu or Raspberry Pi OS, one script installs everything:

```bash
git clone https://github.com/bgencarelle/VideoInterleaving.git
cd VideoInterleaving
sudo ./scripts/setup_app.sh
```

On a Mac, run `./scripts/setup_app.sh` (it uses Homebrew).

To set it up by hand instead, you need Python 3.11 or newer:

```bash
sudo apt update
sudo apt install $(grep -v '^#' system-requirements.txt | tr '\n' ' ')
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python utilities/check_modem_setup.py
```

The last line checks that sound input and output work.

## Your pictures

Put your pictures in a folder with two parts, `face` and `float`. Each holds
folders of numbered frames. A `float` picture is laid over a `face` picture.

```text
images/
├── face/
│   ├── 0_rest/
│   └── 1_subject/
└── float/
    └── 0_overlay/
```

Point the app at it with `--dir images`. Use WebP or JPEG pictures.

## Show pictures

Run these from the repository folder:

```bash
# A window on this screen
.venv/bin/python main.py --mode local --dir images

# A web browser (open http://localhost:8080/)
.venv/bin/python main.py --mode web --dir images

# Text video over Telnet (connect with: nc localhost 2323)
.venv/bin/python main.py --mode ascii --dir images

# Text video in a web browser (open http://localhost:1980/ascii)
.venv/bin/python main.py --mode asciiweb --dir images
```

Useful options:

```text
--dir PATH                 where your pictures are
--rotation 0|90|180|270    turn the picture
--mirror                   flip the picture left to right
--help                     every option
```

For the text modes, you can change the look with `--ascii-contrast`,
`--ascii-brightness` and `--ascii-gamma` (1.0 means no change).

## Draw on an oscilloscope

The easiest way is the scope window:

```bash
./vi.scope-gui
```

Pick your pictures or a video, pick the audio output wired to the scope, and
press Start. The left channel moves the dot sideways and the right channel
moves it up and down.

From the command line:

```bash
.venv/bin/python utilities/convert_to_xy.py -i images -o images_xy
.venv/bin/python main.py --mode scope --dir images --xy-dir images_xy \
  --device "Your audio output"
```

More in [`docs/SCOPE_UPGRADE.md`](docs/SCOPE_UPGRADE.md).

## Send pictures as sound (the modem)

The modem turns pictures into sound on two audio channels. Play that sound
into another computer's audio input (by cable, tape or recorder), and the
receiver turns it back into pictures.

### 1. Start the receiver

On the computer that will show the pictures:

```bash
./vi.modem-receive-gui
```

Choose the audio input in Setup and press Start. The receiver works out on
its own how the sound was sent, so there is nothing else to set. It also
lines up the left and right channels if one arrives a little late.

Keys in the receiver window: `P` shows only the picture, `F` goes full
screen, `I` shows details, `Esc` goes back.

Without the window:

```bash
./vi.modem-receive --device "Your audio input"
```

### 2. Start a sender

There are three ways to send. With no extra options, all three send the same
way, using the best settings found so far.

**The sender window**, for a camera, the screen, or a video file:

```bash
./vi.modem-send-gui
```

Choose the audio output, the source and press Start. Drop a video file on the
window to send it.

**The sender from the command line:**

```bash
./vi.modem-send --device "Your audio output" --source screen
./vi.modem-send --device "Your audio output" --source video --video-source clip.mp4
./vi.modem-send --device "Your audio output" --source camera
```

**Your picture folder**, from the main app:

```bash
.venv/bin/python main.py --mode modem --modem-source images --dir images \
  --device "Your audio output"
```

Add `--modem-pair 0,0` to send one fixed face and float folder. Add
`--modem-wav out.wav --modem-frames 100` to save the sound to a file instead
of playing it.

To list your audio devices:

```bash
.venv/bin/python -m sounddevice
```

### Ways to send

The default sends the picture on both channels. You can pick another way with
`--profile` (in the sender window, it is the Wire profile setting):

- `aspect-fold-500`: both channels, the default
- `aspect-mono-500`: picture on one channel, sound on the other
- `stereo-nested`, `aspect-mono-nested`: more detail, still being tested
- `adaptive-stereo-nested`, `adaptive-mono-nested`: more detail, still being
  tested

**Pixel encode** sends hard square pixels instead of a smooth picture. It is
good for pixel art. Turn it on in the sender window, or add `--pixel-encode`.

### Where the default settings live

The best settings found so far are kept in one file,
`tools/v7_send_defaults.py`. The sender window, the command-line sender and
`main.py` all start from it, so a change there changes all three.

### More reading

- [`docs/MODEM_MODE.md`](docs/MODEM_MODE.md): the modem in detail
- [`docs/transport_v7_spec.md`](docs/transport_v7_spec.md): how the sound is
  built
- [`docs/V7_FIXES.md`](docs/V7_FIXES.md): known problems and what is being
  worked on

## Tests

```bash
.venv/bin/python -m unittest discover -s modem_tests -p 'test_v7_*.py'
.venv/bin/python -m unittest discover -s tests
```

## What is where

| Path | What it is |
| --- | --- |
| `main.py` | Starts the app in the mode you choose |
| `vi.*` | Short commands for the modem and scope windows |
| `tools/` | The modem sender and receiver, and other tools |
| `animation_modem/` | How pictures become sound and back |
| `utilities/` | Picture converters and the setup check |
| `docs/` | Guides and details |
| `scripts/` | Install and start-up scripts |
| `modem_tests/`, `tests/` | Tests |

Logs go to `logs/`.
