# Decoder-only dependencies

These requirements apply to the bundle-level `modem_receive.py` entry point.
They do not install or launch the transmitter, baker, full application or services.
The frozen `source/` tree and `SNAPSHOT.json` remain unchanged.

| Use | Python packages | Native dependencies |
| --- | --- | --- |
| Live audio with display | NumPy, SciPy, Pillow, sounddevice | PortAudio, Python Tk bindings, Pillow Tk bridge |
| Live audio, headless | NumPy, SciPy, Pillow, sounddevice | PortAudio |
| WAV with display | NumPy, SciPy, Pillow | Python Tk bindings, Pillow Tk bridge |
| WAV, headless | NumPy, SciPy, Pillow | No Tk or PortAudio |

Pillow remains necessary for the current headless entry point because its image
module is imported at startup. Removing that dependency would require a later
code change. NumPy/SciPy/Pillow themselves also need platform-compatible wheels
or compiled installations; this list does not guarantee old-macOS wheel support.

## Install

From `modem_bundle/`, run `./setup_decode.command` on macOS/Linux or
`setup_decode.cmd` on Windows. Double-clicking the launchers also works where
file associations permit it. They find Python 3.11+, show failures, and wait
before closing. If Python is missing, they show installation instructions.
Set `MODEM_PYTHON` to an executable path to select a specific interpreter; it
must not contain extra command arguments.

Alternatively, from an open terminal:

```bash
python3 setup_decode.py --install
```

Windows: `py -3 setup_decode.py --install`.
For WAV-only/headless use add `--wav --headless`. Without `--install`, the Python
helper checks the existing bundle environment. Launchers default to `--install`
only when no arguments are given; otherwise they forward your arguments.

The installer creates `modem_bundle/.venv` and runs pip and the import checker
using that environment's Python. It leaves pip errors visible and returns
nonzero on failure. It never deletes/replaces an existing environment belonging
to another interpreter: select its original Python or rename `.venv` and retry.
An incomplete environment after a failed creation likewise needs renaming.
Native packages installed for the selected base Python remain visible through
`--system-site-packages`; pip installs go into the decoder environment.

For unattended use pass `--non-interactive`. The launchers also accept
`MODEM_NO_PAUSE=1` to suppress their closing pause. Native package installation
requires a separate interactive confirmation; unattended setup prints the
command and fails rather than installing native packages without consent.

## Native packages and package managers

`--install --native` offers native package installation on macOS, then creates
or checks the environment and installs Python requirements. It prints the
command and asks before invoking the package manager (and `sudo` for MacPorts).
It requires a Python owned by the detected manager; it does not mix an unrelated
Python installation with MacPorts or Homebrew extensions.

For missing managers, setup can open official installation instructions after
a yes/no prompt. Request that assistance directly with:

```bash
python3 setup_decode.py --package-manager
```

You complete the package-manager installer yourself, reopen the terminal, and
rerun setup. Neither manager is required if your dependencies already work.

### Older Macs / MacPorts

[MacPorts provides a Sierra installer](https://www.macports.org/install.php).
Use the installer for your exact macOS and its matching Apple developer tools.
Individual ports may require compilation or fail on older systems; this does
not guarantee Sierra compatibility for every current package version.

After installing MacPorts, install its Python if needed:

```bash
sudo port install python312
MODEM_PYTHON=/opt/local/bin/python3.12 ./setup_decode.command --install --native
```

For Python 3.12, native setup installs MacPorts `py312-numpy`, `py312-scipy`,
`py312-Pillow`, plus `py312-tkinter` for display and `portaudio` for live input.
Other selected Python versions use their matching port names. The matching
scientific packages can avoid pip attempting unsuitable source builds.
MacPorts Python/Tk extensions cannot be used by an unrelated Python 3.12.

For Homebrew Python, native setup requests the matching versioned
`python-tk@X.Y` and PortAudio as needed. Scientific packages come from pip.
The root `setup_app.sh` remains the full-app Homebrew installer on macOS;
`integration_reference/` remains a frozen archive.

### Windows

Install Python with pip and Tcl/Tk enabled, then run `setup_decode.cmd`.
If Tk imports fail, modify/repair that same Python installation to include
Tcl/Tk. The helper does not replace Python automatically.
[WinGet / App Installer](https://learn.microsoft.com/en-us/windows/package-manager/winget/)
is optional and requires Windows 10 version 1809 or later.
[Sounddevice wheels normally bundle PortAudio on Windows and macOS](https://python-sounddevice.readthedocs.io/en/latest/installation.html).
There is no requirement to install a separate Windows PortAudio package.
`--native` on Windows/Linux prints native dependency guidance; automatic native
package installation is currently limited to MacPorts/Homebrew.

## Receive

A successful setup prints a command using the bundle environment. Replace
`YOUR_DEVICE` and `YOUR_CHANNELS` with your existing working selection. Setup
does not discover, choose, or change devices/channels/sample rates.

macOS/Linux examples:

```bash
.venv/bin/python modem_receive.py --device YOUR_DEVICE --channels YOUR_CHANNELS
.venv/bin/python modem_receive.py --wav recording.wav --headless --fast
```

Windows uses `.venv\Scripts\python.exe` instead. Add `--headless` for live
headless reception. Receiver flags are separate from setup flags. These commands select the original modem 2 packet receiver, matching the utility
with `recovery=False, fast=True`. Progressive/reference are optional via
`--receiver progressive` or `--receiver reference`.

## OS dependencies

- Debian-family systems: `libportaudio2` for live input, `python3-tk` for the GUI.
  When using distro Pillow, its Tk bridge is supplied by `python3-pil.imagetk`.
- On macOS, Python, Tk and Pillow's Tk bridge must be compatible with the same
  interpreter. PortAudio is needed for live input. Neither Homebrew nor MacPorts
  is an application runtime dependency. Use the setup assistance above; do not
  use the archived full-application installer for this decoder setup.
- A visible GUI requires an available desktop/display; the import check does
  not create a window and cannot verify that desktop access works.

The decoder does not require SDL2, OpenGL/GLFW/ModernGL, OpenCV, PyTurboJPEG,
MIDI/mido, web servers, FFmpeg/Rubber Band, baked image libraries or transmitter
settings. Receiving plausible shared-time timestamps still assumes an already
synchronized system clock. Keep the existing chrony/system time setup; this
change neither installs nor reconfigures time synchronization.

The checker verifies imports only. It does not query devices, open audio streams,
probe sample rates, test capture permissions or run decoding benchmarks.
The archived source remains unchanged; the separately retained speed-fix patch
is not applied.
