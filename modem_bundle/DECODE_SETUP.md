# Decoder-only dependencies

These requirements apply to the preserved `source/modem_receive.py` entry point.
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

## Live receiver

From `modem_bundle/`, using your chosen Python environment:

```bash
python -m pip install -r requirements.txt
python check_decode_setup.py
python source/modem_receive.py --receiver reference --device YOUR_DEVICE --channels YOUR_CHANNELS
```

Replace device/channel placeholders with your existing working selection.
For headless reception, pass `--headless` to both the check and receiver.

## WAV-only receiver

```bash
python -m pip install -r requirements-decode-core.txt
python check_decode_setup.py --wav --headless
python source/modem_receive.py --receiver reference --wav recording.wav --headless --fast
```

For a WAV display, omit `--headless` and provide Tk support.

## OS dependencies

- Debian-family systems: `libportaudio2` for live input, `python3-tk` for the GUI.
  When using distro Pillow, its Tk bridge is supplied by `python3-pil.imagetk`.
- On macOS, Python, Tk and Pillow's Tk bridge must be compatible with the same
  interpreter. PortAudio is needed for live input. Neither Homebrew nor MacPorts
  is an application runtime dependency. An older-Mac installer is separate work;
  do not use the archived full-application installer for this decoder setup.
- A visible GUI requires an available desktop/display; the import check does
  not create a window and cannot verify that desktop access works.

The decoder does not require SDL2, OpenGL/GLFW/ModernGL, OpenCV, PyTurboJPEG,
MIDI/mido, web servers, FFmpeg/Rubber Band, baked image libraries or transmitter
settings. Receiving plausible shared-time timestamps still assumes an already
synchronized system clock. Keep the existing chrony/system time setup; this
change neither installs nor reconfigures time synchronization.

The checker verifies imports only. It does not query devices, open audio streams,
probe sample rates, test capture permissions or run decoding benchmarks.
The reference decoder here remains the archived, unpatched baseline; these
requirements do not apply the separately retained speed-fix patch.
