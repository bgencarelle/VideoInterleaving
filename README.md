# VideoInterleaving

**VideoInterleaving** is a timecode-synced image sequence renderer designed for high-performance animation-based installations. It supports dual-layer blending, MIDI and MTC synchronization, client-server WebSocket sync, and real-time OpenGL rendering. Originally built for interactive, installation-based workflows, the system emphasizes frame accuracy, low-latency playback, and folder-based control over assets.

---

## Setup

```bash
# Clone the repository
git clone https://github.com/your-username/VideoInterleaving.git
cd VideoInterleaving

# Run the bootstrap script (checks Python, installs venv + deps)
bash bootstrap.sh
```

This will:

- Check Python version
- Create a virtual environment at `$HOME/PyIntervenv`
- Install all dependencies
- Generate a `runPortrait.sh` launcher in your home directory

### Preparing Image Sequences

Organize images into folders:

- `images_1080/float/` → Main layer
- `images_1080/face/` → Float layer (must be named with prefix `255_`)

Run:

```bash
python make_file_lists.py
```

This will scan the folders, verify image consistency, interleave sequences, and generate CSVs.

### Launching the Player

```bash
bash launchInterleavingScript.sh
```

Or manually:

```bash
source $HOME/PyIntervenv/bin/activate
python main.py
```

---

## Clock Modes

| Mode Name     | Constant | Source            | Behavior                                                                  |
| ------------- | -------- | ----------------- | ------------------------------------------------------------------------- |
| `FREE_CLOCK`  | `255`    | System time       | No external sync. Time drives index. Ideal for random pairing and testing |
| `MIDI_CLOCK`  | `1`      | MIDI clock pulses | Syncs playback tempo to incoming BPM via 24ppqn MIDI clock                |
| `MTC_CLOCK`   | `0`      | MIDI timecode     | Follows absolute SMPTE-style position. Frame-accurate.                    |
| `MIXED_CLOCK` | `2`      | MTC + MIDI clock  | Combines tempo and time for position-aware sync (currently broken)        |
| `CLIENT_MODE` | `3`      | WebSocket         | Follows remote index from another machine. Lightweight network sync.      |

Mode is selected via `settings.py → CLOCK_MODE` or interactively at runtime. `make_file_lists.py` ensures sequence folders match and prepares frame data accordingly.

---

## Overview

### Features

- **Frame-accurate animation playback** using image sequences (`.png`, `.webp`) with no decode delay
- **Dual-folder blending** with alpha-aware float layer
- **Multiple time modes** (see above)
- **FIFO image buffering** with async preloading
- **OpenGL rendering** with aspect ratio, rotation, and mirror control
- **Real-time folder switching** and ping-pong index logic
- **HTTP monitoring dashboard (optional)**
- **Custom bootstrapping and virtual environment setup**

### Project Structure

- `image_display.py` – main loop and render logic
- `renderer.py` – OpenGL shaders and drawing functions
- `midi_control.py` – MIDI/MTC parsing and clock tracking
- `index_calculator.py` – time-to-index logic
- `index_client.py / index_server.py` – network sync layer
- `calculators.py` – index scaling, frame duration presets
- `make_file_lists.py` – image folder scanning and CSV generation

---

## Author

- Ben Gencarelle

---

## Related Tools

A suite of auxiliary scripts supports the animation preparation pipeline, ranging from white balance correction and matte application to benchmarking and compositing:

| Tool                       | Purpose                                                                                                                                         |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `MakeAmatte.py`            | Applies matte alpha to images and restores blacked-out areas using `turntoblack.py`. Essential for combining transparency with visual recovery. |
| `whiteBalance.py`          | Performs static white balance correction using a single reference frame (usually frame 30). Fast and consistent for well-lit sequences.         |
| `mattePlusWhiteBalance.py` | Combines white balancing with matte alpha application. Uses dynamic recalibration (EMA) across frames. Best for long-form adaptive corrections. |
| `turntoblack.py`           | Reconstructs visual content lost in fully transparent areas by compositing over black.                                                          |
| `overlay.py`               | Batch overlays two image sets in both directions. Useful for compositional studies and variant generation.                                      |
| `batch.py`                 | WebP compression pipeline using `cwebp`. Preserves ICC profiles and flags missing metadata.                                                     |
| `background_removal.py`    | Applies contrast enhancement and then removes background using `rembg`. Uses persistent session for consistent results.                         |
| `test_alpha.py`            | Emulates OpenGL-style alpha blending in Python. Helps preview composite behavior without full render.                                           |
| `webpTester.py`            | Benchmarks various image loading libraries for `.webp` performance. Includes OpenGL preview renderer.                                           |
| `mergeImagesToMovie.py`    | Composites paired images from CSV listings into flattened PNGs, using multiprocessing and alpha-preserving logic.                              |
| `csv_list_maker.py`        | Generates randomized or ping-pong index CSVs for dual-image sequencing with folder switching logic.                                             |
| `stripPixels.py`           | Cleans up residual alpha line artifacts from `.webp` sequences. Preserves valid transparency and boosts visual clarity.                        |
| `merge_2_images.py`        | Merges RGB channels from one image with the alpha channel of another, generating hybrid `.webp` output.                                        |
| `image_resize.py`          | Recursively resizes folder trees of images while preserving aspect ratio and sRGB color profiles.                                              |
| `find_missing.py`          | Scans image sequences for gaps based on numeric filename patterns and reports missing frames.                                                  |

White balance options:

- Use `whiteBalance.py` for speed, reproducibility, and stable lighting.
- Use `mattePlusWhiteBalance.py` when smooth temporal correction across frames is needed.

## License

This project is released under the MIT License.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

---

## Acknowledgments

This project originated in the context of real-time animation-based installation work, where minimal latency, predictable behavior, and image-folder-based sequencing were essential. It favors explicit file structures and low-dependency logic over opaque media pipelines.

Development began in early 2023 and involved extensive use of AI-assisted tools, including early and experimental versions of ChatGPT, alongside a range of open-source libraries.

