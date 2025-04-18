# VideoInterleaving

VideoInterleaving is a timecode‑synced image‐sequence renderer designed for high‑performance animation installations. It supports dual‑layer blending, MIDI/MTC synchronization, client‐server WebSocket sync, and real‑time OpenGL rendering.

---

## Prerequisites

1. **Python** 3.8+  (check with `python3 --version`).
2. **System libraries**

   **Debian/Ubuntu/Raspbian:**
   ```bash
   sudo apt update
   sudo apt install python3-venv python3-pip libwebp-dev pkg-config libgl1-mesa-dev \
       libxi-dev libxrandr-dev libxcursor-dev libxinerama-dev
   ```

   **Fedora/CentOS:**
   ```bash
   sudo dnf install python3-venv python3-pip libwebp-devel mesa-libGL-devel \
       libXi-devel libXrandr-devel libXcursor-devel libXinerama-devel
   ```

   **macOS (Homebrew):**
   ```bash
   brew install python webp pkg-config
   ```

   **Windows (Chocolatey):**
   ```powershell
   choco install python webp
   ```

3. **Git**:
   ```bash
   git --version
   ```

4. **Optional MIDI**: a functional MIDI interface or USB‑MIDI adapter for `MIDI_CLOCK` or `MTC_CLOCK` modes.

---

## Setup & Running

1. **Clone the repo**
   ```bash
   git clone https://github.com/bgencarelle/VideoInterleaving.git
   cd VideoInterleaving
   ```

2. **Create & activate a Python venv**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate    # Linux/macOS
   .venv\Scripts\activate      # Windows PowerShell
   ```

3. **Install Python dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Prepare your image sequences**

   - Organize your frames into two parallel folder trees:
     ```text
     images/main/<frame_number>/*.webp
     images/float/<frame_number>/*.webp
     ```
   - **Note**: `main.py` auto‑scans the `images/` folders and regenerates `csv` lists on each run. Use:
     ```bash
     python make_file_lists.py
     ```
     only if you add or remove folders manually.

5. **Launch the player**
   ```bash
   source .venv/bin/activate
   python main.py
   ```
   Or use the convenience script:
   ```bash
   bash launchInterleavingScript.sh
   ```

---

## Bootstrap Script

The **`bootstrap.sh`** (in the project root) automates environment setup:

1. Ensures **Python 3.11+** is installed.
2. Checks for **OpenGL** (`glxinfo`) and **SDL2** (`libsdl2`), warning if missing.
3. Resets the Git repo, pulls the latest code, and re‑executes itself if updated.
4. Deletes any existing venv at **`$HOME/PyIntervenv`**, creates a fresh one there, and installs dependencies.
5. Generates **`runPortrait.sh`** in your home directory, containing:
   ```bash
   #!/bin/bash
   source "$HOME/PyIntervenv/bin/activate"
   cd "$PROJECT_DIR"
   python main.py
   ```
6. Marks it executable and immediately runs **`runPortrait.sh`** to launch the player.

---

## Convenience Launchers

- **`runPortrait.sh`** (in `$HOME`):
  Activates the venv, changes to the project directory, and runs `main.py`.

- **`launchInterleavingScript.sh`** (in project root):
  ```bash
  #!/bin/bash
  set -e
  # Change to project dir
  cd "$HOME/VideoInterleaving"
  # Activate venv (expects .PyIntervenv in project root)
  source .PyIntervenv/bin/activate
  # Run the player
  python main.py
  ```

---

## HTTP Monitor (Webserver)

When `TEST_MODE=True` and `HTTP_MONITOR=True` in `settings.py`, a lightweight HTTP server runs on startup:

- **Port:** configured by `WEB_PORT` (default: 8000)
- **Access:** `http://localhost:<WEB_PORT>/`
- **Dashboard:** live FPS, FIFO stats/misses, folder coverage/entropy, CPU/memory usage, error logs.

---

## requirements.txt

Pinned dependencies include:

- **pygame** – multimedia framework
- **psutil** – system monitoring
- **opencv-python** – fallback image I/O
- **PyOpenGL** – OpenGL bindings
- **websockets** – client/server sync

Install with:
```bash
pip install -r requirements.txt
```

---

## Clock Modes

Mode           | Constant | Description
--------------|:--------:|-----------------------------
FREE_CLOCK    |   255    | System‑time driven ping‑pong
MIDI_CLOCK    |     1    | Sync to MIDI clock pulses
MTC_CLOCK     |     0    | SMPTE‑style timecode sync
MIXED_CLOCK   |     2    | Experimental MIDI + MTC
CLIENT_MODE   |     3    | WebSocket index sync

Set via `CLOCK_MODE` in `settings.py` or interactively at launch.

---

## Project Structure

- **main.py** / **image_display.py** – entrypoint, threading, FIFO, monitor hooks
- **renderer.py** – OpenGL shader & draw routines
- **image_loader.py** – WebP C API loader + OpenCV fallback; FIFOImageBuffer
- **index_calculator.py** – ping‑pong & clock math
- **folder_selector.py** – random folder selection logic
- **event_handler.py** – Pygame event mapping
- **display_manager.py** – display init & aspect‑ratio logic
- **index_client.py** / **index_server.py** – WebSocket sync modes
- **make_file_lists.py** – scans folders, writes CSV lists
- **bootstrap.sh** / **runPortrait.sh** / **launchInterleavingScript.sh** – setup & launch scripts
- **lightweight_monitor.py** – HTTP dashboard (TEST_MODE)
- **settings.py** – configuration constants
- **requirements.txt** – Python dependencies

---

## License

MIT License. See [LICENSE](LICENSE).

*© 2025 Ben Gencarelle*

