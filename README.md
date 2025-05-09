VideoInterleaving is a timecode-synced image-sequence renderer designed for high-performance animation installations. It supports dual-layer blending, MIDI/MTC synchronization, client-server WebSocket sync, and real-time OpenGL rendering.

---

## Development Tools

The following tools are used or recommended for building and running **VideoInterleaving** in a stable and precise time-sensitive environment:

* **WebP Library**:

  * Required for fast decoding of `.webp` image sequences.
  * Ensure `is installed on Linux or` via Homebrew/Chocolatey.
* **Chrony** (Linux only):

  * High-precision NTP client recommended for keeping system time accurate in real-time installations.
  * Install with:

    ```bash
    sudo apt install chrony
    ```
  * Ensures stable **FREE\_CLOCK** or hybrid clock performance.
* **Python 3.8+**, OpenGL, and SDL2 for rendering.
* **requirements.txt** defines all necessary Python packages.

---

## Prerequisites

1. **Python** 3.8+  (check with `python3 --version`).

2. **System libraries**

   **Debian/Ubuntu/Raspbian:**

   ```bash
   sudo apt update
   sudo apt install python3-venv python3-dev python3-pip build-essential cmake pkg-config \
       libwebp-dev libsdl2-dev libasound2-dev libgl1-mesa-dev libglu1-mesa-dev \
       libegl1-mesa-dev chrony
   ```

   **Fedora/CentOS:**

   ```bash
   sudo dnf install python3-venv python3-pip python3-devel build-essential cmake pkgconfig \
       libwebp-devel SDL2-devel alsa-lib-devel mesa-libGL-devel mesa-libGLU-devel \
       mesa-libEGL-devel chrony
   ```

   **macOS (Homebrew):**

   ```bash
   brew install python webp pkg-config sdl2 chrony
   ```

   **Windows (Chocolatey):**

   ```powershell
   choco install python webp sdl2 chrony
   ```

3. **Git**:

   ```bash
   git --version
   ```

4. **Optional MIDI**: a functional MIDI interface or USB-MIDI adapter for `MIDI_CLOCK` or `MTC_CLOCK` modes.

### Chrony Configuration

If you're in Germany or nearby and want highly reliable time servers, here's a sample configuration used during development. You may adapt it to your region's preferred NTP servers:

Edit `/etc/chrony/chrony.conf` (Linux) to include:

```conf
# Primary Tier 1 German time servers (Physikalisch-Technische Bundesanstalt)
server ptbtime1.ptb.de iburst prefer
server ptbtime2.ptb.de iburst
server ptbtime3.ptb.de iburst

# Additional reliable European pool servers
server de.pool.ntp.org iburst
server 0.europe.pool.ntp.org iburst
server 1.europe.pool.ntp.org iburst

# Do not use DHCP-pushed NTP servers
# Ensure this directive is commented out or the directory does not exist:
# sourcedir /run/chrony-dhcp
```

---

## Setup & Running

1. **Clone & prepare the repository**

   ```bash
   git clone https://github.com/bgencarelle/VideoInterleaving.git
   mv VideoInterleaving pyInter
   cd pyInter
   ```

2. **Bootstrap environment**

   ```bash
   ./bootstrap.sh
   ```

   * Pay close attention to any error messages and follow the prompts to install missing dependencies.

3. **Create & activate a Python venv**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate    # Linux/macOS
   .venv\Scripts\activate      # Windows PowerShell
   ```

4. **Install Python dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Prepare your image sequences**

   * Organize your frames into two parallel folder trees:

     ```text
     images/main/<frame_number>/*.webp
     images/float/<frame_number>/*.webp
     ```
   * The player auto-scans the `images/` folders and regenerates `csv` lists on each run. Only use:

     ```bash
     python make_file_lists.py
     ```

     if you manually add or remove folders.

6. **Launch the player**

   ```bash
   source .venv/bin/activate
   python main.py
   ```

   Or use the convenience script:

   ```bash
   bash launchInterleavingScript.sh
   ```

---

## Convenience Launchers

* \`\` (in `$HOME`): Activates the venv, changes to the project directory, and runs `main.py`.

* \`\` (in project root):

  ```bash
  #!/bin/bash
  set -e
  # Change to project dir
  cd "$HOME/pyInter"
  # Activate venv
  source .PyIntervenv/bin/activate
  # Run the player
  python main.py
  ```

---

## HTTP Monitor (Webserver)

When `TEST_MODE=True` and `HTTP_MONITOR=True` in `settings.py`, a lightweight HTTP server runs on startup:

* **Port:** configured by `WEB_PORT` (default: 8000)
* **Access:** `http://localhost:<WEB_PORT>/`
* **Dashboard:** live FPS, FIFO stats/misses, folder coverage/entropy, CPU/memory usage, error logs.

---

## Clock Modes

As of May 2025, only `FREE_CLOCK` mode has been tested and verified in the main branch. Other modes (`MIDI_CLOCK`, `MTC_CLOCK`, `MIXED_CLOCK`, `CLIENT_MODE`) are present in the codebase but should be considered experimental until further testing.

| Mode         | Constant | Description                  |
| ------------ | -------- | ---------------------------- |
| FREE\_CLOCK  | 255      | System-time driven ping-pong |
| MIDI\_CLOCK  | 1        | Sync to MIDI clock pulses    |
| MTC\_CLOCK   | 0        | SMPTE-style timecode sync    |
| MIXED\_CLOCK | 2        | Experimental MIDI + MTC      |
| CLIENT\_MODE | 3        | WebSocket index sync         |

Set via `CLOCK_MODE` in `settings.py` or interactively at launch.

---

## Project Structure

* **main.py** / **image\_display.py** – entrypoint, threading, FIFO, monitor hooks
* **renderer.py** – OpenGL shader & draw routines
* **image\_loader.py** – WebP C API loader + OpenCV fallback; FIFOImageBuffer
* **index\_calculator.py** – ping-pong & clock math
* **folder\_selector.py** – random folder selection logic
* **event\_handler.py** – Pygame event mapping
* **display\_manager.py** – display init & aspect-ratio logic
* **index\_client.py** / **index\_server.py** – WebSocket sync modes
* **make\_file\_lists.py** – scans folders, writes CSV lists
* **bootstrap.sh** / **runPortrait.sh** / **launchInterleavingScript.sh** – setup & launch scripts
* **lightweight\_monitor.py** – HTTP dashboard (TEST\_MODE)
* **settings.py** – configuration constants
* **requirements.txt** – Python dependencies
* **utilities/** – helper scripts and tools that may be useful for developing or preprocessing your own image sequences

---

## License

MIT License. See [LICENSE](LICENSE).

*© 2025 Ben Gencarelle*
