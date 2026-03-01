# VideoInterleaving

**VideoInterleaving** is a timecode-synced image-sequence renderer designed for high-performance animation installations. It supports dual-layer blending, MIDI/MTC synchronization, real-time OpenGL rendering, and multi-format streaming (MJPEG & ASCII).

It is designed to run on everything from high-end workstations to headless Raspberry Pis and remote VPS instances.

---

## Key Features

* **Dual-Layer Compositing**: Blends a "Main" face layer with a "Float" overlay layer in real-time.
* **Multi-Mode Output**:
    * **Local Window**: GPU-accelerated OpenGL display (GLFW).
    * **Web Stream**: Low-latency MJPEG stream for browsers.
    * **ASCII Stream**: Real-time text-mode video over Telnet/TCP.
* **Performance First**: Uses Side-by-Side (SBS) JPEGs and TurboJPEG for maximum throughput on low-power CPUs.
* **Sync**: Supports free-running, MIDI, MTC, and Client/Server index synchronization.

---

## 1. Prerequisites

### Automated Setup (Recommended)

The `setup_app.sh` script handles all dependencies and systemd service configuration automatically:

```bash
git clone [https://github.com/bgencarelle/VideoInterleaving.git](https://github.com/bgencarelle/VideoInterleaving.git)
cd VideoInterleaving
sudo ./setup_app.sh
```

This script will:
* Install all system packages from `system-requirements.txt`
* Create a Python virtual environment with `--system-site-packages` enabled
* Install all Python packages from `requirements.txt`
* Auto-detect your display environment (X11/Wayland/framebuffer)
* Create systemd services for web, ASCII, and local modes

### Manual Setup

#### System Libraries

**Debian/Ubuntu/Raspbian:**
System packages are listed in `system-requirements.txt`. Install them with:

```bash
sudo apt update
sudo apt install $(grep -v '^#' system-requirements.txt | tr '\n' ' ')
```

**Fedora/CentOS (rough equivalents):**

```bash
sudo dnf install python3 python3-pip python3-devel gcc gcc-c++ make cmake pkgconfig \
    libwebp-devel libjpeg-turbo-devel SDL2-devel alsa-lib-devel \
    mesa-libGL-devel mesa-libGLU-devel mesa-libEGL-devel mesa-libGLES-devel \
    libglvnd-devel glfw-devel mesa-utils chrony ninja-build bind-utils
```

**macOS (Homebrew):**

```bash
brew install python webp pkg-config sdl2 chrony jpeg-turbo 
```

#### Python Environment

Clone the repo:

```bash
git clone [https://github.com/bgencarelle/VideoInterleaving.git](https://github.com/bgencarelle/VideoInterleaving.git)
cd VideoInterleaving
```

Create venv with system-site-packages (allows access to system-installed Python packages):

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
```

Install Dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```
> **Note:** `requirements.txt` only contains Python packages (system packages live in `system-requirements.txt`).

---

## 2. Image Preparation (Crucial)

To achieve high framerates on low-end hardware, this engine uses a custom Side-by-Side (SBS) JPEG format instead of RGBA WebP or PNG.

* **Left Half**: Color Data (RGB)
* **Right Half**: Alpha Mask (Grayscale)

**How to Convert Your Images:** We provide a multi-core converter tool that takes your existing folder of WebP/PNGs and generates the optimized SBS JPEGs.

```bash
python tools/convert_to_sbs_fixed.py
# Follow the prompts to select your source 'images' folder.
# It will create a new folder (e.g., 'images_sbs') automatically.
```
> **Note:** Ensure your `settings.py` or CLI arguments point to this new `_sbs` folder.

---

## 3. Running the Player

You can configure the player via `settings.py` (defaults) or override them at runtime using CLI arguments.

### The CLI Way (Recommended)

**1. Start the Web Stream (MJPEG):**
```bash
python main.py --mode web --dir ./images_sbs
```
* View at: `http://<IP>:8080`
* Monitor: `http://<IP>:1978`

**2. Start the ASCII Stream (Telnet):**
```bash
python main.py --mode ascii --dir ./images_tiny
```
* Connect via Terminal: `telnet <IP> 2323` or `nc <IP> 2323`
* *Note: Use smaller resolution images (e.g., 150px wide) for ASCII to save CPU.*

**3. Start Local Mode (Windowed):**
```bash
python main.py --mode local --dir ./images_sbs
```

### The `settings.py` Way (Legacy)

Edit `settings.py` to set your defaults:

```python
SERVER_MODE = True       # Web Stream
ASCII_MODE = False       # ASCII Stream (Overrides SERVER_MODE if True)
IMAGES_DIR = "images_sbs"
```

Then simply run:

```bash
python main.py
```

---

## 4. ASCII Mode Details

The ASCII engine is a dedicated render path that converts video frames into colored ANSI text characters.

**Connecting:** Standard `telnet` works, but `netcat` (`nc`) often provides a smoother frame rate.

```bash
# Auto-reconnect loop for digital signage displays
while true; do nc 192.168.1.50 2323; sleep 1; done
```

**Configuration (`settings.py`):**
* `ASCII_WIDTH` / `HEIGHT`: Resolution of the text grid (e.g., 80x40).
* `ASCII_COLOR`: Enable/Disable ANSI color codes.
* `ASCII_FONT_RATIO`: Corrects aspect ratio for non-square terminal characters (default `0.55`).

**Artistic Tweaks:**
* `ASCII_SATURATION`: Boost color intensity.
* `ASCII_GAMMA`: Lift mid-tones for better visibility on dark terminals.
* `ASCII_PALETTE`: Custom character set sorted by visual density.

---

## 5. Hardware Configuration

### Raspberry Pi (HDMI & Composite Out)

For kiosk deployments using Wayland and hardware video output (HDMI or Composite), use the following configurations.

**1. Boot Configuration (`/boot/firmware/config.txt`)**
Add or modify the following lines to configure the display outputs and DRM drivers:

```ini
# Enable DRM VC4 V3D driver
dtoverlay=vc4-kms-v3d,composite
enable_tvout=1
max_framebuffers=2

# HDMI configuration
#hdmi_ignore_edid=0xa5000080
hdmi_ignore_hotplug=1
hdmi_force_hotplug=0
#hdmi_group=2              # CEA
#hdmi_mode=9               # 800x600p
#hdmi_drive=1              # DVI mode (no HDMI audio)

# Don't have the firmware create an initial video= setting in cmdline.txt.
# Use the kernel's default instead.
disable_fw_kms_setup=1
```

**2. Kernel Command Line (`/boot/firmware/cmdline.txt`)**
Ensure your boot parameters are set correctly to handle composite resolution and margins. *Add this to the start of the single line in the file:*

```text
console=serial0,115200 console=tty1 root=PARTUUID=ce063a65-02 rootfstype=ext4 fsck.repair=yes rootwait quiet splash plymouth.ignore-serial-consoles cfg80211.ieee80211_regdom=DE video=Composite-1:720x576@50ie margin_left=40,margin_right=40,margin_top=32,margin_bottom=32
```

**3. Wayland Display Initialization**
To force the correct output resolution in a Wayland environment (like Sway or Wayfire), run the appropriate `wlr-randr` command before launching the app. This can be edited in  ~/.config/labwc/autostart after running sudo setup_vi_kiosk_wayland.sh :

* **For HDMI:**
  ```bash
  sh -c 'sleep 1; wlr-randr --output HDMI-A-1 --mode 1920x1080 || true'
  ```
* **For Composite:**
  ```bash
  sh -c 'sleep 1; wlr-randr --output Composite-1  --mode 720x576  || true'
  ```

**4. Kiosk Autostart**
To automatically launch and respawn the application in a kiosk environment:

```bash
/usr/bin/lwrespawn /opt/kiosk/run_videointerleaving.sh
```

### Performance Optimization for Low-Power Devices

For Raspberry Pi Zero 2 W and similar low-power devices, consider these optimizations in `settings.py`:

* **Memory Optimization:**
  * `FIFO_LENGTH = 10-15` (default: 30) - Reduces memory usage by limiting pre-loaded frames. Lower values reduce memory footprint but may cause frame drops if loading is slow.
* **Encoding/Decoding Performance:**
  * `JPEG_QUALITY = 40-50` (default: 55) - Lower quality equals faster encode/decode.
* **Frame Rate:**
  * `FPS = 20-25` (default: 30) - Lower target FPS reduces CPU load.
  * `SERVER_CAPTURE_RATE = 10-15` (default: matches FPS) - Lower capture rate for web streaming.
* **Additional Tips:**
  * Use pre-encoded SBS JPEGs (already optimized format).
  * Disable frame counter if not needed: `FRAME_COUNTER_DISPLAY = False`
  * Reduce image resolution if possible (smaller images = faster processing).

### Legacy GPUs (GLES 2.0 / GL 2.1)

Some systems (e.g., Raspberry Pi 2 / older iGPUs / restricted drivers) only expose GLES 2.0 / OpenGL 2.1. In these cases, the app automatically switches to a legacy PyOpenGL renderer for local window mode.

For headless streaming on Wayland where standalone EGL contexts fail, the app will also fall back to a hidden GLFW window + legacy FBO capture path.

**Optional overrides:**
* Force legacy renderer: `FORCE_LEGACY_GL=1`
* Force GLES version attempts: `GLES_REQUIRE_OVERRIDE=200` (or `300`, `310`)

**Wayland note:**
* In local mode on Wayland, fullscreen uses a borderless fullscreen-sized window (not a mode-setting fullscreen) to reduce compositor/session crashes on some drivers.

**TurboJPEG note:**
* If you see `unable to locate turbojpeg library automatically`, install `libturbojpeg0` (Debian/Raspbian) or set `TURBOJPEG_LIB=/path/to/libturbojpeg.so.0`.

### Precision Timing (Chrony)

For installations requiring frame-perfect sync across multiple machines, installing `chrony` is highly recommended.

**Linux Install:**
```bash
sudo apt install chrony
```
*(See `chrony.conf` sample in repo for PTB/German time server config.)*

---

## Project Structure

* `main.py`: Entry point. Parses CLI args and launches threads.
* `renderer.py`: The Unified Rendering Engine. Handles OpenGL (Shader) and CPU (NumPy) compositing.
* `display_manager.py`: Manages window creation (GLFW or Headless FBO) and GL context.
* `image_display.py`: The main loop. Manages time, loading, and feeding the renderer.
* `image_loader.py`: High-speed TurboJPEG loader with FIFO buffering.
* `web_service.py`: Flask-less HTTP server for MJPEG streaming and System Monitoring.
* `ascii_server.py`: Raw TCP server for Telnet streaming.
* `ascii_converter.py`: Vectorized image-to-text conversion engine.
* `settings.py`: Global configuration constants.
* `tools/`: Helper scripts (e.g., `convert_to_sbs_fixed.py`).

---

## License

MIT License. See LICENSE.

© 2026 Ben Gencarelle
