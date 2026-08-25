# settings.py
import os
from constantStorage.bio_constants import *
from constantStorage.midi_constants import *
from constantStorage.timezones import *
from constantStorage.ascii_constants import *
from constantStorage.server_constants import *
from constantStorage.display_constants import *

# -------------------------
# Image Directories and Folder Paths
# -------------------------
# Use images_sbs if it exists, otherwise fall back to images
if os.path.exists("images_sbs"):
    IMAGES_DIR = "images_sbs"
elif os.path.exists("images"):
    IMAGES_DIR = "images"
else:
    # Default to images_sbs if neither exists (will error later if truly missing)
    IMAGES_DIR = "images_sbs"
MAIN_FOLDER_PATH = f"{IMAGES_DIR}/face"
FLOAT_FOLDER_PATH = f"{IMAGES_DIR}/float"

# XY vector/raster libraries for scope mode, baked by utilities/convert_to_xy.py.
# The baker mirrors the source tree, so face/ and float/ folders line up by index.
# Search the same way IMAGES_DIR is found, plus the "<source>_xy" convention --
# a tree called 150_91 bakes to 150_91_xy, which none of the fixed names catch.
def _find_xy_dir(images_dir):
    import glob
    base = os.path.basename(os.path.normpath(images_dir))
    for cand in (f"{images_dir}_xy", f"{base}_xy", "images_xy", "images_sbs_xy"):
        if os.path.isdir(cand):
            return cand
    # last resort: a single unambiguous *_xy directory in the working dir
    found = [d for d in glob.glob("*_xy") if os.path.isdir(d)]
    if len(found) == 1:
        return found[0]
    return f"{images_dir}_xy"          # nonexistent, but a useful error message


XY_DIR = _find_xy_dir(IMAGES_DIR)

# -------------------------
# Display Mode & Performance
# -------------------------
FULLSCREEN_MODE = True
VSYNC = True  # or False, depending on your preference

# Frames per Second and Images Per Second (IPS)
IPS = 30
FPS = 60

# Buffer settings: The BUFFER_SIZE is derived from IPS (e.g., 15 if IPS == 60)
TOLERANCE = 10
FIFO_LENGTH = 15

# Run mode stuff
PINGPONG = True
FROM_BIRTH = True

# web stuff
HTTP_MONITOR = True  # or True
# WEB_PORT is now managed by server_config.py - do not set here
FRAME_COUNTER_DISPLAY = True  # If the monitor is on, skip printing


# If using MIDI-based clock mode.
MIDI_MODE = False

# --- SERVER MODE CONFIGURATION ---
SERVER_MODE = True      # Enable headless streaming
HEADLESS_USE_GL = True  # new: disable ModernGL headless on VPS
SERVER_CAPTURE_RATE = FPS // 2   #  FPS by 2

# --- ASCII MODE SETTINGS ---
ASCII_MODE = False
ASCII_COLOR = True
ASCII_COLOR_BLUR = 7  # 0 = off, odd integer = blur strength (3, 5, 7)
ASCII_FPS = SERVER_CAPTURE_RATE // 2
ASCII_WIDTH = 60 # keep aspect ratio 3:2
ASCII_HEIGHT = 40
ASCII_SOURCE_IMAGE_ASPECT_RATIO = 1.333333333
JPEG_QUALITY = 75# Image quality
HEADLESS_RES = (480, 600)   # Resolution for the virtual screen

# --- SCOPE MODE (XY output via the sound card) ---
# Scope refresh defaults to IPS: one trace per index maximises samples per
# trace, which is the entire resolution budget (samples = rate / fps).
SCOPE_MODE = False
SCOPE_FPS = None          # None -> follow IPS
SCOPE_FIELDS = 1          # raster interlace: traces per picture. 2 or 4 lifts
                          # the refresh rate above flicker fusion without
                          # touching the grid. Needs SCOPE_FPS = N * IPS.
SCOPE_MIN_FEATURE = 0.02  # vector: shortest stroke kept by the occlusion cull
SCOPE_TRIM = 0.02         # raster: drop cells dimmer than this from the sweep
SCOPE_GAMMA = 2.2         # raster contrast
SCOPE_DENSITY = 1.0       # raster samples per cell (1.0 = finest)
SCOPE_LOWPASS = None      # Hz; emulate a softer output chain (Pi headphone
                          # jack, cheap codec, or a physical RC filter).
                          # None = off. See scope_lowpass.py.
SCOPE_OVERSAMPLE = 1      # anti-alias the path: generate N x samples, bandlimit,
                          # decimate. Correct in principle, but measured only a
                          # 5.6% change on portrait content -- the dwell profile
                          # is already mostly below Nyquist. Costs ~10% CPU at 4.
SCOPE_BUFFER_BLOCKS = 6   # realtime mode: 256-sample blocks queued ahead of the
                          # audio callback. Higher survives slower machines;
                          # each block is ~2.7 ms of added latency at 96 kHz.
SCOPE_AUTOFIT = True      # size the grid against cells that survive trim,
                          # not the whole rectangle -- roughly doubles the
                          # usable grid on a subject over a dark background
SCOPE_SWEEP = "alternate" # alternate | palindrome | retrace
SCOPE_ROWS = None         # raster scanline count (None = auto from budget)
SCOPE_MIX = None          # Hz to alternate raster/vector (None = off)
SCOPE_MIX_DUTY = 0.5      # fraction of mixed passes spent on raster