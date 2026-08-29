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
# Scope refresh defaults to IPS. For completed vector/raster passes, one trace
# per index maximises their finite sample budget (samples = rate / fps).
# Stochastic is continuous: a trace is only an audio buffer, not an image.
SCOPE_MODE = False
SCOPE_ROW_BIAS = 1.0      # >1 trades columns for rows at constant cell count.
                          # Faces want ~1.3: their features are horizontal
                          # edges, and rows resolve those. Past ~1.6 the mouth
                          # smears and the silhouette blocks up.
SCOPE_BORDER = 0.0        # fraction of each trace spent drawing a fixed
                          # rectangle at the full extent. Pins the framing so
                          # the picture stops skewing as content changes.
                          # 0.03 is about right; 0 = off.
SCOPE_DC_COMP = None      # Hz. Cancel the output's AC coupling at this corner
                          # so flat regions of a trace do not sag. Start at 30.
                          # Costs amplitude (~26% at a 20 Hz corner, ~47% at
                          # 50 Hz), so raise the scope gain to compensate.
SCOPE_PREVIEW_FPS = 12    # /scope/stream.mjpg frame rate. THE cpu knob for
                          # the web preview, together with ?size=. Rendering
                          # runs on the request thread, so this never costs the
                          # trace deadline -- but on a Pi keep it low.
SCOPE_DEVICE = None       # audio output: name FRAGMENT ("Scarlett", "hw:1")
                          # or index. Prefer the name -- PortAudio indices
                          # reshuffle when hardware is plugged or unplugged, so
                          # an index that works today is a wrong-device bug at
                          # the install. --device overrides this. This is the
                          # only way to pin an output without a CLI, which is
                          # what a systemd unit needs.
SCOPE_DEVICE_SPEC = None  # transient CLI name/index before it is resolved
SCOPE_DEVICE_RESOLVED = False
SCOPE_ASK = False         # interactive device picker; safe only with a tty
SCOPE_FPS = None          # None -> follow IPS
SCOPE_SAMPLES = None      # explicit samples/trace; incompatible with mix
SCOPE_RENDER_MODE = "vector"  # vector | raster | stochastic | stipple | fusion
                          # Stochastic is a luminance-weighted XY walk; fusion
                          # multiplexes corresponding V/R/S array entries.
SCOPE_FUSION = "vrs"      # vrs | vr | sv | sr; round-robin position sources
SCOPE_RASTER = False      # compatibility mirror for older integrations
SCOPE_REALTIME = False    # raster-only low-latency streaming path
SCOPE_LIST_FROM_IMAGES = False  # bypass the baked manifest for legacy bakes
SCOPE_FIELDS = 1          # raster interlace: traces per picture. 2 or 4 lifts
                          # the refresh rate above flicker fusion without
                          # touching the grid. Needs SCOPE_FPS = N * IPS.
SCOPE_FIELDS_EXPLICIT = False  # main.py sets this for --scope-fields, including
                          # explicit 1, which disables mix's automatic fields
SCOPE_MIN_FEATURE = 0.02  # vector: shortest stroke kept by the occlusion cull
SCOPE_TRIM = 0.02         # raster: drop cells dimmer than this from the sweep
SCOPE_GAMMA = 2.2         # raster luminance exponent
SCOPE_DENSITY = 1.0       # raster samples per cell (1.0 = finest)
SCOPE_PRECONDITION = None # None uses compact bake's recommendation; legacy 0
SCOPE_WALK_RADIUS = 10    # stochastic: nearest-neighbour search radius, pixels
SCOPE_WALK_STRIDE = 0     # stochastic: auto-scale (~width/120; 1 at width 128)
SCOPE_WALK_RESEED_MS = 5.0  # stochastic: jump to another bright region
SCOPE_WALK_GAMMA = 2.0    # stochastic: useful portrait default. Equivalent to
                          # Osci Image Threshold 0.1 (exponent = UI * 10 + 1).
SCOPE_WALK_EDGE = 0.0     # stochastic: optional non-Osci edge probability
SCOPE_WALK_HZ = 48000.0   # stochastic target clock; independent of faster DACs
SCOPE_STIPPLE_POINTS = 768 # stable weighted image positions; output rate only
                           # resamples the finished proximity-ordered route
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
SCOPE_MIX = None          # Hz for triangular V -> R -> S -> R trace mix
SCOPE_MIX_DUTY = 0.5      # raster share; remainder splits vector/stochastic
