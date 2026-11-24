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
# Common milliseconds subtracted from the free-running index clock. Keep this
# equal on synchronized units; positive values backdate all of them together.
INDEX_TIME_OFFSET_MS = 0.0

# Buffer settings: The BUFFER_SIZE is derived from IPS (e.g., 15 if IPS == 60)
TOLERANCE = 10
FIFO_LENGTH = 15

# Run mode stuff
PINGPONG = True
FROM_BIRTH = True

# web stuff
# WEB_PORT is now managed by server_config.py - do not set here
FRAME_COUNTER_DISPLAY = True  # If the monitor is on, skip printing


# If using MIDI-based clock mode.
MIDI_MODE = False

# --- SERVER MODE CONFIGURATION ---
SERVER_MODE = True      # Enable headless streaming
HEADLESS_USE_GL = True  # new: disable ModernGL headless on VPS
# Web/ascii capture rate, decoupled from FPS: this is the bandwidth knob for
# slow links. The index still advances at IPS (30), so at 10fps the stream
# carries every third index -- but each frame it carries is the CURRENT one,
# never a backlog. Raise toward IPS on fast links for per-index fidelity.
SERVER_CAPTURE_RATE = 15

# --- ASCII MODE SETTINGS ---
ASCII_MODE = False
ASCII_COLOR = True
ASCII_TRUECOLOR = True  # Full RGB tints; False selects the 256-colour fallback.
ASCII_COLOR_BRIGHTNESS = 1.0  # Truecolour tint exposure, separate from glyph brightness.
ASCII_COLOR_BLUR = 0  # Optional tint smoothing; 0 preserves source colours.
# Independent of SERVER_CAPTURE_RATE: ascii frames are a few KB of text, so
# the telnet/websocket rate is not the bandwidth problem the JPEG stream is.
ASCII_FPS = 15
ASCII_WIDTH = 60 # keep aspect ratio 3:2
ASCII_HEIGHT = 40
ASCII_WEB_MAX_SIZE = (120, 80)  # Live web master; terminal scales remain separate.
ASCII_WEB_SOURCE_SIZE = None  # Enabled only in asciiweb; baked packages keep their own grid.
ASCII_SOURCE_IMAGE_ASPECT_RATIO = 1.333333333

# --- Local ASCII display (--mode asciilocal) ---
LOCAL_ASCII = False              # Set by --mode asciilocal.
LOCAL_ASCII_GRID = (160, 120)    # Finest grid (cols, rows); --ascii-grid 160x120 overrides. None = ASCII_WEB_MAX_SIZE x multiplier. Starting point for the ASCII service.
LOCAL_ASCII_MULTIPLIER = 1       # Local master = ASCII_WEB_MAX_SIZE x this (2 -> 240x160). Sampled on this machine only.
LOCAL_ASCII_MULTIPLY = 'fine'    # 'fine': only the finest end grows; 'all': coarsest and default sizes scale too.
LOCAL_ASCII_ROWS = (None, None)  # Allowed row range (min, max); None = 10 / the master's rows.
LOCAL_ASCII_HOME_ROWS = None     # Size at rest and after reset; None = ASCII_HEIGHT.
LOCAL_ASCII_MANUAL_TIMEOUT = 30  # Seconds after the last wheel/key zoom before random zoom takes over again; 0 = stay manual until 0 is pressed.
LOCAL_ASCII_ZOOM = 'random'      # 'random': folder-selector-style driver; 'manual': wheel and keys only.
LOCAL_ASCII_WALK_RATE = 0        # Mouse/keys: rows per second toward a new size; 0 jumps straight there, as the browser does.
LOCAL_ASCII_RANDOM_TRANSITION = (0.5, 3.0)  # Random driver: each change takes a random time in this range (s), drawn per change, eased; (0, 0) jumps.
LOCAL_ASCII_HOLD = (2.0, 20.0)   # Random driver: at least / at most this long (s) between changes, counted from each pick as the folder selector does.
LOCAL_ASCII_CROSSFADE = False    # True: mid-change, cross-fade the two neighbouring grid sizes (smoother, but less text-mode). False: step a whole grid at a time.
LOCAL_ASCII_LINE_SPACING = 1.05  # As the web viewer's default.
LOCAL_ASCII_DECODE_ROW_PX = 4   # JPEGs decode at 1/2-1/8 size while keeping this many pixels per finest row (only the grid is sampled); 0 = full size.
# --- Receipt printer (--printer: local and asciilocal) ---
# The MICROS-branded Epson TM-T88V (C31CA85A6001) on its Ethernet interface,
# raw ESC/POS on port 9100. The register reaches it through the NetworkManager
# connection "printer" (enp1s0, 172.18.35.180/27, no gateway).
PRINTER = False                  # Set by --printer. P prints without it: to the system's default printer (CUPS / Windows), else to this receipt printer when --printer is on.
PRINTER_MODE = 'screen'          # 'screen': the window's pixels, dithered (any local mode). 'text': the ASCII characters (asciilocal); --printer text.
PRINTER_HOST = '172.18.35.171'   # The receipt printer --printer drives (random prints, double tap; P too when there is no system printer).
PRINTER_PORT = 9100
PRINTER_EVERY = (120, 900)       # Seconds between random prints (2-15 min); --print-every 2-15 sets it in minutes.
PRINTER_DOTS = 576               # 'screen': dots across (576 on 80 mm paper).
PRINTER_ROTATE = None            # Receipts print the picture upright as it stands on screen (R undone), full paper width. 'auto': turn a landscape picture to run along the paper.
PRINTER_FEED_LINES = 10          # Receipts: blank lines fed after the print, then the cut.
PRINTER_SPEED = 2                # Receipt pictures: print speed 1 (slowest) - 13. Slow prints heat fewer dots at once: less peak current. None = the printer's own.
PRINTER_INK = 0.85               # Receipt pictures: share of dots burnt where the picture is black (0-1). Less heat and current; black still reads black.
PRINTER_PAPER = 'white'          # 'white': the frame is drawn again on white for printing, tones kept (ASCII letters in ink as dark as they are bright on screen, no glow). 'black': exactly as on screen (mostly-black frames print nearly solid).
PRINTER_COLUMNS = 64             # 'text': 80 mm paper in font B (576 dots / 9).
PRINTER_DARK = 32                # 'text': cells drawn darker than this on screen (0-255, brightest colour channel) print blank, as the eye barely sees them; 0 prints every character.
LOCAL_ASCII_SHAPE = 'picture'    # Grid shape: 'picture' (whole image, as large as the window allows), 'window' (fills it, image cropped), 'grid' (browser's 3:2, letterboxed).
LOCAL_ASCII_GLOW = 0.8           # --ascii-glow: neighbouring cells' colour under small letters (0 off, 1 full). Starting point for the ASCII service.
LOCAL_ASCII_GLOW_FROM = 32       # Glow starts when cells are this many pixels high...
LOCAL_ASCII_GLOW_FULL = 8        # ...and is at full strength from this height down.
LOCAL_ASCII_STROKES = 0.5        # --ascii-strokes: keep small letters bright (0 averaged as before, 1 brightest pixel). Starting point for the ASCII service.
LOCAL_ASCII_FONT = 'static/fonts/century-schoolbook/CenturySchoolbookBold.ttf'  # or 'static/fonts/schoolbook-mono/InterleaveSchoolbookMono-Black.woff2' (monospaced, weight 900, as on the web)
JPEG_QUALITY = 60# Image quality (lower = fewer bytes per frame on slow links)
HEADLESS_RES = (320, 400)   # Resolution for the virtual screen (web frame size)

# --- SCOPE MODE (scope waveform output via the sound card) ---
# Scope refresh defaults to IPS. For completed vector/raster passes, one trace
# per index maximises their finite sample budget (samples = rate / fps).
# Stochastic is continuous: a trace is only an audio buffer, not an image.
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
SCOPE_PREVIEW_FPS = 15    # /scope/stream.mjpg frame rate. THE cpu knob for
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
SCOPE_SOURCE = "bake"     # "bake" (offline XY) or "images" (runtime thumbnails)
SCOPE_GUI = False          # optional native scope trace preview/tuning window
SCOPE_GUI_IMAGE_ONLY = False  # start the native scope preview without controls
SCOPE_GUI_FULLSCREEN = False   # start the native scope preview fullscreen
SCOPE_CHANNELS = (1, 2)   # 1-based PortAudio X,Y outputs; e.g. (18, 19)
SCOPE_LIVE_SIZE = 128     # live-image thumbnail width; bake format is unchanged
SCOPE_FPS = None          # None -> follow IPS
SCOPE_SAMPLES = None      # explicit samples/trace; incompatible with mix
SCOPE_GEOMETRY_SAMPLES = None  # opt-in independent trajectory detail budget
SCOPE_TRAVERSAL_HZ = None     # opt-in geometry traversal clock
SCOPE_PHYSICAL_DWELL = 0.0    # opt-in physical dwell redistribution [0, 1]
SCOPE_RENDER_MODE = "vector"  # vector | raster | stochastic | stipple | fusion
                          # Stochastic is a luminance-weighted XY walk; fusion
                          # multiplexes corresponding V/R/S entries by light.
SCOPE_FUSION = "vrs"      # vrs | vr | sv | sr; weighted position sources
SCOPE_RASTER = False      # compatibility mirror for older integrations
SCOPE_INVERT = False      # inverse tone; vector approximates it through dwell velocity
SCOPE_REALTIME = False    # raster-only low-latency streaming path
# The X trigger marker is a property of the output, not a mode: shaped as a
# ramp and parked outside the picture box it costs an XY display nothing, and
# it supplies the trigger for single-input Y-T viewing. On by default; turn it
# off with --no-scope-trigger.
SCOPE_TRIGGER = True
SCOPE_X_ONLY = False     # open one output channel and send the X waveform only
SCOPE_TRIGGER_US = 250.0   # marker duration; CLI override is in microseconds
SCOPE_TRIGGER_SHAPE = "ramp"  # "ramp": moving, parked off-picture, invisible on
                          # XY. "step": the original two-dwell marker, for a
                          # scope whose trigger will not take the ramp.
SCOPE_YT_TIMING = "dwell"  # "fixed": equal row slots, so unrelated brightness
                          # cannot move or resize a row. Costs the tonal
                          # balance of whole-trace weighting and rails empty
                          # rows at -0.936; raster only.
SCOPE_YT = False          # deprecated: reads as --scope-yt-timing fixed
SCOPE_YT_TRIGGER_US = None  # deprecated alias for SCOPE_TRIGGER_US
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
SCOPE_PRECONDITION = 0.0  # natural raster tone; opt in only if a display needs it
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
SCOPE_MIX = None          # Hz for V -> R -> S -> R -> T -> R trace mix
SCOPE_MIX_DUTY = 0.5      # raster share; remainder splits vector/stochastic/stipple
