# settings.py

# -------------------------
# Image Directories and Folder Paths
# -------------------------
IMAGES_DIR = ("images")
MAIN_FOLDER_PATH = f"{IMAGES_DIR}/face"
FLOAT_FOLDER_PATH = f"{IMAGES_DIR}/float"

# -------------------------
# Display Mode & Performance
# -------------------------
FULLSCREEN_MODE = True
VSYNC = True  # or False, depending on your preference

# Frames per Second and Images Per Second (IPS)
IPS = 30
FPS = 25

# Buffer settings: The BUFFER_SIZE is derived from IPS (e.g., 15 if IPS == 60)
TOLERANCE = 10
FIFO_LENGTH = 30

# Run mode stuff
PINGPONG = True

FROM_BIRTH = True
BIRTH_TZ = "EST"

BIRTH_TIME = "1978, 11, 17, 7, 11"

TEST_MODE = True

# web stuff
HTTP_MONITOR = True  # or True
WEB_PORT = 1978  # web port is year of birth
FRAME_COUNTER_DISPLAY = True  # If the monitor is on, skip printing

# For testing purposes.
if TEST_MODE:
    SHOW_DELTA = not HTTP_MONITOR
else:
    SHOW_DELTA = False

# Additional Display Settings
BACKGROUND_COLOR = (4, 4, 4)       # Background clear color (RGB)
GAMMA_CORRECTION_ENABLED = False    # Enable gamma correction in fragment shader
ENABLE_SRGB_FRAMEBUFFER = False     # Request sRGB framebuffer if supported

# -------------------------
# Clock Mode Constants
# -------------------------
MTC_CLOCK = 0
MIDI_CLOCK = 1
MIXED_CLOCK = 2
CLIENT_MODE = 3
FREE_CLOCK = 255

# -------------------------
# Image Transformation Settings
# -------------------------
INITIAL_ROTATION = 0
INITIAL_MIRROR = 0
CONNECTED_TO_RCA_HDMI = False  # this is for analog TVs
RCA_HDMI_RESOLUTION = (640, 480)
LOW_RES_FULLSCREEN = False        # Set to True to force low resolution mode
LOW_RES_FULLSCREEN_RESOLUTION = (960, 600)

# -------------------------
# Timing and Clock Buffer Parameters - midi
# -------------------------
TIMEOUT_SECONDS = 1  # Set the timeout as needed
CLOCK_BUFFER_SIZE = 50
CLOCK_MODE = FREE_CLOCK

# If using MIDI-based clock mode.
MIDI_MODE = False

# Valid Clock Modes (for interactive selection or configuration)
VALID_MODES = {
    "MTC_CLOCK": MTC_CLOCK,
    "MIDI_CLOCK": MIDI_CLOCK,
    "MIXED_CLOCK": MIXED_CLOCK,
    "CLIENT_MODE": CLIENT_MODE,
    "FREE_CLOCK": FREE_CLOCK,
}

from timezones import TIMEZONE_OFFSETS
#---- AUDIO

# in settings.py (you’ll make this later)

AUDIO_MODE = "preset"        # "preset" or "api"
AUDIO_PRESET = "hybrid4"     # one of: "product_mod_triad", "midpoint", "hybrid4"
AUDIO_BASE_OCTAVE = 3        # C3-ish region
AUDIO_OCTAVE_SPAN = 3        # how many octaves the index can walk through

# --- SERVER MODE CONFIGURATION ---
SERVER_MODE = False       # Enable headless streaming
HEADLESS_USE_GL = True   # new: disable ModernGL headless on VPS
STREAM_PORT = 8080           # Port for video stream (distinct from monitor port 1978)
STREAM_HOST = '127.0.0.1'      # Listen on all interfaces
SERVER_CAPTURE_RATE = FPS     # Streaming FPS
JPEG_QUALITY = 75# Image quality
HEADLESS_RES = (400, 533)   # Resolution for the virtual screen
HEADLESS_BACKEND = "egl"     # "egl"
MAX_VIEWERS = 20 # Max simultaneous connections


# --- ASCII MODE SETTINGS ---
ASCII_MODE = False
ASCII_COLOR = True
ASCII_PORT = 2323
ASCII_HOST = '0.0.0.0'
ASCII_WIDTH = 90
ASCII_HEIGHT = 60
ASCII_FPS = 15
ASCII_FONT_RATIO = .50

# --- ARTISTIC TWEAKS ---
ASCII_CONTRAST   = 1.2  # tiny bit of punch
ASCII_SATURATION = .8   # slightly muted color
ASCII_BRIGHTNESS = 1.3   # handled in HSV
ASCII_GAMMA      = .7  # neutral given your LUT definition

ASCII_PADDING_CHAR = " "     # <--- NEW: Character for pillar/letterboxing

# --- THE GENCARELLE PALETTE ---
ASCII_PALETTE_LIGHT = "MWB8GRDNHESAVTOLPmevncray97stji1-/., "
ASCII_PALETTE_DARK = " ,.1ijts79yarcnvemCPLOTVASEHNDRG8BWM"
ASCII_PALETTE = ASCII_PALETTE_LIGHT

