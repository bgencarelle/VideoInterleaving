# settings.py
from constantStorage.bio_constants import *
from constantStorage.midi_constants import *
from constantStorage.timezones import *
from constantStorage.ascii_constants import *
from constantStorage.server_constants import *
from constantStorage.display_constants import *

# -------------------------
# Image Directories and Folder Paths
# -------------------------
IMAGES_DIR = ("images_sbs")
MAIN_FOLDER_PATH = f"{IMAGES_DIR}/face"
FLOAT_FOLDER_PATH = f"{IMAGES_DIR}/float"

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
FIFO_LENGTH = 30

# Run mode stuff
PINGPONG = True
FROM_BIRTH = True

# web stuff
HTTP_MONITOR = True  # or True
WEB_PORT = BIRTH_YEAR  # web port is year of birth
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
ASCII_FPS = SERVER_CAPTURE_RATE // 2

JPEG_QUALITY = 80# Image quality
HEADLESS_RES = (480, 640)   # Resolution for the virtual screen
