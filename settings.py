# settings.py
# -------------------------
# Image Directories and Folder Paths
# -------------------------
IMAGES_DIR = "images_1080"

MAIN_FOLDER_PATH = f"{IMAGES_DIR}/float"
FLOAT_FOLDER_PATH = f"{IMAGES_DIR}/face"

# -------------------------
# Display Mode & Performance
# -------------------------
FULLSCREEN_MODE = True

# Frames per Second and Images Per Second (IPS)
IPS = 30
FPS = 600

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
BACKGROUND_COLOR = (9, 10, 10)       # Background clear color (RGB)
GAMMA_CORRECTION_ENABLED = False     # Enable gamma correction in fragment shader
ENABLE_SRGB_FRAMEBUFFER = False      # Request sRGB framebuffer if supported

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
INITIAL_ROTATION = 90
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

# --- Helper for timezone offsets (UTC-based) ---
TIMEZONE_OFFSETS = {
    "UTC": 0,
    "GMT": 0,

    # North America
    "EST": -5,  # Eastern Standard Time
    "EDT": -4,  # Eastern Daylight Time
    "CST": -6,  # Central Standard Time
    "CDT": -5,  # Central Daylight Time
    "MST": -7,  # Mountain Standard Time
    "MDT": -6,  # Mountain Daylight Time
    "PST": -8,  # Pacific Standard Time
    "PDT": -7,  # Pacific Daylight Time
    "AKST": -9,  # Alaska Standard Time
    "AKDT": -8,  # Alaska Daylight Time

    # South America
    "ART": -3,  # Argentina Time
    "BRT": -3,  # Brasilia Time
    "CLT": -4,  # Chile Standard Time
    "CLST": -3, # Chile Summer Time

    # Europe
    "CET": 1,   # Central European Time
    "CEST": 2,  # Central European Summer Time
    "EET": 2,   # Eastern European Time
    "EEST": 3,  # Eastern European Summer Time
    "WET": 0,   # Western European Time
    "WEST": 1,  # Western European Summer Time
    "MSK": 3,   # Moscow Time

    # Africa
    "WAT": 1,   # West Africa Time
    "CAT": 2,   # Central Africa Time
    "EAT": 3,   # East Africa Time

    # Asia
    "IST": 5.5,  # India Standard Time
    "PKT": 5,    # Pakistan Standard Time
    "BST": 6,    # Bangladesh Standard Time
    "ICT": 7,    # Indochina Time
    "CST-Asia": 8,  # China Standard Time
    "JST": 9,    # Japan Standard Time
    "KST": 9,    # Korea Standard Time

    # Australia & Oceania
    "AWST": 8,   # Australian Western Standard Time
    "ACST": 9.5, # Australian Central Standard Time
    "ACDT": 10.5,# Australian Central Daylight Time
    "AEST": 10,  # Australian Eastern Standard Time
    "AEDT": 11,  # Australian Eastern Daylight Time
    "NZST": 12,  # New Zealand Standard Time
    "NZDT": 13,  # New Zealand Daylight Time

    # Other
    "AST": -4,   # Atlantic Standard Time
    "ADT": -3,   # Atlantic Daylight Time
    "CHAST": 12.75, # Chatham Standard Time
    "CHADT": 13.75, # Chatham Daylight Time
}
