# settings.py
# -------------------------
# Image Directories and Folder Paths
# settings.py
IMAGES_DIR = "images_1080"

MAIN_FOLDER_PATH = f"{IMAGES_DIR}/float"
FLOAT_FOLDER_PATH = f"{IMAGES_DIR}/face"

# -------------------------
# Display Mode & Performance
# -------------------------
FULLSCREEN_MODE = True

# Frames per Second and Images Per Second (IPS)
IPS = 30
FPS = 60

# Buffer settings: The BUFFER_SIZE is derived from IPS (e.g., 15 if IPS == 60)
TOLERANCE = 10
FIFO_LENGTH = 30
PINGPONG = True
FROM_BIRTH = True

TEST_MODE = True

# web stuff
HTTP_MONITOR = True  # or True
WEB_PORT = 1978 # web port is year of birth

# For testing purposes.
if TEST_MODE:
    FRAME_COUNTER_DISPLAY = not HTTP_MONITOR  # If the monitor is on, we skip printing
    SHOW_DELTA = not HTTP_MONITOR
else:
    FRAME_COUNTER_DISPLAY = False
    SHOW_DELTA = False

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
CONNECTED_TO_RCA_HDMI = True # this is for analog tvs
# In settings.py, add near the other display parameters:
RCA_HDMI_RESOLUTION = (640, 480)
LOW_RES_FULLSCREEN = False        # Set to True to force low resolution mode
LOW_RES_FULLSCREEN_RESOLUTION = (960, 600)

# -------------------------
# Timing and Clock Buffer Parameters - midi
# -------------------------
TIMEOUT_SECONDS = 1  # Set the timeout value as needed
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

