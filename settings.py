# settings.py

# -------------------------
# Display Mode & Performance
# -------------------------
FULLSCREEN_MODE = True

# Frames per Second and Images Per Second (IPS)
IPS = 60
FPS = 120

# Buffer settings: The BUFFER_SIZE is derived from IPS (e.g., 15 if IPS == 60)
BUFFER_SIZE = IPS // 2
PINGPONG = True
FROM_BIRTH = True

# -------------------------
# Clock Mode Constants
# -------------------------
MTC_CLOCK = 0
MIDI_CLOCK = 1
MIXED_CLOCK = 2
CLIENT_MODE = 3
FREE_CLOCK = 255

# For testing purposes.
TEST_MODE = False

# -------------------------
# Image Transformation Settings
# -------------------------
INITIAL_ROTATION = 270
INITIAL_MIRROR = 0

# -------------------------
# Timing and Clock Buffer Parameters
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

# -------------------------
# Image Directories and Folder Paths
# -------------------------
IMAGES_DIR = "images"
MAIN_FOLDER_PATH = "images/float"
FLOAT_FOLDER_PATH = "images/face"

# -------------------------
# Texture Settings
# -------------------------
TEXTURE_FILTER_TYPE = "LINEAR"  # Options: "NEAREST", "LINEAR", "MIPMAP"
