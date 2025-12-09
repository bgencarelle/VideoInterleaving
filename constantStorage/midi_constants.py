#midi_onstants.py configuration for midi modes
# -------------------------
# Clock Mode Constants
# -------------------------
MTC_CLOCK = 0
MIDI_CLOCK = 1
MIXED_CLOCK = 2
CLIENT_MODE = 3
FREE_CLOCK = 255


# -------------------------
# Timing and Clock Buffer Parameters - midi
# -------------------------
TIMEOUT_SECONDS = 1  # Set the timeout as needed
CLOCK_BUFFER_SIZE = 50
CLOCK_MODE = FREE_CLOCK


# Valid Clock Modes (for interactive selection or configuration)
VALID_MODES = {
    "MTC_CLOCK": MTC_CLOCK,
    "MIDI_CLOCK": MIDI_CLOCK,
    "MIXED_CLOCK": MIXED_CLOCK,
    "CLIENT_MODE": CLIENT_MODE,
    "FREE_CLOCK": FREE_CLOCK,
}

#---- AUDIO