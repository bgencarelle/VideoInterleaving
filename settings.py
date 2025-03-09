# settings.py
# Display Mode

FULLSCREEN_MODE = True
# Clock Mode Constants
MTC_CLOCK = 0
MIDI_CLOCK = 1
MIXED_CLOCK = 2
CLIENT_MODE = 3
FREE_CLOCK = 255
CLOCK_MODE = FREE_CLOCK

# Valid Clock Modes (for interactive selection)
VALID_MODES = {
    "MTC_CLOCK": MTC_CLOCK,
    "MIDI_CLOCK": MIDI_CLOCK,
    "MIXED_CLOCK": MIXED_CLOCK,
    "CLIENT_MODE": CLIENT_MODE,
    "FREE_CLOCK": FREE_CLOCK,
}

# Images Per Second (used in timing calculations)
IPS = 30
