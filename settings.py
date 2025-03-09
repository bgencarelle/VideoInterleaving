# settings.py
# Display Mode

FULLSCREEN_MODE = False
##
IPS = 30
FPS = 60

BUFFER_SIZE = IPS // 4  # e.g., 15 if IPS==60
PINGPONG = True
FROM_BIRTH = True
# Clock Mode Constants
MTC_CLOCK = 0
MIDI_CLOCK = 1
MIXED_CLOCK = 2
CLIENT_MODE = 3
FREE_CLOCK = 255
CLOCK_MODE = FREE_CLOCK

MIDI_MODE = False

# Valid Clock Modes (for interactive selection)
VALID_MODES = {
    "MTC_CLOCK": MTC_CLOCK,
    "MIDI_CLOCK": MIDI_CLOCK,
    "MIXED_CLOCK": MIXED_CLOCK,
    "CLIENT_MODE": CLIENT_MODE,
    "FREE_CLOCK": FREE_CLOCK,
}
