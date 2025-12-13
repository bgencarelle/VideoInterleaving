#ascii_constants.py - configuration for ascii web and telnet modes.

ASCII_PORT = 2323
ASCII_HOST = '127.0.0.1'
ASCII_WIDTH = 90
ASCII_HEIGHT = 60
ASCII_FONT_RATIO = .5

# --- ARTISTIC TWEAKS ---
ASCII_CONTRAST   = 1.2  # tiny bit of punch
ASCII_SATURATION = .9   # slightly muted color
ASCII_BRIGHTNESS = 1.3   # handled in HSV
ASCII_GAMMA      = 1  # neutral given your LUT definition

ASCII_PADDING_CHAR = " "     # <--- NEW: Character for pillar/letterboxing

# --- THE GENCARELLE PALETTE ---
ASCII_PALETTE_LIGHT = "MWB8GRDNHESAVTOLPmevncray97stji1-/., "
ASCII_PALETTE_DARK = " ,.1ijts79yarcnvemCPLOTVASEHNDRG8BWM"
ASCII_PALETTE = ASCII_PALETTE_LIGHT
