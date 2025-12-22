#ascii_constants.py - configuration for ascii web and telnet modes.

ASCII_PORT = 2323
ASCII_HOST = '0.0.0.0'
ASCII_FONT_RATIO = .5

# --- ARTISTIC TWEAKS ---
ASCII_CONTRAST   = 1.2  # tiny bit of punch
ASCII_SATURATION = .9   # slightly muted color
ASCII_BRIGHTNESS = 1.4   # handled in HSV
ASCII_GAMMA      = .9  # neutral given your LUT definition

# Optional pre-HSV grading for ASCII output. Defaults keep the legacy path
# untouched until explicitly enabled.
ASCII_ENABLE_CONTRAST = False
ASCII_ENABLE_RGB_BRIGHTNESS = False
ASCII_RGB_BRIGHTNESS = (1.0, 1.0, 1.0)

ASCII_PADDING_CHAR = " "     # <--- NEW: Character for pillar/letterboxing

# --- THE GENCARELLE PALETTE ---
ASCII_PALETTE_LIGHT = "MWB8GRDNHESAVTOLPmevncray97stji1-/., "
ASCII_PALETTE_DARK = " ,.1ijts79yarcnvemCPLOTVASEHNDRG8BWM"
ASCII_PALETTE = ASCII_PALETTE_LIGHT
