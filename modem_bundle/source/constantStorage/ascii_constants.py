#ascii_constants.py - configuration for ascii web and telnet modes.

ASCII_PORT = 2323
ASCII_HOST = '0.0.0.0'
ASCII_FONT_RATIO = .5

# --- ARTISTIC TWEAKS ---
# Applied in ascii_converter.to_ascii, once each: saturation on S, then
# contrast and brightness on V, then gamma on the grey the character comes from.
ASCII_CONTRAST   = 1.0  # 1.0 = neutral; >1 more punch, <1 flatter
ASCII_SATURATION = .9   # slightly muted color
ASCII_BRIGHTNESS = 1.4   # handled in HSV
ASCII_GAMMA      = .9  # neutral given your LUT definition

# ASCII_CONTRAST read 1.2 here for a long time and did nothing: the stage that
# consumed it referenced globals that were never defined, so it raised
# NameError on every frame until 9adda652 deleted it -- and the setting was
# left behind. It is wired up now, so it starts at 1.0, which is exactly
# neutral, and the shipped picture is byte-identical to before. Set it to 1.2
# for the punch it always claimed to have.
#
# ASCII_ENABLE_CONTRAST, ASCII_ENABLE_RGB_BRIGHTNESS and ASCII_RGB_BRIGHTNESS
# belonged to that same stage. Nothing read them, and the per-channel RGB tint
# they gated exists nowhere else, so they are gone rather than left lying.

ASCII_PADDING_CHAR = " "     # <--- NEW: Character for pillar/letterboxing

# --- THE GENCARELLE PALETTE ---
ASCII_PALETTE_LIGHT = "MWB8GRDNHESAVTOLPmevncray97stji1-/., "
ASCII_PALETTE_DARK = " ,.1ijts79yarcnvemCPLOTVASEHNDRG8BWM"
ASCII_PALETTE = ASCII_PALETTE_LIGHT
