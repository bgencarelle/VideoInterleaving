#display_constants.py

# Additional Display Settings
BACKGROUND_COLOR = (4, 4, 4)       # Background clear color (RGB)
GAMMA_CORRECTION_ENABLED = False    # Enable gamma correction in fragment shader
ENABLE_SRGB_FRAMEBUFFER = False     # Request sRGB framebuffer if supported

# -------------------------
# Image Transformation Settings
# -------------------------
INITIAL_ROTATION = 0
INITIAL_MIRROR = 0
CONNECTED_TO_RCA_HDMI = False  # this is for analog TVs
RCA_HDMI_RESOLUTION = (640, 480)
LOW_RES_FULLSCREEN = False        # Set to True to force low resolution mode
LOW_RES_FULLSCREEN_RESOLUTION = (960, 600)

# -------------------------
# Display Resolution Optimization
# -------------------------
AUTO_OPTIMIZE_DISPLAY_RESOLUTION = True  # Automatically change display resolution to match small images
RESTORE_DISPLAY_ON_EXIT = True  # Restore original display resolution when application exits
