import numpy as np
import cv2
import settings

# 1. LOAD PALETTE
_raw_chars = getattr(settings, 'ASCII_PALETTE', 'MB8NG9SEaemvyznocrtlj17i. ')
CHARS = np.asarray(list(_raw_chars))

# 2. COLOR & GAMMA TABLES
_ansi_colors = [f"\033[38;5;{i}m" for i in range(256)]
_ansi_colors[16] = "\033[38;5;235m"
ANSI_LUT = np.array(_ansi_colors)
RESET_CODE = "\033[0m"

_gamma_val = getattr(settings, 'ASCII_GAMMA', 1.0)
GAMMA_LUT = np.array([((i / 255.0) ** _gamma_val) * 255 for i in range(256)], dtype=np.uint8)

_apply_contrast = getattr(settings, 'ASCII_ENABLE_CONTRAST', False)
_contrast_factor = getattr(settings, 'ASCII_CONTRAST', 1.0)
CONTRAST_LUT = None
if _apply_contrast and _contrast_factor != 1.0:
    CONTRAST_LUT = np.clip(((np.arange(256) - 128) * _contrast_factor) + 128, 0, 255).astype(np.uint8)

_apply_rgb_brightness = getattr(settings, 'ASCII_ENABLE_RGB_BRIGHTNESS', False)
_rgb_brightness = np.array(getattr(settings, 'ASCII_RGB_BRIGHTNESS', (1.0, 1.0, 1.0)), dtype=float)
if _rgb_brightness.shape != (3,):
    _rgb_brightness = np.ones(3, dtype=float)

def to_ascii(frame):
    """
    Converts a frame to ASCII using a 'Cover' (Zoom/Crop) scaling method.
    Optimized to crop the image pixels *before* color grading for efficiency.
    """
    if frame is None:
        return ""

    # --- 1. GET CONSTRAINTS ---
    max_cols = getattr(settings, 'ASCII_WIDTH', 90)
    max_rows = getattr(settings, 'ASCII_HEIGHT', 60)
    font_ratio = getattr(settings, 'ASCII_FONT_RATIO', 0.5)

    sat_mult = getattr(settings, 'ASCII_SATURATION', 1.0)
    contrast_mult = getattr(settings, 'ASCII_CONTRAST', 1.0)
    bright_mult = getattr(settings, 'ASCII_BRIGHTNESS', 1.0)

    # --- 2. CALCULATE GEOMETRY (COVER Scaling) ---
    h, w = frame.shape[:2]

    # Calculate scale needed to fully COVER the terminal area
    scale_x = max_cols / w
    scale_y = max_rows / (h * font_ratio)
    scale = max(scale_x, scale_y)

    # Calculate oversized dimensions
    new_w = int(w * scale)
    new_h = int(h * scale * font_ratio)

    new_w = max(1, new_w)
    new_h = max(1, new_h)

    # --- 3. RESIZE AND CROP PIXELS ---
    # Resize frame to the calculated oversized grid
    frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Calculate offset to extract the center max_cols x max_rows area
    x_off = (new_w - max_cols) // 2
    y_off = (new_h - max_rows) // 2

    # [CHANGE] Crop the pixel array down to the exact final size before processing
    frame_cropped = frame_resized[y_off : y_off + max_rows, x_off : x_off + max_cols]

    # Apply optional pre-HSV grading for ASCII output
    graded_frame = frame_cropped
    pre_hsv_adjustment = False
    if CONTRAST_LUT is not None:
        graded_frame = cv2.LUT(graded_frame, CONTRAST_LUT)
        pre_hsv_adjustment = True
    if _apply_rgb_brightness and not np.allclose(_rgb_brightness, 1.0):
        graded_frame = np.clip(
            graded_frame.astype(np.float32) * _rgb_brightness.reshape(1, 1, 3),
            0,
            255,
        ).astype(np.uint8)
        pre_hsv_adjustment = True

    # --- Step B: Color Grading (Now on the final max_cols x max_rows pixel count) ---
    hsv = cv2.cvtColor(graded_frame, cv2.COLOR_RGB2HSV).astype(float)
    if sat_mult != 1.0: hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_mult, 0, 255)
    if bright_mult != 1.0 and not pre_hsv_adjustment:
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * bright_mult, 0, 255)
    frame_boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # --- Step C & D: Map and Compose ---
    gray = cv2.cvtColor(frame_boosted, cv2.COLOR_RGB2GRAY)
    gray = cv2.LUT(gray, GAMMA_LUT)
    indices = ((255 - gray) / 255 * (len(CHARS) - 1)).astype(int)
    char_array = CHARS[indices]

    if getattr(settings, 'ASCII_COLOR', False):
        small_frame = frame_boosted.astype(int)
        r, g, b = small_frame[:,:,0], small_frame[:,:,1], small_frame[:,:,2]
        ansi_ids = 16 + (36 * (r * 5 // 255)) + (6 * (g * 5 // 255)) + (b * 5 // 255)
        image_grid = np.char.add(ANSI_LUT[ansi_ids], char_array)
    else:
        image_grid = char_array

    # --- 4. OUTPUT ---
    # Output is already guaranteed to be exactly max_rows tall.
    rows = ["".join(row) for row in image_grid]
    return "\r\n".join(rows) + RESET_CODE
