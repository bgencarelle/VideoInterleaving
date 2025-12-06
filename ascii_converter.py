import numpy as np
import cv2
import settings

# 1. LOAD PALETTE
_raw_chars = getattr(settings, 'ASCII_PALETTE', 'MB8NG9SEaemvyznocrtlj17i. ')
CHARS = np.asarray(list(_raw_chars))

# 2. PRE-CALCULATE TABLES
ANSI_LUT = np.array([f"\033[38;5;{i}m" for i in range(256)])
RESET_CODE = "\033[0m"

# Gamma LUT
_gamma_val = getattr(settings, 'ASCII_GAMMA', 1.0)
GAMMA_LUT = np.array([
    ((i / 255.0) ** _gamma_val) * 255
    for i in range(256)
], dtype=np.uint8)


def to_ascii(frame):
    if frame is None:
        return ""

    target_w = getattr(settings, 'ASCII_WIDTH', 80)
    target_h = getattr(settings, 'ASCII_HEIGHT', 24)
    font_ratio = getattr(settings, 'ASCII_FONT_RATIO', 0.55)
    pad_char = getattr(settings, 'ASCII_PADDING_CHAR', '_')

    # Artistic Knobs
    sat_mult = getattr(settings, 'ASCII_SATURATION', 1.0)
    bright_mult = getattr(settings, 'ASCII_BRIGHTNESS', 1.0)

    # --- Step A: Aspect-Correct Resize & Masking ---
    h, w = frame.shape[:2]

    # 1. Calculate Scale
    scale_w = target_w / w
    scale_h = target_h / (h * font_ratio)
    scale = min(scale_w, scale_h)

    new_w = int(w * scale)
    new_h = int(h * scale * font_ratio)
    new_w = max(1, min(new_w, target_w))
    new_h = max(1, min(new_h, target_h))

    # 2. Resize Content
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # 3. Create Canvas (Image) and Mask (Padding Tracker)
    # canvas holds pixel data
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # is_padding: 1 = Empty Space, 0 = Image
    # Initialize full True (all padding)
    is_padding = np.ones((target_h, target_w), dtype=bool)

    # 4. Paste Image into Center
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2

    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    is_padding[y_off:y_off + new_h, x_off:x_off + new_w] = False  # Mark this area as NOT padding

    frame = canvas

    # --- Step B: Color Grading (HSV) ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(float)

    if sat_mult != 1.0:
        hsv[:, :, 1] *= sat_mult
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

    if bright_mult != 1.0:
        hsv[:, :, 2] *= bright_mult
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)

    frame_boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # --- Step C: Geometry Mapping ---
    gray = cv2.cvtColor(frame_boosted, cv2.COLOR_RGB2GRAY)
    gray = cv2.LUT(gray, GAMMA_LUT)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # --- Step D: Map to Palette ---
    indices = ((255 - gray) / 255 * (len(CHARS) - 1)).astype(int)
    char_array = CHARS[indices]

    # --- Step E: Composition (Image vs Padding) ---

    # 1. Generate Image Strings (Color or B&W)
    if getattr(settings, 'ASCII_COLOR', False):
        small_frame = frame_boosted.astype(int)
        r = (small_frame[:, :, 0] * 5 // 255)
        g = (small_frame[:, :, 1] * 5 // 255)
        b = (small_frame[:, :, 2] * 5 // 255)
        ansi_ids = 16 + (36 * r) + (6 * g) + b
        color_prefixes = ANSI_LUT[ansi_ids]
        image_grid = np.char.add(color_prefixes, char_array)
    else:
        image_grid = char_array

    # 2. Generate Padding Grid
    # Create an array filled with the padding char
    # If color is enabled, image_grid has ANSI codes. Padding grid has none (colorless).
    padding_grid = np.full(image_grid.shape, pad_char, dtype=image_grid.dtype)

    # 3. Combine using the Mask
    # Where is_padding is True, use padding_grid. Else use image_grid.
    final_grid = np.where(is_padding, padding_grid, image_grid)

    # --- Step F: Borders & Output ---

    # Generate the 2-row border string
    border_line = (pad_char * target_w) + RESET_CODE

    # Create rows
    rows = ["".join(row) + RESET_CODE for row in final_grid]

    # Add top/bottom borders
    final_output = (
            border_line + "\r\n" +
            border_line + "\r\n" +
            "\r\n".join(rows) + "\r\n" +
            border_line + "\r\n" +
            border_line
    )

    return final_output