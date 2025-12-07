import numpy as np
import cv2
import settings

# 1. LOAD PALETTE
_raw_chars = getattr(settings, 'ASCII_PALETTE', 'MB8NG9SEaemvyznocrtlj17i. ')
CHARS = np.asarray(list(_raw_chars))

# 2. PRE-CALCULATE TABLES
# --- COLOR FIX: Remap pure black (16) to Dark Grey (235) to prevent invisible text ---
_ansi_colors = [f"\033[38;5;{i}m" for i in range(256)]
_ansi_colors[16] = "\033[38;5;235m"  # Remap Black to Dark Grey
ANSI_LUT = np.array(_ansi_colors)

RESET_CODE = "\033[0m"

# Gamma LUT
_gamma_val = getattr(settings, 'ASCII_GAMMA', 1.0)
GAMMA_LUT = np.array([
    ((i / 255.0) ** _gamma_val) * 255
    for i in range(256)
], dtype=np.uint8)


def to_ascii(frame):
    """
    Converts a frame to an ASCII string.
    Uses 'Bounding Box' logic: output size <= Max Width/Height.
    No padding rows are added.
    """
    if frame is None:
        return ""

    # These are now Maximum Constraints (Bounding Box)
    max_w = getattr(settings, 'ASCII_WIDTH', 90)
    max_h = getattr(settings, 'ASCII_HEIGHT', 60)
    font_ratio = getattr(settings, 'ASCII_FONT_RATIO', 0.5)

    # Artistic Knobs
    sat_mult = getattr(settings, 'ASCII_SATURATION', 1.0)
    bright_mult = getattr(settings, 'ASCII_BRIGHTNESS', 1.0)

    # --- Step A: Aspect-Correct Resize (No Canvas) ---
    h, w = frame.shape[:2]

    # 1. Calculate Scale to fit INSIDE the bounding box
    scale_w = max_w / w
    scale_h = max_h / (h * font_ratio)
    scale = min(scale_w, scale_h)

    new_w = int(w * scale)
    new_h = int(h * scale * font_ratio)

    # Clamp to ensure at least 1x1
    new_w = max(1, min(new_w, max_w))
    new_h = max(1, min(new_h, max_h))

    # 2. Resize Content
    # We process this resized frame directly.
    frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # --- Step B: Color Grading ---
    hsv = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2HSV).astype(float)
    if sat_mult != 1.0:
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_mult, 0, 255)
    if bright_mult != 1.0:
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * bright_mult, 0, 255)
    frame_boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # --- Step C: Map to Chars ---
    gray = cv2.cvtColor(frame_boosted, cv2.COLOR_RGB2GRAY)
    gray = cv2.LUT(gray, GAMMA_LUT)

    # Invert mapping: Bright = Dense, Dark = Sparse
    indices = ((255 - gray) / 255 * (len(CHARS) - 1)).astype(int)
    char_array = CHARS[indices]

    # --- Step D: Composition ---
    if getattr(settings, 'ASCII_COLOR', False):
        small_frame = frame_boosted.astype(int)

        # Standard 6x6x6 Color Cube mapping
        r = (small_frame[:, :, 0] * 5 // 255)
        g = (small_frame[:, :, 1] * 5 // 255)
        b = (small_frame[:, :, 2] * 5 // 255)
        ansi_ids = 16 + (36 * r) + (6 * g) + b

        color_prefixes = ANSI_LUT[ansi_ids]
        image_grid = np.char.add(color_prefixes, char_array)
    else:
        image_grid = char_array

    # --- Step E: Output String ---
    # Since we removed the canvas, we simply join the image rows.
    rows = ["".join(row) for row in image_grid]

    # Return the block.
    # The client/server handles the cursor position (Home).
    return "\r\n".join(rows) + RESET_CODE