import numpy as np
import cv2
import settings

# 1. LOAD PALETTE
_raw_chars = getattr(settings, 'ASCII_PALETTE', 'MB8NG9SEaemvyznocrtlj17i. ')
CHARS = np.asarray(list(_raw_chars))

# 2. COLOR & GAMMA TABLES
_ansi_colors = [f"\033[38;5;{i}m" for i in range(256)]
_ansi_colors[16] = "\033[38;5;235m"  # Black crush fix
ANSI_LUT = np.array(_ansi_colors)
RESET_CODE = "\033[0m"

_gamma_val = getattr(settings, 'ASCII_GAMMA', 1.0)
GAMMA_LUT = np.array([((i / 255.0) ** _gamma_val) * 255 for i in range(256)], dtype=np.uint8)

# Cache for grayscale conversion (reuse buffer if frame size doesn't change)
_gray_cache = None
_gray_cache_size = None


def to_ascii(frame, source_aspect_ratio=None):
    """
    Converts a frame to ASCII using a 'Cover' (Zoom/Crop) scaling method.
    Includes Contrast, RGB Brightness, and Saturation adjustments.
    """
    if frame is None:
        return ""

    # --- 1. GET CONSTRAINTS ---
    max_cols = getattr(settings, 'ASCII_WIDTH', 90)
    max_rows = getattr(settings, 'ASCII_HEIGHT', 60)
    font_ratio = getattr(settings, 'ASCII_FONT_RATIO', 0.5)

    # --- 2. CALCULATE GEOMETRY (COVER Scaling) ---
    h, w = frame.shape[:2]
    
    # Adjust font_ratio if source aspect ratio is provided (for aspect ratio matching)
    adjusted_font_ratio = font_ratio
    if source_aspect_ratio is not None:
        # Calculate target aspect ratio
        target_aspect = (max_cols * font_ratio) / max_rows
        # Adjust font_ratio to match source aspect ratio
        if source_aspect_ratio != target_aspect:
            adjusted_font_ratio = font_ratio * (source_aspect_ratio / target_aspect)

    # Calculate scale needed to fully COVER the terminal area
    scale_x = max_cols / w
    scale_y = max_rows / (h * adjusted_font_ratio)
    scale = max(scale_x, scale_y)

    # Calculate oversized dimensions
    new_w = int(w * scale)
    new_h = int(h * scale * adjusted_font_ratio)

    new_w = max(1, new_w)
    new_h = max(1, new_h)

    # --- 3. RESIZE AND CROP PIXELS ---
    # Resize frame to the calculated oversized grid
    # Use INTER_NEAREST for speed in realtime, or INTER_LINEAR if affordable
    frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Calculate offset to extract the center max_cols x max_rows area
    x_off = (new_w - max_cols) // 2
    y_off = (new_h - max_rows) // 2

    # Crop the pixel array down to the exact final size before processing
    frame_cropped = frame_resized[y_off: y_off + max_rows, x_off: x_off + max_cols]

    # --- Step B: Color Grading ---
    # Convert to float for math
    rgb_float = frame_cropped.astype(float)

    # 1. CONTRAST
    contrast = getattr(settings, 'ASCII_CONTRAST', 1.0)
    if contrast != 1.0:
        # Formula: Factor * (Pixel - 128) + 128
        rgb_float = (rgb_float - 128.0) * contrast + 128.0

    # 2. BRIGHTNESS (RGB Multiplier)
    bright_mult = getattr(settings, 'ASCII_BRIGHTNESS', 1.0)
    if bright_mult != 1.0:
        rgb_float = rgb_float * bright_mult

    # Clip back to valid range
    rgb_graded = np.clip(rgb_float, 0, 255).astype(np.uint8)

    # 3. SATURATION (HSV)
    sat_mult = getattr(settings, 'ASCII_SATURATION', 1.0)
    if sat_mult != 1.0:
        hsv = cv2.cvtColor(rgb_graded, cv2.COLOR_RGB2HSV).astype(float)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_mult, 0, 255)
        frame_boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    else:
        frame_boosted = rgb_graded

    # --- Step C & D: Map and Compose ---
    # Cache grayscale conversion buffer if frame size hasn't changed
    global _gray_cache, _gray_cache_size
    if _gray_cache is None or _gray_cache_size != frame_cropped.shape[:2]:
        _gray_cache = np.empty(frame_cropped.shape[:2], dtype=np.uint8)
        _gray_cache_size = frame_cropped.shape[:2]
    
    cv2.cvtColor(frame_boosted, cv2.COLOR_RGB2GRAY, dst=_gray_cache)
    gray = cv2.LUT(_gray_cache, GAMMA_LUT)

    # Map brightness to character index
    # (255 - gray) flips it so Bright Pixels -> Low Index (Dense Chars)
    indices = ((255 - gray) / 255 * (len(CHARS) - 1)).astype(int)
    char_array = CHARS[indices]

    if getattr(settings, 'ASCII_COLOR', False):
        # Optimize color mapping: use vectorized operations
        r, g, b = frame_boosted[:, :, 0], frame_boosted[:, :, 1], frame_boosted[:, :, 2]
        
        # Vectorized xterm-256 color mapping (faster than nested loops)
        ansi_ids = 16 + (36 * (r * 5 // 255)) + (6 * (g * 5 // 255)) + (b * 5 // 255)
        ansi_ids = ansi_ids.astype(np.uint8)

        # Combine ANSI Color Code + Character
        image_grid = np.char.add(ANSI_LUT[ansi_ids], char_array)
    else:
        image_grid = char_array

    # --- 4. OUTPUT (OPTIMIZED) ---
    # Optimize string building: use list comprehension with join (already efficient)
    rows = [''.join(row) for row in image_grid]
    return "\r\n".join(rows).strip() + RESET_CODE