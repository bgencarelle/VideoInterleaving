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


def to_ascii(frame, source_aspect_ratio=None):
    """
    Converts a frame to ASCII using a 'Fit' (Letterbox/Pillarbox) scaling method.
    Scales image to fit within terminal bounds while preserving aspect ratio.
    Includes Contrast, RGB Brightness, and Saturation adjustments.
    """
    if frame is None:
        return ""

    # --- 1. GET CONSTRAINTS ---
    max_cols = getattr(settings, 'ASCII_WIDTH', 90)
    max_rows = getattr(settings, 'ASCII_HEIGHT', 60)
    font_ratio = getattr(settings, 'ASCII_FONT_RATIO', 0.5)
    padding_char = getattr(settings, 'ASCII_PADDING_CHAR', ' ')

    # --- 2. CALCULATE GEOMETRY (FIT Scaling) ---
    h, w = frame.shape[:2]
    
    # Use source_aspect_ratio if provided (from first frame), otherwise calculate from current frame
    if source_aspect_ratio is not None:
        image_aspect = source_aspect_ratio
    else:
        image_aspect = w / h if h > 0 else 1.0
    
    # Calculate terminal display aspect ratio (accounting for font ratio)
    terminal_aspect = (max_cols * font_ratio) / max_rows if max_rows > 0 else 1.0
    
    # Calculate effective font ratio to match image aspect ratio
    effective_font_ratio = font_ratio * (image_aspect / terminal_aspect) if terminal_aspect > 0 else font_ratio

    # Calculate scale to FIT (not cover) - use min to ensure image fits within bounds
    scale_x = max_cols / w if w > 0 else 1.0
    scale_y = max_rows / (h * effective_font_ratio) if h > 0 else 1.0
    scale = min(scale_x, scale_y)  # FIT instead of COVER

    # Calculate scaled dimensions that fit within terminal bounds
    scaled_w = int(w * scale)
    scaled_h = int(h * scale * effective_font_ratio)
    
    # Ensure minimum size
    scaled_w = max(1, scaled_w)
    scaled_h = max(1, scaled_h)

    # --- 3. RESIZE IMAGE ---
    # Resize frame to the calculated scaled size
    frame_resized = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)

    # --- 4. ADD LETTERBOXING/PILLARBOXING ---
    # Calculate padding (letterbox = top/bottom, pillarbox = left/right)
    pad_x = (max_cols - scaled_w) // 2
    pad_y = (max_rows - scaled_h) // 2
    
    # Create padded frame with background color (black)
    frame_cropped = np.zeros((max_rows, max_cols, 3), dtype=np.uint8)
    
    # Place resized image in center
    frame_cropped[pad_y:pad_y + scaled_h, pad_x:pad_x + scaled_w] = frame_resized

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
    gray = cv2.cvtColor(frame_boosted, cv2.COLOR_RGB2GRAY)
    gray = cv2.LUT(gray, GAMMA_LUT)

    # Map brightness to character index
    # (255 - gray) flips it so Bright Pixels -> Low Index (Dense Chars)
    indices = ((255 - gray) / 255 * (len(CHARS) - 1)).astype(int)
    char_array = CHARS[indices]

    if getattr(settings, 'ASCII_COLOR', False):
        small_frame = frame_boosted.astype(int)
        r, g, b = small_frame[:, :, 0], small_frame[:, :, 1], small_frame[:, :, 2]

        # xterm-256 color mapping
        ansi_ids = 16 + (36 * (r * 5 // 255)) + (6 * (g * 5 // 255)) + (b * 5 // 255)

        # Combine ANSI Color Code + Character
        image_grid = np.char.add(ANSI_LUT[ansi_ids], char_array)
    else:
        image_grid = char_array

    # --- 4. OUTPUT ---
    rows = ["".join(row) for row in image_grid]
    return "\r\n".join(rows) + RESET_CODE
