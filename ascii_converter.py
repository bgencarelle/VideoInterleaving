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


def to_ascii(frame, source_aspect_ratio=None, mode=None):
    """
    Converts a frame to ASCII by filling one dimension and padding the other.
    Scales image to fill either width or height completely, then pads the other dimension.
    Includes Contrast, RGB Brightness, and Saturation adjustments.
    
    Args:
        frame: Input image frame (numpy array)
        source_aspect_ratio: Aspect ratio from first frame (w/h) for consistency
        mode: 'telnet' or 'web' - determines font ratio handling (default: None for backward compatibility)
    """
    if frame is None:
        return ""

    # --- 1. GET CONSTRAINTS ---
    max_cols = getattr(settings, 'ASCII_WIDTH', 90)
    max_rows = getattr(settings, 'ASCII_HEIGHT', 60)
    font_ratio = getattr(settings, 'ASCII_FONT_RATIO', 0.5)
    padding_char = getattr(settings, 'ASCII_PADDING_CHAR', ' ')

    # --- 2. CALCULATE GEOMETRY (FIT Scaling, Then Padding) ---
    h, w = frame.shape[:2]
    
    # 1. Get image aspect ratio (from source if available, otherwise from current frame)
    if source_aspect_ratio is not None:
        image_aspect = source_aspect_ratio
    else:
        image_aspect = w / h if h > 0 else 1.0
    
    # 2. Calculate FIT scaling to fit inside ASCII dimensions while maintaining display aspect ratio
    # We want the DISPLAYED aspect ratio to match the image aspect ratio
    # Display aspect = (scaled_w * font_ratio) / scaled_h = image_aspect
    # And: scaled_w <= max_cols, scaled_h <= max_rows
    # We want to FILL one dimension completely to optimize space usage
    
    # Option 1: Fill width completely
    # scaled_w = max_cols
    # (max_cols * font_ratio) / scaled_h = image_aspect
    # scaled_h = (max_cols * font_ratio) / image_aspect
    scaled_h_if_fill_width = (max_cols * font_ratio) / image_aspect if image_aspect > 0 else max_rows
    
    # Option 2: Fill height completely
    # scaled_h = max_rows
    # (scaled_w * font_ratio) / max_rows = image_aspect
    # scaled_w = (max_rows * image_aspect) / font_ratio
    scaled_w_if_fill_height = (max_rows * image_aspect) / font_ratio if font_ratio > 0 else max_cols
    
    # Use FIT: choose the option that fits within both bounds AND fills one dimension
    if scaled_h_if_fill_width <= max_rows:
        # Filling width fits - use this to maximize width usage
        scaled_w_pixels = max_cols
        scaled_h_pixels = int((max_cols * font_ratio) / image_aspect) if image_aspect > 0 else max_rows
        # Calculate the actual scale factor used
        scale = scaled_w_pixels / w if w > 0 else 1.0
    else:
        # Must fit height instead - fill height completely
        scaled_h_pixels = max_rows
        scaled_w_pixels = int((max_rows * image_aspect) / font_ratio) if font_ratio > 0 else max_cols
        # Calculate the actual scale factor used
        scale = scaled_h_pixels / h if h > 0 else 1.0
    
    # Clamp to terminal bounds (safety check)
    scaled_w_pixels = max(1, min(scaled_w_pixels, max_cols))
    scaled_h_pixels = max(1, min(scaled_h_pixels, max_rows))
    
    # 3. Calculate padding (centered, in pixels)
    pad_x = (max_cols - scaled_w_pixels) // 2
    pad_y = (max_rows - scaled_h_pixels) // 2

    # Debug: Print aspect ratio calculations
    terminal_raw_aspect = max_cols / max_rows if max_rows > 0 else 1.0
    terminal_display_aspect = (max_cols * font_ratio) / max_rows if max_rows > 0 else 1.0
    actual_scaled_display_aspect = (scaled_w_pixels * font_ratio) / scaled_h_pixels if scaled_h_pixels > 0 else 1.0
    print(f"[ASCII] Image aspect: {image_aspect:.4f} | Terminal raw: {terminal_raw_aspect:.4f} | "
          f"Terminal display: {terminal_display_aspect:.4f} | "
          f"Scaled dimensions: {scaled_w_pixels}x{scaled_h_pixels} (display aspect: {actual_scaled_display_aspect:.4f})")

    # --- 3. RESIZE IMAGE ---
    # Resize frame to pixel dimensions
    frame_resized = cv2.resize(frame, (scaled_w_pixels, scaled_h_pixels), interpolation=cv2.INTER_NEAREST)

    # --- 4. ADD PADDING ---
    # Create padded frame with background color (black)
    # frame_cropped is a pixel array: (max_rows, max_cols, 3)
    # Each pixel will become one character in the ASCII output
    frame_cropped = np.zeros((max_rows, max_cols, 3), dtype=np.uint8)
    
    # Place resized image in center
    # frame_resized has shape (scaled_h_pixels, scaled_w_pixels, 3)
    # The slice must match frame_resized dimensions exactly
    frame_cropped[pad_y:pad_y + scaled_h_pixels, pad_x:pad_x + scaled_w_pixels] = frame_resized

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
