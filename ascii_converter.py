import numpy as np
import cv2
import settings

# 1. LOAD PALETTE
_raw_chars = getattr(settings, 'ASCII_PALETTE', 'MB8NG9SEaemvyznocrtlj17i. ')
CHARS = np.asarray(list(_raw_chars))

# 2. COLOR & GAMMA TABLES
_ansi_colors = [f"\033[38;5;{i}m" for i in range(256)]
_ansi_colors[16] = "\033[38;5;235m"
ANSI_LUT = _ansi_colors  # Keep as plain list for indexed access in row builder
RESET_CODE = "\033[0m"

def _build_gamma_lut(gamma):
    return np.array([((i / 255.0) ** gamma) * 255 for i in range(256)],
                    dtype=np.uint8)


_gamma_val = getattr(settings, 'ASCII_GAMMA', 1.0)
GAMMA_LUT = _build_gamma_lut(_gamma_val)


def _refresh_gamma_lut():
    """Rebuild GAMMA_LUT when ASCII_GAMMA has moved since it was built.

    The LUT used to be baked once, at import. That made ASCII_GAMMA settable
    only by editing the constant BEFORE anything imported this module -- so a
    command-line flag for it would have worked or not depending purely on
    import order, which is the same trap display_manager had with
    INITIAL_ROTATION. Verified before this changed: setting
    settings.ASCII_GAMMA after import left the picture untouched.

    Gated on the VALUE, not called unconditionally, so the 256-entry rebuild
    happens when someone actually changes gamma and not once per frame -- and
    so a caller that patches GAMMA_LUT directly, as the tests do, keeps its
    patch as long as it has not also moved ASCII_GAMMA.
    """
    global GAMMA_LUT, _gamma_val
    current = getattr(settings, 'ASCII_GAMMA', 1.0)
    if current != _gamma_val:
        _gamma_val = current
        GAMMA_LUT = _build_gamma_lut(current)


def _build_colored_rows(ansi_ids, char_array, max_rows, max_cols):
    """
    Build output rows with run-length color encoding.
    Only emits a color escape when the color changes from the previous character.
    """
    rows = []
    for r in range(max_rows):
        parts = []
        last_id = -1
        id_row = ansi_ids[r]
        ch_row = char_array[r]
        for c in range(max_cols):
            cid = id_row[c]
            if cid != last_id:
                parts.append(ANSI_LUT[cid])
                last_id = cid
            parts.append(ch_row[c])
        rows.append("".join(parts))
    return rows


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

    # --- Step B: Color Grading (Now on the final max_cols x max_rows pixel count) ---
    hsv = cv2.cvtColor(frame_cropped, cv2.COLOR_RGB2HSV).astype(float)
    if sat_mult != 1.0: hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_mult, 0, 255)
    # Contrast, then brightness, both on V, both exactly once.
    #
    # ASCII_CONTRAST used to be read by a SECOND grading stage that ran before
    # this one -- and that stage referenced three module globals that were
    # never defined, so to_ascii() raised NameError on every call until commit
    # 9adda652 deleted it. The setting outlived the code and has done nothing
    # since. It lives here now, in the one stage that exists.
    #
    # Scaling about mid-grey rather than about zero is what makes this contrast
    # rather than brightness: 1.0 is exactly neutral (v - 128 + 128 == v), above
    # 1.0 pushes lights and darks apart, below pulls them together. That
    # neutrality is load-bearing -- it is what lets the setting become live
    # without changing a single character of anyone's existing output.
    if contrast_mult != 1.0:
        hsv[:, :, 2] = np.clip((hsv[:, :, 2] - 128.0) * contrast_mult + 128.0,
                               0, 255)
    if bright_mult != 1.0:
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * bright_mult, 0, 255)
    frame_boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # --- Step C & D: Map and Compose ---
    gray = cv2.cvtColor(frame_boosted, cv2.COLOR_RGB2GRAY)
    _refresh_gamma_lut()          # no-op unless ASCII_GAMMA actually moved
    gray = cv2.LUT(gray, GAMMA_LUT)
    indices = ((255 - gray) / 255 * (len(CHARS) - 1)).astype(int)
    char_array = CHARS[indices]

    if getattr(settings, 'ASCII_COLOR', False):
        # Optional: blur color channels to create longer same-color runs.
        # Characters (detail) stay sharp — only the color tinting is softened.
        # 0 = off, odd integer = kernel size (3 = subtle, 5 = moderate, 7 = heavy)
        color_blur = getattr(settings, 'ASCII_COLOR_BLUR', 0)
        if color_blur > 0:
            color_source = cv2.GaussianBlur(frame_boosted, (color_blur, color_blur), 0)
        else:
            color_source = frame_boosted

        small_frame = color_source.astype(int)
        r, g, b = small_frame[:,:,0], small_frame[:,:,1], small_frame[:,:,2]
        ansi_ids = 16 + (36 * (r * 5 // 255)) + (6 * (g * 5 // 255)) + (b * 5 // 255)
        rows = _build_colored_rows(ansi_ids, char_array, max_rows, max_cols)
    else:
        rows = ["".join(row) for row in char_array]

    # --- 4. OUTPUT ---
    return "\r\n".join(rows) + RESET_CODE
