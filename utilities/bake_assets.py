import os
import cv2
import numpy as np
import settings  # Assuming your settings.py is in the same folder

# --- 1. SETUP CONSTANTS (Mirrors your original setup) ---
_raw_chars = getattr(settings, 'ASCII_PALETTE', 'MB8NG9SEaemvyznocrtlj17i. ')
CHARS = np.asarray(list(_raw_chars))

_gamma_val = getattr(settings, 'ASCII_GAMMA', 1.0)
GAMMA_LUT = np.array([((i / 255.0) ** _gamma_val) * 255 for i in range(256)], dtype=np.uint8)


def image_to_buffer(frame):
    """
    Performs the Resize -> Crop -> Color Grade -> Map logic.
    Returns: Dictionary containing 'chars' (str array) and 'colors' (int array).
    """
    if frame is None:
        return None

    # --- A. GEOMETRY & CROP ---
    max_cols = getattr(settings, 'ASCII_WIDTH', 90)
    max_rows = getattr(settings, 'ASCII_HEIGHT', 60)
    font_ratio = getattr(settings, 'ASCII_FONT_RATIO', 0.5)

    h, w = frame.shape[:2]

    # Cover Scaling
    scale_x = max_cols / w
    scale_y = max_rows / (h * font_ratio)
    scale = max(scale_x, scale_y)

    new_w = int(w * scale)
    new_h = int(h * scale * font_ratio)
    new_w, new_h = max(1, new_w), max(1, new_h)

    # Resize
    frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Center Crop
    x_off = (new_w - max_cols) // 2
    y_off = (new_h - max_rows) // 2
    frame_cropped = frame_resized[y_off: y_off + max_rows, x_off: x_off + max_cols]

    # --- B. COLOR GRADING ---
    sat_mult = getattr(settings, 'ASCII_SATURATION', 1.0)
    bright_mult = getattr(settings, 'ASCII_BRIGHTNESS', 1.0)

    hsv = cv2.cvtColor(frame_cropped, cv2.COLOR_RGB2HSV).astype(float)
    if sat_mult != 1.0: hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_mult, 0, 255)
    if bright_mult != 1.0: hsv[:, :, 2] = np.clip(hsv[:, :, 2] * bright_mult, 0, 255)
    frame_boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # --- C. GENERATE DATA BUFFERS ---

    # 1. Character Map
    gray = cv2.cvtColor(frame_boosted, cv2.COLOR_RGB2GRAY)
    gray = cv2.LUT(gray, GAMMA_LUT)
    indices = ((255 - gray) / 255 * (len(CHARS) - 1)).astype(int)
    char_grid = CHARS[indices]

    # 2. Color ID Map (The critical storage optimization)
    # Maps RGB directly to 0-255 ANSI integer IDs
    small_frame = frame_boosted.astype(int)
    r, g, b = small_frame[:, :, 0], small_frame[:, :, 1], small_frame[:, :, 2]
    # Standard xterm 256 color mapping
    color_ids = 16 + (36 * (r * 5 // 255)) + (6 * (g * 5 // 255)) + (b * 5 // 255)

    return {
        "chars": char_grid,
        "colors": color_ids.astype(np.uint8)  # Save space by using uint8
    }


def batch_convert(input_folder, output_folder):
    """
    Reads all images in input_folder, converts them, and saves to output_folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]

    print(f"Found {len(files)} images in {input_folder}...")

    for i, filename in enumerate(files):
        # 1. Read Image
        input_path = os.path.join(input_folder, filename)
        frame = cv2.imread(input_path)

        # 2. Convert to Buffer
        buffer_data = image_to_buffer(frame)

        if buffer_data is not None:
            # 3. Save Compressed
            # Change extension to .npz
            name_no_ext = os.path.splitext(filename)[0]
            output_path = os.path.join(output_folder, name_no_ext + ".npz")

            np.savez_compressed(
                output_path,
                chars=buffer_data["chars"],
                colors=buffer_data["colors"]
            )
            print(f"[{i + 1}/{len(files)}] Baked: {filename} -> {output_path}")


# --- EXECUTION ---
if __name__ == "__main__":
    # Example Usage
    INPUT_DIR = "./Mo"  # Folder containing your PNGs
    OUTPUT_DIR = "./assets/ascii"  # Folder to save the .npz files

    batch_convert(INPUT_DIR, OUTPUT_DIR)