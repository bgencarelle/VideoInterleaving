import os
import sys
import time
import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor
from PIL import Image  # Requires: pip install pillow

import settings

# --- 1. GLOBAL CONSTANTS ---
_raw_chars = getattr(settings, 'ASCII_PALETTE', 'MB8NG9SEaemvyznocrtlj17i. ')
CHARS = np.asarray(list(_raw_chars))

# Pre-calculate Gamma LUT
_gamma_val = getattr(settings, 'ASCII_GAMMA', 1.0)
GAMMA_LUT = np.array([((i / 255.0) ** _gamma_val) * 255 for i in range(256)], dtype=np.uint8)


def load_image_rgba(filepath):
    """
    Loads an image into a Numpy RGBA array.
    """
    try:
        with Image.open(filepath) as img:
            # FORCE RGBA to preserve transparency
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            return np.array(img)
    except Exception as e:
        # print(f"Error loading {filepath}: {e}") # Silence here, caught in worker
        return None


def image_to_buffer(frame_rgba):
    """
    Converts RGBA numpy array to ASCII buffer data.
    Transparency (Alpha < threshold) becomes ' ' (Space).
    Includes Brightness, Contrast, and Saturation adjustments.
    """
    if frame_rgba is None:
        return None

    # --- A. GEOMETRY & CROP ---
    max_cols = getattr(settings, 'ASCII_WIDTH', 90)
    max_rows = getattr(settings, 'ASCII_HEIGHT', 60)
    font_ratio = getattr(settings, 'ASCII_FONT_RATIO', 0.5)

    h, w = frame_rgba.shape[:2]

    scale_x = max_cols / w
    scale_y = max_rows / (h * font_ratio)
    scale = max(scale_x, scale_y)

    new_w = int(w * scale)
    new_h = int(h * scale * font_ratio)
    new_w, new_h = max(1, new_w), max(1, new_h)

    # High Quality Resizing
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
    frame_resized = cv2.resize(frame_rgba, (new_w, new_h), interpolation=interpolation)

    x_off = (new_w - max_cols) // 2
    y_off = (new_h - max_rows) // 2

    # Crop RGBA
    frame_cropped = frame_resized[y_off: y_off + max_rows, x_off: x_off + max_cols]

    # Split Channels
    rgb_cropped = frame_cropped[:, :, :3].astype(float)  # Float for precision math
    alpha_cropped = frame_cropped[:, :, 3]

    # --- B. COLOR GRADING (Brightness & Contrast) ---
    # We apply this to the RGB channels before converting to HSV or Grayscale

    # 1. CONTRAST (New Feature)
    contrast = getattr(settings, 'ASCII_CONTRAST', 1.0)
    if contrast != 1.0:
        # Formula: Factor * (Pixel - 128) + 128
        rgb_cropped = (rgb_cropped - 128.0) * contrast + 128.0

    # 2. BRIGHTNESS (RGB Multiplier)
    # Applying here often looks more natural than HSV Value boost alone
    bright_mult = getattr(settings, 'ASCII_BRIGHTNESS', 1.0)
    if bright_mult != 1.0:
        rgb_cropped = rgb_cropped * bright_mult

    # Clip values back to valid range after math
    rgb_cropped = np.clip(rgb_cropped, 0, 255).astype(np.uint8)

    # --- C. SATURATION (HSV Space) ---
    sat_mult = getattr(settings, 'ASCII_SATURATION', 1.0)

    # Convert to HSV
    hsv = cv2.cvtColor(rgb_cropped, cv2.COLOR_RGB2HSV).astype(float)

    if sat_mult != 1.0:
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_mult, 0, 255)

    # Convert back to RGB for color mapping
    frame_boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # --- D. GENERATE BUFFERS ---

    # 1. Character Map (Luminance)
    gray = cv2.cvtColor(frame_boosted, cv2.COLOR_RGB2GRAY)

    # Apply Gamma LUT (Non-linear brightness curve)
    gray = cv2.LUT(gray, GAMMA_LUT)

    # Map Grayscale (0-255) to Palette Indices
    # NOTE: We invert the mapping logic here to match standard palettes where
    # Index 0 is often 'Dark/Heavy' and Last Index is 'Light/Empty'.
    # If using 'ASCII_PALETTE_LIGHT' (Dark->Light), bright pixels (255) should map to Low Indices (Dark chars).
    # If using 'ASCII_PALETTE_DARK' (Light->Dark), bright pixels (255) should map to Low Indices (Light chars).
    #
    # Current Default: ASCII_PALETTE_LIGHT = "MW... "
    # We want Bright Pixels -> ' ' (Space) ? NO.
    # On a black terminal, we want Bright Pixels -> 'M' (More pixels lit up).
    # So we need Bright (255) -> Index 0 ('M').

    # (255 - gray) flips it so 255 becomes 0.
    indices = ((255 - gray) / 255 * (len(CHARS) - 1)).astype(int)

    # If the resulting image is "Negative" (Dark pixels became bright chars),
    # swap the palette in settings.py or remove the "255 - " above.
    # Based on your constants, "MW..." is typically density descending.
    char_grid = CHARS[indices]

    # 2. Color ID Map (xterm 256)
    small_frame = frame_boosted.astype(int)
    r, g, b = small_frame[:, :, 0], small_frame[:, :, 1], small_frame[:, :, 2]
    color_ids = 16 + (36 * (r * 5 // 255)) + (6 * (g * 5 // 255)) + (b * 5 // 255)

    # --- E. APPLY TRANSPARENCY ---
    # Anywhere alpha is low, we force the character to be a space ' '
    alpha_mask = alpha_cropped < 50
    char_grid[alpha_mask] = ' '

    return {
        "chars": char_grid,
        "colors": color_ids.astype(np.uint8)
    }


def process_file(args):
    """
    Worker function.
    args: (source_full_path, dest_full_path)
    """
    src, dest = args

    # 1. Ensure destination directory exists (Race condition safe-ish)
    dest_dir = os.path.dirname(dest)
    os.makedirs(dest_dir, exist_ok=True)

    # 2. Load
    img = load_image_rgba(src)
    if img is None:
        return f"Load Error: {src}"

    # 3. Process
    try:
        buffer_data = image_to_buffer(img)
        # 4. Save
        np.savez_compressed(dest, chars=buffer_data["chars"], colors=buffer_data["colors"])
        return None
    except Exception as e:
        return f"Process Error {src}: {e}"


def get_directory_interactive(prompt_text):
    while True:
        path = input(f"{prompt_text}").strip()
        if path: return path  # We validate existence later for output
        print("Please enter a path.")


def main():
    print("--- RECURSIVE ASCII BAKER (Brighter + Contrast Boost) ---")

    # 1. Interactive Input
    input_root = get_directory_interactive("Enter SOURCE root directory: ")
    while not os.path.isdir(input_root):
        print(f"Error: '{input_root}' is not a directory.")
        input_root = get_directory_interactive("Enter SOURCE root directory: ")

    # Suggest an output folder name (Source + _npz)
    default_out = input_root.rstrip(os.sep) + "_npz"
    output_root = input(f"Enter OUTPUT root directory [default: {default_out}]: ").strip()
    if not output_root:
        output_root = default_out

    # 2. Walk the directory tree
    tasks = []
    valid_exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff')

    print(f"\nScanning '{input_root}'...")

    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(valid_exts):
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, input_root)
                dest_path = os.path.join(output_root, os.path.splitext(rel_path)[0] + ".npz")
                tasks.append((src_path, dest_path))

    total_tasks = len(tasks)
    if not total_tasks:
        print("No images found.")
        return

    print(f"Found {total_tasks} images. Baking to '{output_root}'...")
    print(f"Using {os.cpu_count()} CPU cores.\n")

    # 3. Execute
    start_time = time.time()
    errors = []

    # Use ProcessPoolExecutor to utilize all cores
    with ProcessPoolExecutor() as executor:
        # map preserves order
        results = executor.map(process_file, tasks)

        for i, res in enumerate(results):
            # Calculate stats
            count = i + 1
            percent = (count / total_tasks) * 100

            # Status Indicator
            status_char = "."
            if res:
                errors.append(res)
                status_char = "x"

            # Print Progress Bar (Overwrites same line)
            sys.stdout.write(f"\r[ {count}/{total_tasks} ] {percent:5.1f}% | {status_char}")
            sys.stdout.flush()

    duration = time.time() - start_time

    print(f"\n\n--- Done in {duration:.2f}s ---")
    print(f"Processed: {total_tasks}")
    print(f"Errors:    {len(errors)}")

    if errors:
        print("\nErrors:")
        for e in errors: print(e)


if __name__ == "__main__":
    main()