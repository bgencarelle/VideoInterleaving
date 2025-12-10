import os
import sys
import time
import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

import settings

# --- 1. GLOBAL CONSTANTS ---
_raw_chars = getattr(settings, 'ASCII_PALETTE', 'MB8NG9SEaemvyznocrtlj17i. ')
# FORCE S1 (1-byte string) so we can cast to uint8 easily
CHARS = np.asarray(list(_raw_chars), dtype='S1')

_gamma_val = getattr(settings, 'ASCII_GAMMA', 1.0)
GAMMA_LUT = np.array([((i / 255.0) ** _gamma_val) * 255 for i in range(256)], dtype=np.uint8)


def load_image_rgba(filepath):
    try:
        with Image.open(filepath) as img:
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            return np.array(img)
    except Exception:
        return None


def image_to_stacked_array(frame_rgba):
    """
    Converts RGBA -> Stacked (2, H, W) uint8 array.
    Layer 0: Char ASCII codes
    Layer 1: Color IDs
    """
    if frame_rgba is None: return None

    # --- GEOMETRY ---
    max_cols = getattr(settings, 'ASCII_WIDTH', 90)
    max_rows = getattr(settings, 'ASCII_HEIGHT', 60)
    font_ratio = getattr(settings, 'ASCII_FONT_RATIO', 0.5)

    h, w = frame_rgba.shape[:2]
    scale = max(max_cols / w, max_rows / (h * font_ratio))
    new_w, new_h = int(w * scale), int(h * scale * font_ratio)
    new_w, new_h = max(1, new_w), max(1, new_h)

    # Resize & Crop
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
    frame_resized = cv2.resize(frame_rgba, (new_w, new_h), interpolation=interp)

    x_off = (new_w - max_cols) // 2
    y_off = (new_h - max_rows) // 2
    frame = frame_resized[y_off:y_off + max_rows, x_off:x_off + max_cols]

    # Split
    rgb = frame[:, :, :3].astype(float)
    alpha = frame[:, :, 3]

    # --- COLOR GRADING ---
    contrast = getattr(settings, 'ASCII_CONTRAST', 1.0)
    bright_mult = getattr(settings, 'ASCII_BRIGHTNESS', 1.0)

    if contrast != 1.0: rgb = (rgb - 128.0) * contrast + 128.0
    if bright_mult != 1.0: rgb = rgb * bright_mult
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    # Saturation
    sat_mult = getattr(settings, 'ASCII_SATURATION', 1.0)
    if sat_mult != 1.0:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(float)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_mult, 0, 255)
        frame_boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    else:
        frame_boosted = rgb

    # --- MAPPING ---
    gray = cv2.cvtColor(frame_boosted, cv2.COLOR_RGB2GRAY)
    gray = cv2.LUT(gray, GAMMA_LUT)
    indices = ((255 - gray) / 255 * (len(CHARS) - 1)).astype(int)

    # CHARS GRID (S1 type)
    char_grid = CHARS[indices]

    # Apply Transparency (Space char)
    alpha_mask = alpha < 50
    char_grid[alpha_mask] = b' '  # Note the b for bytes

    # COLOR ID GRID (uint8)
    small = frame_boosted.astype(int)
    r, g, b = small[:, :, 0], small[:, :, 1], small[:, :, 2]
    color_ids = 16 + (36 * (r * 5 // 255)) + (6 * (g * 5 // 255)) + (b * 5 // 255)

    # --- OPTIMIZATION: STACKING ---
    # Convert Chars (S1) -> uint8
    chars_uint8 = char_grid.view(np.uint8)
    colors_uint8 = color_ids.astype(np.uint8)

    # Stack into shape (2, H, W)
    # This creates a single contiguous block of memory
    return np.stack([chars_uint8, colors_uint8], axis=0)


def process_file(args):
    src, dest = args
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    # Check if target exists to skip? (Optional)
    # if os.path.exists(dest): return None

    img = load_image_rgba(src)
    if img is None: return f"Load Error: {src}"

    try:
        # Get the stacked array
        packed_data = image_to_stacked_array(img)

        # Save as RAW .npy (Uncompressed)
        # This is much faster to read later
        np.save(dest, packed_data)
        return None
    except Exception as e:
        return f"Error {src}: {e}"


def get_directory_interactive(prompt_text):
    while True:
        path = input(f"{prompt_text}").strip()
        if path: return path
        print("Please enter a path.")


def main():
    print("--- RECURSIVE ASCII BAKER (Ultra-Fast .npy format) ---")
    input_root = get_directory_interactive("Enter SOURCE root directory: ")
    while not os.path.isdir(input_root):
        print("Invalid directory.")
        input_root = get_directory_interactive("Enter SOURCE root directory: ")

    default_out = input_root.rstrip(os.sep) + "_npy"
    output_root = input(f"Enter OUTPUT root directory [default: {default_out}]: ").strip() or default_out

    tasks = []
    valid_exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    print(f"\nScanning '{input_root}'...")

    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(valid_exts):
                src = os.path.join(root, file)
                rel = os.path.relpath(src, input_root)
                # Change extension to .npy
                dest = os.path.join(output_root, os.path.splitext(rel)[0] + ".npy")
                tasks.append((src, dest))

    total = len(tasks)
    if not total: return print("No images found.")

    print(f"Baking {total} images to '{output_root}'...")
    print("Using .npy format (Uncompressed) for maximum read speed.\n")

    start = time.time()
    errors = []
    with ProcessPoolExecutor() as ex:
        results = ex.map(process_file, tasks)
        for i, res in enumerate(results):
            if res: errors.append(res)
            sys.stdout.write(f"\r[ {i + 1}/{total} ] {((i + 1) / total) * 100:5.1f}%")
            sys.stdout.flush()

    print(f"\n\nDone in {time.time() - start:.2f}s. Errors: {len(errors)}")
    if errors:
        for e in errors: print(e)


if __name__ == "__main__":
    main()