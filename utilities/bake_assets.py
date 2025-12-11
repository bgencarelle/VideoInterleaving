import os
import sys
import time
import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

# --- 1. SETUP PATHS ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import settings

# --- 2. CONSTANTS ---
_raw_chars = getattr(settings, 'ASCII_PALETTE', 'MB8NG9SEaemvyznocrtlj17i. ')
CHARS = np.asarray(list(_raw_chars), dtype='S1')

_gamma_val = getattr(settings, 'ASCII_GAMMA', 1.0)
GAMMA_LUT = np.array([((i / 255.0) ** _gamma_val) * 255 for i in range(256)], dtype=np.uint8)

HEADLESS_RES = getattr(settings, 'HEADLESS_RES', (640, 480))


def load_image_rgba(filepath):
    try:
        with Image.open(filepath) as img:
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            return np.array(img)
    except:
        return None


# --- GENERATORS ---
def generate_ascii_stack(frame_rgba):
    max_cols = getattr(settings, 'ASCII_WIDTH', 90)
    max_rows = getattr(settings, 'ASCII_HEIGHT', 60)
    font_ratio = getattr(settings, 'ASCII_FONT_RATIO', 0.5)

    h, w = frame_rgba.shape[:2]
    scale = max(max_cols / w, max_rows / (h * font_ratio))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale * font_ratio))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
    frame = cv2.resize(frame_rgba, (nw, nh), interpolation=interp)

    dx, dy = (nw - max_cols) // 2, (nh - max_rows) // 2
    frame = frame[dy:dy + max_rows, dx:dx + max_cols]

    rgb, alpha = frame[:, :, :3].astype(float), frame[:, :, 3]

    contrast = getattr(settings, 'ASCII_CONTRAST', 1.0)
    bright = getattr(settings, 'ASCII_BRIGHTNESS', 1.0)
    sat = getattr(settings, 'ASCII_SATURATION', 1.0)

    if contrast != 1.0: rgb = (rgb - 128.0) * contrast + 128.0
    if bright != 1.0: rgb *= bright
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    if sat != 1.0:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(float)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat, 0, 255)
        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.LUT(gray, GAMMA_LUT)
    indices = ((255 - gray) / 255 * (len(CHARS) - 1)).astype(int)

    char_grid = CHARS[indices]
    char_grid[alpha < 50] = b' '

    small = rgb.astype(int)
    c_ids = 16 + (36 * (small[:, :, 0] * 5 // 255)) + (6 * (small[:, :, 1] * 5 // 255)) + (small[:, :, 2] * 5 // 255)

    return np.stack([char_grid.view(np.uint8), c_ids.astype(np.uint8)], axis=0)


def generate_headless_img(frame_rgba, posterize=False):
    # 1. Resize
    resized = cv2.resize(frame_rgba, HEADLESS_RES, interpolation=cv2.INTER_AREA)

    # 2. Optimize Channels (Strip Alpha if opaque)
    if resized.shape[2] == 4 and np.min(resized[:, :, 3]) == 255:
        final_img = resized[:, :, :3]
    else:
        final_img = resized

    # 3. Posterize (Entropy Reduction)
    if posterize:
        if final_img.shape[2] == 4:
            rgb = final_img[:, :, :3]
            alpha = final_img[:, :, 3]
            rgb = np.bitwise_and(rgb, 0xFC)  # Zero out lowest 2 bits
            final_img = np.dstack((rgb, alpha))
        else:
            final_img = np.bitwise_and(final_img, 0xFE)

    return final_img


def process_hybrid(args):
    src, ascii_dest, img_dest, do_posterize = args

    os.makedirs(os.path.dirname(ascii_dest), exist_ok=True)
    os.makedirs(os.path.dirname(img_dest), exist_ok=True)

    img = load_image_rgba(src)
    if img is None: return f"Load Error: {src}"

    try:
        # 1. ASCII (.npy)
        np.save(ascii_dest, generate_ascii_stack(img))

        # 2. HEADLESS (.npz)
        optimized_img = generate_headless_img(img, posterize=do_posterize)
        np.savez_compressed(img_dest, image=optimized_img)

        return None
    except Exception as e:
        return f"Error {src}: {e}"


def get_directory_interactive(prompt_text):
    while True:
        path = input(f"{prompt_text}").strip()
        if path: return path
        print("Please enter a path.")


def main():
    print("--- HYBRID ASSET BAKER (v6 Interactive) ---")

    input_root = get_directory_interactive("Enter SOURCE root directory: ")
    while not os.path.isdir(input_root):
        print("Invalid directory.")
        input_root = get_directory_interactive("Enter SOURCE root directory: ")

    # OPTIMIZATION SWITCH
    opt_choice = input("Enable bit-depth reduction (smaller files, slightly lossy)? [Y/n]: ").strip().lower()
    do_posterize = opt_choice != 'n'

    suffix = "_opt" if do_posterize else ""

    # Clean input path to get base name
    clean_path = input_root.rstrip(os.sep)

    # 1. ASCII Naming: Name_WxH_ascii
    w_asc = getattr(settings, 'ASCII_WIDTH', 90)
    h_asc = getattr(settings, 'ASCII_HEIGHT', 60)
    ascii_root = f"{clean_path}_{w_asc}x{h_asc}_ascii"

    # 2. Headless Naming: Name_WxH_headless[_opt]
    w_head, h_head = HEADLESS_RES
    headless_root = f"{clean_path}_{w_head}x{h_head}_headless{suffix}"

    print(f"\nConfiguration:")
    print(f"  Posterization: {'ENABLED' if do_posterize else 'DISABLED'}")
    print(f"\nOutputs:")
    print(f"  ASCII    (.npy) -> {ascii_root}")
    print(f"  HEADLESS (.npz) -> {headless_root}")

    confirm = input("\nProceed? [Y/n]: ").lower()
    if confirm == 'n': return

    tasks = []
    valid_exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    print(f"\nScanning '{input_root}'...")

    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(valid_exts):
                src = os.path.join(root, file)
                rel = os.path.relpath(src, input_root)

                # ASCII DEST
                d_npy = os.path.join(ascii_root, os.path.splitext(rel)[0])

                # HEADLESS DEST
                d_npz = os.path.join(headless_root, os.path.splitext(rel)[0])

                tasks.append((src, d_npy, d_npz, do_posterize))

    total = len(tasks)
    if not total: return print("No images found.")

    print(f"Baking {total} images using {os.cpu_count()} cores...")
    start = time.time()
    errors = []
    processed_count = 0

    with ProcessPoolExecutor() as ex:
        futures = {ex.submit(process_hybrid, t): t for t in tasks}
        for future in futures:
            res = future.result()
            processed_count += 1
            if res: errors.append(res)

            percent = (processed_count / total) * 100
            bar_len = 30
            filled = int(bar_len * processed_count // total)
            bar = '█' * filled + '-' * (bar_len - filled)
            sys.stdout.write(f'\rProgress: |{bar}| {percent:.1f}% ({processed_count}/{total})')
            sys.stdout.flush()

    print(f"\n\nDone in {time.time() - start:.2f}s.")
    print(f"Errors: {len(errors)}")
    if errors:
        for e in errors: print(e)


if __name__ == "__main__":
    main()