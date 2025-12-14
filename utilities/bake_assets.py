import os
import sys
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

# --- 1. SETUP PATHS ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import settings

# Force RGBA for consistency (Main + Float compatibility)
CHANNELS = 4
HEADLESS_RES = getattr(settings, 'HEADLESS_RES', (640, 480))


def get_directory_interactive(prompt_text):
    while True:
        path = input(f"{prompt_text}").strip()
        if path: return path
        print("Please enter a path.")


def process_folder_to_slab(args):
    """
    Reads all images in a folder, resizes them, and writes them
    into a single pre-allocated .npy slab file.
    """
    src_folder, dest_file = args
    src_path = Path(src_folder)

    # 1. Gather Files
    valid = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    files = sorted([p for p in src_path.iterdir() if p.is_file() and p.suffix.lower() in valid])

    if not files:
        return f"Skipped (No images): {src_folder}"

    count = len(files)
    w, h = HEADLESS_RES

    # 2. Pre-allocate Memory Mapped File
    # This creates the full file on disk instantly filled with zeros
    os.makedirs(os.path.dirname(dest_file), exist_ok=True)

    try:
        # Shape: (Frames, Height, Width, RGBA)
        slab = np.lib.format.open_memmap(
            dest_file,
            mode='w+',
            dtype=np.uint8,
            shape=(count, h, w, CHANNELS)
        )

        # 3. Fill the Slab
        for i, fp in enumerate(files):
            with Image.open(fp) as im:
                # Convert & Resize
                im = im.convert('RGBA')
                im = im.resize((w, h), Image.NEAREST)  # Nearest is fastest, use BILINEAR for quality

                # Write directly to disk-backed memory
                slab[i] = np.asarray(im)

        # Flush changes to disk
        slab.flush()
        return None  # Success

    except Exception as e:
        return f"Error processing {src_folder}: {e}"


def main():
    print("--- SLAB BAKER (High Performance Memmap) ---")
    print(f"Target Resolution: {HEADLESS_RES} (RGBA)")
    print("This will pack folders into single .npy files for instant seeking.\n")

    input_root = get_directory_interactive("Enter SOURCE root directory: ")
    while not os.path.isdir(input_root):
        print("Invalid directory.")
        input_root = get_directory_interactive("Enter SOURCE root directory: ")

    clean_path = input_root.rstrip(os.sep)
    output_root = f"{clean_path}_slab"

    print(f"\nSource: {input_root}")
    print(f"Dest:   {output_root}")

    tasks = []

    print("Scanning folders...")
    for root, dirs, files in os.walk(input_root):
        # Check if this folder contains images
        has_images = any(f.lower().endswith(('.png', '.jpg', '.webp')) for f in files)

        if has_images:
            rel_path = os.path.relpath(root, input_root)
            dest_file = os.path.join(output_root, rel_path, "frames.npy")
            tasks.append((root, dest_file))

    total = len(tasks)
    if total == 0:
        print("No image folders found.")
        return

    print(f"Found {total} folders to pack. Starting bake...")

    start = time.time()
    errors = []

    # Run in parallel
    with ProcessPoolExecutor() as ex:
        for i, res in enumerate(ex.map(process_folder_to_slab, tasks)):
            if res: errors.append(res)

            # Simple progress
            percent = ((i + 1) / total) * 100
            sys.stdout.write(f"\rProgress: {percent:.1f}% ({i + 1}/{total})")
            sys.stdout.flush()

    print(f"\n\nDone in {time.time() - start:.2f}s.")
    if errors:
        print(f"Errors ({len(errors)}):")
        for e in errors: print(e)


if __name__ == "__main__":
    import time

    main()