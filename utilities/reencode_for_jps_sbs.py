import os
import cv2
import numpy as np
import time
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def process_file_final(file_path, input_root, output_root, quality=90):
    try:
        # 1. Load Image
        # IMREAD_UNCHANGED lets us see if it has 3 or 4 channels
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            return False, "Load failed"

        h, w = img.shape[:2]

        # 2. Logic to Normalize to [Color | Alpha]
        # We need to construct two arrays: 'color_bgr' and 'alpha_bgr'

        if img.ndim == 2:
            # Case: Grayscale (1 Channel)
            # Color: Convert Gray -> BGR
            color_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # Alpha: Solid White (Fully Visible)
            alpha_bgr = np.full((h, w, 3), 255, dtype=np.uint8)

        elif img.ndim == 3:
            channels = img.shape[2]

            if channels == 4:
                # Case: BGRA (Has Transparency)
                # Split channels (Fast in C++)
                b, g, r, a = cv2.split(img)

                # Recombine BGR
                color_bgr = cv2.merge([b, g, r])

                # Recombine Alpha into a 3-channel grayscale BGR image
                # (We need 3 channels to stack it next to the color image)
                alpha_bgr = cv2.merge([a, a, a])

            elif channels == 3:
                # Case: BGR (No Transparency)
                color_bgr = img
                # Alpha: Solid White
                alpha_bgr = np.full((h, w, 3), 255, dtype=np.uint8)

            else:
                return False, f"Unsupported channel count: {channels}"
        else:
            return False, "Unknown dimensions"

        # 3. Stack Side-by-Side (Horizontal)
        # This creates the [Color | Mask] layout
        sbs = np.hstack([color_bgr, alpha_bgr])

        # 4. Save using OpenCV (Fast C++ I/O)
        rel_path = os.path.relpath(file_path, input_root)
        rel_path_jpg = os.path.splitext(rel_path)[0] + ".jpg"
        out_path = os.path.join(output_root, rel_path_jpg)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # cv2.imwrite is faster because it writes to disk from C++ directly
        success = cv2.imwrite(out_path, sbs, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

        if not success: return False, "Write failed"
        return True, None

    except Exception as e:
        return False, str(e)


def main():
    print("--- Final SBS Converter (Fast I/O + Auto-Alpha) ---")

    default_input = "images"
    input_dir = input(f"Enter source folder path [default: {default_input}]: ").strip() or default_input

    if not os.path.isdir(input_dir):
        print("Directory not found.")
        return

    default_output = input_dir + "_sbs"
    output_dir = input(f"Enter output folder path [default: {default_output}]: ").strip() or default_output

    q_str = input("Quality (1-100) [default: 90]: ").strip()
    quality = int(q_str) if q_str.isdigit() else 90

    files = glob(os.path.join(input_dir, "**", "*.webp"), recursive=True)
    total_files = len(files)

    if total_files == 0: return
    if input(f"Convert {total_files} files? (y/n): ").strip().lower() != 'y': return

    start_time = time.time()
    cpu_count = max(1, os.cpu_count())
    print(f"Engaging {cpu_count} CPU Cores...")

    errors = []

    with ProcessPoolExecutor(max_workers=cpu_count) as executor:
        futures = {
            executor.submit(process_file_final, f, input_dir, output_dir, quality): f
            for f in files
        }

        for future in tqdm(as_completed(futures), total=total_files, unit="img"):
            result = future.result()
            if not result[0]:
                errors.append(f"{futures[future]}: {result[1]}")

    duration = time.time() - start_time
    print(f"\nDone in {duration:.2f}s ({total_files / duration:.1f} fps)")

    if errors:
        print(f"Errors: {len(errors)}")


if __name__ == "__main__":
    main()