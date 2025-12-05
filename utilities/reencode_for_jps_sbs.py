import os
import cv2
import numpy as np
import time
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def process_file(file_path, input_root, output_root, quality=90):
    """
    Reads a WebP/PNG file, ensures it has an alpha mask (generating one if missing),
    stacks them side-by-side, and saves as a high-quality JPEG.
    """
    try:
        # 1. Load Image
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return False, "Load failed"

        h, w = img.shape[:2]

        color_bgr = None
        alpha_single = None

        # 2. Logic to handle different channel counts
        if img.ndim == 2:
            # Case: Grayscale (1 Channel)
            # Convert Gray -> BGR
            color_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # Generate solid white alpha (fully visible)
            alpha_single = np.full((h, w), 255, dtype=np.uint8)

        elif img.ndim == 3:
            channels = img.shape[2]

            if channels == 4:
                # Case: BGRA (Has Transparency)
                # Split channels
                b, g, r, a = cv2.split(img)
                color_bgr = cv2.merge([b, g, r])
                alpha_single = a

            elif channels == 3:
                # Case: BGR (No Transparency)
                color_bgr = img
                # Generate solid white alpha
                alpha_single = np.full((h, w), 255, dtype=np.uint8)

            else:
                return False, f"Unsupported channel count: {channels}"
        else:
            return False, f"Unknown dimensions: {img.shape}"

        # 3. Prepare Alpha for Stacking
        # We need alpha to be 3 channels (grayscale BGR) to stack next to color
        alpha_bgr = cv2.merge([alpha_single, alpha_single, alpha_single])

        # 4. Stack horizontally: [Color | Alpha]
        sbs = np.hstack([color_bgr, alpha_bgr])

        # 5. Construct Output Path
        rel_path = os.path.relpath(file_path, input_root)
        rel_path_jpg = os.path.splitext(rel_path)[0] + ".jpg"
        out_path = os.path.join(output_root, rel_path_jpg)

        # Ensure output dir exists
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # 6. Save as JPEG
        success = cv2.imwrite(out_path, sbs, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

        if not success:
            return False, "Write failed"

        return True, None

    except Exception as e:
        return False, str(e)


def main():
    print("--- Side-by-Side (SBS) JPEG Converter (Fixed) ---")
    print("Converts WebP/PNG -> JPEG (RGB + Alpha Mask side-by-side)")

    # 1. Interactive Input
    default_input = "images"
    input_dir = input(f"Enter source folder path [default: {default_input}]: ").strip()
    if not input_dir:
        input_dir = default_input

    # FORCE ABSOLUTE PATH to avoid 'missing file' errors
    input_dir = os.path.abspath(input_dir)

    if not os.path.isdir(input_dir):
        print(f"Error: Directory '{input_dir}' not found.")
        return

    default_output = input_dir + "_sbs"
    output_dir = input(f"Enter output folder path [default: {default_output}]: ").strip()
    if not output_dir:
        output_dir = default_output

    # FORCE ABSOLUTE PATH
    output_dir = os.path.abspath(output_dir)

    quality_str = input("Enter JPEG Quality (1-100) [default: 90]: ").strip()
    quality = int(quality_str) if quality_str.isdigit() else 90

    print(f"\nScanning '{input_dir}' for .webp files...")
    files = glob(os.path.join(input_dir, "**", "*.webp"), recursive=True)

    total_files = len(files)
    print(f"Found {total_files} files to process.")

    if total_files == 0:
        print("Nothing to do.")
        return

    confirm = input("Start conversion? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Aborted.")
        return

    # 2. Parallel Processing
    start_time = time.time()
    cpu_count = max(1, os.cpu_count())
    print(f"\nStarting conversion using {cpu_count} CPU cores...")

    errors = []

    with ProcessPoolExecutor(max_workers=cpu_count) as executor:
        futures = {
            executor.submit(process_file, f, input_dir, output_dir, quality): f
            for f in files
        }

        for future in tqdm(as_completed(futures), total=total_files, unit="img"):
            file_path = futures[future]
            try:
                success, error_msg = future.result()
                if not success:
                    errors.append(f"{file_path}: {error_msg}")
            except Exception as e:
                errors.append(f"{file_path}: Crash - {e}")

    end_time = time.time()
    duration = end_time - start_time

    print("\n" + "=" * 40)
    print(f"Processing Complete!")
    print(f"Time taken: {duration:.2f} seconds")
    if duration > 0:
        print(f"Average speed: {total_files / duration:.1f} fps")
    print(f"Output saved to: {output_dir}")

    if errors:
        print(f"\n{len(errors)} Errors encountered:")
        for err in errors[:10]:
            print(f" - {err}")
    else:
        print("0 Errors.")


if __name__ == "__main__":
    main()