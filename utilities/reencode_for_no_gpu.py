import os
import cv2
import numpy as np
import time
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def process_file(file_path, input_root, output_root, quality=90):
    """
    Reads a WebP file, extracts color/alpha, stacks them side-by-side,
    and saves as a high-quality JPEG.
    """
    try:
        # 1. Load WebP (BGRA) - Unchanged flag preserves alpha channel
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return False, "Load failed"

        # Ensure we have 4 channels (BGRA)
        if img.ndim != 3 or img.shape[2] != 4:
            # If it's just RGB or Grayscale, we can't extract alpha reliably
            # (or it's fully opaque). We could fake it, but better to warn.
            return False, f"Skipping {file_path}: Not BGRA (Channels: {img.shape[2] if img.ndim == 3 else 1})"

        # 2. Extract channels
        # OpenCV loads as BGR, Alpha is index 3
        b, g, r, a = cv2.split(img)

        # 3. Create Side-by-Side canvas (Double Width)
        # Left: Color (BGR)
        color_bgr = cv2.merge([b, g, r])

        # Right: Alpha (Gray -> BGR for JPEG compatibility)
        # We assume alpha is single channel. Merging it 3 times creates a grayscale BGR image.
        alpha_bgr = cv2.merge([a, a, a])

        # Stack horizontally
        sbs = np.hstack([color_bgr, alpha_bgr])

        # 4. Construct Output Path
        # Determine relative path to maintain folder structure
        rel_path = os.path.relpath(file_path, input_root)
        # Change extension to .jpg
        rel_path_jpg = os.path.splitext(rel_path)[0] + ".jpg"
        out_path = os.path.join(output_root, rel_path_jpg)

        # Ensure output dir exists
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # 5. Save as JPEG
        # Use high quality to preserve mask details
        success = cv2.imwrite(out_path, sbs, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

        return success, None

    except Exception as e:
        return False, str(e)


def main():
    print("--- Side-by-Side (SBS) JPEG Converter ---")
    print("Converts WebP (RGBA) -> JPEG (RGB + Alpha Mask side-by-side)")

    # 1. Interactive Input
    default_input = "images"
    input_dir = input(f"Enter source folder path [default: {default_input}]: ").strip()
    if not input_dir:
        input_dir = default_input

    if not os.path.isdir(input_dir):
        print(f"Error: Directory '{input_dir}' not found.")
        return

    default_output = input_dir + "_sbs"
    output_dir = input(f"Enter output folder path [default: {default_output}]: ").strip()
    if not output_dir:
        output_dir = default_output

    quality_str = input("Enter JPEG Quality (1-100) [default: 90]: ").strip()
    quality = int(quality_str) if quality_str.isdigit() else 90

    print(f"\nScanning '{input_dir}' for .webp files...")
    # Recursive glob for all webp files
    # Using recursive=True with ** pattern
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
    # ProcessPoolExecutor is best for CPU-bound tasks like image encoding
    # It avoids the GIL (Global Interpreter Lock)
    start_time = time.time()
    cpu_count = os.cpu_count()
    print(f"\nStarting conversion using {cpu_count} CPU cores...")

    errors = []

    with ProcessPoolExecutor() as executor:
        # Submit all tasks
        # We map futures to filenames for error reporting
        futures = {
            executor.submit(process_file, f, input_dir, output_dir, quality): f
            for f in files
        }

        # Use tqdm for a nice progress bar
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
        for err in errors[:10]:  # Show first 10
            print(f" - {err}")
        if len(errors) > 10:
            print(f" ... and {len(errors) - 10} more.")
    else:
        print("0 Errors.")


if __name__ == "__main__":
    main()