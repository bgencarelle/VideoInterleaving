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
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            return False, "Load failed"

        h, w = img.shape[:2]

        # 2. Logic to Normalize to [Color | Alpha]
        if img.ndim == 2:
            # Grayscale
            color_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            alpha_bgr = np.full((h, w, 3), 255, dtype=np.uint8)
        elif img.ndim == 3:
            channels = img.shape[2]
            if channels == 4:
                # BGRA
                b, g, r, a = cv2.split(img)
                color_bgr = cv2.merge([b, g, r])
                alpha_bgr = cv2.merge([a, a, a])
            elif channels == 3:
                # BGR
                color_bgr = img
                alpha_bgr = np.full((h, w, 3), 255, dtype=np.uint8)
            else:
                return False, f"Unsupported channel count: {channels}"
        else:
            return False, "Unknown dimensions"

        # 3. Stack Side-by-Side
        sbs = np.hstack([color_bgr, alpha_bgr])

        # 4. Save
        # Calculate structure relative to the input root
        rel_path = os.path.relpath(file_path, input_root)
        rel_path_jpg = os.path.splitext(rel_path)[0] + ".jpg"

        # Join with absolute output root
        out_path = os.path.join(output_root, rel_path_jpg)

        # Ensure directory exists
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        success = cv2.imwrite(out_path, sbs, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

        if not success: return False, "Write failed"

        # Return the absolute path of the written file for verification
        return True, out_path

    except Exception as e:
        return False, str(e)


def main():
    print("--- Absolute Path SBS Converter ---")

    # Get Input
    default_input = "images"
    raw_input = input(f"Enter source folder path [default: {default_input}]: ").strip() or default_input

    # FORCE ABSOLUTE PATH
    input_dir = os.path.abspath(raw_input)

    if not os.path.isdir(input_dir):
        print(f"ERROR: Directory not found at: {input_dir}")
        return

    # Get Output
    default_output = input_dir + "_sbs"
    raw_output = input(f"Enter output folder path [default: {default_output}]: ").strip() or default_output

    # FORCE ABSOLUTE PATH
    output_dir = os.path.abspath(raw_output)

    print(f"\n------------------------------------------------")
    print(f"SOURCE: {input_dir}")
    print(f"TARGET: {output_dir}")
    print(f"------------------------------------------------\n")

    q_str = input("Quality (1-100) [default: 90]: ").strip()
    quality = int(q_str) if q_str.isdigit() else 90

    files = glob(os.path.join(input_dir, "**", "*.webp"), recursive=True)
    total_files = len(files)

    if total_files == 0:
        print("No .webp files found in source.")
        return

    if input(f"Convert {total_files} files? (y/n): ").strip().lower() != 'y': return

    start_time = time.time()
    cpu_count = max(1, os.cpu_count())

    errors = []
    first_success = False

    with ProcessPoolExecutor(max_workers=cpu_count) as executor:
        futures = {
            executor.submit(process_file_final, f, input_dir, output_dir, quality): f
            for f in files
        }

        for future in tqdm(as_completed(futures), total=total_files, unit="img"):
            result = future.result()
            success = result[0]
            payload = result[1]  # Either error msg or file path

            if success:
                if not first_success:
                    print(f"\n[VERIFY] First file written to: {payload}")
                    first_success = True
            else:
                errors.append(f"{futures[future]}: {payload}")

    duration = time.time() - start_time
    print(f"\nDone in {duration:.2f}s")
    print(f"Final Output Location: {output_dir}")

    if errors:
        print(f"Errors: {len(errors)}")


if __name__ == "__main__":
    main()