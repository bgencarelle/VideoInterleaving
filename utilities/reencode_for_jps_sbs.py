import os
import cv2
import numpy as np
import time
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from turbojpeg import TurboJPEG, TJPF_BGR, TJSAMP_420

# --- Global Instance for Workers ---
# This ensures we only load the C-library once per CPU Core
_jpeg = None


def init_worker():
    global _jpeg
    _jpeg = TurboJPEG()


def process_file_fastest(file_path, input_root, output_root, quality=90):
    global _jpeg
    try:
        # 1. Load Image (Unchanged)
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None: return False, "Load failed"

        h, w = img.shape[:2]

        # 2. Allocate ONE contiguous block of memory
        # This is faster than stacking/merging arrays
        sbs = np.empty((h, w * 2, 3), dtype=np.uint8)

        # 3. Fast C++ Memory Operations
        # We use OpenCV's C++ backend to write directly into the 'sbs' buffer slices

        # Define views into the target buffer
        left_view = sbs[:, :w]
        right_view = sbs[:, w:]

        if img.ndim == 2:
            # Grayscale -> BGR (Left)
            cv2.cvtColor(img, cv2.COLOR_GRAY2BGR, dst=left_view)
            # Solid White (Right)
            right_view[:] = 255

        elif img.ndim == 3:
            channels = img.shape[2]
            if channels == 4:
                # BGRA
                # Copy BGR to Left
                left_view[:] = img[:, :, :3]

                # Copy Alpha to Right (Gray -> BGR)
                # This is much faster than sbs[:,w:,0]=a; sbs[:,w:,1]=a...
                cv2.cvtColor(img[:, :, 3], cv2.COLOR_GRAY2BGR, dst=right_view)

            elif channels == 3:
                # BGR
                left_view[:] = img
                right_view[:] = 255
            else:
                return False, f"Bad channels: {channels}"
        else:
            return False, "Bad dimensions"

        # 4. Turbo Encode
        # TJSAMP_420 is the standard JPEG speed/quality balance.
        # It is significantly faster than 4:4:4.
        encoded_bytes = _jpeg.encode(sbs, quality=quality, pixel_format=TJPF_BGR, jpeg_subsample=TJSAMP_420)

        # 5. Write to disk
        rel_path = os.path.relpath(file_path, input_root)
        rel_path_jpg = os.path.splitext(rel_path)[0] + ".jpg"
        out_path = os.path.join(output_root, rel_path_jpg)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(encoded_bytes)

        return True, None

    except Exception as e:
        return False, str(e)


def main():
    print("--- Maximum Velocity SBS Converter ---")

    default_input = "images"
    input_dir = input(f"Enter source folder path [default: {default_input}]: ").strip() or default_input

    if not os.path.isdir(input_dir):
        print("Directory not found.")
        return

    default_output = input_dir + "_sbs_fast"
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

    # Initialize workers with the global TurboJPEG instance
    with ProcessPoolExecutor(max_workers=cpu_count, initializer=init_worker) as executor:
        futures = {
            executor.submit(process_file_fastest, f, input_dir, output_dir, quality): f
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