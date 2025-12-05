import os
import cv2
import numpy as np
import time
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from turbojpeg import TurboJPEG, TJPF_BGR, TJSAMP_444


def process_file_optimized(file_path, input_root, output_root, quality=90):
    # Initialize TurboJPEG instance inside the process (thread-safe isolation)
    jpeg = TurboJPEG()

    try:
        # 1. Load Image (OpenCV is still best for decoding WebP)
        # Load as-is (unchanged) to detect alpha
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            return False, "Load failed"

        h, w = img.shape[:2]

        # 2. Pre-allocate the Side-by-Side Buffer
        # We perform one allocation for the final image.
        # Format: Height, Width*2, 3 Channels (BGR)
        sbs = np.empty((h, w * 2, 3), dtype=np.uint8)

        # 3. Direct Memory Assignment (Faster than split/merge/hstack)

        if img.ndim == 2:
            # Grayscale (1 channel)
            # Left side: Convert Gray to BGR (3 channels) in-place
            cv2.cvtColor(img, cv2.COLOR_GRAY2BGR, dst=sbs[:, :w])
            # Right side: Solid White
            sbs[:, w:] = 255

        elif img.ndim == 3:
            channels = img.shape[2]

            if channels == 4:
                # BGRA
                # Copy BGR channels to Left
                sbs[:, :w] = img[:, :, :3]

                # Copy Alpha to Right
                # We broadcast the (H,W) alpha to (H,W,3) efficiently
                alpha = img[:, :, 3]
                sbs[:, w:, 0] = alpha
                sbs[:, w:, 1] = alpha
                sbs[:, w:, 2] = alpha

            elif channels == 3:
                # BGR
                # Copy BGR to Left
                sbs[:, :w] = img
                # Right side: Solid White
                sbs[:, w:] = 255

            else:
                return False, f"Unsupported channels: {channels}"
        else:
            return False, "Unknown dimensions"

        # 4. TurboJPEG Encoding
        # This is where the massive speedup comes from vs cv2.imwrite
        # pixel_format=TJPF_BGR matches OpenCV's internal layout
        # subsampling=TJSAMP_444 ensures high fidelity for the Alpha Mask (no color bleed)
        encoded_bytes = jpeg.encode(sbs, quality=quality, pixel_format=TJPF_BGR, jpeg_subsample=TJSAMP_444)

        # 5. Write to Disk
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
    print("--- Turbo-Charged SBS Converter ---")
    print("Optimizations: TurboJPEG Encode + NumPy Slice Assignment")

    default_input = "images"
    input_dir = input(f"Enter source folder path [default: {default_input}]: ").strip() or default_input

    if not os.path.isdir(input_dir):
        print("Directory not found.")
        return

    default_output = input_dir + "_sbs_turbo"
    output_dir = input(f"Enter output folder path [default: {default_output}]: ").strip() or default_output

    q_str = input("Quality (1-100) [default: 90]: ").strip()
    quality = int(q_str) if q_str.isdigit() else 90

    files = glob(os.path.join(input_dir, "**", "*.webp"), recursive=True)
    total_files = len(files)
    print(f"Found {total_files} files.")

    if total_files == 0: return
    if input("Start? (y/n): ").strip().lower() != 'y': return

    # Processing
    start_time = time.time()
    # TurboJPEG releases the GIL efficiently, so we can use all cores
    cpu_count = max(1, os.cpu_count())
    print(f"Engaging {cpu_count} Warp Engines...")

    errors = []

    with ProcessPoolExecutor(max_workers=cpu_count) as executor:
        futures = {
            executor.submit(process_file_optimized, f, input_dir, output_dir, quality): f
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
        print(errors[:5])


if __name__ == "__main__":
    main()