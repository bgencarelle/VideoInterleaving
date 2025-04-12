"""
turntoblack.py

Restores perceptual appearance of images whose alpha masks erased dark RGB data,
by compositing the image over black and setting full opacity.

Two modes:
1. Script mode (asks user for input directory)
2. Callable mode (from another script: turntoblack.process_folder(...))
"""

import os
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional


def recover_image(filepath: str, output_folder: str) -> Optional[str]:
    try:
        with Image.open(filepath).convert("RGBA") as img:
            data = np.array(img)
            r, g, b, a = data[..., 0], data[..., 1], data[..., 2], data[..., 3]
            alpha_norm = a.astype(np.float32) / 255.0

            # Composite RGB over black
            r_new = (r.astype(np.float32) * alpha_norm).clip(0, 255).astype(np.uint8)
            g_new = (g.astype(np.float32) * alpha_norm).clip(0, 255).astype(np.uint8)
            b_new = (b.astype(np.float32) * alpha_norm).clip(0, 255).astype(np.uint8)

            new_data = np.stack([r_new, g_new, b_new, np.full_like(a, 255)], axis=-1)
            out_img = Image.fromarray(new_data, "RGBA")

            # Save to output
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, os.path.basename(filepath))

            # Ensure lossless saving for WebP
            ext = os.path.splitext(filepath)[1].lower()
            if ext == ".webp":
                out_img.save(output_path, "WEBP", lossless=True)
            else:
                out_img.save(output_path)

            return output_path

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None


def process_folder(input_folder: str, output_folder: Optional[str] = None, max_workers: int = 8, dry_run: bool = False):
    if output_folder is None:
        output_folder = f"{input_folder}_blackrecovered"
    os.makedirs(output_folder, exist_ok=True)

    files = [os.path.join(input_folder, f) for f in os.listdir(input_folder)
             if f.lower().endswith((".png", ".webp"))]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for f in files:
            if dry_run:
                print(f"Would recover: {f}")
            else:
                futures.append(executor.submit(recover_image, f, output_folder))

        if not dry_run:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    print(f"Recovered: {os.path.basename(result)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        dry_run = '--dry-run' in sys.argv
        input_paths = [arg for arg in sys.argv[1:] if not arg.startswith('--')]
        for input_path in input_paths:
            process_folder(input_path, max_workers=os.cpu_count(), dry_run=dry_run)
    else:
        input_path = input("Enter the path to the folder containing the images: ").strip()
        process_folder(input_path, max_workers=os.cpu_count())
