import os
import re
import cv2
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

# Import the turntoblack recovery function.
try:
    from turntoblack import recover_image  # existing recover_image (used in mode 1)
except ImportError:
    print(
        "WARNING: Could not import 'recover_image' from 'turntoblack.py'. Make sure it's in the same folder or installed.")

########################################
# Natural sort helper
########################################
def natural_sort_key(s: str):
    """Provides a key for natural (human) sorting of filenames."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'([0-9]+)', s)]

########################################
# Channel + alpha utility
########################################
def ensure_four_channels(img: np.ndarray) -> np.ndarray:
    """
    Ensures that the image has 4 channels.
    If the image has 3 channels, converts it from BGR to BGRA.
    If grayscale, converts to BGRA.
    """
    if img is None:
        return None

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    return img

########################################
# Adjust transparent pixels to black
########################################
def adjust_transparent_pixels_cv2(image: np.ndarray) -> np.ndarray:
    """
    Adjusts an image (in BGRA order) so that any pixel with full transparency (alpha==0)
    has its B, G, R channels set to 0.
    """
    if image is None:
        return None

    mask = (image[:, :, 3] == 0)
    image[mask, 0] = 0  # Blue
    image[mask, 1] = 0  # Green
    image[mask, 2] = 0  # Red
    return image

########################################
# Mode 1: Normal matte application function
########################################
def process_file(
        src_path: str,
        file: str,
        current_root: str,
        transparency_path: str,
        target_folder: str,
        dest_root_normal: str,
        dest_root_inv: str,
        recover_first: bool = False,
        tmp_recover_dir: str = "tmp_blackrecover"
) -> None:
    """
    Process one target image using a transparency matte image.

    Steps:
      1) Optionally call recover_image(...) to restore black-based appearance
         for alpha-lost images.
      2) Load both target image and transparency image in BGRA.
      3) Resize transparency to match target.
      4) Create normal matte (alpha = transparency's alpha).
      5) Create inverted matte (alpha = 255 - transparency's alpha).
      6) Ensure fully transparent pixels are (0,0,0,0) in BGRA.
      7) Convert to RGBA and save lossless WebP in a mirrored folder structure.
    """
    try:
        # Optionally recover black-based appearance first.
        if recover_first:
            recovered_path = recover_image(src_path, output_folder=tmp_recover_dir)
            if recovered_path:
                src_path = recovered_path  # use the recovered image

        # Load images (target and transparency) with unchanged flags.
        target_image = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
        transparency_image = cv2.imread(transparency_path, cv2.IMREAD_UNCHANGED)

        target_image = ensure_four_channels(target_image)
        transparency_image = ensure_four_channels(transparency_image)

        if target_image is None or transparency_image is None:
            print(f"Error loading images for {src_path}")
            return

        # Resize transparency if needed.
        if (target_image.shape[0], target_image.shape[1]) != (transparency_image.shape[0], transparency_image.shape[1]):
            transparency_image = cv2.resize(
                transparency_image,
                (target_image.shape[1], target_image.shape[0]),
                interpolation=cv2.INTER_LANCZOS4
            )

        # Extract alpha from transparency.
        alpha_channel = transparency_image[:, :, 3]

        # Create normal matte.
        normal_image = target_image.copy()
        normal_image[:, :, 3] = alpha_channel
        normal_image = adjust_transparent_pixels_cv2(normal_image)

        # Create inverted matte.
        inverted_alpha = 255 - alpha_channel
        inverted_image = target_image.copy()
        inverted_image[:, :, 3] = inverted_alpha
        inverted_image = adjust_transparent_pixels_cv2(inverted_image)

        # Mirror the output folder structure.
        rel_path = os.path.relpath(current_root, target_folder)
        dest_dir_normal = os.path.join(dest_root_normal, rel_path)
        dest_dir_inv = os.path.join(dest_root_inv, rel_path)
        os.makedirs(dest_dir_normal, exist_ok=True)
        os.makedirs(dest_dir_inv, exist_ok=True)

        # Build destination file names (using .webp extension).
        base_name = os.path.splitext(file)[0]
        normal_dest_path = os.path.join(dest_dir_normal, base_name + ".webp")
        inv_dest_path = os.path.join(dest_dir_inv, base_name + ".webp")

        # Convert from BGRA to RGBA and save normal matte.
        try:
            normal_rgba = cv2.cvtColor(normal_image, cv2.COLOR_BGRA2RGBA)
            normal_pil = Image.fromarray(normal_rgba)
            normal_pil.save(normal_dest_path, "WEBP", lossless=True)
            print(f"Saved normal matte image to {normal_dest_path}")
        except Exception as e:
            print(f"Error saving normal image {normal_dest_path}: {e}")

        # Convert from BGRA to RGBA and save inverted matte.
        try:
            inverted_rgba = cv2.cvtColor(inverted_image, cv2.COLOR_BGRA2RGBA)
            inverted_pil = Image.fromarray(inverted_rgba)
            inverted_pil.save(inv_dest_path, "WEBP", lossless=True)
            print(f"Saved inverted matte image to {inv_dest_path}")
        except Exception as e:
            print(f"Error saving inverted image {inv_dest_path}: {e}")

    except Exception as e:
        print(f"Error processing {src_path}: {e}")

########################################
# Mode 2: Black recovery lossless function
########################################
def recover_image_lossless(filepath: str, output_folder: str) -> Optional[str]:
    """
    Recovers the perceptual appearance of an image whose alpha mask erased dark RGB data,
    by compositing the image over black and restoring full opacity.
    Saves the output losslessly if possible (e.g. lossless WebP).
    """
    try:
        with Image.open(filepath).convert("RGBA") as img:
            data = np.array(img)
            r, g, b, a = data[..., 0], data[..., 1], data[..., 2], data[..., 3]
            alpha_norm = a.astype(np.float32) / 255.0

            # Composite RGB over black.
            r_new = (r.astype(np.float32) * alpha_norm).clip(0, 255).astype(np.uint8)
            g_new = (g.astype(np.float32) * alpha_norm).clip(0, 255).astype(np.uint8)
            b_new = (b.astype(np.float32) * alpha_norm).clip(0, 255).astype(np.uint8)

            new_data = np.stack([r_new, g_new, b_new, np.full_like(a, 255)], axis=-1)
            out_img = Image.fromarray(new_data, "RGBA")

            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, os.path.basename(filepath))
            ext = os.path.splitext(filepath)[1].lower()
            # For WebP files, save with lossless encoding.
            if ext == ".webp":
                out_img.save(output_path, "WEBP", lossless=True)
            else:
                out_img.save(output_path)
            return output_path

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def recover_folder_recursive(input_folder: str, output_folder: str, max_workers: int = 8):
    """
    Recursively recovers images (using the lossless black-recovery function),
    preserving the folder structure.
    """
    IMG_EXTS = ('.png', '.jpg', '.jpeg', '.webp')
    tasks = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Walk through every file in the input folder.
        for current_root, dirs, files in os.walk(input_folder):
            for file in files:
                if file.lower().endswith(IMG_EXTS):
                    src_path = os.path.join(current_root, file)
                    # Compute the folder's relative path.
                    rel_path = os.path.relpath(current_root, input_folder)
                    dest_dir = os.path.join(output_folder, rel_path)
                    os.makedirs(dest_dir, exist_ok=True)
                    tasks.append(executor.submit(recover_image_lossless, src_path, dest_dir))
        # Wait for all tasks to complete.
        for future in as_completed(tasks):
            result = future.result()
            if result:
                print(f"Recovered: {result}")

########################################
# Main routine
########################################
def main():
    print("Select processing mode:")
    print("1: Normal matte processing (apply transparency matte and optionally recover black alpha)")
    print("2: Just black recovery (recover images with lossless output, preserving folder structure)")

    mode = input("Enter mode number (1 or 2): ").strip()

    if mode == "2":
        # Mode 2: Just black recover
        input_folder = input("Enter the path to the source folder for black recovery: ").strip()
        # The output folder is placed in the same parent as the input folder, with '_rec' appended.
        parent_folder = os.path.dirname(input_folder)
        basename = os.path.basename(input_folder.rstrip(os.sep))
        output_folder = os.path.join(parent_folder, basename + "_rec")
        max_workers_input = input("Enter the number of worker threads (default 8): ").strip()
        max_workers = int(max_workers_input) if max_workers_input.isdigit() else 8

        print(f"Recovering images from '{input_folder}', saving into '{output_folder}' ...")
        recover_folder_recursive(input_folder, output_folder, max_workers)
    else:
        # Mode 1: Normal matte processing (with optional black recovery before matte application)
        transparency_folder = input("Enter the path to the transparency folder: ").strip()
        target_folder = input("Enter the path to the target folder (source images): ").strip()
        test_mode_input = input("Do you want to run in test mode? (y/n): ").strip().lower()
        test_mode = test_mode_input in ("y", "yes")

        recover_input = input("Recover black-lost alpha first? (y/n): ").strip().lower()
        recover_first = recover_input in ("y", "yes")

        # Output folders for normal and inverted mattes.
        dest_root_normal = target_folder.rstrip(os.sep) + "_normal"
        dest_root_inv = target_folder.rstrip(os.sep) + "_inv"

        IMG_EXTS = ('.png', '.jpg', '.jpeg', '.webp')

        # Gather and sort transparency images.
        transparency_files = sorted(
            [f for f in os.listdir(transparency_folder) if f.lower().endswith(IMG_EXTS)],
            key=natural_sort_key
        )
        if not transparency_files:
            raise ValueError("No transparency images found in the transparency folder.")
        transparency_paths = [os.path.join(transparency_folder, f) for f in transparency_files]

        if test_mode:
            # In test mode, for each folder we process only the same predetermined files.
            # These are the 1-based file positions to be processed:
            test_indices = [20, 80, 140, 200, 260, 320, 380, 440, 500, 1000, 2000]
            # Convert to zero-based indices:
            test_indices_zero = [i - 1 for i in test_indices]

            for current_root, dirs, files in os.walk(target_folder):
                sorted_files = sorted([f for f in files if f.lower().endswith(IMG_EXTS)], key=natural_sort_key)
                if not sorted_files:
                    continue  # skip folders with no matching files
                rel_folder = os.path.relpath(current_root, target_folder)
                for idx in test_indices_zero:
                    if idx < len(sorted_files):
                        file = sorted_files[idx]
                        src_path = os.path.join(current_root, file)
                        # Use the same round-robin pairing as before.
                        transparency_path = transparency_paths[idx % len(transparency_paths)]
                        print(f"(Test Mode) Processing '{file}' from folder '{rel_folder}' with transparency '{os.path.basename(transparency_path)}'")
                        process_file(
                            src_path=src_path,
                            file=file,
                            current_root=current_root,
                            transparency_path=transparency_path,
                            target_folder=target_folder,
                            dest_root_normal=dest_root_normal,
                            dest_root_inv=dest_root_inv,
                            recover_first=recover_first
                        )
        else:
            tasks = []
            with ThreadPoolExecutor() as executor:
                for current_root, dirs, files in os.walk(target_folder):
                    sorted_files = sorted([f for f in files if f.lower().endswith(IMG_EXTS)], key=natural_sort_key)
                    for i, file in enumerate(sorted_files):
                        src_path = os.path.join(current_root, file)
                        # Round-robin pairing of target images with transparency images.
                        transparency_path = transparency_paths[i % len(transparency_paths)]
                        tasks.append(
                            executor.submit(
                                process_file,
                                src_path,
                                file,
                                current_root,
                                transparency_path,
                                target_folder,
                                dest_root_normal,
                                dest_root_inv,
                                recover_first
                            )
                        )
                # Wait for all tasks to complete.
                for future in as_completed(tasks):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error in threaded task: {e}")

if __name__ == "__main__":
    main()
