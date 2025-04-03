import os
import re
import cv2
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

# Import the turntoblack recovery function
try:
    from turntoblack import recover_image
except ImportError:
    print("WARNING: Could not import 'recover_image' from 'turntoblack.py'. Make sure it's in the same folder or installed.")

########################################
# Natural sort helper
########################################
def natural_sort_key(s: str):
    """Provides a key for natural (human) sorting of filenames."""
    # breaks string into alpha + numeric chunks for better sorting
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'([0-9]+)', s)]

########################################
# Channel + alpha utility
########################################
def ensure_four_channels(img: np.ndarray) -> np.ndarray:
    """
    Ensures that the image has 4 channels.
    If the image has 3 channels, converts it from BGR to BGRA.
    If grayscale, convert to BGRA.
    """
    if img is None:
        return None

    if len(img.shape) == 2:
        # Grayscale; convert to BGRA with opaque alpha.
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
    image[mask, 0] = 0  # Blue channel
    image[mask, 1] = 0  # Green channel
    image[mask, 2] = 0  # Red channel
    return image

########################################
# Main matte application function
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
      7) Convert to RGBA + save lossless WebP in mirrored folder structure.
    """
    try:
        # ------------------------------------------
        # 1) Optionally recover black-based appearance
        #    for the target image.
        # ------------------------------------------
        if recover_first:
            recovered_path = recover_image(src_path, output_folder=tmp_recover_dir)
            if recovered_path:
                src_path = recovered_path  # override with recovered

        # ------------------------------------------
        # 2) Load target + transparency (BGRA)
        # ------------------------------------------
        target_image = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
        transparency_image = cv2.imread(transparency_path, cv2.IMREAD_UNCHANGED)

        target_image = ensure_four_channels(target_image)
        transparency_image = ensure_four_channels(transparency_image)

        if target_image is None or transparency_image is None:
            print(f"Error loading images for {src_path}")
            return

        # ------------------------------------------
        # 3) Resize transparency if needed
        # ------------------------------------------
        if (target_image.shape[0], target_image.shape[1]) != (transparency_image.shape[0], transparency_image.shape[1]):
            transparency_image = cv2.resize(
                transparency_image,
                (target_image.shape[1], target_image.shape[0]),
                interpolation=cv2.INTER_LANCZOS4
            )

        # Extract alpha from transparency
        alpha_channel = transparency_image[:, :, 3]

        # ------------------------------------------
        # 4) Create normal matte
        # ------------------------------------------
        normal_image = target_image.copy()
        normal_image[:, :, 3] = alpha_channel
        normal_image = adjust_transparent_pixels_cv2(normal_image)

        # ------------------------------------------
        # 5) Create inverted matte
        # ------------------------------------------
        inverted_alpha = 255 - alpha_channel
        inverted_image = target_image.copy()
        inverted_image[:, :, 3] = inverted_alpha
        inverted_image = adjust_transparent_pixels_cv2(inverted_image)

        # ------------------------------------------
        # 6) Mirror directory structure for outputs
        # ------------------------------------------
        rel_path = os.path.relpath(current_root, target_folder)
        dest_dir_normal = os.path.join(dest_root_normal, rel_path)
        dest_dir_inv = os.path.join(dest_root_inv, rel_path)
        os.makedirs(dest_dir_normal, exist_ok=True)
        os.makedirs(dest_dir_inv, exist_ok=True)

        # use the original file's basename with .webp extension
        base_name = os.path.splitext(file)[0]
        normal_dest_path = os.path.join(dest_dir_normal, base_name + ".webp")
        inv_dest_path = os.path.join(dest_dir_inv, base_name + ".webp")

        # ------------------------------------------
        # 7) Convert BGRA -> RGBA and save
        # ------------------------------------------
        # Normal matte
        try:
            normal_rgba = cv2.cvtColor(normal_image, cv2.COLOR_BGRA2RGBA)
            normal_pil = Image.fromarray(normal_rgba)
            normal_pil.save(normal_dest_path, "WEBP", lossless=True)
            print(f"Saved normal matte image to {normal_dest_path}")
        except Exception as e:
            print(f"Error saving normal image {normal_dest_path}: {e}")

        # Inverted matte
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
# Main routine
########################################
def main():
    # Prompt for user input
    transparency_folder = input("Enter the path to the transparency folder: ").strip()
    target_folder = input("Enter the path to the target folder (source images): ").strip()
    test_mode_input = input("Do you want to run in test mode? (y/n): ").strip().lower()
    test_mode = test_mode_input in ("y", "yes")

    recover_input = input("Recover black-lost alpha first? (y/n): ").strip().lower()
    recover_first = recover_input in ("y", "yes")

    # Destination roots for normal/inverted outputs
    dest_root_normal = target_folder.rstrip(os.sep) + "_normal"
    dest_root_inv = target_folder.rstrip(os.sep) + "_inv"

    # Supported image extensions
    IMG_EXTS = ('.png', '.jpg', '.jpeg', '.webp')

    # Build a sorted list of transparency images
    transparency_files = sorted(
        [f for f in os.listdir(transparency_folder) if f.lower().endswith(IMG_EXTS)],
        key=natural_sort_key
    )
    if not transparency_files:
        raise ValueError("No transparency images found in the transparency folder.")
    transparency_paths = [os.path.join(transparency_folder, f) for f in transparency_files]

    tasks = []
    processed_one = False

    with ThreadPoolExecutor() as executor:
        for current_root, dirs, files in os.walk(target_folder):
            sorted_files = sorted([f for f in files if f.lower().endswith(IMG_EXTS)], key=natural_sort_key)
            for i, file in enumerate(sorted_files):
                src_path = os.path.join(current_root, file)
                # round-robin or sequential pairing
                transparency_path = transparency_paths[i % len(transparency_paths)]

                if test_mode:
                    # do just one file for test
                    print(f"(Test Mode) Would process {file} with {os.path.basename(transparency_path)}")
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
                    processed_one = True
                    break
                else:
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
            if test_mode and processed_one:
                break

        # Wait for threads
        for future in as_completed(tasks):
            try:
                future.result()
            except Exception as e:
                print(f"Error in threaded task: {e}")

if __name__ == "__main__":
    main()
