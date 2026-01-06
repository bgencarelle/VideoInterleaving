import os
import re
import cv2
import numpy as np


def natural_key(string):
    """
    Helper for natural (alphanumeric) sorting.
    Splits the string into a list of integers and non-digit substrings.
    """
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', string)]


def add_alpha_channel(image):
    """
    Ensure the image has an alpha channel.
    If the image has 3 channels, add a fully opaque alpha.
    """
    if image is None:
        return None
    # If grayscale, convert to BGR first.
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    channels = image.shape[2] if len(image.shape) > 2 else 1
    if channels == 3:
        if image.dtype == np.uint8:
            alpha = np.full((image.shape[0], image.shape[1]), 255, dtype=np.uint8)
        elif image.dtype == np.uint16:
            alpha = np.full((image.shape[0], image.shape[1]), 65535, dtype=np.uint16)
        else:
            alpha = np.ones((image.shape[0], image.shape[1]), dtype=image.dtype)
        # Convert BGR to BGRA and set alpha.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        image[:, :, 3] = alpha
    return image


def load_image_as_rgba(path):
    """
    Loads an image with cv2 and ensures it has an alpha channel.
    Returns the image in BGRA order.
    """
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    return add_alpha_channel(image)


def extract_white_patch(rgb, threshold=0.9, alpha=None):
    """
    Given a normalized RGB image (values in [0, 1]),
    find the largest connected white or neutral gray patch and return its average color.
    If an alpha channel is provided, only consider pixels where alpha > 0.
    """
    if alpha is not None:
        visible = alpha > 0
        mask = np.all(rgb >= threshold, axis=2) & visible
    else:
        mask = np.all(rgb >= threshold, axis=2)
    mask = mask.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None  # No patch found.
    # Exclude label 0 (background) and select the largest connected component.
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    white_patch_mask = (labels == largest_label)
    if np.count_nonzero(white_patch_mask) == 0:
        return None
    avg_color = np.mean(rgb[white_patch_mask], axis=0)
    return avg_color


def compute_scaling_factors(reference_color):
    """
    Compute per-channel scaling factors to map the reference white/gray color to pure white [1,1,1].
    """
    target = np.array([1.0, 1.0, 1.0])
    with np.errstate(divide='ignore', invalid='ignore'):
        scaling_factors = np.where(reference_color > 0, target / reference_color, 1.0)
    return scaling_factors


def apply_white_balance(rgb, scaling_factors):
    """
    Applies white balance to a normalized RGB image using the computed scaling factors.
    """
    balanced = rgb * scaling_factors.reshape(1, 1, 3)
    balanced = np.clip(balanced, 0, 1)
    return balanced


def process_directory(input_dir, transparency_dir, output_root, output_root_inv, white_threshold=0.9):
    """
    Process a single folder of images:
      - Lists main images and transparency images via natural sort.
      - Computes an initial white balance from the first main image that yields a valid white patch.
      - Then processes each paired image. Every few frames (update_interval),
        it recalculates the white patch from the current main image and updates the scaling factors
        using an exponential moving average (smoothing_alpha) to avoid sudden jumps.
      - For each frame, two outputs are produced:
          Version A (saved in output_root): the white-balanced RGB is combined with a thresholded alpha,
             where for each pixel if the transparency (normalized) > 0.25 then alpha is set to 0 (fully transparent),
             otherwise it is set to full opacity.
          Version B (saved in output_root_inv): the thresholded alpha is inverted.
      - Both outputs preserve an alpha channel so they can be recombined without visible gaps or feathering.
    """
    supported_exts = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp')

    # List and naturally sort main images.
    main_files = sorted([f for f in os.listdir(input_dir)
                         if f.lower().endswith(supported_exts)], key=natural_key)
    if not main_files:
        print(f"No supported images found in {input_dir}")
        return

    # List and naturally sort transparency files.
    alpha_files = sorted([f for f in os.listdir(transparency_dir)
                          if f.lower().endswith(supported_exts)], key=natural_key)
    if not alpha_files:
        print(f"No supported transparency images found in {transparency_dir}")
        return

    if len(main_files) != len(alpha_files):
        print(
            f"Warning: Number of main images ({len(main_files)}) and transparency images ({len(alpha_files)}) differ. Pairing by index up to the minimum count.")

    pair_count = min(len(main_files), len(alpha_files))

    # Parameters for periodic update and smoothing.
    update_interval = 10  # update white balance every 10 frames
    smoothing_alpha = 0.2  # weight for new scaling factors

    # Find initial white balance reference from the main images.
    initial_ref = None
    ref_filename = None
    for fname in main_files:
        path = os.path.join(input_dir, fname)
        image = load_image_as_rgba(path)
        if image is None:
            print(f"Error loading {path}, skipping.")
            continue
        image_rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        if image_rgba.dtype == np.uint8:
            rgb = image_rgba[:, :, :3] / 255.0
            alpha_img = image_rgba[:, :, 3] / 255.0
        elif image_rgba.dtype == np.uint16:
            rgb = image_rgba[:, :, :3] / 65535.0
            alpha_img = image_rgba[:, :, 3] / 65535.0
        else:
            rgb = image_rgba[:, :, :3]
            alpha_img = image_rgba[:, :, 3]
        candidate = extract_white_patch(rgb, threshold=white_threshold, alpha=alpha_img)
        if candidate is not None:
            initial_ref = candidate
            ref_filename = fname
            break

    if initial_ref is None:
        print(f"No valid white/neutral patch found in any image in {input_dir}; skipping folder.")
        return

    current_scaling = compute_scaling_factors(initial_ref)
    print(f"In folder '{input_dir}', using '{ref_filename}' as initial white balance reference.")
    print(f"Initial reference white/gray color: {initial_ref}")
    print(f"Initial scaling factors: {current_scaling}")

    # Determine maximum value for current image dtype.
    def get_max_val(dtype):
        if dtype == np.uint8:
            return 255
        elif dtype == np.uint16:
            return 65535
        else:
            return 1.0

    # Process each paired image.
    for idx in range(pair_count):
        main_path = os.path.join(input_dir, main_files[idx])
        alpha_path = os.path.join(transparency_dir, alpha_files[idx])

        main_img = load_image_as_rgba(main_path)
        if main_img is None:
            print(f"Error loading {main_path}; skipping.")
            continue
        trans_img = load_image_as_rgba(alpha_path)
        if trans_img is None:
            print(f"Error loading transparency image {alpha_path}; using full opacity instead.")
            if main_img.dtype == np.uint8:
                trans_img = np.ones(main_img.shape, dtype=np.uint8) * 255
            elif main_img.dtype == np.uint16:
                trans_img = np.ones(main_img.shape, dtype=np.uint16) * 65535
            else:
                trans_img = np.ones(main_img.shape, dtype=main_img.dtype)
            trans_img = add_alpha_channel(trans_img)

        main_rgba = cv2.cvtColor(main_img, cv2.COLOR_BGRA2RGBA)
        if main_rgba.dtype == np.uint8:
            rgb = main_rgba[:, :, :3] / 255.0
            alpha_main = main_rgba[:, :, 3] / 255.0
        elif main_rgba.dtype == np.uint16:
            rgb = main_rgba[:, :, :3] / 65535.0
            alpha_main = main_rgba[:, :, 3] / 65535.0
        else:
            rgb = main_rgba[:, :, :3]
            alpha_main = main_rgba[:, :, 3]

        # Periodically update white balance scaling factors.
        if idx % update_interval == 0:
            candidate = extract_white_patch(rgb, threshold=white_threshold, alpha=alpha_main)
            if candidate is not None:
                new_scaling = compute_scaling_factors(candidate)
                current_scaling = smoothing_alpha * new_scaling + (1 - smoothing_alpha) * current_scaling
                print(f"Frame {idx}: Updated scaling factors: {current_scaling}")
            else:
                print(f"Frame {idx}: No valid white patch found; retaining previous scaling factors.")

        balanced_rgb = apply_white_balance(rgb, current_scaling)
        max_val = get_max_val(main_rgba.dtype)
        if main_rgba.dtype == np.uint8:
            balanced_rgb = (balanced_rgb * 255).astype(np.uint8)
        elif main_rgba.dtype == np.uint16:
            balanced_rgb = (balanced_rgb * 65535).astype(np.uint16)

        # Process the transparency image.
        trans_rgba = cv2.cvtColor(trans_img, cv2.COLOR_BGRA2RGBA)
        if trans_rgba.dtype == np.uint8:
            alpha_channel = trans_rgba[:, :, 3] / 255.0
        elif trans_rgba.dtype == np.uint16:
            alpha_channel = trans_rgba[:, :, 3] / 65535.0
        else:
            alpha_channel = trans_rgba[:, :, 3]

        # --- NEW ALPHA PROCESSING ---
        # Threshold: if normalized alpha > 0.25 then set to 0 (full transparency), else 1 (fully opaque).
        threshold_val = 0.10
        thresholded_alpha = alpha_channel#(np.where(alpha_channel > threshold_val, 0.0, 1.0))
        # Invert the thresholded mask.
        inverted_alpha = 1.0 - thresholded_alpha

        # Build two versions:
        # Version A: white balanced image with thresholded alpha.
        balanced_rgba_A = np.dstack((balanced_rgb, (thresholded_alpha * max_val).astype(main_rgba.dtype)))
        balanced_bgra_A = cv2.cvtColor(balanced_rgba_A, cv2.COLOR_RGBA2BGRA)
        # Version B: white balanced image with inverted alpha.
        balanced_rgba_B = np.dstack((balanced_rgb, (inverted_alpha * max_val).astype(main_rgba.dtype)))
        balanced_bgra_B = cv2.cvtColor(balanced_rgba_B, cv2.COLOR_RGBA2BGRA)

        # Determine output paths (preserving folder structure relative to input_folder).
        rel_path = os.path.relpath(input_dir, input_folder)
        output_dir_A = os.path.join(output_root, rel_path)
        os.makedirs(output_dir_A, exist_ok=True)
        out_path_A = os.path.join(output_dir_A, main_files[idx])

        output_dir_B = os.path.join(output_root_inv, rel_path)
        os.makedirs(output_dir_B, exist_ok=True)
        out_path_B = os.path.join(output_dir_B, main_files[idx])

        cv2.imwrite(out_path_A, balanced_bgra_A)
        cv2.imwrite(out_path_B, balanced_bgra_B)
        print(f"Processed {main_path} with transparency from {alpha_path} ->")
        print(f"   Normal: {out_path_A}")
        print(f"   Inverted Alpha: {out_path_B}")


def main():
    global input_folder  # Used for preserving relative paths.
    input_folder = input("Enter the path to the main image folder: ").strip()
    if not os.path.isdir(input_folder):
        print("The provided main image folder is not valid.")
        return

    transparency_folder = input("Enter the path to the transparency folder: ").strip()
    if not os.path.isdir(transparency_folder):
        print("The provided transparency folder is not valid.")
        return

    parent_dir = os.path.dirname(input_folder)
    folder_name = os.path.basename(input_folder)
    output_root = os.path.join(parent_dir, folder_name + "_wb")
    output_root_inv = os.path.join(parent_dir, folder_name + "_wb_alpha_inv")
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(output_root_inv, exist_ok=True)

    # Process each folder (and its subdirectories) that contain supported images.
    for root, dirs, files in os.walk(input_folder):
        if any(file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp')) for file in files):
            # Assume a 1:1 relationship at the folder level between main and transparency folders.
            rel = os.path.relpath(root, input_folder)
            corresponding_transparency_dir = os.path.join(transparency_folder, rel)
            if not os.path.isdir(corresponding_transparency_dir):
                print(f"Transparency folder {corresponding_transparency_dir} not found for {root}; skipping.")
                continue
            print(f"\nProcessing folder: {root}")
            process_directory(root, corresponding_transparency_dir, output_root, output_root_inv, white_threshold=0.9)


if __name__ == "__main__":
    main()
