import os
import re
import cv2
import numpy as np

def natural_key(string):
    """
    A helper function for natural sorting.
    Splits the string into a list of integers and non-digit substrings.
    """
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', string)]

def add_alpha_channel(image):
    """
    Ensure that the image has an alpha channel.
    If the image has 3 channels (BGR), add a fully opaque alpha channel.
    If the image already has 4 channels, return it unchanged.
    """
    if image is None:
        return None
    if len(image.shape) == 2:
        # Grayscale image: convert to BGR first.
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    channels = image.shape[2] if len(image.shape) > 2 else 1
    if channels == 3:
        # Create a fully opaque alpha channel.
        if image.dtype == np.uint8:
            alpha = np.full((image.shape[0], image.shape[1]), 255, dtype=np.uint8)
        elif image.dtype == np.uint16:
            alpha = np.full((image.shape[0], image.shape[1]), 65535, dtype=np.uint16)
        else:
            # Assume float image in range [0,1]
            alpha = np.ones((image.shape[0], image.shape[1]), dtype=image.dtype)
        # Concatenate alpha channel to create BGRA.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        image[:, :, 3] = alpha
    return image

def extract_white_patch(rgb, threshold=0.9, alpha=None):
    """
    Given a normalized RGB image (values in [0, 1]), find the largest connected
    white or neutral gray patch and return its average color.
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
        return None  # No white/neutral patch found.
    # Exclude label 0 (background) and select the largest component.
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
    Applies white balance to a normalized RGB image using the scaling factors.
    """
    balanced = rgb * scaling_factors.reshape(1, 1, 3)
    balanced = np.clip(balanced, 0, 1)
    return balanced

def load_image_as_rgba(path):
    """
    Load an image using cv2 and ensure it has an alpha channel.
    Returns the image in BGRA order.
    """
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    image = add_alpha_channel(image)
    return image

def process_directory(input_dir, output_root, white_threshold=0.9):
    """
    Process one directory of images.
    It uses the first image (in natural sort order) that produces a valid white patch
    as the reference for white balance. All images in the directory are adjusted based
    on this reference.
    """
    supported_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp')
    images = [f for f in os.listdir(input_dir)
              if f.lower().endswith(supported_extensions)]
    if not images:
        print(f"No supported images found in {input_dir}")
        return

    sorted_images = sorted(images, key=natural_key)
    reference_color = None
    ref_image_name_used = None

    # Iterate over candidate images until a valid white patch is found.
    for ref_image_name in sorted_images:
        ref_image_path = os.path.join(input_dir, ref_image_name)
        ref_image = load_image_as_rgba(ref_image_path)
        if ref_image is None:
            print(f"Error reading reference image {ref_image_path}, skipping.")
            continue

        # Convert from BGRA to RGBA.
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGRA2RGBA)
        # Normalize the image.
        if ref_image.dtype == np.uint8:
            rgb = ref_image[:, :, :3] / 255.0
            alpha = ref_image[:, :, 3] / 255.0
        elif ref_image.dtype == np.uint16:
            rgb = ref_image[:, :, :3] / 65535.0
            alpha = ref_image[:, :, 3] / 65535.0
        else:
            rgb = ref_image[:, :, :3]
            alpha = ref_image[:, :, 3]

        candidate = extract_white_patch(rgb, threshold=white_threshold, alpha=alpha)
        if candidate is not None:
            reference_color = candidate
            ref_image_name_used = ref_image_name
            break

    if reference_color is None:
        print(f"No suitable white/neutral patch found in any image in {input_dir}; skipping folder.")
        return

    scaling_factors = compute_scaling_factors(reference_color)
    print(f"In folder '{input_dir}', using {ref_image_name_used} as reference.")
    print(f"Reference white/gray color: {reference_color}")
    print(f"Scaling factors: {scaling_factors}")

    for file in images:
        image_path = os.path.join(input_dir, file)
        image = load_image_as_rgba(image_path)
        if image is None:
            print(f"Error reading {image_path}")
            continue

        # Convert from BGRA to RGBA for processing.
        image_rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        if image_rgba.dtype == np.uint8:
            rgb = image_rgba[:, :, :3] / 255.0
            alpha = image_rgba[:, :, 3] / 255.0
        elif image_rgba.dtype == np.uint16:
            rgb = image_rgba[:, :, :3] / 65535.0
            alpha = image_rgba[:, :, 3] / 65535.0
        else:
            rgb = image_rgba[:, :, :3]
            alpha = image_rgba[:, :, 3]

        balanced_rgb = apply_white_balance(rgb, scaling_factors)
        if image_rgba.dtype == np.uint8:
            balanced_rgb = (balanced_rgb * 255).astype(np.uint8)
        elif image_rgba.dtype == np.uint16:
            balanced_rgb = (balanced_rgb * 65535).astype(np.uint16)

        # Recombine with the alpha channel.
        balanced_rgba = np.dstack((balanced_rgb, (alpha * (255 if image_rgba.dtype == np.uint8 else 65535)).astype(image_rgba.dtype)))
        # Convert back to BGRA for saving.
        balanced_bgra = cv2.cvtColor(balanced_rgba, cv2.COLOR_RGBA2BGRA)

        rel_path = os.path.relpath(input_dir, input_folder)
        output_dir = os.path.join(output_root, rel_path)
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, file)
        cv2.imwrite(out_path, balanced_bgra)
        print(f"Processed {image_path} -> {out_path}")

def main():
    global input_folder  # Used for relative paths in process_directory.
    input_folder = input("Enter the path to the input folder: ").strip()
    if not os.path.isdir(input_folder):
        print("The provided path is not a valid directory.")
        return

    parent_dir = os.path.dirname(input_folder)
    folder_name = os.path.basename(input_folder)
    output_root = os.path.join(parent_dir, folder_name + "_wb")
    os.makedirs(output_root, exist_ok=True)

    # Process all subdirectories (including the input folder itself) that contain supported images.
    for root, dirs, files in os.walk(input_folder):
        if any(file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp')) for file in files):
            print(f"\nProcessing folder: {root}")
            process_directory(root, output_root, white_threshold=0.9)

if __name__ == "__main__":
    main()
