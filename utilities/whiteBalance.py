#!/usr/bin/env python3
"""
white_balance_native_sequential_optimized.py

This module processes images in their native BGRA space with white balancing,
using the 30th image (or the last one if fewer than 30 exist) in each folder as
the white reference. It computes per-channel scaling factors from a white/neutral
patch and then applies these factors to all images in the folder.

The output images are saved as lossless WebP. The full‐resolution input image is
output without resizing.

This version preserves the source folder structure and processes directories
sequentially (while processing images concurrently). It’s optimized by removing
redundant conversions and prints a message for each file as it is finished.
"""

import os
import re
import cv2
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Helper Functions ---

def natural_key(string):
    """Return a list of integers and lower-case substrings for natural sorting."""
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', string)]

def safe_relpath(path, start):
    """
    Compute a relative path from 'start' to 'path'. If path is a subdirectory of start,
    return the full subpath; otherwise, return the basename.
    """
    if path.startswith(start):
        return path[len(start):].lstrip(os.sep)
    return os.path.basename(path)

def add_alpha_channel(image):
    """
    Ensure the image has an alpha channel.
    If the image is 3-channel (BGR), add a fully opaque alpha channel.
    If it already has 4 channels, return it unchanged.
    """
    if image is None:
        return None
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return image
    # If 3-channel, add an alpha channel.
    if image.dtype == np.uint8:
        alpha = np.full((image.shape[0], image.shape[1]), 255, dtype=np.uint8)
    elif image.dtype == np.uint16:
        alpha = np.full((image.shape[0], image.shape[1]), 65535, dtype=np.uint16)
    else:
        alpha = np.ones((image.shape[0], image.shape[1]), dtype=image.dtype)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    image[:, :, 3] = alpha
    return image

def extract_white_patch(rgb, threshold=0.9, alpha=None):
    """
    Given a normalized RGB image (values in [0,1]), find the largest connected
    white (or neutral gray) patch and return its average color.
    Only consider pixels with alpha > 0 (if provided).
    """
    if alpha is not None:
        mask = np.all(rgb >= threshold, axis=2) & (alpha > 0)
    else:
        mask = np.all(rgb >= threshold, axis=2)
    mask = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    white_patch_mask = (labels == largest_label)
    if np.count_nonzero(white_patch_mask) == 0:
        return None
    return np.mean(rgb[white_patch_mask], axis=0)

def compute_scaling_factors(reference_color):
    """
    Compute per-channel scaling factors to adjust the reference white/gray to [1,1,1].
    """
    target = np.array([1.0, 1.0, 1.0])
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(reference_color > 0, target / reference_color, 1.0)

def load_image_as_rgba(path):
    """
    Load an image with cv2.IMREAD_UNCHANGED and ensure it has 4 channels.
    The returned image is in native BGRA order.
    """
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    return add_alpha_channel(image)

def white_balance_image_native(image, white_threshold=0.9, auto_reference=True, reference_color=None):
    """
    White balance an image in native BGRA space.

    Parameters:
      image: Input image in BGRA order.
      white_threshold: Threshold to consider a pixel white.
      auto_reference: If True and no reference_color is given, compute one.
      reference_color: A 3-element array representing the white reference.

    Returns:
      The white-balanced image in BGRA order.
    """
    if image is None:
        return None
    if image.shape[2] != 4:
        image = add_alpha_channel(image)
    # Set normalization factor based on dtype.
    norm = 255.0 if image.dtype == np.uint8 else 65535.0 if image.dtype == np.uint16 else 1.0
    rgb = image[:, :, :3] / norm
    alpha = image[:, :, 3] / norm
    if reference_color is None and auto_reference:
        candidate = extract_white_patch(rgb, threshold=white_threshold, alpha=alpha)
        if candidate is not None:
            reference_color = candidate
        else:
            return image
    if reference_color is None:
        return image
    scaling_factors = compute_scaling_factors(reference_color)
    balanced_rgb = np.clip(rgb * scaling_factors.reshape(1, 1, 3), 0, 1)
    # Convert back to original dtype.
    if image.dtype == np.uint8:
        balanced_rgb = (balanced_rgb * 255).astype(np.uint8)
        alpha_channel = (alpha * 255).astype(np.uint8)
    elif image.dtype == np.uint16:
        balanced_rgb = (balanced_rgb * 65535).astype(np.uint16)
        alpha_channel = (alpha * 65535).astype(np.uint16)
    else:
        alpha_channel = alpha
    return np.dstack((balanced_rgb, alpha_channel))

def save_lossless_webp(image, out_path):
    """
    Save the image as lossless WebP using OpenCV if available, otherwise Pillow.
    """
    try:
        cv2.imwrite(out_path, image, [cv2.IMWRITE_WEBP_LOSSLESS, 1])
    except AttributeError:
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
        pil_img.save(out_path, "WEBP", lossless=True)

# --- Per-Image Processing ---

def process_image(file, input_dir, output_dir, reference_color, white_threshold):
    """
    Process a single image: white balance it and save as lossless WebP.
    Prints a message when finished.
    """
    image_path = os.path.join(input_dir, file)
    image = load_image_as_rgba(image_path)
    if image is None:
        msg = f"Error reading {image_path}"
        print(msg)
        return msg
    balanced_image = white_balance_image_native(
        image, white_threshold=white_threshold, auto_reference=False, reference_color=reference_color
    )
    base_name = os.path.splitext(file)[0]
    out_path = os.path.join(output_dir, base_name + ".webp")
    save_lossless_webp(balanced_image, out_path)
    msg = f"Processed {image_path} -> {out_path}"
    print(msg)
    return msg

# --- Per-Folder Processing ---

def process_directory(input_dir, output_root, white_threshold=0.9):
    """
    Process all images in a directory:
      - Compute a white reference from the 30th (or last) file.
      - White balance each image using that reference.
      - Save images as lossless WebP.
    The source folder structure is preserved.
    """
    supported_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp')
    images = [f for f in os.listdir(input_dir) if f.lower().endswith(supported_extensions)]
    if not images:
        return f"No supported images found in {input_dir}"
    sorted_images = sorted(images, key=natural_key)
    ref_image_name = sorted_images[29] if len(sorted_images) >= 30 else sorted_images[-1]
    ref_image_path = os.path.join(input_dir, ref_image_name)
    ref_image = load_image_as_rgba(ref_image_path)
    if ref_image is None:
        return f"Error reading reference image {ref_image_path}; skipping folder."
    norm = 255.0 if ref_image.dtype == np.uint8 else 65535.0 if ref_image.dtype == np.uint16 else 1.0
    rgb = ref_image[:, :, :3] / norm
    alpha = ref_image[:, :, 3] / norm
    candidate = extract_white_patch(rgb, threshold=white_threshold, alpha=alpha)
    if candidate is None:
        return f"No valid white patch found in {ref_image_path}; skipping folder."
    reference_color = candidate
    print(f"In folder '{input_dir}', using '{ref_image_name}' as reference.")
    print(f"Reference white/gray color: {reference_color}")
    rel_path = safe_relpath(input_dir, BASE_INPUT_FOLDER)
    output_dir = os.path.join(output_root, rel_path)
    os.makedirs(output_dir, exist_ok=True)
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_image, file, input_dir, output_dir, reference_color, white_threshold): file
                   for file in images}
        for future in as_completed(futures):
            _ = future.result()  # Each file prints its own status.
    return f"Finished processing folder: {input_dir}"

# --- Main Function ---

BASE_INPUT_FOLDER = None

def main():
    """
    Standalone entry point.
    Prompts for an input folder, gathers all subdirectories containing supported images,
    and processes each directory sequentially.
    The output folder structure is preserved and a message is printed when each folder is finished.
    """
    global BASE_INPUT_FOLDER
    BASE_INPUT_FOLDER = input("Enter the path to the input folder: ").strip()
    if not os.path.isdir(BASE_INPUT_FOLDER):
        print("The provided path is not a valid directory.")
        return
    parent_dir = os.path.dirname(BASE_INPUT_FOLDER)
    folder_name = os.path.basename(BASE_INPUT_FOLDER)
    output_root = os.path.join(parent_dir, folder_name + "_wb")
    os.makedirs(output_root, exist_ok=True)
    dirs_to_process = []
    supported_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp')
    for root, dirs, files in os.walk(BASE_INPUT_FOLDER):
        if any(file.lower().endswith(supported_extensions) for file in files):
            dirs_to_process.append(root)
    for d in sorted(dirs_to_process, key=natural_key):
        result = process_directory(d, output_root, white_threshold=0.9)
        print(result)

if __name__ == "__main__":
    main()
