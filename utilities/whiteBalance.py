#!/usr/bin/env python3
"""
White Balance Correction

Processes images in their native BGRA space with white balancing,
using the 30th image (or the last one if fewer than 30 exist) in each folder as
the white reference. It computes per-channel scaling factors from a white/neutral
patch and then applies these factors to all images in the folder.

The output images are saved as lossless WebP. The full‐resolution input image is
output without resizing.

This version preserves the source folder structure and processes directories
sequentially (while processing images concurrently).
"""

import argparse
import logging
import os
import re
import sys
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple


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
    Given a normalized RGB image (values in [0,1]), apply a Gaussian blur
    to reduce noise, then find the largest connected white (or neutral gray) patch
    and return its average color. Only consider pixels with alpha > 0 (if provided).
    """
    blurred_rgb = cv2.GaussianBlur(rgb, (5, 5), 0)
    if alpha is not None:
        mask = np.all(blurred_rgb >= threshold, axis=2) & (alpha > 0)
    else:
        mask = np.all(blurred_rgb >= threshold, axis=2)
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

    If no white patch is found and auto_reference is True, this function falls back to a
    generic white balance using overall image statistics (the 95th percentile is used here).

    The computed per-channel scaling factors are dampened and clamped to reduce overcorrection.
    """
    if image is None:
        return None
    if image.shape[2] != 4:
        image = add_alpha_channel(image)

    norm = 255.0 if image.dtype == np.uint8 else 65535.0 if image.dtype == np.uint16 else 1.0
    rgb = image[:, :, :3] / norm
    alpha = image[:, :, 3] / norm

    if reference_color is None and auto_reference:
        candidate = extract_white_patch(rgb, threshold=white_threshold, alpha=alpha)
        if candidate is None:
            print("No white patch found; applying generic white balance using overall image statistics.")
            candidate = np.percentile(rgb, 95, axis=(0, 1))
        reference_color = candidate

    scaling_factors = compute_scaling_factors(reference_color)
    # Dampen the gain: apply only 50% of the computed correction.
    dampening_factor = 0.5
    scaling_factors = 1.0 + dampening_factor * (scaling_factors - 1.0)
    # Clamp to a lower maximum gain to avoid washout.
    scaling_factors = np.clip(scaling_factors, 1.0, 1.2)
    balanced_rgb = np.clip(rgb * scaling_factors.reshape(1, 1, 3), 0, 1)

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

def process_image(file: str, input_dir: Path, output_dir: Path, reference_color: np.ndarray, white_threshold: float) -> str:
    """
    Process a single image: white balance it and save as lossless WebP.
    Returns a status message.
    """
    image_path = input_dir / file
    image = load_image_as_rgba(str(image_path))
    if image is None:
        msg = f"Error reading {image_path}"
        logging.error(msg)
        return msg
    balanced_image = white_balance_image_native(
        image, white_threshold=white_threshold, auto_reference=False, reference_color=reference_color
    )
    base_name = Path(file).stem
    out_path = output_dir / f"{base_name}.webp"
    save_lossless_webp(balanced_image, str(out_path))
    msg = f"Processed {image_path} -> {out_path}"
    logging.debug(msg)
    return msg


# --- Per-Folder Processing ---

def process_directory(input_dir: Path, output_root: Path, base_input_folder: Path, white_threshold: float = 0.9, workers: Optional[int] = None) -> str:
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
    ref_image_path = input_dir / ref_image_name
    ref_image = load_image_as_rgba(str(ref_image_path))
    if ref_image is None:
        return f"Error reading reference image {ref_image_path}; skipping folder."
    norm = 255.0 if ref_image.dtype == np.uint8 else 65535.0 if ref_image.dtype == np.uint16 else 1.0
    rgb = ref_image[:, :, :3] / norm
    alpha = ref_image[:, :, 3] / norm
    candidate = extract_white_patch(rgb, threshold=white_threshold, alpha=alpha)
    if candidate is None:
        logging.warning(
            f"No valid white patch found in {ref_image_path}; applying generic white balance using the 95th percentile of the overall image.")
        candidate = np.percentile(rgb, 95, axis=(0, 1))
    reference_color = candidate
    logging.info(f"In folder '{input_dir}', using '{ref_image_name}' as reference.")
    logging.info(f"Reference white/gray color: {reference_color}")
    rel_path = safe_relpath(str(input_dir), str(base_input_folder))
    output_dir = output_root / rel_path
    output_dir.mkdir(parents=True, exist_ok=True)
    
    max_workers = workers or os.cpu_count() or 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_image, file, input_dir, output_dir, reference_color, white_threshold): file
                   for file in images}
        for future in as_completed(futures):
            _ = future.result()
    return f"Finished processing folder: {input_dir}"


# --- Main Function ---

def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Apply white balance correction to images using per-folder reference images."
    )
    parser.add_argument(
        "-i", "--input-dir",
        type=str,
        required=True,
        help="Input folder path containing images to process"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Output folder path (default: {input_dir}_wb)"
    )
    parser.add_argument(
        "-t", "--white-threshold",
        type=float,
        default=0.9,
        help="White patch detection threshold (0.0-1.0, default: 0.9)"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=None,
        help="Number of parallel workers per folder (default: CPU count)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)"
    )
    return parser.parse_args()


def main() -> None:
    """
    Standalone entry point.
    Processes all subdirectories containing supported images,
    and processes each directory sequentially.
    The output folder structure is preserved.
    """
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    
    base_input_folder = Path(args.input_dir).expanduser().resolve()
    if not base_input_folder.is_dir():
        logging.error(f"The provided path is not a valid directory: {base_input_folder}")
        sys.exit(1)
    
    if args.output_dir:
        output_root = Path(args.output_dir).expanduser().resolve()
    else:
        parent_dir = base_input_folder.parent
        folder_name = base_input_folder.name
        output_root = parent_dir / f"{folder_name}_wb"
    
    output_root.mkdir(parents=True, exist_ok=True)
    
    dirs_to_process: list[Path] = []
    supported_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp')
    for root, dirs, files in os.walk(base_input_folder):
        if any(file.lower().endswith(supported_extensions) for file in files):
            dirs_to_process.append(Path(root))
    
    if not dirs_to_process:
        logging.warning(f"No directories with supported images found in {base_input_folder}")
        sys.exit(0)
    
    logging.info(f"Found {len(dirs_to_process)} directory(ies) to process")
    logging.info(f"Output root: {output_root}")
    logging.info(f"White threshold: {args.white_threshold}")
    
    for d in sorted(dirs_to_process, key=lambda p: natural_key(str(p))):
        result = process_directory(d, output_root, base_input_folder, white_threshold=args.white_threshold, workers=args.workers)
        logging.info(result)


if __name__ == "__main__":
    main()
