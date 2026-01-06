"""
Asset Baker

Packs image folders into single .npy memory-mapped files for instant seeking.
Resizes images to the target resolution and stores as RGBA arrays.
"""
import argparse
import logging
import os
import sys
import time
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Tuple
from PIL import Image

# --- 1. SETUP PATHS ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import settings

# Force RGBA for consistency (Main + Float compatibility)
CHANNELS = 4
HEADLESS_RES = getattr(settings, 'HEADLESS_RES', (640, 480))


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


def process_folder_to_slab(args: Tuple[Path, Path, Tuple[int, int]]) -> Optional[str]:
    """
    Reads all images in a folder, resizes them, and writes them
    into a single pre-allocated .npy slab file.
    
    Returns:
        None on success, error message string on failure
    """
    src_folder, dest_file, resolution = args
    src_path = Path(src_folder)

    # 1. Gather Files
    valid = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    files = sorted([p for p in src_path.iterdir() if p.is_file() and p.suffix.lower() in valid])

    if not files:
        return f"Skipped (No images): {src_folder}"

    count = len(files)
    w, h = resolution

    # 2. Pre-allocate Memory Mapped File
    # This creates the full file on disk instantly filled with zeros
    Path(dest_file).parent.mkdir(parents=True, exist_ok=True)

    try:
        # Shape: (Frames, Height, Width, RGBA)
        slab = np.lib.format.open_memmap(
            dest_file,
            mode='w+',
            dtype=np.uint8,
            shape=(count, h, w, CHANNELS)
        )

        # 3. Fill the Slab
        for i, fp in enumerate(files):
            with Image.open(fp) as im:
                # Convert & Resize
                im = im.convert('RGBA')
                im = im.resize((w, h), Image.NEAREST)  # Nearest is fastest, use BILINEAR for quality

                # Write directly to disk-backed memory
                slab[i] = np.asarray(im)

        # Flush changes to disk
        slab.flush()
        return None  # Success

    except Exception as e:
        return f"Error processing {src_folder}: {e}"


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Pack image folders into single .npy memory-mapped files for instant seeking."
    )
    parser.add_argument(
        "-i", "--input-dir",
        type=str,
        required=True,
        help="Source root directory containing image folders"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Output root directory (default: {input_dir}_slab)"
    )
    parser.add_argument(
        "-r", "--resolution",
        type=str,
        default=None,
        help="Target resolution as WxH (e.g., 640x480). Default: from settings.HEADLESS_RES"
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
    """Main entry point."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Determine resolution
    if args.resolution:
        try:
            w, h = map(int, args.resolution.split('x'))
            resolution = (w, h)
        except ValueError:
            logging.error(f"Invalid resolution format: {args.resolution}. Use WxH (e.g., 640x480)")
            sys.exit(1)
    else:
        resolution = HEADLESS_RES
    
    # Resolve paths
    input_root = Path(args.input_dir).expanduser().resolve()
    if not input_root.is_dir():
        logging.error(f"Invalid directory: {input_root}")
        sys.exit(1)
    
    if args.output_dir:
        output_root = Path(args.output_dir).expanduser().resolve()
    else:
        output_root = input_root.parent / f"{input_root.name}_slab"
    
    logging.info("--- SLAB BAKER (High Performance Memmap) ---")
    logging.info(f"Target Resolution: {resolution} (RGBA)")
    logging.info("This will pack folders into single .npy files for instant seeking.")
    logging.info(f"Source: {input_root}")
    logging.info(f"Dest:   {output_root}")

    tasks: list[Tuple[Path, Path, Tuple[int, int]]] = []

    logging.info("Scanning folders...")
    for root, dirs, files in os.walk(input_root):
        # Check if this folder contains images
        has_images = any(f.lower().endswith(('.png', '.jpg', '.webp')) for f in files)

        if has_images:
            rel_path = Path(root).relative_to(input_root)
            dest_file = output_root / rel_path / "frames.npy"
            tasks.append((Path(root), dest_file, resolution))

    total = len(tasks)
    if total == 0:
        logging.warning("No image folders found.")
        sys.exit(0)

    logging.info(f"Found {total} folder(s) to pack. Starting bake...")

    start = time.time()
    errors: list[str] = []

    # Run in parallel
    with ProcessPoolExecutor() as ex:
        for i, res in enumerate(ex.map(process_folder_to_slab, tasks)):
            if res:
                errors.append(res)

            # Simple progress
            percent = ((i + 1) / total) * 100
            logging.info(f"Progress: {percent:.1f}% ({i + 1}/{total})")

    duration = time.time() - start
    logging.info(f"Done in {duration:.2f}s.")
    if errors:
        logging.warning(f"Errors ({len(errors)}):")
        for e in errors:
            logging.warning(f"  - {e}")
    else:
        logging.info("All folders processed successfully.")


if __name__ == "__main__":
    main()