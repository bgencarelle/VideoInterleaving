"""
Side-by-Side JPEG Converter

Converts WebP/PNG images to JPEG format with side-by-side layout:
RGB image on the left, alpha mask on the right.
Useful for formats that don't support alpha channels natively.
"""
import argparse
import logging
import os
import sys
import cv2
import numpy as np
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import Tuple, Optional


def process_file(file_path: Path, input_root: Path, output_root: Path, quality: int = 90) -> Tuple[bool, Optional[str]]:
    """
    Reads a WebP/PNG file, ensures it has an alpha mask (generating one if missing),
    stacks them side-by-side, and saves as a high-quality JPEG.
    
    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    try:
        # 1. Load Image
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return False, "Load failed"

        h, w = img.shape[:2]

        color_bgr = None
        alpha_single = None

        # 2. Logic to handle different channel counts
        if img.ndim == 2:
            # Case: Grayscale (1 Channel)
            # Convert Gray -> BGR
            color_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # Generate solid white alpha (fully visible)
            alpha_single = np.full((h, w), 255, dtype=np.uint8)

        elif img.ndim == 3:
            channels = img.shape[2]

            if channels == 4:
                # Case: BGRA (Has Transparency)
                # Split channels
                b, g, r, a = cv2.split(img)
                color_bgr = cv2.merge([b, g, r])
                alpha_single = a

            elif channels == 3:
                # Case: BGR (No Transparency)
                color_bgr = img
                # Generate solid white alpha
                alpha_single = np.full((h, w), 255, dtype=np.uint8)

            else:
                return False, f"Unsupported channel count: {channels}"
        else:
            return False, f"Unknown dimensions: {img.shape}"

        # 3. Prepare Alpha for Stacking
        # We need alpha to be 3 channels (grayscale BGR) to stack next to color
        alpha_bgr = cv2.merge([alpha_single, alpha_single, alpha_single])

        # 4. Stack horizontally: [Color | Alpha]
        sbs = np.hstack([color_bgr, alpha_bgr])

        # 5. Construct Output Path
        rel_path = file_path.relative_to(input_root)
        rel_path_jpg = rel_path.with_suffix(".jpg")
        out_path = output_root / rel_path_jpg

        # Ensure output dir exists
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # 6. Save as JPEG
        success = cv2.imwrite(str(out_path), sbs, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

        if not success:
            return False, "Write failed"

        return True, None

    except Exception as e:
        return False, str(e)


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
        description="Convert WebP/PNG images to JPEG with side-by-side RGB+Alpha layout."
    )
    parser.add_argument(
        "-i", "--input-dir",
        type=str,
        required=True,
        help="Source folder path containing WebP/PNG images"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Output folder path (default: {input_dir}_sbs)"
    )
    parser.add_argument(
        "-q", "--quality",
        type=int,
        default=90,
        choices=range(1, 101),
        help="JPEG quality (1-100, default: 90)"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)"
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
    
    # Resolve paths
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        logging.error(f"Directory '{input_dir}' not found.")
        sys.exit(1)
    
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = input_dir.parent / f"{input_dir.name}_sbs"
    
    # Find files
    logging.info(f"Scanning '{input_dir}' for .webp files...")
    files = list(input_dir.rglob("*.webp"))
    total_files = len(files)

    if total_files == 0:
        logging.warning("No .webp files found. Nothing to do.")
        sys.exit(0)
    
    logging.info(f"Found {total_files} file(s) to process.")
    
    # Determine workers
    if args.workers:
        workers = args.workers
        if workers < 1:
            logging.error("Worker count must be >= 1")
            sys.exit(1)
    else:
        workers = max(1, os.cpu_count() or 1)
    
    logging.info(f"Starting conversion using {workers} CPU core(s)...")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"JPEG quality: {args.quality}")
    
    # Parallel Processing
    start_time = time.time()
    errors: list[str] = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_file, f, input_dir, output_dir, args.quality): f
            for f in files
        }

        for future in tqdm(as_completed(futures), total=total_files, unit="img"):
            file_path = futures[future]
            try:
                success, error_msg = future.result()
                if not success:
                    errors.append(f"{file_path}: {error_msg}")
            except Exception as e:
                errors.append(f"{file_path}: Crash - {e}")

    end_time = time.time()
    duration = end_time - start_time

    logging.info("=" * 40)
    logging.info("Processing Complete!")
    logging.info(f"Time taken: {duration:.2f} seconds")
    if duration > 0:
        logging.info(f"Average speed: {total_files / duration:.1f} fps")
    logging.info(f"Output saved to: {output_dir}")

    if errors:
        logging.warning(f"{len(errors)} error(s) encountered:")
        for err in errors[:10]:
            logging.warning(f"  - {err}")
        if len(errors) > 10:
            logging.warning(f"  ... and {len(errors) - 10} more errors")
    else:
        logging.info("0 errors.")


if __name__ == "__main__":
    main()