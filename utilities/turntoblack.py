"""
Turn to Black

Restores perceptual appearance of images whose alpha masks erased dark RGB data,
by compositing the image over black and setting full opacity.

Supports both command-line and interactive modes.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional


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


def recover_image(filepath: Path, output_folder: Path) -> Optional[Path]:
    """
    Recover image by compositing over black and setting full opacity.
    
    Args:
        filepath: Path to input image file
        output_folder: Path to output directory
        
    Returns:
        Path to output file on success, None on failure
    """
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
            output_folder.mkdir(parents=True, exist_ok=True)
            output_path = output_folder / filepath.name

            # Ensure lossless saving for WebP
            ext = filepath.suffix.lower()
            if ext == ".webp":
                out_img.save(output_path, "WEBP", lossless=True)
            else:
                out_img.save(output_path)

            return output_path

    except Exception as e:
        logging.error(f"Error processing {filepath}: {e}")
        return None


def process_folder(
    input_folder: Path,
    output_folder: Optional[Path] = None,
    max_workers: int = 8,
    dry_run: bool = False
) -> None:
    """
    Process all images in a folder.
    
    Args:
        input_folder: Path to input folder
        output_folder: Path to output folder (default: {input_folder}_blackrecovered)
        max_workers: Number of parallel workers
        dry_run: If True, only print what would be processed
    """
    if output_folder is None:
        output_folder = input_folder.parent / f"{input_folder.name}_blackrecovered"
    
    output_folder.mkdir(parents=True, exist_ok=True)

    files = [
        input_folder / f
        for f in os.listdir(input_folder)
        if Path(f).suffix.lower() in (".png", ".webp")
    ]

    if not files:
        logging.warning(f"No PNG or WebP files found in {input_folder}")
        return

    logging.info(f"Found {len(files)} file(s) to process")
    if dry_run:
        logging.info("DRY RUN MODE - No files will be modified")
        for f in files:
            logging.info(f"Would recover: {f}")
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(recover_image, f, output_folder) for f in files]

            for future in as_completed(futures):
                result = future.result()
                if result:
                logging.debug(f"Recovered: {result.name}")
            else:
                logging.warning(f"Failed to recover a file")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Restore perceptual appearance of images whose alpha masks erased dark RGB data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single folder
  python turntoblack.py /path/to/images

  # Process multiple folders
  python turntoblack.py /path/to/images1 /path/to/images2

  # Specify output directory
  python turntoblack.py -i /path/to/images -o /path/to/output

  # Dry run to see what would be processed
  python turntoblack.py -i /path/to/images --dry-run
        """
    )
    parser.add_argument(
        "folders",
        nargs="*",
        help="Input folder path(s) containing images to process"
    )
    parser.add_argument(
        "-i", "--input-dir",
        type=str,
        action="append",
        help="Input folder path (can be specified multiple times)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Output folder path (default: {input_dir}_blackrecovered for each input)"
    )
    parser.add_argument(
        "-w", "--max-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode: show what would be processed without modifying files"
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
    
    # Collect input folders
    input_folders: list[Path] = []
    if args.folders:
        input_folders.extend([Path(f).expanduser().resolve() for f in args.folders])
    if args.input_dir:
        input_folders.extend([Path(f).expanduser().resolve() for f in args.input_dir])
    
    # If no folders specified, use interactive mode
    if not input_folders:
        input_path_str = input("Enter the path to the folder containing the images: ").strip()
        input_folders = [Path(input_path_str).expanduser().resolve()]
    
    # Validate folders
    for folder in input_folders:
        if not folder.exists() or not folder.is_dir():
            logging.error(f"'{folder}' is not a valid directory.")
            sys.exit(1)
    
    # Determine workers
    if args.max_workers:
        max_workers = args.max_workers
        if max_workers < 1:
            logging.error("Max workers must be >= 1")
            sys.exit(1)
    else:
        max_workers = os.cpu_count() or 1
    
    # Process each folder
    for input_folder in input_folders:
        logging.info(f"Processing folder: {input_folder}")
        output_folder = None
        if args.output_dir:
            output_folder = Path(args.output_dir).expanduser().resolve()
        process_folder(
            input_folder,
            output_folder=output_folder,
            max_workers=max_workers,
            dry_run=args.dry_run
        )
        logging.info(f"Finished processing: {input_folder}")


if __name__ == "__main__":
    main()
