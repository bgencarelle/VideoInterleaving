import csv
import logging
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import List, Tuple

from PIL import Image
from tqdm import tqdm
import argparse
import os

# Constants
DEFAULT_CSV_PATTERN = "*.csv"  # Pattern to find CSV files in the current directory
DEFAULT_OUTPUT_FOLDER = "merged_images_png"  # Folder for PNG images
NAME_PREFIX = "NAME_"  # Prefix for output filenames
PADDING = 6  # Number of digits for padding (e.g., 000001)
MAX_RETRIES = 3  # Maximum number of retries for failed processing
DEFAULT_COMPRESSION_LEVEL = 1  # PNG compression level (0-9)

def find_csv_files(pattern: str = DEFAULT_CSV_PATTERN) -> List[Path]:
    """
    Find all CSV files matching the given pattern in the current directory.
    """
    return sorted(Path('.').glob(pattern))

def read_csv(file_path: Path) -> List[List[str]]:
    """
    Read the CSV file and return a sorted list of rows, excluding the header.
    Each row is expected to have at least three columns:
    Absolute Index, Main Image Path, Float Image Path
    Sorting is based on Absolute Index to ensure alphabetical processing.
    """
    rows = []
    with file_path.open(newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None)  # Skip header
        for row in reader:
            if len(row) >= 3:
                rows.append(row)
            else:
                logging.warning(f"Skipping malformed row in '{file_path}': {row}")

    try:
        rows.sort(key=lambda x: int(x[0]))
    except ValueError as e:
        logging.error(f"Error sorting rows in '{file_path}': {e}")

    return rows

def alpha_composite_images(main_image_path: Path, float_image_path: Path) -> Image.Image:
    """
    Composite the float image onto the main image while respecting alpha.
    Returns the merged PIL Image object.
    """
    with Image.open(main_image_path).convert("RGBA") as main_img, \
         Image.open(float_image_path).convert("RGBA") as float_img:
        
        # Ensure images are the same size
        if main_img.size != float_img.size:
            logging.debug(
                f"Resizing '{float_image_path}' from {float_img.size} to {main_img.size} to match '{main_image_path}'."
            )
            float_img = float_img.resize(main_img.size, Image.ANTIALIAS)
        
        # Perform alpha compositing: float_image over main_image
        merged_image = Image.alpha_composite(main_img, float_img)
    
    return merged_image

def save_image_atomic(merged_image: Image.Image, output_path: Path, compression_level: int):
    """
    Save the merged image in PNG format atomically.
    Uses a temporary file first, then renames to final path to ensure atomicity.
    """
    temp_path = output_path.with_suffix('.png.tmp')
    try:
        merged_image.save(temp_path, "PNG", compress_level=compression_level)
        temp_path.rename(output_path)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise e

def process_row(
    row: List[str],
    output_folder: Path,
    name_prefix: str,
    padding: int,
    compression_level: int,
    attempt: int = 1
) -> Tuple[str, str, str]:
    """
    Process a single CSV row: composite images and save the output.
    Implements retry mechanism for robustness.
    Returns a tuple: (output_filename, status, error)
    """
    absolute_index, main_image_str, float_image_str = row
    main_image_path = Path(main_image_str)
    float_image_path = Path(float_image_str)

    # Create padded index
    padded_index = absolute_index.zfill(padding)

    # Create output filename
    output_filename = f"{name_prefix}{padded_index}.png"

    # Define full output path
    output_path = output_folder / output_filename

    try:
        # Composite images
        merged_image = alpha_composite_images(main_image_path, float_image_path)

        # Save image atomically
        save_image_atomic(merged_image, output_path, compression_level)

        return (output_filename, "Success", "")
    except Exception as e:
        if attempt < MAX_RETRIES:
            logging.warning(f"Attempt {attempt} failed for '{output_filename}'. Retrying...")
            return process_row(row, output_folder, name_prefix, padding, compression_level, attempt=attempt + 1)
        else:
            logging.error(f"Failed to process '{output_filename}' after {MAX_RETRIES} attempts: {e}")
            return (output_filename, "Failed", str(e))

def process_all_rows(
    rows: List[List[str]],
    output_folder: Path,
    name_prefix: str,
    padding: int,
    compression_level: int,
    max_workers: int
) -> List[Tuple[str, str, str]]:
    """
    Process all CSV rows using parallel processing.
    Returns a list of results: [(output_filename, status, error), ...]
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Partial function to fix certain parameters except 'row'
        process_func = partial(
            process_row,
            output_folder=output_folder,
            name_prefix=name_prefix,
            padding=padding,
            compression_level=compression_level
        )

        # Submit all tasks to the executor
        futures = {executor.submit(process_func, row): row for row in rows}

        # Use tqdm to display a progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Images"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                # This should not happen as exceptions are handled in process_row
                logging.error(f"Unhandled exception: {e}")
                continue

    return results

def setup_logging(log_level: str = "INFO"):
    """
    Setup logging configuration.
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Composite images based on CSV files and save as PNG.")
    parser.add_argument(
        '-p', '--pattern',
        type=str,
        default=DEFAULT_CSV_PATTERN,
        help=f"Pattern to find CSV files (default: '{DEFAULT_CSV_PATTERN}')"
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=DEFAULT_OUTPUT_FOLDER,
        help=f"Output folder for merged images (default: '{DEFAULT_OUTPUT_FOLDER}')"
    )
    parser.add_argument(
        '-c', '--compression',
        type=int,
        default=DEFAULT_COMPRESSION_LEVEL,
        choices=range(0, 10),
        help=f"PNG compression level (0-9, default: {DEFAULT_COMPRESSION_LEVEL})"
    )
    parser.add_argument(
        '-l', '--log',
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    parser.add_argument(
        '-n', '--name-prefix',
        type=str,
        default=NAME_PREFIX,
        help=f"Prefix for output filenames (default: '{NAME_PREFIX}')"
    )
    parser.add_argument(
        '--padding',
        type=int,
        default=PADDING,
        help=f"Number of digits for filename padding (default: {PADDING})"
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help="Number of parallel workers (default: number of CPU cores)"
    )

    return parser.parse_args()

def main():
    """
    Main function to orchestrate the image merging process.
    """
    args = parse_arguments()

    # Setup logging
    setup_logging(args.log)
    logging.info("Starting image merging process.")

    # Determine output folder
    output_folder = Path(args.output)
    logging.info(f"Output folder: '{output_folder}'")
    logging.info(f"PNG compression level: {args.compression}")

    # Step 1: Find CSV files
    csv_files = find_csv_files(pattern=args.pattern)
    if not csv_files:
        logging.error(f"No CSV files found matching the pattern '{args.pattern}'.")
        return

    logging.info(f"Found {len(csv_files)} CSV file(s): {', '.join(str(f) for f in csv_files)}")

    # Step 2: Process each CSV file
    for csv_file in csv_files:
        logging.info(f"\nProcessing CSV file: '{csv_file}'")
        rows = read_csv(csv_file)
        logging.info(f"Loaded {len(rows)} row(s) from '{csv_file}'.")

        if not rows:
            logging.warning(f"No valid rows to process in '{csv_file}'. Skipping...")
            continue

        # Step 3: Determine the number of workers
        max_workers = args.workers or os.cpu_count()
        logging.info(f"Using {max_workers} parallel worker(s).")

        # Step 4: Process all rows with parallel execution
        results = process_all_rows(
            rows=rows,
            output_folder=output_folder,
            name_prefix=args.name_prefix,
            padding=args.padding,
            compression_level=args.compression,
            max_workers=max_workers
        )

        # Step 5: Summary of results
        success_count = sum(1 for _, status, _ in results if status == "Success")
        failure_count = len(results) - success_count
        logging.info(
            f"Completed processing '{csv_file}'. {success_count} succeeded, {failure_count} failed."
        )

    logging.info(f"\nAll processing completed. Merged images saved to '{output_folder.resolve()}'.")
    logging.info("Image merging process finished successfully.")

if __name__ == "__main__":
    main()
