import os
import csv
from PIL import Image
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm
import logging
import shutil

# Constants
MAIN_FLOAT_CSV_PATTERN = "*.csv"  # Pattern to find CSV files in current directory
OUTPUT_FOLDER_LOSSLESS = "merged_images_lossless"  # Folder for lossless WebP images
OUTPUT_FOLDER_COMPRESSED = "merged_images_compressed"  # Folder for compressed WebP images
NAME_PREFIX = "NAME_"  # Prefix for output filenames
PADDING = 6  # Number of digits for padding (e.g., 0000)
QUALITY_LOSSLESS = 100  # Quality setting for lossless WebP
QUALITY_COMPRESSED = 99  # Quality setting for compressed WebP
MAX_RETRIES = 3  # Maximum number of retries for failed processing


def find_csv_files(pattern=MAIN_FLOAT_CSV_PATTERN):
    """
    Find all CSV files matching the given pattern in the current directory.
    """
    return sorted(glob.glob(pattern))


def read_csv(file_path):
    """
    Read the CSV file and return a sorted list of rows, excluding the header.
    Each row is expected to have at least three columns:
    Absolute Index, Main Image Path, Float Image Path
    Sorting is based on Absolute Index to ensure alphabetical processing.
    """
    rows = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None)  # Skip header
        for row in reader:
            if len(row) >= 3:
                rows.append(row)
            else:
                logging.warning(f"Skipping malformed row: {row}")

    # Sort rows based on Absolute Index (assuming it's an integer)
    try:
        rows.sort(key=lambda x: int(x[0]))
    except ValueError as e:
        logging.error(f"Error sorting rows: {e}")

    return rows


def alpha_composite_images(main_image_path, float_image_path):
    """
    Composite the float image onto the main image while respecting alpha.
    Returns the merged PIL Image object.
    """
    main_image = Image.open(main_image_path).convert("RGBA")
    float_image = Image.open(float_image_path).convert("RGBA")

    # Ensure images are the same size
    if main_image.size != float_image.size:
        logging.warning(
            f"Image sizes do not match for '{float_image_path}' and '{main_image_path}'. Resizing float image.")
        float_image = float_image.resize(main_image.size, Image.ANTIALIAS)

    # Perform alpha compositing: float_image over main_image
    merged_image = Image.alpha_composite(main_image, float_image)

    return merged_image


def save_image_atomic(merged_image, output_path, mode):
    """
    Save the merged image in the specified mode atomically.
    - mode: 'lossless' or 'compressed'
    Uses a temporary file first, then renames to final path to ensure atomicity.
    """
    temp_path = output_path + ".tmp"
    try:
        if mode == "lossless":
            merged_image.save(temp_path, "WEBP", lossless=True, quality=QUALITY_LOSSLESS)
        elif mode == "compressed":
            merged_image.save(temp_path, "WEBP", lossless=False, quality=QUALITY_COMPRESSED)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        # Atomically move temp file to final destination
        shutil.move(temp_path, output_path)
    except Exception as e:
        # If saving fails, ensure temp file is removed
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e


def process_row(row, output_folder, name_prefix, padding, mode, attempt=1):
    """
    Process a single CSV row: composite images and save the output.
    Implements retry mechanism for robustness.
    Returns a tuple: (output_filename, status, error)
    """
    absolute_index, main_image_path, float_image_path = row

    # Create padded index
    padded_index = str(absolute_index).zfill(padding)

    # Create output filename
    output_filename = f"{name_prefix}{padded_index}.webp"

    # Define full output path
    output_path = os.path.join(output_folder, output_filename)

    try:
        # Composite images
        merged_image = alpha_composite_images(main_image_path, float_image_path)

        # Save image atomically based on mode
        save_image_atomic(merged_image, output_path, mode)

        return (output_filename, "Success", None)
    except Exception as e:
        if attempt < MAX_RETRIES:
            logging.warning(f"Attempt {attempt} failed for '{output_filename}'. Retrying...")
            return process_row(row, output_folder, name_prefix, padding, mode, attempt=attempt + 1)
        else:
            logging.error(f"Failed to process '{output_filename}' after {MAX_RETRIES} attempts: {e}")
            return (output_filename, "Failed", str(e))


def process_all_rows(rows, output_folder, name_prefix, padding, mode, max_workers=None):
    """
    Process all CSV rows using parallel processing.
    Returns a list of results: [(output_filename, status, error), ...]
    """
    os.makedirs(output_folder, exist_ok=True)

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Partial function to fix certain parameters except 'row'
        process_func = partial(
            process_row,
            output_folder=output_folder,
            name_prefix=name_prefix,
            padding=padding,
            mode=mode
        )

        # Submit all tasks to the executor
        future_to_row = {executor.submit(process_func, row): row for row in rows}

        # Use tqdm to display a progress bar
        for future in tqdm(as_completed(future_to_row), total=len(future_to_row), desc="Processing Images"):
            try:
                output_filename, status, error = future.result()
                results.append((output_filename, status, error))
            except Exception as e:
                # This should not happen as exceptions are handled in process_row
                logging.error(f"Unhandled exception: {e}")
                continue

    return results


def setup_logging():
    """
    Setup logging configuration.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )


def select_mode():
    """
    Prompt the user to select the output mode: 'lossless' or 'compressed'.
    Returns the selected mode as a string.
    """
    while True:
        print("\nSelect Output Mode:")
        print("1. Lossless WebP")
        print("2. Compressed WebP (GPU-friendly)")
        choice = input("Enter the number corresponding to your choice (1 or 2): ").strip()

        if choice == '1':
            return "lossless"
        elif choice == '2':
            return "compressed"
        else:
            print("Invalid input. Please enter 1 or 2.\n")


def main():
    """
    Main function to orchestrate the image merging process.
    """
    # Setup logging
    setup_logging()

    # Interactive mode selection
    mode = select_mode()
    logging.info(f"Selected mode: {mode.capitalize()}")

    # Determine output folder based on mode
    if mode == "lossless":
        output_folder = OUTPUT_FOLDER_LOSSLESS
    else:  # compressed
        output_folder = OUTPUT_FOLDER_COMPRESSED

    # Step 1: Find CSV files
    csv_files = find_csv_files(pattern=MAIN_FLOAT_CSV_PATTERN)
    if not csv_files:
        logging.error("No CSV files found matching the pattern '%s'.", MAIN_FLOAT_CSV_PATTERN)
        return

    logging.info("Found CSV files: %s", ', '.join(csv_files))

    # Step 2: Process each CSV file
    for csv_file in csv_files:
        logging.info("\nProcessing CSV file: %s", csv_file)
        rows = read_csv(csv_file)
        logging.info("Loaded %d rows from '%s'.", len(rows), csv_file)

        if not rows:
            logging.warning("No valid rows to process in '%s'. Skipping...", csv_file)
            continue

        # Step 3: Determine the number of workers
        max_workers = os.cpu_count()
        logging.info("Using %d parallel workers.", max_workers)

        # Step 4: Process all rows with parallel execution
        results = process_all_rows(
            rows=rows,
            output_folder=output_folder,
            name_prefix=NAME_PREFIX,
            padding=PADDING,
            mode=mode,
            max_workers=max_workers
        )

        # Step 5: Summary of results
        success_count = sum(1 for _, status, _ in results if status == "Success")
        failure_count = len(results) - success_count
        logging.info("Completed processing '%s'. %d succeeded, %d failed.", csv_file, success_count, failure_count)

    logging.info("\nAll processing completed. Merged images saved to '%s'.", output_folder)


if __name__ == "__main__":
    main()
