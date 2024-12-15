import os
import csv
from PIL import Image
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm

# Constants
MAIN_FLOAT_CSV_PATTERN = "*.csv"  # Default pattern to find CSV files
OUTPUT_FOLDER = "merged_images"  # Folder to save the merged images


def find_csv_files(pattern=MAIN_FLOAT_CSV_PATTERN):
    """
    Find all CSV files matching the given pattern in the current directory.
    """
    return glob.glob(pattern)


def read_csv(file_path):
    """
    Read the CSV file and return a list of rows, excluding the header.
    """
    rows = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None)  # Skip header
        for row in reader:
            if len(row) >= 3:
                rows.append(row)
            else:
                print(f"Warning: Skipping malformed row: {row}")
    return rows


def alpha_composite_images(main_image_path, float_image_path, output_path):
    """
    Composite the float image onto the main image while respecting alpha.
    """
    try:
        main_image = Image.open(main_image_path).convert("RGBA")
        float_image = Image.open(float_image_path).convert("RGBA")

        # Perform alpha compositing
        merged_image = Image.alpha_composite(main_image, float_image)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save the result
        merged_image.save(output_path)
        return (output_path, "Success", None)
    except Exception as e:
        return (output_path, "Failed", str(e))


def process_row(row, output_folder):
    """
    Process a single CSV row: composite images and save the output.
    """
    absolute_index, main_image_path, float_image_path = row
    # Derive output filename (e.g., Absolute Index.png)
    # You can customize the naming convention as needed
    output_filename = f"{absolute_index}.png"
    output_path = os.path.join(output_folder, output_filename)

    return alpha_composite_images(main_image_path, float_image_path, output_path)


def process_all_rows(rows, output_folder, max_workers=None):
    """
    Process all CSV rows using parallel processing.
    """
    os.makedirs(output_folder, exist_ok=True)

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Partial function to fix the output_folder parameter
        process_func = partial(process_row, output_folder=output_folder)

        # Submit all tasks to the executor
        future_to_row = {executor.submit(process_func, row): row for row in rows}

        # Use tqdm to display a progress bar
        for future in tqdm(as_completed(future_to_row), total=len(future_to_row), desc="Processing Images"):
            output_path, status, error = future.result()
            if status == "Failed":
                print(f"Error processing {output_path}: {error}")
            results.append((output_path, status, error))

    return results


def main():
    # Step 1: Find CSV files
    csv_files = find_csv_files()
    if not csv_files:
        print("No CSV files found.")
        return

    print(f"Found CSV files: {csv_files}")

    # Step 2: Read the first CSV file (you can modify to handle multiple files if needed)
    csv_file = csv_files[0]
    print(f"Using CSV file: {csv_file}")

    # Step 3: Parse the CSV
    rows = read_csv(csv_file)
    print(f"Loaded {len(rows)} rows from the CSV.")

    if not rows:
        print("No valid rows to process.")
        return

    # Step 4: Determine the number of workers
    # If max_workers is None, it defaults to the number of CPUs on the machine
    max_workers = os.cpu_count()
    print(f"Using {max_workers} parallel workers.")

    # Step 5: Process all rows with parallel execution
    results = process_all_rows(rows, OUTPUT_FOLDER, max_workers=max_workers)

    # Step 6: Summary of results
    success_count = sum(1 for _, status, _ in results if status == "Success")
    failure_count = len(results) - success_count
    print(f"Completed processing. {success_count} succeeded, {failure_count} failed.")
    print(f"Merged images saved to {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
