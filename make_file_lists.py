#!/usr/bin/env python
"""
make_file_lists.py

This combined script:
  1. Scans two image directories (MAIN_FOLDER_PATH and FLOAT_FOLDER_PATH, as set in settings.py)
     recursively to build intermediate CSV files containing folder details.
     It recognizes PNG, WEBP, and JXL files.
  2. Reads these intermediate CSVs, verifies image counts, sorts and interleaves image paths,
     and then writes the final CSV lists.

**Requirement:** This script requires the 'opencv-python' library.
Install it using pip:
   pip install opencv-python

**WARNING:** Standard 'opencv-python' wheels often DO NOT support reading JXL files.
This script MAY FAIL to process JXL files unless you have a custom OpenCV build
with JXL support enabled. It will likely handle PNG and WEBP correctly.
"""

import os
import re
import csv
import sys
import shutil
from itertools import zip_longest
import cv2  # Use OpenCV
import numpy as np # OpenCV uses numpy arrays
from collections import defaultdict

import settings  # settings.py should define IMAGES_DIR, MAIN_FOLDER_PATH, FLOAT_FOLDER_PATH

# Define supported image file extensions
SUPPORTED_EXTENSIONS = ('.png', '.webp', '.jxl')

# ------------------------------
# Utility functions (shared)
# ------------------------------

def get_subdirectories(path):
    """Retrieve all subdirectories within a given path."""
    subdirs = []
    for root, dirs, _ in os.walk(path):
        for d in dirs:
            subdirs.append(os.path.join(root, d))
    return subdirs


def contains_image_files(path):
    """Check if a directory contains any supported image files (PNG, WEBP, JXL)."""
    try:
        return any(file.lower().endswith(SUPPORTED_EXTENSIONS) for file in os.listdir(path))
    except FileNotFoundError:
        return False


def count_image_files(path):
    """Count the number of supported image files (PNG, WEBP, JXL) in a directory."""
    try:
        return len([file for file in os.listdir(path) if file.lower().endswith(SUPPORTED_EXTENSIONS)])
    except FileNotFoundError:
        return 0

# Renamed function as Pillow's concept of 'mode' doesn't directly apply
def image_has_alpha_channel(img_np):
    """Determine if an image loaded by OpenCV (as numpy array) has an alpha channel."""
    if img_np is None:
        return False
    # Check if image has 3 dimensions (height, width, channels) and the channel count is 4
    return len(img_np.shape) == 3 and img_np.shape[2] == 4


def natural_sort_key(s):
    """
    Generates a key for natural sorting of strings containing numbers.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def interleave_lists(lists):
    """
    Interleaves multiple lists into a single list by alternating elements.
    """
    result = []
    for items in zip_longest(*lists, fillvalue=None):
        group = [item for item in items if item is not None]
        result.append(group)
    return result


# ------------------------------
# Intermediate CSV Creation
# ------------------------------

def create_folder_csv_files(folder_counts, processed_dir, script_dir):
    """
    Create CSV files categorizing folders into main and float groups.
    Relies on the details gathered during the initial scan (write_folder_list).
    """
    groups = defaultdict(list)
    float_group = defaultdict(list)

    # folder_counts contains: (folder_path, first_image_filename, width, height, has_alpha, file_count)
    for folder_info in folder_counts:
        folder, first_image_filename, width, height, has_alpha, file_count = folder_info
        folder_relative = os.path.relpath(folder, script_dir)
        # Group based on folder name prefix
        if os.path.basename(folder).startswith('255_'):
            float_group[file_count].append((folder_relative, first_image_filename, width, height, has_alpha, file_count))
        else:
            groups[file_count].append((folder_relative, first_image_filename, width, height, has_alpha, file_count))

    def write_csv(group, file_name_format):
        for file_count, sub_group in group.items():
            # Sort the group based on numeric prefix (if any) and then folder name.
            sub_group.sort(key=lambda x: (
                int(os.path.basename(x[0]).partition('_')[0]) if os.path.basename(x[0]).partition('_')[
                    0].isdigit() else float('inf'),
                os.path.basename(x[0])
            ))
            csv_filename = file_name_format.format(file_count)
            csv_path = os.path.join(processed_dir, csv_filename)
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Define consistent column headers
                # Note: Alpha Match Info column is removed as reliable re-checking without Pillow is harder
                # writer.writerow(['Index', 'Folder Relative Path', 'Dimensions', 'File Count', 'Has Alpha', 'First Image Extension']) # Optional header row
                for index, (folder_rel, first_image_filename, width, height, has_alpha, file_count) in enumerate(sub_group, 1):
                    file_extension = os.path.splitext(first_image_filename)[1].lower() if first_image_filename else 'N/A'

                    writer.writerow([
                        index,
                        folder_rel,
                        f"{width}x{height} pixels", # Dimensions from initial scan
                        file_count,
                        'Yes' if has_alpha else 'No', # Alpha status from initial scan
                        # Removed alpha match column for simplicity with OpenCV
                        file_extension # Extension of the first found file
                    ])
            print(f"CSV file created: {csv_path}")

    write_csv(groups, 'main_folder_{}.csv')
    write_csv(float_group, 'float_folder_{}.csv')


def write_folder_list():
    """
    Scans the two image directories defined in settings (MAIN_FOLDER_PATH and FLOAT_FOLDER_PATH)
    recursively for supported image files (PNG, WEBP, JXL), collects folder details using OpenCV,
    and writes intermediate CSV files.
    **WARNING:** May fail on JXL files if OpenCV build doesn't support them.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(script_dir, "folders_processed")

    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    os.makedirs(processed_dir)

    main_folder_path = os.path.join(script_dir, settings.MAIN_FOLDER_PATH)
    float_folder_path = os.path.join(script_dir, settings.FLOAT_FOLDER_PATH)

    for base_path in [main_folder_path, float_folder_path]:
        if not os.path.exists(base_path) or not os.path.isdir(base_path):
            print(f"Warning: Directory '{base_path}' is missing or not a directory. Skipping scan for this path.")

    folder_dict = {}
    folder_key = 1
    for base_path in [main_folder_path, float_folder_path]:
        if os.path.exists(base_path) and os.path.isdir(base_path):
            for directory_to_scan in [base_path] + get_subdirectories(base_path):
                if contains_image_files(directory_to_scan):
                    if directory_to_scan not in folder_dict.values():
                        folder_dict[folder_key] = directory_to_scan
                        folder_key += 1

    if not folder_dict:
        print(f"No directories containing supported image files ({', '.join(SUPPORTED_EXTENSIONS)}) were found in the specified paths or their subdirectories.")
        sys.exit(1)

    total_images = 0
    folder_counts = [] # Stores tuples: (folder_path, first_image_filename, width, height, has_alpha, file_count)

    print(f"Scanning {len(folder_dict)} folder(s) for details using OpenCV...")
    for folder in folder_dict.values():
        image_files = [f for f in os.listdir(folder) if f.lower().endswith(SUPPORTED_EXTENSIONS)]
        image_files.sort(key=natural_sort_key)

        if image_files:
            first_image_filename = image_files[0]
            first_image_path = os.path.join(folder, first_image_filename)
            width, height, has_alpha = 0, 0, False # Defaults
            try:
                # Use cv2.imread to get info. IMREAD_UNCHANGED attempts to load alpha.
                # ** This may return None for unsupported formats like JXL in standard builds **
                img_np = cv2.imread(first_image_path, cv2.IMREAD_UNCHANGED)

                if img_np is not None:
                    # Shape gives (height, width, channels) or (height, width)
                    height = img_np.shape[0]
                    width = img_np.shape[1]
                    has_alpha = image_has_alpha_channel(img_np)
                else:
                    # OpenCV failed to read the image
                    print(f"Error: OpenCV (cv2.imread) failed to read first image: '{first_image_path}'. Check format support (especially JXL). Using defaults.")
                    first_image_filename = None # Indicate error

            except Exception as e:
                print(f"Exception processing first image '{first_image_path}' with OpenCV: {e}. Using defaults.")
                first_image_filename = None # Indicate error

            file_count = count_image_files(folder)
            total_images += file_count
            folder_counts.append((folder, first_image_filename, width, height, has_alpha, file_count))
        else:
             print(f"Warning: Folder {folder} was selected but no supported images found directly within it during detail scan.")
             folder_counts.append((folder, None, 0, 0, False, 0))

    if not folder_counts:
         print("No image details could be gathered from the found folders.")
         sys.exit(1)

    # Save folder count summary to a text file
    folder_count_filename = f'folder_count_{total_images}_files.txt'
    folder_count_path = os.path.join(processed_dir, folder_count_filename)
    print(f"Writing folder summary to {folder_count_path}...")
    with open(folder_count_path, 'w', encoding='utf-8') as f:
        f.write(f"Total Folders Found: {len(folder_counts)}\n")
        f.write(f"Total Supported Images Found: {total_images}\n---\n")
        for index, (folder, first_image_fname, width, height, has_alpha, count) in enumerate(folder_counts, 1):
            folder_rel = os.path.relpath(folder, script_dir)
            first_image_name_noext = os.path.splitext(first_image_fname)[0] if first_image_fname else "ReadError/NoImage"
            alpha_str = 'Yes' if has_alpha else 'No'
            f.write(f"{index}. Folder: {folder_rel}\n")
            f.write(f"   First Image: {first_image_fname or 'N/A'} ({first_image_name_noext})\n")
            f.write(f"   Dimensions: {width}x{height}, Has Alpha: {alpha_str}, File Count: {count}\n")
    print(f"Folder count file created.")

    # Create intermediate CSV files for main and float folders
    create_folder_csv_files(folder_counts, processed_dir, script_dir)

# ------------------------------
# Final CSV Creation (Image Lists) - Largely unchanged logic
# ------------------------------

# Functions find_default_csvs, check_unequal_img_counts, sort_image_files,
# write_sorted_images, and parse_folder_locations remain largely the same
# as they primarily deal with file paths and CSV manipulation, not image loading.
# However, minor adjustments were made in the provided full script below for robustness.

def parse_folder_locations(csv_path):
    """
    Parses the given CSV file to create a dictionary mapping folder numbers to folder paths.
    Also checks that all folders have the same image count using check_unequal_img_counts.
    """
    folder_dict = {}
    check_unequal_img_counts(csv_path) # Exits if unequal

    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                # Skip header if present (simple check for non-numeric first column)
                # header = next(reader, None)
                # if header and not header[0].isdigit():
                #      pass # Header skipped
                # else:
                #      # Process header as first row if it looked like data
                #      process_row(header, folder_dict) # Hypothetical process_row func

                for i, row in enumerate(reader):
                    if row and len(row) >= 4: # Need at least index, path, dims, count
                         try:
                             number = int(row[0])
                             folder = row[1]
                             if folder:
                                 folder_dict[number] = folder
                         except ValueError:
                              print(f"Warning: Skipping invalid row in {csv_path}: Could not parse index in row {i+1}: {row}")
                         except IndexError:
                              print(f"Warning: Skipping incomplete row in {csv_path} (row {i+1}): {row}")

        except FileNotFoundError:
             print(f"Error: CSV file not found during parsing: {csv_path}")
             return {}
        except Exception as e:
             print(f"Error reading CSV file {csv_path}: {e}")
             return {}

    return folder_dict


def find_default_csvs(processed_dir):
    """
    Find exactly two CSV files: float_folder_XXXX.csv and main_folder_XXXX.csv
    with the same XXXX image count. Returns the list of paths if found, else None.
    Uses the highest common count if multiple pairs exist.
    """
    float_pattern = re.compile(r'^float_folder_(\d+)\.csv$')
    main_pattern = re.compile(r'^main_folder_(\d+)\.csv$')

    float_files = {} # Stores count -> path
    main_files = {}  # Stores count -> path

    if not os.path.isdir(processed_dir):
        print(f"Error: Processed directory '{processed_dir}' not found.")
        return None

    try:
        for file in os.listdir(processed_dir):
            float_match = float_pattern.match(file)
            main_match = main_pattern.match(file)
            if float_match:
                try:
                    count = int(float_match.group(1))
                    float_files[count] = os.path.join(processed_dir, file)
                except ValueError: pass # Ignore files with non-integer counts
            if main_match:
                 try:
                    count = int(main_match.group(1))
                    main_files[count] = os.path.join(processed_dir, file)
                 except ValueError: pass # Ignore files with non-integer counts

    except FileNotFoundError:
         print(f"Error: Processed directory '{processed_dir}' not found during listing.")
         return None
    except Exception as e:
         print(f"Error listing files in '{processed_dir}': {e}")
         return None

    common_counts = sorted(list(set(float_files.keys()) & set(main_files.keys())), reverse=True)

    if common_counts:
        highest_common_count = common_counts[0]
        print(f"Found matching main and float CSVs for image count: {highest_common_count}")
        return [float_files[highest_common_count], main_files[highest_common_count]]
    else:
        print("Error: No matching pairs of main_folder_XXXX.csv and float_folder_XXXX.csv found based on image count (XXXX).")
        print("Found float files (count: path):", float_files)
        print("Found main files (count: path):", main_files)
        return None


def check_unequal_img_counts(csv_path):
    """
    Checks if all folders listed in the CSV have the same number of images.
    Exits the script with an error message if unequal counts are found.
    """
    img_counts = set()
    first_count = None
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if row and len(row) >= 4: # Need index, path, dims, count columns
                    try:
                        count = int(row[3]) # Column 4 (index 3) is the file count
                        img_counts.add(count)
                        if first_count is None:
                            first_count = count
                        if len(img_counts) > 1:
                            print(f"\n--- ERROR: Unequal Image Counts ---")
                            print(f"File: {os.path.basename(csv_path)}")
                            print(f"Expected all folders to have {first_count} images.")
                            print(f"Found folders with counts: {sorted(list(img_counts))}")
                            print(f"Please ensure all folders listed in this CSV have the same number of supported images.")
                            print(f"CSV Path: {csv_path}")
                            print(f"-------------\n")
                            sys.exit(1)
                    except (ValueError, IndexError):
                         # Ignore rows where count isn't valid - parsing function handles folder dict creation
                         pass # Already warned during parsing if needed

    except FileNotFoundError:
         print(f"Error: CSV file not found during count check: {csv_path}")
         sys.exit(1)
    except Exception as e:
         print(f"Error reading CSV file {csv_path} during count check: {e}")
         sys.exit(1)

    if not img_counts and first_count is None: # Check if file was empty or unparsable
        print(f"Warning: No valid image count data found in {csv_path}. Cannot verify count consistency.")


def sort_image_files(folder_dict, script_dir):
    """
    Gets full paths for image files in each folder and sorts them using natural sort.
    Requires script_dir to construct absolute paths from relative paths in folder_dict.
    Stores relative paths in the final list.
    """
    sorted_image_files = []
    for number in sorted(folder_dict.keys()):
        folder_relative = folder_dict[number]
        folder_absolute = os.path.normpath(os.path.join(script_dir, folder_relative)) # Normalize path
        try:
            if os.path.isdir(folder_absolute):
                image_filenames = [f for f in os.listdir(folder_absolute) if f.lower().endswith(SUPPORTED_EXTENSIONS)]
                image_filenames.sort(key=natural_sort_key)
                # Store paths relative to script_dir in the final output list
                image_relative_paths = [os.path.join(folder_relative, f) for f in image_filenames]
                if not image_relative_paths:
                     print(f"Warning: No supported images found in directory {folder_absolute} despite being listed.")
                sorted_image_files.append(image_relative_paths)
            else:
                 print(f"Warning: Folder path '{folder_relative}' (absolute: {folder_absolute}) not found or not a directory. Skipping.")
        except FileNotFoundError:
             print(f"Error: Directory not found when trying to list images: {folder_absolute}. Skipping.")
        except Exception as e:
             print(f"Error processing folder {folder_absolute}: {e}. Skipping.")

    return sorted_image_files


def write_sorted_images(grouped_image_files, output_folder, csv_path):
    """
    Writes the interleaved and sorted image file paths (relative to script dir)
    to a new CSV in the output folder.
    """
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    try:
         # Extract count and type more robustly
         parts = base_name.split('_')
         image_count = parts[-1] if parts[-1].isdigit() else 'unknown_count'
         folder_type = parts[0] if len(parts) > 1 else 'unknown_type'
    except Exception:
         image_count = 'unknown_count'
         folder_type = 'unknown_type'

    output_csv_name = f'{folder_type}_{image_count}_images_list.csv'
    output_csv_path = os.path.join(output_folder, output_csv_name)
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            max_cols = 0
            if grouped_image_files:
                max_cols = max(len(group) for group in grouped_image_files) if grouped_image_files else 0

            header = ['Frame'] + [f'Image_Path_{i+1}' for i in range(max_cols)]
            csv_writer.writerow(header)

            for index, group in enumerate(grouped_image_files, start=0):
                row_data = [index] + list(group)
                csv_writer.writerow(row_data)
        print(f"Interleaved image list CSV written to: {output_csv_path}")
    except IOError as e:
         print(f"Error writing final CSV file {output_csv_path}: {e}")
    except Exception as e:
         print(f"An unexpected error occurred while writing {output_csv_path}: {e}")


# ------------------------------
# Main Process Function (Interface preserved)
# ------------------------------

def process_files():
    """
    Main function using OpenCV to:
      1. Generate intermediate CSV files (folder details) by scanning the image directories
         for supported image types (PNG, WEBP, JXL). **May fail on JXL.**
      2. Process these CSV files to generate interleaved image lists.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(script_dir, 'folders_processed')
    generated_dir = os.path.join(script_dir, 'generated_img_lists')

    print("--- Starting File List Generation (Using OpenCV) ---")
    print("** WARNING: Standard OpenCV often lacks JXL support. Processing may fail for .jxl files. **")

    # Step 1: Generate intermediate CSV files (folder list details)
    print("\n--- Step 1: Generating folder lists and intermediate CSV files ---")
    try:
        write_folder_list()
    except Exception as e:
        print(f"--- FATAL ERROR during Step 1 (Folder List Generation) ---")
        print(f"{e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        print("----------------------------------------------------------")
        sys.exit(1)
    print("--- Step 1 Completed ---")

    # Step 2: Find default CSV files generated above
    print("\n--- Step 2: Finding matching intermediate CSV files ---")
    default_csvs = find_default_csvs(processed_dir)
    if default_csvs:
        print("Matching CSV files found:")
        for csv_file in default_csvs:
            print(f" - {os.path.basename(csv_file)}")
        csv_paths = default_csvs
    else:
        print("--- ERROR: Could not find a matching pair of main/float CSVs. ---")
        print("Please check the contents of the 'folders_processed' directory.")
        print("Ensure both main_folder_XXXX.csv and float_folder_XXXX.csv exist with the same image count (XXXX).")
        print("----------------------------------------------------------------")
        sys.exit(1)
    print("--- Step 2 Completed ---")

    # Step 3: Generate final image list CSVs
    print("\n--- Step 3: Generating final interleaved image list CSVs ---")
    if os.path.exists(generated_dir):
        print(f"Removing existing output directory: {generated_dir}")
        try:
            shutil.rmtree(generated_dir)
        except OSError as e:
            print(f"Error removing directory {generated_dir}: {e}. Please remove manually.")
            sys.exit(1)
    try:
        os.makedirs(generated_dir)
        print(f"Created output directory: {generated_dir}")
    except OSError as e:
         print(f"Error creating directory {generated_dir}: {e}.")
         sys.exit(1)

    processing_errors = False
    for csv_path in csv_paths:
        print(f"\nProcessing intermediate file: {os.path.basename(csv_path)}")
        try:
            folder_dict = parse_folder_locations(csv_path) # Includes count check

            if not folder_dict:
                print(f"Warning: No valid folder locations parsed from {os.path.basename(csv_path)}. Skipping.")
                processing_errors = True
                continue

            sorted_image_files = sort_image_files(folder_dict, script_dir)

            if not sorted_image_files:
                 print(f"Warning: No image files could be sorted from folders listed in {os.path.basename(csv_path)}. Skipping.")
                 processing_errors = True
                 continue

            grouped_image_files = interleave_lists(sorted_image_files)
            write_sorted_images(grouped_image_files, generated_dir, csv_path)

        except SystemExit as e: # Catch exits from check_unequal_img_counts
             print(f"Exiting due to error processing {os.path.basename(csv_path)} (likely unequal counts).")
             processing_errors = True
             break # Stop processing further files if one fails count check
        except Exception as e:
            print(f"--- ERROR processing {os.path.basename(csv_path)} ---")
            print(f"{e}")
            import traceback
            traceback.print_exc()
            print(f"----------------------------------------------------")
            processing_errors = True

    print("\n--- Step 3 Completed ---")

    if processing_errors:
         print("\n--- Script finished with one or more errors or warnings. Please review the output above. ---")
         # Keep exit code 1 to indicate non-successful completion
    else:
         print("\n--- Script finished successfully! ---")
         sys.exit(0) # Explicitly exit with 0 on success

    sys.exit(1) # Exit with 1 if errors occurred


if __name__ == "__main__":
    process_files()