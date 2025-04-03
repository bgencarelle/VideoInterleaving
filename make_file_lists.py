#!/usr/bin/env python
"""
make_file_lists.py

This combined script:
  1. Scans two image directories (MAIN_FOLDER_PATH and FLOAT_FOLDER_PATH, as set in settings.py)
     recursively to build intermediate CSV files containing folder details.
  2. Reads these intermediate CSVs, verifies image counts, sorts and interleaves image paths,
     and then writes the final CSV lists.

The process_files() function is maintained as the main entry point.
"""

import os
import re
import csv
import sys
import shutil
from itertools import zip_longest
import cv2  # Replace PIL with OpenCV
from collections import defaultdict

import settings  # settings.py should define IMAGES_DIR, MAIN_FOLDER_PATH, FLOAT_FOLDER_PATH


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
    """Check if a directory contains any PNG, WEBP, or JXL files."""
    try:
        return any(file.lower().endswith(('.png', '.webp', '.jxl')) for file in os.listdir(path))
    except FileNotFoundError:
        return False


def count_image_files(path):
    """Count the number of PNG, WEBP, or JXL files in a directory."""
    try:
        return len([file for file in os.listdir(path) if file.lower().endswith(('.png', '.webp', '.jxl'))])
    except FileNotFoundError:
        return 0


def has_alpha_channel(image_path):
    """Determine if an image (given by path) has an alpha channel using OpenCV."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return False  # Image failed to load
        return img.shape[2] == 4  # Check for 4 channels (RGBA)
    except cv2.error as e:
        print(f"OpenCV error checking alpha: {e}")
        return False
    except Exception as e:
        print(f"Error checking alpha channel for {image_path}: {e}")
        return False


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
    The grouping logic remains unchanged: if the folder's basename starts with '255_', it goes into float.
    """
    groups = defaultdict(list)
    float_group = defaultdict(list)

    for folder_info in folder_counts:
        folder, first_png, width, height, has_alpha, file_count = folder_info
        folder_relative = os.path.relpath(folder, script_dir)
        if os.path.basename(folder).startswith('255_'):
            float_group[file_count].append((folder_relative, first_png, width, height, has_alpha, file_count))
        else:
            groups[file_count].append((folder_relative, first_png, width, height, has_alpha, file_count))

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
                for index, (folder_rel, first_png, width, height, has_alpha, file_count) in enumerate(sub_group, 1):
                    if first_png:
                        try:
                            image_path = os.path.join(script_dir, folder_rel, first_png)
                            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                            if img is None:
                                print(f"Error: Could not read image {first_png} in folder {folder_rel} with OpenCV.")
                                alpha_match = 'Error'
                                file_extension = 'Unknown'
                            else:
                                file_extension = os.path.splitext(first_png)[1]
                                if has_alpha:
                                    # OpenCV doesn't directly give "size" of alpha channel; compare channels if possible
                                    alpha_channels = cv2.split(img)[-1]  # Get alpha channel
                                    alpha_match = 'Match' if alpha_channels.shape[:2] == img.shape[:2] else 'NoMatch'
                                else:
                                    alpha_match = 'NoAlpha'
                        except Exception as e:
                            print(f"Error processing image {first_png} in folder {folder_rel}: {e}")
                            alpha_match = 'Error'
                            file_extension = 'Unknown'
                    else:
                        alpha_match = 'NoImage'
                        file_extension = 'N/A'

                    writer.writerow([
                        index,
                        folder_rel,
                        f"{width}x{height} pixels",
                        file_count,
                        'Yes' if has_alpha else 'No',
                        alpha_match,
                        file_extension
                    ])
            print(f"CSV file created: {csv_path}")

    write_csv(groups, 'main_folder_{}.csv')
    write_csv(float_group, 'float_folder_{}.csv')


def write_folder_list():
    """
    Scans the two image directories defined in settings (MAIN_FOLDER_PATH and FLOAT_FOLDER_PATH)
    recursively for PNG, WEBP, and JXL files, collects folder details, and writes intermediate CSV files.
    """
    # Determine the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(script_dir, "folders_processed")

    # Remove old processed directory if it exists
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    os.makedirs(processed_dir)

    # Build full paths from settings
    main_folder_path = os.path.join(script_dir, settings.MAIN_FOLDER_PATH)
    float_folder_path = os.path.join(script_dir, settings.FLOAT_FOLDER_PATH)

    # Ensure both directories exist
    for base_path in [main_folder_path, float_folder_path]:
        if not os.path.exists(base_path) or not os.path.isdir(base_path):
            print(f"Error: Directory '{base_path}' is missing. Please create it and populate with images.")
            # Continue scanning the other folder instead of exiting

    # Scan both directories recursively
    folder_dict = {}
    folder_key = 1
    for base_path in [main_folder_path, float_folder_path]:
        if os.path.exists(base_path) and os.path.isdir(base_path):
            for subdirectory in [base_path] + get_subdirectories(base_path):
                if contains_image_files(subdirectory):
                    folder_dict[folder_key] = subdirectory
                    folder_key += 1

    if not folder_dict:
        print("No valid image files were found in the specified directories or their subdirectories.")
        return

    # Collect folder details (folder, first image, dimensions, alpha flag, image count)
    total_images = 0
    folder_counts = []

    for folder in folder_dict.values():
        image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.webp', '.jxl'))]
        if image_files:
            first_image = image_files[0]
            try:
                img = cv2.imread(os.path.join(folder, first_image), cv2.IMREAD_UNCHANGED)
                if img is None:
                    print(f"Error: Could not read image {first_image} in folder {folder} with OpenCV.")
                    width, height, has_alpha = 0, 0, False
                else:
                    height, width = img.shape[:2]  # OpenCV returns height first
                    has_alpha = has_alpha_channel(os.path.join(folder, first_image))
            except Exception as e:
                print(f"Error processing image {first_image} in folder {folder}: {e}")
                first_image = None
                width, height, has_alpha = 0, 0, False
            file_count = count_image_files(folder)
            total_images += file_count
            folder_counts.append((folder, first_image, width, height, has_alpha, file_count))

    # Save folder count to a text file
    folder_count_filename = f'folder_count_{total_images}.txt'
    folder_count_path = os.path.join(processed_dir, folder_count_filename)
    with open(folder_count_path, 'w', encoding='utf-8') as f:
        for index, (folder, first_image, width, height, has_alpha, count) in enumerate(folder_counts, 1):
            folder_rel = os.path.relpath(folder, script_dir)
            first_image_name = os.path.splitext(os.path.basename(first_image))[0] if first_image else "NoImage"
            f.write(f"{index}, {folder_rel}, {first_image_name}, {width}x{height} pixels, {count}\n")
    print(f"Folder count file created: {folder_count_path}")

    # Create intermediate CSV files for main and float folders
    create_folder_csv_files(folder_counts, processed_dir, script_dir)


# ------------------------------
# Final CSV Creation (Image Lists)
# ------------------------------

def parse_folder_locations(csv_path):
    """
    Parses the given CSV file to create a dictionary mapping folder numbers to folder paths.
    Also checks that all folders have the same image count.
    """
    folder_dict = {}
    check_unequal_img_counts(csv_path)

    if os.path.exists(csv_path):
        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    break
                number, folder = int(row[0]), row[1]
                if folder:
                    folder_dict[number] = folder
    return folder_dict


def find_default_csvs(processed_dir):
    """
    Find exactly two CSV files: float_folder_XXXX.csv and main_folder_XXXX.csv
    with the same XXXX. Returns the list of paths if found, else None.
    """
    float_pattern = re.compile(r'^float_folder_(\d{1,30})\.csv$')
    main_pattern = re.compile(r'^main_folder_(\d{1,30})\.csv$')

    float_files = {}
    main_files = {}

    for file in os.listdir(processed_dir):
        float_match = float_pattern.match(file)
        main_match = main_pattern.match(file)
        if float_match:
            float_files[float_match.group(1)] = os.path.join(processed_dir, file)
        if main_match:
            main_files[main_match.group(1)] = os.path.join(processed_dir, file)

    # Find matching XXXX
    for xxxx in float_files:
        if xxxx in main_files:
            return [float_files[xxxx], main_files[xxxx]]
    return None


def check_unequal_img_counts(csv_path):
    """
    Checks if all folders in the CSV have the same number of images.
    Exits the script if unequal counts are found.
    """
    if os.path.exists(csv_path):
        img_counts = set()
        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    break
                img_counts.add(int(row[3]))  # Column 4 (index 3) is the file count
                if len(img_counts) > 1:
                    print("error: unequal file count found, please fix before proceeding")
                    sys.exit(0)


def sort_image_files(folder_dict):
    """
    Sorts image files in each folder using natural sort.
    """
    sorted_image_files = []
    for number in sorted(folder_dict.keys()):
        folder = folder_dict[number]
        image_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.webp', '.jxl'))]
        image_files.sort(key=natural_sort_key)
        sorted_image_files.append(image_files)
    return sorted_image_files


def write_sorted_images(grouped_image_files, output_folder, csv_path):
    """
    Writes the interleaved and sorted image file paths to a new CSV in the output folder.
    """
    output_csv_name = f'{os.path.splitext(os.path.basename(csv_path))[0]}_list.csv'
    output_csv_path = os.path.join(output_folder, output_csv_name)
    with open(output_csv_path, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        for index, group in enumerate(grouped_image_files, start=0):
            csv_writer.writerow([index] + group)
    print(f"CSV written to {output_csv_path}")


# ------------------------------
# Main Process Function (Interface preserved)
# ------------------------------

def process_files():
    """
    Main function to:
      1. Generate intermediate CSV files (folder details) by scanning the image directories.
      2. Process these CSV files to generate interleaved image lists.

    This function is the main entry point and its interface is preserved.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(script_dir, 'folders_processed')
    generated_dir = os.path.join(script_dir, 'generated_img_lists')

    # Step 1: Generate intermediate CSV files (folder list details)
    print("Generating folders and intermediate CSV files...")
    write_folder_list()

    # Step 2: Find default CSV files generated above
    default_csvs = find_default_csvs(processed_dir)
    if default_csvs:
        print("Default CSV files found:")
        for csv_file in default_csvs:
            print(f" - {csv_file}")
        csv_paths = default_csvs
    else:
        print("No default CSV files were generated. Please check the folder list generation.")
        sys.exit(1)

    # Delete the old generated_img_lists folder if it exists
    if os.path.exists(generated_dir):
        shutil.rmtree(generated_dir)
    os.makedirs(generated_dir)

    # For each default CSV file, process it to generate an interleaved image list CSV.
    for csv_path in csv_paths:
        check_unequal_img_counts(csv_path)
        folder_dict = parse_folder_locations(csv_path)
        if not folder_dict:
            print(f"No accessible files found in the processed CSV: {csv_path}")
            continue  # Skip to the next CSV file

        sorted_image_files = sort_image_files(folder_dict)
        grouped_image_files = interleave_lists(sorted_image_files)
        write_sorted_images(grouped_image_files, generated_dir, csv_path)


if __name__ == "__main__":
    process_files()
