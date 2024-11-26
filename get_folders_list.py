# get_folders_list.py
# This script scans directories for PNG and WEBP files, organizes them, and generates CSV lists.

import os
import re
import csv
from PIL import Image
from collections import defaultdict

def get_subdirectories(path):
    """Retrieve all subdirectories within a given path."""
    return [os.path.join(root, d) for root, dirs, _ in os.walk(path) for d in dirs]

def contains_image_files(path):
    """Check if a directory contains any PNG or WEBP files."""
    try:
        return any(file.lower().endswith(('.png', '.webp')) for file in os.listdir(path))
    except FileNotFoundError:
        return False

def count_image_files(path):
    """Count the number of PNG and WEBP files in a directory."""
    try:
        return len([file for file in os.listdir(path) if file.lower().endswith(('.png', '.webp'))])
    except FileNotFoundError:
        return 0

def parse_line(line):
    """Parse a line from folder_locations.txt."""
    match = re.match(r'(\d+)[.,] (.+)', line)
    return (int(match.group(1)), match.group(2)) if match else (None, None)

def has_alpha_channel(image):
    """Determine if an image has an alpha channel."""
    return image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info)

def create_folder_csv_files(folder_counts, processed_dir, script_dir):
    """Create CSV files categorizing folders into main and float groups."""
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
            # Sort the group based on numeric prefix or folder name
            sub_group.sort(key=lambda x: (
                int(os.path.basename(x[0]).partition('_')[0]) if os.path.basename(x[0]).partition('_')[0].isdigit() else float('inf'),
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
                            with Image.open(image_path) as first_png_image:
                                file_extension = os.path.splitext(first_png)[1]
                                if has_alpha:
                                    alpha_match = 'Match' if first_png_image.size == first_png_image.split()[-1].size else 'NoMatch'
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
    """Main function to create folder lists and CSV files."""
    # Determine the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(script_dir, "folders_processed")

    # Ensure the processed directory exists
    if not os.path.exists(processed_dir):
        try:
            os.makedirs(processed_dir)
            print(f"Created directory: {processed_dir}")
        except Exception as e:
            print(f"Failed to create 'folders_processed' directory at {processed_dir}: {e}")
            raise

    # Attempt to add 'images' directory
    images_path = os.path.join(script_dir, "images")

    folder_dict = {}

    if os.path.exists(images_path) and os.path.isdir(images_path):
        # Add 'images' to folder_dict only if it contains image files
        if contains_image_files(images_path):
            folder_dict[1] = images_path

        # Add subdirectories that contain image files
        for subdirectory in get_subdirectories(images_path):
            if contains_image_files(subdirectory) and subdirectory not in folder_dict.values():
                folder_dict[len(folder_dict) + 1] = subdirectory

    # If 'images' was not added (doesn't exist or has no image files), prompt the user
    if not folder_dict:
        print("The default 'images' directory is missing or contains no image files.")
        print("Please enter directories manually.")

        # Load existing folder locations if available
        folder_locations_path = os.path.join(script_dir, 'folder_locations.txt')
        if os.path.exists(folder_locations_path):
            with open(folder_locations_path, 'r', encoding='utf-8') as f:
                for line in f:
                    number, folder = parse_line(line.strip())
                    if folder:
                        # Convert relative path to absolute path
                        folder_abs = os.path.join(script_dir, folder)
                        if os.path.exists(folder_abs) and os.path.isdir(folder_abs):
                            folder_dict[number] = folder_abs
                        else:
                            print(f"Warning: Folder '{folder}' does not exist and will be skipped.")

        # Interactive prompt loop
        while True:
            folder_path = input("Enter individual paths, parent directories or type quit to stop: ").strip()
            if folder_path.lower() == 'quit':
                break

            folder_abs = os.path.abspath(folder_path)
            if not os.path.exists(folder_abs) or not os.path.isdir(folder_abs):
                print("Invalid directory. Please try again.")
                continue

            if contains_image_files(folder_abs) and folder_abs not in folder_dict.values():
                new_key = max(folder_dict.keys(), default=0) + 1
                folder_dict[new_key] = folder_abs
                print(f"Added folder: {folder_abs}")

            # Add subdirectories that contain image files
            for subdirectory in get_subdirectories(folder_abs):
                if contains_image_files(subdirectory) and subdirectory not in folder_dict.values():
                    new_key = max(folder_dict.keys(), default=0) + 1
                    folder_dict[new_key] = subdirectory
                    print(f"Added subdirectory: {subdirectory}")

    # Final check to ensure at least one folder is added
    if not folder_dict:
        raise Exception("No folders added. Please add at least one main_folder.")

    # Calculate the total number of PNGs and WEBPs
    total_pngs = sum(count_image_files(folder) for folder in folder_dict.values())

    # Gather folder counts and image details
    folder_counts = []
    for folder in folder_dict.values():
        image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.webp'))]
        if image_files:
            first_img = image_files[0]
            try:
                with Image.open(os.path.join(folder, first_img)) as img:
                    width, height = img.size
                    has_alpha = has_alpha_channel(img)
            except Exception as e:
                print(f"Error opening image {first_img} in folder {folder}: {e}")
                first_img = None
                width, height, has_alpha = 0, 0, False
            file_count = count_image_files(folder)
            folder_counts.append((folder, first_img, width, height, has_alpha, file_count))

    # Save folder_count_XXXX.txt with relative paths
    folder_count_filename = f'folder_count_{total_pngs}.txt'
    folder_count_path = os.path.join(processed_dir, folder_count_filename)
    try:
        with open(folder_count_path, 'w', encoding='utf-8') as f:
            for index, (folder, first_img, width, height, has_alpha, count) in enumerate(folder_counts, 1):
                folder_rel = os.path.relpath(folder, script_dir)
                first_png_stripped = os.path.splitext(os.path.basename(first_img))[0] if first_img else "NoImage"
                f.write(f"{index}, {folder_rel}, {first_png_stripped}, {width}x{height} pixels, {count}\n")
        print(f"Folder count file created: {folder_count_path}")
    except Exception as e:
        print(f"Failed to write folder count file at {folder_count_path}: {e}")
        raise

    # Create additional folder_locations_XXXX.csv files with relative paths
    try:
        create_folder_csv_files(folder_counts, processed_dir, script_dir)
    except Exception as e:
        print(f"Failed to create folder CSV files: {e}")
        raise

def sort_image_files(folder_dict, script_dir):
    """Sort image files in each folder using natural sort."""
    sorted_image_files = []
    for number in sorted(folder_dict.keys()):
        folder = folder_dict[number]
        image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.webp'))]
        image_files_sorted = sorted(image_files, key=lambda x: natural_sort_key(x))
        # Convert to relative paths
        image_files_rel = [os.path.join(os.path.relpath(folder, script_dir), f) for f in image_files_sorted]
        sorted_image_files.append(image_files_rel)
    return sorted_image_files

def natural_sort_key(s):
    """Generate a key for natural sorting."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

if __name__ == "__main__":
    write_folder_list()
