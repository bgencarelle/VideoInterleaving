# get_folders_list.py
# This script scans directories for PNG and WEBP files, organizes them, and generates CSV lists.

import os
import re
import csv
from PIL import Image
from collections import defaultdict
import shutil

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
    if os.path.exists(processed_dir):
        # Clear the processed directory if it already exists
        shutil.rmtree(processed_dir)
    os.makedirs(processed_dir)

    # Check for the existence of the 'images' directory
    images_path = os.path.join(script_dir, "images")

    if not os.path.exists(images_path) or not os.path.isdir(images_path):
        print(
            "The 'images' directory is missing. Please create the directory and populate it with PNG or WEBP files before running this script again.")
        return  # Exit the function

    # Collect all subdirectories and main images directory
    folder_dict = {}
    folder_key = 1

    # Recursively add the "images" path and its subdirectories
    for subdirectory in [images_path] + get_subdirectories(images_path):
        if contains_image_files(subdirectory):
            folder_dict[folder_key] = subdirectory
            folder_key += 1

    # Abort if no valid folders are found
    if not folder_dict:
        print("No valid image files were found in the 'images' directory or its subdirectories.")
        return  # Exit the function

    # Collect folder details and count images
    total_images = 0
    folder_counts = []

    for folder in folder_dict.values():
        image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.webp'))]
        if image_files:
            first_image = image_files[0]
            try:
                with Image.open(os.path.join(folder, first_image)) as img:
                    width, height = img.size
                    has_alpha = has_alpha_channel(img)
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

    # Create CSV files for main and float folders
    create_folder_csv_files(folder_counts, processed_dir, script_dir)

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
