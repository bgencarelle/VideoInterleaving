# This script turns our list of folders into a list of PNG and WEBP files.

import os
import re
from itertools import zip_longest
import csv
import sys
import get_folders_list

def parse_folder_locations(csv_path):
    """
    Parses the given CSV file to create a dictionary mapping folder numbers to folder paths.
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
    with the same XXXX.
    Returns the list of paths if found, else None.
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

def choose_file(processed_dir):
    """
    Allows the user to choose which CSV files to process if the default pair isn't found.
    """
    available_files = [f for f in os.listdir(processed_dir) if f.endswith('.csv')]

    if not available_files:  # If no files found
        print("No CSV files found, running get_folders_list.")
        get_folders_list.write_folder_list()
        available_files = [f for f in os.listdir(processed_dir) if f.endswith('.csv')]

    print("Available CSV files:")
    for i, file in enumerate(available_files):
        print(f"{i + 1}: {file}")

    if len(available_files) > 2:
        response = input("Multiple CSV files found. Do you want to process all? (y/n): ").strip().lower()
        if response == "y":
            return [os.path.join(processed_dir, f) for f in available_files]
        else:
            while True:
                try:
                    choice = int(input("Enter the number corresponding to the desired file: ")) - 1
                    if 0 <= choice < len(available_files):
                        return [os.path.join(processed_dir, available_files[choice])]
                    else:
                        print("Invalid choice. Please enter a valid number.")
                except ValueError:
                    print("Invalid input. Please enter a valid number.")
    elif len(available_files) == 2:
        return [os.path.join(processed_dir, available_files[0])]
    else:
        return []

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
                img_counts.add(int(row[3]))  # Column 4 has index 3
                if len(img_counts) > 1:
                    print("error: unequal file count found, please fix before proceeding")
                    sys.exit(0)

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
        group = []
        for item in items:
            if item is not None:
                group.append(item)
        result.append(group)
    return result

def write_sorted_images(grouped_image_files, output_folder, csv_path):
    """
    Writes the interleaved and sorted image file paths to a new CSV in the output folder.
    """
    output_csv_name = f'{os.path.splitext(os.path.basename(csv_path))[0]}_list.csv'
    with open(os.path.join(output_folder, output_csv_name), 'w', newline='') as f:
        csv_writer = csv.writer(f)
        # Enumerate starts at 0
        for index, group in enumerate(grouped_image_files, start=0):
            csv_writer.writerow([index] + group)
    print(f"CSV written to {os.path.join(output_folder, output_csv_name)}")

def sort_image_files(folder_dict):
    """
    Sorts image files in each folder using natural sort.
    """
    sorted_image_files = []
    for number in sorted(folder_dict.keys()):
        folder = folder_dict[number]
        image_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.webp'))]
        image_files.sort(key=natural_sort_key)
        sorted_image_files.append(image_files)
    return sorted_image_files

def process_files():
    """
    Main function to process CSV files and generate interleaved image lists.
    """
    # Determine the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(script_dir, 'folders_processed')
    generated_dir = os.path.join(script_dir, 'generated_img_lists')

    # Check if 'folders_processed' exists
    if not os.path.exists(processed_dir):
        print("'folders_processed' directory not found. Running get_folders_list to generate it.")
        get_folders_list.write_folder_list()

    # Refresh the existence after attempting to create
    if not os.path.exists(processed_dir):
        print("Failed to create 'folders_processed' directory. Please check permissions.")
        sys.exit(1)

    # Attempt to find default CSVs
    default_csvs = find_default_csvs(processed_dir)

    if default_csvs:
        print("Default CSV files found:")
        for csv_file in default_csvs:
            print(f" - {csv_file}")
        csv_paths = default_csvs
    else:
        print("Default CSV files not found. Running get_folders_list to generate them.")
        get_folders_list.write_folder_list()
        # Try finding default CSVs again
        default_csvs = find_default_csvs(processed_dir)
        if default_csvs:
            print("Default CSV files found after running get_folders_list:")
            for csv_file in default_csvs:
                print(f" - {csv_file}")
            csv_paths = default_csvs
        else:
            print("Default CSV files still not found. Falling back to user prompts.")
            csv_paths = choose_file(processed_dir)

    # Ensure 'generated_img_lists' exists and is clean
    if not os.path.exists(generated_dir):
        os.makedirs(generated_dir)
    else:
        # Remove existing files in 'generated_img_lists'
        for file in os.listdir(generated_dir):
            file_path = os.path.join(generated_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    for csv_path in csv_paths:
        check_unequal_img_counts(csv_path)
        folder_dict = parse_folder_locations(csv_path)
        if not folder_dict:
            print(f"No accessible files in the 'folders_processed' directory for {csv_path}.")
            continue  # Skip to the next CSV file

        sorted_image_files = sort_image_files(folder_dict)
        grouped_image_files = interleave_lists(sorted_image_files)

        write_sorted_images(grouped_image_files, generated_dir, csv_path)

if __name__ == "__main__":
    process_files()
