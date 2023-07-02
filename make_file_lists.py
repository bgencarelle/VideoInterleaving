# this turns our list of folders into a list of png and webp files.

import os
import re
from itertools import zip_longest
import csv
import sys
import get_folders_list


def parse_folder_locations(csv_path):
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


def choose_file():
    processed_dir = 'folders_processed'
    available_files = [f for f in os.listdir(processed_dir) if f.endswith('.csv')]

    if not available_files:  # If no files found
        "No Folders Found, running get_folders_list"
        get_folders_list.write_folder_list()
        available_files = [f for f in os.listdir(processed_dir) if f.endswith('.csv')]

    print("Available CSV files:")
    for i, file in enumerate(available_files):
        print(f"{i + 1}: {file}")

    if len(available_files) > 1:
        response = input("Multiple CSV files found. Do you want to process all? (y/n): ")
        if response.lower() == "y":
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
    else:
        return [os.path.join(processed_dir, available_files[0])]


def check_unequal_img_counts(csv_path):
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


def parse_line(line):
    match = re.match(r'(\d+)\. (.+)', line)
    if match:
        return int(match.group(1)), match.group(2)
    return None, None


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def interleave_lists(lists):
    result = []
    for items in zip_longest(*lists, fillvalue=None):
        group = []
        for item in items:
            if item is not None:
                group.append(item)
        result.append(group)
    return result


def write_sorted_images(grouped_image_files, output_folder, csv_path):
    with open(os.path.join(output_folder, f'{os.path.splitext(os.path.basename(csv_path))[0]}_list.csv'), 'w', newline='') as f:
        csv_writer = csv.writer(f)
        sorted_grouped_png_files = sorted(enumerate(grouped_image_files), key=lambda x: x[1][0])
        # Sort by the number in C1
        for index, group in sorted_grouped_png_files:
            csv_writer.writerow([index] + group)
    print(f"csv written to {os.path.splitext(os.path.basename(csv_path))[0]}_list.csv")



def process_files():
    csv_paths = choose_file()
    for csv_path in csv_paths:
        check_unequal_img_counts(csv_path)
        folder_dict = parse_folder_locations(csv_path)
        if not folder_dict:
            print(f"No accessible files in the folders_processed directory for {csv_path}.")
            continue  # Skip to the next CSV file

        sorted_image_files = sort_image_files(folder_dict)
        grouped_image_files = interleave_lists(sorted_image_files)

        output_folder = "generated_img_lists"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        write_sorted_images(grouped_image_files, output_folder, csv_path)


def sort_image_files(folder_dict):
    sorted_image_files = []
    for number in sorted(folder_dict.keys()):
        folder = folder_dict[number]
        image_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.png', '.webp'))]
        image_files.sort(key=natural_sort_key)
        sorted_image_files.append(image_files)
    return sorted_image_files



if __name__ == "__main__":
    process_files()
