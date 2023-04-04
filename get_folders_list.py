import os
import re
import csv
from PIL import Image


def get_subdirectories(path):
    return [os.path.join(root, d) for root, dirs, _ in os.walk(path) for d in dirs]


def contains_png_files(path):
    return any(file.endswith('.png') for file in os.listdir(path))


def count_png_files(path):
    return len([file for file in os.listdir(path) if file.endswith('.png')])


def parse_line(line):
    match = re.match(r'(\d+)\. (.+)', line)
    return (int(match.group(1)), match.group(2)) if match else (None, None)


def write_folder_list():
    folder_dict = {}
    processed_dir = "foldersProcessed"

    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    if os.path.exists('folder_locations.txt'):
        with open('folder_locations.txt', 'r') as f:
            for line in f.readlines():
                number, folder = parse_line(line.strip())
                if folder:
                    folder_dict[number] = folder

    while True:
        folder_path = input("Enter a folder path (or type 'quit' to stop): ").strip()
        if folder_path.lower() == 'quit':
            break

        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            print("Invalid directory. Please try again.")
            continue

        if contains_png_files(folder_path) and folder_path not in folder_dict.values():
            folder_dict[max(folder_dict.keys()) + 1 if folder_dict else 1] = folder_path

        for subdirectory in get_subdirectories(folder_path):
            if contains_png_files(subdirectory) and subdirectory not in folder_dict.values():
                folder_dict[max(folder_dict.keys()) + 1 if folder_dict else 1] = subdirectory

    if not folder_dict:
        raise Exception("No folders added. Please add at least one folder.")

    total_pngs = sum(count_png_files(folder) for folder in folder_dict.values())

    folder_counts = sorted([(folder, *next(
        ((f, *Image.open(os.path.join(folder, f)).size) for f in os.listdir(folder) if f.endswith('.png')),
        (None, 0, 0)), count_png_files(folder)) for folder in folder_dict.values()], key=lambda x: x[3])

    # Updated code to output folder_count_XXXX.csv and strip unnecessary details
    with open(os.path.join(processed_dir, f'folder_count_{total_pngs}.csv'), 'w', newline='') as f:
        writer = csv.writer(f)

        for index, (folder, first_png, width, height, count) in enumerate(folder_counts, 1):
            first_png_stripped = os.path.splitext(os.path.basename(first_png))[0]  # Remove the file extension
            parent_dir = os.path.dirname(folder)  # Get parent directory
            grandparent_dir = os.path.dirname(parent_dir)  # Get grandparent directory

            # Keep only parent and grandparent directories in the folder path
            folder_relative = os.path.join(grandparent_dir, os.path.basename(parent_dir), os.path.basename(folder))

            writer.writerow([index, folder_relative, first_png_stripped, f"{width}x{height} pixels", count])

    # Add the following code snippet to replace the previous file writing block
    file_format = input("Choose output format (txt, csv, or both): ").strip().lower()

    while file_format not in ('txt', 'csv', 'both'):
        print("Invalid format. Please choose 'txt', 'csv', or 'both'.")
        file_format = input("Choose output format (txt, csv, or both): ").strip().lower()

    if file_format in ('txt', 'both'):
        with open(os.path.join(processed_dir, 'folder_locations.txt'), 'w') as f:
            current_count = folder_counts[0][4]
            for index, (folder, _, _, _, count) in enumerate(folder_counts, 1):
                if count != current_count:
                    f.write("#########\n")
                    current_count = count
                f.write(f"{index}. {folder}\n")

    if file_format in ('csv', 'both'):
        with open(os.path.join(processed_dir, 'folder_locations.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            current_count = folder_counts[0][4]
            for index, (folder, _, _, _, count) in enumerate(folder_counts, 1):
                if count != current_count:
                    writer.writerow(["#########"])
                    current_count = count
                writer.writerow([index, folder])  # Separate the number and the file location by a comma

    print("Updated folder_locations files in the foldersProcessed directory.")


if __name__ == "__main__":
    write_folder_list()
