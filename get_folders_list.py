# this creates the lists of folders that contain our png and webp files
import os
import re
import csv
from PIL import Image
from collections import defaultdict


def get_subdirectories(path):
    return [os.path.join(root, d) for root, dirs, _ in os.walk(path) for d in dirs]


def contains_image_files(path):
    return any(file.endswith(('.png', '.webp')) for file in os.listdir(path))


def count_image_files(path):
    return len([file for file in os.listdir(path) if file.endswith(('.png', '.webp'))])


def parse_line(line):
    match = re.match(r'(\d+)\. (.+)', line)
    return (int(match.group(1)), match.group(2)) if match else (None, None)


def has_alpha_channel(image):
    return image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info)


def create_folder_csv_files(folder_counts, processed_dir):
    groups = defaultdict(list)
    float_group = defaultdict(list)

    for index, (folder, first_png, width, height, has_alpha, file_count) in enumerate(folder_counts, 1):
        if os.path.basename(folder).startswith('255_'):
            float_group[file_count].append((folder, first_png, width, height, has_alpha, file_count))
        else:
            groups[file_count].append((folder, first_png, width, height, has_alpha, file_count))

    def write_csv(group, file_name_format):
        for file_count, sub_group in group.items():
            # Sort the group based on the numeric prefix in the actual folder name, or the folder name itself
            sub_group.sort(key=lambda x: (int(os.path.basename(x[0]).partition('_')[0]) if os.path.basename(x[0]).partition('_')[0].isdigit() else float('inf'), os.path.basename(x[0])))
            with open(os.path.join(processed_dir, file_name_format.format(file_count)), 'w', newline='') as f:
                writer = csv.writer(f)
                for index, (folder, first_png, width, height, has_alpha, file_count) in enumerate(sub_group, 1):
                    first_png_image = Image.open(os.path.join(folder, first_png))
                    file_extension = os.path.splitext(first_png)[1]
                    if has_alpha:
                        alpha_match = 'Match' if first_png_image.size == first_png_image.split()[-1].size else 'NoMatch'
                    else:
                        alpha_match = 'NoAlpha'
                    writer.writerow([index, folder, f"{width}x{height} pixels", file_count, 'Yes' if has_alpha else 'No', alpha_match, file_extension])

    write_csv(groups, 'main_folder_{}.csv')
    write_csv(float_group, 'float_folder_{}.csv')


def write_folder_list():
    folder_dict = {}
    processed_dir = "folders_processed"
    background_mode = False
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    if os.path.exists('folder_locations.txt'):
        with open('folder_locations.txt', 'r') as f:
            for line in f.readlines():
                number, folder = parse_line(line.strip())
                if folder:
                    folder_dict[number] = folder

    while True:
        folder_path = input("Enter individual paths, parent directories or type quit to stop:").strip()
        if folder_path.lower() == 'quit':
            break

        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            print("Invalid directory. Please try again.")
            continue

        if contains_image_files(folder_path) and folder_path not in folder_dict.values():
            folder_dict[max(folder_dict.keys()) + 1 if folder_dict else 1] = folder_path

        for subdirectory in get_subdirectories(folder_path):
            if contains_image_files(subdirectory) and subdirectory not in folder_dict.values():
                folder_dict[max(folder_dict.keys()) + 1 if folder_dict else 1] = subdirectory

    if not folder_dict:
        raise Exception("No folders added. Please add at least one main_folder.")

    total_pngs = sum(count_image_files(folder) for folder in folder_dict.values())

    folder_counts = [(folder, *next(
        ((f, *Image.open(os.path.join(folder, f)).size, has_alpha_channel(Image.open(os.path.join(folder, f)))) for f in
         os.listdir(folder) if f.endswith(('.png', '.webp'))),
        (None, 0, 0, False)), count_image_files(folder)) for folder in folder_dict.values()]

    # Save folder_count_XXXX.txt
    with open(os.path.join(processed_dir, f'folder_count_{total_pngs}.txt'), 'w') as f:
        for index, (folder, first_img, width, height, has_alpha, count) in enumerate(folder_counts, 1):
            first_png_stripped = os.path.splitext(os.path.basename(first_img))[0]
            f.write(f"{index}, {folder}, {first_png_stripped}, {width}x{height} pixels, {count}\n")

    # Create additional folder_locations_XXXX.csv files
    create_folder_csv_files(folder_counts, processed_dir)


if __name__ == "__main__":
    write_folder_list()
