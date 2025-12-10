#!/usr/bin/env python
"""
make_file_lists.py

Scans image directories recursively, verifies integrity, and generates
interleaved CSV lists for the player.

STRICT FILTERING MODE:
- Main Folders: Must start with 0_ to 254_
- Float Folders: Must start with 255_
- Ignored: Anything else (e.g. 999_backup, temp, or non-numeric prefixes)
"""

import os
import re
import csv
import sys
import shutil
from itertools import zip_longest
from PIL import Image
from collections import defaultdict
import numpy as np  # [ADDED] Needed to read .npz headers

import settings

# --- CONSTANTS ---
# Defaults. main.py overrides these in settings if needed.
PROCESSED_DIR_NAME = getattr(settings, 'PROCESSED_DIR', "folders_processed")
GENERATED_DIR_NAME = getattr(settings, 'GENERATED_LISTS_DIR', "generated_img_lists")


# ------------------------------
# Utility functions
# ------------------------------

def get_subdirectories(path):
    subdirs = []
    for root, dirs, _ in os.walk(path):
        for d in dirs:
            subdirs.append(os.path.join(root, d))
    return subdirs


def contains_image_files(path):
    try:
        return any(f.lower().endswith(('.png', '.webp', '.jpg', 'jpeg', '.npz')) for f in os.listdir(path))
    except FileNotFoundError:
        return False


def count_image_files(path):
    try:
        return len([f for f in os.listdir(path) if f.lower().endswith(('.png', '.webp', '.npz', '.jpg', 'jpeg'))])
    except FileNotFoundError:
        return 0


def has_alpha_channel(image):
    return image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info)


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def interleave_lists(lists):
    result = []
    for items in zip_longest(*lists, fillvalue=None):
        group = [item for item in items if item is not None]
        result.append(group)
    return result


# ------------------------------
# Strict Prefix Logic
# ------------------------------

def check_folder_prefix(folder_path, allowed_type):
    """
    Returns True if the folder name matches the strict numeric rules.
    allowed_type: 'main' (0-254) or 'float' (255)
    """
    folder_name = os.path.basename(folder_path)
    prefix_part = folder_name.partition('_')[0]

    if not prefix_part.isdigit():
        return False  # Ignore non-numeric folders (e.g. "backup", "test")

    prefix = int(prefix_part)

    if allowed_type == 'main':
        return 0 <= prefix <= 254
    elif allowed_type == 'float':
        return prefix == 255

    return False


# ------------------------------
# Scanning Logic
# ------------------------------

def scan_directory_recursive(base_path, script_dir, folder_type):
    """
    Recursively scans a directory for image folders matching the strict prefix rules.
    folder_type: 'main' or 'float'
    """
    results = []
    if not os.path.exists(base_path) or not os.path.isdir(base_path):
        print(f"Error: Directory '{base_path}' is missing.")
        return results

    # Get all subdirectories plus the root itself
    dirs_to_scan = [base_path] + get_subdirectories(base_path)

    for subdir in dirs_to_scan:
        # 1. STRICT FILTER: Check name before scanning content
        if subdir != base_path:
            if not check_folder_prefix(subdir, folder_type):
                continue

        # 2. Check for images
        if not contains_image_files(subdir):
            continue

        image_files = [f for f in os.listdir(subdir) if f.lower().endswith(('.png', '.webp', '.npz', '.jpg', '.jpeg'))]
        if not image_files:
            continue

        # Sort to ensure consistent "First Image" logic
        image_files.sort(key=natural_sort_key)
        first_image = image_files[0]

        width, height, has_alpha = 0, 0, False

        try:
            file_path = os.path.join(subdir, first_image)

            # --- [CHANGED] Detect File Type ---
            if first_image.lower().endswith('.npz'):
                # Handle ASCII Asset
                try:
                    data = np.load(file_path)
                    # .npz chars shape is (Height, Width)
                    h_arr, w_arr = data['chars'].shape
                    width, height = int(w_arr), int(h_arr)

                    # Detect Alpha: Check if any characters are a space ' '
                    # (This aligns with our baker logic where alpha pixels become spaces)
                    has_alpha = bool(np.any(data['chars'] == ' '))
                except Exception as e:
                    print(f"Error reading NPZ {first_image}: {e}")
                    raise e
            else:
                # Handle Standard Image (PIL)
                with Image.open(file_path) as img:
                    width, height = img.size
                    has_alpha = has_alpha_channel(img)

        except Exception as e:
            print(f"Error processing image {first_image} in {subdir}: {e}")
            first_image = None

        file_count = count_image_files(subdir)
        results.append((subdir, first_image, width, height, has_alpha, file_count))

    return results


def create_folder_csv_files(counts_main, counts_float, processed_dir, script_dir):
    """Write intermediate CSVs separated by Main/Float groups."""

    groups_main = defaultdict(list)
    groups_float = defaultdict(list)

    # Helper to pack data
    def pack(info):
        abs_path = info[0]
        rel_path = os.path.relpath(abs_path, script_dir)
        return (rel_path, *info[1:])

    for info in counts_main:
        groups_main[info[5]].append(pack(info))

    for info in counts_float:
        groups_float[info[5]].append(pack(info))

    def write_group_csv(group, name_fmt):
        for f_count, items in group.items():
            # Sort by numeric prefix
            items.sort(key=lambda x: (
                int(os.path.basename(x[0]).partition('_')[0]) if os.path.basename(x[0]).partition('_')[
                    0].isdigit() else float('inf'),
                os.path.basename(x[0])
            ))

            csv_path = os.path.join(processed_dir, name_fmt.format(f_count))
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for idx, (rel_path, first_img, w, h, alpha, count) in enumerate(items):
                    alpha_str = 'Yes' if alpha else 'No'
                    ext = os.path.splitext(first_img)[1] if first_img else 'N/A'
                    writer.writerow([idx, rel_path, f"{w}x{h} pixels", count, alpha_str, 'Match', ext])
            print(f"CSV file created: {csv_path}")

    write_group_csv(groups_main, 'main_folder_{}.csv')
    write_group_csv(groups_float, 'float_folder_{}.csv')


def write_folder_list():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(script_dir, PROCESSED_DIR_NAME)

    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    os.makedirs(processed_dir)

    main_path = os.path.join(script_dir, settings.MAIN_FOLDER_PATH)
    float_path = os.path.join(script_dir, settings.FLOAT_FOLDER_PATH)

    # 1. Scan with STRICT MODE
    print(f"Scanning Main: {main_path} (Allow: 0_-254_)")
    counts_main = scan_directory_recursive(main_path, script_dir, 'main')

    print(f"Scanning Float: {float_path} (Allow: 255_)")
    counts_float = scan_directory_recursive(float_path, script_dir, 'float')

    all_counts = counts_main + counts_float
    total_images = sum(x[5] for x in all_counts)

    if not all_counts:
        print("No valid images found matching prefix rules.")
        # We don't exit here because main.py expects files to exist,
        # but the next steps will handle empty lists gracefully.

    # 2. Write Summary
    count_path = os.path.join(processed_dir, f'folder_count_{total_images}.txt')
    with open(count_path, 'w', encoding='utf-8') as f:
        for i, (path, img, w, h, _, count) in enumerate(all_counts):
            rel = os.path.relpath(path, script_dir)
            f.write(f"{i}, {rel}, {img}, {w}x{h} pixels, {count}\n")
    print(f"Folder count file created: {count_path}")

    # 3. Write Detail CSVs
    create_folder_csv_files(counts_main, counts_float, processed_dir, script_dir)


# ------------------------------
# Matching & Generation
# ------------------------------

def find_default_csvs(processed_dir):
    """Find matching main/float CSV pairs based on image count."""
    float_files = {}
    main_files = {}

    if not os.path.exists(processed_dir):
        return None

    for f in os.listdir(processed_dir):
        if match := re.match(r'^float_folder_(\d+)\.csv$', f):
            float_files[int(match.group(1))] = os.path.join(processed_dir, f)
        elif match := re.match(r'^main_folder_(\d+)\.csv$', f):
            main_files[int(match.group(1))] = os.path.join(processed_dir, f)

    common = set(float_files.keys()) & set(main_files.keys())

    if not common:
        print(f"No matching counts found.\nFloat: {list(float_files.keys())}\nMain: {list(main_files.keys())}")
        return None

    # Pick largest count if multiple
    chosen = max(common)
    if len(common) > 1:
        print(f"Warning: Multiple matches {common}. Selected {chosen}.")

    return [float_files[chosen], main_files[chosen]]


def check_unequal_img_counts(csv_path):
    if os.path.exists(csv_path):
        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            counts = set()
            for row in reader:
                if row: counts.add(int(row[3]))
            if len(counts) > 1:
                print(f"Error: Unequal file count in {csv_path}")
                sys.exit(0)


def sort_image_files(folder_dict):
    sorted_files = []
    for num in sorted(folder_dict.keys()):
        folder = folder_dict[num]
        imgs = [os.path.join(folder, x) for x in os.listdir(folder)
                if x.lower().endswith(('.png', '.npz', '.webp', '.jpg', '.jpeg'))]
        imgs.sort(key=natural_sort_key)
        sorted_files.append(imgs)
    return sorted_files


def parse_folder_locations(csv_path):
    d = {}
    if os.path.exists(csv_path):
        with open(csv_path, 'r', newline='') as f:
            for row in csv.reader(f):
                if row: d[int(row[0])] = row[1]
    return d


def process_files():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(script_dir, PROCESSED_DIR_NAME)
    generated_dir = os.path.join(script_dir, GENERATED_DIR_NAME)

    # 1. Generate Metadata
    write_folder_list()

    # 2. Find Matches
    csv_paths = find_default_csvs(processed_dir)
    if not csv_paths:
        print("Critical: No matching CSV pairs found.")
        sys.exit(1)

    print("Default CSV files found:")
    for c in csv_paths: print(f" - {c}")

    # 3. Generate Final Lists
    if os.path.exists(generated_dir):
        shutil.rmtree(generated_dir)
    os.makedirs(generated_dir)

    for csv_path in csv_paths:
        check_unequal_img_counts(csv_path)
        folder_dict = parse_folder_locations(csv_path)
        if not folder_dict: continue

        sorted_groups = sort_image_files(folder_dict)
        interleaved = interleave_lists(sorted_groups)

        out_name = f'{os.path.splitext(os.path.basename(csv_path))[0]}_list.csv'
        out_path = os.path.join(generated_dir, out_name)
        with open(out_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for idx, grp in enumerate(interleaved):
                writer.writerow([idx] + grp)
        print(f"CSV written to {out_path}")


if __name__ == "__main__":
    process_files()