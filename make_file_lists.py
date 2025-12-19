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
import numpy as np  # [ADDED]

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
        # [MODIFIED] Added new extensions
        valid = ('.png', '.webp', '.jpg', 'jpeg', '.npy', '.npz', '.spz', '.spy')
        return any(f.lower().endswith(valid) for f in os.listdir(path))
    except FileNotFoundError:
        return False


def count_image_files(path):
    try:
        # [MODIFIED] Added new extensions
        valid = ('.png', '.webp', '.jpg', 'jpeg', '.npy', '.npz', '.spz', '.spy')
        return len([f for f in os.listdir(path) if f.lower().endswith(valid)])
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
                # Silently ignore 999_ or wrong types
                continue

        # 2. Check for images
        if not contains_image_files(subdir):
            continue

        # [MODIFIED] Added extensions to list comprehension
        valid = ('.png', '.webp', '.jpg', 'jpeg', '.npy', '.npz', '.spz', '.spy')
        image_files = [f for f in os.listdir(subdir) if f.lower().endswith(valid)]

        if not image_files:
            continue

        # Sort to ensure consistent "First Image" logic
        image_files.sort(key=natural_sort_key)
        first_image = image_files[0]

        width, height, has_alpha = 0, 0, False

        try:
            file_path = os.path.join(subdir, first_image)
            ext = first_image.split('.')[-1].lower()

            # --- [INSERTED] Handle New Types ---
            if ext == 'npy':
                # ASCII Stack
                data = np.load(file_path, mmap_mode='r')
                if data.ndim == 3 and data.shape[0] == 2:
                    width, height = int(data.shape[2]), int(data.shape[1])
                    has_alpha = True
                else:
                    width, height = data.shape[1], data.shape[0]
                    has_alpha = False

            elif ext in ('spz', 'npz'):
                # Compressed Image
                with np.load(file_path) as archive:
                    if 'image' in archive:
                        img = archive['image']
                    elif 'chars' in archive:
                        img = archive['chars']  # Legacy
                    else:
                        img = archive[archive.files[0]]

                    if img.ndim == 3:
                        h_arr, w_arr = img.shape[:2]
                        has_alpha = (img.shape[2] == 4)
                    else:
                        h_arr, w_arr = img.shape
                        has_alpha = False
                    width, height = int(w_arr), int(h_arr)

            elif ext == 'spy':
                # Raw Image
                img = np.load(file_path, mmap_mode='r')
                h_arr, w_arr = img.shape[:2]
                width, height = int(w_arr), int(h_arr)
                has_alpha = (img.shape[2] == 4) if img.ndim == 3 else False

            # --- [ORIGINAL] Standard Images ---
            else:
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

        # [MODIFIED] Added extensions
        valid = ('.png', '.webp', '.jpg', 'jpeg', '.npy', '.npz', '.spz', '.spy')
        imgs = [os.path.join(folder, x) for x in os.listdir(folder) if x.lower().endswith(valid)]

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


# ------------------------------
# Image List Loading Functions
# (Moved from calculators.py)
# ------------------------------

def select_image_list_files():
    """
    Selects main and secondary CSV files from the generated lists directory.
    Returns tuple of (main_csv_path, secondary_csv_path).
    """
    csv_dir = getattr(settings, 'GENERATED_LISTS_DIR', 'generated_img_lists')

    if not os.path.exists(csv_dir):
        print(f"Directory '{csv_dir}' not found. Creating it and generating file lists.")
        os.makedirs(csv_dir)
        process_files()

    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

    if not csv_files:
        print(f"No CSV files found in '{csv_dir}'. Generating file lists.")
        process_files()
        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

    selected_files = {'main': None, 'secondary': None}

    main_folder_file = next((f for f in csv_files if f.startswith('main_folder')), None)
    if main_folder_file:
        selected_files['main'] = os.path.join(csv_dir, main_folder_file)
        # Avoid removing while iterating, just filter
        others = [f for f in csv_files if f != main_folder_file]
        selected_files['secondary'] = os.path.join(csv_dir, others[0]) if others else selected_files['main']
        print("Auto-selected main and secondary files:", selected_files)
        return selected_files['main'], selected_files['secondary']

    if len(csv_files) == 1:
        selected_files['main'] = selected_files['secondary'] = os.path.join(csv_dir, csv_files[0])
        print("Only one CSV file found, defaulting both to:", selected_files['main'])
        return selected_files['main'], selected_files['secondary']

    selected_files['main'] = os.path.join(csv_dir, csv_files[0])
    selected_files['secondary'] = os.path.join(csv_dir, csv_files[1])
    print("Multiple CSV files found. Defaulting main to", selected_files['main'], "and secondary to",
          selected_files['secondary'])
    return selected_files['main'], selected_files['secondary']


def load_image_paths_from_csv(file_path):
    """
    Loads image paths from a CSV file.
    Returns tuple of (image_paths, png_paths_len).
    """
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        png_paths = [row[1:] for row in csv_reader]  # Skip first column (index)
        png_paths_len = len(png_paths)
    return png_paths, png_paths_len


# ------------------------------
# Presets System Functions
# (Moved from calculators.py)
# ------------------------------

def get_midi_length(midi_file_path):
    """Calculate video length in frames from a MIDI file."""
    import mido
    midi_file = mido.MidiFile(midi_file_path)
    midi_length_seconds = midi_file.length
    frames_per_second = 30  # Assuming 30 FPS
    return int(midi_length_seconds * frames_per_second)


def set_video_length(video_name, video_name_length):
    """Write video length to presets CSV file."""
    presets_folder = "presets"
    if not os.path.exists(presets_folder):
        os.makedirs(presets_folder)
    csv_file_path = os.path.join(presets_folder, "set_video_length.csv")
    with open(csv_file_path, mode="a", newline='') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow([video_name, video_name_length])


def get_video_length(video_number=0):
    """Read video length from presets CSV file."""
    presets_folder = "presets"
    csv_file_path = os.path.join(presets_folder, "set_video_length.csv")
    if not os.path.exists(csv_file_path):
        print("No video length data found. Running setup.")
        setup_video_length()
    with open(csv_file_path, mode="r", newline='') as file:
        reader = csv.reader(file)
        video_lengths = list(reader)
        if len(video_lengths) == 1:
            return int(video_lengths[0][1])
        if 0 < video_number <= len(video_lengths):
            return int(video_lengths[video_number - 1][1])
        print("Multiple videos found. Please choose one:")
        for video in video_lengths:
            print(video[0])
        while True:
            selected_video = input("Enter the name of the video: ").strip()
            for video in video_lengths:
                if video[0] == selected_video:
                    return int(video[1])
            print("Invalid video name. Please try again.")


def setup_video_length():
    """
    Interactive setup for video length (from calculate_frame_duration).
    Returns the calculated frame_duration.
    """
    presets_folder = "presets"
    csv_file_path = os.path.join(presets_folder, "set_video_length.csv")
    if not os.path.exists(csv_file_path):
        print("No video length preset found. Please set up video length.")
        mode = input("Enter 'm' for manual entry or 'd' for MIDI-derived length: ").strip().lower()
        if mode == 'm':
            while True:
                try:
                    video_length = int(input("Enter the video length in frames: "))
                    video_length = abs(video_length)
                    break
                except ValueError:
                    print("Invalid input. Please enter a positive integer.")
        elif mode == 'd':
            while True:
                midi_file_path = input("Enter the MIDI file path: ").strip()
                if not os.path.exists(midi_file_path):
                    print("MIDI file not found. Try again.")
                elif not midi_file_path.lower().endswith('.mid'):
                    print("Invalid file type. Please provide a MIDI file.")
                else:
                    try:
                        video_length = get_midi_length(midi_file_path)
                        break
                    except Exception as e:
                        print(f"Error: {e}")
        else:
            print("Invalid mode selected; defaulting to manual entry.")
            while True:
                try:
                    video_length = int(input("Enter the video length in frames: "))
                    video_length = abs(video_length)
                    break
                except ValueError:
                    print("Invalid input. Please enter a positive integer.")
        video_name = input("Enter the name of the video: ").strip()
        set_video_length(video_name, video_length)
    else:
        video_length = get_video_length()
    return video_length


def initialize_image_lists(clock_mode):
    """
    Initializes the preset data, image lists, and frame duration.
    Returns tuple of (csv_main, main_image_paths, float_image_paths).
    Also sets index_calculator module-level variables for MIDI modes.
    """
    from constantStorage.midi_constants import FREE_CLOCK
    import index_calculator
    
    csv_main, csv_float = select_image_list_files()
    main_image_paths, png_paths_len = load_image_paths_from_csv(csv_main)
    float_image_paths, _ = load_image_paths_from_csv(csv_float)
    
    if png_paths_len == 0:
        raise ValueError("No image paths found in CSV file. Cannot initialize image lists.")
    
    video_length = None
    frame_duration = None
    
    if clock_mode == FREE_CLOCK or clock_mode == 255:
        video_length = len(main_image_paths)
        frame_duration = video_length / png_paths_len if png_paths_len > 0 else 1.0
        print("FREE_CLOCK mode: video_length preset set from main folder length =", video_length)
        print("Computed frame_duration =", frame_duration)
    else:
        # Non-FREE_CLOCK mode: use presets system
        video_length = setup_video_length()
        frame_duration = video_length / png_paths_len if png_paths_len > 0 else 1.0
        print("Frame scaling factor for this video:", frame_duration)
    
    # Set module-level variables in index_calculator for MIDI modes
    index_calculator.png_paths_len = png_paths_len
    index_calculator.frame_duration = frame_duration
    
    return csv_main, main_image_paths, float_image_paths


if __name__ == "__main__":
    process_files()