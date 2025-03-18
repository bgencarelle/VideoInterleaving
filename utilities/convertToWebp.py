#!/usr/bin/env python3
import os
import sys
import cv2
import webp
from multiprocessing import Pool

def convert_image(source_file, dest_file):
    """
    Opens an image using OpenCV, ensures it has an alpha channel (RGBA),
    and writes it as a lossless WebP file using the official webp library.
    """
    # Read image using OpenCV (including alpha if present)
    img = cv2.imread(source_file, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Skipping non-image or unreadable file: {source_file}")
        return

    # Ensure the image has an alpha channel.
    # If the image is grayscale, convert to RGBA.
    # If it is BGR (3 channels), convert to RGBA.
    # If it is BGRA (4 channels), convert from BGRA to RGBA.
    if len(img.shape) == 2:
        img_rgba = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
    elif img.shape[2] == 3:
        img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    elif img.shape[2] == 4:
        img_rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    else:
        print(f"Unexpected image shape: {img.shape} in file {source_file}")
        return

    try:
        # Write the image using the official webp library.
        # Using lossless encoding with quality 80.
        success = webp.imwrite(dest_file, img_rgba, lossless=1, quality=80)
        if success:
            print(f"Converted: {source_file} -> {dest_file}")
        else:
            print(f"Failed to write: {dest_file}")
    except Exception as e:
        print(f"Error converting {source_file}: {e}")

def process_task(task):
    source_file, dest_file = task
    convert_image(source_file, dest_file)

def main(source_directory):
    if not os.path.isdir(source_directory):
        print(f"Error: {source_directory} is not a valid directory.")
        sys.exit(1)

    # Define destination directory (mirrored structure)
    abs_source = os.path.abspath(source_directory)
    parent_dir = os.path.dirname(abs_source)
    base_name = os.path.basename(abs_source)
    dest_directory = os.path.join(parent_dir, base_name + "_webp")
    print(f"Creating mirrored directory structure in: {dest_directory}")

    # Build list of tasks (source, destination) and recreate the directory structure.
    tasks = []
    for root, dirs, files in os.walk(source_directory):
        rel_path = os.path.relpath(root, source_directory)
        dest_path = os.path.join(dest_directory, rel_path)
        os.makedirs(dest_path, exist_ok=True)
        for file in files:
            file_name, _ = os.path.splitext(file)
            dest_file = os.path.join(dest_path, file_name + ".webp")
            tasks.append((os.path.join(root, file), dest_file))

    print(f"Found {len(tasks)} files to process.")

    # Use multiprocessing Pool to process files in parallel.
    with Pool() as pool:
        pool.map(process_task, tasks)

if __name__ == "__main__":
    # Prompt for the source directory if not provided via command-line.
    if len(sys.argv) == 2:
        source_dir = sys.argv[1]
    else:
        source_dir = input("Please enter the source directory: ").strip()
    main(source_dir)
