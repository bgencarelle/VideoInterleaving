#!/usr/bin/env python3
import os
import sys
import cv2
import concurrent.futures


def convert_image(source_file, dest_file):
    """
    Reads an image using OpenCV and writes it as a lossless WebP file.
    If the file isn't a valid image, it is skipped.
    """
    try:
        # Read image from the source file.
        img = cv2.imread(source_file, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Skipping non-image or unreadable file: {source_file}")
            return
        # Write image as lossless WebP. The flag cv2.IMWRITE_WEBP_LOSSLESS is set to 1.
        if cv2.imwrite(dest_file, img, [cv2.IMWRITE_WEBP_LOSSLESS, 1]):
            print(f"Converted: {source_file} -> {dest_file}")
        else:
            print(f"Failed to write: {dest_file}")
    except Exception as e:
        print(f"Error converting {source_file}: {e}")


def convert_task(task):
    """
    Wrapper function to unpack the task tuple.
    """
    source_file, dest_file = task
    convert_image(source_file, dest_file)


def main(source_directory):
    # Validate the source directory.
    if not os.path.isdir(source_directory):
        print(f"Error: {source_directory} is not a valid directory.")
        sys.exit(1)

    # Define the destination directory by appending '_webp' to the top folder name.
    abs_source = os.path.abspath(source_directory)
    parent_dir = os.path.dirname(abs_source)
    base_name = os.path.basename(abs_source)
    dest_directory = os.path.join(parent_dir, base_name + "_webp")

    print(f"Creating mirrored directory structure in: {dest_directory}")

    # Build a list of conversion tasks and recreate the directory structure.
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

    # Use a ProcessPoolExecutor for parallel image conversion.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(convert_task, tasks)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_to_webp.py <source_directory>")
        sys.exit(1)

    main(sys.argv[1])
