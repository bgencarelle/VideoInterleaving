#!/usr/bin/env python3
import os
import sys
import concurrent.futures
from PIL import Image
import webp


def convert_image(source_file, dest_file):
    """
    Converts an image to a lossless WebP file with an alpha channel using the official webp library.
    If the file isn’t a valid image, it is skipped.
    """
    try:
        # Open the image using Pillow and always convert to RGBA
        img = Image.open(source_file).convert("RGBA")

        # Create a WebP picture from the PIL image
        pic = webp.WebPPicture.from_pil(img)
        # Set up a configuration with lossless encoding enabled.
        config = webp.WebPConfig.new(preset=webp.WebPPreset.PHOTO, quality=80)
        config.lossless = 1  # Enable lossless encoding

        # Encode the image to WebP
        encoded = pic.encode(config)
        if encoded is not None:
            with open(dest_file, "wb") as f:
                f.write(encoded.buffer())
            print(f"Converted: {source_file} -> {dest_file}")
        else:
            print(f"Failed to encode: {source_file}")
    except Exception as e:
        print(f"Error converting {source_file}: {e}")


def convert_task(task):
    source_file, dest_file = task
    convert_image(source_file, dest_file)


def main(source_directory):
    # Validate the source directory.
    if not os.path.isdir(source_directory):
        print(f"Error: {source_directory} is not a valid directory.")
        sys.exit(1)

    # Define the destination directory by appending '_webp' to the source folder name.
    abs_source = os.path.abspath(source_directory)
    parent_dir = os.path.dirname(abs_source)
    base_name = os.path.basename(abs_source)
    dest_directory = os.path.join(parent_dir, base_name + "_webp")

    print(f"Creating mirrored directory structure in: {dest_directory}")

    # Build a list of conversion tasks and mirror the directory structure.
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

    # Use ProcessPoolExecutor for parallel conversion.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(convert_task, tasks)


if __name__ == "__main__":
    # Use command-line argument if provided; otherwise, prompt for directory.
    if len(sys.argv) == 2:
        source_dir = sys.argv[1]
    else:
        source_dir = input("Please enter the source directory: ").strip()

    main(source_dir)
