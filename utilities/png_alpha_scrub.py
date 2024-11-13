import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, PngImagePlugin


def transparent_to_black(input_file):
    img = Image.open(input_file).convert("RGBA")
    img_data = np.array(img)

    # Set RGB values of transparent pixels to black
    img_data[img_data[..., 3] == 0] = [0, 0, 0, 0]

    # Create a new Image object from the modified NumPy array
    new_img = Image.fromarray(img_data)
    new_img.save(input_file, "PNG")


def process_file(filepath):
    try:
        print(f"Processing {filepath}")
        transparent_to_black(filepath)
    except Exception as e:
        print(f"An error occurred while processing {filepath}: {e}")


def process_directory(directory, max_workers=4):
    png_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".png"):
                filepath = os.path.join(root, file)
                png_files.append(filepath)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_file, png_files)


if __name__ == "__main__":
    try:
        input_directory = input("Please enter the directory path: ")
        if os.path.isdir(input_directory):
            process_directory(input_directory)
        else:
            print("Invalid directory path. Please check the path and try again.")
    except Exception as e:
        print(f"An error occurred: {e}")
