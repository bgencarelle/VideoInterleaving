import os
import subprocess
from concurrent.futures import ThreadPoolExecutor


def convert_to_webp(input_file, output_file):
    try:
        subprocess.run(["cwebp", input_file, "-o", output_file], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while converting file: {input_file}. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred. Details: {e}")


def convert_directory_to_webp(directory):
    base_dir = os.path.basename(directory)
    parent_dir = os.path.dirname(directory)
    new_parent_dir = os.path.join(parent_dir, f"{base_dir}_webp")

    with ThreadPoolExecutor(max_workers=4) as executor:
        for dirpath, dirnames, filenames in os.walk(directory):
            relative_dirpath = os.path.relpath(dirpath, directory)
            new_dirpath = os.path.join(new_parent_dir, relative_dirpath)

            os.makedirs(new_dirpath, exist_ok=True)

            for filename in filenames:
                if filename.endswith(".png") or filename.endswith(".jpg"):
                    old_filepath = os.path.join(dirpath, filename)
                    new_filename = f"{os.path.splitext(filename)[0]}.webp"
                    new_filepath = os.path.join(new_dirpath, new_filename)
                    executor.submit(convert_to_webp, old_filepath, new_filepath)


if __name__ == "__main__":
    directory = input("Please enter a directory path: ")
    convert_directory_to_webp(directory)
