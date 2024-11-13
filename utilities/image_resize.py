import os
import sys
from PIL import Image, ImageCms
import concurrent.futures
import threading

counter_lock = threading.Lock()
processed_files = 0
total_files = 0


def count_png_files(src):
    count = 0
    for root, _, files in os.walk(src):
        for file in files:
            if file.endswith('.png'):
                count += 1
    return count


def create_smol_structure(src, dest, height, num_threads):
    global processed_files
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        for root, dirs, files in os.walk(src):
            relative_path = os.path.relpath(root, src)
            new_dir = os.path.join(dest, relative_path)
            os.makedirs(new_dir, exist_ok=True)
            for file in files:
                if file.endswith('.png'):
                    src_path = os.path.join(root, file)
                    dest_path = os.path.join(new_dir, file)
                    executor.submit(resize_image, src_path, dest_path, height)

    print(f"\nAll {processed_files} images have been resized.")


def resize_image(src_path, dest_path, height):
    global processed_files
    with Image.open(src_path) as img:
        aspect_ratio = float(img.width) / float(img.height)
        new_height = height
        new_width = int(aspect_ratio * new_height)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img = add_srgb_profile(img)
        img.save(dest_path, 'PNG', icc_profile=img.info.get('icc_profile'))

    with counter_lock:
        processed_files += 1
        print(f"Processed {processed_files}/{total_files} images", end='\r')
        sys.stdout.flush()


def add_srgb_profile(img):
    if 'icc_profile' not in img.info:
        srgb_profile = ImageCms.createProfile("sRGB")
        img = ImageCms.profileToProfile(img, srgb_profile, srgb_profile, outputMode='RGBA')
    return img


def main():
    parent_dir = input("Enter the parent directory: ")
    height = int(input("Enter the resizing height: "))
    num_threads = int(input("Enter the number of threads to use: "))

    if not os.path.exists(parent_dir):
        print(f"Error: {parent_dir} does not exist.")
        sys.exit(1)

    smol_dir = parent_dir.rstrip('/') + '_smol'
    os.makedirs(smol_dir, exist_ok=True)

    global total_files
    total_files = count_png_files(parent_dir)

    create_smol_structure(parent_dir, smol_dir, height, num_threads)


if __name__ == "__main__":
    main()
