import os
import re
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define a natural order sort key function.
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'([0-9]+)', s)]

# --- Prompt for user input ---
transparency_folder = input("Enter the path to the transparency folder: ").strip()
target_folder = input("Enter the path to the target folder (source images): ").strip()
invert_input = input("Do you want to invert the transparency? (y/n): ").strip().lower()
invert = invert_input in ("y", "yes")

# Destination root – we append '_snip' to the target folder name.
dest_root = target_folder.rstrip(os.sep) + "_snip"

# Supported image extensions (including WebP)
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.webp')

# --- Build a sorted list of transparency image paths ---
transparency_files = sorted(
    [f for f in os.listdir(transparency_folder) if f.lower().endswith(IMG_EXTS)],
    key=natural_sort_key
)
if not transparency_files:
    raise ValueError("No transparency images found in the transparency folder.")

transparency_paths = [os.path.join(transparency_folder, f) for f in transparency_files]

def process_file(src_path, file, current_root, transparency_path, invert):
    """
    Process one target image by transferring the alpha channel from the given transparency image.
    If invert flag is set, the alpha channel is inverted.
    """
    try:
        # Open the target image and convert it to RGBA.
        target_image = Image.open(src_path).convert("RGBA")
        # Open the transparency image.
        transparency_image = Image.open(transparency_path).convert("RGBA")
    except Exception as e:
        print(f"Error processing {file}: {e}")
        return

    # Resize transparency image if dimensions differ from the target image.
    if target_image.size != transparency_image.size:
        transparency_image = transparency_image.resize(target_image.size, Image.ANTIALIAS)

    # Retrieve the alpha channel from the transparency image.
    new_alpha = transparency_image.getchannel("A")

    # If the invert flag is set, invert the alpha channel.
    if invert:
        new_alpha = Image.eval(new_alpha, lambda a: 255 - a)

    # Update the target image's alpha channel.
    target_image.putalpha(new_alpha)

    # Recreate the directory structure in the destination folder.
    rel_path = os.path.relpath(current_root, target_folder)
    dest_dir = os.path.join(dest_root, rel_path)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, file)

    # Save the processed image; if it's WebP, set quality to 95.
    try:
        if file.lower().endswith('.webp'):
            target_image.save(dest_path, "WEBP", quality=95)
        else:
            target_image.save(dest_path)
        print(f"Saved processed image to {dest_path}")
    except Exception as e:
        print(f"Error saving {dest_path}: {e}")

# --- Process images using ThreadPoolExecutor ---
tasks = []
with ThreadPoolExecutor() as executor:
    for current_root, dirs, files in os.walk(target_folder):
        sorted_files = sorted([f for f in files if f.lower().endswith(IMG_EXTS)], key=natural_sort_key)
        # For each file, select a transparency image by counting.
        for i, file in enumerate(sorted_files):
            src_path = os.path.join(current_root, file)
            transparency_path = transparency_paths[i % len(transparency_paths)]
            tasks.append(executor.submit(process_file, src_path, file, current_root, transparency_path, invert))
    for future in as_completed(tasks):
        try:
            future.result()
        except Exception as e:
            print(f"Error in threaded task: {e}")
