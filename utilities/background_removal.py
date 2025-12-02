#!/usr/bin/env python3
"""
This script processes all images in an input folder by:
1. Applying contrast enhancement (using threading) and saving the result as PNG
   in a temporary folder (alpha_temp).
2. Removing the background using rembg (with a persistent session for "birefnet-portrait")
   in parallel, and saving the final output as PNG in the output folder (alpha).

Reference examples for using rembg:
--------------------------------------------------
# Load the Image
from PIL import Image
from rembg import new_session, remove

input = Image.open('input.png')

# Removing the background (defaults to u2net)
output = remove(input)
output.save('output.png')

# With a specific model:
model_name = "isnet-general-use"
session = new_session(model_name)
output = remove(input, session=session)
output.save('output.png')

# For processing multiple images:
rembg_session = new_session("unet")
for img in images:
    output = remove(img, session=rembg_session)
--------------------------------------------------
"""

import threading
from pathlib import Path
from io import BytesIO
from PIL import Image, ImageEnhance
from rembg import new_session, remove

# Configuration
CONTRAST_FACTOR = 1.2
TEMP_FOLDER = Path("alpha_temp")
OUTPUT_FOLDER = Path("alpha")
# Force using the birefnet-portrait model
MODEL = "birefnet-portrait"

def enhance_contrast(image_path: Path, output_path: Path, contrast_factor=CONTRAST_FACTOR):
    """ac 
    Opens an image, applies contrast enhancement, and saves it as PNG.
    """
    try:
        with Image.open(image_path).convert("RGB") as img:
            enhancer = ImageEnhance.Contrast(img)
            contrasted = enhancer.enhance(contrast_factor)
            contrasted.save(output_path, format="PNG")
            print(f"[✓] Contrast enhanced: {image_path.name} -> {output_path.name}")
    except Exception as e:
        print(f"[!] Error processing {image_path.name}: {e}")

def process_rembg(image_path: Path, output_path: Path, session):
    """
    Uses rembg's remove() function to generate an alpha matte for an image,
    then saves the result as PNG.
    """
    try:
        image_bytes = image_path.read_bytes()
        result_bytes = remove(image_bytes, session=session)
        result_image = Image.open(BytesIO(result_bytes))
        # Save as PNG to avoid recompression artifacts
        result_image.save(output_path, "PNG")
        print(f"[✓] Rembg processed: {image_path.name} -> {output_path.name}")
    except Exception as e:
        print(f"[!] Error in rembg for {image_path.name}: {e}")

def main():
    # Prompt for the input folder path.
    input_folder = input("Enter the path to the input folder: ").strip()
    input_path = Path(input_folder)
    if not input_path.exists() or not input_path.is_dir():
        print(f"Error: '{input_path}' is not a valid directory.")
        return

    # Create TEMP_FOLDER and OUTPUT_FOLDER if they don't exist.
    TEMP_FOLDER.mkdir(exist_ok=True)
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    # Step 1: Apply contrast enhancement to each image (in parallel).
    contrast_threads = []
    for image_file in input_path.iterdir():
        if image_file.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
            temp_output = TEMP_FOLDER / f"{image_file.stem}.png"
            t = threading.Thread(target=enhance_contrast, args=(image_file, temp_output))
            t.start()
            contrast_threads.append(t)
    for t in contrast_threads:
        t.join()
    print("[✓] Contrast enhancement completed for all images.")

    # Step 2: Process each contrast-enhanced image with rembg (in parallel).
    session = new_session(MODEL)
    rembg_threads = []
    for temp_file in TEMP_FOLDER.iterdir():
        if temp_file.suffix.lower() == ".png":
            output_file = OUTPUT_FOLDER / f"{temp_file.stem}.png"
            t = threading.Thread(target=process_rembg, args=(temp_file, output_file, session))
            t.start()
            rembg_threads.append(t)
    for t in rembg_threads:
        t.join()
    print("[✓] Background removal completed for all images.")

if __name__ == "__main__":
    main()
