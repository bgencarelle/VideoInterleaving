#!/usr/bin/env python3
import threading
from pathlib import Path
from io import BytesIO

from PIL import Image, ImageEnhance
from rembg import new_session, remove

# Configuration
CONTRAST_FACTOR = 1.2
TEMP_FOLDER = Path("alpha_temp")
OUTPUT_FOLDER = Path("alpha")

def enhance_contrast(image_path: Path, output_path: Path, contrast_factor=CONTRAST_FACTOR):
    """
    Open an image, apply contrast enhancement, and save it as PNG.
    """
    try:
        with Image.open(image_path).convert("RGB") as img:
            enhancer = ImageEnhance.Contrast(img)
            contrasted = enhancer.enhance(contrast_factor)
            contrasted.save(output_path, format="PNG")
            print(f"[✓] Contrast applied: {image_path.name}")
    except Exception as e:
        print(f"[!] Error processing {image_path.name}: {e}")

def process_rembg(image_path: Path, output_path: Path, session):
    """
    Use rembg to generate a matte for an image and save the result as WebP.
    """
    try:
        image_bytes = image_path.read_bytes()
        result_bytes = remove(image_bytes, session=session)
        result_image = Image.open(BytesIO(result_bytes))
        result_image.save(output_path, "WEBP")
        print(f"[✓] Rembg processed: {image_path.name} -> {output_path.name}")
    except Exception as e:
        print(f"[!] Error in rembg for {image_path.name}: {e}")

def main():
    # Prompt for input folder and model selection
    input_folder = input("Enter the path to the input folder: ").strip()
    input_path = Path(input_folder)
    if not input_path.exists() or not input_path.is_dir():
        print(f"Error: '{input_path}' is not a valid directory.")
        return

    model = "birefnet-portrait"
    print(f"Using model: {model}")

    # Create output folders if they don't exist
    TEMP_FOLDER.mkdir(exist_ok=True)
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    # Step 1: Contrast enhancement on all images (in parallel)
    threads = []
    for image_file in input_path.iterdir():
        if image_file.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
            temp_image_path = TEMP_FOLDER / (image_file.stem + ".png")
            thread = threading.Thread(target=enhance_contrast, args=(image_file, temp_image_path))
            thread.start()
            threads.append(thread)
    for thread in threads:
        thread.join()
    print("[✓] Contrast enhancement completed for all images.")

    # Step 2: Process each pre-processed image with rembg using the chosen model.
    session = new_session(model)
    for temp_image_file in TEMP_FOLDER.iterdir():
        if temp_image_file.suffix.lower() == ".png":
            output_file_path = OUTPUT_FOLDER / (temp_image_file.stem + ".webp")
            process_rembg(temp_image_file, output_file_path, session)

if __name__ == "__main__":
    main()
