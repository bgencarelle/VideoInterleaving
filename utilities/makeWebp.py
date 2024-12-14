import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import webp


def process_image(input_path, output_path, quality=85, max_width=720):
    """
    Process a single image:
    - Opens the image using Pillow.
    - Resizes it to a maximum width of 'max_width' while maintaining aspect ratio.
    - Ensures the image is in RGBA mode to preserve all four channels (including alpha).
    - Encodes the image to WebP using webp.WebPPicture with a photo-optimized preset.
    """
    try:
        # Ensure the parent directory of the output file exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with Image.open(input_path) as img:
            # Convert to RGBA to preserve alpha
            img = img.convert("RGBA")

            # Resize if needed
            width, height = img.size
            if width > max_width:
                aspect_ratio = height / width
                new_width = max_width
                new_height = int(new_width * aspect_ratio)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Encode to WebP in memory with PHOTO preset
            pic = webp.WebPPicture.from_pil(img)
            config = webp.WebPConfig.new(preset=webp.WebPPreset.PHOTO, quality=quality)
            buf = pic.encode(config).buffer()

            # Write the encoded data to disk
            with open(output_path, 'wb') as f:
                f.write(buf)

            print(f"Processed: {input_path} -> {output_path}")

    except UnidentifiedImageError:
        print(f"Skipping invalid image file: {input_path}")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")


def process_directory(input_dir, quality=85, max_width=720, max_workers=6):
    """
    Recursively processes all images in 'input_dir', converting them to WebP with
    the given quality and max_width settings, preserving alpha channels.
    The output directory will be created in the same parent directory as 'input_dir'.
    """
    input_dir = Path(input_dir)
    output_dir = input_dir.parent / f"{input_dir.name}_{quality}q_{max_width}p"

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    image_files = [
        file for file in input_dir.rglob("*")
        if file.suffix.lower() in image_extensions and file.is_file()
    ]

    # Use threading to speed up processing for large numbers of images
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for image_file in image_files:
            relative_path = image_file.relative_to(input_dir)
            output_path = output_dir / relative_path.with_suffix(".webp")
            executor.submit(process_image, image_file, output_path, quality, max_width)

    print(f"Processing complete. Optimized images saved to: {output_dir}")


if __name__ == "__main__":
    source_directory = input("Enter the source directory path: ").strip()
    process_directory(source_directory)
