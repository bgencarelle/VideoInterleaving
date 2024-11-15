import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define the WebP quality level
webp_quality = 95

def convert_image(file_path, output_folder, quality):
    """Converts a single image to WebP format with RGBA mode and specified quality."""
    try:
        with Image.open(file_path) as img:
            img = img.convert("RGBA")
            output_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(file_path))[0]}.webp")
            img.save(output_path, format="WEBP", quality=quality)
            print(f"Converted {os.path.basename(file_path)} to WebP at {quality}% quality.")
    except Exception as e:
        print(f"Could not convert {os.path.basename(file_path)}: {e}")

def convert_images_to_webp(input_folder, quality=webp_quality, max_workers=4):
    # Create the output folder by appending '_webp_{webp_quality}' to the input folder name
    output_folder = f"{input_folder}_webp_{webp_quality}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List of all image files in the input folder
    image_files = [os.path.join(input_folder, filename) for filename in os.listdir(input_folder)]
    
    # Use ThreadPoolExecutor to process images concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(convert_image, file_path, output_folder, quality) for file_path in image_files]
        
        # Wait for all threads to complete
        for future in as_completed(futures):
            future.result()  # This will raise any exceptions caught during processing

# Prompt for input folder
input_folder = input("Enter the path to the folder containing the images: ")
convert_images_to_webp(input_folder, quality=webp_quality)
