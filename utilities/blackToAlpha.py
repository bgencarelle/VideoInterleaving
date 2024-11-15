import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

def make_black_pixels_transparent(file_path, output_folder):
    """Convert black pixels in an image to fully transparent."""
    try:
        with Image.open(file_path) as img:
            # Ensure the image is in RGBA mode
            img = img.convert("RGBA")
            data = img.getdata()
            
            # Replace black pixels with fully transparent pixels
            new_data = [
                (r, g, b, 0) if (r, g, b) <= (4, 4, 4) else (r, g, b, a)
                for (r, g, b, a) in data
            ]
            img.putdata(new_data)
            
            # Save the modified image in the output folder
            output_path = os.path.join(output_folder, os.path.basename(file_path))
            img.save(output_path, format="PNG")
            print(f"Processed {os.path.basename(file_path)}")
    
    except Exception as e:
        print(f"Could not process {os.path.basename(file_path)}: {e}")

def process_images_in_folder(input_folder, max_workers=8):
    # Create the output folder by appending '_transparent' to the input folder name
    output_folder = f"{input_folder}_transparent"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in the input folder
    image_files = [os.path.join(input_folder, filename) for filename in os.listdir(input_folder)]
    
    # Use ThreadPoolExecutor for concurrent processing of images
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(make_black_pixels_transparent, file_path, output_folder) for file_path in image_files]
        
        # Ensure each thread completes and catch any exceptions
        for future in as_completed(futures):
            future.result()

# Prompt for input folder
input_folder = input("Enter the path to the folder containing the images: ")
process_images_in_folder(input_folder, max_workers=os.cpu_count())
