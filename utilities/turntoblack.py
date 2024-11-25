import os
from PIL import Image, features, ImageFilter, ImageChops
import numpy as np
from scipy.ndimage import binary_dilation

# Check if WebP is supported
webp_supported = features.check('webp')
if not webp_supported:
    print("Warning: WebP support is not available in your Pillow installation.")
    print("WebP files will be skipped. To process WebP images, please install Pillow with WebP support.")
    # Remove .webp from supported extensions
    supported_extensions = ('.png', '.tif', '.tiff')
else:
    supported_extensions = ('.png', '.tif', '.tiff', '.webp')

# Prompt the user for the input folder path
input_folder = input("Enter the path to your input folder containing images: ").strip()

# Ensure the input folder exists
if not os.path.isdir(input_folder):
    print(f"The folder '{input_folder}' does not exist. Please check the path and try again.")
    exit(1)

# Define the output folder by appending '_alpha' to the input folder name
output_folder = f"{input_folder.rstrip(os.sep)}_alpha"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize a counter for processed images
processed_images = 0

# Process each image in the input folder
for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)
    if os.path.isfile(file_path) and filename.lower().endswith(supported_extensions):
        input_path = file_path
        output_path = os.path.join(output_folder, filename)

        print(f"Processing file: {input_path}")

        try:
            # Open the image and ensure it has an alpha channel
            img = Image.open(input_path).convert('RGBA')
            r, g, b, a = img.split()

            # Convert channels to NumPy arrays
            r_np = np.array(r)
            g_np = np.array(g)
            b_np = np.array(b)
            a_np = np.array(a)

            # Create a mask where alpha > 0 (visible pixels)
            alpha_mask = a_np > 0

            # Set RGB values to 0 where alpha > 0 (visible pixels become black)
            r_np[alpha_mask] = 0
            g_np[alpha_mask] = 0
            b_np[alpha_mask] = 0

            # Create a soft transition near the border of the alpha channel
            # Erode the alpha mask to find the inner area
            eroded_mask = binary_dilation(alpha_mask, iterations=-5)
            edge_mask = alpha_mask & (~eroded_mask)

            # Reduce alpha values by 25% in the edge area for soft transition
            a_np[edge_mask] = (a_np[edge_mask] * 0.75).astype(a_np.dtype)

            # Reconstruct the image with modified channels
            r_new = Image.fromarray(r_np)
            g_new = Image.fromarray(g_np)
            b_new = Image.fromarray(b_np)
            a_new = Image.fromarray(a_np)

            img_new = Image.merge('RGBA', (r_new, g_new, b_new, a_new))

            # Handle WebP alpha saving issue
            output_extension = os.path.splitext(output_path)[1].lower()
            if output_extension == '.webp':
                # Convert to 'RGBa' mode for WebP
                img_new = img_new.convert('RGBa')
                img_new.save(output_path, format='WEBP')
            else:
                img_new.save(output_path)

            print(f"Saved modified image to: {output_path}")
            processed_images += 1

        except Exception as e:
            print(f"Error processing file '{input_path}': {e}")

# Check if any images were processed
if processed_images == 0:
    print("No images were processed. Please check if the input folder contains supported image files.")
else:
    print(f"Processing complete. {processed_images} images were processed and saved in: {output_folder}")
